"""LINE Beacon proximity: signatures, signed device message, freshness, linking, webhook → receipt."""
import base64
import hashlib
import hmac
import json
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

from flask import Flask

import app
from app.routes import api_checkin, line as line_route
from app.services import line_beacon, proximity_receipt

KEY = bytes(range(32)).hex()
CHANNEL_SECRET = "channel-secret-for-tests"
RECEIPT_SECRET = "r" * 64


class DeviceMessageTests(unittest.TestCase):
    def test_vector_and_round_trip(self):
        dm = line_beacon.device_message(KEY, 0x12345678)
        self.assertEqual(len(dm), 12)
        self.assertEqual(dm[:4], bytes.fromhex("12345678"))
        expected = hmac.new(bytes.fromhex(KEY), b"smartcheck-line-v1" + dm[:4], hashlib.sha256).digest()[:8]
        self.assertEqual(dm[4:], expected)
        self.assertEqual(line_beacon.verify_dm(KEY, dm.hex()), 0x12345678)

    def test_tampered_wrong_key_or_shape(self):
        dm = bytearray(line_beacon.device_message(KEY, 7))
        dm[11] ^= 1
        self.assertIsNone(line_beacon.verify_dm(KEY, dm.hex()))
        self.assertIsNone(line_beacon.verify_dm("ff" * 32, line_beacon.device_message(KEY, 7).hex()))
        self.assertIsNone(line_beacon.verify_dm(KEY, "zz" * 12))
        self.assertIsNone(line_beacon.verify_dm(KEY, "00" * 11))
        self.assertIsNone(line_beacon.verify_dm(KEY, None))
        self.assertIsNone(line_beacon.verify_dm(None, line_beacon.device_message(KEY, 7).hex()))

    def test_signature(self):
        body = b'{"events":[]}'
        sig = base64.b64encode(hmac.new(CHANNEL_SECRET.encode(), body, hashlib.sha256).digest()).decode()
        self.assertTrue(line_beacon.valid_signature(CHANNEL_SECRET, body, sig))
        self.assertFalse(line_beacon.valid_signature(CHANNEL_SECRET, body + b" ", sig))
        self.assertFalse(line_beacon.valid_signature(CHANNEL_SECRET, body, ""))
        self.assertFalse(line_beacon.valid_signature("", body, sig))


class FreshnessTests(unittest.TestCase):
    def client(self, first_seen):
        q = Mock()
        for m in ("upsert", "select", "eq", "maybe_single"):
            getattr(q, m).return_value = q
        q.execute.return_value = SimpleNamespace(data={"first_seen_at": first_seen.isoformat()})
        return SimpleNamespace(table=lambda name: q), q

    def test_first_sighting_window(self):
        now = datetime(2026, 10, 20, 9, 0, tzinfo=timezone.utc)
        client, q = self.client(now - timedelta(seconds=30))
        self.assertTrue(line_beacon.fresh(client, "b1", 5, now=now))
        q.upsert.assert_called_once()
        self.assertTrue(q.upsert.call_args.kwargs["ignore_duplicates"])
        client, _ = self.client(now - timedelta(seconds=line_beacon.FRESH_SECONDS + 1))
        self.assertFalse(line_beacon.fresh(client, "b1", 5, now=now))   # a recording replayed later


class LinkCodeTests(unittest.TestCase):
    def setUp(self):
        self.rows = {"line_link_codes": {}, "users": {"u1": {"id": "u1", "line_user_id": None}}}

    def client(self):
        rows = self.rows

        def table(name):
            q = Mock()
            state = {"filters": {}}
            for m in ("select", "maybe_single"):
                getattr(q, m).return_value = q
            q.eq.side_effect = lambda k, v: (state["filters"].__setitem__(k, v), q)[1]
            q.upsert.side_effect = lambda row, on_conflict=None: (rows[name].__setitem__(row["user_id"], row), q)[1]
            q.update.side_effect = lambda patch_: (state.__setitem__("update", patch_), q)[1]
            q.delete.side_effect = lambda: (state.__setitem__("delete", True), q)[1]

            def execute():
                f = state["filters"]
                if "update" in state:
                    rows[name][f["id"]].update(state["update"]); return SimpleNamespace(data=[])
                if state.get("delete"):
                    rows[name].pop(f["user_id"], None); return SimpleNamespace(data=[])
                if name == "line_link_codes":
                    hit = next((r for r in rows[name].values() if r["code_hash"] == f.get("code_hash")), None)
                    return SimpleNamespace(data=hit)
                hit = next((r for r in rows[name].values() if r.get("line_user_id") == f.get("line_user_id")), None)
                return SimpleNamespace(data=hit)
            q.execute.side_effect = execute
            return q
        return SimpleNamespace(table=table)

    def test_link_flow(self):
        client = self.client()
        code = line_beacon.new_link_code(client, "u1")
        self.assertRegex(code, r"^\d{6}$")
        self.assertNotIn(code, json.dumps(self.rows["line_link_codes"]))     # only the hash is stored
        self.assertEqual(line_beacon.redeem_link_code(client, "Uline", "link 000000" if code != "000000" else "LINK 111111"), "invalid")
        self.assertEqual(line_beacon.redeem_link_code(client, "Uline", f"LINK {code}"), "linked")
        self.assertEqual(self.rows["users"]["u1"]["line_user_id"], "Uline")
        self.assertEqual(self.rows["line_link_codes"], {})                    # one use

    def test_expired_code(self):
        client = self.client()
        code = line_beacon.new_link_code(client, "u1", now=datetime.now(timezone.utc) - timedelta(minutes=11))
        self.assertEqual(line_beacon.redeem_link_code(client, "Uline", f"LINK {code}"), "invalid")


class WebhookTests(unittest.TestCase):
    def setUp(self):
        self.web = Flask(__name__)
        self.web.config.update(SECRET_KEY="t", RATELIMIT_ENABLED=False, LINE_BEACON_ENABLED=True,
                               LINE_CHANNEL_SECRET=CHANNEL_SECRET, LINE_CHANNEL_ACCESS_TOKEN="token",
                               PUBLIC_BASE_URL="https://smartcheck.example", PROXIMITY_RECEIPT_SECRET=RECEIPT_SECRET)
        app.limiter.init_app(self.web)
        self.web.register_blueprint(line_route.line_bp)
        self.client = self.web.test_client()
        self.replies = []
        self.data = {
            "users": {"id": "student-1", "role": "student"},
            "beacons": {"id": "beacon-1", "room_name": "LAB 2", "ble_room_code": "LAB-2", "ble_secret": KEY},
            "sessions": [{"id": "sess-1", "course_id": "c1"}],
            "course_enrollments": [{"course_id": "c1"}],
        }
        self.fresh = True
        db = Mock()

        def table(name):
            q = Mock()
            for m in ("select", "eq", "maybe_single"):
                getattr(q, m).return_value = q
            q.execute.return_value = SimpleNamespace(data=self.data.get(name))
            return q
        db.table.side_effect = table
        for target in (patch.object(line_route, "supabase_admin", db),
                       patch.object(line_beacon, "reply", side_effect=lambda tok, rt, msgs: self.replies.append(msgs)),
                       patch.object(line_beacon, "fresh", side_effect=lambda c, b, n: self.fresh)):
            target.start()
            self.addCleanup(target.stop)

    def post(self, events, secret=CHANNEL_SECRET):
        body = json.dumps({"events": events}).encode()
        sig = base64.b64encode(hmac.new(secret.encode(), body, hashlib.sha256).digest()).decode()
        return self.client.post("/line/webhook", data=body, headers={"X-Line-Signature": sig,
                                                                    "Content-Type": "application/json"})

    def beacon(self, dm=None, kind="enter"):
        return {"type": "beacon", "replyToken": "rt", "timestamp": 1760000000000,
                "source": {"type": "user", "userId": "Uline"},
                "beacon": {"hwid": "0102030405", "type": kind,
                           "dm": dm if dm is not None else line_beacon.device_message(KEY, 42).hex()}}

    def test_enter_issues_a_receipt_link(self):
        r = self.post([self.beacon()])
        self.assertEqual(r.status_code, 200)
        message = self.replies[0][0]
        self.assertEqual(message["type"], "template")
        url = message["template"]["actions"][0]["uri"]
        self.assertTrue(url.startswith("https://smartcheck.example/student/checkin?"))
        q = parse_qs(urlparse(url).query)
        self.assertEqual((q["session_id"][0], q["via"][0], q["room"][0]), ("sess-1", "line", "LAB-2"))
        self.assertIsNone(proximity_receipt.verify(q["r"][0], RECEIPT_SECRET, "student-1", "sess-1", "line",
                                                   "LAB-2", max_age=line_beacon.RECEIPT_TTL_SECONDS))
        # The receipt is bound to this student, session and method.
        self.assertIsNotNone(proximity_receipt.verify(q["r"][0], RECEIPT_SECRET, "student-2", "sess-1", "line", "LAB-2"))
        self.assertIsNotNone(proximity_receipt.verify(q["r"][0], RECEIPT_SECRET, "student-1", "sess-1", "ble", "LAB-2"))

    def test_bad_signature_and_disabled(self):
        self.assertEqual(self.post([self.beacon()], secret="wrong").status_code, 400)
        self.assertEqual(self.replies, [])
        self.web.config["LINE_BEACON_ENABLED"] = False
        self.assertEqual(self.post([self.beacon()]).status_code, 404)

    def test_invalid_or_replayed_device_message_gets_no_receipt(self):
        self.post([self.beacon(dm="00" * 12)])
        self.assertEqual(self.replies[-1][0]["type"], "text")         # "ยืนยันห้องเรียนไม่สำเร็จ"
        self.replies.clear()
        self.fresh = False
        self.post([self.beacon()])
        self.assertEqual(self.replies, [])                             # replay: silent, logged

    def test_unlinked_no_session_and_leave(self):
        self.data["users"] = None
        self.post([self.beacon()])
        self.assertIn("ยังไม่ได้เชื่อมบัญชี", self.replies[-1][0]["text"])
        self.data["users"] = {"id": "student-1", "role": "student"}
        self.data["course_enrollments"] = [{"course_id": "other"}]
        self.post([self.beacon()])
        self.assertIn("ไม่มีคาบ", self.replies[-1][0]["text"])
        before = len(self.replies)
        self.post([self.beacon(kind="leave")])
        self.assertEqual(len(self.replies), before)


class CheckinReceiptMethodTests(unittest.TestCase):
    def test_line_receipts_only_when_enabled(self):
        web = Flask(__name__)
        web.config.update(CHECKIN_PROXIMITY_METHOD="ble", LINE_BEACON_ENABLED=True)
        with web.test_request_context():
            self.assertEqual(api_checkin._receipt_method({"proximity_via": "line"}), ("line", line_beacon.RECEIPT_TTL_SECONDS))
            self.assertEqual(api_checkin._receipt_method({}), ("ble", proximity_receipt.TTL_SECONDS))
            web.config["LINE_BEACON_ENABLED"] = False
            self.assertEqual(api_checkin._receipt_method({"proximity_via": "line"}), ("ble", proximity_receipt.TTL_SECONDS))


if __name__ == "__main__":
    unittest.main()

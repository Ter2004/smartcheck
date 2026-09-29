import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from flask import Flask

import app
from app.routes import api_checkin as route
from app.services import ble_challenge

# Fixed vector shared with the firmware parity check (scripts/ble_hmac_parity.py).
KEY = bytes(range(32)).hex()
NONCE = bytes(range(16)).hex()
VECTOR = "975fe6e237c23b0e5b1263789a3646161a57763c3d512ded8c7e797f5029f48a"


class VerifyTests(unittest.TestCase):
    def challenge(self, **changes):
        return {"nonce": NONCE, "session_id": "s1", "beacon_id": "b1", "issued_at": 1000.0,
                **changes}

    def verify(self, challenge, response=VECTOR, now=1010.0, session_id="s1", beacon_id="b1"):
        return ble_challenge.verify(challenge, session_id, beacon_id, KEY, response, now=now)

    def test_fixed_vector(self):
        self.assertEqual(ble_challenge.expected_response(KEY, NONCE), VECTOR)

    def test_board_answer_passes_in_either_hex_case(self):
        self.assertIsNone(self.verify(self.challenge()))
        self.assertIsNone(self.verify(self.challenge(), response=VECTOR.upper()))

    def test_reject_reasons(self):
        cases = [
            (None, {}, "ble_challenge_missing"),
            (self.challenge(), {"session_id": "s2"}, "ble_challenge_mismatch"),
            (self.challenge(), {"beacon_id": "b2"}, "ble_challenge_mismatch"),
            (self.challenge(), {"now": 1031.0}, "ble_challenge_expired"),
            (self.challenge(), {"now": 999.0}, "ble_challenge_expired"),
            (self.challenge(), {"response": "TEST-101"}, "ble_response_invalid"),
            (self.challenge(), {"response": None}, "ble_response_invalid"),
            (self.challenge(), {"response": "0" * 64}, "ble_response_invalid"),
        ]
        for challenge, changes, reason in cases:
            with self.subTest(reason=reason, changes=changes):
                self.assertEqual(self.verify(challenge, **changes), reason)

    def test_nonces_are_16_random_bytes(self):
        first = ble_challenge.new_challenge("s1", "b1")["nonce"]
        second = ble_challenge.new_challenge("s1", "b1")["nonce"]
        self.assertEqual(len(bytes.fromhex(first)), 16)
        self.assertNotEqual(first, second)


class Database:
    def __init__(self):
        self.secret = KEY

    def table(self, name):
        query = Mock()
        for method in ["select", "eq", "maybe_single"]:
            getattr(query, method).return_value = query
        data = {
            "sessions": {"id": "s1", "course_id": "c1", "is_open": True, "beacon_id": "b1",
                         "beacons": {"ble_room_code": "TEST-101"}},
            "course_enrollments": {"id": "enrollment"},
            "beacons": {"ble_secret": self.secret},
        }.get(name)
        query.execute.return_value = SimpleNamespace(data=data)
        return query


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.db = Database()
        patcher = patch.object(route, "supabase_admin", self.db)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.web = Flask(__name__)
        self.web.config.update(SECRET_KEY="test", RATELIMIT_ENABLED=False,
                               PROXIMITY_RECEIPT_SECRET="r" * 64,
                               CHECKIN_PROXIMITY_METHOD="ble")
        app.limiter.init_app(self.web)
        self.web.register_blueprint(route.api_checkin_bp)
        self.client = self.web.test_client()
        with self.client.session_transaction() as session:
            session.update(user_id="student", user_role="student", csrf_token="csrf")

    def post(self, path, **body):
        return self.client.post(path, json={"session_id": "s1", "room_code": "TEST-101", **body},
                                headers={"X-CSRF-Token": "csrf"})

    def challenge(self):
        return self.post("/api/checkin/ble/challenge")

    def prove(self, response):
        return self.post("/api/checkin/proximity", ble_response=response)

    def answer(self, nonce):
        return ble_challenge.expected_response(KEY, nonce)

    def test_room_code_alone_is_refused(self):
        # The vulnerability: the static code is public, so it must not be enough.
        response = self.post("/api/checkin/proximity")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error_code"], "room_code_invalid")
        self.assertNotIn("proximity_receipt", response.json)

    def test_signed_nonce_gets_a_receipt(self):
        nonce = self.challenge().json["nonce"]
        response = self.prove(self.answer(nonce))
        self.assertEqual(response.status_code, 200, response.json)
        self.assertTrue(response.json["proximity_receipt"])

    def test_answer_cannot_be_reused(self):
        nonce = self.challenge().json["nonce"]
        self.assertEqual(self.prove(self.answer(nonce)).status_code, 200)
        self.assertEqual(self.prove(self.answer(nonce)).status_code, 400)

    def test_wrong_answer_uses_up_the_challenge(self):
        nonce = self.challenge().json["nonce"]
        self.assertEqual(self.prove("0" * 64).status_code, 400)
        self.assertEqual(self.prove(self.answer(nonce)).status_code, 400)

    def test_expired_challenge_is_refused(self):
        nonce = self.challenge().json["nonce"]
        # Age the stored challenge (patching time.time would also expire the session).
        with self.client.session_transaction() as session:
            aged = dict(session["ble_challenge"])
            aged["issued_at"] -= ble_challenge.TTL_SECONDS + 1
            session["ble_challenge"] = aged
        response = self.prove(self.answer(nonce))
        self.assertEqual(response.status_code, 400)
        self.assertIn("หมดเวลา", response.json["error"])

    def test_room_without_secret_is_503(self):
        self.db.secret = None
        response = self.challenge()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json["error_code"], "room_code_unavailable")

    def test_wrong_room_code_gets_no_challenge(self):
        response = self.post("/api/checkin/ble/challenge", room_code="OTHER-1")
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("nonce", response.json)

    def test_challenge_response_never_contains_the_key(self):
        response = self.challenge()
        self.assertNotIn(KEY, response.get_data(as_text=True))
        self.assertEqual(response.headers["Cache-Control"], "no-store")

    def test_totp_mode_has_no_challenge(self):
        self.web.config["CHECKIN_PROXIMITY_METHOD"] = "totp"
        self.assertEqual(self.challenge().status_code, 404)


if __name__ == "__main__":
    unittest.main()

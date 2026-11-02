"""Class-open notifications: Firebase sender, who gets them, once only, register/forget the phone."""
import json
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from flask import Flask

import app
from app.routes import auth, push
from app.services import class_push, fcm

KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
PEM = KEY.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                        serialization.NoEncryption()).decode()
ACCOUNT = {"project_id": "smartcheck-test", "client_email": "push@smartcheck-test.iam.gserviceaccount.com",
           "private_key": PEM}
CONFIG = {"SESSION_OPEN_MODE": "teacher", "CLASS_ON_TIME_MINUTES": 5, "FCM_SERVICE_ACCOUNT_FILE": "x"}
OPENED = datetime(2026, 11, 3, 1, 0, tzinfo=timezone.utc)
TOKEN_A, TOKEN_B = "a" * 40, "b:" + "B" * 40


class SenderTests(unittest.TestCase):
    def setUp(self):
        self.requests = []

        def handler(request):
            self.requests.append(request)
            if request.url.host == "oauth2.googleapis.com":
                return httpx.Response(200, json={"access_token": "at-1", "expires_in": 3600})
            token = json.loads(request.content)["message"]["token"]
            return {TOKEN_A: httpx.Response(200, json={"name": "m"}),
                    TOKEN_B: httpx.Response(404, json={"error": {"status": "NOT_FOUND"}})}.get(
                token, httpx.Response(500, json={"error": {"status": "INTERNAL"}}))
        self.sender = fcm.Sender(ACCOUNT, httpx.Client(transport=httpx.MockTransport(handler)))

    def test_signs_a_service_account_jwt_and_reuses_the_access_token(self):
        self.assertEqual(self.sender.send(TOKEN_A, {"data": {}}), fcm.SENT)
        self.assertEqual(self.sender.send(TOKEN_A, {"data": {}}), fcm.SENT)
        token_requests = [r for r in self.requests if r.url.host == "oauth2.googleapis.com"]
        self.assertEqual(len(token_requests), 1)
        assertion = dict(x.split("=", 1) for x in token_requests[0].content.decode().split("&"))["assertion"]
        claims = jwt.decode(assertion, KEY.public_key(), algorithms=["RS256"], audience=fcm.TOKEN_URL)
        self.assertEqual(claims["iss"], ACCOUNT["client_email"])
        self.assertEqual(claims["scope"], fcm.SCOPE)
        send = self.requests[-1]
        self.assertEqual(str(send.url), "https://fcm.googleapis.com/v1/projects/smartcheck-test/messages:send")
        self.assertEqual(send.headers["Authorization"], "Bearer at-1")

    def test_unregistered_token_and_server_errors(self):
        self.assertEqual(self.sender.send(TOKEN_B, {}), fcm.INVALID_TOKEN)
        self.assertEqual(self.sender.send("other-token", {}), fcm.FAILED)

    def test_no_service_account_means_no_push(self):
        self.assertIsNone(fcm.sender({"FCM_SERVICE_ACCOUNT_FILE": ""}))


class Database:
    """sessions/courses/enrollments/attendance/push_devices; records updates, deletes and is_ filters."""

    def __init__(self, session_row, claim_wins=True):
        self.session_row, self.claim_wins = session_row, claim_wins
        self.updates, self.deleted = [], []

    def table(self, name):
        q = Mock()
        state = {"update": None, "delete": False}
        for method in ["select", "eq", "is_", "in_", "maybe_single", "not_"]:
            getattr(q, method).return_value = q
        q.not_ = q

        def update(values):
            state["update"] = values
            self.updates.append((name, values))
            return q
        q.update.side_effect = update

        def delete():
            state["delete"] = True
            return q
        q.delete.side_effect = delete

        def in_(column, values):
            if state["delete"]:
                self.deleted.extend(values)
            return q
        q.in_.side_effect = in_

        def execute():
            if name == "sessions":
                if state["update"] is not None:
                    return SimpleNamespace(data=[self.session_row] if self.claim_wins else [])
                return SimpleNamespace(data=[self.session_row])
            return SimpleNamespace(data={
                "courses": {"code": "SE101"},
                "course_enrollments": [{"student_id": "s1"}, {"student_id": "s2"}, {"student_id": "s3"}],
                "attendance": [{"student_id": "s2"}],
                "push_devices": [{"token": TOKEN_A}, {"token": TOKEN_B}],
            }[name])
        q.execute.side_effect = execute
        return q


class NotifyTests(unittest.TestCase):
    def setUp(self):
        self.row = {"id": "sess-1", "course_id": "c1", "is_open": True,
                    "opened_at": OPENED.isoformat(), "present_minutes": 5}
        self.sent = []
        sender = Mock()
        sender.send.side_effect = lambda token, msg: (self.sent.append((token, msg)),
                                                      fcm.SENT if token == TOKEN_A else fcm.INVALID_TOKEN)[1]
        patcher = patch.object(fcm, "sender", Mock(return_value=sender))
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_with(self, db, fn, now):
        with patch.object(class_push, "_db", return_value=db):
            return fn("sess-1", CONFIG, now)

    def test_open_goes_to_students_not_checked_in_and_drops_dead_tokens(self):
        db = Database(self.row)
        reached = self.run_with(db, class_push.notify_opened, OPENED)
        self.assertEqual(reached, 1)
        self.assertEqual([t for t, _ in self.sent], [TOKEN_A, TOKEN_B])
        msg = self.sent[0][1]
        self.assertEqual(msg["notification"]["title"], "SE101 เปิดเช็คชื่อแล้ว")
        self.assertIn("08:05", msg["notification"]["body"])
        self.assertEqual(msg["data"]["url"], "/student/checkin?session_id=sess-1")
        self.assertEqual(msg["android"]["notification"], {"channel_id": "checkin", "tag": "sess-1"})
        self.assertEqual(db.deleted, [TOKEN_B])
        self.assertIn(("sessions", {"open_push_at": OPENED.isoformat()}), db.updates)

    def test_recipients_exclude_checked_in_students(self):
        captured = {}
        db = Database(self.row)
        original = db.table

        def table(name):
            q = original(name)
            if name == "push_devices":
                inner = q.in_.side_effect
                q.in_.side_effect = lambda column, values: (captured.setdefault(column, values), inner(column, values))[1]
            return q
        db.table = table
        self.run_with(db, class_push.notify_opened, OPENED)
        self.assertEqual(captured["user_id"], ["s1", "s3"])

    def test_sent_once_when_another_process_claimed_it(self):
        self.assertEqual(self.run_with(Database(self.row, claim_wins=False), class_push.notify_opened, OPENED), 0)
        self.assertEqual(self.sent, [])

    def test_reminder_only_while_still_on_time(self):
        reached = self.run_with(Database(self.row), class_push.notify_reminder, OPENED + timedelta(minutes=4))
        self.assertEqual(reached, 1)
        self.assertEqual(self.sent[0][1]["notification"]["title"], "SE101 ยังไม่ได้เช็คชื่อ")
        self.assertIn("มาเรียนได้แล้ว", self.sent[0][1]["notification"]["body"])
        self.sent.clear()
        self.run_with(Database(self.row), class_push.notify_reminder, OPENED + timedelta(minutes=5))
        self.assertEqual(self.sent, [])

    def test_off_without_teacher_mode(self):
        self.assertEqual(class_push.notify_opened("sess-1", {**CONFIG, "SESSION_OPEN_MODE": "schedule"}), 0)


class CatchUpTests(unittest.TestCase):
    def test_tick_sends_missed_open_and_due_reminder(self):
        rows = [{"id": "new", "opened_at": OPENED.isoformat(), "present_minutes": 5,
                 "open_push_at": None, "reminder_push_at": None},
                {"id": "old", "opened_at": (OPENED - timedelta(minutes=30)).isoformat(), "present_minutes": 5,
                 "open_push_at": None, "reminder_push_at": None}]
        db = Mock()
        q = db.table.return_value
        for method in ["select", "eq", "is_"]:
            getattr(q, method).return_value = q
        q.not_ = q
        q.execute.return_value = SimpleNamespace(data=rows)
        with patch.object(class_push, "_db", return_value=db), \
                patch.object(class_push, "enabled", return_value=True), \
                patch.object(class_push, "notify_opened") as opened, \
                patch.object(class_push, "notify_reminder") as reminder:
            class_push.catch_up(CONFIG, OPENED + timedelta(minutes=4, seconds=10))
        self.assertEqual([c.args[0] for c in opened.call_args_list], ["new"])   # "old" is too late
        self.assertEqual([c.args[0] for c in reminder.call_args_list], ["new"])


class RegisterTests(unittest.TestCase):
    def setUp(self):
        self.web = Flask(__name__)
        self.web.config.update(SECRET_KEY="t", RATELIMIT_ENABLED=False)
        app.limiter.init_app(self.web)
        self.web.register_blueprint(auth.auth_bp)
        self.web.register_blueprint(push.push_bp)
        self.client = self.web.test_client()
        with self.client.session_transaction() as s:
            s.update(user_id="s1", user_role="student", csrf_token="c")
        self.db = Mock()
        q = self.db.table.return_value
        for method in ["upsert", "delete", "eq"]:
            getattr(q, method).return_value = q
        self.query = q
        patcher = patch.object(push, "supabase_admin", self.db)
        patcher.start()
        self.addCleanup(patcher.stop)

    def register(self, token, csrf="c"):
        return self.client.post("/api/push/register", json={"token": token}, headers={"X-CSRF-Token": csrf})

    def test_stores_the_token_for_the_logged_in_student(self):
        self.assertEqual(self.register(TOKEN_B).status_code, 200)
        row = self.query.upsert.call_args.args[0]
        self.assertEqual((row["token"], row["user_id"]), (TOKEN_B, "s1"))
        self.assertEqual(self.query.upsert.call_args.kwargs, {"on_conflict": "token"})

    def test_rejects_bad_tokens_missing_csrf_and_teachers(self):
        self.assertEqual(self.register("short").status_code, 400)
        self.assertEqual(self.register("x" * 30 + "<script>").status_code, 400)
        self.assertEqual(self.register(TOKEN_A, csrf="").status_code, 403)
        with self.client.session_transaction() as s:
            s["user_role"] = "teacher"
        self.assertNotEqual(self.register(TOKEN_A).status_code, 200)
        self.query.upsert.assert_not_called()

    def test_push_script_only_for_students_in_a_firebase_build_when_the_server_can_send(self):
        from flask import render_template_string

        def loaded(user_agent, role="student", enabled=True):
            with self.web.test_request_context("/", headers={"User-Agent": user_agent}), \
                    patch.object(class_push, "enabled", return_value=enabled):
                from flask import session
                session["user_role"] = role
                return render_template_string("{{ 'yes' if push_in_app else 'no' }}") == "yes"
        app_ua = "Mozilla/5.0 (Linux; Android 16; wv) SmartCheckApp SmartCheckPush"
        self.assertTrue(loaded(app_ua))
        self.assertFalse(loaded("Mozilla/5.0 (Linux; Android 16; wv) SmartCheckApp"))
        self.assertFalse(loaded(app_ua, role="teacher"))
        self.assertFalse(loaded(app_ua, enabled=False))

    def test_logout_forgets_this_phone(self):
        self.register(TOKEN_A)
        self.client.post("/logout", data={"csrf_token": "c"})
        self.query.delete.assert_called_once()
        self.assertEqual([c.args for c in self.query.eq.call_args_list], [("token", TOKEN_A), ("user_id", "s1")])


if __name__ == "__main__":
    unittest.main()

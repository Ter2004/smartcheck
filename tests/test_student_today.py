"""Home page poll (/student/today.json): board states + what was recorded (Phase 9)."""
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from flask import Flask

import app
from app.routes import student

TH = timezone(timedelta(hours=7))
NOW = datetime(2026, 11, 6, 10, 3, tzinfo=TH)


def sess(id_, **fields):
    start = datetime(2026, 11, 6, 10, 0, tzinfo=TH).astimezone(timezone.utc).isoformat()
    return {"id": id_, "start_time": start, "is_open": False, "end_time": None,
            "courses": {"code": id_.upper(), "name": "n"}, "beacons": {"room_name": "TEST-101"}, **fields}


class TodayStatesTests(unittest.TestCase):
    def setUp(self):
        self.web = Flask(__name__)
        self.web.config.update(SECRET_KEY="test", RATELIMIT_ENABLED=False,
                               SESSION_OPEN_MODE="teacher", CLASS_ON_TIME_MINUTES=5)
        app.limiter.init_app(self.web)
        self.web.register_blueprint(student.student_bp, url_prefix="/student")
        self.client = self.web.test_client()
        with self.client.session_transaction() as s:
            s.update(user_id="student", user_role="student")

    def get(self, reads):
        with patch.object(student, "_today_board_reads", **reads):
            return self.client.get("/student/today.json")

    def test_states_and_recorded_check_ins(self):
        recorded = {"session_id": "a", "status": "late", "check_in_at": "2026-11-06T03:07:00+00:00"}
        res = self.get({"return_value": ([sess("a", end_time="x"), sess("b"), sess("c", is_open=True)],
                                         [recorded], NOW)})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json["states"], {"a": "boarded", "b": "waiting", "c": "boarding"})
        self.assertEqual(res.json["checked"], {"a": {"status": "late", "check_in_at": recorded["check_in_at"]}})

    def test_read_failure_is_a_503_not_a_crash(self):
        res = self.get({"side_effect": RuntimeError("db down")})
        self.assertEqual(res.status_code, 503)
        self.assertFalse(res.json["ok"])


if __name__ == "__main__":
    unittest.main()

"""Teacher live view: head-count while a class is open (Phase 8d)."""
import unittest

from app.routes.teacher import live_counts


class LiveCountTests(unittest.TestCase):
    def test_counts_checked_late_and_rest(self):
        rows = [{"status": "present"}, {"status": "late"}, {"status": "manual"}, {"status": "absent"}]
        self.assertEqual(live_counts(rows, 6),
                         {"total": 6, "checked": 3, "late": 1, "present": 2, "rest": 3})

    def test_never_negative_when_rows_outnumber_the_roster(self):
        self.assertEqual(live_counts([{"status": "present"}] * 3, 2)["rest"], 0)


class LivePollLimitTests(unittest.TestCase):
    """The 10 s poll is 360/hour: the 200/hour default answered 429 at request 201."""
    def test_poll_has_its_own_per_minute_limit(self):
        from unittest.mock import patch
        from flask import Flask
        import app
        from app.routes import teacher
        web = Flask(__name__)
        web.config.update(SECRET_KEY="test", RATELIMIT_ENABLED=True, RATELIMIT_STORAGE_URI="memory://")
        app.limiter.init_app(web)
        web.register_blueprint(teacher.teacher_bp, url_prefix="/teacher")
        client = web.test_client()
        with client.session_transaction() as s:
            s.update(user_id="t-poll", user_role="teacher")
        with patch.object(teacher, "_owned_session", return_value=None):
            codes = [client.get("/teacher/session/x/live.json").status_code for _ in range(31)]
        self.assertNotIn(429, codes[:30])
        self.assertEqual(codes[30], 429)


if __name__ == "__main__":
    unittest.main()

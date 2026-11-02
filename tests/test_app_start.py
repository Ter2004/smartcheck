"""Android app start: straight to the open class; static URLs carry a version for caching."""
import os
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from flask import Flask, url_for

import app
from app.routes import auth, student

APP_UA = {"User-Agent": "Mozilla/5.0 (Linux; Android 16; wv) Chrome/140 Mobile SmartCheckApp"}
BROWSER_UA = {"User-Agent": "Mozilla/5.0 (Linux; Android 16) Chrome/140 Mobile"}


class AppStartTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.enrolled = True
        self.open_ids = ["s1"]
        self.checked = []
        self.stack.enter_context(patch.object(student, "_enrollment_status",
                                              side_effect=lambda: {"is_enrolled": self.enrolled}))
        self.stack.enter_context(patch.object(student, "_open_sessions_today",
                                              side_effect=lambda uid: ([{"id": i} for i in self.open_ids], set(), None)))
        db = MagicMock()
        db.table.return_value.select.return_value.eq.return_value.in_.return_value.execute.side_effect = \
            lambda: MagicMock(data=[{"session_id": i} for i in self.checked])
        self.stack.enter_context(patch.object(student, "supabase_admin", db))
        self.web = Flask(__name__)
        self.web.config.update(SECRET_KEY="t", RATELIMIT_ENABLED=False)
        app.limiter.init_app(self.web)
        self.web.register_blueprint(auth.auth_bp)
        self.web.register_blueprint(student.student_bp, url_prefix="/student")
        self.client = self.web.test_client()
        with self.client.session_transaction() as s:
            s.update(user_id="u1", user_role="student", csrf_token="c")

    def location(self, path, headers):
        response = self.client.get(path, headers=headers)
        self.assertEqual(response.status_code, 302, path)
        return response.headers["Location"]

    def test_opening_the_app_goes_to_the_one_open_class(self):
        self.assertEqual(self.location("/", APP_UA), "/student/start")
        self.assertEqual(self.location("/student/start", APP_UA), "/student/checkin?session_id=s1")

    def test_browser_still_gets_the_dashboard(self):
        self.assertEqual(self.location("/", BROWSER_UA), "/student/dashboard")

    def test_back_to_the_login_page_does_not_jump_again(self):
        # Back from the check-in page lands on GET /login: dashboard, not the class again.
        self.assertEqual(self.location("/login", APP_UA), "/student/dashboard")

    def test_several_open_classes_show_the_list(self):
        self.open_ids = ["s1", "s2"]
        self.assertEqual(self.location("/student/start", APP_UA), "/student/checkin")

    def test_checked_in_or_nothing_open_goes_to_the_dashboard(self):
        self.checked = ["s1"]
        self.assertEqual(self.location("/student/start", APP_UA), "/student/dashboard")
        self.checked, self.open_ids = [], []
        self.assertEqual(self.location("/student/start", APP_UA), "/student/dashboard")

    def test_not_enrolled_goes_to_the_dashboard(self):
        self.enrolled = False
        self.assertEqual(self.location("/student/start", APP_UA), "/student/dashboard")


class StaticVersionTests(unittest.TestCase):
    def test_static_urls_carry_the_file_mtime(self):
        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, "a.js"), "w") as f:
                f.write("1")
            os.utime(os.path.join(folder, "a.js"), (1_800_000_000, 1_800_000_000))
            web = Flask(__name__, static_folder=folder, static_url_path="/static")
            app._cache_static_files(web)
            with web.test_request_context():
                self.assertEqual(url_for("static", filename="a.js"), "/static/a.js?v=1800000000")
                self.assertEqual(url_for("static", filename="missing.js"), "/static/missing.js")
            self.assertEqual(web.config["SEND_FILE_MAX_AGE_DEFAULT"], 7 * 24 * 3600)


if __name__ == "__main__":
    unittest.main()

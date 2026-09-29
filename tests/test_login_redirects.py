import unittest
from pathlib import Path

from flask import Flask
from werkzeug.exceptions import TooManyRequests

import app
from app.routes import admin, auth, student, teacher


class LoginRedirectTests(unittest.TestCase):
    def setUp(self):
        templates = Path(auth.__file__).resolve().parents[1] / "templates"
        self.web = Flask(__name__, template_folder=str(templates))
        self.web.config.update(SECRET_KEY="test", RATELIMIT_ENABLED=False)
        app.limiter.init_app(self.web)
        self.web.register_blueprint(auth.auth_bp)
        self.web.register_blueprint(admin.admin_bp, url_prefix="/admin")
        self.web.register_blueprint(teacher.teacher_bp, url_prefix="/teacher")
        self.web.register_blueprint(student.student_bp, url_prefix="/student")
        self.web.register_error_handler(429, app.rate_limit_exceeded)

        @self.web.route("/admin/limited")
        def limited():
            raise TooManyRequests()

        self.client = self.web.test_client()

    def login(self, role):
        with self.client.session_transaction() as session:
            session.update(user_id="u1", user_role=role)

    def test_logged_in_user_opening_login_goes_to_their_dashboard(self):
        self.login("admin")
        response = self.client.get("/login")
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.location.endswith("/admin/dashboard"))

    def test_anonymous_user_sees_the_login_form(self):
        response = self.client.get("/login")
        self.assertEqual(response.status_code, 200)

    def test_rate_limit_while_logged_in_shows_a_page_not_a_redirect(self):
        self.login("admin")
        response = self.client.get("/admin/limited")
        self.assertEqual(response.status_code, 429)
        self.assertIn("คำขอมากเกินไป", response.get_data(as_text=True))

    def test_rate_limit_while_anonymous_still_goes_to_login(self):
        response = self.client.get("/admin/limited")
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.location.endswith("/login"))


if __name__ == "__main__":
    unittest.main()

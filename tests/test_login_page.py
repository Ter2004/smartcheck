"""Login page: a failed login shows its message once, inline by the form (not also as a toast)."""
import os
import unittest

from flask import Flask, flash, render_template

import app
from app.routes import auth

ROOT = os.path.join(os.path.dirname(__file__), "..", "app")


class LoginMessageTests(unittest.TestCase):
    def setUp(self):
        self.web = Flask("app", template_folder=os.path.join(ROOT, "templates"),
                         static_folder=os.path.join(ROOT, "static"), static_url_path="/static")
        self.web.config.update(SECRET_KEY="t", RATELIMIT_ENABLED=False)
        app.limiter.init_app(self.web)
        self.web.register_blueprint(auth.auth_bp)

    def page(self, endpoint):
        with self.web.test_request_context("/login"):
            from flask import request
            request.url_rule = type("Rule", (), {"endpoint": endpoint})()
            flash("อีเมล / ID หรือรหัสผ่านไม่ถูกต้อง", "danger")
            return render_template("auth/login.html")

    def test_login_error_is_inline_once_and_announced(self):
        html = self.page("auth.login")
        self.assertEqual(html.count("อีเมล / ID หรือรหัสผ่านไม่ถูกต้อง"), 1)
        self.assertIn('role="alert"', html)
        self.assertNotIn('id="flash-data"', html)

    def test_other_pages_still_get_toasts(self):
        self.assertIn('id="flash-data"', self.page("auth.change_password"))


if __name__ == "__main__":
    unittest.main()

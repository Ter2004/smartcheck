"""Per-login lifetimes: students in the app and teachers ~4 months, admin "permanent", the rest 1 hour."""
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cachelib.file import FileSystemCache
from flask import Flask, session
from flask_session import Session

from app.routes import auth
from app.services import session_lifetime
from app.services.session_lifetime import LIFETIME_KEY

CONFIG = {"STUDENT_APP_SESSION_DAYS": 120, "ADMIN_SESSION_DAYS": 3650, "TEACHER_SESSION_DAYS": 120}
APP_UA = "Mozilla/5.0 (Linux; Android 16; wv) Chrome/140 Mobile SmartCheckApp"
BROWSER_UA = "Mozilla/5.0 (Linux; Android 16) Chrome/140 Mobile"


class LoginLifetimeTests(unittest.TestCase):
    """The login route records the lifetime for the role and where the login came from."""

    def setUp(self):
        self.web = Flask(__name__)
        self.web.secret_key = "test"
        self.web.config.update(CONFIG)
        self.backend = Mock()
        self.backend.auth.sign_in_with_password.return_value = SimpleNamespace(
            user=SimpleNamespace(id="user-id"), session=SimpleNamespace(access_token="token"))

    def lifetime_after_login(self, role, user_agent):
        with self.web.test_request_context("/login", method="POST", headers={"User-Agent": user_agent},
                                           data={"email": "someone", "password": "pw"}):
            with patch.object(auth, "supabase", self.backend), \
                    patch.object(auth, "get_user_by_id", return_value={
                        "role": role, "full_name": "Name", "is_active": True}), \
                    patch.object(auth, "_redirect_by_role", return_value="logged-in"):
                auth.login.__wrapped__()
            return session.get(LIFETIME_KEY)

    def test_student_in_the_app_stays_about_a_term(self):
        self.assertEqual(self.lifetime_after_login("student", APP_UA), 120 * 86400)

    def test_admin_stays_logged_in_anywhere(self):
        self.assertEqual(self.lifetime_after_login("admin", BROWSER_UA), 3650 * 86400)
        self.assertEqual(self.lifetime_after_login("admin", APP_UA), 3650 * 86400)

    def test_teachers_stay_about_a_term_anywhere(self):
        # They press "open class" every class; a 1-hour login meant logging in each time.
        self.assertEqual(self.lifetime_after_login("teacher", BROWSER_UA), 120 * 86400)
        self.assertEqual(self.lifetime_after_login("teacher", APP_UA), 120 * 86400)

    def test_browser_students_keep_one_hour(self):
        self.assertIsNone(self.lifetime_after_login("student", BROWSER_UA))


class StoreLifetimeTests(unittest.TestCase):
    """The session store keeps each login, and sets its cookie, for that login's lifetime."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.cache = FileSystemCache(folder.name, threshold=20000)
        self.cache.set = Mock(wraps=self.cache.set)
        self.web = Flask(__name__)
        self.web.config.update(SECRET_KEY="t", SESSION_TYPE="cachelib", SESSION_CACHELIB=self.cache,
                               SESSION_PERMANENT=True, PERMANENT_SESSION_LIFETIME=3600)
        Session(self.web)
        session_lifetime.enable(self.web)
        session_lifetime.enable(self.web)   # twice is harmless

        @self.web.route("/login/<int:seconds>")
        def login(seconds):
            session["user_id"] = "u"
            if seconds:
                session[LIFETIME_KEY] = seconds
            return "ok"

        @self.web.route("/whoami")
        def whoami():
            return session.get("user_id", "")

        self.client = self.web.test_client()

    def stored_timeout(self):
        # cachelib also calls set() for its own file counter; keep the session writes.
        writes = [c.kwargs for c in self.cache.set.call_args_list
                  if str(c.kwargs.get("key", "")).startswith("session:")]
        return writes[-1]["timeout"]

    def cookie_days(self):
        expires = self.client.get_cookie("session").expires
        return (expires - datetime.now(timezone.utc)) / timedelta(days=1)

    def test_long_login_is_stored_and_cookied_for_its_lifetime(self):
        self.client.get(f"/login/{120 * 86400}")
        self.assertEqual(self.stored_timeout(), 120 * 86400)
        self.assertAlmostEqual(self.cookie_days(), 120, delta=0.01)
        # Each later request renews it for the same lifetime.
        self.assertEqual(self.client.get("/whoami").text, "u")
        self.assertEqual(self.stored_timeout(), 120 * 86400)

    def test_login_without_a_lifetime_keeps_one_hour(self):
        self.client.get("/login/0")
        self.assertEqual(self.stored_timeout(), 3600)
        self.assertAlmostEqual(self.cookie_days() * 24, 1, delta=0.01)


if __name__ == "__main__":
    unittest.main()

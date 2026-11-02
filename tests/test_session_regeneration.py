import os
import tempfile
import unittest

from cachelib.file import FileSystemCache
from flask import Flask, session
from flask_session import Session
from flask_sqlalchemy import SQLAlchemy

from app.routes import auth


class SessionRegenerationTests(unittest.TestCase):
    """Runs the real Flask-Session SQLAlchemy backend on in-memory SQLite."""

    def setUp(self):
        self.db = SQLAlchemy()
        self.web = Flask(__name__)
        self.web.config.update(SECRET_KEY="test", SESSION_TYPE="sqlalchemy",
                               SQLALCHEMY_DATABASE_URI="sqlite://",
                               SESSION_SQLALCHEMY=self.db,
                               SESSION_SQLALCHEMY_TABLE="flask_sessions")
        self.db.init_app(self.web)
        with self.web.app_context():
            Session(self.web)
            self.db.create_all()

        @self.web.route("/seed")
        def seed():
            session["pre_login"] = "value"
            return "ok"

        @self.web.route("/login")
        def login():
            auth._regenerate_session()
            session["user_id"] = "user-1"
            return "ok"

        @self.web.route("/whoami")
        def whoami():
            return session.get("user_id", "") + "|" + session.get("pre_login", "")

        self.client = self.web.test_client()

    def cookie(self):
        return self.client.get_cookie("session").value

    def stored_ids(self):
        with self.web.app_context():
            model = self.web.session_interface.sql_session_model
            return {row.session_id for row in self.db.session.query(model).all()}

    def test_login_gets_new_sid_and_old_row_is_deleted(self):
        self.client.get("/seed")
        old_sid = self.cookie()
        self.assertEqual(len(self.stored_ids()), 1)
        self.client.get("/login")
        new_sid = self.cookie()
        self.assertNotEqual(new_sid, old_sid)
        self.assertEqual(len(self.stored_ids()), 1)
        self.assertFalse(any(old_sid in stored for stored in self.stored_ids()))
        self.assertEqual(self.client.get("/whoami").text, "user-1|")

    def test_empty_session_cookie_is_not_reused(self):
        self.client.set_cookie("session", "attacker-chosen-sid")
        self.client.get("/login")
        self.assertNotEqual(self.cookie(), "attacker-chosen-sid")
        self.assertEqual(self.client.get("/whoami").text, "user-1|")


class FileSessionRegenerationTests(SessionRegenerationTests):
    """The same checks on the default store: cachelib files (Config SESSION_STORE=file)."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.cache = FileSystemCache(folder.name, threshold=20000)
        self.web = Flask(__name__)
        self.web.config.update(SECRET_KEY="test", SESSION_TYPE="cachelib", SESSION_CACHELIB=self.cache)
        Session(self.web)

        @self.web.route("/seed")
        def seed():
            session["pre_login"] = "value"
            return "ok"

        @self.web.route("/login")
        def login():
            auth._regenerate_session()
            session["user_id"] = "user-1"
            return "ok"

        @self.web.route("/whoami")
        def whoami():
            return session.get("user_id", "") + "|" + session.get("pre_login", "")

        self.client = self.web.test_client()

    def stored_ids(self):
        return {sid for sid in self._seen if self.cache.has("session:" + sid)}

    def cookie(self):
        sid = self.client.get_cookie("session").value
        self._seen = getattr(self, "_seen", set()) | {sid}
        return sid

    def test_login_gets_new_sid_and_old_row_is_deleted(self):
        self.client.get("/seed")
        old_sid = self.cookie()
        self.assertEqual(self.stored_ids(), {old_sid})
        self.client.get("/login")
        new_sid = self.cookie()
        self.assertNotEqual(new_sid, old_sid)
        self.assertEqual(self.stored_ids(), {new_sid})
        self.assertEqual(self.client.get("/whoami").text, "user-1|")

    def test_config_defaults_to_the_file_store(self):
        from app.config import Config
        if os.getenv("SESSION_STORE", "file") == "file":
            self.assertEqual(Config.SESSION_TYPE, "cachelib")
            self.assertTrue(Config.SESSION_CACHELIB._path.endswith(os.path.join("instance", "sessions")))


if __name__ == "__main__":
    unittest.main()

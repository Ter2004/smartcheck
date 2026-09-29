import unittest

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


if __name__ == "__main__":
    unittest.main()

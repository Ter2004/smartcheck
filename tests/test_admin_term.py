import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from flask import Flask

import app
from app.routes import admin, auth, student, teacher


class Database:
    def __init__(self):
        self.inserts = []

    def table(self, name):
        query = Mock()
        for method in ["select", "order", "limit"]:
            getattr(query, method).return_value = query
        query.insert.side_effect = lambda row: self.inserts.append((name, row)) or query
        query.execute.return_value = SimpleNamespace(data=[])
        return query


class AdminTermTests(unittest.TestCase):
    def setUp(self):
        templates = Path(admin.__file__).resolve().parents[1] / "templates"
        self.web = Flask(__name__, template_folder=str(templates))
        self.web.config.update(SECRET_KEY="test", RATELIMIT_ENABLED=False)
        app.limiter.init_app(self.web)
        self.web.register_blueprint(auth.auth_bp)
        self.web.register_blueprint(admin.admin_bp, url_prefix="/admin")
        self.web.register_blueprint(teacher.teacher_bp, url_prefix="/teacher")
        self.web.register_blueprint(student.student_bp, url_prefix="/student")
        self.db = Database()
        patcher = patch.object(admin, "supabase_admin", self.db)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = self.web.test_client()
        self.login("admin")

    def login(self, role):
        with self.client.session_transaction() as session:
            session.update(user_id="a1", user_role=role, csrf_token="csrf")

    def post(self, csrf="csrf", **form):
        data = {"name": "1/2569", "start_date": "2026-09-01", "weeks": "16", "csrf_token": csrf}
        return self.client.post("/admin/term", data={**data, **form})

    def terms_written(self):
        return [row for table, row in self.db.inserts if table == "terms"]

    def test_page_renders(self):
        self.assertEqual(self.client.get("/admin/term").status_code, 200)

    def test_admin_sets_the_term_and_it_is_audited(self):
        self.assertEqual(self.post().status_code, 302)
        self.assertEqual(self.terms_written(),
                         [{"name": "1/2569", "start_date": "2026-09-01", "weeks": 16}])
        self.assertIn("term_set", [row["event_type"] for t, row in self.db.inserts if t == "audit_logs"])

    def test_invalid_input_writes_nothing(self):
        for form in ({"start_date": "not-a-date"}, {"weeks": "0"}, {"weeks": "31"}, {"name": " "}):
            with self.subTest(form=form):
                self.post(**form)
        self.assertEqual(self.terms_written(), [])

    def test_csrf_and_role_are_required(self):
        self.assertEqual(self.post(csrf="wrong").status_code, 403)
        for role in ("teacher", "student"):
            self.login(role)
            self.post()
        self.assertEqual(self.terms_written(), [])


if __name__ == "__main__":
    unittest.main()

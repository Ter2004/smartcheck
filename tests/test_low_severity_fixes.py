"""Phase 6: override scope, logout CSRF, error leaks, date filters, session contents."""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from flask import Flask, session

import app
from app.routes import admin, auth, student, teacher


class Query:
    def __init__(self, db, name):
        self.db, self.name = db, name

    def __getattr__(self, method):
        def call(*args, **kwargs):
            self.db.calls.append((self.name, method, args))
            if method in ("insert", "update"):
                self.db.writes.append((self.name, method, args[0]))
            if self.db.fail_on == (self.name, method):
                raise RuntimeError("secret internal detail: relation xyz")
            return self
        return call

    def execute(self):
        return SimpleNamespace(data=self.db.data.get(self.name))


class Database:
    def __init__(self, **data):
        self.data, self.calls, self.writes, self.fail_on = data, [], [], None

    def table(self, name):
        return Query(self, name)


def web_app():
    templates = Path(auth.__file__).resolve().parents[1] / "templates"
    web = Flask(__name__, template_folder=str(templates))
    web.config.update(SECRET_KEY="test", RATELIMIT_ENABLED=False)
    web.jinja_env.filters["thai_time"] = lambda value, fmt=None: str(value)
    app.limiter.init_app(web)
    for bp, prefix in ((auth.auth_bp, None), (admin.admin_bp, "/admin"),
                       (teacher.teacher_bp, "/teacher"), (student.student_bp, "/student")):
        web.register_blueprint(bp, url_prefix=prefix)
    return web


class LowSeverityTests(unittest.TestCase):
    def setUp(self):
        self.web = web_app()
        self.client = self.web.test_client()

    def login(self, role, uid="t1"):
        with self.client.session_transaction() as s:
            s.update(user_id=uid, user_role=role, csrf_token="csrf")

    def use(self, module, db):
        patcher = patch.object(module, "supabase_admin", db)
        patcher.start()
        self.addCleanup(patcher.stop)
        return db

    # 1. Override only for students of the session's course
    def override(self, enrolled):
        self.login("teacher")
        db = self.use(teacher, Database(
            sessions={"course_id": "c1", "courses": {"teacher_id": "t1"}},
            course_enrollments=[{"id": "e1"}] if enrolled else [],
            attendance=None))
        self.client.post("/teacher/session/s1/override",
                         data={"student_id": "u9", "status": "present", "csrf_token": "csrf"})
        return [w for w in db.writes if w[0] == "attendance"]

    def test_override_refuses_a_student_outside_the_course(self):
        self.assertEqual(self.override(enrolled=False), [])

    def test_override_still_works_for_an_enrolled_student(self):
        self.assertEqual(len(self.override(enrolled=True)), 1)

    # 2. Logout needs POST + CSRF
    def test_logout_get_does_not_log_out(self):
        self.login("student")
        self.assertEqual(self.client.get("/logout").status_code, 302)
        with self.client.session_transaction() as s:
            self.assertIn("user_id", s)

    def test_logout_post_without_token_is_refused(self):
        self.login("student")
        self.assertEqual(self.client.post("/logout").status_code, 403)
        with self.client.session_transaction() as s:
            self.assertIn("user_id", s)

    def test_logout_post_with_token_clears_the_session(self):
        self.login("student")
        response = self.client.post("/logout", data={"csrf_token": "csrf"})
        self.assertTrue(response.location.endswith("/login"))
        with self.client.session_transaction() as s:
            self.assertNotIn("user_id", s)

    def test_logout_after_expiry_just_returns_to_login(self):
        self.assertEqual(self.client.post("/logout").status_code, 302)

    def test_sidebar_logs_out_with_a_form(self):
        base = (Path(auth.__file__).resolve().parents[1] / "templates/base.html").read_text(encoding="utf-8")
        self.assertIn("<form method=\"POST\" action=\"{{ url_for('auth.logout') }}\"", base)
        self.assertNotIn("<a href=\"{{ url_for('auth.logout') }}\"", base)

    # 3. No internal error text to the client
    def test_reset_errors_do_not_leak_details(self):
        for module, role, prefix in ((admin, "admin", "/admin"), (teacher, "teacher", "/teacher")):
            with self.subTest(role=role):
                self.login(role)
                db = self.use(module, Database(courses=[{"id": "c1"}], course_enrollments=[{"id": "e1"}]))
                db.fail_on = ("student_biometrics", "update")
                response = self.client.post(f"{prefix}/api/reset-enrollment/u1",
                                            headers={"X-CSRF-Token": "csrf"})
                self.assertEqual(response.status_code, 500)
                self.assertNotIn("secret internal detail", response.get_data(as_text=True))

    # 5. Date filters: Bangkok dates, malformed input ignored
    def test_history_dates_are_validated_and_use_bangkok_time(self):
        self.login("teacher")
        db = self.use(teacher, Database(courses=[], sessions=[]))
        self.assertEqual(self.client.get("/teacher/history?date_from=x'&date_to=2026-13-40").status_code, 200)
        self.assertFalse([c for c in db.calls if c[1] in ("gte", "lte")])
        self.client.get("/teacher/history?date_from=2026-09-01&date_to=2026-09-30")
        filters = {c[1]: c[2][1] for c in db.calls if c[1] in ("gte", "lte")}
        self.assertEqual(filters, {"gte": "2026-09-01T00:00:00+07:00", "lte": "2026-09-30T23:59:59+07:00"})

    # 4. The unused Supabase access token is not kept in the session
    def test_login_does_not_store_the_supabase_access_token(self):
        backend = Mock()
        backend.auth.sign_in_with_password.return_value = SimpleNamespace(
            user=SimpleNamespace(id="u1"), session=SimpleNamespace(access_token="token"))
        with self.web.test_request_context("/login", method="POST",
                                           data={"email": "a@b.c", "password": "pw"}):
            with patch.object(auth, "supabase", backend), \
                    patch.object(auth, "get_user_by_id", return_value={
                        "role": "student", "full_name": "S", "is_active": True}):
                auth.login.__wrapped__()
            self.assertEqual(session["user_id"], "u1")
            self.assertNotIn("access_token", session)


if __name__ == "__main__":
    unittest.main()

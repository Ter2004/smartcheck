"""maybe_single().execute() returns None (not a response) when no row matches.

postgrest 0.19 does this, so reading .data straight off it crashed the
teacher's manual check-in for any student who had not checked in yet, and the
course CSV import for any new student.
"""
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from flask import Flask

import app
from app.routes import admin, auth, student, teacher


class Query:
    def __init__(self, db, name):
        self.db, self.name = db, name
        self.single = False
        self.write = None

    def maybe_single(self):
        self.single = True
        return self

    def insert(self, row):
        self.write = ("insert", row)
        return self

    def update(self, row):
        self.write = ("update", row)
        return self

    def __getattr__(self, name):     # select / eq / limit ... are no-ops here
        return lambda *args, **kwargs: self

    def execute(self):
        if self.write:
            self.db.writes.append((self.name,) + self.write)
            return SimpleNamespace(data=[self.write[1]])
        rows = self.db.rows.get(self.name, [])
        if self.single:
            return SimpleNamespace(data=rows[0]) if rows else None   # real postgrest behaviour
        return SimpleNamespace(data=rows)


class Database:
    def __init__(self, rows):
        self.rows, self.writes = rows, []
        user = SimpleNamespace(user=SimpleNamespace(id="new-student"))
        self.auth = SimpleNamespace(admin=SimpleNamespace(create_user=lambda data: user))

    def table(self, name):
        return Query(self, name)


class MissingRowTests(unittest.TestCase):
    def setUp(self):
        templates = Path(teacher.__file__).resolve().parents[1] / "templates"
        self.web = Flask(__name__, template_folder=str(templates))
        self.web.config.update(SECRET_KEY="test", RATELIMIT_ENABLED=False)
        app.limiter.init_app(self.web)
        for bp, prefix in ((auth.auth_bp, None), (admin.admin_bp, "/admin"),
                           (teacher.teacher_bp, "/teacher"), (student.student_bp, "/student")):
            self.web.register_blueprint(bp, url_prefix=prefix)
        self.client = self.web.test_client()

    def login(self, user_id, role):
        with self.client.session_transaction() as session:
            session.update(user_id=user_id, user_role=role, csrf_token="tok")

    def test_manual_check_in_for_student_without_attendance_row(self):
        self.login("t1", "teacher")
        db = Database({
            "sessions": [{"course_id": "c1", "courses": {"teacher_id": "t1"}}],
            "course_enrollments": [{"id": "e1"}],
            "attendance": [],                                  # never checked in
        })
        with patch.object(teacher, "supabase_admin", db), \
             patch.object(teacher, "log_audit_event") as audit:
            response = self.client.post("/teacher/session/s1/override", data={
                "csrf_token": "tok", "student_id": "u1", "status": "manual", "reason": "phone broke"})

        self.assertEqual(response.status_code, 302)
        [(table, op, row)] = db.writes
        self.assertEqual((table, op), ("attendance", "insert"))
        self.assertEqual((row["student_id"], row["status"], row["override_by"]), ("u1", "manual", "t1"))
        self.assertIsNone(audit.call_args.kwargs["old_value"])

    def test_override_of_missing_session_redirects_instead_of_crashing(self):
        self.login("t1", "teacher")
        db = Database({"sessions": []})
        with patch.object(teacher, "supabase_admin", db):
            response = self.client.post("/teacher/session/gone/override", data={
                "csrf_token": "tok", "student_id": "u1", "status": "present"})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(db.writes, [])

    def test_course_csv_import_creates_and_enrolls_a_new_student(self):
        self.login("a1", "admin")
        db = Database({"users": [], "course_enrollments": []})
        csv_file = (io.BytesIO("email,full_name,student_id\nnew@x.th,New Student,6630000001\n".encode()), "s.csv")
        with patch.object(admin, "supabase_admin", db):
            response = self.client.post("/admin/courses/c1/import-csv", data={
                "csrf_token": "tok", "csv_file": csv_file}, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 302)
        self.assertEqual([(t, op) for t, op, _ in db.writes],
                         [("users", "insert"), ("student_biometrics", "insert"), ("course_enrollments", "insert")])
        self.assertEqual(db.writes[-1][2], {"course_id": "c1", "student_id": "new-student"})


if __name__ == "__main__":
    unittest.main()

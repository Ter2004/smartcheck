import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from flask import Flask

import app
from app.routes import admin, auth, student, teacher

NAME = "x'); alert(1);//"


class Query:
    def __init__(self, data, calls):
        self.data, self.calls = data, calls

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args))
            return self
        return call

    def execute(self):
        return SimpleNamespace(data=self.data)


class Storage:
    def __init__(self):
        self.signed = []

    def from_(self, bucket):
        assert bucket == "face-images"
        return self

    def create_signed_urls(self, paths, expires_in):
        self.signed.append((sorted(paths), expires_in))
        return [{"path": p, "signedURL": f"https://x.supabase.co/sign/{p}?token=t", "error": None}
                for p in paths]


class Database:
    def __init__(self, teacher_id="t1"):
        self.calls, self.storage = [], Storage()
        self.teacher_id = teacher_id

    def table(self, name):
        data = {
            "sessions": {"id": "s1", "course_id": "c1", "title": "T", "start_time": "2026-09-29T03:30:00+00:00",
                         "is_open": False, "courses": {"id": "c1", "code": "TEST01", "name": "n",
                                                       "teacher_id": self.teacher_id},
                         "beacons": {"room_name": "TEST-101"}},
            "attendance": [],
            "course_enrollments": [
                {"student_id": "u1", "users": {"id": "u1", "full_name": NAME, "student_id": "001", "email": "a"}},
                {"student_id": "u2", "users": {"id": "u2", "full_name": "no photo", "student_id": "002", "email": "b"}},
            ],
            "student_biometrics": [{"user_id": "u1", "face_image_url": "u1.jpg"},
                                   {"user_id": "u2", "face_image_url": None}],
        }[name]
        return Query(data, self.calls)


class TeacherPhotoTests(unittest.TestCase):
    def setUp(self):
        templates = Path(teacher.__file__).resolve().parents[1] / "templates"
        self.web = Flask(__name__, template_folder=str(templates))
        self.web.config.update(SECRET_KEY="test", RATELIMIT_ENABLED=False)
        self.web.jinja_env.filters["thai_time"] = lambda value, fmt=None: str(value)  # create_app's filter
        app.limiter.init_app(self.web)
        for bp, prefix in ((auth.auth_bp, None), (admin.admin_bp, "/admin"),
                           (teacher.teacher_bp, "/teacher"), (student.student_bp, "/student")):
            self.web.register_blueprint(bp, url_prefix=prefix)
        self.client = self.web.test_client()
        with self.client.session_transaction() as session:
            session.update(user_id="t1", user_role="teacher")

    def view(self, db):
        with patch.object(teacher, "supabase_admin", db):
            return self.client.get("/teacher/session/s1")

    def test_enrolled_students_show_a_short_lived_photo(self):
        db = Database()
        page = self.view(db).get_data(as_text=True)
        self.assertIn("https://x.supabase.co/sign/u1.jpg?token=t", page)
        self.assertIn("ยังไม่ได้ลงทะเบียนใบหน้า", page)             # u2 placeholder
        self.assertEqual(db.storage.signed, [(["u1.jpg"], teacher.PHOTO_URL_TTL_S)])
        self.assertLessEqual(teacher.PHOTO_URL_TTL_S, 600)

    def test_other_teachers_session_signs_nothing(self):
        db = Database(teacher_id="t2")
        self.assertEqual(self.view(db).status_code, 302)
        self.assertEqual(db.storage.signed, [])

    def test_student_name_cannot_break_out_of_the_override_handler(self):
        page = self.view(Database()).get_data(as_text=True)
        self.assertIn('onclick="openOverride(this.dataset.id, this.dataset.name)"', page)
        self.assertNotIn("alert(1);//')", page)

    def test_storage_failure_hides_photos_instead_of_failing(self):
        db = Database()
        db.storage.create_signed_urls = lambda *a: (_ for _ in ()).throw(RuntimeError("down"))
        response = self.view(db)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("supabase.co/sign", response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()

import csv
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from flask import Flask

import app
from app.routes import teacher

ATTENDANCE = [{
    "users": {"student_id": "6600000001", "full_name": "=HYPERLINK(\"http://x\")",
              "email": "@student.example"},
    "status": "manual", "check_in_at": "2026-09-29T08:00:00+00:00",
    "face_score": 0.91, "ble_rssi": -60, "liveness_action": "head_turn",
    "override_reason": "+cmd|' /C calc'!A0",
}]


class Database:
    def table(self, name):
        query = Mock()
        for method in ["select", "eq", "order", "maybe_single"]:
            getattr(query, method).return_value = query
        data = {"sessions": {"title": "t", "start_time": "2026-09-29T08:00:00+00:00",
                             "courses": {"code": "DE101", "name": "n", "teacher_id": "teacher-1"}},
                "attendance": ATTENDANCE}[name]
        query.execute.return_value = SimpleNamespace(data=data)
        return query


class TeacherExportTests(unittest.TestCase):
    def setUp(self):
        self.web = Flask(__name__)
        self.web.config.update(SECRET_KEY="test", RATELIMIT_ENABLED=False)
        app.limiter.init_app(self.web)
        self.web.register_blueprint(teacher.teacher_bp, url_prefix="/teacher")
        self.client = self.web.test_client()
        with self.client.session_transaction() as session:
            session.update(user_id="teacher-1", user_role="teacher")
        patcher = patch.object(teacher, "supabase_admin", Database())
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_formula_cells_are_escaped_and_numbers_kept(self):
        response = self.client.get("/teacher/session/s1/export")
        self.assertEqual(response.status_code, 200)
        rows = list(csv.reader(io.StringIO(response.get_data(as_text=True).lstrip(chr(0xFEFF)))))
        row = dict(zip(rows[0], rows[1]))
        self.assertEqual(row["full_name"], "'=HYPERLINK(\"http://x\")")
        self.assertEqual(row["email"], "'@student.example")
        self.assertEqual(row["override_reason"], "'+cmd|' /C calc'!A0")
        self.assertEqual(row["ble_rssi"], "-60")
        self.assertEqual(row["student_id"], "6600000001")

    def test_csv_safe_leaves_plain_text(self):
        self.assertEqual(teacher._csv_safe("Somchai"), "Somchai")
        self.assertEqual(teacher._csv_safe(-60), -60)
        self.assertEqual(teacher._csv_safe("-60"), "'-60")


if __name__ == "__main__":
    unittest.main()

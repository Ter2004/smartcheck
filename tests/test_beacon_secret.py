import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from flask import Flask

import app
from app.routes import admin, auth, student, teacher

STORED_SECRET = "ab" * 32
BEACON = {"id": "b1", "uuid": "u", "major": 1, "minor": 1, "room_name": "TEST-101",
          "ble_room_code": "TEST-101", "is_active": True,
          "ble_secret": STORED_SECRET}


class Database:
    """Answers the beacons/audit_logs queries these routes make; records writes."""

    def __init__(self):
        self.updates, self.inserts = [], []

    def table(self, name):
        query = Mock()
        for method in ["select", "eq", "order", "maybe_single"]:
            getattr(query, method).return_value = query
        query.update.side_effect = lambda row: self.updates.append((name, row)) or query
        query.insert.side_effect = lambda row: self.inserts.append((name, row)) or query
        rows = {"beacons": [dict(BEACON)]}.get(name, [])
        query.maybe_single.side_effect = lambda: (setattr(
            query.execute, "return_value", SimpleNamespace(data=rows[0] if rows else None)) or query)
        query.execute.return_value = SimpleNamespace(data=rows)
        return query


class BeaconSecretTests(unittest.TestCase):
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
            session.update(user_id="admin-1", user_role=role, csrf_token="csrf")

    def rotate(self, csrf="csrf"):
        return self.client.post("/admin/beacons/b1/ble-secret", data={"csrf_token": csrf})

    def test_list_page_never_contains_the_stored_secret(self):
        response = self.client.get("/admin/beacons")
        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertNotIn(STORED_SECRET, page)
        self.assertIn("ตั้งแล้ว", page)

    def test_rotation_stores_and_shows_a_new_64_hex_key_once(self):
        response = self.rotate()
        self.assertEqual(response.status_code, 200)
        (table, row), = self.db.updates
        self.assertEqual(table, "beacons")
        self.assertRegex(row["ble_secret"], r"^[0-9a-f]{64}$")
        self.assertNotEqual(row["ble_secret"], STORED_SECRET)
        self.assertIn(f'ROOM_SECRET_HEX[] = "{row["ble_secret"]}"', response.get_data(as_text=True))
        self.assertEqual(response.headers["Cache-Control"], "no-store")

    def test_audit_log_records_rotation_without_the_key(self):
        self.rotate()
        secret = self.db.updates[0][1]["ble_secret"]
        audits = [row for table, row in self.db.inserts if table == "audit_logs"]
        self.assertEqual(len(audits), 1)
        self.assertEqual(audits[0]["event_type"], "beacon_secret_rotated")
        self.assertNotIn(secret, repr(audits[0]))

    def test_rotation_requires_csrf(self):
        self.assertEqual(self.rotate(csrf="wrong").status_code, 403)
        self.assertEqual(self.db.updates, [])

    def test_only_admin_can_rotate(self):
        for role in ("teacher", "student"):
            self.login(role)
            response = self.rotate()
            self.assertEqual(response.status_code, 302)
        self.assertEqual(self.db.updates, [])

    def test_migration_constrains_the_key_format(self):
        sql = (Path(admin.__file__).resolve().parents[2]
               / "database/migrations/20260929_beacon_ble_secret.sql").read_text(encoding="utf-8")
        self.assertTrue(re.search(r"ble_secret ~ '\^\[0-9a-f\]\{64\}\$'", sql))


if __name__ == "__main__":
    unittest.main()

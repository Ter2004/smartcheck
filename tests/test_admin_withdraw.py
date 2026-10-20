"""Admin PDPA withdrawal on a student's behalf: every face record and photo goes, in order."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from flask import Flask

import app
from app.routes import admin, auth
from app.services import biometric_wipe

HEADERS = {"X-CSRF-Token": "csrf"}


class FakeDb:
    def __init__(self, role="student", fail=None):
        self.calls, self.role, self.fail = [], role, fail
        self.bucket = Mock()
        self.bucket.remove.side_effect = self.remove
        self.storage = SimpleNamespace(from_=lambda name: self.bucket)

    def remove(self, paths):
        if self.fail == "photos":
            raise RuntimeError("storage down")
        self.calls.append(("remove", tuple(paths)))

    def table(self, name):
        q = Mock()
        for m in ("select", "eq", "maybe_single", "in_", "update", "insert", "delete"):
            getattr(q, m).return_value = q

        def execute():
            if self.fail == name:
                raise RuntimeError(f"{name} down")
            if name == "users":
                return SimpleNamespace(data={"id": "s1", "role": self.role})
            return SimpleNamespace(data=[])
        q.execute.side_effect = execute
        q.insert.side_effect = lambda row: (self.calls.append(("insert", name, row)), q)[1]
        q.update.side_effect = lambda row: (self.calls.append(("update", name, row)), q)[1]
        q.delete.side_effect = lambda: (self.calls.append(("delete", name)), q)[1]
        return q


class AdminWithdrawTests(unittest.TestCase):
    def setUp(self):
        self.web = Flask(__name__)
        self.web.config.update(SECRET_KEY="t", RATELIMIT_ENABLED=False)
        app.limiter.init_app(self.web)
        self.web.register_blueprint(auth.auth_bp)
        self.web.register_blueprint(admin.admin_bp, url_prefix="/admin")
        self.client = self.web.test_client()
        with self.client.session_transaction() as s:
            s.update(user_id="admin-1", user_role="admin", csrf_token="csrf")
        self.audits = []
        p = patch("app.services.security_service.log_audit_event",
                  side_effect=lambda *a, **k: self.audits.append(k))
        p.start()
        self.addCleanup(p.stop)

    def post(self, db):
        with patch.object(admin, "supabase_admin", db):
            return self.client.post("/admin/api/withdraw-biometrics/s1", json={}, headers=HEADERS)

    def test_full_wipe_in_order(self):
        db = FakeDb()
        r = self.post(db)
        self.assertEqual(r.status_code, 200, r.json)
        kinds = [c[0] + ":" + (c[1] if len(c) > 1 and isinstance(c[1], str) else "") for c in db.calls]
        self.assertEqual(kinds, ["insert:consent_logs", "update:student_biometrics", "delete:face_enrollments", "remove:"])
        consent = db.calls[0][2]
        self.assertFalse(consent["consent_given"])
        self.assertTrue(consent["user_agent"].startswith("admin:admin-1"))
        self.assertIsNone(db.calls[1][2]["face_embeddings"])
        self.assertEqual(db.calls[3][1], ("s1.jpg", "v2/s1.jpg"))
        self.assertEqual(self.audits[0]["event_type"], "consent_withdrawn_data_deleted_by_admin")

    def test_failure_is_reported_with_the_step(self):
        for step in ("consent_logs", "student_biometrics", "face_enrollments", "photos"):
            with self.subTest(step=step):
                r = self.post(FakeDb(fail=step))
                self.assertEqual(r.status_code, 500)
                self.assertIn({"consent_logs": "consent_log", "student_biometrics": "legacy_biometrics",
                               "face_enrollments": "face_enrollments", "photos": "photos"}[step], r.json["message"])

    def test_missing_v2_table_is_not_a_failure(self):
        db = FakeDb()
        with patch.object(biometric_wipe, "_missing_table", return_value=True), \
                patch("app.services.face_store.delete_enrollment", side_effect=RuntimeError("PGRST205")):
            r = self.post(db)
        self.assertEqual(r.status_code, 200, r.json)

    def test_only_students(self):
        self.assertEqual(self.post(FakeDb(role="teacher")).status_code, 404)

    def test_non_admin_is_refused(self):
        with self.client.session_transaction() as s:
            s.update(user_role="student")
        r = self.post(FakeDb())
        self.assertIn(r.status_code, (302, 403))


if __name__ == "__main__":
    unittest.main()

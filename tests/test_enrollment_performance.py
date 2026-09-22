import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from flask import Flask, render_template_string

import app
from app.routes import student
from app.services.request_performance import init_request_performance


class EnrollmentPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.web = Flask(__name__)
        self.web.config.update(SECRET_KEY="test", RATELIMIT_ENABLED=False, ENROLL_FLOW_MODE="classic")
        init_request_performance(self.web)
        app.limiter.init_app(self.web)
        self.web.register_blueprint(student.student_bp, url_prefix="/student")
        self.client = self.web.test_client()
        with self.client.session_transaction() as session:
            session.update(user_id="student", user_role="student")
        self.db = Mock()
        self.query = self.db.table.return_value
        for method in ("select", "eq", "maybe_single"):
            getattr(self.query, method).return_value = self.query
        self.query.execute.return_value = SimpleNamespace(data=None)
        self.db.rpc.return_value.execute.return_value = SimpleNamespace(data=[])
        self.db_patch = patch.object(student, "supabase_admin", self.db)
        self.db_patch.start()
        self.addCleanup(self.db_patch.stop)
        # Exercise the real blueprint context processor, without unrelated UI.
        self.render_patch = patch.object(student, "render_template", side_effect=
            lambda *args, **kwargs: render_template_string("{{ can_enroll_face }}", **kwargs))
        self.render_patch.start()
        self.addCleanup(self.render_patch.stop)

    def test_baseline_and_cached_query_counts(self):
        for cache, endpoint, expected in (
            (False, "enroll", 3), (False, "dashboard", 2),
            (True, "enroll", 1), (True, "dashboard", 1),
        ):
            with self.subTest(cache=cache, endpoint=endpoint):
                self.web.config["ENROLLMENT_STATUS_CACHE"] = cache
                self.query.execute.reset_mock()
                with self.assertLogs("smartcheck.performance", level="INFO") as logs:
                    response = self.client.get("/student/" + endpoint)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(self.query.execute.call_count, expected)
                self.assertIn("status_queries=" + str(expected), logs.output[0])
                self.assertIn("response_ms=", logs.output[0])

    def test_rpc_never_fetches_embeddings_and_preserves_guard(self):
        self.web.config["ENROLLMENT_STATUS_RPC"] = True
        response = self.client.get("/student/enroll")
        self.assertEqual(response.status_code, 200)
        self.db.rpc.assert_called_once_with("get_enrollment_status", {"p_user_id": "student"})
        self.db.table.assert_not_called()
        # A second request observes enrollment completed elsewhere.
        self.db.rpc.return_value.execute.return_value = SimpleNamespace(
            data=[dict(is_enrolled=True, baseline_ear=0.3)])
        response = self.client.post("/student/api/enroll", json={})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json["status"], "already_enrolled")

    def test_rpc_failure_and_invalid_boolean_fail_closed(self):
        self.web.config["ENROLLMENT_STATUS_RPC"] = True
        execute = self.db.rpc.return_value.execute
        execute.side_effect = RuntimeError("offline")
        self.assertEqual(self.client.post("/student/api/enroll").status_code, 503)
        execute.side_effect = None
        execute.return_value = SimpleNamespace(data=[{"is_enrolled": "false"}])
        self.assertEqual(self.client.post("/student/api/enroll").status_code, 503)
        self.db.table.assert_not_called()

    def test_no_cross_request_cache_after_withdrawal(self):
        self.query.execute.return_value = SimpleNamespace(
            data={"face_embeddings": [[1]], "consent_given": True})
        self.assertEqual(self.client.get("/student/dashboard").data, b"False")
        self.query.execute.return_value = SimpleNamespace(
            data={"face_embeddings": None, "consent_given": False})
        self.assertEqual(self.client.get("/student/dashboard").data, b"True")
        self.assertEqual(self.query.execute.call_count, 2)

    def test_pages_show_retryable_503_without_requery_or_redirect(self):
        for rpc in (False, True):
            self.web.config["ENROLLMENT_STATUS_RPC"] = rpc
            execute = (self.db.rpc.return_value if rpc else self.query).execute
            execute.side_effect = RuntimeError("private database error")
            for page in ("dashboard", "enroll", "checkin"):
                with self.subTest(rpc=rpc, page=page):
                    execute.reset_mock()
                    response = self.client.get("/student/" + page)
                    self.assertEqual(response.status_code, 503)
                    self.assertEqual(response.mimetype, "text/html")
                    self.assertEqual(response.headers["Cache-Control"], "no-store")
                    self.assertIn("ลองใหม่", response.get_data(as_text=True))
                    self.assertNotIn("private database error", response.get_data(as_text=True))
                    self.assertNotIn("Location", response.headers)
                    execute.assert_called_once()

    def test_malformed_rpc_result_returns_503_on_pages(self):
        self.web.config["ENROLLMENT_STATUS_RPC"] = True
        self.db.rpc.return_value.execute.return_value = SimpleNamespace(
            data=[{"is_enrolled": "false"}])
        for page in ("dashboard", "enroll", "checkin"):
            with self.subTest(page=page):
                self.assertEqual(self.client.get("/student/" + page).status_code, 503)

    def test_failed_request_timing_and_context_isolation(self):
        self.query.execute.side_effect = RuntimeError("offline")
        with self.assertLogs("smartcheck.performance", level="INFO") as logs:
            self.assertEqual(self.client.post("/student/api/enroll").status_code, 503)
            self.assertEqual(self.client.get("/missing").status_code, 404)
        self.assertIn("status_queries=1", logs.output[0])
        self.assertIn("status_queries=0", logs.output[1])


if __name__ == "__main__":
    unittest.main()

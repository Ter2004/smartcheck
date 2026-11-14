"""Closed accounts are logged out; a lost database connection answers without looping."""
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from flask import Flask

import app
from app.routes import admin, auth, student, teacher
from app.services import broadcast_beacon


class Users:
    """supabase_admin stand-in: one users row (or none), counting the reads."""
    def __init__(self, row):
        self.row, self.reads, self.fail = row, 0, False

    def table(self, name):
        assert name == "users"
        query = Mock()
        for method in ["select", "eq", "maybe_single"]:
            getattr(query, method).return_value = query

        def execute():
            self.reads += 1
            if self.fail:
                raise RuntimeError("database unreachable")
            return SimpleNamespace(data=self.row) if self.row else None
        query.execute.side_effect = execute
        return query


def make_web():
    templates = Path(auth.__file__).resolve().parents[1] / "templates"
    web = Flask(__name__, template_folder=str(templates))
    web.config.update(SECRET_KEY="test", RATELIMIT_ENABLED=False)
    app.limiter.init_app(web)
    web.register_blueprint(auth.auth_bp)
    web.register_blueprint(admin.admin_bp, url_prefix="/admin")
    web.register_blueprint(teacher.teacher_bp, url_prefix="/teacher")
    web.register_blueprint(student.student_bp, url_prefix="/student")

    @web.route("/page")
    def page():
        return "ok"

    @web.route("/api/thing", methods=["POST"])
    def thing():
        return "ok"
    return web


class AccountRecheckTests(unittest.TestCase):
    def setUp(self):
        self.users = Users({"is_active": True, "role": "teacher"})
        patcher = patch.object(auth, "supabase_admin", self.users)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = make_web().test_client()

    def login(self, checked_ago=None):
        with self.client.session_transaction() as session:
            session.update(user_id="u1", user_role="teacher")
            if checked_ago is not None:
                session[auth.ACTIVE_CHECKED_KEY] = time.time() - checked_ago

    def logged_in(self):
        with self.client.session_transaction() as session:
            return "user_id" in session

    def test_recent_check_reads_nothing(self):
        self.login(checked_ago=10)
        self.assertEqual(self.client.get("/page").status_code, 200)
        self.assertEqual(self.users.reads, 0)

    def test_login_from_before_this_check_starts_the_clock_without_a_read(self):
        self.login()
        self.assertEqual(self.client.get("/page").status_code, 200)
        self.assertEqual(self.users.reads, 0)
        with self.client.session_transaction() as session:
            self.assertIn(auth.ACTIVE_CHECKED_KEY, session)

    def test_active_account_is_read_once_per_interval(self):
        self.login(checked_ago=auth.ACTIVE_RECHECK_S + 1)
        self.assertEqual(self.client.get("/page").status_code, 200)
        self.assertEqual(self.client.get("/page").status_code, 200)
        self.assertEqual(self.users.reads, 1)
        self.assertTrue(self.logged_in())

    def test_closed_deleted_or_changed_role_is_logged_out(self):
        for row in [{"is_active": False, "role": "teacher"}, None, {"is_active": True, "role": "student"}]:
            self.users.row = row
            self.login(checked_ago=auth.ACTIVE_RECHECK_S + 1)
            response = self.client.get("/page")
            self.assertEqual(response.status_code, 302, row)
            self.assertTrue(response.location.endswith("/login"))
            self.assertFalse(self.logged_in(), row)

    def test_api_request_of_a_closed_account_gets_json_401(self):
        self.users.row = {"is_active": False, "role": "teacher"}
        self.login(checked_ago=auth.ACTIVE_RECHECK_S + 1)
        response = self.client.post("/api/thing", json={})
        self.assertEqual(response.status_code, 401)
        self.assertFalse(response.json["ok"])

    def test_unreachable_database_keeps_the_login_and_asks_again(self):
        self.users.fail = True
        self.login(checked_ago=auth.ACTIVE_RECHECK_S + 1)
        self.assertEqual(self.client.get("/page").status_code, 200)
        self.assertEqual(self.client.get("/page").status_code, 200)
        self.assertEqual(self.users.reads, 2)
        self.assertTrue(self.logged_in())


class ConnectionLostTests(unittest.TestCase):
    def setUp(self):
        refresh = patch.object(app, "_refresh_clients")
        refresh.start()
        self.addCleanup(refresh.stop)
        web = Flask(__name__)
        web.config.update(SECRET_KEY="test")
        web.register_error_handler(httpx.RemoteProtocolError, app.connection_lost)

        @web.route("/page")
        def page():
            raise httpx.RemoteProtocolError("Server disconnected")

        @web.route("/api/checkin", methods=["POST"])
        def checkin():
            raise httpx.RemoteProtocolError("Server disconnected")

        @web.route("/student/today.json")
        def today():
            raise httpx.RemoteProtocolError("Server disconnected")
        self.client = web.test_client()

    def test_page_is_tried_once_more_then_shows_a_retry_page(self):
        first = self.client.get("/page?x=1")
        self.assertEqual(first.status_code, 302)
        self.assertIn("x=1", first.location)
        self.assertIn("_retry=1", first.location)
        second = self.client.get(first.location)
        self.assertEqual(second.status_code, 503)
        self.assertIn("ลองใหม่", second.get_data(as_text=True))

    def test_post_and_json_requests_get_503_not_a_redirect(self):
        for response in [self.client.post("/api/checkin", json={}), self.client.get("/student/today.json")]:
            self.assertEqual(response.status_code, 503)
            self.assertFalse(response.json["ok"])
            self.assertTrue(response.json["error"])


class FirstSightingTests(unittest.TestCase):
    """fresh() uses the row the write returns (a first sighting) instead of reading it back."""
    def run_fresh(self, returned, stored=None):
        reads = []
        client = Mock()

        def table(_name):
            query = Mock()
            for method in ["select", "eq", "order", "limit", "maybe_single"]:
                getattr(query, method).return_value = query
            state = {"kind": "first"}
            query.order.side_effect = lambda *a, **k: state.update(kind="newest") or query
            query.upsert.side_effect = lambda *a, **k: state.update(kind="write") or query

            def execute():
                reads.append(state["kind"])
                if state["kind"] == "write":
                    return SimpleNamespace(data=returned)
                if state["kind"] == "newest":
                    return SimpleNamespace(data=[{"counter": 100}])
                return SimpleNamespace(data=stored)
            query.execute.side_effect = execute
            return query
        client.table.side_effect = table
        from datetime import datetime, timezone
        now = datetime(2026, 11, 14, 9, 0, tzinfo=timezone.utc)
        return broadcast_beacon.fresh(client, "b1", 100, now=now), sorted(reads), now

    def test_first_sighting_needs_no_read_back(self):
        from datetime import datetime, timezone
        now = datetime(2026, 11, 14, 9, 0, tzinfo=timezone.utc).isoformat()
        ok, reads, _ = self.run_fresh([{"beacon_id": "b1", "counter": 100, "first_seen_at": now}])
        self.assertTrue(ok)
        self.assertEqual(reads, ["newest", "write"])

    def test_seen_before_is_read_back_and_judged_by_its_first_time(self):
        ok, reads, _ = self.run_fresh([], stored={"first_seen_at": "2026-11-14T08:59:50+00:00"})
        self.assertTrue(ok)
        self.assertEqual(reads, ["first", "newest", "write"])
        ok, _, _ = self.run_fresh([], stored={"first_seen_at": "2026-11-14T08:59:00+00:00"})
        self.assertFalse(ok)   # first seen 60 s ago: a replay

    def test_a_returned_old_row_is_still_judged_by_its_time(self):
        ok, _, _ = self.run_fresh([{"first_seen_at": "2026-11-14T08:59:00+00:00"}])
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()

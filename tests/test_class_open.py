"""Teacher opens the class: on-time window, automatic open after 15 minutes, present/late."""
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

from flask import Flask

import app
from app import scheduler
from app.routes import teacher
from app.services import class_open

CONFIG = {"SESSION_OPEN_MODE": "teacher", "CLASS_ON_TIME_MINUTES": 5, "CLASS_ON_TIME_MAX_MINUTES": 60,
          "CLASS_AUTO_OPEN_AFTER_MINUTES": 15, "CLASS_EARLY_OPEN_MINUTES": 15}
START = datetime(2026, 11, 3, 1, 0, tzinfo=timezone.utc)    # 08:00 Bangkok
END = START + timedelta(hours=3)


def at(minutes):
    return START + timedelta(minutes=minutes)


class ScheduledStateTests(unittest.TestCase):
    def state(self, sess, minutes, **config):
        return class_open.scheduled_state(sess, START, END, at(minutes), {**CONFIG, **config})

    def test_waits_for_the_teacher_until_fifteen_minutes_after_the_start(self):
        for minutes in (-20, 0, 14):
            with self.subTest(minutes=minutes):
                self.assertEqual(self.state({}, minutes), ({"is_open": False, "end_time": None}, False))

    def test_opens_by_itself_after_fifteen_minutes_with_the_default_window(self):
        fields, auto = self.state({}, 15)
        self.assertTrue(auto)
        self.assertEqual(fields, {"is_open": True, "end_time": None,
                                  "opened_at": at(15).isoformat(), "present_minutes": 5})

    def test_a_class_the_teacher_opened_stays_open_until_the_end(self):
        opened = {"is_open": True, "opened_at": at(-10).isoformat()}
        self.assertEqual(self.state(opened, 1), ({"is_open": True, "end_time": None}, False))
        self.assertEqual(self.state(opened, 180), ({"is_open": False, "end_time": END.isoformat()}, False))

    def test_a_class_open_from_before_this_mode_is_not_closed(self):
        self.assertEqual(self.state({"is_open": True}, 3), ({"is_open": True, "end_time": None}, False))

    def test_schedule_mode_keeps_the_old_rule(self):
        for minutes, is_open in [(-1, False), (0, True), (179, True), (180, False)]:
            with self.subTest(minutes=minutes):
                fields, auto = self.state({}, minutes, SESSION_OPEN_MODE="schedule")
                self.assertEqual(fields["is_open"], is_open)
                self.assertNotIn("opened_at", fields)
                self.assertFalse(auto)


class StatusTests(unittest.TestCase):
    def test_on_time_inside_the_window_late_after_it(self):
        sess = {"start_time": START.isoformat(), "opened_at": at(-5).isoformat(), "present_minutes": 10}
        self.assertEqual(class_open.checkin_status(sess, at(5), CONFIG), "present")
        self.assertEqual(class_open.checkin_status(sess, at(5) + timedelta(seconds=1), CONFIG), "late")

    def test_missing_window_uses_the_default(self):
        sess = {"start_time": START.isoformat(), "opened_at": at(20).isoformat(), "present_minutes": None}
        self.assertEqual(class_open.checkin_status(sess, at(25), CONFIG), "present")
        self.assertEqual(class_open.checkin_status(sess, at(26), CONFIG), "late")

    def test_sessions_without_opened_at_keep_the_fifteen_minute_rule(self):
        sess = {"start_time": START.isoformat()}
        self.assertEqual(class_open.checkin_status(sess, at(15), CONFIG), "present")
        self.assertEqual(class_open.checkin_status(sess, at(16), CONFIG), "late")


class RefusalTests(unittest.TestCase):
    def sess(self, **fields):
        return {"start_time": START.isoformat(), "is_open": False, "end_time": None, **fields}

    def test_opening_window(self):
        self.assertIsNotNone(class_open.open_refusal(self.sess(), at(-16), CONFIG))
        self.assertIsNone(class_open.open_refusal(self.sess(), at(-15), CONFIG))
        self.assertIsNone(class_open.open_refusal(self.sess(), at(60), CONFIG))
        self.assertIsNotNone(class_open.open_refusal(self.sess(), at(13 * 60), CONFIG))

    def test_open_or_finished_classes_are_refused(self):
        self.assertEqual(class_open.open_refusal(self.sess(is_open=True), at(0), CONFIG), "คาบนี้เปิดอยู่แล้ว")
        self.assertEqual(class_open.open_refusal(self.sess(end_time=END.isoformat()), at(0), CONFIG),
                         "คาบนี้จบไปแล้ว")

    def test_minutes(self):
        for text, minutes in [("5", 5), (" 1 ", 1), ("60", 60), ("0", None), ("61", None), ("x", None), (None, None)]:
            with self.subTest(text=text):
                self.assertEqual(class_open.parse_minutes(text, CONFIG), minutes)


class SchedulerTickTests(unittest.TestCase):
    """auto_manage_sessions in teacher mode: create closed, open by itself only after 15 minutes."""

    def tick(self, minutes, existing):
        db = Mock()
        self.queries = []

        def table(name):
            q = Mock()
            for method in ["select", "eq", "limit", "insert", "update", "is_"]:
                getattr(q, method).return_value = q
            data = {"schedules": [dict(course_id="c", day_of_week=1, start_time="08:00:00", end_time="11:00:00",
                                       courses=dict(id="c", code="T1", name="T", is_active=True))],
                    "beacons": [dict(id="b")]}.get(name, existing)
            q.execute.return_value = SimpleNamespace(data=data)
            self.queries.append((name, q))
            return q
        db.table.side_effect = table
        with patch.object(scheduler, "_close_expired_sessions"), \
                patch.object(scheduler, "_get_config", return_value=CONFIG), \
                patch.object(scheduler, "_get_supabase", return_value=db), \
                patch.object(scheduler, "datetime") as clock:
            clock.now.return_value = at(minutes)
            scheduler.auto_manage_sessions()

    def written(self, method):
        return [(getattr(q, method).call_args.args[0], q) for name, q in self.queries
                if name == "sessions" and getattr(q, method).called]

    def test_created_closed_before_the_teacher_presses(self):
        self.tick(0, existing=[])
        (row, _), = self.written("insert")
        self.assertFalse(row["is_open"])
        self.assertNotIn("opened_at", row)

    def test_opened_by_itself_after_fifteen_minutes_only_if_still_unopened(self):
        self.tick(15, existing=[dict(id="s", is_open=False, end_time=None, opened_at=None)])
        (changes, query), = self.written("update")
        self.assertEqual(changes["is_open"], True)
        self.assertEqual(changes["present_minutes"], 5)
        query.is_.assert_called_once_with("opened_at", "null")   # a teacher press in between wins

    def test_a_teacher_opened_class_is_left_alone(self):
        self.tick(5, existing=[dict(id="s", is_open=True, end_time=None, opened_at=at(-3).isoformat())])
        self.assertEqual(self.written("update"), [])


class TeacherOpenRouteTests(unittest.TestCase):
    def setUp(self):
        self.web = Flask(__name__)
        self.web.config.update(SECRET_KEY="t", RATELIMIT_ENABLED=False, **CONFIG)
        app.limiter.init_app(self.web)
        self.web.register_blueprint(teacher.teacher_bp, url_prefix="/teacher")
        self.client = self.web.test_client()
        with self.client.session_transaction() as s:
            s.update(user_id="teacher-1", user_role="teacher", csrf_token="c")
        self.row = {"id": "s1", "start_time": START.isoformat(), "is_open": False, "end_time": None,
                    "courses": {"code": "T1", "name": "T", "teacher_id": "teacher-1"}}
        self.db = Mock()
        q = self.db.table.return_value
        for method in ["select", "eq", "maybe_single", "update", "is_"]:
            getattr(q, method).return_value = q
        q.execute.side_effect = lambda: SimpleNamespace(data=self.row if not q.update.called else [{"id": "s1"}])
        self.query = q
        for target, value in [("supabase_admin", self.db), ("log_audit_event", Mock())]:
            patcher = patch.object(teacher, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def press(self, minutes="10", now=at(-5)):
        with patch.object(teacher, "datetime") as clock:
            clock.now.return_value = now
            return self.client.post("/teacher/session/s1/open", data={"csrf_token": "c", "minutes": minutes})

    def test_opens_with_the_chosen_minutes(self):
        self.assertEqual(self.press().status_code, 302)
        changes = self.query.update.call_args.args[0]
        self.assertEqual(changes, {"is_open": True, "opened_at": at(-5).isoformat(),
                                   "opened_by": "teacher-1", "present_minutes": 10})
        self.query.is_.assert_called_once_with("end_time", "null")

    def test_refuses_too_early_bad_minutes_and_other_teachers(self):
        self.press(now=at(-30))
        self.press(minutes="0")
        self.row["courses"]["teacher_id"] = "someone-else"
        self.press()
        self.query.update.assert_not_called()

    def test_needs_the_csrf_token(self):
        response = self.client.post("/teacher/session/s1/open", data={"minutes": "5"})
        self.assertEqual(response.status_code, 403)

    def test_schedule_mode_has_no_button(self):
        self.web.config["SESSION_OPEN_MODE"] = "schedule"
        self.assertEqual(self.press().status_code, 404)


if __name__ == "__main__":
    unittest.main()

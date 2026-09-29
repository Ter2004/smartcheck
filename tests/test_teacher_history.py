import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from flask import Flask

import app
from app.routes import teacher

OWN_COURSE = {"id": "own-course", "code": "DE101", "name": "Mine", "section": "1"}


class Database:
    """Answers courses/sessions/attendance and records each sessions .in_() filter."""

    def __init__(self):
        self.session_filters = []

    def table(self, name):
        query = Mock()
        for method in ["select", "eq", "order", "gte", "lte", "limit"]:
            getattr(query, method).return_value = query

        def in_(column, values):
            if name == "sessions" and column == "course_id":
                self.session_filters.append(list(values))
            return query
        query.in_.side_effect = in_
        data = {"courses": [OWN_COURSE],
                "sessions": [{"id": "s1", "course_id": "own-course"}],
                "attendance": []}[name]
        query.execute.return_value = SimpleNamespace(data=data)
        return query


class TeacherHistoryTests(unittest.TestCase):
    def setUp(self):
        self.web = Flask(__name__)
        self.web.config.update(SECRET_KEY="test", RATELIMIT_ENABLED=False)
        app.limiter.init_app(self.web)
        self.web.register_blueprint(teacher.teacher_bp, url_prefix="/teacher")
        self.client = self.web.test_client()
        with self.client.session_transaction() as session:
            session.update(user_id="teacher-1", user_role="teacher")
        self.db = Database()
        for target, value in [("supabase_admin", self.db),
                              ("render_template", Mock(return_value="page"))]:
            patcher = patch.object(teacher, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def rendered(self):
        return teacher.render_template.call_args.kwargs

    def test_other_teachers_course_is_not_queried(self):
        response = self.client.get("/teacher/history?course_id=someone-elses-course")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.db.session_filters, [["own-course"]])
        self.assertEqual(self.rendered()["course_filter"], "")

    def test_own_course_filter_still_applies(self):
        self.client.get("/teacher/history?course_id=own-course")
        self.assertEqual(self.db.session_filters, [["own-course"]])
        self.assertEqual(self.rendered()["course_filter"], "own-course")

    def test_no_filter_lists_all_own_courses(self):
        self.client.get("/teacher/history")
        self.assertEqual(self.db.session_filters, [["own-course"]])


if __name__ == "__main__":
    unittest.main()

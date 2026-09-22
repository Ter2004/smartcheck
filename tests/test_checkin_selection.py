import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from flask import Flask
from app.routes import student


class CheckinSelectionTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.secret_key = 'test'
        self.app.add_url_rule('/student/checkin', endpoint='student.checkin', view_func=lambda: '')
        self.sessions = [
            {'id': 'first', 'course_id': 'course-a'},
            {'id': 'second', 'course_id': 'course-b'},
            {'id': 'foreign', 'course_id': 'other'},
        ]

    def invoke(self, query=''):
        def table(name):
            data = {'sessions': self.sessions,
                    'course_enrollments': [{'course_id': 'course-a'}, {'course_id': 'course-b'}],
                    'attendance': [], 'schedules': []}[name]
            builder = Mock()
            for method in ['select', 'eq', 'gte', 'lt', 'lte', 'in_', 'order', 'limit']:
                getattr(builder, method).return_value = builder
            builder.execute.return_value = SimpleNamespace(data=data)
            return builder
        handler = student.checkin
        while hasattr(handler, '__wrapped__'):
            handler = handler.__wrapped__
        with self.app.test_request_context('/student/checkin' + query):
            from flask import session
            session['user_id'] = 'student'
            with patch.object(student, '_enrollment_status', return_value={'is_enrolled': True}), \
                 patch.object(student, 'supabase_admin') as database, \
                 patch.object(student, 'render_template', side_effect=lambda name, **kwargs: (name, kwargs)):
                database.table.side_effect = table
                return handler()

    def test_first_page_is_selection_without_auto_selected_session(self):
        template, context = self.invoke()
        self.assertEqual(template, 'student/checkin_select.html')
        self.assertEqual([s['id'] for s in context['available_sessions']], ['first', 'second'])

    def test_second_session_is_selected_explicitly(self):
        template, context = self.invoke('?session_id=second')
        self.assertEqual(template, 'student/checkin.html')
        self.assertEqual(context['session_data']['id'], 'second')
        self.assertEqual(context['week_schedules'], [])

    def test_unavailable_or_unenrolled_session_returns_to_selection(self):
        for selected in ['foreign', 'closed', 'invalid']:
            with self.subTest(selected=selected):
                response = self.invoke('?session_id=' + selected)
                self.assertEqual(response.status_code, 302)
                self.assertEqual(response.location, '/student/checkin')

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch
from flask import Flask
import app
from app import scheduler
from app.routes import student, teacher


class EnrollmentTests(unittest.TestCase):
    def setUp(self):
        self.web = Flask(__name__)
        self.web.config.update(SECRET_KEY='test', RATELIMIT_ENABLED=False)
        app.limiter.init_app(self.web)
        self.web.register_blueprint(student.student_bp, url_prefix='/student')
        self.web.register_blueprint(teacher.teacher_bp, url_prefix='/teacher')
        self.client = self.web.test_client()
        with self.client.session_transaction() as session:
            session.update(user_id='student', user_role='student')
        self.db = Mock()
        q = self.db.table.return_value
        q.select.return_value = q
        q.eq.return_value = q
        q.maybe_single.return_value = q
        q.execute.return_value = SimpleNamespace(data={'face_embeddings': [[1]], 'consent_given': True})
        self.patch = patch.object(student, 'supabase_admin', self.db)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_completed_page_redirects(self):
        r = self.client.get('/student/enroll')
        self.assertEqual(r.status_code, 302)
        self.assertTrue(r.location.endswith('/student/dashboard'))

    def test_completed_apis_cannot_mutate(self):
        for endpoint in ['enroll', 'self_verify', 'consent', 'spoof_check', 'reset-liveness']:
            with self.subTest(endpoint=endpoint):
                r = self.client.post('/student/api/' + endpoint, json={})
                self.assertEqual(r.status_code, 409)
                self.assertEqual(r.json['status'], 'already_enrolled')
        self.db.table.return_value.upsert.assert_not_called()
        self.db.table.return_value.update.assert_not_called()

    def test_first_enrollment_can_continue_to_validation(self):
        self.db.table.return_value.execute.return_value = SimpleNamespace(data=None)
        with self.web.test_request_context('/student/api/enroll', method='POST'):
            from flask import session
            session.update(user_id='student', user_role='student')
            self.assertIsNone(student.prevent_completed_enrollment())

    def test_database_failure_blocks_enrollment(self):
        self.db.table.return_value.execute.side_effect = RuntimeError('offline')
        self.assertEqual(self.client.post('/student/api/enroll').status_code, 503)

    def test_checkin_without_completed_biometrics_redirects_to_enrollment(self):
        for result in [None, SimpleNamespace(data=None),
                       SimpleNamespace(data={}),
                       SimpleNamespace(data={'face_embeddings': [], 'consent_given': True}),
                       SimpleNamespace(data={'face_embeddings': [[1]], 'consent_given': False})]:
            with self.subTest(result=result):
                self.db.table.return_value.execute.return_value = result
                response = self.client.get('/student/checkin')
                self.assertEqual(response.status_code, 302)
                self.assertTrue(response.location.endswith('/student/enroll'))

    def test_checkin_with_completed_biometrics_renders(self):
        query = self.db.table.return_value
        for method in ['gte', 'lt']:
            getattr(query, method).return_value = query
        query.execute.side_effect = [
            SimpleNamespace(data={'face_embeddings': [[1]], 'consent_given': True,
                                  'baseline_ear': 0.3}),
            SimpleNamespace(data=[]),  # Open sessions
            SimpleNamespace(data=[]),  # Course enrollments
        ]
        with patch.object(student, 'render_template', return_value='checkin') as render:
            response = self.client.get('/student/checkin')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b'checkin')
        self.assertEqual(render.call_args.args[0], 'student/checkin.html')
        self.assertEqual(render.call_args.kwargs['baseline_ear'], 0.3)

    def test_manual_teacher_endpoints_removed(self):
        with self.client.session_transaction() as session:
            session.update(user_role='teacher')
        for path in ['/teacher/session/create', '/teacher/session/example/toggle']:
            self.assertIn(self.client.post(path).status_code, (404, 405))


class SchedulerTests(unittest.TestCase):
    def run_tick(self, hour, minute=0, exists=True):
        db = Mock()
        calls = []
        def table(name):
            q = Mock()
            for method in ['select', 'eq', 'limit', 'insert', 'update']:
                getattr(q, method).return_value = q
            if name == 'schedules':
                data = [dict(course_id='c', day_of_week=0, start_time='12:00:00', end_time='22:00:00',
                             courses=dict(id='c', code='TEST1', name='Test', is_active=True))]
            elif name == 'beacons':
                data = [dict(id='b')]
            else:
                data = [dict(id='s', is_open=False, end_time=None)] if exists else []
            q.execute.return_value = SimpleNamespace(data=data)
            calls.append((name, q))
            return q
        db.table.side_effect = table
        now = datetime(2026, 9, 21, hour, minute, tzinfo=scheduler.TZ_THAI).astimezone(timezone.utc)
        with patch.object(scheduler, '_get_supabase', return_value=db), patch.object(scheduler, 'datetime') as clock:
            clock.now.return_value = now
            scheduler.auto_manage_sessions()
        return calls

    def test_existing_session_boundaries(self):
        for hour, minute, expected in [(11,59,None),(12,0,True),(21,59,True),(22,0,False)]:
            with self.subTest(hour=hour,minute=minute):
                calls = self.run_tick(hour, minute)
                updates = [q.update.call_args.args[0] for name,q in calls if q.update.called]
                if expected is None:
                    self.assertEqual(updates, [])
                else:
                    self.assertEqual(updates[0]['is_open'], expected)
                    self.assertNotIn('start_time', updates[0])

    def test_created_session_is_open_only_during_schedule(self):
        for hour, expected in [(11,False),(12,True),(22,False)]:
            calls = self.run_tick(hour, exists=False)
            rows = [q.insert.call_args.args[0] for name,q in calls if q.insert.called]
            self.assertEqual(rows[0]['is_open'], expected)
            self.assertEqual(rows[0]['start_time'], '2026-09-21T05:00:00+00:00')


if __name__ == '__main__':
    unittest.main()

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from flask import Flask
import app
from app.routes import student
from app.services import security_service


class WithdrawalTests(unittest.TestCase):
    def setUp(self):
        self.web = Flask(__name__)
        self.web.config.update(SECRET_KEY='test', RATELIMIT_ENABLED=False)
        self.web.add_url_rule('/login', endpoint='auth.login', view_func=lambda: 'login')
        app.limiter.init_app(self.web)
        self.web.register_blueprint(student.student_bp, url_prefix='/student')
        self.client = self.web.test_client()
        self.biometric = dict(face_embeddings=[[1, 0]], face_image_url='student.jpg',
                              consent_given=True)
        self.objects = {'student.jpg', 'other-student.jpg'}
        self.events = []
        self.fail_table = None
        self.fail_storage = False
        self.db = Mock()
        self.db.table.side_effect = self.table
        self.bucket = self.db.storage.from_.return_value
        self.bucket.remove.side_effect = self.remove
        self.db_patch = patch.object(student, 'supabase_admin', self.db)
        self.db_patch.start()
        self.addCleanup(self.db_patch.stop)
        self.audit_patch = patch.object(security_service, 'log_audit_event')
        self.audit = self.audit_patch.start()
        self.addCleanup(self.audit_patch.stop)
        self.sensitive_keys = ('consent_given_at', 'consent_ip', 'liveness_embeddings',
                               'enroll_baseline_ear', 'enroll_retry', 'spoof_check_acc')
        with self.client.session_transaction() as sess:
            sess.update(user_id='student', user_role='student', csrf_token='csrf')
            for key in self.sensitive_keys:
                sess[key] = 'old-state'

    def table(self, name):
        query = Mock()
        query.insert.return_value = query
        query.update.return_value = query
        query.eq.return_value = query

        def execute():
            self.events.append(name)
            if self.fail_table == name:
                raise RuntimeError('database unavailable')
            if name == 'student_biometrics':
                query.eq.assert_called_with('user_id', 'student')
                self.biometric.update(query.update.call_args.args[0])
            return SimpleNamespace(data=[])

        query.execute.side_effect = execute
        return query

    def remove(self, paths):
        self.events.append('storage')
        if self.fail_storage:
            raise RuntimeError('storage unavailable')
        removed = self.objects.intersection(paths)
        self.objects.difference_update(paths)
        return [{'name': name} for name in removed]

    def post(self, **kwargs):
        return self.client.post('/student/api/withdraw-consent',
                                headers={'X-CSRF-Token': 'csrf'}, **kwargs)

    def test_withdrawal_removes_only_own_photo_after_revoking_biometrics(self):
        response = self.post(json={'face_image_url': 'other-student.jpg', 'user_id': 'other-student'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('face_photo', response.json['deleted'])
        self.assertEqual(self.events, ['consent_logs', 'student_biometrics', 'storage'])
        self.assertEqual(self.objects, {'other-student.jpg'})
        self.assertFalse(self.biometric['consent_given'])
        self.assertIsNone(self.biometric['face_embeddings'])
        self.assertIsNone(self.biometric['face_image_url'])
        self.db.storage.from_.assert_called_once_with('face-images')
        self.audit.assert_called_once()
        with self.client.session_transaction() as sess:
            for key in self.sensitive_keys:
                self.assertNotIn(key, sess)

    def test_storage_failure_is_reported_and_can_retry_after_url_is_cleared(self):
        self.fail_storage = True
        response = self.post()
        self.assertEqual(response.status_code, 503)
        self.assertTrue(response.json['consent_withdrawn'])
        self.assertTrue(response.json['photo_cleanup_pending'])
        self.assertFalse(self.biometric['consent_given'])
        self.assertIsNone(self.biometric['face_image_url'])
        self.assertIn('student.jpg', self.objects)
        self.audit.assert_not_called()
        with self.client.session_transaction() as sess:
            self.assertNotIn('liveness_embeddings', sess)
        self.fail_storage = False
        self.assertEqual(self.post().status_code, 200)
        self.assertNotIn('student.jpg', self.objects)

    def test_no_photo_and_repeated_withdrawal_are_successful(self):
        self.objects.remove('student.jpg')
        self.biometric['face_image_url'] = None
        for _ in range(2):
            self.assertEqual(self.post().status_code, 200)
        self.assertEqual(self.objects, {'other-student.jpg'})

    def test_orphan_photo_is_removed_even_without_database_url(self):
        self.biometric['face_image_url'] = None
        self.assertEqual(self.post().status_code, 200)
        self.assertNotIn('student.jpg', self.objects)

    def test_audit_insert_failure_does_not_delete_data(self):
        self.fail_table = 'consent_logs'
        self.assertEqual(self.post().status_code, 500)
        self.assertEqual(self.events, ['consent_logs'])
        self.bucket.remove.assert_not_called()
        self.assertTrue(self.biometric['consent_given'])

    def test_biometrics_failure_does_not_report_success(self):
        self.fail_table = 'student_biometrics'
        with self.assertLogs(self.web.logger, level='ERROR'):
            self.assertEqual(self.post().status_code, 500)
        self.bucket.remove.assert_not_called()
        self.audit.assert_not_called()

    def test_missing_csrf_cannot_remove_photo(self):
        response = self.client.post('/student/api/withdraw-consent')
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.events, [])

    def test_non_student_cannot_remove_photo(self):
        with self.client.session_transaction() as sess:
            sess['user_role'] = 'teacher'
        response = self.post()
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.location, '/login')
        self.assertEqual(self.events, [])

    def test_logged_out_request_cannot_remove_photo(self):
        with self.client.session_transaction() as sess:
            sess.clear()
        self.assertEqual(self.post().status_code, 302)
        self.assertEqual(self.events, [])


if __name__ == '__main__':
    unittest.main()

import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from flask import Flask
import app
from app.routes import student
from app.services import face_service, liveness_challenge

HEADERS = {'X-CSRF-Token': 'csrf'}
FRAME = 'data:image/jpeg;base64,AAAA'
VERIFY_BODY = {'nonce': 'n-1', 'before': FRAME, 'action_frames': [FRAME, FRAME], 'after': FRAME}


def table_mock(data):
    q = Mock()
    for name in ('select', 'eq', 'order', 'limit', 'maybe_single'):
        getattr(q, name).return_value = q
    q.execute.return_value = SimpleNamespace(data=data)
    return q


class LivenessRouteTests(unittest.TestCase):
    def setUp(self):
        self.web = Flask(__name__)
        self.web.config.update(SECRET_KEY='test', RATELIMIT_ENABLED=False, ENROLL_FLOW_MODE='classic')
        app.limiter.init_app(self.web)
        self.web.register_blueprint(student.student_bp, url_prefix='/student')
        self.client = self.web.test_client()
        with self.client.session_transaction() as s:
            s.update(user_id='student', user_role='student', csrf_token='csrf',
                     consent_given_at='2026-09-24T00:00:00Z')
        self.db = Mock()
        tables = {'student_biometrics': table_mock(None),              # not enrolled yet
                  'consent_logs': table_mock([{'consent_given': True}])}
        self.db.table.side_effect = lambda name: tables[name]
        for target in (patch.object(student, 'supabase_admin', self.db),
                       patch.object(face_service, 'server_validate_frame',
                                    return_value={'valid': True, 'reason': 'passed', 'metadata': {}}),
                       patch.object(liveness_challenge, 'decode_frame', return_value=object())):
            target.start()
            self.addCleanup(target.stop)

    def session(self):
        with self.client.session_transaction() as s:
            return dict(s)

    def post(self, path, body=None):
        return self.client.post('/student/api/' + path, json=body or {}, headers=HEADERS)

    def test_challenge_issues_turn_order_and_resets_liveness(self):
        with self.client.session_transaction() as s:
            s.update(liveness_embeddings=[[1.0]], liveness_verified_at=time.time())
        r = self.post('liveness/challenge')
        self.assertEqual(r.status_code, 200)
        self.assertCountEqual(r.json['actions'], ['turn_left', 'turn_right'])
        stored = self.session()
        self.assertEqual(stored['liveness_challenge']['nonce'], r.json['nonce'])
        self.assertNotIn('liveness_embeddings', stored)
        self.assertNotIn('liveness_verified_at', stored)

    def test_verify_pass_sets_continuity_reference(self):
        self.post('liveness/challenge')
        with patch.object(liveness_challenge, 'verify',
                          return_value={'passed': True, 'reason': 'passed', 'yaws': [], 'scores': {}}), \
                patch.object(face_service, 'extract_embedding', return_value=[0.5] * 512):
            r = self.post('liveness/verify', VERIFY_BODY)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json['passed'])
        stored = self.session()
        self.assertEqual(len(stored['liveness_embeddings']), 2)
        self.assertAlmostEqual(stored['liveness_verified_at'], time.time(), delta=5)
        self.assertNotIn('liveness_challenge', stored)

    def test_verify_failure_consumes_challenge(self):
        self.post('liveness/challenge')
        seen = []

        def fake_verify(challenge, *args):
            seen.append(challenge)
            return {'passed': False, 'reason': 'wrong_direction:action_1:frontal', 'yaws': [], 'scores': {}}
        with patch.object(liveness_challenge, 'verify', side_effect=fake_verify):
            first = self.post('liveness/verify', VERIFY_BODY)
            second = self.post('liveness/verify', VERIFY_BODY)
        self.assertEqual((first.status_code, second.status_code), (400, 400))
        self.assertIsNotNone(seen[0])
        self.assertIsNone(seen[1])  # the retry has no challenge left to reuse
        self.assertNotIn('liveness_verified_at', self.session())

    def test_verify_rejects_malformed_frames(self):
        self.post('liveness/challenge')
        with patch.object(liveness_challenge, 'verify') as verify:
            for body in ({}, dict(VERIFY_BODY, action_frames='x'),
                         dict(VERIFY_BODY, action_frames=[FRAME] * 5)):
                self.assertEqual(self.post('liveness/verify', body).status_code, 400)
        verify.assert_not_called()

    def test_enroll_requires_recent_verified_challenge_before_counting_attempt(self):
        body = {'face_images': [FRAME] * 5}
        for verified_at in (None, time.time() - student.LIVENESS_MAX_AGE_S - 1):
            with self.subTest(verified_at=verified_at):
                with self.client.session_transaction() as s:
                    s['liveness_verified_at'] = verified_at
                r = self.post('enroll', body)
                self.assertEqual(r.status_code, 400)
                self.assertIn('Liveness', r.json['message'])
        self.db.rpc.assert_not_called()

    def test_self_verify_requires_verified_challenge(self):
        r = self.post('self_verify', {'face_image': FRAME})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json['status'], 'continuity_fail')

    def test_spoof_check_only_extends_reference_with_the_verified_face(self):
        same, other = [1.0] + [0.0] * 511, [0.0, 1.0] + [0.0] * 510
        real = {'is_real': True, 'confidence': 0.9, 'message': '', 'layers': {}}
        with self.client.session_transaction() as s:
            s.update(liveness_embeddings=[same], liveness_verified_at=time.time())
        with patch.object(face_service, 'spoof_check_with_embedding',
                          side_effect=[dict(real, embedding=other), dict(real, embedding=same)]), \
                patch.object(face_service, '_decode_image', return_value=__import__('numpy').zeros((120, 160, 3), 'uint8')):
            self.post('spoof_check', {'image': FRAME})
            self.assertEqual(len(self.session()['liveness_embeddings']), 1)  # other face ignored
            self.post('spoof_check', {'image': FRAME})
            self.assertEqual(len(self.session()['liveness_embeddings']), 2)

    def test_spoof_check_before_challenge_adds_nothing(self):
        real = {'is_real': True, 'confidence': 0.9, 'message': '', 'layers': {}, 'embedding': [1.0] * 512}
        with patch.object(face_service, 'spoof_check_with_embedding', return_value=real), \
                patch.object(face_service, '_decode_image', return_value=__import__('numpy').zeros((120, 160, 3), 'uint8')):
            self.post('spoof_check', {'image': FRAME})
        self.assertNotIn('liveness_embeddings', self.session())

    def test_completed_students_cannot_use_challenge_endpoints(self):
        self.db.table.side_effect = lambda name: table_mock({'face_embeddings': [[1]], 'consent_given': True})
        for path in ('liveness/challenge', 'liveness/verify'):
            self.assertEqual(self.post(path, VERIFY_BODY).status_code, 409)


if __name__ == '__main__':
    unittest.main()

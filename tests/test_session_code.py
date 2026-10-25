"""Teacher-screen room code (CHECKIN_SESSION_CODE): codes, QR, check-in API, teacher screen, re-enroll banner, LINE unlink."""
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from flask import Flask

import app
from app.routes import admin, api_checkin as route, auth, student, teacher
from app.services import session_code
from tests.test_checkin_totp import Database

SECRET = "r" * 64
T = 1_789_999_980.0          # a fixed clock at the start of a 30-second step (T % 30 == 0)


class CodeTests(unittest.TestCase):
    def test_current_and_previous_step_only(self):
        code, left = session_code.current(SECRET, "s1", now=T + 5)
        self.assertRegex(code, r"^\d{6}$")
        self.assertEqual(left, 25)
        self.assertTrue(session_code.verify(SECRET, "s1", code, now=T + 5))
        self.assertTrue(session_code.verify(SECRET, "s1", code, now=T + 35))     # just changed: still ok
        self.assertFalse(session_code.verify(SECRET, "s1", code, now=T + 65))    # two steps later
        self.assertFalse(session_code.verify(SECRET, "s2", code, now=T + 5))     # another session
        self.assertFalse(session_code.verify("x" * 64, "s1", code, now=T + 5))   # another secret
        for bad in (None, "", "12345", "1234567", "abcdef", 123456):
            self.assertFalse(session_code.verify(SECRET, "s1", bad, now=T + 5))

    def test_qr_is_inline_svg(self):
        svg = session_code.qr_svg("https://example.test/student/checkin?session_id=s1&via=code&c=123456")
        self.assertTrue(svg.lstrip().startswith("<svg"))
        self.assertNotIn("<script", svg)


class ProximityCodeTests(unittest.TestCase):
    """POST /api/checkin/proximity then /api/checkin with proximity_via=code (BLE mode server)."""

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.db = Database()
        self.stack.enter_context(patch.object(route, 'supabase_admin', self.db))
        for name, result in {'server_validate_frame': {'valid': True}, 'combined_spoof_score': {'is_real': True},
                             '_verify_face_v2': (1.0, None)}.items():
            self.stack.enter_context(patch.object(route, name, return_value=result))
        self.stack.enter_context(patch.object(route, '_decode_image',
                                              side_effect=lambda image: np.full((64, 64, 3), 20, dtype=np.uint8)))
        self.web = Flask(__name__)
        self.web.config.update(SECRET_KEY='test', PROXIMITY_RECEIPT_SECRET=SECRET, EMBEDDING_INTEGRITY_SALT='salt',
                               RATELIMIT_ENABLED=False, CHECKIN_PROXIMITY_METHOD='ble', CHECKIN_LIVENESS='passive',
                               CHECKIN_SESSION_CODE=True)
        app.limiter.init_app(self.web)
        self.web.register_blueprint(auth.auth_bp)
        self.web.register_blueprint(route.api_checkin_bp)
        self.client = self.web.test_client()
        with self.client.session_transaction() as s:
            s.update(user_id='student', user_role='student', csrf_token='csrf')

    def proximity(self, code, via='code'):
        return self.client.post('/api/checkin/proximity', headers={'X-CSRF-Token': 'csrf'}, json={
            'session_id': 'session', 'room_code': '', 'proximity_via': via, 'session_code': code})

    def checkin(self, receipt, via='code'):
        return self.client.post('/api/checkin', headers={'X-CSRF-Token': 'csrf'}, json={
            'session_id': 'session', 'room_code': '', 'proximity_receipt': receipt, 'proximity_via': via,
            'liveness_action': 'passive', 'face_image': '20', 'face_images': ['20', '20']})

    def test_valid_code_gets_a_receipt_and_checks_in(self):
        code, _ = session_code.current(SECRET, 'session')
        r = self.proximity(code)
        self.assertEqual(r.status_code, 200, r.json)
        self.assertEqual(self.checkin(r.json['proximity_receipt']).status_code, 200)
        self.assertEqual(len(self.db.records), 1)

    def test_wrong_code_or_other_session_is_refused(self):
        self.assertEqual(self.proximity('000000' if session_code.current(SECRET, 'session')[0] != '000000' else '111111').status_code, 400)
        other, _ = session_code.current(SECRET, 'another-session')
        if other != session_code.current(SECRET, 'session')[0]:
            self.assertEqual(self.proximity(other).status_code, 400)
        self.assertEqual(self.db.records, [])

    def test_disabled_falls_back_to_ble(self):
        self.web.config['CHECKIN_SESSION_CODE'] = False
        code, _ = session_code.current(SECRET, 'session')
        r = self.proximity(code)
        # The BLE path applies again: here the session's board is not configured (503).
        self.assertIn(r.status_code, (400, 503))
        self.assertNotIn('proximity_receipt', r.json)

    def test_code_receipt_is_not_a_ble_receipt(self):
        code, _ = session_code.current(SECRET, 'session')
        receipt = self.proximity(code).json['proximity_receipt']
        r = self.checkin(receipt, via=None)                 # same receipt sent as a BLE one
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.db.records, [])


class TeacherScreenTests(unittest.TestCase):
    def setUp(self):
        self.web = Flask(__name__, template_folder="../app/templates")
        self.web.config.update(SECRET_KEY='t', RATELIMIT_ENABLED=False, PROXIMITY_RECEIPT_SECRET=SECRET,
                               CHECKIN_SESSION_CODE=True, PUBLIC_BASE_URL='https://smartcheck.example')
        app.limiter.init_app(self.web)
        self.web.register_blueprint(auth.auth_bp)
        self.web.register_blueprint(teacher.teacher_bp, url_prefix='/teacher')
        self.client = self.web.test_client()
        with self.client.session_transaction() as s:
            s.update(user_id='teacher-1', user_role='teacher', csrf_token='csrf')
        self.row = {'id': 's1', 'title': 'Lab', 'is_open': True, 'course_id': 'c1',
                    'courses': {'code': 'CS101', 'name': 'x', 'teacher_id': 'teacher-1'}, 'beacons': {'room_name': 'LAB 2'}}
        q = Mock()
        for m in ('select', 'eq', 'maybe_single'):
            getattr(q, m).return_value = q
        q.execute.side_effect = lambda: SimpleNamespace(data=self.row)
        db = Mock()
        db.table.return_value = q
        p = patch.object(teacher, 'supabase_admin', db)
        p.start()
        self.addCleanup(p.stop)

    def test_owner_gets_code_and_qr_for_open_session(self):
        r = self.client.get('/teacher/session/s1/code.json')
        self.assertEqual(r.status_code, 200)
        self.assertRegex(r.json['code'], r'^\d{6}$')
        self.assertTrue(session_code.verify(SECRET, 's1', r.json['code']))
        self.assertIn('<svg', r.json['qr_svg'])
        self.assertEqual(r.headers['Cache-Control'], 'no-store')

    def test_other_teacher_closed_session_and_disabled(self):
        self.row['courses']['teacher_id'] = 'someone-else'
        self.assertEqual(self.client.get('/teacher/session/s1/code.json').status_code, 403)
        self.row['courses']['teacher_id'] = 'teacher-1'
        self.row['is_open'] = False
        r = self.client.get('/teacher/session/s1/code.json')
        self.assertFalse(r.json['ok'])
        self.assertNotIn('code', r.json)
        self.web.config['CHECKIN_SESSION_CODE'] = False
        self.assertEqual(self.client.get('/teacher/session/s1/code.json').status_code, 404)

    def test_qr_points_to_the_public_url(self):
        with self.web.test_request_context('/'):
            self.assertEqual(teacher._checkin_url('s1', '123456'),
                             'https://smartcheck.example/student/checkin?session_id=s1&via=code&c=123456')


class ReenrollBannerTests(unittest.TestCase):
    def test_only_legacy_students_under_arcface(self):
        web = Flask(__name__)
        q = Mock()
        for m in ('select', 'eq', 'maybe_single'):
            getattr(q, m).return_value = q
        db = Mock()
        db.table.return_value = q
        with web.test_request_context('/'), patch.object(student, 'supabase_admin', db):
            web.config['FACE_ENGINE'] = 'arcface'
            q.execute.return_value = SimpleNamespace(data={'enrolled_at': '2026-10-01', 'consent_given': True})
            self.assertTrue(student._has_legacy_enrollment('u1'))
            q.execute.return_value = SimpleNamespace(data={'enrolled_at': None, 'consent_given': False})
            self.assertFalse(student._has_legacy_enrollment('u1'))
            web.config['FACE_ENGINE'] = 'facenet512'
            q.execute.return_value = SimpleNamespace(data={'enrolled_at': '2026-10-01', 'consent_given': True})
            self.assertFalse(student._has_legacy_enrollment('u1'))


class LineUnlinkTests(unittest.TestCase):
    def test_admin_unlinks(self):
        web = Flask(__name__)
        web.config.update(SECRET_KEY='t', RATELIMIT_ENABLED=False)
        app.limiter.init_app(web)
        web.register_blueprint(auth.auth_bp)
        web.register_blueprint(admin.admin_bp, url_prefix='/admin')
        client = web.test_client()
        with client.session_transaction() as s:
            s.update(user_id='admin-1', user_role='admin', csrf_token='csrf')
        calls = []
        q = Mock()
        for m in ('eq',):
            getattr(q, m).return_value = q
        q.update.side_effect = lambda row: (calls.append(('update', row)), q)[1]
        q.delete.side_effect = lambda: (calls.append(('delete',)), q)[1]
        db = Mock()
        db.table.return_value = q
        with patch.object(admin, 'supabase_admin', db), patch.object(admin, 'log_audit_event') as audit:
            r = client.post('/admin/api/line-unlink/u1', json={}, headers={'X-CSRF-Token': 'csrf'})
        self.assertEqual(r.status_code, 200, r.json)
        self.assertEqual(calls, [('update', {'line_user_id': None}), ('delete',)])
        self.assertEqual(audit.call_args.kwargs['event_type'], 'line_unlinked')


if __name__ == '__main__':
    unittest.main()

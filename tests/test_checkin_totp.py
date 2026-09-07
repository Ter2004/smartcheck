import os
import re
import subprocess
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from flask import Flask
import app
from app.routes import api_checkin as route
from app.services import esp32_totp as totp
from app.services.security_service import compute_embedding_integrity_hash

ROOT = Path(__file__).resolve().parents[1]
SECRET = b"integration-test-only"
NOW = 1788758850


class Database:
    def __init__(self):
        self.records = []

    def table(self, name):
        query = Mock()
        for method in ['select', 'eq', 'neq', 'maybe_single']:
            getattr(query, method).return_value = query
        data = {
            'sessions': {'id': 'session', 'course_id': 'course', 'is_open': True},
            'course_enrollments': {'id': 'enrollment'},
            'users': {'device_id': ''},
            'student_biometrics': {
                'face_embeddings': [[1., 0.]],
                'integrity_hash': compute_embedding_integrity_hash('student', [[1., 0.]], 'salt')},
        }.get(name)
        query.execute.return_value = SimpleNamespace(data=data)
        def insert(record):
            self.records.append(record)
            return query
        query.insert.side_effect = insert
        return query


class IntegratedTOTPTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.db = Database()
        self.stack.enter_context(patch.object(route, 'supabase_admin', self.db))
        self.stack.enter_context(patch.object(totp.time, 'time', return_value=NOW))
        for name, result in {
            'server_validate_frame': {'valid': True},
            'detect_screen_moire': {'avg_score': 0., 'is_screen': False},
            'detect_screen_texture': False,
            'combined_spoof_score': {'is_real': True},
            'extract_embedding': [1., 0.],
            'verify_face_multi': {'verified': True, 'best_similarity': 1., 'avg_similarity': 1.},
        }.items():
            self.stack.enter_context(patch.object(route, name, return_value=result))
        self.stack.enter_context(patch.object(route, '_decode_image',
            side_effect=lambda image: np.full((64, 64, 3), int(image), dtype=np.uint8)))
        self.web = Flask(__name__)
        self.web.config.update(SECRET_KEY='test', ESP32_TOTP_SECRET=SECRET,
                               PROXIMITY_RECEIPT_SECRET='r' * 64,
                               EMBEDDING_INTEGRITY_SALT='salt', RATELIMIT_ENABLED=False)
        app.limiter.init_app(self.web)
        self.web.register_blueprint(route.api_checkin_bp)
        self.client = self.web.test_client()
        with self.client.session_transaction() as session:
            session.update(user_id='student', user_role='student', csrf_token='csrf')
        self.payload = dict(session_id='session', liveness_action='passive',
                            face_image='20', face_images=['20', '80', '140'],
                            ear_samples=[.25, .26], room_code=totp.generate_code(SECRET, NOW))

    def post(self):
        preflight = self.client.post('/api/checkin/proximity', json=self.payload,
                                     headers={'X-CSRF-Token': 'csrf'})
        if preflight.status_code != 200:
            return preflight
        self.payload['proximity_receipt'] = preflight.json['proximity_receipt']
        return self.client.post('/api/checkin', json=self.payload,
                                headers={'X-CSRF-Token': 'csrf'})

    def test_correct_checks_in(self):
        response = self.post()
        self.assertEqual(response.status_code, 200, response.json)
        self.assertTrue(response.json['ok'])
        self.assertEqual(len(self.db.records), 1)
        self.assertTrue(self.db.records[0]['face_pass'])
        self.assertTrue(self.db.records[0]['liveness_pass'])

    def preflight(self):
        return self.client.post('/api/checkin/proximity', json=self.payload,
                                headers={'X-CSRF-Token': 'csrf'})

    def test_checkin_audit_metrics_never_reject(self):
        for frames in [None, [], ['20'], ['20', '20'], ['invalid', 'invalid']]:
            self.payload['face_images'] = frames
            with self.subTest(frames=frames), self.assertLogs('smartcheck', level='INFO') as logs:
                self.assertEqual(self.post().status_code, 200)
            self.assertIn('log_only', '\n'.join(logs.output))
        for name in ['detect_screen_moire', 'detect_screen_texture']:
            with patch.object(route, name, side_effect=RuntimeError('audit failure')):
                with self.assertLogs('smartcheck', level='INFO') as logs:
                    self.assertEqual(self.post().status_code, 200)
            self.assertIn('error_log_only', '\n'.join(logs.output))

    def test_model_rejection_still_blocks_with_low_temporal_variance(self):
        self.payload['face_images'] = ['20', '20']
        with patch.object(route, 'combined_spoof_score', return_value={'is_real': False}), \
             patch.object(route, 'is_system_failure', return_value=False):
            response = self.post()
        self.assertEqual(response.status_code, 400)
        self.assertTrue(response.json['spoof'])
        self.assertEqual(self.db.records, [])

    def test_supplied_temporal_frames_cannot_enable_voting(self):
        from app.services import face_service as fs
        with patch.object(fs, 'detect_screen_moire', return_value={'avg_score': 0., 'is_screen': False}), \
             patch.object(fs, 'detect_screen_texture', return_value=False), \
             patch.object(fs, '_run_fasnet_antispoof', return_value=(True, .01)), \
             patch.object(fs, '_run_antispoof', return_value=(True, .99)), \
             patch.object(fs, 'detect_static_image', return_value={'is_static': True, 'temporal_variance': 0.}):
            baseline = fs.combined_spoof_score(np.zeros((64, 64, 3), dtype=np.uint8))
            with self.assertLogs('smartcheck', level='WARNING') as logs:
                supplied = fs.combined_spoof_score(np.zeros((64, 64, 3), dtype=np.uint8), frames_for_temporal=[1, 2])
        self.assertIn('TV-01', '\n'.join(logs.output))
        self.assertEqual(supplied['combined_score'], baseline['combined_score'])
        self.assertEqual(supplied['is_real'], baseline['is_real'])
        self.assertIsNone(supplied['layers']['temporal']['spoof_score'])

    def test_device_token_absence_and_rejection_reasons(self):
        from app.services.security_service import create_device_token
        valid = create_device_token('student', 'test-device', 'test')
        wrong = create_device_token('student', 'test-device', 'different-key')
        with patch.object(totp.time, 'time', return_value=NOW - 121 * 86400):
            expired = create_device_token('student', 'test-device', 'test')
        self.payload['proximity_receipt'] = self.preflight().json['proximity_receipt']
        for header, status, reason in [
            (None, 200, 'step=device_token result=absent'),
            ('DeviceToken', 200, 'step=device_token_bare_scheme result=absent'),
            ('DeviceToken ', 200, 'step=device_token_bare_scheme result=absent'),
            ('DeviceToken broken', 403, 'step=device_token_malformed result=reject'),
            ('DeviceToken ' + wrong, 403, 'step=device_token_signature_mismatch result=reject'),
            ('DeviceToken ' + expired, 403, 'step=device_token_expired result=reject'),
            ('DeviceToken ' + valid, 200, 'step=device_token result=pass'),
        ]:
            with self.subTest(header_kind=reason):
                headers = {'X-CSRF-Token': 'csrf'}
                if header is not None:
                    headers['Authorization'] = header
                with self.assertLogs('smartcheck', level='INFO') as logs:
                    response = self.client.post('/api/checkin', json=self.payload, headers=headers)
                self.assertEqual(response.status_code, status, response.json)
                self.assertIn(reason, '\n'.join(logs.output))
                for token in [valid, wrong, expired]:
                    self.assertNotIn(token, '\n'.join(logs.output))

    def final_post(self):
        return self.client.post('/api/checkin', json=self.payload,
                                headers={'X-CSRF-Token': 'csrf'})

    def test_preflight_does_not_run_face_and_totp_can_rotate_during_capture(self):
        with patch.object(route, 'extract_embedding') as extraction:
            result = self.preflight()
            self.assertEqual(result.status_code, 200)
            extraction.assert_not_called()
        self.payload['proximity_receipt'] = result.json['proximity_receipt']
        with patch.object(totp.time, 'time', return_value=NOW + 65):
            self.assertFalse(totp.verify_code(self.payload['room_code'], SECRET))
            self.assertEqual(self.final_post().status_code, 200)

    def test_receipt_missing_tampered_expired_and_binding(self):
        result = self.preflight()
        receipt = result.json['proximity_receipt']
        self.assertEqual(self.final_post().json['error_code'], 'proximity_receipt_missing')
        self.payload['proximity_receipt'] = receipt + 'tampered'
        self.assertEqual(self.final_post().json['error_code'], 'proximity_receipt_invalid')
        self.payload['proximity_receipt'] = receipt
        for key, value in [('session_id', 'other'), ('room_code', '000000')]:
            previous = self.payload[key]
            self.payload[key] = value
            self.assertEqual(self.final_post().json['error_code'], 'proximity_receipt_invalid')
            self.payload[key] = previous
        with self.client.session_transaction() as session:
            session['user_id'] = 'other-student'
        self.assertEqual(self.final_post().json['error_code'], 'proximity_receipt_invalid')
        with self.client.session_transaction() as session:
            session['user_id'] = 'student'
        with patch.object(totp.time, 'time', return_value=NOW + 91):
            response = self.final_post()
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json['error_code'], 'proximity_receipt_expired')
        self.assertEqual(self.db.records, [])

    def test_ble_room_bound_and_rechecked_without_totp(self):
        self.web.config['CHECKIN_PROXIMITY_METHOD'] = 'ble'
        self.payload['room_code'] = 'TEST-101'
        original_table = self.db.table
        def table(name):
            query = original_table(name)
            if name == 'sessions':
                query.execute.return_value.data['beacons'] = {'ble_room_code': 'TEST-101'}
            return query
        with patch.object(self.db, 'table', side_effect=table), patch.object(route, 'verify_code') as totp_check:
            result = self.preflight()
            self.assertEqual(result.status_code, 200)
            self.payload['proximity_receipt'] = result.json['proximity_receipt']
            self.assertEqual(self.final_post().status_code, 200)
            totp_check.assert_not_called()
            self.web.config['CHECKIN_PROXIMITY_METHOD'] = 'totp'
            self.assertEqual(self.final_post().json['error_code'], 'proximity_receipt_invalid')

    def test_receipt_secret_missing_aborts_boot(self):
        with patch.dict(os.environ), patch.object(app.db, 'init_app') as initialize:
            os.environ.pop('PROXIMITY_RECEIPT_SECRET', None)
            with self.assertRaisesRegex(RuntimeError, 'PROXIMITY_RECEIPT_SECRET'):
                app.create_app()
            initialize.assert_not_called()

    def test_closed_session_is_rechecked_after_preflight(self):
        self.payload['proximity_receipt'] = self.preflight().json['proximity_receipt']
        original_table = self.db.table
        def table(name):
            query = original_table(name)
            if name == 'sessions':
                query.execute.return_value.data['is_open'] = False
            return query
        with patch.object(self.db, 'table', side_effect=table):
            with self.assertLogs('smartcheck', level='INFO') as logs:
                self.assertEqual(self.final_post().status_code, 400)
        self.assertIn('step=session_closed result=reject', '\n'.join(logs.output))
        self.assertEqual(self.db.records, [])

    def test_request_and_spoof_logs_share_id_without_sensitive_payload(self):
        from app.services import face_service
        def spoof(_image):
            face_service._audit.info('[COMBINED_SPOOF] decision=real')
            return {'is_real': True}
        with self.assertLogs('smartcheck', level='INFO') as logs:
            with patch.object(route, 'combined_spoof_score', side_effect=spoof):
                response = self.post()
        rid = response.headers['X-Request-ID']
        text = '\n'.join(logs.output)
        self.assertIn(f'request_id={rid} endpoint=api_checkin.checkin [COMBINED_SPOOF]', text)
        self.assertIn('step=totp result=pass', text)
        for name in ['device_lookup', 'biometrics_lookup', 'extraction', 'matching']:
            for result in ['start', 'complete']:
                self.assertIn(f'step={name} result={result}', text)
        self.assertNotIn(self.payload['room_code'], text)
        self.assertNotIn(SECRET.decode(), text)
        with self.assertLogs('smartcheck', level='INFO'):
            other = self.post()
        self.assertNotEqual(rid, other.headers['X-Request-ID'])

    def test_empty_action_is_visible(self):
        self.payload['liveness_action'] = ''
        with self.assertLogs('smartcheck', level='INFO') as logs:
            response = self.post()
        self.assertEqual(response.status_code, 400)
        self.assertIn('step=liveness_action_invalid result=reject details={"received":""}',
                      '\n'.join(logs.output))

    def test_totp_rejection_and_error_outcomes(self):
        self.payload['room_code'] = totp.generate_code(SECRET, NOW - 60)
        with self.assertLogs('smartcheck', level='INFO') as logs:
            self.assertEqual(self.post().status_code, 400)
        self.assertIn('step=totp result=wrong_or_stale', '\n'.join(logs.output))
        with patch.object(route, 'verify_code', side_effect=RuntimeError(SECRET.decode())):
            with self.assertLogs('smartcheck', level='INFO') as logs:
                self.assertEqual(self.post().status_code, 503)
        self.assertIn('step=totp result=verifier_error', '\n'.join(logs.output))
        self.assertNotIn(SECRET.decode(), '\n'.join(logs.output))

    def test_extraction_error_has_start_error_and_rejection(self):
        with patch.object(route, 'extract_embedding', side_effect=RuntimeError('private-image-data')):
            with self.assertLogs('smartcheck', level='INFO') as logs:
                self.assertEqual(self.post().status_code, 400)
        text = '\n'.join(logs.output)
        self.assertIn('step=extraction result=start', text)
        self.assertIn('step=extraction result=error', text)
        self.assertIn('step=extraction_failed result=reject', text)
        self.assertNotIn('step=extraction result=complete', text)
        self.assertNotIn('private-image-data', text)

    def test_csrf_rejection_is_correlated(self):
        with self.assertLogs('smartcheck', level='INFO') as logs:
            response = self.client.post('/api/checkin', json=self.payload)
        self.assertEqual(response.status_code, 403)
        self.assertIn('step=csrf_rejected result=reject', '\n'.join(logs.output))
        self.assertIn('X-Request-ID', response.headers)

    def test_stale_is_400_and_can_retry_current_code(self):
        self.payload['room_code'] = totp.generate_code(SECRET, NOW - 60)
        response = self.post()
        self.assertEqual(response.status_code, 400)
        self.assertIn('กรุณาดูรหัสปัจจุบัน', response.json['error'])
        self.assertTrue(response.json['retry_room_code'])
        self.assertEqual(self.db.records, [])
        self.payload['room_code'] = totp.generate_code(SECRET, NOW)
        self.assertEqual(self.post().status_code, 200)

    def test_missing_code(self):
        del self.payload['room_code']
        response = self.post()
        self.assertEqual(response.status_code, 400)
        self.assertIn('กรุณากรอกรหัสห้อง', response.json['error'])
        self.assertEqual(self.db.records, [])

    def test_verifier_exception_is_503(self):
        with patch.object(route, 'verify_code', side_effect=RuntimeError('internal failure')):
            response = self.post()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json['error_code'], 'room_code_unavailable')
        self.assertIn('ระบบตรวจสอบรหัสห้องขัดข้อง', response.json['error'])
        self.assertNotIn('internal failure', response.json['error'])
        self.assertEqual(self.db.records, [])

    def test_simulator_console_to_checkin(self):
        env = dict(os.environ, ESP32_TOTP_SECRET=SECRET.decode(), ESP32_TOTP_SIMULATOR='true')
        # Run the actual hardware-free simulator process, then submit its output
        # through the authenticated, CSRF-protected route and attendance insert.
        process = subprocess.Popen([sys.executable, 'scripts/esp32_totp.py', '--simulate'],
                                   cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        try:
            # communicate has a timeout so a broken simulator cannot hang the suite.
            try:
                output, errors = process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                process.terminate()
                output, errors = process.communicate(timeout=5)
            match = re.search(r'unix=(\d+) counter=\d+ code=(\d{6})', output)
            self.assertIsNotNone(match, errors)
            self.payload['room_code'] = match[2]
            with patch.object(totp.time, 'time', return_value=int(match[1])):
                response = self.post()
            self.assertEqual(response.status_code, 200, response.json)
            self.assertEqual(len(self.db.records), 1)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()

    def test_valid_code_does_not_bypass_face(self):
        with patch.object(route, 'verify_face_multi', return_value={
                'verified': False, 'best_similarity': 0., 'avg_similarity': 0.}):
            self.assertEqual(self.post().status_code, 400)
        self.assertEqual(self.db.records, [])

    def test_boot_rejects_missing_or_empty_secret_before_database(self):
        for value in [None, '']:
            with patch.dict(os.environ), patch.object(app.db, 'init_app') as initialize:
                if value is None:
                    os.environ.pop('ESP32_TOTP_SECRET', None)
                else:
                    os.environ['ESP32_TOTP_SECRET'] = value
                with self.assertRaisesRegex(totp.TOTPConfigurationError, 'ESP32_TOTP_SECRET'):
                    app.create_app()
                initialize.assert_not_called()


if __name__ == '__main__':
    unittest.main()

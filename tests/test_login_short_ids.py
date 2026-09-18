import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from flask import Flask
from app.routes import auth


class ShortLoginTests(unittest.TestCase):
    def setUp(self):
        self.web = Flask(__name__)
        self.web.secret_key = 'test'
        self.backend = Mock()
        self.backend.auth.sign_in_with_password.return_value = SimpleNamespace(
            user=SimpleNamespace(id='user-id'), session=SimpleNamespace(access_token='token'))

    def login(self, identifier, active=True):
        with self.web.test_request_context('/login', method='POST',
                                         data={'email': identifier, 'password': 'test-password'}):
            with patch.object(auth, 'supabase', self.backend), \
                 patch.object(auth, 'get_user_by_id', return_value={
                     'role': 'student', 'full_name': 'Student', 'is_active': active}), \
                 patch.object(auth, '_redirect_by_role', return_value='logged-in'):
                # Exercise the route without its rate limiter wrapper.
                return auth.login.__wrapped__()

    def test_all_short_ids_use_real_password_authentication(self):
        for alias in ['std1', 'std2', 'std3', 'std4', 't1', 't2', 't3', 't4', 't5', 'admin']:
            with self.subTest(alias=alias):
                self.assertEqual(self.login(f' {alias.upper()} '), 'logged-in')
                self.backend.auth.sign_in_with_password.assert_called_with(
                    {'email': f'{alias}@smartcheck.local', 'password': 'test-password'})

    def test_email_login_is_preserved(self):
        self.assertEqual(self.login('person@example.com'), 'logged-in')
        self.backend.auth.sign_in_with_password.assert_called_with(
            {'email': 'person@example.com', 'password': 'test-password'})

    def test_unknown_alias_is_not_mapped_to_test_account(self):
        self.login('t6')
        self.backend.auth.sign_in_with_password.assert_called_with(
            {'email': 't6', 'password': 'test-password'})


if __name__ == '__main__':
    unittest.main()

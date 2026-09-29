import unittest
from unittest.mock import patch

from flask import Flask

import app
from app.config import Config
from app.routes import api_checkin as route


class RequestLimitTests(unittest.TestCase):
    def setUp(self):
        self.web = Flask(__name__)
        self.web.config.update(SECRET_KEY="test", RATELIMIT_ENABLED=False,
                               MAX_CONTENT_LENGTH=Config.MAX_CONTENT_LENGTH)
        app.limiter.init_app(self.web)
        self.web.register_blueprint(route.api_checkin_bp)
        self.client = self.web.test_client()
        with self.client.session_transaction() as session:
            session.update(user_id="student", user_role="student", csrf_token="csrf")

    def post(self, path, **json):
        return self.client.post(path, json=json, headers={"X-CSRF-Token": "csrf"})

    def test_body_limit_is_10_mb(self):
        self.assertEqual(Config.MAX_CONTENT_LENGTH, 10 * 1024 * 1024)

    def test_oversized_body_is_refused_before_the_route(self):
        with patch.object(route, "extract_embedding") as extraction, \
                patch.object(route, "combined_spoof_score") as spoof:
            response = self.post("/api/checkin", face_image="A" * (11 * 1024 * 1024))
        self.assertEqual(response.status_code, 413)
        extraction.assert_not_called()
        spoof.assert_not_called()

    def test_passive_antispoof_score_endpoint_is_gone(self):
        # It returned a numeric anti-spoof score: an oracle for tuning fakes.
        self.assertEqual(self.post("/api/antispoof-passive", face_image="x").status_code, 404)


if __name__ == "__main__":
    unittest.main()

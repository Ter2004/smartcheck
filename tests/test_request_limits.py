import base64
import unittest
from unittest.mock import patch

from flask import Flask

import app
from app.config import Config
from app.routes import api_checkin as route


def jpeg_b64(size):
    # JPEG markers around filler: passes the header/footer checks, so only the
    # size limit can reject it.
    body = b"\xff\xd8\xff" + b"\x00" * (size - 5) + b"\xff\xd9"
    return "data:image/jpeg;base64," + base64.b64encode(body).decode()


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

    def post_passive(self, **json):
        return self.client.post("/api/antispoof-passive", json=json,
                                headers={"X-CSRF-Token": "csrf"})

    def test_body_limit_is_10_mb(self):
        self.assertEqual(Config.MAX_CONTENT_LENGTH, 10 * 1024 * 1024)

    def test_oversized_body_is_refused_before_the_route(self):
        with patch.object(route, "check_anti_spoof_with_score") as scorer:
            response = self.post_passive(face_image="A" * (11 * 1024 * 1024))
        self.assertEqual(response.status_code, 413)
        scorer.assert_not_called()

    def test_passive_antispoof_rejects_large_frame_before_decoding(self):
        with patch.object(route, "check_anti_spoof_with_score") as scorer:
            response = self.post_passive(face_image=jpeg_b64(600 * 1024))
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json["real"])
        scorer.assert_not_called()

    def test_passive_antispoof_rejects_non_string_image(self):
        with patch.object(route, "check_anti_spoof_with_score") as scorer:
            response = self.post_passive(face_image=["not", "a", "frame"])
        self.assertEqual(response.status_code, 400)
        scorer.assert_not_called()

    def test_passive_antispoof_scores_a_valid_frame(self):
        with patch.object(route, "server_validate_frame", return_value={"valid": True}), \
                patch.object(route, "check_anti_spoof_with_score", return_value=(True, 0.91)):
            response = self.post_passive(face_image="frame")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"ok": True, "real": True, "score": 0.91})


if __name__ == "__main__":
    unittest.main()

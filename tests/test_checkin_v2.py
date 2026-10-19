"""Check-in face step on the arcface engine: re-enrollment, tampered rows, match threshold."""
import unittest
from unittest.mock import patch

import numpy as np
from flask import Flask

from app.routes import api_checkin
from app.services import face_engine, face_store
from app.services.security_service import compute_template_integrity_hash

CONFIG = {"FACE_ENGINE": "arcface", "FACE_CLAHE": "off", "ARCFACE_MATCH_THRESHOLD": "0.4",
          "ARCFACE_DUP_THRESHOLD": "0.5", "ENROLL_SELF_SIM_MIN": "0.6",   # test values, not production
          "EMBEDDING_INTEGRITY_SALT": "salt"}
MODEL = "buffalo_l/w600k_r50"


def unit(seed, base=None, noise=0.2):
    v = np.random.default_rng(seed).standard_normal(512).astype(np.float32)
    if base is not None:
        v = base + noise * v / np.linalg.norm(v)
    return (v / np.linalg.norm(v)).astype(np.float32)


ME = unit(1)
TEMPLATES = [unit(10 + i, ME) for i in range(5)]


def enrollment(**over):
    row = {"status": "completed", "model_version": MODEL, "templates": TEMPLATES,
           "integrity_hash": compute_template_integrity_hash("s1", MODEL, TEMPLATES, "salt")}
    row.update(over)
    return row


class CheckinV2Tests(unittest.TestCase):
    def setUp(self):
        self.web = Flask(__name__)
        self.web.config.update(SECRET_KEY="t", **CONFIG)
        self.ctx = self.web.test_request_context("/api/checkin")
        self.ctx.push()
        self.addCleanup(self.ctx.pop)
        self.live = unit(50, ME)

    def run_v2(self, row):
        frame = np.zeros((480, 640, 3), np.uint8)
        with patch.object(face_store, "load_enrollment", return_value=row), \
                patch.object(face_engine, "analyze", return_value={"embedding": self.live}):
            return api_checkin._verify_face_v2("s1", frame)

    def test_match_passes(self):
        score, failure = self.run_v2(enrollment())
        self.assertIsNone(failure)
        self.assertGreater(score, 0.4)

    def test_other_person_fails_as_mismatch(self):
        self.live = unit(99)
        score, failure = self.run_v2(enrollment())
        response, code = failure
        self.assertEqual(code, 400)
        self.assertTrue(response.json["retry_face"])

    def test_reenrollment_required_not_a_failed_match(self):
        for row in (None,                                         # only a legacy FaceNet512 row (or none)
                    enrollment(model_version="facenet512"),       # older model
                    enrollment(status="pending"),
                    enrollment(templates=[])):
            with self.subTest(row=None if row is None else {k: row[k] for k in ("model_version", "status")}):
                score, (response, code) = self.run_v2(row)
                self.assertEqual(code, 409)
                self.assertTrue(response.json["reenroll_required"])
                self.assertNotIn("retry_face", response.json)

    def test_tampered_row_fails_integrity(self):
        tampered = [t.copy() for t in TEMPLATES]
        tampered[0][3] += 0.05
        for row in (enrollment(templates=tampered),                # embedding edited in the DB
                    enrollment(integrity_hash="0" * 64),          # hash replaced
                    enrollment(integrity_hash=None)):             # hash removed
            with self.subTest():
                score, (response, code) = self.run_v2(row)
                self.assertEqual(code, 403)
        # Relabelling the row to another model breaks it too (model_version is in the HMAC).
        relabelled = enrollment()
        relabelled["integrity_hash"] = compute_template_integrity_hash("s1", "other/model", TEMPLATES, "salt")
        score, (response, code) = self.run_v2(relabelled)
        self.assertEqual(code, 403)

    def test_no_face_is_a_retry(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        with patch.object(face_store, "load_enrollment", return_value=enrollment()), \
                patch.object(face_engine, "analyze", side_effect=face_engine.NoFaceError()):
            score, (response, code) = api_checkin._verify_face_v2("s1", frame)
        self.assertEqual(code, 400)
        self.assertTrue(response.json["retry_face"])


if __name__ == "__main__":
    unittest.main()

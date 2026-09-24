import unittest
from unittest.mock import patch

import numpy as np
from app.services import face_service as service


class FakeSession:
    def __init__(self, logits):
        self.logits = np.array([logits], dtype=np.float32)
        self.blob = None

    def get_inputs(self):
        return [type("Input", (), {"name": "input"})()]

    def run(self, _, feeds):
        self.blob = feeds["input"]
        return [self.logits]


class AntispoofOnnxInputTests(unittest.TestCase):
    def run_model(self, logits=(0.0, 5.0, 0.0)):
        # Distinct constant per channel so channel order and scale are observable.
        crop = np.zeros((80, 80, 3), np.uint8)
        crop[:, :, 0], crop[:, :, 1], crop[:, :, 2] = 10, 100, 200  # B, G, R
        session = FakeSession(logits)
        with patch.object(service, "_get_antispoof_session", return_value=session), \
                patch.object(service, "_crop_face_for_antispoof", return_value=crop):
            result = service._run_antispoof(np.zeros((120, 160, 3), np.uint8))
        return session.blob, result

    def test_model_receives_bgr_in_0_255_range(self):
        blob, _ = self.run_model()
        self.assertEqual(blob.shape, (1, 3, 80, 80))
        self.assertEqual(blob.dtype, np.float32)
        self.assertTrue(np.all(blob[0, 0] == 10))
        self.assertTrue(np.all(blob[0, 1] == 100))
        self.assertTrue(np.all(blob[0, 2] == 200))

    def test_real_class_is_index_one(self):
        _, (is_real, real_score) = self.run_model((0.0, 5.0, 0.0))
        self.assertTrue(is_real)
        self.assertGreater(real_score, 0.9)
        _, (is_real, real_score) = self.run_model((5.0, 0.0, 0.0))
        self.assertFalse(is_real)
        self.assertLess(real_score, 0.1)


if __name__ == "__main__":
    unittest.main()

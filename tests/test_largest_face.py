"""Main-face selection and the face box anti-spoof scores (face stack v2: SCRFD)."""
import unittest
from unittest.mock import patch

import numpy as np

from app.services import face_engine, fasnet_onnx
from app.services import face_service as service


class LargestFaceTests(unittest.TestCase):
    def test_picks_largest_opencv_box(self):
        # Haar crop for the ONNX audit layer: a small false detection at the edge
        # (LFW Britney_Spears_0001) must not win over the real face.
        boxes = np.array([[0, 142, 78, 78], [70, 67, 113, 113]])
        self.assertEqual(list(service._largest_face(boxes)), [70, 67, 113, 113])

    def test_fasnet_scores_the_scrfd_main_face_box(self):
        seen = {}

        def analyze(img, box):
            seen["box"] = box
            return True, 0.8
        main = {"bbox": (70.4, 68.2, 185.9, 183.7), "det_score": 0.9, "kps": None}
        with patch.object(face_engine, "detect_main", return_value=main), \
                patch.object(fasnet_onnx, "analyze", side_effect=analyze):
            is_real, spoof = service._run_fasnet_antispoof(np.zeros((250, 250, 3), np.uint8))
        self.assertTrue(is_real)
        self.assertAlmostEqual(spoof, 0.2)
        self.assertEqual(seen["box"], (70, 68, 115, 115))

    def test_box_is_clipped_to_the_frame(self):
        # A padded-retry detection can extend past the frame (face fills the frame).
        face = {"bbox": (-30.0, -10.0, 300.0, 280.0)}
        self.assertEqual(face_engine.box_xywh(face, (250, 250, 3)), (0, 0, 249, 249))

    def test_spoof_check_without_embedding_face_is_a_retry(self):
        with patch.object(service, "_decode_image", return_value=np.zeros((80, 80, 3), np.uint8)), \
             patch.object(service, "combined_spoof_score",
                          return_value={"is_real": True, "combined_score": 0.0, "layers": {}}), \
             patch.object(face_engine, "analyze", side_effect=face_engine.NoFaceError()):
            result = service.spoof_check_with_embedding("x")
        self.assertTrue(result["retry_capture"])
        self.assertFalse(result["system_failure"])
        self.assertIsNone(result["embedding"])
        self.assertNotIn("Face could not", result["message"])


if __name__ == "__main__":
    unittest.main()

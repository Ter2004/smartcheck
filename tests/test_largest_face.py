import unittest
from unittest.mock import patch

import numpy as np

from app.services import face_service as service

# A small false detection at the image edge listed before the real face,
# as DeepFace returned for LFW Britney_Spears_0001.
EDGE = {"embedding": [0.0, 1.0], "facial_area": {"x": 0, "y": 167, "w": 73, "h": 73},
        "is_real": False, "antispoof_score": 0.9}
MAIN = {"embedding": [1.0, 0.0], "facial_area": {"x": 70, "y": 68, "w": 115, "h": 115},
        "is_real": True, "antispoof_score": 0.8}


class LargestFaceTests(unittest.TestCase):
    def test_picks_largest_deepface_dict_whatever_the_order(self):
        self.assertIs(service._largest_face([EDGE, MAIN]), MAIN)
        self.assertIs(service._largest_face([MAIN, EDGE]), MAIN)

    def test_picks_largest_opencv_box(self):
        boxes = np.array([[0, 142, 78, 78], [70, 67, 113, 113]])
        self.assertEqual(list(service._largest_face(boxes)), [70, 67, 113, 113])

    def test_extract_embedding_uses_main_face(self):
        with patch.object(service, "_decode_image", return_value=np.zeros((250, 250, 3), np.uint8)), \
             patch.object(service, "_call_deepface", return_value=[EDGE, MAIN]):
            embedding, meta = service.extract_embedding("x", include_metadata=True)
        self.assertEqual(embedding, MAIN["embedding"])
        self.assertEqual(meta["detector_crop"], MAIN["facial_area"])

    def test_fasnet_scores_main_face(self):
        with patch.object(service, "_call_deepface", return_value=[EDGE, MAIN]):
            is_real, spoof = service._run_fasnet_antispoof(np.zeros((250, 250, 3), np.uint8))
        self.assertTrue(is_real)
        self.assertAlmostEqual(spoof, 0.2)


if __name__ == "__main__":
    unittest.main()

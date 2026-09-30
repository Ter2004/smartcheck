import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from app.services import face_service as service


def _face(x1, y1, x2, y2, score=0.99):
    cx, cy, w = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1
    return {"score": score, "facial_area": [x1, y1, x2, y2], "landmarks": {
        "left_eye": [cx + w * 0.2, cy - w * 0.1], "right_eye": [cx - w * 0.2, cy - w * 0.1],
        "nose": [cx, cy], "mouth_left": [cx + w * 0.15, cy + w * 0.2], "mouth_right": [cx - w * 0.15, cy + w * 0.2]}}


# A small false detection at the image edge listed before the real face,
# as the Haar cascade returned for LFW Britney_Spears_0001.
EDGE = _face(0, 167, 73, 240)
MAIN = _face(70, 68, 185, 183)


def _retinaface(faces, seen):
    def detect_faces(img, threshold=0.9, allow_upscaling=True):
        seen.append((img.shape, threshold, allow_upscaling))
        return faces
    return types.SimpleNamespace(RetinaFace=types.SimpleNamespace(detect_faces=detect_faces))


class LargestFaceTests(unittest.TestCase):
    def test_picks_largest_opencv_box(self):
        boxes = np.array([[0, 142, 78, 78], [70, 67, 113, 113]])
        self.assertEqual(list(service._largest_face(boxes)), [70, 67, 113, 113])

    def test_detect_main_face_picks_largest_without_upscaling(self):
        seen = []
        with patch.dict(sys.modules, {"retinaface": _retinaface({"face_1": EDGE, "face_2": MAIN}, seen)}):
            box, left_eye, right_eye = service._detect_main_face(np.zeros((250, 250, 3), np.uint8))
        self.assertEqual(box, (70, 68, 115, 115))
        self.assertGreater(left_eye[0], right_eye[0])   # person's left eye is on the image right
        self.assertEqual(seen, [((250, 250, 3), 0.9, False)])

    def test_detect_main_face_downscales_and_maps_back(self):
        seen = []
        big = _face(140, 100, 370, 330)   # coordinates in the 480x360 copy
        with patch.dict(sys.modules, {"retinaface": _retinaface({"face_1": big}, seen)}):
            box, _, _ = service._detect_main_face(np.zeros((960, 1280, 3), np.uint8))
        self.assertEqual(seen[0][0], (360, 480, 3))
        self.assertEqual(box, (373, 266, 613, 613))

    def test_no_face_raises_the_deepface_message(self):
        with patch.dict(sys.modules, {"retinaface": _retinaface({}, [])}):
            with self.assertRaises(ValueError) as caught:
                service._detect_main_face(np.zeros((250, 250, 3), np.uint8))
        self.assertTrue(str(caught.exception).startswith("Face could not be detected"))

    def test_extract_embedding_uses_main_face_and_skip_detector(self):
        calls = []

        def represent(method, **kw):
            calls.append((method, kw))
            return [{"embedding": [1.0, 0.0]}]
        img = np.zeros((250, 250, 3), np.uint8)
        with patch.dict(sys.modules, {"retinaface": _retinaface({"a": EDGE, "b": MAIN}, [])}), \
             patch.object(service, "_decode_image", return_value=img), \
             patch.object(service, "_call_deepface", side_effect=represent):
            embedding, meta = service.extract_embedding("x", include_metadata=True)
        self.assertEqual(embedding, [1.0, 0.0])
        self.assertEqual(meta["detector_crop"], {"x": 70, "y": 68, "w": 115, "h": 115})
        method, kw = calls[0]
        self.assertEqual((method, kw["detector_backend"], kw["model_name"]), ("represent", "skip", "Facenet512"))
        self.assertGreater(kw["img_path"].shape[0], 100)   # the aligned face crop, not the full frame

    def test_fasnet_scores_main_face_box(self):
        seen = {}

        class Fasnet:
            def analyze(self, img, facial_area):
                seen["box"] = facial_area
                return True, 0.8
        modeling = types.SimpleNamespace(build_model=lambda task, model_name: Fasnet())
        with patch.dict(sys.modules, {"retinaface": _retinaface({"a": EDGE, "b": MAIN}, [])}), \
             patch("deepface.modules.modeling", modeling, create=True):
            import deepface.modules
            with patch.object(deepface.modules, "modeling", modeling, create=True):
                is_real, spoof = service._run_fasnet_antispoof(np.zeros((250, 250, 3), np.uint8))
        self.assertTrue(is_real)
        self.assertAlmostEqual(spoof, 0.2)
        self.assertEqual(seen["box"], (70, 68, 115, 115))

    def test_spoof_check_without_embedding_face_is_a_retry(self):
        with patch.object(service, "_decode_image", return_value=np.zeros((80, 80, 3), np.uint8)), \
             patch.object(service, "combined_spoof_score",
                          return_value={"is_real": True, "combined_score": 0.0, "layers": {}}), \
             patch.object(service, "_face_embedding", side_effect=ValueError(service._NO_FACE_MSG)):
            result = service.spoof_check_with_embedding("x")
        self.assertTrue(result["retry_capture"])
        self.assertFalse(result["system_failure"])
        self.assertIsNone(result["embedding"])
        self.assertNotIn("Face could not", result["message"])


if __name__ == "__main__":
    unittest.main()

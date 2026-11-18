"""One SCRFD detection per check-in frame, shared by Fasnet and ArcFace; models run on the main face only."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.services import face_engine


def _fake_app(boxes):
    calls = {"detect": 0, "recognition": [], "landmark": []}
    kps = np.array([[[25., 30.], [45., 30.], [35., 40.], [28., 55.], [42., 55.]]] * len(boxes))

    def detect(img, max_num=0, metric="default"):
        calls["detect"] += 1
        return np.array(boxes, dtype=np.float32), kps

    def recognition(img, face):
        calls["recognition"].append(tuple(face.bbox))
        face.embedding = np.ones(512, np.float32)

    def landmark(img, face):
        calls["landmark"].append(tuple(face.bbox))
        face.pose = np.array([1., 2., 3.])

    app = SimpleNamespace(det_model=SimpleNamespace(detect=detect),
                          models={"detection": None,
                                  "recognition": SimpleNamespace(get=recognition),
                                  "landmark_3d_68": SimpleNamespace(get=landmark)})
    return app, calls


class SharedDetectionTests(unittest.TestCase):
    def test_detect_main_detection_is_reused_by_analyze(self):
        app, calls = _fake_app([[10, 10, 60, 70, .9]])
        img = np.zeros((80, 80, 3), np.uint8)
        with patch.object(face_engine, "get_app", return_value=app):
            main = face_engine.detect_main(img)
            face = face_engine.analyze(img, clahe=False, detection=main["detection"])
        self.assertEqual(calls["detect"], 1)
        self.assertEqual(face["bbox"], main["bbox"])

    def test_detection_of_another_frame_is_not_reused(self):
        app, calls = _fake_app([[10, 10, 60, 70, .9]])
        a, b = np.zeros((80, 80, 3), np.uint8), np.zeros((80, 80, 3), np.uint8)
        with patch.object(face_engine, "get_app", return_value=app):
            main = face_engine.detect_main(a)
            face_engine.analyze(b, clahe=False, detection=main["detection"])
        self.assertEqual(calls["detect"], 2)

    def test_clahe_detects_again(self):
        app, calls = _fake_app([[10, 10, 60, 70, .9]])
        img = np.full((80, 80, 3), 90, np.uint8)
        with patch.object(face_engine, "get_app", return_value=app):
            main = face_engine.detect_main(img)
            face_engine.analyze(img, clahe=True, detection=main["detection"])
        self.assertEqual(calls["detect"], 2)

    def test_models_run_on_the_main_face_only(self):
        # A small face behind the main one: counted, but no embedding or pose for it.
        app, calls = _fake_app([[50, 50, 60, 60, .8], [10, 10, 60, 70, .9]])
        with patch.object(face_engine, "get_app", return_value=app):
            face = face_engine.analyze(np.zeros((80, 80, 3), np.uint8), clahe=False)
        self.assertEqual(face["faces"], 2)
        self.assertFalse(face["second_large_face"])
        self.assertEqual(calls["recognition"], [(10., 10., 60., 70.)])
        self.assertEqual(calls["landmark"], [(10., 10., 60., 70.)])

    def test_second_large_face_still_rejected(self):
        app, _ = _fake_app([[10, 10, 60, 70, .9], [0, 0, 45, 60, .9]])
        with patch.object(face_engine, "get_app", return_value=app):
            with self.assertRaises(face_engine.MultipleFacesError):
                face_engine.analyze(np.zeros((80, 80, 3), np.uint8), clahe=False)


if __name__ == "__main__":
    unittest.main()

"""Head-turn verification on the arcface engine (yaw in landmark_3d_68 degrees)."""
import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from app.services import face_engine
from app.services import liveness_challenge as lc

IMAGES = Path(__file__).resolve().parent.parent / "test_images"
HAVE_MODEL = importlib.util.find_spec("insightface") is not None and \
    (Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx").exists()


def _emb(seed):
    v = np.random.default_rng(seed).standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def _frame(yaw, roll=0.0, emb=None):
    return {"yaw": yaw, "roll": roll, "embedding": _emb(1) if emb is None else emb, "unit": "deg"}


class ArcfaceVerifyTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(face_engine, "threshold", return_value=0.4)   # test value
        patcher.start()
        self.addCleanup(patcher.stop)

    def _verify(self, actions, before, action_frames, after):
        challenge = {"nonce": "n", "actions": actions, "issued_at": 1000.0}
        frames = {id(f): f for f in [before, after] + action_frames}
        return lc.verify(challenge, "n", before, action_frames, after, now=1001.0,
                         analyze=lambda f: frames[id(f)])

    def test_correct_turn_passes(self):
        for action, yaw in (("turn_left", 25.0), ("turn_right", -25.0)):
            with self.subTest(action=action):
                r = self._verify([action], _frame(3.0), [_frame(yaw)], _frame(-2.0))
                self.assertTrue(r["passed"], r)

    def test_wrong_or_partial_turn_fails(self):
        r = self._verify(["turn_left"], _frame(3.0), [_frame(-25.0)], _frame(0.0))
        self.assertEqual(r["reason"], "wrong_direction:action_1:turn_right")
        r = self._verify(["turn_left"], _frame(3.0), [_frame(19.5)], _frame(0.0))   # between 18.4 and 20.4
        self.assertEqual(r["reason"], "wrong_direction:action_1:partial")

    def test_not_frontal_before(self):
        r = self._verify(["turn_left"], _frame(19.0), [_frame(30.0)], _frame(0.0))
        self.assertEqual(r["reason"], "not_frontal:before")

    def test_identity_uses_arcface_match_threshold(self):
        other = _emb(99)   # a different random unit vector: cosine ~ 0
        r = self._verify(["turn_left"], _frame(0.0), [_frame(25.0, emb=other)], _frame(0.0))
        self.assertEqual(r["reason"], "identity_mismatch:action_1")
        # A swap after the turn is caught on the after frame too.
        r = self._verify(["turn_left"], _frame(0.0), [_frame(25.0)], _frame(0.0, emb=other))
        self.assertEqual(r["reason"], "identity_mismatch:after")

    def test_tilt_limits_in_degrees(self):
        r = self._verify(["tilt_left"], _frame(0.0, roll=0.0), [_frame(30.0, roll=12.0)], _frame(0.0))
        self.assertTrue(r["passed"], r)   # 30 deg yaw is under the 35.7 deg tilt limit
        r = self._verify(["tilt_left"], _frame(0.0, roll=0.0), [_frame(40.0, roll=12.0)], _frame(0.0))
        self.assertEqual(r["reason"], "wrong_direction:action_1:turned")


class ArcfaceAnalyzeDispatchTests(unittest.TestCase):
    def test_dispatch_and_error_mapping(self):
        fake = {"yaw_deg": 22.0, "eye_roll_deg": -3.0, "embedding": _emb(2)}
        with patch.dict(os.environ, {"FACE_ENGINE": "arcface"}), \
                patch.object(face_engine, "analyze", return_value=fake):
            out = lc.analyze_frame(np.zeros((480, 640, 3), np.uint8))
        self.assertEqual((out["yaw"], out["roll"], out["unit"]), (22.0, -3.0, "deg"))
        for error, reason in ((face_engine.NoFaceError(), "no_face"), (face_engine.MultipleFacesError(), "multiple_faces")):
            with self.subTest(reason=reason), patch.dict(os.environ, {"FACE_ENGINE": "arcface"}), \
                    patch.object(face_engine, "analyze", side_effect=error):
                with self.assertRaisesRegex(lc.FrameError, reason):
                    lc.analyze_frame(np.zeros((480, 640, 3), np.uint8))


@unittest.skipUnless(HAVE_MODEL, "insightface buffalo_l not available")
class ArcfaceRealFrameTests(unittest.TestCase):
    def test_directions_on_real_frames(self):
        with patch.dict(os.environ, {"FACE_ENGINE": "arcface", "FACE_CLAHE": "off"}), \
                patch.object(face_engine, "clahe_enabled", return_value=False):
            for name, expected in (("real_05_sharp_face2.jpg", "frontal"),
                                   ("real_lowdetail_dark15.jpg", "turn_left"),
                                   ("real_lowdetail_motion.jpg", "turn_right")):
                with self.subTest(name=name):
                    out = lc.analyze_frame(cv2.imread(str(IMAGES / name)))
                    self.assertEqual(lc.direction(out["yaw"], out["unit"]), expected)


if __name__ == "__main__":
    unittest.main()

"""fasnet_onnx: DeepFace's crop geometry, output shape, and parity when DeepFace is installed."""
import importlib.util
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.services import fasnet_onnx

IMAGES = Path(__file__).resolve().parent.parent / "test_images"


class CropGeometryTests(unittest.TestCase):
    def test_box_scaled_and_centred(self):
        # 100x100 box centred at (320, 240) in a 640x480 frame, scale 2.7 -> 270x270 crop.
        self.assertEqual(fasnet_onnx._get_new_box(640, 480, (270, 190, 100, 100), 2.7), (185, 105, 455, 375))

    def test_scale_capped_and_shifted_inside(self):
        # A box near the left edge is shifted right, never cropped smaller than scale allows.
        x1, y1, x2, y2 = fasnet_onnx._get_new_box(640, 480, (0, 190, 100, 100), 4.0)
        self.assertEqual((x1, x2 - x1), (0, 400))
        # Scale is capped so the crop fits the frame height.
        x1, y1, x2, y2 = fasnet_onnx._get_new_box(640, 480, (270, 140, 200, 200), 4.0)
        self.assertEqual((y1, y2), (0, 479))

    def test_crop_is_80x80_bgr(self):
        img = np.zeros((480, 640, 3), np.uint8)
        img[..., 2] = 200
        out = fasnet_onnx.crop(img, (270, 190, 100, 100), 2.7)
        self.assertEqual(out.shape, (80, 80, 3))
        self.assertEqual(out[0, 0].tolist(), [0, 0, 200])


@unittest.skipUnless(fasnet_onnx.available(), "exported Fasnet ONNX models missing")
class OnnxOutputTests(unittest.TestCase):
    def test_probabilities_and_decision(self):
        img = cv2.imread(str(IMAGES / "real_05_sharp_face2_fit.jpg"))
        h, w = img.shape[:2]
        box = (w // 4, h // 4, w // 2, h // 2)
        probs = fasnet_onnx.probabilities(img, box)
        self.assertEqual([p.shape for p in probs], [(3,), (3,)])
        for p in probs:
            self.assertAlmostEqual(float(p.sum()), 1.0, places=5)
        is_real, score = fasnet_onnx.analyze(img, box)
        self.assertIsInstance(is_real, bool)
        self.assertTrue(0.0 <= score <= 1.0)

    @unittest.skipUnless(importlib.util.find_spec("deepface") and importlib.util.find_spec("torch"),
                         "DeepFace/torch not installed (removed in Phase 5)")
    def test_matches_deepface_fasnet(self):
        from deepface.modules import modeling
        fasnet = modeling.build_model(task="spoofing", model_name="Fasnet")
        for name in ("real_05_sharp_face2_fit.jpg", "spoof_01_phone.jpg"):
            img = cv2.imread(str(IMAGES / name))
            h, w = img.shape[:2]
            box = (w // 4, h // 4, w // 2, h // 2)
            real_t, score_t = fasnet.analyze(img=img, facial_area=box)
            real_o, score_o = fasnet_onnx.analyze(img, box)
            self.assertEqual(bool(real_t), real_o, name)
            self.assertAlmostEqual(float(score_t), score_o, places=5)


if __name__ == "__main__":
    unittest.main()

"""face_engine: config fail-fast, channel-order guard, and the arcface model on real photos."""
import base64
import importlib.util
import os
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.services import face_engine
from app.services.security_service import compute_template_integrity_hash, verify_template_integrity

IMAGES = Path(__file__).resolve().parent.parent / "test_images"
MODELS = Path.home() / ".insightface" / "models" / "buffalo_l"
HAVE_MODEL = importlib.util.find_spec("insightface") is not None and (MODELS / "w600k_r50.onnx").exists()
GOOD = {"FACE_ENGINE": "arcface", "FACE_CLAHE": "off", "ARCFACE_MATCH_THRESHOLD": "0.4",
        "ARCFACE_DUP_THRESHOLD": "0.5", "ENROLL_SELF_SIM_MIN": "0.6"}   # test values, not production


def _frame(name, width=640):
    img = cv2.imread(str(IMAGES / name))
    scale = width / img.shape[1]
    return cv2.resize(img, (width, round(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)


class ConfigTests(unittest.TestCase):
    def test_facenet_default_needs_nothing(self):
        self.assertEqual(face_engine.validate_config({}), {})
        self.assertEqual(face_engine.model_version("facenet512"), "facenet512")
        self.assertEqual(face_engine.model_version("arcface"), "buffalo_l/w600k_r50")

    def test_unknown_engine_fails(self):
        with self.assertRaises(RuntimeError):
            face_engine.validate_config({"FACE_ENGINE": "vgg"})

    def test_arcface_without_each_setting_fails(self):
        for key in ("FACE_CLAHE", "ARCFACE_MATCH_THRESHOLD", "ARCFACE_DUP_THRESHOLD", "ENROLL_SELF_SIM_MIN"):
            with self.subTest(missing=key):
                cfg = dict(GOOD)
                cfg.pop(key)
                with self.assertRaisesRegex(RuntimeError, key):
                    face_engine.validate_config(cfg)

    def test_arcface_rejects_bad_values(self):
        for key, value in (("FACE_CLAHE", "maybe"), ("ARCFACE_MATCH_THRESHOLD", "high"), ("ENROLL_SELF_SIM_MIN", "1.5")):
            with self.subTest(key=key):
                with self.assertRaises(RuntimeError):
                    face_engine.validate_config(dict(GOOD, **{key: value}))

    def test_arcface_parsed(self):
        self.assertEqual(face_engine.validate_config(GOOD), {
            "FACE_CLAHE": False, "ARCFACE_MATCH_THRESHOLD": 0.4,
            "ARCFACE_DUP_THRESHOLD": 0.5, "ENROLL_SELF_SIM_MIN": 0.6})


class ChannelOrderTests(unittest.TestCase):
    def test_rgb_is_flipped_to_bgr(self):
        img = np.zeros((4, 4, 3), np.uint8)
        img[..., 0] = 10   # R in RGB order
        out = face_engine.ensure_bgr(img, "RGB")
        self.assertEqual(out[0, 0].tolist(), [0, 0, 10])
        self.assertTrue(out.flags["C_CONTIGUOUS"])

    def test_bgr_passes_unchanged(self):
        img = np.random.default_rng(0).integers(0, 255, (4, 4, 3), dtype=np.uint8)
        self.assertTrue(np.array_equal(face_engine.ensure_bgr(img), img))

    def test_wrong_inputs_are_refused(self):
        with self.assertRaises(TypeError):
            face_engine.ensure_bgr(np.zeros((4, 4, 3), np.float32))
        with self.assertRaises(TypeError):
            face_engine.ensure_bgr(np.zeros((4, 4), np.uint8))
        with self.assertRaises(ValueError):
            face_engine.ensure_bgr(np.zeros((4, 4, 3), np.uint8), "HSV")


class TemplateHmacTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.embs = [v / np.linalg.norm(v) for v in rng.standard_normal((5, 512)).astype(np.float32)]

    def test_binds_user_and_model_version(self):
        base = compute_template_integrity_hash("u1", "buffalo_l/w600k_r50", self.embs, "salt")
        self.assertNotEqual(base, compute_template_integrity_hash("u1", "facenet512", self.embs, "salt"))
        self.assertNotEqual(base, compute_template_integrity_hash("u2", "buffalo_l/w600k_r50", self.embs, "salt"))
        self.assertEqual(base, compute_template_integrity_hash("u1", "buffalo_l/w600k_r50", self.embs[::-1], "salt"))
        self.assertTrue(verify_template_integrity("u1", "buffalo_l/w600k_r50", self.embs, base, "salt"))

    def test_tampered_or_missing_fails(self):
        base = compute_template_integrity_hash("u1", "buffalo_l/w600k_r50", self.embs, "salt")
        tampered = [e.copy() for e in self.embs]
        tampered[2][7] += 0.01
        self.assertFalse(verify_template_integrity("u1", "buffalo_l/w600k_r50", tampered, base, "salt"))
        self.assertFalse(verify_template_integrity("u1", "buffalo_l/w600k_r50", self.embs, "", "salt"))
        self.assertFalse(verify_template_integrity("u1", "buffalo_l/w600k_r50", [], base, "salt"))


@unittest.skipUnless(HAVE_MODEL, "insightface or ~/.insightface/models/buffalo_l not available")
class ArcfaceModelTests(unittest.TestCase):
    def test_zero_faces(self):
        with self.assertRaises(face_engine.NoFaceError) as ctx:
            face_engine.analyze(np.full((480, 640, 3), 127, np.uint8), clahe=False)
        self.assertIn("could not be detected", str(ctx.exception))   # existing route handlers match on this

    def test_one_face(self):
        face = face_engine.analyze(_frame("real_01_sharp_face1_fit.jpg"), clahe=False)
        self.assertEqual(face["embedding"].dtype, np.float32)
        self.assertEqual(face["embedding"].shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(face["embedding"])), 1.0, places=4)
        self.assertGreater(face["face_px"], 0)
        for key in ("det_score", "yaw_deg", "pitch_deg", "roll_deg", "yaw_ratio"):
            self.assertTrue(np.isfinite(face[key]), key)

    def test_two_faces(self):
        a, b = _frame("real_01_sharp_face1_fit.jpg", 320), _frame("real_05_sharp_face2_fit.jpg", 320)
        h = min(a.shape[0], b.shape[0])
        with self.assertRaises(face_engine.MultipleFacesError):
            face_engine.analyze(np.hstack([a[:h], b[:h]]), clahe=False)

    def test_channel_order_matters_and_is_converted(self):
        bgr = _frame("real_01_sharp_face1_fit.jpg")
        as_bgr = face_engine.analyze(bgr, clahe=False)["embedding"]
        declared_rgb = face_engine.analyze(bgr[:, :, ::-1].copy(), channel_order="RGB", clahe=False)["embedding"]
        self.assertGreater(float(as_bgr @ declared_rgb), 0.9999)
        # Feeding RGB as if it were BGR (the mix-up the guard prevents) changes the embedding.
        mixed = face_engine.analyze(bgr[:, :, ::-1].copy(), clahe=False)["embedding"]
        self.assertLess(float(as_bgr @ mixed), 0.999)

    def test_extract_embedding_same_signature(self):
        ok, buf = cv2.imencode(".jpg", _frame("real_01_sharp_face1_fit.jpg"))
        b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
        prev = {k: os.environ.get(k) for k in GOOD}
        os.environ.update(GOOD)
        try:
            emb, meta = face_engine.extract_embedding(b64, include_metadata=True)
            self.assertEqual(len(emb), 512)
            self.assertEqual(meta["model_version"], "buffalo_l/w600k_r50")
            self.assertEqual(set(meta["detector_crop"]), {"x", "y", "w", "h"})
            self.assertEqual(len(face_engine.extract_embedding(b64)), 512)
        finally:
            for k, v in prev.items():
                os.environ.pop(k) if v is None else os.environ.__setitem__(k, v)


if __name__ == "__main__":
    unittest.main()

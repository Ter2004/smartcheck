"""Face detection, alignment and embedding, switchable between two stacks.

FACE_ENGINE=facenet512 (default) keeps the existing DeepFace path in
face_service.py untouched. FACE_ENGINE=arcface uses InsightFace buffalo_l on
ONNX Runtime CPU: SCRFD det_10g finds the face and its 5 keypoints, the face is
aligned to the ArcFace template, w600k_r50 returns a 512-D embedding, and
1k3d68 gives the head pose (pitch, yaw, roll in degrees).

Input images are BGR uint8, as cv2.imdecode returns them. Callers holding RGB
must say so (channel_order="RGB"); ensure_bgr converts at this boundary so a
channel mix-up cannot reach the models silently.

License: the InsightFace pretrained models (buffalo_l) are for non-commercial
research only. Fine for this capstone; a commercial deployment needs other
weights or a licence from InsightFace.
"""
import logging
import os
import threading
import time

import numpy as np

MODEL_VERSIONS = {"facenet512": "facenet512", "arcface": "buffalo_l/w600k_r50"}
ENGINES = tuple(MODEL_VERSIONS)
ARCFACE_THRESHOLD_KEYS = ("ARCFACE_MATCH_THRESHOLD", "ARCFACE_DUP_THRESHOLD", "ENROLL_SELF_SIM_MIN")
DET_SIZE = (640, 640)
# A second face at least this fraction of the main face's area fails the frame;
# smaller ones (people far behind) are ignored. Same rule and value as the
# head-turn check (liveness_challenge.SECOND_FACE_AREA_RATIO).
SECOND_FACE_AREA_RATIO = 0.5

_log = logging.getLogger("smartcheck.face_engine")
_app = None
_app_lock = threading.Lock()
# One inference at a time per process, like the DeepFace calls (_deepface_lock).
_infer_lock = threading.Lock()
_channel_order_logged = False


class FaceEngineError(ValueError):
    """The frame cannot be evaluated. Messages match DeepFace's so existing handlers apply."""


class NoFaceError(FaceEngineError):
    def __init__(self):
        super().__init__("Face could not be detected in numpy array.")


class MultipleFacesError(FaceEngineError):
    def __init__(self):
        super().__init__("Face could not be detected in numpy array: more than one face.")


# ─── Configuration ───────────────────────────────────────────────────────────

def _config():
    try:
        from flask import current_app, has_app_context
        if has_app_context():
            return current_app.config
    except ImportError:
        pass
    return os.environ


def active_engine(config=None):
    return str((config or _config()).get("FACE_ENGINE") or "facenet512").strip().lower()


def model_version(engine=None):
    return MODEL_VERSIONS[engine or active_engine()]


def _parse_switch(value):
    text = str(value).strip().lower()
    if text in ("on", "1", "true", "yes"):
        return True
    if text in ("off", "0", "false", "no"):
        return False
    raise ValueError(value)


def validate_config(config):
    """Fail fast: an unknown engine, or arcface without its CLAHE switch and thresholds.

    Returns the parsed arcface settings ({} for facenet512).
    """
    engine = active_engine(config)
    if engine not in ENGINES:
        raise RuntimeError(f"[SmartCheck] FACE_ENGINE={engine!r} is invalid (expected one of {ENGINES}).")
    if engine != "arcface":
        return {}
    missing = [k for k in ("FACE_CLAHE",) + ARCFACE_THRESHOLD_KEYS if config.get(k) in (None, "")]
    if missing:
        raise RuntimeError(
            "[SmartCheck] FACE_ENGINE=arcface needs " + ", ".join(missing)
            + " (values from scripts/compare_models.py; there are no defaults).")
    settings = {}
    try:
        settings["FACE_CLAHE"] = _parse_switch(config.get("FACE_CLAHE"))
    except ValueError:
        raise RuntimeError(f"[SmartCheck] FACE_CLAHE={config.get('FACE_CLAHE')!r} must be on or off.") from None
    for key in ARCFACE_THRESHOLD_KEYS:
        try:
            value = float(config.get(key))
        except (TypeError, ValueError):
            raise RuntimeError(f"[SmartCheck] {key}={config.get(key)!r} is not a number.") from None
        if not -1.0 <= value <= 1.0:
            raise RuntimeError(f"[SmartCheck] {key}={value} is outside the cosine range [-1, 1].")
        settings[key] = value
    return settings


def threshold(key):
    """An arcface threshold from the active config (validated at startup)."""
    return validate_config(_config())[key]


def clahe_enabled():
    return validate_config(_config())["FACE_CLAHE"]


# ─── Model ───────────────────────────────────────────────────────────────────

def get_app():
    """The buffalo_l FaceAnalysis singleton (loaded once per process)."""
    global _app
    if _app is None:
        with _app_lock:
            if _app is None:
                from insightface.app import FaceAnalysis
                app = FaceAnalysis(name="buffalo_l",
                                   allowed_modules=["detection", "recognition", "landmark_3d_68"],
                                   providers=["CPUExecutionProvider"])
                app.prepare(ctx_id=-1, det_size=DET_SIZE)
                missing = {"detection", "recognition", "landmark_3d_68"} - set(app.models)
                if missing:
                    raise RuntimeError(f"buffalo_l is missing models: {sorted(missing)}")
                _app = app
    return _app


def warm_up():
    """Load the models and run one dummy inference through each. Returns seconds taken."""
    global _channel_order_logged
    t0 = time.perf_counter()
    app = get_app()
    with _infer_lock:
        app.get(np.zeros((480, 640, 3), dtype=np.uint8))          # detector
        app.models["recognition"].get_feat(np.zeros((112, 112, 3), dtype=np.uint8))
    seconds = time.perf_counter() - t0
    _log.info("[FACE_ENGINE] arcface %s warm-up %.2fs input=BGR uint8 (cv2.imdecode order)",
              model_version("arcface"), seconds)
    _channel_order_logged = True
    return seconds


def ensure_bgr(img, channel_order="BGR"):
    """Return a contiguous BGR uint8 HxWx3 image; convert RGB, reject anything else."""
    global _channel_order_logged
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise TypeError("expected a uint8 HxWx3 image")
    order = str(channel_order).upper()
    if order == "RGB":
        img = img[:, :, ::-1]
    elif order != "BGR":
        raise ValueError(f"unknown channel order {channel_order!r}")
    if not _channel_order_logged:
        _log.info("[FACE_ENGINE] input channel order: BGR uint8 (callers convert at the boundary)")
        _channel_order_logged = True
    return np.ascontiguousarray(img)


def _area(bbox):
    return max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))


def yaw_ratio_from_kps(kps):
    """Nose offset from the eye midpoint along the eye line, in eye distances.

    Same formula as liveness_challenge.yaw_ratio on the same 5-point layout
    (eyes, nose, mouth corners). Positive = nose toward the image's right.
    """
    a, b = np.asarray(kps[0], float), np.asarray(kps[1], float)
    left, right = (a, b) if a[0] <= b[0] else (b, a)
    axis = right - left
    denom = float(np.dot(axis, axis))
    if denom < 1e-6:
        return 0.0
    return float(np.dot(np.asarray(kps[2], float) - (left + right) / 2, axis) / denom)


def roll_deg_from_kps(kps):
    """Eye-line angle in degrees; positive = the image-right eye is lower."""
    a, b = np.asarray(kps[0], float), np.asarray(kps[1], float)
    left, right = (a, b) if a[0] <= b[0] else (b, a)
    return float(np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0])))


def analyze(img, channel_order="BGR", clahe=None, reject_multiple=True):
    """The single main face of a frame: embedding, detection score, size and pose.

    Raises NoFaceError, or MultipleFacesError when a second face is at least
    SECOND_FACE_AREA_RATIO of the main one (exactly one significant face).
    reject_multiple=False takes the largest face instead (offline comparisons
    only, to match the facenet512 path's largest-face selection).
    """
    from app.services.face_service import normalize_illumination

    img = ensure_bgr(img, channel_order)
    if clahe is None:
        clahe = clahe_enabled()
    if clahe:
        img = normalize_illumination(img)
    app = get_app()
    with _infer_lock:
        faces = app.get(img)
    if not faces:
        raise NoFaceError()
    ranked = sorted(faces, key=lambda f: _area(f.bbox), reverse=True)
    main = ranked[0]
    second_large = len(ranked) > 1 and _area(ranked[1].bbox) >= SECOND_FACE_AREA_RATIO * _area(main.bbox)
    if second_large and reject_multiple:
        raise MultipleFacesError()
    x1, y1, x2, y2 = (float(v) for v in main.bbox)
    pitch, yaw, roll = (float(v) for v in main.pose)
    embedding = np.asarray(main.normed_embedding, dtype=np.float32)
    return {
        "embedding": embedding,
        "det_score": float(main.det_score),
        "face_px": int(round(y2 - y1)),          # detection box height
        "bbox": (x1, y1, x2, y2),
        "kps": np.asarray(main.kps, dtype=float),
        "yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll,
        "yaw_ratio": yaw_ratio_from_kps(main.kps),
        "eye_roll_deg": roll_deg_from_kps(main.kps),
        "faces": len(faces), "second_large_face": bool(second_large),
    }


def box_xywh(face, img_shape):
    """The detection box as clipped integer (x, y, w, h), the layout Fasnet expects."""
    h, w = img_shape[:2]
    x1, y1, x2, y2 = face["bbox"]
    x, y = max(0, int(x1)), max(0, int(y1))
    return x, y, max(1, min(w - x - 1, int(x2 - x1))), max(1, min(h - y - 1, int(y2 - y1)))


# ─── Route-facing API (same signature as face_service.extract_embedding) ─────

def extract_embedding(base64_image, include_metadata=False):
    """Decode → (CLAHE if configured) → embedding of the main face as a list of floats."""
    from app.services import face_service

    if active_engine() != "arcface":
        return face_service.extract_embedding(base64_image, include_metadata=include_metadata)
    img = face_service._decode_image(base64_image)
    face = analyze(img)
    embedding = [float(v) for v in face["embedding"]]
    if not include_metadata:
        return embedding
    x, y, w, h = box_xywh(face, img.shape)
    return embedding, {
        "detector_crop": {"x": x, "y": y, "w": w, "h": h},
        "det_score": round(face["det_score"], 4), "face_px": face["face_px"],
        "yaw_deg": round(face["yaw_deg"], 1), "model_version": model_version("arcface"),
    }

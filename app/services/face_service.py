import base64
import logging
import os
import threading
import time
import traceback
import numpy as np
import cv2
import json

# ─── Thresholds (edit here to tune) ──────────────────────────────────────────
SELF_VERIFY_THRESHOLD    = 0.80   # no route uses it (self-verify step removed); calibration script reports it
SAME_DEVICE_THRESHOLD    = 0.70   # check-in, trusted device
NEW_DEVICE_THRESHOLD     = 0.80   # check-in, new / unbound device
CONSISTENCY_THRESHOLD    = 0.80   # pairwise consistency during enrollment
DUPLICATE_THRESHOLD      = 0.65   # reject if another student matches this closely
CONTINUITY_THRESHOLD     = 0.80   # liveness -> capture identity continuity
TEMPORAL_VAR_THRESHOLD   = 4.0   # historical, uncalibrated face-ROI reference
# No calibrated face-crop separation range is retained. The former claim that
# cropped real faces score ~15-25 is contradicted by a cooperative real session
# measured at 3.609. Callers decide whether this reference is audit or enforcement.
DUPLICATE_GRAY_ZONE      = (0.60, 0.70)  # log matches in this range for future tuning

# ─── Weighted spoof detection config ─────────────────────────────────────────
# Each layer outputs spoof_score in [0.0, 1.0] where 0=real, 1=spoof.
# Final decision: weighted sum > SPOOF_DECISION_THRESHOLD → reject.
#
# The Moiré and screen-texture FFT layers were removed on 2026-09-30. They
# never beat the trivial "always real" baseline (16 samples, FRR-1/F-15/Q-15 in
# docs/review/10-moire-frr-investigation.md), and on 2,000 CelebA-Spoof crops
# their AUC was 0.50 and 0.53 (docs/evidence/eval-2026-09-30).
SPOOF_WEIGHTS = {
    # 2026-08-26 rebalance (docs/review/10-moire-frr-investigation.md §13).
    # The prior comment here ("FFT layers more reliable than Fasnet") was
    # never measured and turned out backwards: Fasnet is the only layer with
    # confirmed separation (FULL SEPARATION, gap 0.7562, n=16). Temporal is
    # unmeasured — no burst-capture data exists — so its weight is cut, not
    # zeroed, pending its own validation round. Onnx is left at its prior
    # value (gap≈0 measured, but restructuring it was out of scope here).
    "fasnet":   0.70,
    "temporal": 0.20,  # TV-01: historical nominal weight; effective weight is ALWAYS zero.
    "onnx":     0.10,
}
SPOOF_DECISION_THRESHOLD = 0.50

from app.services.request_audit import RequestLogger
_audit = RequestLogger(logging.getLogger("smartcheck.enrollment"), {})

# DeepFace caches one OpenCV face/eye detector per process. Its native cascade
# buffers must not be used concurrently by enrollment and check-in requests.
_deepface_lock = threading.RLock()
_cascade_lock = threading.Lock()


def _call_deepface(method, **kwargs):
    with _deepface_lock:
        from deepface import DeepFace
        return getattr(DeepFace, method)(**kwargs)


class FaceNotDetectedError(ValueError):
    """The frame cannot be evaluated; this is not a spoof verdict."""


def _largest_face(faces):
    """The biggest detection, from DeepFace dicts or OpenCV (x, y, w, h) boxes.

    Neither returns faces in a fixed order, and the smaller boxes are often a
    person in the background or a false detection at the image edge: taking
    the first one matched the wrong face in 16 of 18 LFW impostor matches
    (docs/evidence/eval-2026-09-30/README.md).
    """
    def area(face):
        box = (face.get("facial_area") or {}) if isinstance(face, dict) else {"w": face[2], "h": face[3]}
        return box.get("w", 0) * box.get("h", 0)
    return max(faces, key=area)


# ─── Face detection for embeddings and Fasnet ─────────────────────────────────
# RetinaFace, as in the head-turn check, on a copy no larger than DETECT_MAX_SIDE
# and without upscaling: DeepFace's own "retinaface" backend upscales every frame
# to 1024 px (~750 ms on CPU). OpenCV's Haar cascade, used until 2026-09-30,
# missed or misplaced the face in 9+ frames of one webcam enrollment in which
# RetinaFace found it every time (docs/evidence/eval-2026-09-30).
DETECT_MAX_SIDE = 480
MIN_FACE_SCORE  = 0.9
# Same text as DeepFace, which _log_face_exception and callers match on.
_NO_FACE_MSG = "Face could not be detected in numpy array."


def _detect_main_face(img_bgr: np.ndarray):
    """(x, y, w, h), left_eye, right_eye of the largest face, in img_bgr pixels.

    Eyes follow DeepFace's convention (the person's own left/right).
    Raises ValueError(_NO_FACE_MSG) when there is no face.
    """
    from retinaface import RetinaFace

    h, w = img_bgr.shape[:2]
    scale = min(1.0, DETECT_MAX_SIDE / max(h, w))
    small = img_bgr if scale == 1.0 else cv2.resize(
        img_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    with _deepface_lock:
        faces = RetinaFace.detect_faces(small, threshold=MIN_FACE_SCORE, allow_upscaling=False)
    if not isinstance(faces, dict) or not faces:
        raise ValueError(_NO_FACE_MSG)
    face = max(faces.values(), key=lambda f: (f["facial_area"][2] - f["facial_area"][0])
                                             * (f["facial_area"][3] - f["facial_area"][1]))
    x1, y1, x2, y2 = (v / scale for v in face["facial_area"])
    x, y = max(0, int(x1)), max(0, int(y1))
    box = (x, y, min(w - x - 1, int(x2 - x1)), min(h - y - 1, int(y2 - y1)))
    eyes = [tuple(int(v / scale) for v in face["landmarks"][k][:2]) for k in ("left_eye", "right_eye")]
    return box, eyes[0], eyes[1]


def _aligned_face(img_bgr: np.ndarray, box, left_eye, right_eye) -> np.ndarray:
    """The face rotated level and cropped exactly as DeepFace's detect-and-align does."""
    from deepface.modules.detection import align_img_wrt_eyes, project_facial_area

    h, w = img_bgr.shape[:2]
    bh, bw = int(0.5 * h), int(0.5 * w)
    padded = cv2.copyMakeBorder(img_bgr, bh, bh, bw, bw, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    x, y, fw, fh = box
    aligned, angle = align_img_wrt_eyes(img=padded, left_eye=(left_eye[0] + bw, left_eye[1] + bh),
                                        right_eye=(right_eye[0] + bw, right_eye[1] + bh))
    x1, y1, x2, y2 = project_facial_area(facial_area=(x + bw, y + bh, x + bw + fw, y + bh + fh),
                                         angle=angle, size=(padded.shape[0], padded.shape[1]))
    return aligned[int(y1):int(y2), int(x1):int(x2)]


def _face_embedding(img_bgr: np.ndarray):
    """(FaceNet512 embedding, box) of the largest face. Raises ValueError without a face."""
    box, left_eye, right_eye = _detect_main_face(img_bgr)
    crop = _aligned_face(img_bgr, box, left_eye, right_eye)
    if crop.size == 0:
        raise ValueError(_NO_FACE_MSG)
    # With detector_backend="skip" DeepFace flips channels once (it expects the RGB
    # face its detectors return), so pass RGB to feed FaceNet what it always got.
    rep = _call_deepface("represent", img_path=np.ascontiguousarray(crop[:, :, ::-1]),
                         model_name="Facenet512", enforce_detection=False, detector_backend="skip")
    return rep[0]["embedding"], box


def _log_face_exception(stage, error):
    # Do not log exception messages: upstream errors can include image inputs.
    # A stable reason and stack locations still identify detector/model failures.
    reason = (
        "face_not_detected"
        if isinstance(error, ValueError) and str(error).startswith("Face could not be detected")
        else "inference_failed"
    )
    locations = " > ".join(
        f"{os.path.basename(frame.filename)}:{frame.lineno}:{frame.name}"
        for frame in traceback.extract_tb(error.__traceback__)[-6:]
    )
    _audit.error("[%s] error=%s reason=%s stack=%s", stage,
                 type(error).__name__, reason, locations)
    if isinstance(error, cv2.error):
        # Native diagnostics only: never log image inputs or full error text.
        _audit.error("[%s] opencv_code=%s opencv_func=%s", stage,
                     error.code, error.func)
    return reason

# ─── Anti-spoof ONNX (Silent-Face MiniFASNetV2) ───────────────────────────────
_antispoof_session    = None
_antispoof_lock       = threading.Lock()
_ANTISPOOF_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "antispoof.onnx")
_ANTISPOOF_INPUT_SIZE = 80
_ANTISPOOF_SCALE      = 2.7

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _get_antispoof_session():
    global _antispoof_session
    if _antispoof_session is None:
        with _antispoof_lock:
            if _antispoof_session is None:
                try:
                    import onnxruntime as ort
                except ImportError:
                    _audit.warning("[ANTISPOOF] onnxruntime not installed — ONNX audit layer disabled")
                    return None
                if not os.path.exists(_ANTISPOOF_MODEL_PATH):
                    _audit.warning(f"[ANTISPOOF] ONNX model not found at {_ANTISPOOF_MODEL_PATH} — audit layer disabled")
                    return None
                _antispoof_session = ort.InferenceSession(
                    _ANTISPOOF_MODEL_PATH, providers=["CPUExecutionProvider"]
                )
                _audit.info(f"[ANTISPOOF] ONNX session loaded from {_ANTISPOOF_MODEL_PATH}")
    return _antispoof_session


def _crop_face_for_antispoof(img_bgr: np.ndarray, scale: float = 2.7, size: int = 80) -> np.ndarray:
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    with _cascade_lock:
        faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    h_img, w_img = img_bgr.shape[:2]
    if len(faces) > 0:
        x, y, w, h = _largest_face(faces)
        cx, cy = x + w // 2, y + h // 2
        nw, nh = int(w * scale), int(h * scale)
        x1 = max(0, cx - nw // 2);    y1 = max(0, cy - nh // 2)
        x2 = min(w_img, cx + nw // 2); y2 = min(h_img, cy + nh // 2)
        crop = img_bgr[y1:y2, x1:x2]
    else:
        side = min(h_img, w_img)
        y1 = (h_img - side) // 2; x1 = (w_img - side) // 2
        crop = img_bgr[y1:y1+side, x1:x1+side]
    if crop.size == 0:
        crop = img_bgr
    return cv2.resize(crop, (size, size))


def _run_antispoof(img_bgr: np.ndarray) -> tuple:
    session    = _get_antispoof_session()
    crop       = _crop_face_for_antispoof(img_bgr)
    # Silent-Face MiniFASNet expects BGR in 0-255 (DeepFace FasNet.to_tensor
    # deliberately drops div(255)); RGB/255 made the output constant.
    blob       = np.transpose(crop.astype(np.float32), (2, 0, 1))[np.newaxis, :]
    input_name = session.get_inputs()[0].name
    raw        = session.run(None, {input_name: blob})[0][0]   # (3,)

    shifted = raw - raw.max()
    exp_out = np.exp(shifted)
    probs   = exp_out / (exp_out.sum() + 1e-8)

    # Official Silent-Face logic: argmax across all 3 classes
    #   class 0 = printed-photo spoof
    #   class 1 = real
    #   class 2 = screen/replay spoof
    label      = int(np.argmax(probs))
    real_score = float(probs[1])
    is_real    = (label == 1)

    # Confidence-margin guard: if argmax picks spoof but class 1 is a
    # close runner-up, be lenient.
    CONFIDENCE_MARGIN = 0.10
    sorted_probs = sorted(probs, reverse=True)
    margin = float(sorted_probs[0] - sorted_probs[1])
    overridden = False
    if not is_real and margin < CONFIDENCE_MARGIN and real_score > 0.25:
        _audit.warning(
            f"[ANTISPOOF] borderline reject overridden — "
            f"label={label} real={real_score:.4f} margin={margin:.4f}"
        )
        is_real = True
        overridden = True

    _audit.info(
        f"[ANTISPOOF] label={label} "
        f"probs=[spoof={probs[0]:.4f}, real={probs[1]:.4f}, "
        f"screen={probs[2]:.4f}] margin={margin:.4f} "
        f"is_real={is_real}{' (overridden)' if overridden else ''}"
    )
    return is_real, real_score


def _run_fasnet_antispoof(img_bgr: np.ndarray) -> tuple:
    """
    Run DeepFace's built-in anti-spoofing (Fasnet/MiniVision Silent-Face).
    Returns (is_real: bool, spoof_score: float) where spoof_score ∈ [0,1]
    and 0 = definitely real, 1 = definitely spoof.
    On exception returns (None, None) — caller redistributes weight.
    """
    try:
        box, _, _ = _detect_main_face(img_bgr)
        with _deepface_lock:
            from deepface.modules import modeling
            fasnet = modeling.build_model(task="spoofing", model_name="Fasnet")
            is_real, raw_score = fasnet.analyze(img=img_bgr, facial_area=box)
        is_real   = bool(is_real)
        raw_score = float(raw_score)
        spoof_score = (1.0 - raw_score) if is_real else raw_score
        spoof_score = max(0.0, min(1.0, spoof_score))
        return is_real, spoof_score
    except Exception as e:
        if _log_face_exception("FASNET", e) == "face_not_detected":
            raise FaceNotDetectedError("face_not_detected") from e
        return None, None


def combined_spoof_score(
    img_bgr: np.ndarray,
    frames_for_temporal: list = None,
) -> dict:
    """
    Run the anti-spoof layers (fasnet, temporal, onnx) and combine them into a
    weighted score. Temporal is audit-only (weight 0, TV-01).

    Args:
        img_bgr: single frame (primary input for single-frame checks)
        frames_for_temporal: optional list of 2+ frames for temporal variance.
            If None or <2 frames, temporal layer is skipped (weight redistributed).

    Returns dict with keys: is_real, combined_score, threshold, layers,
    weights_used, disagreements.

    Fail behavior: Fasnet, ONNX, Temporal fail-open (None → weight
    redistributed to 0). If ALL voting layers fail → fail-close (is_real=False).
    """
    layers = {}
    active_weights = dict(SPOOF_WEIGHTS)

    # ── Layer 3: Temporal Variance (fail-open if no frames) ────────────────
    if frames_for_temporal is not None:
        _audit.warning("[COMBINED_SPOOF] finding=TV-01 step=temporal_supplied result=audit_only decision=log_only effective_weight=0; uncalibrated temporal voting DISABLED")
    if frames_for_temporal is not None and len(frames_for_temporal) >= 2:
        try:
            temporal = detect_static_image(frames_for_temporal)
            variance = temporal["temporal_variance"]
            if variance >= TEMPORAL_VAR_THRESHOLD * 2:
                temporal_spoof = 0.0
            elif variance <= TEMPORAL_VAR_THRESHOLD / 2:
                temporal_spoof = 1.0
            else:
                temporal_spoof = 1.0 - (variance - TEMPORAL_VAR_THRESHOLD / 2) / (TEMPORAL_VAR_THRESHOLD * 1.5)
                temporal_spoof = max(0.0, min(1.0, temporal_spoof))
            layers["temporal"] = {
                "spoof_score": round(temporal_spoof, 4),
                "variance": variance,
                "is_static": temporal["is_static"],
            }
        except Exception as e:
            _audit.warning(f"[COMBINED_SPOOF] temporal error skip: {type(e).__name__}")
            layers["temporal"] = {"spoof_score": None, "variance": None, "error": str(e)[:80]}
            active_weights["temporal"] = 0.0
    else:
        layers["temporal"] = {"spoof_score": None, "variance": None, "reason": "not_enough_frames"}
        active_weights["temporal"] = 0.0

    # TV-01: retain diagnostic computation, but never vote or hard-reject from it.
    active_weights["temporal"] = 0.0
    layers["temporal"]["audit_spoof_score"] = layers["temporal"]["spoof_score"]
    layers["temporal"]["spoof_score"] = None
    layers["temporal"]["decision"] = "log_only"

    # ── Layer 4: DeepFace Fasnet (primary ML, fail-open) ───────────────────
    # TEMP PERF: wall-clock this layer for docs/review/06-performance.md — remove after measurement
    _t0_fasnet = time.perf_counter()
    try:
        fasnet_is_real, fasnet_spoof = _run_fasnet_antispoof(img_bgr)
    except FaceNotDetectedError:
        return {
            "is_real": False, "combined_score": 1.0,
            "threshold": SPOOF_DECISION_THRESHOLD,
            "layers": layers, "weights_used": {},
            "disagreements": ["face_not_detected"], "retry_capture": True,
        }
    _fasnet_ms = round((time.perf_counter() - _t0_fasnet) * 1000, 2)
    if fasnet_spoof is not None:
        layers["fasnet"] = {
            "spoof_score": round(fasnet_spoof, 4),
            "is_real": fasnet_is_real,
        }
    else:
        layers["fasnet"] = {"spoof_score": None, "is_real": None, "error": "inference_failed"}
        active_weights["fasnet"] = 0.0

    # ── Layer 5: Old ONNX (audit layer, fail-open) ─────────────────────────
    # TEMP PERF: wall-clock this layer for docs/review/06-performance.md — remove after measurement
    _t0_onnx = time.perf_counter()
    try:
        onnx_is_real, onnx_raw = _run_antispoof(img_bgr)
        onnx_spoof = 1.0 - onnx_raw
        layers["onnx"] = {
            "spoof_score": round(onnx_spoof, 4),
            "is_real": onnx_is_real,
            "raw_real_score": round(onnx_raw, 4),
        }
    except Exception as e:
        _audit.warning(f"[COMBINED_SPOOF] onnx error skip: {type(e).__name__}")
        layers["onnx"] = {"spoof_score": None, "is_real": None, "raw_real_score": None, "error": str(e)[:80]}
        active_weights["onnx"] = 0.0
    _onnx_ms = round((time.perf_counter() - _t0_onnx) * 1000, 2)

    # ── CRITICAL: fail-close if primary ML layer (Fasnet) is dead ──────────
    # Without Fasnet, only FFT layers remain — insufficient for high-DPI screens.
    fasnet_alive = layers.get("fasnet", {}).get("spoof_score") is not None
    if not fasnet_alive:
        _audit.error(
            "[COMBINED_SPOOF] Fasnet layer unavailable — failing CLOSED "
            "(rejecting frame). FFT-only defense is insufficient for "
            "high-DPI screen attacks."
        )
        return {
            "is_real": False,
            "combined_score": 1.0,
            "threshold": SPOOF_DECISION_THRESHOLD,
            "layers": layers,
            "weights_used": active_weights,
            "disagreements": ["fasnet_unavailable_fail_close"],
        }

    # ── Multi-layer hard-reject rule ────────────────────────────────────────
    # Weighted scoring can be dominated by Fasnet when it's wrong.
    # If BOTH remaining voting layers independently flag suspicious, reject
    # immediately — real faces rarely trigger 2 layers at once.

    def _layer_suspicious(layer_data, threshold):
        score = layer_data.get("spoof_score")
        return score is not None and score >= threshold

    temporal_suspicious = _layer_suspicious(layers.get("temporal", {}), 0.50)
    fasnet_suspicious   = _layer_suspicious(layers.get("fasnet", {}),   0.30)

    suspicious_count = sum([temporal_suspicious, fasnet_suspicious])

    if suspicious_count >= 1:
        _tw = sum(active_weights.values())
        _wbc = round(sum(
            layers[k]["spoof_score"] * v / _tw
            for k, v in active_weights.items()
            if layers.get(k, {}).get("spoof_score") is not None
        ), 4) if _tw > 0 else 1.0
        _pre_susp = (
            ([f"temporal({layers['temporal']['spoof_score']:.4f})"] if temporal_suspicious else []) +
            ([f"fasnet({layers['fasnet']['spoof_score']:.4f})"]   if fasnet_suspicious   else [])
        )
        _audit.warning(
            f"[COMBINED_SPOOF] PRE-HARDREJECT "
            f"all=[fasnet={layers['fasnet'].get('spoof_score')} "
            f"temporal={layers['temporal'].get('spoof_score')} "
            f"onnx={layers['onnx'].get('spoof_score')}] "
            f"suspicious={_pre_susp or ['none']} "
            f"count={suspicious_count} "
            f"would_be_combined={_wbc}"
        )

    if suspicious_count >= 2:
        suspicious_names = []
        if temporal_suspicious: suspicious_names.append(f"temporal({layers['temporal']['spoof_score']:.3f})")
        if fasnet_suspicious:   suspicious_names.append(f"fasnet({layers['fasnet']['spoof_score']:.3f})")
        _audit.warning(
            f"[COMBINED_SPOOF] HARD-REJECT: {suspicious_count} layers suspicious "
            f"({', '.join(suspicious_names)}) — bypassing weighted score"
        )
        return {
            "is_real": False,
            "combined_score": 1.0,
            "threshold": SPOOF_DECISION_THRESHOLD,
            "layers": layers,
            "weights_used": active_weights,
            "disagreements": [f"hard_reject_{suspicious_count}_layers_agree"],
            "hard_reject": True,
            "suspicious_layers": suspicious_names,
        }

    # ── Normalize active weights so they sum to 1.0 ────────────────────────
    total_weight = sum(active_weights.values())
    if total_weight <= 0:
        _audit.error("[COMBINED_SPOOF] all layers failed — fail-close")
        return {
            "is_real": False,
            "combined_score": 1.0,
            "threshold": SPOOF_DECISION_THRESHOLD,
            "layers": layers,
            "weights_used": active_weights,
            "disagreements": ["all_layers_failed"],
        }
    normalized_weights = {k: v / total_weight for k, v in active_weights.items()}

    # ── Compute weighted combined score ────────────────────────────────────
    combined = 0.0
    for layer_name, weight in normalized_weights.items():
        layer_data = layers.get(layer_name, {})
        score = layer_data.get("spoof_score")
        if score is not None and weight > 0:
            combined += score * weight
    combined = round(combined, 4)

    is_real = combined < SPOOF_DECISION_THRESHOLD

    # ── Identify layer disagreements for audit ─────────────────────────────
    disagreements = []
    for layer_name, layer_data in layers.items():
        score = layer_data.get("spoof_score")
        if score is None:
            continue
        layer_says_spoof = score >= 0.5
        final_says_spoof = not is_real
        if layer_says_spoof != final_says_spoof:
            disagreements.append(
                f"{layer_name}(spoof_score={score:.3f},says_{'spoof' if layer_says_spoof else 'real'})"
            )

    _audit.info(
        f"[COMBINED_SPOOF] combined={combined:.4f} threshold={SPOOF_DECISION_THRESHOLD} "
        f"decision={'real' if is_real else 'spoof'} "
        f"layers=["
        f"fasnet={layers['fasnet'].get('spoof_score')}, "
        f"temporal={layers['temporal'].get('spoof_score')}, "
        f"onnx={layers['onnx'].get('spoof_score')}] "
        f"disagreements={disagreements or 'none'}"
    )

    return {
        "is_real": is_real,
        "combined_score": combined,
        "threshold": SPOOF_DECISION_THRESHOLD,
        "layers": layers,
        "weights_used": normalized_weights,
        "disagreements": disagreements,
        # TEMP PERF: additive-only key for docs/review/06-performance.md — remove after measurement
        "timings": {"fasnet_ms": _fasnet_ms, "onnx_ms": _onnx_ms},
    }


# F-16 (docs/review/10-moire-frr-investigation.md §16): combined_spoof_score's two
# early-return fail-close paths (Fasnet unavailable; all voting layers errored) mark
# themselves with these disagreement strings. Callers use this to tell "the anti-spoof
# system itself is broken" apart from "it ran and scored the frame as spoof" — the two
# must not produce the same user-facing message.
_SYSTEM_FAILURE_MARKERS = {"fasnet_unavailable_fail_close", "all_layers_failed"}


def is_system_failure(spoof_result: dict) -> bool:
    """True if combined_spoof_score's is_real=False came from an infra failure, not
    a spoof determination. Callers must keep this distinction server-side only
    (logs) — the user-facing message must not reveal the internal cause."""
    return not spoof_result["is_real"] and bool(
        _SYSTEM_FAILURE_MARKERS & set(spoof_result.get("disagreements", []))
    )


def normalize_illumination(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE to L-channel of LAB colorspace to normalize lighting."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _decode_image(base64_image: str) -> np.ndarray:
    """Decode base64 image string to BGR numpy array."""
    if "," in base64_image:
        base64_image = base64_image.split(",", 1)[1]
    img_bytes = base64.b64decode(base64_image)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("ไม่สามารถอ่านรูปภาพได้")
    return img


def extract_embedding(base64_image: str, include_metadata: bool = False):
    """
    Decode base64 image → CLAHE normalize → FaceNet512 embedding (512-D list).
    Raises ValueError if face not detected or image unreadable.
    """
    img = _decode_image(base64_image)
    img = normalize_illumination(img)

    embedding, (x, y, w, h) = _face_embedding(img)
    if include_metadata:
        return embedding, {"detector_crop": {"x": x, "y": y, "w": w, "h": h}}
    return embedding


def spoof_check_with_embedding(base64_image: str) -> dict:
    """
    Combined spoof detection + FaceNet512 embedding extraction.

    Returns dict with keys: is_real, confidence, combined_score,
    embedding (None if spoof or face not found), message, layers.

    Spoof detection runs first via combined_spoof_score (5 layers).
    Embedding is always attempted for audit but withheld from callers
    if spoof is detected or face extraction fails.
    """
    try:
        img = _decode_image(base64_image)
    except Exception as e:
        return {
            "is_real": False, "confidence": 0.0, "combined_score": 1.0,
            "embedding": None, "message": str(e), "layers": {},
        }

    spoof_result = combined_spoof_score(img)

    if spoof_result.get("retry_capture"):
        return {
            "is_real": False, "confidence": 0.0, "combined_score": 1.0,
            "embedding": None, "layers": spoof_result["layers"],
            "system_failure": False, "retry_capture": True,
            "message": "ไม่พบใบหน้าชัดเจน กรุณามองตรง จัดหน้าให้อยู่กลางกรอบ และเพิ่มแสงด้านหน้า",
        }

    embedding = None
    error_msg = ""
    try:
        embedding, _ = _face_embedding(normalize_illumination(img))
    except Exception as e:
        no_face = _log_face_exception("SPOOF_CHECK_EMBED", e) == "face_not_detected"
        error_msg = "" if no_face else "ไม่สามารถอ่านใบหน้าได้"

    confidence = 1.0 - spoof_result["combined_score"]

    if not spoof_result["is_real"]:
        return {
            "is_real": False,
            "confidence": round(confidence, 4),
            "combined_score": spoof_result["combined_score"],
            "embedding": None,
            "message": "ตรวจพบการปลอมแปลง",
            "layers": spoof_result["layers"],
            # F-16 (docs/review/10-moire-frr-investigation.md §16-17): lets callers
            # tell "anti-spoof system unavailable" apart from "scored as spoof".
            "system_failure": is_system_failure(spoof_result),
        }

    if embedding is None:
        if not error_msg:
            # No face for the embedding: a retake, not a spoof verdict (as FRR-2 at check-in).
            return {
                "is_real": False, "confidence": 0.0, "combined_score": spoof_result["combined_score"],
                "embedding": None, "layers": spoof_result["layers"],
                "system_failure": False, "retry_capture": True,
                "message": "ไม่พบใบหน้าชัดเจน กรุณามองตรง จัดหน้าให้อยู่กลางกรอบ และเพิ่มแสงด้านหน้า",
            }
        return {
            "is_real": False,
            "confidence": round(confidence, 4),
            "combined_score": spoof_result["combined_score"],
            "embedding": None,
            "message": error_msg,
            "layers": spoof_result["layers"],
        }

    return {
        "is_real": True,
        "confidence": round(confidence, 4),
        "combined_score": spoof_result["combined_score"],
        "embedding": embedding,
        "message": "",
        "layers": spoof_result["layers"],
    }


def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """Cosine similarity between two embedding vectors."""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def verify_face_multi(
    checkin_embedding: list,
    stored_embeddings: list,
    threshold: float,
) -> dict:
    """
    Compare checkin_embedding against all stored embeddings.
    Decision is based on best_similarity (max), not average.
    Returns dict with: verified, best_similarity, avg_similarity, matched_index.
    """
    if not stored_embeddings:
        return {"verified": False, "best_similarity": 0.0, "avg_similarity": 0.0, "matched_index": -1}

    similarities = [cosine_similarity(checkin_embedding, emb) for emb in stored_embeddings]
    best_sim  = max(similarities)
    avg_sim   = sum(similarities) / len(similarities)
    best_idx  = similarities.index(best_sim)

    return {
        "verified":        best_sim >= threshold,
        "best_similarity": round(best_sim, 4),
        "avg_similarity":  round(avg_sim, 4),
        "matched_index":   best_idx,
    }


def max_similarity_multi(live_emb: list, stored_embeddings: list) -> float:
    """Return highest cosine similarity between live_emb and any stored embedding."""
    if not stored_embeddings:
        return 0.0
    return max(cosine_similarity(live_emb, emb) for emb in stored_embeddings)


def server_validate_frame(frame_b64: str) -> dict:
    """
    Zero-trust frame validation — run BEFORE any DeepFace call (Sprint 2A).

    Checks (in order):
      1. Payload size: 3 KB – 500 KB
      2. JPEG magic bytes: FF D8 FF … FF D9
      3. Decodable to BGR image via OpenCV
      4. Resolution: 160×120 – 1920×1080
      5. Laplacian blur variance ≥ 8
      6. All color channel std-dev ≥ 2.0 (rejects solid-color / synthetic images)

    Returns {"valid": bool, "reason": str, "metadata": dict}
    """
    result: dict = {"valid": False, "reason": "", "metadata": {}}

    # Strip data-URL prefix if present
    b64_data = frame_b64.split(",", 1)[1] if "," in frame_b64 else frame_b64

    try:
        raw_bytes = base64.b64decode(b64_data)
    except Exception:
        result["reason"] = "base64_decode_failed"
        return result

    size_kb = len(raw_bytes) / 1024
    result["metadata"]["size_kb"] = round(size_kb, 1)

    if size_kb > 500:
        result["reason"] = "frame_too_large"
        return result
    if size_kb < 3:
        result["reason"] = "frame_too_small"
        return result

    # JPEG magic bytes
    if raw_bytes[:3] != b"\xff\xd8\xff":
        result["reason"] = "invalid_jpeg_header"
        return result
    if raw_bytes[-2:] != b"\xff\xd9":
        result["reason"] = "invalid_jpeg_footer"
        return result

    # Decode image
    img_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        result["reason"] = "decode_failed"
        return result

    h, w = frame.shape[:2]
    result["metadata"]["dimensions"] = f"{w}x{h}"

    if w > 1920 or h > 1080:
        result["reason"] = "resolution_too_high"
        return result
    if w < 160 or h < 120:
        result["reason"] = "resolution_too_low"
        return result

    # Blur check
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    result["metadata"]["laplacian_var"] = round(lap_var, 2)
    if lap_var < 8:   # relaxed from 20 — webcam video frames are inherently less sharp
        result["reason"] = "image_too_blurry"
        return result

    # Color naturalness (synthetic / solid images have near-zero channel std-dev)
    ch_stds = [round(float(np.std(frame[:, :, i])), 2) for i in range(3)]
    result["metadata"]["B_std"] = ch_stds[0]
    result["metadata"]["G_std"] = ch_stds[1]
    result["metadata"]["R_std"] = ch_stds[2]
    if min(ch_stds) < 2.0:   # relaxed from 5.0 — allows dim/uniform lighting environments
        result["reason"] = "unnaturally_uniform_color"
        return result

    result["valid"] = True
    result["reason"] = "passed"
    return result


def detect_static_image(frames: list, threshold: float = TEMPORAL_VAR_THRESHOLD) -> dict:
    """
    Measure temporal pixel variance, using a face ROI when Haar finds one.

    The threshold is a historical, uncalibrated comparison reference; no
    retained dataset establishes a separating range for face-cropped real and
    spoof bursts. This function reports the measurement/comparison only. Each
    caller is responsible for choosing audit-only or enforcement behavior.
    Falls back to full frame if no face detected.
    Returns { is_static: bool, temporal_variance: float }
    """
    if len(frames) < 2:
        return {"is_static": False, "temporal_variance": 0.0}

    # ── Detect face ROI from first frame (Haar cascade — bundled in OpenCV) ──
    first_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    with _cascade_lock:
        detected = _face_cascade.detectMultiScale(
            first_gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
        )
    crop = None
    if len(detected) > 0:
        x, y, w, h = detected[0]
        pad = int(min(w, h) * 0.20)
        h_img, w_img = frames[0].shape[:2]
        x1 = max(0, x - pad);        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad); y2 = min(h_img, y + h + pad)
        crop = (x1, y1, x2, y2)

    resized = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if crop:
            x1, y1, x2, y2 = crop
            gray = gray[y1:y2, x1:x2]
        resized.append(cv2.resize(gray, (64, 64)).astype(np.float32))

    stack = np.stack(resized, axis=0)
    mean_var = float(np.mean(np.std(stack, axis=0)))

    _audit.info(f"[TEMPORAL] temporal_variance={mean_var:.3f} threshold={threshold} face_crop={crop is not None}")
    return {
        "is_static":         mean_var < threshold,
        "temporal_variance": round(mean_var, 3),
    }


def check_embedding_consistency(embeddings: list, threshold: float = CONSISTENCY_THRESHOLD) -> dict:
    """
    Check pairwise cosine similarity for all C(n,2) pairs.
    Returns:
      consistent=True                          → all pairs pass
      consistent=False, multi_outlier=False    → single outlier index returned → need_more
      consistent=False, multi_outlier=True     → ≥2 outliers → restart capture entirely
    """
    n = len(embeddings)
    if n < 2:
        return {
            "consistent": True,
            "outlier_indices": [],
            "pairwise_scores": [],
            "average_similarities": [],
            "embedding_diagnostics": [],
            "multi_outlier": False,
        }

    embedding_diagnostics = []
    for idx, embedding in enumerate(embeddings):
        vector = np.asarray(embedding)
        embedding_diagnostics.append({
            "frame": idx + 1,
            "shape": list(vector.shape),
            "dtype": str(vector.dtype),
            "l2_norm": round(float(np.linalg.norm(vector.astype(np.float32))), 4),
        })
    sim_matrix = np.zeros((n, n), dtype=np.float32)
    pairwise = []
    for i in range(n):
        for j in range(i + 1, n):
            s = cosine_similarity(embeddings[i], embeddings[j])
            sim_matrix[i][j] = s
            sim_matrix[j][i] = s
            pairwise.append({"i": i, "j": j, "score": round(float(s), 4)})

    avg_sims = [float(np.sum(sim_matrix[i]) / (n - 1)) for i in range(n)]
    average_similarities = [
        {"frame": i + 1, "average": round(avg, 4)}
        for i, avg in enumerate(avg_sims)
    ]
    failing = [p for p in pairwise if p["score"] < threshold]
    if not failing:
        return {
            "consistent": True,
            "outlier_indices": [],
            "pairwise_scores": pairwise,
            "average_similarities": average_similarities,
            "embedding_diagnostics": embedding_diagnostics,
            "multi_outlier": False,
        }

    min_score = round(float(min(p["score"] for p in pairwise)), 4)

    # Identify all frames whose average similarity to others is below threshold
    bad_indices = [i for i, avg in enumerate(avg_sims) if avg < threshold]

    if len(bad_indices) > 1:
        # Multiple bad frames — cannot fix by replacing one; request full recapture
        return {
            "consistent":      False,
            "outlier_indices": bad_indices,
            "pairwise_scores": pairwise,
            "average_similarities": average_similarities,
            "embedding_diagnostics": embedding_diagnostics,
            "multi_outlier":   True,
            "min_score":       min_score,
        }

    # Single outlier — return the one frame with the lowest avg similarity
    outlier_idx = int(np.argmin(avg_sims))
    return {
        "consistent":      False,
        "outlier_indices": [outlier_idx],
        "pairwise_scores": pairwise,
        "average_similarities": average_similarities,
        "embedding_diagnostics": embedding_diagnostics,
        "multi_outlier":   False,
        "min_score":       min_score,
    }

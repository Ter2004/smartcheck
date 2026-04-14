import base64
import logging
import numpy as np
import cv2
import json

# ─── Thresholds (edit here to tune) ──────────────────────────────────────────
SELF_VERIFY_THRESHOLD    = 0.75   # enrollment self-verify (overridden to 0.80 in student.py)
SAME_DEVICE_THRESHOLD    = 0.70   # check-in, trusted device
NEW_DEVICE_THRESHOLD     = 0.80   # check-in, new / unbound device
CONSISTENCY_THRESHOLD    = 0.80   # pairwise consistency during enrollment
DUPLICATE_THRESHOLD      = 0.65   # reject if another student matches this closely
MOIRE_THRESHOLD          = 0.60   # high-freq energy ratio; above = likely screen replay (multi-frame /api/enroll)
MOIRE_THRESHOLD_SINGLE   = 0.72   # single-frame threshold for /api/spoof_check — more conservative (real face JPEG noise can score 0.50–0.58)
TEMPORAL_VAR_THRESHOLD   = 8.0   # face-ROI temporal std-dev; below = static photo
# Applied to face-crop only (not full frame) → real face ~15-25, static photo ~0.5-2.5
DUPLICATE_GRAY_ZONE      = (0.60, 0.70)  # log matches in this range for future tuning
MOIRE_LOG_RANGE          = (0.45, 0.75)  # log FFT scores near the threshold

_audit = logging.getLogger("smartcheck.enrollment")


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


def extract_embedding(base64_image: str) -> list:
    """
    Decode base64 image → CLAHE normalize → FaceNet512 embedding (512-D list).
    Raises ValueError if face not detected or image unreadable.
    """
    from deepface import DeepFace

    img = _decode_image(base64_image)
    img = normalize_illumination(img)

    result = DeepFace.represent(
        img_path=img,
        model_name="Facenet512",
        enforce_detection=True,
        detector_backend="opencv",
    )

    if not result:
        raise ValueError("ตรวจไม่เจอใบหน้าในรูป")

    return result[0]["embedding"]


def check_anti_spoof(base64_image: str) -> bool:
    """
    Liveness check via face detection (MiniFASNet/PyTorch unavailable on Railway).
    Returns True if a face is detected, False otherwise.
    Liveness is enforced by the blink check in the frontend.
    """
    from deepface import DeepFace

    img = _decode_image(base64_image)
    faces = DeepFace.extract_faces(
        img_path=img,
        anti_spoofing=False,
        detector_backend="opencv",
        enforce_detection=True,
    )
    return bool(faces)


def check_anti_spoof_with_score(base64_image: str) -> tuple:
    """
    Returns (is_real: bool, score: float).
    MiniFASNet unavailable (requires PyTorch); falls back to face detection.
    """
    from deepface import DeepFace

    img = _decode_image(base64_image)
    faces = DeepFace.extract_faces(
        img_path=img,
        anti_spoofing=False,
        detector_backend="opencv",
        enforce_detection=True,
    )
    is_real = bool(faces)
    _audit.debug(f"[ANTISPOOF] face_detected={is_real} (no MiniFASNet — PyTorch not available)")
    return is_real, 1.0 if is_real else 0.0


def spoof_check_with_embedding(base64_image: str) -> dict:
    """
    Face detection + FaceNet512 embedding extraction.
    MiniFASNet anti-spoof removed: requires PyTorch (~700MB) which exceeds
    Railway memory limit. Liveness is enforced by blink check in frontend.
    Returns:
        {
            "is_real":    bool,   # True if face detected
            "confidence": float,  # 1.0 if face found, 0.0 if not
            "embedding":  list | None,
            "message":    str,
        }
    """
    from deepface import DeepFace

    try:
        img = _decode_image(base64_image)
    except Exception as e:
        return {"is_real": False, "confidence": 0.0, "embedding": None, "message": str(e)}

    # Extract FaceNet512 embedding — raises ValueError if no face detected
    try:
        img_clahe = normalize_illumination(img)
        rep = DeepFace.represent(
            img_path=img_clahe,
            model_name="Facenet512",
            enforce_detection=True,
            detector_backend="opencv",
        )
        embedding = rep[0]["embedding"] if rep else None
    except ValueError:
        return {"is_real": False, "confidence": 0.0, "embedding": None, "message": "ไม่พบใบหน้า"}
    except Exception as e:
        _audit.error(f"[SPOOF_CHECK] embedding error: {e}", exc_info=True)
        return {"is_real": False, "confidence": 0.0, "embedding": None, "message": f"ตรวจสอบไม่สำเร็จ: {e}"}

    _audit.info(f"[SPOOF_CHECK] face detected, embedding extracted (no MiniFASNet)")
    return {
        "is_real":    True,
        "confidence": 1.0,
        "embedding":  embedding,
        "message":    "ใบหน้าตรวจพบ",
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


def _adaptive_moire_threshold(frames: list, base: float = MOIRE_THRESHOLD) -> float:
    """
    B8: Adjust Moiré threshold based on average frame brightness.

    Dark frames (<60 mean) → raise threshold by up to +0.05 to avoid false positives
    from JPEG compression noise amplified in low light.
    Bright frames (>180 mean) → lower threshold by up to -0.03 (screens glow brighter).
    """
    if not frames:
        return base
    brightness_vals = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_vals.append(float(np.mean(gray)))
    avg_brightness = sum(brightness_vals) / len(brightness_vals)

    if avg_brightness < 60:
        # Very dark — loosen threshold proportionally (max +0.05)
        adjust = 0.05 * (1.0 - avg_brightness / 60.0)
    elif avg_brightness > 180:
        # Very bright / screen-like — tighten threshold proportionally (max -0.03)
        adjust = -0.03 * ((avg_brightness - 180.0) / 75.0)
    else:
        adjust = 0.0

    result = round(max(0.50, min(0.75, base + adjust)), 4)
    if adjust != 0.0:
        _audit.debug(f"[MOIRE] adaptive threshold: brightness={avg_brightness:.1f} adjust={adjust:+.4f} threshold={result}")
    return result


def detect_screen_moire(frames: list, threshold: float | None = None) -> dict:
    """
    Detect screen replay attacks via FFT-based moiré analysis.
    Real faces have smooth frequency spectra; screens have periodic peaks from pixel grids.

    Args:
        frames: list of BGR numpy arrays (3-5 frames from enrollment or 1 from check-in)
        threshold: override threshold (None → use adaptive threshold based on brightness)
    Returns:
        { is_screen: bool, avg_score: float, per_frame: list[float] }
    """
    effective_threshold = threshold if threshold is not None else _adaptive_moire_threshold(frames)

    scores = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize to fixed size for consistent FFT results
        gray = cv2.resize(gray, (256, 256))

        f_transform = np.fft.fft2(gray.astype(np.float32))
        f_shift     = np.fft.fftshift(f_transform)
        magnitude   = np.abs(f_shift)

        h, w   = magnitude.shape
        cy, cx = h // 2, w // 2
        low_r  = int(min(h, w) * 0.10)  # centre 20% of spectrum = low-frequency

        low_mask = np.zeros_like(magnitude, dtype=bool)
        low_mask[cy - low_r:cy + low_r, cx - low_r:cx + low_r] = True

        total_energy = float(np.sum(magnitude))
        low_energy   = float(np.sum(magnitude[low_mask]))
        high_energy  = total_energy - low_energy

        ratio = high_energy / (total_energy + 1e-8)
        scores.append(ratio)

    avg_score = sum(scores) / len(scores)
    if MOIRE_LOG_RANGE[0] <= avg_score <= MOIRE_LOG_RANGE[1]:
        _audit.info(f"[MOIRE] near-threshold avg_score={avg_score:.4f} threshold={effective_threshold}")
    return {
        "is_screen":  avg_score > effective_threshold,
        "avg_score":  round(avg_score, 4),
        "per_frame":  [round(s, 4) for s in scores],
        "threshold":  effective_threshold,
    }


def detect_screen_texture(
    img_bgr: np.ndarray,
    peak_threshold_multiplier: float = 3.0,
    min_peaks: int = 50,
) -> bool:
    """
    Detect periodic pixel-grid pattern of phone/monitor screens via FFT peak counting.
    Complements Moiré FFT — catches high-res OLED screens that have low overall
    high-freq energy but still show periodic spikes.
    Returns True if a screen is detected.

    Tune min_peaks (default 50) by testing against real faces vs phone screens;
    OLED screens typically score 80-200+, real faces 5-30.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Resize to fixed 256×256 so peak count is resolution-independent
    gray = cv2.resize(gray, (256, 256))
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    h, w = magnitude.shape
    center_y, center_x = h // 2, w // 2
    mask_radius = min(h, w) // 4

    # Zero out low-frequency centre, keep only high-frequency ring
    high_freq = magnitude.copy()
    high_freq[
        center_y - mask_radius: center_y + mask_radius,
        center_x - mask_radius: center_x + mask_radius,
    ] = 0

    threshold = np.mean(high_freq) + peak_threshold_multiplier * np.std(high_freq)
    num_peaks = int(np.sum(high_freq > threshold))
    _audit.debug(f"[SCREEN_TEXTURE] num_peaks={num_peaks} threshold_multiplier={peak_threshold_multiplier}")
    return num_peaks > min_peaks


def server_validate_frame(frame_b64: str) -> dict:
    """
    Zero-trust frame validation — run BEFORE any DeepFace call (Sprint 2A).

    Checks (in order):
      1. Payload size: 3 KB – 500 KB
      2. JPEG magic bytes: FF D8 FF … FF D9
      3. Decodable to BGR image via OpenCV
      4. Resolution: 160×120 – 1920×1080
      5. Laplacian blur variance ≥ 20
      6. All color channel std-dev ≥ 5 (rejects solid-color / synthetic images)

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
    Detect static photo/replay by measuring pixel variance across the time axis.
    Crops face ROI first (Haar cascade) so background doesn't dilute the score.
    Real faces (face-only): breathing + micro-movements → std-dev ~15–25.
    Static photo (face-only): only JPEG noise → std-dev ~0.5–2.5.
    Falls back to full frame if no face detected.
    Returns { is_static: bool, temporal_variance: float }
    """
    if len(frames) < 2:
        return {"is_static": False, "temporal_variance": 0.0}

    # ── Detect face ROI from first frame (Haar cascade — bundled in OpenCV) ──
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    first_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    detected = face_cascade.detectMultiScale(
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
        return {"consistent": True, "outlier_indices": [], "pairwise_scores": [], "multi_outlier": False}

    sim_matrix = np.zeros((n, n), dtype=np.float32)
    pairwise = []
    for i in range(n):
        for j in range(i + 1, n):
            s = cosine_similarity(embeddings[i], embeddings[j])
            sim_matrix[i][j] = s
            sim_matrix[j][i] = s
            pairwise.append({"i": i, "j": j, "score": round(float(s), 4)})

    failing = [p for p in pairwise if p["score"] < threshold]
    if not failing:
        return {"consistent": True, "outlier_indices": [], "pairwise_scores": pairwise, "multi_outlier": False}

    min_score = round(float(min(p["score"] for p in pairwise)), 4)

    # Identify all frames whose average similarity to others is below threshold
    avg_sims = [float(np.sum(sim_matrix[i]) / (n - 1)) for i in range(n)]
    bad_indices = [i for i, avg in enumerate(avg_sims) if avg < threshold]

    if len(bad_indices) > 1:
        # Multiple bad frames — cannot fix by replacing one; request full recapture
        return {
            "consistent":      False,
            "outlier_indices": bad_indices,
            "pairwise_scores": pairwise,
            "multi_outlier":   True,
            "min_score":       min_score,
        }

    # Single outlier — return the one frame with the lowest avg similarity
    outlier_idx = int(np.argmin(avg_sims))
    return {
        "consistent":      False,
        "outlier_indices": [outlier_idx],
        "pairwise_scores": pairwise,
        "multi_outlier":   False,
        "min_score":       min_score,
    }

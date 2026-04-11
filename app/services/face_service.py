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
MOIRE_THRESHOLD          = 0.60   # high-freq energy ratio; above = likely screen replay
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
        detector_backend="retinaface",
    )

    if not result:
        raise ValueError("ตรวจไม่เจอใบหน้าในรูป")

    return result[0]["embedding"]


def check_anti_spoof(base64_image: str) -> bool:
    """
    Anti-spoofing via DeepFace MiniFASNet.
    Returns True = real face, False = spoof.
    Raises ValueError if image unreadable or no face detected.
    """
    from deepface import DeepFace

    img = _decode_image(base64_image)

    faces = DeepFace.extract_faces(
        img_path=img,
        anti_spoofing=True,
        detector_backend="retinaface",
        enforce_detection=True,
    )

    if not faces:
        raise ValueError("ตรวจไม่เจอใบหน้า")

    is_real = faces[0].get("is_real", False)
    score   = faces[0].get("antispoof_score", 0.0)
    print(f"[ANTISPOOF] is_real={is_real} score={score:.4f}")
    return bool(is_real)


def check_anti_spoof_with_score(base64_image: str) -> tuple:
    """
    Returns (is_real: bool, score: float).
    score near 1.0 = real face, near 0.0 = spoof.
    """
    from deepface import DeepFace

    img = _decode_image(base64_image)

    faces = DeepFace.extract_faces(
        img_path=img,
        anti_spoofing=True,
        detector_backend="retinaface",
        enforce_detection=True,
    )

    if not faces:
        raise ValueError("ตรวจไม่เจอใบหน้า")

    is_real = bool(faces[0].get("is_real", False))
    score   = float(faces[0].get("antispoof_score", 0.0))
    print(f"[ANTISPOOF] is_real={is_real} score={score:.4f}")
    return is_real, score


def spoof_check_with_embedding(base64_image: str) -> dict:
    """
    Combined anti-spoof + embedding extraction for inline enrollment checks.
    Runs MiniFASNet first; extracts FaceNet512 embedding only if face is real.
    Returns:
        {
            "is_real":    bool,
            "confidence": float,        # antispoof_score (higher = more real)
            "embedding":  list | None,  # 512-D vector if real, else None
            "message":    str,
        }
    """
    from deepface import DeepFace

    try:
        img = _decode_image(base64_image)
    except Exception as e:
        return {"is_real": False, "confidence": 0.0, "embedding": None, "message": str(e)}

    # Anti-spoof (includes face detection + MiniFASNet)
    try:
        faces = DeepFace.extract_faces(
            img_path=img,
            anti_spoofing=True,
            detector_backend="retinaface",
            enforce_detection=True,
        )
    except ValueError:
        return {"is_real": False, "confidence": 0.0, "embedding": None, "message": "ไม่พบใบหน้า"}
    except Exception as e:
        return {"is_real": False, "confidence": 0.0, "embedding": None, "message": f"ตรวจสอบไม่สำเร็จ: {e}"}

    is_real = bool(faces[0].get("is_real", False))
    score   = float(faces[0].get("antispoof_score", 0.0))
    print(f"[SPOOF_CHECK] is_real={is_real} score={score:.4f}")

    if not is_real:
        return {
            "is_real":    False,
            "confidence": round(score, 3),
            "embedding":  None,
            "message":    "ตรวจพบภาพปลอม กรุณาใช้ใบหน้าจริง",
        }

    # Extract FaceNet512 embedding (for face continuity check)
    try:
        img_clahe = normalize_illumination(img)
        rep = DeepFace.represent(
            img_path=img_clahe,
            model_name="Facenet512",
            enforce_detection=True,
            detector_backend="retinaface",
        )
        embedding = rep[0]["embedding"] if rep else None
    except Exception as e:
        print(f"[SPOOF_CHECK] embedding extraction failed (non-fatal): {e}")
        embedding = None

    return {
        "is_real":    True,
        "confidence": round(score, 3),
        "embedding":  embedding,
        "message":    "ใบหน้าจริง",
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


def detect_screen_moire(frames: list, threshold: float = MOIRE_THRESHOLD) -> dict:
    """
    Detect screen replay attacks via FFT-based moiré analysis.
    Real faces have smooth frequency spectra; screens have periodic peaks from pixel grids.

    Args:
        frames: list of BGR numpy arrays (3-5 frames from enrollment or 1 from check-in)
        threshold: high-freq energy ratio above which the image is flagged as a screen
    Returns:
        { is_screen: bool, avg_score: float, per_frame: list[float] }
    """
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
        _audit.info(f"[MOIRE] near-threshold avg_score={avg_score:.4f} threshold={threshold}")
    return {
        "is_screen": avg_score > threshold,
        "avg_score": round(avg_score, 4),
        "per_frame": [round(s, 4) for s in scores],
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

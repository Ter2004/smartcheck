"""Server checks of the v2 enrollment (FACE_ENGINE=arcface), in the order the route runs them.

frame validation → one face per frame → quality gates → anti-spoof on >= K of N
→ head turn matches the challenge → embeddings → pairwise self-consistency →
1:N duplicate check → keep 5 (3 frontal + 2 turned).

The browser sends up to 8 frames it scored itself (pose bucket, capture time
since it received the challenge, its own yaw reading). Nothing the browser says
decides anything: every frame is re-checked here. A frame that fails a quality
gate is dropped; the attempt fails when too few usable frames remain, with the
one fix that would have helped most.

Gate values are the ones the existing enrollment page already enforces
(enrollment_flow.js): face height >= 0.30 of the frame, brightness >= 55,
edge/centre brightness <= 2.2, Laplacian variance >= 20 at 640x480. Anti-spoof
keeps the existing 4-of-5 ratio. Similarity thresholds come from config
(ARCFACE_MATCH_THRESHOLD, ARCFACE_DUP_THRESHOLD, ENROLL_SELF_SIM_MIN).
"""
import math
import secrets
import time
from collections import Counter

import cv2
import numpy as np

from app.services import liveness_challenge as lc

MIN_FRAMES, MAX_FRAMES = 5, 8
KEEP_FRONTAL, KEEP_TURN = 3, 2
MIN_FACE_FRAC = 0.30        # enrollment_flow.js face-size gate (landmark span >= 0.30)
MIN_BRIGHTNESS = 55         # enrollment_flow.js _checkCameraConditions
MAX_BACKLIGHT = 2.2         # enrollment_flow.js _checkCameraConditions (edge / centre)
MIN_BLUR = 20               # enrollment_flow.js _checkBlur, Laplacian variance at 640x480
SPOOF_PASS_RATIO = 0.8      # the facenet512 flow's MIN_SPOOF_PASS = 4 of 5
TURNS = ("turn_left", "turn_right")
TURN_LABELS = {"turn_left": "ซ้าย", "turn_right": "ขวา"}

FIX = {
    "challenge":      "หมดเวลาหรือคำขอไม่ถูกต้อง — กรุณาเริ่มถ่ายใหม่",
    "bad_request":    "ข้อมูลไม่ครบ — กรุณาเริ่มถ่ายใหม่",
    "order":          "ลำดับการถ่ายไม่ถูกต้อง — กรุณาเริ่มถ่ายใหม่",
    "frame_invalid":  "ภาพเสียหรือไม่ชัด — กรุณาถ่ายใหม่",
    "no_face":        "ไม่พบใบหน้า — จัดหน้าให้อยู่ในกรอบวงรี",
    "multiple_faces": "พบใบหน้ามากกว่าหนึ่งคน — ให้มีเพียงคุณคนเดียวในกล้อง",
    "too_small":      "เข้าใกล้กล้องอีกหน่อย",
    "too_dark":       "แสงน้อยเกินไป — เปิดไฟหรือหันหน้าเข้าหาแสง",
    "backlight":      "แสงจ้าด้านหลังมากเกินไป — หันหน้าเข้าหาแสง",
    "blurry":         "ภาพเบลอ — ถือกล้องให้นิ่ง",
    "spoof":          "ตรวจพบรูปถ่ายหรือหน้าจอ — กรุณาใช้ใบหน้าจริงเท่านั้น",
    "not_frontal":    "ช่วงแรกให้มองตรงเข้ากล้อง",
    "inconsistent":   "ใบหน้าไม่ตรงกันระหว่างการถ่าย — อยู่ในกรอบคนเดียวตลอดแล้วลองใหม่",
    "duplicate":      "ไม่สามารถลงทะเบียนได้ กรุณาติดต่ออาจารย์",
}


class EnrollFail(Exception):
    """One attempt failed. `code` is logged; `fix` is the single message the student sees."""

    def __init__(self, code, fix=None, http=400, retry=True, system=False, detail=None):
        super().__init__(code)
        self.code, self.http, self.retry, self.system, self.detail = code, http, retry, system, detail
        self.fix = fix or FIX.get(code, FIX["bad_request"])


def new_challenge(now=None, rng=None):
    """One random turn direction, bound to a nonce, for the v2 camera session."""
    rng = rng or secrets.SystemRandom()
    return {"nonce": secrets.token_urlsafe(16), "direction": rng.choice(TURNS),
            "issued_at": time.time() if now is None else now}


def parse_frames(data, challenge, now=None):
    """Validate the request shape and the challenge; return [{image, t, pose, client_yaw}]."""
    now = time.time() if now is None else now
    if not challenge:
        raise EnrollFail("challenge")
    if not isinstance(data, dict) or not isinstance(data.get("challenge_id"), str) \
            or not secrets.compare_digest(data["challenge_id"], challenge["nonce"]):
        raise EnrollFail("challenge")
    if not 0 <= now - challenge["issued_at"] <= lc.CHALLENGE_TTL_S:
        raise EnrollFail("challenge")
    raw = data.get("frames")
    if not isinstance(raw, list) or not MIN_FRAMES <= len(raw) <= MAX_FRAMES:
        raise EnrollFail("bad_request")
    frames = []
    for f in raw:
        if not isinstance(f, dict) or not isinstance(f.get("image"), str) or f.get("pose") not in ("frontal", "turn"):
            raise EnrollFail("bad_request")
        try:
            t = float(f.get("t"))
        except (TypeError, ValueError):
            raise EnrollFail("bad_request") from None
        if not math.isfinite(t) or not 0 <= t <= lc.CHALLENGE_TTL_S * 1000:
            raise EnrollFail("order")
        try:
            client_yaw = float(f["client_yaw"]) if f.get("client_yaw") is not None else None
        except (TypeError, ValueError):
            client_yaw = None
        frames.append({"image": f["image"], "t": t, "pose": f["pose"], "client_yaw": client_yaw})
    # Capture order: all frontal frames are taken before the turn starts.
    frontal_t = [f["t"] for f in frames if f["pose"] == "frontal"]
    turn_t = [f["t"] for f in frames if f["pose"] == "turn"]
    if not frontal_t or not turn_t or max(frontal_t) > min(turn_t):
        raise EnrollFail("order")
    return frames


def quality(img):
    """(failed gate or None, metrics) with the enrollment page's own gates, on 640x480."""
    small = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)   # uint8, as server_validate_frame uses
    lum = cv2.resize(gray, (320, 240), interpolation=cv2.INTER_AREA).astype(np.float32)
    brightness = float(lum.mean())
    centre = lum[60:180, 80:240]
    edge_mean = (lum.sum() - centre.sum()) / (lum.size - centre.size)
    backlight = float(edge_mean / max(float(centre.mean()), 1e-6))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    metrics = {"brightness": round(brightness, 1), "backlight": round(backlight, 2), "blur": round(blur, 1)}
    if brightness < MIN_BRIGHTNESS:
        return "too_dark", metrics
    if backlight > MAX_BACKLIGHT:
        return "backlight", metrics
    if blur < MIN_BLUR:
        return "blurry", metrics
    return None, metrics


def _most_common(reasons, default):
    return Counter(reasons).most_common(1)[0][0] if reasons else default


def check(frames, challenge, deps, thresholds, log=lambda *a: None):
    """Run every server check. Returns {"kept": [5 frames], "thumbnail": frame}; raises EnrollFail.

    deps: validate(b64)->{"valid"}, decode(b64)->BGR, analyze(BGR)->face dict,
          spoof(BGR)->combined_spoof_score dict, is_system_failure(dict)->bool,
          match(embedding)->[(student_id, similarity)].
    """
    from app.services import face_engine

    direction_wanted = challenge["direction"]
    dropped = []

    # 1. Frame validation
    usable = []
    for i, f in enumerate(frames):
        if not deps["validate"](f["image"])["valid"]:
            dropped.append("frame_invalid"); log("frame", f"i={i} drop=frame_invalid"); continue
        try:
            f["img"] = deps["decode"](f["image"])
        except Exception:
            dropped.append("frame_invalid"); continue
        usable.append(f)

    # 2. One face per frame (a second person anywhere fails the attempt)
    faced = []
    for f in usable:
        try:
            f["face"] = deps["analyze"](f["img"])
        except face_engine.MultipleFacesError:
            raise EnrollFail("multiple_faces") from None
        except face_engine.NoFaceError:
            dropped.append("no_face"); continue
        faced.append(f)

    # 3. Quality gates
    good = []
    for f in faced:
        h = f["img"].shape[0]
        if f["face"]["face_px"] < MIN_FACE_FRAC * h:
            dropped.append("too_small"); continue
        failed, metrics = quality(f["img"])
        f["quality"] = metrics
        if failed:
            dropped.append(failed); continue
        good.append(f)
    if len(good) < KEEP_FRONTAL + KEEP_TURN:
        raise EnrollFail(_most_common(dropped, "no_face"), detail={"dropped": dropped})

    # 4. Anti-spoof on >= K of N
    passed, system = 0, False
    for f in good:
        try:
            result = deps["spoof"](f["img"])
        except Exception:
            system = True; continue
        f["spoof_ok"] = bool(result["is_real"])
        if result["is_real"]:
            passed += 1
        elif deps["is_system_failure"](result):
            system = True
    needed = math.ceil(SPOOF_PASS_RATIO * len(good))
    log("antispoof", f"passed={passed}/{len(good)} needed={needed} system_failure={system}")
    if passed < needed:
        if system:
            raise EnrollFail("system", "ระบบตรวจสอบใบหน้าขัดข้องชั่วคราว กรุณาลองใหม่อีกครั้ง",
                             http=503, retry=False, system=True)
        raise EnrollFail("spoof")
    good = [f for f in good if f.get("spoof_ok")]

    # 5. Head turn matches the challenge (yaw in degrees, calibrated limits)
    frontal, turned = [], []
    for f in good:
        seen = lc.direction(lc.ARCFACE_YAW_SIGN * f["face"]["yaw_deg"], "deg")
        f["seen"] = seen
        log("turn_yaw", f"pose={f['pose']} client_yaw={f['client_yaw']} server_yaw_deg={f['face']['yaw_deg']:.1f} seen={seen}")
        if f["pose"] == "frontal" and seen == "frontal":
            frontal.append(f)
        elif f["pose"] == "turn":
            if seen in TURNS and seen != direction_wanted:
                raise EnrollFail("wrong_direction", f"หันผิดทาง — กรุณาหันหน้าไปทาง{TURN_LABELS[direction_wanted]}")
            if seen == direction_wanted:
                turned.append(f)
    if len(frontal) < KEEP_FRONTAL:
        raise EnrollFail("not_frontal")
    if len(turned) < KEEP_TURN:
        raise EnrollFail("turn_missing", f"หันหน้าไปทาง{TURN_LABELS[direction_wanted]}ให้มากขึ้นตามคำสั่ง")

    # 6-7. Embeddings and self-consistency on the best frames
    rank = lambda f: (f["face"]["det_score"], f["face"]["face_px"], f["quality"]["blur"])
    frontal = sorted(frontal, key=rank, reverse=True)[:KEEP_FRONTAL]
    turned = sorted(turned, key=rank, reverse=True)[:KEEP_TURN]
    fe = [f["face"]["embedding"] for f in frontal]
    pair_scores = [float(fe[a] @ fe[b]) for a in range(len(fe)) for b in range(a + 1, len(fe))]
    log("self_consistency", f"frontal_pairs={[round(s, 4) for s in pair_scores]} min={thresholds['ENROLL_SELF_SIM_MIN']}")
    if min(pair_scores) < thresholds["ENROLL_SELF_SIM_MIN"]:
        raise EnrollFail("inconsistent")
    for f in turned:
        best = max(float(f["face"]["embedding"] @ e) for e in fe)
        log("turn_identity", f"best={best:.4f} min={thresholds['ARCFACE_MATCH_THRESHOLD']}")
        if best < thresholds["ARCFACE_MATCH_THRESHOLD"]:
            raise EnrollFail("inconsistent")

    kept = frontal + turned

    # 8. 1:N duplicate check (other students, same model)
    for f in kept:
        for other, similarity in deps["match"](f["face"]["embedding"]):
            if similarity >= thresholds["ARCFACE_DUP_THRESHOLD"]:
                raise EnrollFail("duplicate", retry=False, detail={"matched_student": other,
                                                                   "similarity": round(similarity, 4)})
    return {"kept": kept, "thumbnail": frontal[0]}

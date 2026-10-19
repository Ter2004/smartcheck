"""Server-verified head-turn challenge for enrollment and check-in liveness.

The browser performs the gestures, but the server decides whether they happened.
A challenge is a random pick of distinct gestures from ACTIONS (turn_left /
turn_right / tilt_left / tilt_right) — two for enrollment, one for check-in —
bound to a one-time nonce. The client returns one frame taken
before the gestures, one frame per
gesture (captured when the browser detector reports it done) and one after.
The server then checks from the face landmarks (InsightFace landmark_3d_68 yaw,
SCRFD eye line) that each gesture frame shows the head turned or tilted the
requested way, and from ArcFace embeddings that every frame shows the same face
as the frontal reference.

Conventions on the raw (unmirrored) camera frame, as in mediapipe_liveness.js:
turn_left moves the nose toward the image's right; tilt_left (left ear toward
left shoulder) lowers the eye on the image's right. Yaw is measured along the
eye line so a tilt is not read as a turn, and tilt is the roll change from the
frontal "before" frame so a phone held at an angle does not count.

Limits: this stops a single still image (photo, AI face, mannequin, statue)
sent straight to the API. It does not stop a client that submits several
pre-made images of one face in the right poses, or a live deepfake.
"""
import secrets
import time

import numpy as np

from app.services import face_service

# Tilts (calibrated with scripts/calibrate_tilt.py) were off from 2026-09-29 and
# re-enabled 2026-09-30: with turns only, enrollment always asked for both turns,
# so posed photos of the victim in two known poses passed (B2 simulation).
TILT_ACTIONS = ("tilt_left", "tilt_right")
ACTIONS = ("turn_left", "turn_right") + TILT_ACTIONS
ENROLL_ACTIONS = 2
CHECKIN_ACTIONS = 1
CHALLENGE_TTL_S = 120
# Yaw is the nose's horizontal offset from the eye midpoint, in inter-eye
# distances. On LFW, 72% of photos fall within 0.15; the browser accepts a turn
# at noseRelX 0.62, which lands around 0.25 or more on this scale.
FRONTAL_MAX_YAW = 0.18
TURN_MIN_YAW = 0.20
# FaceNet cosine on RetinaFace crops. Turned-vs-frontal is cross-pose, so it is
# lower than CONTINUITY_THRESHOLD. LFW dev-test, turned (|yaw|>=0.20) vs frontal
# photos of the same / different people (n=128 / 136, 2026-09-24): at 0.45,
# 4.7% same-person pairs fall below and 0.7% different-person pairs reach it.
# LFW photos are taken on different days; frames seconds apart score higher.
TURN_IDENTITY_MIN = 0.45
# Roll change from the before frame (degrees) that counts as a tilt. LFW photos
# rotated by known angles (scripts/calibrate_tilt.py, 60 people x 11 angles,
# 2026-09-29): median roll error < 1 degree, single photos off by up to ~6
# degrees (small LFW faces); a 20-degree rotation read no lower than 13.5. The
# browser asks for 18 degrees, so 10 leaves margin for that noise while a
# 5-degree wobble (read 0.9-7.5) does not count.
TILT_MIN_DEG = 10.0
# Yaw (eye-aligned) a tilt frame may show before it counts as a turn instead.
# Eye-aligned yaw does not grow with tilt (p90 0.21-0.27 at 0-25 degrees,
# while the old image-axis yaw reached 0.45 at 25 degrees).
TILT_MAX_YAW = 0.35
FRONTAL_IDENTITY_MIN = face_service.CONTINUITY_THRESHOLD
MIN_FACE_SCORE = 0.9
# Landmark detection resolution (px, longest side), without RetinaFace's default
# upscaling to ~1024 px (750 ms -> ~120 ms per 640x480 frame on CPU). Measured on
# 60 LFW faces placed on 640x480 frames: yaw differs from full resolution by at
# most 0.11 at 480 px; 320 px produced a 1.2 outlier.
DETECT_MAX_SIDE = 480
# A second face at least this fraction of the main face's area fails the frame.
SECOND_FACE_AREA_RATIO = 0.5

# arcface engine: the same limits in landmark_3d_68 yaw degrees, translated from
# the ratio limits above on 1,461 LFW faces measured both ways
# (scripts/calibrate_arcface_yaw.py, docs/face-stack-v2/arcface-yaw-calibration.json):
# corr 0.953, least-squares slope 102.1 deg per ratio unit, same sign. Medians of
# faces near each ratio limit read 19.0 / 20.5 / 33.8 degrees.
ARCFACE_YAW_SIGN = 1.0
ARCFACE_FRONTAL_MAX_YAW_DEG = 18.4   # 0.18
ARCFACE_TURN_MIN_YAW_DEG = 20.4      # 0.20
ARCFACE_TILT_MAX_YAW_DEG = 35.7      # 0.35


class FrameError(ValueError):
    """A frame cannot be evaluated (no face, several faces)."""


def new_challenge(now=None, rng=None, count=ENROLL_ACTIONS):
    """`count` distinct gestures in random order (enrollment: 2; check-in: 1)."""
    rng = rng or secrets.SystemRandom()
    actions = rng.sample(ACTIONS, count)
    return {
        "nonce": secrets.token_urlsafe(16),
        "actions": actions,
        "issued_at": time.time() if now is None else now,
    }


def _eyes(landmarks):
    """(image-left eye, image-right eye), whatever the library calls them."""
    a, b = (np.asarray(landmarks[k], dtype=float)[:2] for k in ("right_eye", "left_eye"))
    if np.hypot(*(b - a)) < 1e-6:
        raise FrameError("degenerate_landmarks")
    return (a, b) if a[0] <= b[0] else (b, a)


def yaw_ratio(landmarks):
    """Signed nose offset from the eye midpoint along the eye line, in eye distances.

    Positive = nose toward the image's right = turn_left in the browser's terms.
    Measured along the eye line, so rolling the head does not change it.
    """
    left, right = _eyes(landmarks)
    nose = np.asarray(landmarks["nose"], dtype=float)[:2]
    axis = right - left
    return float(np.dot(nose - (left + right) / 2, axis) / np.dot(axis, axis))


def roll_deg(landmarks):
    """Eye-line angle in degrees; positive = the image-right eye is lower."""
    left, right = _eyes(landmarks)
    return float(np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0])))


def _yaw_limits(unit):
    """(frontal max, turn min, tilt max) on the frame's yaw scale."""
    if unit == "deg":
        return ARCFACE_FRONTAL_MAX_YAW_DEG, ARCFACE_TURN_MIN_YAW_DEG, ARCFACE_TILT_MAX_YAW_DEG
    return FRONTAL_MAX_YAW, TURN_MIN_YAW, TILT_MAX_YAW


def direction(yaw, unit="ratio"):
    frontal_max, turn_min, _ = _yaw_limits(unit)
    if abs(yaw) <= frontal_max:
        return "frontal"
    if yaw >= turn_min:
        return "turn_left"
    if yaw <= -turn_min:
        return "turn_right"
    return "partial"


def _area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _analyze_frame_arcface(img_bgr):
    """arcface engine: yaw in degrees from landmark_3d_68, roll from the SCRFD eye line.

    Yaw is signed like yaw_ratio (positive = nose toward the image's right =
    turn_left). ArcFace embeds an aligned crop, so no extra levelling is needed.
    """
    from app.services import face_engine

    try:
        face = face_engine.analyze(img_bgr)
    except face_engine.MultipleFacesError:
        raise FrameError("multiple_faces") from None
    except face_engine.NoFaceError:
        raise FrameError("no_face") from None
    return {"yaw": ARCFACE_YAW_SIGN * face["yaw_deg"], "roll": face["eye_roll_deg"],
            "embedding": face["embedding"], "unit": "deg"}


def analyze_frame(img_bgr):
    """Return {"yaw", "roll", "embedding", "unit"} for the single main face in a BGR frame.

    InsightFace since face stack v2: yaw in degrees (landmark_3d_68), roll from
    the SCRFD eye line, ArcFace embedding. The RetinaFace/FaceNet512 reader and
    its ratio-scale yaw were removed with TensorFlow; the ratio thresholds stay
    as the calibration reference of the degree limits.
    """
    return _analyze_frame_arcface(img_bgr)


def gesture(expected, frame, before):
    """What an action frame shows, in the terms of `expected`'s kind."""
    unit = frame.get("unit", "ratio")
    if expected.startswith("tilt_"):
        if abs(frame["yaw"]) > _yaw_limits(unit)[2]:
            return "turned"
        delta = frame["roll"] - before["roll"]
        if delta >= TILT_MIN_DEG:
            return "tilt_left"
        if delta <= -TILT_MIN_DEG:
            return "tilt_right"
        return "level"
    return direction(frame["yaw"], unit)


def _identity_minimums(unit):
    """(frontal after-frame minimum, other frames minimum) for the engine's embeddings."""
    if unit == "deg":
        # Same person, seconds apart: the arcface check-in match threshold
        # (ARCFACE_MATCH_THRESHOLD, chosen with scripts/compare_models.py).
        from app.services import face_engine
        match = face_engine.threshold("ARCFACE_MATCH_THRESHOLD")
        return match, match
    return FRONTAL_IDENTITY_MIN, TURN_IDENTITY_MIN


def verify(challenge, nonce, before, action_frames, after, now=None, analyze=analyze_frame):
    """Check a completed challenge. Frames are BGR arrays.

    Returns {"passed": bool, "reason": str, "yaws": list, "rolls": list, "scores": dict}.
    The caller must discard the challenge before calling (one attempt each).
    """
    now = time.time() if now is None else now
    result = {"passed": False, "reason": "", "yaws": [], "rolls": [], "scores": {}}

    def fail(reason):
        result["reason"] = reason
        return result

    if not challenge:
        return fail("no_challenge")
    if not isinstance(nonce, str) or not secrets.compare_digest(nonce, challenge["nonce"]):
        return fail("nonce_mismatch")
    if not 0 <= now - challenge["issued_at"] <= CHALLENGE_TTL_S:
        return fail("expired")
    if len(action_frames) != len(challenge["actions"]):
        return fail("frame_count")

    named = [("before", before)] + [(f"action_{i + 1}", f) for i, f in enumerate(action_frames)] + [("after", after)]
    analyzed = {}
    for name, frame in named:
        try:
            analyzed[name] = analyze(frame)
        except FrameError as e:
            return fail(f"{e}:{name}")
        result["yaws"].append(round(analyzed[name]["yaw"], 3))
        result["rolls"].append(round(analyzed[name]["roll"], 1))

    unit = analyzed["before"].get("unit", "ratio")
    if direction(analyzed["before"]["yaw"], unit) != "frontal":
        return fail("not_frontal:before")
    for i, expected in enumerate(challenge["actions"]):
        seen = gesture(expected, analyzed[f"action_{i + 1}"], analyzed["before"])
        if seen != expected:
            return fail(f"wrong_direction:action_{i + 1}:{seen}")

    reference = analyzed["before"]["embedding"]
    for name in [n for n, _ in named[1:]]:
        score = float(np.dot(reference, analyzed[name]["embedding"]))
        result["scores"][name] = round(score, 4)
        needed = FRONTAL_IDENTITY_MIN if name == "after" and direction(analyzed[name]["yaw"]) == "frontal" \
            else TURN_IDENTITY_MIN
        if score < needed:
            return fail(f"identity_mismatch:{name}")

    result["passed"] = True
    result["reason"] = "passed"
    return result


def decode_frame(data_url):
    """Decode a data-URL/base64 JPEG to BGR (callers validate with server_validate_frame first)."""
    return face_service._decode_image(data_url)

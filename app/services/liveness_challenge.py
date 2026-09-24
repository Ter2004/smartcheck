"""Server-verified head-turn challenge for enrollment liveness.

The browser performs the gestures, but the server decides whether they happened.
A challenge is a random order of turn_left / turn_right bound to a one-time
nonce. The client returns one frame taken before the gestures, one frame per
gesture (captured when the browser detector reports it done) and one after.
The server then checks from RetinaFace landmarks that each gesture frame shows
the head turned the requested way, and from FaceNet embeddings that every frame
shows the same face as the frontal reference.

Direction convention matches mediapipe_liveness.js noseRelX on the raw
(unmirrored) camera frame: turn_left moves the nose toward the image's right.

Limits: this stops a single still image (photo, AI face, mannequin, statue)
sent straight to the API. It does not stop a client that submits several
pre-made images of one face in the right poses, or a live deepfake.
"""
import secrets
import time

import numpy as np

from app.services import face_service

ACTIONS = ("turn_left", "turn_right")
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
FRONTAL_IDENTITY_MIN = face_service.CONTINUITY_THRESHOLD
MIN_FACE_SCORE = 0.9
# A second face at least this fraction of the main face's area fails the frame.
SECOND_FACE_AREA_RATIO = 0.5


class FrameError(ValueError):
    """A frame cannot be evaluated (no face, several faces)."""


def new_challenge(now=None, rng=None):
    rng = rng or secrets.SystemRandom()
    actions = list(ACTIONS)
    rng.shuffle(actions)
    return {
        "nonce": secrets.token_urlsafe(16),
        "actions": actions,
        "issued_at": time.time() if now is None else now,
    }


def yaw_ratio(landmarks):
    """Signed nose offset from the eye midpoint, in inter-eye distances.

    Positive = nose toward the image's right = turn_left in the browser's terms.
    """
    right_eye, left_eye, nose = (np.asarray(landmarks[k], dtype=float)[:2]
                                 for k in ("right_eye", "left_eye", "nose"))
    eye_distance = abs(left_eye[0] - right_eye[0])
    if eye_distance < 1e-6:
        raise FrameError("degenerate_landmarks")
    return float((nose[0] - (left_eye[0] + right_eye[0]) / 2) / eye_distance)


def direction(yaw):
    if abs(yaw) <= FRONTAL_MAX_YAW:
        return "frontal"
    if yaw >= TURN_MIN_YAW:
        return "turn_left"
    if yaw <= -TURN_MIN_YAW:
        return "turn_right"
    return "partial"


def _area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def analyze_frame(img_bgr):
    """Return {"yaw", "embedding"} for the single main face in a BGR frame."""
    from retinaface import RetinaFace

    # One TF model at a time per process, like the other face calls.
    with face_service._deepface_lock:
        faces = RetinaFace.detect_faces(img_bgr, threshold=MIN_FACE_SCORE)
    if not isinstance(faces, dict) or not faces:
        raise FrameError("no_face")
    ranked = sorted(faces.values(), key=lambda f: _area(f["facial_area"]), reverse=True)
    main = ranked[0]
    if len(ranked) > 1 and _area(ranked[1]["facial_area"]) >= SECOND_FACE_AREA_RATIO * _area(main["facial_area"]):
        raise FrameError("multiple_faces")

    x1, y1, x2, y2 = main["facial_area"]
    mx, my = int((x2 - x1) * 0.1), int((y2 - y1) * 0.1)
    h, w = img_bgr.shape[:2]
    crop = img_bgr[max(0, y1 - my):min(h, y2 + my), max(0, x1 - mx):min(w, x2 + mx)]
    crop = face_service.normalize_illumination(crop)
    rep = face_service._call_deepface("represent", img_path=crop,
                                      model_name="Facenet512", detector_backend="skip")
    embedding = np.asarray(rep[0]["embedding"], dtype=np.float32)
    return {"yaw": yaw_ratio(main["landmarks"]),
            "embedding": embedding / (np.linalg.norm(embedding) + 1e-12)}


def verify(challenge, nonce, before, action_frames, after, now=None, analyze=analyze_frame):
    """Check a completed challenge. Frames are BGR arrays.

    Returns {"passed": bool, "reason": str, "yaws": list, "scores": dict}.
    The caller must discard the challenge before calling (one attempt each).
    """
    now = time.time() if now is None else now
    result = {"passed": False, "reason": "", "yaws": [], "scores": {}}

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

    if direction(analyzed["before"]["yaw"]) != "frontal":
        return fail("not_frontal:before")
    for i, expected in enumerate(challenge["actions"]):
        seen = direction(analyzed[f"action_{i + 1}"]["yaw"])
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

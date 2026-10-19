"""The face work each route does per request, for scripts/bench_face_pipeline.py.

Face stack v2 (InsightFace buffalo_l + Fasnet ONNX). Each function makes the
same service calls as the route, in the same order, minus Supabase and session
handling, and without the early exits, so every step is always timed.
The legacy (FaceNet512/DeepFace) pipeline and its numbers are in commit d2dc68b
(scripts/face_pipelines.py) and docs/face-stack-v2/bench-phase0-legacy.json.
"""
import numpy as np

from app.services import enroll_v2, face_engine
from app.services import face_service as fs
from app.services import liveness_challenge as lc

PIPELINE_NAME = "arcface (InsightFace buffalo_l) + Fasnet ONNX anti-spoof; check-in passive"
_STORED = []


def warm_up(frames):
    """Load every model once (first calls are not timed)."""
    face_engine.warm_up()
    raw = fs._decode_image(frames[0])
    fs.combined_spoof_score(raw)
    face = face_engine.analyze(raw, clahe=False)
    _STORED[:] = [face["embedding"]] * 5   # stands in for the 5 enrolled templates


def enrollment(frames):
    """/student/api/enroll (v2) on 8 frames: validate, decode, analyze, quality, anti-spoof, consistency.

    The 1:N duplicate check is a database call (match_face_templates) and is not timed here.
    """
    eight = (frames * 8)[:8]
    faces = []
    for f in eight:
        fs.server_validate_frame(f)
        img = fs._decode_image(f)
        face = face_engine.analyze(img, clahe=False)
        enroll_v2.quality(img)
        fs.combined_spoof_score(img)
        lc.direction(lc.ARCFACE_YAW_SIGN * face["yaw_deg"], "deg")
        faces.append(face)
    embs = [f["embedding"] for f in faces[:5]]
    _ = [float(embs[a] @ embs[b]) for a in range(3) for b in range(a + 1, 3)]
    _ = [max(float(e @ x) for x in embs[:3]) for e in embs[3:]]
    return embs


def checkin(frames):
    """/api/checkin, CHECKIN_LIVENESS=passive: validate, anti-spoof, embedding, match."""
    face_image = frames[0]
    fs.server_validate_frame(face_image)
    raw = fs._decode_image(face_image)
    fs.combined_spoof_score(raw)
    live = face_engine.analyze(raw, clahe=False)["embedding"]
    return max(float(live @ t) for t in _STORED)


def checkin_head_turn(frames):
    """/api/checkin in head_turn mode on v2 (same 3 frames as the legacy baseline)."""
    face_image = frames[0]
    fs.server_validate_frame(face_image)
    raw = fs._decode_image(face_image)
    fs.combined_spoof_score(raw)
    extra = [fs._decode_image(f) for f in (frames[1:] + frames)[:2]]
    analyzed = [face_engine.analyze(img, clahe=False) for img in [raw] + extra]
    _ = [float(analyzed[0]["embedding"] @ a["embedding"]) for a in analyzed[1:]]
    return max(float(analyzed[0]["embedding"] @ t) for t in _STORED)

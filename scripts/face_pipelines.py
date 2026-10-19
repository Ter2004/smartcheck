"""The face work each route does per request, for scripts/bench_face_pipeline.py.

Each function makes the same service calls as the route, in the same order,
minus Supabase and session handling. Liveness frames go through analyze_frame
directly so a frame that would fail early is still fully timed.
"""
from app.services import face_service as fs
from app.services import liveness_challenge as lc

_STORED = []

PIPELINE_NAME = "facenet512 (DeepFace) + RetinaFace + Fasnet/ONNX anti-spoof, head-turn liveness"


def warm_up(frames):
    """Load every model once (first calls are not timed)."""
    raw = fs._decode_image(frames[0])
    fs.combined_spoof_score(raw)
    fs.extract_embedding(frames[0])
    lc.analyze_frame(raw)
    _STORED[:] = [fs.extract_embedding(frames[0])] * 5   # stands in for the 5 enrolled embeddings


def enrollment(frames):
    """/student/api/liveness/verify (2 gestures) + 5 x /api/spoof_check + /api/enroll."""
    five = (frames * 5)[:5]
    # Liveness verify: before, 2 gesture frames, after; then 2 continuity references.
    gesture = [lc.decode_frame(f) for f in (frames * 4)[:4]]
    for f in (frames * 4)[:4]:
        fs.server_validate_frame(f)
    analyzed = [lc.analyze_frame(img) for img in gesture]
    _ = [float(analyzed[0]["embedding"] @ a["embedding"]) for a in analyzed[1:]]
    refs = [fs.extract_embedding(frames[0]), fs.extract_embedding(frames[-1])]
    # Step-4 capture: one spoof check (+ embedding) per frame.
    for f in five:
        fs.server_validate_frame(f)
        fs.spoof_check_with_embedding(f)
    # /api/enroll
    for f in five:
        fs.server_validate_frame(f)
    raw = [fs._decode_image(f) for f in five]
    fs.detect_static_image(raw)
    for img in raw:
        fs.combined_spoof_score(img)
    embeddings = [fs.extract_embedding(f) for f in five]
    fs.check_embedding_consistency(embeddings, threshold=fs.CONSISTENCY_THRESHOLD)
    _ = [max(fs.cosine_similarity(e, r) for r in refs) for e in embeddings]
    return embeddings


def checkin(frames):
    """/api/checkin in head_turn mode: frontal frame + 1 gesture frame + after frame."""
    face_image = frames[0]
    fs.server_validate_frame(face_image)
    raw = fs._decode_image(face_image)
    fs.combined_spoof_score(raw)
    extra = [lc.decode_frame(f) for f in (frames[1:] + frames)[:2]]
    analyzed = [lc.analyze_frame(img) for img in [raw] + extra]
    _ = [float(analyzed[0]["embedding"] @ a["embedding"]) for a in analyzed[1:]]
    live = fs.extract_embedding(face_image)
    return fs.verify_face_multi(live, _STORED, fs.NEW_DEVICE_THRESHOLD)

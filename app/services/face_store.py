"""Reads and writes of the v2 face tables (face_enrollments, face_templates).

All calls go through the service-role client, the only role with access
(database/migrations/20261019_face_stack_v2.sql). Embeddings travel as
pgvector text literals ("[0.01,...]") and come back the same way.
"""
import json

import numpy as np


def vector_literal(embedding):
    """pgvector text form of a 512-D embedding, float32 like the column stores."""
    return "[" + ",".join(repr(float(v)) for v in np.asarray(embedding, dtype=np.float32).ravel()) + "]"


def parse_vector(value):
    """A pgvector value from PostgREST (text "[...]" or a list) as a float32 array."""
    if isinstance(value, str):
        value = json.loads(value)
    return np.asarray(value, dtype=np.float32)


def load_enrollment(client, student_id):
    """The student's v2 enrollment with its templates, or None."""
    res = (client.table("face_enrollments")
           .select("id, model_version, status, integrity_hash, thumbnail_path, "
                   "face_templates(embedding, pose, det_score, face_px, yaw_deg)")
           .eq("student_id", student_id)
           .maybe_single()
           .execute())
    row = res.data if res else None
    if not row:
        return None
    row["templates"] = [parse_vector(t["embedding"]) for t in (row.pop("face_templates", None) or [])]
    return row


def is_enrolled(client, student_id, model_version):
    res = (client.table("face_enrollments")
           .select("status, model_version")
           .eq("student_id", student_id)
           .maybe_single()
           .execute())
    row = res.data if res else None
    return bool(row and row.get("status") == "completed" and row.get("model_version") == model_version)


def match_duplicates(client, embedding, model_version, exclude_student, limit=5):
    """[(student_id, similarity)] of the closest other completed enrollments, best first."""
    res = client.rpc("match_face_templates", {
        "p_embedding": vector_literal(embedding),
        "p_model_version": model_version,
        "p_exclude_student": exclude_student,
        "p_limit": limit,
    }).execute()
    return [(r["student_id"], float(r["similarity"])) for r in (res.data or [])]


def complete_enrollment(client, student_id, model_version, consent_id, integrity_hash,
                        thumbnail_path, templates):
    """One transaction (RPC): enrollment row + templates, then status=completed. Returns its id."""
    payload = [{
        "embedding": [float(v) for v in np.asarray(t["embedding"], dtype=np.float32)],
        "det_score": t.get("det_score"), "face_px": t.get("face_px"),
        "yaw_deg": t.get("yaw_deg"), "pose": t.get("pose"),
    } for t in templates]
    res = client.rpc("complete_face_enrollment", {
        "p_student_id": student_id, "p_model_version": model_version,
        "p_consent_id": consent_id, "p_integrity_hash": integrity_hash,
        "p_thumbnail_path": thumbnail_path, "p_templates": payload,
    }).execute()
    return res.data


def delete_enrollment(client, student_id):
    """Remove the student's v2 enrollment; templates go with it (ON DELETE CASCADE)."""
    client.table("face_enrollments").delete().eq("student_id", student_id).execute()

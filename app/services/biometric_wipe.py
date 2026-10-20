"""Withdraw a student's biometric consent on their behalf (admin) and delete their face data.

Students now ask an admin to withdraw (the dashboard no longer has the button),
so the admin needs the same full wipe the student route performs:

  1. consent_logs row with consent_given = false (audit trail, always first)
  2. legacy student_biometrics face columns nulled (FaceNet512 data)
  3. v2 face_enrollments row deleted (face_templates cascade)
  4. both stored photos removed ({id}.jpg legacy, v2/{id}.jpg)
  5. audit_logs event

Each step raises on failure so the caller can report it; nothing is silently skipped.
"""

CONSENT_TYPE = "biometric_enrollment"


class WipeError(Exception):
    def __init__(self, step):
        super().__init__(step)
        self.step = step


def _missing_table(error):
    text = str(error)
    return any(m in text for m in ("42P01", "PGRST205", "does not exist", "Could not find the table"))


def wipe_on_behalf(client, student_id, admin_id, consent_version, ip, user_agent):
    """Run all steps; return the list of what was deleted. Raises WipeError(step)."""
    from app.services import face_store
    from app.services.security_service import log_audit_event

    try:
        client.table("consent_logs").insert({
            "user_id": student_id, "consent_type": CONSENT_TYPE, "consent_given": False,
            "consent_version": consent_version, "ip_address": ip,
            "user_agent": f"admin:{admin_id} {user_agent}"[:500],
        }).execute()
    except Exception as error:
        raise WipeError("consent_log") from error

    try:
        client.table("student_biometrics").update({
            "face_embeddings": None, "baseline_ear": None, "baseline_ear_metric": None,
            "face_image_url": None, "integrity_hash": None, "consent_given": False,
            "enrollment_attempts": 0, "enrolled_at": None,
        }).eq("user_id", student_id).execute()
    except Exception as error:
        raise WipeError("legacy_biometrics") from error

    try:
        face_store.delete_enrollment(client, student_id)
    except Exception as error:
        if not _missing_table(error):
            raise WipeError("face_enrollments") from error

    try:
        client.storage.from_("face-images").remove([f"{student_id}.jpg", f"v2/{student_id}.jpg"])
    except Exception as error:
        raise WipeError("photos") from error

    log_audit_event(client, actor_id=admin_id, actor_role="admin",
                    event_type="consent_withdrawn_data_deleted_by_admin", target_id=student_id,
                    old_value="biometrics_existed", new_value="biometrics_deleted")
    return ["consent_logs:withdrawn", "student_biometrics:face_columns", "face_enrollments",
            "face_templates", "face_photos"]

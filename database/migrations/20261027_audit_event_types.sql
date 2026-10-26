-- audit_logs.event_type check: add the event types the code writes but the live
-- check refused (insert failed with 23514, the event was only in the server log).
-- Found 2026-10-27: face_duplicate_blocked during a v2 enrollment test.
-- The ten existing values are kept. Apply in the Supabase SQL editor.

ALTER TABLE public.audit_logs DROP CONSTRAINT IF EXISTS audit_logs_event_type_check;
ALTER TABLE public.audit_logs ADD CONSTRAINT audit_logs_event_type_check CHECK (event_type IN (
    -- existing
    'teacher_override', 'liveness_failure', 'enrollment_failure', 'anti_spoof_trigger',
    'integrity_violation', 'device_mismatch', 'anomaly_detected', 'session_override',
    'embedding_deleted', 'rate_limit_hit',
    -- written by the code (grep event_type= in app/)
    'face_duplicate_blocked', 'consent_withdrawn_data_deleted',
    'consent_withdrawn_data_deleted_by_admin', 'reset_enrollment_attempts',
    'beacon_secret_rotated', 'term_set', 'line_unlinked'
));

-- ── Verification (the last row must read END) ─────────────────────────────
-- SELECT 'check', pg_get_constraintdef(oid) FROM pg_constraint WHERE conname = 'audit_logs_event_type_check'
-- UNION ALL SELECT 'END', 'END';

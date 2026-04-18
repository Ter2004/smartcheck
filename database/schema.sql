-- SmartCheck Database Schema
-- Generated: 2026-04-11
-- Supabase (PostgreSQL 15+) — run in SQL Editor or via psql
-- All tables use UUID primary keys and Row Level Security (RLS).
-- Enable the pgcrypto extension if gen_random_uuid() is not available.

-- ---------------------------------------------------------------------------
-- EXTENSIONS
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ---------------------------------------------------------------------------
-- USERS
-- Central account table. Auth is managed by Supabase Auth (auth.users);
-- this table holds application-level profile data keyed to the same UUID.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id                   uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    email                text        NOT NULL UNIQUE,
    full_name            text        NOT NULL,
    role                 text        NOT NULL CHECK (role IN ('admin', 'teacher', 'student')),
    student_id           text        UNIQUE,                  -- เลขนิสิต (student only)
    device_id            text,                                -- bound device fingerprint
    is_active            boolean     NOT NULL DEFAULT true,
    must_change_password boolean     NOT NULL DEFAULT false,  -- force change on first login
    created_at           timestamptz NOT NULL DEFAULT now(),
    updated_at           timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE  users IS 'Application user profiles — mirrors auth.users by UUID.';
COMMENT ON COLUMN users.student_id           IS 'University student ID (null for admin/teacher).';
COMMENT ON COLUMN users.device_id            IS 'HMAC device fingerprint bound at first check-in.';
COMMENT ON COLUMN users.must_change_password IS 'Set true after CSV import; cleared after first password change.';

CREATE INDEX IF NOT EXISTS idx_users_email    ON users (email);
CREATE INDEX IF NOT EXISTS idx_users_role     ON users (role);
CREATE INDEX IF NOT EXISTS idx_users_student_id ON users (student_id) WHERE student_id IS NOT NULL;

-- ---------------------------------------------------------------------------
-- STUDENT BIOMETRICS
-- One row per student. Stores face embeddings and enrollment state.
-- face_image_url stores the storage path (not a public URL) — generate a
-- signed URL at render time via supabase_admin.storage.create_signed_url().
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS student_biometrics (
    user_id                  uuid        PRIMARY KEY REFERENCES users (id) ON DELETE CASCADE,
    face_embeddings          jsonb,                           -- array of FaceNet512 vectors
    baseline_ear             float,                           -- calibrated Eye Aspect Ratio
    face_image_url           text,                            -- storage path in "face-images" bucket
    consent_given            boolean     NOT NULL DEFAULT false,
    consent_at               timestamptz,
    enrolled_at              timestamptz,
    enrollment_attempts      int         NOT NULL DEFAULT 0,
    last_enrollment_attempt  timestamptz,
    integrity_hash           text,                            -- HMAC-SHA256 of embeddings
    updated_at               timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE  student_biometrics IS 'Face biometric data and enrollment state for each student.';
COMMENT ON COLUMN student_biometrics.face_embeddings     IS 'JSON array of 5× FaceNet512 embeddings (512-dim each).';
COMMENT ON COLUMN student_biometrics.integrity_hash      IS 'HMAC-SHA256(user_id + embeddings, EMBEDDING_INTEGRITY_SALT).';
COMMENT ON COLUMN student_biometrics.face_image_url      IS 'Storage path in private "face-images" bucket — NOT a public URL.';

-- ---------------------------------------------------------------------------
-- CONSENT LOGS
-- Append-only PDPA audit trail. Never delete rows — only insert.
-- Records both consent_given=true (grant) and consent_given=false (withdraw).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS consent_logs (
    id               uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id          uuid        NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    consent_type     text        NOT NULL DEFAULT 'biometric_enrollment',
    consent_given    boolean     NOT NULL,
    consent_version  text        NOT NULL DEFAULT '1.0',
    ip_address       text,
    user_agent       text,
    created_at       timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE  consent_logs IS 'PDPA consent audit trail — append-only, never delete.';
COMMENT ON COLUMN consent_logs.consent_type    IS 'E.g. "biometric_enrollment".';
COMMENT ON COLUMN consent_logs.consent_given   IS 'true = granted, false = withdrawn.';
COMMENT ON COLUMN consent_logs.consent_version IS 'Version of the consent text shown to the user.';

CREATE INDEX IF NOT EXISTS idx_consent_logs_user_id    ON consent_logs (user_id);
CREATE INDEX IF NOT EXISTS idx_consent_logs_created_at ON consent_logs (created_at DESC);

-- ---------------------------------------------------------------------------
-- AUDIT LOGS
-- Application-level audit trail for admin/teacher actions.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_logs (
    id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    actor_id    uuid        REFERENCES users (id) ON DELETE SET NULL,
    actor_role  text,
    event_type  text        NOT NULL,                -- e.g. "user_created", "session_opened"
    target_id   uuid,                                -- affected resource (user, session, …)
    session_id  uuid,                                -- class session context if applicable
    old_value   text,
    new_value   text,
    metadata    jsonb       NOT NULL DEFAULT '{}',
    ip_address  text,
    user_agent  text,
    created_at  timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE  audit_logs IS 'Application-level audit log for security-sensitive events.';
COMMENT ON COLUMN audit_logs.event_type IS 'Dot-separated verb: "enrollment.reset", "user.created", etc.';

CREATE INDEX IF NOT EXISTS idx_audit_logs_actor_id   ON audit_logs (actor_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_target_id  ON audit_logs (target_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type ON audit_logs (event_type);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs (created_at DESC);

-- ---------------------------------------------------------------------------
-- COURSES
-- A course belongs to one teacher. Students are linked via course_enrollments.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS courses (
    id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    code        text        NOT NULL,                -- e.g. "CS101"
    name        text        NOT NULL,
    teacher_id  uuid        NOT NULL REFERENCES users (id) ON DELETE RESTRICT,
    semester    int         NOT NULL CHECK (semester IN (1, 2, 3)),
    section     text,                                -- e.g. "001" (nullable)
    is_active   boolean     NOT NULL DEFAULT true,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE courses IS 'University courses managed in the system.';

CREATE INDEX IF NOT EXISTS idx_courses_teacher_id ON courses (teacher_id);
CREATE INDEX IF NOT EXISTS idx_courses_code       ON courses (code);

-- ---------------------------------------------------------------------------
-- COURSE ENROLLMENTS
-- Many-to-many: students ↔ courses.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS course_enrollments (
    id          uuid    PRIMARY KEY DEFAULT gen_random_uuid(),
    course_id   uuid    NOT NULL REFERENCES courses (id) ON DELETE CASCADE,
    student_id  uuid    NOT NULL REFERENCES users   (id) ON DELETE CASCADE,
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (course_id, student_id)
);

COMMENT ON TABLE course_enrollments IS 'Students enrolled in each course.';

CREATE INDEX IF NOT EXISTS idx_course_enrollments_course_id   ON course_enrollments (course_id);
CREATE INDEX IF NOT EXISTS idx_course_enrollments_student_id  ON course_enrollments (student_id);

-- ---------------------------------------------------------------------------
-- BEACONS
-- BLE iBeacons used to verify physical presence in the classroom.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS beacons (
    id              uuid    PRIMARY KEY DEFAULT gen_random_uuid(),
    uuid            text    NOT NULL,                -- iBeacon UUID string
    major           int     NOT NULL,
    minor           int     NOT NULL,
    room_name       text    NOT NULL,
    rssi_threshold  int     NOT NULL DEFAULT -75,    -- minimum RSSI to count as "present"
    is_active       boolean NOT NULL DEFAULT true,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE  beacons IS 'BLE iBeacon devices installed in classrooms.';
COMMENT ON COLUMN beacons.rssi_threshold IS 'Minimum received signal strength (dBm); e.g. -75.';

CREATE INDEX IF NOT EXISTS idx_beacons_is_active ON beacons (is_active) WHERE is_active = true;

-- ---------------------------------------------------------------------------
-- SESSIONS
-- A class session is one occurrence of a course in a room.
-- The teacher opens/closes it; students check in while is_open=true.
-- checkin_duration limits how many minutes after start_time check-ins are accepted.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    id                uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    course_id         uuid        NOT NULL REFERENCES courses (id) ON DELETE CASCADE,
    beacon_id         uuid        REFERENCES beacons (id) ON DELETE SET NULL,
    title             text        NOT NULL,
    start_time        timestamptz NOT NULL DEFAULT now(),
    end_time          timestamptz,
    is_open           boolean     NOT NULL DEFAULT false,
    checkin_duration  int,                               -- minutes; null = unlimited
    created_at        timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE  sessions IS 'Individual class meeting sessions.';
COMMENT ON COLUMN sessions.checkin_duration IS 'How many minutes after start_time to accept check-ins (null = no limit).';

CREATE INDEX IF NOT EXISTS idx_sessions_course_id  ON sessions (course_id);
CREATE INDEX IF NOT EXISTS idx_sessions_is_open    ON sessions (is_open) WHERE is_open = true;
CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON sessions (start_time DESC);

-- ---------------------------------------------------------------------------
-- SCHEDULES
-- Weekly recurring schedule for a course used by the auto-session scheduler.
-- day_of_week: 0=Monday … 6=Sunday (matches Python datetime.weekday()).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schedules (
    id           uuid  PRIMARY KEY DEFAULT gen_random_uuid(),
    course_id    uuid  NOT NULL REFERENCES courses (id) ON DELETE CASCADE,
    day_of_week  int   NOT NULL CHECK (day_of_week BETWEEN 0 AND 6),
    start_time   time  NOT NULL,
    end_time     time  NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE  schedules IS 'Weekly class schedule — used by the APScheduler auto-open/close job.';
COMMENT ON COLUMN schedules.day_of_week IS '0=Monday … 6=Sunday (Python weekday convention).';

CREATE INDEX IF NOT EXISTS idx_schedules_course_id ON schedules (course_id);

-- ---------------------------------------------------------------------------
-- ATTENDANCE
-- One row per student per session check-in.
-- Stores the results of every verification step (BLE, liveness, face).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS attendance (
    id              uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      uuid        NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    student_id      uuid        NOT NULL REFERENCES users    (id) ON DELETE CASCADE,
    ble_rssi        int,                                -- measured RSSI (null if not available)
    ble_pass        boolean     NOT NULL DEFAULT false,
    liveness_pass   boolean     NOT NULL DEFAULT false,
    liveness_action text        NOT NULL DEFAULT '',    -- e.g. "blink", "nod"
    face_score      float,                              -- cosine similarity (logged server-side)
    face_pass       boolean     NOT NULL DEFAULT false,
    status          text        NOT NULL DEFAULT 'present' CHECK (status IN ('present', 'late', 'absent')),
    check_in_at     timestamptz NOT NULL DEFAULT now(),
    device_id       text,                               -- device fingerprint at time of check-in
    UNIQUE (session_id, student_id)
);

COMMENT ON TABLE  attendance IS 'Student check-in records per session.';
COMMENT ON COLUMN attendance.face_score     IS 'Best cosine similarity vs stored embeddings — stored for audit, never returned to client.';
COMMENT ON COLUMN attendance.liveness_action IS 'Liveness challenge passed (blink / head-nod / etc.).';

CREATE INDEX IF NOT EXISTS idx_attendance_session_id  ON attendance (session_id);
CREATE INDEX IF NOT EXISTS idx_attendance_student_id  ON attendance (student_id);
CREATE INDEX IF NOT EXISTS idx_attendance_check_in_at ON attendance (check_in_at DESC);

-- ---------------------------------------------------------------------------
-- HELPER: updated_at auto-update trigger
-- Call set_updated_at() on tables that have an updated_at column.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

CREATE OR REPLACE TRIGGER trg_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE OR REPLACE TRIGGER trg_student_biometrics_updated_at
    BEFORE UPDATE ON student_biometrics
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE OR REPLACE TRIGGER trg_courses_updated_at
    BEFORE UPDATE ON courses
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE OR REPLACE TRIGGER trg_beacons_updated_at
    BEFORE UPDATE ON beacons
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------------
-- ROW LEVEL SECURITY
-- Enable RLS on all tables. Use service key (supabase_admin) to bypass.
-- Anon key is used only for auth.sign_in_with_password — all data access
-- uses the service key so RLS policies here are a safety net only.
-- ---------------------------------------------------------------------------
ALTER TABLE users               ENABLE ROW LEVEL SECURITY;
ALTER TABLE student_biometrics  ENABLE ROW LEVEL SECURITY;
ALTER TABLE consent_logs        ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_logs          ENABLE ROW LEVEL SECURITY;
ALTER TABLE courses             ENABLE ROW LEVEL SECURITY;
ALTER TABLE course_enrollments  ENABLE ROW LEVEL SECURITY;
ALTER TABLE beacons             ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions            ENABLE ROW LEVEL SECURITY;
ALTER TABLE schedules           ENABLE ROW LEVEL SECURITY;
ALTER TABLE attendance          ENABLE ROW LEVEL SECURITY;

-- All data access goes through supabase_admin (service role) — deny anon by default.
-- Add fine-grained policies here if you ever switch to per-user JWT access.

-- ---------------------------------------------------------------------------
-- STORED PROCEDURES
-- ---------------------------------------------------------------------------

-- Atomically increment enrollment_attempts and check the 24-hour window limit.
-- Returns (current_attempts int, allowed boolean).
-- Uses FOR UPDATE to prevent race conditions under concurrent requests.
CREATE OR REPLACE FUNCTION atomic_enroll_attempt(
    p_user_id  uuid,
    p_max      int,
    p_window_h int
) RETURNS TABLE(current_attempts int, allowed boolean)
LANGUAGE plpgsql AS $$
DECLARE
    v_attempts int;
    v_last     timestamptz;
    v_now      timestamptz := now();
BEGIN
    SELECT enrollment_attempts, last_enrollment_attempt
      INTO v_attempts, v_last
      FROM student_biometrics
     WHERE user_id = p_user_id
       FOR UPDATE;

    IF v_last IS NULL OR v_now - v_last > (p_window_h || ' hours')::interval THEN
        v_attempts := 0;
    END IF;

    v_attempts := v_attempts + 1;

    UPDATE student_biometrics
       SET enrollment_attempts      = v_attempts,
           last_enrollment_attempt  = v_now
     WHERE user_id = p_user_id;

    RETURN QUERY SELECT v_attempts, (v_attempts <= p_max);
END;
$$;

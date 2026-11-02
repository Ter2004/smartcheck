-- Teacher opens the class (SESSION_OPEN_MODE=teacher, app/services/class_open.py)
-- and class-open notifications to the Android app. Safe to re-run.
-- Apply in the Supabase SQL editor BEFORE restarting the server on this code;
-- until then set SESSION_OPEN_MODE=schedule.
--
-- sessions.opened_at        when the class opened (teacher press, or automatic
--                           CLASS_AUTO_OPEN_AFTER_MINUTES after the start)
-- sessions.opened_by        the teacher who pressed; null when it opened by itself
-- sessions.present_minutes  check-ins within this many minutes of opened_at are
--                           present, later ones late

ALTER TABLE public.sessions ADD COLUMN IF NOT EXISTS opened_at timestamptz;
ALTER TABLE public.sessions ADD COLUMN IF NOT EXISTS opened_by uuid REFERENCES public.users (id) ON DELETE SET NULL;
ALTER TABLE public.sessions ADD COLUMN IF NOT EXISTS present_minutes int;
ALTER TABLE public.sessions DROP CONSTRAINT IF EXISTS sessions_present_minutes_check;
ALTER TABLE public.sessions ADD CONSTRAINT sessions_present_minutes_check
    CHECK (present_minutes IS NULL OR present_minutes BETWEEN 1 AND 60);

-- Class-open notifications (app/services/class_push.py, app/routes/push.py)
--
-- sessions.open_push_at      when "class open" was sent; set once by a conditional
-- sessions.reminder_push_at  update, so it is never sent twice ("1 minute left")
-- push_devices               one row per phone: its Firebase token and who is
--                            logged in on it. Server-only (service key).

ALTER TABLE public.sessions ADD COLUMN IF NOT EXISTS open_push_at timestamptz;
ALTER TABLE public.sessions ADD COLUMN IF NOT EXISTS reminder_push_at timestamptz;

CREATE TABLE IF NOT EXISTS public.push_devices (
    token         text        PRIMARY KEY,
    user_id       uuid        NOT NULL REFERENCES public.users (id) ON DELETE CASCADE,
    platform      text        NOT NULL DEFAULT 'android',
    created_at    timestamptz NOT NULL DEFAULT now(),
    last_seen_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_push_devices_user_id ON public.push_devices (user_id);

ALTER TABLE public.push_devices ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.push_devices FROM PUBLIC, anon, authenticated;

-- ── Verification (the last row must read END) ───────────────────────────────
-- SELECT 'column', column_name FROM information_schema.columns
--  WHERE table_name = 'sessions'
--    AND column_name IN ('opened_at', 'opened_by', 'present_minutes', 'open_push_at', 'reminder_push_at')
-- UNION ALL SELECT 'table', table_name FROM information_schema.tables WHERE table_name = 'push_devices'
-- UNION ALL SELECT 'rls', relrowsecurity::text FROM pg_class WHERE relname = 'push_devices'
-- UNION ALL SELECT 'END', 'END';

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

-- ── Verification (the last row must read END) ───────────────────────────────
-- SELECT 'column', column_name FROM information_schema.columns
--  WHERE table_name = 'sessions' AND column_name IN ('opened_at', 'opened_by', 'present_minutes')
-- UNION ALL SELECT 'END', 'END';

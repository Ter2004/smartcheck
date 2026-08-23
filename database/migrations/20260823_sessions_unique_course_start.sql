-- F-10: prevent duplicate sessions for the same course + start_time.
-- Root cause: app/scheduler.py's auto_manage_sessions() does a
-- SELECT-then-INSERT with no DB-level guard, and teacher.py /
-- admin.py can also insert into sessions concurrently (APScheduler
-- runs on its own background thread even with a single gunicorn
-- worker, so this can race a teacher/admin HTTP request too — not
-- just a multi-worker problem). See docs/review/99-summary.md F-10.
--
-- PREREQUISITE — run this BEFORE applying the constraint below, and
-- resolve any rows it returns (see docs/review/99-summary.md F-10 /
-- the SQL Claude provided in chat for the duplicate-check and
-- cleanup queries). This ALTER TABLE will fail with a
-- "could not create unique index" error if duplicates still exist.

ALTER TABLE sessions
    ADD CONSTRAINT sessions_course_id_start_time_key UNIQUE (course_id, start_time);

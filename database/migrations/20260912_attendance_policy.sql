-- Apply after existing migrations, before deploying the matching application.
-- Historical sessions/attendance are retained. Unclassified old sessions cannot
-- accept new automatic check-ins. Never guess their original timetable.
BEGIN;
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS face_enrolled_once boolean NOT NULL DEFAULT false;
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS is_test_account boolean NOT NULL DEFAULT false;
UPDATE public.users u SET face_enrolled_once = true FROM public.student_biometrics b
 WHERE b.user_id = u.id AND (b.enrolled_at IS NOT NULL OR b.face_embeddings IS NOT NULL);
UPDATE public.users SET is_test_account = true
 WHERE email ~ '^loadtest([0-9]+|_teacher)@smartcheck[.]local$';
ALTER TABLE public.courses ADD COLUMN IF NOT EXISTS is_test_course boolean NOT NULL DEFAULT false;
UPDATE public.courses SET is_test_course = true WHERE code = 'LOADTEST101';
ALTER TABLE public.schedules ADD COLUMN IF NOT EXISTS beacon_id uuid REFERENCES public.beacons(id);
ALTER TABLE public.schedules ADD COLUMN IF NOT EXISTS is_active boolean NOT NULL DEFAULT true;
ALTER TABLE public.schedules ADD COLUMN IF NOT EXISTS retired_by uuid REFERENCES public.users(id);
ALTER TABLE public.schedules ADD COLUMN IF NOT EXISTS retire_reason text;
ALTER TABLE public.schedules ADD COLUMN IF NOT EXISTS open_before_minutes integer NOT NULL DEFAULT 0 CHECK (open_before_minutes BETWEEN 0 AND 60);
ALTER TABLE public.schedules ADD COLUMN IF NOT EXISTS late_after_minutes integer NOT NULL DEFAULT 15 CHECK (late_after_minutes BETWEEN 0 AND 1440);
ALTER TABLE public.schedules ADD COLUMN IF NOT EXISTS close_after_minutes integer CHECK (close_after_minutes BETWEEN 1 AND 1440);
ALTER TABLE public.sessions ADD COLUMN IF NOT EXISTS schedule_id uuid REFERENCES public.schedules(id);
ALTER TABLE public.sessions ADD COLUMN IF NOT EXISTS session_kind text NOT NULL DEFAULT 'legacy' CHECK (session_kind IN ('legacy','scheduled','makeup'));
ALTER TABLE public.sessions ADD COLUMN IF NOT EXISTS checkin_opens_at timestamptz;
ALTER TABLE public.sessions ADD COLUMN IF NOT EXISTS late_at timestamptz;
ALTER TABLE public.sessions ADD COLUMN IF NOT EXISTS checkin_closes_at timestamptz;
ALTER TABLE public.sessions ADD COLUMN IF NOT EXISTS change_reason text;
ALTER TABLE public.sessions ADD COLUMN IF NOT EXISTS created_by uuid REFERENCES public.users(id);
ALTER TABLE public.sessions ADD COLUMN IF NOT EXISTS cancelled_at timestamptz;
ALTER TABLE public.sessions ADD COLUMN IF NOT EXISTS cancelled_by uuid REFERENCES public.users(id);
-- Fails safely if old duplicates exist: inspect them rather than deleting records.
CREATE UNIQUE INDEX IF NOT EXISTS sessions_course_start_policy_key ON public.sessions(course_id,start_time);
ALTER TABLE public.attendance ADD COLUMN IF NOT EXISTS override_by uuid REFERENCES public.users(id);
ALTER TABLE public.attendance ADD COLUMN IF NOT EXISTS override_reason text;
ALTER TABLE public.attendance ADD COLUMN IF NOT EXISTS override_at timestamptz;
ALTER TABLE public.attendance DROP CONSTRAINT IF EXISTS attendance_status_check;
ALTER TABLE public.attendance ADD CONSTRAINT attendance_status_check CHECK (status IN ('present','late','absent','manual'));

CREATE TABLE IF NOT EXISTS public.face_change_requests (
 id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
 user_id uuid NOT NULL REFERENCES public.users(id),
 reason text NOT NULL CHECK (length(trim(reason)) BETWEEN 5 AND 1000),
 status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','approved','rejected','consumed')),
 reviewed_by uuid REFERENCES public.users(id), review_reason text,
 created_at timestamptz NOT NULL DEFAULT now(), reviewed_at timestamptz,
 consumed_at timestamptz
);
CREATE UNIQUE INDEX IF NOT EXISTS face_change_one_pending ON public.face_change_requests(user_id) WHERE status IN ('pending','approved');
ALTER TABLE public.face_change_requests ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.face_change_requests FROM anon, authenticated;
GRANT ALL ON public.face_change_requests TO service_role;

-- Serialize identity replacement per user, including concurrent enrollments.
CREATE OR REPLACE FUNCTION public.guard_face_replacement() RETURNS trigger
LANGUAGE plpgsql SET search_path = public AS $$
DECLARE was_enrolled boolean; approval uuid; consent boolean;
BEGIN
 IF NEW.face_embeddings IS NULL OR NEW.face_embeddings = '[]'::jsonb THEN RETURN NEW; END IF;
 IF TG_OP = 'UPDATE' AND NEW.face_embeddings IS NOT DISTINCT FROM OLD.face_embeddings
    AND NEW.enrolled_at IS NOT DISTINCT FROM OLD.enrolled_at THEN RETURN NEW; END IF;
 SELECT face_enrolled_once INTO was_enrolled FROM users WHERE id = NEW.user_id FOR UPDATE;
 SELECT consent_given INTO consent FROM consent_logs WHERE user_id=NEW.user_id
   AND consent_type='biometric_enrollment' ORDER BY created_at DESC LIMIT 1;
 IF consent IS DISTINCT FROM true THEN RAISE EXCEPTION 'enrollment_consent_required'; END IF;
 IF was_enrolled THEN
   SELECT id INTO approval FROM face_change_requests WHERE user_id=NEW.user_id
     AND status='approved' FOR UPDATE;
   IF approval IS NULL THEN RAISE EXCEPTION 'face_replacement_not_approved'; END IF;
   UPDATE face_change_requests SET status='consumed', consumed_at=now() WHERE id=approval;
   INSERT INTO audit_logs(actor_id,actor_role,event_type,target_id,metadata)
     VALUES(NEW.user_id,'student','enrollment.replaced',NEW.user_id,jsonb_build_object('request_id',approval));
 END IF;
 UPDATE users SET face_enrolled_once=true WHERE id=NEW.user_id;
 RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS guard_face_replacement ON public.student_biometrics;
CREATE TRIGGER guard_face_replacement BEFORE INSERT OR UPDATE ON public.student_biometrics
 FOR EACH ROW EXECUTE FUNCTION public.guard_face_replacement();

-- One transaction validates an override and stores its audit record.
-- Automatic attendance is rechecked against current time even after slow models.
CREATE OR REPLACE FUNCTION public.guard_attendance_policy() RETURNS trigger
LANGUAGE plpgsql SET search_path = public AS $$
DECLARE s sessions%ROWTYPE; old_status text; cutoff timestamptz;
BEGIN
 SELECT * INTO s FROM sessions WHERE id=NEW.session_id FOR UPDATE;
 IF NOT EXISTS (SELECT 1 FROM course_enrollments e JOIN users u ON u.id=e.student_id
   WHERE e.course_id=s.course_id AND e.student_id=NEW.student_id AND u.role='student' AND u.is_active) THEN
   RAISE EXCEPTION 'student_not_in_roster';
 END IF;
 IF NEW.override_by IS NOT NULL THEN
   IF NOT EXISTS (SELECT 1 FROM courses c JOIN users u ON u.id=c.teacher_id
     WHERE c.id=s.course_id AND c.teacher_id=NEW.override_by AND u.role='teacher' AND u.is_active)
     OR length(trim(coalesce(NEW.override_reason,''))) < 5 THEN RAISE EXCEPTION 'invalid_override'; END IF;
   IF TG_OP='UPDATE' THEN old_status := OLD.status; END IF;
   NEW.override_at := now();
   INSERT INTO audit_logs(actor_id,actor_role,event_type,target_id,session_id,old_value,new_value,metadata)
     VALUES(NEW.override_by,'teacher','teacher_override',NEW.student_id,NEW.session_id,old_status,NEW.status,
       jsonb_build_object('reason',NEW.override_reason));
 ELSE
   IF s.session_kind='scheduled' AND NOT EXISTS (SELECT 1 FROM schedules WHERE id=s.schedule_id AND is_active) THEN
     RAISE EXCEPTION 'schedule_retired';
   END IF;
   cutoff := s.checkin_closes_at;
   IF s.checkin_duration IS NOT NULL THEN
     IF s.checkin_duration <= 0 THEN RAISE EXCEPTION 'invalid_checkin_duration'; END IF;
     cutoff := least(cutoff, s.start_time + s.checkin_duration * interval '1 minute');
   END IF;
   IF s.session_kind NOT IN ('scheduled','makeup') OR s.cancelled_at IS NOT NULL
      OR s.checkin_opens_at IS NULL OR cutoff IS NULL OR s.end_time IS NULL OR s.late_at IS NULL
      OR now() < s.checkin_opens_at OR now() >= cutoff OR now() >= s.end_time THEN
     RAISE EXCEPTION 'session_outside_checkin_window';
   END IF;
   NEW.status := CASE WHEN now() >= s.late_at THEN 'late' ELSE 'present' END;
   NEW.check_in_at := now();
 END IF;
 RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS guard_attendance_policy ON public.attendance;
CREATE TRIGGER guard_attendance_policy BEFORE INSERT OR UPDATE ON public.attendance
 FOR EACH ROW EXECUTE FUNCTION public.guard_attendance_policy();

CREATE OR REPLACE FUNCTION public.guard_session_window() RETURNS trigger
LANGUAGE plpgsql SET search_path = public AS $$
BEGIN
 IF NEW.cancelled_at IS NOT NULL AND (TG_OP='INSERT' OR OLD.cancelled_at IS NULL) THEN
   IF NOT EXISTS(SELECT 1 FROM users WHERE id=NEW.cancelled_by AND role='admin' AND is_active)
      OR length(trim(coalesce(NEW.change_reason,'')))<5 THEN RAISE EXCEPTION 'invalid_cancellation'; END IF;
   INSERT INTO audit_logs(actor_id,actor_role,event_type,target_id,session_id,metadata)
     VALUES(NEW.cancelled_by,'admin','session.cancelled',NEW.id,NEW.id,jsonb_build_object('reason',NEW.change_reason));
 END IF;
 IF TG_OP='UPDATE' AND OLD.cancelled_at IS NOT NULL AND NEW.cancelled_at IS NULL THEN
   RAISE EXCEPTION 'cannot_reopen_cancelled_session';
 END IF;
 IF NEW.session_kind='legacy' THEN NEW.is_open:=false; RETURN NEW; END IF;
 IF NEW.checkin_opens_at IS NULL OR NEW.checkin_closes_at IS NULL OR NEW.late_at IS NULL OR NEW.end_time IS NULL
   OR NOT (NEW.checkin_opens_at<=NEW.start_time AND NEW.start_time<NEW.end_time
      AND NEW.start_time<=NEW.late_at AND NEW.late_at<=NEW.checkin_closes_at
      AND NEW.checkin_opens_at<NEW.checkin_closes_at AND NEW.checkin_closes_at<=NEW.end_time) THEN
   RAISE EXCEPTION 'invalid_session_window';
 END IF;
 IF NEW.session_kind='scheduled' AND NEW.schedule_id IS NULL THEN RAISE EXCEPTION 'schedule_required'; END IF;
 IF NEW.session_kind='makeup' AND length(trim(coalesce(NEW.change_reason,'')))<5 THEN RAISE EXCEPTION 'makeup_reason_required'; END IF;
 IF TG_OP='INSERT' AND NEW.session_kind='makeup' THEN
   IF NOT EXISTS(SELECT 1 FROM users WHERE id=NEW.created_by AND role='admin' AND is_active) THEN RAISE EXCEPTION 'admin_required'; END IF;
   INSERT INTO audit_logs(actor_id,actor_role,event_type,target_id,metadata)
     VALUES(NEW.created_by,'admin','session.makeup_created',NEW.id,jsonb_build_object('reason',NEW.change_reason));
 END IF;
 NEW.is_open := NEW.cancelled_at IS NULL AND now()>=NEW.checkin_opens_at AND now()<NEW.checkin_closes_at;
 RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS guard_session_window ON public.sessions;
CREATE TRIGGER guard_session_window BEFORE INSERT OR UPDATE ON public.sessions
 FOR EACH ROW EXECUTE FUNCTION public.guard_session_window();
UPDATE public.sessions SET is_open=false WHERE session_kind='legacy' AND is_open;

CREATE OR REPLACE FUNCTION public.audit_face_request() RETURNS trigger
LANGUAGE plpgsql SET search_path = public AS $$
BEGIN
 IF TG_OP='UPDATE' AND NEW.status IN ('approved','rejected') THEN
   IF OLD.status!='pending' OR NOT EXISTS(SELECT 1 FROM users WHERE id=NEW.reviewed_by AND role='admin' AND is_active)
      OR length(trim(coalesce(NEW.review_reason,'')))<5 THEN RAISE EXCEPTION 'invalid_face_review'; END IF;
   INSERT INTO audit_logs(actor_id,actor_role,event_type,target_id,metadata)
     VALUES(NEW.reviewed_by,'admin','enrollment.request_'||NEW.status,NEW.user_id,
       jsonb_build_object('request_id',NEW.id,'reason',NEW.review_reason));
 ELSIF TG_OP='INSERT' THEN
   IF NEW.status!='pending' THEN RAISE EXCEPTION 'pending_request_required'; END IF;
   INSERT INTO audit_logs(actor_id,actor_role,event_type,target_id,metadata)
     VALUES(NEW.user_id,'student','enrollment.change_requested',NEW.user_id,jsonb_build_object('request_id',NEW.id,'reason',NEW.reason));
 END IF;
 RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS audit_face_request ON public.face_change_requests;
CREATE TRIGGER audit_face_request BEFORE INSERT OR UPDATE ON public.face_change_requests
 FOR EACH ROW EXECUTE FUNCTION public.audit_face_request();

CREATE OR REPLACE FUNCTION public.retire_schedule() RETURNS trigger
LANGUAGE plpgsql SET search_path = public AS $$
BEGIN
 IF OLD.is_active AND NOT NEW.is_active THEN
   IF NOT EXISTS(SELECT 1 FROM users WHERE id=NEW.retired_by AND role='admin' AND is_active)
     OR length(trim(coalesce(NEW.retire_reason,'')))<5 THEN RAISE EXCEPTION 'invalid_schedule_retirement'; END IF;
   UPDATE sessions SET cancelled_at=now(), cancelled_by=NEW.retired_by,
      change_reason=NEW.retire_reason, is_open=false
      WHERE schedule_id=NEW.id AND end_time>now() AND cancelled_at IS NULL;
   INSERT INTO audit_logs(actor_id,actor_role,event_type,target_id,metadata)
     VALUES(NEW.retired_by,'admin','schedule.retired',NEW.id,jsonb_build_object('reason',NEW.retire_reason));
 END IF;
 RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS retire_schedule ON public.schedules;
CREATE TRIGGER retire_schedule AFTER UPDATE ON public.schedules FOR EACH ROW EXECUTE FUNCTION public.retire_schedule();
COMMIT;
NOTIFY pgrst, 'reload schema';

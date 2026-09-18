-- Serialize conflicting enrollment saves at the row level, including upserts.
-- Clearing biometric data (withdrawal/admin reset) remains possible.
CREATE OR REPLACE FUNCTION public.prevent_completed_face_replacement()
RETURNS trigger LANGUAGE plpgsql SET search_path = public AS $$
BEGIN
    IF OLD.consent_given
       AND OLD.face_embeddings IS NOT NULL AND OLD.face_embeddings <> '[]'::jsonb
       AND NEW.face_embeddings IS NOT NULL AND NEW.face_embeddings <> '[]'::jsonb
       AND (NEW.face_embeddings IS DISTINCT FROM OLD.face_embeddings
            OR NEW.enrolled_at IS DISTINCT FROM OLD.enrolled_at) THEN
        RAISE EXCEPTION 'already_enrolled' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE TRIGGER prevent_completed_face_replacement
BEFORE UPDATE ON public.student_biometrics
FOR EACH ROW EXECUTE FUNCTION public.prevent_completed_face_replacement();

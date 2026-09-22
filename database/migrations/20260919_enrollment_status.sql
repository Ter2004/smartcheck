-- Minimal server-only projection. No embeddings leave Postgres.
-- Reject malformed JSON instead of silently changing enrollment protection.
CREATE OR REPLACE FUNCTION public.get_enrollment_status(p_user_id uuid)
RETURNS TABLE (is_enrolled boolean, baseline_ear double precision)
LANGUAGE plpgsql STABLE SECURITY INVOKER SET search_path = public AS $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM public.student_biometrics b
        WHERE b.user_id = p_user_id AND b.face_embeddings IS NOT NULL
          AND jsonb_typeof(b.face_embeddings) NOT IN ('array', 'null')
    ) THEN
        RAISE EXCEPTION 'Invalid biometric embedding format';
    END IF;
    RETURN QUERY
    SELECT CASE WHEN jsonb_typeof(b.face_embeddings) = 'array'
                THEN jsonb_array_length(b.face_embeddings) > 0 AND b.consent_given
                ELSE false END,
           b.baseline_ear::double precision
    FROM public.student_biometrics b WHERE b.user_id = p_user_id;
END;
$$;
REVOKE ALL ON FUNCTION public.get_enrollment_status(uuid) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.get_enrollment_status(uuid) TO service_role;
NOTIFY pgrst, 'reload schema';

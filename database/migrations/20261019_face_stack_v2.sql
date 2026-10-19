-- Face stack v2: enrollments and templates in their own tables, one row per
-- enrolled student plus one row per stored embedding (pgvector), with the
-- model version that produced them. student_biometrics and its trigger are not
-- touched: legacy FaceNet512 rows stay where they are until the owner decides.
--
-- Access: RLS on with no policies, and no grants to anon/authenticated, so only
-- the service role (the Flask server) can read or write these tables.
-- Apply in the Supabase SQL editor. Verification queries are at the end.

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;

CREATE TABLE IF NOT EXISTS public.face_enrollments (
    id              uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id      uuid        NOT NULL UNIQUE REFERENCES public.users (id) ON DELETE CASCADE,
    model_version   text        NOT NULL,
    status          text        NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'completed')),
    consent_id      uuid        REFERENCES public.consent_logs (id),
    integrity_hash  text,
    thumbnail_path  text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    completed_at    timestamptz
);

CREATE TABLE IF NOT EXISTS public.face_templates (
    id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    enrollment_id  uuid    NOT NULL REFERENCES public.face_enrollments (id) ON DELETE CASCADE,
    embedding      extensions.vector(512) NOT NULL,
    det_score      real,
    face_px        integer,
    yaw_deg        real,
    pose           text    CHECK (pose IN ('frontal', 'turn')),
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_face_templates_enrollment ON public.face_templates (enrollment_id);
-- 1:N duplicate check at enrollment (cosine distance).
CREATE INDEX IF NOT EXISTS idx_face_templates_embedding_hnsw
    ON public.face_templates USING hnsw (embedding extensions.vector_cosine_ops);

COMMENT ON TABLE  public.face_enrollments IS 'One face enrollment per student (face stack v2). Service role only.';
COMMENT ON COLUMN public.face_enrollments.integrity_hash IS 'HMAC-SHA256(user_id + model_version + embeddings, EMBEDDING_INTEGRITY_SALT).';
COMMENT ON COLUMN public.face_enrollments.thumbnail_path IS 'Private "face-images" bucket path of the best frontal frame — NOT a public URL.';
COMMENT ON TABLE  public.face_templates   IS 'Embeddings of one enrollment (3 frontal + 2 turned), unit length.';

ALTER TABLE public.face_enrollments ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.face_templates   ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.face_enrollments FROM PUBLIC, anon, authenticated;
REVOKE ALL ON public.face_templates   FROM PUBLIC, anon, authenticated;

-- Same semantics as prevent_completed_face_replacement on student_biometrics
-- (20260918): a completed enrollment cannot be replaced or have its templates
-- changed; clearing it (DELETE, as withdrawal and admin reset do) stays possible.
CREATE OR REPLACE FUNCTION public.prevent_completed_face_enrollment_replacement()
RETURNS trigger LANGUAGE plpgsql SET search_path = public AS $$
BEGIN
    IF TG_TABLE_NAME = 'face_enrollments' THEN
        IF OLD.status = 'completed' AND NEW.status = 'completed'
           AND (NEW.model_version  IS DISTINCT FROM OLD.model_version
                OR NEW.integrity_hash IS DISTINCT FROM OLD.integrity_hash
                OR NEW.completed_at   IS DISTINCT FROM OLD.completed_at
                OR NEW.student_id     IS DISTINCT FROM OLD.student_id) THEN
            RAISE EXCEPTION 'already_enrolled' USING ERRCODE = '23514';
        END IF;
        IF OLD.status = 'completed' AND NEW.status <> 'completed' THEN
            RAISE EXCEPTION 'already_enrolled' USING ERRCODE = '23514';
        END IF;
        RETURN NEW;
    END IF;
    -- face_templates: no INSERT or UPDATE once the enrollment is completed.
    IF EXISTS (SELECT 1 FROM public.face_enrollments e
               WHERE e.id = NEW.enrollment_id AND e.status = 'completed') THEN
        RAISE EXCEPTION 'already_enrolled' USING ERRCODE = '23514';
    END IF;
    IF TG_OP = 'UPDATE' AND NEW.enrollment_id IS DISTINCT FROM OLD.enrollment_id
       AND EXISTS (SELECT 1 FROM public.face_enrollments e
                   WHERE e.id = OLD.enrollment_id AND e.status = 'completed') THEN
        RAISE EXCEPTION 'already_enrolled' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE TRIGGER prevent_completed_face_enrollment_replacement
BEFORE UPDATE ON public.face_enrollments
FOR EACH ROW EXECUTE FUNCTION public.prevent_completed_face_enrollment_replacement();

CREATE OR REPLACE TRIGGER prevent_completed_face_template_change
BEFORE INSERT OR UPDATE ON public.face_templates
FOR EACH ROW EXECUTE FUNCTION public.prevent_completed_face_enrollment_replacement();

-- One transaction: enrollment row + its templates + HMAC. A leftover pending
-- row (an attempt that died mid-way) is replaced; a completed one is not.
-- p_templates: [{"embedding": [512 floats], "det_score", "face_px", "yaw_deg", "pose"}, ...]
CREATE OR REPLACE FUNCTION public.complete_face_enrollment(
    p_student_id     uuid,
    p_model_version  text,
    p_consent_id     uuid,
    p_integrity_hash text,
    p_thumbnail_path text,
    p_templates      jsonb
) RETURNS uuid LANGUAGE plpgsql SET search_path = public, extensions AS $$
DECLARE
    v_id uuid;
BEGIN
    IF jsonb_typeof(p_templates) <> 'array' OR jsonb_array_length(p_templates) = 0 THEN
        RAISE EXCEPTION 'templates_missing' USING ERRCODE = '22023';
    END IF;
    IF EXISTS (SELECT 1 FROM public.face_enrollments
               WHERE student_id = p_student_id AND status = 'completed') THEN
        RAISE EXCEPTION 'already_enrolled' USING ERRCODE = '23514';
    END IF;
    DELETE FROM public.face_enrollments WHERE student_id = p_student_id AND status = 'pending';
    INSERT INTO public.face_enrollments (student_id, model_version, status, consent_id, integrity_hash, thumbnail_path)
    VALUES (p_student_id, p_model_version, 'pending', p_consent_id, p_integrity_hash, p_thumbnail_path)
    RETURNING id INTO v_id;
    INSERT INTO public.face_templates (enrollment_id, embedding, det_score, face_px, yaw_deg, pose)
    SELECT v_id, (t->'embedding')::text::extensions.vector(512),
           (t->>'det_score')::real, (t->>'face_px')::integer, (t->>'yaw_deg')::real, t->>'pose'
    FROM jsonb_array_elements(p_templates) AS t;
    UPDATE public.face_enrollments SET status = 'completed', completed_at = now() WHERE id = v_id;
    RETURN v_id;
END;
$$;

-- 1:N duplicate check: the closest completed enrollments of the same model,
-- excluding the student's own rows. similarity = 1 - cosine distance.
-- The inner ORDER BY ... LIMIT is the shape the HNSW index serves; 40 matches
-- pgvector's default hnsw.ef_search. On a small table the planner scans it
-- exactly instead, so the result is exact until the table grows.
CREATE OR REPLACE FUNCTION public.match_face_templates(
    p_embedding        text,
    p_model_version    text,
    p_exclude_student  uuid,
    p_limit            integer DEFAULT 5
) RETURNS TABLE (student_id uuid, similarity double precision)
LANGUAGE sql STABLE SET search_path = public, extensions AS $$
    WITH nearest AS (
        SELECT t.enrollment_id, t.embedding <=> p_embedding::extensions.vector(512) AS distance
        FROM public.face_templates t
        ORDER BY t.embedding <=> p_embedding::extensions.vector(512)
        LIMIT 40
    )
    SELECT e.student_id, max(1 - n.distance) AS similarity
    FROM nearest n
    JOIN public.face_enrollments e ON e.id = n.enrollment_id
    WHERE e.status = 'completed'
      AND e.model_version = p_model_version
      AND e.student_id IS DISTINCT FROM p_exclude_student
    GROUP BY e.student_id
    ORDER BY similarity DESC
    LIMIT greatest(1, least(p_limit, 50));
$$;

REVOKE ALL ON FUNCTION public.complete_face_enrollment(uuid, text, uuid, text, text, jsonb) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.match_face_templates(text, text, uuid, integer) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.complete_face_enrollment(uuid, text, uuid, text, text, jsonb) TO service_role;
GRANT EXECUTE ON FUNCTION public.match_face_templates(text, text, uuid, integer) TO service_role;

-- ── Verification (run after applying; the last row must read END) ──────────
-- SELECT 'extension' AS item, extversion AS value FROM pg_extension WHERE extname = 'vector'
-- UNION ALL SELECT 'rls_' || relname, relrowsecurity::text FROM pg_class WHERE relname IN ('face_enrollments', 'face_templates')
-- UNION ALL SELECT 'trigger', tgname::text FROM pg_trigger WHERE tgname LIKE 'prevent_completed_face_%'
-- UNION ALL SELECT 'index', indexname::text FROM pg_indexes WHERE tablename = 'face_templates'
-- UNION ALL SELECT 'END', 'END';

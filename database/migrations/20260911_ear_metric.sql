-- Run before deploying the corrected enrollment writer. No legacy conversion:
-- existing scalar baselines cannot recover their original landmark geometry.
ALTER TABLE public.student_biometrics
    ADD COLUMN IF NOT EXISTS baseline_ear_metric text;
COMMENT ON COLUMN public.student_biometrics.baseline_ear_metric IS
    'NULL: legacy/unknown metric. pixel-v1: EAR distances in actual input pixels. Enrollment timestamp is enrolled_at.';
NOTIFY pgrst, 'reload schema';

-- Keep existing deployments compatible with the enrollment self-verify flow.
ALTER TABLE student_biometrics
    ADD COLUMN IF NOT EXISTS verify_attempts int NOT NULL DEFAULT 0;

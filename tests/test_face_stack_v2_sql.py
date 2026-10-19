"""The v2 face tables, trigger and RPCs on a real Postgres with pgvector.

Runs only when SMARTCHECK_TEST_PG_DSN points at a throwaway database, e.g.
    docker run -d --name sc-pgtest -e POSTGRES_PASSWORD=... -p 127.0.0.1:54329:5432 pgvector/pgvector:pg17
    set SMARTCHECK_TEST_PG_DSN=postgresql://postgres:...@127.0.0.1:54329/postgres
Each test class creates a fresh schema-less database state: Supabase roles and
the two referenced tables are stubbed, then the migration file is applied.
Never point this at the real project.
"""
import os
import unittest
import uuid
from pathlib import Path

import numpy as np

DSN = os.getenv("SMARTCHECK_TEST_PG_DSN")
MIGRATION = Path(__file__).resolve().parent.parent / "database" / "migrations" / "20261019_face_stack_v2.sql"

STUBS = """
DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;
DROP SCHEMA IF EXISTS extensions CASCADE; CREATE SCHEMA extensions;
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='anon') THEN CREATE ROLE anon NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='authenticated') THEN CREATE ROLE authenticated NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='service_role') THEN CREATE ROLE service_role NOLOGIN; END IF;
END $$;
GRANT USAGE ON SCHEMA public TO anon, authenticated, service_role;
CREATE TABLE public.users (id uuid PRIMARY KEY DEFAULT gen_random_uuid());
CREATE TABLE public.consent_logs (id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid REFERENCES public.users(id) ON DELETE CASCADE);
"""


def _unit(seed):
    v = np.random.default_rng(seed).standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


@unittest.skipUnless(DSN, "SMARTCHECK_TEST_PG_DSN not set (needs a throwaway Postgres with pgvector)")
class FaceStackV2SqlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import psycopg2
        cls.conn = psycopg2.connect(DSN)
        cls.conn.autocommit = True
        with cls.conn.cursor() as cur:
            cur.execute(STUBS)
            cur.execute(MIGRATION.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def _student(self):
        with self.conn.cursor() as cur:
            cur.execute("INSERT INTO public.users DEFAULT VALUES RETURNING id")
            uid = cur.fetchone()[0]
            cur.execute("INSERT INTO public.consent_logs (user_id) VALUES (%s) RETURNING id", (uid,))
            return uid, cur.fetchone()[0]

    def _complete(self, uid, consent, seeds, model="buffalo_l/w600k_r50", hash_="h"):
        from app.services.face_store import vector_literal
        import json
        templates = [{"embedding": json.loads(vector_literal(_unit(s))), "det_score": 0.8,
                      "face_px": 200, "yaw_deg": 0.0, "pose": "frontal"} for s in seeds]
        with self.conn.cursor() as cur:
            cur.execute("SELECT public.complete_face_enrollment(%s, %s, %s, %s, %s, %s::jsonb)",
                        (uid, model, consent, hash_, f"{uid}.jpg", json.dumps(templates)))
            return cur.fetchone()[0]

    def test_rls_and_grants(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT relname, relrowsecurity FROM pg_class WHERE relname IN ('face_enrollments','face_templates') ORDER BY 1")
            self.assertEqual(cur.fetchall(), [("face_enrollments", True), ("face_templates", True)])
            cur.execute("SELECT has_table_privilege('anon','public.face_templates','SELECT'), "
                        "has_table_privilege('authenticated','public.face_enrollments','SELECT'), "
                        "has_function_privilege('anon','public.match_face_templates(text,text,uuid,integer)','EXECUTE')")
            self.assertEqual(cur.fetchone(), (False, False, False))

    def test_complete_then_replacement_is_blocked(self):
        import psycopg2
        uid, consent = self._student()
        enrollment = self._complete(uid, consent, [1, 2, 3, 4, 5])
        with self.conn.cursor() as cur:
            cur.execute("SELECT status, count(t.id) FROM public.face_enrollments e "
                        "JOIN public.face_templates t ON t.enrollment_id = e.id WHERE e.id = %s GROUP BY status", (enrollment,))
            self.assertEqual(cur.fetchone(), ("completed", 5))
        attempts = [
            ("SELECT public.complete_face_enrollment(%s, 'buffalo_l/w600k_r50', NULL, 'x', NULL, '[{\"embedding\": [0]}]'::jsonb)", (uid,)),
            ("UPDATE public.face_enrollments SET integrity_hash = 'tampered' WHERE id = %s", (enrollment,)),
            ("UPDATE public.face_enrollments SET model_version = 'facenet512' WHERE id = %s", (enrollment,)),
            ("UPDATE public.face_enrollments SET status = 'pending' WHERE id = %s", (enrollment,)),
            ("INSERT INTO public.face_templates (enrollment_id, embedding) VALUES (%s, %s::extensions.vector)",
             (enrollment, "[" + ",".join(["0.01"] * 512) + "]")),
            ("UPDATE public.face_templates SET det_score = 0.1 WHERE enrollment_id = %s", (enrollment,)),
        ]
        for sql, params in attempts:
            with self.subTest(sql=sql.split()[0] + " " + sql.split()[1]):
                with self.conn.cursor() as cur, self.assertRaises(psycopg2.Error) as ctx:
                    cur.execute(sql, params)
                self.assertIn("already_enrolled", str(ctx.exception))
        # Clearing (withdrawal / admin reset) stays possible and takes the templates with it.
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM public.face_enrollments WHERE student_id = %s", (uid,))
            cur.execute("SELECT count(*) FROM public.face_templates WHERE enrollment_id = %s", (enrollment,))
            self.assertEqual(cur.fetchone()[0], 0)

    def test_match_excludes_self_and_other_models(self):
        uid_a, consent_a = self._student()
        uid_b, consent_b = self._student()
        uid_c, consent_c = self._student()
        self._complete(uid_a, consent_a, [11, 12, 13, 14, 15])
        self._complete(uid_b, consent_b, [21, 22, 23, 24, 25])
        self._complete(uid_c, consent_c, [11], model="other/model")
        from app.services.face_store import vector_literal
        with self.conn.cursor() as cur:
            cur.execute("SELECT student_id::text, round(similarity::numeric, 4) FROM public.match_face_templates(%s, %s, %s, 5)",
                        (vector_literal(_unit(11)), "buffalo_l/w600k_r50", str(uid_b)))
            rows = cur.fetchall()
        self.assertEqual(rows[0][0], str(uid_a))
        self.assertAlmostEqual(float(rows[0][1]), 1.0, places=3)
        self.assertNotIn(str(uid_b), [r[0] for r in rows])   # excluded: the enrolling student
        self.assertNotIn(str(uid_c), [r[0] for r in rows])   # other model_version

    def test_hmac_survives_pgvector_round_trip(self):
        from app.services.face_store import parse_vector
        from app.services.security_service import compute_template_integrity_hash
        uid, consent = self._student()
        embeddings = [_unit(s) for s in (31, 32, 33, 34, 35)]
        expected = compute_template_integrity_hash(str(uid), "buffalo_l/w600k_r50", embeddings, "salt")
        enrollment = self._complete(uid, consent, [31, 32, 33, 34, 35], hash_=expected)
        with self.conn.cursor() as cur:
            cur.execute("SELECT embedding::text FROM public.face_templates WHERE enrollment_id = %s ORDER BY id", (enrollment,))
            stored = [parse_vector(r[0]) for r in cur.fetchall()]
        self.assertEqual(compute_template_integrity_hash(str(uid), "buffalo_l/w600k_r50", stored, "salt"), expected)


if __name__ == "__main__":
    unittest.main()

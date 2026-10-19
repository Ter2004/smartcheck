# Face stack v2: Phase 2, face engine and new storage (2026-10-19)

## What changed

### `app/services/face_engine.py`

- **Engine switch:** `FACE_ENGINE` is `facenet512` (the default, the existing
  path, untouched) or `arcface`.
- **Model:** `FaceAnalysis("buffalo_l", allowed_modules=[detection, recognition, landmark_3d_68], providers=[CPU])`,
  then `prepare(ctx_id=-1, det_size=(640,640))`.
  - It is a singleton.
  - `warm_up()` runs one dummy detection and one dummy recognition, then logs
    the time and channel order.
  - `create_app()` calls it when `FACE_ENGINE=arcface` and `FACE_ENGINE_WARMUP=1`.
- **Input:** BGR uint8. `ensure_bgr(img, channel_order)` converts RGB at the
  boundary and refuses anything else (wrong dtype or shape gives `TypeError`,
  an unknown channel order gives `ValueError`).
- **One face only:** with no face it raises `NoFaceError`. When a second face
  is ≥ 0.5 of the main face's area it raises `MultipleFacesError`. This is the
  same rule and value as the head-turn check. Smaller background faces are
  ignored, which is a deliberate reading of "exactly one face" for classrooms.
  - Both errors subclass `ValueError` with DeepFace's "Face could not be
    detected" text, so existing route handlers apply unchanged.
- **`analyze()` returns:**
  - `normed_embedding` (float32);
  - `det_score`;
  - `face_px` (box height);
  - yaw/pitch/roll in degrees from `landmark_3d_68`;
  - the 5 keypoints, the kps yaw ratio and the eye-line roll.
- **Other functions:**
  - `extract_embedding(base64, include_metadata)` has the same signature as
    `face_service.extract_embedding` and delegates there for facenet512;
  - `detect_main()` runs SCRFD only, for the anti-spoof box (added in Phase 3).
- **`MODEL_VERSION`:** `facenet512` / `buffalo_l/w600k_r50`.
- **Fail-fast config:** `validate_config()` runs in `create_app()`. An unknown
  engine fails. With `arcface`, a missing or invalid `FACE_CLAHE`,
  `ARCFACE_MATCH_THRESHOLD`, `ARCFACE_DUP_THRESHOLD` or `ENROLL_SELF_SIM_MIN`
  also fails. They have no defaults, and `.env.example` documents them.

### Migration `database/migrations/20261019_face_stack_v2.sql`

**Not applied**, because the project is paused. Apply it in the SQL editor;
the verification query is at the end of the file.

- **`face_enrollments`:**
  - `student_id` unique, `model_version`, `status` (pending/completed);
  - `consent_id` → consent_logs, `integrity_hash`, `thumbnail_path`;
  - `created_at`, `completed_at`.
- **`face_templates`:**
  - `enrollment_id`, `embedding vector(512)`, `det_score`, `face_px`, `yaw_deg`, `pose`;
  - HNSW index `vector_cosine_ops`.
- **Access:** RLS on with no policies, and no grants to anon/authenticated.
- **Triggers:** they keep `prevent_completed_face_replacement`'s semantics.
  - A completed enrollment cannot be replaced or relabelled, and its templates
    cannot be added or changed.
  - DELETE (withdrawal, admin reset) stays possible and cascades to the templates.
- **RPC `complete_face_enrollment`:** enrollment, templates and HMAC in one
  transaction. It replaces a leftover pending row and refuses a completed one.
- **RPC `match_face_templates`:** the 1:N lookup over the same model's
  completed enrollments, excluding the enrolling student.
- `student_biometrics` and its trigger are not touched.

### HMAC and data access

- **HMAC** (`security_service.compute_template_integrity_hash`): user_id +
  model_version + embeddings. Values are taken as float32 first, because
  pgvector stores float4, so a value read back hashes identically. The legacy
  `compute_embedding_integrity_hash` is unchanged.
- **`app/services/face_store.py`:** load, is_enrolled, match, complete and
  delete for the new tables, all through the service role.

## Evidence

`python -m unittest tests.test_face_engine -v`: Ran 15 tests, **OK**.
- config fail-fast;
- channel-order guard;
- HMAC binds the model version and detects tampering;
- 0 / 1 / 2 faces on real frames;
- embedding norm ≈ 1, float32, 512-D;
- RGB declared as RGB gives the same embedding (cosine > 0.9999);
- RGB passed as if it were BGR changes the embedding, which is the mix-up the guard prevents;
- `extract_embedding` keeps its signature.

`SMARTCHECK_TEST_PG_DSN=... python -m unittest tests.test_face_stack_v2_sql -v`:
Ran 4 tests, **OK**. This runs on a throwaway `pgvector/pgvector:pg17`
container (pgvector 0.8.7).
- RLS and grants;
- 6 replacement attempts all raise `already_enrolled`, and DELETE still clears;
- match excludes the enrolling student and other model versions;
- the HMAC survives the pgvector text round trip.

Migration verification query on that container:

```
         item         |                     value
----------------------+-----------------------------------------------
 extension            | 0.8.7
 rls_face_enrollments | true
 rls_face_templates   | true
 trigger              | prevent_completed_face_enrollment_replacement
 trigger              | prevent_completed_face_template_change
 index                | face_templates_pkey
 index                | idx_face_templates_enrollment
 index                | idx_face_templates_embedding_hnsw
 END                  | END
(9 rows)
```

Full suite with the SQL tests: Ran 236 tests, FAILED (failures=1). The one
failure is the same `test_simulator_console_to_checkin` that fails on 43f5f7b.

**Route behaviour is unchanged in this phase.** With the default
`FACE_ENGINE=facenet512` nothing new runs at request time.

## Packaging note

insightface 2.1 (pure Python) is used because 0.7.3 needs MSVC to build here
(see phase0-recon.md). onnx is pinned to 1.16.2 while TensorFlow 2.15 is
installed (ml_dtypes 0.2.0). Both pins go into requirements in Phase 5.

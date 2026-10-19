# Face stack v2: Phase 0 recon and baselines (2026-10-19)

The branch `feat/face-stack-v2` was created from 43f5f7b. The 5 WIP files were
committed to `wip/eyeyaw-turn-yaw` (e5432c6) and not merged, together with
`scripts/test_turn_checker.js`, which tests that WIP. The untracked
`database/migrations/20261001_drop_rssi_columns.sql` is unrelated and stays untracked.

## Environment facts

- Python 3.11.9, Windows, venv at `venv/`. The tests use `unittest`; pytest is
  not installed. The command is `python -m unittest discover -s tests`.
- **insightface 0.7.3 cannot be installed here.** It ships only as an sdist and
  needs a C++ compiler: `error: Microsoft Visual C++ 14.0 or greater is required`.
  **insightface 2.1** (the same library, released 2026-09-08) is a pure-Python
  wheel with the same `FaceAnalysis` / `prepare` / `get` / `pose` API.
  It installed with:
  `pip install insightface==2.1 -c constraints.txt` plus `onnxruntime==1.24.4`
  and `ml-dtypes==0.2.0` as constraints, then `pip install onnx==1.16.2`.
  onnx 1.19 imports `ml_dtypes.float4_e2m1fn`, which needs ml_dtypes ≥ 0.5, but
  TensorFlow 2.15 pins ml_dtypes 0.2.0. `pip check`: "No broken requirements found".
- buffalo_l models are already cached in `~/.insightface/models/buffalo_l`.
- **The Supabase project `smartcheck-dev` is INACTIVE (paused).** Because of that:
  - no live SQL was run;
  - the trigger and consent schemas below come from the repo migrations;
  - SQL tests use a local `pgvector/pgvector:pg17` container instead.

## Legacy face stack usages

`git grep` finds 82 matching lines across `app/`, requirements, Dockerfile and
constraints. The code entry points are:

| Where | What |
|---|---|
| app/services/face_service.py:55-58 | `_call_deepface` → `DeepFace.<method>` under `_deepface_lock` |
| app/services/face_service.py:91-113 | `_detect_main_face`: RetinaFace on ≤480 px, no upscaling, largest face |
| app/services/face_service.py:116-128 | `_aligned_face`: DeepFace `align_img_wrt_eyes` + `project_facial_area` |
| app/services/face_service.py:131-141 | `_face_embedding`: Facenet512 via `represent(detector_backend="skip")` on an RGB crop |
| app/services/face_service.py:261-282 | `_run_fasnet_antispoof`: `modeling.build_model(task="spoofing", model_name="Fasnet")` |
| app/services/liveness_challenge.py:134-166 | `analyze_frame`: RetinaFace landmarks + Facenet512 on a level-rotated crop |
| app/runtime_check.py:10 | startup refuses to run without TensorFlow |
| requirements.txt:6,7,12 / constraints.txt / Dockerfile:5-112 | deepface 0.0.93, retina-face 0.0.18, tensorflow 2.15.0, TF-specific Docker steps |

## Decoding, channel order, CLAHE

- **Decoding:** `face_service._decode_image` (face_service.py:537) uses
  `cv2.imdecode(..., IMREAD_COLOR)`, which gives **BGR**. PIL is not used.
  Enrollment, check-in and liveness all decode through this function.
- **FaceNet input:** it receives RGB. face_service.py:139 flips the aligned BGR
  crop, because DeepFace flips once more when `detector_backend="skip"`.
- **CLAHE** (`normalize_illumination`, face_service.py:528) runs on the LAB
  L-channel, clip 2.0 and 8×8 tiles. It is applied:
  - to the whole frame before detection and embedding in `extract_embedding`
    (face_service.py:555), at both enrollment and check-in;
  - to the crop in `spoof_check_with_embedding` (595);
  - to the crop in `liveness_challenge.analyze_frame` (161).
  - Anti-spoof gets the raw frame, without CLAHE.

## Enrollment JS (app/static/js/enrollment_flow.js)

- Camera: `getUserMedia({facingMode:'user', width:640, height:480})`. Frames are
  `canvas.toDataURL('image/jpeg', 0.82/0.88)` at the video's native size.
- Flow: consent → light check (brightness ≥ 55, backlight edge/centre ≤ 2.2) →
  head-turn challenge → 5 frontal captures 2 s apart.
- Client gates during capture:
  - face height ≥ 0.30 of the frame;
  - frontal ±15°;
  - centering;
  - brightness and backlight;
  - Laplacian blur ≥ 20 at 640×480.
- What it POSTs:
  - `/student/api/liveness/challenge`;
  - `/student/api/liveness/verify` `{nonce, before, action_frames, after}`;
  - `/student/api/spoof_check` `{image}` about 8 times;
  - `/student/api/enroll` `{face_images: [5 JPEG data URLs]}`.

## /api/enroll steps (app/routes/student.py:319-835)

1. Consent:
   - the session flag;
   - the latest `consent_logs` row must have `consent_given=true`. Fail-closed on a DB error.
2. Input:
   - exactly 5 frames;
   - `baseline_ear` range;
   - a passed head-turn challenge within 15 min;
   - session retries < 3.
3. `atomic_enroll_attempt` RPC: at most 5 attempts per 24 h, fail-closed.
4. `server_validate_frame` on all 5 frames: 3-500 KB, JPEG magic bytes,
   160×120 to 1920×1080, Laplacian ≥ 8, channel std ≥ 2.
5. Pre-duplicate check of the session's liveness embeddings against everyone's,
   in batches of 50 (DUPLICATE_THRESHOLD 0.65).
6. Decode; temporal variance and EAR std are logged only.
7. Anti-spoof (`combined_spoof_score`) on all 5 frames; at least 4/5 must pass.
   A system failure returns 503.
8. Facenet512 on each frame. One frame without a face → `need_more`;
   two or more → `spoof_detected`.
9. Pairwise consistency at 0.80. One outlier → `need_more`; several → `restart_capture`.
10. Continuity: each embedding vs the session liveness embeddings must reach ≥ 0.80.
11. Duplicate check against all students, 0.65, with a 0.60-0.70 gray-zone log.
12. Save:
    - upsert `student_biometrics {face_embeddings, integrity_hash, consent, enrolled_at}`;
    - upload the first frame to the private Storage path `face-images/{user_id}.jpg`.

**HMAC payload** (security_service.py:89-112):
`HMAC-SHA256(EMBEDDING_INTEGRITY_SALT, json.dumps({"uid": user_id, "emb": sorted([[round(v, 6) for v in e] for e in embeddings])}, separators=(",", ":")))`

**Trigger** (database/migrations/20260918_prevent_face_reenrollment.sql;
the live definition could not be read while the project is paused). BEFORE
UPDATE on `student_biometrics` raises `already_enrolled` (23514) when:
- OLD has consent and non-empty embeddings;
- NEW has non-empty embeddings;
- and the embeddings or `enrolled_at` differ.

Clearing to NULL or `[]` stays allowed.

## Liveness (app/services/liveness_challenge.py)

- **Challenge:** a random sample from turn_left / turn_right / tilt_left /
  tilt_right. Enrollment gets 2 actions, check-in 1. Nonce, TTL 120 s.
- **Metric:**
  - `yaw_ratio` is the nose offset from the eye midpoint along the eye line,
    in eye distances, from RetinaFace's 5 points. Positive means turn_left.
  - Roll is the eye-line angle.
- **Thresholds:**
  - FRONTAL_MAX_YAW 0.18, TURN_MIN_YAW 0.20;
  - TILT_MIN_DEG 10, TILT_MAX_YAW 0.35;
  - MIN_FACE_SCORE 0.9;
  - a second face ≥ 0.5 of the main face's area fails the frame.
- **Continuity:** Facenet512 cosine between the `before` frame and each later
  frame. A frontal `after` frame needs ≥ CONTINUITY_THRESHOLD 0.80; the others
  need ≥ TURN_IDENTITY_MIN 0.45.
- **What the WIP branch changes:** only the browser. `checkin_flow.js` and
  `mediapipe_liveness.js` measure the turn with the server formula (eyeYaw,
  TURN_YAW 0.30) instead of noseRelX 0.65, plus a comment in
  liveness_challenge.py. Server thresholds are unchanged.

## Anti-spoof fusion (face_service.py:285-508)

- **Layers:**
  - fasnet, weight 0.70 (DeepFace Fasnet, on the RetinaFace box);
  - onnx, weight 0.10 (`antispoof.onnx`, MiniFASNetV2, 2.7 crop of a Haar box,
    BGR 0-255);
  - temporal, weight 0, log only.
- **Decision:** weights are renormalised, and the frame is rejected when the
  weighted score is ≥ 0.50.
- **Fail-closed rules:**
  - Fasnet dead → fail-close;
  - all layers failed → fail-close;
  - no face → `retry_capture`.
- **Loading:**
  - the ONNX session loads on first use (`_get_antispoof_session`);
  - DeepFace models (Facenet512, Fasnet) and RetinaFace also load on first
    request; nothing is warmed at startup;
  - `runtime_check` only checks that TensorFlow is installed.

## Consent

- `/student/api/consent` (student.py:285) sets the session's
  `consent_given_at` and inserts `consent_logs` with
  `{consent_type: biometric_enrollment, consent_given: true, consent_version: "1.1", ip, ua}`.
  A DB error is non-fatal there, but /api/enroll re-checks the DB fail-closed.
- Withdrawal (student.py:1079) inserts a `consent_given=false` row, nulls the
  biometric columns and removes the Storage photo.
- `consent_logs` (schema.sql:71): `id uuid` PK, `user_id`, `consent_type`,
  `consent_given`, `consent_version`, `ip_address`, `user_agent`, `created_at`.

## Baselines

**Tests.** `python -m unittest discover -s tests`: Ran 217 tests in 18.1 s,
**FAILED (failures=1)**. The failure is
`test_checkin_totp.IntegratedTOTPTests.test_simulator_console_to_checkin`
(500 != 200), already failing on 43f5f7b before any change.

**Pipeline timing** (`python scripts/bench_face_pipeline.py --runs 10`;
the raw result is in `bench-phase0-legacy.json`):
- in process, on the same service calls the routes make;
- 640×480 JPEG frames made from `test_images/real_*_sharp_*_ok.jpg`;
- no HTTP and no Supabase, so these are not end-to-end times;
- laptop CPU, Ryzen 7 8845HS.

| | median | p90 | min | max |
|---|---|---|---|---|
| enrollment (liveness verify 2 gestures + 5× spoof_check + /api/enroll) | 8181 ms | 8516 ms | 7945 | 8875 |
| check-in (head_turn: spoof + 3 liveness frames + embedding + match) | 1600 ms | 1744 ms | 1569 | 1746 |

Memory (Windows working set):
- 96 MB at start;
- **1211 MB after warm-up**;
- 1266 MB after 20 runs.

Warm-up, i.e. the first calls that load every model, took 40.3 s.

The end-to-end timing of 10 enrollments and 10 check-ins with test accounts
needs Supabase, which is paused, so it was not run.

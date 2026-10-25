# Face stack v2: final report (2026-10-20)

Go-live steps in order: [../go-live-checklist.md](../go-live-checklist.md).

Branch `feat/face-stack-v2`, made from 43f5f7b. It is not merged and not pushed.
Each phase has its own report with the commands and real output.

| phase | commit | report |
|---|---|---|
| 0 recon and baselines | d2dc68b | [phase0-recon.md](phase0-recon.md) |
| 1 evaluation harness | 084fed2 | [phase1-compare.md](phase1-compare.md) |
| 2 engine and storage | 8d3c058 | [phase2-engine-storage.md](phase2-engine-storage.md) |
| 3 liveness and anti-spoof | 1f5991b | [phase3-liveness-antispoof.md](phase3-liveness-antispoof.md) |
| 4 enrollment flow | f666cd3 (last commit that can still run facenet512) | [phase4-enrollment-flow.md](phase4-enrollment-flow.md) |
| 4b five poses (frontal + left/right/up/down in random order), 2026-10-26 | see `git log -- docs/face-stack-v2/phase4b-five-poses.md` | [phase4b-five-poses.md](phase4b-five-poses.md) |
| 5 check-in cutover, TF removal | a0a6e04, 68e4ef9 | [phase5-cutover.md](phase5-cutover.md) |
| 6 test kits | 7b981af | [../spoof_test.md](../spoof_test.md), [../realclass_test.md](../realclass_test.md) |

The 5 WIP files are parked on `wip/eyeyaw-turn-yaw` (e5432c6).

## What works (with evidence in the phase reports)

**Detection, recognition and pose:**
- InsightFace buffalo_l (SCRFD + ArcFace R50 + 3D-68 pose) on ONNX Runtime CPU.
- A padded retry for faces that fill the frame raised SCRFD from 19 to 101 of
  104 frames.

**Anti-spoof:** the DeepFace Fasnet weights now run on ONNX.
- Parity: max probability difference about 2e-6, and 100% decision agreement
  on 104 real live and spoof frames.

**Head turn:** yaw is read in degrees, with limits translated from the
calibrated ratio limits on 1,461 LFW faces (corr 0.953).

**Storage:**
- `face_enrollments` / `face_templates` with pgvector, RLS, and
  triggers that block replacing a completed enrollment;
- the save is one RPC transaction;
- the HMAC binds the model version;
- the 1:N duplicate search excludes the student's own rows.
- Tested on Postgres 17 + pgvector 0.8.7.

**Enrollment v2** (five poses since Phase 4b):
- three screens and a camera session with no buttons;
- the server issues a random order of left, right, up and down;
- 3 frontal + 4 posed templates are kept (7);
- a duplicate is blocked with a generic message plus a server-side audit row.
- 18 route tests cover: wrong turn, up/down sign, static photo, replay,
  capture order, person swap, second account, second person in frame,
  overwrite, spoof, and 3 failures.

**Check-in:**
- passive liveness;
- `face_templates` matching with HMAC verification;
- another model version gets "re-enrollment required" (409), never a failed match.

**Runtime:**
- no TensorFlow, DeepFace or torch;
- the clean venv and the Docker image both start;
- the image builds with the old stack absent and models warmed.

| measured | legacy | v2 |
|---|---|---|
| Docker image | 4.49 GB | 1.64 GB |
| RSS after warm-up (Windows / Docker) | 1213 / 1381 MB | 825 / 871 MB |
| model warm-up | 36.7 s | 5.2 s |
| check-in, head_turn mode, median | 1688 ms | 1031 ms |
| check-in, passive (new default), median | — | 514 ms |

These times are in-process server CPU work. They exclude network, database and browser.

**Tests:**
- The full suite in the TensorFlow-free venv ran 267 tests with 1 failure and
  1 skip. The failure is the same test that already failed on 43f5f7b; the skip
  is the DeepFace parity test.
- The JavaScript check scripts all pass.

## Bugs found and fixed along the way

1. **SQLAlchemy unpinned.** A fresh install gets 2.1.5, which makes
   `postgresql://` need psycopg v3, so any newly built image could not start.
   Pinned to 2.0.52.
2. **SCRFD missed close-up faces.** Fixed with the padded retry.
3. **`enroll_v2.quality` passed float32 to the Laplacian.** OpenCV rejects
   that, so every v2 enrollment would have failed. A test caught it before any
   real use.
4. **`.dockerignore` did not exclude `firmware/`.** The git-ignored
   `firmware/smartcheck_ble/secret.h` (the ESP32 room key) was copied into any
   image built from the working folder.
   - Images built from a clean checkout do not have it.
   - `firmware/` is now excluded, and the local v2 image was rebuilt and checked
     ("no firmware in image").
   - If an image was ever built from the working folder and pushed or deployed,
     rotate that room key.

## Not done or not verifiable here

These need you, a camera, or a running database.

1. **Thresholds and CLAHE.** Run `scripts/compare_models.py` on your own photos
   and choose `FACE_CLAHE`, `ARCFACE_MATCH_THRESHOLD`, `ARCFACE_DUP_THRESHOLD`
   and `ENROLL_SELF_SIM_MIN`.
   - The app refuses to start without them (fail-fast).
   - The LFW proxy run is only a smoke test of the harness.
2. **Supabase.** The project `smartcheck-dev` was paused for the whole session.
   - Apply `database/migrations/20261019_face_stack_v2.sql` and run its
     verification query.
   - The end-to-end timing of 10 enrollments and 10 check-ins with test
     accounts was not possible.
3. **Real devices.** The v2 enrollment page has not run on a phone or laptop
   camera, including iPhone Safari. The browser-vs-server comparison (≥ 20
   turned and ≥ 20 up/down frames) needs those logs; use `scripts/report_turn_yaw.py`.
4. **Approvals still yours:** the Fasnet ONNX tolerance (Phase 3) and the
   turn-bucket deviation (Phase 4: turned frames must pass 20.4°, not "≤ 15°").
5. **Re-enrollment.** Every student re-enrolls. `student_biometrics` and its
   trigger were left untouched by design.
6. **Spoof and real-class tests:** see docs/spoof_test.md and docs/realclass_test.md.

## Dead code and dead paths left in place (listed as asked)

- **The trusted-device 0.70 path.** `SAME_DEVICE_THRESHOLD` was only reachable
  with a device token, and no route ever issues one (`create_device_token` has
  no caller).
  - It was not enabled.
  - The v2 check-in uses `ARCFACE_MATCH_THRESHOLD` alone.
  - The constant remains in `face_service.py` with a comment.
- **The v1 enrollment page and routes** are no longer reachable from the UI:
  - `enroll_face.html` and `enrollment_flow.js`;
  - `/student/api/spoof_check`, `/student/api/liveness/*` and `/student/api/reset-liveness`;
  - the legacy body of `/student/api/enroll`.
  - Their tests run with `FACE_ENGINE=facenet512` in the test app config.
  - Delete them after the real-class test confirms v2.
- **The admin biometrics page and dashboard count** still read
  `student_biometrics`, so v2 enrollments do not appear there yet. The teacher
  photo view does read the v2 thumbnail.

## License

The InsightFace pretrained models (buffalo_l) are for **non-commercial research
only**. That is fine for this capstone; a commercial deployment needs other
weights or a licence. MiniFASNet (Silent-Face) is Apache-2.0, per the header
of DeepFace's FasNet.py.

## Local artifacts from this session

- **Docker image:** `smartcheck:face-stack-v2` (1.64 GB).
  - The legacy comparison image was removed after measuring; its numbers are
    in `bench-phase5-docker.json` and phase5-cutover.md.
- **Test database container:** `sc-pgtest` (pgvector/pgvector:pg17 on 127.0.0.1:54329).
- **To remove them:** `docker rm -f sc-pgtest && docker rmi smartcheck:face-stack-v2`.

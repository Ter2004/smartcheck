# Face stack v2: Phase 5, check-in cutover and TensorFlow removal (2026-10-20)

## What changed

1. **Check-in face step** (`api_checkin._verify_face_v2`, commit a0a6e04):
   - it reads `face_enrollments` + `face_templates` for the active `model_version`;
   - it verifies the HMAC, which covers user_id, model_version and embeddings;
   - it embeds the live frame with ArcFace and checks max cosine ≥ `ARCFACE_MATCH_THRESHOLD`.
   - A missing enrollment of the active model version, including a student who
     only has a legacy FaceNet512 row, returns **409 "re-enrollment required"**
     (`reenroll_required: true`), never a failed match.
   - The attendance tail (duplicate guard, present/late, insert) moved into
     `_record_attendance` unchanged.
2. **Anti-spoof** always runs `fasnet_onnx` on the SCRFD box, plus the existing
   `antispoof.onnx`. The `ANTISPOOF_FASNET` switch was removed.
3. **`CHECKIN_LIVENESS=passive`** is set in `.env` and `.env.example`.
   `create_app` logs `check-in liveness: passive` (log below).
4. **Removed from runtime:**
   - deepface, tensorflow, tf-keras and retina-face are out of `requirements.txt`;
   - torch was never a runtime pin and is now only in `requirements-dev.txt`,
     together with the libraries needed by the export, parity and old calibration scripts.
   - `face_service` lost every DeepFace/RetinaFace function.
   - `liveness_challenge.analyze_frame` is InsightFace only.
   - `runtime_check` checks onnxruntime, insightface, the buffalo_l files and the Fasnet ONNX files.
   - `FACE_ENGINE` defaults to arcface. `facenet512` is refused at startup with
     a pointer to commit f666cd3, the last state that can run the old stack.
5. **Dockerfile:**
   - two stages, with no compiler needed: insightface 2.1 is pure Python and
     everything else ships wheels;
   - pinned versions;
   - buffalo_l downloaded at build time, keeping only det_10g, w600k_r50 and 1k3d68;
   - a build-time assertion that tensorflow, deepface and torch are absent;
   - a model warm-up check during the build.
   - `.dockerignore` now also excludes `.build/`, the local LFW and CelebA
     datasets, and the private face-photo folders.
6. **Fixed a startup bug that was already there:** SQLAlchemy was unpinned.
   - A fresh install today gets 2.1.5, which maps `postgresql://` to psycopg v3.
     psycopg v3 is not installed, so a newly built image, old or new, would not
     start (`ModuleNotFoundError: No module named 'psycopg'`).
   - The local venv has 2.0.52, which is why this never showed up.
   - Pinned `SQLAlchemy==2.0.52`.

**Not changed:**
- `student_biometrics` and its trigger;
- BLE / TOTP / ESP32, the scheduler, and the attendance status logic;
- the dead 0.70 trusted-device path. It was not enabled and the face step no
  longer reads it; see the final report.

## Evidence

### Clean venv without the old stack

- Built from `requirements.txt` only, plus the same headless-OpenCV swap the Dockerfile does.
- `find_spec`: tensorflow False, deepface False, torch False, retinaface False.
- site-packages is **665 MB**, against 3.2 GB in the dev venv with TensorFlow.
- `pip check` reports only "insightface 2.1 requires opencv-python, which is
  not installed". That is expected: opencv-python-headless provides the same `cv2`.
- Tests: `SMARTCHECK_TEST_PG_DSN=… python -m unittest discover -s tests` gives
  **Ran 267, FAILED (failures=1, skipped=1)**.
  - The failure is the existing `test_simulator_console_to_checkin`, failing
    since 43f5f7b: its test app has no `auth` blueprint, so `login_required`
    cannot build the redirect URL.
  - The skip is the DeepFace parity test, which needs deepface installed.

### Startup

In the clean venv (thresholds are test values passed as environment variables):

```
INFO smartcheck.app — [SmartCheck] proximity method: ble
INFO smartcheck.app — [SmartCheck] check-in liveness: passive
INFO smartcheck.app — [SmartCheck] face engine: arcface (model_version=buffalo_l/w600k_r50)
INFO smartcheck.face_engine — [FACE_ENGINE] arcface buffalo_l/w600k_r50 warm-up 3.37s input=BGR uint8 (cv2.imdecode order)
create_app OK; CHECKIN_LIVENESS = passive ; FACE_ENGINE = arcface
```

Without the thresholds, startup refuses as intended:
`RuntimeError: [SmartCheck] FACE_ENGINE=arcface needs FACE_CLAHE, ARCFACE_MATCH_THRESHOLD, ARCFACE_DUP_THRESHOLD, ENROLL_SELF_SIM_MIN (values from scripts/compare_models.py; there are no defaults).`

Inside the v2 container, against a throwaway Postgres:
`[FACE_ENGINE] … warm-up 4.81s` … `create_app OK in container; CHECKIN_LIVENESS = passive`.

### Images

`docker images`:

| image | size | build time |
|---|---|---|
| `smartcheck:legacy-43f5f7b` (old Dockerfile at 43f5f7b) | **4.49 GB** | 7 m 23 s |
| `smartcheck:face-stack-v2` | **1.64 GB** | 3 m 19 s (2 m 23 s rebuild) |

### Pipeline timing and memory

The benchmark runs in process, on the route-level service calls, with no HTTP
and no Supabase (`scripts/bench_face_pipeline.py --runs 10`).

**Windows, same laptop and same frames as Phase 0**
(`bench-phase0-legacy.json` vs `bench-phase5-v2.json`):

| | legacy (Phase 0) | v2 |
|---|---|---|
| RSS after warm-up | 1213 MB | **825 MB** |
| warm-up (load every model) | 36.7 s | **5.2 s** |
| check-in, head_turn mode (3 frames), median / p90 | 1688 / 1824 ms | **1031 / 1042 ms** |
| check-in, passive (the new default), median / p90 | (not timed) | **514 / 522 ms** |
| enrollment server work, median / p90 | 8499 / 8937 ms * | **4115 / 4189 ms** * |

\* The two enrollment rows measure different work:
- **legacy:** liveness verify (4 frames + 2 embeddings), 5 × `/api/spoof_check`
  and `/api/enroll` on 5 frames;
- **v2:** `/api/enroll` on 8 frames;
- neither includes the database duplicate search.

**Docker (Linux), both images on the same 3 frames, run back to back** (`bench-phase5-docker.json`):

| | legacy image (43f5f7b) | v2 image |
|---|---|---|
| RSS after warm-up | 1381 MB | **871 MB** |
| warm-up | 33.8 s | **3.95 s** |
| check-in, head_turn, median / p90 | 1676 / 1901 ms | **1100 / 1360 ms** |
| check-in, passive, median / p90 | — | **579 / 662 ms** |
| enrollment server work *, median / p90 | 8535 / 10782 ms | **4408 / 4516 ms** |

The legacy p90 values are noisy: two short `docker run` checks overlapped that
run. An earlier legacy run overlapped a whole docker build, so it is discarded.
That run still measured RSS 1395 MB.

## Still to decide or do (yours)

- **Phase 1 decisions:** go/no-go, `FACE_CLAHE` and the three thresholds, taken
  from your own photos. The app does not start until they are set.
- **Apply the migration** `20261019_face_stack_v2.sql`. The project is paused right now.
- **Approve the Fasnet tolerance** (Phase 3).
- **Every student re-enrolls.** Legacy-only students are told
  "re-enrollment required" at check-in, and `student_biometrics` stays untouched.
- **Rollback:** commit f666cd3 still runs the old stack (`FACE_ENGINE=facenet512`
  in a venv that has TensorFlow).

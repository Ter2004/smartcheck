# Go-live checklist (branch `feat/face-stack-v2`)

Do the steps in order. Each step says how to check it worked. Commands assume
the project folder `smartcheck_project/` and its `venv`.

## 1. Database (Supabase SQL editor)

| # | Do | Check |
|---|---|---|
| 1.1 | Run `database/migrations/20261019_face_stack_v2.sql` (the current file: since 2026-10-26 it stores five poses; running it again over an earlier copy updates the pose check). **Done on smartcheck-dev, 2026-10-26.** | Run the verification query at the end of the file. You should see `extension` 0.8.x, RLS `true` on both tables, 2 triggers on `face_enrollments` / `face_templates` (the old `prevent_completed_face_replacement ON student_biometrics` also shows; leave it), 3 indexes, a `pose_check` row listing frontal, left, right, up and down, and `END` as the last row. |
| 1.2 | Optional, only if you use LINE Beacon: run `database/migrations/20261020_line_beacon.sql` | Run its verification query; `END` is the last row. |

## 2. Face settings (`.env`)

| # | Do | Check |
|---|---|---|
| 2.1 | Collect photos with each person's permission, in `dataset/<student>/` (git-ignored), named `<NN>_<pose>_<lighting>.jpg`. Per person: `01`–`03` frontal, `04` left, `05` right, `06` up, `07` down (the enrollment), then `08`+ frontal check-in photos in different lighting, e.g. `08_frontal_dark.jpg`. Use the phone front camera at arm's length. Several people. | The script prints a WARNING for a pose word in the wrong place. |
| 2.2 | Run `python scripts/compare_models.py --dataset dataset --out compare-out` | The tables print per config and per lighting. Read `docs/face-stack-v2/phase1-compare.md` on how to read them. |
| 2.3 | Set `FACE_CLAHE` (on/off), `ARCFACE_MATCH_THRESHOLD`, `ARCFACE_DUP_THRESHOLD` and `ENROLL_SELF_SIM_MIN` from those tables | `python run.py` logs `face engine: arcface` and a warm-up time. If a value is missing, startup stops and names it. |
| 2.4 | Keep `CHECKIN_LIVENESS=passive` (already set) | The startup log shows `check-in liveness: passive`. |

## 3. Fixed https URL (needed for LINE, recommended for everything)

| # | Do | Check |
|---|---|---|
| 3.1 | Get a stable https address: your own domain with a Cloudflare named tunnel, or a cloud deploy (`Dockerfile`; image about 1.6 GB). Quick-tunnel URLs change on every restart. | Open the URL on a phone and log in. |
| 3.2 | Set `PUBLIC_BASE_URL=https://…` | QR codes and LINE buttons open this address. |

## 4. Room proximity: choose what students use

| Option | Turn on | Notes |
|---|---|---|
| BLE board + button (Android Chrome, PC Chrome/Edge) | already on (`CHECKIN_PROXIMITY_METHOD=ble`) | iPhone Safari cannot use it |
| Teacher screen code / QR (any phone) | `CHECKIN_SESSION_CODE=1` | The teacher opens "แสดงรหัสเช็คชื่อ (สำรอง)" on the session page. A code can be forwarded within about 1 minute. |
| LINE Beacon (walk in, tap the LINE button) | `LINE_BEACON_ENABLED=1` plus `LINE_CHANNEL_SECRET`, `LINE_CHANNEL_ACCESS_TOKEN` and `PUBLIC_BASE_URL` | Follow every step in `docs/line_beacon.md`, including the trial measurements |

## 5. Students

| # | Do | Check |
|---|---|---|
| 5.1 | Tell students to re-enroll. The dashboard already shows "ต้องลงทะเบียนใหม่" to anyone with an old enrollment. | The admin page ข้อมูลใบหน้า counts the new enrollments. |
| 5.2 | Withdrawal requests now go through the admin page ข้อมูลใบหน้า, button "ถอนความยินยอม / ลบข้อมูลใบหน้า". | The row goes back to "ยังไม่ได้ลง". |

## 6. Tests with people (the committee's request)

Keep the server log of every test session:
`.\venv\Scripts\python.exe run.py 2>&1 | Tee-Object -FilePath server.log`.
Windows PowerShell 5.1 writes this file as UTF-16; the report scripts read
both UTF-8 and UTF-16.

| # | Do | Kit |
|---|---|---|
| 6.1 | Real class, 20-30 people, timed against a stopwatch roll call | `docs/realclass_test.md` + `scripts/realclass_report.py` |
| 6.2 | Fake faces (printed photo, phone screen, laptop screen), ≥ 20 attempts each | `docs/spoof_test.md` + `scripts/spoof_report.py` |
| 6.3 | Enroll on Android and iPhone until the log has ≥ 20 turned and ≥ 20 up/down frames; compare browser and server angles and look for `up_missing` / `down_missing` failures | `scripts/report_turn_yaw.py` on the server log; `docs/face-stack-v2/phase4b-five-poses.md` |

## 7. After the tests pass

- Merge `feat/face-stack-v2`.
- Then delete the v1 enrollment code (`enroll_face.html`, `enrollment_flow.js`,
  `/student/api/spoof_check`, `/student/api/liveness/*`, the legacy `/api/enroll` body)
  and the tests that pin `FACE_ENGINE=facenet512`.
- Rollback until then: commit `f666cd3` runs the old FaceNet512 stack, in a venv
  that has `requirements-dev.txt`.

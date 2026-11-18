# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Local development
python run.py                    # Flask dev server on localhost:5000

# Production (Railway / Docker)
gunicorn "app:create_app()" --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 300

# Docker
docker build -t smartcheck .
docker run -p 8080:8080 --env-file .env smartcheck
```

Python tests use `unittest` (pytest is not required):

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s tests -p 'test_checkin_selection.py' -v
```

Use the project's virtual environment. Face tests use the installed InsightFace models (~/.insightface/models/buffalo_l) and skip when absent; set SMARTCHECK_TEST_PG_DSN to a throwaway pgvector Postgres to run tests/test_face_stack_v2_sql.py. Dev-only scripts that need DeepFace/torch are listed in requirements-dev.txt. JavaScript checks are in `scripts/test_*.js` and run individually with Node. Browser and real-device checks are still needed for camera/BLE flows, including iPhone.

## Architecture

### Entry point & factory
`run.py` calls `app.create_app()` (`app/__init__.py`).  
`create_app()` wires together: SQLAlchemy (Flask-Session backend), Flask-Limiter, two Supabase clients, five Blueprints, Jinja2 filters, security headers, error handlers, and the APScheduler.

### Blueprints
| Module | URL prefix | Who uses it |
|---|---|---|
| `app/routes/auth.py` | _(none)_ | `/login`, `/logout`, `/` |
| `app/routes/admin.py` | `/admin` | Admin users |
| `app/routes/teacher.py` | `/teacher` | Teachers |
| `app/routes/student.py` | `/student` | Students (incl. enrollment) |
| `app/routes/api_checkin.py` | _(none)_ | `POST /api/checkin` |

All page routes must be guarded with `@login_required` then `@role_required('role')` (both decorators defined in `auth.py`). JSON API endpoints additionally need `@csrf_protect` from `security_service.py`.

### Supabase client pattern
Two clients are imported from `app`:
```python
from app import supabase        # anon key — Supabase Auth sign-in only
from app import supabase_admin  # service key — all data reads/writes (bypasses RLS)
```
All application data access uses `supabase_admin`. `supabase` (anon) is used only for `auth.sign_in_with_password`.

### Face pipeline (face stack v2: `face_engine.py`, `face_service.py`, `enroll_v2.py`)
InsightFace buffalo_l on ONNX Runtime CPU; DeepFace/TensorFlow were removed (docs/face-stack-v2/). Server-side:
1. `server_validate_frame()` — zero-trust JPEG validation (size, magic bytes, blur, color variance)
2. `combined_spoof_score()` — Fasnet (0.70, `fasnet_onnx.py`, same weights as DeepFace's, on the SCRFD box) and ONNX (0.10) vote; temporal variance is logged only. Fasnet unavailable → fail-close.
3. `face_engine.analyze()` — SCRFD detection (padded retry for close-ups), one significant face, ArcFace w600k_r50 embedding (512-D, unit length), 3D-68 head pose; optional CLAHE (`FACE_CLAHE`)
4. Check-in: max cosine against the student's `face_templates` (active `model_version`, HMAC-verified) ≥ `ARCFACE_MATCH_THRESHOLD`; another model version → "re-enrollment required" (409)

Enrollment (`/student/enroll` → `enroll_face_v2.html` + `enrollment_v2.js`, server `app/services/enroll_v2.py`): consent → one camera session (align, look + blink, then left/right/up/down in a server-issued random order; left/right by calibrated yaw, up/down by pitch direction only) → ≤ 12 frames → checks → 3 frontal + 4 posed templates in `face_enrollments`/`face_templates` (migration `20261019_face_stack_v2.sql`) via one RPC transaction, 1:N duplicate block via `match_face_templates`. See docs/face-stack-v2/phase4b-five-poses.md. Check-in liveness: `CHECKIN_LIVENESS=passive` (head_turn still supported; yaw limits in degrees in `liveness_challenge.py`).

Thresholds: `FACE_CLAHE`, `ARCFACE_MATCH_THRESHOLD`, `ARCFACE_DUP_THRESHOLD`, `ENROLL_SELF_SIM_MIN` are required env settings with no defaults (choose with `scripts/compare_models.py`); `SPOOF_DECISION_THRESHOLD = 0.50`. The FaceNet512 constants left in `face_service.py` belong to the unreachable v1 enrollment code.

### Security layer (`app/services/security_service.py`)
- **Device token**: HMAC-SHA256 signed token binding `user_id + device_fingerprint`, verified on every check-in
- **Embedding integrity**: HMAC-SHA256 over stored embeddings using `EMBEDDING_INTEGRITY_SALT`; tamper detection before face compare
- **CSRF**: `@csrf_protect` (JSON APIs) and `@csrf_protect_form` (form POSTs) decorators
- **Rate limiting**: Flask-Limiter per-user (uses `user_id` as key if logged in, falls back to IP). Redis required for multi-worker deployments.

### Scheduler (`app/scheduler.py`)
APScheduler (BackgroundScheduler, Asia/Bangkok timezone) runs two jobs:
- `auto_manage_sessions` every 1 min — creates/closes class sessions from `schedules` table
- `keep_alive` every 3 min — pings Supabase to prevent idle HTTP/2 connection drops

Scheduler startup is deferred to the first request via `before_request`, guarded by a lock so concurrent requests start it only once per process. The first session-management tick is scheduled immediately. A reloader supervisor receives no requests and does not start the scheduler; startup does not depend on `WERKZEUG_RUN_MAIN`. Multiple serving processes each have their own scheduler.

Each tick also closes expired open sessions from previous days using current schedules and Bangkok dates. Sessions that cannot be matched remain open and warn once per session per process until closed or matched.

### Database schema (`database/schema.sql`)
Tables: `users`, `student_biometrics`, `consent_logs`, `audit_logs`, `courses`, `course_enrollments`, `beacons`, `sessions`, `schedules`, `attendance`. All use UUID PKs. RLS is enabled on every table — service key bypasses it.

Key stored procedure: `atomic_enroll_attempt(user_id, max, window_h)` — atomically increments `enrollment_attempts` with `FOR UPDATE` to prevent race conditions.

Migrations live in `database/migrations/`. Apply via Supabase SQL Editor or psql.

### Session & config
Flask-Session uses SQLAlchemy backend (`flask_sessions` table on `DATABASE_URL`). Sessions expire after 1 hour. In production: `SESSION_COOKIE_SECURE=True`, requires HTTPS.

Enrollment is the v2 flow above. The v1 page (`enroll_face.html`, `enrollment_flow.js`) and its routes (`/student/api/spoof_check`, `/student/api/liveness/*`, the legacy body of `/api/enroll`) are no longer reachable from the UI and are kept only until the real-class test confirms v2.

## Conventions
- All templates extend `base.html`; flash categories: `success`, `danger`, `warning`, `info`
- Datetime display: use the `|thai_time` Jinja2 filter (converts UTC → UTC+7)
- The `enroll` endpoint (`/student/api/enroll`) is security-critical — do not modify without explicit approval
- Audit events go through `log_audit_event()` in `security_service.py`
- Environment: Python 3.11, deployed on Railway; `nixpacks.toml` and `Dockerfile` both exist

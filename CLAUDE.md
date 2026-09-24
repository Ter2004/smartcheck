# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Local development
python run.py                    # Flask dev server on localhost:5000

# Production (Railway / Docker)
gunicorn "app:create_app()" --bind 0.0.0.0:$PORT --workers 1 --timeout 120

# Docker
docker build -t smartcheck .
docker run -p 8080:8080 --env-file .env smartcheck
```

Python tests use `unittest` (pytest is not required):

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s tests -p 'test_checkin_selection.py' -v
```

Use the project's virtual environment. Face tests need the installed DeepFace/TensorFlow dependencies; tests mock model calls where appropriate. JavaScript checks are in `scripts/test_*.js` and run individually with Node. Browser and real-device checks are still needed for camera/BLE flows, including iPhone.

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

### Face pipeline (`app/services/face_service.py`)
The core AI pipeline runs entirely server-side:
1. `server_validate_frame()` — zero-trust JPEG validation (size, magic bytes, blur, color variance)
2. `combined_spoof_score()` — only Fasnet (0.70) and ONNX (0.10) vote; Moiré, screen texture and temporal variance are computed and logged only. Fasnet unavailable → fail-close.
3. `extract_embedding()` — CLAHE normalization → FaceNet512 (512-D vector) via DeepFace
4. `verify_face_multi()` — cosine similarity against all stored embeddings; decision on best (not average)

Liveness is decided server-side by a head-turn challenge (`app/services/liveness_challenge.py`): the server picks the turn order, RetinaFace landmarks confirm each turn and FaceNet confirms the same face. Enrollment uses two turns (`/student/api/liveness/*`; `/api/enroll` and `/api/self_verify` require a pass within 15 min). Check-in uses one turn (`/api/checkin/liveness/challenge`) verified inside `/api/checkin` against the submitted frame. `CHECKIN_LIVENESS=passive` reverts check-in to hands-free capture.

Key thresholds (edit in `face_service.py` top section):
- `SAME_DEVICE_THRESHOLD = 0.70`, `NEW_DEVICE_THRESHOLD = 0.80`
- `SELF_VERIFY_THRESHOLD = 0.80`, `DUPLICATE_THRESHOLD = 0.65`
- `SPOOF_DECISION_THRESHOLD = 0.50`

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

`ENROLL_FLOW_MODE` env var controls the enrollment UI variant: `"classic"` (default) or `"circular"`.

## Conventions
- All templates extend `base.html`; flash categories: `success`, `danger`, `warning`, `info`
- Datetime display: use the `|thai_time` Jinja2 filter (converts UTC → UTC+7)
- The `enroll` endpoint (`/student/api/enroll`) is security-critical — do not modify without explicit approval
- Audit events go through `log_audit_event()` in `security_service.py`
- Environment: Python 3.11, deployed on Railway; `nixpacks.toml` and `Dockerfile` both exist

import logging
from flask import Flask
from supabase import create_client, Client
from app.config import Config
from flask_session import Session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

_log = logging.getLogger("smartcheck.app")

# Supabase clients (initialized in create_app)
supabase: Client = None          # ใช้ anon key — เรียกผ่าน RLS
supabase_admin: Client = None    # ใช้ service key — bypass RLS (สำหรับ admin operations)


def get_rate_limit_key():
    """
    Per-user rate limiting for authenticated endpoints.
    ใช้ user_id ถ้า login แล้ว → นักศึกษาหลายคนบน WiFi เดียวกัน (NAT) ไม่กระทบกัน
    Fallback เป็น IP สำหรับ endpoint ที่ยังไม่ได้ login (เช่น login เอง)
    """
    try:
        from flask import session as _session
        uid = _session.get("user_id")
        if uid:
            return f"user:{uid}"
    except Exception:
        pass
    return f"ip:{get_remote_address()}"


def _limiter_storage_uri() -> str:
    """ใช้ Redis ถ้ามี REDIS_URL — มิฉะนั้น fallback เป็น in-memory (single-process only)"""
    from app.config import Config
    redis_url = getattr(Config, "REDIS_URL", "") or ""
    if redis_url:
        return redis_url
    _log.warning(
        "[SmartCheck] REDIS_URL not set — using in-memory rate limiter. "
        "Multi-worker deployments (gunicorn) will have per-process counters."
    )
    return "memory://"


# Rate limiter — created at module level so blueprints can import it before create_app().
# init_app(app) is called inside create_app() to bind it to the Flask instance.
limiter = Limiter(
    key_func=get_rate_limit_key,
    default_limits=["200 per hour"],
    storage_uri=_limiter_storage_uri(),
)


def _refresh_clients():
    """สร้าง supabase client ใหม่ (เรียกเมื่อ connection ตาย)"""
    global supabase, supabase_admin
    supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_ANON_KEY)
    supabase_admin = create_client(Config.SUPABASE_URL, Config.SUPABASE_SERVICE_KEY)


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # --- Structured logging (B10) ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # Suppress noisy third-party loggers
    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    logging.getLogger("deepface").setLevel(logging.WARNING)

    # --- Server-side sessions — Sprint 1A ---
    Session(app)

    # --- Rate limiter — Sprint 1C ---
    limiter.init_app(app)

    # --- Init Supabase ---
    _refresh_clients()

    # --- Register Blueprints ---
    from app.routes.auth import auth_bp
    from app.routes.admin import admin_bp
    from app.routes.teacher import teacher_bp
    from app.routes.student import student_bp
    from app.routes.api_checkin import api_checkin_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp, url_prefix="/admin")
    app.register_blueprint(teacher_bp, url_prefix="/teacher")
    app.register_blueprint(student_bp, url_prefix="/student")
    app.register_blueprint(api_checkin_bp)

    # --- Jinja2 filter: แปลง UTC → Thai time (UTC+7) ---
    from datetime import datetime, timedelta
    def to_thai_time(utc_str, fmt='%d/%m/%y %H:%M'):
        if not utc_str:
            return '—'
        try:
            s = str(utc_str).replace('Z', '+00:00')
            dt = datetime.fromisoformat(s)
            thai = dt + timedelta(hours=7)
            return thai.strftime(fmt)
        except Exception:
            return str(utc_str)[:16]
    app.jinja_env.filters['thai_time'] = to_thai_time

    # --- Auto-reconnect เมื่อ httpx connection ตาย ---
    import httpx
    @app.before_request
    def reconnect_if_needed():
        pass  # placeholder — reconnect จะเกิดใน error handler

    @app.errorhandler(httpx.RemoteProtocolError)
    @app.errorhandler(httpx.ReadTimeout)
    def handle_connection_error(e):
        from flask import redirect, request as freq
        _refresh_clients()
        return redirect(freq.url)

    @app.errorhandler(429)
    def rate_limit_exceeded(e):
        from flask import jsonify, request as freq
        retry_after = getattr(e, "retry_after", None) or getattr(e, "description", None)
        # JSON response for API endpoints; plain redirect for page requests
        if freq.is_json or freq.path.startswith("/api/") or freq.path.startswith("/student/api/"):
            return jsonify({
                "status":      "error",
                "message":     "คำขอมากเกินไป กรุณารอสักครู่แล้วลองใหม่",
                "retry_after": str(retry_after) if retry_after else None,
            }), 429
        from flask import flash, redirect, url_for
        flash("คำขอมากเกินไป กรุณารอสักครู่แล้วลองใหม่", "warning")
        return redirect(url_for("auth.login"))

    # --- Security headers (A9) ---
    @app.after_request
    def add_security_headers(response):
        # Content-Security-Policy — adjust CDN allowlist as needed
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' cdn.tailwindcss.com cdn.jsdelivr.net "
                "'unsafe-inline'; "   # required for inline Tailwind config block
            "style-src 'self' fonts.googleapis.com 'unsafe-inline'; "
            "font-src fonts.gstatic.com data:; "
            "img-src 'self' data: blob:; "
            "media-src 'self' blob:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"]         = "DENY"
        response.headers["Referrer-Policy"]          = "strict-origin-when-cross-origin"
        return response

    # --- Start Scheduler ---
    from app.scheduler import start_scheduler
    start_scheduler()

    return app

import logging
import os
import sys
from dotenv import load_dotenv

_log = logging.getLogger("smartcheck.config")

load_dotenv()

_IS_PRODUCTION = os.getenv("FLASK_ENV") == "production" or os.getenv("FLASK_DEBUG", "1") == "0"


def _require_env(key: str, fallback: str) -> str:
    """
    ใน production: raise RuntimeError ถ้าไม่ set env var
    ใน dev: ใช้ fallback แต่ print warning ชัดเจน
    """
    value = os.getenv(key)
    if value:
        return value
    if _IS_PRODUCTION:
        raise RuntimeError(
            f"[SmartCheck] CRITICAL: environment variable '{key}' is not set. "
            f"Application cannot start in production without it."
        )
    _log.warning(
        f"[SmartCheck] '{key}' not set — using insecure dev fallback. "
        f"DO NOT use in production."
    )
    return fallback


class Config:
    PERFORMANCE_LOG_ENABLED = os.getenv("PERFORMANCE_LOG_ENABLED", "false").lower() == "true"
    ENROLLMENT_STATUS_CACHE = os.getenv("ENROLLMENT_STATUS_CACHE", "true").lower() == "true"
    # Enable only after applying 20260919_enrollment_status.sql.
    ENROLLMENT_STATUS_RPC = os.getenv("ENROLLMENT_STATUS_RPC", "false").lower() == "true"
    # v2 enrollment attempts per student per 24 h (atomic_enroll_attempt). Raise only while testing.
    ENROLL_DAILY_MAX = int(os.getenv("ENROLL_DAILY_MAX", "5"))
    SECRET_KEY = _require_env("FLASK_SECRET_KEY", "dev-fallback-key-not-for-production")
    # Flask answers 413 before a route reads a larger body. The largest real
    # payload is check-in: 7 frames, each accepted only up to 500 KB
    # (server_validate_frame), sent as base64 (about 4.7 MB in total).
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

    # Embedding integrity HMAC salt (Sprint 2B)
    EMBEDDING_INTEGRITY_SALT = _require_env(
        "EMBEDDING_INTEGRITY_SALT",
        "dev-integrity-salt-not-for-production",
    )

    # Redis URL for rate limiter (production)
    REDIS_URL = os.getenv("REDIS_URL", "")

    # Reverse proxies in front of Flask (cloudflared = 1). 0 trusts no
    # X-Forwarded-* header: the client address is the TCP peer.
    TRUSTED_PROXY_HOPS = int(os.getenv("TRUSTED_PROXY_HOPS", "0"))

    # Check-in proximity method: "ble" (default: signed nonce challenge against the
    # room's ESP32 over Web Bluetooth), "totp" (6-digit code from a room screen) or
    # "broadcast" (signed rolling code the room board advertises; read by the app).
    CHECKIN_PROXIMITY_METHOD = os.getenv("CHECKIN_PROXIMITY_METHOD", "ble").lower()
    if CHECKIN_PROXIMITY_METHOD not in ("totp", "ble", "broadcast"):
        _log.warning(
            f"[SmartCheck] CHECKIN_PROXIMITY_METHOD={CHECKIN_PROXIMITY_METHOD!r} is invalid "
            f"(expected 'totp', 'ble' or 'broadcast') — falling back to 'ble'."
        )
        CHECKIN_PROXIMITY_METHOD = "ble"

    # Check-in liveness: "head_turn" (server-verified random turn, default) or
    # "passive" (legacy hands-free capture; anti-spoof only). Rollback switch only.
    CHECKIN_LIVENESS = os.getenv("CHECKIN_LIVENESS", "head_turn").lower()
    if CHECKIN_LIVENESS not in ("head_turn", "passive"):
        _log.warning(
            f"[SmartCheck] CHECKIN_LIVENESS={CHECKIN_LIVENESS!r} is invalid "
            f"(expected 'head_turn' or 'passive') — falling back to 'head_turn'."
        )
        CHECKIN_LIVENESS = "head_turn"

    # Face stack: "arcface" (InsightFace buffalo_l), the only engine since face
    # stack v2 Phase 5. FACE_CLAHE and the three thresholds below must be set
    # (values come from scripts/compare_models.py); the app refuses to start
    # without them (face_engine.validate_config). They have no defaults on purpose.
    FACE_ENGINE = os.getenv("FACE_ENGINE", "arcface").strip().lower()
    FACE_CLAHE = os.getenv("FACE_CLAHE")
    ARCFACE_MATCH_THRESHOLD = os.getenv("ARCFACE_MATCH_THRESHOLD")
    ARCFACE_DUP_THRESHOLD = os.getenv("ARCFACE_DUP_THRESHOLD")
    ENROLL_SELF_SIM_MIN = os.getenv("ENROLL_SELF_SIM_MIN")
    # Room code on the teacher's screen (app/services/session_code.py): fallback for
    # phones without BLE or LINE (e.g. iPhone Safari). Off by default: a code read
    # off the screen can be forwarded within its minute, weaker than BLE/LINE.
    CHECKIN_SESSION_CODE = os.getenv("CHECKIN_SESSION_CODE", "0") == "1"
    # LINE Beacon proximity (optional): the LINE app detects the room board and LINE
    # posts a webhook to /line/webhook. Needs migration 20261020_line_beacon.sql.
    LINE_BEACON_ENABLED = os.getenv("LINE_BEACON_ENABLED", "0") == "1"
    LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
    LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
    PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "")   # fixed https URL students open from LINE
    # Load and run the face models once while the app starts (1) or on first use (0).
    FACE_ENGINE_WARMUP = os.getenv("FACE_ENGINE_WARMUP", "1") == "1"

    # ── Flask-Session: server-side sessions ──
    # "file" (default): files in instance/sessions on this server. The database store
    # cost 1.36 s on every request (6 round trips to Supabase in ap-south-1, measured
    # 2026-11-02). Files work for any number of workers on one machine; several
    # machines need a shared store. "database": the flask_sessions table, as before.
    SESSION_STORE              = os.getenv("SESSION_STORE", "file")
    if SESSION_STORE == "database":
        SESSION_TYPE           = "sqlalchemy"
    else:
        from cachelib.file import FileSystemCache
        SESSION_TYPE           = "cachelib"
        # A threshold this high is never reached by valid sessions (1 hour each);
        # past it cachelib drops expired files first.
        SESSION_CACHELIB       = FileSystemCache(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "instance", "sessions"),
            threshold=20000)
    SESSION_SQLALCHEMY_TABLE   = "flask_sessions"
    SESSION_PERMANENT          = True
    PERMANENT_SESSION_LIFETIME = 3600          # 1 hour: teachers, and students in a browser
    # Longer logins (app/services/session_lifetime.py), counted from the last use.
    # Students in the Android app: about one term. Admin: "permanent" (10 years).
    STUDENT_APP_SESSION_DAYS   = int(os.getenv("STUDENT_APP_SESSION_DAYS", "120"))
    ADMIN_SESSION_DAYS         = int(os.getenv("ADMIN_SESSION_DAYS", "3650"))
    SESSION_COOKIE_HTTPONLY    = True           # JS cannot read the session cookie
    SESSION_COOKIE_SAMESITE    = "Strict"       # CSRF layer 1
    SESSION_COOKIE_SECURE      = _IS_PRODUCTION  # True in production (TLS required)

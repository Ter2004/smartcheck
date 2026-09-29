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

    # Enrollment flow variant: "classic" | "circular"
    ENROLL_FLOW_MODE = os.getenv("ENROLL_FLOW_MODE", "classic")

    # Check-in proximity method: "totp" (6-digit code from a room screen, default)
    # or "ble" (Web Bluetooth GATT connect+read against the room's beacon).
    CHECKIN_PROXIMITY_METHOD = os.getenv("CHECKIN_PROXIMITY_METHOD", "totp").lower()
    if CHECKIN_PROXIMITY_METHOD not in ("totp", "ble"):
        _log.warning(
            f"[SmartCheck] CHECKIN_PROXIMITY_METHOD={CHECKIN_PROXIMITY_METHOD!r} is invalid "
            f"(expected 'totp' or 'ble') — falling back to 'totp'."
        )
        CHECKIN_PROXIMITY_METHOD = "totp"

    # Check-in liveness: "head_turn" (server-verified random turn, default) or
    # "passive" (legacy hands-free capture; anti-spoof only). Rollback switch only.
    CHECKIN_LIVENESS = os.getenv("CHECKIN_LIVENESS", "head_turn").lower()
    if CHECKIN_LIVENESS not in ("head_turn", "passive"):
        _log.warning(
            f"[SmartCheck] CHECKIN_LIVENESS={CHECKIN_LIVENESS!r} is invalid "
            f"(expected 'head_turn' or 'passive') — falling back to 'head_turn'."
        )
        CHECKIN_LIVENESS = "head_turn"

    # ── Flask-Session: server-side SQLAlchemy sessions (Railway deployment) ──
    SESSION_TYPE               = "sqlalchemy"
    SESSION_SQLALCHEMY_TABLE   = "flask_sessions"
    SESSION_PERMANENT          = True
    PERMANENT_SESSION_LIFETIME = 3600          # 1 hour
    SESSION_COOKIE_HTTPONLY    = True           # JS cannot read the session cookie
    SESSION_COOKIE_SAMESITE    = "Strict"       # CSRF layer 1
    SESSION_COOKIE_SECURE      = _IS_PRODUCTION  # True in production (TLS required)

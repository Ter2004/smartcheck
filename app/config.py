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
    SECRET_KEY = _require_env("FLASK_SECRET_KEY", "dev-fallback-key-not-for-production")
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

    # ── Flask-Session: server-side SQLAlchemy sessions (Railway deployment) ──
    SESSION_TYPE               = "sqlalchemy"
    SESSION_SQLALCHEMY_TABLE   = "flask_sessions"
    SESSION_PERMANENT          = True
    PERMANENT_SESSION_LIFETIME = 3600          # 1 hour
    SESSION_COOKIE_HTTPONLY    = True           # JS cannot read the session cookie
    SESSION_COOKIE_SAMESITE    = "Strict"       # CSRF layer 1
    SESSION_COOKIE_SECURE      = _IS_PRODUCTION  # True in production (TLS required)

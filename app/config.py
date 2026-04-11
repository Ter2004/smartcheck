import os
import tempfile
from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-fallback-key")
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

    # Embedding integrity HMAC salt (Sprint 2B)
    EMBEDDING_INTEGRITY_SALT = os.getenv(
        "EMBEDDING_INTEGRITY_SALT",
        "smartcheck-integrity-salt-change-in-production",
    )

    # ── Flask-Session: server-side filesystem sessions (Sprint 1A) ──
    SESSION_TYPE             = "filesystem"
    SESSION_FILE_DIR         = os.path.join(tempfile.gettempdir(), "smartcheck_sessions")
    SESSION_PERMANENT        = True
    PERMANENT_SESSION_LIFETIME = 3600          # 1 hour
    SESSION_COOKIE_HTTPONLY  = True             # JS cannot read the session cookie
    SESSION_COOKIE_SAMESITE  = "Strict"         # CSRF layer 1
    SESSION_COOKIE_SECURE    = False            # Set True when TLS is enabled
    SESSION_FILE_THRESHOLD   = 500             # max cached session files on disk

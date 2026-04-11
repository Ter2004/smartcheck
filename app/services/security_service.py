"""
security_service.py — SmartCheck security utilities

Functions:
  create_device_token()         — HMAC-signed device token (Sprint 1B)
  verify_device_token()         — verify + decode device token
  compute_embedding_integrity_hash() — SHA-256 over embeddings (Sprint 2B)
  verify_embedding_integrity()  — tamper detection for stored embeddings
  csrf_protect()                — decorator for JSON API endpoints (Sprint 1C)
  log_audit_event()             — insert row into audit_logs (Sprint 3B)
"""

import hmac
import hashlib
import time
import json
import base64
import secrets
from functools import wraps
from flask import request, session, jsonify


# ─── Device Token ─────────────────────────────────────────────────────────────

def create_device_token(user_id: str, device_fingerprint: str, secret_key: str) -> str:
    """
    HMAC-SHA256 device token binding user_id + device_fingerprint.
    Payload is base64url-encoded JSON; signature follows after '.'.
    """
    payload = {
        "uid": user_id,
        "did": device_fingerprint,
        "iat": int(time.time()),
    }
    payload_b64 = (
        base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode())
        .decode()
        .rstrip("=")
    )
    sig = hmac.new(
        secret_key.encode(),
        payload_b64.encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"{payload_b64}.{sig}"


def verify_device_token(
    token: str,
    secret_key: str,
    max_age_days: int = 120,
) -> dict | None:
    """
    Verify HMAC signature and expiry.
    Returns the payload dict on success, None on any failure.
    """
    if not token:
        return None
    try:
        payload_b64, sig = token.rsplit(".", 1)
    except ValueError:
        return None

    expected_sig = hmac.new(
        secret_key.encode(),
        payload_b64.encode(),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(sig, expected_sig):
        return None  # tampered

    # Restore stripped base64 padding
    pad = 4 - len(payload_b64) % 4
    if pad != 4:
        payload_b64 += "=" * pad

    try:
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
    except Exception:
        return None

    if time.time() - payload.get("iat", 0) > max_age_days * 86400:
        return None  # expired

    return payload


# ─── Embedding Integrity Hash ─────────────────────────────────────────────────

def compute_embedding_integrity_hash(
    user_id: str,
    embeddings: list,
    integrity_salt: str,
) -> str:
    """
    HMAC-SHA256 over the user's face embeddings.
    Binds user_id (prevents swapping between users) and uses a server-side
    salt (prevents offline pre-computation).
    Embeddings are rounded to 6 d.p. and sorted for order-independence.
    """
    normalized = sorted([
        [round(float(v), 6) for v in emb]
        for emb in embeddings
    ])
    payload = json.dumps(
        {"uid": user_id, "emb": normalized},
        separators=(",", ":"),
    )
    return hmac.new(
        integrity_salt.encode(),
        payload.encode(),
        hashlib.sha256,
    ).hexdigest()


def verify_embedding_integrity(
    user_id: str,
    embeddings: list,
    stored_hash: str,
    integrity_salt: str,
) -> bool:
    """
    Constant-time comparison — returns False if embeddings have been tampered with.
    """
    if not stored_hash:
        return False  # No hash = unverifiable — reject and require re-enrollment
    expected = compute_embedding_integrity_hash(user_id, embeddings, integrity_salt)
    return hmac.compare_digest(expected, stored_hash)


# ─── CSRF Protection Decorator ────────────────────────────────────────────────

def csrf_protect(f):
    """
    CSRF protection for JSON API endpoints.

    Requires the client to send the session CSRF token as:
        X-CSRF-Token: <token>

    The token is injected into base.html as:
        <meta name="csrf-token" content="{{ session.get('csrf_token', '') }}">

    JavaScript reads it with:
        document.querySelector('meta[name="csrf-token"]').content
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        token    = request.headers.get("X-CSRF-Token", "")
        expected = session.get("csrf_token", "")
        if not expected or not hmac.compare_digest(token, expected):
            return jsonify({"ok": False, "error": "CSRF validation failed"}), 403
        return f(*args, **kwargs)
    return decorated


def csrf_protect_form(f):
    """
    CSRF protection for HTML form endpoints (admin/teacher pages).
    Only validates on state-changing methods (POST/PUT/DELETE/PATCH).
    Requires a hidden <input name="csrf_token"> in every form.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            token    = request.form.get("csrf_token", "")
            expected = session.get("csrf_token", "")
            if not expected or not hmac.compare_digest(token, expected):
                from flask import abort
                abort(403)
        return f(*args, **kwargs)
    return decorated


# ─── Audit Log Helper ─────────────────────────────────────────────────────────

def log_audit_event(
    supabase_client,
    actor_id: str,
    actor_role: str,
    event_type: str,
    target_id: str = None,
    session_id: str = None,
    old_value: str = None,
    new_value: str = None,
    metadata: dict = None,
) -> None:
    """
    Insert one row into audit_logs (non-fatal — errors are logged but not raised).
    Uses supabase_admin (service key) so it bypasses RLS.
    """
    try:
        entry = {
            "actor_id":   actor_id,
            "actor_role": actor_role,
            "event_type": event_type,
            "target_id":  target_id,
            "session_id": session_id,
            "old_value":  old_value,
            "new_value":  new_value,
            "metadata":   metadata or {},
            "ip_address": request.remote_addr,
            "user_agent": (request.headers.get("User-Agent", "") or "")[:500],
        }
        supabase_client.table("audit_logs").insert(entry).execute()
    except Exception as e:
        import logging
        logging.getLogger("smartcheck.audit").warning(
            f"audit_log insert failed: {e}"
        )

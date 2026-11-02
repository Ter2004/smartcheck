"""The Android app registers its Firebase token here (app/static/js/push_notifications.js)."""
import logging
import re
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request, session

from app import supabase_admin
from app.routes.auth import login_required, role_required
from app.services.security_service import csrf_protect

push_bp = Blueprint("push", __name__)
_log = logging.getLogger("smartcheck.push")

TOKEN_RE = re.compile(r"^[A-Za-z0-9_:\-]{20,4096}$")
SESSION_KEY = "push_token"   # removed from push_devices on logout (auth.logout)


@push_bp.route("/api/push/register", methods=["POST"])
@login_required
@role_required("student")
@csrf_protect
def register():
    token = (request.get_json(silent=True) or {}).get("token")
    if not isinstance(token, str) or not TOKEN_RE.match(token):
        return jsonify(ok=False, error="token ไม่ถูกต้อง"), 400
    try:
        # One row per phone: logging in as someone else on it moves the row.
        supabase_admin.table("push_devices").upsert({
            "token": token, "user_id": session["user_id"], "platform": "android",
            "last_seen_at": datetime.now(timezone.utc).isoformat(),
        }, on_conflict="token").execute()
    except Exception as e:
        _log.warning("[PUSH] register failed: %s", type(e).__name__)
        return jsonify(ok=False, error="บันทึกการแจ้งเตือนไม่สำเร็จ"), 503
    session[SESSION_KEY] = token
    return jsonify(ok=True)


def forget_this_device():
    """On logout: this phone stops getting the user's notifications."""
    token = session.get(SESSION_KEY)
    if not token or "user_id" not in session:
        return
    try:
        (supabase_admin.table("push_devices").delete()
         .eq("token", token).eq("user_id", session["user_id"]).execute())
    except Exception as e:
        _log.warning("[PUSH] unregister failed: %s", type(e).__name__)

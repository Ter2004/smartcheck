"""LINE Beacon proximity (LINE_BEACON_ENABLED=1): webhook and account linking.

POST /line/webhook       LINE's server → us. Authenticated only by X-Line-Signature.
POST /line/link-code     a logged-in student gets a 6-digit code to send to the LINE account.

On a beacon "enter" event for a linked student in the room of an open session
of their course, the student gets a LINE message with one button that opens the
check-in page carrying a proximity receipt; the face scan then runs as usual.
"""
import logging
import time
from urllib.parse import quote

from flask import Blueprint, abort, current_app, jsonify, request, session

from app import supabase_admin
from app import limiter as _limiter
from app.routes.auth import login_required, role_required
from app.services import line_beacon, proximity_receipt
from app.services.security_service import csrf_protect

line_bp = Blueprint("line", __name__)
_log = logging.getLogger("smartcheck.line")

NOT_LINKED = ("ยังไม่ได้เชื่อมบัญชี LINE กับ SmartCheck — เข้าเว็บ SmartCheck หน้าหลัก "
              "กด \"เชื่อมบัญชี LINE\" แล้วส่งรหัสที่ได้มาที่แชทนี้")
HOW_TO_LINK = "พิมพ์ LINK ตามด้วยรหัส 6 หลักจากหน้าเว็บ SmartCheck เช่น LINK 123456"


def _enabled():
    return bool(current_app.config.get("LINE_BEACON_ENABLED"))


def _reply(event, messages):
    token = event.get("replyToken")
    if not token:
        return
    try:
        line_beacon.reply(current_app.config["LINE_CHANNEL_ACCESS_TOKEN"], token, messages)
    except Exception as error:          # never fail the webhook because LINE's reply API is down
        _log.warning("[LINE] reply failed: %s", type(error).__name__)


@line_bp.route("/line/webhook", methods=["POST"])
@_limiter.exempt
def webhook():
    if not _enabled():
        abort(404)
    body = request.get_data()
    if not line_beacon.valid_signature(current_app.config["LINE_CHANNEL_SECRET"], body,
                                       request.headers.get("X-Line-Signature", "")):
        _log.warning("[LINE] webhook signature mismatch")
        abort(400)
    received_ms = int(time.time() * 1000)
    try:
        events = (request.get_json(silent=True) or {}).get("events") or []
    except Exception:
        events = []
    for event in events:
        try:
            kind = event.get("type")
            if kind == "beacon":
                _on_beacon(event, received_ms)
            elif kind == "message" and (event.get("message") or {}).get("type") == "text":
                _on_text(event)
            elif kind == "follow":
                _reply(event, [line_beacon.text("ยินดีต้อนรับสู่ SmartCheck\n" + HOW_TO_LINK)])
        except Exception as error:     # one bad event must not drop the others
            _log.error("[LINE] event %s failed: %s", event.get("type"), type(error).__name__)
    return "OK"


def _on_text(event):
    line_user = (event.get("source") or {}).get("userId")
    text = (event.get("message") or {}).get("text", "")
    if not line_user or "LINK" not in text.upper():
        _reply(event, [line_beacon.text(HOW_TO_LINK)])
        return
    result = line_beacon.redeem_link_code(supabase_admin, line_user, text)
    _log.info("[LINE] link result=%s", result)
    _reply(event, [line_beacon.text({
        "linked": "เชื่อมบัญชีเรียบร้อย เมื่อเข้าห้องเรียน ระบบจะส่งปุ่มเช็คชื่อมาที่แชทนี้",
        "taken": "บัญชี LINE นี้เชื่อมกับนักศึกษาคนอื่นแล้ว กรุณาติดต่อผู้ดูแลระบบ",
    }.get(result, "รหัสไม่ถูกต้องหรือหมดอายุ — ขอรหัสใหม่จากหน้าเว็บ SmartCheck"))])


def _on_beacon(event, received_ms):
    beacon_info = event.get("beacon") or {}
    line_user = (event.get("source") or {}).get("userId")
    latency = received_ms - int(event.get("timestamp") or received_ms)
    _log.info("[LINE] beacon type=%s hwid=%s dm_present=%s latency_ms=%s",
              beacon_info.get("type"), beacon_info.get("hwid"),
              bool(beacon_info.get("dm") or beacon_info.get("deviceMessage")), latency)
    if beacon_info.get("type") != "enter" or not line_user:
        return

    user = (supabase_admin.table("users").select("id, role").eq("line_user_id", line_user)
            .maybe_single().execute())
    student = (user.data if user else None) or {}
    if student.get("role") != "student":
        _reply(event, [line_beacon.text(NOT_LINKED)])
        return

    hwid = str(beacon_info.get("hwid") or "").lower()
    found = (supabase_admin.table("beacons").select("id, room_name, ble_room_code, ble_secret")
             .eq("line_hwid", hwid).maybe_single().execute())
    room = (found.data if found else None) or {}
    if not room:
        _log.warning("[LINE] unknown hwid=%s", hwid)
        return

    # "dm" per the LINE SDKs; the official example omits it, so accept the long name too.
    dm = beacon_info.get("dm") or beacon_info.get("deviceMessage")
    counter = line_beacon.verify_dm(room.get("ble_secret"), dm)
    if counter is None:
        _log.warning("[LINE] reject=dm_invalid beacon=%s student=%s", room["id"], student["id"])
        _reply(event, [line_beacon.text("ยืนยันห้องเรียนไม่สำเร็จ กรุณาแจ้งอาจารย์")])
        return
    if not line_beacon.fresh(supabase_admin, room["id"], counter):
        _log.warning("[LINE] reject=dm_replayed beacon=%s counter=%s student=%s", room["id"], counter, student["id"])
        return

    sessions = (supabase_admin.table("sessions").select("id, course_id")
                .eq("beacon_id", room["id"]).eq("is_open", True).execute().data or [])
    courses = {r["course_id"] for r in (supabase_admin.table("course_enrollments").select("course_id")
                                        .eq("student_id", student["id"]).execute().data or [])}
    session_row = next((s for s in sessions if s["course_id"] in courses), None)
    if not session_row:
        _reply(event, [line_beacon.text(f"ห้อง {room.get('room_name', '')}: ตอนนี้ไม่มีคาบของคุณที่เปิดเช็คชื่อ")])
        return

    room_code = room.get("ble_room_code") or ""
    receipt = proximity_receipt.issue(current_app.config["PROXIMITY_RECEIPT_SECRET"], student["id"],
                                      session_row["id"], "line", room_code)
    base = current_app.config["PUBLIC_BASE_URL"].rstrip("/")
    url = (f"{base}/student/checkin?session_id={quote(session_row['id'])}&via=line"
           f"&room={quote(room_code)}&r={quote(receipt)}&openExternalBrowser=1")
    _log.info("[LINE] receipt issued student=%s session=%s latency_ms=%s", student["id"], session_row["id"], latency)
    _reply(event, [line_beacon.checkin_button(room.get("room_name", ""), url)])


@line_bp.route("/line/link-code", methods=["POST"])
@login_required
@role_required("student")
@_limiter.limit("5 per minute")
@csrf_protect
def link_code():
    if not _enabled():
        abort(404)
    code = line_beacon.new_link_code(supabase_admin, session["user_id"])
    return jsonify(ok=True, code=code, expires_in_min=line_beacon.LINK_CODE_TTL_MIN,
                   instruction=f"เพิ่มเพื่อนบัญชี LINE ของ SmartCheck แล้วพิมพ์ LINK {code}")

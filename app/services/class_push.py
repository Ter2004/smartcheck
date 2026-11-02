"""Class-open notifications to the Android app (Firebase, app/services/fcm.py).

Two notifications per class, only to enrolled students who have not checked in:
  1. when the class opens (teacher press or automatic open), and
  2. REMINDER_BEFORE the on-time window ends ("มาเรียนได้แล้ว").
Both carry the same Android tag (the session id), so the reminder replaces the
first one in the notification bar instead of stacking. The app removes it once
the student has checked in.

Each notification is claimed with a conditional update (sessions.open_push_at /
reminder_push_at IS NULL), so a teacher press, the scheduler tick and a second
server process can never send it twice. Off unless SESSION_OPEN_MODE=teacher and
FCM_SERVICE_ACCOUNT_FILE are set.
"""
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from dateutil import parser as dtparser

from app.services import class_open, fcm

_log = logging.getLogger("smartcheck.class_push")

CHANNEL_ID = "checkin"                    # created by the app (app/static/js/push_notifications.js)
REMINDER_BEFORE = timedelta(minutes=1)
OPEN_CATCH_UP = timedelta(minutes=10)     # the tick still sends a missed open notification this late
OPEN_TTL_S = 30 * 60                      # an offline phone gets it within this, else never
TH = timezone(timedelta(hours=7))

_send_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="fcm-send")


def enabled(config):
    return class_open.teacher_mode(config) and fcm.sender(config) is not None


def _db():
    from app import supabase_admin
    return supabase_admin


def _time(value):
    return value if isinstance(value, datetime) else dtparser.parse(value)


def _claim(session_id, column, now):
    """The session row if this call won the right to send `column`'s notification, else None."""
    rows = (_db().table("sessions").update({column: now.isoformat()})
            .eq("id", session_id).eq("is_open", True).is_(column, "null")
            .execute().data or [])
    return rows[0] if rows else None


def _course_code(sess):
    res = _db().table("courses").select("code").eq("id", sess["course_id"]).maybe_single().execute()
    return ((res.data if res else None) or {}).get("code") or "คาบเรียน"


def _recipient_tokens(sess):
    db = _db()
    students = {r["student_id"] for r in (db.table("course_enrollments").select("student_id")
                                          .eq("course_id", sess["course_id"]).execute().data or [])}
    checked = {r["student_id"] for r in (db.table("attendance").select("student_id")
                                         .eq("session_id", sess["id"]).execute().data or [])}
    waiting = sorted(students - checked)
    if not waiting:
        return []
    rows = db.table("push_devices").select("token").in_("user_id", waiting).execute().data or []
    return [r["token"] for r in rows]


def message(sess, title, body, ttl_s):
    """FCM v1 message (without the device token)."""
    return {
        "notification": {"title": title, "body": body},
        "data": {"session_id": sess["id"], "url": f"/student/checkin?session_id={sess['id']}"},
        "android": {"priority": "HIGH", "ttl": f"{int(ttl_s)}s",
                    "notification": {"channel_id": CHANNEL_ID, "tag": sess["id"]}},
    }


def _send(config, tokens, msg):
    sender = fcm.sender(config)
    if not sender or not tokens:
        return 0
    results = list(_send_pool.map(lambda t: sender.send(t, msg), tokens))
    invalid = [t for t, r in zip(tokens, results) if r == fcm.INVALID_TOKEN]
    if invalid:
        try:
            _db().table("push_devices").delete().in_("token", invalid).execute()
        except Exception:
            _log.exception("[PUSH] could not delete %d invalid tokens", len(invalid))
    sent = results.count(fcm.SENT)
    _log.info("[PUSH] sent=%d invalid=%d failed=%d", sent, len(invalid), results.count(fcm.FAILED))
    return sent


def notify_opened(session_id, config, now=None):
    """Send "class open" once per session. Returns the number of phones reached."""
    if not enabled(config):
        return 0
    now = now or datetime.now(timezone.utc)
    sess = _claim(session_id, "open_push_at", now)
    if not sess:
        return 0
    until = class_open.on_time_until(sess, config)
    body = (f"แตะเพื่อเช็คชื่อ — มาเรียนทันเวลาถึง {until.astimezone(TH):%H:%M} น."
            if until else "แตะเพื่อเช็คชื่อ")
    msg = message(sess, f"{_course_code(sess)} เปิดเช็คชื่อแล้ว", body, OPEN_TTL_S)
    return _send(config, _recipient_tokens(sess), msg)


def notify_reminder(session_id, config, now=None):
    """Send "1 minute left" once per session, only while it is still on time."""
    if not enabled(config):
        return 0
    now = now or datetime.now(timezone.utc)
    sess = _claim(session_id, "reminder_push_at", now)
    if not sess:
        return 0
    until = class_open.on_time_until(sess, config)
    if until is None or now >= until:
        return 0
    msg = message(sess, f"{_course_code(sess)} ยังไม่ได้เช็คชื่อ",
                  f"อีก {max(1, round((until - now).total_seconds() / 60))} นาทีจะนับสาย — มาเรียนได้แล้ว แตะเพื่อเช็คชื่อ",
                  max(1, (until - now).total_seconds()))
    return _send(config, _recipient_tokens(sess), msg)


def _safely(fn, *args):
    try:
        fn(*args)
    except Exception:
        _log.exception("[PUSH] %s failed", fn.__name__)


def class_opened(session_id, opened_at, present_minutes, config):
    """After a class opens: send "open" now (background) and "1 minute left" on time."""
    if not enabled(config):
        return
    config = dict(config)   # read from other threads after the request ends
    threading.Thread(target=_safely, args=(notify_opened, session_id, config), daemon=True).start()
    if int(present_minutes) <= 1:
        return
    due = _time(opened_at) + timedelta(minutes=int(present_minutes)) - REMINDER_BEFORE
    delay = max(0.0, (due - datetime.now(timezone.utc)).total_seconds())
    timer = threading.Timer(delay, _safely, args=(notify_reminder, session_id, config))
    timer.daemon = True
    timer.start()


def catch_up(config, now):
    """Scheduler tick: send what a restart or a dead thread missed (claims stop repeats)."""
    if not enabled(config):
        return
    rows = (_db().table("sessions").select("id, opened_at, present_minutes, open_push_at, reminder_push_at")
            .eq("is_open", True).not_.is_("opened_at", "null").execute().data or [])
    for sess in rows:
        opened_at = _time(sess["opened_at"])
        if not sess.get("open_push_at") and now - opened_at < OPEN_CATCH_UP:
            _safely(notify_opened, sess["id"], config, now)
        until = class_open.on_time_until(sess, config)
        minutes = sess.get("present_minutes") or config["CLASS_ON_TIME_MINUTES"]
        if (not sess.get("reminder_push_at") and int(minutes) > 1
                and until - REMINDER_BEFORE <= now < until):
            _safely(notify_reminder, sess["id"], config, now)

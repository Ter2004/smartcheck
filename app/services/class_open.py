"""When a class session is open, and whether a check-in counts as on time.

SESSION_OPEN_MODE=teacher (default, migration 20261103_teacher_open_and_push.sql):
  - The teacher presses "เปิดคาบ" and picks the on-time window (default 5 min).
    Pressing works from CLASS_EARLY_OPEN_MINUTES before the scheduled start.
  - If nobody presses it, the class opens by itself CLASS_AUTO_OPEN_AFTER_MINUTES
    after the scheduled start, with the default window.
  - It closes at the scheduled end either way.
  - A check-in inside the window from opened_at is present, after it late.
SESSION_OPEN_MODE=schedule keeps the old rule: open for the whole scheduled time,
late more than 15 minutes after the start. It needs no new columns.
"""
from datetime import datetime, timedelta, timezone

from dateutil import parser as dtparser

LEGACY_LATE_AFTER = timedelta(minutes=15)
# A session whose start is further back than this cannot be opened (an old
# day's row the scheduler has not closed yet).
OPENABLE_FOR = timedelta(hours=12)


def teacher_mode(config):
    return config.get("SESSION_OPEN_MODE", "teacher") == "teacher"


def _time(value):
    if isinstance(value, datetime):
        return value
    parsed = dtparser.parse(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def scheduled_state(sess, start, end, now, config):
    """Fields a scheduled session should have at `now`.

    sess is the row ({} before it exists); start/end are the schedule's times
    today (aware). Returns (fields, auto_opened). An auto-open also sets
    opened_at and present_minutes.
    """
    end_iso = end.astimezone(timezone.utc).isoformat()
    if now >= end:
        return {"is_open": False, "end_time": end_iso}, False
    if not teacher_mode(config):
        return {"is_open": now >= start, "end_time": None}, False
    # Opened by the teacher, or open from before this mode (no opened_at).
    if sess.get("opened_at") or sess.get("is_open"):
        return {"is_open": True, "end_time": None}, False
    if now >= start + timedelta(minutes=int(config["CLASS_AUTO_OPEN_AFTER_MINUTES"])):
        return {"is_open": True, "end_time": None, "opened_at": now.astimezone(timezone.utc).isoformat(),
                "present_minutes": int(config["CLASS_ON_TIME_MINUTES"])}, True
    return {"is_open": False, "end_time": None}, False


def on_time_until(sess, config):
    """Last on-time moment of an opened session, or None when it was not opened by this mode."""
    if not sess.get("opened_at"):
        return None
    minutes = sess.get("present_minutes") or int(config["CLASS_ON_TIME_MINUTES"])
    return _time(sess["opened_at"]) + timedelta(minutes=int(minutes))


def checkin_status(sess, now, config):
    """'present' or 'late' for a check-in at `now`."""
    deadline = on_time_until(sess, config)
    if deadline is None:
        if not sess.get("start_time"):
            return "present"
        deadline = _time(sess["start_time"]) + LEGACY_LATE_AFTER
    return "late" if now > deadline else "present"


def opens_from(sess, config):
    """Earliest moment the teacher may open this session."""
    return _time(sess["start_time"]) - timedelta(minutes=int(config["CLASS_EARLY_OPEN_MINUTES"]))


def open_refusal(sess, now, config):
    """Why the teacher cannot open this session now (Thai), or None."""
    if sess.get("is_open"):
        return "คาบนี้เปิดอยู่แล้ว"
    if sess.get("end_time"):
        return "คาบนี้จบไปแล้ว"
    if now < opens_from(sess, config):
        minutes = int(config["CLASS_EARLY_OPEN_MINUTES"])
        return f"เปิดคาบได้ก่อนเวลาเริ่มไม่เกิน {minutes} นาที"
    if now - _time(sess["start_time"]) > OPENABLE_FOR:
        return "คาบนี้เลยเวลาไปแล้ว"
    return None


def parse_minutes(text, config):
    """The on-time window the teacher typed, as minutes in 1..CLASS_ON_TIME_MAX_MINUTES, or None."""
    try:
        minutes = int(str(text).strip())
    except (TypeError, ValueError):
        return None
    return minutes if 1 <= minutes <= int(config["CLASS_ON_TIME_MAX_MINUTES"]) else None

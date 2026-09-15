"""
scheduler.py — Auto create/close sessions ตาม schedules table
รันทุก 1 นาที:
  - สร้าง session อัตโนมัติสำหรับทุก schedule ที่ตรงกับวันนี้ (ถ้ายังไม่มี)
  - ปิด session ถ้าเลยเวลาจบแล้ว
"""
import logging
import os
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from apscheduler.schedulers.background import BackgroundScheduler

_log = logging.getLogger("smartcheck.scheduler")

TZ_THAI = ZoneInfo("Asia/Bangkok")

DAY_NAMES = ["จันทร์", "อังคาร", "พุธ", "พฤหัส", "ศุกร์", "เสาร์", "อาทิตย์"]


def _get_supabase():
    from app import supabase_admin
    return supabase_admin


def auto_manage_sessions():
    from app.services.session_policy import occurrence, state
    from app.config import Config
    try:
        sb = _get_supabase()
        now = datetime.now(timezone.utc)
        today = now.astimezone(TZ_THAI).date()
        schedules = sb.table("schedules").select("*, courses(id, code, name, is_active, is_test_course)").eq("is_active", True).execute().data or []
        for sch in schedules:
            course = sch.get("courses") or {}
            if not course.get("is_active") or (course.get("is_test_course") and not Config.ALLOW_TEST_ACCOUNTS):
                continue
            if not sch.get("beacon_id"):
                _log.warning("Schedule %s has no room; skipped", sch["id"])
                continue
            # Yesterday covers overnight classes; next seven days populate the calendar.
            for offset in range(-1, 8):
                day = today + timedelta(days=offset)
                if day.weekday() != sch["day_of_week"]:
                    continue
                row = occurrence(sch, day)
                row["title"] = f"{course['code']} {day.isoformat()} ({sch['start_time'][:5]}-{sch['end_time'][:5]})"
                row["is_open"] = state(row, now) == "open"
                # Ignore duplicates; never reopen or overwrite a cancelled occurrence.
                sb.table("sessions").upsert(row, on_conflict="course_id,start_time", ignore_duplicates=True).execute()
        # is_open is only a display cache. APIs always evaluate timestamps themselves.
        rows = sb.table("sessions").select("*").gte("end_time", (now-timedelta(days=1)).isoformat()).execute().data or []
        for row in rows:
            desired = state(row, now) == "open"
            if row.get("is_open") != desired:
                sb.table("sessions").update({"is_open": desired}).eq("id", row["id"]).execute()
    except Exception:
        _log.exception("Session scheduler failed")


def _parse_time(time_str: str):
    """แปลง 'HH:MM:SS' หรือ 'HH:MM' เป็น time object"""
    from datetime import time
    parts = time_str.split(":")
    return time(int(parts[0]), int(parts[1]))


def keep_alive():
    """Ping Supabase ทุก 3 นาที เพื่อป้องกัน HTTP/2 idle connection timeout"""
    try:
        _get_supabase().table("beacons").select("id").limit(1).execute()
    except Exception:
        from app import _refresh_clients
        _refresh_clients()
        _log.info("[KEEP-ALIVE] Reconnected to Supabase")


def start_scheduler(app):
    scheduler = BackgroundScheduler(timezone="Asia/Bangkok")
    scheduler.add_job(auto_manage_sessions, "interval", minutes=1, id="session_manager")
    scheduler.add_job(keep_alive, "interval", minutes=3, id="keep_alive")
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
        if not scheduler.running:
            scheduler.start()
            _log.info("[SCHEDULER] Started — checking every minute")
    return scheduler

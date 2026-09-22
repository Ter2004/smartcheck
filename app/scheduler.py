"""
scheduler.py — Auto create/close sessions ตาม schedules table
รันทุก 1 นาที:
  - สร้าง session อัตโนมัติสำหรับทุก schedule ที่ตรงกับวันนี้ (ถ้ายังไม่มี)
  - ปิด session ถ้าเลยเวลาจบแล้ว
"""
import logging
from threading import Lock
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
    try:
        sb        = _get_supabase()
        now       = datetime.now(timezone.utc)
        local_now = now.astimezone(TZ_THAI)
        today_dow  = local_now.weekday()
        today_date = local_now.date().isoformat()
        now_time   = local_now.time()

        # ─── ดึง schedules ที่ตรงกับวันนี้ ─────────────────────────────
        schedules = (
            sb.table("schedules")
            .select("*, courses(id, code, name, teacher_id, is_active)")
            .eq("day_of_week", today_dow)
            .execute()
            .data or []
        )

        # ดึง beacon แรกที่ active ไว้เป็น default
        beacons = sb.table("beacons").select("id").eq("is_active", True).limit(1).execute().data or []
        default_beacon_id = beacons[0]["id"] if beacons else None

        for sch in schedules:
            course = sch.get("courses") or {}
            if not course.get("is_active"):
                continue

            course_id  = course["id"]
            sch_start  = sch["start_time"][:5]   # "HH:MM"
            sch_end    = sch["end_time"][:5]
            start_time = _parse_time(sch_start)
            end_time   = _parse_time(sch_end)

            # ─── Auto-create: สร้าง session ถ้ายังไม่มีของวันนี้ช่วงนี้ ──
            # คำนวณช่วงเวลาของ schedule เป็น UTC เพื่อ query
            sched_start_dt = local_now.replace(
                hour=start_time.hour, minute=start_time.minute,
                second=0, microsecond=0,
            ).astimezone(timezone.utc)
            sched_end_dt = local_now.replace(
                hour=end_time.hour, minute=end_time.minute,
                second=0, microsecond=0,
            ).astimezone(timezone.utc)
            existing = (
                sb.table("sessions")
                .select("id, is_open, end_time")
                .eq("course_id", course_id)
                .eq("start_time", sched_start_dt.isoformat())
                .execute()
                .data or []
            )
            beacon_id_to_use = sch.get("beacon_id") or default_beacon_id
            if not existing and beacon_id_to_use:
                day_name  = DAY_NAMES[today_dow]
                title     = f"{course['code']} {day_name} {today_date} ({sch_start}–{sch_end})"
                try:
                    sb.table("sessions").insert({
                        "course_id":  course_id,
                        "beacon_id":  beacon_id_to_use,
                        "title":      title,
                        "start_time": sched_start_dt.isoformat(),
                        "end_time":   sched_end_dt.isoformat() if now_time >= end_time else None,
                        "is_open":    start_time <= now_time < end_time,
                    }).execute()
                    _log.info(f"[SCHEDULER] Auto-created: {title}")
                except Exception as insert_err:
                    err_str = str(insert_err)
                    if "23505" in err_str or "duplicate" in err_str.lower() or "unique" in err_str.lower():
                        # F-10: UNIQUE(course_id, start_time) already rejected this —
                        # expected/normal when another scheduler instance (or a
                        # concurrent manual create) won the race this tick, not an error.
                        _log.info(f"[SCHEDULER] Auto-create skipped (already exists): {title}")
                    else:
                        raise

            # Each occurrence follows its own schedule, including adjacent classes.
            desired_open = start_time <= now_time < end_time
            desired_end = sched_end_dt.isoformat() if now_time >= end_time else None
            for sess in existing:
                if sess["is_open"] != desired_open or sess.get("end_time") != desired_end:
                    sb.table("sessions").update({
                        "is_open": desired_open,
                        "end_time": desired_end,
                    }).eq("id", sess["id"]).execute()

    except Exception as e:
        _log.error(f"[SCHEDULER] Error: {e}", exc_info=True)


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
    start_lock = Lock()

    @app.before_request
    def ensure_scheduler_started():
        # Only the serving process receives requests. Debug mode alone cannot
        # tell whether a reloader is enabled (e.g. flask run --no-reload).
        # Defer startup so the reloader supervisor never starts a second job.
        with start_lock:
            if not scheduler.running:
                # The first request may arrive long after app creation. Set the
                # initial run here so APScheduler does not discard it as late.
                scheduler.get_job("session_manager").modify(next_run_time=datetime.now(timezone.utc))
                scheduler.start()
                _log.info("[SCHEDULER] Started — checking every minute")

    return scheduler

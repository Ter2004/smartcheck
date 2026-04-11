"""
scheduler.py — Auto create/close sessions ตาม schedules table
รันทุก 1 นาที:
  - สร้าง session อัตโนมัติสำหรับทุก schedule ที่ตรงกับวันนี้ (ถ้ายังไม่มี)
  - ปิด session ถ้าเลยเวลาจบแล้ว
"""
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from apscheduler.schedulers.background import BackgroundScheduler

TZ_THAI = ZoneInfo("Asia/Bangkok")

DAY_NAMES = ["จันทร์", "อังคาร", "พุธ", "พฤหัส", "ศุกร์", "เสาร์", "อาทิตย์"]


def _get_supabase():
    from app import supabase_admin
    return supabase_admin


def auto_manage_sessions():
    try:
        sb        = _get_supabase()
        now       = datetime.now(timezone.utc)
        local_now = datetime.now(TZ_THAI)
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
                .select("id")
                .eq("course_id", course_id)
                .gte("start_time", sched_start_dt.isoformat())
                .lte("start_time", sched_end_dt.isoformat())
                .execute()
                .data or []
            )
            if not existing and default_beacon_id:
                day_name  = DAY_NAMES[today_dow]
                title     = f"{course['code']} {day_name} {today_date} ({sch_start}–{sch_end})"
                sb.table("sessions").insert({
                    "course_id":  course_id,
                    "beacon_id":  default_beacon_id,
                    "title":      title,
                    "start_time": sched_start_dt.isoformat(),
                    "end_time":   None,
                    "is_open":    False,
                }).execute()
                print(f"[SCHEDULER] Auto-created: {title}")

            # ─── Auto-close: เลยเวลาจบ ────────────────────────────────
            if now_time >= end_time:
                # ใช้ Thai midnight แปลงเป็น UTC เพื่อหา session ที่เริ่มวันนี้
                # (session ที่เริ่มก่อน 07:00 Thai จะมี start_time UTC เป็นวันก่อนหน้า)
                thai_midnight = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
                thai_midnight_utc = thai_midnight.astimezone(timezone.utc)
                open_sessions = (
                    sb.table("sessions")
                    .select("id, title")
                    .eq("course_id", course_id)
                    .eq("is_open", True)
                    .gte("start_time", thai_midnight_utc.isoformat())
                    .execute()
                    .data or []
                )
                for sess in open_sessions:
                    sb.table("sessions").update({
                        "is_open":  False,
                        "end_time": now.isoformat(),
                    }).eq("id", sess["id"]).execute()
                    print(f"[SCHEDULER] Auto-closed: {sess['title']}")

    except Exception as e:
        print(f"[SCHEDULER] Error: {e}")


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
        print("[KEEP-ALIVE] Reconnected to Supabase")


def start_scheduler():
    scheduler = BackgroundScheduler(timezone="Asia/Bangkok")
    scheduler.add_job(auto_manage_sessions, "interval", minutes=1, id="session_manager")
    scheduler.add_job(keep_alive, "interval", minutes=3, id="keep_alive")
    scheduler.start()
    print("[SCHEDULER] Started — checking every minute")
    return scheduler

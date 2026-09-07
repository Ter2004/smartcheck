"""
scripts/seed_load_test.py
==========================
เตรียมข้อมูลสำหรับทดสอบโหลด 50 นักศึกษา (check-in flow):

  - teacher 1 คน
  - course 1 วิชา ผูกกับ teacher นั้น
  - schedule ของวันนี้ (day_of_week = วันนี้, ช่วงเวลาครอบเวลาปัจจุบัน)
  - session ที่เปิดอยู่ (is_open=True) ผูกกับ beacon TEST-101 — สร้างตรง ๆ
    ไม่รอ APScheduler เพราะ load test ต้องเริ่มได้ทันที ไม่ใช่รอ tick ถัดไป
    (ดูหมายเหตุ "ทำไมสร้าง session ตรง ๆ" ด้านล่าง)
  - student 50 คน, enroll เข้า course ครบ
  - student_biometrics: fake face_embeddings (512-dim สุ่ม) + integrity_hash
    ที่คำนวณถูกต้องจริงด้วย compute_embedding_integrity_hash() ตัวเดียวกับ
    ที่ app ใช้ — ถ้า hash ผิด verify_embedding_integrity() จะ reject
    "ก่อน" extract_embedding()/DeepFace.represent() ถูกเรียกเลย
    (api_checkin.py:330-345 — เช็ค integrity ก่อนถึง embed เสมอ)
    ทำให้ timing ที่วัดได้จาก load test ไม่มีความหมาย ถ้า hash ไม่ตรง

ทำไมสร้าง session ตรง ๆ แทนที่จะรอ scheduler:
  1) load test ต้องรันได้ทันที ไม่ต้องรอ APScheduler tick ถัดไป (สูงสุด 60 วิ)
  2) script นี้ควรรันได้แบบ standalone โดยไม่ต้องพึ่ง Flask app/scheduler
     ทำงานอยู่เลยด้วยซ้ำ (idempotent, ทดสอบซ้ำได้)
  3) ยังคง insert แถวใน `schedules` ไว้ด้วย (ตามที่โจทย์ขอ) เพื่อให้
     auto_manage_sessions() เห็น course นี้เป็นส่วนหนึ่งของ N ในการทดสอบ
     เรื่องรวม query — สคริปต์นี้แค่ไม่ต้อง "รอ" มันเพื่อสร้าง session

Idempotent — รันซ้ำได้ปลอดภัย: user/course/schedule เดิมจะถูก reuse ไม่สร้างซ้ำ
session ใหม่จะไม่ถูกสร้างซ้ำถ้ามี open session ของวันนี้อยู่แล้ว
enrollments และ biometrics ใช้ upsert (on_conflict) จึงอัปเดตทับได้เรื่อย ๆ

Usage:
    python scripts/seed_load_test.py --beacon-id <ID>  # skip REST beacon lookup
    python scripts/seed_load_test.py            # seed ข้อมูล
    python scripts/seed_load_test.py --clean    # ลบเฉพาะข้อมูล loadtest ที่ script นี้สร้าง
                                                  # (match ด้วย email pattern / course code เท่านั้น
                                                  #  ไม่แตะ admin หรือ beacon TEST-101)

ต้องตั้งใน .env (project root):
    SEED_BEACON_ID                     — optional existing beacon ID; --beacon-id wins
    SUPABASE_URL, SUPABASE_SERVICE_KEY  — จำเป็นเสมอ
    EMBEDDING_INTEGRITY_SALT            — จำเป็นเฉพาะตอน seed (ไม่ใช่ --clean)
                                           ถ้าไม่ตั้ง สคริปต์จะสร้าง teacher/course/
                                           schedule/session/students/enrollments ให้ครบ
                                           แต่ "หยุด" ก่อนขั้น biometrics แล้วบอกวิธีแก้
"""
import sys
import os
import math
import random
import argparse
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

# ── เพิ่ม project root ลง sys.path (ตามแบบ scripts/backfill_integrity_hash.py) ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows console (cp1252) ไม่รองรับตัวอักษรไทยโดย default — บังคับ utf-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

# ============================================================
# ค่าคงที่ — แก้ตรงนี้ที่เดียว
# ============================================================
LOADTEST_PASSWORD      = "LoadTest2026!"          # ใช้ login ได้ทั้ง teacher และ student 50 คน
TEACHER_EMAIL          = "loadtest_teacher@smartcheck.local"
TEACHER_NAME           = "Load Test Teacher"

STUDENT_COUNT          = 50
STUDENT_EMAIL_FMT      = "loadtest{:03d}@smartcheck.local"   # loadtest001..loadtest050
STUDENT_NAME_FMT       = "Load Test Student {:03d}"
STUDENT_ID_FMT          = "LT2026{:03d}"

COURSE_CODE            = "LOADTEST101"
COURSE_NAME            = "Load Test Course"
COURSE_SEMESTER        = 1

BEACON_ROOM_MATCH      = "TEST-101"   # ต้อง match room_name ของ beacon ที่มีอยู่แล้ว — ไม่สร้างใหม่

EMBEDDING_DIM           = 512
EMBEDDINGS_PER_STUDENT  = 5
BASELINE_EAR            = 0.28

TZ_THAI = ZoneInfo("Asia/Bangkok")

class BeaconLookupError(RuntimeError):
    """A beacon must be supplied manually when REST lookup is unavailable."""


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Seed today's load-test session (Asia/Bangkok).")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--beacon-id", default=os.getenv("SEED_BEACON_ID"),
                        help="Existing beacon ID; overrides SEED_BEACON_ID and skips room lookup")
    return parser.parse_args(argv)


def _resolve_beacon_id(sb, supplied=None):
    if supplied and supplied.strip():
        return supplied.strip()
    try:
        return _get_beacon_id(sb)
    except Exception:
        raise BeaconLookupError(
            "Beacon lookup failed or TEST-101 was not found. Copy its id from the "
            "Supabase SQL editor and rerun with --beacon-id <ID> or set SEED_BEACON_ID "
            "in .env. This skips the REST room-name lookup; completed seed steps can be reused."
        ) from None


# ============================================================
# Helpers
# ============================================================

def _random_unit_embedding(dim: int) -> list:
    """สุ่ม vector แล้ว L2-normalize ให้เหมือนโครงสร้าง FaceNet512 embedding จริง
    (ไม่ต้อง match ใบหน้าใครได้ — load test วัด compute cost ไม่ใช่ match success)"""
    raw = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in raw)) or 1.0
    return [round(v / norm, 6) for v in raw]


def _get_or_create_user(sb, email: str, full_name: str, role: str, student_id: str | None = None) -> str:
    """หา users row จาก email ก่อน — ถ้ามีแล้ว reuse id, ถ้าไม่มีสร้างทั้ง Auth + public.users
    (เช็คจาก public.users เป็น source of truth เดียว เพื่อไม่ต้อง parse exception
    ของ Supabase Auth ตอน email ซ้ำ)"""
    existing = sb.table("users").select("id").eq("email", email).maybe_single().execute()
    if existing and existing.data:
        return existing.data["id"]

    auth_res = sb.auth.admin.create_user({
        "email": email,
        "password": LOADTEST_PASSWORD,
        "email_confirm": True,
    })
    uid = str(auth_res.user.id)

    row = {
        "id": uid,
        "email": email,
        "full_name": full_name,
        "role": role,
        "is_active": True,
        "must_change_password": False,   # กัน F-11 flow มาขวาง load test
    }
    if student_id:
        row["student_id"] = student_id
    sb.table("users").insert(row).execute()
    return uid


def _get_or_create_course(sb, teacher_id: str) -> str:
    existing = sb.table("courses").select("id").eq("code", COURSE_CODE).execute()
    if existing.data:
        return existing.data[0]["id"]
    res = sb.table("courses").insert({
        "code": COURSE_CODE,
        "name": COURSE_NAME,
        "teacher_id": teacher_id,
        "semester": COURSE_SEMESTER,
        "is_active": True,
    }).execute()
    return res.data[0]["id"]


def _get_or_create_schedule(sb, course_id: str, today_dow: int) -> str:
    existing = (
        sb.table("schedules").select("id")
        .eq("course_id", course_id).eq("day_of_week", today_dow)
        .execute()
    )
    if existing.data:
        return existing.data[0]["id"]

    now_local = datetime.now(TZ_THAI)
    start_t = max(now_local - timedelta(minutes=5), now_local.replace(hour=0, minute=0, second=0)).strftime("%H:%M:00")
    end_t   = (now_local.replace(hour=min(now_local.hour + 2, 23))).strftime("%H:%M:00")

    res = sb.table("schedules").insert({
        "course_id": course_id,
        "day_of_week": today_dow,
        "start_time": start_t,
        "end_time": end_t,
    }).execute()
    return res.data[0]["id"]


def _get_beacon_id(sb) -> str:
    res = sb.table("beacons").select("id, room_name").ilike("room_name", f"%{BEACON_ROOM_MATCH}%").execute()
    if not res.data:
        raise BeaconLookupError("No matching beacon")
    return res.data[0]["id"]


def _get_or_create_open_session(sb, course_id: str, beacon_id: str) -> dict:
    now_local = datetime.now(TZ_THAI)
    thai_midnight_utc = now_local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    tomorrow_utc = thai_midnight_utc + timedelta(days=1)
    existing = (
        sb.table("sessions").select("id, title, start_time")
        .eq("course_id", course_id).eq("is_open", True)
        .gte("start_time", thai_midnight_utc.isoformat())
        .lt("start_time", tomorrow_utc.isoformat())
        .eq("beacon_id", beacon_id)
        .execute()
    )
    if existing.data:
        return existing.data[0]

    today_str = now_local.date().isoformat()
    title = f"{COURSE_CODE} Load Test {today_str}"
    try:
        res = sb.table("sessions").insert({
            "course_id": course_id,
            "beacon_id": beacon_id,
            "title": title,
            "start_time": now_local.astimezone(timezone.utc).isoformat(),
            "end_time": None,
            "checkin_duration": None,
            "is_open": True,
        }).execute()
        return res.data[0]
    except Exception as e:
        # F-10: UNIQUE(course_id, start_time) — ถ้าชนแสดงว่ามีแถวอยู่แล้วจากรอบก่อนหน้า
        # (เผื่อ race กับ scheduler ตัวจริงที่รันคู่กันอยู่) — ดึงแถวที่มีอยู่มาใช้แทน
        err_str = str(e)
        if "23505" in err_str or "duplicate" in err_str.lower() or "unique" in err_str.lower():
            print(f"  (session insert ชน UNIQUE constraint — ดึงแถวที่มีอยู่แล้วมาใช้แทน)")
            existing = (
                sb.table("sessions").select("id, title, start_time")
                .eq("course_id", course_id)
                .eq("is_open", True).eq("beacon_id", beacon_id)
                .gte("start_time", thai_midnight_utc.isoformat())
                .lt("start_time", tomorrow_utc.isoformat())
                .order("start_time", desc=True).limit(1)
                .execute()
            )
            if existing.data:
                return existing.data[0]
        raise


# ============================================================
# Seed
# ============================================================

def _seed(sb, beacon_id=None):
    print("=== Seed load-test data ===\n")

    # ── 1) Teacher ──────────────────────────────────────────
    teacher_id = _get_or_create_user(sb, TEACHER_EMAIL, TEACHER_NAME, "teacher")
    print(f"[1/6] teacher: {TEACHER_EMAIL} ({teacher_id})")

    # ── 2) Course ───────────────────────────────────────────
    course_id = _get_or_create_course(sb, teacher_id)
    print(f"[2/6] course: {COURSE_CODE} ({course_id})")

    # ── 3) Schedule (วันนี้) ────────────────────────────────
    today_dow = datetime.now(TZ_THAI).weekday()
    schedule_id = _get_or_create_schedule(sb, course_id, today_dow)
    print(f"[3/6] schedule: day_of_week={today_dow} ({schedule_id})")

    # ── 4) Session เปิดอยู่ ผูก beacon TEST-101 ─────────────
    beacon_id = _resolve_beacon_id(sb, beacon_id)
    session = _get_or_create_open_session(sb, course_id, beacon_id)
    print(f"[4/6] session: {session['title']} is_open=True ({session['id']})")

    # ── 5) Student 50 คน + enroll ───────────────────────────
    student_ids = []
    for i in range(1, STUDENT_COUNT + 1):
        email = STUDENT_EMAIL_FMT.format(i)
        name  = STUDENT_NAME_FMT.format(i)
        sid   = STUDENT_ID_FMT.format(i)
        uid   = _get_or_create_user(sb, email, name, "student", student_id=sid)
        student_ids.append(uid)
    print(f"[5/6] students: {len(student_ids)} คน (loadtest001..loadtest{STUDENT_COUNT:03d}@smartcheck.local)")

    enroll_rows = [{"course_id": course_id, "student_id": sid} for sid in student_ids]
    sb.table("course_enrollments").upsert(enroll_rows, on_conflict="course_id,student_id").execute()
    print(f"       enrolled ทั้งหมดเข้า {COURSE_CODE} แล้ว (upsert)")

    # ── 6) Fake face embeddings + integrity_hash ────────────
    salt = os.getenv("EMBEDDING_INTEGRITY_SALT")
    if not salt:
        print("\n" + "=" * 70)
        print("หยุดที่ขั้นตอน biometrics — EMBEDDING_INTEGRITY_SALT ไม่ได้ตั้งใน .env")
        print("=" * 70)
        print(
            "ทุกอย่างก่อนหน้านี้ (teacher/course/schedule/session/students/enrollments)\n"
            "สร้างเสร็จแล้วและ idempotent — รันสคริปต์นี้ซ้ำได้เรื่อย ๆ โดยไม่สร้างซ้ำ\n\n"
            "แต่ student_biometrics (fake face_embeddings + integrity_hash) ยังไม่ถูกสร้าง\n"
            "เพราะ api_checkin.py:330-345 เช็ค verify_embedding_integrity() ก่อนเรียก\n"
            "extract_embedding() (DeepFace.represent, ~350ms) เสมอ — ถ้า hash ผิดหรือไม่มี\n"
            "request จะถูก reject ตั้งแต่ก่อนเรียก DeepFace เลย ทำให้ timing ที่วัดได้จาก\n"
            "load test ไม่มีความหมาย (วัดแค่เวลาที่ reject ไว ๆ ไม่ใช่เวลา compute จริง)\n\n"
            "ตั้งค่า EMBEDDING_INTEGRITY_SALT ใน .env ให้ตรงกับค่าที่ Flask app ที่จะรับ\n"
            "load test จริงใช้อยู่ (ค่าเดียวกับที่ create_app() อ่านผ่าน Config) แล้วรัน\n"
            "สคริปต์นี้ซ้ำอีกครั้ง — ขั้นตอน 1-5 ด้านบนจะถูก skip อัตโนมัติ (idempotent)\n"
            "เหลือแค่ขั้นตอนนี้ที่จะทำต่อให้จบ"
        )
        print("=" * 70)
        _print_summary(teacher_id, course_id, schedule_id, session, student_ids, biometrics_done=False)
        sys.exit(1)

    from app.services.security_service import compute_embedding_integrity_hash

    bio_rows = []
    for uid in student_ids:
        embeddings = [_random_unit_embedding(EMBEDDING_DIM) for _ in range(EMBEDDINGS_PER_STUDENT)]
        integrity_hash = compute_embedding_integrity_hash(uid, embeddings, salt)
        bio_rows.append({
            "user_id": uid,
            "face_embeddings": embeddings,
            "baseline_ear": BASELINE_EAR,
            "consent_given": True,
            "consent_at": datetime.now(timezone.utc).isoformat(),
            "enrolled_at": datetime.now(timezone.utc).isoformat(),
            "integrity_hash": integrity_hash,
        })
    sb.table("student_biometrics").upsert(bio_rows, on_conflict="user_id").execute()
    print(f"[6/6] student_biometrics: {len(bio_rows)} แถว (fake {EMBEDDING_DIM}-dim x{EMBEDDINGS_PER_STUDENT} + integrity_hash ถูกต้อง)")

    _print_summary(teacher_id, course_id, schedule_id, session, student_ids, biometrics_done=True)


def _print_summary(teacher_id, course_id, schedule_id, session, student_ids, biometrics_done):
    print("\n=== สรุป ===")
    print(f"teacher_id:   {teacher_id}  ({TEACHER_EMAIL} / {LOADTEST_PASSWORD})")
    print(f"course_id:    {course_id}  ({COURSE_CODE})")
    print(f"schedule_id:  {schedule_id}")
    print(f"session_id (open): {session['id']}  — {session['title']}")
    print(f"students:     {len(student_ids)} คน")
    print(f"  email:      {STUDENT_EMAIL_FMT.format(1)} .. {STUDENT_EMAIL_FMT.format(STUDENT_COUNT)}")
    print(f"  password:   {LOADTEST_PASSWORD}  (เหมือนกันทุกคน)")
    print(f"biometrics:   {'สร้างครบแล้ว — พร้อม check-in load test' if biometrics_done else 'ยังไม่ได้สร้าง — ดูคำอธิบายด้านบน'}")


# ============================================================
# Clean
# ============================================================

def _clean(sb):
    print("=== Clean load-test data ===\n")
    print("(match เฉพาะ course.code == LOADTEST101 และ users.email ILIKE 'loadtest%@smartcheck.local'")
    print(" ไม่แตะ admin หรือ beacon TEST-101)\n")

    # ต้องลบ course ก่อน users เพราะ courses.teacher_id เป็น ON DELETE RESTRICT
    # (ลบ course ก่อน cascade ไปที่ sessions/schedules/enrollments/attendance ให้เอง)
    course_res = sb.table("courses").select("id, code").eq("code", COURSE_CODE).execute()
    for c in (course_res.data or []):
        sb.table("courses").delete().eq("id", c["id"]).execute()
        print(f"  deleted course {c['code']} ({c['id']}) — cascades sessions/schedules/enrollments/attendance")

    users_res = sb.table("users").select("id, email").ilike("email", "loadtest%@smartcheck.local").execute()
    loadtest_users = users_res.data or []
    for u in loadtest_users:
        sb.table("users").delete().eq("id", u["id"]).execute()   # cascades student_biometrics/consent_logs
        try:
            sb.auth.admin.delete_user(u["id"])
        except Exception as e:
            print(f"  warn: ลบ Supabase Auth user ไม่สำเร็จสำหรับ {u['email']}: {e}", file=sys.stderr)
        print(f"  deleted user {u['email']} ({u['id']})")

    print(f"\nDone — removed {len(course_res.data or [])} course(s), {len(loadtest_users)} user(s)")


# ============================================================
def main(argv=None):
    args = _parse_args(argv)
    from supabase import create_client

    supabase_url = os.getenv("SUPABASE_URL")
    service_key  = os.getenv("SUPABASE_SERVICE_KEY")
    if not supabase_url or not service_key:
        print("ERROR: SUPABASE_URL / SUPABASE_SERVICE_KEY ต้องตั้งใน .env", file=sys.stderr)
        sys.exit(1)

    sb = create_client(supabase_url, service_key)

    try:
        if args.clean:
            _clean(sb)
        else:
            _seed(sb, args.beacon_id)
    except BeaconLookupError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

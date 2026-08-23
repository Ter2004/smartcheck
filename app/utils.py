"""
utils.py — small shared helpers with no Flask/Supabase-specific dependencies.
Kept dependency-free (stdlib only) so any layer (routes, services) can import
it without circular-import risk.
"""


def friendly_error(e: Exception) -> str:
    """แปลง exception จาก Supabase/DB เป็นข้อความภาษาไทยที่อ่านได้"""
    msg = str(e)
    if "23505" in msg or "duplicate key" in msg:
        if "sessions_course_id_start_time_key" in msg or ("course_id" in msg and "start_time" in msg):
            return "มี session ของวิชานี้ในเวลาเดียวกันอยู่แล้ว"
        if "student_id" in msg:
            return "รหัสนักศึกษานี้มีในระบบแล้ว"
        if "email" in msg:
            return "อีเมลนี้มีในระบบแล้ว"
        return "ข้อมูลซ้ำในระบบ"
    if "23503" in msg or "foreign key" in msg:
        return "ข้อมูลอ้างอิงไม่ถูกต้อง"
    if "already registered" in msg or "User already registered" in msg:
        return "อีเมลนี้ถูกลงทะเบียนแล้ว"
    if "invalid" in msg.lower() and "email" in msg.lower():
        return "รูปแบบอีเมลไม่ถูกต้อง"
    return "เกิดข้อผิดพลาด กรุณาลองใหม่"

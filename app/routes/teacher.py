import csv
import io
from datetime import datetime, timezone, date, timedelta
from flask import (Blueprint, render_template, request, redirect,
                   url_for, flash, session, jsonify, Response)
from app.routes.auth import login_required, role_required
from app import supabase_admin
from app.services.security_service import log_audit_event, csrf_protect, csrf_protect_form

teacher_bp = Blueprint("teacher", __name__)


# ─── Session History ──────────────────────────────────────────

@teacher_bp.route("/history")
@login_required
@role_required("teacher")
@csrf_protect_form
def history():
    teacher_id  = session["user_id"]
    course_filter = request.args.get("course_id", "")
    date_from     = request.args.get("date_from", "")
    date_to       = request.args.get("date_to", "")

    courses = (
        supabase_admin.table("courses")
        .select("id, code, name, section")
        .eq("teacher_id", teacher_id)
        .eq("is_active", True)
        .order("code")
        .execute()
        .data or []
    )

    course_ids = [c["id"] for c in courses] or ["00000000-0000-0000-0000-000000000000"]

    query = (
        supabase_admin.table("sessions")
        .select("*, courses(code, name, section)")
        .in_("course_id", course_ids if not course_filter else [course_filter])
        .order("start_time", desc=True)
    )
    if date_from:
        query = query.gte("start_time", date_from + "T00:00:00+00:00")
    if date_to:
        query = query.lte("start_time", date_to + "T23:59:59+00:00")

    sessions_data = query.limit(200).execute().data or []

    # นับจำนวนเข้าเรียนต่อ session
    if sessions_data:
        session_ids = [s["id"] for s in sessions_data]
        att_counts_raw = (
            supabase_admin.table("attendance")
            .select("session_id, status")
            .in_("session_id", session_ids)
            .in_("status", ["present", "late", "manual"])
            .execute()
            .data or []
        )
        att_map = {}
        for a in att_counts_raw:
            att_map[a["session_id"]] = att_map.get(a["session_id"], 0) + 1
        for s in sessions_data:
            s["present_count"] = att_map.get(s["id"], 0)

    return render_template(
        "teacher/history.html",
        courses=courses,
        sessions=sessions_data,
        course_filter=course_filter,
        date_from=date_from,
        date_to=date_to,
    )


# ─── Dashboard ────────────────────────────────────────────────

@teacher_bp.route("/dashboard")
@login_required
@role_required("teacher")
@csrf_protect_form
def dashboard():
    teacher_id = session["user_id"]

    courses = (
        supabase_admin.table("courses")
        .select("*")
        .eq("teacher_id", teacher_id)
        .eq("is_active", True)
        .order("code")
        .execute()
        .data or []
    )

    # sessions ล่าสุด 10 รายการ
    recent_sessions = (
        supabase_admin.table("sessions")
        .select("*, courses(code, name)")
        .in_("course_id", [c["id"] for c in courses] or ["00000000-0000-0000-0000-000000000000"])
        .order("start_time", desc=True)
        .limit(10)
        .execute()
        .data or []
    )

    beacons = (
        supabase_admin.table("beacons")
        .select("id, room_name, uuid, rssi_threshold")
        .eq("is_active", True)
        .order("room_name")
        .execute()
        .data or []
    )

    # แปลง start_time เป็นวันที่ไทย (UTC+7) เพื่อใช้จัดกลุ่มใน template
    from datetime import datetime, timedelta as _td
    _TH_OFFSET = _td(hours=7)
    for s in recent_sessions:
        raw = s.get("start_time", "")
        if raw:
            try:
                utc_dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                s["thai_date"] = (utc_dt + _TH_OFFSET).date().isoformat()
            except Exception:
                s["thai_date"] = raw[:10]
        else:
            s["thai_date"] = ""

    return render_template(
        "teacher/dashboard.html",
        courses=courses,
        recent_sessions=recent_sessions,
        beacons=beacons,
        today_str=date.today().isoformat(),
    )


# ─── Create Session ───────────────────────────────────────────

@teacher_bp.route("/session/create", methods=["POST"])
@login_required
@role_required("teacher")
@csrf_protect_form
def session_create():
    teacher_id = session["user_id"]
    course_id  = request.form.get("course_id")
    beacon_id  = request.form.get("beacon_id")
    title      = request.form.get("title", "").strip()
    start_time = request.form.get("start_time")
    end_time   = request.form.get("end_time")

    if not all([course_id, beacon_id, title, start_time, end_time]):
        flash("กรุณากรอกข้อมูลให้ครบ", "danger")
        return redirect(url_for("teacher.dashboard"))

    # ตรวจสอบว่า course เป็นของ teacher คนนี้
    course = (
        supabase_admin.table("courses")
        .select("id")
        .eq("id", course_id)
        .eq("teacher_id", teacher_id)
        .maybe_single()
        .execute()
        .data
    )
    if not course:
        flash("ไม่มีสิทธิ์สร้าง session ให้วิชานี้", "danger")
        return redirect(url_for("teacher.dashboard"))

    try:
        res = supabase_admin.table("sessions").insert({
            "course_id":  course_id,
            "beacon_id":  beacon_id,
            "title":      title,
            "start_time": start_time,
            "end_time":   end_time,
            "is_open":    True,
        }).execute()
        new_id = res.data[0]["id"]
        flash(f"สร้างคาบเรียน '{title}' สำเร็จ", "success")
        return redirect(url_for("teacher.session_view", session_id=new_id))
    except Exception as e:
        flash(f"สร้างไม่สำเร็จ: {e}", "danger")
        return redirect(url_for("teacher.dashboard"))


# ─── Session View ─────────────────────────────────────────────

@teacher_bp.route("/session/<session_id>")
@login_required
@role_required("teacher")
@csrf_protect_form
def session_view(session_id):
    teacher_id = session["user_id"]

    sess = (
        supabase_admin.table("sessions")
        .select("*, courses(id, code, name, teacher_id), beacons(room_name)")
        .eq("id", session_id)
        .maybe_single()
        .execute()
        .data
    )
    if not sess or not sess.get("courses") or sess["courses"]["teacher_id"] != teacher_id:
        flash("ไม่พบ session หรือไม่มีสิทธิ์", "danger")
        return redirect(url_for("teacher.dashboard"))

    attendance = (
        supabase_admin.table("attendance")
        .select("*, users!attendance_student_id_fkey(full_name, student_id, email)")
        .eq("session_id", session_id)
        .order("check_in_at")
        .execute()
        .data or []
    )

    # นักศึกษาทั้งหมดในวิชานี้
    all_students = (
        supabase_admin.table("course_enrollments")
        .select("*, users(id, full_name, student_id, email)")
        .eq("course_id", sess["course_id"])
        .execute()
        .data or []
    )

    # att_map: student_id → attendance row (สร้างใน Python เพื่อใช้ใน template)
    att_map = {a["student_id"]: a for a in attendance}

    return render_template(
        "teacher/session_view.html",
        sess=sess,
        attendance=attendance,
        all_students=all_students,
        att_map=att_map,
    )


# ─── Toggle Session Open/Close ────────────────────────────────

@teacher_bp.route("/session/<session_id>/toggle", methods=["POST"])
@login_required
@role_required("teacher")
@csrf_protect_form
def session_toggle(session_id):
    teacher_id = session["user_id"]

    sess = (
        supabase_admin.table("sessions")
        .select("is_open, start_time, course_id, courses(id, teacher_id)")
        .eq("id", session_id)
        .maybe_single()
        .execute()
        .data
    )
    if not sess or not sess.get("courses") or sess["courses"]["teacher_id"] != teacher_id:
        flash("ไม่มีสิทธิ์", "danger")
        return redirect(url_for("teacher.dashboard"))

    new_state = not sess["is_open"]
    now_dt    = datetime.now(timezone.utc)

    # ─── ถ้าจะเปิด: ตรวจ schedule ว่าอยู่ในช่วงเวลาที่อนุญาตไหม ──────
    if new_state:
        from zoneinfo import ZoneInfo
        from datetime import timedelta
        local_now  = now_dt.astimezone(ZoneInfo("Asia/Bangkok"))
        today_dow  = local_now.weekday()
        now_time   = local_now.time()

        schedules = (
            supabase_admin.table("schedules")
            .select("start_time, end_time")
            .eq("course_id", sess["course_id"])
            .eq("day_of_week", today_dow)
            .execute()
            .data or []
        )

        if schedules:
            from datetime import time as dtime
            in_window = False
            for sch in schedules:
                if not sch.get("start_time") or not sch.get("end_time"):
                    continue
                parts_s = sch["start_time"].split(":")
                parts_e = sch["end_time"].split(":")
                s_time  = dtime(int(parts_s[0]), int(parts_s[1]))
                e_time  = dtime(int(parts_e[0]), int(parts_e[1]))
                if s_time <= now_time <= e_time:
                    in_window = True
                    break
            if not in_window:
                flash("ไม่สามารถเปิดคาบได้ — อยู่นอกช่วงเวลาที่กำหนดในตารางเรียน", "danger")
                return redirect(url_for("teacher.session_view", session_id=session_id))

    update_data = {"is_open": new_state}
    if new_state:
        update_data["start_time"] = now_dt.isoformat()
        update_data["end_time"]   = None
        duration_str = request.form.get("checkin_duration", "").strip()
        if duration_str.isdigit() and int(duration_str) > 0:
            update_data["checkin_duration"] = int(duration_str)
        else:
            update_data["checkin_duration"] = None
    if not new_state:
        update_data["end_time"] = now_dt.isoformat()

    supabase_admin.table("sessions").update(update_data).eq("id", session_id).execute()
    flash(f"{'เปิด' if new_state else 'ปิด'}การเช็คชื่อแล้ว", "success")
    return redirect(url_for("teacher.session_view", session_id=session_id))


# ─── Manual Override ──────────────────────────────────────────

@teacher_bp.route("/session/<session_id>/override", methods=["POST"])
@login_required
@role_required("teacher")
@csrf_protect_form
def override_attendance(session_id):
    teacher_id   = session["user_id"]
    student_id   = request.form.get("student_id")
    new_status   = request.form.get("status")
    reason       = request.form.get("reason", "").strip()

    if new_status not in ("present", "late", "absent", "manual"):
        flash("สถานะไม่ถูกต้อง", "danger")
        return redirect(url_for("teacher.session_view", session_id=session_id))

    # ตรวจสอบสิทธิ์
    sess = (
        supabase_admin.table("sessions")
        .select("courses(teacher_id)")
        .eq("id", session_id)
        .maybe_single()
        .execute()
        .data
    )
    if not sess or not sess.get("courses") or sess["courses"]["teacher_id"] != teacher_id:
        flash("ไม่มีสิทธิ์", "danger")
        return redirect(url_for("teacher.dashboard"))

    now = datetime.now(timezone.utc).isoformat()

    # upsert attendance row
    existing = (
        supabase_admin.table("attendance")
        .select("id, status")
        .eq("session_id", session_id)
        .eq("student_id", student_id)
        .maybe_single()
        .execute()
        .data
    )

    if existing:
        supabase_admin.table("attendance").update({
            "status":          new_status,
            "override_by":     teacher_id,
            "override_reason": reason,
            "override_at":     now,
        }).eq("id", existing["id"]).execute()
    else:
        supabase_admin.table("attendance").insert({
            "session_id":      session_id,
            "student_id":      student_id,
            "status":          new_status,
            "override_by":     teacher_id,
            "override_reason": reason,
            "override_at":     now,
            "check_in_at":     now,
            "ble_pass":        False,
            "liveness_pass":   False,
            "face_pass":       False,
        }).execute()

    # Sprint 3B: audit log — every manual override is recorded permanently
    log_audit_event(
        supabase_admin,
        actor_id   = teacher_id,
        actor_role = "teacher",
        event_type = "teacher_override",
        target_id  = student_id,
        session_id = session_id,
        old_value  = existing["status"] if existing and isinstance(existing, dict) else None,
        new_value  = new_status,
        metadata   = {"reason": reason or ""},
    )

    flash(f"บันทึกสถานะ '{new_status}' สำเร็จ", "success")
    return redirect(url_for("teacher.session_view", session_id=session_id))


# ─── Export CSV ───────────────────────────────────────────────

@teacher_bp.route("/session/<session_id>/export")
@login_required
@role_required("teacher")
@csrf_protect_form
def export_csv(session_id):
    teacher_id = session["user_id"]

    sess = (
        supabase_admin.table("sessions")
        .select("title, start_time, courses(code, name, teacher_id)")
        .eq("id", session_id)
        .maybe_single()
        .execute()
        .data
    )
    if not sess or not sess.get("courses") or sess["courses"]["teacher_id"] != teacher_id:
        flash("ไม่มีสิทธิ์", "danger")
        return redirect(url_for("teacher.dashboard"))

    attendance = (
        supabase_admin.table("attendance")
        .select("*, users!attendance_student_id_fkey(full_name, student_id, email)")
        .eq("session_id", session_id)
        .order("check_in_at")
        .execute()
        .data or []
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "student_id", "full_name", "email",
        "status", "check_in_at",
        "face_score", "ble_rssi",
        "liveness_action", "override_reason",
    ])
    for a in attendance:
        u = a.get("users") or {}
        writer.writerow([
            u.get("student_id", ""),
            u.get("full_name", ""),
            u.get("email", ""),
            a.get("status", ""),
            (a.get("check_in_at") or "")[:19],
            a.get("face_score", ""),
            a.get("ble_rssi", ""),
            a.get("liveness_action", ""),
            a.get("override_reason", ""),
        ])

    course_code = (sess.get("courses") or {}).get("code", "unknown").replace("/", "-")
    date_str = (sess.get("start_time") or "")[:10]
    filename = f"attendance_{course_code}_{date_str}.csv"

    return Response(
        "\ufeff" + output.getvalue(),  # BOM สำหรับ Excel ภาษาไทย
        mimetype="text/csv; charset=utf-8-sig",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ─── Enrollment Retry Reset ───────────────────────────────────

@teacher_bp.route("/api/reset-enrollment/<student_id>", methods=["POST"])
@login_required
@role_required("teacher")
@csrf_protect
def api_reset_enrollment(student_id):
    teacher_id = session["user_id"]

    # Verify the student is enrolled in at least one of this teacher's courses
    teacher_courses = (
        supabase_admin.table("courses")
        .select("id")
        .eq("teacher_id", teacher_id)
        .execute()
    )
    teacher_course_ids = [c["id"] for c in (teacher_courses.data or [])]
    if not teacher_course_ids:
        return jsonify({"status": "error", "message": "ไม่พบนักศึกษาในรายวิชาของคุณ"}), 403

    enroll_res = (
        supabase_admin.table("course_enrollments")
        .select("id")
        .eq("student_id", student_id)
        .in_("course_id", teacher_course_ids)
        .limit(1)
        .execute()
    )
    if not (enroll_res and enroll_res.data):
        return jsonify({"status": "error", "message": "ไม่พบนักศึกษาในรายวิชาของคุณ"}), 403

    try:
        supabase_admin.table("student_biometrics").update({
            "enrollment_attempts":     0,
            "last_enrollment_attempt": None,
        }).eq("user_id", student_id).execute()

        log_audit_event(
            supabase_admin,
            actor_id=teacher_id,
            actor_role="teacher",
            event_type="reset_enrollment_attempts",
            target_id=student_id,
            new_value="0",
        )
        return jsonify({"status": "ok", "message": "รีเซ็ตจำนวนครั้งลงทะเบียนสำเร็จ"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

import csv
import io
import secrets
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from app.routes.auth import login_required, role_required
from app import supabase_admin
from app.services.security_service import log_audit_event, csrf_protect

admin_bp = Blueprint("admin", __name__)


def _friendly_error(e: Exception) -> str:
    """แปลง exception จาก Supabase/DB เป็นข้อความภาษาไทยที่อ่านได้"""
    msg = str(e)
    if "23505" in msg or "duplicate key" in msg:
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


# ─── Dashboard ────────────────────────────────────────────────

@admin_bp.route("/dashboard")
@login_required
@role_required("admin")
def dashboard():
    users_res = supabase_admin.table("users").select("*", count="exact").execute()
    beacons_res = supabase_admin.table("beacons").select("*", count="exact").execute()
    bio_res = (
        supabase_admin.table("student_biometrics")
        .select("*", count="exact")
        .not_.is_("face_embeddings", "null")
        .execute()
    )
    stats = {
        "total_users": users_res.count or 0,
        "total_beacons": beacons_res.count or 0,
        "enrolled_biometrics": bio_res.count or 0,
    }
    return render_template("admin/dashboard.html", stats=stats)


# ─── User List ────────────────────────────────────────────────

@admin_bp.route("/users")
@login_required
@role_required("admin")
def users():
    role_filter = request.args.get("role", "")
    query = supabase_admin.table("users").select("*").order("created_at", desc=True)
    if role_filter:
        query = query.eq("role", role_filter)
    users_data = query.execute().data or []
    return render_template("admin/users.html", users=users_data, role_filter=role_filter)


# ─── CSV Import ───────────────────────────────────────────────

@admin_bp.route("/import-csv", methods=["GET", "POST"])
@login_required
@role_required("admin")
def import_csv():
    if request.method == "GET":
        return render_template("admin/import_csv.html")

    file = request.files.get("csv_file")
    if not file or not file.filename.endswith(".csv"):
        flash("กรุณาอัพโหลดไฟล์ .csv", "danger")
        return redirect(url_for("admin.import_csv"))

    success_count = 0
    error_rows = []
    created_users = []  # เก็บ {full_name, email, role, temp_password} เพื่อแสดงหลัง import

    stream = io.StringIO(file.stream.read().decode("utf-8-sig"))
    reader = csv.DictReader(stream)

    for i, row in enumerate(reader, start=2):
        email = row.get("email", "").strip()
        full_name = row.get("full_name", "").strip()
        role = row.get("role", "student").strip().lower()
        student_id = row.get("student_id", "").strip() or None

        if not email or not full_name:
            error_rows.append(f"แถว {i}: email หรือ full_name ว่าง")
            continue

        if role not in ("student", "teacher"):
            role = "student"

        temp_password = secrets.token_urlsafe(12)

        try:
            auth_res = supabase_admin.auth.admin.create_user({
                "email": email,
                "password": temp_password,
                "email_confirm": True,
            })
            uid = str(auth_res.user.id)

            supabase_admin.table("users").insert({
                "id": uid,
                "email": email,
                "full_name": full_name,
                "role": role,
                "student_id": student_id,
            }).execute()

            if role == "student":
                supabase_admin.table("student_biometrics").insert({
                    "user_id": uid,
                }).execute()

            success_count += 1
            created_users.append({
                "full_name":     full_name,
                "email":         email,
                "student_id":    student_id or "—",
                "role":          role,
                "temp_password": temp_password,
            })

        except Exception as e:
            error_rows.append(f"แถว {i} ({email}): {_friendly_error(e)}")

    # ถ้ามีการสร้าง user สำเร็จ → แสดงหน้า password summary
    if created_users:
        return render_template(
            "admin/import_result.html",
            created_users=created_users,
            error_rows=error_rows,
        )

    for err in error_rows:
        flash(err, "danger")
    return redirect(url_for("admin.import_csv"))


# ─── Beacon Management ────────────────────────────────────────

@admin_bp.route("/beacons")
@login_required
@role_required("admin")
def beacons():
    beacons_data = (
        supabase_admin.table("beacons").select("*").order("room_name").execute().data or []
    )
    return render_template("admin/beacons.html", beacons=beacons_data)


@admin_bp.route("/beacons/add", methods=["POST"])
@login_required
@role_required("admin")
def beacon_add():
    try:
        supabase_admin.table("beacons").insert({
            "uuid": request.form["uuid"].strip(),
            "major": int(request.form["major"]),
            "minor": int(request.form["minor"]),
            "room_name": request.form["room_name"].strip(),
            "rssi_threshold": int(request.form.get("rssi_threshold", -75)),
            "is_active": True,
        }).execute()
        flash("เพิ่ม Beacon สำเร็จ", "success")
    except Exception as e:
        flash(f"เพิ่มไม่สำเร็จ: {_friendly_error(e)}", "danger")
    return redirect(url_for("admin.beacons"))


@admin_bp.route("/beacons/<beacon_id>/edit", methods=["POST"])
@login_required
@role_required("admin")
def beacon_edit(beacon_id):
    try:
        supabase_admin.table("beacons").update({
            "uuid": request.form["uuid"].strip(),
            "major": int(request.form["major"]),
            "minor": int(request.form["minor"]),
            "room_name": request.form["room_name"].strip(),
            "rssi_threshold": int(request.form.get("rssi_threshold", -75)),
            "is_active": request.form.get("is_active") == "on",
        }).eq("id", beacon_id).execute()
        flash("แก้ไข Beacon สำเร็จ", "success")
    except Exception as e:
        flash(f"แก้ไขไม่สำเร็จ: {_friendly_error(e)}", "danger")
    return redirect(url_for("admin.beacons"))


@admin_bp.route("/beacons/<beacon_id>/delete", methods=["POST"])
@login_required
@role_required("admin")
def beacon_delete(beacon_id):
    try:
        supabase_admin.table("beacons").delete().eq("id", beacon_id).execute()
        flash("ลบ Beacon แล้ว", "success")
    except Exception as e:
        flash(f"ลบไม่สำเร็จ: {_friendly_error(e)}", "danger")
    return redirect(url_for("admin.beacons"))


# ─── Session Management ───────────────────────────────────────

@admin_bp.route("/sessions")
@login_required
@role_required("admin")
def sessions():
    sessions_data = (
        supabase_admin.table("sessions")
        .select("*, courses(code, name, users(full_name)), beacons(room_name)")
        .order("start_time", desc=True)
        .execute()
        .data or []
    )
    courses = (
        supabase_admin.table("courses")
        .select("id, code, name, teacher_id, users(id, full_name)")
        .eq("is_active", True)
        .order("code")
        .execute()
        .data or []
    )
    teachers = (
        supabase_admin.table("users")
        .select("id, full_name, email")
        .eq("role", "teacher")
        .order("full_name")
        .execute()
        .data or []
    )
    beacons = (
        supabase_admin.table("beacons")
        .select("id, room_name")
        .eq("is_active", True)
        .order("room_name")
        .execute()
        .data or []
    )
    return render_template("admin/sessions.html",
                           sessions=sessions_data, courses=courses,
                           beacons=beacons, teachers=teachers)


@admin_bp.route("/sessions/create", methods=["POST"])
@login_required
@role_required("admin")
def session_create():
    course_id = request.form.get("course_id")
    beacon_id = request.form.get("beacon_id")

    if not all([course_id, beacon_id]):
        flash("กรุณากรอกข้อมูลให้ครบ", "danger")
        return redirect(url_for("admin.sessions"))

    course = (
        supabase_admin.table("courses").select("code, name")
        .eq("id", course_id).maybe_single().execute().data or {}
    )
    title = f"{course.get('code', '')} — {course.get('name', '')}"

    try:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        supabase_admin.table("sessions").insert({
            "course_id":  course_id,
            "beacon_id":  beacon_id,
            "title":      title,
            "start_time": now,
            "is_open":    False,
        }).execute()
        flash(f"สร้างคาบเรียน '{title}' สำเร็จ — อาจารย์กดเปิดเองได้เลย", "success")
    except Exception as e:
        flash(f"สร้างไม่สำเร็จ: {_friendly_error(e)}", "danger")
    return redirect(url_for("admin.sessions"))


@admin_bp.route("/sessions/<session_id>/delete", methods=["POST"])
@login_required
@role_required("admin")
def session_delete(session_id):
    try:
        supabase_admin.table("sessions").delete().eq("id", session_id).execute()
        flash("ลบคาบเรียนแล้ว", "success")
    except Exception as e:
        flash(f"ลบไม่สำเร็จ: {_friendly_error(e)}", "danger")
    return redirect(url_for("admin.sessions"))


# ─── Biometrics Status ────────────────────────────────────────

# ─── Course Management ────────────────────────────────────────

@admin_bp.route("/courses")
@login_required
@role_required("admin")
def courses():
    courses_data = (
        supabase_admin.table("courses")
        .select("*, users(full_name)")
        .eq("is_active", True)
        .order("code")
        .execute()
        .data or []
    )
    teachers = (
        supabase_admin.table("users")
        .select("id, full_name")
        .eq("role", "teacher")
        .order("full_name")
        .execute()
        .data or []
    )

    # จัดกลุ่มโดย (code, name, teacher_id)
    groups = {}
    for c in courses_data:
        key = (c["code"], c["name"], c["teacher_id"])
        if key not in groups:
            groups[key] = {"code": c["code"], "name": c["name"],
                           "teacher": c.get("users") or {}, "semester": c["semester"],
                           "sections": []}
        groups[key]["sections"].append(c)
    course_groups = list(groups.values())

    return render_template("admin/courses.html", course_groups=course_groups, teachers=teachers)


@admin_bp.route("/courses/add", methods=["POST"])
@login_required
@role_required("admin")
def course_add():
    code       = request.form.get("code", "").strip()
    name       = request.form.get("name", "").strip()
    teacher_id = request.form.get("teacher_id", "").strip()
    semester   = request.form.get("semester", "1").strip()
    section    = request.form.get("section", "").strip() or None

    if not all([code, name, teacher_id, semester]):
        flash("กรุณากรอกข้อมูลให้ครบ", "danger")
        return redirect(url_for("admin.courses"))

    try:
        supabase_admin.table("courses").insert({
            "code":       code,
            "name":       name,
            "teacher_id": teacher_id,
            "semester":   int(semester),
            "section":    section,
            "is_active":  True,
        }).execute()
        flash(f"เพิ่มวิชา {code} สำเร็จ", "success")
    except Exception as e:
        flash(f"เพิ่มวิชาไม่สำเร็จ: {_friendly_error(e)}", "danger")

    return redirect(url_for("admin.courses"))


@admin_bp.route("/courses/<course_id>/add-section", methods=["POST"])
@login_required
@role_required("admin")
def section_add(course_id):
    # ดึง course ต้นแบบ
    parent = (
        supabase_admin.table("courses")
        .select("code, name, teacher_id, semester")
        .eq("id", course_id)
        .maybe_single()
        .execute()
        .data
    )
    if not parent:
        flash("ไม่พบวิชา", "danger")
        return redirect(url_for("admin.courses"))

    # หา section สูงสุดของวิชานี้
    siblings = (
        supabase_admin.table("courses")
        .select("section")
        .eq("code", parent["code"])
        .eq("teacher_id", parent["teacher_id"])
        .eq("semester", parent["semester"])
        .execute()
        .data or []
    )
    max_sec = 1
    for s in siblings:
        try:
            v = int(s["section"] or 0)
            if v > max_sec:
                max_sec = v
        except (ValueError, TypeError):
            pass
    next_sec = str(max_sec + 1)

    try:
        supabase_admin.table("courses").insert({
            "code":       parent["code"],
            "name":       parent["name"],
            "teacher_id": parent["teacher_id"],
            "semester":   parent["semester"],
            "section":    next_sec,
            "is_active":  True,
        }).execute()
        flash(f"เพิ่ม sec {next_sec} สำเร็จ", "success")
    except Exception as e:
        flash(f"เพิ่มไม่สำเร็จ: {_friendly_error(e)}", "danger")

    return redirect(url_for("admin.courses"))


@admin_bp.route("/courses/<course_id>")
@login_required
@role_required("admin")
def course_detail(course_id):
    course = (
        supabase_admin.table("courses")
        .select("*, users(full_name)")
        .eq("id", course_id)
        .maybe_single()
        .execute()
        .data
    )
    if not course:
        flash("ไม่พบวิชานี้", "danger")
        return redirect(url_for("admin.courses"))

    # นักศึกษาที่ enroll แล้ว
    enrolled = (
        supabase_admin.table("course_enrollments")
        .select("id, student_id, users(full_name, email, student_id)")
        .eq("course_id", course_id)
        .execute()
        .data or []
    )
    enrolled_ids = {e["student_id"] for e in enrolled}

    # นักศึกษาทั้งหมดที่ยังไม่ได้ enroll
    all_students = (
        supabase_admin.table("users")
        .select("id, full_name, email, student_id")
        .eq("role", "student")
        .order("full_name")
        .execute()
        .data or []
    )
    available = [s for s in all_students if s["id"] not in enrolled_ids]

    schedules = (
        supabase_admin.table("schedules")
        .select("*")
        .eq("course_id", course_id)
        .order("day_of_week")
        .execute()
        .data or []
    )

    return render_template("admin/course_detail.html",
                           course=course, enrolled=enrolled,
                           available=available, schedules=schedules)


@admin_bp.route("/courses/<course_id>/enroll", methods=["POST"])
@login_required
@role_required("admin")
def course_enroll(course_id):
    student_ids = request.form.getlist("student_ids")
    if not student_ids:
        flash("กรุณาเลือกนักศึกษาอย่างน้อย 1 คน", "danger")
        return redirect(url_for("admin.course_detail", course_id=course_id))

    rows = [{"course_id": course_id, "student_id": sid} for sid in student_ids]
    try:
        supabase_admin.table("course_enrollments").insert(rows).execute()
        flash(f"เพิ่มนักศึกษา {len(rows)} คน สำเร็จ", "success")
    except Exception as e:
        flash(f"เพิ่มไม่สำเร็จ: {_friendly_error(e)}", "danger")

    return redirect(url_for("admin.course_detail", course_id=course_id))


@admin_bp.route("/courses/<course_id>/unenroll/<student_id>", methods=["POST"])
@login_required
@role_required("admin")
def course_unenroll(course_id, student_id):
    try:
        supabase_admin.table("course_enrollments") \
            .delete() \
            .eq("course_id", course_id) \
            .eq("student_id", student_id) \
            .execute()
        flash("ลบนักศึกษาออกจากวิชาสำเร็จ", "success")
    except Exception as e:
        flash(f"ลบไม่สำเร็จ: {_friendly_error(e)}", "danger")

    return redirect(url_for("admin.course_detail", course_id=course_id))


# ─── Schedule Management ──────────────────────────────────────

@admin_bp.route("/courses/<course_id>/import-csv", methods=["POST"])
@login_required
@role_required("admin")
def course_import_csv(course_id):
    file = request.files.get("csv_file")
    if not file or not file.filename.endswith(".csv"):
        flash("กรุณาอัพโหลดไฟล์ .csv", "danger")
        return redirect(url_for("admin.course_detail", course_id=course_id))

    stream = io.StringIO(file.stream.read().decode("utf-8-sig"))
    reader = csv.DictReader(stream)

    success_count = 0
    enrolled_count = 0
    error_rows = []

    for i, row in enumerate(reader, start=2):
        email      = row.get("email", "").strip()
        full_name  = row.get("full_name", "").strip()
        student_id = row.get("student_id", "").strip() or None

        if not email or not full_name:
            error_rows.append(f"แถว {i}: email หรือ full_name ว่าง")
            continue

        # หา user ที่มีอยู่แล้ว
        existing = (
            supabase_admin.table("users")
            .select("id")
            .eq("email", email)
            .maybe_single()
            .execute()
            .data
        )

        if existing:
            uid = existing["id"]
        else:
            try:
                auth_res = supabase_admin.auth.admin.create_user({
                    "email": email,
                    "password": secrets.token_urlsafe(12),
                    "email_confirm": True,
                })
                uid = str(auth_res.user.id)
                supabase_admin.table("users").insert({
                    "id": uid, "email": email,
                    "full_name": full_name, "role": "student",
                    "student_id": student_id,
                }).execute()
                supabase_admin.table("student_biometrics").insert({"user_id": uid}).execute()
                success_count += 1
            except Exception as e:
                error_rows.append(f"แถว {i} ({email}): {_friendly_error(e)}")
                continue

        # enroll เข้า course (ถ้ายังไม่มี)
        dup = (
            supabase_admin.table("course_enrollments")
            .select("id")
            .eq("course_id", course_id)
            .eq("student_id", uid)
            .maybe_single()
            .execute()
            .data
        )
        if not dup:
            try:
                supabase_admin.table("course_enrollments").insert({
                    "course_id": course_id, "student_id": uid,
                }).execute()
                enrolled_count += 1
            except Exception as e:
                error_rows.append(f"แถว {i} ({email}) enroll ไม่สำเร็จ: {_friendly_error(e)}")

    if success_count:
        flash(f"สร้าง account ใหม่ {success_count} คน", "success")
    if enrolled_count:
        flash(f"Enroll เข้า section สำเร็จ {enrolled_count} คน", "success")
    for err in error_rows:
        flash(err, "danger")

    return redirect(url_for("admin.course_detail", course_id=course_id))


@admin_bp.route("/courses/<course_id>/schedules/add", methods=["POST"])
@login_required
@role_required("admin")
def schedule_add(course_id):
    day_of_week = request.form.get("day_of_week")
    start_time  = request.form.get("start_time")
    end_time    = request.form.get("end_time")

    if not all([day_of_week, start_time, end_time]):
        flash("กรุณากรอกข้อมูลให้ครบ", "danger")
        return redirect(url_for("admin.course_detail", course_id=course_id))

    try:
        supabase_admin.table("schedules").insert({
            "course_id":   course_id,
            "day_of_week": int(day_of_week),
            "start_time":  start_time,
            "end_time":    end_time,
        }).execute()
        flash("เพิ่มตารางเรียนสำเร็จ", "success")
    except Exception as e:
        flash(f"เพิ่มไม่สำเร็จ: {_friendly_error(e)}", "danger")

    return redirect(url_for("admin.course_detail", course_id=course_id))


@admin_bp.route("/courses/<course_id>/schedules/<schedule_id>/delete", methods=["POST"])
@login_required
@role_required("admin")
def schedule_delete(course_id, schedule_id):
    try:
        supabase_admin.table("schedules").delete().eq("id", schedule_id).execute()
        flash("ลบตารางเรียนสำเร็จ", "success")
    except Exception as e:
        flash(f"ลบไม่สำเร็จ: {_friendly_error(e)}", "danger")

    return redirect(url_for("admin.course_detail", course_id=course_id))


# ─── Biometrics ───────────────────────────────────────────────

@admin_bp.route("/biometrics")
@login_required
@role_required("admin")
def biometrics():
    res = (
        supabase_admin.table("student_biometrics")
        .select("*, users(full_name, email, student_id)")
        .order("updated_at", desc=True)
        .execute()
    )
    records = res.data or []

    # Fetch latest consent_logs entry per student for PDPA status column
    consent_map = {}
    try:
        user_ids = [r["user_id"] for r in records if r.get("user_id")]
        if user_ids:
            consent_res = (
                supabase_admin.table("consent_logs")
                .select("user_id, consent_given, consent_version, created_at")
                .in_("user_id", user_ids)
                .eq("consent_type", "biometric_enrollment")
                .order("created_at", desc=True)
                .execute()
            )
            # Keep only the latest row per user_id
            for row in (consent_res.data or []):
                uid = row["user_id"]
                if uid not in consent_map:
                    consent_map[uid] = row
    except Exception:
        pass  # Non-fatal — table may not exist yet in older envs

    return render_template("admin/biometrics.html", records=records, consent_map=consent_map)


@admin_bp.route("/api/reset-enrollment/<student_id>", methods=["POST"])
@login_required
@role_required("admin")
@csrf_protect
def api_reset_enrollment(student_id):
    admin_id = session["user_id"]
    try:
        supabase_admin.table("student_biometrics").update({
            "enrollment_attempts":     0,
            "last_enrollment_attempt": None,
        }).eq("user_id", student_id).execute()

        log_audit_event(
            supabase_admin,
            actor_id=admin_id,
            actor_role="admin",
            event_type="reset_enrollment_attempts",
            target_id=student_id,
            new_value="0",
        )
        return jsonify({"status": "ok", "message": "รีเซ็ตจำนวนครั้งลงทะเบียนสำเร็จ"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

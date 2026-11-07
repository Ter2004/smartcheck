import csv
import io
import logging
import secrets
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify, make_response
from app.routes.auth import login_required, role_required
from app import supabase_admin
from app.services.security_service import log_audit_event, csrf_protect, csrf_protect_form
from app.utils import friendly_error as _friendly_error

admin_bp = Blueprint("admin", __name__)
_log = logging.getLogger("smartcheck.admin")


# ─── Dashboard ────────────────────────────────────────────────

@admin_bp.route("/dashboard")
@login_required
@role_required("admin")
def dashboard():
    # Five independent reads side by side (one after another took ~0.6 s, 2026-11-07).
    from concurrent.futures import ThreadPoolExecutor
    from app.services.request_performance import ContextPool
    with ThreadPoolExecutor(max_workers=5, thread_name_prefix="admin-read") as raw:
        pool = ContextPool(raw)
        users = pool.submit(lambda: supabase_admin.table("users").select("id", count="exact").execute())
        students = pool.submit(lambda: supabase_admin.table("users").select("id", count="exact")
                               .eq("role", "student").execute())
        beacons = pool.submit(lambda: supabase_admin.table("beacons").select("id", count="exact").execute())
        enrolled = pool.submit(_enrolled_face_count)
        has_term = pool.submit(_has_term)
        stats = {
            "total_users": users.result().count or 0,
            "students": students.result().count or 0,
            "total_beacons": beacons.result().count or 0,
            "enrolled_biometrics": enrolled.result(),
            "has_term": has_term.result(),
        }
    from datetime import datetime, timedelta, timezone
    today = datetime.now(timezone(timedelta(hours=7))).date()
    return render_template("admin/dashboard.html", stats=stats, today=today)


def _has_term():
    """A term is set (the whole-term export needs it); True before the terms migration."""
    try:
        return bool(supabase_admin.table("terms").select("name").limit(1).execute().data)
    except Exception:
        return True   # table missing: do not nag about a feature that is not installed


def _enrolled_face_count():
    """Completed v2 face enrollments of the active model; legacy rows if the v2 table is absent."""
    from app.services import face_engine
    try:
        res = (supabase_admin.table("face_enrollments").select("id", count="exact")
               .eq("status", "completed").eq("model_version", face_engine.model_version()).execute())
        return res.count or 0
    except Exception:
        res = (supabase_admin.table("student_biometrics").select("*", count="exact")
               .not_.is_("face_embeddings", "null").execute())
        return res.count or 0


def _v2_enrollments(user_ids):
    """student_id -> face_enrollments row (status, model_version, thumbnail, dates); {} before the migration."""
    if not user_ids:
        return {}
    try:
        rows = (supabase_admin.table("face_enrollments")
                .select("student_id, model_version, status, thumbnail_path, completed_at")
                .in_("student_id", user_ids).execute().data or [])
        return {r["student_id"]: r for r in rows}
    except Exception:
        return {}


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
@csrf_protect_form
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
                "id":                   uid,
                "email":                email,
                "full_name":            full_name,
                "role":                 role,
                "student_id":           student_id,
                "must_change_password": True,   # A6: force change on first login
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
    from flask import current_app
    line_col = ", line_hwid" if current_app.config.get("LINE_BEACON_ENABLED") else ""
    beacons_data = (
        supabase_admin.table("beacons")
        .select("id, uuid, major, minor, room_name, "
                "ble_room_code, is_active, ble_secret" + line_col)
        .order("room_name").execute().data or []
    )
    # Rows are embedded as JSON for the edit modal: never send the key itself.
    for b in beacons_data:
        b["has_ble_secret"] = bool(b.pop("ble_secret", None))
    return render_template("admin/beacons.html", beacons=beacons_data)


@admin_bp.route("/beacons/<beacon_id>/ble-secret", methods=["POST"])
@login_required
@role_required("admin")
@csrf_protect_form
def beacon_ble_secret(beacon_id):
    """Generate a new BLE HMAC key and show it once for the room's firmware.

    The previous key stops working at once: the board must be reflashed.
    """
    res = (supabase_admin.table("beacons").select("id, room_name, ble_room_code")
           .eq("id", beacon_id).maybe_single().execute())
    beacon = res.data if res else None
    if not beacon:
        flash("ไม่พบอุปกรณ์", "danger")
        return redirect(url_for("admin.beacons"))
    secret = secrets.token_hex(32)
    try:
        supabase_admin.table("beacons").update({"ble_secret": secret}) \
            .eq("id", beacon_id).execute()
    except Exception as e:
        flash(f"สร้าง secret ไม่สำเร็จ: {_friendly_error(e)}", "danger")
        return redirect(url_for("admin.beacons"))
    log_audit_event(supabase_admin, actor_id=session["user_id"], actor_role="admin",
                    event_type="beacon_secret_rotated", target_id=beacon_id)
    response = make_response(render_template(
        "admin/beacon_secret.html", beacon=beacon, secret=secret))
    response.headers["Cache-Control"] = "no-store"
    return response


def _line_hwid_field():
    """{"line_hwid": ...} from the form when LINE Beacon is on (10 hex chars from LINE OA Manager)."""
    from flask import current_app
    if not current_app.config.get("LINE_BEACON_ENABLED"):
        return {}
    value = (request.form.get("line_hwid") or "").strip().lower() or None
    if value is not None and not (len(value) == 10 and all(c in "0123456789abcdef" for c in value)):
        raise ValueError("LINE HWID ต้องเป็นเลขฐานสิบหก 10 ตัว")
    return {"line_hwid": value}


@admin_bp.route("/beacons/add", methods=["POST"])
@login_required
@role_required("admin")
@csrf_protect_form
def beacon_add():
    try:
        supabase_admin.table("beacons").insert({
            "uuid": request.form["uuid"].strip(),
            "major": int(request.form["major"]),
            "minor": int(request.form["minor"]),
            "room_name": request.form["room_name"].strip(),
            "ble_room_code": (request.form.get("ble_room_code") or "").strip() or None,
            "is_active": True,
            **_line_hwid_field(),
        }).execute()
        flash("เพิ่ม Beacon สำเร็จ", "success")
    except Exception as e:
        flash(f"เพิ่มไม่สำเร็จ: {_friendly_error(e)}", "danger")
    return redirect(url_for("admin.beacons"))


@admin_bp.route("/beacons/<beacon_id>/edit", methods=["POST"])
@login_required
@role_required("admin")
@csrf_protect_form
def beacon_edit(beacon_id):
    try:
        supabase_admin.table("beacons").update({
            "uuid": request.form["uuid"].strip(),
            "major": int(request.form["major"]),
            "minor": int(request.form["minor"]),
            "room_name": request.form["room_name"].strip(),
            "ble_room_code": (request.form.get("ble_room_code") or "").strip() or None,
            "is_active": request.form.get("is_active") == "on",
            **_line_hwid_field(),
        }).eq("id", beacon_id).execute()
        flash("แก้ไข Beacon สำเร็จ", "success")
    except Exception as e:
        flash(f"แก้ไขไม่สำเร็จ: {_friendly_error(e)}", "danger")
    return redirect(url_for("admin.beacons"))


@admin_bp.route("/beacons/<beacon_id>/delete", methods=["POST"])
@login_required
@role_required("admin")
@csrf_protect_form
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
    # Group by course for the collapsible view. sessions_data is newest-first,
    # so the course with the latest session lands on top, and each group's
    # sessions stay newest-first.
    session_groups = {}
    for s in sessions_data:
        group = session_groups.setdefault(s["course_id"], {"course": s.get("courses"), "sessions": []})
        group["sessions"].append(s)
    return render_template("admin/sessions.html",
                           session_groups=list(session_groups.values()), courses=courses,
                           beacons=beacons, teachers=teachers)


@admin_bp.route("/sessions/create", methods=["POST"])
@login_required
@role_required("admin")
@csrf_protect_form
def session_create():
    course_id = request.form.get("course_id")
    beacon_id = request.form.get("beacon_id")

    if not all([course_id, beacon_id]):
        flash("กรุณากรอกข้อมูลให้ครบ", "danger")
        return redirect(url_for("admin.sessions"))

    course = (supabase_admin.table("courses").select("code, name")
              .eq("id", course_id).maybe_single().execute())
    course = (course.data if course else None) or {}
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
@csrf_protect_form
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
@csrf_protect_form
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
@csrf_protect_form
def section_add(course_id):
    # ดึง course ต้นแบบ
    parent = (
        supabase_admin.table("courses")
        .select("code, name, teacher_id, semester")
        .eq("id", course_id)
        .maybe_single()
        .execute()
    )
    parent = parent.data if parent else None
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
    )
    course = course.data if course else None
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
        .select("*, beacons(room_name)")
        .eq("course_id", course_id)
        .order("day_of_week")
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

    return render_template("admin/course_detail.html",
                           course=course, enrolled=enrolled,
                           available=available, schedules=schedules,
                           beacons=beacons)


@admin_bp.route("/courses/<course_id>/enroll", methods=["POST"])
@login_required
@role_required("admin")
@csrf_protect_form
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
@csrf_protect_form
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
@csrf_protect_form
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
        )
        existing = existing.data if existing else None

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
                    "must_change_password": True,   # A6: force change on first login
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
        )
        dup = dup.data if dup else None
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
@csrf_protect_form
def schedule_add(course_id):
    day_of_week = request.form.get("day_of_week")
    start_time  = request.form.get("start_time")
    end_time    = request.form.get("end_time")
    beacon_id   = request.form.get("beacon_id") or None

    if not all([day_of_week, start_time, end_time]):
        flash("กรุณากรอกข้อมูลให้ครบ", "danger")
        return redirect(url_for("admin.course_detail", course_id=course_id))

    try:
        supabase_admin.table("schedules").insert({
            "course_id":   course_id,
            "day_of_week": int(day_of_week),
            "start_time":  start_time,
            "end_time":    end_time,
            "beacon_id":   beacon_id,
        }).execute()
        flash("เพิ่มตารางเรียนสำเร็จ", "success")
    except Exception as e:
        flash(f"เพิ่มไม่สำเร็จ: {_friendly_error(e)}", "danger")

    return redirect(url_for("admin.course_detail", course_id=course_id))


@admin_bp.route("/courses/<course_id>/schedules/<schedule_id>/delete", methods=["POST"])
@login_required
@role_required("admin")
@csrf_protect_form
def schedule_delete(course_id, schedule_id):
    try:
        supabase_admin.table("schedules").delete().eq("id", schedule_id).execute()
        flash("ลบตารางเรียนสำเร็จ", "success")
    except Exception as e:
        flash(f"ลบไม่สำเร็จ: {_friendly_error(e)}", "danger")

    return redirect(url_for("admin.course_detail", course_id=course_id))


# ─── Term (whole-term Excel export) ───────────────────────────

@admin_bp.route("/term", methods=["GET", "POST"])
@login_required
@role_required("admin")
@csrf_protect_form
def term():
    """One term for all courses; the latest start_date is the current term."""
    if request.method == "POST":
        from datetime import date
        name = request.form.get("name", "").strip()
        try:
            start = date.fromisoformat(request.form.get("start_date", ""))
            weeks = int(request.form.get("weeks", "16"))
        except ValueError:
            start, weeks = None, 0
        if not name or start is None or not 1 <= weeks <= 30:
            flash("กรุณากรอกชื่อภาคเรียน วันเปิดเทอม และจำนวนสัปดาห์ (1–30)", "danger")
            return redirect(url_for("admin.term"))
        try:
            supabase_admin.table("terms").insert({
                "name": name, "start_date": start.isoformat(), "weeks": weeks,
            }).execute()
        except Exception as e:
            flash(f"บันทึกไม่สำเร็จ: {_friendly_error(e)}", "danger")
            return redirect(url_for("admin.term"))
        log_audit_event(supabase_admin, actor_id=session["user_id"], actor_role="admin",
                        event_type="term_set", new_value=f"{name} {start.isoformat()} {weeks}w")
        flash(f"ตั้งภาคเรียน {name} สำเร็จ", "success")
        return redirect(url_for("admin.term"))

    try:
        terms = (supabase_admin.table("terms").select("name, start_date, weeks, created_at")
                 .order("start_date", desc=True).limit(5).execute().data or [])
    except Exception:
        terms = None  # migration 20260929_terms.sql not applied yet
    return render_template("admin/term.html", terms=terms)


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

    # Generate signed URLs for face images (1-hour expiry — bucket is private)
    for r in records:
        path = r.get("face_image_url")
        if path and not path.startswith("http"):
            try:
                signed = supabase_admin.storage.from_("face-images") \
                    .create_signed_url(path, expires_in=3600)
                r["face_image_url"] = signed.get("signedURL") or signed.get("signed_url") or None
            except Exception:
                r["face_image_url"] = None

    # Face stack v2 enrollments (the ones check-in uses) and their thumbnails.
    from app.services import face_engine
    active_model = face_engine.model_version()
    v2_map = _v2_enrollments([r["user_id"] for r in records if r.get("user_id")])
    for r in records:
        v2 = v2_map.get(r.get("user_id"))
        r["v2"] = v2
        r["v2_current"] = bool(v2 and v2.get("status") == "completed" and v2.get("model_version") == active_model)
        if v2 and v2.get("thumbnail_path"):
            try:
                signed = supabase_admin.storage.from_("face-images").create_signed_url(
                    v2["thumbnail_path"], expires_in=3600)
                r["face_image_url"] = signed.get("signedURL") or signed.get("signed_url") or r.get("face_image_url")
            except Exception:
                pass

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


@admin_bp.route("/api/withdraw-biometrics/<student_id>", methods=["POST"])
@login_required
@role_required("admin")
@csrf_protect
def api_withdraw_biometrics(student_id):
    """PDPA withdrawal on a student's behalf: consent row, all face data and photos deleted."""
    from app.services.biometric_wipe import WipeError, wipe_on_behalf
    admin_id = session["user_id"]
    target = (supabase_admin.table("users").select("id, role").eq("id", student_id)
              .maybe_single().execute())
    if not (target and target.data and target.data.get("role") == "student"):
        return jsonify({"status": "error", "message": "ไม่พบนักศึกษา"}), 404
    try:
        deleted = wipe_on_behalf(supabase_admin, student_id, admin_id, consent_version="admin",
                                 ip=request.remote_addr or "unknown",
                                 user_agent=request.headers.get("User-Agent", ""))
    except WipeError as error:
        _log.error("withdraw_biometrics failed at %s for %s", error.step, student_id)
        return jsonify({"status": "error",
                        "message": f"ลบข้อมูลไม่สำเร็จ (ขั้น {error.step}) กรุณาลองใหม่"}), 500
    return jsonify({"status": "ok", "message": "ถอนความยินยอมและลบข้อมูลใบหน้าแล้ว", "deleted": deleted})


@admin_bp.route("/api/line-unlink/<user_id>", methods=["POST"])
@login_required
@role_required("admin")
@csrf_protect
def api_line_unlink(user_id):
    """Remove a student's LINE link (new phone or LINE account); pending link codes go too."""
    try:
        supabase_admin.table("users").update({"line_user_id": None}).eq("id", user_id).execute()
        supabase_admin.table("line_link_codes").delete().eq("user_id", user_id).execute()
    except Exception as error:
        _log.error("line_unlink failed: %s", type(error).__name__)
        return jsonify({"status": "error", "message": "ยกเลิกไม่สำเร็จ (ยังไม่ได้รัน migration LINE?) กรุณาลองใหม่"}), 500
    log_audit_event(supabase_admin, actor_id=session["user_id"], actor_role="admin",
                    event_type="line_unlinked", target_id=user_id)
    return jsonify({"status": "ok", "message": "ยกเลิกการเชื่อม LINE แล้ว"})


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
        _log.error("reset_enrollment_attempts failed: %s", type(e).__name__)
        return jsonify({"status": "error", "message": "รีเซ็ตไม่สำเร็จ กรุณาลองใหม่"}), 500

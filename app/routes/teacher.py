import logging
import re
from datetime import date, datetime, timezone, timedelta
from flask import (Blueprint, render_template, request, redirect,
                   url_for, flash, session, jsonify, Response, abort, current_app)
from app.routes.auth import login_required, role_required
from app import supabase_admin
from app import limiter as _limiter
from app.services import attendance_export, class_open, class_push
from app.services.security_service import log_audit_event, csrf_protect, csrf_protect_form
from concurrent.futures import ThreadPoolExecutor
from app.services.request_performance import ContextPool

# Reads that do not depend on each other run side by side: each Supabase query is
# ~120-180 ms from here (ap-south-1), and these pages made them one after another
# (dashboard 4 = 0.5-0.7 s, session page 5 = 0.8-1.0 s; measured 2026-11-07).
_reads = ContextPool(ThreadPoolExecutor(max_workers=8, thread_name_prefix="teacher-read"))

teacher_bp = Blueprint("teacher", __name__)
_log = logging.getLogger("smartcheck.teacher")


# ─── Session History ──────────────────────────────────────────

@teacher_bp.route("/history")
@login_required
@role_required("teacher")
def history():
    teacher_id  = session["user_id"]
    course_filter = request.args.get("course_id", "")
    date_from     = _valid_date(request.args.get("date_from", ""))
    date_to       = _valid_date(request.args.get("date_to", ""))

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
    # A course_id from the query string is only a filter over the teacher's own
    # courses; any other id is dropped rather than queried.
    if course_filter and course_filter not in course_ids:
        flash("ไม่มีสิทธิ์ดูประวัติของวิชานี้", "warning")
        course_filter = ""

    query = (
        supabase_admin.table("sessions")
        .select("*, courses(code, name, section)")
        .in_("course_id", [course_filter] if course_filter else course_ids)
        .order("start_time", desc=True)
    )
    if date_from:
        query = query.gte("start_time", date_from + "T00:00:00+07:00")   # Bangkok dates
    if date_to:
        query = query.lte("start_time", date_to + "T23:59:59+07:00")

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


def _valid_date(text):
    """YYYY-MM-DD from a query string, or "" (a malformed value used to reach the DB)."""
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        return ""


# ─── Dashboard ────────────────────────────────────────────────

@teacher_bp.route("/dashboard")
@login_required
@role_required("teacher")
def dashboard():
    teacher_id = session["user_id"]

    import datetime as _datetime_mod
    _TH = _datetime_mod.timezone(_datetime_mod.timedelta(hours=7))
    _today_th = _datetime_mod.datetime.now(_TH).date()

    # All four reads at once. The session reads are joined to the teacher's active
    # courses, so they no longer wait for the course list (2 round trips -> 1, 2026-11-10).
    beacons_read = _reads.submit(lambda: (
        supabase_admin.table("beacons")
        .select("id, room_name, uuid")
        .eq("is_active", True)
        .order("room_name")
        .execute()
        .data or []
    ))
    courses_read = _reads.submit(lambda: (
        supabase_admin.table("courses")
        .select("*")
        .eq("teacher_id", teacher_id)
        .eq("is_active", True)
        .order("code")
        .execute()
        .data or []
    ))
    # sessions ล่าสุด
    recent_read = _reads.submit(lambda: (
        supabase_admin.table("sessions")
        .select("*, courses!inner(code, name, teacher_id, is_active)")
        .eq("courses.teacher_id", teacher_id)
        .eq("courses.is_active", True)
        .order("start_time", desc=True)
        .limit(30)
        .execute()
        .data or []
    ))
    today_read = _reads.submit(lambda: (
        supabase_admin.table("sessions")
        .select("*, courses!inner(id, code, name, teacher_id, is_active), beacons(room_name)")
        .eq("courses.teacher_id", teacher_id)
        .eq("courses.is_active", True)
        .gte("start_time", f"{_today_th}T00:00:00+07:00")
        .lte("start_time", f"{_today_th}T23:59:59+07:00")
        .order("start_time")
        .execute()
        .data or []
    ))
    courses = courses_read.result()
    recent_sessions = recent_read.result()
    beacons = beacons_read.result()

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

    today_sessions = today_read.result()

    now = _now()
    config = current_app.config
    for s in today_sessions:
        s["on_time_until"] = _iso(class_open.on_time_until(s, config))
        s["can_open"] = class_open.teacher_mode(config) and class_open.open_refusal(s, now, config) is None
        s["opens_from"] = _iso(class_open.opens_from(s, config)) if s.get("start_time") else None

    return render_template(
        "teacher/dashboard.html",
        open_mode_teacher=class_open.teacher_mode(config),
        on_time_default=config["CLASS_ON_TIME_MINUTES"],
        on_time_max=config["CLASS_ON_TIME_MAX_MINUTES"],
        courses=courses,
        recent_sessions=recent_sessions,
        beacons=beacons,
        today_str=_today_th.isoformat(),   # Bangkok date, same as today_sessions
        today_sessions=today_sessions,
    )


# ─── Session View ─────────────────────────────────────────────

@teacher_bp.route("/session/<session_id>")
@login_required
@role_required("teacher")
def session_view(session_id):
    teacher_id = session["user_id"]

    # The attendance and roster reads start with the session read instead of after it
    # (one round trip less, 2026-11-10). Nothing from them is used until the session
    # is known to belong to this teacher.
    attendance_read = _reads.submit(lambda: (
        supabase_admin.table("attendance")
        .select("*, users!attendance_student_id_fkey(full_name, student_id, email)")
        .eq("session_id", session_id)
        .order("check_in_at")
        .execute()
        .data or []
    ))
    # นักศึกษาทั้งหมดในวิชานี้ (the course is found through the session)
    students_read = _reads.submit(lambda: (
        supabase_admin.table("course_enrollments")
        .select("*, users(id, full_name, student_id, email), courses!inner(sessions!inner(id))")
        .eq("courses.sessions.id", session_id)
        .execute()
        .data or []
    ))
    sess = (
        supabase_admin.table("sessions")
        .select("*, courses(id, code, name, teacher_id), beacons(room_name)")
        .eq("id", session_id)
        .maybe_single()
        .execute()
    )
    sess = sess.data if sess else None
    if not sess or not sess.get("courses") or sess["courses"]["teacher_id"] != teacher_id:
        for read in (attendance_read, students_read):
            read.cancel()
        flash("ไม่พบคาบนี้หรือไม่มีสิทธิ์", "danger")
        return redirect(url_for("teacher.dashboard"))

    attendance = attendance_read.result()
    all_students = students_read.result()
    for row in all_students:
        row.pop("courses", None)   # join filter only

    # att_map: student_id → attendance row (สร้างใน Python เพื่อใช้ใน template)
    att_map = {a["student_id"]: a for a in attendance}
    all_students = order_roster(all_students, att_map, bool(sess.get("is_open")))
    sess["on_time_until"] = _iso(class_open.on_time_until(sess, current_app.config)) if sess.get("is_open") else None

    return render_template(
        "teacher/session_view.html",
        sess=sess,
        attendance=attendance,
        all_students=all_students,
        att_map=att_map,
        counts=live_counts(attendance, len(all_students)),
        photos=_enrollment_photos([s["student_id"] for s in all_students]),
    )


CHECKED_STATUSES = ("present", "late", "manual")


def order_roster(enrollments, att_map, is_open):
    """By name; while the class is open, the students not checked in yet come first."""
    by_name = sorted(enrollments, key=lambda e: ((e.get("users") or {}).get("full_name") or ""))
    if not is_open:
        return by_name
    return sorted(by_name, key=lambda e: e.get("student_id") in att_map)   # stable: keeps name order


def live_counts(attendance, total):
    """Head-count for the teacher's live view. "rest" is not checked in yet while the
    class is open, and absent once it has closed (the template words it)."""
    checked = [a for a in attendance if a.get("status") in CHECKED_STATUSES]
    late = sum(1 for a in checked if a.get("status") == "late")
    return {"total": total, "checked": len(checked), "late": late,
            "present": len(checked) - late, "rest": max(0, total - len(checked))}


@teacher_bp.route("/session/<session_id>/live.json")
# Own limit instead of the 200/hour default: the page polls every 10 s (360/hour),
# which hit 429 at request 201, about 33 minutes into a class (measured 2026-11-06).
@_limiter.limit("30 per minute")
@login_required
@role_required("teacher")
def session_live(session_id):
    """Counts and per-student status for the open-class view (polled; no page reload)."""
    # Polled every 10 s: both reads start with the ownership read instead of after it
    # (0.27 s -> one round trip, 2026-11-10). They are only used once the session is
    # known to belong to this teacher; the roster finds its course through the session.
    attendance_read = _reads.submit(lambda: (supabase_admin.table("attendance")
                                             .select("student_id, status, check_in_at")
                                             .eq("session_id", session_id).execute().data or []))
    enrolled_read = _reads.submit(lambda: (supabase_admin.table("course_enrollments")
                                           .select("student_id, courses!inner(sessions!inner(id))")
                                           .eq("courses.sessions.id", session_id)
                                           .execute().data or []))
    sess = _owned_session(session_id, session["user_id"])
    if not sess:
        for read in (attendance_read, enrolled_read):
            read.cancel()
        return jsonify(ok=False, error="ไม่พบคาบนี้หรือไม่มีสิทธิ์"), 403
    attendance, enrolled = attendance_read.result(), enrolled_read.result()
    return jsonify(ok=True, is_open=bool(sess.get("is_open")),
                   counts=live_counts(attendance, len(enrolled)),
                   rows={a["student_id"]: {"status": a.get("status"), "check_in_at": a.get("check_in_at")}
                         for a in attendance})


def _now():
    return datetime.now(timezone.utc)


def _iso(moment):
    return moment.astimezone(timezone.utc).isoformat() if moment else None


@teacher_bp.route("/session/<session_id>/open", methods=["POST"])
@login_required
@role_required("teacher")
@csrf_protect_form
def session_open(session_id):
    """The teacher opens the class; check-ins inside the chosen minutes are on time."""
    config = current_app.config
    if not class_open.teacher_mode(config):
        abort(404)
    teacher_id = session["user_id"]
    res = (supabase_admin.table("sessions").select("*, courses(code, name, teacher_id)")
           .eq("id", session_id).maybe_single().execute())
    sess = res.data if res else None
    if not sess or not sess.get("courses") or sess["courses"]["teacher_id"] != teacher_id:
        flash("ไม่พบคาบนี้หรือไม่มีสิทธิ์", "danger")
        return redirect(url_for("teacher.dashboard"))
    minutes = class_open.parse_minutes(request.form.get("minutes"), config)
    if minutes is None:
        flash(f"ใส่เวลาเช็คชื่อทันเวลาเป็นนาที 1–{config['CLASS_ON_TIME_MAX_MINUTES']}", "warning")
        return redirect(url_for("teacher.dashboard"))
    now = _now()
    refusal = class_open.open_refusal(sess, now, config)
    if refusal:
        flash(refusal, "warning")
        return redirect(url_for("teacher.dashboard"))
    opened = (supabase_admin.table("sessions")
              .update({"is_open": True, "opened_at": now.isoformat(), "opened_by": teacher_id,
                       "present_minutes": minutes})
              .eq("id", session_id).eq("is_open", False).is_("end_time", "null")
              .execute().data or [])
    if not opened:   # the scheduler opened or closed it meanwhile
        flash("คาบนี้เปิดอยู่แล้วหรือจบไปแล้ว", "warning")
        return redirect(url_for("teacher.dashboard"))
    log_audit_event(supabase_admin, teacher_id, "teacher", "session_override",
                    session_id=session_id, new_value="opened", metadata={"present_minutes": minutes})
    class_push.class_opened(session_id, now, minutes, config)
    until = (now + timedelta(minutes=minutes)).astimezone(timezone(timedelta(hours=7)))
    sent = " · ส่งแจ้งเตือนถึงนักศึกษาแล้ว" if class_push.enabled(config) else ""
    flash(f"เปิดคาบ {sess['courses']['code']} แล้ว — เช็คชื่อทันเวลาได้ถึง {until:%H:%M}{sent}", "success")
    # Land on the live head-count, which is what the teacher watches next.
    return redirect(url_for("teacher.session_view", session_id=session_id))


def _owned_session(session_id, teacher_id):
    """The session row if this teacher owns its course, else None."""
    res = (supabase_admin.table("sessions")
           .select("id, title, is_open, course_id, courses(code, name, teacher_id), beacons(room_name)")
           .eq("id", session_id).maybe_single().execute())
    sess = res.data if res else None
    if not sess or not sess.get("courses") or sess["courses"]["teacher_id"] != teacher_id:
        return None
    return sess


def _checkin_url(session_id, code):
    from urllib.parse import quote
    base = (current_app.config.get("PUBLIC_BASE_URL") or request.host_url).rstrip("/")
    return f"{base}/student/checkin?session_id={quote(session_id)}&via=code&c={code}"


@teacher_bp.route("/session/<session_id>/code")
@login_required
@role_required("teacher")
def session_code_screen(session_id):
    """Full-screen room code + QR for students who cannot use BLE or LINE."""
    if not current_app.config.get("CHECKIN_SESSION_CODE"):
        abort(404)
    sess = _owned_session(session_id, session["user_id"])
    if not sess:
        flash("ไม่พบคาบนี้หรือไม่มีสิทธิ์", "danger")
        return redirect(url_for("teacher.dashboard"))
    return render_template("teacher/session_code.html", sess=sess)


@teacher_bp.route("/session/<session_id>/code.json")
# Own limit, like live.json: the code screen asks every 30 s (120/hour), which with
# the teacher's other pages can pass the 200/hour default during one class.
@_limiter.limit("30 per minute")
@login_required
@role_required("teacher")
def session_code_json(session_id):
    if not current_app.config.get("CHECKIN_SESSION_CODE"):
        abort(404)
    sess = _owned_session(session_id, session["user_id"])
    if not sess:
        return jsonify(ok=False, error="ไม่พบคาบนี้หรือไม่มีสิทธิ์"), 403
    if not sess.get("is_open"):
        return jsonify(ok=False, closed=True, error="คาบนี้ปิดการเช็คชื่อแล้ว")
    from app.services import session_code
    code, expires_in = session_code.current(current_app.config["PROXIMITY_RECEIPT_SECRET"], session_id)
    url = _checkin_url(session_id, code)
    response = jsonify(ok=True, code=code, expires_in=expires_in, qr_svg=session_code.qr_svg(url))
    response.headers["Cache-Control"] = "no-store"
    return response


PHOTO_URL_TTL_S = 600


def _enrollment_photos(student_ids):
    """student_id -> short-lived signed URL of the enrollment photo (private bucket).

    Lets the teacher compare the enrolled face with who is in the room. Only
    called after the course-ownership check; failures just hide the photos.
    """
    if not student_ids:
        return {}
    try:
        def v2_rows():   # v2 enrollments (face_enrollments.thumbnail_path); absent before the migration
            try:
                return (supabase_admin.table("face_enrollments").select("student_id, thumbnail_path")
                        .in_("student_id", student_ids).eq("status", "completed").execute().data or [])
            except Exception:
                return []
        v2_read = _reads.submit(v2_rows)   # alongside the legacy table
        rows = (supabase_admin.table("student_biometrics").select("user_id, face_image_url")
                .in_("user_id", student_ids).execute().data or [])
        owners = {r["face_image_url"]: r["user_id"] for r in rows if r.get("face_image_url")}
        owners.update({r["thumbnail_path"]: r["student_id"] for r in v2_read.result() if r.get("thumbnail_path")})
        if not owners:
            return {}
        signed = supabase_admin.storage.from_("face-images").create_signed_urls(
            list(owners), PHOTO_URL_TTL_S)
        return {owners[item["path"]]: item["signedURL"]
                for item in signed if not item.get("error") and item.get("path") in owners}
    except Exception:
        return {}


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
        .select("course_id, courses(teacher_id)")
        .eq("id", session_id)
        .maybe_single()
        .execute()
    )
    sess = sess.data if sess else None
    if not sess or not sess.get("courses") or sess["courses"]["teacher_id"] != teacher_id:
        flash("ไม่มีสิทธิ์", "danger")
        return redirect(url_for("teacher.dashboard"))

    # Owning the session is not enough: the student must belong to its course.
    enrolled = (
        supabase_admin.table("course_enrollments")
        .select("id")
        .eq("course_id", sess["course_id"])
        .eq("student_id", student_id)
        .limit(1)
        .execute()
        .data
    )
    if not enrolled:
        flash("นักศึกษาคนนี้ไม่ได้ลงทะเบียนในวิชานี้", "danger")
        return redirect(url_for("teacher.session_view", session_id=session_id))

    now = datetime.now(timezone.utc).isoformat()

    # upsert attendance row
    existing = (
        supabase_admin.table("attendance")
        .select("id, status")
        .eq("session_id", session_id)
        .eq("student_id", student_id)
        .maybe_single()
        .execute()
    )
    # maybe_single().execute() returns None (not an empty response) on 0 rows.
    existing = existing.data if existing else None

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


# ─── Whole-term Excel export ──────────────────────────────────

@teacher_bp.route("/course/<course_id>/export")
@login_required
@role_required("teacher")
def export_course(course_id):
    """Every scheduled date of the current term for one of the teacher's courses."""
    course = (
        supabase_admin.table("courses")
        .select("id, code, name, section, teacher_id")
        .eq("id", course_id)
        .maybe_single()
        .execute()
    )
    course = course.data if course else None
    if not course or course.get("teacher_id") != session["user_id"]:
        flash("ไม่มีสิทธิ์", "danger")
        return redirect(url_for("teacher.dashboard"))

    terms = (
        supabase_admin.table("terms")
        .select("name, start_date, weeks")
        .order("start_date", desc=True)
        .limit(1)
        .execute()
        .data or []
    )
    if not terms:
        flash("ยังไม่ได้ตั้งวันเปิดเทอม กรุณาแจ้งแอดมิน", "warning")
        return redirect(url_for("teacher.history", course_id=course_id))

    today = datetime.now(attendance_export.TZ).date()
    dates, students = attendance_export.collect(supabase_admin, course_id, terms[0], today)
    content = attendance_export.build_workbook(course, terms[0], dates, students)

    safe = lambda text: re.sub(r"[^A-Za-z0-9_-]+", "-", str(text or "")).strip("-") or "x"
    filename = (f"attendance_{safe(course.get('code'))}"
                f"{'_sec' + safe(course['section']) if course.get('section') else ''}"
                f"_{safe(terms[0]['name'])}.xlsx")
    return Response(
        content,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}",
                 "Cache-Control": "no-store"},
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
        _log.error("reset_enrollment_attempts failed: %s", type(e).__name__)
        return jsonify({"status": "error", "message": "รีเซ็ตไม่สำเร็จ กรุณาลองใหม่"}), 500

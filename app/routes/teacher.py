import csv
import io
from datetime import datetime, timezone, date, timedelta
from flask import (Blueprint, render_template, request, redirect,
                   url_for, flash, session, jsonify, Response)
from app.routes.auth import login_required, role_required
from app import supabase_admin
from app.services.security_service import log_audit_event, csrf_protect, csrf_protect_form
from app.services.session_eligibility import (
    DEFAULT_CHECKIN_DURATION_MINUTES, window_status, parse_start_time, extend_duration_from_now,
)
from app.utils import friendly_error

teacher_bp = Blueprint("teacher", __name__)


def _parse_requested_minutes(raw):
    """Parse a teacher-submitted 'checkin_duration' form field into whole
    minutes for the ACCEPTANCE-WINDOW REQUEST — not the cumulative stored
    value used by extend_duration_from_now() when reopening/refreshing a
    stale session, which can legitimately exceed this range.

    Blank/missing -> (DEFAULT_CHECKIN_DURATION_MINUTES, None): the approved
    default. A non-blank value that isn't a plain positive integer, or is
    outside the existing UI's 1-120 minute range, is a validation error —
    it must not be silently coerced to the default, and must not raise.

    Returns (minutes, error_message); error_message is None on success.
    """
    raw = (raw or "").strip()
    if not raw:
        return DEFAULT_CHECKIN_DURATION_MINUTES, None
    if not raw.isdigit():
        return None, "เวลารับเช็คชื่อไม่ถูกต้อง กรุณากรอกจำนวนนาทีเป็นตัวเลขเต็มบวก 1-120"
    # str.isdigit() is True for some inputs int() still rejects: non-ASCII
    # digit characters like U+00B2 ('²') that int() doesn't accept, and (as
    # of Python 3.11) numeric strings longer than sys.get_int_max_str_digits()
    # (4300 by default), which raise ValueError rather than converting. Both
    # are confirmed reproductions, not hypothetical — caught here so they
    # become the same validation error, not an unhandled 500.
    try:
        minutes = int(raw)
    except ValueError:
        return None, "เวลารับเช็คชื่อไม่ถูกต้อง กรุณากรอกจำนวนนาทีเป็นตัวเลขเต็มบวก 1-120"
    if not (1 <= minutes <= 120):
        return None, "เวลารับเช็คชื่อต้องอยู่ระหว่าง 1-120 นาที"
    return minutes, None


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
        .limit(30)
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

    import datetime as _datetime_mod
    _TH = _datetime_mod.timezone(_datetime_mod.timedelta(hours=7))
    _today_th = _datetime_mod.datetime.now(_TH).date()

    today_sessions = (
        supabase_admin.table("sessions")
        .select("*, courses(id, code, name, teacher_id), beacons(room_name)")
        .in_("course_id", [c["id"] for c in courses] or ["00000000-0000-0000-0000-000000000000"])
        .gte("start_time", f"{_today_th}T00:00:00+07:00")
        .lte("start_time", f"{_today_th}T23:59:59+07:00")
        .order("start_time")
        .execute()
        .data or []
    )

    return render_template(
        "teacher/dashboard.html",
        courses=courses,
        recent_sessions=recent_sessions,
        beacons=beacons,
        today_str=date.today().isoformat(),
        today_sessions=today_sessions,
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

    checkin_duration, duration_error = _parse_requested_minutes(request.form.get("checkin_duration"))
    if duration_error:
        flash(duration_error, "danger")
        return redirect(url_for("teacher.dashboard"))

    try:
        res = supabase_admin.table("sessions").insert({
            "course_id":  course_id,
            "beacon_id":  beacon_id,
            "title":      title,
            "start_time": start_time,
            "end_time":   end_time,
            "is_open":    True,
            "checkin_duration": checkin_duration,
        }).execute()
        new_id = res.data[0]["id"]
        flash(f"สร้างคาบเรียน '{title}' สำเร็จ", "success")
        return redirect(url_for("teacher.session_view", session_id=new_id))
    except Exception as e:
        flash(f"สร้างไม่สำเร็จ: {friendly_error(e)}", "danger")
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

    # "Open" and "accepting check-ins" are different things (see
    # app/services/session_eligibility.py) — the template needs to show
    # which one applies, not just is_open.
    checkin_window = window_status(sess)
    deadline_local = None
    if checkin_window in ("accepting", "expired"):
        from app.services.session_eligibility import checkin_deadline
        from zoneinfo import ZoneInfo
        deadline = checkin_deadline(sess)
        if deadline is not None:
            deadline_local = deadline.astimezone(ZoneInfo("Asia/Bangkok"))

    return render_template(
        "teacher/session_view.html",
        sess=sess,
        attendance=attendance,
        all_students=all_students,
        att_map=att_map,
        checkin_window=checkin_window,
        deadline_local=deadline_local,
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
        .select("is_open, start_time, end_time, course_id, courses(id, teacher_id)")
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

    # Teachers may open a session manually regardless of whether a recurring
    # schedule exists for today, and regardless of whether the current time
    # falls inside that schedule's window — a same-day makeup/extra session
    # is a legitimate manual open, not a mistake to block. Ownership (above)
    # and CSRF (@csrf_protect_form) remain the only gates on this action.
    update_data = {"is_open": new_state}
    if new_state:
        requested_minutes, duration_error = _parse_requested_minutes(request.form.get("checkin_duration"))
        if duration_error:
            flash(duration_error, "danger")
            return redirect(url_for("teacher.session_view", session_id=session_id))
        update_data["end_time"] = None

        # Every application write path this codebase was inspected for
        # (scheduler auto-create, admin create, teacher.session_create,
        # scripts/seed_load_test.py) leaves end_time null at creation and
        # teacher.session_create never creates is_open=False — so the only
        # application code path that produces is_open=False with a non-null
        # end_time is this function's own close branch below. Within that
        # inspected set, end_time IS NOT NULL is reliable evidence of a
        # reopen. It is not proof against a row a direct manual DB edit put
        # into an equivalent state outside the application.
        is_reopen = sess.get("end_time") is not None
        if is_reopen:
            # Reopening must not overwrite the original class start — it
            # drives late-arrival classification and historical/weekly
            # grouping elsewhere (see the session-eligibility start_time
            # trace). Extend checkin_duration instead, the same way
            # session_set_window does, via the one shared function.
            original_start = parse_start_time(sess)
            if original_start is None:
                flash("ไม่สามารถเปิดคาบซ้ำได้ — ไม่พบเวลาเริ่มคาบเดิมที่ถูกต้อง", "danger")
                return redirect(url_for("teacher.session_view", session_id=session_id))
            checkin_duration, starts_in_future = extend_duration_from_now(
                original_start, requested_minutes, now_dt)
            update_data["checkin_duration"] = checkin_duration
            message = (f"เปิดคาบซ้ำแล้ว (จะรับเช็คชื่อ {requested_minutes} นาทีเมื่อถึงเวลาคาบ)"
                       if starts_in_future else
                       f"เปิดคาบซ้ำแล้ว (รับ {requested_minutes} นาทีจากนี้)")
        else:
            # First opening of a session that has never been closed before
            # (auto-created ahead of time, or admin-created with a
            # placeholder start_time) — now() is genuinely the class's
            # actual start.
            update_data["start_time"] = now_dt.isoformat()
            update_data["checkin_duration"] = requested_minutes
            message = f"เปิดการเช็คชื่อแล้ว (รับ {requested_minutes} นาที)"
    else:
        update_data["end_time"] = now_dt.isoformat()
        message = "ปิดการเช็คชื่อแล้ว"

    supabase_admin.table("sessions").update(update_data).eq("id", session_id).execute()
    flash(message, "success")
    return redirect(url_for("teacher.session_view", session_id=session_id))


# ─── Set/refresh check-in acceptance window (session stays open) ──
#
# Distinct from session_toggle: a session with is_open=true but a null
# checkin_duration is "open" but not accepting check-ins (see
# app/services/session_eligibility.py). This lets a teacher establish (or
# reset) that window without closing the session — closing would stamp
# end_time and read as "session ended" in history, which is wrong for a
# still-running class that just needs its window (re)established.

@teacher_bp.route("/session/<session_id>/set-window", methods=["POST"])
@login_required
@role_required("teacher")
@csrf_protect_form
def session_set_window(session_id):
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
    if not sess.get("is_open"):
        flash("คาบนี้ปิดอยู่ — กรุณาเปิดคาบก่อน", "danger")
        return redirect(url_for("teacher.session_view", session_id=session_id))

    requested_minutes, duration_error = _parse_requested_minutes(request.form.get("checkin_duration"))
    if duration_error:
        flash(duration_error, "danger")
        return redirect(url_for("teacher.session_view", session_id=session_id))

    # start_time is the original class start — it also drives late-arrival
    # classification (api_checkin.py), calendar-day/week grouping (teacher
    # dashboard/history, student weekly table), and the scheduler's
    # auto-close lookup. It must not be reset here (see the trace in the
    # session-eligibility work). Instead, extend checkin_duration so the
    # deadline (start_time + checkin_duration, computed in
    # session_eligibility.checkin_deadline) lands at now + requested_minutes
    # — or, if the class hasn't started yet, at original_start +
    # requested_minutes — without touching start_time at all. This means
    # checkin_duration's raw stored value stops reading as "minutes from
    # class start" once this has been used on a stale session — the teacher
    # UI shows the derived accepting/expired status, not this raw number,
    # specifically because of that.
    original_start = parse_start_time(sess)
    if original_start is None:
        flash("ไม่สามารถตั้งเวลาได้ — ไม่พบเวลาเริ่มคาบเดิมที่ถูกต้อง", "danger")
        return redirect(url_for("teacher.session_view", session_id=session_id))

    now_dt = datetime.now(timezone.utc)
    checkin_duration, starts_in_future = extend_duration_from_now(
        original_start, requested_minutes, now_dt)

    supabase_admin.table("sessions").update({
        "checkin_duration": checkin_duration,
    }).eq("id", session_id).execute()
    message = (f"ตั้งเวลารับเช็คชื่อแล้ว (จะรับเช็คชื่อ {requested_minutes} นาทีเมื่อถึงเวลาคาบ)"
               if starts_in_future else
               f"ตั้งเวลารับเช็คชื่อใหม่แล้ว (รับอีก {requested_minutes} นาทีจากนี้)")
    flash(message, "success")
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

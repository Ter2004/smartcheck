import logging
import json
import time
import cv2
import numpy as np
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify, current_app, g, Response
from app.services.request_performance import execute_status_query
from app.routes.auth import login_required, role_required
from app import supabase_admin
from app.services.security_service import (
    create_device_token, csrf_protect,
    compute_embedding_integrity_hash,
)
from app.services.face_service import (
    SELF_VERIFY_THRESHOLD,
    DUPLICATE_THRESHOLD,
    DUPLICATE_GRAY_ZONE,
    CONTINUITY_THRESHOLD,
    cosine_similarity,
)
from app import limiter as _limiter

student_bp = Blueprint("student", __name__)


class EnrollmentStatusUnavailable(Exception):
    """The current enrollment state could not be established."""


@student_bp.errorhandler(EnrollmentStatusUnavailable)
def enrollment_status_unavailable(error):
    # Standalone response: rendering the shared layout would query status again.
    return Response(
        '<!doctype html><html lang="th"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>ตรวจสอบสถานะไม่สำเร็จ</title><body>'
        '<h1>ตรวจสอบสถานะลงทะเบียนไม่สำเร็จ</h1>'
        '<p>กรุณาลองใหม่อีกครั้งในภายหลัง</p><a href="">ลองใหม่</a>'
        '</body></html>',
        status=503, mimetype="text/html", headers={"Cache-Control": "no-store"},
    )


def _enrollment_status():
    try:
        return _load_enrollment_status()
    except Exception as error:
        raise EnrollmentStatusUnavailable() from error


def _load_enrollment_status():
    """Share a fresh status read within this request, never across sessions."""
    user_id = session["user_id"]
    cache = g.setdefault("enrollment_status", {})
    use_cache = current_app.config.get("ENROLLMENT_STATUS_CACHE", True)
    if use_cache and user_id in cache:
        return cache[user_id]
    if current_app.config.get("ENROLLMENT_STATUS_RPC", False):
        result = execute_status_query(supabase_admin.rpc(
            "get_enrollment_status", {"p_user_id": user_id}))
        rows = result.data if result else None
        status = rows[0] if rows else {"is_enrolled": False, "baseline_ear": None}
        if type(status.get("is_enrolled")) is not bool:
            raise ValueError("Invalid enrollment status response")
    else:
        result = execute_status_query(supabase_admin.table("student_biometrics")
            .select("face_embeddings, baseline_ear, consent_given")
            .eq("user_id", user_id).maybe_single())
        bio = result.data if result else None
        status = {
            "is_enrolled": bool(bio and bio.get("face_embeddings") and bio.get("consent_given")),
            "baseline_ear": bio.get("baseline_ear") if bio else None,
        }
    if use_cache:
        cache[user_id] = status
    return status


@student_bp.before_request
def prevent_completed_enrollment():
    """Do not allow completed students to restart or overwrite enrollment."""
    if request.endpoint not in {
        "student.enroll_face", "student.record_consent", "student.api_enroll",
        "student.api_self_verify", "student.api_spoof_check", "student.api_reset_liveness",
        "student.api_liveness_challenge", "student.api_liveness_verify",
    } or session.get("user_role") != "student" or not session.get("user_id"):
        return None
    try:
        status = _enrollment_status()
    except Exception:
        if request.endpoint == "student.enroll_face":
            return enrollment_status_unavailable(None)
        return jsonify(status="error", message="ตรวจสอบสถานะลงทะเบียนไม่สำเร็จ กรุณาลองใหม่"), 503
    if status["is_enrolled"]:
        if request.endpoint == "student.enroll_face":
            flash("คุณลงทะเบียนใบหน้าสำเร็จแล้ว", "info")
            return redirect(url_for("student.dashboard"))
        return jsonify(status="already_enrolled", message="คุณลงทะเบียนใบหน้าสำเร็จแล้ว ไม่สามารถลงทะเบียนซ้ำได้"), 409


@student_bp.context_processor
def enrollment_navigation():
    if session.get("user_role") != "student" or not session.get("user_id"):
        return {}
    try:
        return {"can_enroll_face": not _enrollment_status()["is_enrolled"]}
    except Exception:
        return {"can_enroll_face": False}

# ─── Enrollment thresholds ────────────────────────────────────────────────────
MAX_RETRY             = 3      # max outlier-retry rounds (server-enforced — A4)
LIVENESS_MAX_AGE_S    = 15 * 60  # a passed head-turn challenge is valid this long


def _liveness_verified():
    """True if this session passed the server-verified head-turn challenge recently."""
    verified_at = session.get("liveness_verified_at")
    return bool(verified_at) and 0 <= time.time() - verified_at <= LIVENESS_MAX_AGE_S

# ─── Audit logger (D1) ───────────────────────────────────────────────────────
_audit = logging.getLogger("smartcheck.enrollment")


import re as _re
_IP_RE = _re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')

def _safe_ip():
    """M13: Return client IP — validate X-Forwarded-For to prevent log spoofing."""
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        first = xff.split(",")[0].strip()
        if _IP_RE.match(first):
            return first
    return request.remote_addr or "unknown"


def _log(student_id, step, result, details=""):
    """D1: structured audit log for every enrollment pipeline step."""
    ip = _safe_ip()
    ua = request.headers.get("User-Agent", "unknown")[:80]
    _audit.info(f"student={student_id} step={step} result={result} details={details} ip={ip} ua={ua}")


# ─────────────────────────────────────────────────────────────────────────────
# Page routes
# ─────────────────────────────────────────────────────────────────────────────

@student_bp.route("/dashboard")
@login_required
@role_required("student")
def dashboard():
    enrolled = _enrollment_status()["is_enrolled"]
    return render_template("student/dashboard.html", enrolled=enrolled)


@student_bp.route("/enroll")
@login_required
@role_required("student")
def enroll_face():
    already_enrolled = _enrollment_status()["is_enrolled"]
    # Reset server-side retry counters when page is (re)loaded
    session.pop("enroll_retry", None)
    session.pop("consent_given_at", None)
    return render_template("student/enroll_face.html",
                           already_enrolled=already_enrolled,
                           flow_mode=current_app.config["ENROLL_FLOW_MODE"])


@student_bp.route("/checkin")
@login_required
@role_required("student")
def checkin():
    import re
    user_id = session["user_id"]

    status = _enrollment_status()
    if not status["is_enrolled"]:
        flash("กรุณาลงทะเบียนใบหน้าก่อน", "warning")
        return redirect(url_for("student.enroll_face"))

    baseline_ear = status.get("baseline_ear") or 0.25

    import datetime as _dt_mod
    _TH_S = _dt_mod.timezone(_dt_mod.timedelta(hours=7))
    _now_th = _dt_mod.datetime.now(_TH_S)
    _today_start = _now_th.replace(hour=0, minute=0, second=0, microsecond=0)
    _tomorrow_start = _today_start + _dt_mod.timedelta(days=1)

    open_sessions = (
        supabase_admin.table("sessions")
        .select("*, courses(id, code, name), beacons(uuid, rssi_threshold, room_name)")
        .eq("is_open", True)
        .gte("start_time", _today_start.isoformat())
        .lt("start_time", _tomorrow_start.isoformat())
        .execute()
        .data or []
    )

    enrolled_course_ids = {
        row["course_id"]
        for row in (
            supabase_admin.table("course_enrollments")
            .select("course_id")
            .eq("student_id", user_id)
            .execute()
            .data or []
        )
    }

    available_sessions = [s for s in open_sessions if s["course_id"] in enrolled_course_ids]
    selected_id = request.args.get("session_id")
    session_data = next((s for s in available_sessions if s["id"] == selected_id), None)
    if selected_id and session_data is None:
        flash("คาบนี้ไม่เปิดเช็คชื่อหรือคุณไม่ได้ลงทะเบียน กรุณาเลือกคาบใหม่", "warning")
        return redirect(url_for("student.checkin"))

    already_checked = False
    if session_data:
        res = (
            supabase_admin.table("attendance")
            .select("id")
            .eq("session_id", session_data["id"])
            .eq("student_id", user_id)
            .limit(1)
            .execute()
        )
        if res and res.data:
            already_checked = True

    _week_start = (_now_th.date() - _dt_mod.timedelta(days=_now_th.weekday())).isoformat()
    _week_end   = (_now_th.date() + _dt_mod.timedelta(days=6 - _now_th.weekday())).isoformat()

    week_schedules = (
        supabase_admin.table("schedules")
        .select("*, courses(id, code, name), beacons(room_name)")
        .in_("course_id", list(enrolled_course_ids))
        .order("day_of_week")
        .execute()
        .data or []
    ) if enrolled_course_ids else []

    week_sessions = (
        supabase_admin.table("sessions")
        .select("id, course_id, is_open, start_time, end_time")
        .in_("course_id", list(enrolled_course_ids))
        .gte("start_time", f"{_week_start}T00:00:00+07:00")
        .lte("start_time", f"{_week_end}T23:59:59+07:00")
        .execute()
        .data or []
    ) if enrolled_course_ids else []

    week_session_map = {s["course_id"]: s for s in week_sessions}

    checked_session_ids = set()
    if week_sessions:
        _wids = [s["id"] for s in week_sessions]
        _wa = (
            supabase_admin.table("attendance")
            .select("session_id")
            .eq("student_id", user_id)
            .in_("session_id", _wids)
            .execute()
            .data or []
        )
        checked_session_ids = {a["session_id"] for a in _wa}

    ua = request.headers.get("User-Agent", "")
    ios_warning = bool(re.search(r"iPhone|iPad|iPod", ua, re.I))

    if not selected_id:
        return render_template(
            "student/checkin_select.html", available_sessions=available_sessions,
            checked_session_ids=checked_session_ids, week_schedules=week_schedules,
        )

    return render_template(
        "student/checkin.html",
        session_data=session_data,
        baseline_ear=baseline_ear,
        ios_warning=ios_warning,
        already_checked=already_checked,
        week_schedules=[],
        week_session_map=week_session_map,
        checked_session_ids=checked_session_ids,
        today_dow=_now_th.weekday(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# A5: Consent endpoint — record PDPA consent server-side before enrollment
# ─────────────────────────────────────────────────────────────────────────────

@student_bp.route("/api/consent", methods=["POST"])
@login_required
@role_required("student")
@_limiter.limit("10 per minute")
@csrf_protect
def record_consent():
    from datetime import datetime, timezone
    user_id = session["user_id"]
    now_iso = datetime.now(timezone.utc).isoformat()
    session["consent_given_at"] = now_iso
    session["consent_ip"]       = request.remote_addr
    _log(user_id, "consent", "recorded", f"at={now_iso}")

    # PDPA: persist consent record to DB (audit trail — never deleted)
    try:
        supabase_admin.table("consent_logs").insert({
            "user_id":          user_id,
            "consent_type":     "biometric_enrollment",
            "consent_given":    True,
            "consent_version":  "1.1",
            "ip_address":       _safe_ip(),
            "user_agent":       request.headers.get("User-Agent", "")[:500],
        }).execute()
    except Exception as consent_err:
        _log(user_id, "consent_db", "error", str(consent_err)[:80])
        # Non-fatal — session flag is backup; enrollment still allowed

    return jsonify({"status": "ok"})


# ─────────────────────────────────────────────────────────────────────────────
# Enrollment API
# ─────────────────────────────────────────────────────────────────────────────

@student_bp.route("/api/enroll", methods=["POST"])
@login_required
@role_required("student")
@_limiter.limit("5 per minute")
@csrf_protect
def api_enroll():
    """
    Receive 5 frontal frames, run full anti-spoof pipeline, check consistency, save pending.

    Pipeline order (A1) — see docs/review/04-student.md "Enrollment Pipeline Order":
      1a. Consent check — session (consent_given_at)
      1b. Consent check — DB (latest consent_logs row)
      1c. Input: frame count == 5
      1d. Input: baseline_ear range + type
      1e. Retry limit: session["enroll_retry"] >= MAX_RETRY
      2.  DB attempt limit: atomic_enroll_attempt RPC (max 5/24h)
      3.  Zero-trust frame validation: server_validate_frame x5
      4.  Pre-duplicate check (vs session["liveness_embeddings"])
      5.  Decode all 5 frames (_decode_image)
      6.  Moiré FFT: detect_screen_moire, all 5 frames
      7.  Screen Texture Detection: detect_screen_texture, all 5 frames (>=2/5)
      8.  Temporal variance: detect_static_image
      9.  EAR std (client-reported, logged only — not blocking)
      10. MiniFASNet Anti-Spoof: combined_spoof_score x5 (>=4/5 must pass)
      11. Extract FaceNet512 embeddings (all 5 frames, face detection happens here)
      12. Embedding consistency check (B1: handles multi-outlier)
      13. Face continuity vs session["liveness_embeddings"] (CONTINUITY_THRESHOLD)
      14. Duplicate face check (A6: gray-zone logging)
      15. Save pending to DB + profile image upload

    Response statuses:
      pending_verify   : all 5 consistent, saved as pending (consent_given=False)
      need_more        : single outlier or detect failure — frontend re-shoots that frame
      restart_capture  : ≥2 outliers — frontend resets Step 4 entirely
      error            : validation failure, max retries, duplicate
      spoof_detected   : any anti-spoof layer triggered
    """
    from app.services.face_service import (
        extract_embedding, check_embedding_consistency,
        max_similarity_multi, detect_screen_moire, detect_screen_texture,
        detect_static_image, combined_spoof_score, is_system_failure,
        _decode_image, server_validate_frame,
    )
    # M2: DUPLICATE_THRESHOLD and DUPLICATE_GRAY_ZONE defined at module level above — use those

    user_id = session["user_id"]
    data    = request.get_json()

    # ── 1. Validate ───────────────────────────────────────────────────────────
    # A5: consent must be recorded server-side via /api/consent before enrolling
    if not session.get("consent_given_at"):
        return jsonify({"status": "error", "message": "ต้องยินยอม PDPA ก่อน"}), 400

    # PDPA: verify consent exists in DB (session alone is insufficient — 1-hour expiry)
    # Bug fix: do NOT filter by consent_given=True — that silently skips withdrawn rows
    # and lets a user who withdrew consent re-enroll using a stale True row.
    # Instead, fetch the latest row unconditionally and check its value in Python.
    try:
        consent_check = (
            supabase_admin.table("consent_logs")
            .select("consent_given")
            .eq("user_id", user_id)
            .eq("consent_type", "biometric_enrollment")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if not consent_check.data:
            _log(user_id, "enroll_consent", "no_db_record", "consent_logs empty")
            return jsonify({"status": "error", "message": "กรุณายอมรับข้อตกลงก่อนลงทะเบียน"}), 400
        # Explicit withdrawal check: if latest record is False → block enrollment
        if not consent_check.data[0].get("consent_given"):
            _log(user_id, "enroll_consent", "withdrawn", "latest consent_given=False")
            return jsonify({"status": "error",
                            "message": "คุณได้ถอนความยินยอมไปแล้ว กรุณายอมรับข้อตกลงใหม่ก่อนลงทะเบียน"}), 403
    except Exception as consent_err:
        _log(user_id, "enroll_consent", "db_error", str(consent_err)[:80])
        # Fail-close: DB error → deny enrollment to prevent bypassing consent requirement
        return jsonify({"status": "error", "message": "ไม่สามารถตรวจสอบความยินยอมได้ กรุณาลองใหม่"}), 500

    if not data:
        return jsonify({"status": "error", "message": "ไม่พบข้อมูล"}), 400

    face_images  = data.get("face_images", [])
    # L2: validate baseline_ear before float() — malformed value causes unhandled ValueError
    _raw_ear = data.get("baseline_ear")
    if _raw_ear is not None:
        try:
            baseline_ear = float(_raw_ear)
            if not (0.0 < baseline_ear < 1.0):
                baseline_ear = None  # out of plausible EAR range — ignore
        except (ValueError, TypeError):
            return jsonify({"status": "error", "message": "ข้อมูล baseline_ear ไม่ถูกต้อง"}), 400
    else:
        baseline_ear = None
    # A4: retry_count is server-enforced — ignore client-sent value
    retry_count = session.get("enroll_retry", 0)

    if len(face_images) != 5:
        return jsonify({"status": "error",
                        "message": f"ต้องการรูป 5 รูป (ได้รับ {len(face_images)})"}), 400

    # Server-verified head-turn challenge must pass first; browser checks alone
    # can be skipped by calling this API directly. Checked before the DB attempt
    # counter so an unverified request does not use up an attempt.
    if not _liveness_verified():
        _log(user_id, "enroll_liveness", "blocked", "no recent server-verified challenge")
        return jsonify({"status": "error",
                        "message": "กรุณาทำ Liveness Check ก่อน — กรุณาเริ่มใหม่"}), 400

    if retry_count >= MAX_RETRY:
        session.pop("enroll_retry", None)
        _log(user_id, "enroll_validate", "blocked", f"retry_count={retry_count}")
        return jsonify({
            "status":  "error",
            "message": f"ลองใหม่เกิน {MAX_RETRY} รอบ — กรุณาเริ่มลงทะเบียนใหม่",
        }), 400

    # ── 1b. DB-level enrollment attempt check ────────────────────────────────────
    DB_MAX_ATTEMPTS = 5
    try:
        rpc_res = supabase_admin.rpc(
            "atomic_enroll_attempt",
            {"p_user_id": user_id, "p_max": DB_MAX_ATTEMPTS, "p_window_h": 24},
        ).execute()
        if not rpc_res.data:
            raise RuntimeError("atomic_enroll_attempt returned no rows")
        row = rpc_res.data[0]
        db_attempts_now = row["current_attempts"]
        allowed         = row["allowed"]
        _log(user_id, "enroll_attempt", "counted",
             f"db_attempts={db_attempts_now}/{DB_MAX_ATTEMPTS}")
        if not allowed:
            _log(user_id, "enroll_attempt", "db_blocked",
                 f"db_attempts={db_attempts_now}")
            return jsonify({
                "status":      "blocked",
                "message":     "ลงทะเบียนเกินจำนวนครั้งที่กำหนด กรุณาติดต่ออาจารย์",
                "db_attempts": db_attempts_now,
            }), 403
    except Exception as db_err:
        _log(user_id, "enroll_attempt", "db_error_blocked", str(db_err)[:80])
        return jsonify({
            "status":  "error",
            "message": "ไม่สามารถตรวจสอบสิทธิ์ได้ชั่วคราว กรุณาลองใหม่อีกครั้ง",
        }), 500

    # ── 2. Zero-trust frame validation (Sprint 2A) — before any DeepFace call ──
    for idx, img in enumerate(face_images):
        v = server_validate_frame(img)
        if not v["valid"]:
            _log(user_id, "frame_validate", "fail",
                 f"frame={idx+1} reason={v['reason']} meta={v['metadata']}")
            return jsonify({
                "status":       "error",
                "failed_frame": idx + 1,
                "reason":       v["reason"],
                "message":      f"รูปที่ {idx+1} ไม่ถูกต้อง ({v['reason']}) — กรุณาถ่ายใหม่",
            }), 400

    # ── 2b. Pre-Duplicate Check (ใช้ liveness_embeddings จาก session) ───────────
    # รันก่อน spoof checks ทั้งหมด เพื่อให้ผู้ใช้เห็น "duplicate" แทน "spoof_detected"
    # เมื่อส่งใบหน้าที่ลงทะเบียนไปแล้ว liveness_embeddings ถูก extract ไว้ใน session
    # ระหว่าง liveness check (FaceNet512 512-D) — ใช้ซ้ำได้โดยไม่ต้องรัน DeepFace เพิ่ม
    _pre_dup_embs = session.get("liveness_embeddings", [])
    if _pre_dup_embs:
        try:
            _pre_dup_offset = 0
            _pre_dup_found  = False
            while not _pre_dup_found:
                _pre_batch = (
                    supabase_admin.table("student_biometrics")
                    .select("user_id, face_embeddings")
                    .not_.is_("face_embeddings", "null")
                    .neq("user_id", user_id)
                    .range(_pre_dup_offset, _pre_dup_offset + 49)
                    .execute()
                    .data or []
                )
                if not _pre_batch:
                    break
                for _bio in _pre_batch:
                    _stored = _bio.get("face_embeddings") or []
                    if not _stored:
                        continue
                    for _emb in _pre_dup_embs:
                        _sim = max_similarity_multi(_emb, _stored)
                        if _sim >= DUPLICATE_THRESHOLD:
                            _log(user_id, "pre_duplicate_check", "blocked",
                                 f"sim={_sim:.4f} other_user={_bio['user_id']}")
                            _pre_dup_found = True
                            break
                    if _pre_dup_found:
                        break
                if _pre_dup_found or len(_pre_batch) < 50:
                    break
                _pre_dup_offset += 50
            if _pre_dup_found:
                return jsonify({
                    "status":  "duplicate",
                    "message": "ใบหน้านี้ถูกลงทะเบียนในระบบแล้ว",
                }), 400
            _log(user_id, "pre_duplicate_check", "pass",
                 f"liveness_embs={len(_pre_dup_embs)}")
        except Exception as _pre_dup_err:
            # Non-fatal — ถ้า pre-check crash ให้ดำเนินการต่อ (definitive check ที่ step 12 จะรับ)
            _log(user_id, "pre_duplicate_check", "error", str(_pre_dup_err)[:80])

    # ── 3. Decode all frames once (shared across checks below) ───────────────
    try:
        raw_frames = [_decode_image(img) for img in face_images]
    except Exception as e:
        return jsonify({"status": "error", "message": "ไม่สามารถอ่านรูปภาพได้"}), 400

    # ── 4. Moiré FFT (all 5 frames) — computed + logged only, does not reject (FRR-1/F-15/Q-15) ──
    moire_checked = False
    try:
        moire = detect_screen_moire(raw_frames)
        moire_checked = True
        _log(user_id, "moire_fft", "screen" if moire["is_screen"] else "pass",
             f"avg_score={moire['avg_score']}")
    except Exception as e:
        _log(user_id, "moire_fft", "error_log_only",
             f"error={str(e)[:80]} decision=log_only")

    # ── 5. Screen Texture Detection (A3, all 5 frames) — computed + logged only, does not reject ──
    try:
        screen_count = sum(1 for f in raw_frames if detect_screen_texture(f, min_peaks=30))
        _log(user_id, "screen_texture", "screen" if screen_count >= 2 else "pass",
             f"screen_frames={screen_count}/5")
    except Exception as e:
        _log(user_id, "screen_texture", "error_log_only",
             f"error={str(e)[:80]} decision=log_only")

    # ── 5b. Temporal variance — detect static photo / phone screen ───────────
    try:
        temporal = detect_static_image(raw_frames)
        _log(user_id, "temporal_var",
             "static_log_only" if temporal["is_static"] else "pass_log_only",
             f"variance={temporal['temporal_variance']} "
             "reference_threshold=4.0 decision=log_only")
    except Exception as e:
        _log(user_id, "temporal_var", "error_log_only",
             f"error={str(e)[:80]} decision=log_only")

    # ── 5c. EAR temporal variance — client-reported (defence-in-depth) ───────
    # Blocking disabled: passive 5-frame (1.25s) capture does not guarantee blink,
    # causing high FRR for real users. EAR is logged only for audit/future tuning.
    _raw_ear_std = data.get("ear_std")
    try:
        ear_std = float(_raw_ear_std) if _raw_ear_std not in (None, "") else None
        if ear_std is not None and not np.isfinite(ear_std):
            ear_std = None
    except (ValueError, TypeError):
        ear_std = None
    if ear_std is None:
        _log(user_id, "ear_std", "absent",
             "ear_std=absent (blocking disabled for passive capture)")
    else:
        _log(user_id, "ear_std", "low_but_pass" if ear_std < 0.003 else "pass",
             f"ear_std={ear_std:.5f} (blocking disabled for passive capture)")

    # ── 6. MiniFASNet Anti-Spoof (A2: all 5 frames) — fail-close ─────────────
    # Require MIN_SPOOF_PASS frames to pass; exception = fail, not skip
    # Calls combined_spoof_score() directly (not the check_anti_spoof() bool wrapper)
    # so is_system_failure() can tell "Fasnet crashed" apart from "Fasnet said spoof"
    # — F-16, docs/review/10-moire-frr-investigation.md §16. raw_frames[idx] is
    # already-decoded (step 3 above) — no need to re-decode face_images[idx].
    MIN_SPOOF_PASS = 4   # at least 4/5 frames must clear MiniFASNet
    spoof_pass_count = 0
    first_spoof_frame = None
    any_system_failure = False
    for idx in range(len(raw_frames)):
        try:
            spoof_result = combined_spoof_score(raw_frames[idx])
            is_real = spoof_result["is_real"]
            _log(user_id, "minifasnet", "pass" if is_real else "spoof", f"frame={idx+1}")
            if not is_real:
                if is_system_failure(spoof_result):
                    any_system_failure = True
                if first_spoof_frame is None:
                    first_spoof_frame = idx + 1
                # ไม่หยุดทันที — นับต่อเพื่อ fail_close threshold
            else:
                spoof_pass_count += 1
        except Exception as e:
            _log(user_id, "minifasnet", "exception", f"frame={idx+1} {str(e)[:60]}")
            any_system_failure = True
            # Exception counts as fail — do NOT skip

    if spoof_pass_count < MIN_SPOOF_PASS:
        _log(user_id, "minifasnet", "fail_close",
             f"only {spoof_pass_count}/{len(raw_frames)} passed first_spoof={first_spoof_frame} "
             f"system_failure={any_system_failure}")
        if any_system_failure:
            # F-16: at least one frame failed because the anti-spoof system itself
            # was unavailable, not because it scored the frame as spoof. Retaking
            # photos can't fix this, so don't tell the user "spoof detected".
            return jsonify({
                "status":  "error",
                "message": "ระบบตรวจสอบใบหน้าขัดข้องชั่วคราว กรุณาลองใหม่อีกครั้ง หรือแจ้งเจ้าหน้าที่หากยังพบปัญหา",
            }), 503
        return jsonify({
            "status":       "spoof_detected",
            "failed_frame": first_spoof_frame,
            "reason":       "minifasnet",
            "message":      "ตรวจพบการปลอมแปลงใบหน้า — กรุณาใช้ใบหน้าจริงเท่านั้น",
        }), 400

    # ── 7. Extract FaceNet512 embeddings (face detection happens here) ────────
    embeddings     = []
    detector_crops = []
    failed_indices = []
    for idx, img_b64 in enumerate(face_images):
        try:
            embedding, extraction_meta = extract_embedding(img_b64, include_metadata=True)
            embeddings.append(embedding)
            detector_crop = extraction_meta.get("detector_crop")
            detector_crops.append({"frame": idx + 1, "crop": detector_crop})
            _log(user_id, "extract_embedding", "pass",
                 f"frame={idx+1} detector_crop={detector_crop}")
        except Exception as e:
            msg = str(e)
            no_face = "could not be detected" in msg or "numpy array" in msg
            _log(user_id, "extract_embedding", "no_face" if no_face else "error",
                 f"frame={idx+1} {msg[:80]}")
            failed_indices.append(idx)

    if failed_indices:
        all_failed = len(failed_indices) >= len(face_images)
        if all_failed:
            return jsonify({
                "status":       "error",
                "failed_frame": failed_indices[0] + 1,
                "reason":       "no_face_detected",
                "message":      "ตรวจจับใบหน้าไม่ได้เลย — กรุณาเพิ่มแสงและมองตรงกล้อง",
            }), 400
        # ≥2 frames undetectable → suspicious, block
        if len(failed_indices) >= 2:
            return jsonify({
                "status":       "spoof_detected",
                "failed_frame": failed_indices[0] + 1,
                "reason":       "no_face_multi",
                "message":      "ตรวจพบภาพที่ไม่ใช่ใบหน้าจริง — กรุณาลองใหม่อีกครั้ง",
            }), 400
        # Single frame failed → re-shoot that frame only
        new_retry = retry_count + 1
        session["enroll_retry"] = new_retry
        return jsonify({
            "status":          "need_more",
            "removed_indices": failed_indices,
            "failed_frame":    failed_indices[0] + 1,
            "message":         f"รูปที่ {failed_indices[0]+1} ชัดไม่พอ — ระบบจะถ่ายใหม่อัตโนมัติ",
        })

    # ── 8. Embedding consistency check (B1: multi-outlier aware) ─────────────
    # Circular mode sends varied poses (front/right/left/up/down); cross-pose
    # cosine similarity with FaceNet512 can be 0.75–0.85, so use a slightly
    # lower threshold. Classic mode keeps the stricter 0.80 (all frontal frames).
    _flow_mode = current_app.config["ENROLL_FLOW_MODE"]
    _consistency_threshold = 0.75 if _flow_mode == "circular" else 0.80
    _log(user_id, "consistency_threshold", "set",
         f"flow_mode={_flow_mode} threshold={_consistency_threshold}")
    consistency = check_embedding_consistency(embeddings, threshold=_consistency_threshold)
    averages_by_index = {
        item["frame"] - 1: item["average"]
        for item in consistency["average_similarities"]
    }
    flagged_frames = [
        {
            "frame": idx + 1,
            "average": averages_by_index.get(idx),
            "threshold": _consistency_threshold,
            "reason": "average_below_threshold",
        }
        for idx in consistency["outlier_indices"]
    ]
    consistency_diagnostics = {
        "threshold": _consistency_threshold,
        "pairwise_scores": consistency["pairwise_scores"],
        "average_similarities": consistency["average_similarities"],
        "embedding_diagnostics": consistency["embedding_diagnostics"],
        "detector_crops": detector_crops,
        "flagged_indices_zero_based": consistency["outlier_indices"],
        "flagged_frames": flagged_frames,
        "classifier_reason": "per_frame_average_below_threshold",
        "multi_outlier": consistency["multi_outlier"],
    }
    _log(user_id, "consistency_diagnostics",
         "pass" if consistency["consistent"] else "fail",
         json.dumps(consistency_diagnostics, separators=(",", ":"), sort_keys=True))
    if not consistency["consistent"]:
        _log(user_id, "consistency", "fail",
             f"min_score={consistency.get('min_score')} "
             f"outliers={consistency['outlier_indices']} "
             f"multi={consistency['multi_outlier']}")

        if consistency["multi_outlier"]:
            # ≥2 bad frames → reset capture entirely (B1)
            session.pop("enroll_retry", None)
            return jsonify({
                "status":  "restart_capture",
                "message": "ภาพไม่สม่ำเสมอ — กรุณาถ่ายภาพใหม่ทั้งหมด",
            })

        new_retry = retry_count + 1
        session["enroll_retry"] = new_retry
        outlier_num = consistency["outlier_indices"][0] + 1
        return jsonify({
            "status":          "need_more",
            "removed_indices": consistency["outlier_indices"],
            "message":         f"รูปที่ {outlier_num} ไม่สอดคล้อง — ถ่ายใหม่",
        })

    _log(user_id, "consistency", "pass")

    # ── 9a. Server-side Face Continuity Check ─────────────────────────────────
    # เปรียบเทียบแต่ละ embedding กับ liveness_embeddings จาก session (Step 2-3)
    # ป้องกัน client bypass ที่อาจ patch JS เพื่อส่งหน้าคนอื่นหลัง liveness ผ่าน
    liveness_embeddings = session.get("liveness_embeddings", [])
    if not liveness_embeddings:
        _log(user_id, "continuity", "blocked", "no liveness embeddings in session")
        return jsonify({
            "status":  "error",
            "message": "กรุณาทำ Liveness Check ก่อน — กรุณาเริ่มใหม่",
        }), 400

    # M1: use module-level CONTINUITY_THRESHOLD (defined at top of file)
    for idx, emb in enumerate(embeddings):
        max_sim = max(
            cosine_similarity(emb, ref) for ref in liveness_embeddings
        )
        if max_sim < CONTINUITY_THRESHOLD:
            _log(user_id, "continuity", "fail",
                 f"frame={idx+1} max_sim={max_sim:.4f} threshold={CONTINUITY_THRESHOLD}")
            return jsonify({
                "status":  "continuity_fail",
                "message": "ตรวจพบใบหน้าไม่ตรงกับ Liveness Check — กรุณาเริ่มใหม่",
            }), 400

    _log(user_id, "continuity", "pass",
         f"all {len(embeddings)} frames passed liveness_count={len(liveness_embeddings)}")

    # ── 9. Duplicate face check — batch pagination + early exit (A7) ─────────
    # ดึงทีละ 50 rows แทน full table scan เพื่อ O(batch) แทน O(N)
    BATCH_SIZE  = 50
    dup_offset  = 0
    dup_blocked = False
    while not dup_blocked:
        batch = (
            supabase_admin.table("student_biometrics")
            .select("user_id, face_embeddings")
            .not_.is_("face_embeddings", "null")
            .neq("user_id", user_id)
            .range(dup_offset, dup_offset + BATCH_SIZE - 1)
            .execute()
            .data or []
        )
        if not batch:
            break
        for bio in batch:
            stored = bio.get("face_embeddings") or []
            if not stored:
                continue
            for emb in embeddings:
                sim = max_similarity_multi(emb, stored)
                if sim >= DUPLICATE_THRESHOLD:
                    _log(user_id, "duplicate_check", "blocked",
                         f"sim={sim:.4f} other_user={bio['user_id']}")
                    dup_blocked = True
                    break
                if DUPLICATE_GRAY_ZONE[0] <= sim < DUPLICATE_THRESHOLD:
                    _audit.info(
                        f"[DUPLICATE_GRAY_ZONE] student={user_id} "
                        f"other={bio['user_id']} sim={sim:.4f}"
                    )
            if dup_blocked:
                break
        if len(batch) < BATCH_SIZE:
            break   # last page
        dup_offset += BATCH_SIZE

    if dup_blocked:
        return jsonify({
            "status":  "duplicate",
            "message": "ใบหน้านี้ถูกลงทะเบียนในระบบแล้ว",
        }), 400

    _log(user_id, "duplicate_check", "pass")

    # ── 10. Save embeddings — finalize immediately (self-verify step removed) ────
    from datetime import datetime, timezone
    import base64 as _b64

    now_iso = datetime.now(timezone.utc).isoformat()

    integrity_hash = None
    try:
        integrity_hash = compute_embedding_integrity_hash(
            user_id,
            embeddings,
            current_app.config["EMBEDDING_INTEGRITY_SALT"],
        )
    except Exception as ih_err:
        _log(user_id, "integrity_hash", "warning", str(ih_err)[:80])

    try:
        supabase_admin.table("student_biometrics").upsert({
            "user_id":         user_id,
            "face_embeddings": embeddings,
            "baseline_ear":    baseline_ear,
            "baseline_ear_metric": "pixel-v1" if baseline_ear is not None and data.get("baseline_ear_metric") == "pixel-v1" else None,
            "consent_given":   True,
            "consent_at":      session.get("consent_given_at") or now_iso,
            "enrolled_at":     now_iso,
            "integrity_hash":  integrity_hash,
            "verify_attempts": 0,
        }, on_conflict="user_id").execute()
    except Exception as error:
        if "already_enrolled" in str(error):
            return jsonify(status="already_enrolled", message="Enrollment already completed"), 409
        raise

    # Upload first capture frame as profile image (non-fatal)
    try:
        raw = face_images[0].split(",")[1] if "," in face_images[0] else face_images[0]
        face_path = f"{user_id}.jpg"
        supabase_admin.storage.from_("face-images").upload(
            face_path, _b64.b64decode(raw),
            file_options={"content-type": "image/jpeg", "upsert": "true"},
        )
        supabase_admin.table("student_biometrics") \
            .update({"face_image_url": face_path}).eq("user_id", user_id).execute()
    except Exception as upload_err:
        _log(user_id, "image_upload", "warning", str(upload_err)[:80])

    session["enroll_baseline_ear"] = baseline_ear
    session.pop("enroll_retry", None)
    _log(user_id, "enroll_save", "success")

    return jsonify({"status": "pending_verify", "message": "ลงทะเบียนใบหน้าสำเร็จ!"})


# ─────────────────────────────────────────────────────────────────────────────
# Self-Verify API (B2: retry up to 2 times before wiping)
# ─────────────────────────────────────────────────────────────────────────────

@student_bp.route("/api/self_verify", methods=["POST"])
@login_required
@role_required("student")
@_limiter.limit("10 per minute")
@csrf_protect
def api_self_verify():
    """
    Verify one live shot against the 5 pending embeddings stored in DB.
    B2: user gets up to 2 attempts before pending embeddings are wiped.
    A7: threshold raised to 0.80 (from 0.75).
    On success: set consent_given=True and upload profile image.

    Pipeline: validate_frame → spoof_check → extract_embedding →
              continuity_check → verify_face_multi → finalize
    """
    from app.services.face_service import (
        extract_embedding, verify_face_multi,
        server_validate_frame, check_anti_spoof,
    )

    user_id = session["user_id"]

    data       = request.get_json()
    verify_img = (data or {}).get("face_image")
    if not verify_img:
        return jsonify({"status": "error", "message": "ไม่พบรูปภาพ"}), 400

    if not _liveness_verified():
        _log(user_id, "self_verify_liveness", "blocked", "no recent server-verified challenge")
        return jsonify({"status": "continuity_fail",
                        "message": "กรุณาทำ Liveness Check ก่อน — กรุณาเริ่มใหม่"}), 400

    # ── Zero-trust frame validation ───────────────────────────────────────────
    v = server_validate_frame(verify_img)
    if not v["valid"]:
        _log(user_id, "self_verify_validate", "fail",
             f"reason={v['reason']} meta={v['metadata']}")
        return jsonify({
            "status":  "error",
            "reason":  v["reason"],
            "message": f"รูปภาพไม่ถูกต้อง ({v['reason']}) — กรุณาถ่ายใหม่",
        }), 400

    # ── Server-side spoof check (MiniFASNet) ──────────────────────────────────
    try:
        is_real = check_anti_spoof(verify_img)
        _log(user_id, "self_verify_spoof", "pass" if is_real else "spoof")
        if not is_real:
            return jsonify({
                "status":  "spoof_detected",
                "message": "ตรวจพบภาพปลอม — กรุณาใช้ใบหน้าจริงเท่านั้น",
            }), 400
    except Exception as e:
        _log(user_id, "self_verify_spoof", "exception", str(e)[:80])
        return jsonify({
            "status":  "error",
            "message": "ไม่สามารถตรวจสอบใบหน้าได้ — กรุณาลองใหม่",
        }), 400

    # B2 / H2: track verify attempts in DB (not session) to prevent concurrent-tab bypass
    MAX_VERIFY_ATTEMPTS = 2

    # ── Load pending embeddings + current attempt count ───────────────────────
    bio_res = (
        supabase_admin.table("student_biometrics")
        .select("face_embeddings, baseline_ear, verify_attempts")
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )
    if not bio_res or not bio_res.data:
        return jsonify({"status": "error", "message": "ไม่พบข้อมูล enrollment — กรุณาเริ่มใหม่"}), 400

    try:
        verify_attempts = int(bio_res.data.get("verify_attempts") or 0)
    except (ValueError, TypeError):
        verify_attempts = 0

    # ── Check attempt limit BEFORE expensive DeepFace call ───────────────────
    if verify_attempts >= MAX_VERIFY_ATTEMPTS:
        return jsonify({
            "status":  "error",
            "message": "ยืนยันตัวตนเกินจำนวนครั้งที่กำหนด — กรุณาเริ่มลงทะเบียนใหม่",
        }), 400
    stored_embeddings = bio_res.data.get("face_embeddings") or []
    if not stored_embeddings:
        return jsonify({"status": "error", "message": "ไม่พบ embedding — กรุณาเริ่มใหม่"}), 400

    # ── Extract verify shot ───────────────────────────────────────────────────
    try:
        live_emb = extract_embedding(verify_img)
    except Exception:
        return jsonify({"status": "error", "message": "ตรวจจับใบหน้าไม่สำเร็จ — กรุณาจัดหน้าให้อยู่ในกรอบ"}), 400

    # ── Server-side Face Continuity Check (self-verify) ───────────────────────
    liveness_embeddings = session.get("liveness_embeddings", [])
    if not liveness_embeddings:
        _log(user_id, "self_verify_continuity", "blocked", "no liveness embeddings in session")
        return jsonify({
            "status":  "continuity_fail",
            "message": "กรุณาทำ Liveness Check ก่อน — กรุณาเริ่มใหม่",
        }), 400

    max_sim = max(cosine_similarity(live_emb, ref) for ref in liveness_embeddings)
    if max_sim < CONTINUITY_THRESHOLD:
        _log(user_id, "self_verify_continuity", "fail", f"max_sim={max_sim:.4f} threshold={CONTINUITY_THRESHOLD}")
        return jsonify({
            "status":  "continuity_fail",
            "message": "ตรวจพบใบหน้าไม่ตรงกับ Liveness Check — กรุณาเริ่มใหม่",
        }), 400

    _log(user_id, "self_verify_continuity", "pass", f"max_sim={max_sim:.4f}")

    # ── Compare against stored embeddings ─────────────────────────────────────
    verify_result = verify_face_multi(live_emb, stored_embeddings, SELF_VERIFY_THRESHOLD)
    best_sim = verify_result["best_similarity"]
    _log(user_id, "self_verify",
         "pass" if verify_result["verified"] else "fail",
         f"best_sim={best_sim:.4f} avg={verify_result['avg_similarity']:.4f} "
         f"threshold={SELF_VERIFY_THRESHOLD} attempt={verify_attempts+1}/{MAX_VERIFY_ATTEMPTS}")

    if not verify_result["verified"]:
        new_attempts = verify_attempts + 1
        if new_attempts >= MAX_VERIFY_ATTEMPTS:
            # Wipe pending embeddings + reset counter — must re-enroll from scratch
            supabase_admin.table("student_biometrics") \
                .update({"face_embeddings": None, "verify_attempts": 0}) \
                .eq("user_id", user_id).execute()
            session.pop("enroll_baseline_ear", None)
            _log(user_id, "self_verify", "wiped", "max_attempts_reached")
            return jsonify({
                "status":  "failed",
                "message": "ยืนยันตัวตนไม่สำเร็จ — กรุณาลงทะเบียนใบหน้าใหม่ตั้งแต่ต้น",
            })
        # Still have attempts left — increment DB counter then let user retry
        supabase_admin.table("student_biometrics") \
            .update({"verify_attempts": new_attempts}) \
            .eq("user_id", user_id).execute()
        remaining = MAX_VERIFY_ATTEMPTS - new_attempts
        return jsonify({
            "status":             "retry",
            "message":            f"ยืนยันไม่ผ่าน — กรุณาลองอีกครั้ง (เหลืออีก {remaining} ครั้ง)",
            "remaining_attempts": remaining,
        })

    # ── Finalize: set consent_given=True ──────────────────────────────────────
    from datetime import datetime, timezone
    baseline_ear = session.get("enroll_baseline_ear") or bio_res.data.get("baseline_ear")
    now_iso = datetime.now(timezone.utc).isoformat()

    try:
        # Sprint 2B: compute integrity hash before writing
        integrity_hash = compute_embedding_integrity_hash(
            user_id,
            stored_embeddings,
            current_app.config["EMBEDDING_INTEGRITY_SALT"],
        )

        supabase_admin.table("student_biometrics").update({
            "consent_given":   True,
            "consent_at":      session.get("consent_given_at") or now_iso,
            "enrolled_at":     now_iso,
            "baseline_ear":    baseline_ear,
            "baseline_ear_metric": None,  # obsolete flow has no metric provenance
            "integrity_hash":  integrity_hash,
            "verify_attempts": 0,   # H2: reset counter on successful enrollment
        }).eq("user_id", user_id).execute()

        # Upload self-verify shot as profile image (non-fatal)
        # PDPA: bucket must be set to PRIVATE in Supabase Dashboard → Storage → face-images
        # Access is via signed URL generated at render time (1-hour expiry)
        try:
            import base64 as _b64
            raw = verify_img.split(",")[1] if "," in verify_img else verify_img
            face_path = f"{user_id}.jpg"
            supabase_admin.storage.from_("face-images").upload(
                face_path, _b64.b64decode(raw),
                file_options={"content-type": "image/jpeg", "upsert": "true"},
            )
            # Store path only (not a public URL) — signed URL generated on demand
            supabase_admin.table("student_biometrics") \
                .update({"face_image_url": face_path}).eq("user_id", user_id).execute()
        except Exception as upload_err:
            _log(user_id, "image_upload", "warning", str(upload_err)[:80])

        # Sprint 1B: generate HMAC device token bound to this device fingerprint
        device_fingerprint = (data or {}).get("device_fingerprint", "")
        device_token = None
        if device_fingerprint:
            device_token = create_device_token(
                user_id,
                device_fingerprint,
                current_app.config["SECRET_KEY"],
            )

        # Clean up session (รวม liveness embeddings)
        session.pop("enroll_baseline_ear", None)
        session.pop("consent_given_at", None)
        session.pop("liveness_embeddings", None)
        session.pop("liveness_verified_at", None)

        _log(user_id, "enroll_finalize", "success",
             f"best_sim={best_sim:.4f} device_bound={bool(device_fingerprint)}")
        return jsonify({
            "status":       "success",
            "similarity":   best_sim,
            "message":      "ลงทะเบียนใบหน้าสำเร็จ!",
            "device_token": device_token,   # None if no fingerprint sent
        })

    except Exception as e:
        _log(user_id, "enroll_finalize", "error", str(e)[:80])
        return jsonify({"status": "error", "message": "บันทึกข้อมูลไม่สำเร็จ — กรุณาลองใหม่"}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Inline Spoof Check API (Step 2, 3, 4 of enrollment)
# ─────────────────────────────────────────────────────────────────────────────

@student_bp.route("/api/spoof_check", methods=["POST"])
@login_required
@role_required("student")
@_limiter.limit("30 per minute")
@csrf_protect
def api_spoof_check():
    """
    Single-frame spoof check for enrollment flow.
    Called ~8 times per enrollment (Step 2×1, Step 3×2, Step 4×5).

    ถ้า is_real=True → เก็บ embedding ลง session["liveness_embeddings"] (server-side)
    Response: { is_real, confidence, message } — ไม่ส่ง embedding กลับ client
    """
    import base64 as _b64
    from app.services.face_service import (
        spoof_check_with_embedding, server_validate_frame,
        detect_screen_moire, detect_screen_texture, _decode_image,
        MOIRE_THRESHOLD_SINGLE,
    )

    user_id = session["user_id"]
    data    = request.get_json()
    img_b64 = (data or {}).get("image")

    if not img_b64:
        return jsonify({"is_real": False, "confidence": 0.0,
                        "message": "ไม่พบรูปภาพ"}), 400

    # ── 1. Zero-trust frame validation ───────────────────────────────────────
    v = server_validate_frame(img_b64)
    if not v["valid"]:
        metadata = v.get("metadata", {})
        diagnostic_fields = [f"reason={v['reason']}"]
        for key in ("size_kb", "dimensions", "laplacian_var", "B_std", "G_std", "R_std"):
            if key in metadata:
                diagnostic_fields.append(f"{key}={metadata[key]}")
        _log(user_id, "spoof_check", "frame_invalid", " ".join(diagnostic_fields))
        return jsonify({"is_real": False, "confidence": 0.0,
                        "message": "รูปภาพไม่ถูกต้อง"}), 400

    # ── 2. Decode once — shared by all checks below ───────────────────────────
    try:
        raw = _decode_image(img_b64)
    except Exception as e:
        return jsonify({"is_real": False, "confidence": 0.0,
                        "message": "อ่านรูปภาพไม่ได้"}), 400

    # ── 3. Single-frame Moiré FFT — audit-only; verdict/error never rejects ──
    try:
        moire = detect_screen_moire([raw], threshold=MOIRE_THRESHOLD_SINGLE)
        _log(user_id, "liveness_moire", "screen" if moire["is_screen"] else "pass",
             f"score={moire['avg_score']} threshold={MOIRE_THRESHOLD_SINGLE}")
    except Exception as e:
        _log(user_id, "liveness_moire", "error_log_only",
             f"error={str(e)[:80]} decision=log_only")

    # ── 4. Single-frame screen texture — audit-only; verdict/error never rejects ──
    try:
        is_screen_tex = detect_screen_texture(raw, min_peaks=30)
        _log(user_id, "liveness_texture", "screen" if is_screen_tex else "pass", "")
    except Exception as e:
        _log(user_id, "liveness_texture", "error_log_only",
             f"error={str(e)[:80]} decision=log_only")

    # ── 5. Accumulated temporal variance (สะสม thumbnail ใน session) ─────────
    # เก็บ 64×64 grayscale PNG (~1-2 KB ต่อเฟรม) เพื่อเช็ค inter-frame variance
    try:
        gray_small = cv2.resize(
            cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY), (64, 64)
        )
        _, buf = cv2.imencode(".png", gray_small)
        thumb_b64 = _b64.b64encode(buf.tobytes()).decode("utf-8")

        acc = session.get("spoof_check_acc", [])
        acc.append(thumb_b64)
        acc = acc[-6:]   # สะสมไม่เกิน 6 เฟรม (Steps 2–3–4 รวมกัน)
        session["spoof_check_acc"] = acc

        if len(acc) >= 3:
            frames_gray = []
            for tb in acc:
                arr = np.frombuffer(_b64.b64decode(tb), dtype=np.uint8)
                g = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                if g is not None:
                    frames_gray.append(g.astype(np.float32))

            if len(frames_gray) >= 3:
                stack    = np.stack(frames_gray, axis=0)
                mean_var = float(np.mean(np.std(stack, axis=0)))
                _log(user_id, "liveness_temporal",
                     "static_log_only" if mean_var < 6.0 else "pass_log_only",
                     f"variance={mean_var:.3f} frames={len(frames_gray)} "
                     "reference_threshold=6.0 decision=log_only")
    except Exception as e:
        _log(user_id, "liveness_temporal", "error", str(e)[:80])

    # ── 6. DeepFace face detection + embedding ────────────────────────────────
    try:
        result = spoof_check_with_embedding(img_b64)
    except Exception as e:
        _log(user_id, "spoof_check", "exception", str(e)[:80])
        return jsonify({"is_real": False, "confidence": 0.0,
                        "message": "ไม่สามารถตรวจสอบได้ กรุณาลองใหม่อีกครั้ง"}), 500

    # ถ้า real face → บันทึก embedding ลง session (สำหรับ server-side continuity check)
    # Only after the head-turn challenge passed, and only for the same face: the
    # reference set starts from challenge frames and each addition must match it,
    # so a still image sent here later cannot become the continuity reference.
    stored = session.get("liveness_embeddings", [])
    if (result["is_real"] and result.get("embedding") and _liveness_verified() and stored
            and max(cosine_similarity(result["embedding"], ref) for ref in stored) >= CONTINUITY_THRESHOLD):
        stored.append(result["embedding"])
        # Bug fix: use [-5:] not [:5].
        # /api/spoof_check is called up to 8 times (liveness ×3, capture ×5).
        # [:5] kept only the first 5 embeddings, so capture-phase embeddings were
        # discarded and face-continuity in /api/enroll compared against stale liveness frames.
        # [-5:] keeps the 5 most-recent embeddings, ensuring continuity is checked
        # against the frames closest to the final enrollment capture.
        session["liveness_embeddings"] = stored[-5:]   # keep 5 most-recent embeddings

    if result.get("retry_capture"):
        _log(user_id, "spoof_check", "face_not_detected", "retry_capture=true")
        return jsonify({"is_real": False, "confidence": 0.0,
                        "retry_capture": True, "message": result["message"]})

    if result.get("system_failure"):
        # F-16 (docs/review/10-moire-frr-investigation.md §16-17): anti-spoof system
        # itself failed to run — not a spoof determination. Called up to 8x per
        # enrollment; without this split a user would see "spoof detected" that
        # many times for something retrying can't fix. _callSpoofCheckSafe() in
        # enrollment_flow.js already treats any non-2xx/429 status as a soft
        # network error (fail-open, no penalty to retry counters) — no frontend
        # change needed for that part.
        _log(user_id, "spoof_check", "system_failure",
             f"confidence={result['confidence']}")
        return jsonify({"is_real": False, "confidence": 0.0,
                        "message": "ระบบตรวจสอบใบหน้าขัดข้องชั่วคราว กรุณาลองใหม่อีกครั้ง หรือแจ้งเจ้าหน้าที่หากยังพบปัญหา"}), 503

    _log(user_id, "spoof_check",
         "real" if result["is_real"] else "spoof",
         f"confidence={result['confidence']} "
         f"liveness_count={len(session.get('liveness_embeddings', []))}")

    # ไม่ส่ง embedding กลับ client — ป้องกัน JS inspection/bypass
    return jsonify({
        "is_real":    result["is_real"],
        "confidence": result["confidence"],
        "message":    result.get("message", ""),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Reset Liveness Session (เรียกตอน fullRestart เพื่อล้าง liveness embeddings)
# ─────────────────────────────────────────────────────────────────────────────

@student_bp.route("/api/reset-liveness", methods=["POST"])
@login_required
@role_required("student")
@_limiter.limit("10 per minute")
@csrf_protect
def api_reset_liveness():
    """ล้าง session['liveness_embeddings'] และ spoof_check_acc — เรียกเมื่อ user เริ่มลงทะเบียนใหม่ตั้งแต่ Step 2"""
    session.pop("liveness_embeddings", None)
    session.pop("spoof_check_acc", None)
    session.pop("liveness_challenge", None)
    session.pop("liveness_verified_at", None)
    _log(session.get("user_id", ""), "reset_liveness", "cleared")
    return jsonify({"status": "cleared"})


# ─────────────────────────────────────────────────────────────────────────────
# Server-verified head-turn challenge (Step 4 liveness)
# ─────────────────────────────────────────────────────────────────────────────

@student_bp.route("/api/liveness/challenge", methods=["POST"])
@login_required
@role_required("student")
@_limiter.limit("10 per minute")
@csrf_protect
def api_liveness_challenge():
    """Issue a random turn order bound to a one-time nonce; resets liveness state."""
    from app.services.liveness_challenge import new_challenge, CHALLENGE_TTL_S

    challenge = new_challenge()
    session["liveness_challenge"] = challenge
    session.pop("liveness_embeddings", None)
    session.pop("liveness_verified_at", None)
    _log(session["user_id"], "liveness_challenge", "issued", f"actions={challenge['actions']}")
    return jsonify({"nonce": challenge["nonce"], "actions": challenge["actions"],
                    "expires_in": CHALLENGE_TTL_S})


@student_bp.route("/api/liveness/verify", methods=["POST"])
@login_required
@role_required("student")
@_limiter.limit("10 per minute")
@csrf_protect
def api_liveness_verify():
    """Check the frames of a completed challenge. One attempt per challenge.

    Body: {nonce, before, action_frames: [one per action], after} (JPEG data URLs).
    On success the before/after embeddings become the continuity reference used
    by /api/enroll and /api/self_verify.
    """
    from app.services.face_service import server_validate_frame, extract_embedding
    from app.services.liveness_challenge import verify, decode_frame

    user_id   = session["user_id"]
    data      = request.get_json(silent=True) or {}
    challenge = session.pop("liveness_challenge", None)  # consumed even on failure
    retry_msg = "ตรวจท่าทางไม่ผ่าน — กรุณาทำใหม่ตามคำสั่ง"

    action_frames = data.get("action_frames")
    if not isinstance(action_frames, list) or not 1 <= len(action_frames) <= 4:
        _log(user_id, "liveness_verify", "fail", "reason=bad_request")
        return jsonify({"passed": False, "message": retry_msg}), 400
    frames = [data.get("before")] + action_frames + [data.get("after")]
    if not all(isinstance(f, str) and f for f in frames):
        _log(user_id, "liveness_verify", "fail", "reason=bad_request")
        return jsonify({"passed": False, "message": retry_msg}), 400
    for idx, frame in enumerate(frames):
        v = server_validate_frame(frame)
        if not v["valid"]:
            _log(user_id, "liveness_verify", "fail", f"reason=frame_invalid frame={idx} detail={v['reason']}")
            return jsonify({"passed": False, "message": retry_msg}), 400

    try:
        decoded = [decode_frame(f) for f in frames]
        result = verify(challenge, data.get("nonce"), decoded[0], decoded[1:-1], decoded[-1])
    except Exception as e:
        _log(user_id, "liveness_verify", "error", f"{type(e).__name__}: {str(e)[:80]}")
        return jsonify({"passed": False,
                        "message": "ระบบตรวจสอบใบหน้าขัดข้องชั่วคราว กรุณาลองใหม่อีกครั้ง"}), 503

    _log(user_id, "liveness_verify", "pass" if result["passed"] else "fail",
         f"reason={result['reason']} yaws={result['yaws']} scores={result['scores']}")
    if not result["passed"]:
        return jsonify({"passed": False, "message": retry_msg}), 400

    # Continuity references in the same pipeline /api/enroll uses.
    refs = []
    for frame in (frames[0], frames[-1]):
        try:
            refs.append(extract_embedding(frame))
        except Exception:
            pass
    if not refs:
        _log(user_id, "liveness_verify", "fail", "reason=reference_embedding_failed")
        return jsonify({"passed": False, "message": retry_msg}), 400
    session["liveness_embeddings"]  = refs
    session["liveness_verified_at"] = time.time()
    return jsonify({"passed": True})


# ─── PDPA: Consent Withdrawal ─────────────────────────────────────────────────

@student_bp.route("/api/withdraw-consent", methods=["POST"])
@login_required
@role_required("student")
@csrf_protect
def api_withdraw_consent():
    """
    PDPA Right to Withdraw.
    Order of operations (MUST be strict):
      1. Insert consent_given=False into consent_logs (audit trail — always first)
      2. Hard-delete sensitive biometric columns (fail LOUD if this fails)
      3. Clear session state
      4. Remove the stored profile photo (retryable if storage is unavailable)
      5. Audit log event_type=consent_withdrawn_data_deleted
    consent_logs rows are NEVER deleted — they are the audit trail.
    """
    from app.services.security_service import log_audit_event
    user_id = session["user_id"]

    # ── 1. Record withdrawal in consent_logs (audit trail) ───────────────────
    try:
        supabase_admin.table("consent_logs").insert({
            "user_id":         user_id,
            "consent_type":    "biometric_enrollment",
            "consent_given":   False,
            "consent_version": "1.1",
            "ip_address":      _safe_ip(),
            "user_agent":      request.headers.get("User-Agent", "")[:500],
        }).execute()
    except Exception as e:
        _log(user_id, "withdraw_consent", "consent_log_error", str(e)[:80])
        return jsonify({"status": "error", "message": "ไม่สามารถบันทึกการถอนความยินยอม"}), 500

    # ── 2. Hard delete sensitive biometric columns ───────────────────────────
    # CRITICAL: if this fails after consent is withdrawn = PDPA violation.
    # Must fail loud — no silent fallthrough.
    try:
        supabase_admin.table("student_biometrics").update({
            "face_embeddings":   None,
            "baseline_ear":      None,
            "baseline_ear_metric": None,
            "face_image_url":    None,
            "integrity_hash":    None,
            "consent_given":     False,
            "enrollment_attempts": 0,
            "enrolled_at":       None,
        }).eq("user_id", user_id).execute()
    except Exception as e:
        current_app.logger.error(
            f"CRITICAL: Failed to delete biometrics after consent withdrawal "
            f"user={user_id} error={e}"
        )
        _log(user_id, "withdraw_consent", "biometrics_delete_failed", str(e)[:80])
        return jsonify({
            "status":  "error",
            "message": "ไม่สามารถลบข้อมูลได้ กรุณาติดต่อผู้ดูแลระบบ",
        }), 500

    # ── 3. Clear session state ────────────────────────────────────────────────
    session.pop("consent_given_at",    None)
    session.pop("consent_ip",          None)
    session.pop("liveness_embeddings", None)
    session.pop("enroll_baseline_ear", None)
    session.pop("enroll_retry", None)
    session.pop("spoof_check_acc", None)
    _log(user_id, "withdraw_consent", "biometrics_deleted")

    # Both enrollment upload paths use this server-derived key. Remove it even
    # when face_image_url was already cleared, so failed cleanup can be retried
    # and orphaned uploads are covered. Never accept a deletion path from input.
    try:
        supabase_admin.storage.from_("face-images").remove([f"{user_id}.jpg"])
    except Exception as error:
        _log(user_id, "withdraw_consent", "photo_delete_failed", type(error).__name__)
        return jsonify({
            "status": "error",
            "consent_withdrawn": True,
            "photo_cleanup_pending": True,
            "message": "ถอนความยินยอมแล้ว แต่ยังลบรูปไม่สำเร็จ กรุณาลองอีกครั้งหรือติดต่อผู้ดูแลระบบ",
        }), 503

    # ── 5. Audit log (non-fatal — consent + deletion already completed) ──────
    try:
        log_audit_event(
            supabase_admin,
            actor_id=user_id,
            actor_role="student",
            event_type="consent_withdrawn_data_deleted",
            target_id=user_id,
            old_value="biometrics_existed",
            new_value="biometrics_deleted",
        )
    except Exception:
        pass

    return jsonify({
        "status":  "ok",
        "message": "ถอนความยินยอมและลบข้อมูลชีวมิติเรียบร้อยแล้ว",
        "deleted": ["face_embeddings", "baseline_ear", "face_image_url", "integrity_hash", "face_photo"],
    })

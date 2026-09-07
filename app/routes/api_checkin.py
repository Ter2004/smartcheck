import logging
import time
import cv2
import numpy as np
from datetime import datetime, timezone
from dateutil import parser as dtparser
from flask import Blueprint, request, jsonify, session, current_app
from app.routes.auth import login_required, role_required
from app import supabase_admin

from app.services.request_audit import RequestLogger, audited_checkin, event, reject, stage
_log = RequestLogger(logging.getLogger("smartcheck.checkin"), {})
from app.services.face_service import (
    extract_embedding, verify_face_multi,
    check_anti_spoof, check_anti_spoof_with_score, combined_spoof_score,
    is_system_failure,
    detect_screen_moire, detect_screen_texture,
    MOIRE_THRESHOLD_SINGLE, _decode_image, server_validate_frame,
    SAME_DEVICE_THRESHOLD, NEW_DEVICE_THRESHOLD,
)
from app.services.security_service import (
    verify_device_token_details, verify_embedding_integrity, csrf_protect,
)
from app.services.esp32_totp import verify_code
from app.services import proximity_receipt
from app import limiter as _limiter

api_checkin_bp = Blueprint("api_checkin", __name__)

# ─── Face verification thresholds (FaceNet512) ───────────────────────────────
@api_checkin_bp.route("/api/checkin", methods=["POST"])
@audited_checkin
@login_required
@role_required("student")
@_limiter.limit("5 per minute")
@csrf_protect
def checkin():
    # TEMP PERF: instrumentation for docs/review/06-performance.md — remove after measurement
    _t0 = time.perf_counter()
    _perf = {}
    student_id = session["user_id"]
    data = request.get_json()

    if not isinstance(data, dict) or not data:
        reject("payload_empty")
        return jsonify({"ok": False, "error": "ไม่พบข้อมูล"}), 400

    proximity_method = current_app.config.get("CHECKIN_PROXIMITY_METHOD", "totp")

    session_id      = data.get("session_id")
    ble_rssi        = data.get("ble_rssi")
    liveness_action = data.get("liveness_action", "") or ""
    face_image      = data.get("face_image")
    ear_samples     = data.get("ear_samples") or []

    receipt_error = _verify_receipt(data)
    if receipt_error is not None:
        return receipt_error

    # M7: whitelist liveness_action — reject arbitrary strings
    _ALLOWED_LIVENESS_ACTIONS = {"passive", "blink", "turn_left"}
    if not isinstance(liveness_action, str) or liveness_action not in _ALLOWED_LIVENESS_ACTIONS:
        reject("liveness_action_invalid", received=data.get("liveness_action"))
        return jsonify({"ok": False, "error": "ข้อมูลไม่ถูกต้อง"}), 400

    # M6: validate ble_rssi before int() conversion
    if ble_rssi is not None:
        try:
            ble_rssi = int(ble_rssi)
            if not (-120 <= ble_rssi <= 0):
                ble_rssi = None  # out of realistic RSSI range — ignore silently
        except (ValueError, TypeError):
            reject("ble_value_invalid")
            return jsonify({"ok": False, "error": "ข้อมูล BLE ไม่ถูกต้อง"}), 400

    if not all([session_id, face_image]):
        reject("required_fields_missing")
        return jsonify({"ok": False, "error": "ข้อมูลไม่ครบ"}), 400

    # ─── 0. Device token verification (Sprint 1B) — cheapest check first ──────
    authorization = request.headers.get("Authorization", "").strip()
    parts = authorization.split(None, 1)
    bare_scheme = parts == ["DeviceToken"]
    raw_token = "" if bare_scheme else (parts[1] if len(parts) == 2 and parts[0] == "DeviceToken" else authorization)
    device_payload, token_reason = verify_device_token_details(raw_token, current_app.config["SECRET_KEY"])
    if bare_scheme:
        event("device_token_bare_scheme", "absent")
    elif not raw_token:
        event("device_token", "absent")
    elif device_payload is None:
        reject("device_token_" + token_reason)
        return jsonify({"ok": False, "error": "Device token \u0e44\u0e21\u0e48\u0e16\u0e39\u0e01\u0e15\u0e49\u0e2d\u0e07"}), 403
    else:
        event("device_token", "pass")
    if device_payload is not None and device_payload.get("uid") != student_id:
        # Token is valid but belongs to a different user — reject immediately
        reject("device_token_user_mismatch")
        return jsonify({"ok": False, "error": "Device token ไม่ตรงกับบัญชีนี้"}), 403
    # device_payload=None + no raw_token = legacy / first check-in — allowed
    _perf["validate"] = round((time.perf_counter() - _t0) * 1000, 2); _t1 = time.perf_counter()

    try:
        sess, eligibility_error = _eligible(data)
    except Exception as error:
        reject("proximity_verifier_error", exception_type=type(error).__name__)
        return jsonify(ok=False, error=_receipt_message("proximity_verifier_error"), retry_room_code=True), 503
    if eligibility_error is not None:
        return eligibility_error

    # ─── 1b. Zero-trust frame validation (Sprint 2A) ─────────────────────────
    frame_check = server_validate_frame(face_image)
    if not frame_check["valid"]:
        _log.info(f"[FRAME_VALIDATE] fail reason={frame_check['reason']} meta={frame_check['metadata']}")
        reject("frame_invalid")
        return jsonify({
            "ok":        False,
            "error":     "รูปภาพไม่ถูกต้อง — กรุณาถ่ายใหม่อีกครั้ง",
            "retry_face": True,
        }), 400

    _perf["session"] = round((time.perf_counter() - _t1) * 1000, 2); _t2 = time.perf_counter()

    # ─── 2. BLE RSSI check ───────────────────────────────────────────────────
    if current_app.config.get("BLE_CHECK_ENABLED", False):
        rssi_threshold = -70  # dBm — must be within range
        ble_skip       = data.get("ble_skip", False)
        if not ble_skip and (ble_rssi is None or ble_rssi < rssi_threshold):
            _log.warning(f"[BLE] RSSI fail: rssi={ble_rssi} threshold={rssi_threshold}")
            reject("ble_proximity_failed")
            return jsonify({"ok": False, "error": "ไม่พบสัญญาณ Beacon ในห้องเรียน"}), 400
        ble_pass = True
    else:
        _log.debug("[BLE] check skipped (BLE_CHECK_ENABLED=false)")
        ble_pass = True
    _perf["ble"] = round((time.perf_counter() - _t2) * 1000, 2); _t3 = time.perf_counter()

    # ─── 3. Server-side EAR liveness check ──────────────────────────────────
    server_liveness_pass = False
    try:
        ear_arr = np.asarray(ear_samples, dtype=float)
        if ear_arr.ndim != 1 or len(ear_arr) < 2 or not np.all(np.isfinite(ear_arr)):
            raise ValueError("invalid EAR samples")
        if np.any((ear_arr < 0.0) | (ear_arr > 1.0)):
            raise ValueError("EAR samples out of range")

        # Only a blink challenge can be proven from EAR. Other challenge types
        # still have to pass temporal and anti-spoof checks below.
        if liveness_action == "blink":
            ear_std = float(np.std(ear_arr))
            ear_min = float(np.min(ear_arr))
            _log.info(f"[LIVENESS] ear std={ear_std:.4f} min={ear_min:.4f} n={len(ear_arr)}")
            if ear_std < 0.03 or ear_min >= 0.18:
                reject("blink_failed")
                return jsonify({
                    "ok":        False,
                    "error":     "ไม่ผ่านการตรวจสอบความมีชีวิต — กรุณากะพริบตาตามธรรมชาติขณะเช็คชื่อ",
                    "retry_face": True,
                }), 400
    except (ValueError, TypeError) as ear_err:
        _log.warning(f"[LIVENESS] EAR validation failed: {type(ear_err).__name__}")
        reject("ear_invalid")
        return jsonify({
            "ok": False,
            "error": "ข้อมูลตรวจสอบความมีชีวิตไม่ถูกต้อง กรุณาลองใหม่",
            "retry_face": True,
        }), 400
    _perf["ear"] = round((time.perf_counter() - _t3) * 1000, 2); _t4 = time.perf_counter()

    # ─── 4a. Moiré (computed + logged only — FRR-1/F-15/Q-15, does not reject) ───
    try:
        raw_frame = _decode_image(face_image)
    except Exception as error:
        reject("frame_decode_error", exception_type=type(error).__name__)
        return jsonify(ok=False, error="\u0e44\u0e21\u0e48\u0e2a\u0e32\u0e21\u0e32\u0e23\u0e16\u0e2d\u0e48\u0e32\u0e19\u0e20\u0e32\u0e1e\u0e44\u0e14\u0e49 \u0e01\u0e23\u0e38\u0e13\u0e32\u0e16\u0e48\u0e32\u0e22\u0e43\u0e2b\u0e21\u0e48", retry_face=True), 400
    try:
        moire       = detect_screen_moire([raw_frame], threshold=MOIRE_THRESHOLD_SINGLE)
        _log.info(f"[MOIRE] avg_score={moire['avg_score']} is_screen={moire['is_screen']} threshold={MOIRE_THRESHOLD_SINGLE}")
    except Exception as moire_err:
        event("moire", "error_log_only", exception_type=type(moire_err).__name__, decision="log_only")
    _perf["moire"] = round((time.perf_counter() - _t4) * 1000, 2); _t5 = time.perf_counter()

    # ─── 4a-2. Screen Texture (computed + logged only — FRR-1/F-15/Q-15, does not reject) ───
    try:
        is_screen_tex = detect_screen_texture(raw_frame, min_peaks=30)
        _log.info(f"[SCREEN_TEXTURE] is_screen={is_screen_tex}")
    except Exception as tex_err:
        event("texture", "error_log_only", exception_type=type(tex_err).__name__, decision="log_only")
    _perf["texture"] = round((time.perf_counter() - _t5) * 1000, 2); _t6 = time.perf_counter()

    # ─── 4a-3. Temporal variance (uncalibrated audit only) ─
    face_images_list = data.get("face_images")
    try:
        if not isinstance(face_images_list, list) or len(face_images_list) < 2:
            raise ValueError("not enough submitted temporal frames")
        frames_gray = []
        for img_b64 in face_images_list[-3:]:
            try:
                frame_bgr = _decode_image(img_b64)
                gray = cv2.resize(
                    cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY),
                    (64, 64)
                )
                frames_gray.append(gray.astype(np.float32))
            except Exception as error:
                event("temporal_frame", "error_log_only", exception_type=type(error).__name__, decision="log_only")
                continue
        if len(frames_gray) >= 2:
            stack = np.stack(frames_gray, axis=0)
            temporal_var = float(np.mean(np.std(stack, axis=0)))
            event("temporal", "measured_log_only", variance=round(temporal_var, 3),
                  frames=len(frames_gray), reference_threshold=4.0, decision="log_only")
        else:
            raise ValueError("not enough valid temporal frames")
    except Exception as temp_err:
        event("temporal", "error_log_only", exception_type=type(temp_err).__name__, decision="log_only", reference_threshold=4.0)
    _perf["temporal"] = round((time.perf_counter() - _t6) * 1000, 2); _t7 = time.perf_counter()

    # ─── 4b. Anti-spoofing via MiniFASNet ────────────────────────────────────
    try:
        with stage("antispoof"):
            spoof_result = combined_spoof_score(raw_frame)
        is_real = spoof_result["is_real"]
        _spoof_timings = spoof_result.get("timings", {})
        _perf["fasnet"] = _spoof_timings.get("fasnet_ms", 0.0)
        _perf["onnx"]   = _spoof_timings.get("onnx_ms", 0.0)
        if not is_real:
            if is_system_failure(spoof_result):
                # F-16 (docs/review/10-moire-frr-investigation.md §16): Fasnet (or all
                # voting layers) failed to run — this is an infra failure, not a spoof
                # determination. Retaking the photo can't fix it, so don't tell the
                # user "spoof detected".
                _log.error(f"[ANTISPOOF] system failure, not a spoof determination — "
                           f"disagreements={spoof_result.get('disagreements')}")
                reject("antispoof_unavailable")
                return jsonify({
                    "ok": False,
                    "error": "ระบบตรวจสอบใบหน้าขัดข้องชั่วคราว กรุณาลองใหม่อีกครั้ง หรือแจ้งเจ้าหน้าที่หากยังพบปัญหา",
                    "retry_face": True,
                }), 503
            reject("spoof_detected")
            return jsonify({
                "ok": False,
                "error": "ตรวจพบรูปถ่ายหรือหน้าจอ — กรุณาใช้ใบหน้าจริงเท่านั้น",
                "spoof": True,
                "retry_face": True,
            }), 400
        server_liveness_pass = True
    except Exception as e:
        _log.error(f"[ANTISPOOF] check error (fail-close): {type(e).__name__}")
        reject("antispoof_error")
        return jsonify({
            "ok": False,
            "error": "ไม่สามารถตรวจสอบใบหน้าได้ — กรุณาถ่ายใหม่อีกครั้ง",
            "retry_face": True,
        }), 400
    _t8 = time.perf_counter()

    # ─── 5. Device binding (determines threshold) ─────────────────────────────
    device_id = request.headers.get("X-Device-ID", "")
    with stage("device_lookup"):
        user_res  = (
            supabase_admin.table("users")
            .select("device_id")
            .eq("id", student_id)
            .maybe_single()
            .execute()
        )
    user_device = ((user_res and user_res.data) or {}).get("device_id") or ""
    if user_device and device_id and user_device != device_id:
        reject("device_binding_mismatch")
        return jsonify({"ok": False, "error": "Device ไม่ตรง — ต้องใช้อุปกรณ์ที่ผูกไว้"}), 400
    # Bind device on first check-in
    # M5: verify device_id not already bound to another student before binding
    if not user_device and device_id:
        existing = (
            supabase_admin.table("users")
            .select("id")
            .eq("device_id", device_id)
            .neq("id", student_id)
            .maybe_single()
            .execute()
        )
        if existing and existing.data:
            _log.warning(f"[DEVICE_BIND] device_id already bound to another student={existing.data.get('id')} attempt by={student_id}")
            reject("device_owned_by_other")
            return jsonify({"ok": False, "error": "อุปกรณ์นี้ถูกผูกกับบัญชีอื่นแล้ว"}), 403
        supabase_admin.table("users").update({"device_id": device_id}).eq("id", student_id).execute()

    # Sprint 1B: HMAC device token counts as trusted regardless of DB binding
    token_trusted  = device_payload is not None   # cryptographic proof of device
    db_trusted     = bool(user_device and device_id and user_device == device_id)
    device_trusted = token_trusted or db_trusted
    face_threshold = SAME_DEVICE_THRESHOLD if device_trusted else NEW_DEVICE_THRESHOLD

    # ─── 6. Face verification (multi-embedding) ───────────────────────────────
    with stage("biometrics_lookup"):
        bio_res = (
            supabase_admin.table("student_biometrics")
            .select("face_embeddings, integrity_hash")
            .eq("user_id", student_id)
            .maybe_single()
            .execute()
        )
    if not bio_res or not bio_res.data:
        reject("biometrics_missing")
        return jsonify({"ok": False, "error": "ยังไม่ได้ลงทะเบียนใบหน้า"}), 400

    stored_embeddings = bio_res.data.get("face_embeddings") or []
    if not stored_embeddings:
        reject("embeddings_empty")
        return jsonify({"ok": False, "error": "ยังไม่ได้ลงทะเบียนใบหน้า"}), 400

    # Sprint 2B: verify embedding integrity before using them
    stored_hash = bio_res.data.get("integrity_hash") or ""
    if not verify_embedding_integrity(
        student_id, stored_embeddings, stored_hash,
        current_app.config["EMBEDDING_INTEGRITY_SALT"],
    ):
        _log.warning(f"[INTEGRITY] VIOLATION student={student_id}")
        reject("embedding_integrity_failed")
        return jsonify({
            "ok":   False,
            "error": "ข้อมูลชีวมาตรไม่สมบูรณ์ — กรุณาลงทะเบียนใบหน้าใหม่อีกครั้ง",
        }), 403

    try:
        with stage("extraction"):
            live_embedding = extract_embedding(face_image)
    except Exception as e:
        _log.warning(f"[FACE] extract_embedding failed: {type(e).__name__}")
        # L1: don't expose internal error details to client
        reject("extraction_failed")
        return jsonify({"ok": False, "error": "ตรวจใบหน้าไม่สำเร็จ กรุณาถ่ายใหม่อีกครั้ง", "retry_face": True}), 400
    _perf["embed"] = round((time.perf_counter() - _t8) * 1000, 2); _t9 = time.perf_counter()

    with stage("matching"):
        verify_result = verify_face_multi(live_embedding, stored_embeddings, face_threshold)
    score = verify_result["best_similarity"]
    _log.info(f"[FACE] best={score:.4f} avg={verify_result['avg_similarity']:.4f} "
              f"threshold={face_threshold} trusted={device_trusted} pass={verify_result['verified']}")

    if not verify_result["verified"]:
        reject("face_mismatch")
        return jsonify({
            "ok": False,
            "error": "ใบหน้าไม่ตรง — กรุณาถ่ายรูปใหม่",
            "retry_face": True,
        }), 400
    _perf["verify"] = round((time.perf_counter() - _t9) * 1000, 2); _t10 = time.perf_counter()

    # ─── 7. Duplicate check-in guard ─────────────────────────────────────────
    with stage("attendance_lookup"):
        dup = (
            supabase_admin.table("attendance")
            .select("id, status")
            .eq("session_id", session_id)
            .eq("student_id", student_id)
            .maybe_single()
            .execute()
        )
    if dup and dup.data:
        reject("already_checked")
        return jsonify({"ok": False, "already_checked": True, "error": f"เช็คชื่อแล้ว (สถานะ: {dup.data['status']})"}), 400

    # ─── 8. Determine attendance status (present / late) ─────────────────────
    now = datetime.now(timezone.utc)
    start_time_str = sess.get("start_time", "")
    status = "present"
    if start_time_str:
        start_time = dtparser.parse(start_time_str)
        if now > start_time and (now - start_time).total_seconds() > 900:
            status = "late"

    # ─── 9. Insert attendance record ─────────────────────────────────────────
    # ใช้ upsert + on_conflict เพื่อป้องกัน TOCTOU race condition:
    # ถ้า 2 requests เข้าพร้อมกัน ผ่าน duplicate check แล้ว insert พร้อมกัน
    # DB unique constraint บน (session_id, student_id) จะ reject request ที่ 2
    try:
        supabase_admin.table("attendance").insert({
            "session_id":      session_id,
            "student_id":      student_id,
            "ble_rssi":        ble_rssi,  # already int or None from validation above
            "ble_pass":        ble_pass,
            "liveness_pass":   server_liveness_pass,
            "liveness_action": liveness_action or "",
            "face_score":      round(score, 4),
            "face_pass":       True,
            "status":          status,
            "check_in_at":     now.isoformat(),
            "device_id":       device_id or None,
        }).execute()
    except Exception as insert_err:
        err_str = str(insert_err)
        _log.warning(f"[CHECKIN] insert failed (possible duplicate): {type(insert_err).__name__}")
        # Duplicate key violation (PostgreSQL error code 23505)
        if "23505" in err_str or "duplicate" in err_str.lower() or "unique" in err_str.lower():
            reject("attendance_duplicate")
            return jsonify({"ok": False, "already_checked": True,
                            "error": "เช็คชื่อแล้ว"}), 400
        reject("attendance_insert_failed")
        return jsonify({"ok": False, "error": "บันทึกข้อมูลไม่สำเร็จ กรุณาลองใหม่"}), 500
    _perf["db"] = round((time.perf_counter() - _t10) * 1000, 2)
    _perf["total"] = round((time.perf_counter() - _t0) * 1000, 2)
    _log.info(
        "[PERF] total={total}ms validate={validate}ms session={session}ms ble={ble}ms "
        "ear={ear}ms moire={moire}ms texture={texture}ms temporal={temporal}ms "
        "fasnet={fasnet}ms onnx={onnx}ms embed={embed}ms verify={verify}ms db={db}ms".format(**_perf)
    )

    status_label = "มาเรียน" if status == "present" else "มาสาย"
    return jsonify({"ok": True, "message": f"เช็คชื่อสำเร็จ — {status_label}"})


# ─── Passive anti-spoof (hybrid liveness) ────────────────────────────────────

@api_checkin_bp.route("/api/antispoof-passive", methods=["POST"])
@login_required
@role_required("student")
@_limiter.limit("20 per minute")
@csrf_protect
def antispoof_passive():
    data       = request.get_json()
    face_image = data.get("face_image") if data else None
    if not face_image:
        return jsonify({"ok": False, "real": False, "score": 0.0}), 400
    try:
        is_real, score = check_anti_spoof_with_score(face_image)
        return jsonify({"ok": True, "real": is_real, "score": round(score, 4)})
    except Exception as e:
        _log.error(f"[ANTISPOOF-PASSIVE] error: {type(e).__name__}")
        # Fail-close: exception → treat as spoof, not real
        return jsonify({"ok": False, "real": False, "score": 0.0,
                        "message": "ไม่สามารถตรวจสอบได้ กรุณาลองใหม่"}), 500


def _eligible(data):
    student_id = session["user_id"]
    session_id = data.get("session_id")
    proximity_method = current_app.config.get("CHECKIN_PROXIMITY_METHOD", "totp")
    # ─── 1. Verify session is still open ─────────────────────────────────────
    with stage("session_lookup"):
        sess_res = (
            supabase_admin.table("sessions")
            .select("id, course_id, is_open, beacon_id, start_time, end_time, checkin_duration, beacons(rssi_threshold, ble_room_code)")
            .eq("id", session_id)
            .maybe_single()
            .execute()
        )
    if not sess_res or not sess_res.data:
        reject("session_missing")
        return None, (jsonify({"ok": False, "error": "ไม่พบ session"}), 404)
    sess = sess_res.data
    if not sess.get("is_open"):
        reject("session_closed")
        return None, (jsonify({"ok": False, "error": "คาบเรียนนี้ปิดการเช็คชื่อแล้ว"}), 400)

    # ─── 1c. BLE room proximity check (replaces TOTP when enabled) ───────────
    if proximity_method == "ble":
        beacon = sess.get("beacons") or {}
        configured_room = (beacon.get("ble_room_code") or "").strip()
        if not configured_room:
            _log.error(f"[BLE_ROOM] beacon not configured for session={session_id}")
            reject("ble_beacon_unconfigured")
            return None, (jsonify({"ok": False, "error_code": "room_code_unavailable",
                            "error": "ระบบตรวจสอบตำแหน่งห้องเรียนขัดข้องชั่วคราว กรุณาแจ้งอาจารย์",
                            "retry_room_code": True}), 503)
        submitted_room = data.get("room_code")
        submitted_room = submitted_room.strip() if isinstance(submitted_room, str) else ""
        if not submitted_room:
            reject("ble_room_code_missing")
            return None, (jsonify({"ok": False, "error_code": "room_code_invalid",
                            "error": "กรุณากดหาอุปกรณ์ในห้องเรียนก่อนเช็คชื่อ",
                            "retry_room_code": True}), 400)
        if submitted_room != configured_room:
            _log.warning(f"[BLE_ROOM] mismatch student={student_id} session={session_id}")
            reject("ble_room_mismatch")
            return None, (jsonify({"ok": False, "error_code": "room_code_invalid",
                            "error": "อุปกรณ์ที่เชื่อมต่อไม่ใช่ของห้องเรียนนี้ กรุณาลองใหม่ในห้องที่ถูกต้อง",
                            "retry_room_code": True}), 400)
        event("ble_room", "pass")

    # A session id is not authorization. The student must belong to the course.
    with stage("enrollment_lookup"):
        enrollment = (
            supabase_admin.table("course_enrollments")
            .select("id")
            .eq("course_id", sess["course_id"])
            .eq("student_id", student_id)
            .maybe_single()
            .execute()
        )
    if not enrollment or not enrollment.data:
        _log.warning(
            f"[CHECKIN] student={student_id} attempted session={session_id} "
            "without course enrollment"
        )
        reject("course_not_enrolled")
        return None, (jsonify({"ok": False, "error": "คุณไม่ได้ลงทะเบียนในรายวิชานี้"}), 403)

    # ─── Check-in window (checkin_duration minutes from start) ───────────────
    checkin_duration = sess.get("checkin_duration")
    if checkin_duration and sess.get("start_time"):
        from datetime import timedelta
        open_at  = dtparser.parse(sess["start_time"])
        deadline = open_at + timedelta(minutes=int(checkin_duration))
        if datetime.now(timezone.utc) > deadline:
            reject("checkin_deadline_exceeded")
            return None, (jsonify({"ok": False, "error": f"หมดเวลาเช็คชื่อแล้ว (รับ {checkin_duration} นาที)"}), 400)
    return sess, None


def _receipt_message(reason):
    if reason == "proximity_receipt_expired":
        return "ผลยืนยันตำแหน่งห้องเรียนหมดอายุ กรุณายืนยันใหม่ก่อนเช็คชื่อ"
    if reason == "proximity_verifier_error":
        return "ระบบตรวจสอบตำแหน่งห้องเรียนขัดข้องชั่วคราว กรุณาลองใหม่อีกครั้ง"
    return "กรุณายืนยันตำแหน่งห้องเรียนใหม่ก่อนเช็คชื่อ"


def _verify_receipt(data):
    try:
        reason = proximity_receipt.verify(data.get("proximity_receipt"),
            current_app.config["PROXIMITY_RECEIPT_SECRET"], session["user_id"],
            data.get("session_id"), current_app.config.get("CHECKIN_PROXIMITY_METHOD", "totp"),
            data.get("room_code"))
    except Exception:
        reason = "proximity_verifier_error"
    if reason:
        reject(reason)
        return jsonify(ok=False, error_code=reason, retry_room_code=True,
                       error=_receipt_message(reason)), 503 if reason == "proximity_verifier_error" else 400
    event("proximity_receipt", "pass")


@api_checkin_bp.route("/api/checkin/proximity", methods=["POST"])
@audited_checkin
@login_required
@role_required("student")
@_limiter.limit("5 per minute")
@csrf_protect
def checkin_proximity():
    data = request.get_json()
    if not isinstance(data, dict) or not data:
        reject("payload_empty")
        return jsonify(ok=False, error="ไม่พบข้อมูล"), 400
    proximity_method = current_app.config.get("CHECKIN_PROXIMITY_METHOD", "totp")
    # Room possession factor. TOTP: cheap, checked before any DB/costly work,
    # exactly as before. BLE: needs the session's beacon, so it happens in
    # §1c below, right after the session is loaded — see that block.
    if proximity_method == "totp":
        # Configuration is validated at boot; unexpected verifier failures are F-16 503.
        try:
            room_code_valid = verify_code(data.get("room_code"), current_app.config["ESP32_TOTP_SECRET"])
        except Exception:
            event("totp", "verifier_error")
            _log.error("[ROOM_CODE] verifier system failure")
            reject("totp_verifier_error")
            return jsonify({"ok": False, "error_code": "room_code_unavailable",
                            "error": "ระบบตรวจสอบรหัสห้องขัดข้องชั่วคราว กรุณาลองใหม่อีกครั้ง หรือแจ้งอาจารย์หากยังพบปัญหา",
                            "retry_room_code": True}), 503
        event("totp", "pass" if room_code_valid else "wrong_or_stale")
        if not room_code_valid:
            message = ("กรุณากรอกรหัสห้อง 6 หลักจากจอในห้องเรียน" if not data.get("room_code") else
                       "รหัสห้องไม่ถูกต้องหรือหมดอายุ กรุณาดูรหัสปัจจุบันจากจอในห้องเรียนแล้วกรอกใหม่")
            reject("totp_invalid")
            return jsonify({"ok": False, "error_code": "room_code_invalid",
                            "error": message, "retry_room_code": True}), 400

    try:
        sess, error = _eligible(data)
        if error is not None:
            return error
        token = proximity_receipt.issue(current_app.config["PROXIMITY_RECEIPT_SECRET"],
            session["user_id"], data.get("session_id"), proximity_method, data.get("room_code"))
    except Exception as error:
        reject("proximity_verifier_error", exception_type=type(error).__name__)
        return jsonify(ok=False, error=_receipt_message("proximity_verifier_error")), 503
    event("proximity_preflight", "pass")
    response = jsonify(ok=True, proximity_receipt=token, expires_in=proximity_receipt.TTL_SECONDS)
    response.headers["Cache-Control"] = "no-store"
    return response

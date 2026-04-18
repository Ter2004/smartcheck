import logging
from datetime import datetime, timezone
from dateutil import parser as dtparser
from flask import Blueprint, request, jsonify, session, current_app
from app.routes.auth import login_required, role_required
from app import supabase_admin

_log = logging.getLogger("smartcheck.checkin")
from app.services.face_service import (
    extract_embedding, verify_face_multi,
    check_anti_spoof, check_anti_spoof_with_score,
    detect_screen_moire, _decode_image, server_validate_frame,
)
from app.services.security_service import (
    verify_device_token, verify_embedding_integrity, csrf_protect,
)
from app import limiter as _limiter

api_checkin_bp = Blueprint("api_checkin", __name__)

# ─── Face verification thresholds (FaceNet512) ───────────────────────────────
FACE_THRESHOLD_TRUSTED = 0.70   # trusted / previously bound device
FACE_THRESHOLD_NEW     = 0.80   # new or unbound device


@api_checkin_bp.route("/api/checkin", methods=["POST"])
@login_required
@role_required("student")
@_limiter.limit("5 per minute")
@csrf_protect
def checkin():
    student_id = session["user_id"]
    data = request.get_json()

    if not data:
        return jsonify({"ok": False, "error": "ไม่พบข้อมูล"}), 400

    session_id      = data.get("session_id")
    ble_rssi        = data.get("ble_rssi")
    liveness_pass   = data.get("liveness_pass", False)
    liveness_action = data.get("liveness_action", "") or ""
    face_image      = data.get("face_image")
    ear_samples     = data.get("ear_samples") or []

    # M7: whitelist liveness_action — reject arbitrary strings
    _ALLOWED_LIVENESS_ACTIONS = {"", "blink", "nod", "turn_left", "turn_right", "smile", "raise_eyebrows"}
    if liveness_action not in _ALLOWED_LIVENESS_ACTIONS:
        return jsonify({"ok": False, "error": "ข้อมูลไม่ถูกต้อง"}), 400

    # M6: validate ble_rssi before int() conversion
    if ble_rssi is not None:
        try:
            ble_rssi = int(ble_rssi)
            if not (-120 <= ble_rssi <= 0):
                ble_rssi = None  # out of realistic RSSI range — ignore silently
        except (ValueError, TypeError):
            return jsonify({"ok": False, "error": "ข้อมูล BLE ไม่ถูกต้อง"}), 400

    if not all([session_id, face_image]):
        return jsonify({"ok": False, "error": "ข้อมูลไม่ครบ"}), 400

    # ─── 0. Device token verification (Sprint 1B) — cheapest check first ──────
    raw_token = request.headers.get("Authorization", "").replace("DeviceToken ", "").strip()
    device_payload = verify_device_token(raw_token, current_app.config["SECRET_KEY"])
    # M4: distinguish 3 cases — (a) no token, (b) invalid token, (c) valid wrong user
    if raw_token and device_payload is None:
        # Token was provided but failed verification (invalid/expired/tampered)
        _log.warning(f"[DEVICE_TOKEN] invalid token rejected student={student_id}")
        return jsonify({"ok": False, "error": "Device token ไม่ถูกต้อง"}), 403
    if device_payload is not None and device_payload.get("uid") != student_id:
        # Token is valid but belongs to a different user — reject immediately
        return jsonify({"ok": False, "error": "Device token ไม่ตรงกับบัญชีนี้"}), 403
    # device_payload=None + no raw_token = legacy / first check-in — allowed

    # ─── 0b. Zero-trust frame validation (Sprint 2A) ─────────────────────────
    frame_check = server_validate_frame(face_image)
    if not frame_check["valid"]:
        _log.info(f"[FRAME_VALIDATE] fail reason={frame_check['reason']} meta={frame_check['metadata']}")
        return jsonify({
            "ok":        False,
            "error":     "รูปภาพไม่ถูกต้อง — กรุณาถ่ายใหม่อีกครั้ง",
            "retry_face": True,
        }), 400

    # ─── 1. Verify session is still open ─────────────────────────────────────
    sess_res = (
        supabase_admin.table("sessions")
        .select("id, is_open, beacon_id, start_time, end_time, checkin_duration, beacons(rssi_threshold)")
        .eq("id", session_id)
        .maybe_single()
        .execute()
    )
    if not sess_res or not sess_res.data:
        return jsonify({"ok": False, "error": "ไม่พบ session"}), 404
    sess = sess_res.data
    if not sess.get("is_open"):
        return jsonify({"ok": False, "error": "คาบเรียนนี้ปิดการเช็คชื่อแล้ว"}), 400

    # ─── Check-in window (checkin_duration minutes from start) ───────────────
    checkin_duration = sess.get("checkin_duration")
    if checkin_duration and sess.get("start_time"):
        from datetime import timedelta
        open_at  = dtparser.parse(sess["start_time"])
        deadline = open_at + timedelta(minutes=int(checkin_duration))
        if datetime.now(timezone.utc) > deadline:
            return jsonify({"ok": False, "error": f"หมดเวลาเช็คชื่อแล้ว (รับ {checkin_duration} นาที)"}), 400

    # ─── 2. BLE RSSI check ───────────────────────────────────────────────────
    if current_app.config.get("BLE_CHECK_ENABLED", False):
        rssi_threshold = -70  # dBm — must be within range
        ble_skip       = data.get("ble_skip", False)
        if not ble_skip and (ble_rssi is None or ble_rssi < rssi_threshold):
            _log.warning(f"[BLE] RSSI fail: rssi={ble_rssi} threshold={rssi_threshold}")
            return jsonify({"ok": False, "error": "ไม่พบสัญญาณ Beacon ในห้องเรียน"}), 400
        ble_pass = True
    else:
        _log.debug("[BLE] check skipped (BLE_CHECK_ENABLED=false)")
        ble_pass = True

    # ─── 3. Server-side EAR liveness check ──────────────────────────────────
    if not ear_samples:
        _log.warning("[LIVENESS] ear_samples missing — passing with warn (lenient mode)")
    else:
        try:
            import numpy as _np
            ear_arr = _np.array(ear_samples, dtype=float)
            ear_std = float(_np.std(ear_arr))
            ear_min = float(_np.min(ear_arr))
            _log.info(f"[LIVENESS] ear std={ear_std:.4f} min={ear_min:.4f} n={len(ear_arr)}")
            if ear_std < 0.03 or ear_min >= 0.18:
                return jsonify({
                    "ok":        False,
                    "error":     "ไม่ผ่านการตรวจสอบความมีชีวิต — กรุณากะพริบตาตามธรรมชาติขณะเช็คชื่อ",
                    "retry_face": True,
                }), 400
        except Exception as ear_err:
            _log.warning(f"[LIVENESS] EAR validation error (passing): {ear_err}")

    # ─── 4a. Moiré / screen-replay detection (FFT — faster than MiniFASNet) ───
    try:
        raw_frame   = _decode_image(face_image)
        moire       = detect_screen_moire([raw_frame])
        _log.info(f"[MOIRE] avg_score={moire['avg_score']} is_screen={moire['is_screen']} threshold={moire.get('threshold')}")
        if moire["is_screen"]:
            return jsonify({
                "ok":    False,
                "error": "ตรวจพบหน้าจอมือถือ — กรุณาใช้ใบหน้าจริงเท่านั้น",
                "spoof": True,
                "retry_face": True,
            }), 400
    except Exception as moire_err:
        _log.error(f"[MOIRE] check error (fail-close): {moire_err}")
        return jsonify({
            "ok": False,
            "error": "ไม่สามารถตรวจสอบภาพได้ — กรุณาถ่ายใหม่อีกครั้ง",
            "retry_face": True,
        }), 400

    # ─── 4b. Anti-spoofing via MiniFASNet ────────────────────────────────────
    try:
        is_real = check_anti_spoof(face_image)
        if not is_real:
            return jsonify({
                "ok": False,
                "error": "ตรวจพบรูปถ่ายหรือหน้าจอ — กรุณาใช้ใบหน้าจริงเท่านั้น",
                "spoof": True,
                "retry_face": True,
            }), 400
    except Exception as e:
        _log.error(f"[ANTISPOOF] check error (fail-close): {e}")
        return jsonify({
            "ok": False,
            "error": "ไม่สามารถตรวจสอบใบหน้าได้ — กรุณาถ่ายใหม่อีกครั้ง",
            "retry_face": True,
        }), 400

    # ─── 5. Device binding (determines threshold) ─────────────────────────────
    device_id = request.headers.get("X-Device-ID", "")
    user_res  = (
        supabase_admin.table("users")
        .select("device_id")
        .eq("id", student_id)
        .maybe_single()
        .execute()
    )
    user_device = ((user_res and user_res.data) or {}).get("device_id") or ""
    if user_device and device_id and user_device != device_id:
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
            return jsonify({"ok": False, "error": "อุปกรณ์นี้ถูกผูกกับบัญชีอื่นแล้ว"}), 403
        supabase_admin.table("users").update({"device_id": device_id}).eq("id", student_id).execute()

    # Sprint 1B: HMAC device token counts as trusted regardless of DB binding
    token_trusted  = device_payload is not None   # cryptographic proof of device
    db_trusted     = bool(user_device and device_id and user_device == device_id)
    device_trusted = token_trusted or db_trusted
    face_threshold = FACE_THRESHOLD_TRUSTED if device_trusted else FACE_THRESHOLD_NEW

    # ─── 6. Face verification (multi-embedding) ───────────────────────────────
    bio_res = (
        supabase_admin.table("student_biometrics")
        .select("face_embeddings, integrity_hash")
        .eq("user_id", student_id)
        .maybe_single()
        .execute()
    )
    if not bio_res or not bio_res.data:
        return jsonify({"ok": False, "error": "ยังไม่ได้ลงทะเบียนใบหน้า"}), 400

    stored_embeddings = bio_res.data.get("face_embeddings") or []
    if not stored_embeddings:
        return jsonify({"ok": False, "error": "ยังไม่ได้ลงทะเบียนใบหน้า"}), 400

    # Sprint 2B: verify embedding integrity before using them
    stored_hash = bio_res.data.get("integrity_hash") or ""
    if not verify_embedding_integrity(
        student_id, stored_embeddings, stored_hash,
        current_app.config["EMBEDDING_INTEGRITY_SALT"],
    ):
        _log.warning(f"[INTEGRITY] VIOLATION student={student_id}")
        return jsonify({
            "ok":   False,
            "error": "ข้อมูลชีวมาตรไม่สมบูรณ์ — กรุณาลงทะเบียนใบหน้าใหม่อีกครั้ง",
        }), 403

    try:
        live_embedding = extract_embedding(face_image)
    except Exception as e:
        _log.warning(f"[FACE] extract_embedding failed: {e}")
        # L1: don't expose internal error details to client
        return jsonify({"ok": False, "error": "ตรวจใบหน้าไม่สำเร็จ กรุณาถ่ายใหม่อีกครั้ง", "retry_face": True}), 400

    verify_result = verify_face_multi(live_embedding, stored_embeddings, face_threshold)
    score = verify_result["best_similarity"]
    _log.info(f"[FACE] best={score:.4f} avg={verify_result['avg_similarity']:.4f} "
              f"threshold={face_threshold} trusted={device_trusted} pass={verify_result['verified']}")

    if not verify_result["verified"]:
        return jsonify({
            "ok": False,
            "error": "ใบหน้าไม่ตรง — กรุณาถ่ายรูปใหม่",
            "retry_face": True,
        }), 400

    # ─── 7. Duplicate check-in guard ─────────────────────────────────────────
    dup = (
        supabase_admin.table("attendance")
        .select("id, status")
        .eq("session_id", session_id)
        .eq("student_id", student_id)
        .maybe_single()
        .execute()
    )
    if dup and dup.data:
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
            "liveness_pass":   False,
            "liveness_action": liveness_action or "",
            "face_score":      round(score, 4),
            "face_pass":       True,
            "status":          status,
            "check_in_at":     now.isoformat(),
            "device_id":       device_id or None,
        }).execute()
    except Exception as insert_err:
        err_str = str(insert_err)
        _log.warning(f"[CHECKIN] insert failed (possible duplicate): {err_str[:120]}")
        # Duplicate key violation (PostgreSQL error code 23505)
        if "23505" in err_str or "duplicate" in err_str.lower() or "unique" in err_str.lower():
            return jsonify({"ok": False, "already_checked": True,
                            "error": "เช็คชื่อแล้ว"}), 400
        return jsonify({"ok": False, "error": "บันทึกข้อมูลไม่สำเร็จ กรุณาลองใหม่"}), 500

    status_label = "มาเรียน" if status == "present" else "มาสาย"
    return jsonify({"ok": True, "message": f"เช็คชื่อสำเร็จ — {status_label}"})


# ─── Passive anti-spoof (hybrid liveness) ────────────────────────────────────

@api_checkin_bp.route("/api/antispoof-passive", methods=["POST"])
@login_required
@role_required("student")
@_limiter.limit("20 per minute")
def antispoof_passive():
    data       = request.get_json()
    face_image = data.get("face_image") if data else None
    if not face_image:
        return jsonify({"ok": False, "real": False, "score": 0.0}), 400
    try:
        is_real, score = check_anti_spoof_with_score(face_image)
        return jsonify({"ok": True, "real": is_real, "score": round(score, 4)})
    except Exception as e:
        _log.error(f"[ANTISPOOF-PASSIVE] error: {e}")
        # Fail-close: exception → treat as spoof, not real
        return jsonify({"ok": False, "real": False, "score": 0.0,
                        "message": "ไม่สามารถตรวจสอบได้ กรุณาลองใหม่"}), 500

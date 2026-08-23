# Code Review: app/routes/student.py

ตรวจสอบ ณ วันที่ 2026-08-20  
ไฟล์: `app/routes/student.py` — 1 183 LOC  
บริบท: localhost เท่านั้น, ไม่มี ESP32, ยังไม่ deploy Railway

---

## Function Inventory

| ฟังก์ชัน | line | LOC | ถูกเรียกจากไหน | ต้นทุนต่อ call | คำตัดสิน | เหตุผล |
|---|---|---|---|---|---|---|
| `_safe_ip` | 31 | 8 | student.py:43 (`_log`), student.py:238 (`record_consent`) | ~0 ms | KEEP | ป้องกัน X-Forwarded-For spoofing ใน audit log |
| `_log` | 41 | 5 | student.py (ทุก step ทั่วไฟล์ — puluhan บรรทัด) | ~0 ms (I/O logging) | KEEP | D1: uniform audit log; เป็น glue ทั้งไฟล์ |
| `_cosine_sim` | 48 | 7 | student.py:627 (api_enroll continuity), student.py:836 (api_self_verify continuity) | ~0 ms | MERGE | โค้ด pure-Python cosine เหมือนกับ `face_service._cosine_sim` ทุกตัวอักษร; ควร import จาก face_service แทน |
| `dashboard` | 64 | 12 | auth.py:180, base.html:144, dashboard.html:21,76, enroll_face.html:98,272,347, checkin.html:59,148,193 | 1 SELECT (student_biometrics) | KEEP | ทำงานถูกต้อง; ฟังก์ชัน thin |
| `enroll_face` | 81 | 17 | base.html:154, dashboard.html:21,76 | 1 SELECT + session.pop ×2 | KEEP | reset server-side retry counters บน page load — ถูกต้อง |
| `checkin` | 103 | 109 | base.html (nav link), auth.py ผ่าน redirect หลัง login | **7 SELECT** (student_biometrics, sessions, course_enrollments, attendance, schedules, sessions, attendance) | REFACTOR | GET page route หนักเกิน; 7 sequential Supabase calls บน localhost ≈ 300–600 ms |
| `record_consent` | 223 | 23 | enrollment_flow.js:318 via `ENROLL_CONFIG.consentUrl` (enroll_face.html:444) | 1 INSERT (consent_logs) non-fatal + session write | KEEP | PDPA audit trail; non-fatal INSERT ถูกต้อง |
| `api_enroll` | 257 | 475 | enrollment_flow.js:1290, enrollment_circular.js:443 via `ENROLL_CONFIG.enrollUrl` | 3+ SELECT/RPC + 5×(Moiré+Texture+MiniFASNet+FaceNet512) ≈ **5–15 s** | SPLIT | 475 LOC, 15 pipeline steps ในฟังก์ชันเดียว — อ่าน/debug/test ยากมาก |
| `api_self_verify` | 743 | 201 | `ENROLL_CONFIG.selfVerifyUrl` นิยามใน enroll_face.html:446 — **ไม่พบ `fetch()` ใน .js ไฟล์ใดเลย** | server_validate_frame + check_anti_spoof + extract_embedding + 2 SELECT/UPDATE | DELETE (**ตรวจก่อน**) | api_enroll:687 บอก "self-verify step removed — finalize immediately"; endpoint ยังค้างอยู่แต่ไม่มีผู้เรียกจาก JS |
| `api_spoof_check` | 955 | 127 | enrollment_flow.js:229, enrollment_circular.js:379 via `ENROLL_CONFIG.spoofCheckUrl` | server_validate_frame + _decode_image + Moiré + Texture + temporal + spoof_check_with_embedding ≈ **200–800 ms** | KEEP | เป็น liveness_embeddings populator สำหรับ continuity check; ใจกลาง F-6 mitigation |
| `api_reset_liveness` | 1088 | 6 | enrollment_flow.js:1463 via `ENROLL_CONFIG.resetLivenessUrl` | session.pop ×2 | KEEP | จำเป็นสำหรับ fullRestart; ขนาดเหมาะสม |
| `api_withdraw_consent` | 1107 | 76 | **ไม่พบ `fetch()`, `form action`, หรือ link ใน .js หรือ .html ใดเลย** | 2 Supabase calls (INSERT + UPDATE) + log_audit_event | REFACTOR | มี endpoint แต่ไม่มี front-end trigger — ผู้ใช้ไม่สามารถใช้สิทธิ์ PDPA ได้จาก UI จริง |

---

## Server-side Re-validation

ทุก client-sent value ที่ server ได้รับและวิธีที่ server ตรวจสอบ:

| Client value | Endpoint | Server validation | สถานะ |
|---|---|---|---|
| `face_images` (list base64) | /api/enroll | `server_validate_frame` ทุก frame (size ≤ 800 KB, magic bytes JPEG, blur, color variance); exact count == 5 | ✅ |
| `baseline_ear` (float) | /api/enroll | Range check: 0.0 < x < 1.0; `ValueError` caught; ค่านอกช่วง → `None` | ⚠️ เก็บใน DB แต่ไม่ใช้เป็น gate security |
| `ear_std` (float) | /api/enroll | logged only; gate ถูก disable (บรรทัด 506-507: "blocking disabled for passive capture") | ⚠️ informational เท่านั้น |
| `flow_mode` (str) | /api/enroll | **ไม่มี whitelist** — ถ้า == `"circular"` ลด consistency_threshold จาก 0.80 → 0.75; ค่าอื่นๆ ใช้ 0.80 | ⚠️ **client ควบคุม threshold ได้** (ดูรายละเอียดใน Top 3) |
| `retry_count` (int) | /api/enroll | **ไม่อ่านจาก client** — ใช้ `session["enroll_retry"]` เสมอ | ✅ |
| `face_image` (base64) | /api/self_verify | `server_validate_frame` + `check_anti_spoof` + `extract_embedding` | ✅ |
| `device_fingerprint` (str) | /api/self_verify | ไม่มีการ validate format/length — ส่งตรงเข้า `create_device_token` | ⚠️ ไม่มีขอบเขต input |
| `image` (base64) | /api/spoof_check | `server_validate_frame` + `_decode_image` | ✅ |
| (body ว่าง) | /api/consent | user_id จาก session เท่านั้น — ไม่อ่าน body | ✅ |
| (body ว่าง) | /api/reset-liveness | ไม่อ่าน body | ✅ |
| (body ว่าง) | /api/withdraw-consent | ไม่อ่าน body | ✅ |

---

## Enrollment Pipeline Order

ขั้นตอนใน `api_enroll` ตามลำดับที่รัน พร้อม fail mode:

| # | ขั้นตอน | fail mode | บรรทัด |
|---|---|---|---|
| 1a | Consent check — session (`consent_given_at`) | → 400 | 291 |
| 1b | Consent check — DB (latest consent_logs row) | fail-close: DB error → 500 deny | 298–319 |
| 1c | Input: frame count == 5 | → 400 | 339 |
| 1d | Input: `baseline_ear` range + type | → 400 ถ้า malformed | 326–335 |
| 1e | Retry limit: `session["enroll_retry"]` ≥ 3 | → 400 | 343–349 |
| 2 | DB attempt limit: `atomic_enroll_attempt` RPC (max 5/24h) | fail-close: DB error → 500 deny | 352–378 |
| 3 | Zero-trust frame validation: `server_validate_frame` ×5 | → 400 per frame | 381–391 |
| 4 | Pre-duplicate check (vs `session["liveness_embeddings"]`) | non-fatal: crash → continue (step 14 เป็น definitive) | 393–439 |
| 5 | Decode all 5 frames (`_decode_image`) | → 400 | 442–445 |
| 6 | Moiré FFT: `detect_screen_moire` all 5 frames | fail-close: exception → 400 block | 447–465 |
| 7 | Screen Texture: `detect_screen_texture` all 5 frames (≥2/5) | fail-close: exception → 400 block | 467–483 |
| 8 | Temporal variance: `detect_static_image` | fail-close: exception → 400 block | 485–500 |
| 9 | EAR std (client-reported) | logged only — ไม่ block | 505–507 |
| 10 | MiniFASNet: `check_anti_spoof` ×5 (ต้องผ่าน ≥4/5) | fail-close: exception = fail | 509–536 |
| 11 | FaceNet512 extraction: `extract_embedding` ×5 | 0 ผ่าน → error; ≥2 fail → spoof; 1 fail → retry | 538–577 |
| 12 | Embedding consistency: `check_embedding_consistency` | multi-outlier → restart_capture; single → need_more | 579–611 |
| 13 | Face continuity vs `session["liveness_embeddings"]` | hard block ถ้า empty หรือ cosine < CONTINUITY_THRESHOLD | 613–638 |
| 14 | Duplicate face check (batch 50, all enrolled) | → 400 duplicate | 640–685 |
| 15 | Save to DB (upsert student_biometrics) + image upload | image upload non-fatal | 687–731 |

**หมายเหตุ**: api_enroll docstring (บรรทัด 261–278) ล้าสมัย — ระบุ 9 ขั้นตอน แต่โค้ดจริงมี 15 ขั้นตอน (เพิ่ม pre-duplicate, temporal, EAR, continuity ทีหลัง)

---

## F-6 Server Side

**คำถาม**: ถ้า client block `/api/spoof_check` ระหว่าง liveness steps (ตามที่วิเคราะห์ใน 03-enrollment-flow.md) — server ตรวจพบได้หรือไม่?

### สิ่งที่ server ทำได้

**1. Continuity check (step 13) บล็อก "full block" attack**

ถ้าผู้โจมตี block `/api/spoof_check` *ทุก* call (Steps 2, 3, และ 4):
- `session["liveness_embeddings"]` จะว่างเปล่าตลอด
- api_enroll:618 → `"no liveness embeddings in session"` → return 400
- **Attack ถูกบล็อก**

**2. Independent spoof pipeline ใน api_enroll ไม่ขึ้นกับ api_spoof_check**

api_enroll รัน Moiré + Texture + Temporal + MiniFASNet ซ้ำอิสระบน 5 frames ที่ส่งมา — ผลของ `/api/spoof_check` ไม่มีผลต่อ steps เหล่านี้เลย การ block `/api/spoof_check` **ไม่ bypass** anti-spoof ใน api_enroll

### ช่องโหว่ที่ยังเหลือ

**"Partial block" attack** (จาก 03-enrollment-flow.md F-6):
1. Block `/api/spoof_check` ระหว่าง Steps 2–3 (liveness challenge)
2. Unblock ระหว่าง Step 4 (capture phase, 5 calls)
3. Capture-phase calls สำเร็จ → `liveness_embeddings` ถูก populate ด้วย capture frames
4. api_enroll step 13: `liveness_embeddings` ไม่ว่าง → continuity check ผ่าน (เทียบ embed ชุดเดียวกัน)

**สิ่งที่ server ไม่รู้**: server ไม่รู้ว่า `liveness_embeddings` มาจาก liveness phase หรือ capture phase — มันเป็น sliding window 5 most-recent (api_spoof_check:1069)

**ผลจริง**: Interactive liveness challenge (blink, head-turn EAR) — client-side เท่านั้น ไม่มีหลักฐานบน server ว่า challenge นี้ถูกทำจริง `ear_std` ที่รับมาจาก client ไม่ถูก gate (student.py:506-507)

### สรุป exposure ระดับ server

| เรื่อง | ถูก F-6 bypass ไหม |
|---|---|
| Moiré FFT (5 frames ใน api_enroll) | ❌ ไม่ bypass — รัน independent |
| Screen Texture (5 frames) | ❌ ไม่ bypass |
| Temporal variance (5 frames) | ❌ ไม่ bypass |
| MiniFASNet (5 frames) | ❌ ไม่ bypass |
| Continuity check (liveness_embeddings) | ⚠️ บล็อก "full block"; แต่ "partial block" (liveness skip + capture unblock) ผ่านได้ |
| Interactive blink/head-pose check | ✅ bypass ได้ — client-side เท่านั้น |

---

## Top 3

### 1. `flow_mode` client-controlled consistency threshold — REFACTOR

**บรรทัด**: api_enroll:583–584  
```python
_flow_mode = data.get("flow_mode", "classic")
_consistency_threshold = 0.75 if _flow_mode == "circular" else 0.80
```
Client ส่ง `"flow_mode": "circular"` ใน POST body → server ลด embedding consistency threshold จาก **0.80 → 0.75** โดยไม่ตรวจสอบ ไม่ match กับ `ENROLL_FLOW_MODE` env var ที่กำหนดใน config จริง

ผลกระทบ: ผู้ใช้ในโหมด classic สามารถลด threshold ด้วยตัวเองได้ผ่าน DevTools → อนุมัติ embedding ที่ inconsistent ขึ้นเล็กน้อย

แก้ไข: ใช้ `current_app.config["ENROLL_FLOW_MODE"]` แทน `data.get("flow_mode")`

### 2. SPLIT `api_enroll` (475 LOC, 15 steps) — SPLIT

ฟังก์ชันยาวสุดในระบบ (475 LOC) ครอบคลุม 15 pipeline steps ที่แตกต่างกัน อ่าน/debug/test ยาก Steps ควรถูก extract เป็น private helpers เช่น `_validate_consent()`, `_run_spoof_pipeline()`, `_check_continuity()`, `_save_enrollment()` ทำให้สามารถ unit test แต่ละ step ได้แยกกัน

### 3. `api_withdraw_consent` ไม่มี front-end trigger — REFACTOR

Endpoint ทำงานถูกต้อง (PDPA-compliant, audit trail ก่อน delete, fail-loud) แต่ไม่มี button/link/fetch ใน .html หรือ .js ใดเลย — ผู้ใช้ใช้สิทธิ์ PDPA Right to Withdraw ไม่ได้จาก UI จริง ควรเพิ่ม button ใน dashboard.html หรือ enroll_face.html

---

## ข้อสังเกตเพิ่มเติม

**`_cosine_sim` duplicate** (student.py:48): โค้ดเหมือนกับ `face_service` ทุกบรรทัด; ควร `from app.services.face_service import _cosine_sim` (ถ้า export) หรือ move ขึ้น shared utility

**`api_self_verify` dead endpoint** (student.py:743): api_enroll docstring บรรทัด 687 บอก "self-verify step removed — finalize immediately" แต่ endpoint 201 LOC ยังค้างอยู่ `ENROLL_CONFIG.selfVerifyUrl` นิยามใน template แต่ไม่มี fetch() ใน .js ใดเลย — ควรยืนยันแล้ว DELETE ถ้าไม่ใช้จริง

**`checkin` route 7 SELECT** (student.py:103): GET page รัน 7 Supabase calls บน localhost ≈ 300–600 ms ไม่ critical ตอนนี้ แต่ควร track เมื่อ scale

**docstring ล้าสมัย** (api_enroll:261–278): ระบุ 9 steps แต่โค้ดจริงมี 15 — ควร sync หลัง split

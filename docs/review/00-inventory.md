# Code Inventory — SmartCheck

สแกน ณ วันที่ 2026-08-20  
ข้ามไฟล์: `venv/`, `migrations/`, `__pycache__/`, ไฟล์ `__init__.py` ที่ว่างเปล่า

---

## ตารางไฟล์

| ไฟล์ | LOC | ฟังก์ชัน | หน้าที่ (1 บรรทัด) | กลุ่มโมดูล |
|---|---|---|---|---|
| `run.py` | 7 | 0 | Entry point — เรียก `create_app()` แล้วรัน dev server | Entry Point |
| `app/__init__.py` | 181 | 10 | App factory: เชื่อม Blueprint, Limiter, Session, Supabase, security headers | App Factory |
| `app/config.py` | 62 | 1 | โหลด env vars และ validate ใน production mode | Config |
| `app/models/user_model.py` | 14 | 2 | DB helper — ดึง user โดย id / email | Models |
| `app/routes/auth.py` | 181 | 8 | Login, logout, change-password, session fixation protection, auth decorators | Routes |
| `app/routes/admin.py` | 738 | 22 | CRUD ผู้ใช้/วิชา/Beacon/session, CSV import, biometrics admin | Routes |
| `app/routes/teacher.py` | 523 | 8 | Dashboard, session toggle, manual attendance override, CSV export | Routes |
| `app/routes/student.py` | 1 183 | 12 | Enrollment pipeline (consent → spoof → embed → verify), check-in page, PDPA withdrawal | Routes |
| `app/routes/api_checkin.py` | 415 | 2 | `POST /api/checkin` — pipeline เช็คชื่อ (device token → frame → BLE → liveness → face) | Routes |
| `app/services/face_service.py` | 887 | 20 | FaceNet512 embedding, 5-layer anti-spoof, Moiré/Texture/Temporal FFT, frame validation | Services |
| `app/services/security_service.py` | 211 | 7 | HMAC device token, embedding integrity hash, CSRF decorators, audit log helper | Services |
| `app/scheduler.py` | 143 | 4 | APScheduler: auto-create/close session ตาม schedule, keep-alive ping Supabase | Scheduler |
| `migrate_to_facenet512.py` | 78 | 0 | One-shot script: ล้าง embedding 128-D และ force re-enrollment ทุกคน | Scripts |
| `scripts/backfill_integrity_hash.py` | 81 | 1 | One-shot script: คำนวณ HMAC integrity_hash สำหรับ rows ที่ยังไม่มี | Scripts |
| `app/services/__init__.py` | 1 | 0 | (ว่าง) | — |
| `app/models/__init__.py` | 1 | 0 | (ว่าง) | — |
| `app/routes/__init__.py` | 1 | 0 | (ว่าง) | — |

**รวม:** ~3 708 LOC (ไม่นับ `__init__.py` ว่าง), 97 ฟังก์ชัน

---

## ลำดับการ Review (Risk × Call Frequency)

เรียงจากมากสุด → น้อยสุด

---

### 🔴 Tier 1 — ความเสี่ยงสูงมาก × ถูกเรียกทุก request

#### 1. `app/routes/api_checkin.py` (415 LOC)
**เหตุผล:** endpoint เดียวที่รับ face_image จากนักศึกษาทุกคนในทุกคาบ — ถ้ามี bug ที่นี่ นักศึกษาทุกคนได้รับผลกระทบทันที  
pipeline มี 9 ขั้นตอนตรวจ security เรียงกัน → ต้องตรวจว่าลำดับถูก, ไม่มี early-return ที่ข้ามขั้นตอน, TOCTOU ป้องกันครบ  
Rate limit 5/min ต่อ user → ตรวจว่า key function ถูก (`user_id` ไม่ใช่ IP)

#### 2. `app/services/face_service.py` (887 LOC)
**เหตุผล:** ไลบรารีหัวใจของทั้งระบบ — ถูกเรียกจาก enrollment (~8 ครั้งต่อ session) และ checkin ทุกครั้ง  
มี threshold hardcode หลายตัว (MOIRE, TEMPORAL, SPOOF_DECISION) → ต้องตรวจว่าค่าปัจจุบันสมดุลระหว่าง FRR/FAR  
`combined_spoof_score` มี fail-close logic หลายชั้น → ตรวจว่า weight renormalization ถูกต้องเมื่อ layer ใด layer หนึ่งตาย  
`server_validate_frame` เป็น gatekeeper แรก → ตรวจว่า threshold ไม่หลวมเกินหรือแน่นเกิน

#### 3. `app/services/security_service.py` (211 LOC)
**เหตุผล:** ถูก import โดยทุก route ที่มี protected endpoint  
`csrf_protect` และ `csrf_protect_form` ป้องกัน CSRF ทั้งระบบ → หาก bypass ได้ → ทุก POST ถูก forge  
`verify_embedding_integrity` ใช้ `hmac.compare_digest` → ต้องตรวจว่าเรียกถูกต้องและ `stored_hash = ""` ถูก reject  
`create_device_token` ใช้ `hmac.new` ไม่ใช่ `hmac.new(...)` → **typo** ที่ต้องตรวจ (ควรเป็น `hmac.new` → Python `hmac.new` ไม่มี, ต้อง `hmac.HMAC` หรือ `hmac.digest`)

---

### 🟠 Tier 2 — ความเสี่ยงสูง × ถูกเรียกทุก session

#### 4. `app/routes/student.py` (1 183 LOC)
**เหตุผล:** ไฟล์ยาวที่สุด — รวม `api_enroll` (9+ ขั้นตอน), `api_spoof_check` (~8 เรียก/session), `api_self_verify`, `api_withdraw_consent`  
`api_enroll` เป็น endpoint ที่ห้ามแตะโดยไม่ได้รับอนุมัติ → review อย่างระมัดระวัง  
`api_withdraw_consent` ต้องลบ biometric ก่อน return → ตรวจ order of operations  
มี `session["liveness_embeddings"]` ที่ใช้ state ใน session → ตรวจ race condition กรณีหลาย tab

#### 5. `app/__init__.py` (181 LOC)
**เหตุผล:** app factory — `create_app()` ตั้งค่า security headers, rate limiter, session, error handlers ครั้งเดียวสำหรับทุก request  
CSP header อนุญาต `unsafe-inline` + `unsafe-eval` → ตรวจว่าจำเป็นจริงหรือ scope กว้างเกินไป  
`reconnect_if_needed` เป็น `pass` placeholder → ตรวจว่า reconnect logic พอแล้วหรือยัง

---

### 🟡 Tier 3 — ความเสี่ยงปานกลาง × ถูกเรียกบ่อย (admin/teacher)

#### 6. `app/routes/auth.py` (181 LOC)
**เหตุผล:** login ทุก user ผ่านที่นี่ — session fixation prevention, CSRF token generation  
Password change flow ใช้ Supabase Admin API → ตรวจว่า error leak ไปสู่ client หรือไม่  
`role_required` decorator เป็น gatekeeper ทุก protected page → ตรวจ edge case เช่น session ขาดกลางคัน

#### 7. `app/routes/admin.py` (738 LOC)
**เหตุผล:** CSV import สร้าง Supabase Auth users โดยตรง → ตรวจ injection ผ่าน CSV fields  
`api_reset_enrollment` เป็น JSON API ที่ใช้ `@csrf_protect` (ไม่ใช่ form) → ตรวจว่า decorator ลำดับถูก  
`beacon_edit` รับ `int()` จาก form โดยไม่ try/except บาง field → ตรวจ ValueError

#### 8. `app/routes/teacher.py` (523 LOC)
**เหตุผล:** `override_attendance` เขียน audit log ทุกครั้ง → ตรวจว่า `log_audit_event` ถูกเรียกก่อน return  
`session_toggle` มี schedule window check → ตรวจ timezone edge case (DST, midnight)  
`export_csv` ใส่ BOM UTF-8 สำหรับ Excel → ตรวจ injection ผ่าน field ที่มี comma หรือ newline

---

### 🟢 Tier 4 — ความเสี่ยงต่ำ / ไม่ได้รันใน production path ปกติ

#### 9. `app/scheduler.py` (143 LOC)
background job รันทุก 1 นาที → ตรวจว่า error handling ป้องกัน loop crash และ double-start ป้องกันถูกต้อง

#### 10. `app/config.py` (62 LOC)
โหลด env vars — ตรวจ fallback ใน dev mode ไม่หลุดไป production

#### 11. `app/models/user_model.py` (14 LOC)
얇은 wrapper — ตรวจว่า `maybe_single()` handle `None` ได้ถูกต้องทุก caller

#### 12. `scripts/backfill_integrity_hash.py` (81 LOC)
one-shot script — ตรวจ `--dry-run` และ error handling รายแถว

#### 13. `migrate_to_facenet512.py` (78 LOC)
one-shot script — ตรวจว่า fallback เมื่อ `face_centroid` column ไม่มีทำงานถูกต้อง

---

## สรุป Tier

| Tier | ไฟล์ | LOC รวม | เหตุผลหลัก |
|---|---|---|---|
| 🔴 1 | api_checkin, face_service, security_service | 1 513 | รับ input จากภายนอก, crypto, ถูกเรียกทุก request |
| 🟠 2 | student, app/__init__ | 1 364 | enrollment pipeline ยาว, app-wide config |
| 🟡 3 | auth, admin, teacher | 1 442 | auth flow, CSV import, audit override |
| 🟢 4 | scheduler, config, models, scripts | 378 | background/utility, ไม่รับ untrusted input โดยตรง |

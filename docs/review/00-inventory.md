# Code Inventory — SmartCheck

สแกน ณ วันที่ 2026-08-20  
ข้ามไฟล์: `venv/`, `migrations/`, `__pycache__/`, ไฟล์ `__init__.py` ที่ว่างเปล่า

---

## Python

| ไฟล์ | LOC | หน้าที่ (1 บรรทัด) | กลุ่ม |
|---|---|---|---|
| `run.py` | 7 | Entry point — เรียก `create_app()` แล้วรัน dev server | Entry Point |
| `app/__init__.py` | 181 | App factory: เชื่อม Blueprint, Limiter, Session, Supabase, security headers | App Factory |
| `app/config.py` | 62 | โหลด env vars และ validate ใน production mode | Config |
| `app/models/user_model.py` | 14 | DB helper — ดึง user โดย id / email | Models |
| `app/routes/auth.py` | 181 | Login, logout, change-password, session fixation protection, auth decorators | Routes |
| `app/routes/admin.py` | 738 | CRUD ผู้ใช้/วิชา/Beacon/session, CSV import, biometrics admin | Routes |
| `app/routes/teacher.py` | 523 | Dashboard, session toggle, manual attendance override, CSV export | Routes |
| `app/routes/student.py` | 1 183 | Enrollment pipeline (consent → spoof → embed → verify), check-in page, PDPA withdrawal | Routes |
| `app/routes/api_checkin.py` | 415 | `POST /api/checkin` — pipeline เช็คชื่อ (device token → frame → BLE → liveness → face) | Routes |
| `app/services/face_service.py` | 887 | FaceNet512 embedding, 5-layer anti-spoof, Moiré/Texture/Temporal FFT, frame validation | Services |
| `app/services/security_service.py` | 211 | HMAC device token, embedding integrity hash, CSRF decorators, audit log helper | Services |
| `app/scheduler.py` | 143 | APScheduler: auto-create/close session ตาม schedule, keep-alive ping Supabase | Scheduler |
| `migrate_to_facenet512.py` | 78 | One-shot script: ล้าง embedding 128-D และ force re-enrollment ทุกคน | Scripts |
| `scripts/backfill_integrity_hash.py` | 81 | One-shot script: คำนวณ HMAC integrity_hash สำหรับ rows ที่ยังไม่มี | Scripts |

**รวม Python:** ~3 708 LOC

---

## JavaScript (`app/static/js/`)

| ไฟล์ | LOC | หน้าที่ (1 บรรทัด) | กลุ่ม |
|---|---|---|---|
| `enrollment_flow.js` | 1 518 | Classic enrollment UI: consent → light check → EAR baseline → liveness → 5-frame capture → `/api/enroll` | Enrollment |
| `enrollment_circular.js` | 645 | Circular enrollment UI: 5 pose zones + blink challenge; override `startLivenessChallenge` | Enrollment |
| `mediapipe_liveness.js` | 503 | MediaPipe FaceMesh wrapper — shared singleton `_sharedFM`, EAR, blink, head-pose helpers | Shared Lib |
| `checkin_flow.js` | 454 | Check-in UI: BLE scan → EAR liveness → frame capture → `POST /api/checkin` | Check-in |
| `ble_scanner.js` | 153 | Web Bluetooth BLE scanner — scan/connect beacons, expose `getBleResult()` | Shared Lib |
| `rt_analyze.js` | 135 | Real-time face analysis UI (admin biometrics preview) | Admin |
| `camera_guard.js` | 55 | Virtual camera detection guard — blocks OBS/manycam before enrollment | Shared Lib |

**รวม JS:** ~3 463 LOC

---

## HTML Templates (`app/templates/`)

| ไฟล์ | LOC | หน้าที่ (1 บรรทัด) | กลุ่ม |
|---|---|---|---|
| `student/enroll_face.html` | 458 | Enrollment page — โหลด enrollment_flow.js + enrollment_circular.js (conditional) | Student |
| `student/checkin.html` | 224 | Check-in page — โหลด checkin_flow.js + ble_scanner.js | Student |
| `admin/beacons.html` | 208 | Beacon CRUD UI + map/status display | Admin |
| `base.html` | 284 | Layout ฐาน: nav, flash messages, CSP meta, shared CSS/JS | Shared |
| `teacher/session_view.html` | 307 | รายชื่อนักศึกษาใน session + attendance override inline | Teacher |
| `admin/course_detail.html` | 305 | รายละเอียดวิชา — enrollment list, schedule, session history | Admin |
| `teacher/dashboard.html` | 273 | Teacher dashboard — วิชาทั้งหมด + session toggle buttons | Teacher |
| `auth/login.html` | 166 | Login form | Auth |
| `admin/courses.html` | 150 | Course list + create/edit modal | Admin |
| `admin/sessions.html` | 132 | Session list + manual open/close | Admin |
| `admin/users.html` | 127 | User list + role management | Admin |
| `teacher/history.html` | 126 | Attendance history per course | Teacher |
| `admin/import_result.html` | 124 | ผลการ import CSV (success/error per row) | Admin |
| `admin/biometrics.html` | 124 | Biometrics admin — reset enrollment, real-time preview | Admin |
| `student/dashboard.html` | 116 | Student dashboard — วิชาที่ลงทะเบียน + enrollment status | Student |
| `admin/import_csv.html` | 74 | CSV import form | Admin |
| `admin/dashboard.html` | 100 | Admin overview stats | Admin |
| `auth/change_password.html` | 40 | Change password form | Auth |
| `auth/register.html` | 48 | Register form (admin-created users) | Auth |

**รวม HTML:** ~3 570 LOC

---

**รวมทั้งโปรเจกต์:** ~10 741 LOC

---

## ลำดับการ Review (Risk × Call Frequency)

เรียงจากมากสุด → น้อยสุด

---

### 🔴 Tier 1 — ความเสี่ยงสูงมาก × ถูกเรียกทุก request

#### 1. `app/routes/api_checkin.py` (415 LOC)
**เหตุผล:** endpoint เดียวที่รับ face_image จากนักศึกษาทุกคนในทุกคาบ — pipeline 9 ขั้นตอนเรียงกัน → ต้องตรวจลำดับ, ไม่มี early-return ข้ามขั้น, TOCTOU ป้องกันครบ; rate limit key ต้องเป็น `user_id` ไม่ใช่ IP

#### 2. `app/services/face_service.py` (887 LOC)
**เหตุผล:** ไลบรารีหัวใจ — ถูกเรียกทุก enrollment และทุก check-in; `combined_spoof_score` มี fail-close/fail-open หลายชั้น; weight renormalization เมื่อ layer ตาย; threshold หลายตัว hardcode

#### 3. `app/services/security_service.py` (211 LOC)
**เหตุผล:** ถูก import โดยทุก route ที่มี protected endpoint; CSRF bypass → ทุก POST ถูก forge; `verify_embedding_integrity` ต้องตรวจ empty-hash rejection

#### 4. `app/static/js/enrollment_flow.js` (1 518 LOC)
**เหตุผล:** ไฟล์ JS ใหญ่สุด — ควบคุม UX enrollment ทั้งหมด; มี dead code 5 items ยืนยันแล้ว; `_deviceFingerprint` ไม่มี caller แต่ยังนิยามอยู่; liveness bypass ที่ client-side มีผลต่อ frame ที่ส่งขึ้น server

---

### 🟠 Tier 2 — ความเสี่ยงสูง × ถูกเรียกทุก session

#### 5. `app/routes/student.py` (1 183 LOC)
**เหตุผล:** ไฟล์ Python ยาวสุด — รวม `api_enroll` (security-critical, ห้ามแตะโดยไม่ได้รับอนุมัติ), `api_spoof_check`, `api_self_verify`, `api_withdraw_consent`; `liveness_embeddings` ใน session → ตรวจ race condition หลาย tab

#### 6. `app/static/js/enrollment_circular.js` (645 LOC)
**เหตุผล:** override `window.startLivenessChallenge` — ถ้า override ผิดจะ bypass liveness ทั้งหมด; ขึ้นอยู่กับ globals จาก enrollment_flow.js 8 ตัว → ตรวจว่า dependency chain ไม่ขาด

#### 7. `app/__init__.py` (181 LOC)
**เหตุผล:** app factory — CSP อนุญาต `unsafe-inline` + `unsafe-eval`; `reconnect_if_needed` เป็น `pass` placeholder; session config ครั้งเดียวสำหรับทุก request

---

### 🟡 Tier 3 — ความเสี่ยงปานกลาง × ถูกเรียกบ่อย

#### 8. `app/routes/auth.py` (181 LOC)
**เหตุผล:** login ทุก user ผ่านที่นี่; session fixation prevention; `role_required` decorator เป็น gatekeeper ทุก protected page

#### 9. `app/routes/admin.py` (738 LOC)
**เหตุผล:** CSV import สร้าง Supabase Auth users โดยตรง → injection ผ่าน CSV fields; `beacon_edit` รับ `int()` โดยไม่มี try/except บาง field

#### 10. `app/routes/teacher.py` (523 LOC)
**เหตุผล:** `override_attendance` → ตรวจ audit log ถูกเรียกก่อน return; `export_csv` → injection ผ่าน field ที่มี comma/newline

#### 11. `app/static/js/checkin_flow.js` (454 LOC)
**เหตุผล:** ส่ง face frame ไป `POST /api/checkin` — ตรวจว่า CSRF token แนบทุก request; BLE result ถูก validate ก่อนส่ง

#### 12. `app/templates/student/enroll_face.html` (458 LOC)
**เหตุผล:** template ที่ใหญ่สุด — โหลด JS สองตัว; ตรวจ Jinja2 conditional ป้องกัน circular โหลดทั้งสองพร้อมกันใน mode ผิด

---

### 🟢 Tier 4 — ความเสี่ยงต่ำ / utility / ไม่รับ untrusted input โดยตรง

#### 13. `app/static/js/mediapipe_liveness.js` (503 LOC)
shared lib — MediaPipe wrapper; ตรวจว่า singleton `_sharedFM` destroy/reinit ถูกต้อง

#### 14. `app/templates/teacher/session_view.html` (307 LOC)
#### 15. `app/templates/admin/course_detail.html` (305 LOC)
#### 16. `app/templates/teacher/dashboard.html` (273 LOC)
#### 17. `app/templates/base.html` (284 LOC)
templates ขนาดกลาง — ตรวจ XSS escape และ CSRF token ใน form

#### 18. `app/scheduler.py` (143 LOC)
background job — ตรวจ error handling ป้องกัน loop crash

#### 19. `app/config.py` (62 LOC)
env vars — ตรวจ fallback ใน dev mode ไม่หลุดไป production

#### 20. `app/models/user_model.py` (14 LOC)
얇은 wrapper — ตรวจว่า `maybe_single()` handle `None` ถูกต้อง

#### 21. `app/static/js/ble_scanner.js` (153 LOC)
#### 22. `app/static/js/camera_guard.js` (55 LOC)
#### 23. `app/static/js/rt_analyze.js` (135 LOC)
utility JS — ความเสี่ยงต่ำ, ไม่ส่งข้อมูล biometric

#### 24–31. Templates อื่น ๆ (auth, admin minor pages)
`login.html`, `beacons.html`, `courses.html`, `sessions.html`, `users.html`, `biometrics.html`, `import_csv.html`, `import_result.html`, `history.html`, `dashboard.html` (admin/teacher/student), `change_password.html`, `register.html`

#### 32–33. Scripts one-shot
`migrate_to_facenet512.py` (78 LOC), `scripts/backfill_integrity_hash.py` (81 LOC)

---

## สรุป Tier

| Tier | ไฟล์หลัก | LOC รวม | เหตุผลหลัก |
|---|---|---|---|
| 🔴 1 | api_checkin, face_service, security_service, enrollment_flow.js | ~3 031 | รับ input จากภายนอก, crypto, ถูกเรียกทุก request, JS ขนาดใหญ่ |
| 🟠 2 | student, enrollment_circular.js, app/__init__ | ~2 009 | enrollment pipeline ยาว, override liveness, app-wide config |
| 🟡 3 | auth, admin, teacher, checkin_flow.js, enroll_face.html | ~2 355 | auth flow, CSV import, audit override, JS check-in, template ใหญ่ |
| 🟢 4 | scheduler, config, models, scripts, templates อื่น ๆ, utility JS | ~3 346 | background/utility, ไม่รับ untrusted input โดยตรง |

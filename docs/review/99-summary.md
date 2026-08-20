# Pass 3 — Summary Review: SmartCheck

ตรวจสอบ ณ วันที่ 2026-08-20  
ครอบคลุม: api_checkin.py, face_service.py, enrollment_flow.js, student.py, security_service.py, auth.py (6 ไฟล์)  
ยังไม่ครอบคลุม: admin.py, teacher.py, app/__init__.py และอื่นๆ (ดูหัวข้อ "ยังไม่ได้ review")

---

## Cross-cutting Findings

### CC-1 — Liveness Challenge ไม่มีหลักฐานบน server (root cause ของ F-1 + F-6)

**F-1** (api_checkin.py:45–146): client ส่ง `liveness_action` จากตัวเอง; server ตรวจ EAR เฉพาะเมื่อค่า == `"blink"` — ค่าอื่นทุกค่าใน whitelist ข้าม EAR ไปตลอด

**F-6** (enrollment_flow.js:737–826): `_networkError` ทำให้ client ข้าม pre/post-challenge spoof check ได้ด้วยการ block URL เดียว; server enrollment pipeline ยังทำงานแต่ interactive liveness check หายไปทั้ง 2 จุด

**Root cause เดียวกัน**: ทั้ง F-1 และ F-6 เกิดจากปัญหาเดียวกัน — **server ไม่มีหลักฐานว่า liveness challenge ถูกทำจริง** server รับ `liveness_action` (F-1) หรือ embedding (F-6) จาก client โดยไม่มี nonce/token ที่ผูกกับ challenge ที่ server สั่ง ทำให้ client ข้ามหรือ replay challenge ได้เสมอ

**ความแตกต่าง:**
- F-1 = check-in; ข้าม EAR blink verification ด้วย string เดียว
- F-6 = enrollment; ข้าม liveness spoof checks ด้วยการ block network

**ทางแก้ร่วม:** server สุ่ม challenge token เก็บใน session ก่อน step 2; ตอน submit ตรวจว่า token ตรงกับที่สั่ง — client ปลอม action หรือ skip network ไม่ได้ เพราะ token ผูกกับ session

---

### CC-2 — Client-controlled Values ปรากฏใน 3 ไฟล์

| ค่า | ไฟล์:line | server validation | ความเสี่ยง |
|---|---|---|---|
| `liveness_action` (str) | api_checkin.py:46 | whitelist 7 ค่า แต่ gate ทำงานเฉพาะ `"blink"` | 🔴 F-1 |
| `flow_mode` (str) | student.py:583 | ไม่มี whitelist — ส่ง `"circular"` ลด threshold 0.80→0.75 | 🟠 security |
| `capturedImages` array | enrollment_flow.js:45, 1228 | server ตรวจทุก frame ด้วย combined_spoof_score | 🟡 defense-in-depth ยังอยู่ |
| `baseline_ear` (float) | enrollment_flow.js:1298, student.py:326 | range (0,1) เท่านั้น — stored ใน DB | 🟡 gate ปัจจุบัน disabled |
| `ear_std` (float) | enrollment_flow.js:1299, student.py:505 | logged เท่านั้น ไม่ reject | 🟢 informational |

**Pattern:** client-side ค่าใดๆ ที่ส่งขึ้น server ควรถูก validate หรือ derive จาก server state ทุกครั้ง `flow_mode` เป็นกรณีที่เร่งด่วนที่สุดเพราะ derive ได้จาก `current_app.config["ENROLL_FLOW_MODE"]` ทันที

---

### CC-3 — Dead Code ปรากฏในทุกไฟล์ที่ review

| รายการ | ไฟล์ | ประเภท |
|---|---|---|
| `startCamera`, `calcEAR`, `dist`, `_deviceFingerprint`, `_PROGRESS_MSGS_VERIFY`, `calibEARValues`, `calibrating`, `calibStream`, `verifyStream`, `_log`, `_warn` | enrollment_flow.js | 11 items; รวม 4 functions + 7 variables |
| `api_self_verify` | student.py:743 | 201 LOC endpoint — ไม่มี fetch() ใน .js ใดเลย; api_enroll:687 ระบุ "self-verify step removed" |
| `FASNET_REAL_THRESHOLD` | face_service.py:38 | constant 0.50 — ไม่มี caller ใดเลย |
| `else: raise ValueError` | api_checkin.py:236 | unreachable branch หลัง outer guard ที่ line 210 |
| whitelist entries `nod`, `turn_right`, `smile`, `raise_eyebrows` | api_checkin.py:45 | 4 ค่า — client ไม่เคยส่ง (grep ยืนยัน) |

**รวม dead code ยืนยันแล้ว: ~17 items** ลบได้ทั้งหมดโดยไม่กระทบ runtime

---

### CC-4 — Haar Cascade โหลดซ้ำทุก Request

`cv2.CascadeClassifier(...)` ถูกสร้างใหม่ใน 2 จุดในไฟล์เดียว:
- `face_service.py:71` (`_crop_face_for_antispoof`)
- `face_service.py:803` (`detect_static_image`)

ทั้งสองถูกเรียกในทุก check-in request ผ่าน `combined_spoof_score` → **62 ms ต่อ request** (วัดจริง: 31 ms × 2)  
แก้: module-level `_face_cascade = cv2.CascadeClassifier(...)` 1 บรรทัด — zero behavior impact

---

### CC-5 — Threshold ที่ตั้งโดยไม่ calibrate ร่วมกัน

| threshold | file:line | ปัญหา |
|---|---|---|
| `fasnet_suspicious = 0.30` | face_service.py:296 | ต่ำกว่าทุก layer อื่น (0.50–0.55) — ค่ามาจาก commit `0485e04` วัตถุประสงค์เดิมคือ `is_real` gate ไม่ใช่ hard-reject vote; borderline real face trigger suspicious ได้ง่าย |
| `CONSISTENCY_THRESHOLD = 0.80` (classic) vs 0.75 (circular) | student.py:583–584 | client เลือกค่าเองได้ผ่าน `flow_mode`; ควร derive จาก config |
| `detect_screen_texture` min_peaks default 50 | face_service.py:673 | ค่า default ไม่มี caller ใช้ — ทุก caller ส่ง 30 โดยตรง; function signature misleading |
| `server_validate` blur 8, color std 2.0 | face_service.py:772, 781 | relax จาก 20 และ 5.0 ตาม commit `fe71b22` ก่อน `fasnet 0.30` ถูกตั้ง — ไม่ได้ calibrate ร่วมกัน |

---

## Master Findings Table

| ID | เรื่อง | ไฟล์:line | ประเภท | ความยากแก้ | สถานะ |
|---|---|---|---|---|---|
| **F-7** | `/change-password` ขาด CSRF + ไม่ตรวจ current_password — takeover ผ่าน CSRF ได้ | auth.py:118, change_password.html | security 🔴 | ต่ำมาก | **แก้แล้ว** (commit f7b7773) |
| **F-1** | EAR liveness ข้ามได้ด้วย `"passive"` | api_checkin.py:45, 146 | security 🔴 | สูง (design change) | ยังไม่แก้ |
| **F-3** | `antispoof_passive()` ขาด `@csrf_protect` | api_checkin.py:401 | security 🟡 | ต่ำมาก (1 บรรทัด) | **แก้แล้ว** (commit f7b7773) |
| **F-5** | `fasnet_suspicious = 0.30` ไม่สมดุลกับ layer อื่น | face_service.py:296 | security/quality 🟠 | ต่ำ (1 บรรทัด) — รอ log | รอข้อมูล PRE-HARDREJECT log |
| **F-6** | `_networkError` fail-open ข้าม liveness spoof check | enrollment_flow.js:741, 818 | security 🟠 | ปานกลาง | ยังไม่แก้ |
| **CC-1a** | `flow_mode` client-controlled consistency threshold | student.py:583 | security 🟠 | ต่ำ (1 บรรทัด) | **แก้แล้ว** (commit 948885b) |
| **CC-1b** | liveness challenge ไม่มีหลักฐานบน server (root) | api_checkin.py, student.py | security 🟠 | สูง | ยังไม่แก้ |
| **P-1** | Haar Cascade โหลดซ้ำ 2 จุด (~62 ms/request) | face_service.py:71, 803 | perf 🟡 | ต่ำมาก | **แก้แล้ว** (commit 948885b) |
| **P-2** | Double decode + double Moiré (~13 ms/request) | api_checkin.py:166, 247 | perf 🟡 | ต่ำ | **แก้แล้ว** (commit 7f764c5) |
| **P-3** | `checkin()` 7 SELECT ต่อ GET page load | student.py:103 | perf 🟡 | ปานกลาง | ยังไม่แก้ |
| **Q-1** | `api_enroll` 475 LOC, 15 pipeline steps | student.py:257 | quality 🟢 | สูง (SPLIT) | ยังไม่แก้ |
| **Q-2** | `checkin()` (api_checkin) 368 LOC, 14 pipeline steps | api_checkin.py:26 | quality 🟢 | สูง (SPLIT) | ยังไม่แก้ |
| **Q-3** | `combined_spoof_score` 234 LOC | face_service.py:163 | quality 🟢 | ปานกลาง (SPLIT) | ยังไม่แก้ |
| **Q-4** | `startCaptureWithDetection` 223 LOC, onResults 170+ LOC | enrollment_flow.js:1056 | quality 🟢 | ปานกลาง (SPLIT) | ยังไม่แก้ |
| **Q-5** | `api_self_verify` 201 LOC — dead endpoint ไม่มี caller | student.py:743 | quality 🟡 | ต่ำ (ยืนยันก่อน delete) | ยังไม่แก้ |
| **Q-6** | Dead code 11 รายการ ใน enrollment_flow.js | enrollment_flow.js | quality 🟢 | ต่ำ | ยังไม่แก้ |
| **Q-7** | Dead code: `FASNET_REAL_THRESHOLD`, `else: raise ValueError`, whitelist 4 ค่า | face_service.py:38, api_checkin.py:45, 236 | quality 🟢 | ต่ำมาก | ยังไม่แก้ |
| **Q-8** | `_cosine_sim` duplicate ใน student.py vs face_service.py | student.py:48 | quality 🟢 | ต่ำ (MERGE) | ยังไม่แก้ |
| **Q-9** | `check_anti_spoof` + `check_anti_spoof_with_score` near-duplicate | face_service.py:443, 458 | quality 🟢 | ต่ำ (MERGE) | ยังไม่แก้ |
| **Q-10** | `spoof_check_with_embedding` ทำ DeepFace.represent ซ้ำแทนที่จะ delegate | face_service.py:474 | quality 🟢 | ปานกลาง | ยังไม่แก้ |
| **Q-11** | BLE FAIL-OPEN by default (`BLE_CHECK_ENABLED=False`) | api_checkin.py:124 | config 🟡 | ต่ำ (env var) | ยังไม่แก้ |
| **Q-12** | Frame validation (0b) ก่อน session check (1) — เสีย CPU | api_checkin.py:75, 85 | quality 🟡 | ต่ำ | ยังไม่แก้ |
| **Q-13** | api_enroll docstring ล้าสมัย (9 steps vs 15 จริง) | student.py:261 | quality 🟢 | ต่ำมาก | ยังไม่แก้ |
| **L-1** | `api_withdraw_consent` ไม่มี front-end trigger | student.py:1107 | legal/PDPA 🟡 | ต่ำ (เพิ่ม UI) | **แก้แล้ว** (commit ed342e0) |
| **D-1** | `onnxruntime` ไม่อยู่ใน requirements.txt | requirements.txt | deploy ⚠️ | ต่ำมาก | **แก้แล้ว** (commit 948885b) |
| **D-2** | PRE-HARDREJECT log ถอดหลัง F-5 สรุปเสร็จ | face_service.py:~298 | cleanup 🟢 | ต่ำมาก | รอ F-5 สรุป |
| **F-8** | Device token หมดอายุ 120 วัน — ไม่มี revocation กลางสาย | security_service.py:24 | security 🟡 | ปานกลาง | ยังไม่แก้ |
| **F-2** | Dead whitelist entries (nod/turn_right/smile/raise_eyebrows) | api_checkin.py:45 | quality 🟢 | ต่ำมาก | ยังไม่แก้ |
| **F-4** | `else: raise ValueError` unreachable | api_checkin.py:236 | quality 🟢 | ต่ำมาก | ยังไม่แก้ |

---

## แก้ก่อน (เรียงตาม ผลลัพธ์ ÷ เวลา)

| ลำดับ | ID | การแก้ | เวลาประมาณ | ผลลัพธ์ | เหตุผลที่ทำก่อน |
|---|---|---|---|---|---|
| ~~—~~ | ~~F-7~~ | ~~@csrf_protect_form + current_password verify บน /change-password~~ | ~~5 นาที~~ | ~~ป้องกัน CSRF account takeover~~ | **แก้แล้ว** commit f7b7773 |
| ~~—~~ | ~~F-3~~ | ~~@csrf_protect บน antispoof_passive~~ | ~~1 บรรทัด~~ | ~~ปิด cross-origin oracle~~ | **แก้แล้ว** commit f7b7773 |
| ~~—~~ | ~~D-1~~ | ~~เพิ่ม onnxruntime ใน requirements.txt~~ | ~~2 นาที~~ | ~~deploy ไม่พัง~~ | **แก้แล้ว** commit 948885b |
| ~~—~~ | ~~P-1~~ | ~~ย้าย Haar Cascade ไป module-level~~ | ~~10 นาที~~ | ~~-62 ms/request~~ | **แก้แล้ว** commit 948885b |
| ~~—~~ | ~~CC-1a~~ | ~~แทน data.get("flow_mode") → current_app.config~~ | ~~10 นาที~~ | ~~client ควบคุม threshold ไม่ได้~~ | **แก้แล้ว** commit 948885b |
| ~~—~~ | ~~P-2~~ | ~~combined_spoof_score(raw_frame) แทน check_anti_spoof(face_image)~~ | ~~10 นาที~~ | ~~-13 ms/request~~ | **แก้แล้ว** commit 7f764c5 |
| ~~—~~ | ~~L-1~~ | ~~Withdraw Consent button ใน dashboard.html~~ | ~~30 นาที~~ | ~~ผู้ใช้ใช้สิทธิ์ PDPA ได้จาก UI~~ | **แก้แล้ว** commit ed342e0 |
| 1 | F-5 | ปรับ `fasnet_suspicious: 0.30 → 0.50` หลังดู PRE-HARDREJECT log | **15 นาที** (+ รอ log) | ลด FRR เมื่อ borderline real face อยู่ใน dim light | รอข้อมูลก่อน ห้ามแก้ตอนนี้ |
| 4 | Q-6 | ลบ dead code 11 รายการใน enrollment_flow.js | **30 นาที** | -~50 LOC; ลด confusion trace `_deviceFingerprint` / `calcEAR` | ไม่มี risk, cleaner codebase |
| 5 | F-1 + CC-1b | Server-side challenge token: server สุ่ม + เก็บใน session ก่อน challenge; ตรวจตอน submit | **2–4 ชั่วโมง** | ปิด liveness bypass ทั้ง F-1 (check-in) และ F-6 (enrollment) พร้อมกัน | design change ใหญ่ รอเวลาที่เหมาะสม |
| 6 | Q-1, Q-2, Q-3, Q-4 | SPLIT api_enroll / checkin / combined_spoof_score / startCaptureWithDetection | **1–2 วัน** | แต่ละ security layer test ได้อิสระ; onboard ง่าย | refactor ใหญ่ ทำหลัง security fix ทั้งหมด |

---

## ยังไม่ได้ Review

ไฟล์ด้านล่างอยู่นอก scope pass นี้ ความเสี่ยงสรุปโดยย่อตาม 00-inventory.md:

| ไฟล์ | LOC | Tier | ความเสี่ยงหลักที่รอ |
|---|---|---|---|
| `app/__init__.py` | 181 | 🟠 2 | CSP อนุญาต `unsafe-inline` + `unsafe-eval`; `reconnect_if_needed` เป็น `pass` placeholder |
| `app/routes/admin.py` | 738 | 🟡 3 | CSV import สร้าง Supabase Auth users — injection ผ่าน CSV fields; beacon_edit int() ไม่มี try/except |
| `app/routes/teacher.py` | 523 | 🟡 3 | `override_attendance` audit log ก่อน return; `export_csv` injection ผ่าน comma/newline |
| `app/static/js/checkin_flow.js` | 454 | 🟡 3 | BLE result validation ก่อนส่ง server (CSRF header ยืนยันแล้วว่ามี) |
| `app/static/js/mediapipe_liveness.js` | 503 | 🟢 4 | singleton `_sharedFM` destroy/reinit ถูกต้องหรือไม่ |
| `app/static/js/enrollment_circular.js` | 645 | 🟠 2 | override `window.startLivenessChallenge` — ถ้า override ผิดจะ bypass liveness ทั้งหมด; globals dependency จาก enrollment_flow.js 8 ตัว |
| `app/scheduler.py` | 143 | 🟢 4 | error handling ป้องกัน loop crash |
| `app/config.py` | 62 | 🟢 4 | fallback ใน dev mode ไม่หลุดไป production |
| Templates ทั้งหมด | ~3570 | 🟢 4 | XSS escape, CSRF token ใน form, Jinja2 injection |

**ความเสี่ยงที่ unreviewed มากที่สุด:** `app/__init__.py` (CSP + session config) — ส่งผลต่อทุก request ในระบบ

**Review เสร็จแล้วใน session นี้:** `security_service.py` (05-security-service.md) และ `auth.py` (ครอบคลุมใน 05-security-service.md + แก้ F-7 แล้ว)

---

## PDPA / Data Handling

### สิ่งที่ทำถูกต้อง (จากไฟล์ที่ review มาแล้ว)

| เรื่อง | ไฟล์:line | สถานะ |
|---|---|---|
| Consent audit trail (consent_logs) ไม่ถูกลบ — INSERT เท่านั้น | student.py:223, 1120 | ✅ |
| ตรวจ latest consent_logs row ก่อน enroll (ไม่กรองแค่ `consent_given=True`) | student.py:295–315 | ✅ bug fix ถูกต้อง |
| Hard delete biometrics ก่อน return success ใน withdrawal | student.py:1138–1156 | ✅ fail-loud |
| ลบทุก column biometric: face_embeddings, baseline_ear, face_image_url, integrity_hash | student.py:1138 | ✅ |
| Audit log `consent_withdrawn_data_deleted` ผ่าน `log_audit_event()` | student.py:1166 | ✅ |
| Clear session state (liveness_embeddings, consent_given_at) หลัง withdrawal | student.py:1159 | ✅ |
| face-images bucket access ผ่าน signed URL (1-hour expiry) — comment ระบุ bucket ต้อง PRIVATE | student.py:901 | ✅ (per code comment) |
| `consent_version: "1.0"` บันทึกทุก record | student.py:237 | ✅ |

### ช่องว่างที่พบ

| เรื่อง | ไฟล์:line | ความรุนแรง |
|---|---|---|
| ~~ไม่มี front-end trigger สำหรับ withdrawal~~ — **แก้แล้ว** commit ed342e0 | student.py:1107 | ✅ L-1 |
| `consent_version` hardcode `"1.0"` — ไม่มีกลไก bump version เมื่อ policy เปลี่ยน | student.py:237, 1123 | 🟡 |
| ไม่มี data retention policy ใน code — ระบบไม่ auto-delete biometrics ของ user ที่ไม่ active | — | 🟡 (นโยบาย ไม่ใช่ bug) |
| face-images bucket privacy ไม่ได้ verified จาก code — ต้องตรวจ Supabase Dashboard | student.py:901 | 🟡 (ต้องตรวจ manually) |
| `ip_address` ใน consent_logs มาจาก `request.headers.get("X-Forwarded-For", request.remote_addr)` โดยตรง ไม่ผ่าน `_safe_ip()` | student.py:1127 | 🟢 minor (log spoofing เท่านั้น) |

### สรุปสถานะ PDPA

enrollment consent flow ออกแบบถูกต้องตาม PDPA — audit trail ก่อน, delete ทีหลัง, fail-loud ถ้า delete ล้มเหลว  
**gap ที่แก้แล้ว:** L-1 — เพิ่ม withdrawal UI ใน dashboard.html (commit ed342e0)  
**gap ที่เหลือก่อน deploy:** ตรวจ face-images bucket privacy ใน Supabase Dashboard (manual)

---

## สถิติสรุป

| หมวด | จำนวน finding | แก้แล้ว | แก้ได้ < 1 ชั่วโมง |
|---|---|---|---|
| security 🔴🟠🟡 | 9 | 3 (F-3, F-7, CC-1a) | 4 |
| perf 🟡 | 3 | 2 (P-1, P-2) | 2 |
| quality 🟢 | 14 | 0 | 10 |
| legal/PDPA 🟡 | 1 | 1 (L-1) | 1 |
| deploy ⚠️ | 2 | 1 (D-1) | 2 |
| **รวม** | **29** | **7** | **19** |

ไฟล์ที่ review ครอบคลุม ~4 036 LOC จากทั้งหมด ~10 741 LOC (~38%) — auth.py และ security_service.py review เสร็จแล้ว

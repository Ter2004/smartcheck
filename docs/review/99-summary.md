# Pass 3 — Summary Review: SmartCheck

## FAR/FRR update — temporal audit policy (2026-09-07)

Observed temporal ordering contradicts the low-variance spoof assumption.
User-reported live-camera measurements on this machine:

  Real check-in face:       1.774
  Real enrollment faces:   2.1–8.5
  Handheld phone screen:   39.1, 43.4, 42.8, 41.6

At check-in's <4.0 rule, the real sample is rejected while all four phone
measurements lie above the cutoff. Phone values are approximately 22–24×
the real check-in value. Hand movement is a plausible explanation, not
an isolated causal measurement.

These observations strongly contradict using low temporal variance as a
spoof rejection gate in this setup. They are not population FAR/FRR
estimates, and passing this metric does not establish that a phone passes
the complete anti-spoof pipeline. Enrollment/check-in preprocessing also
differs, so the values should retain their endpoint labels.

Applied: `/api/checkin` computes temporal variance with reference_threshold=4.0,
decision=log_only, result=measured_log_only. Missing/insufficient bursts and
computation exceptions log and continue. Moiré/Texture computation exceptions
also log and continue. Main image decoding remains required by FasNet and has
its own frame_decode_error rejection. Model decisions remain enforced.

**TV-01 — Historical temporal weight is unexercised and uncalibrated.** The
nominal 0.20 in SPOOF_WEIGHTS never contributed on current HTTP paths, because
none supplied frames_for_temporal. Retained diagnostic branch now warns at
WARNING with TV-01 whenever frames are supplied. Its effective weight is always
zero and voting spoof_score is None, preventing participation in weighted or
hard-reject decisions even if a future caller supplies frames accidentally.
Diagnostic audit_spoof_score remains available; re-enabling enforcement requires
an explicit code change and calibration, not merely passing the parameter.

Current inventory: `/api/checkin` image variance (4.0), `/api/enroll` image
variance (4.0), `/api/spoof_check` accumulated variance (6.0), and shared scorer
temporal diagnostics are all nonblocking. `/api/enroll` EAR standard deviation
is also audit-only. `/api/checkin` blink EAR validation remains enforced as a
separate signal. `/api/self_verify` and `/api/antispoof-passive` call the shared
scorer without temporal bursts. This update supersedes earlier descriptions
of check-in temporal enforcement and dormant temporal voting capability.

## Device trust findings — 2026-09-07 (open, not fixed)

- **DT-01 — Device trust is unwired end to end.** Current frontend sends no
  `device_fingerprint`, stores no returned device token, and sends no `X-Device-ID`.
  The signed `did` is not compared against a current fingerprint. The conditional
  token issuance path has likely never executed through this frontend; historical
  execution is not proven. Issuance and fingerprint enforcement remain out of scope.
- **DT-02 — Copied bearer token reduces biometric strictness (security finding).**
  A valid device token lowers the face threshold from 0.80 to 0.70 without checking
  the current fingerprint. Copying a token therefore weakens biometric verification
  for its account. Neither the thresholds nor this trust policy were changed.
- **DT-03 — Predictable device-token signing key (security finding).** The running
  `FLASK_SECRET_KEY` was confirmed to equal the placeholder
  `change-this-to-a-random-string`. Device tokens signed with this known value are
  forgeable. Rotation is planned by the user before the demo; not performed here.

Narrow fix applied separately: omit Authorization when storage contains no token;
interpret a bare `DeviceToken` scheme as absent under the existing optional-token
policy. Supplied invalid tokens still return 403. Diagnostic reasons now distinguish
`device_token_malformed`, `device_token_signature_mismatch`, `device_token_expired`,
and `device_token_bare_scheme` (result=absent, not reject). No tokens are logged.

## Proximity preflight and receipt — 2026-09-07

Implemented BLE/TOTP verification before camera access. `/api/checkin/proximity`
checks room/code, open session, enrollment and check-in deadline, then issues a
server-signed receipt. Final `/api/checkin` requires the receipt, checks its age
and student/session/method/room binding, and rechecks eligibility and the BLE room
configuration before face processing. Existing rejection logs/Thai messages are
retained in shared validation; new receipt failures have named rejection events.

Lifetime: **90 seconds**, selected as an operational allowance, not a measured
check-in percentile. User-reported enrollment lasted about 39 seconds. Check-in
code waits 1.5 seconds then needs 25 consecutive acceptable frames, with a
40-second alignment timeout after camera permission. There is no measured real
check-in capture duration in this investigation. Ninety seconds allows one
full attempt with margin and potentially a prompt second attempt; it does not
guarantee a retry after slow permission, network, or user delays. A visible timer
requires explicit fresh proximity verification at expiry. No automatic second
BLE connection occurs on submission. Browser capture completion logs elapsed_ms
to support later measurement. Receipt age is checked before server face work,
not after inference finishes. The browser timer starts before preflight network
transit, making its estimate conservative.

Signing: dedicated random 32-byte `PROXIMITY_RECEIPT_SECRET` (stored as 64 hex
characters), HMAC-SHA256 via ItsDangerous timed serialization with a purpose salt.
It does not reuse `EMBEDDING_INTEGRITY_SALT`, `ESP32_TOTP_SECRET`, or the Flask
session key. Missing/short keys abort startup. Raw code/receipt/key are not logged.
The signed payload holds a keyed room commitment instead of a raw TOTP code.

**Accepted limitation:** the receipt proves proximity was verified at a point
in time, not continued presence. Static BLE room identifiers remain forgeable
by anyone who has read one. The receipt authenticates the server's earlier
validation, not the physical board. It may be reused by the same student/session
within its lifetime for retries; the existing attendance duplicate guard remains.
TOTP is validated at preflight and may rotate during capture without invalidating
an otherwise valid receipt.

Environment diagnosis: the saved `.env` lacked CHECKIN_PROXIMITY_METHOD entirely;
it had valid UTF-8 without BOM and a trailing CRLF. Docker was not dropping the
last line. The saved setting is now `ble`, and the dedicated key was generated
locally. Startup logs `[SmartCheck] proximity method: ble` before database init.
Recreate the container with an explicit `-e CHECKIN_PROXIMITY_METHOD=ble` override.

Validation: 38 Python tests passed, plus the browser flow harness for both modes
and 6 BLE disconnect scenarios. Database/face inference are test doubles; no live
camera capture or container replacement was performed in this implementation.

## Demo risk — Supabase REST availability (2026-09-07)

Build-time dependency: Torch is explicitly installed from the separate CPU-wheel
index `download.pytorch.org`, with no configured fallback. This external build
dependency is additional to Supabase REST at runtime, which also has no fallback.
The Docker install chain now confines `|| true` to the OpenCV uninstall only;
failed installs stop the build. ONNX Runtime now has an explicit import check
alongside NumPy, Torch, TensorFlow, and DeepFace.

The system depends on Supabase's REST API (PostgREST), with no fallback when that
service is unavailable. The user reports PostgREST instability for weeks and an
open Supabase incident. In the observed seed failure, the beacon room-name
`ilike` request returned Cloudflare 1101 “Worker threw exception”, while the
equivalent SQL-editor query succeeded; schema reload did not resolve it.
These incident details are user-reported, not independently verified here.
No infrastructure fallback is being implemented today.

`scripts/seed_load_test.py` accepts `--beacon-id` or `SEED_BEACON_ID` to skip the
beacon lookup entirely (CLI takes precedence). This bypasses only that query,
not PostgREST: session queries/inserts, student lookups/inserts, enrollment and
biometric upserts still use REST. The only other `ilike` is the user-email lookup
in `--clean`; it is not executed during seeding. Auth user creation/deletion uses
the separate Supabase Auth API. Do not assume a successful manual beacon override
means the wider REST outage is resolved.

Session dates are computed at runtime in `Asia/Bangkok`, never hardcoded to
2026-08-24 or 2026-09-07. Session reuse, including the duplicate-insert recovery
query, is bounded to today's Bangkok day and the chosen beacon.

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
| ~~`else: raise ValueError`~~ | api_checkin.py:236 | **แก้ไข 2026-08-23:** ไม่ใช่ dead code จริง — ผูกกับ `if len(frames_gray)>=2:` คนละเงื่อนไขกับ wrapper `if` ที่ line 211 จึง reachable จริง เก็บไว้ (ดู F-4); item ที่ลบจริงคือ wrapper `if` ที่ line 211 |
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
| **F-5** | `fasnet_suspicious = 0.30` ไม่สมดุลกับ layer อื่น | face_service.py:296 | security/quality 🟠 | ต่ำ (1 บรรทัด) — รอข้อมูลเพิ่ม | **อัปเดต 2026-08-25 (รอบ 5):** วัดกับชุดเต็มแล้ว (14 real รวม real-degraded / 3 spoof) ยัง FULL SEPARATION เหมือนเดิม (gap 0.7273) — หลักฐานคัดค้าน finding เดิมหนักแน่นขึ้น (ทดสอบผ่านทั้ง real-sharp และ real-degraded แล้ว ไม่ใช่แค่ sharp) แต่ n=3 spoof ยังเล็กเกินจะปิดเคส — เปิดไว้รอตัวอย่างเพิ่ม ดู `10-moire-frr-investigation.md` หัวข้อ 8 |
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
| **Q-6** | Dead code 11 รายการ ใน enrollment_flow.js | enrollment_flow.js | quality 🟢 | ต่ำ | **แก้แล้ว** (commit 22d07d7) |
| **Q-7** | Dead code: `FASNET_REAL_THRESHOLD` (whitelist 4 ค่า → ดู F-2; `else: raise ValueError` ไม่ใช่ dead code → ดู F-4) | face_service.py:38 | quality 🟢 | ต่ำมาก | **แก้แล้ว** (commit 8281668) |
| **Q-8** | `_cosine_sim` duplicate ใน student.py vs face_service.py | student.py:48 | quality 🟢 | ต่ำ (MERGE) | **แก้แล้ว** (commit 8281668) |
| **Q-9** | `check_anti_spoof` + `check_anti_spoof_with_score` near-duplicate | face_service.py:443, 458 | quality 🟢 | ต่ำ (MERGE) | ยังไม่แก้ |
| **Q-10** | `spoof_check_with_embedding` ทำ DeepFace.represent ซ้ำแทนที่จะ delegate | face_service.py:474 | quality 🟢 | ปานกลาง | ยังไม่แก้ |
| **Q-11** | BLE FAIL-OPEN by default (`BLE_CHECK_ENABLED=False`) | api_checkin.py:124 | config 🟡 | ต่ำ (env var) | ยังไม่แก้ |
| **Q-12** | Frame validation (0b) ก่อน session check (1) — เสีย CPU | api_checkin.py:75, 85 | quality 🟡 | ต่ำ | **แก้แล้ว** (commit 8281668) |
| **Q-13** | api_enroll docstring ล้าสมัย (9 steps vs 15 จริง) | student.py:261 | quality 🟢 | ต่ำมาก | **แก้แล้ว** (commit 8281668) |
| **Q-15** | Anti-spoof decision policy (Moiré/Texture) กระจายอยู่ 7 จุดใน 3 ไฟล์ (`api_checkin.py`, `student.py` ×2, `face_service.py`) ด้วย 4 threshold value ต่างกันสำหรับ metric เดียวกัน — ไม่มีจุดเดียวที่แสดงภาพรวม policy ทำให้ threshold-tuning 5 รอบในอดีตแก้ได้แค่บางส่วนของระบบต่อครั้ง — ดู `10-moire-frr-investigation.md` หัวข้อ 12 (Q-14 อยู่ที่ `07-auth.md`, ไม่อยู่ในตารางนี้) | api_checkin.py:172-210, student.py:449-485/994-1015, face_service.py:188-217 | quality/maintainability 🟢 | สูง (ต้อง centralize) | บันทึกไว้ — **ผลกระทบด้าน decision แก้แล้ว 2026-08-26** (ทั้ง 7 จุด log-only แทน reject) แต่ตัวโครงสร้าง "กระจาย 7 จุด" เองยังไม่ centralize **อัปเดต §20–21 (2026-08-27): พบ Temporal enforcement ที่เดิมนับตกสองจุด — accumulated 6.0 ใน `/api/spoof_check` (คนจริงตก 3/8) และ face-cropped 4.0 ใน final `/api/enroll` (คนจริง 3.609) แก้ทั้งคู่เป็น `static_log_only`; Moiré/Texture/Temporal audit exceptions ในสอง endpoint นี้เป็น log-and-continue แล้ว ไม่มี audit-only computation terminate request ได้อีก Source claim cropped-real 15–25 ถูกลบเพราะไม่มี calibration และขัดกับ 3.609 นอกจากนี้ `frames_for_temporal` ไม่มี HTTP caller ทำให้ Temporal weight 0.20 ทำงานบนศูนย์ HTTP path — weight ใน dict ไม่ใช่ production decision weightจริง** F-16 fix ยังคงแยก system failure/spoof ผ่าน `is_system_failure()` แต่โครงสร้าง policy รวมยังไม่ centralize |
| **L-1** | `api_withdraw_consent` ไม่มี front-end trigger | student.py:1107 | legal/PDPA 🟡 | ต่ำ (เพิ่ม UI) | **แก้แล้ว** (commit ed342e0) |
| **D-1** | `onnxruntime` ไม่อยู่ใน requirements.txt | requirements.txt | deploy ⚠️ | ต่ำมาก | **แก้แล้ว** (commit 948885b) |
| **D-2** | PRE-HARDREJECT log ถอดหลัง F-5 สรุปเสร็จ | face_service.py:~298 | cleanup 🟢 | ต่ำมาก | รอ F-5 สรุป |
| **F-8** | Device token หมดอายุ 120 วัน — ไม่มี revocation กลางสาย | security_service.py:24 | security 🟡 | ปานกลาง | ยังไม่แก้ |
| **F-2** | Dead whitelist entries (nod/turn_right/smile/raise_eyebrows) | api_checkin.py:45 | quality 🟢 | ต่ำมาก | **แก้แล้ว** (commit 8281668) |
| **F-4** | Redundant wrapper `if` ที่ line 211 (เดิมเข้าใจผิดว่า `else: raise ValueError` ที่ 236 unreachable — ไม่จริง, reachable จริง ห้ามลบ) | api_checkin.py:211 | quality 🟢 | ต่ำมาก | **แก้บางส่วน** (commit 8281668) — ดูหมายเหตุ: ลบเฉพาะ wrapper if ที่ 211, `else: raise ValueError` ที่ 236 เก็บไว้เพราะ reachable จริง |
| **F-9** | `auto_manage_sessions()` (APScheduler) ไม่มี lock/unique constraint กันการรันซ้อนกันข้าม process — ถ้ามี >1 worker แต่ละ worker รัน scheduler อิสระ, auto-create session ชนกันได้ (race, ไม่มี `UNIQUE` บนตาราง `sessions`); `keep_alive` ก็ซ้ำ N ชุดแต่ไม่มีผลเสีย (แค่ redundant read) | scheduler.py:25-142, `app/__init__.py:172-173` | reliability/scale 🔴 | ปานกลาง (advisory lock) ถึงสูง (single-instance gating ผ่าน infra) | **BLOCKER** — ต้องแก้ก่อนเพิ่ม `--workers` (ดู docs/review/06-performance.md ข้อ 6, 8) ยังไม่แก้ |
| **F-15** | `detect_screen_texture()` คำนวณ threshold (`mean+3σ`) จากทั้งอาเรย์ 256×256 รวมพื้นที่ mask 0 ไว้ 25% แทนที่จะคำนวณเฉพาะพื้นที่ high-freq จริง — ผลคือ threshold สูงเกินจะ trigger ได้แทบทุกกรณี (ยืนยันด้วย synthetic worst-case checkerboard ก็ยังได้แค่ 1 peak vs `min_peaks=30` ที่ต้องการ) layer นี้ (weight 15%) จึงแทบไม่เคย contribute การตรวจจับจริงเลย ไม่ใช่แค่สัญญาณอ่อน — ดู `10-moire-frr-investigation.md` หัวข้อ 9 | face_service.py:738 (root cause), 706–741 | security (defense-in-depth อ่อนกว่าคิด) 🟠 | ต่ำ-ปานกลาง | **แก้แล้ว 2026-08-25** (`min_peaks` ไม่แตะ) — re-measure: `FA@cur=0` (จับ spoof ทั้ง 3 ที่มีได้ครบที่ threshold เดิม) แต่ `FR@cur=10/19` (over-trigger บน real มือสั่น + real ที่ยังไม่ผ่าน crop/resize) ทิศทางสัญญาณถูกแล้ว แต่ n=22 เล็กเกินจะสรุป cutoff — **2026-08-26: ถอดออกจาก vote/gate ทั้งหมดแล้ว** (ยัง compute+log อยู่ ไม่โหวต) ดูหัวข้อ 10, 14 |
| **F-16** | **[severity-critical] — แก้ครบแล้ว 2026-08-26, verify end-to-end ผ่านทุกขั้นตอน (§18)** `Dockerfile`'s image-size cleanup step (`find /usr/local/lib/python3.11 -depth -type d -name test -exec rm -rf {} +`) deletes `tensorflow/_api/v2/__internal__/test/` — a real, load-bearing part of TensorFlow's public API (`tf.__internal__.test`), not a bundled test suite. Every `from deepface import DeepFace` crashes with `ImportError: cannot import name 'test' from partially initialized module...` on **every single call, deterministically** (confirmed: bare `docker run` with zero Flask/gunicorn/threading crashes identically; fresh-restarted container fails on its very first request — not a race, not import-order, not `TF_USE_LEGACY_KERAS`). This hits `combined_spoof_score`'s existing `fasnet_alive=False` fail-close path (face_service.py:289-305), which returns the *same* 400 `"ตรวจพบรูปถ่ายหรือหน้าจอ..."` / `spoof:true` response as a genuine spoof rejection — **100% of checkins fail in this Docker build, indistinguishable from normal operation at the client or in INFO-level logs.** `/health` (`app/__init__.py:176-178`) is a static `"ok"`, never exercises Fasnet, so no healthcheck would catch this. **อัปเดต: ไม่มี Railway deployment จริง (user ยืนยัน — ไม่เคย subscribe) ไม่มีผู้ใช้ได้รับผลกระทบ พบตั้งแต่ก่อน deploy จริงใด ๆ** — ประเด็น Railway จึงปิดแล้ว ไม่ใช่ open question อีกต่อไป Root cause: the cleanup `RUN` layer executes *after* the Dockerfile's own import-verification (Step 4) and Fasnet build-time smoke test, both of which pass against the pre-cleanup filesystem — and the smoke test's `except` only re-raises on 2 narrow substrings (`"Numpy is not available"`, `"cuInit"`), so it wouldn't have caught this even if run after cleanup. **ยืนยันซ้ำอิสระอีกรอบ 2026-08-26 (คนละ session):** fresh-restart + single-request ยังพังตั้งแต่ request แรก, path ที่หายไปยืนยันตรง ๆ (`ls` ใน container ว่าง — ไม่ใช่แค่ infer จาก error message) ดู `10-moire-frr-investigation.md` §17 Cross-ref **FRR-2**: same defect (fail-close message doesn't distinguish "system broken" from "spoof detected") at maximum severity — FRR-2 is the edge-case version (~8% of dim-room real samples), F-16 is the same message ambiguity realized at 100% outage. See `10-moire-frr-investigation.md` §16-17 | Dockerfile (cleanup RUN layer, near end); face_service.py (new `is_system_failure()` helper); api_checkin.py:242-262; student.py:501-528; app/__init__.py:176-178 (`/health`) | security/reliability 🔴 → ✅ **FIXED** | ต่ำ (แก้ Dockerfile find pattern + ลำดับ layer) ถึงปานกลาง (ถ้าจะแยก message/status ด้วย) | **แก้ครบ 2026-08-26 (§17-18):** Fix #1 (แยก 400 spoof / 503 system-failure) ครบ 3 endpoint (checkin, enroll final-submit, `/api/spoof_check`) + Fix #2 (root cause: reorder cleanup layer ก่อน import checks, narrow `find` pattern ไม่ให้ลบ tensorflow's `test/`, widen Fasnet smoke test's except) verify ผ่าน 3-step ตามที่ user กำหนด (reorder-only ยังพัง → +widen ยังพังจุดเดิม → +narrow pattern ผ่านจริง ยืนยันด้วย `ls` ตรง ๆ ในอิมเมจ ไม่ใช่แค่เชื่อ build log) end-to-end confirm ครบ 2 ผลลัพธ์: 503 ก่อนแก้ root cause, 400 "ใบหน้าไม่ตรง" หลังแก้ — Eager-import+`--preload` (fork-safety กับ SQLAlchemy เป็นคนละประเด็น) และ `/health` real-check ยัง hold ไว้ตามคำสั่ง user (record only) — open item เดียวที่เหลือ: Step 4 capture loop อาจ retry เงียบไม่จบถ้า 503 เป็นปัญหาถาวร ไม่มี error message ให้เห็น (ต้องแก้ JS ถ้าจะปิด — ยังไม่ implement) |
| **F-10** | ตาราง `sessions` ไม่มี `UNIQUE` constraint บน `(course_id, start_time)` เลย — **มีอยู่แล้วแม้ตอนนี้ที่ workers=1** เพราะ `auto_manage_sessions` รันบน APScheduler background thread ของตัวเอง ซึ่งเป็นคนละ thread จาก thread ที่ serve HTTP request เสมอ (ไม่ว่าจะกี่ worker) — ชนกับ `teacher.py:188` (`session_create`, insert ด้วย `start_time` ที่ teacher พิมพ์เอง ความละเอียดระดับนาที ชนกับ auto-create ได้จริงถ้าตรง schedule) ได้โดยตรงแม้ single-worker; `admin.py:282` ใช้ `start_time=now()` ความละเอียด microsecond จึงชนยากกว่ามาก (แทบเป็นไปไม่ได้ในทางปฏิบัติ) — F-9 (multi-worker) เป็นแค่ตัวขยายปัญหาเดิมนี้ให้ชนบ่อยขึ้น ไม่ใช่ต้นเหตุ | `database/schema.sql:181-191`, `app/scheduler.py:81`, `app/routes/teacher.py:188`, `app/routes/admin.py:282` | reliability 🟠 | ต่ำ (migration + code fix ทำแล้ว) | **แก้แล้ว** — migration `database/migrations/20260823_sessions_unique_course_start.sql` + `scheduler.py` catch 23505 skip เงียบ; **ต้อง apply migration เองใน Supabase SQL editor** (agent รันให้ไม่ได้ ไม่มี DB access) หลังเช็ค/ล้าง duplicate ก่อนตามที่ให้ SQL ไว้ในแชท; `teacher.py:188` ยังใช้ raw `except Exception as e: flash(f"...: {e}")` — ถ้า constraint reject จะโชว์ raw Postgres error ให้ teacher เห็น (ยังไม่แก้ตามที่สั่งไม่ให้แตะนอก scope) |

---

## FRR Findings (แยก series จาก Master Findings Table)

**Legend เพิ่ม:** `FRR-` = False-Reject / usability — **คนละความหมายกับ `F-`** (security: "attacker ผ่านเข้ามาได้") `FRR-` หมายถึง "ผู้ใช้จริงถูกปฏิเสธ" คนละทิศทางกัน ไม่ควรนับรวมในสถิติ security severity เดียวกัน

| ID | เรื่อง | ไฟล์:line | ประเภท | ความยากแก้ | สถานะ | เกี่ยวข้องกับ |
|---|---|---|---|---|---|---|
| **FRR-1** | Moiré FFT: `low_r=0.10` (วง low-freq แคบเกิน) + คำนวณบน full frame ไม่ crop หน้า → ภาพใบหน้าจริงจากกล้องมือถือติดคะแนน 0.69–0.77 ชนกับ `MOIRE_THRESHOLD_SINGLE=0.70` เป็นประจำ ไม่ใช่ image-prep artifact — **อัปเดต 2026-08-25 (รอบ 5, ชุดเต็ม 19 real / 3 spoof รวม real-degraded 7 ไฟล์):** ~~อาจเป็น inverted~~ → **non-separable ในทุกทิศทาง** — sweep ทั้งสองทิศทางแล้ว ทั้งคู่แย่เท่ากับไม่ใช้ layer เลย (native_err=flip_err=3=trivial baseline) เพราะ spoof (0.6375–0.6997) แทรกอยู่คนละจุดกับทั้ง real-degraded (0.36–0.66) และ real-sharp (0.69–0.77) พร้อมกัน ข้อเสนอพลิก polarity **tested และ rejected แล้ว** อธิบายได้แล้วว่าทำไม 5 รอบ tune threshold ในอดีตไม่เคยได้ผลถาวร — ปัญหาคือ distribution ปนกัน ไม่ใช่ตัวเลข cutoff ผิด — ดู `10-moire-frr-investigation.md` หัวข้อ 8 | face_service.py:18, 683, 692 | usability/FRR 🟠 | **สูง** — ไม่ใช่แค่ปรับ `low_r`/threshold แล้วจบ ต้องคิด metric ใหม่ทั้งหมดหรือรับว่า Moiré ไม่ช่วยอะไรกับ full-frame FFT approach นี้ | **อัปเดต 2026-08-26 (รอบเก้า): implement แล้ว — option (C)** ทั้ง 7 จุด (6 standalone gate + `combined_spoof_score` gate #2/#3) เปลี่ยนเป็น log-only ไม่ reject จาก Moiré/Texture อีก `SPOOF_WEIGHTS` ปรับเป็น `fasnet=0.70, temporal=0.20, onnx=0.10` (moire/texture ออกจาก dict ทั้งหมด, explicit skip-set) smoke test ยืนยันด้วย `combined_spoof_score` โดยตรง: ภาพที่ moire=1.0 (เคย hard-reject) ตอนนี้ `is_real=True` ถูกต้อง — ดู `10-moire-frr-investigation.md` หัวข้อ 14 | **F-5** (ดูหมายเหตุ), **F-15** (Texture ก็ไม่ทำงานเช่นกัน คนละสาเหตุ), **Q-15** (โครงสร้าง 7 จุด) |
| **FRR-2** | Fasnet's face detector (`DeepFace.extract_faces(enforce_detection=True)`, face_service.py:149-174) หาหน้าไม่เจอในภาพมืด → `combined_spoof_score` fail-close ทันที (`fasnet_alive=False`, บรรทัด 289-305) ไม่มี layer อื่นได้โหวต — วัดได้ **1/13 (~8%)** unprocessed real samples (`real_lowdetail_dark15.jpg`, ~15 lux) — pre-existing, คนละกลไกจาก FRR-1 (detector limitation ไม่ใช่ threshold-calibration) และตอนนี้เด่นขึ้นเพราะ Fasnet ได้ weight 70% หลังแก้ FRR-1 — checkin: 1 เฟรมตก = reject 400 ทันที (`"ตรวจพบรูปถ่ายหรือหน้าจอ..."`); enroll แย่กว่า: `MIN_SPOOF_PASS=4/5` ทำให้ 2/5 เฟรมมืดพอจะตกทั้ง burst (`"ตรวจพบการปลอมแปลงใบหน้า..."`) — สำรวจข้อความที่ผู้ใช้เห็นทั้ง 7 จุด (checkin ×1 + enroll gate ×6 รวม `/api/spoof_check` 3 จุด) มีแค่ **1 จุดเดียว** (`enrollment_flow.js:1151`, Step 4 หมดโควตารวม) ที่บอกใบ้เรื่องแสง — ที่เหลือบอกว่า "ปลอมแปลง/หน้าจอ/สลับใบหน้า" ทั้งหมด ผู้ใช้ในห้องมืดจึงไม่รู้ว่าต้องแก้ด้วยแสง | face_service.py:149-174, 289-305; api_checkin.py:242-262; student.py:501-528; enrollment_flow.js:690-780, 1122-1159 | usability/FRR 🟠 | ต่ำ (แก้แค่ข้อความ) ถึงสูง (ถ้าจะแก้ detector/preprocessing) | **บันทึกไว้ 2026-08-26 — ยังไม่แก้อะไร** นอก scope รอบนี้ (ห้ามแตะ threshold/detection settings ของ Fasnet) | **F-5**, **FRR-1** (คนละ layer คนละกลไก บน anti-spoof stack เดียวกัน), **Q-15**, **F-16** (severity-critical — เดียวกัน แต่ 100% outage แทน edge case 8%, ดู `10-moire-frr-investigation.md` §16) |

**ความเกี่ยวข้องกับ F-5:** ทั้งคู่เป็น threshold-calibration problem บน anti-spoof stack เดียวกัน (`combined_spoof_score`, face_service.py:166) แต่คนละทิศทาง — F-5 (`fasnet_suspicious=0.30` ต่ำกว่า layer อื่นมาก, face_service.py:296) เสี่ยงให้ borderline **real face** ติด hard-reject vote ง่ายเกิน (ผลลัพธ์ = FRR สูงขึ้นเหมือนกัน แม้ F-5 จะถูกจัดเป็น security/quality เพราะรากมาจาก weight ที่ไม่สมดุล ไม่ใช่ metric เพี้ยนแบบ FRR-1) ทั้งสองข้อชี้ไปทางเดียวกัน: **anti-spoof stack ทั้งชุดต้อง calibrate ใหม่ด้วยข้อมูลจริง (real + spoof samples) พร้อมกันในรอบเดียว ไม่ใช่ขยับทีละ threshold แยกจุด** — ดูสถานะรวบรวม spoof sample ที่ `10-moire-frr-investigation.md`

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
| ~~—~~ | ~~Q-6~~ | ~~ลบ dead code 9 รายการใน enrollment_flow.js~~ | ~~30 นาที~~ | ~~-58 LOC (1518→1460)~~ | **แก้แล้ว** commit 22d07d7 |
| 1 | F-5 | ~~ปรับ `fasnet_suspicious: 0.30 → 0.50`~~ — **อัปเดต 2026-08-25:** วัดจริงครั้งแรกชี้ว่า 0.30 อาจ*ไม่*ต้องขยับ (อยู่กลาง gap ระหว่าง real max 0.294 กับ spoof min 0.454 พอดี) ทิศทาง "ขยับเป็น 0.50" ข้างต้นอาจผิดทาง — รอข้อมูลมากกว่านี้ก่อนตัดสินทิศทาง ไม่ใช่แค่รอ log แล้วปรับตามแผนเดิม | **15 นาที** (+ รอข้อมูลเพิ่ม) | ลด FRR เมื่อ borderline real face อยู่ใน dim light — **แต่ต้องยืนยันทิศทางก่อน** | รอข้อมูลก่อน ห้ามแก้ตอนนี้ |
| 4 | Q-6 | ลบ dead code 11 รายการใน enrollment_flow.js | **30 นาที** | -~50 LOC; ลด confusion trace `_deviceFingerprint` / `calcEAR` | ไม่มี risk, cleaner codebase |
| 5 | F-1 + CC-1b | Server-side challenge token: server สุ่ม + เก็บใน session ก่อน challenge; ตรวจตอน submit | **2–4 ชั่วโมง** | ปิด liveness bypass ทั้ง F-1 (check-in) และ F-6 (enrollment) พร้อมกัน | design change ใหญ่ รอเวลาที่เหมาะสม |
| 6 | Q-1, Q-2, Q-3, Q-4 | SPLIT api_enroll / checkin / combined_spoof_score / startCaptureWithDetection | **1–2 วัน** | แต่ละ security layer test ได้อิสระ; onboard ง่าย | refactor ใหญ่ ทำหลัง security fix ทั้งหมด |
| 0 | **F-9** | **BLOCKER — ต้องแก้ก่อนเพิ่ม `--workers`** (ไม่ใช่ optional): gate scheduler ให้รันแค่ 1 instance หรือใส่ Postgres advisory lock | **30 นาที – 2 ชั่วโมง** (ดูตัวเลือก (a)/(b) ใน docs/review/06-performance.md) | ป้องกัน session ซ้ำเมื่อ scale worker — โดยตรงเปิดทางให้ทำ capacity plan ในข้อ F-9/06-performance.md ได้จริง | ค้นพบระหว่างประเมิน capacity — เพิ่ม `--workers` ตอนนี้ไม่ปลอดภัยจนกว่าจะแก้ข้อนี้ |

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
| ~~`ip_address` ใน consent_logs มาจาก `request.headers.get("X-Forwarded-For", request.remote_addr)` โดยตรง ไม่ผ่าน `_safe_ip()`~~ — **แก้แล้ว** commit 8281668 (ทั้ง student.py:1127 และ `log_audit_event` ใน security_service.py เปลี่ยนไปใช้ `_safe_ip()`) | student.py:1127, security_service.py:202 | ✅ |

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
| quality 🟢 | 14 | 1 (Q-6) | 10 |
| legal/PDPA 🟡 | 1 | 1 (L-1) | 1 |
| deploy ⚠️ | 2 | 1 (D-1) | 2 |
| reliability/scale ~~🔴~~ 🟡 | 1 | 0 | 0 (F-9 — ~~BLOCKER, ปานกลาง–สูง~~ ไม่ใช่ blocker แล้ว ปานกลาง — ดูอัปเดตด้านล่าง) |
| reliability 🟠 | 1 | 1 (F-10 — โค้ด+migration พร้อม รอ apply เอง) | 1 |
| **รวม** | **31** | **9** | **20** |

> **อัปเดต (2026-08-25):** F-9 (`reliability/scale`) ปรับจาก 🔴 BLOCKER → 🟡 ปานกลาง — ตรวจ `auto_manage_sessions`/`keep_alive` จริงแล้วพบว่า auto-create มี UNIQUE constraint จาก F-10 คุ้มครองอยู่ (catch `23505` แล้ว skip), auto-close เป็น `UPDATE ... WHERE is_open=true` ซึ่ง idempotent โดยธรรมชาติ, `keep_alive` read-only — รัน N ชุดพร้อมกัน (ตอนเพิ่ม `--workers`) ไม่ทำให้ข้อมูลพัง เหลือแค่ wasted HTTP call + log noise ซ้ำ N เท่า **ไม่ใช่ blocker ก่อนเพิ่ม `--workers`/ก่อน load test อีกต่อไป** รายละเอียดเต็มดู `06-performance.md:233` (F-9 เองยังควรแก้อยู่ดีหลัง load test เสร็จ ด้วย `fcntl.flock` คุม `start_scheduler()`)

### Verification gap: real `/api/enroll` consistency path

As of 2026-08-27, there is no retained log or repository artifact proving that embedding consistency has ever passed through the real `/api/enroll` HTTP path. Existing route tests mock `check_embedding_consistency()` as successful, so they assert downstream happy-path behavior rather than exercising real DeepFace embeddings and the shipped pairwise consistency calculation. This is a verification gap, not proof that the endpoint has never succeeded.

### Live-camera measurements / TOR §6.3 — 2026-08-27

First retained real-HTTP measurements are now recorded in `docs/review/11-live-camera-measurements.md`: duplicate detection blocked the same person enrolling a second account at similarity `0.9340` against threshold `0.65`; live Fasnet scores separated six real-face frames (`0.0001–0.0011`) from four handheld phone-screen frames (`0.9980–0.9998`) with an observed edge gap of `0.9969` and no errors in this small sample; temporal variance ran in the unsafe direction for this scenario (`5.925–6.330` real/still versus `39.104–43.373` handheld phone), meaning the historical `6.0` gate could reject the real user while passing the replay. These are starting measurements only (`n=1` subject, one device/session environment), not validated FAR/FRR rates.

**หมายเหตุ:** คอลัมน์ "แก้แล้ว"/"แก้ได้ <1ชม." ของหมวดอื่น (โดยเฉพาะ quality 🟢) ยังไม่ได้ reconcile กับสถานะล่าสุดหลัง commit `8281668` (F-2/F-4/Q-7/Q-8/Q-12/Q-13 แก้แล้ว) — ตัวเลขในตารางนี้จึง**ต่ำกว่าความจริง**ในบางหมวด ยังไม่ได้แก้เพราะไม่ใช่ scope ของ task นี้ (เพิ่ม F-9 อย่างเดียว) ต้อง reconcile ทั้งตารางแยกต่างหากถ้าต้องการตัวเลขที่แม่นยำ 100%

ไฟล์ที่ review ครอบคลุม ~4 036 LOC จากทั้งหมด ~10 741 LOC (~38%) — auth.py และ security_service.py review เสร็จแล้ว

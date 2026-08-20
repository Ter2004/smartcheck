# Review: `app/services/security_service.py`

วันที่: 2026-08-20  
Scope: localhost เท่านั้น — ไม่รวม BLE / production deploy  
ไฟล์: 211 LOC — ถูก import โดยทุก protected route  
บริบท: Tier 🔴 1 ตัวสุดท้าย — ถูก import ทุก Blueprint ที่ต้องการ CSRF/auth

---

## Function Inventory

| ฟังก์ชัน | line | LOC | ถูกเรียกจากไหน | คำตัดสิน | เหตุผล |
|---|---|---|---|---|---|
| `create_device_token` | 25 | 21 | student.py:921 (api_self_verify finalize) | KEEP | HMAC-SHA256 ถูกต้อง; payload ชัดเจน (uid+did+iat) |
| `verify_device_token` | 48 | 39 | api_checkin.py:63 | KEEP | compare_digest ถูกต้อง; expiry check ครบ |
| `compute_embedding_integrity_hash` | 91 | 24 | student.py:695 (api_enroll), student.py:885 (api_self_verify), security_service.py:128 (internal), scripts/backfill_integrity_hash.py:65 | KEEP | HMAC-SHA256; sorted+normalized embedding → order-independent ✅ |
| `verify_embedding_integrity` | 117 | 13 | api_checkin.py:315 | KEEP | empty hash → False (fail-close) ✅; compare_digest ✅ |
| `csrf_protect` | 134 | 21 | api_checkin.py:30; student.py:222,256,742,954,1092,1106; admin.py:718; teacher.py:480 (decorator) | KEEP | compare_digest ✅; empty session token → 403 ✅ |
| `csrf_protect_form` | 157 | 16 | admin.py (15 routes); teacher.py (7 routes) (decorator) | KEEP | only validates POST/PUT/DELETE/PATCH → GET ผ่านถูกต้อง; compare_digest ✅ |
| `log_audit_event` | 177 | 35 | student.py:1166; teacher.py:396, 512; admin.py:727 | REFACTOR | non-fatal ✅; แต่ใช้ `request.remote_addr` แทน `_safe_ip()` → IP ใน audit_logs spoofable ผ่าน X-Forwarded-For ใน production |

---

## CSRF Coverage

grep ครบ: ทุก route ที่มี `methods=["POST"]` หรือ `methods=["GET", "POST"]`

| endpoint | file:line | decorator | สถานะ | ถ้าไม่มี ผลคืออะไร |
|---|---|---|---|---|
| `POST /api/checkin` | api_checkin.py:26 | `@csrf_protect` | ✅ | — |
| `POST /api/antispoof-passive` | api_checkin.py:398 | **ไม่มี** | ❌ F-3 | cross-origin forge ส่ง image probe MiniFASNet score ได้ |
| `POST /api/consent` | student.py:218 | `@csrf_protect` | ✅ | — |
| `POST /api/enroll` | student.py:252 | `@csrf_protect` | ✅ | — |
| `POST /api/self_verify` | student.py:738 | `@csrf_protect` | ✅ | — |
| `POST /api/spoof_check` | student.py:950 | `@csrf_protect` | ✅ | — |
| `POST /api/reset-liveness` | student.py:1088 | `@csrf_protect` | ✅ | — |
| `POST /api/withdraw-consent` | student.py:1103 | `@csrf_protect` | ✅ | — |
| `POST /teacher/session/create` | teacher.py:157 | `@csrf_protect_form` | ✅ | — |
| `POST /teacher/session/<id>/toggle` | teacher.py:257 | `@csrf_protect_form` | ✅ | — |
| `POST /teacher/session/<id>/override` | teacher.py:334 | `@csrf_protect_form` | ✅ | — |
| `POST /teacher/api/reset-enrollment/<id>` | teacher.py:477 | `@csrf_protect` | ✅ | — |
| `GET+POST /admin/import-csv` | admin.py:70 | `@csrf_protect_form` | ✅ | — |
| `POST /admin/beacons/add` | admin.py:165 | `@csrf_protect_form` | ✅ | — |
| `POST /admin/beacons/<id>/edit` | admin.py:185 | `@csrf_protect_form` | ✅ | — |
| `POST /admin/beacons/<id>/delete` | admin.py:205 | `@csrf_protect_form` | ✅ | — |
| `POST /admin/sessions/create` | admin.py:261 | `@csrf_protect_form` | ✅ | — |
| `POST /admin/sessions/<id>/delete` | admin.py:295 | `@csrf_protect_form` | ✅ | — |
| `POST /admin/courses/add` | admin.py:348 | `@csrf_protect_form` | ✅ | — |
| `POST /admin/courses/<id>/add-section` | admin.py:379 | `@csrf_protect_form` | ✅ | — |
| `POST /admin/courses/<id>/enroll` | admin.py:495 | `@csrf_protect_form` | ✅ | — |
| `POST /admin/courses/<id>/unenroll/<id>` | admin.py:515 | `@csrf_protect_form` | ✅ | — |
| `POST /admin/courses/<id>/import-csv` | admin.py:535 | `@csrf_protect_form` | ✅ | — |
| `POST /admin/courses/<id>/schedules/add` | admin.py:622 | `@csrf_protect_form` | ✅ | — |
| `POST /admin/courses/<id>/schedules/<id>/delete` | admin.py:651 | `@csrf_protect_form` | ✅ | — |
| `POST /admin/api/reset-enrollment/<id>` | admin.py:715 | `@csrf_protect` | ✅ | — |
| `GET+POST /login` | auth.py:52 | **ไม่มี** | ⚠️ ยอมรับได้ | login CSRF impact ต่ำ — session ยังไม่มี csrf_token ตอน GET |
| `GET+POST /change-password` | auth.py:117 | **ไม่มี** | ❌ **ช่องโหว่** | attacker CSRF ให้ logged-in user เปลี่ยน password โดยไม่รู้ตัว (ไม่ check old password) |

### สรุป CSRF: 25/27 ✅ — 2 ที่ขาด

- **F-3** `/api/antispoof-passive` — impact ต่ำ (ไม่ write DB) แต่ probe oracle ได้
- **`/change-password`** — **impact สูง**: POST โดยไม่มี CSRF token + ไม่ยืนยัน old password → attacker CSRF เปลี่ยน password user เป็นค่าที่รู้ → เข้า account ได้

---

## Crypto Review

| ฟังก์ชัน | line | อัลกอริทึม | key มาจากไหน | ใช้ compare_digest? | ปัญหาที่พบ |
|---|---|---|---|---|---|
| `create_device_token` | 25 | HMAC-SHA256 (hexdigest) | `SECRET_KEY` env var (app.config) | ไม่ — สร้าง token เท่านั้น | `max_age_days=120` ใน verify: expiry นาน 4 เดือน, ไม่มี revocation |
| `verify_device_token` | 48 | HMAC-SHA256 | `SECRET_KEY` env var | **ใช่** (line 70) ✅ | expiry ตรวจจาก `iat` — ถ้า clock drift หรือ NTP ผิดพลาด token อาจ accept/reject ผิด |
| `compute_embedding_integrity_hash` | 91 | HMAC-SHA256 (hexdigest) | `EMBEDDING_INTEGRITY_SALT` env var | ไม่ — compute เท่านั้น | ถ้า `EMBEDDING_INTEGRITY_SALT` ว่างเปล่าใน dev → HMAC key = `b""` → ทุก attack offline compute hash ได้ |
| `verify_embedding_integrity` | 117 | HMAC-SHA256 (via internal call) | `EMBEDDING_INTEGRITY_SALT` env var | **ใช่** (line 129) ✅ | ✅ |
| `csrf_protect` | 134 | constant-time compare session token | per-session `secrets.token_hex(32)` | **ใช่** (line 151) ✅ | ✅ |
| `csrf_protect_form` | 157 | constant-time compare session token | per-session `secrets.token_hex(32)` | **ใช่** (line 168) ✅ | ✅ |

### `verify_embedding_integrity` — empty hash behavior (โค้ดดิบ)

```python
# security_service.py:117–129
def verify_embedding_integrity(
    user_id: str,
    embeddings: list,
    stored_hash: str,
    integrity_salt: str,
) -> bool:
    if not stored_hash:
        return False  # No hash = unverifiable — reject and require re-enrollment
    expected = compute_embedding_integrity_hash(user_id, embeddings, integrity_salt)
    return hmac.compare_digest(expected, stored_hash)
```

`stored_hash = ""` → `not stored_hash` = True → **return False** — **ปฏิเสธทันที ✅**  
นี่คือ fail-close ที่ถูกต้อง: embedding ที่ไม่มี hash (ก่อน backfill) ถูกบล็อกไม่ให้ check-in

### Device Token — expiry และ revocation

```python
# security_service.py:48 (signature)
def verify_device_token(token, secret_key, max_age_days=120) -> dict | None:
    ...
    if time.time() - payload.get("iat", 0) > max_age_days * 86400:
        return None  # expired
```

- **expiry: 120 วัน** (4 เดือน) — hardcode ใน default; ถ้า device token ถูกขโมย attacker มีเวลา 4 เดือน
- **ไม่มี revocation**: ไม่มี server-side token store → ไม่สามารถ invalidate token ที่ออกไปแล้วได้ (logout ไม่ invalidate device token)
- caller ที่ api_checkin.py:63: FAIL-OPEN ถ้า token = None → ใช้ NEW_DEVICE_THRESHOLD 0.80 แทน (รับรู้แล้วใน 01-api-checkin.md)

---

## Audit Log Coverage

`log_audit_event()` ถูกเรียกจาก 4 จุด:

| จุดที่เรียก | file:line | event_type | เมื่อไหร่ |
|---|---|---|---|
| api_withdraw_consent | student.py:1166 | `consent_withdrawn_data_deleted` | ผู้ใช้ถอน PDPA |
| teacher override_attendance | teacher.py:396 | `teacher_override` | อาจารย์แก้ attendance |
| teacher api_reset_enrollment | teacher.py:512 | `reset_enrollment_attempts` | อาจารย์รีเซ็ต attempt counter |
| admin api_reset_enrollment | admin.py:727 | `reset_enrollment_attempts` | แอดมินรีเซ็ต attempt counter |

### Actions สำคัญที่ **ไม่อยู่ใน audit_logs** (ขาด)

| action | ไปที่ | ความเสี่ยง |
|---|---|---|
| Enrollment สำเร็จ (บันทึก face embedding) | `_log()` (Python logger เท่านั้น) — ไม่ถึง audit_logs DB | 🟠 ไม่มี queryable trail ว่าใครลงทะเบียนเมื่อไหร่ |
| Check-in สำเร็จ (บันทึก attendance) | INSERT ลง `attendance` table แต่ไม่มี audit event | 🟡 attendance table เป็น evidence อยู่แล้ว แต่ไม่มี event log |
| Login สำเร็จ / ล้มเหลว | ไม่มีเลย | 🟡 ไม่รู้ว่า account ถูก brute-force หรือไม่ |
| Logout | ไม่มีเลย | 🟢 low impact |
| Password change | ไม่มีเลย | 🟠 ไม่รู้ว่ามีใครเปลี่ยน password คนอื่น (ถ้า CSRF สำเร็จ) |
| Admin สร้าง user / CSV import | ไม่มีเลย | 🟠 ไม่มีหลักฐานว่า admin สร้าง account ใดเมื่อไหร่ |
| Admin ลบ session / unenroll student | ไม่มีเลย | 🟡 |

**ข้อสังเกต:** `_log()` ใน student.py (Python audit logger) บันทึกทุก enrollment step ละเอียด แต่ไปที่ log file/stdout เท่านั้น — ไม่ queryable ผ่าน Supabase; admin ที่ต้องการ query "enrollment ของ student X เมื่อวาน" ทำได้เฉพาะถ้า log file เข้าถึงได้

---

## ข้อสังเกตเพิ่มเติม

### `log_audit_event` ใช้ `request.remote_addr` แทน `_safe_ip()`

```python
# security_service.py:202
"ip_address": request.remote_addr,
```

`request.remote_addr` คือ IP ของ proxy/load-balancer ใน production เสมอ — ไม่ใช่ client จริง  
`_safe_ip()` ใน student.py:31 ทำ X-Forwarded-For validation อยู่แล้ว แต่ไม่ถูก import ไว้ใน security_service  
ผลคือ IP ทุก audit event จาก teacher override, admin reset, consent withdrawal จะแสดงเป็น Railway/proxy IP แทน user จริง

### `compute_embedding_integrity_hash` — INTEGRITY_SALT ว่างเปล่า

ถ้า `EMBEDDING_INTEGRITY_SALT=""` (ค่า default ที่ไม่ได้ตั้ง):
```python
hmac.new(b"", payload.encode(), hashlib.sha256)
```
Key ว่างเปล่า → HMAC ยังทำงาน แต่ attacker ที่รู้ format สามารถ compute hash ล่วงหน้าได้ (no secret)  
ควรเพิ่ม startup check ใน `config.py` ว่า `EMBEDDING_INTEGRITY_SALT` ไม่ว่างเปล่า ก่อน app เริ่ม (คล้ายกับที่ config.py ตรวจ SECRET_KEY)

### `csrf_protect` บน GET-only routes ใน admin.py

admin.py:35, 58 มี `@csrf_protect_form` บน route ที่เป็น GET-only  
`csrf_protect_form` check `if request.method in ("POST",...)` ก่อน → GET ผ่านโดยไม่ validate  
ไม่เป็นปัญหา — เป็น defensive pattern ที่ครอบ route แม้ว่าอนาคตจะเพิ่ม POST; ไม่ต้องแก้

---

## Top 3

### 1. เพิ่ม `@csrf_protect_form` บน `/change-password` (auth.py:117) — 5 นาที

```python
@auth_bp.route("/change-password", methods=["GET", "POST"])
@login_required
@csrf_protect_form     # ← เพิ่มบรรทัดนี้
def change_password():
```

และเพิ่ม `<input type="hidden" name="csrf_token" value="{{ session.csrf_token }}">` ใน `change_password.html`

ผลลัพธ์: ปิด CSRF account takeover ผ่าน password change; impact สูงที่สุดของไฟล์นี้

---

### 2. ใช้ `_safe_ip()` ใน `log_audit_event` แทน `request.remote_addr` — 10 นาที

```python
# security_service.py:202 — แก้เป็น:
"ip_address": _extract_safe_ip(),
```

โดย extract logic จาก student.py:31–38 ขึ้นมาเป็น helper ใน security_service.py เองหรือ shared util  
ผลลัพธ์: IP ใน audit_logs ทุกแถวเป็น client จริง ไม่ใช่ proxy — สำคัญเมื่อ forensic teacher override

---

### 3. เพิ่ม `log_audit_event` ที่ enrollment สำเร็จ และ login สำเร็จ/ล้มเหลว — 30 นาที

enrollment (student.py:729 หลัง upsert สำเร็จ):
```python
log_audit_event(supabase_admin, actor_id=user_id, actor_role="student",
                event_type="enrollment_completed", target_id=user_id)
```

login สำเร็จ (auth.py:110 ก่อน redirect):
```python
log_audit_event(supabase_admin, actor_id=str(sb_user.id), actor_role=user["role"],
                event_type="login_success", target_id=str(sb_user.id))
```

ผลลัพธ์: audit_logs table ที่ queryable ผ่าน Supabase ครอบคลุม event สำคัญ 2 อันดับแรกที่หายไป

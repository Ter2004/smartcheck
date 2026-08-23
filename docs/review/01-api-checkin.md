# Review — `app/routes/api_checkin.py`

รีวิว ณ วันที่ 2026-08-20 | สแกนจากไฟล์จริง, grep จริงทุก caller

---

## Function Table

| ฟังก์ชัน | file:line | LOC | ถูกเรียกจากไหน | ต้นทุนต่อ call | คำตัดสิน | เหตุผล |
|---|---|---|---|---|---|---|
| `checkin()` | api_checkin.py:26 (รวม decorators) | 368 | HTTP POST `/api/checkin` ← `checkin_flow.js:14` (`this.apiUrl = opts.apiUrl \|\| '/api/checkin'`); Blueprint ลงทะเบียนที่ `app/__init__.py:101,107` | สูงมาก — 5 Supabase queries + frame decode + Moiré FFT + Temporal decode (×3 frames) + MiniFASNet inference + FaceNet512 embedding + cosine similarity | SPLIT | ฟังก์ชันเดียว 368 LOC ทำงาน 14 ขั้น; แต่ละ security layer ควรแยกเป็นฟังก์ชัน เพื่อให้ทดสอบและ audit แยกได้ — logic ถูกต้องแต่ไม่สามารถ unit-test แต่ละชั้นได้เลย |
| `antispoof_passive()` | api_checkin.py:398 (รวม decorators) | 17 | HTTP POST `/api/antispoof-passive` ← `checkin_flow.js:227` (`fetch('/api/antispoof-passive', ...)`) | ปานกลาง — MiniFASNet inference เท่านั้น, ไม่มี DB query | REFACTOR | ขนาดเล็ก, single responsibility ดี; แต่ขาด `@csrf_protect` (ดู Security Notes ด้านล่าง) — เพิ่ม 1 decorator |

---

## Pipeline Order

ลำดับที่โค้ดรันจริงใน `checkin()` ตั้งแต่ request เข้า → บันทึก attendance

### ชั้น Decorator (ก่อน function body — ใครทำ fail ก่อน exit ก่อน)

| ลำดับ | ขั้นตอน | file:line | Fail mode | หมายเหตุ |
|---|---|---|---|---|
| D-1 | `@login_required` | auth.py (decorator def) / api_checkin.py:27 | **FAIL-CLOSE** → redirect login | ตรวจ `session["user_id"]` ขาด |
| D-2 | `@role_required("student")` | auth.py / api_checkin.py:28 | **FAIL-CLOSE** → 403 | ตรวจ `session["role"] != "student"` |
| D-3 | `@_limiter.limit("5 per minute")` | api_checkin.py:29 | **FAIL-CLOSE** → 429 | key = `user_id` (ตาม CLAUDE.md) ถ้า login; fall-back IP ถ้าไม่ login (แต่ D-1 กัน case นั้นแล้ว) |
| D-4 | `@csrf_protect` | security_service.py / api_checkin.py:30 | **FAIL-CLOSE** → 403 | ตรวจ CSRF token ใน JSON/header |

### Function Body

| ลำดับ | ขั้นตอน | file:line | Fail mode | Early-return ข้ามอะไร |
|---|---|---|---|---|
| 0-pre | Input whitelist: `liveness_action` | api_checkin.py:46–47 | **FAIL-CLOSE** → 400 | ข้ามทุกขั้นหลัง (ถูก reject ก่อน DB แรก) |
| 0-pre | Input validate: `ble_rssi` range | api_checkin.py:50–56 | **FAIL-CLOSE** → 400 (ถ้าส่งค่า non-int); silent ignore ถ้า RSSI นอก range (-120…0) | ข้ามทุกขั้นหลัง |
| 0-pre | Required fields: `session_id`, `face_image` | api_checkin.py:58–59 | **FAIL-CLOSE** → 400 | ข้ามทุกขั้นหลัง |
| **0** | **Device token HMAC** | api_checkin.py:62–72 | **FAIL-OPEN** ถ้าไม่มี token เลย (legacy/first check-in); **FAIL-CLOSE** → 403 ถ้าส่ง token แต่ invalid หรือ uid ไม่ตรง | ถ้า reject: ข้าม 0b, 1–9 ทั้งหมด |
| **0b** | **Zero-trust frame validation** (`server_validate_frame`) | api_checkin.py:75–82 | **FAIL-CLOSE** → 400 | ถ้า reject: ข้าม 1–9; **สังเกต:** decode image ที่ 0b ก่อนรู้ว่า session valid (ขั้น 1) — เสีย CPU ถ้า session ปิดอยู่ |
| **1** | **Session open check** | api_checkin.py:85–96 | **FAIL-CLOSE** → 404/400 | ถ้า reject: ข้าม enrollment, window, BLE, liveness, face |
| 1a | Course enrollment check | api_checkin.py:99–112 | **FAIL-CLOSE** → 403 | ถ้า reject: ข้าม window, BLE, liveness, face |
| 1b | Check-in window (checkin_duration) | api_checkin.py:115–121 | **FAIL-CLOSE** → 400 (conditional: ข้ามถ้า field ว่าง) | ถ้า reject: ข้าม BLE, liveness, face |
| **2** | **BLE RSSI** | api_checkin.py:124–133 | **FAIL-OPEN ทั้งหมด** เมื่อ `BLE_CHECK_ENABLED=False` (default) → `ble_pass=True` เสมอ; FAIL-CLOSE → 400 เมื่อ enabled | ถ้า reject: ข้าม EAR, spoof layers, face |
| **3** | **EAR liveness (blink เท่านั้น)** | api_checkin.py:136–162 | **FAIL-CLOSE** → 400 (array invalid หรือ blink ไม่ผ่าน `ear_std<0.03` / `ear_min≥0.18`); action อื่น (`passive`,`nod`,…) ผ่าน array-validity check แต่ไม่ตรวจ EAR liveness จริง | `server_liveness_pass` ยังเป็น `False` จนถึงขั้น 4b — EAR ไม่ได้ set flag |
| **4a** | **Moiré FFT** (`detect_screen_moire`) | api_checkin.py:165–182 | **FAIL-CLOSE** → 400 (ทั้งตรวจเจอ screen และ exception) | ถ้า reject: ข้าม texture, temporal, MiniFASNet, face |
| **4a-2** | **Screen Texture FFT** (`detect_screen_texture`) | api_checkin.py:185–201 | **FAIL-CLOSE** → 400 (ทั้งตรวจเจอ screen และ exception) | ถ้า reject: ข้าม temporal, MiniFASNet, face |
| **4a-3** | **Temporal Variance** (ตรวจภาพนิ่ง) | api_checkin.py:204–243 | **FAIL-CLOSE** → 400; ต้องมี `face_images` list ≥ 2 frames; exception → fail-close | ถ้า reject: ข้าม MiniFASNet, face |
| **4b** | **MiniFASNet anti-spoof** (`check_anti_spoof`) | api_checkin.py:245–262 | **FAIL-CLOSE** → 400 (ทั้ง `is_real=False` และ exception); **ตรงนี้เท่านั้นที่ set `server_liveness_pass=True`** | ถ้า reject: ข้าม device binding, face |
| **5** | **Device binding** (threshold selector) | api_checkin.py:265–296 | **FAIL-CLOSE** → 400/403 ถ้า device_id ไม่ตรง / ถูกผูกกับคนอื่น; **FAIL-OPEN** ถ้าไม่มี device_id เลย → ใช้ `NEW_DEVICE_THRESHOLD` | device_id ว่างเปล่า → ข้าม device-collision check, threshold = 0.80 |
| **6** | **Embedding integrity** (`verify_embedding_integrity`) | api_checkin.py:315–323 | **FAIL-CLOSE** → 403 | ถ้า reject: ข้าม embedding extract, face compare |
| 6a | **Extract live embedding** (`extract_embedding`) | api_checkin.py:326–330 | **FAIL-CLOSE** → 400 | ถ้า fail: ข้าม face compare |
| 6b | **Face verification** (`verify_face_multi`) | api_checkin.py:332–342 | **FAIL-CLOSE** → 400 | ถ้า fail: ข้าม duplicate check, insert |
| **7** | **Duplicate check-in guard** | api_checkin.py:345–354 | **FAIL-CLOSE** → 400 (`already_checked=True`); TOCTOU guard อยู่ที่ DB unique constraint (**ขั้น 9**) | ถ้า duplicate: ข้าม status calc, insert |
| 8 | Attendance status (present/late) | api_checkin.py:357–363 | ไม่มี fail — ค่า default `"present"` | — |
| **9** | **DB insert + TOCTOU guard** | api_checkin.py:369–393 | **FAIL-CLOSE** → 400 (duplicate key 23505) / 500 (error อื่น); จับ race condition ผ่าน DB `UNIQUE(session_id, student_id)` | สุดท้าย |

---

### Early-return ที่ข้ามขั้นตอนหลัง (สรุป)

| ตำแหน่ง | เหตุ | ขั้นที่ถูกข้ามทั้งหมด |
|---|---|---|
| api_checkin.py:47 | `liveness_action` ไม่อยู่ใน whitelist | ทุกขั้น (0–9) |
| api_checkin.py:56 | `ble_rssi` ไม่ใช่ int | ทุกขั้น (0–9) |
| api_checkin.py:59 | ขาด `session_id`/`face_image` | ทุกขั้น (0–9) |
| api_checkin.py:68 | token ส่งมาแต่ invalid | 0b, 1–9 |
| api_checkin.py:71 | token valid แต่ `uid` ไม่ตรง | 0b, 1–9 |
| api_checkin.py:77–82 | frame ไม่ valid | 1–9 |
| api_checkin.py:96 | session ปิด | 1a, 1b, 2–9 |
| api_checkin.py:112 | ไม่ได้ enroll ในวิชา | 1b, 2–9 |
| api_checkin.py:121 | เกินเวลา check-in | 2–9 |
| api_checkin.py:129 | BLE RSSI ต่ำกว่า threshold | 3–9 |
| api_checkin.py:151–155 | blink EAR ไม่ผ่าน | 4a–9 |
| api_checkin.py:158–162 | EAR array invalid | 4a–9 |
| api_checkin.py:169–182 | Moiré screen ตรวจเจอ / exception | 4a-2, 4a-3, 4b–9 |
| api_checkin.py:188–201 | Texture screen / exception | 4a-3, 4b–9 |
| api_checkin.py:229–243 | Temporal static / exception | 4b–9 |
| api_checkin.py:248–262 | MiniFASNet fail / exception | 5–9 |
| api_checkin.py:275 | device_id ไม่ตรง | 6–9 |
| api_checkin.py:323 | integrity violation | 6a, 6b, 7, 9 |
| api_checkin.py:330 | embedding extract fail | 6b, 7, 9 |
| api_checkin.py:337–342 | face similarity ต่ำกว่า threshold | 7, 9 |

---

## Security Notes (จุดที่น่าสังเกต)

### 1. `antispoof_passive()` ขาด `@csrf_protect`
`api_checkin.py:398–401` — endpoint รับ `face_image` จาก client โดยไม่มี CSRF guard  
แม้ไม่ write DB แต่ทำให้ attacker cross-origin ส่ง image ไป probe MiniFASNet score ได้  
→ เพิ่ม `@csrf_protect` ตามแบบเดียวกับ `checkin()`

### 2. BLE ทั้งชั้น FAIL-OPEN by default
`api_checkin.py:124` — `BLE_CHECK_ENABLED` ค่า default = `False`  
→ นักศึกษาสามารถ check-in จากที่ไหนก็ได้ใน production ถ้าลืม set env var  
→ ควร default `True` หรืออย่างน้อย log warning ตอน startup ถ้า disabled

### 3. Frame validation (0b) วางไว้ก่อน session check (1)
`api_checkin.py:75` ก่อน `api_checkin.py:85`  
→ decode + validate image ทุกครั้งก่อนรู้ว่า session ยังเปิดอยู่  
→ ถ้า session ปิดบ่อย เสีย CPU ไปกับ frame validation ที่ไม่จำเป็น  
→ อาจย้าย session check ขึ้นก่อน 0b ได้ (ถูกกว่า 1 DB query vs full image decode)

### 4. Redundant `if` ที่ line 211
```python
# line 205-210 (outer guard):
if not isinstance(face_images_list, list) or len(face_images_list) < 2:
    return ...  # <-- early return

# line 211 (inner check — always True after outer guard):
if isinstance(face_images_list, list) and len(face_images_list) >= 2:
```
เงื่อนไข `line 211` เป็น `True` เสมอหลังผ่าน outer guard — `else: raise ValueError` ที่ line 236 ไม่มีทางถูกเรียก  
→ dead code; ลบ inner `if` ออก เหลือแค่ body ข้างใน

### 5. `from datetime import timedelta` อยู่ในฟังก์ชัน (line 117)
Python cache module ไว้แล้ว แต่การ import ใน function scope ยังมี overhead เล็กน้อยทุก call  
→ ย้ายขึ้น top-level import

### 6. Device token step 0: FAIL-OPEN ถ้าไม่มี token
`api_checkin.py:72` — "legacy / first check-in — allowed"  
→ นักศึกษาที่ไม่มี device token ผ่านขั้น 0 ได้เสมอ และใช้ `NEW_DEVICE_THRESHOLD = 0.80`  
→ ไม่ใช่ bug แต่ควรระวัง: ถ้า token requirement ถูก enforce ในอนาคต ต้องแก้ที่นี่ด้วย

---

---

## Q: liveness_action whitelist กับ EAR ขั้น 3

### Whitelist บรรทัดดิบ (api_checkin.py:45)

```python
_ALLOWED_LIVENESS_ACTIONS = {"passive", "blink", "nod", "turn_left", "turn_right", "smile", "raise_eyebrows"}
```

7 ค่า — `passive`, `blink`, `nod`, `turn_left`, `turn_right`, `smile`, `raise_eyebrows`

### Client ส่งค่าไหนมาจริง

| file:line | ค่าที่ส่ง | เงื่อนไข |
|---|---|---|
| `checkin_flow.js:310` | `'passive'` | default (spoof score ≥ 0.98 — passive ผ่าน) |
| `checkin_flow.js:318` | `'blink'` หรือ `'turn_left'` (random 50/50) | spoof score < 0.98 → ขอ manual challenge |
| `checkin_flow.js:330` | ค่าจาก `action` ข้างต้น | ถ้า liveness detector ผ่าน |

Client **ไม่เคย** ส่ง `nod`, `turn_right`, `smile`, `raise_eyebrows` — whitelist มี 7 ค่าแต่ client ใช้แค่ 3 ค่า

### ถ้าส่งค่าที่ไม่ใช่ "blink" → ขั้น 3 ถูกข้ามใช่หรือไม่?

**ใช่**

โค้ด api_checkin.py:146:
```python
if liveness_action == "blink":
    ear_std = float(np.std(ear_arr))
    ear_min = float(np.min(ear_arr))
    if ear_std < 0.03 or ear_min >= 0.18:
        return ...  # fail
```

เมื่อ `liveness_action` เป็น `"passive"`, `"turn_left"` หรือค่าอื่นใน whitelist — `if liveness_action == "blink":` เป็น False ทั้งหมด  
โค้ดก็ยัง validate รูปแบบ EAR array (lines 139–142) แต่ **ไม่ได้ตัดสิน pass/fail จาก EAR**  
`server_liveness_pass` ยังเป็น `False` จนกว่าจะผ่าน MiniFASNet ที่ขั้น 4b

**ผลลัพธ์จริง:**
- ส่ง `"passive"` → ขั้น 3 = array format check เท่านั้น, liveness decision ทั้งหมดตกไปที่ MiniFASNet
- ส่ง `"turn_left"` → เหมือนกัน
- ส่ง `"blink"` → EAR ตรวจจริง **และ** MiniFASNet ตรวจซ้ำ

whitelist 7 ค่า แต่มีเพียงค่าเดียว (`"blink"`) ที่ทำให้ขั้น 3 มีความหมาย — อีก 6 ค่า EAR layer เป็น dead path

---

## Top 3 ที่คุ้มสุดถ้าแก้

### 🥇 1. เพิ่ม `@csrf_protect` บน `antispoof_passive()` (api_checkin.py:401)
**ความยาก:** ต่ำมาก — เพิ่ม 1 บรรทัด  
**ผลลัพธ์:** ปิด cross-origin oracle ที่ attacker สามารถ probe MiniFASNet score ซ้ำ ๆ ได้  
**ผลข้างเคียง:** `checkin_flow.js:227` ต้องส่ง CSRF header ด้วย — ตรวจว่า fetch ณ จุดนั้นแนบ header อยู่แล้วหรือไม่

### 🥈 2. ย้าย session check (ขั้น 1) ขึ้นก่อน frame validation (0b)
**ความยาก:** ต่ำ — ย้าย block ≈20 บรรทัด  
**ผลลัพธ์:** ลด CPU waste: session closed → reject ด้วย 1 DB query แทนที่จะ decode + validate ภาพก่อน  
**ผลข้างเคียง:** ไม่มี — frame validation ยังอยู่ครบ เพียงแต่ทำหลัง session check

### 🥉 3. แก้ redundant `if` ที่ line 211 + SPLIT `checkin()` เป็น sub-functions
**ความยาก:** ปานกลาง (refactor ไม่ใช่ bug fix)  
**ผลลัพธ์:** ลบ dead code branch; แต่ละ security layer (validate_input, check_session, check_liveness, check_spoof, verify_face, record_attendance) กลายเป็น function ที่ทดสอบ/audit แยกได้  
**ผลข้างเคียง:** ต้องระวัง shared state (`raw_frame`, `server_liveness_pass`, `face_threshold`) ที่ต้องส่งผ่าน parameter หรือ dataclass

---

## Critical Findings

| ID | ความรุนแรง | สถานะ |
|---|---|---|
| F-1 | 🔴 สูง | ยังไม่แก้ |
| F-2 | 🟡 ต่ำ | ยังไม่แก้ |
| F-3 | 🟡 ต่ำ | ยังไม่แก้ |
| F-4 | 🟢 Cosmetic | ยังไม่แก้ |

---

### F-1 — EAR liveness ถูกข้ามได้ทุกครั้งที่ client ส่ง "passive"

**ตำแหน่ง:** `api_checkin.py:45` (whitelist), `api_checkin.py:146` (`if liveness_action == "blink":`)

**สิ่งที่เกิด:**
- server รับ `liveness_action` จาก client โดยตรง และตรวจแค่ว่าอยู่ใน whitelist 7 ค่า
- EAR check ที่ขั้น 3 ทำงานจริงเฉพาะเมื่อ `liveness_action == "blink"` เท่านั้น
- `checkin_flow.js:310` ส่ง `'passive'` เมื่อ spoof score ≥ 0.98 — ซึ่งเป็นเส้นทางปกติของคนจริง
- client ที่แก้ JS หรือส่ง request ตรงสามารถส่ง `"passive"` ได้เสมอ → EAR ถูกข้ามทุกครั้ง

**ผลกระทบ:** ไม่มี challenge-response server-enforced เหลือแต่ passive detection (MiniFASNet ขั้น 4b) เท่านั้น — ถ้า MiniFASNet โดน bypass ได้ด้วย liveness layer ไม่มีชั้นสำรอง

**ทางแก้ที่เป็นไปได้:**
- **(a)** ลบ `if liveness_action == "blink":` ที่ line 146 ให้ EAR รันทุกครั้งไม่ว่า action จะเป็นอะไร
- **(b)** server-side challenge: server สุ่ม action เก็บใน session ก่อน check-in, ตอน submit ตรวจว่า action ตรงกับที่ server สั่ง — client ปลอม action ไม่ได้

**หมายเหตุ:** ทางแก้ (b) ปิดปัญหาได้ถาวรกว่า (a) เพราะ (a) ยังให้ client เลือก action เองอยู่

---

### F-2 — Dead entries ใน whitelist

**ตำแหน่ง:** `api_checkin.py:45`

`"nod"`, `"turn_right"`, `"smile"`, `"raise_eyebrows"` อยู่ใน whitelist แต่ `checkin_flow.js` ไม่เคยส่งค่าเหล่านี้  
(grep ทั้งโปรเจกต์ไม่พบ caller นอกจาก `"blink"` และ `"turn_left"`)  
ถ้า F-1 ยังไม่ถูกแก้ ค่าเหล่านี้ก็เป็นเพียง whitelist noise ที่ไม่มีผลใด ๆ

---

### F-3 — `antispoof_passive()` ขาด `@csrf_protect`

**ตำแหน่ง:** `api_checkin.py:398–401`

endpoint ไม่มี CSRF guard ต่างจาก `checkin()` ที่มี `@csrf_protect`  
**ผลกระทบต่ำ** — endpoint นี้ไม่เขียน DB, ทำเพียง MiniFASNet inference แล้ว return score  
ความเสี่ยงจำกัดที่ cross-origin probe score ซ้ำ ๆ เพื่อ calibrate spoof image

---

### F-4 — `else: raise ValueError` unreachable (line 236)

**ตำแหน่ง:** `api_checkin.py:211`, `api_checkin.py:236`

```python
# line 205–210: outer guard — early return ถ้า list ไม่ครบ
if not isinstance(face_images_list, list) or len(face_images_list) < 2:
    return ...

# line 211: inner check — condition นี้ True เสมอหลัง outer guard
if isinstance(face_images_list, list) and len(face_images_list) >= 2:
    ...
else:
    raise ValueError("not enough valid temporal frames")  # line 236 — ไม่มีทางถึง
```

`else` ที่ line 236 เป็น dead code — ไม่กระทบ security แต่ทำให้โค้ดอ่านเข้าใจผิดว่า path นั้นเป็นไปได้

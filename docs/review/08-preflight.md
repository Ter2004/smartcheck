# Preflight: ตัดสิน F-13 impact + หารูของ F-11 fix

วันที่: 2026-08-24 — ต่อจาก `07-auth.md` (อ่านอย่างเดียว ไม่แก้โค้ด ไม่รัน server/test)

---

## งานที่ 1 — ขนาดผลกระทบจริงของ F-13

### 1. `get_rate_limit_key()` คืนค่าอะไรตอนไหน (app/__init__.py:21-34)

```python
def get_rate_limit_key():
    try:
        from flask import session as _session
        uid = _session.get("user_id")
        if uid:
            return f"user:{uid}"
    except Exception:
        pass
    return f"ip:{get_remote_address()}"
```

**Fallback ไป IP เกิดเฉพาะตอน `session.get("user_id")` เป็น falsy เท่านั้น** — คือ **ก่อน login** (หรือหลัง logout/session หมดอายุ) เท่านั้น หลัง login สำเร็จ key จะเป็น `user:{uid}` เสมอ **ไม่แตะ IP เลย**

**ข้อสำคัญ:** ค่านี้คือ key function ของ `Limiter` object เอง (`app/__init__.py:52-56`, `key_func=get_rate_limit_key`) — ใช้เป็น**ค่า default**สำหรับทุก `@_limiter.limit(...)` ที่**ไม่ได้ระบุ `key_func` ของตัวเอง** route ที่ตรวจใน checkin flow (`/api/checkin` api_checkin.py:30, `/api/antispoof-passive` api_checkin.py:425) **ไม่ได้ระบุ `key_func`** → ใช้ตัวนี้ → **หลัง login แล้ว ทั้งคู่ key ด้วย `user:{uid}` ไม่ใช่ IP** — F-13 (IP collapse) **ไม่กระทบ 2 endpoint นี้เลย**

route เดียวที่ระบุ `key_func` เป็นอย่างอื่นชัดเจนคือ `/login` POST (auth.py:54, `key_func=get_remote_address`) — บังคับใช้ raw IP เสมอไม่ว่าจะ login หรือไม่ (สมเหตุสมผลเพราะยังไม่มี user_id ให้ใช้)

### 2. `default_limits` — ตั้งไว้เท่าไหร่ ครอบคลุมอะไรบ้าง

`app/__init__.py:54` → `default_limits=["200 per hour"]`

อ่านซอร์ส `flask_limiter` จริงที่ติดตั้งในโปรเจกต์ (`venv/Lib/site-packages/flask_limiter/extension.py:504`, `manager.py:73-133`) ยืนยันว่า:
- `.limit(...)` decorator มี `override_defaults: bool = True` เป็นค่า default — **route ที่มี `@_limiter.limit(...)` ของตัวเอง จะไม่โดน `default_limits` ซ้อนทับด้วย** (ใช้แค่ limit เฉพาะของตัวเอง) — โค้ดในโปรเจกต์นี้ไม่มีที่ไหน pass `override_defaults=False` เลย (grep ยืนยัน)
- แต่ `Limit.method_exempt` (`wrappers.py:120-124`) เช็คว่า request method ตรงกับ `methods=[...]` ของ decorator ไหม — `/login` มี `methods=["POST"]` เฉพาะ ดังนั้น**สำหรับ GET `/login`** limit เฉพาะนี้ไม่ apply (`method_exempt=True`) → `manager.py:111,128-133` (`explicit_limits_exempt = all(limit.method_exempt ...)`) ทำให้ **`default_limits` กลับมาใช้กับ GET `/login` แทน** (key = `ip:{remote_addr}` เพราะยังไม่ login)

**สรุป:** `default_limits (200/hour)` ครอบคลุม:
- ทุก route ที่ไม่มี `@_limiter.limit` เป็นของตัวเองเลย (เช่น `/`, `/register`, `/logout`, `/student/dashboard`, `/student/checkin`, `/student/enroll`, เกือบทุก route ใน admin.py/teacher.py)
- **GET** `/login` เท่านั้น (ไม่ใช่ POST เพราะ POST มี limit เฉพาะที่ override ไปแล้ว)

ไม่ครอบคลุม: ทุก route ที่มี `@_limiter.limit` ของตัวเอง ตอนที่ method ตรง (POST `/login`, `/api/checkin`, `/api/antispoof-passive`, `/student/api/consent`, `/api/enroll`, `/api/self_verify`, `/api/spoof_check`, `/api/reset-liveness`)

### 3. Request จริงต่อการเช็คชื่อ 1 ครั้ง

ไล่จาก `app/static/js/checkin_flow.js` (grep `fetch(` ทั้งไฟล์ — เจอ 2 จุดเท่านั้น คือบรรทัด 227 และ 354) + route ที่ผ่านก่อนหน้า:

| ลำดับ | Request | Endpoint | Limiter bucket (หลัง login) | ครั้ง |
|---|---|---|---|---|
| 1 | GET หน้า login | `/login` | `default_limits`, key=IP (ยังไม่ login) | 1 |
| 2 | POST login | `/login` | เฉพาะ `10/min`, key=**raw IP เสมอ** | 1 |
| 3 | GET dashboard (redirect หลัง login) | `/student/dashboard` | `default_limits`, key=`user:{uid}` | 1 |
| 4 | GET หน้าเช็คชื่อ | `/student/checkin` | `default_limits`, key=`user:{uid}` | 1 |
| 5 | POST passive spoof check (auto, ครั้งเดียว — `passiveSent` flag กันยิงซ้ำ, checkin_flow.js:225) | `/api/antispoof-passive` | เฉพาะ `20/min`, key=`user:{uid}` | 1 |
| 6 | POST ส่งเช็คชื่อ (`_submitCheckin`, checkin_flow.js:348-354) | `/api/checkin` | เฉพาะ `5/min`, key=`user:{uid}` | 1 |

**รวม 6 requests/คน สำหรับ 1 รอบ login+เช็คชื่อ ครั้งแรกของวัน** (ไม่นับ static asset — ดูข้อ 5) ถ้าเข้าเช็คชื่อคาบถัดไปแบบ session login เดิมยังอยู่ (ไม่ต้อง login ใหม่) เหลือแค่ข้อ 3-6 = **4 requests/รอบ**

หมายเหตุ: `/student/api/spoof_check` (student.py:952, "เรียก ~8 ครั้งต่อ enrollment" ตาม docstring บรรทัด 960) เป็น endpoint ของ**ตอนลงทะเบียนใบหน้าครั้งแรก (enrollment)** ไม่ใช่ check-in — `checkin_flow.js` ไม่เรียก endpoint นี้เลย (ไม่เจอใน grep) — ถ้า 50 คนที่ทดสอบ**ยังไม่เคย enroll มาก่อน** ต้องบวก enrollment flow เพิ่ม (`/api/consent` ×1, `/api/spoof_check` ×~8, `/api/enroll` ×1) แยกเป็นอีกเคสหนึ่ง ไม่ใช่ส่วนของ "เช็คชื่อ"

### 4. ตัวเลข 50 คน — ชนไหม ชนตอนไหน

ถ้า `remote_addr` ยุบเหลือ IP เดียวจริง (สมมติฐานตั้งต้นของ F-13):

- **`default_limits` (200/hour)**: 50 คน × (GET /login 1 ครั้ง) = 50 requests ในบัคเก็ตนี้ (ถ้าเช็คชื่อรอบแรกของวันทุกคน) — **ไม่ชน** เพราะ 50 < 200/hour ขาดอีกมาก เว้นแต่มีการ refresh หน้า login ซ้ำหลายรอบต่อคน (ไม่นับใน static analysis นี้ — ต้องยืนยันเพิ่มถ้าอยากรู้ pattern refresh จริงของผู้ใช้)
- **`/login` POST (10/min, IP เสมอ)**: **นี่คือจุดชนจริง** ถ้านักศึกษา 50 คนพยายาม login ภายในหน้าต่าง 60 วินาทีเดียวกัน (เช่น ต้นคาบเรียนที่ทุกคนเปิดแอปพร้อมกัน) — **คนที่ 11 เป็นต้นไปในหน้าต่างนั้นจะโดน HTTP 429 ทันที** ไม่ว่าจะเป็นความพยายามแรกของแต่ละคนก็ตาม เพราะ bucket เดียวกันหมด (fixed-window strategy — ยืนยันจาก `extension.py:346-347`, ค่า default `"fixed-window"`, โปรเจกต์นี้ไม่ได้ override เป็นอย่างอื่น)
- **`/api/checkin`, `/api/antispoof-passive`**: key เป็น `user:{uid}` เสมอหลัง login — **ไม่ชนเพราะ 50 คนเลย ไม่ว่า IP จะยุบหรือไม่** (คนละ bucket ต่อคน, 5/min และ 20/min ต่อคนก็เกินพอสำหรับเช็คชื่อ 1 ครั้ง)

**ช่วงตัวเลข (ต่ำ-สูง):** ถ้ากระจาย login เกิน 60 วินาที ในอัตรา ≤10 คน/นาที → **ไม่ชนเลย** (0 คนโดน 429) ถ้า login พร้อมกันในหน้าต่างเดียว (burst) → **ชนที่คนที่ 11** ของหน้าต่างนั้น, คนที่ 11-50 (สูงสุด 40 คน) โดน 429 และต้องรอ/retry รอบถัดไป (10 คน/นาทีถัดไปเรื่อย ๆ) — ตัวเลขจริงขึ้นกับ**อัตรา** ไม่ใช่แค่จำนวนรวม 50 คน

### 5. `@limiter.exempt` / `limiter.enabled = False`

grep ทั้ง `app/` ไม่พบทั้งคู่ — **ไม่มี exemption ใด ๆ ที่ตั้งไว้เอง** endpoint เดียวที่ยกเว้นอัตโนมัติคือ Flask `static` (ยืนยันจาก `flask_limiter/extension.py:975`, `endpoint.split(".")[-1] == "static"` — hardcode ไว้ในไลบรารีเอง ไม่ใช่โค้ดโปรเจกต์) → **static asset (CSS/JS/รูป) ไม่นับรวมในการนับข้อ 3-4 เลย ถูกต้องแล้วที่ไม่นับ**

### 6. ข้อสรุป F-13

**ไม่ต้องแก้ก่อนทดสอบโหลด 50 คนสำหรับ flow เช็คชื่อ** เพราะ endpoint ที่ทดสอบโหลดจริง ๆ (`/api/checkin`, `/api/antispoof-passive`, และ GET dashboard/checkin page) ทั้งหมด key ด้วย `user:{uid}` ไม่ใช่ IP อยู่แล้วหลัง login — F-13 **ไม่แตะ flow เหล่านี้เลย** (ต่างจากที่ `07-auth.md` เขียนไว้กว้าง ๆ ว่า "rate limiter" มีปัญหา — จริง ๆ แคบกว่านั้นมาก คือกระทบแค่ `POST /login` จุดเดียว)

**ข้อควรระวังสำหรับตัว load test เอง (ไม่ใช่เหตุผลให้ต้องแก้ F-13 ก่อน):** ถ้า load test สั่ง login 50 virtual user จากเครื่องทดสอบเครื่องเดียว (IP เดียวจริง ๆ ไม่ใช่แค่ IP ที่ยุบจาก proxy) จะชน `10/min` เหมือนกันไม่ว่า F-13 จะถูกแก้หรือไม่ — เพราะ ProxyFix แก้ปัญหาการยุบ IP จาก reverse proxy เท่านั้น ไม่ได้ทำให้ load test ที่มาจาก IP เดียวจริง ๆ กลายเป็นหลาย IP **ทางแก้ของ load test คือ pace การ login ให้ ≤10 ครั้ง/นาที (หรือ stagger เกิน 1 นาที) ไม่เกี่ยวกับ F-13**

**ลำดับ F-13:** คงไว้ที่ 🟡 ปานกลาง ตามที่ `07-auth.md` ประเมิน แต่ **ปรับ scope ให้แคบลง**: กระทบแค่ `POST /login` (และ GET /login เล็กน้อยแต่ไม่มีทางชนในทางปฏิบัติ) ไม่กระทบ check-in throughput เลย — แก้เมื่อไหร่ก็ได้ ไม่ใช่ blocker ของการทดสอบโหลด 50 คนรอบนี้

---

## งานที่ 2 — รูที่ทำให้ fix ของ F-11 ไม่ครบ

### 1-2. ตาราง route ทั้งหมด (grep `@*_bp.route` + `@login_required` + `@role_required` ทุกไฟล์ใน `app/routes/`)

**admin.py (prefix `/admin`, 21 routes):** ทุก route มี `@login_required` + `@role_required("admin")` ครบ **100%** (บรรทัด 15-700, ตรวจทีละคู่แล้วไม่มีข้อยกเว้น)

**teacher.py (prefix `/teacher`, 8 routes):** ทุก route มี `@login_required` + `@role_required("teacher")` ครบ **100%** (บรรทัด 16-480)

**student.py (prefix `/student`, 9 routes):** ทุก route มี `@login_required` + `@role_required("student")` ครบ **100%** (บรรทัด 53-1105)

**api_checkin.py (ไม่มี prefix, 2 routes):** ทั้งคู่มี `@login_required` + `@role_required("student")` ครบ **100%** (บรรทัด 27-424)

**auth.py (ไม่มี prefix, 5 route definitions):**

| Route | Method | `login_required` | เช็ค auth ยังไง | เข้าถึงข้อมูลอะไรไหม |
|---|---|---|---|---|
| `/` | GET | ❌ | เช็คเอง `"user_id" not in session` (auth.py:48) | ไม่ — แค่ redirect ต่อ |
| `/login` | GET+POST | ❌ | ไม่ต้อง — ตั้งใจให้ public (pre-auth) | ไม่ |
| `/change-password` | GET+POST | ✅ (auth.py:119) | — | ใช่ (ข้อมูลของตัวเองเท่านั้น) |
| `/register` | GET | ❌ | ไม่ต้อง — public, ปิดสนิทอยู่แล้ว | ไม่ (แค่ flash ข้อความ) |
| `/logout` | GET | ❌ | ไม่ต้อง — ตั้งใจให้เรียกได้เสมอ | ไม่ |

**สรุปข้อ 2:** **ไม่พบ route ไหนที่เข้าถึงข้อมูลได้โดยไม่ผ่าน `login_required`** — 40 routes ใน admin/teacher/student/api_checkin.py มีครบทุกตัว ส่วน 4 routes ที่ไม่มีใน auth.py (`/`, `/login`, `/register`, `/logout`) ล้วนไม่ส่งข้อมูลอะไรกลับให้ — **ไม่ใช่ช่องโหว่**

**กรณี `/` (index) โดยเฉพาะ:** เป็น route เดียวที่ "ดูเหมือน" จะเป็นรูตามสมมติฐานของโจทย์ (เช็ค session เองใน body แบบเดียวกับที่ระบุ) แต่ไล่โค้ดจริง (auth.py:45-50) พบว่า `/` ทำแค่ 2 อย่าง: ยังไม่ login → redirect ไป `/login`, login แล้ว → `_redirect_by_role()` ส่งต่อไป `/student/dashboard` (หรือ teacher/admin) **ทันที** ซึ่ง route ปลายทางนั้นมี `@login_required` ครบอยู่แล้ว → ถ้าใส่เช็ค `must_change_password` เข้าไปใน `login_required` ตามที่ 07-auth.md เสนอ นักศึกษาที่เข้า `/` จะโดน redirect ต่อไปอีกทอดไปที่ `/change-password` เหมือนกัน (แค่เพิ่ม 1 hop) — **ไม่ใช่รูที่ทำให้ fix ไม่ครบ**

**ข้อสรุป:** ไม่พบรูที่ทำให้ fix ของ F-11 (ใส่เช็คใน `login_required`) ไม่ครบคลุมตามสมมติฐานของโจทย์ — เพราะทุก route ที่มีข้อมูล/การกระทำจริงผูกกับ `login_required` เป็น 1:1 อยู่แล้วทั้งโปรเจกต์

### 3. Route ที่เรียกด้วย `fetch()`/`XMLHttpRequest` — fix แบบ `redirect()` จะพังเงียบ ๆ

grep `fetch(` ทั้ง `app/` (static JS + templates) พบ 9 จุดเรียก 8 endpoint ที่ไม่ซ้ำกัน:

| Endpoint | เรียกจาก |
|---|---|
| `/api/checkin` | checkin_flow.js:354 |
| `/api/antispoof-passive` | checkin_flow.js:227 |
| `/student/api/consent` | enrollment_flow.js:304 |
| `/student/api/enroll` | enrollment_flow.js:1240, enrollment_circular.js:443 |
| `/student/api/spoof_check` | enrollment_flow.js:215, enrollment_circular.js:379 |
| `/student/api/reset-liveness` | enrollment_flow.js:1405 |
| `/student/api/withdraw-consent` | dashboard.html:223 |
| `/teacher/session/<id>/override` | teacher/session_view.html:302 |

ทั้ง 8 endpoint นี้มี `@login_required` (จึงโดน must_change_password gate หลังแก้ F-11 ด้วย) แต่ fix แบบ `return redirect(url_for("auth.change_password"))` จะทำให้ `fetch()` ฝั่ง client ได้ response เป็น HTML (หน้า change-password) แทน JSON ที่ code คาดหวัง — `res.json()` จะ throw → error handler ทั่วไปโชว์ข้อความกำกวมแบบ "ไม่สามารถเชื่อมต่อ server ได้" (ดูตัวอย่างจริงใน checkin_flow.js:386-390 catch block) แทนที่จะบอกตรง ๆ ว่า "ต้องเปลี่ยนรหัสผ่านก่อน" — **ไม่ใช่ security bypass** (การกระทำจริงยังถูกบล็อกอยู่ เพราะ redirect เกิดก่อนถึง view function) **แต่เป็น UX gap ที่ fix ต้องคำนึงถึง**: ต้องแยกเงื่อนไขตอบกลับระหว่าง request ที่เป็น JSON/fetch (ควรตอบ 403 + JSON error) กับ page navigation ปกติ (ตอบ redirect ตามเดิม) — ไม่ลงรายละเอียด fix ตามที่โจทย์ขอ แค่บันทึกไว้ว่าเป็นสิ่งที่ fix ของ F-11 ต้องแก้ไม่ให้พลาด

**route อื่นที่เหลือ** (dashboard, enroll page, checkin page, ทุกหน้า admin/teacher management, form POST ทั้งหมดใน admin.py/teacher.py) เป็น full-page navigation (GET หรือ `<form method="POST">` ธรรมดา) — `redirect()` ทำงานถูกต้องตามปกติ ไม่มีปัญหา

**หมายเหตุนอกขอบเขตแต่เจอระหว่างทาง:** `/student/api/self_verify` (student.py:740), `/admin/api/reset-enrollment/<id>` (admin.py:698), `/teacher/api/reset-enrollment/<id>` (teacher.py:478) — grep `fetch(`/`reset-enrollment` ทั้งโปรเจกต์ไม่เจอ caller เลยสักที่ (self_verify ตรงกับ Q-5 ที่ 99-summary.md บันทึกไว้แล้วว่าเป็น dead code — ยืนยันซ้ำ ณ ที่นี้)

### 4. `role_required` ไม่มี `login_required` คู่กัน

**ไม่พบ** — grep ยืนยันว่าทุกจุดที่มี `@role_required(...)` (40 จุดใน admin/teacher/student/api_checkin.py) มี `@login_required` วางอยู่บรรทัดก่อนหน้าเสมอ ไม่มีข้อยกเว้น

ลำดับ decorator ที่เจอทุกที่คือ:
```python
@xxx_bp.route(...)
@login_required
@role_required(...)
def view(): ...
```
ตาม semantics ของ Python decorator stacking — decorator ที่อยู่**บนสุด**ห่อ (wrap) เป็นชั้นนอกสุด และรันโค้ดของตัวเอง**ก่อน** decorator ที่อยู่ล่างกว่าเสมอ ดังนั้น `login_required` รันก่อน `role_required` ทุกครั้ง — การเพิ่มเช็ค `must_change_password` เข้าไปใน `login_required` (auth.py:17-25) จะทำงาน**ก่อน**ถึง `role_required` และก่อนถึงตัว view function เสมอ ไม่มีทางให้ `role_required` หรือ view รันไปก่อนโดยไม่ผ่านเช็คใหม่นี้

---

## สรุปสั้น

| ประเด็น | ข้อสรุป |
|---|---|
| F-13 กระทบแค่ไหน | แคบกว่าที่ `07-auth.md` บอก — เฉพาะ `POST /login` (IP-keyed เสมอ) เท่านั้น check-in endpoints ทั้งหมด key ด้วย `user:{uid}` ไม่โดน |
| ต้องแก้ F-13 ก่อนทดสอบโหลด 50 คนไหม | ไม่ต้อง — endpoint ที่ทดสอบโหลด (`/api/checkin`, `/api/antispoof-passive`) ไม่ถูกกระทบ ถ้า test script login ทีละคน/pace ให้ ≤10 คนต่อนาทีก็ผ่านสบาย |
| F-11 fix มีรูไหม | ไม่พบรูที่ทำให้ bypass ได้จริง — ทุก route ที่มีข้อมูลผูกกับ `login_required` 1:1 ครบ 100% (40/40 routes นอก auth.py) |
| ข้อควรระวังตอน implement F-11 | 8 endpoint ที่เรียกด้วย `fetch()` ต้องได้ JSON response ไม่ใช่ redirect ไม่งั้น UX จะพัง (ไม่ใช่ security bug แต่ทำให้ error message งง) |

# Review: `app/routes/auth.py`

วันที่: 2026-08-24
ไฟล์: 199 LOC — blueprint `auth_bp`, ไม่มี URL prefix (`/`, `/login`, `/change-password`, `/register`, `/logout`)
บริบท: ต่อจาก `05-security-service.md` (ซึ่งครอบคลุม auth.py บางส่วนแล้ว และพบ F-7 — CSRF บน `/change-password` แก้ไปแล้วด้วย commit `f7b7773`)
สมมติฐานรอบนี้: หา "ช่องแบบ F-7" ที่เหลือ — คือ control ที่ *ดูเหมือน* ป้องกันอยู่ แต่จริง ๆ ไม่ enforce หรือ enforce ไม่ครบ ไม่ใช่แค่ไล่ checklist

**สถานะ F-7 ที่ตรวจซ้ำ:** `@csrf_protect_form` อยู่ที่ auth.py:120 ✅, current-password verify อยู่ที่ auth.py:140–152 ✅ — ยืนยันว่าแก้จริงและถูกต้อง ไม่มีข้อสังเกตเพิ่ม

---

## 1. Route Table — CSRF / Rate Limit / Login / Role

| Route | Method | `@login_required` | `@role_required` | CSRF | Rate limit | หมายเหตุ |
|---|---|---|---|---|---|---|
| `/` (`index`) | GET | ไม่ใช้ decorator, เช็ค `session` เองใน body (auth.py:48) | — | N/A (GET, ไม่มี side effect) | ไม่มีเฉพาะ (ใช้ `default_limits: 200/hour` จาก app-level) | เจตนาไม่ล็อกอินได้ก็ผ่าน (redirect ไป login) — ถูกต้อง |
| `/login` GET | GET | ไม่ต้อง | — | N/A | ไม่มี | render ฟอร์มเฉย ๆ — ไม่มี state change |
| `/login` POST | POST | ไม่ต้อง (pre-auth) | — | ❌ ไม่มี — **ยอมรับได้** (ประเมินซ้ำจาก 05-security-service.md: session ยังไม่มี csrf_token ก่อน login, login CSRF ไม่ทำให้ attacker เข้าบัญชีเหยื่อได้เพราะ regenerate session หลัง login) | ✅ `10 per minute` (auth.py:54) แต่ดู **F-13** — key มาจาก raw IP ที่อาจถูก proxy บีบให้เหลือ IP เดียว | — |
| `/change-password` GET | GET | ✅ (auth.py:119) | — | N/A | ไม่มีเฉพาะ | — |
| `/change-password` POST | POST | ✅ | — | ✅ `@csrf_protect_form` (auth.py:120) | ไม่มีเฉพาะ — อาศัย default 200/hour เท่านั้น ดู **F-14** | current-password verify ผ่าน Supabase sign-in — ดู **F-12** เรื่อง shared client |
| `/register` | GET | ไม่ต้อง | — | N/A | ไม่มี | route ปิดถาวร, ไม่มี POST handler เลย — ไม่มีทางส่ง role มาเองได้ **(ตรวจแล้วผ่าน — ไม่เหมือน CC-1a)** |
| `/logout` | GET | ไม่ต้อง | — | N/A (GET, ไม่มี CSRF token ต้องใช้เพราะ effect คือ "ทำให้ตัวเอง log out" — impact ต่ำสุดถ้าโดน forge) | ไม่มี | ดู **F-12** เรื่อง `supabase.auth.sign_out()` |

ไม่มีช่องว่างใหม่ในตารางนี้ตัวมันเอง (ทุก POST ที่ต้อง login มี CSRF ครบ) — ปัญหาที่เจอรอบนี้ไม่ใช่ "ลืมใส่ decorator" แบบ F-7 แต่เป็น **decorator ใส่ครบแล้วแต่ logic ข้างในไม่ทำตามที่ควร** (F-11, F-12) และ **rate limit ใส่แล้วแต่ key function ผิด context การ deploy** (F-13)

---

## 2. Login (auth.py:53–115)

| หัวข้อ | ผล | รายละเอียด |
|---|---|---|
| Session fixation | ✅ ผ่าน (มี caveat คุณภาพ) | regenerate sid ก่อน set ข้อมูล user ทุกครั้ง (auth.py:83–96) — เช็คกับ flask_session 0.8.0 จริงในโปรเจกต์ (`venv/Lib/site-packages/flask_session`) ยืนยันว่า `save_session()` ใช้ `session.sid` ที่ถูก reassign แล้วตอน insert แถวใหม่ → sid เก่าที่ attacker fixate ไว้ไม่มีทางได้รับสถานะ authenticated จริง ดู **Q-14** สำหรับปัญหาคุณภาพ (ไม่ใช่ security) ของ cleanup โค้ดส่วนนี้ |
| User enumeration / timing | ✅ ผ่าน | exception ทุกเส้นทาง (email ผิด, password ผิด) ถูกจับรวมที่ `except Exception` (auth.py:113) → ข้อความเดียวกันหมด "อีเมลหรือรหัสผ่านไม่ถูกต้อง" ส่วนข้อความที่ต่างกัน ("ไม่พบบัญชี", "บัญชีถูกปิด" — auth.py:70,74) เกิด**หลัง**ผ่าน Supabase Auth สำเร็จแล้วเท่านั้น คือ attacker ต้องมี password ถูกต้องอยู่แล้วถึงจะเห็นข้อความเหล่านี้ ไม่ใช่ enumeration vector สำหรับคนที่ไม่มี credential — timing-based enumeration (bcrypt compare timing ต่างจาก user-not-found) เป็นเรื่องของ Supabase Auth backend เอง ไม่ใช่โค้ดในไฟล์นี้ **ต้องยืนยันเพิ่ม** ถ้าต้องการฟันธง 100% ต้องวัด response time จริงกับ Supabase Auth API ซึ่งอยู่นอกขอบเขตการอ่านโค้ด |
| Lockout / rate limit | ⚠️ มีแต่คุณภาพต่ำกว่าที่ตั้งใจ | มี `10 per minute` ต่อ IP (auth.py:54) — ไม่มี per-account lockout (ตั้งใจแบบนี้ก็สมเหตุสมผลสำหรับระบบเล็ก) แต่ดู **F-13**: key function ไม่รองรับ reverse proxy |
| Password hash | N/A | ไม่มี logic hash ในไฟล์นี้ — password ทั้งหมดจัดการผ่าน Supabase Auth (GoTrue) ภายนอก โค้ดฝั่งนี้ไม่เคยเห็น hash เลย ปลอดภัยโดยการไม่แตะต้อง |
| `must_change_password` enforcement | ❌ **F-11** | ดูรายละเอียดด้านล่าง |

### F-11 — `must_change_password` เป็นแค่ redirect ครั้งเดียวตอน login ไม่ได้ enforce ที่ session/route level

**ไฟล์:บรรทัด:** auth.py:108–109, auth.py:125 (ตรวจครบด้วย grep `must_change_password` ทั้งโปรเจกต์ — พบแค่ 2 จุดที่ auth.py และ 2 จุดที่ admin.py:102,568 ตอน insert `must_change_password: True`)

**ความร้ายแรง:** 🟠 ปานกลาง-สูง

**เกิดอะไรได้:**
Flow ที่ตั้งใจ (comment "A6: force password change on first login after CSV import") คือ: admin import CSV → สร้าง user ด้วย `temp_password = secrets.token_urlsafe(12)` (admin.py:86, 560 — สุ่มจริง entropy สูง ไม่ใช่ปัญหา) → student login ครั้งแรกด้วย temp password → ถูก redirect ไป `/change-password` (auth.py:108–109) → บังคับเปลี่ยน

ปัญหาคือ **ทุกจุดที่เช็ค `must_change_password` อยู่ใน `login()` (โยน redirect เดียว) กับ `change_password()` GET (ใช้แค่โชว์ banner "forced") เท่านั้น** — `login_required` (auth.py:17–25) ซึ่งเป็น decorator ที่คุมทุก route ที่ต้อง login (student/teacher/admin dashboard ทั้งหมด) เช็คแค่ `"user_id" not in session` ไม่เคยเช็ค `must_change_password` เลย และไม่มี `before_request` hook ระดับ app ที่เช็คเรื่องนี้ (`app/__init__.py:126` มี `reconnect_if_needed` แต่เป็น `pass` placeholder เฉย ๆ)

ผลคือ: student ที่ login ด้วย temp password ครั้งแรก **ไม่จำเป็นต้องกดปุ่มเปลี่ยนรหัสผ่านเลย** — แค่พิมพ์ URL เอง (เช่น `/student/dashboard`) แทนที่จะกด submit บนหน้า change-password ที่ redirect ไปให้ ก็เข้าใช้งานได้ปกติทันที และ `must_change_password` ในตาราง `users` จะค้างเป็น `True` ตลอดไปโดยไม่กระทบการใช้งานอะไรเลย — บัญชีนั้นจะใช้ temp password เดิมได้ **ตลอดไป**

**คนร้ายต้องมีอะไรอยู่ในมือก่อน:** ต้องรู้ temp password ของ user คนใดคนหนึ่งที่ยังไม่ได้เปลี่ยนรหัส — ค่านี้ถูกโชว์ครั้งเดียวในหน้า `admin/import_result.html` ตอน import (ดู admin.py:116, 116 `"temp_password": temp_password`) เท่ากับ: staff ที่ทำ import, ใครก็ตามที่เห็นหน้าจอ/สกรีนช็อตตอนนั้น, หรือ log/history ใด ๆ ที่ยังเก็บค่านี้ไว้ (ไม่ได้ตรวจว่าเก็บที่ไหนอีกบ้าง — นอก scope ไฟล์นี้) จะสามารถเข้าบัญชีนั้นได้ถาวรโดย user เจ้าของบัญชีไม่รู้ตัวว่า "ต้องเปลี่ยนรหัสผ่าน" เป็นแค่คำแนะนำ ไม่ใช่ requirement จริง

**ทางแก้:** ใส่เช็คนี้ใน `login_required` (auth.py:17–25) ให้ block ทุก route ยกเว้น `/change-password` เองและ `/logout` เมื่อ `session.get("must_change_password")` เป็น True — วิธีที่ตรงที่สุดคือ cache ค่านี้ลง session ตอน login (บรรทัด 99–105 เพิ่ม `session["must_change_password"] = user.get("must_change_password")`) แล้วเช็คใน decorator แทนการ query DB ทุก request:

```python
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("กรุณาเข้าสู่ระบบก่อน", "warning")
            return redirect(url_for("auth.login"))
        if session.get("must_change_password") and request.endpoint not in (
            "auth.change_password", "auth.logout"
        ):
            return redirect(url_for("auth.change_password"))
        return f(*args, **kwargs)
    return decorated
```

ต้องอัปเดต `session["must_change_password"]` เป็น `False` ตอนเปลี่ยนรหัสสำเร็จด้วย (change_password() บรรทัด 158–160 อัปเดต DB แล้ว แต่ session เดิมยังมีค่าเก่าค้างอยู่จนกว่าจะ login ใหม่ — บรรทัด 162 มี `session.clear()` อยู่แล้วก่อน redirect ไป login ดังนั้นจริง ๆ ไม่ต้องแก้เพิ่ม เพราะรอบถัดไปที่ login จะดึงค่าใหม่จาก DB)

**เวลาที่ใช้แก้โดยประมาณ:** 20–30 นาที (แก้ + ทดสอบ flow เปลี่ยนรหัสผ่านครั้งแรกให้แน่ใจว่าไม่ redirect loop)

---

## 3. Reset / Forgot Password

**ตรวจแล้ว: ไม่มี route นี้ในระบบ** — grep `forgot|reset.password|reset_password|password_reset|recover` ทั้งโปรเจกต์ (`*.py`) ไม่พบไฟล์ใดเลย ระบบนี้ไม่มี self-service password reset — การรีเซ็ตรหัสผ่านทำได้ทาง admin เท่านั้น (ผ่าน CSV re-import หรือ Supabase Dashboard โดยตรง ซึ่งไม่ใช่โค้ดในไฟล์นี้) ไม่มี token/entropy/expiry ให้ตรวจในหัวข้อนี้ — **N/A**

---

## 4. Logout (auth.py:176–184)

| หัวข้อ | ผล |
|---|---|
| Flask session ฝั่ง server ถูกลบครบไหม | ✅ ผ่าน — `session.clear()` (182) ทำให้ session เป็น falsy (`ServerSideSession.__bool__`) + `modified=True` → `save_session()` ของ flask_session เข้าเงื่อนไข "empty + modified" (`base.py:293–299`) ลบแถวใน `flask_sessions` table จริงและสั่ง `delete_cookie` — ยืนยันจากโค้ดจริงใน `venv/Lib/site-packages/flask_session/base.py:274-299` ไม่ใช่แค่เดา |
| Device token (BLE/check-in) ถูก revoke ไหม | ❌ ไม่ — ตรงกับ **F-8** ที่มีอยู่แล้วใน `99-summary.md` (security_service.py:24, ไม่มี revocation store) ไม่รายงานซ้ำเป็น finding ใหม่ แต่ยืนยันว่า auth.py:176–184 ไม่มี logic ใด ๆ ที่แตะ device token เลย — สอดคล้องกับที่ F-8 อธิบายไว้ |
| `supabase.auth.sign_out()` ทำงานถูกกับ "ตัวเอง" จริงไหม | ❌ **F-12** — ดูด้านล่าง |

### F-12 — `supabase.auth.sign_out()` ใช้ shared global client → logout อาจไป revoke session ของคนละ user

**ไฟล์:บรรทัด:** auth.py:179 (`supabase.auth.sign_out()`), เกี่ยวโยงกับ auth.py:64 (`supabase.auth.sign_in_with_password`) และ auth.py:146–149 (current-password verify ก็เรียก `sign_in_with_password` บน client ตัวเดียวกัน)

**ความร้ายแรง:** 🟡 ปานกลาง (ผลกระทบจริงในระบบนี้ต่ำ แต่ตัว logic ผิดจริงและเป็นความเสี่ยงเชิงสถาปัตยกรรม)

**หลักฐาน:** `supabase` เป็น module-level global object สร้างครั้งเดียวใน `_refresh_clients()` (`app/__init__.py:59–63`) แชร์กันทุก request ไม่ใช่ per-request/per-user อ่านซอร์ส GoTrue client จริงที่ติดตั้งในโปรเจกต์ (`venv/Lib/site-packages/gotrue/_sync/gotrue_client.py`) ยืนยันว่า:
- `sign_in_with_password()` (บรรทัด 256) เรียก `self._remove_session()` ก่อน แล้ว **เขียนทับ session ที่เก็บใน `self._storage` (in-memory, ผูกกับ client instance)** ด้วย session ของ user ที่เพิ่ง login (ผ่าน `_save_session()` บรรทัด 1031)
- `sign_out()` (บรรทัด 734) **ไม่รับ user/token เป็น argument** — มันเรียก `self.get_session()` ซึ่งอ่านจาก `self._storage` เดียวกันนี้ (บรรทัด 745) แล้วเอา token ที่ได้ไป revoke (`self.admin.sign_out(access_token, scope)`)

เพราะ deployment เป็น `gunicorn --workers 1` (sync worker, ประมวลผลทีละ request ตาม `CLAUDE.md`) request ทำงานเรียงกัน ไม่ race กัน แต่ **state ใน shared client ไม่ถูกล้างระหว่าง request** — ดังนั้นถ้า:
1. Alice login (auth.py:64) → shared client เก็บ session ของ Alice
2. Bob login (auth.py:64) → shared client **เขียนทับ** เป็น session ของ Bob
3. Alice กด logout (auth.py:179) → `sign_out()` อ่าน `get_session()` ได้ session ของ **Bob** (คนล่าสุดที่ auth ผ่าน client ตัวนี้ ไม่ว่าจะผ่าน login หรือผ่าน current-password verify ตอน change-password ก็ตาม) → ไป revoke refresh token ของ Bob ไม่ใช่ของ Alice

**ผลกระทบจริงในระบบนี้ (ทำไมไม่ใช่ 🔴):** grep `access_token` ทั้ง `app/` พบว่า `session["access_token"]` (auth.py:102) **ไม่เคยถูกอ่านที่ไหนอีกเลยในทั้งโปรเจกต์** — ระบบนี้ใช้ Flask server-side session (`session["user_id"]`) เป็นกลไก auth หลักทั้งหมด ไม่ได้พึ่ง Supabase access/refresh token ต่อเนื่อง ดังนั้นการ revoke ผิดคนจึงไม่ทำให้ใครเข้า/ออกจากระบบผิดคน หรือเสีย session ของตัวเองไป — Bob จะไม่รู้สึกอะไรเลย (Flask session ของเขายังอยู่จนครบ 1 ชั่วโมงตามปกติ) และ Alice เองก็ log out จาก Flask ได้สำเร็จตามปกติ (บรรทัด 182 ทำงานเสมอไม่ว่า try-block ด้านบนจะสำเร็จหรือ throw)

**ความเสี่ยงที่แท้จริง:** เป็นสถาปัตยกรรมที่ผิดหลัก (per-request state ไปแชร์บน global object) — ตอนนี้ไม่มีอันตรายเพราะ `access_token` เป็น dead value แต่ถ้าในอนาคตมีใครเอา `session["access_token"]` ไปใช้จริง (เช่น เรียก Supabase ด้วย user-scoped RLS token แทน `supabase_admin`) code path นี้จะกลายเป็นบั๊กที่ทำให้ผู้ใช้ถูกเตะออกจากระบบสลับกันแบบสุ่มทันที — เป็น latent bug ที่ควรแก้ตอนนี้ก่อนที่จะมีอะไรมาพึ่งพา field นี้

**ทางแก้:** เรียก `sign_out` โดยส่ง token ของ user ที่ต้องการ logout จริง ๆ อย่างชัดเจน แทนการพึ่ง state ที่เดาไม่ได้ของ shared client:

```python
@auth_bp.route("/logout")
def logout():
    token = session.get("access_token")
    if token:
        try:
            supabase_admin.auth.admin.sign_out(token, "global")
        except Exception:
            pass
    session.clear()
    flash("ออกจากระบบแล้ว", "info")
    return redirect(url_for("auth.login"))
```

(ใช้ `supabase_admin.auth.admin.sign_out(jwt, scope)` ซึ่งรับ token ตรง ๆ เป็น argument ไม่ผ่าน shared in-memory session — ยืนยัน signature จาก `venv/Lib/site-packages/gotrue/_sync/gotrue_admin_api.py`) เช่นเดียวกัน ควรพิจารณาว่า current-password verify ใน change_password() (146–149) มีปัญหาเดียวกัน — ทุกครั้งที่มีคน verify password จะเขียนทับ shared client state เหมือนกัน แม้ตอนนี้ยังไม่กระทบอะไรเพราะไม่มีใครอ่าน state นั้นต่อ

**เวลาที่ใช้แก้โดยประมาณ:** 15–20 นาที

---

## 5. Registration

**ตรวจแล้วผ่าน** — `/register` (auth.py:169–173) เป็น GET-only, ไม่มี POST handler เลย แค่ flash ข้อความ "การสมัครสมาชิกถูกปิด" แล้ว redirect ไป login ไม่มีทางส่ง role หรือข้อมูลอื่นจาก client ได้เลย — ตรงข้ามกับ pattern ของ CC-1a (`flow_mode` ที่ client คุม threshold ได้) ในไฟล์นี้ไม่มี pattern แบบนั้นเกิดขึ้น การสร้าง user ทั้งหมดอยู่ที่ admin.py (CSV import) ซึ่งอยู่นอกขอบเขตไฟล์นี้

---

## 6. Input ที่เชื่อจาก Client โดยไม่ Validate ฝั่ง Server

| ค่า | ไฟล์:บรรทัด | validate ยังไง | ความเสี่ยง |
|---|---|---|---|
| `email`, `password` (login) | auth.py:59–60 | ไม่ validate format เลย ส่งตรงเข้า Supabase Auth | 🟢 ต่ำ — Supabase Auth SDK ใช้ HTTPS API ไม่ใช่ raw SQL, ไม่มี injection surface; รูปแบบ email ผิดก็แค่ auth ล้มเหลว |
| `current_password`, `new_password`, `confirm_password` | auth.py:128–130 | เช็คแค่ `len(new_pw) >= 8` (132) และ match กับ confirm (136) — ไม่มี upper bound, ไม่มี complexity rule | 🟢 ต่ำ — ไม่มี max length เปิดช่องให้ส่ง payload ยาวมาก ๆ ได้ (ส่งผลเป็น DoS เล็กน้อยต่อ Supabase Auth call) แต่ปัญหานี้เป็นระดับ app-wide (ไม่มี `MAX_CONTENT_LENGTH` ใน `config.py` เลย) ไม่ใช่จุดอ่อนเฉพาะของ auth.py — ไม่นับเป็น finding แยกในไฟล์นี้ |

ไม่พบ pattern แบบ `flow_mode`/`baseline_ear` (CC-2) ที่ client ส่งค่ามาคุม business logic/threshold ในไฟล์นี้ — **ตรวจแล้วผ่าน**

---

## 7. Logging

**ตรวจแล้วผ่าน (ไม่มีปัญหาการหลุด PII/credential)** — auth.py ไม่มี `import logging` และไม่มีการเรียก log function ใด ๆ เลยทั้งไฟล์ (grep ยืนยัน) จึงไม่มีความเสี่ยงที่ password/token จะหลุดเข้า log จากไฟล์นี้โดยตรง

**แต่เป็นช่องว่างที่มีอยู่แล้ว ไม่ใช่ finding ใหม่:** ตรงกับที่ `05-security-service.md` บันทึกไว้แล้วในหัวข้อ "Actions สำคัญที่ไม่อยู่ใน audit_logs" — login success/failure และ password change ไม่มี `log_audit_event()` เรียกเลยสักจุด (ยืนยันซ้ำจากการอ่าน auth.py รอบนี้ — ไม่มีจุดใดใน login()/change_password() เรียก `log_audit_event`) คำแนะนำเดิมใน 05-security-service.md ("Top 3" ข้อ 3) ยังใช้ได้ ไม่ต้องรายงานซ้ำเป็น F ใหม่

---

## 8. Rate Limit Key ไม่รองรับ Reverse Proxy

### F-13 — `get_remote_address` ใช้ raw `remote_addr` ไม่รองรับ X-Forwarded-For — rate limit ต่อ IP เสี่ยงยุบเหลือ bucket เดียวหลัง proxy

**ไฟล์:บรรทัด:** auth.py:6, auth.py:54 (`key_func=get_remote_address`) — root cause เดียวกันกับ `app/__init__.py:21-34` (`get_rate_limit_key()` fallback ก็ใช้ `get_remote_address()` ตัวเดียวกัน ไม่ใช่แค่ auth.py แต่ auth.py:54 เป็นจุดที่ตั้งใจใช้เพื่อกัน brute-force login โดยเฉพาะ จึงกระทบตรงที่สุดที่นี่)

**ความร้ายแรง:** 🟡 ปานกลาง (reliability เป็นหลัก, security เป็นผลข้างเคียง)

**หลักฐาน:** `get_remote_address` (จาก `flask_limiter.util`) คืนค่า `request.remote_addr` ของ Werkzeug ตรง ๆ ไม่อ่าน `X-Forwarded-For` เลย และในโปรเจกต์นี้ไม่มีการติดตั้ง `ProxyFix` middleware ที่ไหนเลย (grep `ProxyFix` ทั้ง `app/` ไม่พบ) ขณะที่ `student.py` มี `_safe_ip()` เป็น custom X-Forwarded-For parser ของตัวเองสำหรับ audit log แต่ฟังก์ชันนี้**ไม่ได้ถูกใช้กับ rate limiter เลย** — เป็นคนละ code path

`CLAUDE.md` ระบุว่า deploy บน Railway (`gunicorn ... --bind 0.0.0.0:$PORT`) ซึ่งเป็น PaaS ที่ proxy request ผ่าน edge layer ของตัวเองเป็นมาตรฐาน (ไม่ใช่ raw TCP passthrough) ถ้าเป็นจริงตามรูปแบบทั่วไปของ Railway, `request.remote_addr` ที่ Flask เห็นจะเป็น IP ของ Railway's internal proxy สำหรับ**ทุก request** ไม่ใช่ IP ของ client จริง — **ต้องยืนยันเพิ่ม**: ต้องเช็ค production log จริงหรือ deploy config ของ Railway ว่า remote_addr ที่ Flask เห็นเป็น IP เดียวกันทุก request หรือไม่ (ยืนยันไม่ได้จากการอ่านโค้ดอย่างเดียว) — แต่ตัว **code-level defect ยืนยันได้แน่นอน**: ไม่ว่า Railway จะ behave แบบไหน การไม่มี ProxyFix/XFF handling ใน rate limiter คือความเสี่ยงที่มีอยู่จริงสำหรับ deployment ใด ๆ ที่อยู่หลัง reverse proxy

**เกิดอะไรได้ (ถ้า remote_addr ยุบเหลือ IP เดียวจริง):**
1. **Rate limit กลายเป็น global แทนที่จะเป็น per-attacker** — `10 per minute` ที่ตั้งใจจำกัด "1 คน 10 ครั้ง/นาที" จะกลายเป็น "รวมทุกคนบนเว็บ 10 ครั้ง/นาที" ทำให้การจำกัดต่อผู้โจมตีจริง ๆ ไม่เกิดขึ้น (attacker ไม่ได้ประโยชน์จากการ rotate IP เพราะ bucket เดียวกันหมดอยู่แล้ว แต่ก็ไม่มี "per-IP" ที่จะแยกเขาออกจาก traffic ปกติได้เช่นกัน — ป้องกันไม่เต็มร้อยแต่ไม่ได้พังทั้งหมด)
2. **False-positive lockout ต่อผู้ใช้จริง** — ถ้ามีนักศึกษา 11+ คนพยายาม login พร้อมกันภายใน 1 นาที (เช่น ตอนต้นคาบเรียน) คนที่ 11 เป็นต้นไปจะโดน 429 ทั้งที่เป็นผู้ใช้ปกติ — นี่คือผลกระทบที่มีโอกาสเกิดจริงมากกว่าในระบบเช็คชื่อของมหาวิทยาลัย

**คนร้ายต้องมีอะไรอยู่ในมือก่อน:** ไม่ต้องมีอะไรพิเศษ — เป็นผลจาก deployment topology ไม่ใช่ต้อง exploit อะไร แต่ก็ไม่ได้ทำให้ attacker ได้เปรียบขึ้นชัดเจน (ดูข้อ 1) ผลกระทบหลักคือฝั่ง availability/UX ของผู้ใช้จริงมากกว่าการเปิดช่องให้ brute-force ง่ายขึ้น

**ทางแก้:** ใส่ `ProxyFix` ใน `app/__init__.py` (เชื่อ 1 hop จาก Railway's edge — ต้องยืนยันจำนวน hop จริงกับ Railway ก่อน ใส่ผิดจำนวนจะยิ่งแย่กว่าไม่ใส่เพราะ header spoof ได้):

```python
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
```

หลังใส่แล้ว `request.remote_addr` จะสะท้อน client IP จริงจาก `X-Forwarded-For` โดยอัตโนมัติ ทั้ง `get_remote_address()` ในทุกจุด (auth.py:54 และ `app/__init__.py`'s `get_rate_limit_key`) จะถูกต้องไปพร้อมกันโดยไม่ต้องแก้ทีละจุด

**เวลาที่ใช้แก้โดยประมาณ:** 15 นาที (+ ต้องยืนยัน hop count กับ Railway ก่อน deploy จริง)

---

## 9. Rate Limit บน Current-Password Oracle

### F-14 — `/change-password` POST ไม่มี rate limit เฉพาะสำหรับการยืนยัน current password

**ไฟล์:บรรทัด:** auth.py:118–121 (route + decorators), auth.py:146–149 (จุดที่ verify current password)

**ความร้ายแรง:** 🟢 ต่ำ

**เกิดอะไรได้:** `change_password()` เรียก `supabase.auth.sign_in_with_password()` เพื่อยืนยัน current password (140–152) — เป็น password-guessing oracle ตัวหนึ่ง แต่ไม่มี rate limit เฉพาะ อาศัยแค่ `default_limits: 200 per hour` ระดับ app (ประมาณ 1 ครั้ง/18 วินาที ถัวเฉลี่ยตลอดชั่วโมง) เทียบกับ `/login` ที่มี `10 per minute` เฉพาะ — ไม่สมดุลกัน ทั้งที่ทั้งคู่คือ password oracle เหมือนกัน

**คนร้ายต้องมีอะไรอยู่ในมือก่อน:** ต้องมี valid session ของเหยื่ออยู่แล้ว (`@login_required`) เช่น ขโมย session cookie มา หรือใช้เครื่องที่เหยื่อ login ค้างไว้ — ถ้ามีระดับนี้อยู่แล้ว attacker เข้าถึง app ในฐานะเหยื่อได้โดยตรงอยู่แล้ว **แต่** ยังได้ประโยชน์เพิ่มจากช่องนี้คือ: brute-force หา password ตัวจริงของเหยื่อ (ไม่ใช่แค่ session) เพื่อเอาไปลองกับระบบอื่นที่เหยื่อ reuse password เดียวกัน (credential-reuse attack) — เป็นเหตุผลที่ยังควรแก้แม้ prerequisite จะสูง

**ทางแก้:**
```python
@auth_bp.route("/change-password", methods=["GET", "POST"])
@login_required
@_limiter.limit("5 per minute", methods=["POST"])
@csrf_protect_form
def change_password():
```

**เวลาที่ใช้แก้โดยประมาณ:** 5 นาที

---

## 10. Quality Finding

### Q-14 — Session-regeneration cleanup เป็น dead code เพราะเช็คผิด backend

**ไฟล์:บรรทัด:** auth.py:90–96

```python
# ลบ session file เก่าออกจาก filesystem (best-effort)
if _old_sid and hasattr(current_app.session_interface, "cache"):
    try:
        _prefix = getattr(current_app.session_interface, "key_prefix", "session:")
        current_app.session_interface.cache.delete(_prefix + _old_sid)
    except Exception:
        pass
```

**ประเภท:** quality (ไม่ใช่ security vulnerability — ดูเหตุผลด้านล่างว่าทำไม)

**เกิดอะไรได้:** comment บอกว่า "flask-session filesystem backend" แต่ `config.py:55` ตั้ง `SESSION_TYPE = "sqlalchemy"` จริง — คนละ backend กับที่ comment สมมติไว้ อ่านซอร์ส `SqlAlchemySessionInterface` จริงในโปรเจกต์ (`venv/Lib/site-packages/flask_session/sqlalchemy/sqlalchemy.py`) ยืนยันว่า class นี้มี attribute `.client` (SQLAlchemy instance) ไม่มี `.cache` เลย → `hasattr(current_app.session_interface, "cache")` เป็น `False` เสมอ → **โค้ดลบ session แถวเก่าใน `flask_sessions` table ไม่เคยทำงานจริงสักครั้งเดียว**

**ทำไมไม่ใช่ security vulnerability:** ตรวจ `save_session()` ใน `flask_session/base.py:274-305` แล้วยืนยันว่ากลไก anti-fixation หลัก (เปลี่ยน `session.sid` เป็นค่าสุ่มใหม่ก่อน set ข้อมูล user) ทำงานถูกต้องอยู่แล้วโดยไม่ต้องพึ่งการลบแถวเก่า — เพราะ `_upsert_session` ใช้ `session.sid` ที่ถูก reassign แล้ว ณ ตอน save เสมอ แถวเก่า (sid ที่ attacker อาจ fixate ไว้) จะไม่มีทางได้รับข้อมูลของ user ที่ login สำเร็จ ไม่ว่า cleanup จะทำงานหรือไม่ — ผลที่เกิดจริงมีแค่: **แถว session เก่า (ไม่มีข้อมูล auth ใด ๆ) ค้างอยู่ใน `flask_sessions` table ตลอดไป** ทุกครั้งที่มี login สำเร็จ 1 ครั้ง เพราะ `SqlAlchemySessionInterface.ttl = False` (บรรทัด 74 ของไฟล์เดียวกัน) และไม่มี scheduled cleanup job ไหนใน `app/scheduler.py` ที่เรียก `session_cleanup` เลย ตาราง `flask_sessions` จึงโตขึ้นเรื่อย ๆ แบบไม่มีวันจบ (ไม่ใช่แค่จาก login — ทุก anonymous visit ที่ไม่ login ก็ทิ้งแถวไว้เหมือนกัน แต่ login แต่ละครั้งเพิ่มอีก 1 แถวที่รู้แน่ ๆ ว่าจะไม่มีวันถูกเปิดซ้ำอีก)

**ทางแก้:** flask_session 0.8.0 (เวอร์ชันที่ติดตั้งจริงในโปรเจกต์) มี built-in method `session_interface.regenerate(session)` ทำสิ่งเดียวกันแต่ถูกต้องครบ (ลบแถวเก่า + สุ่ม sid ใหม่ + mark modified ในบรรทัดเดียว — ดู `flask_session/base.py:261-270`) แทนที่บรรทัด 83–96 ทั้งหมดด้วย:

```python
_old_sid = getattr(session, "sid", None)
session.clear()
if _old_sid and hasattr(current_app.session_interface, "regenerate"):
    current_app.session_interface.regenerate(session)
```

**เวลาที่ใช้แก้โดยประมาณ:** 10 นาที

---

## สรุปตาราง Findings

| ID | เรื่อง | ไฟล์:line | ประเภท | ความร้ายแรง | ต้องมีอะไรก่อนถึงใช้ได้ | เวลาแก้ |
|---|---|---|---|---|---|---|
| **F-11** | `must_change_password` ไม่ enforce จริง — bypass ได้ด้วยการพิมพ์ URL เอง | auth.py:108-109, 125 | security 🟠 | ปานกลาง-สูง | ต้องรู้ temp_password ของ user ที่ยังไม่เปลี่ยนรหัส (โชว์ครั้งเดียวตอน admin import) | 20-30 นาที |
| **F-12** | `supabase.auth.sign_out()` ใช้ shared global client — logout อาจ revoke session คนละ user | auth.py:179 (+64, 146-149) | security 🟡 | ปานกลาง (impact ปัจจุบันต่ำ เพราะ access_token ไม่ถูกใช้ที่ไหนอีก — เป็น latent bug) | ไม่ต้องมีอะไรพิเศษ — เกิดเองจาก concurrent login/logout ตามปกติ | 15-20 นาที |
| **F-13** | Rate limiter ใช้ raw `remote_addr` ไม่รองรับ X-Forwarded-For หลัง Railway proxy | auth.py:6, 54 | security/reliability 🟡 | ปานกลาง (เสี่ยง false-lockout ผู้ใช้จริงมากกว่าเปิดช่องให้ attacker) | ไม่ต้องมีอะไรพิเศษ — ขึ้นกับ deployment topology (ต้องยืนยันเพิ่มกับ Railway) | 15 นาที |
| **F-14** | `/change-password` current-password check ไม่มี rate limit เฉพาะ | auth.py:118-121 | security 🟢 | ต่ำ | ต้องมี valid session ของเหยื่ออยู่แล้ว | 5 นาที |
| **Q-14** | Session-regen cleanup เป็น dead code (เช็ค `.cache` ผิด backend) → DB row ค้างไม่มีวันหมด | auth.py:90-96 | quality 🟢 | ต่ำ (ไม่ใช่ vulnerability — fixation defense ยังทำงานถูกอยู่แล้ว) | — | 10 นาที |

## หัวข้อที่ตรวจแล้วผ่าน (ไม่มี finding)

| หัวข้อ | ผล |
|---|---|
| Session fixation | ✅ regenerate sid ก่อน set ข้อมูล user ถูกต้อง (ยืนยันจากซอร์ส flask_session จริง) |
| User enumeration / timing บน login error | ✅ error message เดียวกันหมดสำหรับคนที่ไม่มี valid credential |
| Password hashing | N/A — delegate ให้ Supabase Auth ทั้งหมด |
| Reset/forgot password | N/A — ไม่มี route นี้ในระบบ |
| Registration / role จาก client | ✅ `/register` ปิดสนิท ไม่มี POST handler |
| Client input validation (นอกเหนือจากที่ระบุ) | ✅ ไม่มี pattern แบบ `flow_mode`/CC-2 ในไฟล์นี้ |
| Password/token/PII หลุดใน log | ✅ ไฟล์นี้ไม่ log อะไรเลย (แต่ไม่มี audit log login success/failure — เป็นช่องว่างเดิมจาก 05-security-service.md ไม่ใช่ finding ใหม่) |
| Logout ลบ Flask session ฝั่ง server ครบไหม | ✅ ยืนยันจากซอร์ส flask_session จริงว่าแถว + cookie ถูกลบ |

---

## ลำดับที่ควรแก้ก่อนหลัง (เรียงตาม ผลลัพธ์ ÷ เวลา)

| ลำดับ | ID | เหตุผลที่ทำก่อน |
|---|---|---|
| 1 | **F-11** | Impact สูงสุดในไฟล์นี้ — บัญชีที่ import มาอาจไม่เคยถูกบังคับเปลี่ยนรหัสผ่านเลยตลอดอายุการใช้งาน แก้ไม่ยาก (เพิ่ม logic ใน decorator เดียว) |
| 2 | **F-12** | ยังไม่มีอันตรายตอนนี้ (access_token dead) แต่เป็น bug ที่ถูกต้องแล้วอาจกลายเป็นปัญหาใหญ่ทันทีถ้ามีโค้ดในอนาคตมาพึ่ง `session["access_token"]` — แก้ตอนนี้ถูกกว่าแก้ตอนที่มันกลายเป็น production incident |
| 3 | **F-13** | ต้องยืนยัน Railway topology ก่อน (hop count) ถึงจะใส่ ProxyFix ได้ถูกต้อง — priority ปานกลางเพราะผลกระทบหลักคือ reliability ของผู้ใช้จริงมากกว่าช่องโหว่ที่ attacker ใช้ประโยชน์ได้ตรง ๆ |
| 4 | **Q-14** | ไม่มี security risk แต่แก้ง่ายมาก (built-in method มีอยู่แล้ว) ทำพร้อม F-12 ได้เลยเพราะแก้ในโค้ดบล็อกใกล้กัน |
| 5 | **F-14** | ต่ำสุด — ต้องมี session ของเหยื่ออยู่แล้วถึงจะใช้ประโยชน์ได้ ทำเมื่อมีเวลาว่าง |

---

## หมายเหตุขอบเขต

ไฟล์นี้ import และเรียกใช้ `app.services.security_service.csrf_protect_form` เท่านั้น (ไม่มีการเรียก `csrf_protect`, `create_device_token`, `verify_device_token`, `compute_embedding_integrity_hash` เลยในไฟล์นี้) — ฟังก์ชันเหล่านั้นถูก review ไปแล้วใน `05-security-service.md` และ**ไม่ได้ตรวจซ้ำที่นี่** ยกเว้นในส่วนที่จำเป็นเพื่อยืนยัน finding ของไฟล์นี้เอง (เช่น อ่าน `sign_out`/`sign_in_with_password` ใน GoTrue SDK เพื่อยืนยัน F-12, อ่าน `flask_session` internals เพื่อยืนยัน session fixation และ Q-14)

รวม finding ใหม่จากไฟล์นี้: **4 security (F-11 ถึง F-14) + 1 quality (Q-14)** — ไม่มี finding ที่ยกระดับเป็น 🔴 วิกฤต (ต่างจาก F-7 รอบก่อน) แต่ F-11 ถือว่าเป็นข้อที่ควรแก้เร่งด่วนที่สุดในกลุ่มนี้เพราะกระทบ "ความหมาย" ของ control ที่มีอยู่แล้วโดยตรง (คล้าย F-7 ตรงที่ decorator/logic ดูเหมือนป้องกันอยู่ แต่จริง ๆ ข้ามได้ง่าย)

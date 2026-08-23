# Pre-flight Checks — SmartCheck

ตรวจสอบ ณ วันที่ 2026-08-20  
ห้ามแก้โค้ดก่อน review เสร็จ — ไฟล์นี้บันทึกผลตรวจสอบเบื้องต้นทั้ง 4 หัวข้อ

---

## A. .env Security

### คำสั่งที่รัน
```powershell
git check-ignore -v .env
git log --oneline -- .env
git remote -v
```

### Output ดิบ
```
=== git check-ignore ===
.gitignore:1:.env	.env

=== git log -- .env ===
(ไม่มี output — ไม่เคยมี commit ที่มี .env)

=== git remote -v ===
origin	https://github.com/Ter2004/smartcheck.git (fetch)
origin	https://github.com/Ter2004/smartcheck.git (push)
```

### สรุป
`.env` ถูก ignore โดย `.gitignore` บรรทัด 1 และไม่เคยถูก commit ลง git ตลอดประวัติทั้งหมด — **ผ่าน**  
Remote คือ GitHub repo `Ter2004/smartcheck` — ยืนยันว่า working directory นี้คือ repo จริง ไม่ใช่ snapshot

---

## B. JS Conflict: enrollment_flow.js vs enrollment_circular.js

### คำสั่งที่รัน
```powershell
# 1. หา IDs ใน enrollment_flow.js
Select-String -Path '...\enrollment_flow.js' -Pattern 'getElementById\(["\x27]([^"x27]+)["\x27]\)' -AllMatches | ...

# 2. หา IDs ใน enrollment_circular.js
Select-String -Path '...\enrollment_circular.js' -Pattern 'getElementById\(["\x27]([^"x27]+)["\x27]\)' -AllMatches | ...

# 3. ตรวจ IDs ที่ซ้ำกัน (intersection)
Compare-Object $flowIds $circIds -IncludeEqual | Where-Object { $_.SideIndicator -eq '==' }

# 4. ตรวจ top-level declarations ซ้ำ
$flowDecl = ...; $circDecl = ...; Compare-Object $flowDecl $circDecl -IncludeEqual | Where { $_.SideIndicator -eq '==' }

# 5. ตรวจ window.startLivenessChallenge override
Select-String -Path '...\enrollment_circular.js' -Pattern 'startLivenessChallenge'
```

### Output ดิบ
```
=== IDs ที่ซ้ำกันทั้ง 2 ไฟล์ ===
(ไม่มี output — ว่างเปล่า)

=== Top-level declarations ซ้ำ ===
(ไม่มี output — ว่างเปล่า)

=== window.startLivenessChallenge ใน enrollment_circular.js ===
enrollment_circular.js:639:  window.startLivenessChallenge = function () { startCircularCapture(); }

=== Globals ที่ enrollment_circular.js เรียกจาก enrollment_flow.js ===
_getSharedFM      : บรรทัด 339
_computeEAR       : บรรทัด 175, 247
_checkBlur        : บรรทัด 195
_checkCameraConditions : บรรทัด 196
_captureFrameFromVideo : บรรทัด 199
_csrfToken        : บรรทัด 383, 447
_stdDev           : บรรทัด 437
fullRestart        : บรรทัด 485
```

### สรุป
ไม่มี DOM element ID หรือ top-level variable declaration ที่ซ้ำกัน — **ไม่มี JS conflict**  
`window.startLivenessChallenge` ที่บรรทัด 639 เป็น override ตั้งใจ (มี comment `// ─── Override startLivenessChallenge`) เพื่อให้ circular flow เข้าแทนที่ classic liveness challenge — ถูกต้องตามการออกแบบ  
enrollment_circular.js โหลดทีหลัง enrollment_flow.js เสมอ (Jinja2 conditional ใน enroll_face.html บรรทัด 453–455) — ลำดับโหลดถูกต้อง

---

## C. Dead Code ใน enrollment_flow.js

### คำสั่งที่รัน
```powershell
# LOC
(Get-Content '...\enrollment_flow.js').Count

# Top-level functions
Select-String -Path '...\enrollment_flow.js' -Pattern '^(async\s+)?function\s+\w+|^const\s+\w+\s*=\s*(async\s+)?\('

# ตรวจแต่ละ function ว่ามีผู้เรียกหรือไม่
Select-String -Path '...\*.js','...\*.html' -Pattern 'startCamera'
Select-String -Path '...\*.js','...\*.html' -Pattern 'calcEAR'
Select-String -Path '...\*.js','...\*.html' -Pattern '\bdist\b'
Select-String -Path '...\*.js','...\*.html' -Pattern '_deviceFingerprint'
Select-String -Path '...\*.js','...\*.html' -Pattern '_PROGRESS_MSGS_VERIFY'
# (เปรียบเทียบกับ functions ที่มีผู้เรียก)
Select-String -Path '...\*.js','...\*.html' -Pattern 'randomChallengeActions'
Select-String -Path '...\*.js','...\*.html' -Pattern 'InteractiveChallengeDetector'
Select-String -Path '...\*.js','...\*.html' -Pattern 'detectVirtualCamera'
```

### Output ดิบ
```
=== LOC ===
1519

=== Dead functions (ไม่มี caller) ===
startCamera           : นิยาม enrollment_flow.js:333  — ไม่มี caller ใน *.js หรือ *.html
calcEAR               : นิยาม enrollment_flow.js:365  — ไม่มี caller (_calcEAR ใน checkin_flow.js คือ function คนละตัว)
dist                  : นิยาม enrollment_flow.js:361  — ถูกเรียกเฉพาะโดย calcEAR (line 367) ซึ่งเป็น dead code เอง
_deviceFingerprint    : นิยาม enrollment_flow.js:203  — ไม่มี caller ใน *.js หรือ *.html
_PROGRESS_MSGS_VERIFY : นิยาม enrollment_flow.js:1378 — ไม่ถูก pass ให้ _startProgress เลย

=== Functions ที่มี caller (ตัวอย่าง — ไม่ใช่ dead code) ===
randomChallengeActions    : มี caller (enrollment_flow.js)
InteractiveChallengeDetector : มี caller (enrollment_flow.js)
detectVirtualCamera        : มี caller (enrollment_flow.js)
```

### สรุป
enrollment_flow.js มี **5 dead items**: `startCamera`, `calcEAR`, `dist`, `_deviceFingerprint`, `_PROGRESS_MSGS_VERIFY`  
ไม่มีผลต่อ runtime (ไม่ถูกเรียก) แต่เพิ่ม cognitive load และขนาดไฟล์ 1,519 LOC — **สามารถลบได้ใน cleanup pass** (ไม่ urgent, ไม่กระทบ review ความปลอดภัย)

---

## D. Dependencies

### คำสั่งที่รัน
```powershell
# ตรวจ onnxruntime import ในโค้ด
Select-String -Path '...\face_service.py' -Pattern 'onnxruntime'

# pip show
pip show onnxruntime

# เปรียบเทียบ imports ใน .py กับ requirements.txt
# (script PowerShell — ดึง imports จาก .py ทุกไฟล์ เทียบกับ requirements.txt)

# tests/ folder
Test-Path '...\tests'
```

### Output ดิบ
```
=== onnxruntime import ===
face_service.py:56:    import onnxruntime as ort

=== pip show onnxruntime ===
Name: onnxruntime
Version: 1.24.4
...
Required-by: (ว่าง)

=== imports ไม่อยู่ใน requirements.txt ===
cv2          → opencv-python (ชื่อต่าง)
dateutil     → python-dateutil (ชื่อต่าง)
dotenv       → python-dotenv (ชื่อต่าง)
flask_limiter  → flask-limiter (ชื่อต่าง)
flask_session  → flask-session (ชื่อต่าง)
flask_sqlalchemy → flask-sqlalchemy (ชื่อต่าง)
onnxruntime  → *** ไม่อยู่ใน requirements.txt เลย ***

=== tests/ folder ===
False (ไม่มีโฟลเดอร์ tests/)
```

### สรุป
`onnxruntime` เป็น package เดียวที่ import จริงแต่**ไม่มีในรายการ requirements.txt** — อย่างไรก็ตาม import อยู่ใน `try/except ImportError` (face_service.py:56) ดังนั้น ONNX layer จะ disabled อัตโนมัติถ้าไม่ได้ติดตั้ง (fail-open สำหรับ ONNX layer เฉพาะ, ยังผ่าน 4 layer ที่เหลือ)  
ไม่มี test suite (`tests/` ไม่มี) — ต้องทดสอบผ่าน browser ตาม CLAUDE.md

---

## สรุปรวม

| เรื่อง | สถานะ | Action |
|---|---|---|
| A. .env ไม่เคย commit | ✅ ผ่าน | — |
| A. Remote = GitHub repo จริง | ✅ ผ่าน | — |
| B. JS conflict (DOM IDs / declarations) | ✅ ผ่าน | — |
| B. `startLivenessChallenge` override | ✅ ผ่าน (ตั้งใจ) | — |
| B. `ENROLL_FLOW_MODE` บน Railway | ⚠️ ต้องเช็คนอกเครื่อง | เช็คใน Railway dashboard |
| C. Dead code 5 items ใน enrollment_flow.js | ℹ️ ไม่ urgent | ลบใน cleanup pass หลัง review เสร็จ |
| D. `onnxruntime` ไม่อยู่ใน requirements.txt | ⚠️ ต้องแก้ก่อน deploy | เพิ่ม `onnxruntime` ใน requirements.txt |
| D. ไม่มี test suite | ℹ️ รับทราบ | ทดสอบผ่าน browser per CLAUDE.md |

TODO ก่อน deploy: เพิ่ม onnxruntime ใน requirements.txt

TODO: ถอด PRE-HARDREJECT log ออกหลังสรุป F-5 เสร็จ (face_service.py:~298)

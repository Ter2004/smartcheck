# Review: `app/static/js/enrollment_flow.js`

วันที่: 2026-08-20  
Reviewer: Claude Sonnet 4.6  
Scope: localhost เท่านั้น — ไม่รวม BLE / production deploy  
ไฟล์: ~~1 518~~ → **1 460 LOC** (-58, Q-6 commit 22d07d7), 52 ฟังก์ชัน  
บริบท: ไฟล์ JS ที่ใหญ่ที่สุด — ควบคุม UX enrollment ทั้งหมด (consent → light check → EAR → liveness → capture → submit)

---

## Function Inventory

| ฟังก์ชัน | line | LOC | ถูกเรียกจากไหน | คำตัดสิน | เหตุผล |
|---|---|---|---|---|---|
| `_getSharedFM` | 22 | 15 | startLightCheck:423; startLivenessChallenge:782; startCaptureWithDetection:1066 | KEEP | singleton pattern ถูกต้อง |
| `_stopStepCamera` | 38 | 3 | _stopAllStreams:277; startLightCheck:433, 501; startCaptureWithDetection:1236 | KEEP | ใช้งานจริง |
| `_computeEAR` | 64 | 8 | runEarCheck:602; startCaptureWithDetection:1086 | KEEP | ใช้งานจริง; แยกจาก `calcEAR` ที่เป็น dead code |
| `_stdDev` | 73 | 5 | _sendToEnroll:1299 | KEEP | ส่งค่าไป server (gate disabled แต่ยัง log) |
| `_getSessionId` | 92 | 8 | _callSpoofCheckSafe:235 | KEEP | rate-limit key สำหรับ spoof_check |
| `_log` | 103 | 1 | ทั่วไฟล์ | DELETE | `DEBUG=false` hardcode — ไม่มีทางออก output ได้เลยใน runtime |
| `_warn` | 104 | 1 | ทั่วไฟล์ | DELETE | เหมือน `_log` |
| `_showStepModal` | 118 | 17 | runEarCheck:652; proceedFromLightCheck:502; startLivenessChallenge:836 | KEEP | ใช้งานจริง |
| `_dismissStepModal` | 136 | 6 | enroll_face.html:419 (onclick) | KEEP | ใช้งานจริง |
| `goToStep` | 168 | 16 | goToLiveness:326; proceedFromLightCheck:504; runEarCheck:657; startLivenessChallenge:843; _showResult:1431; fullRestart:1489; restartCapture:1516 | KEEP | ใช้งานจริง |
| `_csrfToken` | 197 | 5 | goToLiveness:318; _callSpoofCheckSafe:233; _sendToEnroll:1294; fullRestart:1464 | KEEP | ใช้งานจริง |
| `_deviceFingerprint` | 203 | 9 | **ไม่พบผู้เรียก** | DELETE | dead code — ไม่มี caller ใน .js หรือ .html |
| `_captureFrameFromVideo` | 215 | 8 | startLivenessChallenge:730, 814; startCaptureWithDetection:1166 | KEEP | ใช้งานจริง |
| `_callSpoofCheckSafe` | 225 | 43 | startLivenessChallenge:737, 816; startCaptureWithDetection:1172 | KEEP | retry + backoff ถูกต้อง |
| `_stopAllStreams` | 270 | 9 | beforeunload:280; _showResult:1430; fullRestart:1468 | KEEP | ใช้งานจริง |
| `_setSpoofLabel` | 282 | 9 | startLivenessChallenge:738, 817; startCaptureWithDetection:1173 | KEEP | ใช้งานจริง |
| `_clearSpoofLabel` | 292 | 4 | startLivenessChallenge:739, 829; fullRestart:1475, 1476; restartCapture:1513 | KEEP | ใช้งานจริง |
| `_showSpoofWarn` | 298 | 11 | startLightCheck:431; startLivenessChallenge:746, 820; startCaptureWithDetection:1094, 1190 | KEEP | ใช้งานจริง |
| `goToLiveness` | 313 | 16 | enroll_face.html:156 (onclick) | KEEP | entry point ของ flow |
| `startCamera` | 333 | 20 | **ไม่พบผู้เรียก** | DELETE | dead code — ทุก step เรียก `getUserMedia` โดยตรง |
| `stopStream` | 354 | 3 | startCaptureWithDetection:1238; restartCapture:1505 | KEEP | ใช้งานจริง |
| `dist` | 361 | 3 | calcEAR:367 (ซึ่ง dead) เท่านั้น | DELETE | dead code — caller เดียวเป็น dead ด้วย |
| `calcEAR` | 365 | 4 | **ไม่พบผู้เรียก** | DELETE | dead code — ซ้ำกับ `_computeEAR` ที่ใช้งานจริง |
| `stopLightCheck` | 383 | 7 | _stopAllStreams:271; startLightCheck:432; proceedFromLightCheck:500 | KEEP | ใช้งานจริง |
| `startLightCheck` | 391 | 106 | goToLiveness:327; startLightCheck:440 (recursive retry); fullRestart:1490 | REFACTOR | 106 LOC รวม camera setup + brightness poll + FaceMesh spoof; ควรแยก `_startLightCheckCamera()` |
| `proceedFromLightCheck` | 498 | 9 | startLightCheck:484; enroll_face.html:180 (onclick) | KEEP | ใช้งานจริง |
| `stopEarCheck` | 513 | 7 | _stopAllStreams:271; runEarCheck:656 | KEEP | ใช้งานจริง |
| `startEarCheck` | 521 | 30 | proceedFromLightCheck:505 | KEEP | ใช้งานจริง |
| `runEarCheck` | 552 | 110 | enroll_face.html:204 (onclick) | REFACTOR | 110 LOC; สร้าง FaceMesh instance ใหม่เอง (ไม่ใช้ `_getSharedFM`) ขัดกับ singleton pattern; debug log `console.log` ไม่ผ่าน `_warn` |
| `_buildChallengePills` | 677 | 13 | startLivenessChallenge:696, 777 | KEEP | ใช้งานจริง |
| `startLivenessChallenge` | 691 | 155 | runEarCheck:658; overridden โดย enrollment_circular.js:639 | SPLIT | 155 LOC รวม pre-spoof-check + challenge + post-spoof-check; ควรแยก `_runPreSpoofCheck()` และ `_runPostSpoofCheck()` |
| `drawFaceFeatures` | 854 | 18 | startCaptureWithDetection:1083, 1140 | MERGE | logic ซ้ำกับ checkin_flow.js:415 (`_drawFaceFeatures`) ควรย้ายไป shared lib |
| `prepareCaptureStep` | 878 | 6 | startLivenessChallenge:844; restartCapture:1517 | KEEP | ใช้งานจริง |
| `onStartCaptureClicked` | 885 | 5 | enroll_face.html:237 (onclick) | KEEP | ใช้งานจริง |
| `_checkFrontal` | 892 | 13 | startCaptureWithDetection:1127 | KEEP | ใช้งานจริง |
| `_checkCentering` | 914 | 7 | startCaptureWithDetection:1130 | KEEP | ใช้งานจริง |
| `_checkNeutral` | 933 | 26 | startCaptureWithDetection:1134 | KEEP | ใช้งานจริง |
| `_checkCameraConditions` | 963 | 38 | startLightCheck:459; startCaptureWithDetection:1150 | KEEP | ใช้งานจริง |
| `_checkBlur` | 1004 | 26 | startCaptureWithDetection:1158 | KEEP | ใช้งานจริง |
| `_updateCaptureDots` | 1031 | 11 | startCaptureWithDetection:1062; fullRestart:1488; restartCapture:1515 | KEEP | ใช้งานจริง |
| `_flashCapture` | 1043 | 5 | startCaptureWithDetection:1231 | KEEP | ใช้งานจริง |
| `_addThumbnail` | 1049 | 6 | startCaptureWithDetection:1229; _sendToEnroll:1314 | KEEP | ใช้งานจริง |
| `startCaptureWithDetection` | 1056 | 223 | onStartCaptureClicked:888 | SPLIT | ฟังก์ชันใหญ่สุด 223 LOC; รวม gate checks + spoof + capture + stream + timeout + FaceMesh callback ทั้งหมด; onResults callback คือ 180+ LOC ฝังใน |
| `_sendToEnroll` | 1284 | 82 | startCaptureWithDetection:1239 | KEEP | 82 LOC แต่ if-chain สำหรับ server status อ่านได้ |
| `_startProgress` | 1386 | 14 | _sendToEnroll:1288 | KEEP | ใช้งานจริง |
| `_finishProgress` | 1401 | 7 | _sendToEnroll:1302 | KEEP | ใช้งานจริง |
| `_clearProgress` | 1409 | 8 | _startProgress:1387; _hideChecking:1425 | KEEP | ใช้งานจริง |
| `_showChecking` | 1418 | 5 | _sendToEnroll:1287 | KEEP | ใช้งานจริง |
| `_hideChecking` | 1423 | 4 | _sendToEnroll:1304, 1358 | KEEP | ใช้งานจริง |
| `_showResult` | 1428 | 16 | _sendToEnroll:1326, 1332, 1338, 1342, 1347, 1355 | KEEP | ใช้งานจริง |
| `fullRestart` | 1446 | 46 | startLivenessChallenge:824; startCaptureWithDetection:1099, 1197; enrollment_circular.js:485; enroll_face.html:284 (onclick) | KEEP | ใช้งานจริง |
| `restartCapture` | 1494 | 25 | startCaptureWithDetection:1263, 1274; _sendToEnroll:1317, 1323, 1339, 1361 | KEEP | ใช้งานจริง |

---

## Client-side Security

จุดที่ client ตัดสินใจเองแล้วส่งผลให้ server เชื่อ หรือข้ามขั้นตอนได้

### 1. `challengeAttempts` counter (line 52, อ้างอิงที่ 748–804)

Client นับจำนวนครั้งที่ liveness challenge ล้มเหลว (`MAX_CHALLENGE_ATTEMPTS = 5`) ถ้าครบ → cooldown 30 วินาที  
**DevTools bypass:** `challengeAttempts = 0` หรือ reload หน้า → counter归零, cooldown ไม่เกิด  
**ผลลัพธ์:** ลอง liveness challenge ได้ไม่จำกัดครั้ง; server throttle ด้วย `atomic_enroll_attempt` ต่าง session ไม่ได้ป้องกันกรณีนี้

### 2. `step4SpoofFailConsecutive` + `step4SpoofFailTotal` (lines 80–83, 1095–1202)

Client นับจำนวน consecutive spoof fail (max 3) และ total (max 5) ใน capture step → trigger `fullRestart()`  
**DevTools bypass:** `step4SpoofFailConsecutive = 0; step4SpoofFailTotal = 0` ระหว่างถ่ายรูป → gate ไม่ทำงาน  
**ผลลัพธ์:** ลอง capture ซ้ำหลัง spoof fail ได้ไม่จำกัด; server ยังตรวจ spoof ต่อ frame ที่ส่งมา แต่ client-side "eject" logic ไม่มีผล

### 3. `baseline_ear` ส่งไป `/api/enroll` (line 1298)

Client วัด EAR ฝั่งตัวเองและส่งค่าไป server ซึ่ง store ลง `student_biometrics.baseline_ear`  
**DevTools bypass:** `baselineEAR = 0.01` หรือ intercept request และแก้ค่าก่อน submit  
**ผลลัพธ์:** server validate range (0.0 < x < 1.0) ที่ student.py:330 — ค่า extreme เช่น 0.01 หรือ 0.99 ผ่าน validation; ค่าที่ stored ใน DB อาจมีผลต่อ check-in liveness comparison ในอนาคต; ปัจจุบัน api_checkin.py ใช้ค่าที่วัดเองจาก webcam frame ไม่ใช่ค่าจาก DB โดยตรง → ความเสี่ยงต่ำในปัจจุบัน

### 4. `ear_std` ส่งไป `/api/enroll` (line 1299)

Client คำนวณ std-dev ของ EAR samples ระหว่าง capture แล้วส่ง  
**DevTools bypass:** แก้ค่าใน request body เป็นอะไรก็ได้  
**ผลลัพธ์:** ไม่มีผล — student.py:507 comment ระบุ "blocking disabled for passive capture"; server log เท่านั้น ไม่ reject

### 5. `_networkError` skip (lines 741–744, 818–820 ใน `startLivenessChallenge`)

ถ้า spoof check API ไม่ตอบสนอง (network error, 429, timeout) → client set `_networkError: true` → ข้าม pre/post challenge spoof check ไปเลย  
**DevTools bypass:** ไม่ต้อง bypass — block request ใน DevTools Network tab ด้วย "Block request URL" คือเพียงพอ  
**ผลลัพธ์:** spoof check ก่อนและหลัง liveness challenge ถูกข้าม; server ยังตรวจ spoof ต่อ frame ใน `/api/enroll` แต่ liveness spoof gate หายไป; intentional fail-open แต่ exploitable

### 6. `capturedImages` array injection (line 45, 1228)

Client push frame เข้า array เมื่อผ่าน gate checks (frontal, neutral, blur, spoof) ครบ 5 → ส่งไป `/api/enroll`  
**DevTools bypass:** `capturedImages.push(atob('...arbitrary base64 JPEG...'))` → frame ที่ inject ผ่าน gate checks ทั้งหมดบน client  
**ผลลัพธ์:** server ยังตรวจ spoof per frame ผ่าน `combined_spoof_score`; FaceNet512 embedding ก็ extract จาก frame นั้น; ถ้า attacker inject รูปหน้าจริงที่ถ่ายล่วงหน้า → client gate ทั้งหมดไม่มีความหมาย; server spoof เป็น gate สุดท้ายเดียวที่เหลือ

### 7. `step4Timer` (line 87, 1268)

2-minute timeout trigger `restartCapture()` ถ้า capture ไม่เสร็จใน 2 นาที  
**DevTools bypass:** `clearTimeout(step4Timer); step4Timer = null` → timeout ไม่มีผล  
**ผลลัพธ์:** อาจใช้เวลาเตรียม frame นานเท่าใดก็ได้ก่อน submit; ไม่กระทบ security โดยตรง

---

## Dead Code

### ผลลัพธ์ Q-6 (commit 22d07d7) — ลบแล้วทั้งหมด

| รายการ | line เดิม | ประเภท | สถานะ |
|---|---|---|---|
| `startCamera` | 333 | function | **ลบแล้ว** |
| `calcEAR` | 365 | function | **ลบแล้ว** |
| `dist` | 361 | function | **ลบแล้ว** |
| `_deviceFingerprint` | 203 | function | **ลบแล้ว** |
| `_PROGRESS_MSGS_VERIFY` | 1378 | const | **ลบแล้ว** |
| `calibEARValues` | 13 | let | **ลบแล้ว** |
| `calibrating` | 14 | let | **ลบแล้ว** |
| `calibStream` | 8 | let + _stopAllStreams refs | **ลบแล้ว** (รวม refs ใน _stopAllStreams:273,276) |
| `verifyStream` | 10 | let + _stopAllStreams refs | **ลบแล้ว** (รวม refs ใน _stopAllStreams:273,276) |
| `_log` | 103 | function | **เก็บไว้** — DEBUG เปลี่ยนเป็น `localStorage.getItem('sc_debug') === '1'` |
| `_warn` | 104 | function | **เก็บไว้** — เปิด debug ด้วย `localStorage.setItem('sc_debug','1')` ใน devtools |

---

## Top 3

### 1. `startCaptureWithDetection` — SPLIT (223 LOC)

ฟังก์ชันใหญ่สุดในไฟล์และใน project JS ทั้งหมด  
`onResults` callback ฝังใน (line ~1068–1241) มี 170+ LOC เป็นฟังก์ชันไม่มีชื่อที่รวม: real-time Moiré, gate checks, spoof check, frame capture, counter logic, และ timeout management  
แยกเป็น:
- `_onCaptureResults(results)` — รับ callback แยก
- `_runCaptureGates(lm, video)` — gate checks รวม
- `_handleSpoofFail(consec, total)` — counter + restart logic

**ผลลัพธ์:** ง่าย test/debug แต่ละ gate; ลด context ของ `_sendToEnroll` ที่ฝังในด้วย

---

### 2. ลบ dead variables/functions ทั้ง 11 รายการ — **แก้แล้ว commit 22d07d7**

9 รายการลบแล้ว; `_log`/`_warn` เก็บไว้และ DEBUG เปลี่ยนเป็น `localStorage`  
ผล: 1518 → 1460 LOC (-58)

---

### 3. F-6: network-error fail-open — ปิด liveness spoof gate ด้วยการ block request

`_networkError` bypass ใน liveness stage ข้าม spoof check ก่อนและหลัง challenge ได้ด้วยการ block URL เดียว  
counter (`challengeAttempts`, `step4SpoofFailConsecutive`, `step4SpoofFailTotal`) เป็น UX guard ไม่ใช่ security control เพราะ server ตรวจ spoof ทุก frame ที่ `/api/enroll` อยู่แล้ว — ย้ายไป server ไม่ได้ปิดช่องจริง  
แก้: ดู F-6 ด้านล่าง

---

## F-6: network-error fail-open

### 1. โค้ดดิบจุดที่ `_networkError` ถูก set และผลที่เกิด

**`_callSpoofCheckSafe` (line 225–267) — ที่มาของ `_networkError`**

```js
// line 225-267
async function _callSpoofCheckSafe(imageB64) {
    const BACKOFF_MS = [2000, 4000, 8000];
    for (let attempt = 0; attempt < 2; attempt++) {       // retry 2 ครั้ง (attempt 0 + 1)
        try {
            const res = await fetch(ENROLL_CONFIG.spoofCheckUrl, { ... });

            if (res.status === 429) {
                // ...wait then retry once...
                if (attempt === 0) { await sleep(waitMs); continue; }
                return { is_real: false, _networkError: true, _rateLimited: true };  // line 253-255
            }

            if (!res.ok) throw new Error(`HTTP ${res.status}`);             // line 258 → 5xx, 4xx
            return await res.json();                                         // success
        } catch (e) {
            _warn(`spoof_check attempt ${attempt + 1} failed:`, e);
            if (attempt === 0) await new Promise(r => setTimeout(r, 1000)); // wait 1s between attempts
        }
    }
    return { is_real: false, confidence: 0, _networkError: true };          // line 265-266
}
```

`_networkError: true` เกิดใน **3 กรณี**:

| กรณี | trigger | line |
|---|---|---|
| 429 rate-limit หลัง retry | res.status === 429 และ attempt === 1 | 253–255 |
| HTTP error (5xx / 4xx ยกเว้น 429) | `throw new Error(HTTP ${res.status})` → catch → retry หมด | 258, 265–266 |
| Network fail (blocked, timeout, offline) | fetch() throw → catch → retry หมด | 260, 265–266 |

backoff: attempt 0 → wait 1000ms → attempt 1 → return `_networkError`  
**ไม่มี `AbortController` timeout** — ถ้า server รับ request แต่ไม่ตอบ (hang) จะรอไม่จำกัดเวลา ไม่เกิด `_networkError` เอง

---

**pre-challenge spoof check (line 737–761)**

```js
// line 737-761
const sc1 = await _callSpoofCheckSafe(frame1);
if (!sc1.is_real) {
    if (sc1._networkError) {
        // Server validation/network error — skip pre-check, proceed to challenge
        _warn('spoof_check pre-challenge network error — skipping');   // line 743
        // fall through → challenge proceeds
    } else {
        _showSpoofWarn();
        challengeAttempts++;
        // ... retry liveness ...
        return;
    }
}
```

`_networkError` → ข้าม gate นี้ทันที ไม่แสดง warning ต่อ user

---

**post-challenge spoof check (line 816–826)**

```js
// line 816-826
const sc2 = await _callSpoofCheckSafe(frame2);
if (!sc2.is_real && !sc2._networkError) {   // line 818: _networkError exempt
    // Hard spoof detected — fullRestart()
}
// network error → falls through to Step 4 (capture) silently
```

`_networkError` → condition `!sc2.is_real && !sc2._networkError` = false → ข้าม gate ไปเงียบๆ

---

**capture step spoof check (line 1172–1187)**

```js
// line 1175-1187
if (!sc.is_real) {
    if (sc._networkError) {
        capturePaused = false;
        return;   // skip frame, don't push to capturedImages
    }
    // hard spoof → counter++
}
```

behavior แตกต่าง: `_networkError` ในขั้น capture → **ข้าม frame นั้น** (ไม่ push ไม่นับ fail) แล้วรอ frame ต่อไป  
ถ้า block ตลอด → `capturedImages` ไม่เต็ม 5 → timeout 2 นาที → `restartCapture()` ไม่มี enrollment เกิดขึ้น

---

### 2. `_callSpoofCheckSafe` — retry และ return

| | |
|---|---|
| retry | **2 ครั้ง** (attempt 0, 1); loop `for (let attempt = 0; attempt < 2; attempt++)` |
| backoff ระหว่าง attempt | **1000ms** hardcode (`if (attempt === 0) await sleep(1000)`) บรรทัด 262 |
| backoff array `BACKOFF_MS` | นิยามไว้ที่ line 226 แต่ใช้เฉพาะในกรณี 429 (ล.247) ไม่ได้ใช้ใน catch block |
| return หลัง retry หมด | `{ is_real: false, confidence: 0, message: '...', _networkError: true }` (line 265–266) |

---

### 3. สิ่งที่เกิดถ้า block `/api/spoof_check` ใน DevTools

**ตอบตรงๆ:**

| check | ถูกข้ามหรือไม่ |
|---|---|
| pre-challenge spoof check (line 737–761) | **ใช่ — ถูกข้าม** อย่างชัดเจน (line 741-743) |
| post-challenge spoof check (line 816–826) | **ใช่ — ถูกข้าม** (condition line 818 exempt networkError) |
| capture per-frame spoof check (line 1172–1187) | **ไม่ถูกข้าม** แต่เปลี่ยนเป็น skip-frame → enrollment หยุดชะงัก ไม่เสร็จถ้า block ตลอด |

**attack path ที่ใช้งานได้จริง:**  
Block `/api/spoof_check` เฉพาะ **ระหว่าง liveness stage** (ก่อน + หลัง challenge) → ทั้งสอง gate ข้ามไป → unblock ระหว่าง capture stage → capture ดำเนินตามปกติ  
ผลลัพธ์: enrollment ผ่านโดยไม่มี liveness spoof check เลย

**gate ที่ยังทำงานฝั่ง server:**

| gate | file:line |
|---|---|
| `server_validate_frame` per frame | student.py:382 |
| `detect_screen_moire` per frame | student.py:450 |
| `detect_screen_texture` per frame | student.py:469 |
| `detect_static_image` per frame | student.py:487 |
| `check_anti_spoof` per frame | student.py:516 |
| `check_embedding_consistency` | student.py:587 |
| duplicate check ต่อ stored students | student.py:662 |

server ยังตรวจสอบ spoof ทุก frame ผ่าน `combined_spoof_score` ใน `/api/enroll` — liveness spoof gate เป็น defense-in-depth ไม่ใช่ gate เดียว แต่ข้ามได้ฟรีด้วยการ block URL

---

### 4. ทางแก้สองทาง

**(a) fail-close: network error = ถือว่าไม่ผ่าน ให้ retry**

แก้ใน `startLivenessChallenge` (line 741-744 และ line 818):

```js
// เดิม
if (sc1._networkError) {
    _warn('...skipping');
    // fall through
}

// ใหม่
if (sc1._networkError) {
    document.getElementById('livenessStatus').textContent =
        'ไม่สามารถตรวจสอบได้ — กรุณาตรวจสอบการเชื่อมต่อและลองใหม่';
    _livenessRetryTimer = setTimeout(() => startLivenessChallenge(), 3000);
    return;
}
```

| | |
|---|---|
| **ข้อดี** | ปิดช่อง bypass อย่างสมบูรณ์; ไม่ต้องแก้ server; เปลี่ยน 2-3 บรรทัด |
| **ข้อเสีย** | นักศึกษาที่ WiFi ไม่เสถียรอาจ retry ซ้ำหลายรอบ; ถ้า `/api/spoof_check` down ทั้ง instance → enrollment block หมด |

**(b) server-side flag: บันทึกว่า spoof check ทำหรือไม่ แล้วเข้มที่ `/api/enroll`**

- `/api/spoof_check` ทุก call ที่ผ่าน → set `session["liveness_spoof_ok"] = True`
- `/api/enroll` ตรวจ: ถ้า `liveness_spoof_ok` ไม่ได้ตั้ง → เพิ่มความเข้มของ spoof check (เช่น ใช้ threshold เข้มขึ้น หรือ reject ทันที)

| | |
|---|---|
| **ข้อดี** | enrollment ยังทำงานได้เมื่อ WiFi ไม่เสถียร; server รู้ว่า spoof check ถูกข้าม; สามารถ audit ย้อนหลังได้ |
| **ข้อเสีย** | ต้องแก้ `/api/spoof_check` route + `/api/enroll` route + เพิ่ม session key; ซับซ้อนกว่า; ไม่ป้องกัน attacker ที่ตั้งใจ block (server จะตรวจเข้มขึ้น แต่ก็ยังผ่านได้ถ้า frame จริง) |

**ข้อแนะนำ:** ทาง (a) ปิดช่องได้ตรงกว่า ใช้เวลา < 30 นาที; ทาง (b) เหมาะถ้ากังวลเรื่อง FRR จาก network ไม่เสถียรใน real deployment

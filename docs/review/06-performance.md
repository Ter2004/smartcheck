# Performance — Check-in Path (`/api/checkin`)

วันที่: 2026-08-23

## 0. Environment caveat — อ่านก่อนเชื่อตัวเลขใดๆ ในไฟล์นี้

ทุกตัวเลขในไฟล์นี้วัดจริงบน **sandbox environment นี้** (Windows, Python 3.11) **ไม่ใช่** container Railway ที่ deploy จริง และ package versions ต่างจากที่ pin ไว้ใน `requirements.txt` / `constraints.txt`:

| package | pinned (production) | ติดตั้งจริงใน sandbox นี้ |
|---|---|---|
| numpy | `1.26.4` | `2.4.4` |
| tensorflow | `2.15.0` | `2.20.0` |
| tf-keras | `2.15.1` | `2.20.1` |
| torch | `2.2.2` (CPU wheel) | ไม่ได้ติดตั้ง (Fasnet ผ่าน DeepFace ยัง fallback ได้) |
| onnxruntime | `1.24.4` | `1.24.4` (ตรงกัน) |
| opencv | `>=4.8,<4.12` | `4.13.0` |

**ผลกระทบ:** ตัวเลข RAM/เวลาที่วัดได้เป็น**ค่าบ่งชี้สัดส่วน** (ขั้นไหนกินเวลา/RAM มากกว่ากันกี่เท่า) ได้แม่นยำ แต่**ค่าสัมบูรณ์อาจต่างจาก production จริง** เพราะ TF 2.20 vs 2.15 มี memory footprint ต่างกันได้

ไม่มีรูปหน้าคนจริงในโปรเจกต์นี้ให้ทดสอบ (grep ทั้ง repo หา `.jpg/.jpeg/.png` ไม่เจอไฟล์รูปเลย) และไม่มีกล้อง — การวัด "เวลาโมเดล" และ "RAM โมเดล" ในข้อ 2–3 จึงใช้ synthetic image (`numpy` random noise) กับ `enforce_detection=False` เพื่อบังคับให้ TensorFlow graph / Fasnet / ONNX โหลดและรัน forward pass จริง (สิ่งที่กิน RAM/เวลา คือการโหลด+รัน model ไม่ใช่ว่าจะเจอหน้าจริงหรือไม่) — **แต่ตัวเลข "detection accuracy" ไม่มีความหมายใดๆ**

ไม่มีการเชื่อมต่อ Supabase จริงจาก environment นี้ (ไม่มี credentials/network) → **DB round-trip time วัดไม่ได้เลย ระบุว่า "ไม่ทราบ" ไม่เดา** — ตัวเลข capacity ท้ายไฟล์จึงมี assumption ส่วนนี้กำกับไว้ชัดเจน

---

## 1. Timing instrumentation (ชั่วคราว — commit แล้ว)

เพิ่มใน `app/routes/api_checkin.py` (`checkin()`) และ `app/services/face_service.py` (`combined_spoof_score()`) — ทั้งสองจุด comment กำกับ `# TEMP PERF: ... — remove after measurement` **ไม่เปลี่ยน logic เดิมเลย** (ไม่มี branch/condition/return ใดถูกแก้ มีแค่ `time.perf_counter()` เก็บลง dict + log บรรทัดเดียวก่อน return สำเร็จ)

**Log format จริงที่จะเห็นใน production log:**
```
[PERF] total=Xms validate=Xms session=Xms ble=Xms ear=Xms moire=Xms texture=Xms
temporal=Xms fasnet=Xms onnx=Xms embed=Xms verify=Xms db=Xms
```
Log บรรทัดนี้ยิงเฉพาะ **happy path** (request ที่ผ่านทุกขั้นจนถึง success) เพราะวางไว้บรรทัดสุดท้ายก่อน `return jsonify({"ok": True, ...})` — request ที่ reject กลางทาง (session ปิด, spoof detected, ฯลฯ) จะไม่มี log บรรทัดนี้ (ตาม field list ที่ขอ ซึ่งมีครบทุก stage แปลว่าต้องเป็น happy path เท่านั้น)

### Field → code mapping (ประกาศไว้ชัดเจนเพราะไม่มี 1:1 ตายตัวจาก field name ที่ให้มา)

| field | ครอบคลุมโค้ดส่วนไหน | เหตุผลที่จัดกลุ่มแบบนี้ |
|---|---|---|
| `validate` | input payload validation + device-token verify (ก่อน query session) | ทุกอย่างก่อนแตะ DB |
| `session` | session query + course-enrollment query + `server_validate_frame` (1b) + checkin-window check | ทุกอย่างที่ตัดสิน "request นี้มีสิทธิ์ดำเนินต่อมั้ย" ก่อนเข้า AI pipeline |
| `ble` | BLE RSSI block | ตรงตัว |
| `ear` | EAR liveness block | ตรงตัว |
| `moire` | Moiré FFT แบบ standalone (4a) | ตรงตัว — **ไม่รวม** Moiré ที่รันซ้ำข้างในข้อ 4b (ดูข้อ 2 ล่าง) |
| `texture` | Screen Texture แบบ standalone (4a-2) | เหมือน moire |
| `temporal` | Temporal variance (4a-3) | ตรงตัว |
| `fasnet` | ดึงจาก `combined_spoof_score()["timings"]["fasnet_ms"]` — วัดจาก**ข้างใน** `face_service.py` เพราะ api_checkin.py เห็นแค่ผลรวม | ต้องแตะ `face_service.py` เพิ่ม `timings` key แบบ additive-only (ได้รับอนุมัติแล้วระหว่างทำงาน) |
| `onnx` | ดึงจาก `combined_spoof_score()["timings"]["onnx_ms"]` | เหมือน fasnet |
| `embed` | device binding (step 5) + biometrics fetch + integrity check + `extract_embedding()` รวมกัน | ไม่มี field แยกสำหรับ device-binding ใน field list ที่ขอ จึงพับรวมเข้า embed (เป็นก้อนก่อน extract_embedding ที่ถูกที่สุดในบรรดา field ที่มี) |
| `verify` | `verify_face_multi()` | ตรงตัว |
| `db` | duplicate-check query + status calc + insert attendance | ตรงตัว |

**สถานะ:** commit แล้วที่ `cb13353` (message: `perf(instrument): add temporary [PERF] timing logs to checkin path`) — **ยังไม่ push** และมี TODO บันทึกไว้ใน `docs/review/00b-preflight.md` ให้ถอดออกหลังเก็บข้อมูลเสร็จ รอคุณรัน `python run.py` ทดสอบเองแล้วดู log จริงจาก [PERF] บรรทัดนี้

---

## 2. จำนวนครั้งที่ detect face ต่อ 1 request (นับจาก source จริง)

**ยืนยันจากการอ่าน `app/services/face_service.py` และ `app/routes/api_checkin.py` จริง (ไม่ใช่เดา) — check-in สำเร็จ 1 ครั้งเรียก face detection 3 รอบ:**

| # | จุดเรียก | file:line | เรียกจาก | detector |
|---|---|---|---|---|
| 1 | `DeepFace.extract_faces(detector_backend="opencv", anti_spoofing=True, enforce_detection=True)` | `face_service.py:147` (ใน `_run_fasnet_antispoof`) | `combined_spoof_score()` Layer 4 (Fasnet) ← `api_checkin.py:246` | DeepFace's own "opencv" backend |
| 2 | `_face_cascade.detectMultiScale(...)` | `face_service.py:76` (ใน `_crop_face_for_antispoof`) | `_run_antispoof()` (`face_service.py:96`) ← `combined_spoof_score()` Layer 5 (ONNX) ← `api_checkin.py:246` | โมดูล `cv2.CascadeClassifier` (Haar) ของเราเอง — คนละตัวกับ #1 |
| 3 | `DeepFace.represent(model_name="Facenet512", detector_backend="opencv", enforce_detection=True)` | `face_service.py:458` (ใน `extract_embedding`) | `api_checkin.py:326` | DeepFace's "opencv" backend (เหมือน #1) |

**ไม่ถูกเรียก:** `detect_static_image()` (`face_service.py:818`, ใช้ Haar cascade เช่นกันที่ `face_service.py:832`) — ไม่ถูกเรียกใน check-in เพราะ `combined_spoof_score(raw_frame)` ที่ `api_checkin.py:246` เรียกโดยไม่ส่ง `frames_for_temporal` (ใช้ default `None`) จึง layer Temporal ข้างในถูก skip — Temporal ที่ check-in ใช้จริงคือ block แยกที่ `api_checkin.py:211-242` ซึ่งไม่เรียก face detection เลย (ทำ grayscale variance บนทั้งภาพ ไม่ crop หน้า)

**สิ่งที่พบเพิ่ม (ไม่ได้ถามตรงๆ แต่เกี่ยวข้อง):** Moiré FFT และ Screen Texture FFT ก็ถูกคำนวณ **ซ้ำ 2 รอบ** เช่นกัน — รอบแรก standalone ที่ `api_checkin.py:164-201` (4a, 4a-2) แล้วรอบสองซ้ำอีกครั้งข้างใน `combined_spoof_score()` (`face_service.py:189, 209` — Layer 1/2) เพราะ `combined_spoof_score()` ถูกออกแบบให้รันครบ 5 layer เองเสมอไม่ว่าผู้เรียกจะรันมาก่อนหรือไม่ — นี่คือสิ่งที่ P-2 (`99-summary.md`) พยายามแก้บางส่วนแล้ว (เปลี่ยนจาก `check_anti_spoof(face_image)` เป็น `combined_spoof_score(raw_frame)` เพื่อไม่ decode ซ้ำ) แต่การคำนวณ Moiré/Texture ซ้ำภายในยังเหลืออยู่ — ไม่ใช่ target ของ task นี้ (ห้ามแก้ logic) จึงแค่บันทึกไว้

### ตอบ: crop จากรอบแรกส่งต่อไปรอบถัดไปได้มั้ย?

**ได้บางส่วน** — เฉพาะ crop จาก **รอบ #1** (`DeepFace.extract_faces` ใน Fasnet layer) เท่านั้นที่นำไปใช้กับรอบ **#3** (`extract_embedding`) ได้ เพราะทั้งคู่ใช้ detector backend เดียวกัน (`"opencv"` ของ DeepFace) ส่วนรอบ #2 (Haar cascade ของเราเอง, crop scale 2.7x, resize 80×80 สำหรับโมเดล ONNX antispoof) เป็นคนละ detector คนละ crop scale — เอาไปใช้กับ FaceNet512 (ต้องการ input alignment/scale ต่างกัน) ไม่ได้

ยืนยันจาก DeepFace source จริง (`deepface.modules.detection.extract_faces` docstring, ตรวจสอบแล้วในเครื่อง): `detector_backend` รองรับค่า `"skip"` อย่างเป็นทางการ — ถ้า pass ภาพที่ crop มาแล้วเข้าไปพร้อม `detector_backend="skip"` DeepFace จะข้ามขั้นตอน detect กรอบหน้าไปเลย (ใช้ทั้งภาพที่ส่งมาเป็นหน้าเลย)

**ถ้าจะทำจริง ต้องแก้อะไรบ้าง (ยังไม่แก้ แค่รายงาน):**

1. `_run_fasnet_antispoof()` (`face_service.py:138`) ต้อง **return crop ด้วย** — ตอนนี้ return แค่ `(is_real, spoof_score)` ไม่ได้ return `faces[0]['face']` ที่ได้จาก `DeepFace.extract_faces` (ทั้งที่ตัวแปร `face` มีอยู่แล้วใน local scope บรรทัด 155)
2. `combined_spoof_score()` (`face_service.py:166`) ต้อง thread ค่า crop นั้นผ่าน return dict ของตัวเอง (เพิ่ม key ใหม่ เช่น `"fasnet_crop"`) ให้ผู้เรียกดึงไปใช้ได้
3. `api_checkin.py:246` (จุดเรียก `combined_spoof_score`) ต้องดึง crop นั้นออกมา แล้วส่งต่อให้ `extract_embedding()`
4. `extract_embedding()` (`face_service.py:448`) ต้องมี branch ใหม่รับ pre-cropped face array (เรียก `DeepFace.represent(img_path=crop, detector_backend="skip", enforce_detection=False)` แทนที่จะ decode+detect เอง) — **ตอนนี้รับแค่ base64 string แล้ว decode+CLAHE+detect เองทั้งหมด** ต้องเพิ่ม parameter หรือแยกฟังก์ชันใหม่
5. **จุดที่ต้องระวังเรื่อง correctness:** `extract_embedding()` ปัจจุบันรัน `normalize_illumination()` (CLAHE) บน**ภาพเต็มก่อน detect** (`face_service.py:455-456`) — ถ้าใช้ crop จาก Fasnet ซึ่ง detect จาก **ภาพดิบที่ยังไม่ทำ CLAHE** embedding ที่ได้จะต่างจากปัจจุบัน (ไม่ผ่าน CLAHE) → ต้อง apply CLAHE กับภาพเต็มก่อน pass เข้า Fasnet ตั้งแต่ต้น (เปลี่ยนลำดับ) ไม่งั้นค่า embedding จะเบี่ยงจากที่ threshold (`SAME_DEVICE_THRESHOLD`/`NEW_DEVICE_THRESHOLD`) เคย calibrate ไว้ — **ต้อง re-validate threshold ใหม่หลังเปลี่ยน**
6. **ผลข้างเคียงด้าน accuracy:** `DeepFace.extract_faces(..., align=True)` (default) ทำ face alignment (หมุนตามแนวตา) ให้ระหว่าง detect — แต่ `detector_backend="skip"` จะ**ไม่ทำ alignment** เพราะไม่มีขั้นตอน detect ให้ดึง landmark มาใช้หมุน ผลคือ embedding อาจแม่นยำลดลงเล็กน้อยถ้าหน้าที่ capture มาไม่ตรงกล้องพอดี (ความเสี่ยงต่ำเพราะ flow ปัจจุบันมี frontal/centering gate ฝั่ง client อยู่แล้วก่อนถ่ายจริง แต่ยังเป็นความเสี่ยงที่ต้องรู้ไว้)

**สรุป:** ทำได้จริง ตัดรอบ detection จาก 3 → 2 รอบ (ยังเหลือ Haar cascade ของ ONNX แยกอยู่ เพราะ crop scale ไม่ตรงกัน) แต่ไม่ใช่ config-flip — ต้องแก้ signature 3 ฟังก์ชัน + validate threshold ใหม่ ประเมินเป็นงานระดับ "ปานกลาง" ไม่ใช่ "ต่ำ"

---

## 3. RAM measurement

Script ชั่วคราว: `/tmp/perf_ram_check.py` (mapped จริงคือ `C:\Users\LENOVO\AppData\Local\Temp\perf_ram_check.py`) — วัดด้วย `psutil.Process().memory_info().rss` (ติดตั้ง `psutil` เพิ่มชั่วคราวเพื่อวัด — ไม่ใช่ dependency ของโปรเจกต์ ไม่ได้เพิ่มใน requirements.txt)

**หมายเหตุสำคัญ:** เพื่อ import `face_service.py` โดยไม่ต้องลง `flask-sqlalchemy` (ไม่มีใน sandbox นี้ และ `face_service.py` เองไม่ได้ import Flask อะไรเลย) script โหลดไฟล์ตรงด้วย `importlib` แทนการ `import app.services.face_service` ตามปกติ — ผลลัพธ์ RAM/เวลาของโค้ด `face_service.py` เองไม่กระทบจากวิธี import นี้

### ผลวัดจริง (คำสั่งเดียว รันครั้งเดียว บันทึกดิบ)

| ขั้นตอน | RSS หลังทำ | Δ (เพิ่มจากขั้นก่อน) | เวลาที่ใช้ |
|---|---|---|---|
| baseline (python เปล่า ก่อนโหลดอะไร) | 29.2 MB | — | — |
| หลัง `import face_service` (cv2, numpy, Haar cascade module-level) | 39.1–39.5 MB | +9.9–10.3 MB | 0.08–0.17s |
| หลัง `DeepFace.represent()` ครั้งแรก (โหลด TF graph + น้ำหนัก Facenet512) | 712.5–713.2 MB | **+673–674 MB** | 15.3–17.7s |
| หลัง `DeepFace.extract_faces(anti_spoofing=True)` ครั้งแรก (โหลดโมเดล Fasnet เพิ่ม) | 882.4–882.8 MB | +169.6–169.9 MB | 4.2–5.8s |
| หลัง โหลด ONNX session (`_get_antispoof_session()`) | **905.8–907.9 MB** | +23–25 MB | 0.23–0.28s |

**RAM ต่อ 1 worker หลัง warm ครบ: ~906 MB** (วัดซ้ำ 2 รอบ ได้ 905.8 และ 907.9 MB — สอดคล้องกัน)

**สังเกต:** ตัวที่กิน RAM มากที่สุดคือ TensorFlow graph + น้ำหนัก Facenet512 ตอนโหลดครั้งแรก (~673 MB, 74% ของ RAM ทั้งหมด) — ไม่ใช่ ONNX (เบามาก, +23-25 MB เท่านั้น) ก็ไม่ใช่ Haar cascade (รวมอยู่ใน +10 MB แรก) การโหลด TensorFlow/Keras คือต้นทุนหลักของ RAM ทั้งหมด

---

## 4. Gunicorn / deployment config (raw)

**`Procfile`:**
```
web: gunicorn "app:create_app()" --bind 0.0.0.0:$PORT --workers 1 --timeout 120
```

**`railway.json`:**
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE"
  },
  "deploy": {
    "healthcheckPath": "/health",
    "healthcheckTimeout": 600,
    "restartPolicyType": "ON_FAILURE"
  }
}
```

**`Dockerfile`** (บรรทัดสร้าง entrypoint จริง):
```dockerfile
RUN printf '#!/bin/sh\nexport TF_USE_LEGACY_KERAS=1\nexport TF_CPP_MIN_LOG_LEVEL=2\nexec gunicorn "app:create_app()" --bind "0.0.0.0:${PORT:-8080}" --workers 1 --timeout 300\n' > /start.sh && chmod +x /start.sh
ENTRYPOINT ["/start.sh"]
```

### สรุปค่าจริงที่ใช้

| ค่า | Procfile | Dockerfile (`/start.sh`) | ค่าที่ deploy จริงใช้ |
|---|---|---|---|
| workers | `1` | `1` | **1** (ตรงกันทั้งคู่) |
| timeout | `120` | `300` | **ขัดแย้งกัน — ดูหมายเหตุ** |
| worker_class | ไม่ได้ตั้ง | ไม่ได้ตั้ง | **`sync`** (default ของ gunicorn — ยืนยันจาก source `gunicorn/config.py` ที่ติดตั้งจริง เวอร์ชัน 23.0.0: `class WorkerClass: default = "sync"`) |

**หมายเหตุ — Procfile กับ Dockerfile ขัดแย้งกัน:** `railway.json` ระบุ `"builder": "DOCKERFILE"` ชัดเจน แปลว่า Railway build ด้วย Dockerfile และรันผ่าน `ENTRYPOINT ["/start.sh"]` ของ Dockerfile — **ไม่ใช้ Procfile** (Procfile เป็น convention ของ Heroku/nixpacks buildpack ซึ่งไม่ได้ถูกใช้เมื่อ builder เป็น `DOCKERFILE` โดยตรง) ดังนั้น timeout ที่ใช้จริงน่าจะเป็น **300 วินาที** จาก Dockerfile ไม่ใช่ 120 จาก Procfile — **Procfile น่าจะเป็นไฟล์ตกค้าง (dead config)** ควรลบหรือแก้ให้ตรงกันเพื่อไม่ให้คนอ่านโค้ดสับสนว่าใช้ค่าไหนจริง (ระดับความมั่นใจ: สูง แต่ไม่ใช่ 100% เพราะไม่ได้ verify กับ Railway deployment จริง)

**ถ้าไม่ตั้ง `--timeout` เลย:** default ของ gunicorn คือ **30 วินาที** (ยืนยันจาก source `gunicorn/config.py` ที่ติดตั้งจริง: `class Timeout: default = 30`) — ทั้งสองไฟล์ในโปรเจกต์นี้ตั้งไว้สูงกว่า default มาก (120/300) อยู่แล้ว ไม่ใช่ปัญหาปัจจุบัน

**Worker class `sync` หมายความว่าอะไร:** 1 worker process จัดการได้ **ครั้งละ 1 request เท่านั้น** (blocking, ไม่มี concurrency ภายใน worker เดียว) — เมื่อรวมกับ `--workers 1` แปลว่า **ทั้งระบบรับได้ครั้งละ 1 check-in request เท่านั้น** request อื่นที่มาพร้อมกันจะเข้าคิวรอที่ OS socket backlog จนกว่า worker จะว่าง — นี่คือปัจจัยที่กระทบตัวเลข "50 คนพร้อมกัน" ในข้อ 6 มากที่สุด มากกว่าการ optimize AI pipeline ใดๆ

---

## 5. Warm per-call latency (วัดจริง, models โหลดแล้ว = สภาพจริงตอน serve request ที่ 2 เป็นต้นไป)

รันบน synthetic image เดียวกัน (480×640 random noise), แต่ละอันวัด 5 ครั้ง (`combined_spoof_score` วัด 3 ครั้งเพราะช้ากว่า):

| ฟังก์ชัน | min | median | max |
|---|---|---|---|
| `detect_screen_moire` | 3.6ms | 4.2ms | 7.3ms |
| `detect_screen_texture` | 3.9ms | 4.0ms | 8.4ms |
| temporal inline (ตาม logic ใน `api_checkin.py:211-235`, 3 เฟรม) | 0.20ms | 0.27ms | 7.05ms |
| `_crop_face_for_antispoof` (Haar cascade เดี่ยวๆ) | 15.6ms | 18.6ms | 19.4ms |
| `_run_antispoof` (ONNX เต็ม รวม Haar) | 22.5ms | 26.0ms | 28.0ms |
| `_run_fasnet_antispoof` (DeepFace, warm) | 59.0ms | 66.3ms | 75.5ms |
| `DeepFace.represent` (Facenet512, warm) — **นี่คือ `embed`** | 333.1ms | 357.4ms | 388.8ms |
| `combined_spoof_score` (ครบ 5 layer, warm) | 90.3ms | 94.4ms | 97.5ms |

**คำเตือนเรื่อง bias ของตัวเลข `_run_fasnet_antispoof` / `combined_spoof_score`:** synthetic image เป็น random noise ล้วนๆ ทำให้ DeepFace **ตรวจไม่เจอหน้าเลยทุกครั้ง** (raise `Face could not be detected`) — เวลาที่วัดได้คือเวลาที่ใช้ "พยายาม detect แล้วล้มเหลว" ไม่ใช่เวลา "detect สำเร็จ + รัน antispoof classifier" ซึ่งบนหน้าจริงมักจะ**นานกว่านี้** (มีขั้นตอน classify เพิ่มหลัง detect สำเร็จ) → เลข `fasnet` และ `combined_spoof_score` ในตารางนี้ควรมองเป็น **lower bound (ค่าต่ำสุดที่เป็นไปได้)** ไม่ใช่ค่าเฉลี่ยจริงบนใบหน้าจริง

**Insight ที่ชัดที่สุดจากตารางนี้:** `DeepFace.represent` (การสร้าง embedding FaceNet512) กินเวลา **~350ms** — มากกว่า `combined_spoof_score` ทั้งก้อน (~94ms) เกือบ 4 เท่า และมากกว่าทุก layer อื่นรวมกันหลายเท่าตัว **นี่คือคอขวดที่ใหญ่ที่สุดของ AI compute ทั้งหมดในเส้นทาง check-in** ไม่ใช่ anti-spoof pipeline อย่างที่อาจคาดไว้

---

## 6. สรุปประเมิน Capacity — 1 / 2 / 4 workers

### สูตรที่ใช้ (ระบุ assumption ทุกจุด)

**AI-compute per request (happy path, วัดจริง รวมกัน):**
`moire(4.2) + texture(4.0) + temporal(0.3) + combined_spoof_score(94.4) + embed(357.4)` ≈ **~460ms**
(ใช้ค่า median จากข้อ 5; `verify_face_multi` เป็น numpy cosine similarity ล้วน ไม่ได้วัดแยกแต่คาดว่า <5ms แน่นอนสำหรับ embedding จำนวนน้อยต่อ user)

**DB round-trips (session, enrollment, device-binding, biometrics fetch, duplicate check, insert — รวม 5-6 calls ไป Supabase):** **ไม่ทราบ — ไม่สามารถวัดจาก environment นี้ได้** (ไม่มี network ไป Supabase) สมมติฐานที่ใช้ด้านล่างคือ **assumption ที่ต้อง verify กับของจริง** ไม่ใช่ตัวเลขวัด

| สถานการณ์ | เวลา/request โดยประมาณ | หมายเหตุ |
|---|---|---|
| Optimistic (DB latency ~0) | ~0.46s | ไม่สมจริง — ใช้เป็น floor เท่านั้น |
| Assumption กลาง (DB รวม ~200-400ms, ทั่วไปสำหรับ Supabase ต่างภูมิภาค) | **~0.7–0.9s** | **ค่าที่ใช้คำนวณด้านล่าง — เป็นการสมมติ ไม่ใช่ค่าวัด** |
| Assumption แย่ (DB latency สูง/cold connection) | ~1.2–1.5s | ถ้า network ไป Supabase ช้า |

### RAM ต่อจำนวน worker

`--workers N` แต่ละ worker เป็น process แยก, gunicorn ไม่ได้ตั้ง `--preload` (ไม่มีใน Procfile/Dockerfile ทั้งคู่) และโมเดลทั้งหมดโหลดแบบ lazy (module-level `_antispoof_session = None`, สร้างจริงตอนถูกเรียกครั้งแรกต่อ process) → **แต่ละ worker กิน RAM เต็มก้อนแยกกัน ไม่แชร์กัน** ประมาณ N × ~906 MB:

| workers | RAM โดยประมาณ (N × 906 MB) |
|---|---|
| 1 | **~0.9 GB** |
| 2 | **~1.8 GB** |
| 4 | **~3.6 GB** |

*(ไม่ทราบ RAM limit ของ Railway plan ที่ใช้อยู่ — ไฟล์ config ในโปรเจกต์ไม่ได้ระบุไว้ ต้องเช็คจาก Railway dashboard เอง ก่อนตัดสินใจว่า 2 หรือ 4 workers เป็นไปได้จริงหรือไม่)*

### เวลาสำหรับ 50 คนพร้อมกัน

**ข้อจำกัดสำคัญที่สุด:** worker class เป็น `sync` (default, ไม่ได้ตั้งเป็นอย่างอื่น) → **1 worker รับได้ครั้งละ 1 request เท่านั้น ไม่มี concurrency ภายใน worker เดียว** ดังนั้น 50 requests พร้อมกันจะถูก**เรียงคิว (serialize)** ตามจำนวน worker ทั้งหมด — ไม่ใช่ทำพร้อมกันจริง

*(ตั้งข้อสังเกตเพิ่ม: การรันแบบ multi-worker แข่งกันใช้ CPU จริงได้ก็ต่อเมื่อ host มี CPU core เพียงพอ — จำนวน core ของ Railway plan ที่ใช้อยู่ก็ไม่ทราบเช่นกัน ตัวเลขด้านล่างสมมติว่ามี core เพียงพอให้ N workers ทำงานขนานได้เต็มที่ ซึ่งเป็น assumption ที่ดีที่สุดเท่าที่ทำได้โดยไม่มีข้อมูล)*

ใช้ assumption กลาง (~0.8s/request):

| workers | เวลาโดยประมาณสำหรับ 50 คน (ต่อคิว) | RAM |
|---|---|---|
| **1** | 50 × 0.8s ≈ **~40s** (คนสุดท้ายรอคิวเกือบ 40 วินาทีก่อนแม้แต่จะเริ่ม process) | ~0.9 GB |
| **2** | ~25 requests/worker × 0.8s ≈ **~20s** | ~1.8 GB |
| **4** | ~12-13 requests/worker × 0.8s ≈ **~10s** | ~3.6 GB |

ถ้าใช้ assumption แย่กว่า (~1.5s/request, DB ช้า): 1 worker ≈ **75s**, 2 workers ≈ **38s**, 4 workers ≈ **19s**

### บทสรุป

จากตัวเลขที่วัดได้จริง ปัจจัยที่กระทบ capacity เรียงตามผลกระทบ:

1. **`--workers 1` + `sync` worker class คือคอขวดที่ใหญ่ที่สุดโดยไม่ต้องสงสัย** — ไม่ใช่ AI pipeline ช้า แต่เพราะระบบรับได้ครั้งละ 1 request เท่านั้น เพิ่ม worker เป็นตัวเดียวที่ทำให้เวลารวมลดลงเป็นสัดส่วนตรง (2 workers = ครึ่งเวลา, 4 workers = 1/4 เวลา) **ก่อนจะไปแตะ AI pipeline เลย**
2. **`DeepFace.represent()` (embed) ~350ms คือคอขวดที่ใหญ่ที่สุดในบรรดา AI compute เอง** — มากกว่า anti-spoof ทั้งก้อนเกือบ 4 เท่า (ดูข้อ 2 สำหรับแนวทางลด detection ซ้ำที่อาจช่วยตรงนี้ได้บางส่วน)
3. **RAM ต่อ worker ~906 MB เป็นข้อจำกัดตัวจริงว่าจะเพิ่ม worker ได้กี่ตัว** — ก่อนเพิ่ม `--workers` ต้องรู้ RAM limit ของ Railway plan ก่อน (ไม่ทราบ ต้องเช็คเอง) ถ้า plan มี RAM จำกัด เพิ่ม worker อาจทำให้ container ถูก OOM-kill แทนที่จะเร็วขึ้น
4. **DB round-trip time เป็นตัวแปรสำคัญที่วัดไม่ได้จาก environment นี้เลย** — ตัวเลข "50 คนใช้กี่วินาที" ทั้งหมดข้างต้นมี assumption ส่วนนี้กำกับอยู่ ถ้าต้องการตัวเลขที่เชื่อถือได้จริง ต้องรัน [PERF] log (ข้อ 1) กับ production/staging จริงที่ต่อ Supabase ได้ แล้วดู `session=`, `db=` ในบรรทัด log จริง

**ข้อเสนอสำหรับตัดสินใจ optimize (ไม่ได้แก้ตามนี้ ตามที่สั่งห้ามแตะ logic):** ถ้าต้องเลือกอย่างเดียวก่อน คุ้มสุดคือเพิ่ม `--workers` (ถ้า RAM/CPU ของ plan รองรับ) เพราะให้ผลเชิงเส้นทันทีโดยไม่ต้องแตะโค้ด AI เลย ส่วนการลด detection ซ้ำจาก 3→2 รอบ (ข้อ 2) เป็นงานระดับกลางที่ช่วยลด `embed` ได้บางส่วนแต่ต้อง re-validate threshold ใหม่

> **แก้ไขข้อเสนอด้านบน (พบเพิ่มระหว่างตอบข้อ 8 ด้านล่าง — แต่ข้อสรุปนี้ถูก supersede แล้ว ดูย่อหน้าถัดไป):** การเพิ่ม `--workers` เฉยๆ **ไม่ปลอดภัย 100%** อย่างที่เขียนไว้ข้างต้น — `create_app()` (`app/__init__.py:172-173`) เรียก `start_scheduler(app)` ทุกครั้งที่ worker process ถูกสร้าง (ไม่มี `--preload` ตอนนี้ แปลว่าแต่ละ worker import/call `create_app()` เองอิสระหลัง fork) และ `start_scheduler()` (`app/scheduler.py:138`) จะ `scheduler.start()` จริงทุกครั้งที่ `not app.debug` (True เสมอใน gunicorn) — เพิ่ม `--workers` เป็น N ตอนนี้ = APScheduler ทำงาน N ชุดพร้อมกัน แต่ละชุดยิง `auto_manage_sessions` ทุก 1 นาที และ `keep_alive` ทุก 3 นาที ซ้ำกัน N เท่า

> **อัปเดต (2026-08-25, ตรวจ `auto_manage_sessions`/`keep_alive` จริงแล้ว — F-9 ไม่ใช่ blocker):** ตรวจโค้ด `app/scheduler.py` ทั้งฟังก์ชันแล้ว การรัน N ชุดพร้อมกัน **ไม่ทำให้ข้อมูลพัง**:
> - **Auto-create** (`scheduler.py:81-99`): insert แล้ว catch `23505`/`duplicate`/`unique` แล้ว skip — มี UNIQUE constraint คุ้มครองจริงจาก F-10 (`sb.table("sessions").insert(...)` ในเงื่อนไข `if not existing and beacon_id_to_use`) → N ชุดชนกันแค่ 1 ชุดสร้างสำเร็จ ที่เหลือ insert fail แล้ว skip เงียบ ๆ
> - **Auto-close** (`scheduler.py:101-121`): `UPDATE sessions SET is_open=False WHERE is_open=True ...` — idempotent โดยธรรมชาติ ชุดที่ 2 เป็นต้นไป query `is_open=True` จะไม่เจอ row ที่ชุดแรกปิดไปแล้ว ไม่มีอะไรให้ทำซ้ำ
> - **`keep_alive`** (`scheduler.py:134-141`): read-only (`select("id").limit(1)`), ไม่มี state ให้พังอยู่แล้ว
>
> **สรุป: รัน N ชุดพร้อมกันเหลือแค่ wasted HTTP call ไป Supabase + log noise ซ้ำ N เท่า ไม่ใช่ correctness bug — เพิ่ม `--workers` ได้เลยโดยไม่ต้องรอ F-9** (F-9 ยังควรแก้อยู่ดีเพื่อลด wasted call/log noise แต่เป็นงานหลัง load test ไม่ใช่ blocker ก่อนหน้า — ทางที่เลือกไว้คือ `fcntl.flock` คุม `start_scheduler()` ไม่ให้รันซ้อนข้าม process ไม่ใช่ DB advisory lock)

---

## 7. FaceNet512 แปลงเป็น ONNX ได้มั้ย

**สรุปสั้น: ทำได้ในทางเทคนิค (Keras `Model` ธรรมดา ไม่มีอะไรแปลกพิเศษ) แต่มีจุดที่ต้องระวังจริงจัง 2 จุด: preprocessing ต้อง reimplement เองนอกโมเดล และต้อง validate embedding เทียบของเดิมอย่างเป็นระบบก่อนใช้จริง — ไม่ใช่งาน "แปลงแล้วจบ"**

### DeepFace เก็บ weights ไว้ที่ไหน (ยืนยันจาก path จริงในเครื่องนี้)

```
C:\Users\LENOVO\.deepface\weights\
  facenet512_weights.h5           90.6 MB   ← ตัวที่ใช้จริงใน face_service.py (model_name="Facenet512")
  facenet_weights.h5               87.9 MB   (FaceNet-128d, ไม่ได้ใช้ในโปรเจกต์นี้)
  retinaface.h5                   113.2 MB   (ไม่ได้ใช้ — detector_backend ที่ใช้จริงคือ "opencv")
  2.7_80x80_MiniFASNetV2.pth        1.8 MB   ← Fasnet antispoof (PyTorch, คนละโมเดลคนละ pipeline)
  4_0_0_80x80_MiniFASNetV1SE.pth    1.8 MB
```
บน Linux/Mac path เดียวกันคือ `~/.deepface/weights/` — โครงสร้างนี้ตรงกับ path ที่ Dockerfile ใช้ตอน pre-download (`RUN python - <<'EOF' ... DeepFace.build_model("Facenet512") ...`) เพื่อ bake ไว้ใน image เดียวกัน

**สถาปัตยกรรมโมเดล** (ยืนยันจาก `deepface/models/facial_recognition/Facenet.py` จริง): `FaceNet512dClient.model` เป็น **`tensorflow.keras.models.Model` ธรรมดา** (functional API, `InceptionResNetV1`, `input_shape=(160,160)`, output 512-D) — ไม่ใช่ subclassed model หรือ custom training loop ที่แปลงยาก แต่มี **Lambda layer 21 จุด** (`Lambda(scaling, arguments={"scale": 0.1-1.0})` — residual scaling ใน Inception-ResNet block) ทุกจุดใช้ฟังก์ชัน `scaling(x, scale) = x * scale` เดียวกัน (elementwise multiply ธรรมดา) — Lambda layer ประเภทนี้ tf2onnx แปลงได้ปกติ (ปัญหากับ Lambda มักเกิดตอน logic ข้างในไม่ใช่ TF op ล้วนๆ ซึ่งไม่ใช่กรณีนี้)

### เครื่องมือแปลง: `tf2onnx` ใช้ได้จริงมั้ย

`tf2onnx` **ไม่ได้ติดตั้งใน environment นี้แต่มีบน PyPI จริง** (เช็คแล้ว: เวอร์ชันล่าสุด `1.17.0`, ย้อนหลังถึง `0.3.1`) เป็นเครื่องมือมาตรฐานสำหรับแปลง TF/Keras SavedModel หรือ `.h5` → ONNX

**จุดที่เข้าทางโปรเจกต์นี้:** Dockerfile ตั้ง `ENV TF_USE_LEGACY_KERAS=1` ไว้แล้ว (comment ในไฟล์บอกตรงๆ ว่า "belt-and-suspenders even though tensorflow 2.15 defaults to Keras 2") — โมเดลที่โหลดจริงคือ Keras 2-style (legacy), ไม่ใช่ Keras 3 — `tf2onnx` รองรับ Keras 2 / TF 2.x SavedModel ได้ดีกว่ามาก เมื่อเทียบกับ Keras 3 (ที่ `tf2onnx` มีปัญหา compatibility ที่รู้จักกันเยอะ เพราะ dev ของ `tf2onnx` ช้ากว่าการอัปเดต Keras 3) — บังเอิญว่าการตั้งค่า legacy Keras ที่มีอยู่แล้ว (เพื่อเหตุผลอื่น) ทำให้เส้นทางแปลง ONNX เป็นไปได้ง่ายกว่าที่คิด

วิธีคร่าวๆ (ไม่ได้ลองจริงในรอบนี้ — ยังไม่ต้องทำตามที่สั่ง):
```
model = DeepFace.build_model("Facenet512").model   # ดึง keras.Model ออกมา
model.save("facenet512_saved_model")                 # export เป็น SavedModel format
python -m tf2onnx.convert --saved-model facenet512_saved_model --output facenet512.onnx
```

### Preprocessing ที่ต้อง reimplement เอง (นี่คือจุดที่พลาดง่ายที่สุด)

ONNX เก็บแค่ **กราฟของโมเดล** (จาก input tensor `(1,160,160,3)` → output embedding 512-D) — ทุกอย่างที่ DeepFace ทำ **ก่อน** เข้าโมเดลไม่ได้ติดไปด้วย ต้องเขียนเองแยกนอก ONNX runtime ยืนยันจาก source จริง (`deepface/modules/preprocessing.py`, `deepface/modules/representation.py`) ลำดับที่ DeepFace ทำจริงคือ:

1. **Detect + align** (`detection.extract_faces`, `detector_backend="opencv"`, `align=True` default) — หา facial area ด้วย detector แล้วหมุนภาพตามตำแหน่งตา (eye-based rotation) — **ส่วนนี้ยังต้องมี detector อยู่ดี ONNX ไม่ได้ตัดขั้นตอนนี้ออก** เพียงแต่แทนที่ตัว embedding model เท่านั้น
2. **`resize_image()`** (`preprocessing.py`): resize แบบรักษา aspect ratio ให้พอดีกับ `target_size=(160,160)` แล้ว **pad ด้วย pixel ดำ** ให้ครบขนาด (ไม่ใช่ resize ธรรมดาแบบยืด/บีบภาพ) จากนั้น scale ค่า pixel ให้อยู่ช่วง `[0,1]` (หาร 255)
3. **`normalize_input(img, normalization="base")`** — เนื่องจาก `extract_embedding()` ใน `face_service.py:458` เรียก `DeepFace.represent(...)` **โดยไม่ส่ง `normalization` param** ค่า default คือ `"base"` ซึ่งจาก source จริง (`preprocessing.py`) แปลว่า **ไม่ทำอะไรเพิ่มเลย** (return ภาพที่ resize+scale แล้วตรงๆ ไม่มี mean/std normalization ต่อ) — เข้าใจผิดง่ายว่าโมเดลนี้ต้อง normalize แบบ Facenet-specific (mean/std) แต่จริงๆ ไม่ได้ทำในโค้ดปัจจุบัน

**สรุป preprocessing ที่ต้องเขียนเองถ้าย้ายไป ONNX runtime:** detect+align (ยังต้องใช้ opencv/Haar หรือ detector อื่นเหมือนเดิม) → resize-with-pad ไป 160×160 → หาร 255 → feed เข้า ONNX session ตรงๆ (ไม่มี mean/std ต้องลบ) — เขียนใหม่ได้ไม่ยาก เพราะ logic สั้น (`resize_image` มีแค่ ~20 บรรทัด) แต่ **ถ้าลืม replicate ขั้นตอนไหนแม้แต่ขั้นตอนเดียว (เช่น ลืม pad แล้ว resize ยืดภาพตรงๆ แทน) embedding จะเพี้ยนทันทีแบบเงียบๆ ไม่มี error ให้เห็น**

### ความเสี่ยง: embedding จะตรงกับของเดิมมั้ย + วิธีเทียบให้มั่นใจ

**ความเสี่ยงจริง (ไม่ใช่แค่ทฤษฎี):**
- Op-level numerical difference ระหว่าง TensorFlow runtime กับ ONNXRuntime — ปกติต่างกันในระดับ floating-point เล็กน้อย (1e-5 ~ 1e-6) จาก op ที่ implement ต่างกัน (เช่น BatchNorm epsilon handling, padding convention ของ Conv2D "same") — ระดับนี้ไม่กระทบ cosine similarity ในทางปฏิบัติ
- **แต่** ถ้า preprocessing (ข้อบน) reimplement ผิดแม้เพียงจุดเดียว (ลืม pad, ลืมหาร 255, resize method ต่างจาก `cv2.resize` default interpolation) — embedding จะเบี่ยงมากกว่านั้นมาก และจะไม่มี exception ใดๆ บอกเลย (โมเดลรับ input ผิด shape ที่ resize มาแล้วยังคำนวณต่อได้ปกติ แค่ผลลัพธ์ผิด) — **นี่คือความเสี่ยงหลักจริงๆ ไม่ใช่ตัว conversion เอง**
- Threshold ที่ calibrate ไว้ (`SAME_DEVICE_THRESHOLD=0.70`, `NEW_DEVICE_THRESHOLD=0.80`, `DUPLICATE_THRESHOLD=0.65`, ฯลฯ ใน `face_service.py`) ผูกกับค่า embedding จาก TF path เดิม — ถ้า ONNX path ให้ embedding ที่เบี่ยงแม้เล็กน้อยอย่างเป็นระบบ (systematic bias ไม่ใช่ noise) threshold เดิมอาจไม่ valid อีกต่อไป

**วิธีเทียบให้มั่นใจก่อนใช้จริง (ขั้นตอนมาตรฐานสำหรับ model-conversion validation):**
1. เตรียมชุดภาพทดสอบจริง (ไม่ใช่ synthetic) อย่างน้อยหลักสิบ-ร้อยภาพ ครอบคลุม lighting/pose หลากหลาย — ควรใช้ภาพจริงจาก enrollment flow เดิม (ถ้ามี consent เก็บไว้เพื่อ QA — ต้องเคารพ PDPA เรื่องนี้ด้วย)
2. รันทั้งสอง path (TF ของเดิม vs ONNX ใหม่) บนภาพชุดเดียวกัน **หลัง preprocessing เดียวกันเป๊ะ**
3. เทียบ **cosine similarity ระหว่าง embedding_TF กับ embedding_ONNX ของภาพเดียวกัน** — ถ้า conversion ถูกต้อง ค่านี้ควรอยู่ที่ **>0.999** แทบทุกภาพ (ใกล้ 1.0 มาก เพราะเป็นโมเดลเดียวกัน input เดียวกัน ต่างแค่ runtime) — ถ้าต่ำกว่านั้นชัดเจน (เช่น 0.95, 0.90) แปลว่า preprocessing หรือ conversion มีจุดผิดต้องไล่หา ไม่ใช่แค่ noise ปกติ
4. เทียบผลการตัดสินใจปลายทาง ไม่ใช่แค่ embedding ดิบ — รัน `verify_face_multi()` แบบเดิมด้วย embedding จากทั้งสอง path เทียบกับ dataset จำลอง (คนเดียวกัน vs คนละคน) แล้วดูว่า **decision (`verified: True/False`) เปลี่ยนไปกี่ % ของเคส** ที่ threshold เดิม — เป้าหมายคือ 0% หรือใกล้ 0% ก่อนจะกล้าเปลี่ยน production
5. ถ้าผ่านข้อ 3-4 ค่อยพิจารณาว่าต้อง re-calibrate threshold ใหม่หรือไม่ (ถ้า bias เป็นระบบแต่คงที่ อาจแค่ shift threshold แทนที่จะ debug conversion)

**ประเมินภาพรวม:** งานระดับ "ปานกลาง-สูง" ไม่ใช่งานเร็ว — ตัว conversion เองอาจใช้เวลาไม่กี่ชั่วโมง แต่ validation loop (ข้อ 3-4) ที่ทำให้มั่นใจว่า production ใช้ได้จริงคือส่วนที่กินเวลาและต้องมีชุดภาพทดสอบที่เป็นตัวแทนเพียงพอ — **ยังไม่ได้ประเมิน speedup ที่จะได้จริง** (ONNXRuntime มักเร็วกว่า TF สำหรับ inference-only บน CPU แต่ไม่ได้วัดเทียบกันในรอบนี้ — ควรวัดคู่กับ validation ข้างต้นไปพร้อมกัน)

---

## 8. `--preload` ใน gunicorn ใช้กับ setup นี้ได้มั้ย

**สรุปสั้น: ไม่แนะนำให้เปิดตอนนี้ — เสี่ยง 2 เรื่องที่แยกจากกัน (TF+fork และ scheduler+fork) และไม่ได้ประโยชน์อัตโนมัติอย่างที่คาดด้วยซ้ำถ้าไม่แก้โค้ดเพิ่ม**

### `--preload` ทำอะไรจริงๆ (และทำไมโมเดล lazy-load ไม่ได้ผลอัตโนมัติ)

`--preload` สั่งให้ gunicorn **master process** import แอป (เรียก `app:create_app()`) **ครั้งเดียวก่อน fork** worker ทั้งหมด แทนที่ปกติ (ไม่มี `--preload`, ปัจจุบันของโปรเจกต์นี้) ที่แต่ละ worker import/call `create_app()` **เองอิสระหลัง fork** — ประโยชน์ปกติของ `--preload` คือ: โค้ด/memory ที่ import ไว้ก่อน fork จะถูก **แชร์แบบ copy-on-write** ข้าม worker processes (ประหยัด RAM) และ error ตอน import จะ fail เร็วตั้งแต่ master ไม่ต้องรอ worker ตายทีละตัว

**แต่ — ตอบคำถามแรกตรงๆ: ใช่ ต้องเปลี่ยนให้โหลดตอน import แทน ถึงจะได้ประโยชน์จริง** เพราะโมเดลทั้งหมดใน `face_service.py` เป็น **lazy load** (`_antispoof_session = None` module-level, สร้างจริงตอนถูกเรียกครั้งแรก — ยืนยันจาก source ที่อ่านไปก่อนหน้านี้ในไฟล์นี้) `create_app()` และการ import blueprint ทั้งหมดไม่ได้แตะ `face_service`'s lazy globals เลย — ถ้าเปิด `--preload` เฉยๆ โดยไม่แก้โค้ดเพิ่ม **models จะยังไม่ถูกโหลดตอน master import อยู่ดี** (ยังคง lazy load ตอน worker เจอ request แรกเหมือนเดิมทุกประการ) `--preload` เปล่าๆ จึงแทบไม่ได้ประโยชน์อะไรสำหรับปัญหา cold-start ที่วัดได้ในข้อ 3 (~20 วินาทีสำหรับ TF graph + Facenet512) — ต้องเพิ่มโค้ด warm-up เรียก `combined_spoof_score()`/`extract_embedding()` ครั้งหนึ่งใน `create_app()` ด้วย ถึงจะมีอะไรให้ preload จริง

### TensorFlow กับ `fork()` — ปัญหาที่รู้จักกันจริง (ระดับความมั่นใจ: สูง เป็น pattern ที่รู้จักกันกว้างในวงการ ML deployment แต่ไม่ได้ทดสอบ fork() จริงในรอบนี้เพราะ sandbox นี้เป็น Windows ซึ่งไม่มี `fork()` เลย — เทียบไม่ได้กับ production ที่เป็น Linux ใน Dockerfile)

ถ้า **มีการ trigger ให้ TensorFlow รัน op จริงอย่างน้อย 1 ครั้งก่อน fork** (เช่น เพิ่ม warm-up call ตามข้อบน) TensorFlow's C++ runtime จะสร้าง internal thread pool (Eigen threadpool executor) ขึ้นมาในตอนนั้น — เมื่อ `fork()` เกิดขึ้นหลังจากนั้น (Linux `os.fork()` ที่ gunicorn master ใช้สร้าง worker) **child process จะได้แค่ thread ที่เรียก fork() เท่านั้น** thread อื่นๆ ของ TF thread pool **ไม่ถูกก็อปปี้ไปด้วย** (พฤติกรรมมาตรฐานของ POSIX `fork()` ไม่ใช่บั๊กเฉพาะ TF) — ถ้า thread ที่หายไปนั้นถือ mutex/lock ค้างอยู่ ณ ขณะ fork พอดี child process ที่พยายามใช้ TF ครั้งแรกอาจ **hang หรือ deadlock** โดยไม่มี error message ชัดเจน — เป็นปัญหาที่รู้จักกันกว้างขวางในวงการ deploy ML model ด้วย gunicorn/uWSGI (`fork()` + framework ที่ pre-spawn threads เช่น TF/PyTorch/gRPC) จนเป็นเหตุผลมาตรฐานที่แนะนำกันว่า **"อย่า initialize deep learning framework ก่อน fork"**

### จุดเสี่ยงที่ 2 ที่เจอเพิ่มระหว่างตรวจ (เฉพาะโปรเจกต์นี้ ไม่ใช่ TF ทั่วไป): scheduler thread ก็เจอปัญหาแบบเดียวกัน

`create_app()` (`app/__init__.py:172-173`) เรียก `start_scheduler(app)` ซึ่ง **สร้าง background thread จริง** (`BackgroundScheduler` จาก `apscheduler`, `app/scheduler.py:134-140`) — ถ้า `--preload` ทำให้ `create_app()` รันใน master ก่อน fork เหมือนกัน scheduler thread นั้นจะถูกสร้างใน master **ก่อน** fork ด้วย ผลที่เกิดได้ 2 แบบขึ้นกับ implementation detail ของ `apscheduler`/OS:
- Thread ของ scheduler ไม่รอดจาก fork (เหมือนปัญหา TF ข้างบน) → **ไม่มี worker ไหนรัน scheduler จริงเลย** หลัง fork — `auto_manage_sessions`/`keep_alive` หยุดทำงานเงียบๆ ไม่มี error
- หรือถ้า apscheduler มีกลไก detect fork บางส่วน อาจพฤติกรรมไม่แน่นอนระหว่าง worker

**นี่คือเหตุผลที่ต้องบอกว่า "เสี่ยง" ไม่ใช่แค่ TF อย่างเดียว** — ถ้าจะใช้ `--preload` ต้องแก้ทั้งสองจุดพร้อมกัน (ทั้งโมเดล warm-up และ scheduler) ไม่ใช่แค่เพิ่ม flag เฉยๆ

### สรุปความเสี่ยงเป็นข้อๆ

1. **`--preload` เปล่าๆ ไม่มี code เปลี่ยน = ไม่ได้ประโยชน์อะไรเลย** (โมเดลยัง lazy load เหมือนเดิม, scheduler ยังสร้างหลัง fork เหมือนเดิมเพราะยังไม่มีอะไรให้ preload จริง) — แทบไม่มีความเสี่ยงแต่ก็ไม่มีประโยชน์ ไม่คุ้มจะเปิด
2. **ถ้าเพิ่ม warm-up code ให้ preload มีความหมายจริง** → เปิดความเสี่ยง TF+fork deadlock (เสี่ยงสูง เพราะ Dockerfile บอกชัดว่า production คือ Linux ที่มี `fork()` จริง ต่างจาก sandbox นี้ที่ทดสอบไม่ได้เลย) — ต้อง test บน Linux container จริงก่อนใช้ ห้าม deploy ตรงๆ โดยไม่ทดสอบ
3. **scheduler ต้องย้ายไปเรียกใน gunicorn `post_fork` hook แทน ไม่ใช่ใน `create_app()`** ถ้าจะใช้ `--preload` — ไม่งั้น auto check-in session management จะหยุดทำงานเงียบๆ (ผลกระทบร้ายแรงกว่า RAM/เวลาที่ประหยัดได้จาก preload มาก เพราะกระทบ business logic ตรง ไม่ใช่แค่ performance)
4. **ทางเลือกที่ปลอดภัยกว่าถ้าอยากลด cold-start โดยไม่แตะ `--preload` เลย:** ใช้ gunicorn `post_fork(server, worker)` hook ใน `gunicorn.conf.py` เรียก warm-up **หลัง** fork แทน (แต่ละ worker load โมเดลเองหลัง fork เหมือนเดิมทุกประการ เพียงแต่ trigger ทันทีตอน worker เริ่มแทนที่จะรอ request แรกจาก user) — วิธีนี้หลีกเลี่ยงทั้งปัญหา TF+fork (โหลดหลัง fork เหมือนเดิม ไม่มี thread หายไปไหน) และปัญหา scheduler (ไม่ต้องย้ายอะไร) ได้พร้อมกัน — **เป็นข้อเสนอ ไม่ใช่การแก้ ยังไม่ได้ implement ตามที่สั่งห้ามแตะ logic**

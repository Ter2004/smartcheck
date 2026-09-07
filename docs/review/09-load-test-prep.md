# Load-test prep: pipeline order ยืนยัน + spec ภาพทดสอบ

วันที่: 2026-08-24 — เตรียมความพร้อมก่อนรัน `scripts/load_test.py` จริงบน Railway
(ยังไม่ได้รัน — รอยืนยัน deploy พร้อม)

---

## 1. Pipeline order ของ `/api/checkin` (ยืนยันจากโค้ดจริง `api_checkin.py` ปัจจุบัน — ไม่ใช่จาก `01-api-checkin.md` เพราะ line number ขยับไปแล้วหลังแก้ F-2/F-4)

| # | ขั้นตอน | บรรทัด | ตกแล้ว... |
|---|---|---|---|
| 1 | Input whitelist (`liveness_action`, `ble_rssi`, required fields) | 42-63 | **จบเลย** (400) |
| 2 | Device token HMAC verify | 65-77 | **จบเลย** ถ้า token invalid/uid ไม่ตรง (403) — ไม่มี token เลย = ไปต่อ (fail-open) |
| 3 | Session open check + course enrollment | 79-107 | **จบเลย** (404/403) |
| 4 | Zero-trust frame validation (`server_validate_frame`) | 109-117 | **จบเลย** (400) |
| 5 | Check-in window (`checkin_duration`) | 119-126 | **จบเลย** ถ้าเกินเวลา (400) |
| 6 | BLE RSSI | 129-140 | **จบเลย** เฉพาะตอน `BLE_CHECK_ENABLED=True` (default False → ข้ามเสมอ) |
| 7 | EAR liveness | 142-170 | **จบเลย** ถ้า `liveness_action=="blink"` และ EAR ไม่ผ่าน; action อื่นแค่เช็ค format |
| 8 | **Moiré FFT** | 172-191 | **จบเลย** ถ้าเจอ screen (400, `"spoof": true`) |
| 9 | **Screen Texture FFT** | 193-211 | **จบเลย** ถ้าเจอ screen (400, `"spoof": true`) |
| 10 | **Temporal Variance** (ต้องมี `face_images` >= 2 เฟรม) | 213-253 | **จบเลย** ถ้าเฟรมเหมือนกัน/นิ่ง (400, `"spoof": true`) |
| 11 | **MiniFASNet anti-spoof** (`combined_spoof_score`) | 255-277 | **จบเลย** ถ้า `is_real=False` (400, `"spoof": true`) |
| 12 | Device binding (เลือก threshold) | 279-311 | **จบเลย** เฉพาะ device_id ชนกับคนอื่น (403); ไม่มี device_id = ไปต่อ (fail-open) |
| 13 | Embedding integrity (`verify_embedding_integrity`) | 313-338 | **จบเลย** (403) — **นี่คือจุดที่ seed script ต้องได้ salt ถูกต้อง** |
| 14 | **`extract_embedding()` = `DeepFace.represent(model_name="Facenet512")` (~350ms)** | 340-346 | **จบเลย** ถ้า decode/detect หน้าไม่เจอ (400) |
| 15 | `verify_face_multi` (cosine similarity) | 348-359 | **จบเลย** ถ้าไม่ตรง threshold (400) — **นี่คือผลลัพธ์ที่คาดหวังจาก fake embeddings ของเรา** |
| 16 | Duplicate check-in guard | 361-371 | **จบเลย** ถ้าเช็คแล้ว (400) |
| 17 | Status calc (present/late) | 373-380 | ไม่ fail |
| 18 | DB insert (+ TOCTOU guard บน UNIQUE constraint) | 382-407 | จบด้วย 200 หรือ 400/500 |

## คำตอบคำถามหลัก: ถ้าเป็น spoof จะยัง DeepFace ไหม

**ไม่ — return ก่อนเสมอ** ขั้นที่ 8-11 (Moiré, Screen Texture, Temporal Variance, MiniFASNet) **ทั้งหมดอยู่ก่อน**ขั้นที่ 14 (`extract_embedding`) และทุกขั้นเป็น fail-close (`return` ทันทีถ้าตัดสินว่าเป็น spoof) — ยืนยันจากอ่านโค้ดตรง ๆ ไม่ใช่เดา

**สรุปสำหรับภาพทดสอบ:**
- ต้องเป็นภาพถ่ายจริงจากกล้อง (ไม่ใช่ภาพจากจอ/สิ่งพิมพ์) — ไม่งั้นโดนขั้น 8/9/11 ตัดก่อน
- ต้องมี `face_images` >= 2 เฟรมที่ **แตกต่างกันจริง** (ขยับหน้าเล็กน้อยระหว่างถ่ายแบบ burst shot) — ถ้าใช้ไฟล์เดียวกันซ้ำ ขั้น 10 (Temporal Variance) จะเห็น std=0 แล้วตัดสินว่า "ภาพนิ่ง" (spoof=True) ทั้งที่เป็นภาพจริง 100% — **กับดักที่ไม่ชัดเจนที่สุด**
- ผลลัพธ์ที่ "สำเร็จ" ของ load test ไม่ใช่ HTTP 200 — เพราะ `student_biometrics.face_embeddings` เป็นเลขสุ่ม (จาก `seed_load_test.py`) ทุก request ที่ผ่าน pipeline เต็ม (รวม DeepFace) จะจบที่ **400 "ใบหน้าไม่ตรง"** (ขั้น 15) เสมอ — นี่คือผลลัพธ์ที่ถูกต้องแล้ว ไม่ใช่ bug ของ test

## 2. Spec ภาพทดสอบ

จาก `checkin_flow.js:354-371` (`_submitCheckin`) payload ที่ client จริงส่ง:
```js
fetch('/api/checkin', {
    method: 'POST',
    headers: {
        'Content-Type':  'application/json',
        'X-CSRF-Token':  <จาก meta[name="csrf-token"]>,
        'Authorization': `DeviceToken ${localStorage.getItem('sc_device_token') || ''}`,
    },
    body: JSON.stringify({
        session_id, ble_rssi, ble_skip, liveness_action, liveness_pass: true,
        face_image:  <base64 JPEG เฟรมล่าสุด, จาก canvas.toDataURL('image/jpeg', 0.85)>,
        face_images: <array base64 JPEG หลายเฟรม>,
        ear_samples: <array float>,
    }),
});
```

**Format ภาพ:** base64 data-URL (`data:image/jpeg;base64,...`) — server รองรับทั้งมี/ไม่มี prefix (`_decode_image`/`server_validate_frame` เช็ค `"," in b64` แล้ว strip เอง)

**ข้อกำหนดของ `server_validate_frame` (face_service.py:744-823) — ต้องผ่านทุกข้อ:**
| เงื่อนไข | ค่า |
|---|---|
| Format | JPEG เท่านั้น (magic bytes `FF D8 FF ... FF D9`) |
| ขนาดไฟล์ | 3 KB – 500 KB |
| ความละเอียด | 160×120 – 1920×1080 |
| Laplacian blur variance | >= 8 |
| สี channel std-dev (แต่ละ B/G/R) | >= 2.0 |

**`load_test.py` เช็คเงื่อนไขเหล่านี้ให้อัตโนมัติ** (`_load_images()`) — เตือนถ้าไฟล์นอกช่วงขนาด, error ถ้าไม่ใช่ JPEG, และ **เตือนถ้า 2 ไฟล์เหมือนกันไบต์ต่อไบต์** (กับดัก Temporal Variance ข้างต้น)

**สิ่งที่ต้องเตรียม:** ภาพถ่ายจริง (เช่นจากมือถือ/เว็บแคม) อย่างน้อย **2-3 ไฟล์** เป็น burst shot ของหน้าคนจริงคนเดียวกัน (ขยับหน้า/กะพริบตาเล็กน้อยระหว่างช็อต) — ไม่ต้องเป็นใบหน้าที่ตรงกับใครใน DB เพราะ embedding ที่เก็บไว้เป็นเลขสุ่มอยู่แล้ว ใช้ภาพชุดเดียวกันกับทั้ง 50 คนจำลองได้ (แต่ละ request เป็นอิสระต่อกัน)

---

## เกี่ยวกับ `scripts/load_test.py`

ดู docstring บนสุดของไฟล์สำหรับ usage เต็ม — สรุปสั้น:

- `login` — login 50 คนทีละคน ≤8/min, backoff+retry บน 429, เก็บ cookies+csrf token ไว้ที่ `scripts/load_test_state.json`
- `checkin` — ยิง `/api/checkin` พร้อมกันทั้ง 50 request (ThreadPoolExecutor), วัด p50/p95/p99 + แยก error ตามชนิด, บันทึกผลเป็น `scripts/load_test_results/<label>_<timestamp>.json`
- `all` — ทำทั้งสองเฟสรวด
- `compare` — เทียบผลหลายไฟล์ (เช่น 1 worker vs 4 workers) เป็นตารางเดียว

Session id ของ session ที่เปิดอยู่หาอัตโนมัติจาก Supabase (course `LOADTEST101`) หรือระบุเองผ่าน `--session-id`

**สถานะ:** เขียนเสร็จ, syntax ผ่าน (`py_compile`), `--help` ทำงานถูกต้อง — **ยังไม่ได้รันจริงกับ server** ตามที่สั่ง รอยืนยัน Railway deploy พร้อมก่อน

---

## 3. Runbook: วัดผล 5 configs (workers x TF thread capping)

เป้าหมาย: ยืนยันว่า cap `TF_NUM_INTRAOP_THREADS`/`TF_NUM_INTEROP_THREADS` ช่วยจริงไหมตอน `--workers` > 1 (สงสัยว่า TF ใช้ thread เท่าจำนวน core โดย default ต่อ worker → N workers ชนกันเอง) — **5 configs:** workers=1 default, workers=2 default, workers=2 capped, workers=4 default, workers=4 capped (ตัด "workers=1 capped" ออกเพราะ worker เดียวไม่มี oversubscription ให้ cap)

รวมเวลาประมาณ 15-20 นาที (5 configs x redeploy+warmup+burst) — **ต้องทำ 4 ขั้นตอนนี้เป๊ะเหมือนกันทุก config** ไม่งั้นตัวแปรที่เปลี่ยนจะไม่ใช่แค่ thread setting อย่างเดียว เทียบผลกันไม่ได้

### ขั้น 1 — ตั้งค่าบน Railway → redeploy

- ตั้งจำนวน `--workers` ของ config นี้ (ผ่านค่าที่ deploy อยู่จริง)
- ตั้ง `TF_NUM_INTRAOP_THREADS` / `TF_NUM_INTEROP_THREADS` **เฉพาะ config ที่ "capped"** ผ่าน **Railway dashboard → Variables** (ไม่ใช่แก้ Dockerfile — redeploy จาก env var เปลี่ยนเร็วกว่ามาก ไม่ rebuild image ทั้งก้อนที่มี torch/tensorflow install อยู่)
- รอสถานะ deploy เป็น **Active** ก่อนไปขั้นถัดไปเสมอ — อย่ายิงระหว่าง building/restarting เพราะ response ช่วงนั้นไม่สะท้อน config จริง

### ขั้น 2 — Warmup

```
python scripts/load_test.py warmup --workers N --images photo1.jpg photo2.jpg
```
(`--count` default = 3×N ปรับเพิ่มได้ถ้ายังไม่ warm ครบ)

**เช็คว่า warm ครบทุก worker จริง — ใช้ 2 สัญญาณคู่กัน อย่าเชื่ออย่างใดอย่างหนึ่งเพียงอย่างเดียว:**

1. Elapsed ของ 3 request สุดท้ายจาก `warmup` ใกล้เคียงกัน (สคริปต์เตือนอัตโนมัติถ้าต่างเกิน 2 เท่า) — **สัญญาณนี้ไม่สมบูรณ์:** request ยิงทีละตัวเรียงกัน ไม่รับประกันว่ากระจายไปครบทุก worker เพราะ gunicorn sync worker แข่งกันด้วย accept-lock ก่อนรับ request แต่ละครั้ง worker ที่ warm แล้ว (ตอบเร็วกว่า) มีแนวโน้มกลับไปรอ lock รอบใหม่ได้เร็วกว่า worker ที่ยังโหลดโมเดลอยู่ (ช้ากว่า) จึงชนะ lock ซ้ำ ๆ ได้ — อาจเห็น "last 3 ใกล้กัน" ทั้งที่จริงมี worker เย็นเหลืออยู่ที่ไม่เคยถูกเรียกเลยตลอด warmup
2. **RAM plateau บน Railway → service ที่ deploy อยู่ → แท็บ Metrics (ground truth ตัวจริง):** เฝ้าดู RAM usage จนนิ่งที่ระดับ **~906MB × N** (เช่น N=4 → รอจนถึง ~3.6GB — เลข 906MB มาจาก sandbox ไม่ใช่ production วัดจริง ดู `06-performance.md:7-18`, ถ้าตัวเลขจริงต่างจากนี้มากให้บันทึกของจริงไว้แทน) ถ้า RAM ยังไม่ถึงระดับนี้ = ยังมี worker อย่างน้อย 1 ตัวไม่ได้โหลดโมเดล ยิง warmup เพิ่ม (`--count` สูงขึ้น) จนกว่า RAM จะนิ่งก่อนไปขั้น 3

**อย่าเริ่ม burst 50 คนจนกว่า RAM จะ plateau ที่ ~906MB × N จริง** — นี่คือเงื่อนไขผ่านของขั้นนี้ ไม่ใช่แค่ elapsed variance ต่ำ

### ขั้น 3 — Burst 50 คน

```
python scripts/load_test.py checkin --images photo1.jpg photo2.jpg --label cfgN
```
(ใช้ session state จาก `login` ที่ทำไว้รอบแรกได้เลย ไม่ต้อง login ใหม่ถ้า session ยังไม่หมดอายุ)

### ขั้น 4 — บันทึกผล

ต่อ 1 config เก็บอย่างน้อย:

- p50/p95/p99 + wall time (จาก output `checkin` มาตรฐาน — บันทึกอัตโนมัติที่ `scripts/load_test_results/<label>_<timestamp>.json` อยู่แล้ว)
- **RAM ที่ plateau จริงตอน warm** (จาก Railway metrics ขั้น 2) — เทียบกับ N × 906MB ที่คาดไว้ บันทึกตัวเลขจริงถ้าต่างเยอะ
- **CPU usage ช่วง burst** (จาก Railway metrics) — ถ้าแตะเพดาน ~100% ตลอดช่วง burst แปลว่า CPU เป็นคอขวดจริงตามสมมติฐาน ถ้าไม่ถึง แปลว่ามีคอขวดอื่นปน (DB round-trip, lock contention) ต้องแยกวิเคราะห์เพิ่ม
- categories breakdown (`spoof_reject`/`no_match`/`rate_limited`/`timeout` ฯลฯ จาก output `checkin`) — ถ้า config ไหนมี `rate_limited`/`timeout` โผล่ผิดปกติเทียบกับ config อื่น ให้สงสัยว่า deploy ยังไม่ warm จริง ไม่ใช่ผลของ thread setting

---

### คำถามที่เกี่ยวกับ warmup: มี endpoint เบากว่า checkin ที่บังคับโหลดโมเดลได้ไหม

ตรวจโค้ดจริงแล้ว — **ไม่มี**:

- **`/health`** (`app/__init__.py:176-178`) — return `"ok", 200` เฉย ๆ ไม่แตะ DeepFace/cv2/onnxruntime เลย ใช้ warm อะไรไม่ได้
- **`/api/antispoof-passive`** (`api_checkin.py:422-439`, เบากว่า `/api/checkin` จริง) — เรียก `check_anti_spoof_with_score` → `combined_spoof_score` ซึ่ง warm ได้แค่ Fasnet/Moiré/Texture/Temporal เท่านั้น **ไม่เรียก `extract_embedding()`/`DeepFace.represent(model_name="Facenet512")` เลย** — ตัวที่กิน AI compute มากที่สุด (~350ms จาก ~460ms รวม, `06-performance.md:183`) จะยังเย็นอยู่ ใช้ endpoint นี้ warm อย่างเดียวไม่พอ ต้องยิง `/api/checkin` เต็ม pipeline เท่านั้นถึงจะ warm ครบทุกโมเดล

**เรื่อง "already_checked" path:** ไล่โค้ด `api_checkin.py` แล้ว — duplicate check-in guard (ขั้น 16, บรรทัด 361-371) มาทีหลัง `extract_embedding()`/`verify_face_multi` (ขั้น 14-15, บรรทัด 340-359) เสมอ และ embedding ที่ seed ไว้เป็นเลขสุ่ม (`seed_load_test.py`) ทำให้ `verify_result["verified"]` เป็น `False` แทบทุกครั้ง → request จบที่ **400 "ใบหน้าไม่ตรง" ก่อนถึงขั้น duplicate check เสมอ** ไม่เคย insert attendance จริง **สรุป: warmup ด้วย checkin ซ้ำหลายรอบไม่มีทาง "burn attendance row" และทุกรอบยังคงวิ่งผ่าน DeepFace เต็ม pipeline จริง (ไม่ได้ short-circuit ที่ already_checked) — ใช้ warm ได้ตามที่ตั้งใจ**

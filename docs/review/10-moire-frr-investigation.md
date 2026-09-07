# Investigation: Moiré FFT layer ปฏิเสธภาพใบหน้าจริง (FRR)

วันที่: 2026-08-25
Reviewer: Claude Sonnet 5
Scope: `detect_screen_moire()` เท่านั้น (face_service.py) + จุดเรียกใน `api_checkin.py`
Trigger: ผู้ใช้รายงานว่า selfie จริงจากมือถือ (ไม่เคยถ่ายจากหน้าจอ) โดน reject ที่ `/api/checkin` ด้วยข้อความ "ตรวจพบหน้าจอมือถือ — กรุณาใช้ใบหน้าจริงเท่านั้น"
เครื่องมือ: `scripts/debug_moire.py` (standalone, import `face_service.py` ตรง ๆ, ไม่แก้ `app/`)

---

## 1. กลไกของ detector

`detect_screen_moire()` — face_service.py:657–703

- resize เป็น grayscale 256×256 คงที่ (face_service.py:675)
- FFT2 + fftshift → magnitude spectrum (676–679)
- แบ่งเป็น "low-frequency" = วงกลางขนาด `low_r = min(h,w) * 0.10` รอบจุดศูนย์กลางทั้งสองแกน (คือ **แถบกลาง 20% ของสเปกตรัมเท่านั้น**, บรรทัด 683)
- `score = high_energy / total_energy` โดย `high_energy = total_energy - low_energy` (688–692) — **ทุกอย่างนอกวงกลาง 20% ถือเป็น "high-frequency" หมด**
- คำนวณจาก **ทั้งเฟรม (full frame)** ไม่ใช่ face-crop — เรียกด้วย `[raw_frame]` ที่ decode ตรงจาก payload โดยไม่มีการ crop ใบหน้าเลย ทั้งที่ `api_checkin.py:174-175` และภายใน `combined_spoof_score` (face_service.py:190) ต่างจาก `detect_static_image` (temporal layer) ที่ crop ด้วย Haar cascade ก่อน (face_service.py:838+)
- threshold: `MOIRE_THRESHOLD_SINGLE = 0.70` (face_service.py:18, ใช้กับ check-in เฟรมเดียว), `MOIRE_THRESHOLD = 0.60` (บรรทัด 17, multi-frame `/api/enroll`)

**Provenance ของ threshold** (`git log -L`):

| commit | ค่า | หมายเหตุ |
|---|---|---|
| fe71b22 (11 เม.ย.) | เพิ่ม `MOIRE_THRESHOLD=0.60` | ค่าตั้งต้น ไม่มี single-frame แยก |
| d1c150d (15 เม.ย.) | เพิ่ม `MOIRE_THRESHOLD_SINGLE=0.72` | แยก single-frame ออกจาก multi-frame |
| 689a173 (19 เม.ย.) | 0.72 → 0.55 | "high-DPI phone screens ให้คะแนน 0.50–0.58" |
| ce78184 (19 เม.ย.) | 0.55 → 0.65 | "real faces 0.40-0.55, phone screens 0.55-0.75" |
| 2d39cd6 (20 เม.ย.) | 0.65 → 0.70 | "fix: adjust Moiré threshold to prevent false rejects on iPhone" |

**ตั้งแต่ commit แรก (`fe71b22`) จนถึงปัจจุบัน `low_r = 0.10` (นิยามของ "low frequency") ไม่เคยถูกแก้เลยแม้แต่ครั้งเดียว** — ทุกรอบที่มีคน false-reject บ่น มีแต่การขยับตัวเลข threshold บนสุด (5 ครั้งใน git log) ไม่มีรอบไหนตั้งคำถามกับตัว metric เอง

---

## 2. ผลวัด — `scripts/debug_moire.py` รันกับ `test_images/*.jpg` ทั้งหมด

`fasnet_*` เป็น `None` ในรอบวัดแรก เพราะ venv นี้ยังไม่ได้ติดตั้ง `torch` (DeepFace Fasnet ต้องการ) — **อัปเดต 2026-08-25 รอบสอง:** ติดตั้ง `torch==2.2.2` (CPU, PyTorch index เดียวกับที่ Dockerfile pin ไว้) แล้ว ไม่ชนกับ `numpy==1.26.4` ที่ pin อยู่เดิม (`pip install --dry-run` ยืนยันก่อนติดตั้งจริง, สอดคล้องกับ ABI check ใน Dockerfile) คอลัมน์ fasnet ด้านล่างคือค่าจริงจากรอบสอง

| file | WxH | EXIF | moire score | threshold | Moiré | screen_texture | fasnet_is_real | fasnet_spoof |
|---|---|---|---|---|---|---|---|---|
| face1.jpg (original) | 1152×2048 | no | **0.7425** | 0.70 | **FAIL** | False | True | 0.0520 |
| face1_r.jpg | 562×1000 | no | 0.7225 | 0.70 | **FAIL** | False | True | 0.0480 |
| face1_ok.jpg | 571×900 | no | 0.7507 | 0.70 | **FAIL** | False | None (DeepFace หา face ไม่เจอ) | — |
| face1_r_ok.jpg | 570×900 | no | 0.7415 | 0.70 | **FAIL** | False | None | — |
| face2.jpg (original) | 1152×2048 | no | **0.7430** | 0.70 | **FAIL** | False | True | 0.0744 |
| face2_r.jpg | 562×1000 | no | 0.7208 | 0.70 | **FAIL** | False | True | 0.0356 |
| face2_ok.jpg | 571×900 | no | 0.7680 | 0.70 | **FAIL** | False | None | — |
| face2_r_ok.jpg | 570×900 | no | 0.7581 | 0.70 | **FAIL** | False | None | — |
| face3.jpg (original) | 1152×2048 | no | **0.7112** | 0.70 | **FAIL** | False | True | 0.1753 |
| face3_r.jpg | 562×1000 | no | 0.6922 | 0.70 | PASS | False | True | 0.0625 |
| face3_ok.jpg | 571×900 | no | 0.7206 | 0.70 | **FAIL** | False | True | 0.2938 |
| face3_r_ok.jpg | 570×900 | no | 0.7110 | 0.70 | **FAIL** | False | True | 0.2305 |

Fasnet อ่านถูก **8/8 ไฟล์ที่ตรวจจับใบหน้าได้** ว่าเป็นคนจริง (`is_real=True`, spoof score 0.035–0.294) — 4 ไฟล์ที่เหลือ (`*_ok.jpg`, `*_r_ok.jpg` ของ face1/face2) DeepFace's OpenCV detector หา face ไม่เจอเลย (`enforce_detection=True`) ตรงกับที่ Haar cascade ในหัวข้อ 3d ก็ตรวจจับไม่เจอไฟล์เดียวกันนี้เช่นกัน — ไม่ใช่เรื่องบังเอิญ (crop ที่ `prep_images.py` ทำไว้แน่นเกินไปสำหรับ face detector ทั้งสองตัว)

**11 ใน 12 ไฟล์ fail** รวมทั้ง 3 ไฟล์ original ที่ยังไม่เคยผ่าน PIL re-encode ของเราเลย — คะแนนอยู่ในช่วง 0.71–0.74 ซึ่งสูงกว่า threshold single-frame (0.70) อยู่แล้วตั้งแต่ต้นทาง

**Screen Texture layer ไม่ false-positive สักไฟล์เดียว** — ปัญหาอยู่ที่ Moiré เท่านั้น ไม่ใช่ FFT layer ทั้งหมด

หมายเหตุ: ไฟล์ "original" เหล่านี้ (face1/2/3.jpg) เองก็ไม่มี EXIF และมี marker เป็น `JFIF` ไม่ใช่ `Exif` — แปลว่าไฟล์เหล่านี้ก็ผ่านการ re-encode มาอย่างน้อย 1 รอบมาก่อนแล้ว (เช่น ผ่านแอปแชท) ไม่ใช่ camera-original แท้ ๆ ไม่มีไฟล์ SOOC จริงในชุดทดสอบนี้ให้เทียบ แต่ไม่กระทบข้อสรุปข้อ 3 เพราะไฟล์ที่ "ผ่านการประมวลผลน้อยที่สุด" เท่าที่มีอยู่ก็ยัง fail

---

## 3. ทดสอบสมมติฐาน "double JPEG re-encode ทำให้เกิด Moiré ปลอม"

ทดสอบกับ `face1.jpg`: (a) resize ด้วย LANCZOS + บันทึก quality=95 เทียบกับ (b) **crop เท่านั้น (ไม่ resample เลย)** ให้ขนาดเป็นทวีคูณของ 8 ทั้งสองแกน + บันทึกด้วย `subsampling=0, quality=95` (คง DCT grid เดิมไว้ ไม่สร้าง misalignment ใหม่)

| variant | moire score |
|---|---|
| original (ไม่แตะเลย) | 0.7425 |
| (a) LANCZOS resize + q95 | 0.7221 |
| (b) crop-only (8-aligned) + q95/ss0 | **0.7424** |

ถ้าสมมติฐาน double-encode/misaligned-DCT-grid เป็นสาเหตุจริง (b) ควรต่ำกว่า (a) อย่างชัดเจน (สมมติฐานของ user ระบุไว้ตรง ๆ) — **ผลจริงคือ (b) แทบเท่ากับ original เป๊ะ (0.7424 vs 0.7425) และไม่ต่ำกว่า (a) เลย** สมมติฐานนี้ตกไป

---

## 4. สรุป

**ไม่ใช่ image-prep artifact — เป็น threshold/detector-design ที่ aggressive เกินไปจริง**

หลักฐานที่หักล้าง hypothesis เดิม:
1. ไฟล์ต้นทาง (`face1/2/3.jpg`) ที่ **ไม่เคยผ่าน PIL ของเราเลย** ก็ fail อยู่แล้ว (0.71–0.74 > 0.70)
2. การทดสอบแบบควบคุม (crop-only ไม่ resample เทียบ resize) ให้คะแนนแทบเหมือนต้นฉบับ ไม่สนับสนุน double-DCT-grid theory เลย

Root cause จริง: `low_r = 0.10` (face_service.py:683) นิยาม "low-frequency" แคบมาก (วงกลางแค่ 20% ของสเปกตรัม) ทำให้ high-frequency energy ที่รวม texture ผิวหนัง เส้นผม และ sharpening ปกติของกล้องมือถือ ถูกนับเป็น "หลักฐานหน้าจอ" ไปด้วย — ผลักคะแนนภาพหน้าคนจริงให้อยู่แถว 0.69–0.77 เป็นปกติ ซึ่งชนกับ threshold 0.70 พอดี ประกอบกับคำนวณบน **full frame ไม่ crop หน้า** (ต่างจาก temporal layer) พื้นหลัง เช่น ผ้าม่าน กระเบื้อง ผนังลาย ที่ห้องเรียนจริงมักมี จะยิ่งดันคะแนนสูงขึ้นไปอีก

ห้าปีการ tune ค่า threshold (0.72→0.55→0.65→0.70) ไล่ตามอาการ false-reject โดยไม่เคยแก้ที่ตัว metric (`low_r`) เอง คือรูปแบบ "แก้ symptom ซ้ำแทนที่จะแก้ cause"

**บันทึกเป็น FRR-1** (แยก series จาก `F-` — ดูเหตุผลด้านล่าง)

**ทำไมไม่ใช่ F-15:** ทุก finding ที่ขึ้นต้น `F-` ในซีรีส์นี้แปลว่า "attacker ผ่านเข้ามาได้" (security) ข้อนี้เป็นทิศตรงข้าม — **ผู้ใช้จริงถูกล็อกออก** (false-reject / usability) จึงแยกเป็น series ใหม่ `FRR-` (False-Reject Rate) ไม่นับรวมกับสถิติ security severity ของ `F-`

| ID | เรื่อง | ไฟล์:line | ประเภท | ความยากแก้ | สถานะ |
|---|---|---|---|---|---|
| **FRR-1** | Moiré FFT: `low_r=0.10` (วง low-freq แคบเกิน) + คำนวณบน full frame ไม่ crop หน้า → ภาพใบหน้าจริงจากกล้องมือถือ (มี texture/sharpening ปกติ) ติดคะแนน 0.69–0.77 ชนกับ `MOIRE_THRESHOLD_SINGLE=0.70` เป็นประจำ; ยิ่งแย่ลงถ้าพื้นหลังมีลาย (ผ้าม่าน กระเบื้อง ผนังลาย) — **อัปเดต 2026-08-25 (หัวข้อ 6.1):** วัดกับ spoof samples จริงแล้ว ไม่ใช่แค่ miscalibrated — ในทิศทางที่โค้ดใช้อยู่ (score สูง=spoof) ไม่มี threshold ไหนแยก real/spoof ได้เลย (best case = ปิด layer, ผิด 3/16); พลิกทิศทางกลับเกือบแยกได้สนิท (ผิด 1/16) — น่าจะเป็น **inverted ไม่ใช่แค่แคบเกินไป** (n=3 spoof, ยังสรุปเด็ดขาดไม่ได้) | face_service.py:18, 683, 692 | usability/FRR 🟠 | ปานกลาง→**อาจสูงกว่าที่ประเมินไว้เดิม** ถ้า inverted จริง การขยาย `low_r` เพียงอย่างเดียวจะไม่แก้ (ดูหัวข้อ 6.1) | รอ user ตัดสินใจ — ยังไม่แก้ threshold ใด ๆ ตามคำสั่ง |

**เกี่ยวข้องกับ F-5:** `F-5` (`fasnet_suspicious=0.30` ไม่สมดุลกับ layer อื่น, face_service.py:296, ดู `02-face-service.md` และ `99-summary.md` Master Findings Table) กับ `FRR-1` เป็น threshold-calibration problem บน **anti-spoof stack เดียวกัน** (`combined_spoof_score`, face_service.py:166) — F-5 ต่ำกว่า layer อื่นมากพอที่ borderline real face จะติด hard-reject vote ง่ายเกิน (ผลคือ FRR สูงขึ้นเช่นกัน แม้ root cause ต่างจาก FRR-1 คือ weight ไม่สมดุล ไม่ใช่ metric เพี้ยน) ทั้งสองข้อชี้ไปทางเดียวกัน: **stack นี้ควร calibrate ใหม่พร้อมกันทั้งชุดด้วย real+spoof samples จริง ไม่ใช่ขยับทีละ threshold แยกจุดแบบที่ผ่านมา** (ดู git log ในหัวข้อ 1 — 5 รอบ tune `MOIRE_THRESHOLD_SINGLE` อย่างเดียว)

เกี่ยวข้องกับ NFR-9 (FRR) ตามที่ผู้ใช้อ้างถึงจาก `docs/TOR.md` — **หมายเหตุ: หาไฟล์ `docs/TOR.md` ในโปรเจกต์นี้ไม่เจอ** (ทั้งใน `smartcheck_project/smartcheck_project/` และโฟลเดอร์แม่) กรุณายืนยัน path ที่ถูกต้อง หรือแนบเอกสารให้ เพื่อผูก finding นี้กับ NFR-9 ให้ตรงจริง — เนื้อหา finding ข้างต้นยืนยันได้เองจากการวัดจริง ไม่ผูกกับ path นั้น

---

## 5. คำแนะนำการเตรียมภาพทดสอบ (ไม่เกี่ยวกับ root cause แต่ถามมา)

ทั้ง `_r` และ `_ok` เตรียมด้วย PIL LANCZOS + JPEG re-save เป็นมาตรฐานที่ยอมรับได้สำหรับการทดสอบ (ข้อ 3 พิสูจน์แล้วว่าวิธี re-encode ไม่ใช่ตัวแปรที่ทำให้ fail) ไม่มี "วิธีเตรียมภาพที่ถูกต้อง" ที่จะทำให้ภาพใบหน้าจริงผ่าน Moiré ได้ในสภาพ threshold ปัจจุบัน เพราะปัญหาอยู่ที่ detector ไม่ใช่ pipeline เตรียมภาพ

---

## 6. ผลวัดกับ spoof samples จริงชุดแรก (2026-08-25)

User ถ่าย spoof samples ชุดแรก + real-face control เพิ่ม 1 ภาพ แล้ววัดด้วย `scripts/debug_moire.py` (single-image mode) รอบนี้เป็น **report-only ตามคำสั่ง — ไม่แก้ `low_r`, `MOIRE_THRESHOLD_SINGLE`, `fasnet_suspicious`, weight ใด ๆ ในโค้ดจริง**

### ข้อมูลดิบ

| Sample | ประเภท | moire | m.PASS | fasnet_spoof | fasnet verdict |
|---|---|---|---|---|---|
| face1–3 + `_r`/`_ok` (12 ไฟล์, ดูตารางเต็มในหัวข้อ 2) | real | 0.71–0.77 (12 ค่า) | FAIL 11/12 | 0.035–0.294 (8/12 ตรวจจับได้) | real ✓ (8/8) |
| real_13 (selfie, ผนังเปล่า) | real | 0.6918 | PASS | 0.0034 | real ✓ |
| spoof_01_phone (จอมือถือ) | spoof | 0.6386 | PASS ✗ | 0.9939 | spoof ✓ |
| spoof_02_laptop (จอ laptop) | spoof | 0.6375 | PASS ✗ | 0.4536 | spoof ✓ |
| spoof_03_ipad (จอ iPad) | spoof | 0.6997 | PASS ✗ | 1.0000 | spoof ✓ |

Screen Texture คืนค่า `False` และ ONNX audit คืนค่า `is_real=False` **คงที่ทุกตัวอย่างทั้ง 16 ไฟล์** (real และ spoof เหมือนกันหมด) — ดูหัวข้อ 6.3 Temporal ยังวัดไม่ได้ (ต้องการ ≥2 เฟรมต่อ subject, ตัวอย่างที่มีเป็น single-frame ทั้งหมด)

หมายเหตุ: ตัวเลข `moire` ของกลุ่ม 12 ไฟล์เดิมคือ 0.6922–0.7680 (ไม่ใช่ 0.71–0.74 ตามที่ร่างต้นฉบับสรุปไว้ — ดูตารางเต็มหัวข้อ 2) แก้ให้ตรงตัวเลขจริงก่อนสรุปต่อ

### 6.1 FRR-1 — ยกระดับจาก "miscalibrated" เป็น "inverted"

**ตรวจสอบคำกล่าวอ้าง "spoof scores ต่ำกว่า real-face range ทั้งหมด" ก่อนบันทึก (Kalāma — verify ก่อน state เป็นข้อเท็จจริง):** ไม่ตรงเป๊ะ — `spoof_03_ipad = 0.6997` สูงกว่า `real_13 = 0.6918` และสูงกว่า `face3_r.jpg = 0.6922` (ทั้งคู่เป็น real) จึงมี overlap เล็กน้อยที่ปลายล่างของกลุ่ม real ไม่ใช่ "ต่ำกว่าทั้งหมด" แบบ clean-cut แต่ **ข้อสรุปหลักไม่เปลี่ยน** — full separation (0 error) เป็นไปไม่ได้จริงในทิศทางปัจจุบันของ detector เพราะจุดนี้แหละที่ขวางไว้

**คำนวณ exhaustive threshold sweep แบบเดียวกับที่ `--compare` ใช้ (ทิศทางเดิมของโค้ด: score สูง = spoof) กับทั้ง 16 ตัวอย่าง (13 real, 3 spoof):**

- ที่ threshold ปัจจุบัน (0.70): FR (real ติด FAIL ผิด) = 11/13, FA (spoof หลุดผ่าน) = 3/3 → ผิด 14/16
- threshold ที่ดีที่สุดเท่าที่ทิศทางนี้ทำได้: **FR=0, FA=3 (ผิด 3/16)** — ทำได้ก็ต่อเมื่อขยับ threshold สูงจนไม่ trigger อะไรเลย (เทียบเท่ากับ "ปิด layer นี้ทิ้ง") **ไม่มี threshold ไหนในทิศทางนี้ที่ทำได้ดีกว่าการไม่ใช้ layer นี้เลย** — ตรงกับที่ user สรุปไว้ ("ไม่มี threshold ไหนแยกสองกลุ่มได้")

**ส่วนที่มากกว่าที่ user ตรวจ — ลองสลับทิศทางการตัดสิน (score ต่ำ = spoof แทน):**

ถ้าสลับกติกาเป็น "score < t → spoof" (ตรงข้ามกับที่โค้ดใช้อยู่ทุกวันนี้) แล้ว sweep threshold ใหม่: ที่ t ≈ 0.69 (ช่วง 0.6386–0.6918) ได้ **FR=0, FA=1 → ผิดแค่ 1/16 (93.75% ถูก)** ตัวอย่างเดียวที่ผิดคือ `spoof_03_ipad` (0.6997, จุดเดียวกับที่ทำให้ full separation ในทิศทางเดิมเป็นไปไม่ได้ด้วยตามที่ตรวจสอบไว้ข้างบน)

**นี่คือหลักฐานที่หนักแน่นกว่าคำว่า "inverted" เดิม** — ไม่ใช่แค่ "ทิศทางปัจจุบันแยกไม่ได้" แต่ "ทิศทางตรงข้ามเกือบแยกได้สนิท ด้วยตัวเลข threshold ที่ใกล้เคียงกับ 0.70 ที่มีอยู่แล้วในโค้ดมาก" — ดูกลไกที่เสนอในหัวข้อ 6.1.1

**เครื่องมืออัปเดตแล้ว (2026-08-25 รอบสี่):** `find_best_threshold()` ใน `scripts/debug_moire.py` เดิม sweep แค่ทิศทางเดียว (ทิศทางที่โค้ดจริงใช้อยู่) — ตอนนี้แก้ให้ sweep **ทั้งสองทิศทาง** อัตโนมัติแล้ว (negate คะแนนแล้ว sweep ซ้ำด้วยฟังก์ชันเดิม แล้วเทียบ error กับทิศทางเดิม) ตรวจสอบด้วย unit test เทียบกับตัวเลขที่คำนวณด้วยมือข้างบน — ได้ผลตรงกันเป๊ะ: `{'errors': 1, 'threshold': 0.6652, 'direction': 'low=spoof (INVERTED)', 'native_errors': 3, 'flipped_errors': 1}` ตอนนี้ `--compare` จะโชว์ทั้งสองทิศทางและ flag "[INVERTED beats native!]" ให้เองโดยไม่ต้องคำนวณมือแบบนี้อีกในรอบต่อไป

**Verdict การนับ "ถูก/ผิด" ราย sample (แก้เลข "0/5" ในร่างต้นฉบับ):** ที่ threshold ปัจจุบัน 0.70, `real_13` จัดว่า **ถูก** (PASS, ground truth real) — ไม่ใช่ผิด ตัวเลขที่ตรวจสอบแล้วคือ: **11/13 real ผิด (false reject), 3/3 spoof ผิด (false accept)** = ผิด 14/16 sample ที่ threshold ปัจจุบัน; ทิศทางปัจจุบันจับ spoof ได้ **0/3** ไม่ว่าจะตั้ง threshold ตรงไหนก็ตาม (ข้อสรุปนี้เหมือนร่างต้นฉบับ — เปลี่ยนแค่วิธีนับ)

**สรุป FRR-1:** เดิมบันทึกไว้ว่า "threshold ตั้งไว้แคบ/aggressive เกินไป" (`low_r` แคบเกิน) — ข้อมูลรอบนี้ (แม้ n เล็ก) ชี้ว่าปัญหาลึกกว่านั้น: **ในทิศทางที่โค้ดใช้อยู่ ไม่มี threshold ไหนทำงานได้เลย (best case = ปิด layer)** การขยาย `low_r` จาก 0.10 → 0.15 ในหัวข้อ 3c/3d (ที่ทำให้ real ผ่านหมด) จึง **ไม่ใช่ทางแก้** — มันแค่ลด false-reject ในขณะที่ layer นี้ปล่อย spoof ผ่านอยู่แล้วทั้ง 3/3 ตัวอย่าง ไม่ได้ทำให้จับ spoof ได้เพิ่มขึ้นเลย

### 6.1.1 กลไกที่เสนอ (สมมติฐาน — ยังไม่พิสูจน์, n=3)

**อธิบายว่าทำไมทิศทางกลับข้างถึงเข้ากับข้อมูลได้ดีกว่า:** สมมติฐานตั้งต้นของ `detect_screen_moire` คือ "จอจะให้ high-frequency energy สูงกว่าหน้าคนจริง" (pixel grid ของจอสร้าง moiré pattern) แต่เส้นทางการถ่ายจริงในการทดสอบนี้คือ **ถ่ายรูปคนจริง (สร้าง reference image) → แสดงบนจอ → ถ่ายจอนั้นซ้ำด้วยกล้องอีกตัว** (recapture) กระบวนการ recapture นี้มักจะ **ลด** high-frequency energy ลง ไม่ใช่เพิ่ม เพราะ:
- กล้องที่ถ่ายจอไม่ได้ focus ที่ pixel grid ของจอโดยตรงเสมอไป (depth-of-field, ระยะห่าง)
- การ downsample/anti-alias ของทั้งจอที่แสดงผลและกล้องที่ถ่ายซ้อนกัน 2 ชั้น มักเบลอ detail มากกว่าสร้าง moiré ใหม่ที่ชัดพอจะเกิน real face's ผิว/ผมซึ่งมี high-freq energy สูงอยู่แล้วโดยธรรมชาติ
- ตรงข้ามกับ real face ที่ผิวหนัง เส้นผม และ sharpening ปกติของกล้องมือถือ (ตามที่พบในหัวข้อ 4) ให้ high-freq energy สูงเป็นทุนเดิมอยู่แล้ว

**สถานะ:** เป็นสมมติฐานอธิบายกลไกเท่านั้น **ยังไม่ใช่ข้อสรุปที่ยืนยันแล้ว** — มีเพียง 3 spoof samples (จอมือถือ 1, laptop 1, iPad 1) ไม่พอสรุปว่าเป็นจริงเสมอสำหรับทุกชนิดจอ/ระยะ/มุม ต้องรอ spoof samples เพิ่มก่อนถือเป็นข้อเท็จจริง

### 6.2 F-5 — หลักฐานคัดค้าน finding เดิม, แนะนำเปิดไว้พร้อมเงื่อนไข (ไม่ปิด)

**นับจำนวนตัวอย่างให้ถูกก่อน:** Fasnet มีค่าเฉพาะไฟล์ที่ DeepFace ตรวจจับใบหน้าเจอเท่านั้น = **9 real** (8 จาก 12 ไฟล์เดิม + `real_13`), **3 spoof** — คนละจำนวนกับ Moiré (13 real/3 spoof) เพราะ 4 ไฟล์ `*_ok`/`*_r_ok` เดิม DeepFace หา face ไม่เจอ (ดูหัวข้อ 2)

- real (n=9): 0.0034 – 0.2938
- spoof (n=3): 0.4536, 0.9939, 1.0000 (min = 0.4536)
- gap = 0.4536 − 0.2938 = **0.1598 ≈ 0.16**, `fasnet_suspicious = 0.30` (face_service.py:305) อยู่กลาง gap นี้พอดี

**Exhaustive threshold sweep (ทิศทางเดิม, score สูง = spoof):** `max(real)=0.2938 < min(spoof)=0.4536` → **แยกได้สมบูรณ์ (0 error)** ที่ threshold ใดก็ได้ในช่วง (0.2938, 0.4536] รวมถึง 0.30 ที่ใช้อยู่จริง — ต่างจาก Moiré ที่แยกไม่ได้เลยแม้จะลอง threshold ทุกจุด

**เงื่อนไขที่ต้องบันทึกคู่กัน (ตามที่ user ระบุ, ตรวจแล้วตรง):**
- spoof จอ laptop (0.4536) ใกล้เส้นกว่า spoof อีก 2 ตัวมาก (จอมือถือ 0.9939, iPad 1.0000) — จอใหญ่/low-DPI ดูเหมือนเป็นเคสที่อ่อนที่สุด
- real สูงสุด (`face3_ok.jpg` = 0.2938) ห่างจาก threshold 0.30 แค่ **0.0062** — ระยะปลอดภัยบางมาก
- n เล็ก: **9 real / 3 spoof สำหรับ Fasnet โดยเฉพาะ** (ไม่ใช่ 13/3 — นั่นคือ n ของ Moiré)

**สถานะ F-5:** เปลี่ยนจาก "รอข้อมูล PRE-HARDREJECT log" → **"มีหลักฐานคัดค้าน finding เดิม แต่ n เล็กเกินจะปิดเคส — เปิดไว้รอตัวอย่างเพิ่ม"** ไม่ใช่ "ปิด" เพราะ 3 spoof ไม่พอสรุปว่า 0.4536 (จอ laptop) เป็นค่าต่ำสุดที่จะเจอได้จริงในสนาม — ยังไม่แก้ `fasnet_suspicious` ตามคำสั่ง

### 6.3 คำถามเปิด — Screen Texture กับ ONNX audit มีประโยชน์จริงไหม

ทั้งสอง layer คืนค่า**เดียวกันทุกภาพ** ตลอด 16 ตัวอย่าง (real 13 + spoof 3):
- Screen Texture: `is_screen=False` เสมอ (เหมือนกับ "ทายว่า real ตลอด" — ถูกทุกครั้งฝั่ง real, ผิดทุกครั้งฝั่ง spoof)
- ONNX audit: `is_real=False` เสมอ (เหมือนกับ "ทายว่า spoof ตลอด" — ผิดทุกครั้งฝั่ง real, ถูกทุกครั้งฝั่ง spoof)

ค่าคงที่ไม่ว่า input จะเป็นอะไร (ไม่ขยับตามภาพเลยแม้แต่นิดเดียวใน 16 ตัวอย่างนี้) แปลว่า mutual information ระหว่าง output กับ label = 0 โดยนิยาม **ไม่สรุปเพิ่มเติมตามที่สั่ง** — บันทึกไว้เป็นคำถามเปิด รอ n มากกว่านี้ก่อนตัดสินว่าเป็นเพราะ (ก) layer ไม่มีความสามารถแยกแยะจริง ๆ กับ 16 ตัวอย่างนี้บังเอิญ หรือ (ข) มีบั๊ก/ค่า threshold ภายในที่ทำให้ trigger ไม่ได้เลยไม่ว่า input จะเป็นอะไร (เช่น `min_peaks=30`/`peak_threshold_multiplier=3.0` ของ Texture หรือ `CONFIDENCE_MARGIN=0.10` override ของ ONNX) — ทั้งสองทางต้องดูโค้ด+ข้อมูลเพิ่มก่อนฟันธง

---

## 7. ก่อนพิจารณาพลิก polarity — ต้องเช็คความเสี่ยงด้าน FRR ก่อน (ยังไม่ทำ, ยังไม่ตัดสินใจ)

**คำสั่ง user: ห้ามพลิก polarity ตอนนี้.** หัวข้อ 6.1 พบว่าพลิกทิศทางการตัดสินของ Moiré (score ต่ำ = spoof แทนสูง) เกือบแยก real/spoof ได้สนิทกับข้อมูลชุดปัจจุบัน แต่ยังไม่พิสูจน์ว่าปลอดภัย เพราะ "score ต่ำ" ทางเทคนิคแปลว่า **"ภาพมี high-frequency detail น้อย"** ซึ่งไม่ได้เกิดจากการเป็น spoof เท่านั้น — ภาพเบลอ/แสงน้อย/กล้องเก่า/ไกลจากกล้อง ก็ให้ high-freq energy ต่ำได้เหมือนกันโดยเป็นคนจริงแท้ ๆ ถ้าพลิก polarity โดยไม่เช็คก่อน อาจแค่ย้าย FRR ปัญหาจาก "จอมือถือ" ไปเป็น "ห้องเรียนแสงน้อย/มือสั่น/กล้องรุ่นเก่า" ซึ่งอาจแย่กว่าเดิม (ตรงกับความกังวลเรื่อง NFR-9 ที่ยกมาตั้งแต่ต้น — ห้องเรียนจริงมีทั้งพื้นหลังลายและแสงน้อย)

### สิ่งที่ต้องถ่ายเพิ่ม (real face เท่านั้น, low-detail แต่คนจริงแท้ ๆ — ไม่ใช่ spoof)

ถ่ายในรอบเดียวได้ (9 ภาพ) ตั้งชื่อ prefix `real_lowdetail_` เพื่อให้ `--compare` อ่านเป็น real ได้ทันที (ยังคงขึ้นต้นด้วย `real_`):

| # | ไฟล์ | เงื่อนไข | กลไกที่ทดสอบ |
|---|---|---|---|
| 1 | `real_lowdetail_blur_camera.jpg` | ขยับกล้อง/มือสั่นระหว่างถ่าย ให้เห็น motion blur ชัดเจน ระยะเท่า check-in ปกติ | camera shake |
| 2 | `real_lowdetail_blur_subject.jpg` | คนอยู่นิ่ง กล้องนิ่ง แต่ผู้ถูกถ่ายขยับหัว (พยักหน้า/หันเล็กน้อย) ระหว่าง exposure | subject motion |
| 3 | `real_lowdetail_outoffocus_bg.jpg` | บังคับให้กล้อง focus ที่พื้นหลังแทนใบหน้า (เช่น แตะโฟกัสที่ผนังก่อนถ่าย) หน้าจะเบลอ | autofocus miss |
| 4 | `real_lowdetail_outoffocus_close.jpg` | เอาหน้าเข้าใกล้กล้องเกินระยะโฟกัสขั้นต่ำของกล้องตัวนั้น (ใกล้กว่า macro range) | out-of-focus (ระยะ) |
| 5 | `real_lowdetail_dim30lux.jpg` | ห้องแสงน้อยแบบห้องเรียนจริง (~30–50 lux — ปิดไฟ เหลือแสงจากหน้าต่าง/ผ้าม่านปิด) ระยะ+มุมเหมือน check-in ปกติ | low light |
| 6 | `real_lowdetail_dim10lux.jpg` | แสงน้อยกว่านั้นอีก (~10–20 lux, เกือบมืด) เคสที่แย่สุดที่ยังพอถ่ายติด | low light (worst case) |
| 7 | `real_lowdetail_oldcam.jpg` | ถ่ายด้วยกล้อง/มือถือรุ่นเก่ากว่าที่ใช้ปกติ (หรือถ้าไม่มี ใช้กล้อง webcam ของ laptop แทน — คุณภาพเซนเซอร์ต่ำกว่ากล้องมือถือทั่วไปอยู่แล้ว) แสงปกติ | เซนเซอร์/เลนส์คุณภาพต่ำ |
| 8 | `real_lowdetail_distance_plain.jpg` | ถอยห่างจากกล้องจนหน้าเหลือสัดส่วนเล็กในเฟรม (~1 เมตร แทนระยะ selfie ปกติ) พื้นหลังเรียบ | หน้าเล็ก/detail ต่อพื้นที่น้อย, แยกจาก background texture effect |
| 9 (ถ้ามีเวลา) | `real_lowdetail_distance_pattern.jpg` | เหมือนข้อ 8 แต่พื้นหลังมีลาย (เช่น ผ้าม่าน/กระเบื้อง) | หน้าเล็ก + background texture รวมกัน (เคส NFR-9 ตรง ๆ) |

ไม่ต้องถ่ายเพิ่มสำหรับ "บีบอัด JPEG หนัก" — สร้างได้เองจากภาพ real ที่มีอยู่แล้วด้วย PIL (`quality=40` เป็นต้น) จะทำให้ตอนรัน

### เกณฑ์ตัดสินว่าพลิก polarity ปลอดภัยหรือไม่

- **ปลอดภัยพอจะพิจารณาต่อ:** ทั้ง 9 ภาพ (หรือส่วนใหญ่ที่ชัดว่า degrade จริง) ยังได้ moire score **สูงกว่า** โซนอันตราย (~0.665–0.70 จากข้อมูล spoof ชุดปัจจุบัน) อย่างมีระยะห่างพอสมควร ไม่ใช่แค่ 1 ภาพเดียวที่รอด
- **ไม่ปลอดภัย / ยังพลิกไม่ได้:** มีภาพ real-แต่-degrade ตั้งแต่ 1 ภาพขึ้นไป (โดยเฉพาะถ้ามาจากกลไกคนละแบบกัน เช่น ทั้ง blur และ dim light) ตกลงมาอยู่ในโซน ≤0.70 — แปลว่าพลิก polarity จะ trade ปัญหาเดิม (จอมือถือหลุดผ่าน) ไปเป็นปัญหาใหม่ (คนจริงในสภาพแสงน้อย/มือสั่น/กล้องเก่าโดนปฏิเสธ) ซึ่งอาจกว้างกว่าเดิม เพราะสภาพแบบนี้เกิดในห้องเรียนจริงบ่อยกว่าคนถือจอมือถือมาสวมรอย
- ไม่ว่าผลจะออกมาทางไหน **9 ตัวอย่างก็ยังเล็กเกินจะฟันธง** — ใช้เป็นสัญญาณเตือนล่วงหน้า (early warning) ไม่ใช่ข้อสรุปสุดท้าย เหมือนกับที่ spoof n=3 ในหัวข้อ 6 ก็ยังเล็กอยู่

~~ยังไม่ทำตอนนี้~~ — **อัปเดต 2026-08-25 รอบห้า: ทำแล้ว, ผลออกมาแล้ว, ดูหัวข้อ 8** — user ถ่าย 7 จาก 9 ภาพที่ขอ (ครบทุก mechanism หลัก: motion/shake/dim/far/webcam, ขาด outoffocus_bg กับ distance_pattern) ผลสรุปสั้น: **พลิก polarity ไม่ปลอดภัย — ข้อมูลชี้ว่าไม่ต้องพลิกด้วยซ้ำ เพราะพลิกก็ไม่ช่วยอะไร** (ทั้งสองทิศทางแยก real/spoof ไม่ได้เลยเมื่อรวม real-degraded เข้ามา) รายละเอียดเต็มในหัวข้อ 8 — **ข้อเสนอพลิก polarity จากหัวข้อ 6.1.1 ถือว่า tested และ rejected แล้ว ด้วยข้อมูลจริง**

---

## 8. Full real-set rerun (2026-08-25 รอบห้า) — Moiré ไม่แยก real/spoof ได้ในทิศทางไหนเลย

### เตรียมข้อมูล

**เปลี่ยนชื่อไฟล์:** 12 ไฟล์ real เดิม (`face1.jpg` ฯลฯ) ไม่ขึ้นต้นด้วย `real_` เลยตกหล่นจาก `--compare` ในรอบก่อน (`real_* : 7/7` ตอนนั้นนับแค่ `real_lowdetail_*` 7 ไฟล์) เปลี่ยนชื่อเป็น `real_01_sharp_face1.jpg` … `real_12_sharp_face3_r_ok.jpg` (คงชื่อเดิมต่อท้ายไว้เพื่อ trace กลับได้ — ดู mapping):

| ชื่อใหม่ | ชื่อเดิม | ชื่อใหม่ | ชื่อเดิม |
|---|---|---|---|
| real_01_sharp_face1.jpg | face1.jpg | real_07_sharp_face2_r.jpg | face2_r.jpg |
| real_02_sharp_face1_ok.jpg | face1_ok.jpg | real_08_sharp_face2_r_ok.jpg | face2_r_ok.jpg |
| real_03_sharp_face1_r.jpg | face1_r.jpg | real_09_sharp_face3.jpg | face3.jpg |
| real_04_sharp_face1_r_ok.jpg | face1_r_ok.jpg | real_10_sharp_face3_ok.jpg | face3_ok.jpg |
| real_05_sharp_face2.jpg | face2.jpg | real_11_sharp_face3_r.jpg | face3_r.jpg |
| real_06_sharp_face2_ok.jpg | face2_ok.jpg | real_12_sharp_face3_r_ok.jpg | face3_r_ok.jpg |

**⚠️ `real_13.jpg` ที่อ้างถึงในหัวข้อ 6 ไม่มีอยู่จริงใน `test_images/` แล้ว — สาเหตุยืนยันแล้ว:** user ถ่ายทับไฟล์นี้ตอนถ่าย spoof phone-screen capture ใหม่ (`spoof_01_phone.jpg`) ไม่ใช่ลบทิ้งเฉย ๆ **ตัวเลขในหัวข้อ 6 (moire=0.6918, fasnet=0.0034) ยังใช้ยืนอ้างอิงได้ตามที่บันทึกไว้ แต่ verify ซ้ำจากไฟล์จริงไม่ได้อีกแล้ว** ชุดข้อมูลปัจจุบันจึงมี **12 sharp + 7 `real_lowdetail_*` = 19 real, + 3 `spoof_*` = รวม 22 ไฟล์ (ไม่ใช่ 23)**

**เพิ่มการรายงานราย-ไฟล์:** `run_compare()` ใน `scripts/debug_moire.py` เพิ่มฟังก์ชัน `_print_per_file()` พิมพ์ตารางราย-ไฟล์ (เรียงตาม moire score) ก่อนตารางสรุป ตามที่ขอ

### ผลราย-ไฟล์ (moire, เรียงจากต่ำไปสูง)

| ไฟล์ | ประเภท | moire | หมายเหตุ |
|---|---|---|---|
| real_lowdetail_tooclose.jpg | real, degraded | 0.3632 | โฟกัสพลาด (ใกล้เกินระยะ) |
| real_lowdetail_shake.jpg | real, degraded | 0.3807 | กล้องสั่น |
| real_lowdetail_dim30.jpg | real, degraded | 0.5618 | แสงน้อย ~30 lux |
| real_lowdetail_motion.jpg | real, degraded | 0.5909 | subject ขยับ |
| real_lowdetail_webcam.jpg | real, degraded | 0.6344 | กล้อง webcam แทนมือถือ |
| **spoof_02_laptop.jpg** | **spoof** | **0.6375** | จอ laptop |
| **spoof_01_phone.jpg** | **spoof** | **0.6386** | จอมือถือ |
| real_lowdetail_far.jpg | real, degraded | 0.6593 | ถอยไกล หน้าเล็ก |
| real_lowdetail_dark15.jpg | real, degraded | 0.6612 | แสงน้อยมาก ~15 lux |
| real_11_sharp_face3_r.jpg | real, sharp | 0.6922 | (ต่ำสุดในกลุ่ม sharp) |
| real_12…real_06 (7 ไฟล์) | real, sharp | 0.7110–0.7680 | กลุ่ม sharp ที่เหลือ |
| **spoof_03_ipad.jpg** | **spoof** | **0.6997** | จอ iPad — แทรกอยู่ระหว่าง `real_11` (0.6922) กับ `real_12` (0.7110) |

**นี่คือ interleaving จริง ไม่ใช่แค่ "spoof อยู่ระหว่างสองกลุ่ม" แบบภาพง่าย ๆ ตามที่ user สรุปไว้ — มันแทรกอยู่ 2 จุดคนละที่กัน:** `spoof_02`/`spoof_01` (0.6375/0.6386) แทรกอยู่ระหว่าง `real_lowdetail_webcam` (0.6344) กับ `real_lowdetail_far` (0.6593) — ฝั่ง degraded-real ล้วน ๆ ส่วน `spoof_03_ipad` (0.6997) แทรกอยู่ระหว่าง `real_11_sharp` (0.6922) กับ `real_12_sharp` (0.7110) — ฝั่ง sharp-real ล้วน ๆ **spoof ทั้ง 3 ตัวไม่ได้อยู่เป็นกลุ่มเดียวกันคั่นกลาง — กระจายอยู่คนละจุดของ spectrum ของ real เอง** ยิ่งตอกย้ำว่าไม่มีทางแยกได้ด้วย threshold เดียวไม่ว่าทิศทางไหน

### Bipolar sweep กับชุดเต็ม (19 real, 3 spoof) — ยืนยันด้วย `--compare`

```
moire   n_real=19  n_spoof=3  mean_real=0.6649  mean_spoof=0.6586  gap=-0.0063
        cur_rule=0.70  FR@cur=11  FA@cur=3
        best_thr=1.7680 (=ไม่มี threshold ในช่วงข้อมูลจริงที่ดีกว่า "ปิด layer")
        direction=high=spoof (native)   FR@best=0  FA@best=3
        native_err=3  flip_err=3   ← เท่ากันทั้งสองทิศทาง
        verdict: NO SEPARATION — remove from vote
```

**native_err = flip_err = 3 = trivial_best (min(19,3)=3)** — ทั้งสองทิศทางแย่เท่ากับการไม่ใช้ layer นี้เลย (ทายว่า "real ตลอด") **นี่คือคำตอบสุดท้ายของคำถามในหัวข้อ 6.1.1: ไม่ใช่แค่ทิศทางเดิมใช้ไม่ได้ ทิศทางกลับก็ใช้ไม่ได้เหมือนกัน เพราะระยะห่างระหว่าง real-degraded กับ real-sharp (0.36–0.77) กว้างกว่าระยะห่างระหว่าง spoof (0.6375–0.6997) กับ real ทั้งหมดไปแล้ว**

**ทำไมผลรอบก่อน (หัวข้อ 6.1.1, n=16) ถึงดูเหมือน "เกือบแยกได้สนิท" (flip_err=1):** เพราะตอนนั้น real sample ทั้งหมดเป็น "sharp" ล้วน (13 ไฟล์ครอบคลุมแค่ 0.69–0.77) ไม่มี real-degraded อยู่ในชุดข้อมูลเลย ทำให้ล่างสุดของ real (real_13=0.6918) บังเอิญสูงกว่า spoof ส่วนใหญ่พอดี พอเพิ่ม real-degraded (0.36–0.66) เข้ามา — ซึ่งเป็นเงื่อนไขที่**เกิดขึ้นจริงในห้องเรียน**ไม่ใช่เคสสมมติ — ช่องว่างที่เคยดูสะอาดก็หายไปทันที **ผลรอบ n=16 ถือว่าถูก supersede แล้วโดยรอบนี้** (เก็บไว้ในหัวข้อ 6.1.1 เป็นบันทึกประวัติเท่านั้น อย่าอ้างอิงเป็นข้อสรุปปัจจุบัน)

**สรุป (แก้ไขหัวข้อ 6.1/6.1.1 และ 7):** ~~FRR-1 อาจเป็น "inverted"~~ → **FRR-1 คือ non-separable ในทุกทิศทาง** Moiré ไม่มีค่า threshold ไหน (ทิศทางใดก็ตาม) ที่แยก real จาก spoof ได้ดีกว่าการไม่ใช้ layer นี้เลย ข้อเสนอพลิก polarity (หัวข้อ 6.1.1) **tested and rejected** ด้วยข้อมูลจริงชุดนี้ — ไม่ใช่เพราะพลิกแล้วเสี่ยง FRR เพิ่ม (ที่กลัวไว้ในหัวข้อ 7) แต่เพราะพลิกแล้ว**ไม่ได้ผลลัพธ์ที่ดีขึ้นเลยแม้แต่นิดเดียว** — คำถามเรื่องความเสี่ยง FRR จากการพลิกจึงตกไปเองด้วย (ไม่ต้องพลิกก็ไม่มีอะไรจะเสี่ยง)

**อธิบาย git history ได้แล้ว:** 5 รอบ tune `MOIRE_THRESHOLD_SINGLE`/`MOIRE_THRESHOLD` (0.60→0.72→0.55→0.65→0.70, หัวข้อ 1) ไม่เคยได้ผลถาวรเพราะ **ปัญหาไม่ใช่ตัวเลข cutoff เลย — คือ distribution ของ real กับ spoof แทรกกันอยู่แล้วโดยธรรมชาติ** ไม่มีตัวเลขไหนแก้ปัญหานี้ได้ตราบใดที่ metric (`low_r=0.10`, full-frame, high-freq-ratio) ยังเหมือนเดิม

### สถานะ Fasnet (ตอบข้อ 4)

จาก `--compare` ชุดเต็ม: `n_real=14(-5)` `n_spoof=3` `mean_real=0.0885` `mean_spoof=0.8158` `gap=+0.7273` **FULL SEPARATION, FR@cur=0, FA@cur=0** (แก้ตัวเลข gap จากร่าง user ที่ระบุ 0.7713 — ตัวเลขที่วัดได้จริงคือ 0.7273, ไม่กระทบข้อสรุป)

**Face-detect ล้มเหลว 5/19 ไม่ใช่ 1/7** — ไม่ใช่แค่ 1 ไฟล์จาก `real_lowdetail_*` ตามที่ระบุไว้ (นั่นถูกแค่บางส่วน): 4 ไฟล์เดิมคือ `real_02/04/06/08` (`*_ok`/`*_r_ok`, ปัญหาเดิมจาก crop แน่นเกินที่เคยพบในหัวข้อ 2/3d) **บวกอีก 1 ไฟล์ใหม่คือ `real_lowdetail_dark15.jpg`** (มืดเกินกว่า DeepFace's detector จะหาหน้าเจอ — สมเหตุสมผล คนละสาเหตุกับ 4 ไฟล์แรก) รวม **5/19 (ไม่ใช่ 5/7 หรือ 1/7)**

**บันทึกตามที่ user สั่ง: Fasnet คือ layer เดียวที่มีความสามารถแยกแยะจริงในข้อมูลชุดนี้** (Moiré = ไม่มี, Texture = ดูหัวข้อ 9, ONNX = ดูด้านล่าง) FULL SEPARATION ด้วย margin กว้าง (real สูงสุด 0.294 ต่ำกว่า spoof ต่ำสุด 0.4536 — gap 0.16) และตอนนี้ทดสอบผ่านทั้ง real-sharp และ real-degraded แล้ว ไม่ใช่แค่ real-sharp เหมือนรอบก่อน — **หลักฐานยิ่งหนักแน่นขึ้นสำหรับ F-5 "เปิดไว้รอข้อมูลเพิ่ม" ไม่ใช่ "ปิด"**

### ข้อควรระวังเรื่อง ONNX — "INVERTED beats native" ที่ไม่ใช่สัญญาณจริง (ตอบข้อ 5)

`--compare` รายงาน: `mean_real=0.9940` `mean_spoof=0.9928` **`gap=-0.0013`** `direction=low=spoof (INVERTED)` `flip_err=2` (ดีกว่า `native_err=3`) verdict โชว์ `[INVERTED beats native!]`

**นี่คือ noise ไม่ใช่ signal อย่างชัดเจน** — real และ spoof แทบจะให้ค่าเดียวกันเป๊ะ (0.9940 vs 0.9928, ต่างกันแค่ 0.0013 หรือ 0.13%) การที่ทิศทางกลับ "ชนะ" ด้วย error 2 แทน 3 คือผลของ threshold sweep ไปเจอจุดตัดที่บังเอิญแยกตัวอย่าง 1-2 ตัวออกได้พอดีจาก n ที่เล็กมาก (22 ตัวอย่างรวม) ไม่ใช่เพราะ ONNX มีสัญญาณจริงที่กลับทิศทาง

**คำเตือนสำหรับเครื่องมือ (บันทึกไว้ตามที่สั่ง ไม่แก้โค้ดรอบนี้):** `find_best_threshold()`/`--compare` **จะเลือก "ผู้ชนะ" ระหว่าง 2 ทิศทางเสมอ แม้ทั้งสองทิศทางไม่มีสัญญาณจริงเลยก็ตาม** เพราะ exhaustive sweep เป็น deterministic minimization — ไม่มีกลไกตรวจว่า gap มีนัยสำคัญทางสถิติหรือไม่ ต้องดู **`gap`** และ **`n`** คู่กับ verdict เสมอ ก่อนเชื่อ flag `[INVERTED beats native!]` — แถวนี้ (ONNX, gap≈0, n=22) คือตัวอย่างสอนบทเรียนนี้โดยตรง ไม่ควรนำ threshold ที่ sweep ได้จากแถวนี้ไปใช้งานอะไรทั้งสิ้น

---

## 9. F-15 (ใหม่) — Screen Texture layer คำนวณผิดโครงสร้าง ไม่ใช่แค่สัญญาณอ่อน

**คำถามเปิดจากหัวข้อ 6.3 ตอบได้แล้ว: เป็นข้อ (ข) — มีบั๊ก ไม่ใช่ (ก) สัญญาณอ่อนจริง**

`detect_screen_texture()` (face_service.py:706–741) คำนวณ `threshold = mean(high_freq) + 3×std(high_freq)` โดย `high_freq` คือ magnitude spectrum 256×256 **ทั้งอาเรย์ที่ mask กลาง (25% ของพื้นที่ = 128×128) ให้เป็น 0 แล้ว** (บรรทัด 733–736) — ปัญหาคือ `np.mean()`/`np.std()` ที่ตามมา (บรรทัด 738) **คำนวณจากทั้งอาเรย์รวม 0 ที่ถูก mask ไว้ด้วย ไม่ได้คำนวณเฉพาะพื้นที่ high-frequency จริงที่เหลือ**

**หลักฐาน (diagnostic script นอก `app/`, ไม่แก้โค้ดจริง):**

| input | mean (ทั้งอาเรย์) | std | threshold | max ค่าจริงใน high_freq | num_peaks |
|---|---|---|---|---|---|
| spoof_01_phone.jpg (จอมือถือจริง) | 5.6295 | 3.3205 | 15.5909 | 11.1442 | **0** |
| spoof_02_laptop.jpg | 5.1750 | 3.0793 | 14.4130 | 9.7456 | **0** |
| spoof_03_ipad.jpg | 5.9067 | 3.4708 | 16.3191 | 11.2129 | **0** |
| random noise (synthetic) | 7.1745 | 4.1784 | 19.7096 | 11.0073 | **0** |
| **checkerboard 2px (synthetic, worst-case screen-grid)** | 0.0002 | 0.0623 | 0.1870 | 15.9385 | **1** (เพียง 1!) |

แม้แต่ checkerboard สังเคราะห์ 2 พิกเซล/คาบ (สัญญาณ periodic ที่แรงที่สุดเท่าที่จะเป็นไปได้ — แรงกว่าจอจริงที่ผ่านกล้อง/JPEG มาแล้วมาก) ก็ยังได้ `num_peaks=1` **ห่างจาก `min_peaks=30` (ค่าที่ caller ทุกจุดใช้จริง) มหาศาล** — แปลว่าฟังก์ชันนี้แทบไม่มีทางคืน `True` ได้เลยไม่ว่า input จะเป็นอะไร

**เปรียบเทียบ:** ถ้าคำนวณ `mean`/`std` เฉพาะพื้นที่ high-frequency จริง (ไม่รวมพื้นที่ mask 0) แทน — ทดสอบกับ `spoof_01_phone.jpg` เดียวกัน: `mean=7.5060 std=0.7846 threshold=9.8598 → num_peaks=113` (เทียบกับ 0 จากสูตรปัจจุบัน) **113 vs 0 — ต่างกันคนละโลก**

**สาเหตุ:** พื้นที่ mask 25% ที่ถูกบังคับเป็น 0 ปนเข้าไปใน `mean`/`std` เหมือนเป็น "ข้อมูลจริง" ทั้งที่มันคือ placeholder ที่ตั้งใจตัดออก — การปนกันนี้ดึง mean ลงและ**ดัน std ขึ้นมาก** (เพราะ array กลายเป็น bimodal: กองใหญ่ที่ 0 + กระจายที่ค่าจริง) ทำให้ `threshold = mean+3σ` สูงเกินกว่าค่าจริงในพื้นที่ high-freq จะแตะถึงได้เลย ไม่ว่า input จะมี pattern แรงแค่ไหน

**ผลกระทบ:** Screen Texture ที่ได้รับ weight 15% ใน `SPOOF_WEIGHTS["texture"]` (face_service.py:35) และเป็นหนึ่งใน layer หลักที่ตั้งใจจับ "high-DPI screens ที่ Moiré จับไม่ได้" (ตามคอมเมนต์ที่ face_service.py:713) **แทบไม่เคย contribute อะไรเลยในทางปฏิบัติ** — ไม่ใช่แค่ "อ่อน" แต่ "ปิดตัวเองอยู่โดยโครงสร้าง" การที่ layer นี้ "ไม่เคย false-positive" ตลอดการสืบสวนนี้ (16 ตัวอย่างในหัวข้อ 6, 22 ตัวอย่างในหัวข้อ 8) ไม่ใช่เพราะ specificity ดี — เป็นเพราะแทบไม่เคย trigger อะไรเลยไม่ว่าจะเจออะไร

**บันทึกเป็น F-15** (F-number เพราะเป็นเรื่อง defense-in-depth ที่อ่อนกว่าที่ระบบคิดไว้จริง — layer ที่ควรช่วยจับ screen replay ไม่ทำงาน ไม่ใช่ FRR- เพราะไม่ได้ทำให้ real user ถูกปฏิเสธเพิ่ม ตรงข้าม — มันไม่ reject อะไรเลย):

| ID | เรื่อง | ไฟล์:line | ประเภท | ความยากแก้ | สถานะ |
|---|---|---|---|---|---|
| **F-15** | `detect_screen_texture()` คำนวณ `mean`/`std` ของ threshold จากทั้งอาเรย์ (รวมพื้นที่ mask 0 ไว้ 25%) แทนที่จะคำนวณเฉพาะพื้นที่ high-frequency จริง ทำให้ `threshold=mean+3σ` สูงเกินกว่าจะ trigger ได้แทบทุกกรณี (ยืนยันด้วย synthetic worst-case checkerboard ก็ยังได้แค่ 1 peak) layer นี้จึงแทบไม่เคย contribute การตรวจจับจริงแม้จะมี weight 15% ในระบบ | face_service.py:738 (root cause), 706–741 (ทั้งฟังก์ชัน) | security (defense-in-depth อ่อนกว่าที่คิด) 🟠 | ต่ำ-ปานกลาง | **แก้แล้ว 2026-08-25 (รอบหก)** — minimal fix, `min_peaks`/multiplier ไม่แตะ ดูผลวัดใหม่หัวข้อ 10 |

**ไม่ใช่ FRR:** ตรงข้ามกับ FRR-1 (Moiré) ที่ reject คนจริงเกินไป, F-15 คือ layer ที่ **ไม่ reject อะไรเลย** (แม้แต่ spoof ตรง ๆ) จึงเป็นความเสี่ยงด้าน security (การป้องกันอ่อนกว่าที่ weight ในระบบสมมติไว้) ไม่ใช่ความเสี่ยงด้าน FRR

---

## 10. F-15 fix applied + re-measurement (2026-08-25 รอบหก)

### การแก้ (`app/` — ได้รับอนุญาตแล้ว, การแก้เดียวที่ทำในรอบนี้)

`face_service.py:731–748` (`detect_screen_texture`) — เพิ่ม `ring_mask` (boolean array, `False` ตรงพื้นที่ mask กลางเดียวกับที่ `high_freq` ถูก zero ไว้) แล้วคำนวณ `mean`/`std` จาก `high_freq[ring_mask]` แทนที่จะเป็น `high_freq` ทั้งอาเรย์ **ไม่แตะ `min_peaks=30` หรือค่าอื่นใดในฟังก์ชัน** ตามคำสั่ง — บรรทัด `num_peaks = int(np.sum(high_freq > threshold))` และ `return num_peaks > min_peaks` เหมือนเดิมทุกตัวอักษร มีแค่ตัว `threshold` ที่คำนวณถูกต้องขึ้น

**Smoke test ก่อนรัน `--compare`:** เรียกฟังก์ชันจริงจาก `app/services/face_service.py` (ไม่ใช่ reimplementation) กับ `spoof_01_phone.jpg`: `detect_screen_texture(img, min_peaks=30)` → **`True`** (เดิมคือ `False` เสมอไม่ว่า input อะไร) ยืนยันว่าการแก้ทำงาน

**อัปเดต `scripts/debug_moire.py`:** `_texture_num_peaks()` (ใช้กับ `--compare` เพื่อโชว์ raw count แทน bool) แก้ให้ตรงกับสูตรใหม่เช่นกัน — ไม่งั้น instrumentation จะไม่ตรงกับ `app/` อีกต่อไป

### ผลวัดราย-ไฟล์หลังแก้ (`--compare test_images`, texture column)

| ไฟล์ | ประเภท | moire | **texture (peaks)** |
|---|---|---|---|
| real_lowdetail_tooclose.jpg | real, degraded | 0.3632 | 43 |
| real_lowdetail_shake.jpg | real, degraded | 0.3807 | **404** |
| real_lowdetail_dim30.jpg | real, degraded | 0.5618 | 8 |
| real_lowdetail_motion.jpg | real, degraded | 0.5909 | 0 |
| real_lowdetail_webcam.jpg | real, degraded | 0.6344 | 87 |
| real_lowdetail_far.jpg | real, degraded | 0.6593 | 10 |
| real_lowdetail_dark15.jpg | real, degraded | 0.6612 | 179 |
| real_11_sharp_face3_r.jpg | real, sharp | 0.6922 | 107 |
| real_12_sharp_face3_r_ok.jpg | real, sharp | 0.7110 | 0 |
| real_09_sharp_face3.jpg | real, sharp | 0.7112 | 98 |
| real_10_sharp_face3_ok.jpg | real, sharp | 0.7206 | 0 |
| real_07_sharp_face2_r.jpg | real, sharp | 0.7208 | 95 |
| real_03_sharp_face1_r.jpg | real, sharp | 0.7225 | 92 |
| real_04_sharp_face1_r_ok.jpg | real, sharp | 0.7415 | 0 |
| real_01_sharp_face1.jpg | real, sharp | 0.7425 | 74 |
| real_05_sharp_face2.jpg | real, sharp | 0.7430 | 81 |
| real_02_sharp_face1_ok.jpg | real, sharp | 0.7507 | 0 |
| real_08_sharp_face2_r_ok.jpg | real, sharp | 0.7581 | 0 |
| real_06_sharp_face2_ok.jpg | real, sharp | 0.7680 | 0 |
| **spoof_02_laptop.jpg** | **spoof** | 0.6375 | **39** |
| **spoof_01_phone.jpg** | **spoof** | 0.6386 | **113** |
| **spoof_03_ipad.jpg** | **spoof** | 0.6997 | **157** |

### สรุปจาก `--compare`

```
texture  n_real=19  n_spoof=3  mean_real=67.2632  mean_spoof=103.0000  gap=+35.7368
         cur_rule(min_peaks)=30   FR@cur=10   FA@cur=0
         best_thr=110.0   direction=high=spoof (native)   FR@best=2   FA@best=1
         native_err=3   flip_err=3
         verdict: NO SEPARATION — remove from vote
```

### ตอบคำถามตรง ๆ: "แยก 3 spoof จาก 19 real ได้ไหม"

**ไม่สะอาด แต่ก็ไม่ใช่ "ไม่ทำอะไรเลย" เหมือนก่อนแก้ — สองคำตอบต่างกันสำหรับสองคำถามต่างกัน:**

1. **"มี threshold เดียวที่แยกสองกลุ่มได้สมบูรณ์ไหม" → ไม่มี** best-threshold sweep ได้ error ต่ำสุด = 3 (FR=2, FA=1) เท่ากับ trivial baseline (min(19,3)=3) พอดี — ตรงนิยาม "NO SEPARATION" ของเครื่องมือ เพราะ `real_lowdetail_shake.jpg` (real, มือสั่น) ได้ 404 peaks — **สูงกว่า spoof ทั้ง 3 ตัวรวมกัน** (motion blur สร้าง directional streak ใน FFT ที่นับเป็น "peak" ได้เหมือนกัน) ในขณะที่ 6 ไฟล์ real ที่ผ่าน crop+LANCZOS resize ของ `prep_images.py` (`*_ok`/`*_r_ok` ทั้ง 6 ไฟล์ไม่มีข้อยกเว้น) ได้ 0 peaks พอดี — LANCZOS เป็น low-pass filter ที่ลบ high-freq outlier ออกไปเหมือนกัน (สอดคล้องกับที่พบไปแล้วในหัวข้อ 3/4 ว่า resize แบบนี้ลด high-freq energy) **ไม่ใช่บั๊กใหม่ — เป็นผลจาก image prep ที่รู้สาเหตุแล้ว**

2. **"ที่ threshold ปัจจุบัน (`min_peaks=30`) จับ spoof ทั้ง 3 ตัวที่มีอยู่ได้ไหม → ได้ครบ (`FA@cur=0`)** แต่แลกกับ false-reject ของ real 10/19 (ทุกไฟล์ real ที่ยังไม่ผ่าน crop+resize ของ `prep_images.py` รวมถึงไฟล์มือสั่น) — เทียบกับก่อนแก้ที่ `FA@cur` เป็น **3/3 เสมอ ไม่ว่า threshold จะตั้งเท่าไหร่** (เพราะ `num_peaks` ไม่เคยเกิน ~1) การแก้นี้เปลี่ยน Texture จาก "ไม่เคย catch อะไรเลย" เป็น "catch ทุกอย่างที่มี high-frequency content แรง ไม่ว่าจะเป็น spoof หรือ real motion blur" — ทิศทางของสัญญาณ (`gap=+35.7`, spoof เฉลี่ยสูงกว่า real) **ถูกทางแล้ว** ตามที่ตั้งใจออกแบบไว้ตอนแรก (คอมเมนต์ face_service.py:713: "OLED screens typically score 80-200+, real faces 5-30") แต่ในทางปฏิบัติ real faces ในชุดนี้ไม่ได้อยู่ในช่วง 5-30 เสมอไป (มีตั้งแต่ 0 ถึง 404)

**สรุป (คำของ user, บันทึกตรงตามนี้): "การแก้คืนความสามารถมองเห็น ไม่ใช่ความสามารถใช้งาน" (restored visibility, not usability)** — ก่อนแก้ layer นี้เป็น black box ที่ vote "pass" แบบไม่มีเงื่อนไขเสมอ (0 error บน spoof เป็นภาพลวงจาก num_peaks ที่แตะ 30 ไม่ได้เลย ไม่ใช่เพราะแยกแยะได้จริง) การแก้คุ้มค่าที่จะทำเพราะเปิดให้เห็นสัญญาณที่ซ่อนอยู่ (ทิศทางถูก, `FA@cur=0`) แต่ **layer ที่แก้แล้วนี้ก็ยังไม่แยกแยะได้จริงที่ n=22** — best-threshold ยัง tie กับ trivial baseline เหมือนเดิม

### ตัดไฟล์ prep_images.py ออกจากการวิเคราะห์ต่อจากนี้ (คำสั่ง user)

**เหตุผล:** ทั้ง 6 ไฟล์ `*_ok`/`*_r_ok` (ผ่าน `prep_images.py` — crop + LANCZOS resize เป็น 900px + quality=90) ได้ texture=0 พอดีทั้ง 6 ไฟล์ไม่มีข้อยกเว้น — นี่คือผลของสคริปต์ resize ของ user เองที่ทำลาย high-frequency content ไม่ใช่คุณสมบัติของภาพถ่ายจริงตามธรรมชาติ เอาไว้ในชุดข้อมูลต่อจะดึงค่าเฉลี่ย/threshold ให้เพี้ยนไปในทางที่ไม่สะท้อนการใช้งานจริง (กล้องจริงไม่ได้ resize แบบนี้ก่อนส่งเข้า detector)

**Effective sample ตั้งแต่นี้ไป: n=16 รวม (13 real ไม่ผ่านการประมวลผล + 3 spoof)** ไม่ใช่ 19 real/22 รวมเหมือนหัวข้อ 8-10 ด้านบน — 13 real = 6 sharp-unprocessed (`real_01/03/05/07/09/11`) + 7 `real_lowdetail_*`

**Re-verify ด้วย `--compare` บนชุด n=16 (ไม่แก้ threshold ใด ๆ แค่กรองไฟล์):**

```
moire    n_real=13  n_spoof=3  mean_real=0.6295  mean_spoof=0.6586  gap=+0.0291
         FR@cur=5  FA@cur=3  best_thr=1.7430  native_err=3  flip_err=3  → NO SEPARATION (เหมือนเดิม)
texture  n_real=13  n_spoof=3  mean_real=98.3077  mean_spoof=103.0000  gap=+4.6923  ← แคบลงมาก (จาก +35.7)
         FR@cur=10  FA@cur=0  best_thr=110.0  native_err=3  flip_err=3  → NO SEPARATION (เหมือนเดิม)
fasnet   n_real=12(-1)  n_spoof=3  mean_real=0.0596  mean_spoof=0.8158  gap=+0.7562  ← กว้างขึ้น (จาก +0.7273)
         FULL SEPARATION, FR@cur=0, FA@cur=0 (เหมือนเดิม แน่นขึ้น)
onnx     gap=-0.0012 (เหมือนเดิม, ยัง noise)
```

**ผลของการตัด 6 ไฟล์ที่ผ่าน prep_images.py ออก:**
- **Fasnet แข็งแรงขึ้น** (gap กว้างขึ้นจาก 0.16 เดิม เป็น 0.76 บน spoof_score scale — เพราะ real max ลดจาก 0.2938 เหลือ 0.2355 หลังตัด `face3_ok`/`face3_r_ok` ที่เคยดันค่าสูงสุดออก)
- **Texture's gap หดจาก +35.7 เหลือ +4.7** — พิสูจน์ตรงคำเตือนของ user: ตัวเลข gap เดิม (35.7) ส่วนใหญ่มาจากไฟล์ 6 ไฟล์ที่ resize จน texture=0 ไม่ใช่จากความแตกต่างจริงระหว่าง real-camera กับ spoof-screen — เมื่อเทียบแบบยุติธรรม (ภาพกล้องจริงล้วน ไม่ผ่าน resize พิเศษ) signal ที่เหลือมีน้อยกว่าที่ตัวเลขเดิมทำให้ดูเหมือน
- **Moiré ไม่เปลี่ยนข้อสรุป** — ยัง NO SEPARATION ทั้งก่อนและหลังตัด (จุดที่ทำให้แยกไม่ได้อยู่ที่ปลายล่างของ real-degraded กับปลายบนของ real-sharp ที่เหลืออยู่ ไม่ใช่ไฟล์ที่ถูกตัดออก)

---

## 11. Step 3 — ข้อเสนอถอด Moiré/Texture ออกจาก vote (ตัวเลข+โค้ดเท่านั้น ยังไม่ implement)

### 11.0 ก่อนอ่านข้อเสนอ: "vote" ไม่ใช่จุดเดียวที่ Moiré/Texture ตัดสินใจ — นี่คือจุดสำคัญที่สุดของหัวข้อนี้

`combined_spoof_score()` **ไม่ใช่**จุดเดียวที่ Moiré/Texture ปฏิเสธผู้ใช้ได้ มี **standalone gate อีก 3 จุด** ที่เรียก `detect_screen_moire`/`detect_screen_texture` ตรง ๆ แล้ว fail-close ทันที **ก่อน**ที่ `combined_spoof_score` จะถูกเรียกด้วยซ้ำ — คนละ code path, ไม่ผ่าน `SPOOF_WEIGHTS` เลย:

| # | จุด | ไฟล์:line | threshold | ความถี่ |
|---|---|---|---|---|
| 1 | check-in: single-frame Moiré | `api_checkin.py:172–191` | `MOIRE_THRESHOLD_SINGLE=0.70` | 1 ครั้ง/check-in — **นี่คือจุดที่ error message ต้นเรื่อง `"ตรวจพบหน้าจอมือถือ"` มาจริง ๆ** |
| 2 | check-in: single-frame Texture | `api_checkin.py:193–210` | `min_peaks=30` | 1 ครั้ง/check-in |
| 3 | enroll: multi-frame Moiré (final submit) | `student.py:449–467` | `MOIRE_THRESHOLD=0.60` (**เข้มกว่า checkin**) | 1 ครั้ง/enroll (เฉลี่ย 5 เฟรม) |
| 4 | enroll: multi-frame Texture (final submit, ≥2/5) | `student.py:469–485` | `min_peaks=30` | 1 ครั้ง/enroll |
| 5 | enroll: liveness pre-check Moiré | `student.py:994–1005` | `MOIRE_THRESHOLD_SINGLE=0.70` | **สูงสุด 8 ครั้ง/enroll** (docstring บรรทัด 960: "Step 2×1, Step 3×2, Step 4×5") |
| 6 | enroll: liveness pre-check Texture | `student.py:1007–1015` | `min_peaks=30` | สูงสุด 8 ครั้ง/enroll |

**ผลที่ตามมาตรง ๆ: การแก้ `SPOOF_WEIGHTS`/`combined_spoof_score` (ไม่ว่าจะเลือกข้อไหนในหัวข้อ 11.2) จะไม่แก้ gate ทั้ง 6 จุดนี้เลยสักจุดเดียว** เพราะเป็นโค้ดคนละบล็อกที่เรียก `detect_screen_moire`/`detect_screen_texture` ตรง ๆ ไม่ผ่าน weighted-vote ระบบ **อาการที่รายงานตั้งแต่ต้นการสืบสวนนี้ (checkin ปฏิเสธด้วยข้อความ "ตรวจพบหน้าจอมือถือ") มาจากจุด #1 ไม่ใช่จาก `combined_spoof_score`** — ถ้าเป้าหมายคือแก้อาการต้นเรื่องจริง ๆ ต้องแก้ที่ 6 จุดนี้ ไม่ใช่แค่ weight ใน `combined_spoof_score`

**enroll โดนหนักกว่า checkin:** enroll มี gate มากถึง 4 จุด (3-6) เทียบกับ checkin 2 จุด (1-2) และ threshold ของ gate #3 (`0.60`) เข้มกว่า checkin's #1 (`0.70`) — ต่อ 1 ครั้งการลงทะเบียน ผู้ใช้ต้องผ่าน Moiré evaluation ได้มากถึง **9 ครั้ง** (8 liveness pre-check + 1 final submit) เทียบกับ checkin ที่แค่ 1 ครั้ง — ยิ่งจำนวนครั้งมาก ยิ่งมีโอกาสเจอ false-positive อย่างน้อย 1 ครั้งสูงขึ้นตามหลักความน่าจะเป็น (แม้ต่อครั้งจะมี FRR เท่ากัน)

**คำถามข้อ 11.1–11.3 ด้านล่างตอบตามที่ user ถามตรง ๆ (ขอบเขต `combined_spoof_score` เท่านั้น) แต่ต้องอ่านคู่กับ 11.0 เสมอ — การแก้แค่ `combined_spoof_score` แก้ปัญหาที่รายงานมาตั้งแต่ต้นไม่ได้**

### 11.1 โครงสร้างปัจจุบันของ `combined_spoof_score` (face_service.py:166–432)

**Weights (face_service.py:29–38):**
```python
SPOOF_WEIGHTS = {
    "fasnet":   0.15,   # was 0.35 — demoted; DeepFace Fasnet weak on high-DPI
    "moire":    0.30,   # was 0.20 — best pixel-grid signal
    "temporal": 0.30,   # was 0.20 — best static-photo signal
    "texture":  0.15,   # unchanged — complements Moiré
    "onnx":     0.10,   # unchanged — audit only, usually disabled
}
SPOOF_DECISION_THRESHOLD = 0.50
```
คอมเมนต์บรรทัด 31 ("FFT-based layers (Moiré, Temporal) are more reliable for screen replay attacks") **คือเหตุผลที่ครั้งหนึ่งเคยลด weight ของ Fasnet จาก 0.35→0.15 แล้วเพิ่ม Moiré จาก 0.20→0.30** — ข้อมูลที่วัดได้ในการสืบสวนนี้ (Fasnet = layer เดียวที่แยกแยะได้จริง, Moiré = แยกไม่ได้เลย) **ชี้ตรงข้ามกับสมมติฐานที่ใช้ตัดสินใจ rebalance ครั้งนั้น**

**ลำดับการทำงาน (มี 3 gate ที่ทำงานนอกเหนือ weighted-sum, เรียงตามลำดับจริงในโค้ด):**

1. **Fail-close ถ้า Fasnet ตายทั้งหมด** (line 275–291) — ไม่ขึ้นกับ weight เลย ดูหัวข้อ 11.3
2. **Hard-reject ถ้า ≥2 layers "suspicious" พร้อมกัน** (line 293–353) — เช็คจาก `spoof_score` ของแต่ละ layer โดยตรง (`moire≥0.55`, `texture≥0.50`, `temporal≥0.50`, `fasnet≥0.30`) **ไม่สนใจ weight เลย** ถ้า layer ยังถูกคำนวณอยู่ (แค่ weight=0 ใน `SPOOF_WEIGHTS`) ค่า `spoof_score` ของมันก็ยังเข้าเงื่อนไขนี้ได้ตามปกติ
3. **Hard-reject ถ้า Moiré เดี่ยว ≥0.85** (line 355–371) — `moire_spoof_score` ถูก map ให้เป็น `1.0` พอดีทันทีที่ `moire_avg >= MOIRE_THRESHOLD_SINGLE` (line 195–196) แปลว่า **ภาพ real ที่ fail Moiré แบบ raw (avg≥0.70) จะมี `spoof_score=1.0` เสมอ ซึ่ง ≥0.85 อัตโนมัติ — gate นี้แทบจะเป็น mirror ของ raw Moiré check ตัวเดิม เพียงย้ายมาอยู่ใน `combined_spoof_score`**
4. **Weighted sum** (line 373–396) — normalize weight ที่เหลือให้รวมเป็น 1.0 แล้วถ่วงน้ำหนัก, `is_real = combined < 0.50`

### 11.2 ตัวเลือกสำหรับ Moiré/Texture ใน `combined_spoof_score`

| ตัวเลือก | กลไก | ผลจริงต่อ FRR | หมายเหตุ |
|---|---|---|---|
| **(A) weight=0 ใน `SPOOF_WEIGHTS`** | `"moire": 0.0, "texture": 0.0` — ยังคำนวณ `detect_screen_moire`/`detect_screen_texture` เหมือนเดิมทุกครั้ง แค่ไม่บวกเข้า weighted sum | **แทบไม่ช่วยอะไรเลย** — gate #2 และ #3 ในหัวข้อ 11.1 เช็ค `spoof_score` ตรง ๆ ไม่ผ่าน weight เลย ภาพ real ที่ moire_avg≥0.70 จะยังโดน hard-reject ผ่าน gate #3 เหมือนเดิมทุกประการ แม้ weight=0 | ง่ายที่สุดจะแก้ (1 บรรทัด) แต่ **แก้ปัญหาไม่จริง** — เป็นการเปลี่ยนที่ "ดูเหมือนแก้" แต่ FRR ไม่ลดจริง |
| **(B) เอาออกจาก `combined_spoof_score` ทั้งหมด** (ไม่เรียก `detect_screen_moire`/`detect_screen_texture` ในฟังก์ชันนี้เลย, ลบออกจาก `layers` dict) | `layers.get("moire", {})` จะ return `{}` → `spoof_score=None` → `_layer_suspicious` return `False` เสมอ → gate #2/#3 ไม่มีทาง trigger จาก Moiré/Texture อีกเลย | **ได้ผลจริง** ตัดทั้ง 2 gate ที่ bypass weight ออกไปด้วย ไม่ใช่แค่ weighted sum | ต้องแก้โค้ดมากกว่า (ลบ block ทั้งก้อน + ปรับ `SPOOF_WEIGHTS` ให้เหลือ 3 keys) เสีย audit log ของ 2 layer นี้ไปจาก `combined_spoof_score`'s output (แต่ยังมี log จาก gate 6 จุดในหัวข้อ 11.0 อยู่ดี ถ้ายังไม่แก้จุดนั้น) |
| **(C) log ไว้แต่ไม่โหวต** (คำนวณเหมือนเดิม ใส่ใน `layers` dict เพื่อ audit/log แต่ข้ามใน `_layer_suspicious` check + ไม่บวก weighted sum) | ต้องเพิ่ม logic ใหม่ (เช่น `HARD_REJECT_ELIGIBLE = {"temporal","fasnet"}` แล้ว filter gate #2/#3 ด้วย set นี้) ไม่ใช่แค่ลบ/ใส่ 0 | **ได้ผลเหมือน (B)** ต่อ FRR แต่เก็บข้อมูลไว้ calibrate รอบหน้าได้ (ตรงกับที่เคยแนะนำไว้ตอนเริ่มสืบสวน — เก็บ signal ไว้ดูแม้ยังไม่เชื่อ) | ซับซ้อนกว่า (B) นิดหน่อยแต่ **น่าจะเป็นตัวเลือกที่ดีที่สุดถ้าจะเก็บทางเลือกในอนาคตไว้** — เห็น Moiré/Texture score ใน audit log ทุก request โดยไม่กระทบการตัดสินใจ |
| **threshold อื่นที่ "ทำให้ใช้งานได้"?** | — | **ไม่พบจากข้อมูลชุดนี้** best-threshold sweep (ทั้งสองทิศทาง, n=16 หลังตัดไฟล์ประมวลผล) ให้ error ต่ำสุดเท่ากับ trivial baseline เสมอสำหรับทั้ง Moiré และ Texture — ไม่มีตัวเลข cutoff ไหนในข้อมูลที่มีอยู่ตอนนี้ที่ทำให้ layer ไหนดีกว่าการไม่ใช้เลย | ไม่ได้แปลว่าไม่มีทางเป็นไปได้เลยในอนาคต (n เล็กมาก) แค่ไม่มีในข้อมูลตอนนี้ |

**ไม่ว่าจะเลือก (A)/(B)/(C) — 6 gate ในหัวข้อ 11.0 ยังอยู่เหมือนเดิมทั้งหมด ต้องแก้แยกถ้าต้องการผลจริงกับอาการที่รายงานมา**

### 11.3 ถ้า Fasnet แบกรับการตัดสินใจคนเดียว — พฤติกรรมตอนนี้เมื่อ Fasnet ใช้งานไม่ได้ (คำถามที่สำคัญที่สุด)

**Quote โค้ดตรง ๆ (face_service.py:275–291):**
```python
    # ── CRITICAL: fail-close if primary ML layer (Fasnet) is dead ──────────
    # Without Fasnet, only FFT layers remain — insufficient for high-DPI screens.
    fasnet_alive = layers.get("fasnet", {}).get("spoof_score") is not None
    if not fasnet_alive:
        _audit.error(
            "[COMBINED_SPOOF] Fasnet layer unavailable — failing CLOSED "
            "(rejecting frame). FFT-only defense is insufficient for "
            "high-DPI screen attacks."
        )
        return {
            "is_real": False,
            "combined_score": 1.0,
            ...
        }
```

**คำตอบตรง ๆ: FAIL-CLOSED — และเป็นพฤติกรรมที่มีอยู่แล้ว *ตอนนี้* ไม่ใช่ผลจากข้อเสนอในหัวข้อ 11.2** gate นี้อยู่ **ก่อน** ทั้ง 3 gate ใน 11.1 (line 275 มาก่อน 293/355/373) — ถ้า Fasnet หาหน้าไม่เจอ ฟังก์ชัน `return` ทันทีโดยไม่แม้แต่ไปถึง weighted-sum เลย **แปลว่า Moiré/Texture ไม่เคยมีโอกาส "โหวตค้าน" การ fail-close นี้อยู่แล้วแม้ตอนนี้ที่ยังไม่ได้ตัดออก** — Fasnet "แบกรับการตัดสินใจคนเดียว" ในเคสนี้อยู่แล้วโดยพฤตินัย ข้อเสนอในหัวข้อ 11.2 **ไม่เปลี่ยนพฤติกรรมของ gate นี้แม้แต่นิดเดียว**

**สิ่งที่เปลี่ยนจริงถ้าถอด Moiré/Texture ออก (option B/C) คือ กรณีที่ Fasnet ทำงาน** (`fasnet_alive=True`) — ตอนนี้ weighted sum ยังมี Moiré (30%) + Texture (15%) ร่วมโหวตอยู่ (แม้จะแยกแยะไม่ได้จริงตามที่วัดมา) ถอดออกแล้ว decision จะพึ่ง Fasnet (เดิม 15% → normalize ใหม่ ~60% ถ้า temporal ทำงานด้วย, เกือบ 100% ถ้า temporal ไม่มีข้อมูล) + ONNX (10%, เป็น noise ตามที่วัดในหัวข้อ 6/8) เกือบทั้งหมด

**เชื่อมกับตัวเลขที่วัดได้จริง:** ชุดทดสอบนี้ Fasnet หาหน้าไม่เจอ **5/19 (26%)** ของภาพ real (ก่อนตัดไฟล์ prep_images.py: 4 ไฟล์เพราะ crop แน่นเกิน + `real_lowdetail_dark15.jpg` เพราะมืดเกินไป — หลังตัด 6 ไฟล์ที่ผ่าน prep_images.py ออกแล้ว เหลือ **1/13 (~8%)** ที่ยังหาไม่เจอ คือ `dark15` เท่านั้น) **`dark15` สำคัญเป็นพิเศษ** เพราะเป็นเงื่อนไขที่เกิดขึ้นจริงในสนาม (ห้องเรียนแสงน้อย) ไม่ใช่ปัญหาจาก image-prep ของ user — **ถ้า Fasnet กลายเป็น layer หลักเกือบเดี่ยว ความเสี่ยงจริงคือ: ห้องเรียนแสงน้อยจะทำให้ผู้ใช้จริงโดน fail-close จาก gate นี้บ่อยขึ้น ไม่ใช่เพราะ Fasnet บอกว่าเป็น spoof แต่เพราะ Fasnet's face detector หาหน้าไม่เจอเลย** — เป็นความเสี่ยงเดียวกับที่ F-5/FRR-1 พูดถึง (แสงน้อย) แต่คนละกลไก (face-detection failure ไม่ใช่ threshold miscalibration)

### 11.4 enroll กับ checkin — เรียก stack เดียวกันจริงไหม

**ยืนยันแล้ว: เรียกฟังก์ชันเดียวกัน (`combined_spoof_score`) จริง** ผ่านคนละ wrapper:
- checkin: `api_checkin.py:257` → `combined_spoof_score(raw_frame)` ตรง ๆ, 1 ครั้ง/request
- enroll: `student.py:518` → `check_anti_spoof(face_images[idx])` (face_service.py:479–491, wrapper ที่เรียก `combined_spoof_score` แล้วคืน `result["is_real"]`) **เรียก 5 ครั้ง** (1 ต่อเฟรมที่ capture) ต้องผ่าน **≥4/5** (`MIN_SPOOF_PASS=4`, student.py:512) — เฟรมไหน exception หรือ `is_real=False` นับเป็น fail ทั้งคู่ (student.py:520–527)

**เพราะ enroll เรียก 5 ครั้งอิสระต่อกัน** ถ้า Fasnet หาหน้าไม่เจอ 2/5 เฟรม (เกิดได้ง่ายถ้าแสงน้อย/มุมเปลี่ยนระหว่าง capture — matches `dark15` เคส) แต่ละเฟรมนั้น fail-close ทันทีตาม 11.3 → `spoof_pass_count` ไม่ถึง 4 → enrollment ทั้งชุดถูกปฏิเสธ ทั้งที่อีก 3 เฟรมอาจจะผ่านสบาย ๆ **enroll จึงเสี่ยงกับ Fasnet-unavailable มากกว่า checkin โดยธรรมชาติของ >=4/5 vote เอง** (ไม่เกี่ยวกับหัวข้อ 11.2 เลย)

**สรุปคำตอบข้อ 4: การแก้ `combined_spoof_score` (weights หรือ gate) กระทบ enroll กับ checkin "เหมือนกัน" ตรงที่ทั้งคู่เรียกฟังก์ชันเดียวกัน** แต่ **ผลลัพธ์ปลายทางไม่เท่ากัน** เพราะ (1) enroll ผ่าน 6 standalone gate ในหัวข้อ 11.0 มากกว่า checkin (4 จุดเทียบ 2 จุด, threshold เข้มกว่าด้วย) ซึ่งไม่ถูกแตะโดยข้อเสนอนี้เลย และ (2) enroll เรียก `combined_spoof_score` 5 ครั้งผ่าน `>=4/5` vote ทำให้ fail-close ของ Fasnet (หัวข้อ 11.3) มีโอกาสสะสมทำให้ enrollment ล้มทั้งชุดได้ง่ายกว่า checkin ที่เรียกครั้งเดียว — **ไม่ต้องแก้ต่างกันสำหรับตัว `combined_spoof_score` เอง (โค้ดเดียวกัน) แต่ถ้าจะแก้ปัญหา FRR จริงจังต้องดูทั้ง 6 gate ในหัวข้อ 11.0 แยกทั้งสองไฟล์ ไม่ใช่แค่จุดเดียว**

---

~~ยังไม่ implement ตามคำสั่ง~~ — **อัปเดต 2026-08-25 (รอบแปด): user ตัดสินใจแล้ว — เลือก option (C)** สำหรับทั้ง 6 standalone gate (หัวข้อ 11.0) และ gate #2/#3 ใน `combined_spoof_score` (ลบ moire/texture ออกจาก `_layer_suspicious` check, ลบ gate #3 moire-alone≥0.85 ทิ้งทั้งก้อนเพราะซ้ำกับ raw check) ขอบเขต: **Moiré/Texture เท่านั้น** ไม่แตะ threshold ของ Fasnet/Temporal เก็บ log ทุกจุดไว้ (เอาแค่ `return 400`/hard-reject ออก ไม่เอา detector call หรือ log ออก) — รายละเอียด weight proposal และสถานะ implementation ดูหัวข้อ 12-13

---

## 12. Q-15 (ใหม่) — Anti-spoof policy กระจายอยู่ 7 จุด, 4 threshold ต่างกัน, ไม่มีจุดไหนเห็นภาพรวม

**บันทึกตามคำสั่ง user:** โครงสร้างที่เจอในหัวข้อ 11.0 (6 standalone gate + `combined_spoof_score` เอง = **7 จุดรวม**) เป็นปัญหาเชิงโครงสร้างที่ควรบันทึกแยกจาก FRR-1/F-15 เพราะเป็น**สาเหตุร่วม**ที่ทำให้การ tune threshold ในอดีตไม่เคยได้ผลถาวร

**หลักฐาน:**

| จุด | ไฟล์:line | threshold ที่ใช้ | เรียกกี่ครั้ง |
|---|---|---|---|
| combined_spoof_score (moire layer) | face_service.py:190 | `MOIRE_THRESHOLD_SINGLE=0.70` | ต่อ checkin/enroll-frame |
| combined_spoof_score (texture layer) | face_service.py:210 | `min_peaks=30` | ต่อ checkin/enroll-frame |
| checkin standalone Moiré | api_checkin.py:175 | `MOIRE_THRESHOLD_SINGLE=0.70` | 1/checkin |
| checkin standalone Texture | api_checkin.py:195 | `min_peaks=30` | 1/checkin |
| enroll final-submit Moiré | student.py:452 | `MOIRE_THRESHOLD=0.60` (**คนละค่ากับข้างบน**) | 1/enroll |
| enroll final-submit Texture | student.py:471 | `min_peaks=30`, **≥2/5 เฟรม** (คนละ voting rule) | 1/enroll |
| enroll liveness pre-check Moiré | student.py:996 | `MOIRE_THRESHOLD_SINGLE=0.70` | สูงสุด 8/enroll |
| enroll liveness pre-check Texture | student.py:1009 | `min_peaks=30` | สูงสุด 8/enroll |

**4 threshold ที่ไม่เท่ากัน** สำหรับ metric เดียวกัน (`MOIRE_THRESHOLD_SINGLE=0.70`, `MOIRE_THRESHOLD=0.60`, `min_peaks=30` แบบ single-frame, `min_peaks=30` แบบ ≥2/5-เฟรม) **กระจายอยู่ 7 code path** ไม่มี config เดียวหรือฟังก์ชันเดียวที่แสดง "นี่คือ policy การตัดสินสปูฟทั้งหมดของระบบ" — คนอ่านโค้ดต้องไล่ grep `detect_screen_moire`/`detect_screen_texture` ทั้ง repo ถึงจะเห็นภาพครบ (แบบที่ investigation นี้ต้องทำเองในหัวข้อ 11.0)

**อธิบาย git history ได้อีกชั้น:** 5 รอบ tune `MOIRE_THRESHOLD_SINGLE`/`MOIRE_THRESHOLD` (หัวข้อ 1) แต่ละรอบแก้ **ตัวแปรเดียว** ซึ่งกระทบแค่บางจุดใน 7 จุดนี้ (เช่นแก้ `MOIRE_THRESHOLD_SINGLE` กระทบ 3 จุด แต่ไม่กระทบ `MOIRE_THRESHOLD` ของ enroll final-submit เลย) — เป็นไปได้ที่บางรอบ "แก้แล้วดูเหมือนหาย" เพราะบังเอิญไปกระทบ code path ที่ user กำลังทดสอบอยู่ตอนนั้น ไม่ใช่เพราะปัญหาถูกแก้จริงทั้งระบบ

**บันทึกเป็น Q-15:**

| ID | เรื่อง | ไฟล์:line | ประเภท | ความยากแก้ | สถานะ |
|---|---|---|---|---|---|
| **Q-15** | Anti-spoof decision policy (Moiré/Texture) กระจายอยู่ 7 จุดใน 3 ไฟล์ ด้วย 4 threshold value ต่างกันสำหรับ metric เดียวกัน ไม่มีจุดเดียวที่แสดงภาพรวม policy — ทำให้ threshold-tuning ในอดีตแก้ได้แค่บางส่วนของระบบต่อครั้ง | api_checkin.py:172-210, student.py:449-485/994-1015, face_service.py:188-217 | quality/maintainability 🟢 | สูง (ต้อง centralize เป็นจุดเดียว หรืออย่างน้อย config เดียวที่ทุกจุดอ้างอิง) | บันทึกไว้ — เกี่ยวข้องกับการแก้ FRR-1/F-15 รอบนี้โดยตรง (หัวข้อ 13) แต่ยังไม่ centralize เต็มรูปแบบ |

---

## 13. Fasnet weight proposal — หยุดรอ user ตัดสินใจตามคำสั่ง (ยังไม่ implement)

รายละเอียดเต็มอยู่ในข้อความตอบ user โดยตรง (ไม่ซ้ำที่นี่เพื่อไม่ให้เอกสารกับคำตอบเพี้ยนกัน) — สรุปสั้น: เสนอ 3 ทางเลือกสำหรับ `fasnet`/`temporal`/`onnx` หลังตัด `moire`/`texture` ออกจาก weighted sum: (1) renormalize ตามสัดส่วนเดิม (ไม่แนะนำ — เท่ากับเชื่อ temporal เต็มที่ทั้งที่ยังไม่มีข้อมูลวัดเลย ซ้ำความผิดพลาดเดิมที่เคยเชื่อ Moiré), (2) evidence-weighted (fasnet สูงสุด, temporal ปานกลางเพราะ "ยังไม่วัด" ไม่ใช่ "วัดแล้วแย่", onnx ต่ำสะท้อน gap≈0 ที่วัดได้จริง — **ข้อเสนอหลัก**), (3) fasnet เดี่ยว (temporal/onnx = 0 ด้วย) — ระบุชัดว่าถ้าเลือกทางนี้ fail-close gate ในหัวข้อ 11.3 จะกลายเป็น anti-spoof stack ทั้งหมดของระบบ ไม่ใช่แค่ fallback

~~ยังไม่แก้ `SPOOF_WEIGHTS` หรือไฟล์ใดใน `app/` จนกว่า user จะเลือก~~ — **อัปเดต 2026-08-26: user เลือก evidence-weighted (fasnet=0.70, temporal=0.20, onnx=0.10) ผ่าน AskUserQuestion แล้ว implement แล้ว — ดูหัวข้อ 14**

---

## 14. Implementation record (2026-08-26)

**ขอบเขต: Moiré/Texture เท่านั้น — ไม่แตะ threshold ของ Fasnet/Temporal, ไม่แตะ `min_peaks=30`/`MOIRE_THRESHOLD*` ค่าไหนเลย** เอาแค่ "จะทำอะไรกับผลลัพธ์" (reject → log-only) กับ "น้ำหนักโหวต" (weight → skip-set) ออก ไม่แตะตัว detector หรือ threshold คำนวณ

### ไฟล์ที่แก้ (3 ไฟล์, ทั้งหมด compile ผ่านแล้ว)

**1. `app/services/face_service.py`**

`SPOOF_WEIGHTS` (บรรทัด ~29-45):
```python
SPOOF_WEIGHTS = {
    # 2026-08-26 rebalance (docs/review/10-moire-frr-investigation.md §13).
    # The prior comment here ("FFT layers more reliable than Fasnet") was
    # never measured and turned out backwards: Fasnet is the only layer with
    # confirmed separation (FULL SEPARATION, gap 0.7562, n=16). Temporal is
    # unmeasured — no burst-capture data exists — so its weight is cut, not
    # zeroed, pending its own validation round. Onnx is left at its prior
    # value (gap≈0 measured, but restructuring it was out of scope here).
    "fasnet":   0.70,
    "temporal": 0.20,
    "onnx":     0.10,
}
```
`moire`/`texture` ไม่อยู่ใน dict นี้อีกต่อไป (explicit skip-set ตามที่ user สั่ง แทนที่จะใส่ `0.0`) — `active_weights = dict(SPOOF_WEIGHTS)` จึงมีแค่ 3 key โดยธรรมชาติ ไม่ต้องแก้ loop คำนวณ weighted-sum เลย (บรรทัด `for layer_name, weight in normalized_weights.items()` วนแค่ 3 key ที่มีอยู่)

Gate #2 (`_layer_suspicious` hard-reject) — ตัด `moire_suspicious`/`texture_suspicious` ออกจาก `suspicious_count` (เหลือ `sum([temporal_suspicious, fasnet_suspicious])` — เดิม "≥2 จาก 4" ตอนนี้กลายเป็น "ทั้ง 2 ตัวที่เหลือต้องเห็นตรงกัน" โดยธรรมชาติ เพราะเหลือแค่ 2 ตัวให้โหวต)

Gate #3 (moire-alone ≥0.85) — **ลบทั้งบล็อกออก** (เคยเป็น near-duplicate ของ raw single-layer check อยู่แล้วตามที่ตรวจสอบไว้ในหัวข้อ 11.1)

`layers["moire"]`/`layers["texture"]` **ยังคำนวณและเก็บเหมือนเดิมทุกอย่าง** (fail-close เป็น `spoof_score=1.0` ตอน error เหมือนเดิม, ยังอยู่ใน `_audit.info` log บรรทัดสุดท้ายที่ log ทุก layer, ยังอยู่ใน `disagreements` loop) — **ไม่มีผลต่อ decision อีกต่อไป แต่ยัง log ครบ** ตรงตามคำสั่ง "computed and logged, excluded from decisions"

**2. `app/routes/api_checkin.py`** (2 จุด: gate #1-2 ในหัวข้อ 11.0)
- Moiré (บรรทัด ~172-182 เดิม): ลบ `if moire["is_screen"]: return jsonify(...), 400` ออก เหลือแค่คำนวณ + `_log.info(...)` — **นี่คือจุดที่ error message ต้นเรื่อง `"ตรวจพบหน้าจอมือถือ"` เคยมาจาก ตอนนี้ข้อความนี้ return ไม่ได้อีกแล้วจากจุดนี้**
- Texture (บรรทัด ~193-203 เดิม): เหมือนกัน ลบ reject ออก เหลือ log
- exception handling (decode/detector crash) **ไม่แตะ** — ยัง fail-close ด้วย 400 เหมือนเดิม (คนละเรื่องกับ "ตรวจพบหน้าจอ")

**3. `app/routes/student.py`** (4 จุด: gate #3-6 ในหัวข้อ 11.0)
- Enroll final-submit Moiré (~449-460): ลบ reject, เหลือ log — `moire_checked` (ตัวแปรที่ไม่เคยถูกใช้ที่ไหนอยู่แล้ว) เก็บไว้เหมือนเดิม ไม่แตะ (นอก scope)
- Enroll final-submit Texture (~469-478): ลบ reject (`screen_count >= 2`), เหลือ log
- Liveness pre-check Moiré (~984-991): ลบ reject, เหลือ log
- Liveness pre-check Texture (~997-1003): ลบ reject, เหลือ log
- exception handling ทั้ง 4 จุด **ไม่แตะ** เหมือนกัน — crash ยัง fail-close

### Smoke test ยืนยันก่อนรัน Docker

เรียก `combined_spoof_score()` จริงตรง ๆ (ไม่ผ่าน HTTP) กับ `real_01_sharp_face1.jpg`:
```
SPOOF_WEIGHTS = {'fasnet': 0.7, 'temporal': 0.2, 'onnx': 0.1}
layers: moire=1.0, texture=1.0 (ยังเป็น "สูงสุด" เหมือนก่อนแก้ — คำนวณเหมือนเดิม)
fasnet=0.052
combined_score=0.1698 → is_real=True  ← ก่อนแก้ไฟล์นี้จะ hard-reject จาก moire=1.0≥0.85 (gate #3 เดิม) ทันที
weights_used={'fasnet': 0.875, 'temporal': 0.0, 'onnx': 0.125}  (temporal ไม่มีข้อมูล multi-frame ใน smoke test เดี่ยว ๆ นี้ จึง redistribute)
```
ยืนยันว่า moire=1.0/texture=1.0 (ค่าที่เคย hard-reject) ไม่ทำให้ decision เปลี่ยนอีกต่อไป — `is_real=True` ถูกต้องตามที่ควรเป็นสำหรับภาพใบหน้าจริง

### ตาราง before/after — ทั้ง 7 จุดบังคับใช้ (จากหัวข้อ 11.0 + 11.1)

| # | จุด | ก่อนแก้ | หลังแก้ |
|---|---|---|---|
| 1 | checkin standalone Moiré | reject 400 "ตรวจพบหน้าจอมือถือ" ถ้า `is_screen` | log เท่านั้น, ไม่ reject จาก detection (crash ยัง 400) |
| 2 | checkin standalone Texture | reject 400 "ตรวจพบภาพจากหน้าจอ" ถ้า `is_screen` | log เท่านั้น |
| 3 | enroll final-submit Moiré | reject 400 "spoof_detected" ถ้า `is_screen` | log เท่านั้น |
| 4 | enroll final-submit Texture | reject 400 ถ้า `screen_count>=2/5` | log เท่านั้น |
| 5 | enroll liveness Moiré (≤8×) | reject (200, `is_real:false`) ถ้า `is_screen` | log เท่านั้น |
| 6 | enroll liveness Texture (≤8×) | reject ถ้า `is_screen` | log เท่านั้น |
| 7 | `combined_spoof_score` gate #2 | ≥2/4 layers suspicious (รวม moire/texture) → hard-reject | ≥2/2 (fasnet+temporal ต้องตรงกันทั้งคู่) → hard-reject |
| 7b | `combined_spoof_score` gate #3 | moire spoof_score≥0.85 เดี่ยว → hard-reject | **ลบออก** |
| 7c | `combined_spoof_score` weighted sum | fasnet 15% + moire 30% + temporal 30% + texture 15% + onnx 10% | fasnet 70% + temporal 20% + onnx 10% (moire/texture ไม่โหวต) |

### หัวข้อ 2 (enroll static analysis) — คำตอบตรงประเด็น

**Gate ทั้ง 6 จุดในหัวข้อ 11.0 — 2 จุดอยู่ checkin (#1-2), 4 จุดอยู่ enroll (#3-6)** ทั้งหมด return แค่ log แล้วผ่านต่อ ไม่มี early-return จาก detection อีกต่อไป (มีแค่ exception path ที่ยัง 400/500)

**`check_anti_spoof()` (student.py:518, → `combined_spoof_score`) ยังทำให้เฟรมล้มเหลวได้ — บนเงื่อนไข 3 อย่าง (เรียงตามผลกระทบจริง):**
1. **Fasnet หาหน้าไม่เจอ** (`fasnet_alive=False`) → fail-close ทันที (gate #275-291, ไม่เปลี่ยนจากก่อนหน้านี้เลย)
2. **Fasnet + Temporal ต้อง "สงสัย" พร้อมกันทั้งคู่** (`fasnet_suspicious` ≥0.30 **และ** `temporal_suspicious` ≥0.50) → hard-reject (gate #7 ใหม่)
3. **Weighted sum ≥0.50** — ในทางปฏิบัติ ผูกกับ Fasnet's `spoof_score` เป็นหลัก (70% weight) Temporal มีผลแค่ปรับ margin (20%) Onnx (10%) แทบไม่มีผลจริง (วัดได้ค่าคงที่ ~0.006-0.01 spoof_score แทบทุกภาพไม่ว่า real/spoof)

**ตอบตรง ๆ ตามที่ user ถาม: ไม่ใช่ "เฉพาะ Fasnet detection failure" เท่านั้น — แต่ Fasnet's เอง (ทั้งการหาหน้าเจอไหม และ spoof_score ที่ได้) เป็นตัวตัดสินหลักแทบทั้งหมดในทางปฏิบัติ** Temporal ยังมีบทบาทเสริม (margin + joint hard-reject) Onnx มีบทบาทแทบเป็นศูนย์จากข้อมูลที่วัดได้จริง (ไม่ได้ถูกถอดออกเชิงโครงสร้าง แต่ถอดออกโดยพฤตินัยจากตัวมันเอง)

**`MIN_SPOOF_PASS=4` (student.py:512) ยัง reachable โครงสร้างเดิมทุกอย่าง** ไม่ได้แตะเลย สิ่งที่เปลี่ยนคือ **โอกาสจะ "reachable" ในทางปฏิบัติสูงขึ้น** เพราะกลไกที่เคยทำให้ real face ติด false-reject (moire/texture ผ่าน gate #2/#3 เดิม) ถูกถอดออกแล้ว — เฟรมที่เคย fail จาก moire=1.0 (เช่นตัวอย่าง smoke test ข้างบน) จะผ่านได้แล้วถ้า Fasnet อ่านถูกว่าเป็นคนจริง

**ยืนยัน:** liveness pre-check gate (#5-6, เรียกได้ถึง 8 ครั้ง/enrollment) **ไม่ return 400 จาก detection อีกแล้ว แต่ยัง log ทุกครั้ง** (`_log(user_id, "liveness_moire"/"liveness_texture", ...)`) ตรงตามที่ขอยืนยัน

**คำเตือนที่ user ขอให้บันทึกไว้ชัด ๆ:** **enroll ในห้องแสงน้อยยังคงเป็น failure mode ที่เหลืออยู่ — แต่ตอนนี้มาจาก Fasnet's face-detection โดยตรง (dark15 case, ~8% ของ real samples ที่วัดได้) ไม่ใช่จาก Moiré/Texture อีกต่อไป** เป็นความเสี่ยงที่**มีอยู่แล้วก่อนการแก้รอบนี้** (หัวข้อ 11.3) การแก้รอบนี้ไม่ได้เพิ่มความเสี่ยงนี้ แต่ก็ไม่ได้ลดด้วย — ยังเป็น open risk ที่ยังไม่ถูกแตะ (นอก scope ที่สั่งไว้ชัดเจนว่า "ห้ามแตะ threshold ของ fasnet/temporal")

### Checkin smoke test (Docker) — ผลจริง 2026-08-26, **BLOCKED โดยบั๊กที่ไม่เกี่ยวกับ FRR-1 เลย**

`docker build` ใหม่เสร็จแล้ว (image `9acb62090e3f`, container `distracted_gagarin`) รัน `scripts/load_test.py smoke` ด้วย `real_01_sharp_face1_fit.jpg` + `real_05_sharp_face2_fit.jpg` (1000px LANCZOS resize, ไม่ crop) — ผล: `400 "ตรวจพบรูปถ่ายหรือหน้าจอ — กรุณาใช้ใบหน้าจริงเท่านั้น"` (spoof=true)

**ตอนแรกดูเหมือนจะเป็น FRR-2 (Fasnet หาหน้าไม่เจอ) แต่ตรวจ log จริงใน container แล้วไม่ใช่:**

```
ERROR smartcheck.enrollment — [FASNET] inference error: cannot import name 'test' from
partially initialized module 'tensorflow._api.v2.__internal__' (most likely due to a
circular import) (/usr/local/lib/python3.11/site-packages/tensorflow/_api/v2/__internal__/__init__.py)
ERROR smartcheck.enrollment — [COMBINED_SPOOF] Fasnet layer unavailable — failing CLOSED
```

**Fasnet ไม่ได้ "หาหน้าไม่เจอ" — มัน crash ก่อนจะได้รันด้วยซ้ำ** (`ImportError`, ไม่ใช่ `enforce_detection` no-face exception) เข้า path `fasnet_alive=False` fail-close เดียวกับ FRR-2 (face_service.py:289-305) ก็จริง แต่คนละสาเหตุโดยสิ้นเชิง — เป็น TensorFlow/tf-keras circular-import bug ใน Docker image เอง ไม่เกี่ยวกับภาพ ความละเอียด แสง หรือ Moiré/Texture fix ที่เพิ่งทำไปเลย

**ยืนยันว่า deterministic ไม่ใช่ race แบบสุ่ม:** เรียกซ้ำ 2 ครั้งห่างกัน ~6 นาที (08:15:21 และ 08:21:06) ได้ error message เดียวกันตัวต่อตัวทั้งคู่ — **อัปเดต (ดูหัวข้อ 16): สมมติฐาน "sys.modules poisoned จาก race แล้วพังซ้ำ" ที่เขียนไว้ตรงนี้เดิม ถูกหักล้างแล้วด้วยหลักฐานตรง — restart container สด ๆ + ยิง request แรกสุดครั้งเดียวก็พังทันที และ `docker run` เปล่า ๆ (ไม่มี Flask/gunicorn/thread ใด ๆ เลย) ก็พังเหมือนกันเป๊ะ ไม่ใช่ race เลยแม้แต่น้อย เป็น build artifact ที่หายไปจริง (ดูหัวข้อ 16 สำหรับ root cause ที่ยืนยันแล้ว)**

**เทียบกับ local venv (Windows, คนละ environment จาก Docker/Linux):** เรียก `combined_spoof_score()` ตรงผ่าน base64 round-trip เดียวกับที่ `api_checkin.py` ทำ กับไฟล์ทั้งสอง (`real_01_sharp_face1_fit.jpg`, `real_05_sharp_face2_fit.jpg`) ได้ `is_real=True` ทั้งคู่ ทำซ้ำ 3 รอบต่อไฟล์ ได้ผลเดิมทุกครั้ง (`fasnet_spoof=0.0644`/`0.0234` — ชัดเจนว่าเป็นหน้าจริง) **สรุปว่าทั้งสองไฟล์ผ่านทั้ง `server_validate_frame` และ Fasnet's ความต้องการจริงอยู่แล้ว — ปัญหาไม่ใช่การเตรียมภาพ**

**ข้อเท็จจริงที่แก้ premise เดิมของคำถาม:** ไม่มี resize server-side ระหว่าง decode กับ Fasnet เลยในโค้ด — `server_validate_frame` เช็ค `1920x1080` เป็นแค่ upper-bound reject gate (`w > 1920 or h > 1080` → reject, face_service.py:805-807) ไม่ใช่ downscale operation และ `_decode_image` (บรรทัด 440-449) ก็แค่ `cv2.imdecode` ตรง ๆ ไม่มี resize แทรก ดังนั้น "endpoint บังคับ downscale ภาพ" ไม่ตรงกับโค้ดจริง — ภาพ 562×1000 เข้าไปเป็น 562×1000 เป๊ะ ไม่มีขั้นตอนไหนย่อขนาดเลย

**สถานะ:** smoke test **ยังไม่สามารถยืนยัน FRR-1 fix ได้จริง** เพราะ request ไม่เคยไปถึงจุดที่ Moiré/Texture score มีผลต่อ decision เลย (Fasnet crash ตัดจบก่อน) — ต้องแก้บั๊ก TensorFlow/tf-keras import นี้ก่อนถึงจะ smoke test ผ่านได้จริง **ยังไม่แตะ requirements.txt/Dockerfile/threshold ใด ๆ ตามคำสั่ง — รายงานเท่านั้น รอการตัดสินใจ**

---

## 15. FRR-2 (ใหม่) — Fasnet face-detection failure ในห้องแสงน้อย: FRR exposure ที่ใหญ่ที่สุดที่เหลืออยู่ (2026-08-26)

**แยก series/หัวข้อจาก FRR-1 เพราะคนละ layer คนละกลไก:** FRR-1 คือ Moiré/Texture *metric* คำนวณผิด (threshold-calibration problem บน full-frame FFT — แก้ไปแล้ว, ดูหัวข้อ 9-10, 14) FRR-2 คือ Fasnet's face **detector** (`DeepFace.extract_faces(..., enforce_detection=True)`, เรียกจาก `_run_fasnet_antispoof`, face_service.py:149-174) หาหน้าไม่เจอเลยเมื่อภาพมืดเกินไป — ไม่ใช่ threshold ผิดค่า แต่เป็นข้อจำกัดของ detector เอง **และเป็นปัญหาที่มีอยู่ก่อนการสืบสวน FRR-1 ทั้งหมด** ไม่ใช่สิ่งที่การถอด Moiré/Texture ออกจาก vote (หัวข้อ 14) สร้างขึ้นมาใหม่ — แค่ทำให้เด่นขึ้นเพราะตอนนี้ Fasnet แบกภาระการตัดสินใจเกือบทั้งหมด (ดูหัวข้อ 11.3, 620).

**หลักฐานเชิงตัวเลข:** วัดจากชุด unprocessed real samples 13 ไฟล์ (ตัด 6 ไฟล์ที่ผ่าน `prep_images.py` ออกแล้ว ดูหัวข้อ 8 ท้าย) — `real_lowdetail_dark15.jpg` (ห้องแสงน้อยมาก ~15 lux) เป็น**เพียงไฟล์เดียว**ที่ Fasnet หาหน้าไม่เจอ = **1/13 (~8%)** (ดูหัวข้อ 11 ท้าย, บรรทัด 487) — n เล็กเกินจะสรุป prevalence แน่นอน แต่เป็นเงื่อนไขที่เกิดขึ้นได้จริงในสนาม (ห้องเรียนปิดไฟดู projector) ไม่ใช่ image-prep artifact แบบไฟล์ที่ถูกตัดออก.

### กลไก: เมื่อ Fasnet หาหน้าไม่เจอ เกิดอะไรขึ้น

`_run_fasnet_antispoof()` เรียก `enforce_detection=True` — หาหน้าไม่เจอ → raise → catch → return `(None, None)`. `combined_spoof_score()` เห็น `fasnet_spoof is None` → `active_weights["fasnet"] = 0.0` (บรรทัด 269-270) → `fasnet_alive = False` → **fail-close ทันที ไม่มีเงื่อนไขอื่นให้ตรวจต่อ** (บรรทัด 289-305):
```python
if not fasnet_alive:
    ...
    return {"is_real": False, "combined_score": 1.0, ...,
            "disagreements": ["fasnet_unavailable_fail_close"]}
```
Temporal/ONNX ไม่ได้โหวตเลยในกรณีนี้ — ตัดสินจบที่ layer เดียว ไม่ว่า layer อื่นจะเห็นว่าเป็นหน้าจริงชัดแค่ไหนก็ตาม.

### Checkin: single unconditional reject

`api_checkin.py:242-262` เรียก `combined_spoof_score(raw_frame)` ครั้งเดียวต่อ 1 คำขอ — ถ้า Fasnet หาหน้าไม่เจอในเฟรมนั้น จะ return ทันที:
```python
return jsonify({
    "ok": False,
    "error": "ตรวจพบรูปถ่ายหรือหน้าจอ — กรุณาใช้ใบหน้าจริงเท่านั้น",
    "spoof": True,
    "retry_face": True,
}), 400
```
**ผู้ใช้เห็น:** "ตรวจพบรูปถ่ายหรือหน้าจอ — กรุณาใช้ใบหน้าจริงเท่านั้น" (แปล: "Photo or screen detected — please use a real face only") พร้อม `retry_face: True` ให้ถ่ายใหม่ได้ — **ข้อความไม่บอกเลยว่าปัญหาคือแสงน้อย** นักศึกษาในห้องมืด (ปิดไฟดู projector) จะเห็นข้อความกล่าวหาว่ากำลังใช้ "รูปถ่ายหรือหน้าจอ" ทั้งที่เป็นหน้าจริง — ไม่มีทางรู้จากข้อความนี้ว่าต้องเปิดไฟ/หาที่แสงดีกว่า ถ่ายใหม่กี่ครั้งก็เจอปัญหาเดิมถ้าห้องยังมืดเท่าเดิม.

### Enroll: แย่กว่า — `MIN_SPOOF_PASS=4/5` ทำให้ 2 เฟรมมืดพอจะทำให้ enrollment ทั้งชุดตก

`student.py:501-528` (final `/api/enroll` submit) เรียก `check_anti_spoof()` แยกอิสระทีละเฟรมจาก 5 เฟรม ต้องผ่าน**อย่างน้อย 4/5** (`MIN_SPOOF_PASS = 4`, บรรทัด 503):
```python
if spoof_pass_count < MIN_SPOOF_PASS:
    return jsonify({
        "status":       "spoof_detected",
        "failed_frame": first_spoof_frame,
        "reason":       "minifasnet",
        "message":      "ตรวจพบการปลอมแปลงใบหน้า — กรุณาใช้ใบหน้าจริงเท่านั้น",
    }), 400
```
5 เฟรมถ่ายในช่วงเวลาสั้น ๆ ต่อกัน (burst capture) — ถ้าห้องมืด **ทุกเฟรมมีโอกาสมืดใกล้เคียงกันหมด ไม่ใช่แค่เฟรมเดียว** แค่ 2/5 เฟรม Fasnet หาหน้าไม่เจอ (`spoof_pass_count` เหลือ 3) ก็พอทำให้**enrollment ทั้งชุดตกทันที** แม้อีก 3 เฟรมจะผ่านสบาย ๆ (สอดคล้องกับที่วิเคราะห์ไว้แล้วในหัวข้อ 11.3 บรรทัด 495 ว่า enroll เสี่ยงกว่า checkin โดยธรรมชาติของ ≥4/5 vote) — นี่คือความเสี่ยงที่**แย่กว่า checkin เชิงโครงสร้าง** เพราะ checkin แค่ 1 เฟรมตก = ถ่ายใหม่ได้ทันที แต่ enroll ต้องเริ่ม burst 5 เฟรมใหม่ทั้งชุดถ้าตก.

**ผู้ใช้เห็น:** "ตรวจพบการปลอมแปลงใบหน้า — กรุณาใช้ใบหน้าจริงเท่านั้น" (แปล: "Face spoofing detected — please use a real face only") — **ไม่บอกเรื่องแสงเช่นกัน** และคำว่า "ตรวจพบการปลอมแปลง" ฟังดูรุนแรงกว่าข้อความ checkin ("หน้าจอ/รูปถ่าย") — นักศึกษาที่ลงทะเบียนใบหน้าครั้งแรกในห้องแสงน้อยจะโดนข้อความกล่าวหาว่า "ปลอมแปลง" ทั้งที่จริงเป็นปัญหาแสง.

### บริบทเพิ่มเติม: เส้นทาง enrollment มี gate ก่อนหน้า final-submit อีก 3 จุด (`/api/spoof_check`, เรียกได้ถึง 8 ครั้ง) ที่ก็เจอ Fasnet-fail แบบเดียวกันได้ — สำรวจ `enrollment_flow.js` แล้วมีแค่ 1 ใน 6 จุดที่บอกเรื่องแสง

| จุด | ไฟล์:line | ข้อความที่ผู้ใช้เห็น | บอกเรื่องแสงไหม |
|---|---|---|---|
| Step 2 pre-challenge (≤5 ครั้ง) | enrollment_flow.js:707 | `(server message \|\| 'ตรวจพบภาพปลอม') + " — กรุณาลองใหม่ (N/5)"` — server message = `"ตรวจพบการปลอมแปลง"` (face_service.py:554) | ❌ ไม่บอก |
| Step 2 หมดโควตา (5 ครั้ง) | enrollment_flow.js:703 | `"ลองเกินจำนวนครั้งที่กำหนด — กรุณารอ 30 วินาที"` | ❌ ไม่บอก |
| Step 3 post-challenge (ครั้งเดียว ไม่มีโควตา retry) | enrollment_flow.js:772-773 | `"ตรวจพบการสลับใบหน้า — กรุณาเริ่มใหม่"` → บังคับ `fullRestart()` ทันที | ❌ ไม่บอก (และคำว่า "สลับใบหน้า"/face-swap รุนแรงกว่าเดิมอีก) |
| Step 4 capture loop (ทีละครั้ง จนกว่าจะครบโควตา) | enrollment_flow.js:1156-1159 | `` `ตรวจพบความผิดปกติ — เหลือโอกาสอีก ${remaining} ครั้ง กรุณาใช้ใบหน้าจริง` `` | ❌ ไม่บอก |
| **Step 4 หมดโควตารวม** (`MAX_STEP4_SPOOF_TOTAL=5`) | enrollment_flow.js:1151 | `"ตรวจพบความผิดปกติหลายครั้ง — กรุณาเริ่มใหม่ในสภาพแวดล้อมที่มีแสงเพียงพอ"` | ✅ **จุดเดียวที่บอก** |
| Final `/api/enroll` submit (`MIN_SPOOF_PASS` gate) | student.py:527 | `"ตรวจพบการปลอมแปลงใบหน้า — กรุณาใช้ใบหน้าจริงเท่านั้น"` | ❌ ไม่บอก |

**สรุป:** รวม checkin เป็น 7 จุดที่ Fasnet-fail ทำให้ผู้ใช้เห็นข้อความ (checkin ×1 + enroll ×6) มีแค่ **1 จุดเดียว** ที่บอกใบ้เรื่องแสง และเป็นจุดที่ลึกที่สุด/เจอยากที่สุด (ต้องพลาดสะสมครบ 5 ครั้งใน Step 4 ถึงจะเห็น) ส่วนเส้นทางที่ผู้ใช้เจอบ่อยที่สุด — checkin, final enroll submit, ทุกครั้งแรกของแต่ละ retry — ล้วนบอกว่า "ปลอมแปลง/หน้าจอ/สลับใบหน้า" ไม่ใช่ "แสงไม่พอ" นักศึกษาในห้องเรียนมืดจึงแทบไม่มีทางรู้จากข้อความจริงที่เห็นว่าทางแก้คือเปิดไฟ/หาที่แสงดีกว่า.

### สถานะ

**ยังไม่แก้อะไรทั้งสิ้น — บันทึกเป็น finding แยกตามคำสั่ง user, ห้ามแตะ threshold/detection settings ของ Fasnet ในรอบนี้.** ตัวเลือกที่เป็นไปได้ (ยังไม่เสนอรายละเอียด/ยังไม่ implement เพราะ user ขอแค่บันทึกไว้ก่อน): (a) ปรับข้อความ error ให้บอกเรื่องแสงชัดเจนขึ้นในทุกจุดที่ Fasnet fail-close ตรง ๆ ได้ — ต้นทุนต่ำ ไม่กระทบ security logic ใด ๆ (แค่เปลี่ยนข้อความ), (b) เพิ่ม pre-check ความสว่างของภาพก่อนส่งเข้า Fasnet (ต้นทุนสูงกว่า ต้องออกแบบ threshold ใหม่).

---

## 16. F-16 (ใหม่, **severity-critical**) — Fasnet พังสมบูรณ์ใน Docker image ทุก request: fail-close message ปิดบัง 100% outage ให้ดูเหมือน "ตรวจพบการปลอมแปลง" ปกติ (2026-08-26)

**อัปเดตหัวข้อ 14:** สิ่งที่ดูเหมือนจะเป็น FRR-2 firing ในการ smoke test จริง ๆ แล้วไม่ใช่เลย — สืบจน root cause แล้ว เป็นบั๊กที่ **severity-critical กว่า FRR-2 หลายเท่า และเป็นคนละกลไกสิ้นเชิง** ถึงแม้จะแสดงอาการผ่าน fail-close path เดียวกัน (`fasnet_alive=False`, face_service.py:289-305)

**Cross-ref FRR-2 (หัวข้อ 15):** ทั้งสอง finding ชี้ไปที่ defect เดียวกัน — **fail-close message ไม่แยกแยะระหว่าง "ตรวจพบการปลอมแปลงจริง" กับ "ระบบตรวจสอบเองพังหรือใช้งานไม่ได้"** FRR-2 คือกรณี edge (ห้องมืด, ~8% ของ real samples) ที่ระบบยังทำงานถูกต้องแต่ detector มีข้อจำกัด — ผู้ใช้เห็นข้อความเข้าใจผิดเป็นครั้งคราว **F-16 คือ defect เดียวกันที่ maximum severity: ระบบไม่ได้ทำงานเลย (0% ของ checkin ผ่านได้ ไม่ว่าภาพจะเป็นอย่างไร) แต่ข้อความที่ผู้ใช้เห็นเหมือนกันทุกประการกับกรณีปกติที่ detector ทำงานถูกต้องแล้วเจอ spoof จริง** ไม่มีทางแยกจาก client-side เลยว่าเจอกรณีไหน — นี่คือสิ่งที่ทำให้ user เตรียมภาพผิดทาง 3 รอบก่อนจะมีใครไปดู container log จริง

### หลักฐาน (เรียงตามลำดับที่ตรวจ)

**1. Bare reproduction — ไม่มี Flask/gunicorn/thread ใด ๆ เลย:**
```
docker run --rm --entrypoint python smartcheck -c "from deepface import DeepFace"
```
พังทันทีด้วย traceback เดียวกันเป๊ะกับที่เห็นใน production log:
```
File ".../deepface/DeepFace.py", line 15, in <module>
    import tensorflow as tf
  File ".../tensorflow/__init__.py", line 48, in <module>
    from tensorflow._api.v2 import __internal__
  File ".../tensorflow/_api/v2/__internal__/__init__.py", line 22, in <module>
    from tensorflow._api.v2.__internal__ import test
ImportError: cannot import name 'test' from partially initialized module...
```
**ตัดความเป็นไปได้เรื่อง concurrency/thread-race ออกทั้งหมด** — พังจาก `import tensorflow` ธรรมดา ไม่มี thread สองตัวชิงกันเลยแม้แต่น้อย

**2. Fresh-container determinism (ตามที่ user สั่งให้ทดสอบ):** `docker restart` container (ยืนยันจาก log ว่า worker PID เปลี่ยนใหม่จริง) แล้วยิง request เดียว — **พังตั้งแต่ request แรกสุดของ process ใหม่เอี่ยม** เหมือนกันทุกประการ (`08:36:26 ERROR ... [FASNET] inference error: ...`) **หักล้างสมมติฐาน "sys.modules ถูก poison จาก import ที่พังครั้งแรก แล้วพังซ้ำทุกครั้งหลังจากนั้น" ที่เขียนไว้ในหัวข้อ 14 เดิม — ไม่ใช่ poison, เป็น broken-by-design ตั้งแต่ก่อน process เริ่มด้วยซ้ำ**

**3. ยืนยัน directory ที่หายไปจริง:**
```
docker run --rm --entrypoint sh smartcheck -c "ls tensorflow/_api/v2/__internal__/"
```
`__init__.py` ของ `tensorflow/_api/v2/__internal__/` import 12 submodule เรียงตามตัวอักษร: `graph_util, mixed_precision, monitoring, nest, ops, saved_model, smart_cond, test, tf2, tracking, train, types` — **listing จริงมีครบ 11 จาก 12 ตัว ขาดแค่ `test/` ตัวเดียว** อยู่ตรงตำแหน่งที่ควรอยู่พอดี (ระหว่าง `smart_cond` กับ `tf2`) — ไม่ใช่ความเสียหายแบบสุ่ม เป็นการ**ลบไปตรง ๆ ทั้งโฟลเดอร์**

### Root cause: Dockerfile ลบ `tensorflow/_api/v2/__internal__/test/` เอง ด้วยความตั้งใจดี

```dockerfile
# Remove pycache and test files to reduce image size
RUN find /usr/local/lib/python3.11 -depth -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11 -depth -type d -name tests -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11 -depth -type d -name test -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11 -name "*.pyc" -delete 2>/dev/null || true
```
บรรทัด `-type d -name test -exec rm -rf` แมทช์**ทุกโฟลเดอร์ที่ชื่อ `test` เป๊ะ ๆ ใต้ `site-packages` ทั้งหมด** ไม่จำกัดว่าเป็น test suite ของ package ไหน — และ `tensorflow` ใช้ชื่อโฟลเดอร์ `test` เป็น**ส่วนหนึ่งของ public API จริง** (`tf.__internal__.test`) ไม่ใช่ unit test ที่ bundle มาเฉย ๆ โดนลบไปด้วยความเข้าใจผิดของ command

**ทำไม build-time verification ถึงผ่าน แต่ runtime พัง — เพราะลำดับ RUN layer:**
1. `pip install` + Step 4 (`from deepface import DeepFace` verification) — **ผ่าน**, ตอนนี้ `test/` ยังอยู่
2. Pre-download model weights + Fasnet smoke test (`DeepFace.extract_faces(..., anti_spoofing=True, enforce_detection=False)`) — **ผ่าน** (หรืออย่างน้อยไม่ raise เพราะเงื่อนไข raise แคบมาก ดูหัวข้อถัดไป), ตอนนี้ `test/` ก็ยังอยู่
3. `RUN find ... -name test -exec rm -rf` (**"Remove pycache and test files to reduce image size"**) — ลบ `test/` ทิ้ง **หลังจาก** ทั้ง 2 checks ข้างบนผ่านไปแล้ว
4. `COPY . .` + ตั้ง entrypoint — image สุดท้ายที่ ship ออกไปมี `test/` หายไปแล้วถาวร

ทุก container ที่รันจาก image นี้จึงพังตั้งแต่ request/call แรกสุด — ไม่ใช่ environment-specific, ไม่ใช่ race, เป็นผลของลำดับขั้นตอน build เอง

**ช่องโหว่ที่ซ้อนอยู่ใน Fasnet smoke test เอง (ทำไมมันไม่จับบั๊กนี้ได้ตั้งแต่ตอน build แม้จะรันก่อน cleanup):**
```python
try:
    faces = DeepFace.extract_faces(..., anti_spoofing=True, enforce_detection=False)
    print(f"Fasnet smoke test OK — {len(faces)} faces detected on random image")
except Exception as e:
    err_str = str(e)
    if "Numpy is not available" in err_str or "cuInit" in err_str:
        raise  # เฉพาะ 2 เงื่อนไขนี้เท่านั้นที่ทำให้ build fail
    print(f"Fasnet smoke test ran (no face expected on random image): {err_str[:100]}")
```
เงื่อนไข `raise` แคบมาก เจาะจงแค่ 2 substring — ข้อความ error แบบ `ImportError: cannot import name 'test'...` ไม่ตรงกับทั้งสอง จึงแค่ print แล้วผ่านต่อไปเงียบ ๆ **เท่ากับว่าแม้จะย้าย cleanup step ไปก่อน smoke test แล้ว smoke test เองก็ยังไม่มีทางจับบั๊กแบบนี้ได้อยู่ดี — ต้องแก้ทั้ง 2 จุดพร้อมกัน** (ดูหัวข้อ fix options)

### Q1 — กระทบ Railway ไหม

`railway.json` ยืนยันว่า Railway build ด้วย `"builder": "DOCKERFILE"` — Dockerfile เดียวกันเป๊ะ รันเป็น RUN layer เรียงลำดับเดียวกัน บน `linux/amd64` เดียวกับที่ local Docker Desktop build ให้ (เช็คแล้ว: `docker version` และ `uname -m` ในคอนเทนเนอร์ตรงกัน) เพราะบั๊กนี้พิสูจน์แล้วว่าเกิดจาก**ไฟล์ที่หายไปในตัว image เอง** ไม่เกี่ยวกับ host/thread/env var ใด ๆ (bare `docker run` ก็พัง) จึงมั่นใจสูงว่า Railway build เดียวกันนี้จะพังเหมือนกันทุกประการ

**วิธีเช็คโดยไม่ redeploy:** อ่าน log ที่มีอยู่แล้ว (`railway logs` หรือ dashboard) หา `[FASNET] inference error` / `[COMBINED_SPOOF] Fasnet layer unavailable — failing CLOSED` — ถ้าเคยมีใคร checkin ผ่าน production มาก่อน log จะมี signature นี้อยู่แล้วโดยไม่ต้อง deploy ซ้ำเลย **agent ไม่มี Railway CLI/credentials ใน environment นี้ ตรวจให้ไม่ได้เอง — ต้อง user เช็คเอง** ข้อมูลเสริมที่ทำให้เรื่องนี้น่ากังวลกว่าที่คิด: `/health` (`app/__init__.py:176-178`) return `"ok", 200` เฉย ๆ ไม่แตะ Fasnet/DeepFace เลย — Railway healthcheck จะรายงานว่า deploy นี้ "healthy" ตลอดเวลาที่ checkin พังอยู่ 100%

### Q2/Q3 คำตอบสรุป (ดูหลักฐานเต็มด้านบน)

- **Q2 (fresh-container):** พังตั้งแต่ request แรกสุดของ process ใหม่ — deterministic 100% ไม่ใช่ race/poison
- **Q3 (build ผ่าน runtime พัง):** ลำดับ RUN layer — cleanup step ที่ลบ `test/` รันหลัง verification steps ทั้งหมด ไม่ใช่ concurrent import, ไม่ใช่ TF_USE_LEGACY_KERAS (env var นี้ set เหมือนกันทั้ง build-time verification และ runtime ผ่าน Dockerfile `ENV` ที่ persist ข้าม RUN layer ทั้งหมด — เช็คแล้วไม่ใช่ตัวแปร), ไม่ใช่ import order ต่างกันระหว่าง build-check กับ request path (bare `python -c "from deepface import DeepFace"` พังเหมือนกับที่ request path พังเป๊ะ)

### ทำไมถึงเป็น severity-critical ไม่ใช่แค่ deploy blocker

1. **Silent 100% outage** — ไม่ใช่ FRR แบบ edge case, ไม่มี checkin ไหนผ่านได้เลยไม่ว่าภาพจะดีแค่ไหน
2. **Message ปิดบัง**: response ที่ client เห็น (`"ตรวจพบรูปถ่ายหรือหน้าจอ..."`, 400, `spoof:true`) เหมือนกับกรณีตรวจพบ spoof จริงทุกประการ — ไม่มีทางแยกจาก client หรือแม้แต่จาก log ระดับ INFO ปกติ ต้องขุด log ระดับ ERROR ที่ระบุ `[FASNET] inference error` โดยเฉพาะถึงจะเห็นความต่าง
3. **Health check ไม่ครอบคลุม** — `/health` ไม่ตรวจ Fasnet เลย ระบบ monitoring ที่มีอยู่ (ถ้าอิง `/health` เป็นหลัก) จะไม่มีทาง alert เรื่องนี้เลย
4. **เสี่ยงเกิดขึ้นจริงบน production แล้ว** — Railway ใช้ Dockerfile เดียวกัน และไม่มีอะไรใน host ที่จะเลี่ยงบั๊กนี้ได้ ถ้า deploy ล่าสุดสร้างจาก Dockerfile เวอร์ชันนี้ checkin ทั้งระบบอาจพังมาระยะหนึ่งแล้วโดยไม่มีใครรู้

### ทางเลือกการแก้ (เสนอ trade-off เท่านั้น — ยังไม่ implement ตามคำสั่ง)

**A. แก้ root cause ตรง ๆ — จำกัดขอบเขตของ cleanup `find`:**
เปลี่ยนจาก blanket match ทุกโฟลเดอร์ชื่อ `test`/`tests` เป็นการลบเจาะจง package ที่รู้ว่าปลอดภัย (เช่น scope เฉพาะ `site-packages/<pkg>/tests` ของ package ที่ไม่ใช่ tensorflow) หรือ exclude `tensorflow` ทั้ง tree ออกจาก pattern นี้ไปเลย — **ต้นทุนต่ำที่สุด แก้ปัญหาตรงจุด** แต่เสี่ยงพลาด package อื่นที่ใช้ pattern เดียวกัน (เช่น `torch`, `onnxruntime` อาจมีโฟลเดอร์ชื่อ `test` ที่จำเป็นเหมือนกันก็ได้ — ต้อง audit ทีละ package ก่อนเชื่อว่าปลอดภัย)

**B. แก้ลำดับ + แก้เงื่อนไข smoke test พร้อมกัน (แนะนำเป็นแนวทางที่ robust ที่สุด แต่ต้องทำ 2 จุด):**
ย้าย "Remove pycache and test files" RUN block ไปไว้**ก่อน** ไม่ใช่หลัง verification/smoke-test steps และเปลี่ยนเงื่อนไข `raise` ใน Fasnet smoke test จาก 2 substring แคบ ๆ เป็น **fail build บน exception ใด ๆ ที่ไม่คาดคิด** (whitelist เฉพาะ "no face detected" ซึ่งเป็นผลปกติของภาพ random noise ไม่ใช่ blacklist 2 คำ) — ถ้าทำครบทั้งคู่ build จะ fail ทันทีตอนสร้าง image แทนที่จะ ship image พังไปเงียบ ๆ **ทำให้ "fail-close ปิดบัง outage" ไม่มีทางเกิดขึ้นได้อีก เพราะจะไม่มี image ที่พังหลุดออกไปตั้งแต่ต้น**

**C. Eager import ตอน app startup แทน lazy import ในทุก request (ตามที่ user ถามเจาะจง):**
ปัจจุบัน `from deepface import DeepFace` อยู่ *ข้างใน* `_run_fasnet_antispoof()`/`extract_embedding()`/`spoof_check_with_embedding()` (function-local, lazy) — เรียกครั้งแรกก็ต่อเมื่อ request แรกมาถึงจุดนั้นจริง ๆ
- **ข้อดี:** ถ้า import พังตั้งแต่ตอน app boot (module-level import หรือเรียกใน `create_app()`) gunicorn worker จะ crash ตั้งแต่ startup — Railway restart policy เป็น `ON_FAILURE` (`railway.json`) จะเห็น container crash-loop ทันที เป็นสัญญาณที่ชัดกว่า 400 response ที่หน้าตาเหมือน spoof ปกติมาก — ตรงกับที่ user ต้องการ (ไม่ปิดบัง outage)
- **เกี่ยวกับ `--preload` โดยตรงตามที่ user ถาม:** `gunicorn --preload` โหลด WSGI app ใน master process **ก่อน** fork worker แล้วให้ worker share memory หน้าเดิมผ่าน copy-on-write — แต่ **ใช้ได้เฉพาะกับสิ่งที่ import ตอน module-load เท่านั้น** ตอนนี้ `--workers 1` (Dockerfile:95) จึงไม่มีประโยชน์จาก `--preload` อยู่แล้วไม่ว่าจะ eager หรือ lazy (worker เดียวไม่มีใครแชร์ด้วย) **แต่ถ้าจะขยับไป `--workers N>1` ตามที่ F-9/06-performance.md พูดถึงไว้** lazy import แบบตอนนี้จะทำให้ **แต่ละ worker โหลด TF/DeepFace/Facenet512 model เป็นสำเนาของตัวเองอิสระ** ตอน request แรกของแต่ละ worker (memory ไม่ share กันเลย, คูณ N เท่า) — เปลี่ยนเป็น eager import ที่ module-level (หรือเรียก 1 ครั้งใน `create_app()` ก่อน fork) + เปิด `--preload` จะทำให้ N worker share memory หน้าเดิมของ model ผ่าน COW ได้จริง **ลด per-worker memory ได้อย่างมีนัยสำคัญเมื่อ scale เกิน 1 worker** — ตอบคำถาม user ตรง ๆ ว่า "ใช่ eager import คือ precondition ที่ทำให้ `--preload` มีประโยชน์กับ model sharing ได้เลย lazy import แบบปัจจุบันทำให้ `--preload` ไม่มีผลอะไรกับ TF/DeepFace เลย"
- **ต้นทุน (เชิงคุณภาพ ยังไม่ได้วัดจริง — ต้อง flag ตรง ๆ ว่าเป็นการประมาณ ไม่ใช่ตัวเลขวัดจริง):** boot time เพิ่มขึ้นแน่นอน (สังเกตจาก smoke test ครั้งแรกที่เจอ error ก็ยังใช้เวลาถึง ~7.9s ก่อนจะ error — import+partial-init ของ TF เพียงอย่างเดียวก็มี overhead หลักวินาทีอยู่แล้ว) `healthcheckTimeout: 600` (railway.json) มี headroom เหลือเฟือสำหรับเรื่องนี้ไม่น่าเป็นปัญหา ส่วน memory baseline จะขึ้นสูงขึ้นตั้งแต่ boot แทนที่จะขึ้นทีหลังตอน request แรก — แต่ในแอปนี้ anti-spoof เป็น core ของ feature หลัก (checkin) อยู่แล้ว การขยับเวลาที่ cost เกิดขึ้น (จาก "request แรกของ user" ไปเป็น "ตอน container boot") ไม่ได้เพิ่ม total memory footprint ที่จำเป็นต้องใช้อยู่ดี แค่เปลี่ยนจังหวะที่จ่าย

**D. เพิ่ม `/health` (หรือ endpoint แยก) ให้ตรวจ Fasnet จริง ไม่ใช่แค่ static "ok":**
เรียก dummy inference สั้น ๆ (เหมือนที่ build-time smoke test ทำ) เป็นส่วนหนึ่งของ healthcheck หรือทำเป็น readiness probe แยกจาก liveness — ทำให้ Railway (หรือ monitoring อื่น) เห็นว่า deploy "unhealthy" จริงถ้า Fasnet ใช้งานไม่ได้ แทนที่จะรายงาน "ok" ทั้งที่ core feature พังอยู่ — **เป็น defense-in-depth แยกจาก A/B/C ทำคู่กันได้**

**E. แยก error message/HTTP status ระหว่าง "Fasnet unavailable" กับ "Fasnet says spoof" (ตรงกับที่ cross-ref FRR-2 ไว้ด้านบน):**
`fasnet_alive=False` path (face_service.py:289-305) กับ path ที่ Fasnet ทำงานปกติแต่ตัดสินว่า spoof ควรคืนข้อความ/status คนละแบบ — เช่น `503` + ข้อความ "ระบบตรวจสอบใบหน้าขัดข้องชั่วคราว กรุณาลองใหม่ภายหลัง หรือแจ้งอาจารย์" แทน `400 spoof:true` — ทำให้ทั้งผู้ใช้และคนอ่าน log แยกออกได้ทันทีว่าเป็นกรณีไหน (ปัจจุบันต้องขุด ERROR-level log ถึงจะรู้ ซึ่งเป็นสาเหตุที่ user เตรียมภาพผิดทาง 3 รอบ) — **นี่คือ fix ที่ตรงกับ "severity-critical" framing ของ user มากที่สุด เพราะแก้ที่ตัวปัญหาจริง (message ปิดบัง) ไม่ใช่แค่แก้ environment bug เดียว**

**สถานะ:** ยังไม่ implement ตัวเลือกไหนทั้งสิ้น — ไม่แตะ `app/`, `Dockerfile`, `requirements.txt` ตามคำสั่ง รอ user ตัดสินใจว่าจะเลือกทางไหน/รวมกันกี่ทาง

---

## 17. F-16 fix, ทำทีละขั้น (2026-08-26 รอบถัดมา) — user ตัดสินใจแล้ว: split 400/503 ก่อน → แก้ build smoke test → root cause; eager-import/`--preload` ถือไว้ (record only), `/health` รอหลัง demo

**Cross-ref FRR-2 ยืนยันจาก user เอง:** "This is FRR-2's message ambiguity at maximum severity" — ตรงกับที่บันทึกไว้ในหัวข้อ 16 แล้ว ไม่มีอะไรต้องแก้เพิ่มในส่วน cross-ref

### Independent re-verification ของทั้งสองข้อที่ user ขอให้เช็คซ้ำ (ไม่เชื่อผลจาก session อื่นเฉย ๆ)

**1. Fresh-container, single-request test — รันซ้ำเองอีกรอบ (คนละรอบจาก session ก่อน, image เดิม `9acb62090e3f` ไม่มีการ build ใหม่ระหว่างนี้):** `docker restart` container (worker PID ใหม่ยืนยันจาก log timestamp `08:47:05`) ยิง request เดียว — **พังตั้งแต่ request แรกสุด** (`08:47:22 ERROR ... [FASNET] inference error`, ไม่มี Fasnet call ไหนก่อนหน้าใน log ของ process นี้เลย) **ยืนยันซ้ำอีกครั้ง: deterministic ตั้งแต่ request แรก ไม่ใช่ poison-after-race**

**2. `find -name test` เป็น root cause ที่เป็นไปได้จริง — ยืนยัน path ที่หายไปตรง ๆ:** `docker run --rm --entrypoint sh smartcheck` แล้ว `ls tensorflow/_api/v2/__internal__/` — path เต็มที่หายไปคือ `/usr/local/lib/python3.11/site-packages/tensorflow/_api/v2/__internal__/test` (ยืนยันจาก sibling directory listing: มีครบ 18 รายการยกเว้น `test` ที่ควรอยู่ระหว่าง `smart_cond` กับ `tf2` ตามลำดับ import ใน `__init__.py` บรรทัด 22: `from tensorflow._api.v2.__internal__ import test`) **ทำไม build check ผ่าน:** `grep -n` ยืนยัน line number ตรง ๆ ใน Dockerfile — Step 4 verification (`from deepface import DeepFace`, บรรทัด 39) และ Fasnet build-time smoke test (บรรทัด 48-86) รันก่อน cleanup layer (`find ... -name test`, บรรทัด 88-90) เสมอ ตามลำดับ `RUN` instruction ใน Dockerfile — ทั้งสอง check จึงเห็น filesystem ที่ `test/` ยังอยู่ครบ ผ่านได้จริง ไม่ใช่ false-positive

### ทำไมเดิมพันว่า "poison" ไม่ใช่คำตอบที่ถูก (สรุปสั้น ๆ)

ข้อ 16 เขียนสมมติฐานนี้ไว้ตอนแรกแล้วแก้ไปแล้ว — ยืนยันซ้ำอีกทีในรอบนี้ด้วยข้อมูลเดียวกัน ไม่มีอะไรเปลี่ยน

### Fix #1 — implemented: แยก 503 (system failure) ออกจาก 400 (spoof จริง)

**ไฟล์ที่แก้ (4 ไฟล์):**

**1. `app/services/face_service.py`** — เพิ่ม helper ใหม่ท้าย `combined_spoof_score()` (ไม่แตะ logic ข้างในฟังก์ชันเลย):
```python
_SYSTEM_FAILURE_MARKERS = {"fasnet_unavailable_fail_close", "all_layers_failed"}

def is_system_failure(spoof_result: dict) -> bool:
    return not spoof_result["is_real"] and bool(
        _SYSTEM_FAILURE_MARKERS & set(spoof_result.get("disagreements", []))
    )
```
ใช้ marker string ที่ `combined_spoof_score()` set ไว้อยู่แล้วทั้งสองจุด (`fasnet_alive=False` fail-close, บรรทัด ~289-305; และ `total_weight <= 0` all-layers-failed, บรรทัด ~371-380) — ไม่ต้องแก้ตัว `combined_spoof_score()` เอง ไม่แตะ threshold/detection logic ใด ๆ ตามข้อจำกัด

**2. `app/routes/api_checkin.py`** — antispoof gate (step 4b): เพิ่ม branch เช็ค `is_system_failure(spoof_result)` ก่อน return 400 เดิม — ถ้าใช่ ให้ log ERROR (เก็บ `disagreements` ไว้ฝั่ง server) แล้ว return **503** ข้อความ `"ระบบตรวจสอบใบหน้าขัดข้องชั่วคราว กรุณาลองใหม่อีกครั้ง หรือแจ้งเจ้าหน้าที่หากยังพบปัญหา"` (ไม่มี `spoof:true`) — ไม่ใช่ก็ตกไป 400 เดิมเป๊ะ ไม่เปลี่ยนพฤติกรรม genuine-spoof เลย

**3. `app/routes/student.py`** — enroll final-submit `MIN_SPOOF_PASS` loop (step 6): เปลี่ยนจาก `check_anti_spoof(face_images[idx])` (bool wrapper, ทิ้งข้อมูล `disagreements`) เป็นเรียก `combined_spoof_score(raw_frames[idx])` ตรง — ได้ผลพลอยได้คือเลิก decode ซ้ำด้วย (`raw_frames[idx]` decode ไว้แล้วตั้งแต่ step 3 ของ endpoint นี้) เก็บ `any_system_failure` ตลอด loop (นับรวมทั้งกรณี `is_system_failure()` true และกรณี exception ที่ยัง fail-close เหมือนเดิม) ถ้า `spoof_pass_count < MIN_SPOOF_PASS` **และ** มีอย่างน้อย 1 เฟรมที่เป็น system failure → return **503** เดียวกัน แทน `spoof_detected` 400 — สมเหตุสมผลเพราะแค่ 1 เฟรมระบบพังก็พอทำให้ทั้ง burst ตกได้ (ตามที่วิเคราะห์ไว้ในหัวข้อ FRR-2) `check_anti_spoof` import ออกจาก enroll function's local import (ไม่ได้ใช้ที่อื่นในฟังก์ชันนี้แล้ว — ยังใช้อยู่ใน `api_self_verify`, dead endpoint แยกต่างหาก ไม่แตะ) อัปเดต docstring บรรทัด 267 ให้ตรงกับโค้ดจริงด้วย (`check_anti_spoof x5` → `combined_spoof_score x5`)

**4. `scripts/load_test.py`** — `_diagnose_smoke()`: เพิ่ม branch เฉพาะสำหรับ 503 + ข้อความ "ขัดข้องชั่วคราว" ก่อนถึง generic `>=500` catch เดิม — พิมพ์ชัดว่า "ไม่ใช่การตัดสินว่าเป็น spoof" และชี้ให้ไปดู `[FASNET] inference error`/`[COMBINED_SPOOF] ... failing CLOSED` ใน log แทนที่จะสงสัยภาพเทส (ตรงกับที่ user สั่ง "Make 503 visible in diagnose output")

**Scope ที่ตั้งใจไม่รวม (flag ไว้ ไม่ implement เอง):** `/api/spoof_check` (student.py, pre-check ระหว่าง enrollment Step 2-4, เรียกได้ถึง 8 ครั้ง) เรียก `spoof_check_with_embedding()` ซึ่งก็เจอ defect เดียวกันได้ (message เดิม `"ตรวจพบการปลอมแปลง"` ไม่แยก system-failure) — **user สั่งเจาะจงแค่ "checkin และ enroll" ซึ่งตีความว่าหมายถึง 2 gate หลักที่คุยกันมาตลอด** (`/api/checkin` decisive gate, `/api/enroll` final-submit `MIN_SPOOF_PASS` gate) ไม่ใช่ pre-check ระหว่างทางที่ retry ได้เองอยู่แล้ว — ยังไม่แก้ `/api/spoof_check` รอ user ยืนยันว่าจะรวมด้วยไหม

### การ verify (ไม่ rebuild Docker เต็ม — container ที่รันอยู่ไม่มี volume mount เข้ากับ source ที่แก้ ต้อง rebuild เท่านั้นถึงจะเห็นผลจริงใน container เดิม, `docker inspect distracted_gagarin --format '{{json .Mounts}}'` ยืนยัน `[]`)

1. `py_compile` ทั้ง 4 ไฟล์ที่แก้ — ผ่านหมด ไม่มี syntax error
2. Unit-test `is_system_failure()` ตรง ๆ ผ่าน local venv's `face_service.py` (import แบบเดียวกับ `debug_moire.py`) กับ 5 กรณีที่ตรงกับ shape จริงที่ `combined_spoof_score()` ผลิตได้: `fasnet_unavailable_fail_close` → True, `all_layers_failed` → True, `hard_reject_N_layers_agree` → False, per-layer disagreement string (เช่น spoof จริงจาก weighted score) → False, `is_real=True` (guard กันกรณี disagreements ค้างมาผิด) → False — **ผ่านครบทั้ง 5**
3. `grep` ยืนยันไม่มี call site เก่า (`check_anti_spoof(face_images[idx])`) เหลือค้างอยู่ และ import ใหม่ (`combined_spoof_score`, `is_system_failure`) ต่อสายครบทั้ง 2 ไฟล์route

**สิ่งที่ยัง verify ไม่ได้จนกว่าจะ rebuild (เป็นข้อจำกัดที่ยอมรับได้ — root cause fix ยังไม่ทำ, container ปัจจุบันจึงพัง 100% อยู่ดี ไม่มีทางเห็น 503 จริงจนกว่าจะถึงขั้นตอนที่ 2/3):** end-to-end HTTP response จริงว่า return 503 แทน 400 จริงหรือไม่ — จะ verify ได้เป็นธรรมชาติพร้อมกับขั้นตอนที่ 2 (แก้ build smoke test) เพราะต้อง rebuild image อยู่ดี

### Holding (บันทึกเหตุผลตามคำสั่ง user — ไม่ implement)

**Eager import + `--preload`:** user ระบุเหตุผลเอง 2 ข้อ — (1) ไม่มีประโยชน์ที่ `--workers 1` (ตรงกับที่วิเคราะห์ไว้ในหัวข้อ 16 ตัวเลือก C พอดี), (2) **entangles กับ SQLAlchemy fork safety** — จุดนี้ไม่เคยตรวจสอบมาก่อนในเอกสารนี้ (`app/__init__.py` ใช้ SQLAlchemy เป็น Flask-Session backend ตาม `CLAUDE.md`) `gunicorn --preload` fork worker หลัง master โหลด app เสร็จ — ถ้า SQLAlchemy engine/connection pool ถูกสร้างตอน import/`create_app()` (ก่อน fork) แล้ว worker ที่ fork ออกมาแชร์ file descriptor ของ connection เดิมกันหมด เป็นปัญหาคลาสสิกที่ต้องแก้ด้วย `engine.dispose()` ใน post-fork hook — **ยังไม่ได้ตรวจโค้ดจริงว่าโปรเจกต์นี้มีปัญหานี้อยู่แล้วไหมถ้าเปิด `--preload` วันนี้ (ไม่เกี่ยวกับ eager-import เลย เป็นเรื่องแยกที่จะโผล่มาเหมือนกันถ้าเปิด `--preload` ไม่ว่า deepface จะ eager หรือ lazy)** — บันทึกไว้เป็นความเสี่ยงที่ user รู้แล้ว ไม่ implement ตามคำสั่ง

**`/health` ตรวจ Fasnet จริง:** user บอกว่า "worth doing before the demo, not before checkin works" — เข้าใจตรงกัน ไม่ implement ตอนนี้

### สถานะ

**Fix #1 (split 400/503) — code เขียนเสร็จ verify เท่าที่ทำได้โดยไม่ rebuild แล้ว รอ diff review จาก user ก่อนนับว่า "เสร็จ"** ยังไม่เริ่ม Fix #2 (แก้ build-time smoke test) ตามคำสั่ง "one change at a time, verified before the next" — รอ user ยืนยัน Fix #1 ก่อน

---

## 18. F-16 — Fix #1 confirmed, ขยายไป `/api/spoof_check`, Fix #2 (root cause) แก้เสร็จและ verify ผ่านครบ end-to-end (2026-08-26 รอบสุดท้าย)

**User confirm Fix #1** (evidence: missing `test/` ตรงตำแหน่ง alphabetical + cleanup layer รันหลัง checks) **และรับ correction เรื่อง `--preload`/SQLAlchemy fork-safety ว่าเป็นคนละประเด็นจาก eager-import จริง**

### Fix #1 ขยายไป `/api/spoof_check` (จุดที่ 3 จาก 3 ที่ต้องแก้)

User ตัดสินใจรวม `/api/spoof_check` เข้าด้วย เพราะ "the reasoning applies more strongly there, not less" — เรียกได้ถึง 8 ครั้ง/enrollment, retryable ทำให้แย่กว่าเดิม (ผู้ใช้ retry ซ้ำ ๆ กับสิ่งที่ retry ไม่มีทางสำเร็จ)

**ไฟล์ที่แก้เพิ่ม (2 ไฟล์):**
1. `face_service.py` — `spoof_check_with_embedding()`: เพิ่ม key `"system_failure": is_system_failure(spoof_result)` ใน return dict ของ branch `not spoof_result["is_real"]`
2. `student.py` — `/api/spoof_check` route: เช็ค `result.get("system_failure")` ก่อน log/return ปกติ — ถ้าใช่ log `"system_failure"` แล้ว return **503** ข้อความเดียวกับ checkin/enroll แทน 200 + `is_real:false` + `"ตรวจพบการปลอมแปลง"` เดิม

**พบว่า frontend ไม่ต้องแก้เลย — ตรวจ `_callSpoofCheckSafe()` (enrollment_flow.js:211-253) แล้ว:** `if (!res.ok) throw new Error(...)` — 503 ทำให้ `res.ok=false` เข้า `catch`, retry 1 ครั้ง, สุดท้าย fallback เป็น `{ is_real: false, message: 'ไม่สามารถตรวจสอบได้...', _networkError: true }` **โค้ดเดิมทุกจุดที่เรียก (Step 2/3/4) เช็ค `_networkError` อยู่แล้วและปฏิบัติต่อมันแบบ fail-open** (Step 2/3: ข้าม pre-check แล้วเดินหน้าต่อ, Step 4: skip เฟรมนี้ ไม่นับ fail counter) — 503 จึงไม่โชว์ "ตรวจพบการปลอมแปลง" ให้ผู้ใช้เห็นอีกต่อไปโดยธรรมชาติ ไม่ต้องแก้ JS

**⚠️ ข้อสังเกตที่พบระหว่างตรวจ (flag ไว้ ไม่ implement เพราะนอก scope backend-only ที่ user สั่ง):** Step 4's capture loop (`startCaptureWithDetection`, ~line 1122-1138) เมื่อเจอ `_networkError` จะ `capturePaused = false; return;` แล้ว**ไม่เพิ่ม fail counter เลย** — ถ้า 503 เป็นปัญหาถาวร (แบบ F-16 ก่อนแก้ root cause) ไม่ใช่ปัญหาชั่วคราว การ retry จะวนซ้ำไม่จบ **โดยไม่มี error message ให้ผู้ใช้เห็นเลย** (ต่างจาก Step 2/3 ที่ fail-open แบบเดินหน้าต่อ ไม่ใช่วนซ้ำ) เดิมที (ก่อนแก้ F-16) เฟรมที่ fail แบบนี้จะนับเป็น "spoof" เข้า fail counter แล้วสุดท้าย exhaust ไปเจอข้อความ "กรุณาเริ่มใหม่ในสภาพแวดล้อมที่มีแสงเพียงพอ" (จุดเดียวที่บอกเรื่องแสงตาม FRR-2) — **การแก้ F-16 อาจทำให้ผู้ใช้ไม่มีทางเจอข้อความนั้นอีกเลยถ้า Fasnet ยัง broken ถาวรใน Step 4 โดยเฉพาะ** ยังไม่แก้ตามคำสั่ง "same constraints" (backend only) — บันทึกเป็นความเสี่ยงที่ต้องรู้ ถ้าจะแก้ต้องแตะ JS

**ยืนยัน scope ครบ 7 จุดตามที่ user ขอ (เชื่อมกับ Q-15):** Q-15 เดิมบ่นว่า anti-spoof policy กระจาย 7 จุด ไม่มีจุดไหนเห็นภาพรวม — ตอนนี้ทั้ง 7 จุด (checkin ×1, enroll final-submit ×1, `/api/spoof_check` ×1 [เรียกได้ 8 ครั้ง], `combined_spoof_score` gate ภายใน ×4 ที่เหลือยังไม่ต้องแก้เพราะไม่ return ตรงไปหา client) **ผ่านจุดตัดสินใจเดียวกัน (`is_system_failure()`) แล้ว — คนละสาเหตุ (system failure vs spoof) แยกออกจากกันได้ตอนนี้ทุกจุดที่ user-facing แล้ว** ยังไม่ centralize เป็น 1 policy object ตามที่ Q-15 เสนอไว้เดิม (นอก scope รอบนี้) แต่ "แยกสาเหตุได้" ซึ่งเป็นข้อร้องเรียนที่คมกว่าของ Q-15 แก้ครบแล้ว

### Fix #2 — 3-step verification ตามที่ user สั่ง (ผลจริงจากการรัน ไม่ใช่การคาดเดา)

**Step 1 — reorder เท่านั้น (ไม่ widen except, ไม่ narrow find pattern):** build ล้มเหลว **ที่ Step 4 (`import tensorflow` เปล่า ๆ)** ไม่ใช่ที่ Fasnet smoke test เลย เพราะ Step 4 ไม่เคยมี except คลุมอยู่แล้วตั้งแต่ต้น — **คำตอบสำหรับคำถาม user ตรง ๆ: build ไม่ผ่าน ("does the build still pass? If yes...") คำตอบคือ "ไม่" ดังนั้นข้อสรุปที่แม่นยำกว่าคือ: การ reorder เพียงอย่างเดียวก็เพียงพอจะจับบั๊กนี้ได้แล้ว เพราะ Step 4 ไม่มี allowlist ให้ผ่านตั้งแต่แรก — allowlist ที่แคบใน Fasnet smoke test ไม่ใช่สิ่งที่ปิดบังบั๊กตัวนี้โดยตรง (มันปิดบังปัญหาคนละคลาสที่ยังไม่เคยเกิด — เช่นถ้า `import tensorflow` ผ่านแต่ `extract_faces(anti_spoofing=True)` ล้มเหลวเพราะเหตุอื่น)** ยังคุ้มค่าที่จะ widen อยู่ดีเป็น defense-in-depth แยกเรื่อง ไม่ใช่เพราะมันคือสาเหตุของบั๊กนี้

**Step 2 — widen except ด้วย (ยัง reorder, ยัง blanket find pattern):** build ล้มเหลว **ที่จุดเดียวกันเป๊ะ** (`Step 4`) ด้วยเหตุผลเดียวกัน — Fasnet smoke test ไม่เคยถูกรันถึงเลยเพราะ Step 4 ตายก่อน ยืนยันข้อสรุปจาก step 1 ว่า ordering คือสาเหตุทั้งหมด ไม่ใช่ allowlist

**Step 3 — narrow find pattern ด้วย (`-not -path "*/tensorflow/*"`), ยัง reorder + widen except:** build **ผ่าน** — `tensorflow 2.15.0`, `DeepFace import OK`, `Fasnet smoke test OK — 1 faces detected on random image` (ก่อนหน้านี้เคย fallback message "no face expected" อยู่เสมอ — ตอนนี้ extract_faces สำเร็จจริง ไม่ error เลย)

**⚠️ พบข้อผิดพลาดของตัวเองระหว่างทำ step 3 — เก็บบันทึกไว้ตรง ๆ ไม่ปิดบัง:** ตอนแรกลืมลบ cleanup block เดิมที่อยู่ตำแหน่งท้ายไฟล์ (หลัง Fasnet smoke test) ทิ้ง — เหลือ cleanup **สองรอบ**: รอบใหม่ (มี `-not -path` exclude tensorflow, อยู่ก่อน Step 4) และรอบเก่า (blanket pattern เดิม ไม่มี exclude อยู่หลัง Fasnet smoke test) build "ผ่าน" ตามที่ log บอก แต่ **เป็นการผ่านที่ผิด** เพราะรอบเก่าลบ `test/` ทิ้งอีกรอบหลังจากที่ checks ผ่านไปแล้ว — ตรวจพบจากการ `docker run` เข้าไป `ls` ไฟล์จริงในอิมเมจที่ build เสร็จ (ไม่เชื่อแค่ log ว่า "OK") เจอว่า `test/` หายไปเหมือนเดิม **นี่คือเหตุผลที่ user ย้ำว่า "confirm the build passes for the right reason" สำคัญ** — ลบ cleanup block ซ้ำทิ้ง (เหลือรอบเดียว) แล้ว build ใหม่ (cache hit ทุก layer ที่ไม่เปลี่ยน, เร็วมาก) ตรวจซ้ำด้วย `ls` ตรง ๆ อีกครั้ง: **`test/` อยู่จริงคราวนี้** และ bare `docker run --entrypoint python ... -c "from deepface import DeepFace"` (repro เดิมที่เคย crash 100%) **ผ่านสะอาด ไม่ error เลย**

### Verify Fix #1 end-to-end — ครบทั้ง 2 ผลลัพธ์ที่ user ขอให้เห็น

**ผลลัพธ์ที่ 1 — Fix #1 (app code) มีอยู่ แต่ root cause (Dockerfile) ยังไม่แก้:** revert Dockerfile กลับไปเป็นเวอร์ชันเดิมชั่วคราว (เก็บเวอร์ชันแก้แล้วไว้ที่ scratchpad ก่อน) build image ใหม่ (cache hit install layer ทั้งหมด เพราะเนื้อหาเหมือนเดิมเป๊ะ) รัน container แยกที่ port 8081 ยิง smoke test:
```
HTTP status: 503
{"error": "ระบบตรวจสอบใบหน้าขัดข้องชั่วคราว กรุณาลองใหม่อีกครั้ง หรือแจ้งเจ้าหน้าที่หากยังพบปัญหา",
 "ok": false, "retry_face": true}
```
ยืนยันจาก container log ว่ามาจาก Fasnet crash ตัวเดิมเป๊ะ: `[FASNET] inference error: cannot import name 'test' ...` → `[ANTISPOOF] system failure, not a spoof determination — disagreements=['fasnet_unavailable_fail_close']` — **Fix #1 ทำงานถูกต้อง แยก 503 ออกจาก 400 ได้จริงภายใต้เงื่อนไข Fasnet พังจริง ก่อนจะแก้ root cause**

**ผลลัพธ์ที่ 2 — root cause แก้แล้ว (Fix #2 ครบ):** restore Dockerfile กลับเป็นเวอร์ชันแก้แล้ว (diff ยืนยันตรงกับที่เก็บไว้) build (cache hit ทุกอย่าง เร็วมาก) restart container หลัก ยิง smoke test:
```
HTTP status: 400
{"error": "ใบหน้าไม่ตรง — กรุณาถ่ายรูปใหม่", "ok": false, "retry_face": true}
```
ตรงตามคาด — anti-spoof ผ่านจริง (Fasnet ทำงานสำเร็จ) pipeline เดินไปถึง face-verification แล้วปฏิเสธเพราะ seed embedding เป็นเลขสุ่ม ไม่ตรงกับภาพเทสจริง (**ผลลัพธ์ที่ถูกต้องตามที่ควรเป็น** ไม่ใช่ error)

**ทั้งสองผลลัพธ์ตรงกับที่คาดไว้ทุกประการ — F-16 ปิดจริงแล้ว ทั้ง symptom (400 spoof ปลอม) และ root cause (test/ หาย) และ message-masking defect (cross-ref FRR-2) ครบทุกจุด**

### ทำความสะอาด environment หลัง verify

ลบ image tag ชั่วคราวที่ใช้ระหว่าง verify (`f16-step3` ซ้ำกับ `latest`, `f16-fix1-only` ใช้แล้วหมดหน้าที่) เหลือแค่ `smartcheck:latest` (image ที่ verify แล้วครบ) container หลักตอนนี้ชื่อ `smartcheck-fixed` รันจาก image นี้ที่ port 8080 (container เก่า `distracted_gagarin` ลบไปแล้ว) — container เก่า 3 ตัวจาก image คนละอันตั้งแต่ 35 ชม.ก่อน (`epic_bardeen`, `reverent_germain`, `confident_wiles`) **ไม่แตะ** เผื่อ user เก็บไว้ตั้งใจ

### สถานะ

**F-16 แก้ครบทั้งสามจุด (Fix #1 split 400/503 ×3 endpoint, Fix #2 root cause Dockerfile) verify end-to-end ผ่านทุกขั้นตอนตามที่ user กำหนด — ปิด finding นี้ได้** เหลือ open item เดียวที่ flag ไว้ไม่ implement: Step 4 capture loop's silent-retry-forever gap (ด้านบน) รอ user ตัดสินใจว่าจะแก้ JS ด้วยไหม

---

## 19. Step 4 failure-cap harness and same-origin browser walkthrough (2026-08-26)

### Harness

Run `node scripts/test_step4_failure_tracker.js`. The harness imports the browser-shipped
`app/static/js/step4_failure_tracker.js`; `enrollment_flow.js` uses that same file at runtime.
It covers 503/increment, 429/reset, genuine spoof/reset, genuine pass/reset,
connection-refused/increment, `5 + 429 + 5`/no trip, and exactly six/trip once.

### Revised walkthrough setup: BROKEN backend owns port 8080

Do not run the broken image on 8081 and do not override `spoofCheckUrl`. Stop the fixed
container, then bind the broken image to the original host and container port:

```powershell
docker stop f16-broken-8081
docker stop smartcheck-fixed
docker run --rm --name f16-broken-8080 --env-file .env -e PORT=8080 -p 8080:8080 smartcheck:f16-known-broken
```

Keep the browser at `http://localhost:8080`; the injected URL remains
`/student/api/spoof_check`. Because scheme, host, and port are unchanged, the existing
session/CSRF cookies remain first-party and no CORS or `credentials` workaround is involved.

In DevTools Network, the decisive request is `POST /student/api/spoof_check`. Under the
known F-16 broken image it must show status **503** and this JSON response:

```json
{
  "confidence": 0.0,
  "is_real": false,
  "message": "ระบบตรวจสอบใบหน้าขัดข้องชั่วคราว กรุณาลองใหม่อีกครั้ง หรือแจ้งเจ้าหน้าที่หากยังพบปัญหา"
}
```

A CORS error, `(failed)`, connection refusal, 401, 400, or any different body does not
confirm this walkthrough. `_callSpoofCheckSafe` retries once, so two 503 requests can appear
for one capture attempt; each completed call contributes one Step 4 strike after its retry.

After the walkthrough, stop the foreground broken container (Ctrl+C) and restore the fixed
backend with `docker start smartcheck-fixed`. The stopped `f16-broken-8081` container is left
intact; start it again only if the old 8081 diagnostic endpoint is still wanted.

---

## 20. `/api/spoof_check` temporal demo blocker and audit-only fail-open (2026-08-27)

Live browser evidence on one cooperative real face measured eight accumulated-temporal
evaluations from 1.419–7.314 (mean 5.399); the uncalibrated 6.0 early gate rejected 3/8
(37.5%) before Fasnet could run. The gate is now **log-only**: the rolling six-frame
accumulator and variance computation remain, with `static_log_only`, variance, frame count,
`reference_threshold=6.0`, and `decision=log_only` in the audit log, but low variance no
longer clears the accumulator or returns early.

Moiré and Texture were already audit-only on their verdicts after FRR-1/F-15, but their
exception branches could still terminate `/api/spoof_check` with 500. Those exception paths
now also log `error_log_only` and continue. No audit-only computation in this endpoint can
terminate the request: Moiré, Texture, and accumulated Temporal all continue on either a
flagged result or a computation error.

Frame-validation policy is unchanged. Rejection logging now includes every diagnostic already
available at the point of failure: `size_kb`, dimensions, Laplacian variance, and B/G/R channel
standard deviations. This makes a future sub-3 KB capture diagnosable without storing image
content.

**Important policy finding:** `combined_spoof_score(..., frames_for_temporal=...)` is called
with `frames_for_temporal` by **no HTTP route**. `/api/checkin`, `/api/enroll`, and
`/api/spoof_check` all call it with one image only, so its nominal 0.20 Temporal weight runs on
**zero HTTP paths** and is redistributed away every time. The weight in `SPOOF_WEIGHTS` is not
an effective production decision weight. Check-in and final enrollment still enforce their
separate standalone Temporal gates at 4.0; only `/api/spoof_check` changed to log-only.

---

## 21. Final `/api/enroll` Temporal blocker removed (2026-08-27)

The first successful five-frame browser burst after §20 reached `/api/enroll`, where the
standalone face-cropped Temporal gate rejected the cooperative real face at 3.609 against the
historical 4.0 threshold. Git history shows this threshold moved 8→4→8→4 in under one hour;
the face-crop implementation had no retained calibration dataset, and its source claim that
cropped real faces score 15–25 is directly contradicted by the measured 3.609.

Final-enrollment Temporal is now audit-only (`static_log_only`/`pass_log_only`, variance,
`reference_threshold=4.0`, `decision=log_only`). Its computation exception also logs
`error_log_only` and continues. The unsupported 15–25 comment and docstring claim were
replaced with the actual calibration status; the numeric 4.0 reference was not retuned.

Final-enrollment Moiré and Texture verdicts were already audit-only, but their computation
exceptions still returned 400. Both exceptions now log `error_log_only` and continue. Together
with the existing non-blocking EAR, integrity-hash, and profile-upload audits, no audit-only
computation in `/api/enroll` can terminate enrollment.

Optional `ear_std` parsing is now guarded: missing, empty, nonnumeric, NaN, and infinite values
are logged as `absent` and continue; valid finite values retain `low_but_pass`/`pass`. No frame
validation, 4/5 combined-spoof, embedding consistency, continuity, duplicate, Moiré threshold,
or client behavior changed.

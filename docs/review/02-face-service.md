# Review: `app/services/face_service.py`

วันที่: 2026-08-20  
Reviewer: Claude Sonnet 4.6  
Scope: localhost เท่านั้น — ไม่รวม BLE / production deploy  
บริบท: ไฟล์นี้เป็น core AI pipeline ที่ถูกเรียกทุก enrollment และทุก check-in

---

## Function Inventory

| ฟังก์ชัน | file:line | LOC | ถูกเรียกจากไหน | ต้นทุนต่อ call | คำตัดสิน | เหตุผล |
|---|---|---|---|---|---|---|
| `_get_antispoof_session` | face_service.py:50 | 18 | face_service.py:92 (`_run_antispoof`) | O(1) หลัง lazy load; one-time ONNX load ~100–500 ms | KEEP | double-checked locking ถูกต้อง, lazy singleton pattern |
| `_crop_face_for_antispoof` | face_service.py:70 | 19 | face_service.py:93 (`_run_antispoof`) | Haar cascade + resize ~10–30 ms; สร้าง `CascadeClassifier` ใหม่ทุก call | MERGE | Haar cascade logic ซ้ำกับ `detect_static_image`:803; ควรแยก shared helper |
| `_run_antispoof` | face_service.py:91 | 42 | face_service.py:254 (`combined_spoof_score`) | crop + ONNX inference ~15–40 ms | REFACTOR | `CONFIDENCE_MARGIN = 0.10` hardcode ใน body:114 ไม่อยู่ใน Threshold Map บนสุด |
| `_run_fasnet_antispoof` | face_service.py:135 | 26 | face_service.py:242 (`combined_spoof_score`) | `DeepFace.extract_faces` + anti_spoofing ~500–2000 ms | KEEP | clean fail-open wrapper, single responsibility |
| `combined_spoof_score` | face_service.py:163 | 234 | face_service.py:451, 465, 495 (internal wrappers) | ผลรวมทุก layer ~600–2500 ms | SPLIT | 234 LOC รวม hard-reject logic (ล.284–337) + weight normalization + audit logging; ควรแยก `_check_hard_reject()` helper; `_layer_suspicious` ควรอยู่ระดับ module |
| `normalize_illumination` | face_service.py:399 | 7 | face_service.py:428, 500 (internal) | CLAHE on BGR ~1–5 ms | KEEP | simple, clear |
| `_decode_image` | face_service.py:408 | 10 | face_service.py:427, 450, 465, 487 (internal); api_checkin.py:166, 216; student.py:443, 987 | base64 + `cv2.imdecode` ~1–5 ms | KEEP | แต่ caller pattern ใน api_checkin.py บังคับ decode ซ้ำ (ดู Model Loading) |
| `extract_embedding` | face_service.py:420 | 21 | api_checkin.py:326; student.py:543, 823 | CLAHE + FaceNet512 ~200–800 ms (warm) | KEEP | clean pipeline; ปัญหาอยู่ที่ `spoof_check_with_embedding` ที่ duplicate logic |
| `check_anti_spoof` | face_service.py:443 | 13 | api_checkin.py:247; student.py:516, 778 | _decode + combined_spoof_score ~600–2500 ms | MERGE | identical body กับ `check_anti_spoof_with_score` ต่างแค่ return type; ควร merge เป็น API เดียว |
| `check_anti_spoof_with_score` | face_service.py:458 | 14 | api_checkin.py:408 | เหมือน check_anti_spoof | MERGE | ← เข้า `check_anti_spoof`; near-duplicate, return tuple vs bool เท่านั้น |
| `spoof_check_with_embedding` | face_service.py:474 | 70 | student.py:1053 | combined_spoof + DeepFace.represent ~800–3000 ms | REFACTOR | ทำ `DeepFace.represent` ซ้ำ (ล.501–507) แทนที่จะเรียก `extract_embedding`; ควร delegate |
| `cosine_similarity` | face_service.py:546 | 9 | face_service.py:570, 586, 853 (internal) | dot product 512-D <1 ms | KEEP | simple math utility |
| `verify_face_multi` | face_service.py:557 | 24 | api_checkin.py:332; student.py:847 | N × cosine_sim <1 ms | KEEP | decision on best (not avg) — correct |
| `max_similarity_multi` | face_service.py:583 | 5 | student.py:419, 662 | N × cosine_sim <1 ms | MERGE | 1-liner ซ้ำ logic ของ `verify_face_multi`; ควร delegate |
| `_adaptive_moire_threshold` | face_service.py:590 | 29 | face_service.py:632 (`detect_screen_moire`) | grayscale + mean ~1–2 ms | KEEP | brightness-adaptive tuning ชัดเจน, ไม่มีซ้ำ |
| `detect_screen_moire` | face_service.py:621 | 47 | face_service.py:187 (`combined_spoof_score`); api_checkin.py:167; student.py:450, 994 | FFT 256×256 per frame ~5–15 ms/frame | KEEP | core detection, ถูกเรียก 3 ทางที่ต่างกัน |
| `detect_screen_texture` | face_service.py:670 | 36 | face_service.py:207 (`combined_spoof_score`); api_checkin.py:186; student.py:469, 1007 | FFT + log + peak count ~5–10 ms | KEEP | complements Moiré; default `min_peaks=50` ไม่มี caller ใช้ (ทุก caller ส่ง 30) |
| `server_validate_frame` | face_service.py:708 | 80 | api_checkin.py:75; student.py:382, 766, 979 | base64 + imdecode + Laplacian ~5–15 ms | KEEP | zero-trust gate สำคัญ; checks ชัดเจนเรียงตามลำดับ |
| `detect_static_image` | face_service.py:790 | 45 | face_service.py:219 (`combined_spoof_score`); student.py:487 | Haar cascade + resize + std-dev ~20–50 ms | REFACTOR | สร้าง `CascadeClassifier` ใหม่ทุก call:803; Haar cascade ที่ 3 ในไฟล์ (ดู Model Loading) |
| `check_embedding_consistency` | face_service.py:837 | 50 | student.py:587 | C(n,2) cosine sims <1 ms สำหรับ n≤5 | KEEP | logic ซับซ้อนแต่จำเป็น; caller เดียว |

---

## Threshold Map

| ชื่อ | file:line | ค่าปัจจุบัน | ใช้ตัดสินอะไร | ตั้งสูงไปเกิดอะไร | ตั้งต่ำไปเกิดอะไร |
|---|---|---|---|---|---|
| `SELF_VERIFY_THRESHOLD` | face_service.py:10 | 0.80 | self-verify ช่วง enrollment | FRR↑ reject ใบหน้าจริงมากขึ้น | FAR↑ ยอม verify ใบหน้าต่างคน |
| `SAME_DEVICE_THRESHOLD` | face_service.py:11 | 0.70 | check-in อุปกรณ์ที่เคย bind แล้ว | FRR↑ นักศึกษา check-in ไม่ผ่าน | FAR↑ อุปกรณ์เดิมหลวมเกิน |
| `NEW_DEVICE_THRESHOLD` | face_service.py:12 | 0.80 | check-in อุปกรณ์ใหม่/ไม่ผูก | FRR↑ มาก | FAR↑ อุปกรณ์ใหม่หลวมเกิน |
| `CONSISTENCY_THRESHOLD` | face_service.py:13 | 0.80 | pairwise consistency ของ 5 frames ตอน enrollment | ถ่ายซ้ำบ่อย เมื่อ pose เปลี่ยนนิดเดียว | ยอม embedding ที่ inconsistent เข้า DB |
| `DUPLICATE_THRESHOLD` | face_service.py:14 | 0.65 | reject ถ้า student อื่นตรงนี้ | ยากลง enroll สำหรับหน้าคล้ายกัน | ยอม duplicate enrollment |
| `CONTINUITY_THRESHOLD` | face_service.py:15 | 0.80 | liveness → capture identity continuity | FRR↑ | ยอมภาพคนละคนผ่าน continuity |
| `MOIRE_THRESHOLD` | face_service.py:16 | 0.60 | high-freq FFT energy ratio (multi-frame, enrollment) | ยอม screen replay ผ่านง่าย | reject real face ใน low-light / JPEG noise |
| `MOIRE_THRESHOLD_SINGLE` | face_service.py:17 | 0.70 | high-freq FFT energy ratio (single frame, check-in) | ยอม screen replay single-frame | reject real face |
| `TEMPORAL_VAR_THRESHOLD` | face_service.py:19 | 4.0 | std-dev across face-ROI frames | ยอม static photo ผ่าน | reject real face ที่นิ่งมากใน passive capture |
| `DUPLICATE_GRAY_ZONE` | face_service.py:22 | (0.60, 0.70) | log range เท่านั้น ไม่ตัดสิน | — | — |
| `MOIRE_LOG_RANGE` | face_service.py:23 | (0.45, 0.75) | log near-threshold เท่านั้น ไม่ตัดสิน | — | — |
| `SPOOF_WEIGHTS["fasnet"]` | face_service.py:31 | 0.15 | น้ำหนัก Fasnet ML layer | ลด sensitivity ML spoof | Fasnet ผิดพลาดแล้ว dominate ผล |
| `SPOOF_WEIGHTS["moire"]` | face_service.py:32 | 0.30 | น้ำหนัก Moiré FFT layer | ลด sensitivity screen replay | — |
| `SPOOF_WEIGHTS["temporal"]` | face_service.py:33 | 0.30 | น้ำหนัก Temporal Variance layer | ลด sensitivity static photo | — |
| `SPOOF_WEIGHTS["texture"]` | face_service.py:34 | 0.15 | น้ำหนัก Screen Texture layer | ลด sensitivity OLED screens | — |
| `SPOOF_WEIGHTS["onnx"]` | face_service.py:35 | 0.10 | น้ำหนัก ONNX audit layer | — | — |
| `SPOOF_DECISION_THRESHOLD` | face_service.py:37 | 0.50 | weighted score cutoff (>0.50 = spoof) | ยอม spoof ผ่าน | reject real face มากขึ้น |
| `FASNET_REAL_THRESHOLD` | face_service.py:38 | 0.50 | **ไม่ถูกใช้ที่ไหนเลย** (dead constant) | — | — |
| `CONFIDENCE_MARGIN` | face_service.py:114 | 0.10 | override borderline ONNX reject เมื่อ margin แคบ | ยอม uncertain override มากขึ้น | ONNX hard reject เพิ่มขึ้น |
| hard-reject `moire_suspicious` | face_service.py:293 | 0.55 | moire layer ถือว่า suspicious ใน multi-layer vote | ยอม screen ผ่านก่อน hard-reject | false reject real face ง่าย |
| hard-reject `texture_suspicious` | face_service.py:294 | 0.50 | texture layer suspicious threshold | เหมือนข้างบน | เหมือนข้างบน |
| hard-reject `temporal_suspicious` | face_service.py:295 | 0.50 | temporal layer suspicious threshold | เหมือนข้างบน | เหมือนข้างบน |
| hard-reject `fasnet_suspicious` | face_service.py:296 | 0.30 | fasnet layer suspicious threshold (ต่ำกว่า layer อื่น) | hard-reject ยากขึ้น | Fasnet 0.30 trigger เร็วมาก ใน borderline cases |
| hard-reject Moiré strong | face_service.py:323 | 0.85 | Moiré alone พอ hard-reject | ยอม near-screen ผ่าน | — |
| adaptive brightness low | face_service.py:606 | 60 (mean px) | threshold ของ dark frame | ปล่อย threshold ไม่ขยับ → FPR↑ dark | — |
| adaptive brightness high | face_service.py:609 | 180 (mean px) | threshold ของ bright/screen-like frame | ปล่อย threshold ไม่ขยับ → FNR↑ bright | — |
| adaptive max adjust dark | face_service.py:608 | +0.05 | เพิ่ม threshold เมื่อ dark (loosen) | ยอม screen replay ใน dark room | — |
| adaptive max adjust bright | face_service.py:610 | −0.03 | ลด threshold เมื่อ bright (tighten) | — | false reject real face ใน bright light |
| adaptive clamp | face_service.py:615 | (0.50, 0.75) | clamp adaptive threshold ไม่ให้เกิน range | — | — |
| `server_validate` size min | face_service.py:739 | 3 KB | payload lower bound | reject frame เล็กเกินจริง | ยอม near-empty / corrupt frames |
| `server_validate` size max | face_service.py:736 | 500 KB | payload upper bound | — | ยอม large payload (potential slow path) |
| `server_validate` blur | face_service.py:772 | 8 (Laplacian var) | minimum sharpness (relaxed จาก 20) | ยอม blurry frames | reject webcam frames ที่ inherently soft |
| `server_validate` color std-dev | face_service.py:781 | 2.0 (relaxed จาก 5.0) | naturalness / synthetic image check | ยอม synthetic images | reject dim/uniform environment |
| `detect_screen_texture` peak multiplier | face_service.py:672 | 3.0 (default, ไม่มี caller ใช้) | FFT peak height above mean | ยอม screen ผ่าน | false peaks จาก noise |
| `detect_screen_texture` min_peaks default | face_service.py:673 | 50 (default, ไม่มี caller ใช้) | callers ทุกเจ้าส่ง 30 แทน | ยอม less-periodic screens | false reject real faces |
| `detect_static_image` face pad | face_service.py:813 | 20% (0.20) | padding รอบ ROI Haar crop | ROI หลวม background เข้ามา | crop แน่น พลาด micro-movement ขอบ |
| `detect_static_image` resize | face_service.py:825 | 64×64 | grayscale resize ก่อน std-dev | ยอม low-res analysis | เพิ่ม compute แต่ละ frame |
| `_ANTISPOOF_SCALE` | face_service.py:47 | 2.7 | crop scale สำหรับ ONNX input | หลวม — context รอบใบหน้ามากเกิน | miss face context |
| `_ANTISPOOF_INPUT_SIZE` | face_service.py:48 | 80 | ONNX model input resolution (px) | — | — |

---

## Model Loading

### โมเดลที่ใช้และจุดโหลด

| โมเดล | โหลดที่ | กลไก cache | โหลดซ้ำต่อ request? |
|---|---|---|---|
| **FaceNet512** (DeepFace) | `DeepFace.represent()` ครั้งแรก — lazy | DeepFace module-level cache ภายใน library | ไม่ — warm หลัง call แรก |
| **Fasnet** (DeepFace anti-spoofing) | `DeepFace.extract_faces(anti_spoofing=True)` ครั้งแรก | DeepFace module-level cache ภายใน library | ไม่ — warm หลัง call แรก |
| **ONNX (antispoof.onnx)** | `_get_antispoof_session()` ครั้งแรก — face_service.py:50 | `_antispoof_session` module-level + threading.Lock | ไม่ — lazy singleton ถูกต้อง |
| **Haar Cascade** (frontalface) | สร้างใหม่ทุก call ใน 2 จุด: `_crop_face_for_antispoof`:71 และ `detect_static_image`:803 | **ไม่มี cache** — `CascadeClassifier()` ทุก call | **ใช่ — โหลดซ้ำทุก call** |

### ปัญหาที่พบ

**1. Haar Cascade โหลดซ้ำทุก call (face_service.py:71, 803)**  
ทั้ง `_crop_face_for_antispoof` และ `detect_static_image` สร้าง `cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")` ใหม่ทุกครั้งที่ถูกเรียก ไม่มี module-level instance  
→ ต่อ api_checkin request: `_run_antispoof` + `detect_static_image` (ใน `combined_spoof_score`:219) = **อย่างน้อย 2 ครั้ง**  
แก้: ย้ายไป module-level `_face_cascade = cv2.CascadeClassifier(...)`

**2. Double decode + double Moiré ใน api_checkin.py**  
- `api_checkin.py:166` → `_decode_image(face_image)` แล้ว `detect_screen_moire([raw_frame])` (ล.167)  
- `api_checkin.py:247` → `check_anti_spoof(face_image)` → `_decode_image` อีกครั้ง → `combined_spoof_score` → `detect_screen_moire` อีกครั้ง  
→ **decode 2 ครั้ง, Moiré FFT 2 ครั้ง** ต่อ check-in request เดียว (~10–30 ms เพิ่มขึ้น)  
เป็นปัญหาของ api_checkin.py ไม่ใช่ face_service.py — แต่ interface ของ face_service บังคับให้เป็นแบบนี้ (รับ base64 แทน decoded array)

**3. `FASNET_REAL_THRESHOLD` (face_service.py:38) — dead constant**  
นิยามแต่ไม่มีที่ใดในโปรเจกต์ใช้ ควรลบออกหรือ wire เข้า `_run_fasnet_antispoof`

---

## Top 3 ที่คุ้มสุดถ้าแก้

### 1. ย้าย Haar Cascade ไป module-level (10 นาที, zero risk)

```python
# บรรทัดหลัง imports
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
```

แล้วแทนที่ทุก `cv2.CascadeClassifier(...)` ใน `_crop_face_for_antispoof`:71 และ `detect_static_image`:803 ด้วย `_face_cascade`  
**ผลลัพธ์:** ลด file-load ~20–50 ms ต่อ check-in + enrollment request; ไม่มีผลกระทบต่อ behavior

---

### 2. Merge `check_anti_spoof` + `check_anti_spoof_with_score` (15 นาที)

```python
def check_anti_spoof(base64_image: str, return_score: bool = False):
    ...
    if return_score:
        return result["is_real"], round(1.0 - result["combined_score"], 4)
    return result["is_real"]
```

ลด 14 LOC ซ้ำ; callers ใน api_checkin.py:247 และ :408 ปรับเพียงเพิ่ม argument  
**ผลลัพธ์:** API ชัดขึ้น, ไม่มีโอกาส diverge behavior ระหว่างสองฟังก์ชัน

---

### 3. Split `combined_spoof_score` แยก hard-reject logic (30 นาที)

234 LOC เป็นฟังก์ชันเดียว ทำให้อ่านยากและ test ยาก  
แยก block ล.284–337 เป็น:

```python
def _check_hard_reject(layers: dict) -> dict | None:
    """Return reject-result dict if hard-reject triggered, else None."""
    ...
```

**ผลลัพธ์:** `combined_spoof_score` ลดเหลือ ~180 LOC; `_check_hard_reject` test ได้อิสระ; hard-reject thresholds (ล.293–296, 323) อยู่ในฟังก์ชันเดียว ง่าย tune

---

## Follow-up

### 1. Haar Cascade — ต้นทุนจริงที่วัดได้

วัดด้วย script (10 รอบ, Python process เดียวกัน):

```
per-load avg : 31.010 ms
min          : 18.398 ms
max          : 70.021 ms
× 2 per check-in : 62.020 ms
```

ทุก check-in request กระทบ **2 ครั้ง** — `_crop_face_for_antispoof`:71 (เรียกผ่าน `_run_antispoof` → `combined_spoof_score`) และ `detect_static_image`:803 (เรียกผ่าน `combined_spoof_score` เช่นกัน)  
→ **~62 ms ที่หายไปทุก check-in** ก่อนถึง DeepFace แม้แต่ครั้งแรก  
แก้: ย้าย 1 บรรทัดขึ้น module level ไม่กระทบ behavior

---

### 2. `fasnet_suspicious = 0.30` (face_service.py:296) — ประวัติ commit

ค่า 0.30 ไม่ได้ตั้งมาพร้อม 2-of-N hard-reject rule — มีที่มาสองขั้น:

**ขั้นที่ 1 — commit `0485e04` (15 เม.ย. 2026)**

> "Force Docker cache bust + lower anti-spoof threshold to 0.3"  
> Lower antispoof_score threshold from DeepFace default **0.5 to 0.3** to reduce false positives from webcam captures with varying lighting/angle.

ตอนนั้นค่านี้เป็น `is_real = score >= 0.30` ใน `spoof_check_with_embedding` เก่า (ก่อน refactor) — ตั้งเพื่อลด FRR บน webcam lighting ที่แตกต่าง ไม่ใช่เพื่อ hard-reject voting

**ขั้นที่ 2 — commit `c908cfc`**

> "fix(spoof): rebalance weights toward FFT layers, add 2-of-N hard-reject rule"

เพิ่ม `fasnet_suspicious = _layer_suspicious(layers.get("fasnet", {}), 0.30)` — ค่า 0.30 ถูกนำมาใช้ใหม่ใน context ต่างกัน (จาก "ขีดแบ่ง is_real" กลายเป็น "ขีดเรียก suspicious") โดยไม่มี commit message อธิบาย

**เปรียบเทียบกับ blur/std:**  
`git log -S "blur"` และ `-S "2.0"` ชี้ commit เดียวกัน: `fe71b22` (initial commit ใหญ่)  
→ blur 20→8 และ std 5.0→2.0 ถูก relax **ก่อน** ค่า 0.30 จะถูกตั้ง (0485e04 มาทีหลัง fe71b22)  
→ ค่า 0.30 จึงถูกตั้งบนสภาพแวดล้อมที่ blur และ std ผ่อนแล้ว ไม่ได้ calibrate ร่วมกัน

**ความเสี่ยงปัจจุบัน:** `fasnet_suspicious` ที่ 0.30 ต่ำกว่าทุก layer อื่น (moire 0.55, texture/temporal 0.50) — หมายความว่า Fasnet ที่ spoof_score ≥ 0.30 นับ 1 ใน 2-of-N vote ทั้งที่ layer อื่นต้องผ่านขีดสูงกว่า ถ้า Fasnet borderline (เช่น webcam glare) จะ trigger hard-reject ง่ายกว่า layer อื่น

---

### 3. Double decode — ยืนยันด้วยโค้ดดิบ

**api_checkin.py:164–175 (ขั้น 4a)**

```python
# ─── 4a. Moiré / screen-replay detection (FFT — faster than MiniFASNet) ───
try:
    raw_frame   = _decode_image(face_image)          # decode ครั้งที่ 1
    moire       = detect_screen_moire([raw_frame], threshold=MOIRE_THRESHOLD_SINGLE)
    ...
    if moire["is_screen"]:
        return jsonify({...}), 400
except Exception as moire_err:
    ...
```

**api_checkin.py:245–254 (ขั้น 4b)**

```python
# ─── 4b. Anti-spoofing via MiniFASNet ────────────────────────────────────
try:
    is_real = check_anti_spoof(face_image)           # ส่ง base64 ซ้ำ
    ...
```

`check_anti_spoof` (face_service.py:450) เรียก `_decode_image(base64_image)` อีกครั้ง และ `combined_spoof_score` (face_service.py:187) เรียก `detect_screen_moire([img_bgr])` อีกครั้ง

**`raw_frame` จาก :166 ไม่ถูกนำมาใช้ที่ :247** — `check_anti_spoof` รับ `base64_image: str` เท่านั้น ไม่รับ BGR array

**วิธีแก้ที่ไม่ต้องแก้ signature ของ face_service:**

เปลี่ยนใน `api_checkin.py` บรรทัด 247 จาก:

```python
is_real = check_anti_spoof(face_image)
```

เป็น:

```python
spoof_result = combined_spoof_score(raw_frame)   # ใช้ raw_frame ที่ decode ไว้แล้วที่ :166
is_real = spoof_result["is_real"]
```

`combined_spoof_score` รับ `img_bgr: np.ndarray` อยู่แล้ว — ไม่ต้องแก้ face_service.py เลย  
ประหยัดได้: decode 1 ครั้ง (~3 ms) + moire FFT 1 ครั้ง (~10 ms) = **~13 ms ต่อ check-in**

---

### Top 3 ใหม่ — เรียงโดย ms ที่ประหยัดได้จริงต่อ check-in request

| ลำดับ | การแก้ | ms ประหยัด/request | ความถี่ | ความยาก |
|---|---|---|---|---|
| **1** | Haar Cascade → module-level (1 บรรทัด) | **~62 ms** (วัดจริง: 31 ms × 2) | ทุก check-in | ต่ำมาก |
| **2** | ใช้ `combined_spoof_score(raw_frame)` แทน `check_anti_spoof(face_image)` ใน api_checkin.py:247 | **~13 ms** (decode ~3 ms + moire FFT ~10 ms) | ทุก check-in | ต่ำ (1 บรรทัด, ไม่แก้ face_service) |
| **3** | `spoof_check_with_embedding` delegate `DeepFace.represent` → `extract_embedding()` | **~200–800 ms** (warm FaceNet512) แต่เกิดเฉพาะ enrollment self-verify path | enrollment เท่านั้น | ปานกลาง |

**หมายเหตุ:** อันดับ 3 (Split `combined_spoof_score`, Merge `check_anti_spoof`) ถูกถอดออกจาก Top 3 เพราะ ms = 0 — เป็น code quality ไม่ใช่ performance

---

## F-5: fasnet_suspicious Threshold Mismatch

### 1. โค้ดดิบ block hard-reject (face_service.py:284–337) และวิธีนับ

```python
# face_service.py:284-337

# ── Multi-layer hard-reject rules ──────────────────────────────────────
# Weighted scoring can be dominated by Fasnet when it's wrong.
# If TWO OR MORE independent layers independently flag suspicious,
# reject immediately — real faces never trigger 2+ layers at once.

def _layer_suspicious(layer_data, threshold):
    score = layer_data.get("spoof_score")
    return score is not None and score >= threshold

moire_suspicious    = _layer_suspicious(layers.get("moire", {}),    0.55)  # ← threshold สูง
texture_suspicious  = _layer_suspicious(layers.get("texture", {}),  0.50)  # ← threshold กลาง
temporal_suspicious = _layer_suspicious(layers.get("temporal", {}), 0.50)  # ← threshold กลาง
fasnet_suspicious   = _layer_suspicious(layers.get("fasnet", {}),   0.30)  # ← threshold ต่ำกว่าทุก layer

suspicious_count = sum([moire_suspicious, texture_suspicious, temporal_suspicious, fasnet_suspicious])

if suspicious_count >= 2:
    # ... hard-reject ทันที, ไม่ผ่าน weighted score
    return {"is_real": False, "combined_score": 1.0, ...}

# Moiré alone ≥ 0.85 → hard-reject เดี่ยว (ไม่เกี่ยวกับ Fasnet)
moire_score = layers.get("moire", {}).get("spoof_score")
if moire_score is not None and moire_score >= 0.85:
    return {"is_real": False, "combined_score": 1.0, ...}
```

**วิธีนับ:** `sum([bool, bool, bool, bool])` — แต่ละ layer ให้ได้ 0 หรือ 1 คะแนน ถ้ารวม ≥ 2 → hard-reject เกิดขึ้นก่อนที่ weighted score จะถูกคำนวณ

**ปัญหาของ fasnet_suspicious = 0.30:**

`spoof_score` ของ Fasnet คำนวณใน `_run_fasnet_antispoof` (face_service.py:155):

```python
spoof_score = (1.0 - raw_score) if is_real else raw_score
```

โดย `raw_score = face["antispoof_score"]` จาก DeepFace (0.0–1.0 = confidence ที่ว่าเป็นหน้าจริง) และ DeepFace ตัดสิน `is_real = antispoof_score >= 0.5` ภายใน

จึงได้:

| antispoof_score (DeepFace) | is_real | fasnet spoof_score | fasnet_suspicious (≥0.30)? |
|---|---|---|---|
| 0.90 | True | **0.10** | ไม่ — real face ชัดเจน |
| 0.75 | True | **0.25** | ไม่ |
| 0.70 | True | **0.30** | **ใช่** — borderline real ติด vote |
| 0.60 | True | **0.40** | **ใช่** — weakly real ติด vote |
| 0.51 | True | **0.49** | **ใช่** — barely real ติด vote |
| 0.49 | False | **0.49** | **ใช่** — barely spoof ติด vote |

**ช่วง antispoof_score ที่ติด suspicious**: ทุก real face ที่มีความมั่นใจ < 0.70 (antispoof_score 0.50–0.70) จะมี spoof_score 0.30–0.50 → ติด fasnet_suspicious ทั้งหมด

เทียบกับ layer อื่น: moire/texture/temporal ใช้ threshold 0.50–0.55 ซึ่งหมายถึง "ผ่านครึ่งทางสู่ spoof" — Fasnet ใช้เพียง 0.30 ซึ่งหมายถึง "real ที่มั่นใจน้อย" ก็นับเป็น suspicious แล้ว comment ในโค้ด (ล.29) ระบุ Fasnet "fooled by OLED/high-DPI screens in production" จึง demote weight เหลือ 0.15 — แต่ threshold hard-reject ไม่ได้ถูกปรับตาม

---

### 2. Logger ใน `combined_spoof_score` — อะไรบันทึกได้ อะไรหายไป

มี log 2 จุดหลักในฟังก์ชัน:

**จุดที่ 1 — ทุก hard-reject (WARNING, face_service.py:306–308)**

```
[COMBINED_SPOOF] HARD-REJECT: 2 layers suspicious (moire(0.612), fasnet(0.340)) — bypassing weighted score
```

Fields ที่มี: จำนวน layer suspicious, ชื่อ + spoof_score ของ layer ที่ suspicious เท่านั้น  
Fields ที่ **ไม่มี**: คะแนน layer ที่ไม่ suspicious (temporal, texture, onnx ที่ผ่าน), combined score สุดท้าย, decision path

**จุดที่ 2 — เฉพาะเมื่อ hard-reject ไม่เกิด (INFO, face_service.py:377–387)**

```
[COMBINED_SPOOF] combined=0.3821 threshold=0.5 decision=real layers=[fasnet=0.2100, moire=0.3200, temporal=0.1500, texture=0.0000, onnx=0.6700] disagreements=none
```

Fields ที่มี: combined score, threshold, decision, คะแนนทุก layer, disagreements  
**ปัญหา: log นี้ไม่ทำงานเมื่อ hard-reject เกิดขึ้น** — request ที่ตกหลุม F-5 จะสร้างเฉพาะ WARNING จุดที่ 1 ซึ่งไม่มีคะแนน layer ที่ผ่าน

**log ที่ช่วยเสริมได้:**
- `[ANTISPOOF] label=... probs=[spoof=X, real=X, screen=X] ...` (INFO, ล.126) — ONNX raw probs
- `[TEMPORAL] temporal_variance=X threshold=4.0 face_crop=True` (INFO, ล.830) — temporal score
- `[MOIRE] near-threshold avg_score=X threshold=X` (INFO, ล.661) — เฉพาะเมื่อ score อยู่ใน MOIRE_LOG_RANGE (0.45–0.75)

**สรุป:** ย้อนดูคะแนนจริงได้บางส่วนจาก individual layer logs แต่ต้องตาม request_id หรือ timestamp ข้ามหลาย log lines; ไม่มี log บรรทัดเดียวที่รวบรวมทุก layer score พร้อมกันเมื่อ hard-reject เกิดขึ้น

---

### 3. สรุป F-5

**อาการที่คาดว่าจะเกิด**

หน้าจริงถูก hard-reject ในสถานการณ์:
- นักศึกษาอยู่ในแสงสลัว (DeepFace antispoof_score = 0.55–0.70 → fasnet spoof_score 0.30–0.45 → suspicious)
- ร่วมกับ JPEG noise จาก webcam คุณภาพต่ำ → moire score ขยับเข้า 0.55+ หรือ texture ขยับเข้า 0.50+
- ผลลัพธ์: fasnet + moire (หรือ texture) = 2 layers → hard-reject ทั้งที่ combined score จะผ่าน

สถานการณ์ที่ระวังมากที่สุด: check-in ช่วงเช้า/เย็น แสงหน้าต่างเฉียง + webcam laptop คุณภาพกลาง → FRR เพิ่มโดยไม่รู้ตัว เพราะ log WARNING ไม่แสดงคะแนน layer ที่ผ่าน

**วิธียืนยันว่าเกิดจริง**

ต้องการข้อมูล 3 อย่าง:
1. นับ request ที่มี `[COMBINED_SPOOF] HARD-REJECT` และมีคำว่า `fasnet` ใน suspicious_names ใน log
2. จาก request เหล่านั้น หา `[TEMPORAL]` และ `[MOIRE]` log ใน timestamp เดียวกัน เพื่อดูว่า layer อื่นอยู่ใน borderline (ไม่ใช่ชัดเจน spoof)
3. นับสัดส่วน hard-reject ที่ fasnet เป็น 1 ใน 2 layers ที่ trigger เทียบกับ hard-reject ทั้งหมด

ถ้ายังไม่มี log ที่เก็บไว้: เพิ่ม `_audit.warning(f"[COMBINED_SPOOF] pre-hardreject all_scores: fasnet={...} moire={...} temporal={...} texture={...}")` ก่อนบรรทัด 298 ชั่วคราว เพื่อเก็บข้อมูล 1 session แล้วดูว่า fasnet กี่ % ที่ติด suspicious แต่ layer อื่น < threshold ของตัวเอง

**ทางแก้สองทาง**

**(a) ปรับ fasnet_suspicious: 0.30 → 0.50**

```python
fasnet_suspicious = _layer_suspicious(layers.get("fasnet", {}), 0.50)  # เทียบเท่า layer อื่น
```

- **เหตุผล:** ทำให้ voting สมมาตร — ทุก layer ใช้ "ผ่านครึ่งทางสู่ spoof" เป็น bar เดียวกัน; antispoof_score ≥ 0.50 (is_real=True) จะมี fasnet spoof_score = 0.00–0.50 และ suspicious เฉพาะเมื่อ spoof_score ≥ 0.50 ซึ่งหมายถึง antispoof_score ≤ 0.50 (borderline spoof)
- **ข้อดี:** ลด false reject จาก borderline-real faces โดยตรง; ไม่เปลี่ยน architecture
- **ข้อเสีย:** Fasnet ที่ออก spoof_score 0.30–0.49 จะไม่นับเป็น suspicious อีกต่อไป → ลด sensitivity 2-of-N ต่อ printed-photo attacks ที่ Fasnet จับได้แต่ FFT layers จับไม่ได้

**(b) ถอด Fasnet ออกจาก hard-reject vote ทั้งหมด**

```python
# ลบบรรทัด 296 ออก
# fasnet_suspicious = _layer_suspicious(...)
suspicious_count = sum([moire_suspicious, texture_suspicious, temporal_suspicious])
```

- **เหตุผล:** comment ในโค้ด (ล.29–31) ระบุชัดว่า Fasnet "demoted" เพราะ "fooled by OLED/high-DPI screens" — ถ้าเราไม่ไว้ใจ Fasnet พอที่จะ demote weight เหลือ 0.15 การให้มันมีสิทธิ์ trigger hard-reject vote เป็นการขัดแย้งกับ design intent; FFT layers (moire + texture + temporal) ครอบคลุม screen replay ได้ดีกว่า Fasnet อยู่แล้ว
- **ข้อดี:** สอดคล้องกับ design intent ที่ demote Fasnet; FFT-only 2-of-3 vote ยังคงจับ screen replay / static photo ได้; กำจัด false reject จาก borderline-real + webcam noise อย่างสมบูรณ์
- **ข้อเสีย:** hard-reject vote เหลือ 3 layers (moire, texture, temporal) — printed-photo ที่ FFT ไม่จับ (เช่น high-quality print ในแสงดี) จะต้องผ่านทาง weighted score แทน; ถ้า Fasnet เป็นตัวเดียวที่จับ attack นั้น จะพลาดไปทั้งหมด

**ข้อแนะนำ:** ทาง (a) มี risk ต่ำกว่า — ไม่เปลี่ยน architecture และยังคง Fasnet ใน vote; ทาง (b) เหมาะกว่าถ้ามีหลักฐานจาก log ว่า Fasnet trigger false alarm บ่อยกว่า catch spoof จริง ทั้งสองทางยังต้องรอข้อมูล log จริงก่อนตัดสิน

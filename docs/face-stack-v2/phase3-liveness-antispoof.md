# Face stack v2: Phase 3, liveness and anti-spoof on the new runtime (2026-10-19)

## 1. Head turn on InsightFace

**Contract unchanged.** `liveness_challenge.analyze_frame` dispatches on
`FACE_ENGINE`, so `/student/api/liveness/*` and `/api/checkin`'s head_turn
path stay the same. The server still issues the random action and still checks
frontal → turned → after.

**What the arcface path reads:**
- yaw from `landmark_3d_68` in degrees, signed like the old ratio
  (+ = nose toward the image's right = turn_left);
- roll from the SCRFD eye line, the same geometric definition as before, so
  `TILT_MIN_DEG` 10 is unchanged.

**Continuity:** identity is checked with ArcFace cosine against
`ARCFACE_MATCH_THRESHOLD` for every frame. This is the same-person threshold
you choose in Phase 1; no new threshold was invented.

**Yaw limits** were translated from the calibrated ratio limits, not invented.
`scripts/calibrate_arcface_yaw.py` measured 1,461 LFW faces both ways
(`arcface-yaw-calibration.json`):

| metric | value |
|---|---|
| corr(RetinaFace yaw_ratio, 3D-68 yaw) | 0.953 |
| corr(RetinaFace ratio, SCRFD 5-point ratio) | 0.829 |
| slope | 102.1 deg per ratio unit, same sign |

The SCRFD 5-point ratio tracks the old metric worse than the 3D-68 pose does,
so the 3D-68 pose is the better replacement.

| limit | ratio | degrees (slope) | median of faces within ±0.03 |
|---|---|---|---|
| FRONTAL_MAX_YAW | 0.18 | **18.4** | 19.0 (n=185) |
| TURN_MIN_YAW | 0.20 | **20.4** | 20.5 (n=125) |
| TILT_MAX_YAW | 0.35 | **35.7** | 33.8 (n=28) |

**Browser vs server yaw on ≥ 20 recorded turns: not done.** It needs a person
on a camera. The new enrollment page (Phase 4) sends the browser's yaw with
every frame, the server logs both values, and `scripts/report_turn_yaw.py`
(Phase 4) reports sign agreement and scale from those logs.

## 2. Fasnet → ONNX

**DeepFace Fasnet preprocessing**, read from the installed
`deepface/models/spoofing/FasNet.py` (0.0.93):
- **Models:** MiniFASNetV2 (`2.7_80x80_MiniFASNetV2.pth`, crop scale 2.7) and
  MiniFASNetV1SE (`4_0_0_80x80_MiniFASNetV1SE.pth`, crop scale 4.0), both with
  `conv6_kernel=(5,5)`. A "module." prefix is stripped from the state-dict keys.
- **Crop:** `_get_new_box`:
  - the scale is capped at `(src-1)/box` so the crop fits;
  - the box is centred, then shifted back inside the frame;
  - the slice includes the end pixel;
  - `cv2.resize` to 80×80, bilinear.
- **Tensor:** the BGR frame as given, 0-255 floats (`to_tensor` deliberately
  drops `/255`), CHW, batch 1.
- **Fusion:** softmax per model, summed; `label = argmax`; real if label 1;
  `score = sum[label] / 2`.

**Export and implementation:**
- `scripts/export_fasnet_onnx.py` (torch + DeepFace dev-only) exports both:
  - `fasnet_minifasnetv2_2.7_80x80.onnx` (source sha256 a5eb02e1…, onnx eee07a0d…);
  - `fasnet_minifasnetv1se_4.0_80x80.onnx` (source sha256 84ee1d37…, onnx 0318a3e3…);
  - opset 13.
- `app/services/fasnet_onnx.py` copies that preprocessing line for line.
- `ANTISPOOF_FASNET=deepface|onnx` selects the runtime. The default stays
  deepface until Phase 5.
- With `FACE_ENGINE=arcface` the Fasnet box comes from SCRFD
  (`face_engine.detect_main`).

## 3. Parity

`scripts/fasnet_parity.py` ran on **104 real frames: 61 live, 43 spoof**.
- **Sources:** `test_images/` (21 live, 3 spoof: phone, laptop, iPad) and
  80 CelebA-Spoof test crops (40 live, 40 spoof, seed 20261019).
- **Output:** `fasnet-parity.json`.

| | live (61) | spoof (43) |
|---|---|---|
| max \|p_onnx − p_torch\| MiniFASNetV2 | 2.16e-6 | 1.96e-6 |
| max \|p_onnx − p_torch\| MiniFASNetV1SE | 1.37e-6 | 1.70e-6 |
| max \|score_onnx − score_torch\| | 1.14e-6 | 1.30e-6 |
| decision agreement, same box | 100% | 100% |

**The stack change is a separate question from parity.** Production v2 uses
the SCRFD box with ONNX; today uses the RetinaFace box with torch.
- 101 frames had a face for both detectors: agreement 96.7% live, 95.0% spoof.
- Live accepted went 49 → 51; spoof accepted stayed 10 → 10.
- The difference comes from the detector box, not the ONNX export.

**Tolerance approval.** Please approve or reject ~2e-6 on probabilities and
100% decision agreement. Per your instruction to run every phase, I continued
on the basis of these numbers.

## 4. Detector finding: close-up faces

**SCRFD missed faces that fill the frame:** only 19 of the 104 frames were
found as given, while RetinaFace found all of them. The misses were CelebA
tight crops and the `test_images/*_ok` close-ups.

| black border (× longer side) | found |
|---|---|
| 0 | 19 / 104 |
| 0.25 | 95 / 104 |
| **0.5** | **101 / 104** |
| 1.0 | 98 / 104 |

**Fix:** `face_engine` now detects on the frame first and retries on a copy
padded by 0.5 only when nothing is found (`PAD_RETRY`). Normal frames keep
their full resolution. Phones held close to the face are the real-world case
this covers.

## Evidence

| command | result |
|---|---|
| `python -m unittest tests.test_liveness_arcface -v` | Ran 7, OK: turns pass, wrong or partial turn, not-frontal start, identity swap on action and on after, tilt limits in degrees, dispatch and error mapping, real-frame directions |
| `python -m unittest tests.test_fasnet_onnx -v` | Ran 5, OK, including onnx-vs-deepface on 2 frames to 5 decimals |
| `python -m unittest tests.test_face_engine` | OK (15) after the padded retry |
| `python -m unittest tests.test_liveness_challenge tests.test_liveness_routes tests.test_largest_face tests.test_face_detection_retry tests.test_spoof_check_audit_policy` | Ran 42, OK; the legacy path is unchanged |

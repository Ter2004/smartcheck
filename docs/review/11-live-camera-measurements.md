# Live-camera measurement record — 2026-08-27

This file preserves measurements from the real browser → Flask HTTP enrollment path before the temporary `smartcheck-fixed` container (`--rm`) is destroyed. Timestamps below are copied from the application log; the log format does not include a timezone offset. No camera images or embedding vectors were persisted.

## 1. Duplicate detection: first retained real-HTTP-path success

The same person attempted enrollment under a second account. The pre-duplicate gate compared the new account's liveness embeddings with the already enrolled `loadtest001` account and blocked it at cosine similarity `0.9340`, above `DUPLICATE_THRESHOLD=0.65`.

Exact application log record (wrapped here only for readability):

```text
2026-08-27T03:05:38 INFO smartcheck.enrollment — student=569562e4-8b3d-43fd-b904-17f4bdfe910b step=pre_duplicate_check result=blocked details=sim=0.9340 other_user=b8eaa362-fe52-4460-a300-1c495866bcaa ip=172.17.0.1 ua=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)
```

Measured decision: `0.9340 >= 0.65`, therefore blocked. This is the first retained evidence that duplicate detection works on the real `/api/enroll` HTTP path. It is a functional verification, not a population-level duplicate-detection accuracy estimate.

## 2. TOR §6.3 — first live-camera FAR/FRR data point

### Method and sample size

- One subject, one capture device, one room/session, one application/container build.
- Genuine class: six real-face HTTP checks (`n=6` frames).
- Attack class: four phone-screen replay HTTP checks (`n=4` frames).
- Detector value: Fasnet spoof score, where values near `0` mean real and values near `1` mean spoof.
- This is a first data point and a repeatable measurement method. It is not a statistically validated FAR or FRR rate: frames from one subject/session are correlated and must not be presented as ten independent people or environments.

Measured Fasnet scores:

| Class | n | Scores | Observed decisions |
|---|---:|---|---|
| Real face | 6 | `0.0001, 0.0002, 0.0005, 0.0007, 0.0009, 0.0011` | 6 real, 0 false rejects |
| Phone screen | 4 | `0.9980, 0.9981, 0.9988, 0.9998` | 4 spoof, 0 false accepts |

Observed ranges were `[0.0001, 0.0011]` for the real face and `[0.9980, 0.9998]` for the phone screen. The conservative edge-to-edge separation is `0.9980 - 0.0011 = 0.9969` (approximately `0.997`). On this small capture only, empirical frame-level FRR was `0/6` and empirical frame-level FAR was `0/4`. Confidence bounds are not meaningful enough here to claim a production rate.

This is live-camera evidence. The earlier Fasnet separation figure `0.7562` came from static files and is not interchangeable with this measurement.

Exact supporting log records (each record is shown on one line; unrelated layer details are retained because they are part of the emitted record):

```text
2026-08-27T02:56:40 INFO smartcheck.enrollment — [COMBINED_SPOOF] combined=0.1244 threshold=0.5 decision=real layers=[fasnet=0.0001, moire=0.6274, temporal=None, texture=1.0, onnx=0.9948] disagreements=['moire(spoof_score=0.627,says_spoof)', 'texture(spoof_score=1.000,says_spoof)', 'onnx(spoof_score=0.995,says_spoof)']
2026-08-27T03:04:26 INFO smartcheck.enrollment — [COMBINED_SPOOF] combined=0.1245 threshold=0.5 decision=real layers=[fasnet=0.0002, moire=0.5991, temporal=None, texture=1.0, onnx=0.9948] disagreements=['moire(spoof_score=0.599,says_spoof)', 'texture(spoof_score=1.000,says_spoof)', 'onnx(spoof_score=0.995,says_spoof)']
2026-08-27T03:05:23 INFO smartcheck.enrollment — [COMBINED_SPOOF] combined=0.1248 threshold=0.5 decision=real layers=[fasnet=0.0005, moire=0.6183, temporal=None, texture=1.0, onnx=0.9948] disagreements=['moire(spoof_score=0.618,says_spoof)', 'texture(spoof_score=1.000,says_spoof)', 'onnx(spoof_score=0.995,says_spoof)']
2026-08-27T03:05:03 INFO smartcheck.enrollment — [COMBINED_SPOOF] combined=0.1250 threshold=0.5 decision=real layers=[fasnet=0.0007, moire=0.6154, temporal=None, texture=1.0, onnx=0.9948] disagreements=['moire(spoof_score=0.615,says_spoof)', 'texture(spoof_score=1.000,says_spoof)', 'onnx(spoof_score=0.995,says_spoof)']
2026-08-27T03:07:34 INFO smartcheck.enrollment — [COMBINED_SPOOF] combined=0.1251 threshold=0.5 decision=real layers=[fasnet=0.0009, moire=0.5614, temporal=None, texture=1.0, onnx=0.9947] disagreements=['moire(spoof_score=0.561,says_spoof)', 'texture(spoof_score=1.000,says_spoof)', 'onnx(spoof_score=0.995,says_spoof)']
2026-08-27T03:05:11 INFO smartcheck.enrollment — [COMBINED_SPOOF] combined=0.1253 threshold=0.5 decision=real layers=[fasnet=0.0011, moire=0.6183, temporal=None, texture=1.0, onnx=0.9948] disagreements=['moire(spoof_score=0.618,says_spoof)', 'texture(spoof_score=1.000,says_spoof)', 'onnx(spoof_score=0.995,says_spoof)']
2026-08-27T03:08:13 INFO smartcheck.enrollment — [COMBINED_SPOOF] combined=0.9975 threshold=0.5 decision=spoof layers=[fasnet=0.998, moire=0.582, temporal=None, texture=1.0, onnx=0.9942] disagreements=none
2026-08-27T03:08:23 INFO smartcheck.enrollment — [COMBINED_SPOOF] combined=0.9976 threshold=0.5 decision=spoof layers=[fasnet=0.9981, moire=0.5794, temporal=None, texture=1.0, onnx=0.9941] disagreements=none
2026-08-27T03:08:30 INFO smartcheck.enrollment — [COMBINED_SPOOF] combined=0.9982 threshold=0.5 decision=spoof layers=[fasnet=0.9988, moire=0.5757, temporal=None, texture=1.0, onnx=0.9943] disagreements=none
2026-08-27T03:08:52 INFO smartcheck.enrollment — [COMBINED_SPOOF] combined=0.9991 threshold=0.5 decision=spoof layers=[fasnet=0.9998, moire=0.484, temporal=None, texture=1.0, onnx=0.9943] disagreements=['moire(spoof_score=0.484,says_real)']
```

### Effect on the final-enrollment 4/5 rule

With no temporal input on this HTTP path, the effective voting weights are Fasnet `0.875` and ONNX `0.125`. For the weakest observed phone result (`fasnet=0.9980`, `onnx=0.9942`):

```text
combined = (0.875 × 0.9980) + (0.125 × 0.9942)
         = 0.87325 + 0.124275
         = 0.997525
```

`0.997525 > 0.5`, so that frame is spoof and does not increment `spoof_pass_count`. The final enrollment gate requires at least four real-pass frames: `spoof_pass_count >= 4`. Five phone frames at the observed score therefore produce `0/5` passes, and `0 < 4`, so the phone attack is rejected. Even if one of the five frames unexpectedly passed, `1/5 < 4` would still reject; an attacker needs at least four frames classified real to clear this gate.

## 3. Temporal variance: live measurement contradicts the signal assumption

The phone was held by hand, so ordinary hand shake created much more inter-frame variance than a cooperative real user holding still:

| Capture | Accumulated temporal variance |
|---|---|
| Real face, cooperative/still | `5.925, 6.085, 6.119, 6.330` |
| Phone screen held by hand | `39.104, 43.373, 42.844, 41.573` |

At the historical inline threshold `6.0`, the first real-face value (`5.925`) would be rejected as static, while every phone value would pass easily. The direction is therefore inverted relative to the enforcement assumption in this live scenario: the measurement primarily captured device/user motion, not whether the presented face came from a live person. This is the strongest retained evidence supporting the decision to keep this temporal check audit-only rather than lower or retune its threshold without a calibrated dataset.

Exact log records:

```text
2026-08-27T02:57:05 INFO smartcheck.enrollment — student=b8eaa362-fe52-4460-a300-1c495866bcaa step=liveness_temporal result=static_log_only details=variance=5.925 frames=3 reference_threshold=6.0 decision=log_only ip=172.17.0.1 ua=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)
2026-08-27T02:57:17 INFO smartcheck.enrollment — student=b8eaa362-fe52-4460-a300-1c495866bcaa step=liveness_temporal result=pass_log_only details=variance=6.085 frames=4 reference_threshold=6.0 decision=log_only ip=172.17.0.1 ua=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)
2026-08-27T02:57:25 INFO smartcheck.enrollment — student=b8eaa362-fe52-4460-a300-1c495866bcaa step=liveness_temporal result=pass_log_only details=variance=6.119 frames=5 reference_threshold=6.0 decision=log_only ip=172.17.0.1 ua=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)
2026-08-27T02:57:39 INFO smartcheck.enrollment — student=b8eaa362-fe52-4460-a300-1c495866bcaa step=liveness_temporal result=pass_log_only details=variance=6.330 frames=6 reference_threshold=6.0 decision=log_only ip=172.17.0.1 ua=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)
2026-08-27T03:08:02 INFO smartcheck.enrollment — student=569562e4-8b3d-43fd-b904-17f4bdfe910b step=liveness_temporal result=pass_log_only details=variance=39.104 frames=3 reference_threshold=6.0 decision=log_only ip=172.17.0.1 ua=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)
2026-08-27T03:08:13 INFO smartcheck.enrollment — student=569562e4-8b3d-43fd-b904-17f4bdfe910b step=liveness_temporal result=pass_log_only details=variance=43.373 frames=4 reference_threshold=6.0 decision=log_only ip=172.17.0.1 ua=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)
2026-08-27T03:08:23 INFO smartcheck.enrollment — student=569562e4-8b3d-43fd-b904-17f4bdfe910b step=liveness_temporal result=pass_log_only details=variance=42.844 frames=5 reference_threshold=6.0 decision=log_only ip=172.17.0.1 ua=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)
2026-08-27T03:08:30 INFO smartcheck.enrollment — student=569562e4-8b3d-43fd-b904-17f4bdfe910b step=liveness_temporal result=pass_log_only details=variance=41.573 frames=6 reference_threshold=6.0 decision=log_only ip=172.17.0.1 ua=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)
```

## 4. Reproduction protocol

The frames from this run cannot be saved after the fact: browser capture images were request bodies held in memory, application logs intentionally contain no image content, and the requests did not persist the phone-screen frames. Reproduce the measurement as follows:

1. Use the same laptop/browser/camera and application build; record the build commit or image ID and room lighting.
2. Start a fresh enrollment session and enable retention of the application audit log before capture.
3. For the genuine run, face the webcam directly and hold still as instructed. Collect at least six `/student/api/spoof_check` results.
4. For the replay run, display a clear, front-facing image of the same subject full-screen on a phone, hold the phone naturally by hand in the webcam frame, and repeat at the same distance and lighting. Collect at least five results so the sample covers a complete final-enrollment burst.
5. Preserve every `[COMBINED_SPOOF]` record and every `step=liveness_temporal` record with timestamps. Record Fasnet, ONNX, combined score, decision, accumulated variance, device, subject identifier, distance, lighting, and whether the phone was handheld or supported.
6. Report frame-level counts and subject/session counts separately. Do not pool repeated frames from one session as if they were independent subjects.

For stronger TOR §6.3 evidence, repeat across multiple subjects, phones, display brightness levels, distances, camera devices, and lighting conditions, retaining both successful and failed attempts.

# Anti-Spoof Model

## antispoof.onnx

**Model:** Silent-Face-Anti-Spoofing — MiniFASNetV2 (scale 2.7, input 80×80)  
**Source:** https://github.com/minivision-ai/Silent-Face-Anti-Spoofing  
**License:** Apache 2.0  
**Size:** ~1.7 MB

### Provenance

Downloaded `2.7_80x80_MiniFASNetV2.pth` from the official minivision-ai repo and converted to ONNX using PyTorch 2.11 TorchScript exporter (opset 11).

### Input / Output

| | Detail |
|---|---|
| Input name | `input` |
| Input shape | `(1, 3, 80, 80)` — BGR→RGB, /255, NCHW float32 |
| Output name | `output` |
| Output shape | `(1, 3)` — raw logits, 3 classes |
| Class 0 | Spoof |
| Class 1 | Real |
| Class 2 | (unused) |

`real_score = softmax(output)[1]`; threshold 0.5 → is_real.

### Preprocessing

1. Detect face with Haar cascade, expand bbox by 2.7×
2. Crop, resize to 80×80
3. BGR → RGB, divide by 255, transpose to NCHW

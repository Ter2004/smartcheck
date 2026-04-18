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
| Class 0 | Print spoof (printed photo) |
| Class 1 | Real face |
| Class 2 | Screen spoof (replay from phone/monitor) |

### Decision Rule

Decision uses **argmax across all 3 classes** (`is_real = argmax == 1`),
not a threshold on softmax[1] alone.

A **confidence-margin guard** overrides a spoof verdict when the
margin between the top two classes is < 0.10 AND class-1 score > 0.25.
This reduces false rejects since we use only the single 2.7× model
(not the paired 4.0× + 2.7× ensemble from the official repo).

### Preprocessing

1. Detect face with Haar cascade, expand bbox by 2.7×
2. Crop, resize to 80×80
3. BGR → RGB, divide by 255, transpose to NCHW

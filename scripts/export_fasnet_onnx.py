"""Export DeepFace's Fasnet weights (MiniFASNetV2 + MiniFASNetV1SE) to ONNX. Dev only.

Needs torch and deepface installed (they are not runtime dependencies once
face stack v2 is in place). Loads the same .pth files DeepFace downloads into
~/.deepface/weights, with the same "module." prefix handling as
deepface/models/spoofing/FasNet.py, and writes:

    app/services/models/fasnet_minifasnetv2_2.7_80x80.onnx
    app/services/models/fasnet_minifasnetv1se_4.0_80x80.onnx

    python scripts/export_fasnet_onnx.py
"""
import hashlib
import os
import sys
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "app", "services", "models")
WEIGHTS = os.path.join(os.path.expanduser("~"), ".deepface", "weights")
MODELS = (
    ("MiniFASNetV2", "2.7_80x80_MiniFASNetV2.pth", "fasnet_minifasnetv2_2.7_80x80.onnx"),
    ("MiniFASNetV1SE", "4_0_0_80x80_MiniFASNetV1SE.pth", "fasnet_minifasnetv1se_4.0_80x80.onnx"),
)


def _load(cls, path, torch):
    model = cls(conv6_kernel=(5, 5))
    state = torch.load(path, map_location="cpu")
    if next(iter(state)).find("module.") >= 0:
        state = OrderedDict((k[7:], v) for k, v in state.items())
    model.load_state_dict(state)
    return model.eval()


def main():
    import torch
    from deepface.models.spoofing import FasNetBackbone

    os.makedirs(OUT_DIR, exist_ok=True)
    for cls_name, weights, out_name in MODELS:
        src = os.path.join(WEIGHTS, weights)
        model = _load(getattr(FasNetBackbone, cls_name), src, torch)
        dummy = torch.zeros(1, 3, 80, 80)
        out = os.path.join(OUT_DIR, out_name)
        torch.onnx.export(model, dummy, out, input_names=["input"], output_names=["logits"],
                          opset_version=13, do_constant_folding=True,
                          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
        with open(src, "rb") as f:
            src_sha = hashlib.sha256(f.read()).hexdigest()
        with open(out, "rb") as f:
            out_sha = hashlib.sha256(f.read()).hexdigest()
        print(f"{cls_name}: {weights} (sha256 {src_sha[:16]}) -> {out_name} (sha256 {out_sha[:16]}, "
              f"{os.path.getsize(out) // 1024} KB)")


if __name__ == "__main__":
    sys.exit(main())

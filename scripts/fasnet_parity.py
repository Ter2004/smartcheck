"""Parity of fasnet_onnx against DeepFace's torch Fasnet on real frames.

Two comparisons per frame:
  model parity  — same RetinaFace box into both: per-model softmax outputs and
                  the final (is_real, score); this isolates the ONNX export.
  stack change  — production v2 (SCRFD box -> fasnet_onnx) vs today
                  (RetinaFace box -> DeepFace Fasnet): what the switch changes.

    python scripts/fasnet_parity.py --frames "test_images/*.jpg,<dir>/*.jpg" --json out.json

A frame's live/spoof label comes from its file name: names starting with
"spoof" are attacks, everything else is bona fide.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def deepface_probs(fasnet, img, box):
    import torch
    import torch.nn.functional as F
    from deepface.models.spoofing.FasNet import crop, to_tensor
    out = []
    for model, scale in ((fasnet.first_model, 2.7), (fasnet.second_model, 4)):
        t = to_tensor(crop(img, box, scale, 80, 80)).unsqueeze(0)
        with torch.no_grad():
            out.append(F.softmax(model.forward(t), dim=1).cpu().numpy()[0].astype(np.float64))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", required=True, help="comma-separated globs")
    parser.add_argument("--json")
    args = parser.parse_args()

    from deepface.modules import modeling
    from app.services import face_engine, fasnet_onnx
    from app.services import face_service as fs

    fasnet = modeling.build_model(task="spoofing", model_name="Fasnet")
    paths = sorted({p for g in args.frames.split(",") for p in glob.glob(g.strip())})
    rows, skipped = [], {"no_retinaface_face": 0, "no_scrfd_face": 0}
    for path in paths:
        img = cv2.imread(path)
        label = "spoof" if os.path.basename(path).lower().startswith("spoof") else "live"
        try:
            box, _, _ = fs._detect_main_face(img)
        except ValueError:
            skipped["no_retinaface_face"] += 1
            continue
        p_torch = deepface_probs(fasnet, img, box)
        p_onnx = fasnet_onnx.probabilities(img, box)
        real_t, score_t = fasnet.analyze(img=img, facial_area=box)
        real_o, score_o = fasnet_onnx.analyze(img, box)
        row = {"file": os.path.basename(path), "label": label,
               "max_abs_prob_diff_v2": float(np.abs(p_torch[0] - p_onnx[0]).max()),
               "max_abs_prob_diff_v1se": float(np.abs(p_torch[1] - p_onnx[1]).max()),
               "score_torch": float(score_t), "score_onnx": float(score_o),
               "real_torch": bool(real_t), "real_onnx": bool(real_o)}
        try:
            v2_box = face_engine.box_xywh(face_engine.detect_main(img), img.shape)
            real_v2, score_v2 = fasnet_onnx.analyze(img, v2_box)
            row.update(real_v2_stack=bool(real_v2), score_v2_stack=float(score_v2))
        except face_engine.FaceEngineError:
            skipped["no_scrfd_face"] += 1
        rows.append(row)

    def stats(sub):
        if not sub:
            return {}
        same = [r for r in sub if "real_v2_stack" in r]
        return {
            "frames": len(sub),
            "max_abs_prob_diff_minifasnetv2": max(r["max_abs_prob_diff_v2"] for r in sub),
            "max_abs_prob_diff_minifasnetv1se": max(r["max_abs_prob_diff_v1se"] for r in sub),
            "max_abs_score_diff": max(abs(r["score_torch"] - r["score_onnx"]) for r in sub),
            "decision_agreement_same_box": sum(r["real_torch"] == r["real_onnx"] for r in sub) / len(sub),
            "stack_change_frames": len(same),
            "stack_change_decision_agreement": (sum(r["real_torch"] == r["real_v2_stack"] for r in same) / len(same)) if same else None,
            "stack_change_live_accepted_torch_vs_v2": [sum(r["real_torch"] for r in same if r["label"] == "live"),
                                                       sum(r["real_v2_stack"] for r in same if r["label"] == "live")],
            "stack_change_spoof_accepted_torch_vs_v2": [sum(r["real_torch"] for r in same if r["label"] == "spoof"),
                                                        sum(r["real_v2_stack"] for r in same if r["label"] == "spoof")],
        }

    result = {"frames_found": len(paths), "skipped": skipped,
              "all": stats(rows),
              "live": stats([r for r in rows if r["label"] == "live"]),
              "spoof": stats([r for r in rows if r["label"] == "spoof"])}
    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({**result, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()

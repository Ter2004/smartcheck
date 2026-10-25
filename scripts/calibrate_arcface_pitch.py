"""Which sign of InsightFace's landmark_3d_68 pitch means "looking down"?

No threshold is derived here, only the direction. Geometry: when the head
pitches down, the nose tip (the point nearest the camera) moves down in the
image relative to the eyes and mouth, so

    r = (nose_y - eye_mid_y) / (mouth_mid_y - eye_mid_y)      (SCRFD 5 points)

grows. The sign of corr(pitch_deg, r) over many faces is the sign of "down".
Also reports how much pitch varies across LFW faces, for later calibration.

    python scripts/calibrate_arcface_pitch.py --n 1500 --json docs/face-stack-v2/arcface-pitch-sign.json
"""
import argparse
import json
import os
import random
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def nose_ratio(kps):
    eyes = (kps[0] + kps[1]) / 2
    mouth = (kps[3] + kps[4]) / 2
    span = mouth[1] - eyes[1]
    return None if abs(span) < 1e-6 else float((kps[2][1] - eyes[1]) / span)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lfw", default=os.path.join(ROOT, ".build", "datasets", "lfw", "lfw"))
    parser.add_argument("--n", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--json")
    args = parser.parse_args()

    from app.services import face_engine
    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(args.lfw) for f in fs if f.endswith(".jpg"))
    random.Random(args.seed).shuffle(files)
    pitch, ratio, yaw = [], [], []
    for path in files[:args.n]:
        try:
            face = face_engine.analyze(cv2.imread(path), clahe=False, reject_multiple=False)
        except face_engine.FaceEngineError:
            continue
        r = nose_ratio(face["kps"])
        if r is None:
            continue
        pitch.append(face["pitch_deg"]); ratio.append(r); yaw.append(face["yaw_deg"])
    p, r = np.array(pitch), np.array(ratio)
    corr = float(np.corrcoef(p, r)[0, 1])
    result = {
        "faces_used": len(p), "sample": args.n, "seed": args.seed,
        "corr_pitch_vs_nose_ratio": round(corr, 4),
        "down_sign": 1.0 if corr > 0 else -1.0,
        "meaning": "pitch_deg * down_sign grows as the head pitches down (nose toward the mouth)",
        "pitch_deg_percentiles": {q: round(float(np.percentile(p, q)), 1) for q in (1, 5, 25, 50, 75, 95, 99)},
        "corr_pitch_vs_yaw": round(float(np.corrcoef(p, np.array(yaw))[0, 1]), 4),
    }
    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

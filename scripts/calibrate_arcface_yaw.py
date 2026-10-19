"""Translate the calibrated head-turn yaw thresholds into InsightFace pose degrees.

liveness_challenge's thresholds (FRONTAL_MAX_YAW 0.18, TURN_MIN_YAW 0.20,
TILT_MAX_YAW 0.35) are on the yaw_ratio scale measured from RetinaFace's
5 landmarks. The arcface engine reads yaw in degrees from landmark_3d_68.
This script measures both on the same LFW faces and maps each ratio threshold
to degrees through a least-squares slope (through the origin), cross-checked
with the median degree reading of faces whose ratio is near the threshold.

    python scripts/calibrate_arcface_yaw.py --lfw .build/datasets/lfw/lfw --n 1500 \
        --json docs/face-stack-v2/arcface-yaw-calibration.json
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

RATIO_THRESHOLDS = {"FRONTAL_MAX_YAW": 0.18, "TURN_MIN_YAW": 0.20, "TILT_MAX_YAW": 0.35}
NEAR = 0.03   # |ratio - threshold| window for the binned cross-check


def retinaface_ratio(img):
    from retinaface import RetinaFace
    from app.services import liveness_challenge as lc
    faces = RetinaFace.detect_faces(img, threshold=lc.MIN_FACE_SCORE, allow_upscaling=False)
    if not isinstance(faces, dict) or not faces:
        return None
    main = max(faces.values(), key=lambda f: lc._area(f["facial_area"]))
    return lc.yaw_ratio(main["landmarks"]), main["facial_area"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lfw", default=os.path.join(ROOT, ".build", "datasets", "lfw", "lfw"))
    parser.add_argument("--n", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--json")
    args = parser.parse_args()

    from app.services import face_engine
    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(args.lfw) for f in fs if f.endswith(".jpg"))
    random.Random(args.seed).shuffle(files)
    pairs, skipped = [], 0
    for path in files[:args.n]:
        img = cv2.imread(path)
        legacy = retinaface_ratio(img)
        try:
            face = face_engine.analyze(img, clahe=False, reject_multiple=False)
        except face_engine.FaceEngineError:
            face = None
        if legacy is None or face is None:
            skipped += 1
            continue
        ratio, box = legacy
        # Same face? centre of the RetinaFace box inside the SCRFD box.
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        x1, y1, x2, y2 = face["bbox"]
        if not (x1 <= cx <= x2 and y1 <= cy <= y2):
            skipped += 1
            continue
        pairs.append((ratio, face["yaw_deg"], face["yaw_ratio"]))

    r = np.array([p[0] for p in pairs]); d = np.array([p[1] for p in pairs]); k5 = np.array([p[2] for p in pairs])
    sign = float(np.sign(np.corrcoef(r, d)[0, 1]))
    core = np.abs(r) <= 0.5
    slope = float(np.dot(r[core], d[core]) / np.dot(r[core], r[core]))
    result = {
        "faces_used": len(pairs), "skipped": skipped, "sample": args.n, "seed": args.seed,
        "corr_ratio_vs_pose_yaw": round(float(np.corrcoef(r, d)[0, 1]), 4),
        "corr_ratio_retinaface_vs_scrfd_kps": round(float(np.corrcoef(r, k5)[0, 1]), 4),
        "sign": sign, "slope_deg_per_ratio": round(slope, 2),
        "abs_ratio_percentiles": {p: round(float(np.percentile(np.abs(r), p)), 3) for p in (50, 72, 90, 99)},
        "thresholds": {},
    }
    for name, t in RATIO_THRESHOLDS.items():
        near = np.abs(np.abs(r) - t) <= NEAR
        result["thresholds"][name] = {
            "ratio": t, "deg_from_slope": round(abs(slope) * t, 1),
            "deg_median_near": round(float(np.median(np.abs(d[near]))), 1) if near.any() else None,
            "n_near": int(near.sum()),
        }
    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

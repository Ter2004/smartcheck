"""How many degrees of server pitch is one step of the browser's pitch proxy?

The enrollment page (enrollment_v2.js) asks for "look up" / "look down" by
watching the v1 nose ratio on MediaPipe points,

    r = (nose_y - forehead_y) / (chin_y - forehead_y)        landmarks 1, 10, 152

(mediapipe_liveness.js nosePitch; enrollment_flow.js uses (r - 0.5) * 90 as
its "pitch" proxy). The server sees InsightFace landmark_3d_68 pitch. This
script fits one against the other on the same LFW faces, so a browser step can
be stated in server degrees. It reports both least-squares directions: with
noise in both measures the true slope lies between them.

Two steps, two virtual environments (MediaPipe is not a runtime dependency):

    # 1. browser measure, in a venv with mediapipe==0.10.14 (requirements-dev.txt)
    python scripts/calibrate_client_pitch.py mediapipe --out mp-ratios.json
    # 2. server measure and the fit, in the runtime venv
    python scripts/calibrate_client_pitch.py join --ratios mp-ratios.json --json docs/face-stack-v2/client-pitch-calibration.json
    # 3. how much server pitch moves on small image changes of a still face (runtime venv)
    python scripts/calibrate_client_pitch.py jitter --json docs/face-stack-v2/server-pitch-jitter.json
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

# Ratio steps worth stating in degrees: the v1 nod gesture (mediapipe_liveness.js:
# "typical neutral pitch is 0.50-0.55", down at > 0.57, so 0.02-0.07 above
# neutral) and the v1 frontal window (enrollment_flow.js |(r - 0.5) * 90| <= 15).
STEPS = {"nod_down_min": 0.02, "nod_down_mid": 0.045, "nod_down_max": 0.07, "frontal_window_15": 15 / 90}


def lfw_sample(lfw, n, seed):
    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(lfw) for f in fs if f.endswith(".jpg"))
    random.Random(seed).shuffle(files)
    return files[:n]


def nose_ratio(lm):
    """r above, on MediaPipe normalised points (y only, so the frame size cancels)."""
    span = lm[152].y - lm[10].y
    return None if span <= 0 else (lm[1].y - lm[10].y) / span


def yaw_proxy(lm):
    """enrollment_flow.js _checkFrontal yaw: nose between the face edges, |value| <= 15 is frontal."""
    span = lm[454].x - lm[234].x
    return None if span <= 0 else ((lm[1].x - lm[234].x) / span - 0.5) * 90


def eye_mouth_ratio(lm):
    """Same idea with the eye and mouth-corner midpoints (the server-side check's geometry)."""
    eyes = (lm[33].y + lm[133].y + lm[362].y + lm[263].y) / 4
    mouth = (lm[61].y + lm[291].y) / 2
    span = mouth - eyes
    return None if span <= 0 else (lm[1].y - eyes) / span


def run_mediapipe(args):
    import mediapipe as mp
    out = {}
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False,
                                         min_detection_confidence=0.5) as mesh:
        for path in lfw_sample(args.lfw, args.n, args.seed):
            img = cv2.imread(path)
            if img is None:
                continue
            res = mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                continue
            lm = res.multi_face_landmarks[0].landmark
            r, em, yp = nose_ratio(lm), eye_mouth_ratio(lm), yaw_proxy(lm)
            if r is not None and em is not None and yp is not None:
                out[os.path.relpath(path, args.lfw).replace(os.sep, "/")] = {
                    "nose_ratio": r, "eye_mouth_ratio": em, "yaw_proxy": yp}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"lfw_sample": args.n, "seed": args.seed, "mediapipe": mp.__version__, "faces": out}, f)
    print(f"{len(out)} faces with landmarks -> {args.out}")


def fit(x, y):
    """Least-squares slopes y~x and (inverted) x~y, correlation."""
    c = float(np.corrcoef(x, y)[0, 1])
    k_yx = float(np.polyfit(x, y, 1)[0])
    k_xy_inv = 1.0 / float(np.polyfit(y, x, 1)[0])
    return c, k_yx, k_xy_inv


def run_join(args):
    from app.services import face_engine
    from app.services import liveness_challenge as lc
    with open(args.ratios, encoding="utf-8") as f:
        data = json.load(f)
    pitch, measures, gate = [], {"nose_ratio": [], "eye_mouth_ratio": []}, []
    for rel, m in data["faces"].items():
        try:
            face = face_engine.analyze(cv2.imread(os.path.join(args.lfw, rel)), clahe=False, reject_multiple=False)
        except face_engine.FaceEngineError:
            continue
        pitch.append(face["pitch_deg"])
        for name in measures:
            measures[name].append(m[name])
        # the server's check on frontal, up and down frames (enroll_v2.check)
        gate.append((abs(m["yaw_proxy"]) <= 15, lc.direction(lc.ARCFACE_YAW_SIGN * face["yaw_deg"], "deg") == "frontal"))
    p = np.array(pitch)
    browser_frontal = [server for browser, server in gate if browser]
    result = {"faces_used": len(p), "lfw_sample": data["lfw_sample"], "seed": data["seed"],
              "mediapipe": data["mediapipe"], "server_pitch": "landmark_3d_68 pitch_deg, positive = looking up",
              "pitch_deg_percentiles": {q: round(float(np.percentile(p, q)), 1) for q in (5, 50, 95)},
              "frontal_yaw_gate": {"browser_frontal": len(browser_frontal),
                                   "of_those_server_frontal": sum(browser_frontal),
                                   "share": round(sum(browser_frontal) / max(len(browser_frontal), 1), 4)}}
    for name, values in measures.items():
        r = np.array(values)
        c, k_ratio_per_deg, k_ratio_per_deg_inv = fit(p, r)
        lo, hi = sorted((abs(k_ratio_per_deg), abs(k_ratio_per_deg_inv)))
        entry = {"corr_with_pitch": round(c, 4),
                 "ratio_per_deg_ols": round(k_ratio_per_deg, 5),
                 "ratio_per_deg_inverse_ols": round(k_ratio_per_deg_inv, 5),
                 "ratio_percentiles": {q: round(float(np.percentile(r, q)), 3) for q in (5, 50, 95)}}
        if name == "nose_ratio":
            # degrees per step: the larger slope gives the smaller angle, and vice versa
            entry["steps_in_deg"] = {s: {"ratio": round(v, 4), "deg_low": round(v / hi, 1), "deg_high": round(v / lo, 1)}
                                     for s, v in STEPS.items()}
        result[name] = entry
    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


def perturb(img):
    """Small changes a still head goes through between camera frames (a proxy, not real video)."""
    h, w = img.shape[:2]
    yield "shift_2px", cv2.warpAffine(img, np.float32([[1, 0, 2], [0, 1, 1]]), (w, h), borderMode=cv2.BORDER_REPLICATE)
    for s in (0.97, 1.03):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), 0, s)
        yield f"scale_{s}", cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    yield "jpeg_70", cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1], cv2.IMREAD_COLOR)
    yield "bright_1.1", cv2.convertScaleAbs(img, alpha=1.1, beta=0)


def run_jitter(args):
    """Spread of server pitch over perturbed copies of one image, per face."""
    from app.services import face_engine
    spread, worst = [], []
    for path in lfw_sample(args.lfw, args.faces, args.seed):
        img = cv2.imread(path)
        try:
            base = face_engine.analyze(img, clahe=False, reject_multiple=False)["pitch_deg"]
            values = [base] + [face_engine.analyze(v, clahe=False, reject_multiple=False)["pitch_deg"]
                               for _, v in perturb(img)]
        except face_engine.FaceEngineError:
            continue
        spread.append(float(np.std(values)))
        worst.append(float(max(abs(v - base) for v in values)))
    result = {"faces_used": len(spread), "copies_per_face": 6,
              "pitch_std_deg": {q: round(float(np.percentile(spread, q)), 2) for q in (50, 95)},
              "max_abs_change_deg": {q: round(float(np.percentile(worst, q)), 2) for q in (50, 95, 99)}}
    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lfw", default=os.path.join(ROOT, ".build", "datasets", "lfw", "lfw"))
    parser.add_argument("--n", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20261026)
    sub = parser.add_subparsers(dest="step", required=True)
    a = sub.add_parser("mediapipe")
    a.add_argument("--out", required=True)
    b = sub.add_parser("join")
    b.add_argument("--ratios", required=True)
    b.add_argument("--json")
    c = sub.add_parser("jitter")
    c.add_argument("--faces", type=int, default=300)
    c.add_argument("--json")
    args = parser.parse_args()
    {"mediapipe": run_mediapipe, "join": run_join, "jitter": run_jitter}[args.step](args)


if __name__ == "__main__":
    main()

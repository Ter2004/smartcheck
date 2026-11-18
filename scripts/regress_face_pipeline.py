"""Record what the face pipeline decides for a fixed image set, and compare two runs.

Used to check that a speed change does not change any decision:
    venv\\Scripts\\python.exe scripts\\regress_face_pipeline.py run  OUT.json [--set DIR]
    venv\\Scripts\\python.exe scripts\\regress_face_pipeline.py diff OLD.json NEW.json

The set is dataset/p01-p05 (5 people x 7 poses) and test_images (real, low detail,
phone/laptop/iPad screen photos), each at its own size and resized to 640 px like a
phone frame, plus composites made from the dataset: a small face in the background
and a second face as large as the main one. --set DIR writes the composites there
(default: a folder next to OUT.json).

Per image it records:
- check-in path: combined_spoof_score (is_real, scores) then analyze on the same
  frame (sharing the detection when the code supports it), the error if any, and the
  similarity to every person's 3 frontal enrollment images; pass = spoof pass and
  best own-person similarity >= ARCFACE_MATCH_THRESHOLD.
- enrollment path: analyze alone (embedding, det score, size, yaw/pitch/roll).
"""
import argparse
import inspect
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PEOPLE = [f"p0{i}" for i in range(1, 6)]


def resize_max(img, side=640):
    s = side / max(img.shape[:2])
    return cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def build_set(set_dir):
    """Return [(name, BGR image)]; composites are also written to set_dir for inspection."""
    set_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for p in PEOPLE:
        for f in sorted((ROOT / "dataset" / p).glob("*.jpg")):
            img = cv2.imread(str(f))
            items.append((f"{p}/{f.stem}", img))
            items.append((f"{p}/{f.stem}@640", resize_max(img)))
    for f in sorted((ROOT / "test_images").glob("*.jpg")):
        img = cv2.imread(str(f))
        items.append((f"test/{f.stem}", img))
        items.append((f"test/{f.stem}@640", resize_max(img)))
    for i, p in enumerate(PEOPLE):
        other = PEOPLE[(i + 1) % len(PEOPLE)]
        main = resize_max(cv2.imread(str(ROOT / "dataset" / p / "01_frontal_normal.jpg")))
        back = cv2.imread(str(ROOT / "dataset" / other / "01_frontal_normal.jpg"))
        h, w = main.shape[:2]
        for frac, tag in ((0.30, "bg_small"), (0.45, "bg_medium")):
            small = cv2.resize(back, (int(w * frac), int(back.shape[0] * w * frac / back.shape[1])))
            comp = main.copy()
            sh, sw = min(small.shape[0], h), small.shape[1]
            comp[:sh, w - sw:] = small[:sh]                      # top-right corner
            items.append((f"comp/{p}+{other}_{tag}", comp))
        pair = np.hstack([main, cv2.resize(resize_max(back), (main.shape[1], main.shape[0]))])
        items.append((f"comp/{p}+{other}_side_by_side", pair))
    for name, img in items:
        if name.startswith("comp/"):
            cv2.imwrite(str(set_dir / (name.split("/", 1)[1] + ".jpg")), img)
    return items


def err_name(e):
    return f"{type(e).__name__}: {e}"


def run(out_path, set_dir):
    from app import create_app
    app = create_app()
    with app.app_context():
        from app.services import face_engine as fe
        from app.services.face_service import combined_spoof_score
        threshold = float(app.config["ARCFACE_MATCH_THRESHOLD"])
        share = "detection" in inspect.signature(fe.analyze).parameters
        fe.warm_up()

        refs = {}
        for p in PEOPLE:
            refs[p] = [fe.analyze(cv2.imread(str(ROOT / "dataset" / p / f"0{k}_frontal_normal.jpg")))["embedding"]
                       for k in (1, 2, 3)]

        rows = {}
        for name, img in build_set(set_dir):
            row = {}
            # check-in path
            try:
                spoof = combined_spoof_score(img)
                layers = spoof.get("layers") or {}
                row["spoof"] = {"is_real": spoof.get("is_real"), "spoof_score": spoof.get("combined_score"),
                                "decision": spoof.get("disagreements"),
                                "fasnet": (layers.get("fasnet") or {}).get("spoof_score"),
                                "onnx": (layers.get("onnx") or {}).get("spoof_score")}
                kwargs = {"detection": spoof.get("detection")} if share else {}
                face = fe.analyze(img, **kwargs)
                sims = {p: max(float(np.dot(face["embedding"], r)) for r in refs[p]) for p in PEOPLE}
                row["checkin"] = {"sims": sims}
            except Exception as e:  # noqa: BLE001 — the error itself is the result
                row.setdefault("spoof", None)
                row["checkin_error"] = err_name(e)
            # enrollment path
            try:
                f = fe.analyze(img)
                row["analyze"] = {"embedding": [float(x) for x in f["embedding"]],
                                  **{k: f[k] for k in ("det_score", "face_px", "yaw_deg", "pitch_deg", "roll_deg",
                                                       "yaw_ratio", "eye_roll_deg", "faces", "second_large_face")},
                                  "bbox": [float(v) for v in f["bbox"]]}
            except Exception as e:  # noqa: BLE001
                row["analyze_error"] = err_name(e)
            owner = name.split("/")[1][:3] if name.startswith(("p0", "comp/")) else None
            if name.startswith("p0"):
                owner = name[:3]
            sims = (row.get("checkin") or {}).get("sims")
            spoof_ok = bool(row.get("spoof") and row["spoof"].get("is_real"))
            row["pass_spoof"] = spoof_ok
            row["pass_match_owner"] = bool(sims and owner and sims[owner] >= threshold)
            row["matches_at_threshold"] = sorted(p for p, s in (sims or {}).items() if s >= threshold)
            rows[name] = row
            print(f"{name:45s} spoof={'real' if spoof_ok else 'fail'} "
                  f"match={row['matches_at_threshold']} {row.get('checkin_error') or row.get('analyze_error') or ''}")
        Path(out_path).write_text(json.dumps({"threshold": threshold, "share_detection": share, "rows": rows},
                                             indent=1, default=str), encoding="utf-8")
        print(f"\n{len(rows)} images -> {out_path}")


def diff(old_path, new_path):
    old = json.loads(Path(old_path).read_text(encoding="utf-8"))["rows"]
    new = json.loads(Path(new_path).read_text(encoding="utf-8"))["rows"]
    assert old.keys() == new.keys(), "different image sets"
    decision_changes, score = [], {"spoof_score": 0.0, "sim": 0.0, "embedding_cos_min": 1.0,
                                   "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "det_score": 0.0, "bbox_px": 0.0}
    for name in old:
        o, n = old[name], new[name]
        for k in ("pass_spoof", "pass_match_owner", "matches_at_threshold", "checkin_error", "analyze_error"):
            if o.get(k) != n.get(k):
                decision_changes.append((name, k, o.get(k), n.get(k)))
        if o.get("spoof") and n.get("spoof"):
            for k in ("is_real", "decision"):
                if o["spoof"].get(k) != n["spoof"].get(k):
                    decision_changes.append((name, f"spoof.{k}", o["spoof"].get(k), n["spoof"].get(k)))
            for k in ("spoof_score", "fasnet", "onnx"):
                a, b = o["spoof"].get(k), n["spoof"].get(k)
                if (a is None) != (b is None):
                    decision_changes.append((name, f"spoof.{k}", a, b))
                elif a is not None:
                    score["spoof_score"] = max(score["spoof_score"], abs(a - b))
        if o.get("checkin") and n.get("checkin"):
            for p in o["checkin"]["sims"]:
                score["sim"] = max(score["sim"], abs(o["checkin"]["sims"][p] - n["checkin"]["sims"][p]))
        if o.get("analyze") and n.get("analyze"):
            a, b = np.array(o["analyze"]["embedding"]), np.array(n["analyze"]["embedding"])
            score["embedding_cos_min"] = min(score["embedding_cos_min"], float(np.dot(a, b)))
            for k, key in (("yaw", "yaw_deg"), ("pitch", "pitch_deg"), ("roll", "roll_deg"), ("det_score", "det_score")):
                score[k] = max(score[k], abs(o["analyze"][key] - n["analyze"][key]))
            score["bbox_px"] = max(score["bbox_px"], max(abs(x - y) for x, y in zip(o["analyze"]["bbox"], n["analyze"]["bbox"])))
            for k in ("faces", "second_large_face", "face_px"):
                if o["analyze"][k] != n["analyze"][k]:
                    decision_changes.append((name, f"analyze.{k}", o["analyze"][k], n["analyze"][k]))
    print(f"{len(old)} images compared")
    print(f"decision changes: {len(decision_changes)}")
    for c in decision_changes:
        print("  ", c)
    print("largest score differences:", json.dumps(score, indent=1))
    return 1 if decision_changes else 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("out"); r.add_argument("--set")
    d = sub.add_parser("diff"); d.add_argument("old"); d.add_argument("new")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.out, Path(args.set) if args.set else Path(args.out).with_suffix("") .parent / "regression_set")
    else:
        sys.exit(diff(args.old, args.new))


if __name__ == "__main__":
    main()

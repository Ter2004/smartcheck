"""Compare face configs on your own photos: the input for choosing the face settings.

    python scripts/compare_models.py --dataset dataset --out compare-out

Dataset layout: dataset/<person_id>/<NN>_<pose>_<lighting>.jpg, e.g. 01_frontal_normal.jpg.
The lighting tag is the text after the last underscore. Mirrors production
(app/services/enroll_v2.py keeps 3 frontal + left, right, up, down; check-in
takes the max cosine over all of them). Per person, sorted by file name:
  images 1-3  frontal enrollment shots  -> pairwise scores, for ENROLL_SELF_SIM_MIN
  images 4-7  left, right, up, down     -> each vs the best frontal, which must
                                           pass ARCFACE_MATCH_THRESHOLD at enrollment
  images 8-   check-in probes           -> max over the 7 templates of each person:
                                           genuine vs impostor, for ARCFACE_MATCH_THRESHOLD
                                           and ARCFACE_DUP_THRESHOLD
A file name whose pose word does not fit its place is reported before the run.

Configs: arcface with CLAHE off and on (FACE_CLAHE). The facenet512 baseline was
removed with DeepFace in Phase 5; its LFW proxy numbers stay in
docs/face-stack-v2/phase1-compare.md. Nothing is written to the database.

Outputs in --out:
  scores.csv          every probe-vs-person score (genuine and impostor)
  enroll_pairs.csv    pairwise scores among each person's 3 frontal enrollment images
  enroll_poses.csv    each posed enrollment image vs that person's best frontal image
  failures.csv        images where a config found no usable face
  summary.json        all tables below in machine-readable form
  hist_<config>.png   genuine vs impostor histogram per config
and prints, per config and per lighting tag: min genuine, max impostor, margin,
EER, FRR at FAR = 0 / 0.1% / 1% (with the threshold that gives it), mean
per-image latency, and the two enrollment score distributions.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ENROLL_FRONTAL = 3
ENROLL_POSES = ("left", "right", "up", "down")
ENROLL_N = ENROLL_FRONTAL + len(ENROLL_POSES)          # 7, as enroll_v2 keeps
EXPECTED_POSES = ("frontal",) * ENROLL_FRONTAL + ENROLL_POSES
CONFIGS = ("arcface_clahe_off", "arcface_clahe_on")
FAR_POINTS = (0.0, 0.001, 0.01)


def lighting_tag(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.rsplit("_", 1)[1].lower() if "_" in stem else "untagged"


def pose_word(path):
    """The word before the lighting tag when it names a pose, else None."""
    parts = os.path.splitext(os.path.basename(path))[0].lower().split("_")
    return parts[-2] if len(parts) >= 3 and parts[-2] in EXPECTED_POSES else None


def name_warnings(people):
    """Enrollment files whose pose word does not match their place (1-3 frontal, then left, right, up, down)."""
    out = []
    for person, files in people.items():
        for place, (path, expected) in enumerate(zip(files, EXPECTED_POSES), start=1):
            word = pose_word(path)
            if word is not None and word != expected:
                out.append(f"{person}/{os.path.basename(path)}: image {place} should be {expected}")
    return out


def load_dataset(folder):
    people = {}
    for person in sorted(os.listdir(folder)):
        files = sorted(p for p in glob.glob(os.path.join(folder, person, "*"))
                       if p.lower().endswith((".jpg", ".jpeg", ".png")))
        if len(files) > ENROLL_N:
            people[person] = files
    return people


def embedder(config):
    """config -> function(BGR image) -> unit-length float32 embedding (raises without a face)."""
    from app.services import face_engine
    clahe = config.endswith("_on")
    # Largest face. Frames with a second large face (rejected in production) are counted separately.
    def run(img):
        face = face_engine.analyze(img, clahe=clahe, reject_multiple=False)
        run.second_large += face["second_large_face"]
        return face["embedding"]
    run.second_large = 0
    return run


def embed_all(config, people):
    run = embedder(config)
    run(cv2.imread(next(iter(people.values()))[0]))          # load the model before timing
    embeddings, latencies, failures = {}, [], []
    for person, files in people.items():
        for path in files:
            img = cv2.imread(path)
            t = time.perf_counter()
            try:
                embeddings[path] = run(img)
            except Exception as error:
                failures.append({"config": config, "person": person, "file": os.path.basename(path),
                                 "error": type(error).__name__})
                continue
            finally:
                latencies.append((time.perf_counter() - t) * 1000)
    return embeddings, latencies, failures, getattr(run, "second_large", None)


def operating_points(genuine, impostor):
    g, i = np.sort(np.asarray(genuine)), np.sort(np.asarray(impostor))
    out = {"n_genuine": int(g.size), "n_impostor": int(i.size)}
    if not g.size or not i.size:
        return out
    out.update(min_genuine=round(float(g[0]), 4), max_impostor=round(float(i[-1]), 4),
               margin=round(float(g[0] - i[-1]), 4))
    for far in FAR_POINTS:
        # Smallest threshold whose impostor accept rate (score >= t) is <= far.
        allowed = int(np.floor(far * i.size))
        t = float(np.nextafter(i[-1], np.inf)) if allowed == 0 else float(i[i.size - allowed])
        if allowed and (i >= t).sum() > allowed:
            t = float(np.nextafter(t, np.inf))
        frr = float((g < t).mean())
        out[f"far_{far:g}"] = {"threshold": round(t, 4), "far": round(float((i >= t).mean()), 6), "frr": round(frr, 4)}
    thresholds = np.unique(np.concatenate([g, i]))
    far = np.array([(i >= t).mean() for t in thresholds])
    frr = np.array([(g < t).mean() for t in thresholds])
    k = int(np.argmin(np.abs(far - frr)))
    out["eer"] = {"eer": round(float((far[k] + frr[k]) / 2), 4), "threshold": round(float(thresholds[k]), 4)}
    return out


def histogram_png(path, title, genuine, impostor, bins=60):
    w, h, pad = 900, 420, 40
    canvas = np.full((h, w, 3), 255, np.uint8)
    edges = np.linspace(-0.2, 1.0, bins + 1)
    gh, _ = np.histogram(genuine, edges)
    ih, _ = np.histogram(impostor, edges)
    gh = gh / max(1, gh.sum()); ih = ih / max(1, ih.sum())
    top = max(gh.max(initial=0), ih.max(initial=0)) or 1
    bw = (w - 2 * pad) / bins
    for k in range(bins):
        x0 = int(pad + k * bw)
        for hist, color, off in ((ih, (60, 60, 220), 0), (gh, (60, 160, 60), bw / 2)):
            bh = int((h - 2 * pad) * hist[k] / top)
            cv2.rectangle(canvas, (int(x0 + off), h - pad - bh), (int(x0 + off + bw / 2) - 1, h - pad), color, -1)
    cv2.line(canvas, (pad, h - pad), (w - pad, h - pad), (0, 0, 0), 1)
    for v in (-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        x = int(pad + (v + 0.2) / 1.2 * (w - 2 * pad))
        cv2.putText(canvas, f"{v:.1f}", (x - 12, h - pad + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(canvas, title, (pad, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(canvas, "green = genuine, red = impostor (share of each set per bin)", (pad, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    cv2.imwrite(path, canvas)


def evaluate(config, people, embeddings):
    """(probe rows, frontal pairs, pose-vs-frontal rows), the way enroll_v2 and check-in score them."""
    templates, probes, pairs, poses = {}, [], [], []
    for person, files in people.items():
        enrolled = [embeddings.get(p) for p in files[:ENROLL_N]]
        if any(e is None for e in enrolled):
            continue                                   # an enrollment image without a face: person skipped
        templates[person] = np.stack(enrolled)
        frontal = templates[person][:ENROLL_FRONTAL]
        for a in range(ENROLL_FRONTAL):
            for b in range(a + 1, ENROLL_FRONTAL):
                pairs.append({"config": config, "person": person, "i": a, "j": b,
                              "score": round(float(frontal[a] @ frontal[b]), 4)})
        for k, pose in enumerate(ENROLL_POSES):
            path = files[ENROLL_FRONTAL + k]
            poses.append({"config": config, "person": person, "pose": pose, "file": os.path.basename(path),
                          "score": round(float((frontal @ embeddings[path]).max()), 4)})
        probes += [(person, p) for p in files[ENROLL_N:] if p in embeddings]
    rows = []
    for person, path in probes:
        if person not in templates:
            continue
        v = embeddings[path]
        for target, tpl in templates.items():
            rows.append({"config": config, "probe_person": person, "probe_file": os.path.basename(path),
                         "lighting": lighting_tag(path), "target_person": target,
                         "score": round(float((tpl @ v).max()), 4), "genuine": int(target == person)})
    return rows, pairs, poses


def spread(items):
    """min / p1 / p5 / median of item scores and the 5 lowest per-person minimums."""
    scores = np.asarray([x["score"] for x in items]) if items else np.zeros(0)
    if not scores.size:
        return {}
    per_person = {}
    for x in items:
        per_person[x["person"]] = min(per_person.get(x["person"], 1.0), x["score"])
    return {"n": int(scores.size), "min": round(float(scores.min()), 4),
            "p1": round(float(np.percentile(scores, 1)), 4), "p5": round(float(np.percentile(scores, 5)), 4),
            "median": round(float(np.median(scores)), 4),
            "lowest_person_minimums": sorted(per_person.items(), key=lambda kv: kv[1])[:5]}


def summarize(rows, pairs, poses, latencies, failures, n_images):
    out = {"overall": operating_points([r["score"] for r in rows if r["genuine"]],
                                       [r["score"] for r in rows if not r["genuine"]])}
    out["by_lighting"] = {}
    for tag in sorted({r["lighting"] for r in rows}):
        sub = [r for r in rows if r["lighting"] == tag]
        out["by_lighting"][tag] = operating_points([r["score"] for r in sub if r["genuine"]],
                                                   [r["score"] for r in sub if not r["genuine"]])
    out["enroll_frontal_pairwise"] = spread(pairs)
    out["enroll_pose_vs_frontal"] = spread(poses)
    out["enroll_pose_vs_frontal_min_by_pose"] = {
        pose: round(min(x["score"] for x in poses if x["pose"] == pose), 4)
        for pose in ENROLL_POSES if any(x["pose"] == pose for x in poses)}
    out["latency_ms_per_image"] = {"mean": round(float(np.mean(latencies)), 1),
                                   "median": round(float(np.median(latencies)), 1)} if latencies else {}
    out["no_face"] = {"count": len(failures), "of_images": n_images}
    return out


def print_table(config, summary):
    def line(name, s):
        if "min_genuine" not in s:
            return f"  {name:<14} n_gen={s.get('n_genuine', 0)} n_imp={s.get('n_impostor', 0)} (not enough scores)"
        pts = "  ".join(f"FAR={k[4:]}: t={v['threshold']} FRR={v['frr']:.2%}" for k, v in s.items() if k.startswith("far_"))
        return (f"  {name:<14} n_gen={s['n_genuine']:<5} n_imp={s['n_impostor']:<7} min_gen={s['min_genuine']:<7} "
                f"max_imp={s['max_impostor']:<7} margin={s['margin']:<8} EER={s['eer']['eer']:.2%} @ {s['eer']['threshold']}"
                f"\n  {'':<14} {pts}")
    print(f"\n== {config}  (latency/image {summary['latency_ms_per_image']}, no face {summary['no_face']})")
    print(line("overall", summary["overall"]))
    for tag, s in summary["by_lighting"].items():
        print(line(f"light={tag}", s))
    print(f"  enrollment, frontal pairwise (images 1-{ENROLL_FRONTAL}): {summary['enroll_frontal_pairwise']}")
    print(f"  enrollment, pose vs best frontal (images {ENROLL_FRONTAL + 1}-{ENROLL_N}): {summary['enroll_pose_vs_frontal']}")
    print(f"  enrollment, lowest pose vs frontal per pose: {summary['enroll_pose_vs_frontal_min_by_pose']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=os.path.join(ROOT, "dataset"))
    parser.add_argument("--out", default=os.path.join(ROOT, "compare-out"))
    parser.add_argument("--configs", default=",".join(CONFIGS))
    args = parser.parse_args()

    people = load_dataset(args.dataset)
    if len(people) < 2:
        raise SystemExit(f"need at least 2 people with more than {ENROLL_N} images in {args.dataset}")
    for warning in name_warnings(people):
        print("WARNING", warning)
    n_images = sum(len(f) for f in people.values())
    os.makedirs(args.out, exist_ok=True)
    print(f"dataset: {len(people)} people, {n_images} images (per person: images 1-{ENROLL_FRONTAL} frontal, "
          f"{ENROLL_FRONTAL + 1}-{ENROLL_N} {'/'.join(ENROLL_POSES)}, the rest check-in probes)")

    all_rows, all_pairs, all_poses, all_failures, summaries = [], [], [], [], {}
    for config in [c.strip() for c in args.configs.split(",") if c.strip()]:
        if config not in CONFIGS:
            raise SystemExit(f"unknown config {config!r}; choose from {CONFIGS}")
        embeddings, latencies, failures, second_large = embed_all(config, people)
        rows, pairs, poses = evaluate(config, people, embeddings)
        summaries[config] = summarize(rows, pairs, poses, latencies, failures, n_images)
        if second_large is not None:
            summaries[config]["second_large_face_frames"] = second_large
        print_table(config, summaries[config])
        histogram_png(os.path.join(args.out, f"hist_{config}.png"), f"{config}: max-of-{ENROLL_N} cosine",
                      [r["score"] for r in rows if r["genuine"]], [r["score"] for r in rows if not r["genuine"]])
        all_rows += rows; all_pairs += pairs; all_poses += poses; all_failures += failures

    for name, data, fields in (
            ("scores.csv", all_rows, ["config", "probe_person", "probe_file", "lighting", "target_person", "score", "genuine"]),
            ("enroll_pairs.csv", all_pairs, ["config", "person", "i", "j", "score"]),
            ("enroll_poses.csv", all_poses, ["config", "person", "pose", "file", "score"]),
            ("failures.csv", all_failures, ["config", "person", "file", "error"])):
        with open(os.path.join(args.out, name), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(data)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"dataset": os.path.abspath(args.dataset), "people": len(people), "images": n_images,
                   "configs": summaries}, f, indent=2)
    print(f"\nwrote {args.out}/scores.csv, enroll_pairs.csv, enroll_poses.csv, failures.csv, summary.json, hist_*.png")


if __name__ == "__main__":
    main()

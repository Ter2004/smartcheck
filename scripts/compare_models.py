"""Compare face stacks on your own photos: go/no-go input for the migration.

    python scripts/compare_models.py --dataset dataset --out compare-out

Dataset layout: dataset/<person_id>/<name>_<lighting>.jpg, e.g. 6501_01_bright.jpg.
The lighting tag is the text after the last underscore. Mirrors production:
per person, the first 5 images (sorted by file name) are the enrollment, the
rest are probes, and a probe's score against a person is the max cosine over
that person's 5 enrollment embeddings.

Configs: facenet512 (current pipeline, CLAHE as today), arcface with CLAHE off,
arcface with CLAHE on. Nothing is written to the database.

Outputs in --out:
  scores.csv          every probe-vs-person score (genuine and impostor)
  enroll_pairs.csv    pairwise scores within each person's first 5 images
  failures.csv        images where a config found no usable face
  summary.json        all tables below in machine-readable form
  hist_<config>.png   genuine vs impostor histogram per config
and prints, per config and per lighting tag: min genuine, max impostor, margin,
EER, FRR at FAR = 0 / 0.1% / 1% (with the threshold that gives it), mean
per-image latency, and the pairwise enrollment score distribution.
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

ENROLL_N = 5
CONFIGS = ("facenet512", "arcface_clahe_off", "arcface_clahe_on")
FAR_POINTS = (0.0, 0.001, 0.01)


def lighting_tag(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.rsplit("_", 1)[1].lower() if "_" in stem else "untagged"


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
    if config == "facenet512":
        from app.services import face_service as fs

        def run(img):
            emb, _ = fs._face_embedding(fs.normalize_illumination(img))
            v = np.asarray(emb, dtype=np.float32)
            return v / (np.linalg.norm(v) + 1e-12)
        return run
    from app.services import face_engine
    clahe = config.endswith("_on")
    # Largest face, like the facenet512 path, so both see the same face. Frames
    # with a second large face (rejected in production) are counted separately.
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
    templates, probes, pairs = {}, [], []
    for person, files in people.items():
        enrolled = [embeddings[p] for p in files[:ENROLL_N] if p in embeddings]
        if len(enrolled) == ENROLL_N:
            templates[person] = np.stack(enrolled)
            for a in range(ENROLL_N):
                for b in range(a + 1, ENROLL_N):
                    pairs.append({"config": config, "person": person, "i": a, "j": b,
                                  "score": round(float(enrolled[a] @ enrolled[b]), 4)})
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
    return rows, pairs


def summarize(rows, pairs, latencies, failures, n_images):
    out = {"overall": operating_points([r["score"] for r in rows if r["genuine"]],
                                       [r["score"] for r in rows if not r["genuine"]])}
    out["by_lighting"] = {}
    for tag in sorted({r["lighting"] for r in rows}):
        sub = [r for r in rows if r["lighting"] == tag]
        out["by_lighting"][tag] = operating_points([r["score"] for r in sub if r["genuine"]],
                                                   [r["score"] for r in sub if not r["genuine"]])
    scores = np.asarray([p["score"] for p in pairs]) if pairs else np.zeros(0)
    per_person_min = {}
    for p in pairs:
        per_person_min[p["person"]] = min(per_person_min.get(p["person"], 1.0), p["score"])
    out["enroll_pairwise"] = {} if not scores.size else {
        "n_pairs": int(scores.size), "min": round(float(scores.min()), 4),
        "p1": round(float(np.percentile(scores, 1)), 4), "p5": round(float(np.percentile(scores, 5)), 4),
        "median": round(float(np.median(scores)), 4),
        "lowest_person_minimums": sorted(per_person_min.items(), key=lambda kv: kv[1])[:5],
    }
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
    print(f"  enroll pairwise (first {ENROLL_N} per person): {summary['enroll_pairwise']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=os.path.join(ROOT, "dataset"))
    parser.add_argument("--out", default=os.path.join(ROOT, "compare-out"))
    parser.add_argument("--configs", default=",".join(CONFIGS))
    args = parser.parse_args()

    people = load_dataset(args.dataset)
    if len(people) < 2:
        raise SystemExit(f"need at least 2 people with more than {ENROLL_N} images in {args.dataset}")
    n_images = sum(len(f) for f in people.values())
    os.makedirs(args.out, exist_ok=True)
    print(f"dataset: {len(people)} people, {n_images} images (first {ENROLL_N} per person enrolled)")

    all_rows, all_pairs, all_failures, summaries = [], [], [], {}
    for config in [c.strip() for c in args.configs.split(",") if c.strip()]:
        if config not in CONFIGS:
            raise SystemExit(f"unknown config {config!r}; choose from {CONFIGS}")
        embeddings, latencies, failures, second_large = embed_all(config, people)
        rows, pairs = evaluate(config, people, embeddings)
        summaries[config] = summarize(rows, pairs, latencies, failures, n_images)
        if second_large is not None:
            summaries[config]["second_large_face_frames"] = second_large
        print_table(config, summaries[config])
        histogram_png(os.path.join(args.out, f"hist_{config}.png"), f"{config}: max-of-{ENROLL_N} cosine",
                      [r["score"] for r in rows if r["genuine"]], [r["score"] for r in rows if not r["genuine"]])
        all_rows += rows; all_pairs += pairs; all_failures += failures

    for name, data, fields in (
            ("scores.csv", all_rows, ["config", "probe_person", "probe_file", "lighting", "target_person", "score", "genuine"]),
            ("enroll_pairs.csv", all_pairs, ["config", "person", "i", "j", "score"]),
            ("failures.csv", all_failures, ["config", "person", "file", "error"])):
        with open(os.path.join(args.out, name), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(data)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"dataset": os.path.abspath(args.dataset), "people": len(people), "images": n_images,
                   "configs": summaries}, f, indent=2)
    print(f"\nwrote {args.out}/scores.csv, enroll_pairs.csv, failures.csv, summary.json, hist_*.png")


if __name__ == "__main__":
    main()

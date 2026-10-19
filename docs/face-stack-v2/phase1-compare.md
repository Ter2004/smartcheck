# Face stack v2: Phase 1 evaluation harness (2026-10-19)

`scripts/compare_models.py` is the go/no-go input for the migration. It writes
nothing to the database.

```
python scripts/compare_models.py --dataset dataset --out compare-out
```

## Inputs and method

**Dataset layout:** `dataset/<person_id>/<name>_<lighting>.jpg`. The lighting
tag is the text after the last `_`.

**Method (mirrors production):**
- Per person, the first 5 files by name are the enrollment and the rest are probes.
- A probe's score against a person is the max cosine over that person's 5 enrollment embeddings.

**Configs:**
- `facenet512`: the current pipeline, CLAHE, RetinaFace largest face;
- `arcface_clahe_off`;
- `arcface_clahe_on`.

**Faces:** every config uses the largest face, so all of them score the same
face. The number of frames with a second large face is reported separately;
production arcface rejects those frames.

## Outputs

**Files:**
- `scores.csv`, `enroll_pairs.csv`, `failures.csv`, `summary.json`;
- `hist_<config>.png`.

**Tables per config and per lighting tag:**
- min genuine, max impostor, margin;
- EER and its threshold;
- FRR at FAR = 0 / 0.1% / 1%, each with the threshold that achieves it;
- mean and median latency per image;
- pairwise scores within the first 5 images (min, p1, p5, median, and the
  5 lowest per-person minimums), for `ENROLL_SELF_SIM_MIN`.

**Privacy:** `dataset/` and `compare-out/` are git-ignored, because they hold
biometric data.

## Proxy run (NOT your data, not for choosing thresholds)

Your dataset is not available yet, so the harness was run on a proxy:
- **Data:** the first 40 LFW people with ≥ 10 images, 10 images each, tagged
  `_lfw`, so there is no lighting split.
- **Scores:** 200 genuine and 7,800 impostor.
- **Machine:** laptop CPU.

LFW is celebrity press photos with background people and some label noise,
which inflates both tails. So FAR = 0 and the min/max rows are dominated by
wrong-face picks.

### largest-face (`compare-lfw-proxy/largest-face/summary.json`)

| config | min gen | max imp | EER @ thr | FRR @ FAR 0.1% (thr) | FRR @ FAR 1% (thr) | latency/img median |
|---|---|---|---|---|---|---|
| facenet512 | 0.2296 | 0.7098 | 1.58% @ 0.5037 | 10.50% (0.6635) | 2.00% (0.5292) | 321 ms |
| arcface CLAHE off | -0.0603 | 0.6938 | 1.50% @ 0.1717 | 1.50% (0.3319) | 1.50% (0.1849) | 159 ms |
| arcface CLAHE on | 0.0304 | 0.6719 | 1.00% @ 0.1894 | 1.00% (0.2848) | 1.00% (0.1894) | 162 ms |

Pairwise enrollment scores (first 5 images):

| config | min | p1 | p5 | median |
|---|---|---|---|---|
| facenet512 | 0.1825 | 0.3262 | 0.471 | 0.6998 |
| arcface off | -0.2208 | -0.0887 | 0.365 | 0.6729 |
| arcface on | 0.0025 | 0.0318 | 0.434 | 0.6431 |

Negative pairwise minimums come from LFW frames where the largest face is
another person, which production would refuse as a second large face.

### Production face rule (`compare-lfw-proxy/production-face-rule/summary.json`)

The same run, with frames that have a second large face rejected:
- **Rejected frames:** 42 of 400 with CLAHE off, 39 with CLAHE on, all
  `MultipleFacesError`.
- **People dropped:** those that lost an enrollment frame.
- **Result:** min genuine 0.4049 / 0.347 and max impostor 0.3319 / 0.2848,
  giving margins of +0.073 / +0.062.

The two rows cover different subsets, so they cannot be compared with each other.

## Decision needed from you

Run the script on your own photos and choose:
- go/no-go;
- `FACE_CLAHE`;
- `ARCFACE_MATCH_THRESHOLD`, `ARCFACE_DUP_THRESHOLD` and `ENROLL_SELF_SIM_MIN`.

Your own photos means real students, several sessions, the lighting tags you
care about, and more than 5 images each.

The app refuses to start with `FACE_ENGINE=arcface` until all four are set.

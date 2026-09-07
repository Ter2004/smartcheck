"""
Standalone instrumentation for the anti-spoof FFT layers (app/services/face_service.py).

Does NOT modify anything in app/ — loads face_service.py directly from its file
path via importlib so we skip the Flask/Supabase app package init entirely
(face_service.py itself only imports base64/logging/os/threading/time/numpy/cv2/json
at module scope, so this is safe to import in isolation).

The low_r sweep (--sweep) reimplements detect_screen_moire's ratio formula with
a variable low-frequency fraction, because the real function hardcodes low_r=0.10
(face_service.py:683) and can't be parameterized without editing app/. Everything
else calls the real functions directly.

Usage:
    venv/Scripts/python.exe scripts/debug_moire.py test_images/*.jpg
    venv/Scripts/python.exe scripts/debug_moire.py test_images/*.jpg --sweep
    venv/Scripts/python.exe scripts/debug_moire.py test_images/*.jpg --sweep --fracs 0.10,0.15,0.20,0.25,0.30
    venv/Scripts/python.exe scripts/debug_moire.py path/to/one.jpg --hypothesis
    venv/Scripts/python.exe scripts/debug_moire.py --burst test_images/burst_face1/*.jpg
"""
import sys
import os
import glob
import argparse
import importlib.util

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_SERVICE_PATH = os.path.join(REPO_ROOT, "app", "services", "face_service.py")


def _load_face_service():
    spec = importlib.util.spec_from_file_location("face_service_standalone", FACE_SERVICE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fs = _load_face_service()


def has_exif(path: str) -> bool:
    """Proxy for 'unmodified camera original' — a JPEG re-saved by PIL/most
    editors drops EXIF unless explicitly preserved, and swaps the APP0 marker
    from Exif to bare JFIF."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(32)
        if b"Exif" not in head:
            return False
        im = Image.open(path)
        exif = im.getexif()
        return len(exif) > 0
    except Exception:
        return False


# ─── Core per-image layer scoring (the 4 anti-spoof layers + ONNX audit) ────

def score_image(path: str, temporal_variance=None) -> dict:
    """
    Scores the 4 layers combined_spoof_score() weighs (moire, texture, fasnet,
    temporal) plus the onnx audit layer, each called standalone so a failure
    in one layer (e.g. missing torch) can't hide the others — unlike
    combined_spoof_score() itself, which fails CLOSED the instant Fasnet is
    unavailable (face_service.py:277-291) and would make every row here a
    trivial reject regardless of image content.
    """
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return {"path": path, "error": "cv2.imread failed to decode"}

    h, w = img_bgr.shape[:2]

    moire = fs.detect_screen_moire([img_bgr], threshold=fs.MOIRE_THRESHOLD_SINGLE)

    try:
        is_screen_tex = fs.detect_screen_texture(img_bgr, min_peaks=30)
    except Exception as e:
        is_screen_tex = f"ERROR: {e}"

    try:
        fasnet_is_real, fasnet_spoof_score = fs._run_fasnet_antispoof(img_bgr)
    except Exception as e:
        fasnet_is_real, fasnet_spoof_score = None, f"ERROR: {e}"

    try:
        onnx_is_real, onnx_real_score = fs._run_antispoof(img_bgr)
    except Exception as e:
        onnx_is_real, onnx_real_score = None, f"ERROR: {e}"

    return {
        "path": path,
        "width": w,
        "height": h,
        "has_exif": has_exif(path),
        "moire_score": moire["avg_score"],
        "moire_threshold": moire["threshold"],
        "moire_pass": not moire["is_screen"],
        "texture_is_screen": is_screen_tex,
        "fasnet_is_real": fasnet_is_real,
        "fasnet_spoof_score": fasnet_spoof_score,
        "onnx_is_real": onnx_is_real,
        "onnx_real_score": onnx_real_score,
        "temporal_variance": temporal_variance if temporal_variance is not None
            else "N/A (single frame — use --burst to score 2+ frames of one subject)",
    }


def print_table(results: list):
    header = (f'{"file":34} {"WxH":>11} {"exif":>5} {"moire":>7} {"m.PASS":>6} '
              f'{"tex_scrn":>9} {"fasnet_real":>11} {"fasnet_spf":>10} {"onnx_real":>9} {"temporal_var":>14}')
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r and len(r) == 2:
            print(f'{os.path.basename(r["path"]):34} ERROR: {r["error"]}')
            continue
        fname = os.path.basename(r["path"])
        wh = f'{r["width"]}x{r["height"]}'
        exif = "yes" if r["has_exif"] else "no"
        moire = f'{r["moire_score"]:.4f}'
        passf = "PASS" if r["moire_pass"] else "FAIL"
        tex = str(r["texture_is_screen"])
        freal = str(r["fasnet_is_real"])
        fspoof = f'{r["fasnet_spoof_score"]:.4f}' if isinstance(r["fasnet_spoof_score"], float) else str(r["fasnet_spoof_score"])
        oreal = str(r["onnx_is_real"])
        tvar = f'{r["temporal_variance"]:.3f}' if isinstance(r["temporal_variance"], float) else "N/A"
        print(f"{fname:34} {wh:>11} {exif:>5} {moire:>7} {passf:>6} {tex:>9} {freal:>11} {fspoof:>10} {oreal:>9} {tvar:>14}")


# ─── Burst mode: temporal-variance layer needs 2+ frames of one subject ────

def score_burst(paths: list):
    """
    detect_static_image() (temporal layer) is meaningless on a single static
    test image — duplicate-frame variance is trivially ~0 and always reads
    "static". This scores it properly against a real multi-frame burst
    (e.g. frames pulled from a phone video, or a rapid-fire photo sequence),
    matching how /api/checkin actually feeds it (face_images_list[-3:],
    api_checkin.py:223).
    """
    frames = [cv2.imread(p, cv2.IMREAD_COLOR) for p in paths]
    frames = [f for f in frames if f is not None]
    if len(frames) < 2:
        print(f"Need >=2 decodable frames for burst mode, got {len(frames)}")
        return
    temporal = fs.detect_static_image(frames)
    print(f"\n=== Burst temporal-variance ({len(frames)} frames) ===")
    print(f"temporal_variance = {temporal['temporal_variance']}  "
          f"threshold = {fs.TEMPORAL_VAR_THRESHOLD}  "
          f"is_static = {temporal['is_static']}")
    # Also report the 4-layer table on the first frame, with the real
    # temporal_variance substituted in instead of the single-frame N/A.
    r = score_image(paths[0], temporal_variance=temporal["temporal_variance"])
    print_table([r])


# ─── Task 4 (kept): double-JPEG-encode hypothesis, already disproven ──────

def run_hypothesis_test(src_path: str):
    print(f"\n=== Hypothesis test on {src_path} ===")
    im = Image.open(src_path).convert("RGB")
    w, h = im.size

    target_h = 900
    scale = target_h / h
    im_a = im.resize((int(w * scale), target_h), Image.LANCZOS)
    path_a = os.path.join(os.path.dirname(src_path), "_hyp_a_lanczos_q95.jpg")
    im_a.save(path_a, quality=95)

    w8 = (w // 8) * 8
    h8 = (h // 8) * 8
    im_b = im.crop((0, 0, w8, h8))
    path_b = os.path.join(os.path.dirname(src_path), "_hyp_b_crop8_subsamp0_q95.jpg")
    im_b.save(path_b, quality=95, subsampling=0)

    results = [score_image(path_a), score_image(path_b)]
    print_table(results)

    a_score = results[0]["moire_score"]
    b_score = results[1]["moire_score"]
    print(f"\n(a) LANCZOS resize + q95              moire_score = {a_score:.4f}")
    print(f"(b) crop-to-8 (no resample) + q95/ss0  moire_score = {b_score:.4f}")
    if a_score - b_score > 0.05:
        print(">>> (b) markedly lower than (a): supports the double-encode / resample hypothesis.")
    else:
        print(">>> (b) NOT markedly lower than (a): hypothesis not supported by this pair.")


# ─── Tasks (c) and (d): low_r sweep, full-frame and face-cropped ──────────

def _moire_ratio_at_frac(img_bgr: np.ndarray, low_r_frac: float) -> float:
    """
    Mirrors detect_screen_moire()'s ratio formula exactly (face_service.py:657-703)
    but with low_r_frac as a parameter instead of the hardcoded 0.10 (line 683).
    Same resize-to-256, same fft2/fftshift/magnitude, same ratio = high/total.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256))

    f_transform = np.fft.fft2(gray.astype(np.float32))
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    low_r = int(min(h, w) * low_r_frac)

    low_mask = np.zeros_like(magnitude, dtype=bool)
    low_mask[cy - low_r:cy + low_r, cx - low_r:cx + low_r] = True

    total_energy = float(np.sum(magnitude))
    low_energy = float(np.sum(magnitude[low_mask]))
    high_energy = total_energy - low_energy

    return high_energy / (total_energy + 1e-8)


def crop_face_like_temporal(img_bgr: np.ndarray):
    """
    Mirrors detect_static_image()'s Haar-cascade face crop exactly
    (face_service.py:838-850): same cascade instance (fs._face_cascade),
    same scaleFactor/minNeighbors/minSize, same 20% padding.
    Returns (cropped_bgr, True) or (img_bgr, False) if no face was detected
    (fallback to full frame, same as the real function's documented behavior).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    detected = fs._face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
    )
    if len(detected) == 0:
        return img_bgr, False
    x, y, w, h = detected[0]
    pad = int(min(w, h) * 0.20)
    h_img, w_img = img_bgr.shape[:2]
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(w_img, x + w + pad); y2 = min(h_img, y + h + pad)
    return img_bgr[y1:y2, x1:x2], True


def run_lowr_sweep(paths: list, fracs: list, threshold: float, cropped: bool):
    label = "FACE-CROPPED (Haar, matches temporal layer's crop)" if cropped else "FULL FRAME (current production behavior)"
    print(f"\n=== low_r sweep — {label} — threshold held at {threshold} ===")
    col_w = 15
    header = f'{"file":34} {"face?":>6}' + "".join(f'{("low_r="+str(f)):>{col_w}}' for f in fracs)
    print(header)
    print("-" * len(header))

    col_totals = {f: [] for f in fracs}
    for p in paths:
        img_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"{os.path.basename(p):34}  ERROR: could not decode")
            continue
        face_found = "n/a"
        target = img_bgr
        if cropped:
            target, found = crop_face_like_temporal(img_bgr)
            face_found = "yes" if found else "NO(fallback)"

        row = f"{os.path.basename(p):34} {face_found:>6}"
        for frac in fracs:
            ratio = _moire_ratio_at_frac(target, frac)
            col_totals[frac].append(ratio)
            mark = "FAIL" if ratio > threshold else "pass"
            row += f'{f"{ratio:.4f}({mark})":>{col_w}}'
        print(row)

    print("-" * len(header))
    avg_row = f'{"AVERAGE":34} {"":>6}'
    fail_row = f'{"FAIL COUNT":34} {"":>6}'
    for frac in fracs:
        vals = col_totals[frac]
        avg = sum(vals) / len(vals) if vals else float("nan")
        fails = sum(1 for v in vals if v > threshold)
        avg_row += f'{avg:>{col_w}.4f}'
        fail_row += f'{f"{fails}/{len(vals)}":>{col_w}}'
    print(avg_row)
    print(fail_row)


# ─── Real-vs-spoof separability table ─────────────────────────────────────
# Naming convention: files prefixed `real_` and `spoof_` (any of .jpg/.jpeg/.png)
# in one directory, e.g. real_face1_plain.jpg, spoof_phoneX_fill_normal.jpg.

def _texture_num_peaks(img_bgr: np.ndarray, peak_threshold_multiplier: float = 3.0) -> int:
    """
    Mirrors detect_screen_texture() exactly (face_service.py:706-746, post-F-15-fix)
    but returns the raw peak COUNT instead of the bool the real function exposes.
    detect_screen_texture only returns `num_peaks > min_peaks` — there is no
    standalone way to get the continuous score without reimplementing the
    same handful of lines the real function already computes internally.

    Kept in sync with the F-15 fix: threshold is computed from the unmasked
    high-frequency ring only (ring_mask), not the full array including the
    zeroed-out centre block — matching face_service.py after the fix.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256))
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    h, w = magnitude.shape
    center_y, center_x = h // 2, w // 2
    mask_radius = min(h, w) // 4

    high_freq = magnitude.copy()
    high_freq[
        center_y - mask_radius: center_y + mask_radius,
        center_x - mask_radius: center_x + mask_radius,
    ] = 0

    ring_mask = np.ones_like(high_freq, dtype=bool)
    ring_mask[
        center_y - mask_radius: center_y + mask_radius,
        center_x - mask_radius: center_x + mask_radius,
    ] = False
    ring_values = high_freq[ring_mask]

    threshold = np.mean(ring_values) + peak_threshold_multiplier * np.std(ring_values)
    return int(np.sum(high_freq > threshold))


def _compare_layer_scores(path: str) -> dict:
    """
    Raw spoof-likelihood score per layer for one image, all oriented so that
    HIGHER = more spoof-like (needed for a uniform threshold sweep):
      moire   -> avg_score              (native direction already)
      texture -> num_peaks              (native direction already)
      fasnet  -> spoof_score, or None if DeepFace's own detector found no face
      onnx    -> spoof_score = 1 - raw_real_score, or None on inference error
    Temporal is intentionally excluded — it needs 2+ frames of one subject,
    which the flat real_/spoof_ single-file convention doesn't provide (see
    --burst for that layer instead).
    """
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None

    moire = fs.detect_screen_moire([img_bgr], threshold=fs.MOIRE_THRESHOLD_SINGLE)
    num_peaks = _texture_num_peaks(img_bgr)

    try:
        fasnet_is_real, fasnet_spoof = fs._run_fasnet_antispoof(img_bgr)
    except Exception:
        fasnet_is_real, fasnet_spoof = None, None

    try:
        onnx_is_real, onnx_raw_real = fs._run_antispoof(img_bgr)
        onnx_spoof = 1.0 - onnx_raw_real
    except Exception:
        onnx_is_real, onnx_spoof = None, None

    return {
        "path": path,
        "moire": moire["avg_score"],
        "texture": float(num_peaks),
        "fasnet": fasnet_spoof,       # None if DeepFace found no face
        "fasnet_is_real": fasnet_is_real,
        "onnx": onnx_spoof,           # None on inference error
        "onnx_is_real": onnx_is_real,
    }


def _sweep_high_is_spoof(real_scores: list, spoof_scores: list):
    """
    Exhaustive 1-D threshold sweep for ONE polarity: score > t => spoof.
    Tries the midpoint between every pair of adjacent sorted values plus the
    two unbounded ends, so it's guaranteed to find the global minimum-error
    threshold for this direction of decision rule (there's no gain from a
    finer grid — the error count only changes at those midpoints).

    Returns (min_errors, best_threshold, false_rejects, false_accepts).
    false_reject = real sample scored > threshold (a genuine user rejected)
    false_accept = spoof sample scored <= threshold (a spoof let through)
    """
    all_scores = sorted(set(real_scores + spoof_scores))
    if not all_scores:
        return None
    candidates = [all_scores[0] - 1.0]
    candidates += [(all_scores[i] + all_scores[i + 1]) / 2 for i in range(len(all_scores) - 1)]
    candidates += [all_scores[-1] + 1.0]

    best = None
    for t in candidates:
        fr = sum(1 for s in real_scores if s > t)
        fa = sum(1 for s in spoof_scores if s <= t)
        err = fr + fa
        if best is None or err < best[0]:
            best = (err, t, fr, fa)
    return best


def find_best_threshold(real_scores: list, spoof_scores: list):
    """
    Sweeps BOTH polarities and returns whichever wins, so a signal that's
    present but inverted (spoof scores lower than real, not higher — see
    FRR-1 §6.1: Moiré's own current threshold direction can't separate the
    12-real/3-spoof sample at all, but flipping the decision direction gets
    15/16 right) surfaces automatically instead of needing a by-hand check.

    'low=spoof' is computed by negating both score lists and re-running the
    same high-is-spoof sweep, then negating the winning threshold back —
    exact mirror image, no separate implementation to keep in sync.

    Returns a dict: {errors, threshold, false_rejects, false_accepts,
    direction, native_errors, flipped_errors} — the last two so callers can
    show both directions' error counts even when only one is "best".
    """
    hi = _sweep_high_is_spoof(real_scores, spoof_scores)
    lo_raw = _sweep_high_is_spoof([-s for s in real_scores], [-s for s in spoof_scores])
    lo = (lo_raw[0], -lo_raw[1], lo_raw[2], lo_raw[3])

    native_errors, flipped_errors = hi[0], lo[0]
    winner = hi if hi[0] <= lo[0] else lo
    direction = "high=spoof (native)" if hi[0] <= lo[0] else "low=spoof (INVERTED)"

    return {
        "errors": winner[0],
        "threshold": winner[1],
        "false_rejects": winner[2],
        "false_accepts": winner[3],
        "direction": direction,
        "native_errors": native_errors,
        "flipped_errors": flipped_errors,
    }


def _current_threshold_fr_fa(layer: str, real_scores: list, spoof_scores: list,
                              real_extra: list, spoof_extra: list):
    """
    False-reject / false-accept counts using each layer's CURRENT production
    decision rule — not an arbitrary cutoff on the raw score:
      moire   -> score > MOIRE_THRESHOLD_SINGLE (the actual /api/checkin gate)
      texture -> num_peaks > 30 (min_peaks value every real caller passes)
      fasnet  -> DeepFace's own is_real (argmax across 3 classes + confidence-
                 margin override inside _run_fasnet_antispoof/DeepFace — not
                 a single scalar cutoff, so there is no "threshold" number to
                 report here; only FR/FA counts)
      onnx    -> _run_antispoof's own is_real (same caveat, argmax + 0.10
                 confidence-margin override, face_service.py:94-135)
    """
    if layer == "moire":
        thr = fs.MOIRE_THRESHOLD_SINGLE
        fr = sum(1 for s in real_scores if s > thr)
        fa = sum(1 for s in spoof_scores if s <= thr)
        return f"{thr:.2f}", fr, fa
    if layer == "texture":
        thr = 30
        fr = sum(1 for s in real_scores if s > thr)
        fa = sum(1 for s in spoof_scores if s <= thr)
        return f"{thr}", fr, fa
    if layer in ("fasnet", "onnx"):
        fr = sum(1 for is_real in real_extra if is_real is False)
        fa = sum(1 for is_real in spoof_extra if is_real is True)
        return "argmax+margin (no scalar cutoff)", fr, fa
    raise ValueError(layer)


def _print_per_file(label: str, rows: list):
    layers = ["moire", "texture", "fasnet", "onnx"]
    header = f'{"file":38} ' + "".join(f'{l:>10}' for l in layers)
    print(f"\n--- {label} ({len(rows)} files) ---")
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda r: r["moire"]):
        fname = os.path.basename(r["path"])
        cells = []
        for l in layers:
            v = r[l]
            cells.append(f'{v:>10.4f}' if isinstance(v, float) else f'{"n/a":>10}')
        print(f"{fname:38} " + "".join(cells))


def run_compare(directory: str):
    exts = (".jpg", ".jpeg", ".png")
    all_files = [f for f in glob.glob(os.path.join(directory, "*")) if f.lower().endswith(exts)]
    real_paths = sorted(f for f in all_files if os.path.basename(f).lower().startswith("real_"))
    spoof_paths = sorted(f for f in all_files if os.path.basename(f).lower().startswith("spoof_"))

    if not real_paths or not spoof_paths:
        print(f"Need at least 1 'real_*' and 1 'spoof_*' file in {directory}")
        print(f"Found: {len(real_paths)} real_*, {len(spoof_paths)} spoof_*")
        return
    if len(real_paths) < 3 or len(spoof_paths) < 3:
        print(f"WARNING: only {len(real_paths)} real / {len(spoof_paths)} spoof samples — "
              f"means and best-threshold below will be statistically weak with this few.\n")

    real_rows = [r for r in (_compare_layer_scores(p) for p in real_paths) if r]
    spoof_rows = [r for r in (_compare_layer_scores(p) for p in spoof_paths) if r]

    print(f"real_*  : {len(real_rows)}/{len(real_paths)} decoded")
    print(f"spoof_* : {len(spoof_rows)}/{len(spoof_paths)} decoded")

    _print_per_file("real_* (sorted by moire score, ascending)", real_rows)
    _print_per_file("spoof_* (sorted by moire score, ascending)", spoof_rows)
    print()

    layers = ["moire", "texture", "fasnet", "onnx"]
    col_w = 12
    header = (f'{"layer":9} {"n_real":>7} {"n_spoof":>8} {"mean_real":>{col_w}} {"mean_spoof":>{col_w}} '
              f'{"gap":>{col_w}} {"cur_rule":>20} {"FR@cur":>7} {"FA@cur":>7} '
              f'{"best_thr":>{col_w}} {"direction":>20} {"FR@best":>8} {"FA@best":>8} '
              f'{"native_err":>10} {"flip_err":>9} {"verdict":>30}')
    print(header)
    print("-" * len(header))

    for layer in layers:
        real_scores = [r[layer] for r in real_rows if r[layer] is not None]
        spoof_scores = [r[layer] for r in spoof_rows if r[layer] is not None]
        n_real_skipped = len(real_rows) - len(real_scores)
        n_spoof_skipped = len(spoof_rows) - len(spoof_scores)

        if not real_scores or not spoof_scores:
            print(f"{layer:9} — no usable scores (layer failed on every real_* or every spoof_* sample)")
            continue

        mean_real = sum(real_scores) / len(real_scores)
        mean_spoof = sum(spoof_scores) / len(spoof_scores)
        gap = mean_spoof - mean_real

        real_extra = [r["fasnet_is_real"] if layer == "fasnet" else r["onnx_is_real"] for r in real_rows]
        spoof_extra = [r["fasnet_is_real"] if layer == "fasnet" else r["onnx_is_real"] for r in spoof_rows]
        cur_thr_label, fr_cur, fa_cur = _current_threshold_fr_fa(layer, real_scores, spoof_scores, real_extra, spoof_extra)

        best = find_best_threshold(real_scores, spoof_scores)
        min_errors, best_thr = best["errors"], best["threshold"]
        fr_best, fa_best = best["false_rejects"], best["false_accepts"]
        direction = best["direction"]

        always_real_errors = len(spoof_scores)   # call everyone real -> every spoof is a false accept
        always_spoof_errors = len(real_scores)    # call everyone spoof -> every real is a false reject
        trivial_best = min(always_real_errors, always_spoof_errors)

        inverted_flag = " [INVERTED beats native!]" if "INVERTED" in direction and best["flipped_errors"] < best["native_errors"] else ""
        if min_errors == 0:
            verdict = f"FULL SEPARATION{inverted_flag}"
        elif min_errors < trivial_best:
            verdict = f"PARTIAL ({min_errors} err, beats guessing){inverted_flag}"
        else:
            verdict = "NO SEPARATION — remove from vote"

        n_row = f"{len(real_scores)}" + (f"(-{n_real_skipped})" if n_real_skipped else "")
        s_row = f"{len(spoof_scores)}" + (f"(-{n_spoof_skipped})" if n_spoof_skipped else "")

        print(f"{layer:9} {n_row:>7} {s_row:>8} {mean_real:>{col_w}.4f} {mean_spoof:>{col_w}.4f} "
              f"{gap:>+{col_w}.4f} {cur_thr_label:>20} {fr_cur:>7} {fa_cur:>7} "
              f"{best_thr:>{col_w}.4f} {direction:>20} {fr_best:>8} {fa_best:>8} "
              f"{best['native_errors']:>10} {best['flipped_errors']:>9} {verdict:>30}")

    print("\nFR = false reject (real sample scored as spoof)  |  FA = false accept (spoof sample scored as real)")
    print("'best_thr'/'direction' come from sweeping BOTH polarities (score>t=>spoof, and score<t=>spoof) and")
    print("taking whichever wins — 'direction'=low=spoof (INVERTED) means the signal is real but backwards versus")
    print("what the code currently assumes. 'native_err'/'flip_err' show both directions' error counts side by")
    print("side even when only one wins, so a close call is visible, not just the winner.")
    print("This is NOT a recommendation to change app/ — it only tells you whether a threshold (in either")
    print("direction) separates these two sets at all for that layer.")
    print("fasnet/onnx FR@cur/FA@cur use each layer's own current is_real decision (argmax+margin), not a cutoff")
    print("on the raw score — their best_thr/direction/FR@best/FA@best columns still show what a scalar cutoff,")
    print("in either direction, COULD do.")


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--hypothesis", action="store_true",
                         help="Task 4: double-JPEG-encode hypothesis test on the first path")
    parser.add_argument("--sweep", action="store_true",
                         help="Tasks (c)+(d): sweep low_r on full frame AND face-cropped frame")
    parser.add_argument("--fracs", type=str, default="0.10,0.15,0.20,0.25,0.30",
                         help="Comma-separated low_r fractions to sweep")
    parser.add_argument("--threshold", type=float, default=0.70,
                         help="Threshold held constant during the sweep")
    parser.add_argument("--burst", action="store_true",
                         help="Treat all given paths as frames of ONE subject for the temporal layer")
    parser.add_argument("--compare", type=str, default=None, metavar="DIR",
                         help="Directory with real_*/spoof_* files — real-vs-spoof separability table per layer")
    args = parser.parse_args()

    if args.compare:
        run_compare(args.compare)
        return

    paths = []
    for a in args.paths:
        paths.extend(sorted(glob.glob(a)) if any(ch in a for ch in "*?[") else [a])
    if not paths and not args.burst:
        paths = sorted(glob.glob(os.path.join(REPO_ROOT, "test_images", "*.jpg")))

    if args.burst:
        score_burst(paths)
        return

    print(f"MOIRE_THRESHOLD_SINGLE = {fs.MOIRE_THRESHOLD_SINGLE}  (face_service.py)")
    print(f"MOIRE_THRESHOLD (multi-frame) = {fs.MOIRE_THRESHOLD}\n")

    results = [score_image(p) for p in paths]
    print_table(results)

    if args.hypothesis and paths:
        run_hypothesis_test(paths[0])

    if args.sweep:
        fracs = [float(x) for x in args.fracs.split(",")]
        run_lowr_sweep(paths, fracs, args.threshold, cropped=False)
        run_lowr_sweep(paths, fracs, args.threshold, cropped=True)


if __name__ == "__main__":
    main()

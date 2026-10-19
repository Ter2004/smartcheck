"""Time the server-side face work of one enrollment and one check-in, in process.

No HTTP, no database: the same service calls the routes make, on frames built
from test_images/ (resized to 640x480 JPEG like the browser sends). Network,
Supabase and the browser are not included, so these are CPU-side baselines for
comparing face stacks, not end-to-end latencies.

    python scripts/bench_face_pipeline.py --runs 10
    python scripts/bench_face_pipeline.py --runs 10 --json out.json
"""
import argparse
import base64
import ctypes
import glob
import json
import os
import statistics
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def rss_mb():
    """Resident set size of this process in MB (Windows working set or Linux VmRSS)."""
    if sys.platform == "win32":
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return float("nan")
        return pmc.WorkingSetSize / 2**20
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def browser_frame(path):
    """A 640x480 JPEG data URL, the size the enrollment and check-in pages capture."""
    img = cv2.imread(path)
    h, w = img.shape[:2]
    scale = max(640 / w, 480 / h)
    img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    y0, x0 = (img.shape[0] - 480) // 2, (img.shape[1] - 640) // 2
    img = img[y0:y0 + 480, x0:x0 + 640]
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def summary(samples):
    ordered = sorted(samples)
    p90 = ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))]
    return {"median_ms": round(statistics.median(ordered), 1), "p90_ms": round(p90, 1),
            "min_ms": round(ordered[0], 1), "max_ms": round(ordered[-1], 1), "n": len(ordered)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=10)
    # Frames both stacks find a face in (the *_ok close-ups are too tight for SCRFD).
    parser.add_argument("--images", default=",".join(os.path.join(ROOT, "test_images", g) for g in (
        "real_0[13579]_sharp_face?.jpg", "real_0[37]_sharp_face?_r.jpg", "real_11_sharp_face3_r.jpg")),
        help="comma-separated globs")
    parser.add_argument("--json", help="write the result to this file")
    args = parser.parse_args()

    from scripts import face_pipelines as pipelines
    from scripts.face_pipelines import PIPELINE_NAME, warm_up, enrollment, checkin
    extra = getattr(pipelines, "checkin_head_turn", None)

    paths = sorted({p for pattern in args.images.split(",") for p in glob.glob(pattern)})
    if len(paths) < 2:
        raise SystemExit(f"need at least 2 images matching {args.images}")
    frames = [browser_frame(p) for p in paths]
    rss_start = rss_mb()
    t0 = time.perf_counter()
    warm_up(frames)
    warm_s = time.perf_counter() - t0
    rss_warm = rss_mb()

    enroll_ms, checkin_ms, turn_ms = [], [], []
    for i in range(args.runs):
        rotated = frames[i % len(frames):] + frames[:i % len(frames)]
        t = time.perf_counter(); enrollment(rotated); enroll_ms.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter(); checkin(rotated); checkin_ms.append((time.perf_counter() - t) * 1000)
        if extra:
            t = time.perf_counter(); extra(rotated); turn_ms.append((time.perf_counter() - t) * 1000)

    result = {
        "pipeline": PIPELINE_NAME, "images": [os.path.basename(p) for p in paths],
        "python": sys.version.split()[0], "platform": sys.platform,
        "rss_mb": {"start": round(rss_start, 1), "after_warm_up": round(rss_warm, 1), "after_runs": round(rss_mb(), 1)},
        "warm_up_s": round(warm_s, 2),
        "enrollment": summary(enroll_ms), "checkin": summary(checkin_ms),
    }
    if turn_ms:
        result["checkin_head_turn"] = summary(turn_ms)
    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as out:
            json.dump(result, out, indent=2)


if __name__ == "__main__":
    main()

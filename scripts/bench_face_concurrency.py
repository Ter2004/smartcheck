"""Time the check-in face pipeline for 1 request and for N requests at once.

Runs the same server functions /api/checkin calls on its face_image, in this
process, without login, session or BLE proof:
    decode -> server_validate_frame -> combined_spoof_score (SCRFD + Fasnet + ONNX)
    -> face_engine.analyze (3D-68 + ArcFace; SCRFD again only on code that
       cannot share Fasnet's detection, i.e. before 2026-11-18)
The DB part (Supabase reads/writes) is not run here; take it from the [PERF]
lines in checkin-diagnostics.log.

Usage (project venv, from the repo root):
    venv\\Scripts\\python.exe scripts\\bench_face_concurrency.py [image] [--n 10] [--rounds 3]
"""
import argparse
import inspect
import base64
import os
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHARE = False
sys.path.insert(0, str(ROOT))


def load_b64(path):
    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def one_request(b64, fs, face_engine):
    """Per-stage wall time in ms for one face_image, as the check-in route runs it."""
    t = {}
    t0 = time.perf_counter()
    check = fs.server_validate_frame(b64)
    t["validate_frame"] = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    img = fs._decode_image(b64)
    t["decode"] = (time.perf_counter() - t1) * 1000
    t2 = time.perf_counter()
    shared = {}
    spoof = (fs.combined_spoof_score(img, detection_out=shared) if SHARE
             else fs.combined_spoof_score(img))
    t["spoof_total"] = (time.perf_counter() - t2) * 1000
    t["fasnet(incl. SCRFD)"] = spoof.get("timings", {}).get("fasnet_ms", 0.0)
    t["onnx_antispoof"] = spoof.get("timings", {}).get("onnx_ms", 0.0)
    t3 = time.perf_counter()
    face_engine.analyze(img, detection=shared.get("detection")) if SHARE else face_engine.analyze(img)
    t["arcface(SCRFD+pose+embed)"] = (time.perf_counter() - t3) * 1000
    t["total"] = (time.perf_counter() - t0) * 1000
    t["_valid"] = check.get("valid")
    return t


def run_concurrent(n, b64, fs, face_engine):
    barrier = threading.Barrier(n)
    results = [None] * n

    def worker(i):
        barrier.wait()
        results[i] = one_request(b64, fs, face_engine)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    t0 = time.perf_counter()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return (time.perf_counter() - t0) * 1000, results


def summarize(label, results, wall_ms=None):
    keys = [k for k in results[0] if not k.startswith("_")]
    print(f"\n== {label} ==")
    if wall_ms is not None:
        print(f"  wall time, all done : {wall_ms:8.0f} ms")
    for k in keys:
        vals = [r[k] for r in results]
        print(f"  {k:28s} mean {statistics.mean(vals):7.0f} ms   max {max(vals):7.0f} ms")


def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except ImportError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default=str(ROOT / "test_images" / "real_01_sharp_face1.jpg"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    from app import create_app
    app = create_app()
    with app.app_context():
        from app.services import face_service as fs, face_engine
        import onnxruntime as ort
        global SHARE
        SHARE = "detection_out" in inspect.signature(fs.combined_spoof_score).parameters
        print(f"shared detection: {SHARE}")
        print(f"onnxruntime {ort.__version__} providers={ort.get_available_providers()} "
              f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} cpu_count={os.cpu_count()}")
        b64 = load_b64(args.image)
        print(f"image {args.image}  {len(b64) * 3 // 4 // 1024} KB")

        t = time.perf_counter()
        face_engine.warm_up()
        one_request(b64, fs, face_engine)       # loads Fasnet + ONNX antispoof too
        print(f"warm-up {time.perf_counter() - t:.1f}s  RSS after load: {rss_mb() or 'n/a'} MB")

        first = one_request(b64, fs, face_engine)
        if not first["_valid"]:
            print("WARNING: server_validate_frame rejected this image; times may not match a real check-in")

        seq = [one_request(b64, fs, face_engine) for _ in range(args.rounds * 3)]
        summarize(f"1 request at a time ({len(seq)} runs)", seq)

        walls, all_conc = [], []
        for _ in range(args.rounds):
            wall, res = run_concurrent(args.n, b64, fs, face_engine)
            walls.append(wall)
            all_conc.extend(res)
        summarize(f"{args.n} requests at once ({args.rounds} rounds)", all_conc, statistics.mean(walls))
        slowest = [max(r["total"] for r in all_conc[i * args.n:(i + 1) * args.n]) for i in range(args.rounds)]
        print(f"  slowest request per round: {[round(s) for s in slowest]} ms")
        print(f"  RSS now: {rss_mb() or 'n/a'} MB")


if __name__ == "__main__":
    main()

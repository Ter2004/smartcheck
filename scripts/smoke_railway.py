"""Smoke test a deployed SmartCheck: N requests at once to pages that need no login.

    venv\\Scripts\\python.exe scripts\\smoke_railway.py https://YOUR-APP.up.railway.app [--n 20] [--paths /health /login]

Reports per path: status codes, errors, fastest/median/slowest time and the wall
time for the whole burst. Exit code 1 if any request failed or was not 2xx/3xx.
Only GET requests; nothing is written.
"""
import argparse
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request


def fetch(url, timeout):
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "smartcheck-smoke/1"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
            return r.status, None, (time.perf_counter() - t0) * 1000
    except urllib.error.HTTPError as e:
        return e.code, None, (time.perf_counter() - t0) * 1000
    except Exception as e:  # noqa: BLE001 — report any network error
        return None, f"{type(e).__name__}: {e}", (time.perf_counter() - t0) * 1000


def burst(url, n, timeout):
    barrier = threading.Barrier(n)
    out = [None] * n

    def work(i):
        barrier.wait()
        out[i] = fetch(url, timeout)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out, (time.perf_counter() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--paths", nargs="+", default=["/health", "/login"])
    ap.add_argument("--timeout", type=float, default=30)
    args = ap.parse_args()
    base = args.base.rstrip("/")
    bad = 0
    for path in args.paths:
        url = base + path
        status, err, ms = fetch(url, args.timeout)          # warm-up, not counted
        print(f"\n{url}  (warm-up: {status or err}, {ms:.0f} ms)")
        results, wall = burst(url, args.n, args.timeout)
        times = sorted(r[2] for r in results)
        codes = {}
        for s, e, _ in results:
            key = s if e is None else e
            codes[key] = codes.get(key, 0) + 1
        fails = sum(1 for s, e, _ in results if e is not None or not (200 <= s < 400))
        bad += fails
        print(f"  {args.n} at once: wall {wall:.0f} ms | min {times[0]:.0f}  median {statistics.median(times):.0f}"
              f"  max {times[-1]:.0f} ms")
        print(f"  results: {codes}" + (f"  <-- {fails} failed" if fails else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

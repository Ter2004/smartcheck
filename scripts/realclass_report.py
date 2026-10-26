"""Real-class test report: enrollment and check-in times vs a stopwatch roll call.

    python scripts/realclass_report.py --attempts attempts.csv --rollcall-seconds 145 \
        --room "LAB 2" --class-size 28 --log server.log [--json out.json]

attempts.csv (one row per attempt; template in docs/realclass_test.md):
  student   pseudonym (S01, S02, ...), never the real name or student id
  phase     enroll | checkin
  attempt   1, 2, 3 ... per student and phase
  start     clock time the student started (HH:MM:SS), e.g. opened the page / pressed start
  end       clock time the result appeared (HH:MM:SS)
  success   1 / 0
  reason    optional: message shown on failure

--log: server log(s); check-in face scores are read from lines like
  [FACE] engine=arcface best=0.6123 threshold=0.4 pass=True
They are genuine scores only if the log holds just this test's real check-ins.
"""
import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime

SCORE = re.compile(r"\[FACE\] engine=arcface best=(-?[\d.]+) threshold=(-?[\d.]+) pass=(True|False)")


def seconds(start, end):
    fmt = "%H:%M:%S"
    delta = (datetime.strptime(end.strip(), fmt) - datetime.strptime(start.strip(), fmt)).total_seconds()
    return delta + 86400 if delta < 0 else delta


def dist(values):
    if not values:
        return {}
    v = sorted(values)
    p90 = v[min(len(v) - 1, int(round(0.9 * (len(v) - 1))))]
    return {"n": len(v), "median_s": round(statistics.median(v), 1), "p90_s": round(p90, 1),
            "min_s": round(v[0], 1), "max_s": round(v[-1], 1)}


def phase_report(rows):
    by_student = defaultdict(list)
    for r in rows:
        by_student[r["student"]].append(r)
    first_ok, per_student_total, attempt_times, clock = 0, [], [], []
    for student, items in by_student.items():
        items.sort(key=lambda r: int(r["attempt"]))
        first_ok += items[0]["success"] == "1"
        done = [r for r in items if r["success"] == "1"]
        attempt_times += [seconds(r["start"], r["end"]) for r in items]
        if done:
            per_student_total.append(seconds(items[0]["start"], done[0]["end"]))
        clock += [r["start"] for r in items] + [r["end"] for r in items]
    n = len(by_student)
    out = {"students": n,
           "first_attempt_success": round(first_ok / n, 4) if n else None,
           "first_attempt_reject_rate": round(1 - first_ok / n, 4) if n else None,
           "never_succeeded": n - len(per_student_total),
           "per_attempt_time": dist(attempt_times),
           "per_student_time_to_success": dist(per_student_total),
           "failure_reasons": dict(sorted(defaultdict(int, {
               k: sum(1 for r in rows if r["success"] != "1" and (r.get("reason") or "(none)") == k)
               for k in {r.get("reason") or "(none)" for r in rows if r["success"] != "1"}}).items()))}
    if clock:
        ordered = sorted(clock, key=lambda t: datetime.strptime(t.strip(), "%H:%M:%S"))
        out["class_elapsed_s"] = round(seconds(ordered[0], ordered[-1]), 1)
    return out


def read_text_lines(path):
    """Lines of a log saved as UTF-8 or UTF-16 (Windows PowerShell 5.1 Tee-Object and > write UTF-16)."""
    with open(path, "rb") as f:
        raw = f.read()
    encoding = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
    return raw.decode(encoding, errors="replace").splitlines()


def scores(paths, lowest=5):
    found = []
    for path in paths or []:
        for line in read_text_lines(path):
            m = SCORE.search(line)
            if m:
                found.append((float(m.group(1)), float(m.group(2)), m.group(3) == "True"))
    if not found:
        return {}
    values = sorted(s for s, _, _ in found)
    threshold = found[0][1]
    return {"n": len(found), "threshold": threshold, "min": values[0], "median": round(statistics.median(values), 4),
            "below_threshold": sum(1 for s, t, _ in found if s < t),
            "lowest_margins": [round(s - threshold, 4) for s in values[:lowest]],
            "histogram_0.05": {f"{b / 20:.2f}": sum(1 for s in values if b / 20 <= s < (b + 1) / 20)
                               for b in range(int(values[0] * 20), int(values[-1] * 20) + 1)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--attempts", required=True)
    parser.add_argument("--rollcall-seconds", type=float, help="stopwatch time to call the whole class")
    parser.add_argument("--room", default="")
    parser.add_argument("--class-size", type=int)
    parser.add_argument("--log", nargs="*")
    parser.add_argument("--json")
    args = parser.parse_args()
    with open(args.attempts, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("student") and r.get("phase")]
    out = {"room": args.room, "class_size": args.class_size,
           "enroll": phase_report([r for r in rows if r["phase"].strip() == "enroll"]),
           "checkin": phase_report([r for r in rows if r["phase"].strip() == "checkin"]),
           "rollcall_seconds": args.rollcall_seconds,
           "genuine_scores": scores(args.log)}
    if args.rollcall_seconds and out["checkin"].get("class_elapsed_s"):
        out["checkin_vs_rollcall"] = {
            "smartcheck_class_elapsed_s": out["checkin"]["class_elapsed_s"],
            "rollcall_s": args.rollcall_seconds,
            "ratio": round(out["checkin"]["class_elapsed_s"] / args.rollcall_seconds, 3)}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

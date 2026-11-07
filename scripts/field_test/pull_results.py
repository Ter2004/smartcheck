"""Field test results: pull every round of a test day from the database and the
check-in log, and write a summary for the report and the poster.

Read-only: it only reads (sessions, attendance, course_enrollments, the log file).

Run from the project folder, with the project's venv:

    venv\\Scripts\\python.exe scripts\\field_test\\pull_results.py --course FT-01 --date 2026-11-12 ^
        --roll-call 31.4 --roll-call 29.8 --out docs\\field-test\\results\\2026-11-12

  --course / --date  every class of that course that day is one round, in time order
  --session ID       or give the rounds yourself (repeat; label with --label R1 ...)
  --roll-call SEC    a stopwatch time for calling every name once (repeat per trial)
  --roll-call-n N    how many names were called (default: students enrolled in the course)

Output folder:
  checkins.csv   one row per recorded check-in (participants as P01, P02 ... no names)
  rejects.csv    one row per refused attempt seen in the log during a round
  summary.json   the numbers make_charts.py draws
  summary.md     the same numbers as tables (Thai), ready to paste
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
TH = timezone(timedelta(hours=7))
LOG_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ ")
REJECT = re.compile(r"endpoint=api_checkin\.(\w+) step=(\w+) result=reject")
PERF = re.compile(r"endpoint=api_checkin\.checkin \[PERF\] total=([\d.]+)ms")


def parse_time(value):
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def pct(values, q):
    """Nearest-rank percentile; enough for 10-30 values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, round(q / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def read_log(paths):
    """[(local datetime, line)] from checkin-diagnostics.log and its rotated copies."""
    rows = []
    for path in paths:
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    match = LOG_LINE.match(line)
                    if match:
                        when = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=TH)
                        rows.append((when, line))
        except OSError:
            pass
    rows.sort(key=lambda r: r[0])
    return rows


def find_rounds(db, args):
    if args.session:
        rows = []
        for sid in args.session:
            res = db.table("sessions").select("*, courses(code, name)").eq("id", sid).maybe_single().execute()
            if not res or not res.data:
                sys.exit(f"session not found: {sid}")
            rows.append(res.data)
        return rows
    course = db.table("courses").select("id, code, name").eq("code", args.course).execute().data
    if not course:
        sys.exit(f"course not found: {args.course}")
    day = date.fromisoformat(args.date)
    start = datetime(day.year, day.month, day.day, tzinfo=TH)
    rows = (db.table("sessions").select("*, courses(code, name)")
            .eq("course_id", course[0]["id"])
            .gte("start_time", start.isoformat())
            .lt("start_time", (start + timedelta(days=1)).isoformat())
            .order("start_time").execute().data or [])
    if not rows:
        sys.exit(f"no classes of {args.course} on {args.date}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--course")
    ap.add_argument("--date")
    ap.add_argument("--session", action="append")
    ap.add_argument("--label", action="append")
    ap.add_argument("--roll-call", action="append", type=float, default=[])
    ap.add_argument("--roll-call-n", type=int)
    ap.add_argument("--log", default=str(ROOT / "checkin-diagnostics.log"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not args.session and not (args.course and args.date):
        ap.error("give --course and --date, or --session")

    import app as smartcheck   # loads .env through app.config; prints nothing secret
    smartcheck._refresh_clients()
    db = smartcheck.supabase_admin

    rounds = find_rounds(db, args)
    labels = args.label or [f"R{i + 1}" for i in range(len(rounds))]
    log = read_log(sorted(glob.glob(args.log + "*")))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    course_id = rounds[0]["course_id"]
    enrolled = db.table("course_enrollments").select("student_id").eq("course_id", course_id).execute().data or []
    people = {row["student_id"]: f"P{i + 1:02d}" for i, row in enumerate(sorted(enrolled, key=lambda r: r["student_id"]))}

    summary = {"course": rounds[0]["courses"]["code"], "date": args.date, "enrolled": len(enrolled),
               "roll_call": {"seconds": args.roll_call, "names": args.roll_call_n or len(enrolled)},
               "rounds": []}
    checkin_rows, reject_rows = [], []

    for label, sess in zip(labels, rounds):
        opened = parse_time(sess.get("opened_at"))
        attendance = (db.table("attendance").select("student_id, status, check_in_at")
                      .eq("session_id", sess["id"]).order("check_in_at").execute().data or [])
        seconds = []
        for a in attendance:
            when = parse_time(a.get("check_in_at"))
            since = round((when - opened).total_seconds(), 1) if (when and opened) else None
            if since is not None:
                seconds.append(since)
            checkin_rows.append({"round": label, "participant": people.get(a["student_id"], "outside-course"),
                                 "status": a.get("status"), "seconds_since_open": since,
                                 "check_in_time": when.astimezone(TH).strftime("%H:%M:%S") if when else ""})

        # Refusals and server times written to the log while this round was open.
        if opened:
            closes = parse_time(sess.get("end_time")) or (opened + timedelta(minutes=30))
            last = max((parse_time(a["check_in_at"]) for a in attendance if a.get("check_in_at")), default=opened)
            window_end = min(closes, last + timedelta(minutes=10)) if attendance else closes
        rejects, server_ms = {}, []
        for when, line in log if opened else []:
            if opened.astimezone(TH) <= when <= window_end.astimezone(TH) + timedelta(seconds=1):
                rej = REJECT.search(line)
                if rej:
                    reason = f"{rej.group(1)}:{rej.group(2)}"
                    rejects[reason] = rejects.get(reason, 0) + 1
                    reject_rows.append({"round": label, "time": when.strftime("%H:%M:%S"), "reason": reason})
                perf = PERF.search(line)
                if perf:
                    server_ms.append(float(perf.group(1)))

        face_rejects = sum(n for r, n in rejects.items() if r.startswith("checkin:"))
        attempts = len(attendance) + face_rejects
        summary["rounds"].append({
            "label": label, "session_id": sess["id"], "title": sess.get("title"),
            "opened_at": opened.astimezone(TH).isoformat() if opened else None,
            "on_time_minutes": sess.get("present_minutes"),
            "checked_in": len(attendance), "enrolled": len(enrolled),
            "late": sum(1 for a in attendance if a.get("status") == "late"),
            "seconds": sorted(seconds),
            "first_s": min(seconds) if seconds else None,
            "median_s": statistics.median(seconds) if seconds else None,
            "p90_s": pct(seconds, 90),
            "last_s": max(seconds) if seconds else None,
            "rejects": rejects,
            "face_step_attempts": attempts,
            "face_step_success_rate": round(len(attendance) / attempts, 3) if attempts else None,
            "server_ms": server_ms,
            "server_median_ms": statistics.median(server_ms) if server_ms else None,
        })

    rc = summary["roll_call"]
    if rc["seconds"]:
        rc["mean_s"] = round(statistics.mean(rc["seconds"]), 1)
        rc["per_name_s"] = round(rc["mean_s"] / rc["names"], 2) if rc["names"] else None

    with open(out / "checkins.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["round", "participant", "status", "seconds_since_open", "check_in_time"])
        w.writeheader()
        w.writerows(checkin_rows)
    with open(out / "rejects.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["round", "time", "reason"])
        w.writeheader()
        w.writerows(reject_rows)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(render_markdown(summary))
    print(f"\nwritten: {out}")


def fmt(value, unit=""):
    return "—" if value is None else f"{value:g}{unit}"


def render_markdown(s):
    lines = [f"# ผลทดสอบภาคสนาม {s['course']} {s.get('date') or ''}".rstrip(), "",
             f"นักศึกษาในวิชา: {s['enrolled']} คน · เวลานับจากอาจารย์กด \"เปิดคาบ\" (sessions.opened_at) "
             "ถึงเวลาที่ระบบบันทึก (attendance.check_in_at) [วัด]", "",
             "| รอบ | เช็คได้ | สาย | คนแรก | มัธยฐาน | P90 | คนสุดท้าย | ถูกปฏิเสธ | เซิร์ฟเวอร์ (มัธยฐาน) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in s["rounds"]:
        rej = ", ".join(f"{k} ×{v}" for k, v in sorted(r["rejects"].items())) or "0"
        lines.append(f"| {r['label']} | {r['checked_in']}/{r['enrolled']} | {r['late']} | {fmt(r['first_s'], ' s')} "
                     f"| {fmt(r['median_s'], ' s')} | {fmt(r['p90_s'], ' s')} | {fmt(r['last_s'], ' s')} | {rej} "
                     f"| {fmt(round(r['server_median_ms']) if r['server_median_ms'] else None, ' ms')} |")
    rc = s["roll_call"]
    if rc.get("seconds"):
        lines += ["", f"เรียกชื่อ {rc['names']} คน: " + ", ".join(f"{x:g} s" for x in rc["seconds"])
                  + f" (เฉลี่ย {rc['mean_s']:g} s, {rc['per_name_s']:g} s ต่อชื่อ) [วัด ด้วยนาฬิกาจับเวลา]"]
    lines += ["", "หมายเหตุ: \"ถูกปฏิเสธ\" นับจาก checkin-diagnostics.log ในช่วงเวลาของรอบนั้น "
              "(checkin:* = ตอนส่งภาพหน้า, checkin_proximity:* = ตอนยืนยันห้อง)"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

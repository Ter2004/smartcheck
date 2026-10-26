"""Browser vs server head pose, from v2 enrollment logs.

Every posed frame of a v2 enrollment is logged as
    step=pose_angle result=measured details=pose=left client_yaw=0.412 client_pitch=None
         server_yaw_deg=27.3 server_pitch_delta=1.2 seen=turn_left
(client_yaw: MediaPipe eye-line ratio in the browser, + = turn_left;
client_pitch: change of the v1 nose ratio from the frontal baseline, + = up;
server_yaw_deg / server_pitch_delta: InsightFace landmark_3d_68, the pitch
relative to the same person's frontal frames, + = up). Older logs have
step=turn_yaw lines with pose=turn; they count as left/right frames.

The report checks, separately for turns (left/right) and nods (up/down), that
both sides agree on the direction and how many server degrees one browser unit
reads, so the browser's TURN_YAW (0.30) and PITCH_STEP (0.07) can be compared
with the server's limits (turn 20.4 deg; up/down: direction only, yaw frontal).

    python scripts/report_turn_yaw.py app.log [more.log ...]   (or pipe the log on stdin)
Needs at least 20 frames of a kind for a verdict on it.
"""
import json
import re
import sys

import numpy as np

NUM = r"(-?[\d.]+|None)"
POSE_LINE = re.compile(rf"step=pose_angle .*?pose=(\w+) client_yaw={NUM} client_pitch={NUM} "
                       rf"server_yaw_deg=(-?[\d.]+) server_pitch_delta=(-?[\d.]+) seen=(\w+)")
OLD_LINE = re.compile(rf"step=turn_yaw .*?pose=(\w+) client_yaw={NUM} server_yaw_deg=(-?[\d.]+) seen=(\w+)")
MIN_FRAMES = 20
TURN_YAW, PITCH_STEP = 0.30, 0.07      # enrollment_v2.js EV2


def read_text_lines(path):
    """Lines of a log saved as UTF-8 or UTF-16 (Windows PowerShell 5.1 Tee-Object and > write UTF-16)."""
    with open(path, "rb") as f:
        raw = f.read()
    encoding = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
    return raw.decode(encoding, errors="replace").splitlines()


def _num(text):
    return None if text == "None" else float(text)


def parse(lines):
    rows = []
    for line in lines:
        m = POSE_LINE.search(line)
        if m:
            rows.append({"pose": m.group(1), "client_yaw": _num(m.group(2)), "client_pitch": _num(m.group(3)),
                         "server_yaw": float(m.group(4)), "server_pitch": float(m.group(5)), "seen": m.group(6)})
            continue
        m = OLD_LINE.search(line)
        if m and m.group(1) == "turn":
            rows.append({"pose": "turn", "client_yaw": _num(m.group(2)), "client_pitch": None,
                         "server_yaw": float(m.group(3)), "server_pitch": None, "seen": m.group(4)})
    return rows


def _compare(c, s):
    """Sign agreement, server units per browser unit (least squares through 0), correlation."""
    c, s = np.array(c), np.array(s)
    varies = len(c) > 2 and c.std() > 0 and s.std() > 0
    return {"sign_agreement": round(float(np.mean(np.sign(c) == np.sign(s))), 4),
            "server_per_client_unit": round(float(np.dot(c, s) / np.dot(c, c)), 1),
            "corr": round(float(np.corrcoef(c, s)[0, 1]), 4) if varies else None}


def report(rows):
    turns = [r for r in rows if r["pose"] in ("left", "right", "turn") and r["client_yaw"] is not None]
    nods = [r for r in rows if r["pose"] in ("up", "down") and r["client_pitch"] is not None]
    out = {"frames": len(rows), "turn_frames": len(turns), "up_down_frames": len(nods)}
    if len(turns) < MIN_FRAMES:
        out["turns"] = f"need >= {MIN_FRAMES} turned frames"
    else:
        t = _compare([r["client_yaw"] for r in turns], [r["server_yaw"] for r in turns])
        t["server_below_turn_limit"] = int(sum(r["seen"] not in ("turn_left", "turn_right") for r in turns))
        t[f"client_at_{TURN_YAW}_reads_deg"] = round(TURN_YAW * t["server_per_client_unit"], 1)
        out["turns"] = t
    if len(nods) < MIN_FRAMES:
        out["up_down"] = f"need >= {MIN_FRAMES} up/down frames"
    else:
        n = _compare([r["client_pitch"] for r in nods], [r["server_pitch"] for r in nods])
        n["server_yaw_not_frontal"] = int(sum(r["seen"] != "frontal" for r in nods))
        n[f"client_at_{PITCH_STEP}_reads_deg"] = round(PITCH_STEP * n["server_per_client_unit"], 1)
        n["lfw_fit_for_comparison"] = "0.07 = 8.7-13.3 deg (docs/face-stack-v2/client-pitch-calibration.json)"
        out["up_down"] = n
    return out


def main():
    lines = []
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            lines += read_text_lines(path)
    else:
        lines = sys.stdin.readlines()
    print(json.dumps(report(parse(lines)), indent=2))


if __name__ == "__main__":
    main()

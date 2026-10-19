"""Browser vs server head-turn yaw, from v2 enrollment logs.

Every frame of a v2 enrollment is logged as
    step=turn_yaw result=measured details=pose=turn client_yaw=0.412 server_yaw_deg=27.3 seen=turn_left
(client_yaw: MediaPipe eye-line ratio in the browser; server_yaw_deg: InsightFace
landmark_3d_68). This reports, over turned frames, whether both sides agree on
the direction and how degrees scale with the browser ratio, so the browser's
TURN_YAW (0.30) can be checked against the server's 20.4-degree limit.

    python scripts/report_turn_yaw.py app.log [more.log ...]   (or pipe the log on stdin)
Needs at least 20 turned frames for a verdict.
"""
import json
import re
import sys

import numpy as np

LINE = re.compile(r"step=turn_yaw .*?pose=(\w+) client_yaw=(-?[\d.]+|None) server_yaw_deg=(-?[\d.]+) seen=(\w+)")
MIN_TURNS = 20


def parse(lines):
    rows = []
    for line in lines:
        m = LINE.search(line)
        if m and m.group(2) != "None":
            rows.append({"pose": m.group(1), "client": float(m.group(2)),
                         "server": float(m.group(3)), "seen": m.group(4)})
    return rows


def report(rows):
    turns = [r for r in rows if r["pose"] == "turn"]
    out = {"frames": len(rows), "turn_frames": len(turns)}
    if len(turns) < MIN_TURNS:
        out["verdict"] = f"need >= {MIN_TURNS} turned frames"
        return out
    c = np.array([r["client"] for r in turns]); s = np.array([r["server"] for r in turns])
    out["sign_agreement"] = round(float(np.mean(np.sign(c) == np.sign(s))), 4)
    out["deg_per_client_unit"] = round(float(np.dot(c, s) / np.dot(c, c)), 1)
    out["corr"] = round(float(np.corrcoef(c, s)[0, 1]), 4) if len(turns) > 2 else None
    out["server_below_turn_limit"] = int(sum(r["seen"] not in ("turn_left", "turn_right") for r in turns))
    out["client_at_0.30_reads_deg"] = round(0.30 * out["deg_per_client_unit"], 1)
    return out


def main():
    lines = []
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            with open(path, encoding="utf-8", errors="replace") as f:
                lines += f.readlines()
    else:
        lines = sys.stdin.readlines()
    print(json.dumps(report(parse(lines)), indent=2))


if __name__ == "__main__":
    main()

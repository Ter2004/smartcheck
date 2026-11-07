"""Charts for the report and the poster from pull_results.py's output folder.

Needs matplotlib (the system Python 3.11 on this PC has it; the project venv does not):

    python scripts\\field_test\\make_charts.py docs\\field-test\\results\\2026-11-12

Writes, next to summary.json, each as .png (report, 300 dpi) and .svg (poster, sharp at any size):
  class_time   whole class: SmartCheck rounds vs calling the roll (seconds)
  per_student  every check-in, seconds after the teacher opened the class
  projection   bigger classes: roll call grows per name; SmartCheck flat (estimate beyond the test)

Colours follow the data-viz palette (slot 1 blue = SmartCheck, slot 2 orange = roll call),
validated with the dataviz skill's validate_palette.js; every bar and line end carries
its value, so colour is never the only cue. Labels are English (report and poster language).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SMARTCHECK = "#2a78d6"
ROLL_CALL = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 13,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def save(fig, folder, name):
    for ext, dpi in (("png", 300), ("svg", None)):
        fig.savefig(folder / f"{name}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def class_time(s, folder):
    rows = []
    rc = s.get("roll_call") or {}
    if rc.get("seconds"):
        trials = len(rc["seconds"])
        rows.append((f"Roll call ({rc['names']} names" + (f", mean of {trials}" if trials > 1 else "") + ")",
                     rc["mean_s"], ROLL_CALL, ""))
    for r in s["rounds"]:
        if r["last_s"] is not None:
            note = "" if r["checked_in"] >= r["enrolled"] else f"  ({r['checked_in']}/{r['enrolled']} checked in)"
            rows.append((f"SmartCheck {r['label']}", r["last_s"], SMARTCHECK, note))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(10, 0.75 * len(rows) + 1.4))
    top = max(v for _, v, _, _ in rows)
    ax.set_xlim(0, top * 1.25)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    for i, (label, value, color, note) in enumerate(rows):
        ax.barh(i, value, height=0.55, color=color, linewidth=0)
        ax.text(value + top * 0.015, i, f"{value:g} s{note}", va="center", color=INK, fontsize=13)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], color=INK)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlabel("Seconds until the whole class is recorded (SmartCheck: from \"open class\" to the last check-in)")
    ax.set_title("Whole-class time: SmartCheck vs calling the roll", loc="left", color=INK, fontsize=15, pad=12)
    save(fig, folder, "class_time")


def per_student(s, folder):
    rounds = [r for r in s["rounds"] if r["seconds"]]
    if not rounds:
        return
    fig, ax = plt.subplots(figsize=(10, 0.8 * len(rounds) + 1.5))
    top = max(max(r["seconds"]) for r in rounds)
    for i, r in enumerate(rounds):
        xs = r["seconds"]
        ax.scatter(xs, [i] * len(xs), s=90, color=SMARTCHECK, edgecolors=SURFACE, linewidths=2, zorder=3)
        med = r["median_s"]
        ax.plot([med, med], [i - 0.28, i + 0.28], color=INK, linewidth=2, zorder=4)
        ax.text(med, i - 0.34, f"median {med:.1f} s", ha="center", va="bottom", color=INK, fontsize=12)
    ax.set_yticks(range(len(rounds)), [f"{r['label']}  ({r['checked_in']}/{r['enrolled']})" for r in rounds], color=INK)
    ax.set_ylim(len(rounds) - 0.5, -0.75)
    ax.set_xlim(0, top * 1.12)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlabel("Seconds after the teacher opened the class (one dot per student)")
    ax.set_title("When each student was recorded", loc="left", color=INK, fontsize=15, pad=12)
    save(fig, folder, "per_student")


def projection(s, folder):
    rc = s.get("roll_call") or {}
    measured = [r["last_s"] for r in s["rounds"] if r["last_s"] is not None and r["checked_in"] >= r["enrolled"]]
    if not rc.get("per_name_s") or not measured:
        return
    n_test = rc["names"]
    smart = max(measured)            # the slowest complete round, not the best one
    ns = list(range(0, 61, 5))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    # measured part solid, estimate dashed
    ax.plot([0, n_test], [0, rc["per_name_s"] * n_test], color=ROLL_CALL, linewidth=2, solid_capstyle="round")
    ax.plot([n_test, ns[-1]], [rc["per_name_s"] * n_test, rc["per_name_s"] * ns[-1]], color=ROLL_CALL,
            linewidth=2, linestyle=(0, (5, 4)))
    ax.plot([n_test, n_test], [smart, smart], marker="o", markersize=9, color=SMARTCHECK,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
    ax.plot([n_test, ns[-1]], [smart, smart], color=SMARTCHECK, linewidth=2, linestyle=(0, (5, 4)))
    ax.plot([n_test], [rc["per_name_s"] * n_test], marker="o", markersize=9, color=ROLL_CALL,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
    ax.text(ns[-1] + 0.8, rc["per_name_s"] * ns[-1], f"Roll call\n{rc['per_name_s'] * ns[-1]:.0f} s at {ns[-1]}",
            color=INK, va="center", fontsize=12)
    ax.text(ns[-1] + 0.8, smart, f"SmartCheck\n~{smart:g} s", color=INK, va="center", fontsize=12)
    ax.set_xlim(0, ns[-1])
    ax.set_ylim(0, max(rc["per_name_s"] * ns[-1], smart) * 1.1)
    ax.set_xticks(ns)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Students in the class (dots = measured with {n_test}; dashed = estimate)")
    ax.set_ylabel("Seconds")
    ax.set_title("Bigger classes: roll call grows per name, SmartCheck students check in at once",
                 loc="left", color=INK, fontsize=15, pad=12)
    ax.legend(handles=[plt.Line2D([], [], color=ROLL_CALL, linewidth=2, label="Calling the roll"),
                       plt.Line2D([], [], color=SMARTCHECK, linewidth=2, label="SmartCheck")],
              loc="upper left", frameon=False)
    save(fig, folder, "projection")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    folder = Path(sys.argv[1])
    s = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    class_time(s, folder)
    per_student(s, folder)
    projection(s, folder)
    print("charts written to", folder)


if __name__ == "__main__":
    main()

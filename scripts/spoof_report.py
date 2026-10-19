"""APCER per attack type and BPCER from a presentation-attack test log (ISO/IEC 30107-3 terms).

    python scripts/spoof_report.py spoof_attempts.csv [--json out.json]

CSV columns (one row per attempt; template in docs/spoof_test.md):
  stage      enrollment | checkin
  attack     bona_fide | printed_photo | phone_screen | laptop_screen  (other names are kept as given)
  accepted   1 if the system accepted the presentation (enrolled / checked in), else 0
  subject    optional: who presented (for counting subjects, not identities)
  reason     optional: what the system said when it refused

APCER (attack presentation classification error rate) = attacks accepted / attack
attempts, reported per attack type (PAI species) and as the maximum over types.
BPCER (bona fide presentation classification error rate) = bona fide refused /
bona fide attempts. 95% Wilson intervals are given because 20-attempt samples are
small; an observed 0/20 still allows up to ~16%.
"""
import argparse
import csv
import json
import math
from collections import defaultdict

MIN_ATTEMPTS = 20


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def report(rows):
    out = {}
    by_stage = defaultdict(list)
    for r in rows:
        by_stage[r["stage"].strip().lower()].append(r)
    for stage, items in sorted(by_stage.items()):
        groups = defaultdict(list)
        for r in items:
            groups[r["attack"].strip().lower()].append(int(r["accepted"]))
        stage_out = {"attack_types": {}, "warnings": []}
        bona = groups.pop("bona_fide", [])
        if bona:
            refused = sum(1 for a in bona if not a)
            stage_out["BPCER"] = {"rate": round(refused / len(bona), 4), "refused": refused,
                                  "attempts": len(bona), "ci95": wilson(refused, len(bona))}
            if len(bona) < MIN_ATTEMPTS:
                stage_out["warnings"].append(f"bona_fide: {len(bona)} < {MIN_ATTEMPTS} attempts")
        else:
            stage_out["warnings"].append("no bona_fide attempts: BPCER unknown")
        worst = None
        for attack, accepted in sorted(groups.items()):
            k = sum(accepted)
            entry = {"rate": round(k / len(accepted), 4), "accepted": k, "attempts": len(accepted),
                     "ci95": wilson(k, len(accepted))}
            stage_out["attack_types"][attack] = entry
            if len(accepted) < MIN_ATTEMPTS:
                stage_out["warnings"].append(f"{attack}: {len(accepted)} < {MIN_ATTEMPTS} attempts")
            if worst is None or entry["rate"] > worst[1]["rate"]:
                worst = (attack, entry)
        if worst:
            stage_out["APCER_max"] = {"attack": worst[0], **worst[1]}
        out[stage] = stage_out
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv")
    parser.add_argument("--json")
    args = parser.parse_args()
    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("stage") and r.get("attack") and r.get("accepted") != ""]
    result = report(rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

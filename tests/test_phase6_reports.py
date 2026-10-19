"""scripts/spoof_report.py and scripts/realclass_report.py on small synthetic sheets."""
import unittest

from scripts import realclass_report, spoof_report


class SpoofReportTests(unittest.TestCase):
    def rows(self, stage, attack, accepted, refused):
        return [{"stage": stage, "attack": attack, "accepted": "1"}] * accepted + \
               [{"stage": stage, "attack": attack, "accepted": "0"}] * refused

    def test_apcer_per_type_max_and_bpcer(self):
        rows = (self.rows("checkin", "bona_fide", 18, 2) + self.rows("checkin", "printed_photo", 1, 19)
                + self.rows("checkin", "phone_screen", 3, 17) + self.rows("checkin", "laptop_screen", 0, 20))
        r = spoof_report.report(rows)["checkin"]
        self.assertEqual(r["BPCER"]["rate"], 0.1)
        self.assertEqual(r["attack_types"]["printed_photo"]["rate"], 0.05)
        self.assertEqual(r["APCER_max"]["attack"], "phone_screen")
        self.assertEqual(r["APCER_max"]["rate"], 0.15)
        low, high = r["attack_types"]["laptop_screen"]["ci95"]
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.15)          # 0/20 still allows ~16%
        self.assertEqual(r["warnings"], [])

    def test_small_samples_are_flagged(self):
        r = spoof_report.report(self.rows("enrollment", "printed_photo", 0, 5))["enrollment"]
        self.assertIn("no bona_fide attempts: BPCER unknown", r["warnings"])
        self.assertTrue(any("printed_photo: 5 < 20" in w for w in r["warnings"]))


class RealclassReportTests(unittest.TestCase):
    ROWS = [
        {"student": "S01", "phase": "enroll", "attempt": "1", "start": "09:00:00", "end": "09:00:20", "success": "1"},
        {"student": "S02", "phase": "enroll", "attempt": "1", "start": "09:00:05", "end": "09:00:35", "success": "0", "reason": "turn"},
        {"student": "S02", "phase": "enroll", "attempt": "2", "start": "09:00:40", "end": "09:01:00", "success": "1"},
        {"student": "S01", "phase": "checkin", "attempt": "1", "start": "09:30:00", "end": "09:30:06", "success": "1"},
        {"student": "S02", "phase": "checkin", "attempt": "1", "start": "09:30:02", "end": "09:30:10", "success": "1"},
    ]

    def test_phase_numbers(self):
        enroll = realclass_report.phase_report([r for r in self.ROWS if r["phase"] == "enroll"])
        self.assertEqual(enroll["students"], 2)
        self.assertEqual(enroll["first_attempt_success"], 0.5)
        self.assertEqual(enroll["per_student_time_to_success"]["max_s"], 55.0)   # S02: 09:00:05 -> 09:01:00
        self.assertEqual(enroll["failure_reasons"], {"turn": 1})
        checkin = realclass_report.phase_report([r for r in self.ROWS if r["phase"] == "checkin"])
        self.assertEqual(checkin["first_attempt_reject_rate"], 0.0)
        self.assertEqual(checkin["class_elapsed_s"], 10.0)

    def test_scores_from_log(self):
        import os
        import tempfile
        lines = ["x [FACE] engine=arcface best=0.6123 threshold=0.4 pass=True\n",
                 "x [FACE] engine=arcface best=0.3900 threshold=0.4 pass=False\n",
                 "x [FACE] engine=arcface best=0.7000 threshold=0.4 pass=True\n"]
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False, encoding="utf-8") as f:
            f.writelines(lines)
        try:
            s = realclass_report.scores([f.name])
        finally:
            os.unlink(f.name)
        self.assertEqual((s["n"], s["below_threshold"], s["min"]), (3, 1, 0.39))
        self.assertEqual(s["lowest_margins"][0], -0.01)


if __name__ == "__main__":
    unittest.main()

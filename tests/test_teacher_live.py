"""Teacher live view: head-count while a class is open (Phase 8d)."""
import unittest

from app.routes.teacher import live_counts


class LiveCountTests(unittest.TestCase):
    def test_counts_checked_late_and_rest(self):
        rows = [{"status": "present"}, {"status": "late"}, {"status": "manual"}, {"status": "absent"}]
        self.assertEqual(live_counts(rows, 6),
                         {"total": 6, "checked": 3, "late": 1, "present": 2, "rest": 3})

    def test_never_negative_when_rows_outnumber_the_roster(self):
        self.assertEqual(live_counts([{"status": "present"}] * 3, 2)["rest"], 0)


if __name__ == "__main__":
    unittest.main()

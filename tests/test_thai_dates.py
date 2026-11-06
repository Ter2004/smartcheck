"""One date format on every screen: "6 พ.ย. 69" (Phase 11)."""
import unittest
from datetime import date

from app.services.thai_dates import thai_date


class ThaiDateTests(unittest.TestCase):
    def test_utc_timestamp_is_shown_in_thai_time(self):
        # 17:04 UTC on the 5th is 00:04 on the 6th in Thailand.
        self.assertEqual(thai_date("2026-11-05T17:04:29.98+00:00"), "6 พ.ย. 69")
        self.assertEqual(thai_date("2026-11-05T17:04:29Z", with_time=True), "6 พ.ย. 69 00:04")

    def test_plain_date_is_not_shifted(self):
        self.assertEqual(thai_date("2026-09-01"), "1 ก.ย. 69")
        self.assertEqual(thai_date(date(2027, 1, 15)), "15 ม.ค. 70")

    def test_empty_and_unreadable(self):
        self.assertEqual(thai_date(None), "—")
        self.assertEqual(thai_date(""), "—")
        self.assertEqual(thai_date("not a date"), "not a date")


if __name__ == "__main__":
    unittest.main()

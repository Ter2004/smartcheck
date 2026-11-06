"""Student home: today's classes as boarding passes (boarded / boarding / closed / waiting)."""
import unittest
from datetime import datetime, timedelta, timezone

from app.routes.student import board_hero, thai_day_label, today_board

CONFIG = {"SESSION_OPEN_MODE": "teacher", "CLASS_ON_TIME_MINUTES": 5}
TH = timezone(timedelta(hours=7))
NOW = datetime(2026, 11, 6, 8, 3, tzinfo=TH)


def sess(id_, hour, **fields):
    start = datetime(2026, 11, 6, hour, 0, tzinfo=TH).astimezone(timezone.utc).isoformat()
    return {"id": id_, "start_time": start, "is_open": False, "end_time": None,
            "courses": {"code": id_.upper(), "name": "Class " + id_}, "beacons": {"room_name": "TEST-101"}, **fields}


class BoardTests(unittest.TestCase):
    def test_states_and_countdown(self):
        opened = datetime(2026, 11, 6, 8, 0, tzinfo=TH).astimezone(timezone.utc).isoformat()
        board = today_board([
            sess("a", 7, end_time="x"),
            sess("b", 8, is_open=True, opened_at=opened, present_minutes=5),
            sess("c", 13),
            sess("d", 6, end_time="x"),
        ], [{"session_id": "a", "status": "present", "check_in_at": "t"}], NOW, CONFIG)
        self.assertEqual([r["state"] for r in board], ["boarded", "boarding", "waiting", "closed"])
        boarding = board[1]
        self.assertEqual(boarding["on_time_until"], datetime(2026, 11, 6, 1, 5, tzinfo=timezone.utc).isoformat())
        self.assertFalse(boarding["late_now"])
        self.assertEqual((boarding["code"], boarding["room"]), ("B", "TEST-101"))

    def test_hero_prefers_boarding_then_next_then_last(self):
        self.assertEqual(board_hero([{"state": "boarded"}, {"state": "waiting", "id": 2}, {"state": "boarding", "id": 3}])["id"], 3)
        self.assertEqual(board_hero([{"state": "boarded", "id": 1}, {"state": "waiting", "id": 2}])["id"], 2)
        self.assertEqual(board_hero([{"state": "closed", "id": 1}, {"state": "boarded", "id": 2}])["id"], 2)
        self.assertIsNone(board_hero([]))

    def test_thai_day_label(self):
        self.assertEqual(thai_day_label(NOW), "วันศุกร์ 6 พ.ย.")


if __name__ == "__main__":
    unittest.main()

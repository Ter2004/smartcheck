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


class TimetableTests(unittest.TestCase):
    def test_blocks_are_placed_by_time_and_carry_this_weeks_state(self):
        from app.routes.student import week_timetable

        def sch(dow, start, end, cid):
            return {"day_of_week": dow, "start_time": start, "end_time": end, "course_id": cid,
                    "courses": {"code": cid.upper(), "name": "n"}, "beacons": {"room_name": "ห้อง TEST-101"}}
        th_iso = lambda d, h: datetime(2026, 11, d, h, 0, tzinfo=TH).isoformat()   # noqa: E731
        table = week_timetable(
            [sch(0, "08:00:00", "12:00:00", "a"), sch(4, "13:00:00", "16:00:00", "b"),
             sch(2, "09:00:00", "10:00:00", "c"), sch(6, "19:00:00", "23:59:00", "d"), sch(4, "00:00:00", "05:00:00", "e")],
            [{"id": "s-a", "course_id": "a", "start_time": th_iso(2, 8), "end_time": "x"},
             {"id": "s-b", "course_id": "b", "start_time": th_iso(6, 13), "end_time": None},
             {"id": "s-c", "course_id": "c", "start_time": th_iso(4, 9), "end_time": "x"}],
            checked_ids={"s-a"}, open_ids={"s-b"}, now_th=NOW)
        monday, wednesday, friday = table["days"][0], table["days"][2], table["days"][4]
        self.assertEqual((monday["blocks"][0]["top"], monday["blocks"][0]["height"]), (round(60 / 720 * 100, 3), round(240 / 720 * 100, 3)))
        self.assertEqual(monday["blocks"][0]["state"], "boarded")
        self.assertEqual((friday["blocks"][0]["state"], friday["blocks"][0]["session_id"]), ("open", "s-b"))
        self.assertEqual(wednesday["blocks"][0]["state"], "closed")
        self.assertTrue(friday["today"])
        self.assertEqual(sorted(b["code"] for b in table["outside"]), ["D", "E"])
        self.assertEqual(table["hours"], list(range(7, 19)))


if __name__ == "__main__":
    unittest.main()

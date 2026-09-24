import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

from app import scheduler


class MemoryDB:
    def __init__(self, sessions, schedules):
        self.rows = {'sessions': sessions, 'schedules': schedules, 'beacons': []}

    def table(self, name):
        return Query(self.rows[name])


class Query:
    def __init__(self, rows):
        self.rows = rows
        self.filters = []
        self.bounds = (0, 999)
        self.change = None

    def select(self, _):
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def order(self, _):
        return self

    def range(self, start, end):
        self.bounds = (start, end)
        return self

    def limit(self, count):
        return self.range(0, count - 1)

    def update(self, change):
        self.change = change
        return self

    def execute(self):
        rows = [r for r in self.rows if all(r.get(k) == v for k, v in self.filters)]
        if self.change is not None:
            for row in rows:
                row.update(self.change)
        lo, hi = self.bounds
        return SimpleNamespace(data=[r.copy() for r in rows[lo:hi + 1]])


def schedule(start='12:00:00', end='13:00:00', day=0, course='c'):
    return dict(id=start, course_id=course, day_of_week=day,
                start_time=start, end_time=end)


def session(start='2026-09-21T05:05:00+00:00', **kwargs):
    return dict(dict(id='s', course_id='c', start_time=start,
                     is_open=True, end_time=None), **kwargs)


class ExpiryTests(unittest.TestCase):
    def setUp(self):
        scheduler._warned_unmatched.clear()

    def close(self, rows, schedules, now='2026-09-21T07:00:00+00:00'):
        scheduler._close_expired_sessions(MemoryDB(rows, schedules), datetime.fromisoformat(now))

    def test_manual_start_closes_at_scheduled_end(self):
        row = session()
        self.close([row], [schedule()])
        self.assertFalse(row['is_open'])
        self.assertEqual(row['end_time'], '2026-09-21T06:00:00+00:00')

    def test_recovers_previous_days_including_six_day_outage(self):
        rows = [session('2026-09-23T03:30:00+00:00'),
                session('2026-09-18T05:00:00+00:00', id='old')]
        self.close(rows, [schedule('10:30:00', '22:00:00', 2),
                          schedule('12:00:00', '22:00:00', 4)], '2026-09-24T01:00:00+00:00')
        self.assertTrue(all(not r['is_open'] for r in rows))
        self.assertEqual(rows[1]['end_time'], '2026-09-18T15:00:00+00:00')

    def test_adjacent_later_class_remains_open(self):
        rows = [session(), session('2026-09-21T06:00:00+00:00', id='later')]
        self.close(rows, [schedule(), schedule('13:00:00', '15:00:00')])
        self.assertFalse(rows[0]['is_open'])
        self.assertTrue(rows[1]['is_open'])

    def test_exact_end_boundary_and_before_end(self):
        for now, expected in [('2026-09-21T05:59:59+00:00', True),
                              ('2026-09-21T06:00:00+00:00', False)]:
            row = session()
            self.close([row], [schedule()], now)
            self.assertEqual(row['is_open'], expected)

    def test_uses_thai_date_when_utc_date_is_previous_day(self):
        row = session('2026-09-20T18:05:00Z')  # Monday 01:05 Bangkok
        self.close([row], [schedule('01:00:00', '02:00:00')])
        self.assertEqual(row['end_time'], '2026-09-20T19:00:00+00:00')

    def test_unmatched_and_other_course_are_preserved_with_warning(self):
        row = session()
        with self.assertLogs(scheduler._log, level='WARNING'):
            self.close([row], [schedule(course='other')])
        self.assertTrue(row['is_open'])

    def test_overlapping_windows_do_not_close_manual_session_early(self):
        row = session()
        self.close([row], [schedule(), schedule('11:00:00', '15:00:00')])
        self.assertTrue(row['is_open'])

    def test_reads_all_pages_before_mutating_open_rows(self):
        rows = [session(id=str(i)) for i in range(401)]
        self.close(rows, [schedule()])
        self.assertTrue(all(not row['is_open'] for row in rows))

    def test_malformed_row_does_not_block_other_closures(self):
        rows = [session('invalid'), session(id='valid')]
        with self.assertLogs(scheduler._log, level='ERROR'):
            self.close(rows, [schedule()])
        self.assertFalse(rows[1]['is_open'])

    def test_tick_recovers_old_session_without_todays_schedule(self):
        row = session()
        db = MemoryDB([row], [schedule()])
        with patch.object(scheduler, '_get_supabase', return_value=db), \
                patch.object(scheduler, 'datetime', wraps=datetime) as clock:
            clock.now.return_value = datetime.fromisoformat('2026-09-24T01:00:00+00:00')
            scheduler.auto_manage_sessions()
        self.assertFalse(row['is_open'])

    def test_unmatched_warns_once_until_it_matches_again(self):
        row = session()
        with self.assertLogs(scheduler._log, level='WARNING'):
            self.close([row], [schedule(course='other')])
        with self.assertNoLogs(scheduler._log, level='WARNING'):
            self.close([row], [schedule(course='other')])
        self.close([row], [schedule()], '2026-09-21T05:30:00+00:00')  # matched, still open
        with self.assertLogs(scheduler._log, level='WARNING'):
            self.close([row], [schedule(course='other')])

    def test_each_page_uses_a_fresh_query(self):
        class OneShot(Query):
            def range(self, start, end):
                assert self.bounds == (0, 999), 'range() reused on the same builder'
                return super().range(start, end)
        rows = [session(id=str(i)) for i in range(401)]
        db = MemoryDB(rows, [schedule()])
        db.table = lambda name: OneShot(db.rows[name])
        scheduler._close_expired_sessions(db, datetime.fromisoformat('2026-09-21T07:00:00+00:00'))
        self.assertTrue(all(not r['is_open'] for r in rows))


if __name__ == '__main__':
    unittest.main()

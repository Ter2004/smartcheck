"""Tests for the check-in acceptance-window policy (app/services/session_eligibility.py)
and its wiring into api_checkin._eligible and teacher.py's open/reopen routes.

"is_open" (session exists, not closed) is distinct from "accepting check-ins"
(is_open AND an established, unexpired checkin_duration window). A null
checkin_duration means no window has been established — it is rejected, not
treated as unlimited.
"""
import unittest
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from flask import Flask

from app.routes import api_checkin as api_route
from app.routes import teacher as teacher_route
from app.services.security_service import compute_embedding_integrity_hash
from app.services.session_eligibility import (
    DEFAULT_CHECKIN_DURATION_MINUTES, checkin_deadline, is_accepting_checkins,
    window_status, extend_duration_from_now, NOT_SET, NOT_STARTED, EXPIRED, ACCEPTING,
)

NOW = 1_700_000_000  # arbitrary fixed unix time, safely in the past of real wall-clock time


def iso(unix_ts):
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).isoformat()


def at(unix_ts):
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# Pure logic: app/services/session_eligibility.py — no Flask, no DB fake.
# ─────────────────────────────────────────────────────────────────────────────

class SessionEligibilityUnitTests(unittest.TestCase):
    def test_null_duration_has_no_deadline_and_is_not_accepting(self):
        sess = {'is_open': True, 'start_time': iso(NOW), 'checkin_duration': None}
        self.assertIsNone(checkin_deadline(sess))
        self.assertFalse(is_accepting_checkins(sess, at(NOW)))

    def test_closed_session_is_not_accepting_even_with_a_valid_duration(self):
        sess = {'is_open': False, 'start_time': iso(NOW), 'checkin_duration': 30}
        self.assertFalse(is_accepting_checkins(sess, at(NOW)))

    def test_expiry_boundary_is_inclusive_at_the_deadline_instant(self):
        sess = {'is_open': True, 'start_time': iso(NOW), 'checkin_duration': 30}
        self.assertTrue(is_accepting_checkins(sess, at(NOW + 30 * 60 - 1)))
        self.assertTrue(is_accepting_checkins(sess, at(NOW + 30 * 60)))       # exactly at deadline: still accepted
        self.assertFalse(is_accepting_checkins(sess, at(NOW + 30 * 60 + 1)))  # one second past: rejected

    def test_custom_duration_changes_the_deadline(self):
        sess = {'is_open': True, 'start_time': iso(NOW), 'checkin_duration': 5}
        self.assertTrue(is_accepting_checkins(sess, at(NOW + 5 * 60)))
        self.assertFalse(is_accepting_checkins(sess, at(NOW + 5 * 60 + 1)))

    def test_zero_duration_is_treated_as_not_set_not_an_instantly_open_window(self):
        sess = {'is_open': True, 'start_time': iso(NOW), 'checkin_duration': 0}
        self.assertIsNone(checkin_deadline(sess))
        self.assertEqual(window_status(sess, at(NOW)), NOT_SET)

    def test_negative_duration_is_treated_as_not_set_not_an_always_expired_window(self):
        # Negative duration must not be silently "handled" by happening to
        # always compare as expired — it's invalid input and should be
        # reported the same as "no window established", not "expired".
        sess = {'is_open': True, 'start_time': iso(NOW), 'checkin_duration': -5}
        self.assertIsNone(checkin_deadline(sess))
        self.assertEqual(window_status(sess, at(NOW)), NOT_SET)

    def test_malformed_duration_type_does_not_raise(self):
        for bad_duration in ('not-a-number', [], {}, object()):
            with self.subTest(bad_duration=bad_duration):
                sess = {'is_open': True, 'start_time': iso(NOW), 'checkin_duration': bad_duration}
                self.assertIsNone(checkin_deadline(sess))
                self.assertEqual(window_status(sess, at(NOW)), NOT_SET)

    def test_malformed_start_time_does_not_raise(self):
        for bad_start in ('not-a-timestamp', 12345, [], {}):
            with self.subTest(bad_start=bad_start):
                sess = {'is_open': True, 'start_time': bad_start, 'checkin_duration': 30}
                self.assertIsNone(checkin_deadline(sess))
                self.assertEqual(window_status(sess, at(NOW)), NOT_SET)

    def test_missing_start_time_does_not_raise(self):
        sess = {'is_open': True, 'checkin_duration': 30}
        self.assertIsNone(checkin_deadline(sess))
        self.assertEqual(window_status(sess, at(NOW)), NOT_SET)

    def test_future_start_time_is_not_started_not_silently_accepting(self):
        # A session opened with a start_time ahead of "now" (e.g. created via
        # teacher.session_create with an arbitrary future start_time while
        # is_open=true) must not appear to accept check-ins before it starts,
        # even though "now <= deadline" would trivially hold since the
        # deadline is even further in the future.
        future_start = NOW + 3600
        sess = {'is_open': True, 'start_time': iso(future_start), 'checkin_duration': 30}
        self.assertEqual(window_status(sess, at(NOW)), NOT_STARTED)
        self.assertFalse(is_accepting_checkins(sess, at(NOW)))
        # Once "now" reaches start_time, it becomes a normal accepting window.
        self.assertEqual(window_status(sess, at(future_start)), ACCEPTING)
        self.assertEqual(window_status(sess, at(future_start + 30 * 60 + 1)), EXPIRED)

    def test_timezone_naive_start_time_is_not_silently_assumed_utc(self):
        # A valid, parseable timestamp that simply lacks offset/zone info.
        # This is a distinct failure mode from "malformed" above — dateutil
        # parses it successfully as a naive datetime, so it must be rejected
        # explicitly rather than accidentally accepted or guessed at.
        naive = datetime.fromtimestamp(NOW, tz=timezone.utc).replace(tzinfo=None).isoformat()
        sess = {'is_open': True, 'start_time': naive, 'checkin_duration': 30}
        self.assertIsNone(checkin_deadline(sess))
        self.assertEqual(window_status(sess, at(NOW)), NOT_SET)

    def test_overflowing_duration_does_not_raise(self):
        # 10**9 minutes (~1900 years) added to a 2026 start does NOT overflow
        # datetime (verified: lands around year 3928, well under
        # datetime.MAXYEAR=9999) — that would be the wrong value to test
        # with. 10**10 is the actual boundary; 10**12 is comfortably past it
        # and confirmed to raise OverflowError when added to a 2026 date.
        sess = {'is_open': True, 'start_time': iso(NOW), 'checkin_duration': 10**12}
        self.assertIsNone(checkin_deadline(sess))
        self.assertEqual(window_status(sess, at(NOW)), NOT_SET)

    def test_infinite_duration_does_not_raise(self):
        # int(float('inf')) raises OverflowError (confirmed), distinct from
        # the ValueError/TypeError already covered by the malformed-type test.
        sess = {'is_open': True, 'start_time': iso(NOW), 'checkin_duration': float('inf')}
        self.assertIsNone(checkin_deadline(sess))
        self.assertEqual(window_status(sess, at(NOW)), NOT_SET)


class ExtendDurationFromNowTests(unittest.TestCase):
    def test_rounds_elapsed_time_up_never_short_changing_the_request(self):
        # 5 minutes 30 seconds elapsed — deliberately NOT an exact minute
        # boundary, to actually exercise ceil() rather than a coincidentally
        # exact case.
        original_start = at(NOW - 330)
        duration, starts_in_future = extend_duration_from_now(original_start, 20, at(NOW))
        self.assertFalse(starts_in_future)
        deadline = original_start + timedelta(minutes=duration)
        nominal = at(NOW) + timedelta(minutes=20)
        # The only guarantee: at least the requested window, and at most
        # just under 60 extra seconds from integer-minute rounding. This is
        # NOT second-level accuracy — do not tighten this tolerance.
        self.assertGreaterEqual((deadline - nominal).total_seconds(), 0)
        self.assertLess((deadline - nominal).total_seconds(), 60)

    def test_rounds_up_even_for_a_single_extra_second(self):
        # 1 second past an exact minute boundary must still round up to the
        # next whole minute, not truncate back down to it.
        original_start = at(NOW - 61)
        duration, _ = extend_duration_from_now(original_start, 20, at(NOW))
        self.assertEqual(duration, 2 + 20)  # ceil(61/60) = 2

    def test_exact_minute_boundary_does_not_round_up_unnecessarily(self):
        original_start = at(NOW - 300)  # exactly 5 minutes
        duration, _ = extend_duration_from_now(original_start, 20, at(NOW))
        self.assertEqual(duration, 5 + 20)

    def test_future_start_uses_original_start_not_now(self):
        future_start = at(NOW + 3600)
        duration, starts_in_future = extend_duration_from_now(future_start, 20, at(NOW))
        self.assertTrue(starts_in_future)
        self.assertEqual(future_start + timedelta(minutes=duration),
                          future_start + timedelta(minutes=20))


# ─────────────────────────────────────────────────────────────────────────────
# Teacher-side: default/custom duration on open, off-schedule manual open,
# and the new session_set_window action.
# ─────────────────────────────────────────────────────────────────────────────

class TeacherDatabase:
    """Same fake-table shape as tests/test_checkin_totp.py's Database, minimal
    for teacher.py's session_create / session_toggle / session_set_window."""
    def __init__(self, session_row=None, course_row=None):
        self.session_row = session_row
        self.course_row = course_row
        self.updates = []
        self.inserts = []
        self.tables_queried = []

    def table(self, name):
        self.tables_queried.append(name)
        query = Mock()
        for method in ['select', 'eq', 'maybe_single']:
            getattr(query, method).return_value = query
        data = {'sessions': self.session_row, 'courses': self.course_row}.get(name)
        query.execute.return_value = SimpleNamespace(data=data)
        def update(record):
            self.updates.append(record)
            return query
        query.update.side_effect = update
        def insert(record):
            self.inserts.append(record)
            return SimpleNamespace(execute=lambda: SimpleNamespace(data=[{'id': 'new-session'}]))
        query.insert.side_effect = insert
        return query


class ParseRequestedMinutesTests(unittest.TestCase):
    """Direct tests for teacher._parse_requested_minutes — confirmed
    reproductions where str.isdigit() is True but int() still raises:
    a non-ASCII digit character int() rejects, and (Python 3.11's
    sys.get_int_max_str_digits() default of 4300) a numeric string long
    enough to trip the integer-string-conversion length guard."""

    def test_non_ascii_digit_character_is_a_validation_error_not_a_crash(self):
        # '²' (superscript two) — confirmed: ''.isdigit() is True,
        # int() raises ValueError.
        self.assertTrue('²'.isdigit())
        with self.assertRaises(ValueError):
            int('²')
        minutes, error = teacher_route._parse_requested_minutes('²')
        self.assertIsNone(minutes)
        self.assertIsNotNone(error)

    def test_extremely_long_digit_string_is_a_validation_error_not_a_crash(self):
        long_digits = '1' * 5000
        self.assertTrue(long_digits.isdigit())
        with self.assertRaises(ValueError):
            int(long_digits)
        minutes, error = teacher_route._parse_requested_minutes(long_digits)
        self.assertIsNone(minutes)
        self.assertIsNotNone(error)

    def test_blank_still_defaults_to_30(self):
        for blank in (None, '', '   '):
            with self.subTest(blank=blank):
                minutes, error = teacher_route._parse_requested_minutes(blank)
                self.assertEqual(minutes, DEFAULT_CHECKIN_DURATION_MINUTES)
                self.assertIsNone(error)

    def test_range_boundaries_still_enforced(self):
        self.assertEqual(teacher_route._parse_requested_minutes('1'), (1, None))
        self.assertEqual(teacher_route._parse_requested_minutes('120'), (120, None))
        self.assertEqual(teacher_route._parse_requested_minutes('121')[0], None)
        self.assertEqual(teacher_route._parse_requested_minutes('0')[0], None)


class TeacherWindowTests(unittest.TestCase):
    def _session_row(self, **overrides):
        row = {
            'id': 'session', 'is_open': False, 'start_time': iso(NOW),
            'course_id': 'course', 'checkin_duration': None, 'end_time': None,
            'courses': {'id': 'course', 'teacher_id': 'teacher'},
        }
        row.update(overrides)
        return row

    def _client(self, session_row=None, course_row=None):
        db = TeacherDatabase(session_row, course_row)
        web = Flask(__name__)
        web.config.update(SECRET_KEY='test')
        web.register_blueprint(teacher_route.teacher_bp, url_prefix='/teacher')
        client = web.test_client()
        with client.session_transaction() as sess:
            sess.update(user_id='teacher', user_role='teacher', csrf_token='csrf')
        return db, client

    def test_open_with_blank_duration_defaults_to_30_minutes(self):
        db, client = self._client(self._session_row(is_open=False))
        with patch.object(teacher_route, 'supabase_admin', db):
            response = client.post('/teacher/session/session/toggle',
                                    data={'csrf_token': 'csrf'})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(db.updates[-1]['checkin_duration'], DEFAULT_CHECKIN_DURATION_MINUTES)
        self.assertTrue(db.updates[-1]['is_open'])

    def test_open_honors_a_custom_duration(self):
        db, client = self._client(self._session_row(is_open=False))
        with patch.object(teacher_route, 'supabase_admin', db):
            response = client.post('/teacher/session/session/toggle',
                                    data={'csrf_token': 'csrf', 'checkin_duration': '45'})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(db.updates[-1]['checkin_duration'], 45)

    def test_open_rejects_invalid_nonblank_duration_without_silently_defaulting(self):
        for bad_value in ('abc', '-5', '0', '3.5', '121', '999', '²', '1' * 5000):
            with self.subTest(bad_value=bad_value):
                db, client = self._client(self._session_row(is_open=False))
                with patch.object(teacher_route, 'supabase_admin', db):
                    response = client.post('/teacher/session/session/toggle',
                                            data={'csrf_token': 'csrf', 'checkin_duration': bad_value})
                self.assertEqual(response.status_code, 302)
                self.assertEqual(db.updates, [])  # rejected, not silently coerced to 30

    def test_open_accepts_the_full_1_to_120_range(self):
        for edge_value in ('1', '120'):
            with self.subTest(edge_value=edge_value):
                db, client = self._client(self._session_row(is_open=False))
                with patch.object(teacher_route, 'supabase_admin', db):
                    response = client.post('/teacher/session/session/toggle',
                                            data={'csrf_token': 'csrf', 'checkin_duration': edge_value})
                self.assertEqual(response.status_code, 302)
                self.assertEqual(db.updates[-1]['checkin_duration'], int(edge_value))

    def test_off_schedule_manual_open_is_still_allowed(self):
        # No 'schedules' row for any course/day — the plain "no recurring
        # schedule at all" case. Policy #2 says this must remain allowed.
        db, client = self._client(self._session_row(is_open=False))
        with patch.object(teacher_route, 'supabase_admin', db):
            response = client.post('/teacher/session/session/toggle',
                                    data={'csrf_token': 'csrf'})
        self.assertEqual(response.status_code, 302)
        self.assertTrue(db.updates[-1]['is_open'])

    def test_manual_open_succeeds_even_when_schedule_exists_today_outside_its_window(self):
        # Corrective fix: previously, a schedule row existing for today but
        # not covering the current time (±30min buffer) blocked opening
        # outright. session_toggle must no longer even query 'schedules' for
        # this decision — ownership + CSRF are the only gates. Proven here by
        # asserting 'schedules' is never queried at all, not just that the
        # (now-nonexistent) block didn't fire.
        db, client = self._client(self._session_row(is_open=False))
        with patch.object(teacher_route, 'supabase_admin', db):
            response = client.post('/teacher/session/session/toggle',
                                    data={'csrf_token': 'csrf'})
        self.assertEqual(response.status_code, 302)
        self.assertTrue(db.updates[-1]['is_open'])
        self.assertNotIn('schedules', db.tables_queried)

    def test_first_open_of_a_never_closed_session_still_resets_start_time(self):
        # end_time is null (never closed before) -> genuinely a first open,
        # where now() really is the class's actual start. This must keep
        # working exactly as before the reopen-detection change.
        db, client = self._client(self._session_row(is_open=False, end_time=None,
                                                      start_time=iso(NOW - 5 * 86400)))
        with patch.object(teacher_route, 'supabase_admin', db), \
             patch.object(teacher_route, 'datetime', Mock(now=Mock(return_value=at(NOW)))):
            response = client.post('/teacher/session/session/toggle',
                                    data={'csrf_token': 'csrf'})
        self.assertEqual(response.status_code, 302)
        update = db.updates[-1]
        self.assertEqual(update['start_time'], iso(NOW))
        self.assertEqual(update['checkin_duration'], DEFAULT_CHECKIN_DURATION_MINUTES)

    def test_reopen_of_a_previously_closed_session_preserves_start_time(self):
        # end_time IS NOT NULL -> verified evidence (see teacher.py comment)
        # that this session was opened and closed before. Reopening it must
        # not overwrite the original class start.
        stale_start = NOW - 5 * 86400
        db, client = self._client(self._session_row(
            is_open=False, start_time=iso(stale_start), end_time=iso(NOW - 3600)))
        with patch.object(teacher_route, 'supabase_admin', db), \
             patch.object(teacher_route, 'datetime', Mock(now=Mock(return_value=at(NOW)))):
            response = client.post('/teacher/session/session/toggle',
                                    data={'csrf_token': 'csrf', 'checkin_duration': '20'})
        self.assertEqual(response.status_code, 302)
        update = db.updates[-1]
        self.assertNotIn('start_time', update)  # original class start preserved
        deadline = at(stale_start) + timedelta(minutes=update['checkin_duration'])
        self.assertGreaterEqual((deadline - at(NOW)).total_seconds(), 20 * 60)
        self.assertLess((deadline - at(NOW)).total_seconds(), 20 * 60 + 60)

    def test_reopen_rejected_when_original_start_time_is_malformed(self):
        db, client = self._client(self._session_row(
            is_open=False, start_time='not-a-timestamp', end_time=iso(NOW - 3600)))
        with patch.object(teacher_route, 'supabase_admin', db):
            response = client.post('/teacher/session/session/toggle',
                                    data={'csrf_token': 'csrf', 'checkin_duration': '20'})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(db.updates, [])

    def test_reopen_rejects_invalid_nonblank_duration_without_silently_defaulting(self):
        for bad_value in ('500', '²', '1' * 5000):
            with self.subTest(bad_value=bad_value):
                db, client = self._client(self._session_row(
                    is_open=False, start_time=iso(NOW - 3600), end_time=iso(NOW - 1800)))
                with patch.object(teacher_route, 'supabase_admin', db):
                    response = client.post('/teacher/session/session/toggle',
                                            data={'csrf_token': 'csrf', 'checkin_duration': bad_value})
                self.assertEqual(response.status_code, 302)
                self.assertEqual(db.updates, [])

    def test_set_window_preserves_start_time_and_extends_duration_to_cover_elapsed_time(self):
        # This is the LOADTEST101 shape: is_open=true, checkin_duration=null,
        # stale (5-day-old) start_time. start_time must NOT be reset — it
        # drives late-arrival classification and historical/weekly grouping
        # elsewhere (see the trace in the session-eligibility corrective
        # work). Instead, checkin_duration is extended so the deadline
        # (start_time + checkin_duration) lands at now + requested minutes,
        # rounded up to whole minutes (never short of the request; up to
        # just under 60 extra seconds — not second-level accuracy).
        stale_start = NOW - 5 * 86400
        db, client = self._client(self._session_row(is_open=True, checkin_duration=None,
                                                     start_time=iso(stale_start)))
        with patch.object(teacher_route, 'supabase_admin', db), \
             patch.object(teacher_route, 'datetime', Mock(now=Mock(return_value=at(NOW)))):
            response = client.post('/teacher/session/session/set-window',
                                    data={'csrf_token': 'csrf', 'checkin_duration': '20'})
        self.assertEqual(response.status_code, 302)
        update = db.updates[-1]
        self.assertNotIn('start_time', update)  # original class start is preserved untouched
        self.assertNotIn('is_open', update)
        computed_deadline = at(stale_start) + timedelta(minutes=update['checkin_duration'])
        self.assertGreaterEqual((computed_deadline - at(NOW)).total_seconds(), 20 * 60)
        self.assertLess((computed_deadline - at(NOW)).total_seconds(), 20 * 60 + 60)

    def test_set_window_on_a_not_yet_started_session_uses_original_start_not_now(self):
        # start_time is in the future (e.g. teacher.session_create with a
        # planned future start while is_open=true). The window must be
        # phrased and computed relative to that future start, not "from now".
        future_start = NOW + 3600
        db, client = self._client(self._session_row(is_open=True, checkin_duration=None,
                                                     start_time=iso(future_start)))
        with patch.object(teacher_route, 'supabase_admin', db), \
             patch.object(teacher_route, 'datetime', Mock(now=Mock(return_value=at(NOW)))):
            response = client.post('/teacher/session/session/set-window',
                                    data={'csrf_token': 'csrf', 'checkin_duration': '20'})
        self.assertEqual(response.status_code, 302)
        update = db.updates[-1]
        self.assertEqual(update['checkin_duration'], 20)  # no elapsed time to add
        deadline = at(future_start) + timedelta(minutes=update['checkin_duration'])
        self.assertEqual(deadline, at(future_start) + timedelta(minutes=20))

    def test_set_window_rejects_invalid_nonblank_duration_without_silently_defaulting(self):
        for bad_value in ('abc', '-5', '0', '121', '²', '1' * 5000):
            with self.subTest(bad_value=bad_value):
                db, client = self._client(self._session_row(is_open=True, checkin_duration=30))
                with patch.object(teacher_route, 'supabase_admin', db):
                    response = client.post('/teacher/session/session/set-window',
                                            data={'csrf_token': 'csrf', 'checkin_duration': bad_value})
                self.assertEqual(response.status_code, 302)
                self.assertEqual(db.updates, [])

    def test_set_window_rejected_when_session_is_closed(self):
        db, client = self._client(self._session_row(is_open=False))
        with patch.object(teacher_route, 'supabase_admin', db):
            response = client.post('/teacher/session/session/set-window',
                                    data={'csrf_token': 'csrf', 'checkin_duration': '20'})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(db.updates, [])  # no update happened

    def test_set_window_rejected_when_original_start_time_is_missing_or_malformed(self):
        for bad_start in (None, '', 'not-a-timestamp'):
            with self.subTest(bad_start=bad_start):
                db, client = self._client(self._session_row(is_open=True, start_time=bad_start))
                with patch.object(teacher_route, 'supabase_admin', db):
                    response = client.post('/teacher/session/session/set-window',
                                            data={'csrf_token': 'csrf', 'checkin_duration': '20'})
                self.assertEqual(response.status_code, 302)
                self.assertEqual(db.updates, [])


class SessionCreateTests(unittest.TestCase):
    """teacher.session_create — checkin_duration must be validated the same
    way as session_toggle/session_set_window (blank -> default, invalid
    nonblank -> rejected, never silently coerced or crashing)."""

    def _client(self):
        db = TeacherDatabase(course_row={'id': 'course'})
        web = Flask(__name__)
        web.config.update(SECRET_KEY='test')
        web.register_blueprint(teacher_route.teacher_bp, url_prefix='/teacher')
        client = web.test_client()
        with client.session_transaction() as sess:
            sess.update(user_id='teacher', user_role='teacher', csrf_token='csrf')
        return db, client

    def _form(self, **overrides):
        form = {'csrf_token': 'csrf', 'course_id': 'course', 'beacon_id': 'beacon',
                'title': 'Test', 'start_time': iso(NOW), 'end_time': iso(NOW + 3600)}
        form.update(overrides)
        return form

    def test_blank_duration_defaults_to_30_minutes(self):
        db, client = self._client()
        with patch.object(teacher_route, 'supabase_admin', db):
            response = client.post('/teacher/session/create', data=self._form())
        self.assertEqual(response.status_code, 302)
        self.assertEqual(db.inserts[-1]['checkin_duration'], DEFAULT_CHECKIN_DURATION_MINUTES)

    def test_custom_duration_is_honored(self):
        db, client = self._client()
        with patch.object(teacher_route, 'supabase_admin', db):
            response = client.post('/teacher/session/create',
                                    data=self._form(checkin_duration='45'))
        self.assertEqual(response.status_code, 302)
        self.assertEqual(db.inserts[-1]['checkin_duration'], 45)

    def test_invalid_nonblank_duration_is_rejected_not_silently_defaulted(self):
        for bad_value in ('abc', '-5', '0', '121', '²', '1' * 5000):
            with self.subTest(bad_value=bad_value):
                db, client = self._client()
                with patch.object(teacher_route, 'supabase_admin', db):
                    response = client.post('/teacher/session/create',
                                            data=self._form(checkin_duration=bad_value))
                self.assertEqual(response.status_code, 302)
                self.assertEqual(db.inserts, [])


# ─────────────────────────────────────────────────────────────────────────────
# API-level: _eligible() runs at both /api/checkin/proximity and /api/checkin.
# ─────────────────────────────────────────────────────────────────────────────

class ApiDatabase:
    def __init__(self, session_row, enrollment_row='present'):
        self.session_row = session_row
        self.enrollment_row = enrollment_row  # 'present' | None (simulates zero-row maybe_single())
        self.records = []

    def table(self, name):
        query = Mock()
        for method in ['select', 'eq', 'neq', 'maybe_single', 'limit']:
            getattr(query, method).return_value = query
        if name == 'course_enrollments':
            data = {'id': 'enrollment'} if self.enrollment_row == 'present' else None
        else:
            data = {
                'sessions': self.session_row,
                'users': {'device_id': ''},
                'student_biometrics': {
                    'face_embeddings': [[1., 0.]],
                    'integrity_hash': compute_embedding_integrity_hash('student', [[1., 0.]], 'salt')},
            }.get(name)
        query.execute.return_value = SimpleNamespace(data=data)
        def insert(record):
            self.records.append(record)
            return query
        query.insert.side_effect = insert
        return query


class ApiWindowTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name, result in {
            'server_validate_frame': {'valid': True},
            'detect_screen_moire': {'avg_score': 0., 'is_screen': False},
            'detect_screen_texture': False,
            'combined_spoof_score': {'is_real': True},
            'extract_embedding': [1., 0.],
            'verify_face_multi': {'verified': True, 'best_similarity': 1., 'avg_similarity': 1.},
        }.items():
            self.stack.enter_context(patch.object(api_route, name, return_value=result))
        self.stack.enter_context(patch.object(api_route, '_decode_image',
            side_effect=lambda image: np.full((64, 64, 3), int(image), dtype=np.uint8)))

    def _client(self, session_row, enrollment_row='present'):
        db = ApiDatabase(session_row, enrollment_row)
        web = Flask(__name__)
        web.config.update(SECRET_KEY='test', ESP32_TOTP_SECRET=b'secret',
                           PROXIMITY_RECEIPT_SECRET='r' * 64,
                           EMBEDDING_INTEGRITY_SALT='salt', RATELIMIT_ENABLED=False,
                           CHECKIN_PROXIMITY_METHOD='ble')
        import app as app_module
        app_module.limiter.init_app(web)
        web.register_blueprint(api_route.api_checkin_bp)
        self.stack.enter_context(patch.object(api_route, 'supabase_admin', db))
        client = web.test_client()
        with client.session_transaction() as sess:
            sess.update(user_id='student', user_role='student', csrf_token='csrf')
        payload = dict(session_id='session', room_code='ROOM-1', liveness_action='passive',
                       face_image='20', face_images=['20', '80', '140'], ear_samples=[.25, .26])
        return db, client, payload

    def _session_row(self, **overrides):
        row = {'id': 'session', 'course_id': 'course', 'is_open': True,
               'start_time': iso(NOW), 'checkin_duration': None,
               'beacons': {'ble_room_code': 'ROOM-1'}}
        row.update(overrides)
        return row

    def _post(self, client, db, payload, now=None):
        # Sessions in this file are anchored at NOW (a fixed past unix time),
        # not real wall-clock time, so "now" for the request must be pinned
        # too — otherwise a real `datetime.now()` decades after NOW would
        # make every duration look expired regardless of what's being tested.
        with patch.object(api_route, 'datetime', Mock(now=Mock(return_value=now or at(NOW)))):
            preflight = client.post('/api/checkin/proximity', json=payload,
                                    headers={'X-CSRF-Token': 'csrf'})
            final = None if preflight.status_code != 200 else client.post(
                '/api/checkin',
                json={**payload, 'proximity_receipt': preflight.json['proximity_receipt']},
                headers={'X-CSRF-Token': 'csrf'})
        return preflight, final

    def test_null_duration_rejected_at_preflight(self):
        db, client, payload = self._client(self._session_row(checkin_duration=None))
        preflight, _ = self._post(client, db, payload)
        self.assertEqual(preflight.status_code, 400)
        self.assertEqual(preflight.json['error_code'], 'checkin_window_not_set')
        self.assertEqual(db.records, [])

    def test_null_duration_rejected_at_final_submit_even_if_preflight_state_changes(self):
        # Preflight passes (duration set), then the teacher clears the window
        # (or it was never really set — same shape) before final submit.
        db, client, payload = self._client(self._session_row(checkin_duration=30))
        with patch.object(api_route, 'datetime', Mock(now=Mock(return_value=at(NOW)))):
            preflight = client.post('/api/checkin/proximity', json=payload,
                                    headers={'X-CSRF-Token': 'csrf'})
            self.assertEqual(preflight.status_code, 200)
            db.session_row['checkin_duration'] = None
            final = client.post('/api/checkin',
                                json={**payload, 'proximity_receipt': preflight.json['proximity_receipt']},
                                headers={'X-CSRF-Token': 'csrf'})
        self.assertEqual(final.status_code, 400)
        self.assertEqual(final.json['error_code'], 'checkin_window_not_set')
        self.assertEqual(db.records, [])

    def test_default_and_custom_duration_allow_checkin_within_window(self):
        for duration in (DEFAULT_CHECKIN_DURATION_MINUTES, 5, 120):
            with self.subTest(duration=duration):
                db, client, payload = self._client(self._session_row(checkin_duration=duration))
                preflight, final = self._post(client, db, payload)
                self.assertEqual(preflight.status_code, 200)
                self.assertEqual(final.status_code, 200, final.json)
                self.assertEqual(len(db.records), 1)

    def test_expired_window_rejected_at_final_submit_with_no_scheduler_involved(self):
        # Nothing in this process runs app/scheduler.py — the rejection here is
        # purely the synchronous deadline comparison inside _eligible, proving
        # it holds even if the scheduler is stopped/never started.
        db, client, payload = self._client(self._session_row(checkin_duration=30))
        with patch.object(api_route, 'datetime', Mock(now=Mock(return_value=at(NOW)))):
            preflight = client.post('/api/checkin/proximity', json=payload,
                                    headers={'X-CSRF-Token': 'csrf'})
        self.assertEqual(preflight.status_code, 200)
        past_deadline = at(NOW + 31 * 60)
        with patch.object(api_route, 'datetime', Mock(now=Mock(return_value=past_deadline))):
            final = client.post('/api/checkin',
                                json={**payload, 'proximity_receipt': preflight.json['proximity_receipt']},
                                headers={'X-CSRF-Token': 'csrf'})
        self.assertEqual(final.status_code, 400)
        self.assertEqual(final.json['error_code'], 'checkin_deadline_exceeded')
        self.assertEqual(db.records, [])

    def test_future_start_time_rejected_as_not_started_not_a_crash(self):
        db, client, payload = self._client(
            self._session_row(checkin_duration=30, start_time=iso(NOW + 3600)))
        preflight, _ = self._post(client, db, payload)
        self.assertEqual(preflight.status_code, 400)
        self.assertEqual(preflight.json['error_code'], 'checkin_not_started')
        self.assertEqual(db.records, [])

    def test_zero_and_negative_duration_rejected_as_window_not_set_not_a_503(self):
        for bad_duration in (0, -5):
            with self.subTest(bad_duration=bad_duration):
                db, client, payload = self._client(self._session_row(checkin_duration=bad_duration))
                preflight, _ = self._post(client, db, payload)
                self.assertEqual(preflight.status_code, 400)
                self.assertEqual(preflight.json['error_code'], 'checkin_window_not_set')
                self.assertEqual(db.records, [])

    def test_malformed_start_time_rejected_cleanly_not_a_503(self):
        db, client, payload = self._client(
            self._session_row(checkin_duration=30, start_time='not-a-timestamp'))
        preflight, _ = self._post(client, db, payload)
        self.assertEqual(preflight.status_code, 400)
        self.assertEqual(preflight.json['error_code'], 'checkin_window_not_set')
        self.assertEqual(db.records, [])

    def test_expired_message_reports_the_local_deadline_time(self):
        db, client, payload = self._client(self._session_row(checkin_duration=30))
        with patch.object(api_route, 'datetime', Mock(now=Mock(return_value=at(NOW)))):
            preflight = client.post('/api/checkin/proximity', json=payload,
                                    headers={'X-CSRF-Token': 'csrf'})
        self.assertEqual(preflight.status_code, 200)
        with patch.object(api_route, 'datetime', Mock(now=Mock(return_value=at(NOW + 31 * 60)))):
            final = client.post('/api/checkin',
                                json={**payload, 'proximity_receipt': preflight.json['proximity_receipt']},
                                headers={'X-CSRF-Token': 'csrf'})
        self.assertEqual(final.status_code, 400)
        self.assertEqual(final.json['error_code'], 'checkin_deadline_exceeded')
        self.assertIn('ปิดรับเมื่อ', final.json['error'])

    def test_cross_section_rejection_returns_403_on_verified_zero_row_maybe_single(self):
        # course_enrollments.maybe_single() on zero matching rows returns Python
        # None (verified this session against the installed postgrest package:
        # SyncMaybeSingleRequestBuilder.execute() catches the "0 rows" APIError
        # and returns None, not an object with .data=None). api_checkin.py's
        # `if not enrollment or not enrollment.data:` guard handles that shape
        # correctly, short-circuiting before touching `.data` on None.
        db, client, payload = self._client(self._session_row(checkin_duration=30),
                                            enrollment_row=None)
        preflight, final = self._post(client, db, payload)
        self.assertEqual(preflight.status_code, 403)
        self.assertEqual(db.records, [])


if __name__ == '__main__':
    unittest.main()

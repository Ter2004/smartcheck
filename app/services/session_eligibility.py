"""Shared check-in acceptance-window rules.

"is_open" means the session exists and hasn't been closed. It does not by
itself mean check-ins are being accepted right now — that also requires an
established, unexpired `checkin_duration` window that has actually begun.
A null, zero, negative, or unparseable `checkin_duration` (or `start_time`)
means no acceptance window has been established — it is NOT treated as
unlimited, and it never raises: malformed data is treated the same as
"not set" rather than crashing the caller. A `start_time` without timezone
information is treated as unparseable — never silently assumed to be UTC
or any other zone. This module is the single place this is computed — both
student-facing session selection and the API's authorization call into it,
so the two can't independently drift. Teacher routes that need to extend an
acceptance window without resetting start_time (session_toggle's reopen
path, session_set_window) call extend_duration_from_now() below, so that
math too is computed in exactly one place.

Existing rows are read as-is and never rewritten here.
"""
import math
from datetime import datetime, timedelta, timezone
from dateutil import parser as dtparser

DEFAULT_CHECKIN_DURATION_MINUTES = 30

# window_status() outcomes.
NOT_SET = "not_set"          # no valid checkin_duration/start_time established
NOT_STARTED = "not_started"  # now is before start_time
EXPIRED = "expired"          # now is past the deadline
ACCEPTING = "accepting"      # now is within [start_time, deadline]


def parse_start_time(sess):
    """Parse sess['start_time'] into a timezone-aware UTC-comparable
    datetime, or None if it's missing, unparseable, or lacks timezone
    information. A naive (timezone-less) timestamp is never assumed to be
    UTC or any other zone — it's treated the same as unparseable, because
    guessing would silently misplace the deadline by an unknown offset."""
    start_time = sess.get("start_time")
    if not start_time:
        return None
    try:
        parsed = dtparser.parse(start_time)
    except (ValueError, TypeError, OverflowError):
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _duration_minutes(sess):
    raw = sess.get("checkin_duration")
    if raw is None:
        return None
    try:
        minutes = int(raw)
    except (ValueError, TypeError, OverflowError):
        return None
    return minutes if minutes > 0 else None


def checkin_deadline(sess):
    """UTC datetime after which check-ins stop being accepted, or None if no
    acceptance window has been established. Never raises — this includes an
    absurdly large checkin_duration overflowing datetime's representable
    range, which is treated as "not set" rather than propagating
    OverflowError."""
    minutes = _duration_minutes(sess)
    start_time = parse_start_time(sess)
    if minutes is None or start_time is None:
        return None
    try:
        return start_time + timedelta(minutes=minutes)
    except OverflowError:
        return None


def window_status(sess, now=None):
    """Classify the acceptance window. Assumes the caller already knows the
    session is_open (both current call sites filter on that beforehand); a
    closed session is reported as NOT_SET since it isn't meaningful to ask
    whether a closed session's window is expired vs. accepting. Never raises
    — malformed start_time/checkin_duration are treated as NOT_SET."""
    if not sess.get("is_open"):
        return NOT_SET
    deadline = checkin_deadline(sess)
    if deadline is None:
        return NOT_SET
    start_time = parse_start_time(sess)
    now = now or datetime.now(timezone.utc)
    if now < start_time:
        return NOT_STARTED
    if now > deadline:
        return EXPIRED
    return ACCEPTING


def is_accepting_checkins(sess, now=None):
    """True only if the session is open AND has an unexpired, already-started
    acceptance window."""
    return window_status(sess, now) == ACCEPTING


def extend_duration_from_now(original_start, requested_minutes, now=None):
    """Compute a whole-minute checkin_duration such that, combined with the
    UNCHANGED original_start, the acceptance window covers at least
    requested_minutes starting from now — or, if the class hasn't started
    yet (original_start is still in the future), starting from
    original_start itself, since there is no "now" to extend from before
    the window can even begin.

    Returns (checkin_duration, starts_in_future). Callers must use
    starts_in_future to phrase their message correctly — "from now" is
    wrong when the window actually starts later.

    original_start must already be a timezone-aware datetime (see
    parse_start_time); this function does no parsing or validation itself.
    The returned checkin_duration is the cumulative stored value (elapsed
    time since original_start, plus requested_minutes) and is not bounded
    to the 1-120 range used to validate a teacher's requested-minutes input
    — that range applies to the request, not to this derived total.

    Rounding: elapsed wall-clock time is converted to whole minutes with
    math.ceil, so the resulting deadline can land up to just under 60
    seconds later than the nominal "now + requested_minutes" — never
    earlier, and never guaranteed to the second. Callers must not claim
    second-level accuracy from this value.
    """
    now = now or datetime.now(timezone.utc)
    if original_start >= now:
        return requested_minutes, True
    elapsed_minutes = math.ceil((now - original_start).total_seconds() / 60)
    return elapsed_minutes + requested_minutes, False

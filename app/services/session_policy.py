"""One Bangkok timetable policy for rendering, preflight and final check-in."""
from datetime import datetime, timedelta, timezone, time
from zoneinfo import ZoneInfo

THAI = ZoneInfo('Asia/Bangkok')


def instant(value):
    if not isinstance(value, str):
        raise ValueError('Timestamp required')
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Timezone required')
    return result.astimezone(timezone.utc)


def bounds(row):
    start, end = instant(row['start_time']), instant(row['end_time'])
    opens = instant(row.get('checkin_opens_at') or row['start_time'])
    closes = instant(row.get('checkin_closes_at') or row['end_time'])
    late = instant(row['late_at']) if row.get('late_at') else start + timedelta(minutes=15)
    if row.get('checkin_duration') is not None:
        duration = int(row['checkin_duration'])
        if duration <= 0:
            raise ValueError('Invalid check-in duration')
        closes = min(closes, start + timedelta(minutes=duration))
    if not opens <= start < end or not opens < closes <= end or not start <= late <= closes:
        raise ValueError('Invalid session window')
    return opens, late, closes


def state(row, now=None):
    now = now or datetime.now(timezone.utc)
    if row.get('cancelled_at'):
        return 'cancelled'
    if row.get('session_kind') not in ('scheduled', 'makeup'):
        return 'unconfigured'
    try:
        opens, _, closes = bounds(row)
    except (KeyError, TypeError, ValueError, OverflowError):
        return 'unconfigured'
    if now < opens:
        return 'pending'
    return 'open' if now < closes else 'closed'


def decorate(row, now=None):
    row['policy_state'] = state(row, now)
    row['is_open'] = row['policy_state'] == 'open'
    row['state_label'] = {'pending': 'รอเปิดตามเวลา', 'open': 'เปิดเช็คชื่อ',
                          'closed': 'ปิดเช็คชื่อ', 'cancelled': 'ยกเลิก',
                          'unconfigured': 'คาบเดิม — ต้องตรวจสอบตาราง'}[row['policy_state']]
    return row


def occurrence(schedule, day):
    start = datetime.combine(day, time.fromisoformat(schedule['start_time']), THAI)
    end = datetime.combine(day, time.fromisoformat(schedule['end_time']), THAI)
    if end <= start:
        end += timedelta(days=1)
    early = int(schedule.get('open_before_minutes') or 0)
    late_minutes = int(schedule.get('late_after_minutes', 15))
    close_minutes = schedule.get('close_after_minutes')
    if not 0 <= early <= 60 or not 0 <= late_minutes <= 1440:
        raise ValueError('Invalid schedule offsets')
    closes = end if close_minutes is None else start + timedelta(minutes=int(close_minutes))
    result = dict(start_time=start.isoformat(), end_time=end.isoformat(),
                  checkin_opens_at=(start-timedelta(minutes=early)).isoformat(),
                  late_at=(start+timedelta(minutes=late_minutes)).isoformat(),
                  checkin_closes_at=closes.isoformat(), session_kind='scheduled',
                  schedule_id=schedule['id'], course_id=schedule['course_id'],
                  beacon_id=schedule.get('beacon_id'))
    bounds(result)
    return result

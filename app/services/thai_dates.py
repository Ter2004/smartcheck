"""One date format for every screen: "6 พ.ย. 69" (Buddhist year, two digits).

Owner decision 2026-11-07. Timestamps from the DB are UTC ("...+00:00" or "Z");
they are shown in Thai time. A plain date ("2026-11-06") is shown as it is.
"""
from datetime import date, datetime, timedelta, timezone

TH = timezone(timedelta(hours=7))
MONTHS = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]


def _moment(value):
    """(date, time-or-None) in Thai time, or None when the value cannot be read."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        return value, None
    else:
        text = str(value or "").strip()
        if not text:
            return None
        if len(text) == 10:   # a plain date has no zone to convert
            try:
                return date.fromisoformat(text), None
            except ValueError:
                return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(TH)
    return dt.date(), dt.time()


def thai_date(value, with_time=False, empty="—"):
    """'6 พ.ย. 69', or '6 พ.ย. 69 08:05' with with_time."""
    moment = _moment(value)
    if moment is None:
        return empty if not value else str(value)
    day, clock = moment
    text = f"{day.day} {MONTHS[day.month - 1]} {(day.year + 543) % 100:02d}"
    if with_time and clock is not None:
        text += f" {clock.hour:02d}:{clock.minute:02d}"
    return text

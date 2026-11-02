"""How long a login lasts, chosen per login instead of one app-wide lifetime.

The login route stores the lifetime in the session (LIFETIME_KEY); the session
store then keeps that login, and sets its cookie, for that long. Logins without
the key use PERMANENT_SESSION_LIFETIME (1 hour). Every request renews the login
(SESSION_REFRESH_EACH_REQUEST), so a lifetime counts from the last use.
"""
from datetime import datetime, timedelta, timezone

LIFETIME_KEY = "_lifetime_s"


def login_lifetime_seconds(role, in_app, config):
    """Seconds this login should last, or None for the default (PERMANENT_SESSION_LIFETIME)."""
    if role == "admin":
        return int(config["ADMIN_SESSION_DAYS"]) * 86400
    if role == "teacher":
        return int(config["TEACHER_SESSION_DAYS"]) * 86400
    if role == "student" and in_app:
        return int(config["STUDENT_APP_SESSION_DAYS"]) * 86400
    return None


class _PerLoginLifetime:
    """Mixed into the Flask-Session interface: storage time and cookie expiry follow LIFETIME_KEY."""

    def _lifetime(self, app, session):
        seconds = session.get(LIFETIME_KEY)
        return timedelta(seconds=seconds) if seconds else app.permanent_session_lifetime

    def _upsert_session(self, session_lifetime, session, store_id):
        seconds = session.get(LIFETIME_KEY)
        if seconds:
            session_lifetime = timedelta(seconds=seconds)
        return super()._upsert_session(session_lifetime, session, store_id)

    def get_expiration_time(self, app, session):
        if session.permanent:
            return datetime.now(timezone.utc) + self._lifetime(app, session)
        return None


def enable(app):
    """Give the app's Flask-Session interface per-login lifetimes (call after Session(app))."""
    interface = app.session_interface
    base = type(interface)
    if not issubclass(base, _PerLoginLifetime):
        # Same object and settings, one more class in its MRO.
        interface.__class__ = type("PerLogin" + base.__name__, (_PerLoginLifetime, base), {})

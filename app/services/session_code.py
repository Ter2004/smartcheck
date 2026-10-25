"""Room code shown on the teacher's screen: the fallback when a phone cannot do BLE or LINE.

CHECKIN_SESSION_CODE=1 turns it on. The teacher opens the code screen of an
open session (projector or laptop); students type the 6 digits or scan the QR,
which opens the check-in page with the code filled in. The face scan still runs.

    code = 6 digits from HMAC-SHA256(PROXIMITY_RECEIPT_SECRET,
                                     "smartcheck-session-code-v1:" + session_id + ":" + step)
    step = unix time // STEP_SECONDS

A code belongs to one session and lasts one step plus the previous one (a code
read just before it changes still works). Limit, by design: a code read off the
screen can be forwarded to someone outside the room within that minute, which
is weaker than BLE or LINE. The teacher shows it only when needed, and the
screen says so.
"""
import hashlib
import hmac
import re
import time

STEP_SECONDS = 30
CONTEXT = "smartcheck-session-code-v1"


def _code(secret, session_id, step):
    msg = f"{CONTEXT}:{session_id}:{step}".encode()
    digest = hmac.new(secret.encode(), msg, hashlib.sha256).digest()
    offset = digest[-1] & 0x0F
    value = int.from_bytes(digest[offset:offset + 4], "big") & 0x7FFFFFFF
    return f"{value % 1_000_000:06d}"


def current(secret, session_id, now=None):
    """(code, seconds until it changes)."""
    now = time.time() if now is None else now
    step = int(now // STEP_SECONDS)
    return _code(secret, session_id, step), int(STEP_SECONDS - (now % STEP_SECONDS))


def verify(secret, session_id, code, now=None):
    """True for the current or the previous step's code of this session."""
    if not secret or not session_id or not isinstance(code, str) or not re.fullmatch(r"[0-9]{6}", code):
        return False
    now = time.time() if now is None else now
    step = int(now // STEP_SECONDS)
    return any(hmac.compare_digest(code, _code(secret, session_id, s)) for s in (step, step - 1))


def qr_svg(url, scale=8):
    """The URL as an inline SVG QR code (no external script, CSP unchanged)."""
    import io
    import segno
    out = io.BytesIO()
    segno.make(url, error="m").save(out, kind="svg", scale=scale, border=2, xmldecl=False, svgns=True,
                                    dark="#0f172a", light="#ffffff")
    return out.getvalue().decode()

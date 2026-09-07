"""ESP32-compatible ASCII-key SHA256 TOTP; no Flask or hardware dependency."""
import hashlib
import hmac
import os
import re
import time


class TOTPConfigurationError(RuntimeError):
    pass


def load_secret():
    """Call at startup, including simulator startup. Never normalize key bytes."""
    secret = os.environ.get("ESP32_TOTP_SECRET")
    if not secret:
        raise TOTPConfigurationError("ESP32_TOTP_SECRET is required; startup aborted")
    try:
        return secret.encode("ascii")
    except UnicodeEncodeError:
        raise TOTPConfigurationError("ESP32_TOTP_SECRET must be plain ASCII") from None


def _code(secret: bytes, counter: int) -> str:
    if not isinstance(secret, bytes) or not secret or not secret.isascii():
        raise TOTPConfigurationError("TOTP requires a nonempty ASCII secret")
    digest = hmac.new(secret, counter.to_bytes(8, "big"), hashlib.sha256).digest()
    offset = digest[31] & 0x0F
    value = int.from_bytes(digest[offset:offset + 4], "big") & 0x7FFFFFFF
    return f"{value % 1_000_000:06d}"


def generate_code(secret: bytes, unix_time=None) -> str:
    timestamp = time.time() if unix_time is None else unix_time
    return _code(secret, int(timestamp // 30))


def verify_code(code, secret: bytes, unix_time=None) -> bool:
    timestamp = time.time() if unix_time is None else unix_time
    counter = int(timestamp // 30)
    # Validate configuration even when the submitted code is malformed.
    candidates = [_code(secret, n) for n in (counter - 1, counter, counter + 1) if n >= 0]
    if not isinstance(code, str) or re.fullmatch(r"[0-9]{6}", code) is None:
        return False
    return any(hmac.compare_digest(code, expected) for expected in candidates)

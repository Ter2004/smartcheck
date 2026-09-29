"""Server-issued BLE challenge for the room's ESP32 (proximity proof).

The browser connects to the room board, asks the server for a nonce, writes
the 16 nonce bytes to the board and reads back
HMAC-SHA256(room key, CONTEXT + nonce). Only a board holding the room key
(beacons.ble_secret, flashed as firmware secret.h) can answer, so knowing the
static room code no longer proves presence.

Limit: someone in the room can still relay a live nonce to the board for a
student elsewhere within TTL_SECONDS.
"""
import hashlib
import hmac
import re
import secrets
import time

TTL_SECONDS = 30
CONTEXT = b"smartcheck-ble-v1"
NONCE_BYTES = 16
_HEX64 = re.compile(r"[0-9a-f]{64}")


def new_challenge(session_id, beacon_id, now=None):
    return {"nonce": secrets.token_hex(NONCE_BYTES), "session_id": session_id,
            "beacon_id": beacon_id, "issued_at": time.time() if now is None else now}


def expected_response(secret_hex, nonce_hex):
    return hmac.new(bytes.fromhex(secret_hex), CONTEXT + bytes.fromhex(nonce_hex),
                    hashlib.sha256).hexdigest()


def verify(challenge, session_id, beacon_id, secret_hex, response, now=None):
    """None when the room board answered this challenge, else a reject reason."""
    now = time.time() if now is None else now
    if not challenge:
        return "ble_challenge_missing"
    if challenge.get("session_id") != session_id or challenge.get("beacon_id") != beacon_id:
        return "ble_challenge_mismatch"
    if not 0 <= now - challenge["issued_at"] <= TTL_SECONDS:
        return "ble_challenge_expired"
    if not isinstance(response, str) or not _HEX64.fullmatch(response.lower()):
        return "ble_response_invalid"
    if not hmac.compare_digest(expected_response(secret_hex, challenge["nonce"]), response.lower()):
        return "ble_response_invalid"
    return None

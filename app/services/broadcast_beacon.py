"""Broadcast room board (firmware/smartcheck_broadcast): signed rolling counter.

The board accepts no connections; every phone in the room reads the same
advertisement at once. Manufacturer data payload (23 bytes, after company ID 0xFFFF):

    "SC" | version 0x01 | board id (8 bytes, ble_room_code padded with 0x00)
    | counter (4 bytes, big-endian)
    | HMAC-SHA256(room key, CONTEXT || board id || counter)[:8]

The board has no clock: its counter steps every 5 s and only moves forward, even
across reboots. The server records when each counter was first seen for a board.
A counter is accepted only within FRESH_SECONDS of that first sighting and only if
it is not more than MAX_BEHIND steps older than the newest counter seen for the
board, so a recording replayed later is refused.

Known limit: a counter no student has submitted yet counts as "first seen" when it
arrives. A code recorded in the room and sent out before anyone in the room uses
it still works for up to FRESH_SECONDS. The face check still applies.
"""
import hashlib
import hmac
import struct
from datetime import datetime, timedelta, timezone

CONTEXT = b"smartcheck-bcast-v1"   # must match the firmware
VERSION = 0x01
BOARD_ID_LEN = 8
MAC_LEN = 8
PAYLOAD_LEN = 2 + 1 + BOARD_ID_LEN + 4 + MAC_LEN   # 23
FRESH_SECONDS = 30
MAX_BEHIND = 6                     # counters (5 s each) older than the newest one seen
TABLE = "broadcast_sightings"


def _board_id(room_code):
    raw = room_code.encode("ascii")
    if not 1 <= len(raw) <= BOARD_ID_LEN:
        raise ValueError("room code must be 1-8 ASCII characters")
    return raw.ljust(BOARD_ID_LEN, b"\0")


def payload(secret_hex, room_code, counter):
    """The 23 bytes the board advertises for `counter` (also used by tests)."""
    head = b"SC" + bytes([VERSION]) + _board_id(room_code) + struct.pack(">I", counter & 0xFFFFFFFF)
    mac = hmac.new(bytes.fromhex(secret_hex), CONTEXT + head[3:], hashlib.sha256).digest()[:MAC_LEN]
    return head + mac


def parse(value):
    """(room_code, counter, raw) from a hex payload, or None if it is not a broadcast payload."""
    if not isinstance(value, str) or len(value) != 2 * PAYLOAD_LEN:
        return None
    try:
        raw = bytes.fromhex(value)
    except ValueError:
        return None
    if raw[:2] != b"SC" or raw[2] != VERSION:
        return None
    try:
        room = raw[3:3 + BOARD_ID_LEN].rstrip(b"\0").decode("ascii")
    except UnicodeDecodeError:
        return None
    if not room:
        return None
    counter = struct.unpack(">I", raw[3 + BOARD_ID_LEN:3 + BOARD_ID_LEN + 4])[0]
    return room, counter, raw


def verify(secret_hex, room_code, value):
    """The counter if `value` is this room's board signing for `room_code`, else None."""
    parsed = parse(value)
    if not parsed or not secret_hex or not isinstance(room_code, str):
        return None
    room, counter, raw = parsed
    if room != room_code.strip():
        return None
    try:
        expected = payload(secret_hex, room, counter)
    except ValueError:
        return None
    return counter if hmac.compare_digest(expected, raw) else None


def fresh(client, beacon_id, counter, now=None):
    """True if this board's counter is recent (records first sightings)."""
    now = now or datetime.now(timezone.utc)
    newest = (client.table(TABLE).select("counter").eq("beacon_id", beacon_id)
              .order("counter", desc=True).limit(1).execute())
    rows = (newest.data if newest else None) or []
    if rows and counter < int(rows[0]["counter"]) - MAX_BEHIND:
        return False
    client.table(TABLE).upsert(
        {"beacon_id": beacon_id, "counter": counter, "first_seen_at": now.isoformat()},
        on_conflict="beacon_id,counter", ignore_duplicates=True).execute()
    row = (client.table(TABLE).select("first_seen_at")
           .eq("beacon_id", beacon_id).eq("counter", counter).maybe_single().execute())
    first = (row.data if row else None) or {}
    if not first.get("first_seen_at"):
        return False
    seen = datetime.fromisoformat(first["first_seen_at"].replace("Z", "+00:00"))
    return timedelta(seconds=-2) <= now - seen <= timedelta(seconds=FRESH_SECONDS)

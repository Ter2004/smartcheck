"""LINE Beacon proximity: webhook signature, signed device message, link codes, replies.

The room board advertises a LINE Simple Beacon (service 0xFE6F, frame 0x02,
HWID, TxPower, device message). The LINE app detects it and LINE's server posts
a "beacon" webhook event with the LINE userId, the HWID and the device message.

A Simple Beacon cannot stop impersonation by itself (LINE's spec says so), so
the board puts a signed counter in the 12-byte device message:

    dm = counter (4 bytes, big-endian) || HMAC-SHA256(room key, CONTEXT || counter)[:8]

The room key is the board's existing BLE key (beacons.ble_secret). The board has
no clock, so the counter only moves forward (one step per 10 s from a random
start at boot); the server records when each counter was first seen for a board
and accepts it only within FRESH_SECONDS of that first sighting. A recording
replayed later is refused; a live relay is not (as with any proximity signal),
which is why check-in still requires the face scan.
"""
import base64
import hashlib
import hmac
import json
import secrets
import struct
import urllib.request
from datetime import datetime, timedelta, timezone

CONTEXT = b"smartcheck-line-v1"
DM_LEN = 12
FRESH_SECONDS = 120            # a counter first seen longer ago than this is a replay
LINK_CODE_TTL_MIN = 10
RECEIPT_TTL_SECONDS = 300      # time from the LINE message to the finished face scan
REPLY_URL = "https://api.line.me/v2/bot/message/reply"


# ─── Webhook authenticity ────────────────────────────────────────────────────

def valid_signature(channel_secret, body, signature):
    """X-Line-Signature = base64(HMAC-SHA256(channel secret, raw body))."""
    if not channel_secret or not signature or not isinstance(body, (bytes, bytearray)):
        return False
    expected = base64.b64encode(hmac.new(channel_secret.encode(), body, hashlib.sha256).digest()).decode()
    return hmac.compare_digest(expected, signature)


# ─── Device message ──────────────────────────────────────────────────────────

def device_message(secret_hex, counter):
    """The 12 bytes the board advertises for `counter` (also used by tests)."""
    c = struct.pack(">I", counter & 0xFFFFFFFF)
    return c + hmac.new(bytes.fromhex(secret_hex), CONTEXT + c, hashlib.sha256).digest()[:8]


def parse_dm(value):
    """The device message from the webhook (hex string) as bytes, or None."""
    if not isinstance(value, str):
        return None
    try:
        raw = bytes.fromhex(value.strip())
    except ValueError:
        return None
    return raw if len(raw) == DM_LEN else None


def verify_dm(secret_hex, value):
    """The counter if the device message carries a valid signature for this room key, else None."""
    raw = parse_dm(value)
    if raw is None or not secret_hex:
        return None
    counter = struct.unpack(">I", raw[:4])[0]
    if not hmac.compare_digest(device_message(secret_hex, counter), raw):
        return None
    return counter


def fresh(client, beacon_id, counter, now=None):
    """True if this board's counter was first seen within FRESH_SECONDS (records first sightings)."""
    now = now or datetime.now(timezone.utc)
    client.table("line_beacon_sightings").upsert(
        {"beacon_id": beacon_id, "counter": counter, "first_seen_at": now.isoformat()},
        on_conflict="beacon_id,counter", ignore_duplicates=True).execute()
    row = (client.table("line_beacon_sightings").select("first_seen_at")
           .eq("beacon_id", beacon_id).eq("counter", counter).maybe_single().execute())
    first = (row.data if row else None) or {}
    if not first.get("first_seen_at"):
        return False
    seen = datetime.fromisoformat(first["first_seen_at"].replace("Z", "+00:00"))
    return timedelta(0) <= now - seen <= timedelta(seconds=FRESH_SECONDS) or abs((now - seen).total_seconds()) < 2


# ─── Linking a LINE account to a SmartCheck student ──────────────────────────

def _hash_code(code):
    return hashlib.sha256(f"smartcheck-line-link:{code}".encode()).hexdigest()


def new_link_code(client, user_id, now=None):
    """A 6-digit code the student sends to the LINE account ("LINK 123456"); replaces any older one."""
    now = now or datetime.now(timezone.utc)
    code = f"{secrets.randbelow(10 ** 6):06d}"
    client.table("line_link_codes").upsert({
        "user_id": user_id, "code_hash": _hash_code(code),
        "expires_at": (now + timedelta(minutes=LINK_CODE_TTL_MIN)).isoformat(),
    }, on_conflict="user_id").execute()
    return code


def redeem_link_code(client, line_user_id, text, now=None):
    """Link the sender to the student whose code this is. Returns "linked", "taken", "invalid"."""
    now = now or datetime.now(timezone.utc)
    words = (text or "").strip().upper().replace("LINK", "").split()
    code = words[0] if words else ""
    if not (len(code) == 6 and code.isdigit()):
        return "invalid"
    row = (client.table("line_link_codes").select("user_id, expires_at")
           .eq("code_hash", _hash_code(code)).maybe_single().execute())
    pending = (row.data if row else None) or {}
    if not pending or datetime.fromisoformat(pending["expires_at"].replace("Z", "+00:00")) < now:
        return "invalid"
    owner = (client.table("users").select("id").eq("line_user_id", line_user_id).maybe_single().execute())
    if owner and owner.data and owner.data.get("id") != pending["user_id"]:
        return "taken"      # this LINE account already belongs to another student
    client.table("users").update({"line_user_id": line_user_id}).eq("id", pending["user_id"]).execute()
    client.table("line_link_codes").delete().eq("user_id", pending["user_id"]).execute()
    return "linked"


# ─── Replies ─────────────────────────────────────────────────────────────────

def reply(access_token, reply_token, messages, timeout=5):
    """POST the reply (reply messages do not count against the message quota)."""
    body = json.dumps({"replyToken": reply_token, "messages": messages}).encode()
    req = urllib.request.Request(REPLY_URL, data=body, method="POST", headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {access_token}"})
    with urllib.request.urlopen(req, timeout=timeout) as res:   # noqa: S310 (fixed https URL)
        return res.status


def text(message):
    return {"type": "text", "text": message[:5000]}


def checkin_button(room_name, url):
    """A message with one button that opens the check-in page in the phone's browser."""
    return {"type": "template", "altText": f"อยู่ในห้อง {room_name} — แตะเพื่อสแกนหน้าเช็คชื่อ",
            "template": {"type": "buttons", "text": f"อยู่ในห้อง {room_name}"[:160],
                         "actions": [{"type": "uri", "label": "สแกนหน้าเช็คชื่อ", "uri": url}]}}

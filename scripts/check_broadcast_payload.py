"""Check a code copied from nRF Connect against the room key (docs/broadcast_ble.md).

    python scripts/check_broadcast_payload.py 0x534301544553542D...

Reads ROOM_SECRET_HEX from firmware/smartcheck_broadcast/secret.h and never prints it.
Accepts the manufacturer data with or without the leading company ID (FFFF).
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services import broadcast_beacon  # noqa: E402

SECRET_H = Path(__file__).resolve().parents[1] / "firmware" / "smartcheck_broadcast" / "secret.h"


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    value = re.sub(r"[^0-9a-fA-F]", "", argv[1].lower().removeprefix("0x"))
    if len(value) == 2 * broadcast_beacon.PAYLOAD_LEN + 4 and value.startswith("ffff"):
        value = value[4:]
    parsed = broadcast_beacon.parse(value)
    if not parsed:
        print(f"NOT A BOARD CODE: expected {broadcast_beacon.PAYLOAD_LEN} bytes starting with 53 43 01")
        return 1
    match = re.search(r'ROOM_SECRET_HEX\[\]\s*=\s*"([0-9a-fA-F]{64})"', SECRET_H.read_text())
    if not match:
        print(f"KEY MISSING: {SECRET_H}")
        return 1
    room, counter, _ = parsed
    ok = broadcast_beacon.verify(match.group(1), room, value) is not None
    print(f"room={room} counter={counter} signature={'OK' if ok else 'BAD'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))

"""Standalone parity check / opt-in console simulator; never starts Flask."""
import argparse
import importlib.util
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("esp32_totp", ROOT / "app/services/esp32_totp.py")
totp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(totp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", type=int, help="Exact Unix second printed by the ESP32")
    parser.add_argument("--expect", help="Six-digit serial code to compare (preserve leading zeroes)")
    parser.add_argument("--simulate", action="store_true", help="Print a fresh code and countdown each second")
    args = parser.parse_args()
    if args.simulate and (args.timestamp is not None or args.expect is not None):
        parser.error("--simulate cannot be combined with --timestamp/--expect")
    if args.expect is not None and args.timestamp is None:
        parser.error("--expect requires --timestamp")
    # Optional dotenv support; environment variables take precedence.
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv(ROOT / ".env")
    if args.simulate and os.getenv("ESP32_TOTP_SIMULATOR", "false").lower() != "true":
        parser.error("Simulator disabled: set ESP32_TOTP_SIMULATOR=true explicitly")
    secret = totp.load_secret()
    while True:
        now = args.timestamp if args.timestamp is not None else int(time.time())
        code = totp.generate_code(secret, now)
        print(f"unix={now} counter={now // 30} code={code} remaining={30 - now % 30}", flush=True)
        if args.expect is not None:
            matches = code == args.expect
            print("MATCH" if matches else "MISMATCH: check timestamp and identical ASCII secret")
            return 0 if matches else 1
        if not args.simulate:
            return 0
        time.sleep(1)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except totp.TOTPConfigurationError as error:
        raise SystemExit(str(error))
    except KeyboardInterrupt:
        pass

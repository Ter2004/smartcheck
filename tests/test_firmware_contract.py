"""The BLE challenge is defined in three places: firmware, browser and server.

Nothing runs the .ino here, so this pins the values they must share; the
firmware checks the same HMAC vector itself at boot (Serial: self-test PASS).
"""
import re
import unittest
from pathlib import Path

from app.services import ble_challenge

ROOT = Path(__file__).resolve().parents[1]
SKETCH = (ROOT / "firmware/smartcheck_ble/smartcheck_ble.ino").read_text(encoding="utf-8")
SCANNER = (ROOT / "app/static/js/ble_room_scanner.js").read_text(encoding="utf-8")
VECTOR = "975fe6e237c23b0e5b1263789a3646161a57763c3d512ded8c7e797f5029f48a"


class FirmwareContractTests(unittest.TestCase):
    def test_challenge_uuid_matches_browser(self):
        firmware = re.search(r'CHALLENGE_UUID\[\] = "([0-9a-f-]+)"', SKETCH).group(1)
        browser = re.search(r"CHALLENGE_UUID\s*=\s*'([0-9a-f-]+)'", SCANNER).group(1)
        self.assertEqual(firmware, browser)

    def test_hmac_context_matches_server(self):
        firmware = re.search(r'HMAC_CONTEXT\[\] = "([^"]+)"', SKETCH).group(1)
        self.assertEqual(firmware.encode(), ble_challenge.CONTEXT)

    def test_boot_self_test_vector_matches_server(self):
        body = re.search(r"expected\[MAC_LEN\] = \{(.*?)\};", SKETCH, re.S).group(1)
        firmware = "".join(f"{int(b, 16):02x}" for b in re.findall(r"0x([0-9a-f]{2})", body))
        self.assertEqual(firmware, VECTOR)
        self.assertEqual(ble_challenge.expected_response(bytes(range(32)).hex(),
                                                         bytes(range(16)).hex()), VECTOR)

    def test_room_key_file_is_git_ignored(self):
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("firmware/smartcheck_ble/secret.h", ignored)

    def test_key_is_never_printed(self):
        for line in SKETCH.splitlines():
            if "Serial.print" in line:
                code = re.sub(r'"[^"]*"', '""', line)   # names inside messages are fine
                self.assertNotIn("roomKey", code)
                self.assertNotIn("ROOM_SECRET_HEX", code)


if __name__ == "__main__":
    unittest.main()

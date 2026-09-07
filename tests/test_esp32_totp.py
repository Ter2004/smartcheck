import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "esp32_totp", Path(__file__).resolve().parents[1] / "app/services/esp32_totp.py")
totp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(totp)


class TOTPTests(unittest.TestCase):
    # Public RFC 6238 SHA256 test key, never an application credential.
    secret = b"12345678901234567890123456789012"

    def test_sha256_reference_vectors_reduced_to_six_digits(self):
        for timestamp, expected in [(59, "119246"), (1111111109, "084774"),
                                    (1111111111, "062674"), (1234567890, "819424"),
                                    (2000000000, "698825"), (20000000000, "737706")]:
            with self.subTest(timestamp=timestamp):
                self.assertEqual(totp.generate_code(self.secret, timestamp), expected)

    def test_current_and_adjacent_windows(self):
        for delta in [-30, 0, 30]:
            self.assertTrue(totp.verify_code(totp.generate_code(self.secret, 1234567890 + delta),
                                             self.secret, 1234567890))

    def test_expired_and_two_windows_future(self):
        for delta in [-300, -60, 60]:
            self.assertFalse(totp.verify_code(totp.generate_code(self.secret, 1234567890 + delta),
                                              self.secret, 1234567890))

    def test_wrong_and_malformed(self):
        for code in ["000000", "12345", "1234567", "１２３４５６", 819424, None]:
            self.assertFalse(totp.verify_code(code, self.secret, 1234567890))

    def test_expiry_boundary(self):
        code = totp.generate_code(self.secret, 120)
        self.assertTrue(totp.verify_code(code, self.secret, 179))
        self.assertFalse(totp.verify_code(code, self.secret, 180))

    def test_startup_secret_validation(self):
        for env in [{}, {"ESP32_TOTP_SECRET": ""}, {"ESP32_TOTP_SECRET": "é"}]:
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(totp.TOTPConfigurationError):
                    totp.load_secret()

    def test_secret_bytes_preserved(self):
        with patch.dict(os.environ, {"ESP32_TOTP_SECRET": " literal ASCII "}):
            self.assertEqual(totp.load_secret(), b" literal ASCII ")

    def test_invalid_configuration_is_not_wrong_code(self):
        with self.assertRaises(totp.TOTPConfigurationError):
            totp.verify_code("000000", b"", 120)


if __name__ == "__main__":
    unittest.main()

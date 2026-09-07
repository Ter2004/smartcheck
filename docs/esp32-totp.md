# ESP32 parity gate

**Flow update (2026-09-07):** proximity now precedes camera capture in both
BLE and TOTP modes. The server issues a 90-second signed proximity receipt using
the independent `PROXIMITY_RECEIPT_SECRET`; final submission requires it. Older
descriptions below of code entry after capture are superseded. See
`docs/review/99-summary.md`, “Proximity preflight and receipt”, for timing,
validation and accepted limitations.

Status: integrated after user-confirmed hardware parity in two windows:
`1788758828 / 59625294 / 291694` and `1788758850 / 59625295 / 384478`, both MATCH.
The device rollover at 1788758850 reset remaining to 30.
Flask startup validates the ASCII secret before database initialization.

Set `ESP32_TOTP_SECRET` in `.env` beside `EMBEDDING_INTEGRITY_SALT` to the exact
firmware ASCII string. Do not Base32-decode, trim, or otherwise normalize it.
A mismatch makes every code fail and looks like a logic bug. Never share the
secret in serial logs or chat. The blank entry is intentional; no key is invented.

From the inner `smartcheck_project` directory:

```powershell
python scripts/esp32_totp.py --timestamp 1234567890 --expect 819424
```

Those example values are a public test vector, valid only with the public RFC
SHA256 test key used in the tests, not your board key. For the real comparison,
print `unix=<timestamp> code=<six digits> remaining=<seconds>` in one firmware
serial line using the SAME captured `time(nullptr)` value for all three fields.
Replace the command's timestamp and expected code with that line's values.
The script prints its counter, code, countdown, and MATCH/MISMATCH, returning
exit status 0/1. A recorded timestamp works later; running both commands live
is unnecessary. Compare several lines spanning a 30-second rollover. A line
with only `code=402711 remaining=18` cannot identify an absolute counter.

The implementation uses HMAC-SHA256 of `(unix_time // 30).to_bytes(8, 'big')`,
offset `digest[31] & 15`, four bytes interpreted big-endian, masked with
`0x7fffffff`, modulo one million, padded to six digits.

Verification accepts server counter -1, current, and +1. This guarantees clock
drift of up to 30 seconds in either direction. At server phase r (0..29), accepted
device timestamps range from server time minus (30+r) to server time plus
(59-r). Thus 31..59 seconds may pass depending on phase; 60 seconds is outside
the accepted counter range. Network/typing delay also consumes that allowance.
A code's counter is accepted over 90 seconds total, but this is not ±90 seconds
of drift. Six-digit collisions are possible; acceptance concerns matching codes
in the allowed counters, not proof of which counter originally produced a code.

## Console simulator

Set `ESP32_TOTP_SIMULATOR=true` in `.env`, then run:

```powershell
python scripts/esp32_totp.py --simulate
```

It shows a fresh code/countdown each second using the same secret and algorithm;
Ctrl+C stops it. Missing/empty/non-ASCII secrets abort the tool at startup.
It requires no hardware and offers no authentication bypass. Keep this console
on the instructor's machine for a demo; do not expose a public code endpoint.

## Integrated student flow

After camera capture, the student enters the displayed six-digit classroom code
and clicks “ยืนยันเช็คชื่อ”. The API validates `room_code` before expensive face
work or database writes. Face and liveness checks remain required. Leading zeros
are preserved. On a room-code failure, “ลองใหม่” returns to the code input and
retains the captured frames for retry.

| Failure | HTTP | Student message |
|---|---|---|
| Missing code | 400 | กรุณากรอกรหัสห้อง 6 หลักจากจอในห้องเรียน |
| Wrong, malformed, stale | 400 | รหัสห้องไม่ถูกต้องหรือหมดอายุ กรุณาดูรหัสปัจจุบันจากจอในห้องเรียนแล้วกรอกใหม่ |
| Verifier exception | 503 | ระบบตรวจสอบรหัสห้องขัดข้องชั่วคราว กรุณาลองใหม่อีกครั้ง หรือแจ้งอาจารย์หากยังพบปัญหา |

The browser also requires six numeric digits before submitting the form.
Missing/empty configuration aborts Flask boot with `ESP32_TOTP_SECRET is required;
startup aborted`, rather than serving misleading invalid-code responses.

The verifier does not contact the board and cannot detect board connectivity.
A powered board with a valid clock can keep generating codes without WiFi.
If the board dies, the instructor starts the opt-in console simulator and displays
its codes; students follow the same flow. Without a working display/simulator,
students cannot obtain fresh codes and must ask the instructor to restore it.
There will be no automatic bypass or fabricated 503 based on an invalid code.

Tests: `python -m unittest discover -s tests -p '*totp.py' -v` and
`node scripts/test_checkin_room_code.js`.
The integration suite runs the actual simulator subprocess, reads its console
code, submits through the authenticated/CSRF-protected Flask route with the real
TOTP verifier, and checks attendance insertion. Database and face inference are
test doubles; this is not a live camera/Supabase deployment test.

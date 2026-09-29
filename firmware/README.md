# BLE classroom peripheral

Open `smartcheck_ble/smartcheck_ble.ino` in Arduino IDE. Select **ESP32 Dev Module**,
Espressif ESP32 core **3.3.11**, and the board's COM port. Libraries: **Adafruit
SSD1306**, **Adafruit GFX Library**, **Adafruit BusIO**. Use the BLE library bundled
with the core, not a separately installed legacy ESP32 BLE library.

OLED: SSD1306 I2C address `0x3C`, SDA GPIO21, SCL GPIO22, common GND and appropriate
module power (normally 3.3 V). Sketch defaults to 128x64; change `OLED_HEIGHT` to 32
if the existing panel is 128x32. Serial Monitor: 115200 baud.

| Browser contract | Exact value |
|---|---|
| Name | `SmartCheck-TEST101` |
| Service UUID | `7c6b1000-9f3a-4b27-8d15-6e2a90c4f801` |
| Read characteristic UUID | `7c6b1001-9f3a-4b27-8d15-6e2a90c4f801` |
| Value | `TEST-101` (8 ASCII bytes, no NUL/newline) |
| Challenge characteristic UUID (write + read) | `7c6b1002-9f3a-4b27-8d15-6e2a90c4f801` |
| Challenge | write 16-byte nonce, read `HMAC-SHA256(key, "smartcheck-ble-v1" + nonce)` (32 bytes) |

The service UUID is in the primary advertisement; the name is in scan-response
data to fit legacy BLE's packet limits. No WiFi, NTP, credentials, or clock are
needed. This is a connectable GATT peripheral, not an iBeacon/RSSI broadcaster.

## Room key (required)

The room value is public: it only names the room. Presence is proven by the board
signing a one-time server nonce with the room key, which the server also holds
(`beacons.ble_secret`).

1. Admin page → **อุปกรณ์ห้องเรียน** → **สร้าง secret** for the room. The key is
   shown once.
2. Save it as `smartcheck_ble/secret.h` (format in `secret.h.example`). The file is
   git-ignored; never commit or share it. Without it the sketch does not compile.
3. Flash, then open Serial Monitor: `HMAC self-test PASS` and `key=OK` in the status
   lines. `KEY ERROR` means `secret.h` is not the 64-hex key from the admin page.
4. Generating a new key in the admin page invalidates the flashed one: reflash.

The key is never printed. Anyone who reads it (from `secret.h` or the board's
flash) can make another board answer for the room; keep the board where students
cannot take it.

## Panel sequence

- Boot: device name and `BLE starting...`.
- Waiting: `ADV: ON`, `Client: NONE`, `Read: waiting`.
- Connected: `ADV: OFF (busy)`, `Client: CONNECTED`.
- Characteristic read: `Read: TEST-101 OK`.
- Nonce signed: `Signed: N` counts answers; `KEY ERROR` / `HMAC ERROR` replace it
  when the key is missing/invalid or the boot self-test failed.
- Disconnect: advertising restarts; `Client: NONE`. Last-read confirmation stays
  visible until the next connection, so a quick connection is visible on the panel.
- Advertising failure: `ADV: ERROR`; reset and inspect Serial.
- OLED absent: Serial reports it; BLE still starts.

The sketch intentionally accepts one client at a time: advertising stops when
connected and restarts on disconnect. ESP32 can support advertising alongside
connections with a multi-client configuration; the OLED does not prohibit it.
OLED rendering runs at 4 Hz in `loop()`, never in BLE callbacks. Keep the browser
connection open to demonstrate the connected status, then disconnect to let the
next student use the board.

## Browser test (no attendance submission)

From the project directory:

```powershell
python -m http.server 8001 --bind 127.0.0.1 --directory firmware
```

Open `http://localhost:8001/ble-test.html` in Chrome/Edge on Windows with Bluetooth
enabled. Click **Connect and read room**, select **SmartCheck-TEST101**, verify
`GATT read: TEST-101`, click **Sign random nonce** and check a 32-byte answer
(`Signed:` on the OLED goes up), then click **Disconnect**. The page cannot check
the answer against the key; a real check-in does. Web Bluetooth
requires a secure context (HTTPS or localhost) and a user gesture for device
selection. No experimental advertisement/RSSI APIs are used.

## Check-in flow and limits

`app/static/js/ble_room_scanner.js` connects, reads the room, asks the server
(`/api/checkin/ble/challenge`) for a nonce, has the board sign it and sends the
answer to `/api/checkin/proximity`, which verifies it before issuing the 90 s
proximity receipt. A room without a key answers 503; old firmware without the
challenge characteristic is reported as needing an update.

This proves a live connection to a board holding the room key within 30 s of the
nonce. It does not measure room boundaries, and someone in the room can still
relay a live nonce for a student elsewhere.

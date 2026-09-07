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

The service UUID is in the primary advertisement; the name is in scan-response
data to fit legacy BLE's packet limits. No WiFi, NTP, credentials, or clock are
needed. This is a connectable GATT peripheral, not an iBeacon/RSSI broadcaster.

## Panel sequence

- Boot: device name and `BLE starting...`.
- Waiting: `ADV: ON`, `Client: NONE`, `Read: waiting`.
- Connected: `ADV: OFF (busy)`, `Client: CONNECTED`.
- Characteristic read: `Read: TEST-101 OK`.
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
`GATT read: TEST-101`, inspect the OLED, then click **Disconnect**. Web Bluetooth
requires a secure context (HTTPS or localhost) and a user gesture for device
selection. No experimental advertisement/RSSI APIs are used.

## Integration boundary and fallback

The current app scanner still filters `battery_service` and uses an RSSI-oriented
flow; it is not wired to this new service. This firmware task does not change
check-in or remove its TOTP requirement. Browser/server BLE integration remains
separate work. The test page includes the exact requestDevice/connect/read flow.

An honest browser reading GATT demonstrates a radio connection to a peripheral
offering this service. A fixed name/UUID/room value does NOT authenticate the
physical board to the server, measure room boundaries, or prevent a forged HTTP
payload. The server can check that the room matches the session, but that alone
is not cryptographic proximity proof. A device-signed challenge would be needed
for stronger server verification.

All repository TOTP code remains intact. No original hardware sketch was present
in this workspace, so save your working TOTP sketch before flashing BLE: flashing
replaces the firmware on this board. The existing Python TOTP simulator remains
available for the unchanged check-in flow.

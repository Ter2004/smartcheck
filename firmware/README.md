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
- Waiting: `ADV: ON`, `Clients: 0/3`, `Read: waiting`.
- One or two connected: advertising restarts, `Clients: 1/3` or `2/3`.
- Three connected: `ADV: OFF (full)`, `Clients: 3/3`.
- Characteristic read: `Read: TEST-101 OK`.
- Disconnect: the count decreases; advertising restarts if it was off. Last-read confirmation stays
  visible until the next connection, so a quick connection is visible on the panel.
- Advertising failure: `ADV: ERROR`; reset and inspect Serial.
- OLED absent: Serial reports it; BLE still starts.

The sketch accepts up to three clients concurrently using the existing
BLE/Bluedroid library. The installed ESP32 core 3.3.11 configures
`CONFIG_BTDM_CTRL_BLE_MAX_CONN=3`; a compile-time check rejects a sketch limit
above the core's capacity. Advertising resumes after each connection while
slots remain. A fourth client must wait for a disconnect; there is no queue.
Connection IDs are tracked individually, so disconnecting one client does not
mark the other clients disconnected. OLED rendering runs at 4 Hz in `loop()`,
never in BLE callbacks. The read indicator is board-wide, not per student.

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

### Three-device acceptance check

Use three separate Bluetooth-capable devices with the test page on a secure
origin (localhost on each computer, or HTTPS for phones; a LAN HTTP address is
not sufficient). The test page deliberately holds each connection open.

1. Connect device A and read `TEST-101`. Keep it connected: expect `Clients: 1/3`
   and advertising back `ON`.
2. Connect B while A stays connected, then C while both stay connected. Each
   must read `TEST-101`; expect `Clients: 3/3`, `ADV: OFF (full)`.
3. Disconnect B: expect `Clients: 2/3`, `ADV: ON`, with A and C still connected.
4. Reconnect B and read again; disconnect all three and expect `Clients: 0/3`,
   `ADV: ON`. Repeat several rounds and check Serial for advertising errors.
5. Run the real student BLE/face check-in flow on all three devices. The app
   releases BLE after reading the room, so face processing can overlap.

Compilation does not prove concurrent radio operation. Complete these checks
on the actual board before using this firmware in the demo.

## Integration boundary and fallback

The app's `ble_room_scanner.js` reads this room characteristic and disconnects
after the read. This firmware upgrade preserves that browser contract. The
standalone test page holds its connection open to exercise concurrent clients.

An honest browser reading GATT demonstrates a radio connection to a peripheral
offering this service. A fixed name/UUID/room value does NOT authenticate the
physical board to the server, measure room boundaries, or prevent a forged HTTP
payload. The server can check that the room matches the session, but that alone
is not cryptographic proximity proof. A device-signed challenge would be needed
for stronger server verification.

Save the working single-client sketch before flashing this upgrade: flashing
replaces the firmware on this board. TOTP code and the application's proximity
configuration are unaffected by this firmware change.

# LINE Beacon proximity (trial kit)

With this kit, students check in by walking into the room. They do not press a
Bluetooth button.

1. The LINE app on their phone detects the room board.
2. LINE's server tells SmartCheck.
3. The student receives a LINE message with one button, "สแกนหน้าเช็คชื่อ".
4. The button opens the check-in page with the room already confirmed. The face
   scan (face stack v2, passive) does the rest.

Everything is off by default (`LINE_BEACON_ENABLED=0`). The BLE-button and TOTP
flows keep working unchanged.

## What LINE requires on the phone

From LINE's developer docs, LINE Beacon partner page:
<https://developers.line.biz/en/docs/partner-docs/line-beacon/>

**Both systems:**
- Bluetooth is on.
- LINE → Settings → Privacy → Provide usage data → LINE Beacon is turned on.
- The student has added the SmartCheck account as a friend.

**iPhone:**
- While LINE is open, nothing else is needed.
- For LINE to work in the background (running but not on screen), these are also needed:
  - Location Services on;
  - LINE's location set to "Always";
  - Precise Location on.

**Android:**
- LINE must be **open on screen**. LINE states that background receiving is
  not available on Android.
- Location must be on, with LINE's location set to "While using the app",
  Precise location on, and Nearby devices allowed.

**If LINE was swiped away (not running):** LINE says the behaviour is undefined.

## Setup

1. **LINE account and channel.**
   - Create a LINE Official Account (free plan) and enable the Messaging API.
   - In LINE Developers, copy the channel secret and a channel access token
     into `.env` as `LINE_CHANNEL_SECRET` and `LINE_CHANNEL_ACCESS_TOKEN`.
   - Set the webhook URL to `<PUBLIC_BASE_URL>/line/webhook` and turn
     "Use webhook" on.
2. **HWID.** In LINE Official Account Manager, issue a LINE Simple Beacon HWID
   for each room board. It is 10 hex characters.
3. **Database.** Apply `database/migrations/20261020_line_beacon.sql` in the
   Supabase SQL editor and run its verification query.
4. **Server settings.**
   - Set `PUBLIC_BASE_URL` to a **fixed https** URL (your domain plus a named
     tunnel, or a cloud deploy). A quick-tunnel URL changes on every restart
     and breaks both the webhook and the links already sent.
   - Set `LINE_BEACON_ENABLED=1` and restart. The app refuses to start if any
     of the three LINE settings is missing.
5. **Admin.** On the อุปกรณ์ห้องเรียน page, edit the room and fill in "LINE
   Beacon HWID". The board also needs a room key (สร้าง secret) if it does not
   have one yet.
6. **Board.**
   - Flash `firmware/line_beacon/line_beacon.ino` on an ESP32 (Arduino core).
   - Before flashing, copy `secret.h.example` to `secret.h` and fill in the
     room key and the HWID.
   - Compiled here with esp32 core 3.3.12, using 83% of flash.
   - The board prints `counter=… dm=…` on serial every 10 s.
7. **Students.**
   - On the dashboard, tap "เชื่อมบัญชี LINE" to get a 6-digit code (valid 10 minutes).
   - Add the SmartCheck LINE account as a friend and send `LINK 123456`.

## How the room is proven

A plain LINE Simple Beacon can be copied: LINE's spec says it has no protection
against impersonation. The board therefore puts a signed counter in the
12-byte device message:

```
dm = counter (4 bytes, big-endian) || HMAC-SHA256(room key, "smartcheck-line-v1" || counter)[:8]
```

**What the server checks on each `enter` event:**
1. the webhook signature (`X-Line-Signature`, HMAC with the channel secret);
2. the HWID maps to a room;
3. the HMAC is valid for that room's key;
4. the counter was first seen less than 120 s ago (`line_beacon_sightings`);
5. the LINE user is linked to a student;
6. that student is enrolled in an open session in that room.

Only then does it issue a proximity receipt (method `line`, valid 300 s),
bound to the student, the session and the room.

**Limits:**
- **A recording replayed later is refused.** A live relay from inside the room
  at that moment is not; that holds for any radio proximity signal.
- **A phone in the room is not a person in the room.** That is why the face
  scan is still required.

## Trial: what to measure before relying on it

LINE's docs say nothing about detection time or repeated `enter` events. Every
event is logged with its delivery delay:

```
[LINE] beacon type=enter hwid=… dm_present=True latency_ms=…
[LINE] receipt issued student=… session=… latency_ms=…
```

Measure each of these, on the iPhone and Android phones you have:

| test | how |
|---|---|
| time to detect | note the clock when you walk in; compare with the log line time. ≥ 10 walk-ins per phone type |
| iPhone in the background / screen off | lock the phone before walking in |
| Android with LINE open vs closed | LINE says closed (background) will not work; confirm |
| re-entry | walk out and back in after 1, 5, 15 minutes: does a second `enter` arrive? |
| field name | does `dm_present=True` appear? If not, LINE delivers the device message under another name; send me one raw event |
| camera from the LINE button | the link adds `openExternalBrowser=1` to open the phone's browser; check that the camera opens and the student is still logged in |

**Two server warnings to watch for:**
- `reject=dm_invalid` means the board's key or HWID do not match the admin page.
- `reject=dm_replayed` means the counter was older than 120 s.

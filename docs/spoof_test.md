# Presentation-attack test (face stack v2)

This test measures how often SmartCheck accepts a fake face, and how often it
refuses a real one, at enrollment and at check-in. The terms follow ISO/IEC
30107-3:

- **APCER** (attack presentation classification error rate): attacks accepted
  divided by attack attempts. It is reported for each attack type.
- **BPCER** (bona fide presentation classification error rate): real faces
  refused divided by real attempts.

Report the numbers with `scripts/spoof_report.py`.

## Setup

**Accounts.** Use test accounts only, never real students' rows.

**Server.** Run the server with the production settings you chose:
`FACE_ENGINE=arcface`, your `FACE_CLAHE` and thresholds, and
`CHECKIN_LIVENESS=passive`. Write them into the result file.

**Devices.**
- One phone and one laptop for capture.
- For each attack, a printer, a second phone and a second laptop or monitor.
  Write down each device's model.

**Attack material.** Make it from the *victim's* own enrolled-quality photos:
- a frontal photo;
- a short video that also turns the head.

**Where to run it.**
- Run it in the room and lighting of the real class.
- Also run it in one bright and one dim setting if you can.

## Protocol (≥ 20 attempts per row, per stage)

| `attack` | what is shown to the camera | notes |
|---|---|---|
| `bona_fide` | the real person | several people, normal use; this row gives BPCER |
| `printed_photo` | A4 colour print of the victim's frontal photo | held flat, then curved; at 30-50 cm |
| `phone_screen` | the victim's photo or video on a phone, brightness at max | include the turn video at enrollment |
| `laptop_screen` | the victim's photo or video full-screen on a laptop or monitor | include the turn video at enrollment |

**Enrollment.** Each attempt is one full v2 session: consent, camera, result.
`accepted = 1` if the page says "ลงทะเบียนสำเร็จ".
- After an accepted attack, have an admin clear the test account before the next attempt.
- The server allows 5 attempts per 24 h per account (`atomic_enroll_attempt`).
  Spread the attempts over several test accounts, or reset the counter as an
  admin. Do not change the code.

**Check-in.** Use an account enrolled by the genuine person, in an open session.
`accepted = 1` if the check-in succeeds.
- Delete the attendance row between attempts. A second check-in in the same
  session is refused as a duplicate, not as a spoof.

**What to record for each attempt:** stage, attack, accepted, subject, and the
message shown when it was refused (`reason`). Keep these out of the CSV:
- photos of people who did not consent;
- the victim's real name.

## CSV template

```
stage,attack,accepted,subject,reason
enrollment,bona_fide,1,P01,
enrollment,printed_photo,0,P01,ตรวจพบรูปถ่ายหรือหน้าจอ — กรุณาใช้ใบหน้าจริงเท่านั้น
checkin,phone_screen,0,P02,ตรวจพบรูปถ่ายหรือหน้าจอ — กรุณาใช้ใบหน้าจริงเท่านั้น
```

## Report

```
python scripts/spoof_report.py spoof_attempts.csv --json spoof_report.json
```

**Output, for each stage:**
- APCER for each attack type, with a 95% Wilson interval;
- APCER_max, the worst attack type;
- BPCER with its interval;
- warnings for any row with fewer than 20 attempts.

**Read the intervals.** With 20 attempts, 0 accepted still allows a true rate
up to about 16%. State that limit next to any "0% APCER".

**Out of scope for this kit:** 3D masks, deepfake video injected as a virtual
camera (the page blocks known virtual-camera names only), and replays sent
directly to the API.

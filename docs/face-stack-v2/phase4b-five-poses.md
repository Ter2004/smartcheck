# Face stack v2: Phase 4b, five-pose enrollment (2026-10-26)

This replaces the single head turn of Phase 4 with five poses. You approved the
design on 2026-10-26: frontal, then four poses in a random order chosen by the
server. Left and right are checked against the calibrated yaw limit. Up and
down are checked for direction only, and are kept as templates. The page shows
arrows, a ring with five parts, a tone, and a vibration on Android. Only
`FACE_ENGINE=arcface` is affected.

## What the student sees (`enroll_face_v2.html`, `enrollment_v2.js`)

1. **Consent screen.** Notice version **1.3**. It now says 7 sets: frontal,
   left, right, up and down.
2. **Camera screen.** The video is mirrored, so the student's left is on the
   left of the screen. The frames sent to the server are drawn from the raw
   stream.
   - **Align:** face in the oval and frontal for 0.6 s.
   - **Look:** frontal for at least 2 s, with one blink. The page keeps the
     best 4 frontal frames and records this person's nose-ratio baseline.
   - **Four poses** in the order the server issued: left, right, up and down,
     shuffled. There are 24 orders, so a recording made in advance does not
     fit.
     - Each pose shows a blinking arrow next to the oval.
     - When 2 frames of the pose are captured, the page plays a short tone,
       vibrates (Android only) and shows ✅. One ring part turns green.
     - The student has 12 s per pose. After that the session fails with
       "ยังไม่เจอท่า …".
   - **Submit:** up to 12 frames (4 frontal + 2 per pose).
3. **Result screen.** Unchanged.

## Pose rules

| pose | browser captures when | server accepts when |
|---|---|---|
| left / right | eyeYaw beyond ±0.30 (unchanged) | yaw past 20.4° in that direction. The opposite turn fails with "หันผิดทาง" |
| up / down | yaw inside the v1 frontal gate (\|proxy\| ≤ 15) and the nose ratio moved **0.07** from the baseline | yaw frontal (≤ 18.4°) and pitch moved the right way from the same person's frontal mean (**sign only**) |

On the server, a missing pose fails with "{pose}อีกนิดตามลูกศร" and counts as a
retry; three retries send the student to staff.

The server also checks:

- the capture order: frontal first, then the issued order. A late frontal
  frame, a swapped pose or a missing pose fails;
- identity: each pose's best frame must match the frontal frames
  (`ARCFACE_MATCH_THRESHOLD`);
- duplicates: all 7 kept frames are checked against other accounts
  (`ARCFACE_DUP_THRESHOLD`).

Every posed frame is logged as
`step=pose_angle … client_yaw client_pitch server_yaw_deg server_pitch_delta seen`.
Both client and server pitch use positive = up.

**Kept:** 3 frontal templates plus the best frame of each pose, in the
challenge order (7 rows). The pose `CHECK` in
`20261019_face_stack_v2.sql` is now `frontal | left | right | up | down`.
- The file also drops and re-adds that check by name. Running it again over
  the earlier copy therefore upgrades the check; `CREATE TABLE IF NOT EXISTS`
  alone would keep `'turn'`.
- If old `turn` rows exist, the re-run stops with a check violation. Delete
  those test enrollments first.
- The verification query now prints a `pose_check` row.

## Evidence for the up/down numbers

| question | how | result | file |
|---|---|---|---|
| Which sign of server pitch means up? | corr(pitch, nose-to-mouth ratio) on 1,500 LFW faces | −0.79, so positive pitch = up | `arcface-pitch-sign.json` |
| How many server degrees is a browser step? | MediaPipe 0.10.14 nose ratio vs server pitch on the same 1,500 faces; both least-squares directions | corr −0.81. Step 0.07 = **8.7–13.3°**. The v1 frontal window (15 → 0.167) = 20.8–31.8° | `client-pitch-calibration.json` |
| Can server noise flip the sign? | 300 LFW faces × 6 perturbed copies (2 px shift, ±3 % scale, JPEG 70, +10 % brightness) | per-face spread: median 0.18°, 95th percentile 0.48°. Largest change: 95th 1.26°, 99th 7.21° | `server-pitch-jitter.json` |
| Will the server read up/down frames as frontal yaw? | browser frontal gate vs server direction on LFW | 958 of 968 (99.0 %) | `client-pitch-calibration.json` |

**Why the step is 0.07.** The first draft used 15 proxy degrees, the v1 frontal
window.
- The fit shows 15 needs 21–32° of pitch. That is as wide as the whole 5–95 %
  spread of the ratio on LFW (0.454–0.615), so many students would never reach
  it.
- 0.07 is the top of the v1 nod gesture's range (`mediapipe_liveness.js`:
  neutral 0.50–0.55, down at > 0.57; 0.60 had been "too far").
- It reads 8.7–13.3° on the server, about 7× the 95th-percentile noise.

**Limits of this evidence.**
- LFW is still photos of many people, not phone video of one person.
- The perturbed copies only stand in for frame-to-frame noise.
- The real check is the phone log: `scripts/report_turn_yaw.py` now reports
  up/down too. It shows sign agreement, server degrees per browser unit, and
  frames whose yaw was not frontal.

Reproduce:
`python scripts/calibrate_arcface_pitch.py`, then
`python scripts/calibrate_client_pitch.py mediapipe|join|jitter`. The mediapipe
step needs a separate venv with `mediapipe==0.10.14`; the other steps use the
runtime venv.

## Tests (2026-10-26, clean venv without TensorFlow, `sc-pgtest` Postgres)

Full suite: **303 tests OK**, 1 skipped (the DeepFace parity test, whose
dependency is no longer installed). All 10 JavaScript check scripts pass.

- `tests/test_enroll_v2.py` (18 tests) covers:
  - 7 templates in the challenge order;
  - wrong turn;
  - up/down with the wrong sign;
  - up with a turned head;
  - a still photo;
  - an unknown or reused challenge;
  - capture order: late frontal, swapped poses, a missing pose;
  - person swap, duplicate account, second person, overwrite, spoof, three
    failures;
  - log lines that `report_turn_yaw.parse` reads;
  - random orders.
- `tests/test_face_stack_v2_sql.py`:
  - the five labels save through the RPC;
  - `turn` is refused by the CHECK, and nothing is half-saved;
  - re-running the file over the old `('frontal', 'turn')` check upgrades it;
  - with an old `turn` row present, the re-run stops.
- `tests/test_phase6_reports.py`: the report on `pose_angle` lines and on old
  `turn_yaw` lines.
- `scripts/test_enrollment_v2.js`: nose-ratio sign and the pose rules.
- `scripts/test_enrollment_v2_flow.js` (new) drives the page controller with
  synthetic landmarks: align → look + blink → four poses → submit.
  - Checks: 12 frames, capture order, pitch and yaw signs, 5 vibrations, the
    ring;
  - the opposite turn and "up with a turned head" are not captured;
  - the 12 s timeout;
  - a second face restarts the hold.
  - Three deliberate code mutations each made this test fail: a step of 0.15,
    a flipped up sign, and no yaw gate.

## Not verified here

- **No real camera run.** Android Chrome, laptop and iPhone Safari are all
  untested.
  - It is unknown whether 0.07 is comfortable for most students.
  - iPhones do not vibrate: Safari has no Vibration API.
  - The tone needs the start tap. It may be silent when the iPhone's silent
    switch is on.
- **What to watch in the trial log:** `up_missing` / `down_missing` failures.
  The browser accepted the frame, but the server's pitch read too close to
  frontal.
- **Database.** Apply the current `20261019_face_stack_v2.sql`; its pose
  `CHECK` changed in this phase. It had not been applied to Supabase before.

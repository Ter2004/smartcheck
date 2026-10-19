# Face stack v2: Phase 4, new enrollment flow (2026-10-19)

The flow below applies only with `FACE_ENGINE=arcface`. With facenet512 the
v1 page and `/api/enroll` run exactly as before (checked: `/student/enroll`
still renders `enrollment_flow.js`). `/api/enroll` was changed with your
approval of 2026-10-19.

## UI (`templates/student/enroll_face_v2.html`, `static/js/enrollment_v2.js`)

### Screen 1: consent and start

- A 3-line summary and a link to the full notice (`/student/pdpa`, version **1.2**).
- The checkbox is unchecked by default.
- The start button posts the consent and calls `getUserMedia`, so the camera
  permission prompt comes from the same tap.

### Screen 2: camera session

There are no buttons. The page shows one instruction at a time and a progress
ring around the oval.

1. **Face in the oval.** The face must be in the oval, frontal and large
   enough. The ring turns green and the page moves on after 0.6 s.
2. **Look at the camera for about 2 s.**
   - The page keeps the best frontal frames and watches for a natural blink.
   - It asks "กะพริบตา 1 ครั้ง" only if no blink is seen within 3 s.
3. **Turn the head once,** in the direction the server chose.

### Screen 3: result

- **Success**, or one Thai fix plus "ลองใหม่". Retrying gets a new challenge
  without asking for consent again.
- After 3 failed attempts the server answers `contact_staff`, and the page
  tells the student to enroll with the teacher or TA.

### Client checks

- **Scoring:** each frame is checked for pose, face size, centring,
  brightness, backlight, sharpness and eyes open. Mouth openness ranks the
  frames. The page keeps the best 5 frontal and best 3 turned frames and posts
  at most 8, each with `t` (ms since the challenge) and its own `client_yaw`.
- **Gate values** are reused from the v1 page:

| gate | value |
|---|---|
| face height | ≥ 0.30 |
| centre | 25-75% x / 20-80% y |
| frontal proxy | ±15° |
| brightness | ≥ 55 |
| edge/centre brightness | ≤ 2.2 |
| blur | ≥ 20 |

- **Turn:** eyeYaw > 0.30. This is the WIP-branch calibration: 0.30 here
  passes the server's turn limit for 94% of 3,870 LFW faces.
- **Blink:** EAR below 0.70 × the open-eye baseline, then back. This is the
  old v1 blink rule.

## Server (`app/services/enroll_v2.py`, `routes/student.py`)

### New and changed routes

- **`POST /student/api/enroll/challenge`** returns `{challenge_id, direction ∈ {turn_left, turn_right}, expires_in: 120}`.
- **`POST /student/api/enroll` (arcface)** runs, in this order:
  1. consent: the session flag plus the latest `consent_logs` row, fail-closed;
  2. one attempt per challenge, which is popped from the session;
  3. nonce, TTL, frame count 5-8, timestamps in range, every frontal frame before every turn frame;
  4. `atomic_enroll_attempt`: at most 5 per 24 h;
  5. **frame validation** (`server_validate_frame`);
  6. **one face per frame**: a second large face anywhere fails the attempt;
  7. **quality gates** (the same values as the page);
  8. **anti-spoof on ≥ ⌈0.8·N⌉ frames**, which is v1's 4-of-5 ratio;
  9. **the head turn matches the challenge**, using yaw degrees and the
     Phase 3 limits; any turned frame in the opposite direction fails;
  10. **embeddings**;
  11. **self-consistency**: frontal pairs must be ≥ `ENROLL_SELF_SIM_MIN`, and
      each turned frame must reach ≥ `ARCFACE_MATCH_THRESHOLD` against a frontal
      one, which catches a person swap;
  12. **1:N duplicate** (`match_face_templates`, excluding own rows, ≥ `ARCFACE_DUP_THRESHOLD`);
  13. **keep 5** (3 frontal + 2 turned, ranked by det score, then size, then sharpness);
  14. **one transaction** (`complete_face_enrollment`: enrollment + templates + HMAC);
  15. **best frontal frame** as the private thumbnail (non-fatal, like v1's photo).
- **Duplicate hit:**
  - the response says only `"ไม่สามารถลงทะเบียนได้ กรุณาติดต่ออาจารย์"`;
  - an `audit_logs` row `face_duplicate_blocked` keeps `{matched_student, similarity, model_version}` server-side;
  - the matched account never reaches the client (checked by a test).
- **`/student/api/consent`** writes consent version 1.2 under arcface and keeps
  the `consent_logs.id` for `face_enrollments.consent_id`.
- **Enrollment status** under arcface means a completed `face_enrollments` row
  for the active model version. That status drives the dashboard, the
  check-in redirect and the overwrite guard.
- **Withdrawal** also deletes the v2 enrollment (its templates cascade) and
  fails loudly on any error except "table missing".
- **Teacher photo view** also reads `face_enrollments.thumbnail_path`.

### Deliberate deviations from the brief

- **Turn bucket.** The brief describes "slight turn ≤ ~15°", but the server
  only counts a turn from 20.4° (calibrated in Phase 3). Frames at ≤ 15°
  cannot prove a turn. So the client keeps frames past the turn limit (eyeYaw
  0.30), and those become the 2 turned templates. Narrowing this needs a
  decision from you.
- **The blink is checked only in the browser.** The server cannot see a blink
  in ≤ 8 still frames, so the blink is a UX and liveness hint, not a server check.
- **Replayed video.** A recording of the student that turns in the direction
  the server happens to ask for (50%), sent with forged timestamps, passes the
  geometry checks. Only anti-spoof stands in its way. The tests cover the
  replays the server can catch: a wrong challenge, a reused challenge, frames
  in the wrong order, and the wrong direction.

## Evidence

**Python tests.** `python -m unittest tests.test_enroll_v2 -v`: **Ran 13, OK**.
- `test_correct_turn_enrolls_3_frontal_2_turned`: saves 3+2 templates, the HMAC, the consent id and the thumbnail.
- `test_wrong_turn_direction`.
- `test_static_photo_has_no_turn`.
- `test_replay_is_refused`: wrong id, out-of-order frames, reused challenge.
- `test_person_swap_mid_capture`.
- `test_same_face_into_second_account_is_blocked_without_revealing_it`.
- `test_second_person_in_frame`.
- `test_overwrite_attempt`: the status guard gives 409, and the DB/RPC refusal also gives 409.
- `test_spoof_frames_fail`.
- `test_three_failures_send_student_to_staff`.
- `test_withdrawn_consent_blocks`.
- `test_withdrawal_deletes_v2_enrollment`.
- `test_gates_on_real_frames`. This test found a real bug: the Laplacian ran
  on a float32 image, which OpenCV rejects. It now runs on uint8 like
  `server_validate_frame`.
- The tampered-row (HMAC) case belongs to check-in and is tested in Phase 5.

**JavaScript checks.**
- `node scripts/test_enrollment_v2.js`: all assertions pass. They cover yaw
  sign and scale, the frontal and oval gates, EAR and blink, the keeper, and
  lighting and blur.
- The other `scripts/test_*.js` files pass as well.

**Page render.** `/student/enroll` and `/student/pdpa` both return 200 under
arcface. Under facenet512, `/student/enroll` still serves the v1 page.

**Full suite.** Ran 261, FAILED (failures=1). The failure is the existing
`test_simulator_console_to_checkin`.

**Not tested.**
- A real camera on a phone or laptop, including iPhone Safari: no camera is
  available to this session.
- The ≥ 20 recorded turns for browser vs server yaw.
  `scripts/report_turn_yaw.py` reads the `step=turn_yaw` log lines that every
  v2 enrollment writes. It was only smoke-tested on synthetic lines.

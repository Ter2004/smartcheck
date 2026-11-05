# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

The student app is a Capacitor wrapper (`android-app/`) around the same Flask site. It adds native room-board scanning and notifications; the screens themselves are web pages. iPhones and laptops use the site in a browser.

## Users

- **Students (primary):** they are in a classroom, holding their own Android phone with the SmartCheck app logged in for the whole term. Their job is to check in for the class that just opened in a few seconds and go back to the lecture. Their attention is split, and they may be walking into the room.
- **Teachers:** use their own **phone** to open the class and choose the on-time minutes, then watch who has checked in. A laptop is secondary.
- **Admin:** a staff member who sets up users, courses, schedules, room boards and the term, at a desk.

## Product Purpose

SmartCheck replaces calling the roll. A teacher opens the class; every student's phone is notified; each student proves they are in the room (room board) and that they are themselves (face). Attendance is then recorded as present or late with no further work from the teacher.

**Success right now:** a full class checks in faster than a teacher calling the roll. This is the senior-project committee's main question after the 2026-10-05 presentation.

## Positioning

The whole room checks in at the same time, with no queue. A small ESP32 board in the room broadcasts a signed code that changes every 5 s; phones only listen to it. The server then verifies the student's face (ArcFace) with anti-spoofing. Roll call is slow, and QR codes or room codes can be forwarded; this method is fast and needs presence plus identity.

## Operating Context

- **Phase:** a senior project (Project II) at PSU Phuket. The aim now is a committee demonstration plus a real test in one classroom. It is designed so it could keep being used afterwards, but campus-wide rollout is not the goal now.
- **Class flow:**
  1. The teacher presses "เปิดคาบ" and picks the on-time minutes (default 5).
  2. Students get a notification, tap it, the app finds the board, and they scan their face.
  3. A check-in inside the window is "มา"; after it, "สาย".
  4. If the teacher forgets, the class opens itself 15 minutes after the scheduled start. It closes at the scheduled end.
- **Devices:** students use Android phones from low to mid range (tested on an OPPO CPH2339 and a Redmi Note 14 5G). The server runs on a PC behind a Cloudflare tunnel. The database is Supabase (Mumbai).
- **Language:** all UI copy is Thai.

## Capabilities and Constraints

- **Roles:** student, teacher and admin, each with its own pages (`app/templates/{student,teacher,admin}`).
- **Face enrollment:** one camera session, 5 poses, with consent (PDPA page). The `/api/enroll` endpoint must not be changed without explicit approval.
- **Check-in:** board scan, then the camera, then face capture, then the server's face match and anti-spoof check (about 1 s on the server). Measured on 2026-11-06: 9–10 s from opening the check-in page to success.
- **Fallbacks:** iPhones and laptops cannot scan the board, so they use a rotating code or QR on the teacher's screen (`CHECKIN_SESSION_CODE`). They get no notifications.
- **Not allowed:** LINE in any form; the user rejected it.
- **Front end:**
  - Styling: Tailwind, prebuilt to `app/static/css/tailwind.css`. Run `npm run build` in `tools/tailwind` after adding classes; a test fails on missing classes.
  - JavaScript: plain, no framework.
  - Keep element ids used by `checkin_flow.js`, `enrollment_v2.js` and the tests.
- **Not decided yet:**
  - A teacher button to close a class early.
  - Make-up classes and cancelled days.
  - A fixed URL / cloud hosting (currently a quick tunnel).

## Brand Commitments

- The name is "SmartCheck". There is no logo or brand guide on file.

## Evidence on Hand

- **Measured timings:** in memory notes and commit messages (e.g. Phases 4c–5j).
- **Real-phone check-ins with two phones at once (2026-11-06):** a push arrives about 1 s after the teacher presses; check-in 21–25 s after that press.
- **No user testimonials, class-wide results or committee approvals exist yet.** Do not invent them.

## Product Principles

1. **Seconds matter.** Every screen a student passes through is measured against roll call. Remove waits, taps and reading.
2. **Presence before face.** The face camera never opens until the phone has found the room board.
3. **Tell people what to do next.** Every error says the fix in plain Thai (turn on Location, come into the room, try again).
4. **The teacher does one thing.** Opening the class is the only required action; everything else runs by itself.
5. **Honest status.** Present, late or failed is shown exactly as recorded, never optimistically.

## Accessibility & Inclusion

- Students use the app on a phone, often one-handed, while walking into a room. Use large touch targets and readable Thai text at phone sizes.
- The design must work on low-end Android phones: keep pages light and avoid heavy effects.

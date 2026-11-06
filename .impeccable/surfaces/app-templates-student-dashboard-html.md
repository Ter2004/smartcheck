---
version: 1
slug: "app-templates-student-dashboard-html"
primary_target: "app/templates/student/dashboard.html"
related_targets: ["app/templates/student/checkin.html","app/templates/student/checkin_select.html"]
---

Scope: student surfaces on phones (home/today, class list, check-in flow). Mode: Operate.
Audience and job: a student walking into class, one hand, a few seconds: see whether the class is open, how long until late, check in, see it recorded.
Constraints: Thai copy; keep every element id used by checkin_flow.js / enrollment_v2.js / push_notifications.js; /api/enroll untouched; Tailwind is prebuilt (npm run build in tools/tailwind).

## Direction contract
THESIS: Each class is a boarding pass and the classroom is the gate: the gate opens when the teacher opens the class, the pass says BOARDING, and checking in is boarding. Refuses the category default of white SaaS cards on grey with a blue button.
OWN-WORLD: Tarmac navy #0B1D3A shell (status bar to tab bar), white pass cards with a perforated tear line (dashed rule plus two half-moon notches), gate amber #FFB000 band reserved for the class boarding now, go green #12855B for the board action and boarded, red #C9343A for closed/failed (both darkened 2026-11-06 from #19A974 / #E5484D: white text was 3.0:1 / 3.9:1), slate #51607A secondary ink. Anuphan for display and numerals (tabular), Noto Sans Thai for body. Standard bottom tab bar, standard buttons.
STORY: The student understands at a glance which class is boarding and the minutes left, believes a check-in is recorded the moment the pass turns BOARDED, and does one tap to board.
FIRST VIEWPORT: Navy shell. Small date + greeting. One full-width pass for the class boarding now (or the next one): amber status band with time-to-late, course code at 30px display, grid of room / start / time left, perforation, full-width green "สแกนหน้าเช็คชื่อ". Below on navy: "คาบเรียน" departure rows (time, course · room, status chip). Bottom tab bar: วันนี้ / เช็คชื่อ / บัญชี.
FORM: Boarding pass + departures board (Phuket airport), position 1 of 7 on the ordered list (picked over the roll's assignment, "ตรายางสมุดเช็คชื่อ", by the user). seed:29924065f7997d79
SIGNATURE: The status band behaves like a gate display: BOARDING · เหลือ m:ss counts down live, flips to CLOSING in the last minute and to LATE / CLOSED as state changes; a pass flips to BOARDED with the check-in time.
FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

## Finish review (2026-11-06, in-thread pass, not the separate finish-reviewer agent)
Checked at 390px and 1440px: student home, class list + timetable, check-in, account, enroll v2, PDPA, login, teacher home/results, admin overview/users/beacons/sessions/courses/course detail/term.
- THESIS / FIRST VIEWPORT / SIGNATURE: met. Navy shell, hero pass with live band, green board action, departures rows, tab bar; band flips BOARDING → CLOSING → สาย.
- Fixed in this pass: white-on-green and white-on-red contrast (tokens darkened); Thai page titles overlapping subtitles (line-height); doubled "ห้อง" on teacher pages.
- Extended to all roles: navy sidebar/top bar with amber active state, Tailwind indigo remapped to navy, login as a pass, teacher "คาบวันนี้" as gate passes.
- Open (owner decision): band/chip words are English + Thai (BOARDING/BOARDED/CLOSING/CLOSED); Thai-only labels proposed but not applied.
- Rasters: none shipped (code-led), so no provenance needed.
VERDICT: finished for this scope, with one open copy decision.
DESIGN.md + .impeccable/design.json rewritten for this world on 2026-11-06 (the old Apple-blue version is in git history).

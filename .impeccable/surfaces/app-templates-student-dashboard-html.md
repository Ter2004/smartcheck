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
OWN-WORLD: Tarmac navy #0B1D3A shell (status bar to tab bar), white pass cards with a perforated tear line (dashed rule plus two half-moon notches), gate amber #FFB000 band reserved for the class boarding now, go green #19A974 for the board action and boarded, red #E5484D for closed/failed, slate #51607A secondary ink. Anuphan for display and numerals (tabular), Noto Sans Thai for body. Standard bottom tab bar, standard buttons.
STORY: The student understands at a glance which class is boarding and the minutes left, believes a check-in is recorded the moment the pass turns BOARDED, and does one tap to board.
FIRST VIEWPORT: Navy shell. Small date + greeting. One full-width pass for the class boarding now (or the next one): amber status band with time-to-late, course code at 30px display, grid of room / start / time left, perforation, full-width green "สแกนหน้าเช็คชื่อ". Below on navy: "เที่ยวบินวันนี้" departure rows (time, course · room, status chip). Bottom tab bar: วันนี้ / เช็คชื่อ / บัญชี.
FORM: Boarding pass + departures board (Phuket airport), position 1 of 7 on the ordered list (picked over the roll's assignment, "ตรายางสมุดเช็คชื่อ", by the user). seed:29924065f7997d79
SIGNATURE: The status band behaves like a gate display: BOARDING · เหลือ m:ss counts down live, flips to CLOSING in the last minute and to LATE / CLOSED as state changes; a pass flips to BOARDED with the check-in time.
FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

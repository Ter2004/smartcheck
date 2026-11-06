---
name: SmartCheck
description: Classroom attendance in seconds. Each class is a boarding pass, the classroom is the gate.
colors:
  tarmac: "#0B1D3A"
  tarmac-2: "#12284D"
  tarmac-line: "rgb(255 255 255 / 12%)"
  gate: "#FFB000"
  go: "#12855B"
  go-deep: "#0E6B49"
  stop: "#C9343A"
  late: "#F08C2A"
  pass: "#FFFFFF"
  pass-band-neutral: "#E6EAF2"
  pass-ink: "#0B1D3A"
  pass-muted: "#51607A"
  pass-rule: "#CBD3E1"
  work-ground: "#F3F5FA"
  navy-500: "#2E4A7A"
  navy-600: "#12284D"
  navy-700: "#0B1D3A"
typography:
  display:
    fontFamily: "Anuphan, Noto Sans Thai, system-ui, sans-serif"
    fontWeight: 700
    note: "pass codes, band labels, times, page titles"
  pass-code:
    fontFamily: "Anuphan"
    fontSize: "30px"
    fontWeight: 700
    lineHeight: 1.3
  band:
    fontFamily: "Anuphan"
    fontSize: "14px"
    fontWeight: 700
    letterSpacing: "0.03em"
  page-title:
    fontFamily: "Anuphan"
    fontSize: "clamp(1.65rem, 3vw, 2.2rem)"
    fontWeight: 600
    lineHeight: 1.4
  body:
    fontFamily: "Noto Sans Thai, system-ui, sans-serif"
    fontSize: "15px"
    lineHeight: 1.6
  mono:
    fontFamily: "JetBrains Mono, ui-monospace, monospace"
    note: "IDs, UUIDs, dates in admin tables only"
rounded:
  pass: "20px"
  button: "14px"
  chip: "7px"
  timetable-block: "8px"
  control: "12px"
spacing:
  pass-padding: "18px"
  board-action-height: "54px"
  quiet-action-height: "50px"
  tab-height: "52px"
components:
  pass-band:
    states: { boarding: "{colors.gate} + ink", closing: "{colors.stop} + white", late: "{colors.late} + ink", boarded: "{colors.go} + white", failed: "{colors.stop} + white", closed: "{colors.pass-band-neutral} + muted", waiting: "{colors.pass-band-neutral} + ink" }
  btn-board:
    backgroundColor: "{colors.go}"
    hover: "{colors.go-deep}"
    textColor: "#FFFFFF"
    height: "{spacing.board-action-height}"
    rounded: "{rounded.button}"
  btn-quiet:
    backgroundColor: "transparent"
    border: "1.5px {colors.pass-rule}"
    textColor: "{colors.pass-ink}"
    height: "{spacing.quiet-action-height}"
  btn-gate:
    backgroundColor: "{colors.gate}"
    textColor: "{colors.pass-ink}"
  admin-primary:
    backgroundColor: "{colors.navy-600}"
    hover: "{colors.navy-700}"
    note: "Tailwind indigo-* classes are remapped to this navy scale"
---

# Design System: SmartCheck

## Overview

**Creative North Star: "Each class is a boarding pass; the classroom is the gate."**

The student opens the app the way a traveller checks a departures board: what is boarding now, what is next, am I on it. The page answers that in one glance and offers one action. The airport world gives the product a vocabulary every student already reads without instruction:
- **Pass:** one class.
- **Band:** its state right now.
- **Gate amber:** "open now, go".
- **Green:** "you're on board".

This replaced the earlier Apple-blue system (Nov 2026 redesign, Phase 7). The old look is in git history and is an anti-reference: no soft gray canvas with one blue, and no generic white cards.

**Two grounds, one world**
- **Students, phone-first:** the *tarmac*, a deep navy page. White passes sit on it, with a departures board and a bottom tab bar (วันนี้ / เช็คชื่อ / บัญชี).
- **Teachers and admins, desk and phone:** a light *work ground* (#F3F5FA) for scanning tables. They share the same navy sidebar, amber active state, navy actions and Anuphan titles. The teacher's "คาบวันนี้" are passes on the light ground (`.pass-on-light`).

**Key characteristics**
- The pass is the signature component: band, body, perforation with notches, foot.
- Colour is state, never decoration:
  - amber = boarding/open;
  - green = checked in / do the check-in;
  - red = closing/closed;
  - orange = late;
  - neutral = waiting or done.
- Anuphan for anything you read at a glance: codes, times, bands, titles. Noto Sans Thai for sentences.
- Big targets: the main action is 54px tall and full width inside the pass.

## Colors

### Ground
- **Tarmac #0B1D3A:** the student page background, sidebar and top bar. Also the ink colour on passes, so the pass reads as cut from the same world.
- **Tarmac-2 #12284D:** raised surfaces on the tarmac (notes, tab bar, timetable frame).
- **Tarmac line (white 12%):** rules between departures-board rows.
- **Work ground #F3F5FA:** teacher/admin page background.

### State
| Token | Value | Meaning | Text on it |
|---|---|---|---|
| gate | #FFB000 | boarding / open now, active nav item | ink #0B1D3A |
| go | #12855B | checked in; the "board" action | white (4.65:1) |
| stop | #C9343A | last minute before late; check-in failed (nothing recorded) | white (5.2:1) |
| late | #F08C2A | still open but now counts as late; checked in late | ink (6.8:1) |
| band neutral | #E6EAF2 | waiting / finished | ink or muted |

go and stop were darkened from #19A974 / #E5484D after a contrast check: white text measured 3.0:1 and 3.9:1, below WCAG AA.

### Navy action scale (teacher/admin)
Tailwind's `indigo-*` classes are remapped in `tools/tailwind/tailwind.config.js` to navy:
- 50 #EEF2F8
- 100 #DDE5F1
- 200 #BCC9DE
- 300 #8FA2C3
- 400 #5B729B
- 500 #2E4A7A
- 600 #12284D
- 700 #0B1D3A
- 800 #071328
- 900 #040B18

Every existing `bg-indigo-600` button became navy without template edits. Rebuild Tailwind after changing it (`cd tools/tailwind && npm run build`).

### Named rules
- **One meaning per colour.** Amber is never used for "warning text", and green is never a decorative accent.
- **No side stripes.** State fills the whole band or block, never a coloured left border.

## Typography

- **Display: Anuphan** (Google Fonts, weights 600/700). Used for pass codes (30px/700), band labels (14px/700, +0.03em), departures-board times, page titles and button labels on `.btn-board`. Times use `font-variant-numeric: tabular-nums` so countdowns do not jitter.
- **Body: Noto Sans Thai** for sentences, labels and helper text, at 15px on student screens and 14px in admin tables.
- **Mono: JetBrains Mono** only for IDs, UUIDs and dates in admin/teacher tables.
- **Page titles** (`main h1`) force `line-height: 1.4 !important`. Tailwind's `text-xl` line-height would otherwise let Thai upper marks hit the subtitle.

## Layout

- **Student pages:** a single column. Head = small grey date line + white title; then the hero pass; then the board. `#main-content` gets bottom padding for the fixed tab bar. The tab bar hides at ≥1024px, where the sidebar takes over.
- **Weekly timetable** (เช็คชื่อ tab):
  - days are rows (Mon–Fri), hours are columns (07:00–19:00);
  - it scrolls sideways with the day names sticky;
  - blocks are positioned by % of the 12-hour span, from `week_timetable()` in `app/routes/student.py`;
  - classes outside 07–19 or at weekends are listed under "นอกตาราง".
- **Teacher/admin:** the existing sidebar + content layout. Tables sit in white rounded containers on the work ground.

## Components

### Pass (`.pass`)
White card, 20px radius, overflow hidden. Parts:
1. `.pass-band`: state strip. Left = state label in Thai, right = time or room. `data-state` values:
   - boarding: เปิดเช็คชื่อ
   - closing: ใกล้สาย, the last minute
   - late: เปิดอยู่ · ตอนนี้นับสาย / เช็คชื่อแล้ว · มาสาย
   - boarded: เช็คชื่อแล้ว
   - failed: ยังไม่ได้บันทึก
   - closed: ปิดแล้ว / ถ่ายใหม่อีกครั้ง
   - waiting: รอเปิดคาบ

   The label flips (`.flip.turn`, rotateX) when the countdown or the result changes it. Status words are Thai only (owner decision 2026-11-06); the airport world lives in the shapes, not in English words.
2. `.pass-body`: `.pass-code` (course code), `.pass-name`, optional `.pass-grid` of three fields (ห้อง / เริ่ม / เหลือ-สถานะ).
3. `.pass-perf`: 2px dashed rule with two notches, coloured as the ground. `.pass-on-light` recolours the notches for the light ground.
4. `.pass-foot`: one action, or a one-line explanation.

### Actions
- `.btn-board`: green, 54px, full width, Anuphan 18px. The one thing to do (check in, open class, sign in).
- `.btn-quiet`: outlined, 50px. Secondary (view results, log out).
- `.btn-gate`: amber, for an "open now" jump on the tarmac.
- Admin/teacher primary: navy (`bg-indigo-600`, remapped).

### Departures board (`.board-title`, `.board-row`, `.chip`)
Rows of time / class + room / chip. The chips:
- `boarding`: amber;
- `boarded`: green, "มา 07:58";
- `late`: orange, "สาย 10:07";
- `closed`: red;
- default: translucent white, "รอเปิด".

### Check-in flow
The whole flow lives inside one pass:
- the band (`#checkinBand`, `#checkinStageTitle`) names the step;
- `.ci-steps` dots show progress (amber active, green done);
- `.ci-radar` shows while scanning for the room board (stops when nothing was found);
- `.ci-code` holds the fallback code inputs.

The camera opens only after the room board is found.

During camera + checking, `body.ci-focus` hides the tab bar and back link, so a stray thumb cannot leave the scan. The room-receipt time is mentioned only in its last 15 s.

The result is the band itself:
- green "เช็คชื่อแล้ว · มาเรียน" (orange when late) with the recorded time large;
- red "ยังไม่ได้บันทึก" on failure;
- neutral "ถ่ายใหม่อีกครั้ง" when a live face could not be confirmed. Never accuse ("cheating").

Icons are drawn SVG, not emoji. With no retry, "กลับหน้าวันนี้" is the green action.

### Teacher live view (`teacher/session_view.html`)
- **Hero:** one pass with the band (เปิดเช็คชื่ออยู่ · ทันเวลาถึง HH:MM) and a big "เช็คแล้ว N/M", plus มาเรียน / มาสาย / ยังไม่เช็ค. The last becomes ขาด only after close.
- **Live updates:** the page polls `/teacher/session/<id>/live.json` every 10 s while visible, with no full reload.
- **Phones:** rows become cards.
- **After opening:** opening a class lands here.

### Navigation
- **Student phone:** bottom `.tabbar`. Active tab is amber with `aria-current="page"`. Logout lives in the บัญชี tab.
- **Sidebar (all roles, ≥1024px):** navy. The active item has an amber label/icon on a tarmac-2 pill. The logo tile is amber.

### Dates
Every screen shows dates as "6 พ.ย. 69" (Buddhist year, two digits), with the time after it when it matters ("6 พ.ย. 69 08:05"). Use the `thai_date` filter (app/services/thai_dates.py). It converts UTC timestamps to Thai time; never slice `start_time[:10]`, because that is the UTC date. Browser date inputs keep the browser's own format.

### Student account (บัญชี tab)
A pass answering "am I set up?":
- the band says พร้อมเช็คชื่อ / ยังไม่พร้อมเช็คชื่อ;
- face status, with an enrol button when missing;
- notification permission, read from the Android app;
- then password and logout.

Consent withdrawal stays "contact admin" (owner decision 2026-11-07).

### Segmented choice (`.seg`)
Real radio inputs drawn as 48 px segments. The selected segment is ink-filled. Used for the teacher's on-time minutes (5 / 10 / 15), so opening a class needs no keyboard.

## Do's and Don'ts

### Do
- Put any new per-class surface in a pass, with a band that states its state in words.
- Keep one green action per screen.
- Check white-on-colour contrast before adding a state colour (≥4.5:1 for text under 24px).
- Restart the Flask server after template edits; templates are cached.

### Don't
- Reintroduce the Apple-blue #0071e3 or plain gray-canvas cards.
- Put English airport words (BOARDING, CLOSED…) in status labels. Students read Thai at a glance; the metaphor is carried by the pass, band and board shapes.
- Use loaded or militaristic aviation imagery. The world is a civilian airport only.
- Add coloured left-border stripes, gradients on text, or decorative glass blur.

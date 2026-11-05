---
name: SmartCheck
description: Classroom attendance in seconds — room board plus face, on the student's phone.
colors:
  action-blue: "#0071e3"
  action-blue-deep: "#0062c4"
  action-blue-wash: "#eaf3ff"
  canvas: "#f5f5f7"
  surface: "#ffffff"
  ink: "#1d1d1f"
  ink-secondary: "#424245"
  ink-muted: "#6e6e73"
  hairline: "#e5e5ea"
  field-stroke: "#d2d2d7"
  present-green: "#047857"
  present-wash: "#ecfdf5"
  late-amber: "#b45309"
  late-wash: "#fffbeb"
  error-rose: "#be123c"
  error-wash: "#fff1f2"
typography:
  headline:
    fontFamily: "Noto Sans Thai, system-ui, sans-serif"
    fontSize: "clamp(1.65rem, 3vw, 2.2rem)"
    fontWeight: 600
    lineHeight: 1.45
    letterSpacing: "-0.035em"
  title:
    fontFamily: "Noto Sans Thai, system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 600
    lineHeight: 1.5
  body:
    fontFamily: "Noto Sans Thai, system-ui, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: 1.6
  label:
    fontFamily: "Noto Sans Thai, system-ui, sans-serif"
    fontSize: "0.75rem"
    fontWeight: 500
    lineHeight: 1.5
  mono:
    fontFamily: "JetBrains Mono, ui-monospace, monospace"
    fontSize: "0.875rem"
    fontWeight: 500
rounded:
  control: "12px"
  card: "22px"
  pill: "9999px"
spacing:
  control-height: "44px"
  card-padding: "26px"
  card-padding-phone: "20px"
components:
  button-primary:
    backgroundColor: "{colors.action-blue}"
    textColor: "{colors.surface}"
    rounded: "{rounded.control}"
    height: "{spacing.control-height}"
    padding: "10px 16px"
  button-primary-hover:
    backgroundColor: "{colors.action-blue-deep}"
  input-field:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    height: "{spacing.control-height}"
    padding: "10px 16px"
  card:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.card}"
    padding: "{spacing.card-padding}"
  status-present:
    backgroundColor: "{colors.present-wash}"
    textColor: "{colors.present-green}"
    rounded: "{rounded.pill}"
  status-late:
    backgroundColor: "{colors.late-wash}"
    textColor: "{colors.late-amber}"
    rounded: "{rounded.pill}"
  status-error:
    backgroundColor: "{colors.error-wash}"
    textColor: "{colors.error-rose}"
    rounded: "{rounded.pill}"
---

# Design System: SmartCheck

## Overview

**Creative North Star: "The Classroom Door That Opens Itself"**

SmartCheck should feel like walking through a door that opens as you arrive. It is quiet, clean and fast. At each moment it shows only the next thing to do. Students and teachers arrive at a screen, get its one job done, and leave. Nothing on the page competes with the job.

The system is calm and Apple-adjacent:
- a soft gray canvas;
- white cards with generous corners;
- one blue for every action;
- Thai text set in Noto Sans Thai.

Colour carries meaning, not decoration. Blue means "do this". Green, amber and rose are reserved for the attendance truth: present, late, failed. Density is low on student screens, which are used one-handed on a phone. It is moderate on teacher and admin tables.

**Key Characteristics:**
- One action color (Action Blue); status colors only for status.
- White cards on a gray canvas, 22px corners, near-invisible shadow.
- 44px minimum controls, 12px control corners.
- Thai-first typography; headline tracking tightened, body left at normal tracking.

## Colors

The palette is a near-neutral gray canvas with a single saturated blue for action, plus three status hues used only for attendance outcomes.

### Primary
- **Action Blue** (#0071e3): buttons, links, the active nav item, focus rings, progress rings and step dots. It is the only color that says "tap here". In Tailwind it is `indigo-600`, because the `indigo` scale is remapped to this blue in `tools/tailwind/tailwind.config.js`.
- **Action Blue Deep** (#0062c4): hover and pressed state of primary actions, and active nav text.
- **Action Blue Wash** (#eaf3ff): selected or active backgrounds and info notes.

### Neutral
- **Canvas** (#f5f5f7): the page background behind every card.
- **Surface** (#ffffff): cards, forms, the mobile top bar, the login form panel.
- **Ink** (#1d1d1f): headings and primary text.
- **Ink Secondary** (#424245): secondary text and help summaries.
- **Ink Muted** (#6e6e73): captions, timestamps, helper text.
- **Hairline** (#e5e5ea): card borders and dividers.
- **Field Stroke** (#d2d2d7): input and select borders.

### Status (attendance truth)
- **Present Green** (#047857 on #ecfdf5): "มา", success, the open-class pulse.
- **Late Amber** (#b45309 on #fffbeb): "สาย", warnings such as "turn on Location".
- **Error Rose** (#be123c on #fff1f2): failed check-in, errors, destructive actions such as log out on hover.

### Named Rules
**The One Blue Rule.** Every tappable primary action is Action Blue. The legacy indigo (#4f46e5, `--primary` in `style.css`, the `brand` scale) is retired. Any screen still showing it is a bug to fix, not a second accent.

**The Status-Only Rule.** Green, amber and rose appear only when they mean present, late or failed (or success, warning, error). Never use them as decoration.

## Typography

**Body Font:** Noto Sans Thai (with system-ui, sans-serif)
**Mono Font:** JetBrains Mono (with ui-monospace), for codes, room IDs and timestamps that need fixed width

**Character:** one humanist Thai sans for everything, with weight and size doing the hierarchy. Mono appears only where characters must line up or be read exactly, such as room codes.

### Hierarchy
- **Headline** (600, clamp(1.65rem, 3vw, 2.2rem), 1.45): page titles (`main h1`). Thai needs the tall line height for its upper vowels and tone marks.
- **Title** (600, 1rem, 1.5): card and section titles, course codes in lists.
- **Body** (400, 0.875rem, 1.6): content and descriptions.
- **Label** (500, 0.75rem, 1.5): field labels, table headers, badges and meta lines.
- **Mono** (500, 0.875rem): room codes, the session code, technical IDs.

### Named Rules
**The Thai Line-Height Rule.** Never set Thai text below line-height 1.45, or upper vowels and tone marks collide with the line above.

**The Phone Floor Rule.** On student screens, any text a student must read to act is at least 0.875rem. This includes errors, the next step and status. Use 0.75rem only for meta that no one needs to act on.

## Layout

- **Signed-in pages:**
  - Desktop (lg and up): a fixed 256px frosted sidebar on the left.
  - Phones: a sticky white top bar with a menu button.
- **Content:** a centered `max-w-5xl` column, padded 16px on phones and 32px on desktop, with 48px top padding on desktop and 28px on tablets.
- **Login:**
  - Desktop: a two-column screen, with a gray intro panel on the left and the white form panel on the right.
  - Phones: the intro panel is hidden.
- **Spacing rhythm:** 4px Tailwind steps. Stacks inside cards use 16–24px; cards are 26px inside (20px on phones).
- **Tables:** scroll sideways inside their card on small screens; the page itself never scrolls sideways.

## Elevation & Depth

Depth comes mostly from tonal layering: white surfaces on a gray canvas, separated by a hairline border. Shadows are ambient and nearly invisible.

### Shadow Vocabulary
- **Card ambient** (`box-shadow: 0 4px 24px rgb(0 0 0 / 3%)`): every bordered card.
- **Login symbol** (`box-shadow: 0 12px 40px rgb(0 0 0 / 5%)`): the one lifted object, on the login intro.
- **Sidebar glass** (`backdrop-filter: blur(24px)` over rgb(250 250 252 / 94%)): the sidebar only.

### Named Rules
**The Flat Card Rule.** Cards never get stronger shadows on hover. State is shown with color (blue wash, border) rather than lift.

## Shapes

Generous, friendly corners, scaled by object size:
- 22px for cards and large containers;
- 20px for camera frames and help panels;
- 12px for buttons, inputs and nav items;
- full pills for status badges and avatar circles.

The camera stage uses a soft oval face guide with a fill ring. It is the only non-rectangular silhouette in the system.

## Components

### Buttons
- **Shape:** gently rounded (12px), at least 44px tall.
- **Primary:** Action Blue fill, white text, medium weight. On a phone, a form's main action is full width.
- **Hover / Pressed:** darkens to Action Blue Deep (`indigo-700`, then `indigo-800` when pressed). The focus ring is a 3px Action Blue outline, offset 3px.
- **Secondary:** white with a Hairline border and Ink Secondary text (for example "ดูผล").
- **Disabled:** Canvas or slate-100 fill with muted text and a not-allowed cursor.

### Cards / Containers
- **Corner Style:** 22px.
- **Background:** Surface on Canvas.
- **Shadow Strategy:** card ambient only (see Elevation & Depth).
- **Border:** 1px Hairline.
- **Internal Padding:** 26px (20px on phones).

### Inputs / Fields
- **Style:** white, 1px Field Stroke, 12px corners, at least 44px tall; 16px text on phones so iOS does not zoom.
- **Focus:** a 2px Action Blue ring, and the border goes transparent.
- **Icons:** an optional leading 16px line icon in muted ink.

### Status badges
- **Style:** pill, wash background, matching deep text, 0.75rem medium weight.
- **Variants:** "เปิดอยู่" / "มา" (green), "สาย" (amber), "ไม่สำเร็จ" (rose), "เสร็จแล้ว" / "ยังไม่เริ่ม" (slate).

### Navigation
- **Sidebar links:** 44px rows with 12px corners, a 16px line icon and a 14px label.
- **Active link:** #e6edf7 background, Action Blue Deep text, semibold.
- **Phones:** the sidebar slides in over a 40% black overlay.

### Check-in camera stage (signature)
- A dark 3:4 stage with mirrored video.
- An SVG oval face guide that turns green or red.
- A progress ring that fills while the face is held still.
- A thin progress bar while the server checks the face.
- It must look ready the moment it appears: no spinner over a live camera.

## Do's and Don'ts

### Do:
- **Do** use Action Blue (#0071e3) for every primary action and nothing else.
- **Do** keep every tappable control at least 44px tall, and a phone form's main action full width.
- **Do** write every error as the fix, in Thai: what to turn on, where to stand, what to press.
- **Do** keep Thai text at line-height 1.45 or more.
- **Do** rebuild `app/static/css/tailwind.css` (`npm run build` in `tools/tailwind`) after adding a Tailwind class.

### Don't:
- **Don't** reintroduce the legacy indigo (#4f46e5) or a second accent color.
- **Don't** use green, amber or rose for anything but attendance or result status.
- **Don't** add hover lift or heavy shadows to cards.
- **Don't** put text a student needs to act on below 0.875rem on student screens.
- **Don't** change element ids used by `checkin_flow.js`, `enrollment_v2.js` or `push_notifications.js` while restyling.

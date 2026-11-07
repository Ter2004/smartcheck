# SmartCheck poster (A1 portrait, English)

- Source: [poster.html](poster.html) · Output: [SmartCheck_poster_A1.pdf](SmartCheck_poster_A1.pdf) (594 × 841 mm, 1 page)
- Phone screens in `assets/` are real screenshots of the app's pages rendered with sample data (`TEST1`, `std1`).
- Structure follows the College of Computing example poster: Motivation / Goal, How a check-in works, Security and privacy, System overview, Results, Conclusion, Acknowledgements and references.

## Still to do
1. **PSU / College of Computing logo** — replace the dashed box in the header (`<div class="logo">`) with `<img src="assets/psu-logo.png" style="width:100%">` using the official file from the faculty.
2. **Field test charts** — after the test, run `scripts/field_test/pull_results.py` and `make_charts.py` (see `docs/field-test/kit.html`, last page), then replace the three `<div class="chart">` boxes with
   `<img src="../docs/field-test/results/<date>/class_time.svg">`, `per_student.svg`, `projection.svg`,
   and add the field-test numbers (whole-class time vs roll call, first-try success, R4 security results) to the Results table.
3. **Conclusion** — remove "(to be confirmed by the field test)" only if the test confirms it; otherwise rewrite it to what the test showed.
4. Check the faculty's required size/format (A1 vs A0, template) before printing.

## Print to PDF (Chrome, no extra software)
```
"C:\Program Files\Google\Chrome\Application\chrome.exe" --headless=new --no-pdf-header-footer ^
  --virtual-time-budget=15000 --print-to-pdf=poster\SmartCheck_poster_A1.pdf poster\poster.html
```
Fonts (Anuphan, Noto Sans) load from Google Fonts, so print while online. Check that the PDF is 1 page.

## Every number on the poster and where it comes from
| Poster claim | Source |
|---|---|
| Notification ≈ 1 s after "open class" (2 phones) | live test 2026-11-05/06, OPPO + Redmi |
| Board found 0.13–0.45 s | findRoom timing on the Redmi, 2026-11-02 |
| Server check-in 0.83–1.10 s (7 check-ins) | `[PERF] total` lines, checkin-diagnostics.log, 2026-11-05/06 |
| Tap → recorded 9–10 s (2 phones) | live test 2026-11-05/06 |
| FRR 0.84 %, FAR < 0.001 % at threshold 0.40 | LFW offline evaluation 2026-10-07 (6,409 images; celebrity photos, not classroom data) |
| Code changes every 5 s, accepted for 30 s | firmware counter step; `FRESH_SECONDS = 30` in app/services/broadcast_beacon.py |
| 7 templates from a 5-pose enrollment | PDPA notice / enrollment v2 |

"""Phase 3 front-end hardening, pinned at source level (no browser here)."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "app/templates"


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


class FrontendHardeningTests(unittest.TestCase):
    def test_toast_message_is_text_not_html(self):
        base = read("app/templates/base.html")
        self.assertNotIn("${message}", base)
        self.assertIn("textContent = message", base)

    def test_csv_preview_does_not_build_html_from_the_file(self):
        detail = read("app/templates/admin/course_detail.html")
        preview = detail[detail.index("function previewCSV"):]
        self.assertNotIn("innerHTML", preview)

    def test_no_template_value_is_written_into_inline_javascript(self):
        # Jinja's HTML escaping is undone by the browser before an on* handler
        # runs, so values go through data-* attributes and this.dataset instead.
        for page in TEMPLATES.rglob("*.html"):
            with self.subTest(page=page.name):
                self.assertIsNone(re.search(r'\son[a-z]+="[^"]*\{\{',page.read_text(encoding="utf-8")))

    def test_every_external_script_is_pinned_with_integrity(self):
        for page in TEMPLATES.rglob("*.html"):
            for tag in re.findall(r"<script[^>]+src=\"https?://[^>]+>", page.read_text(encoding="utf-8")):
                with self.subTest(page=page.name, tag=tag):
                    self.assertRegex(tag, r"@\d[\d.]*/", "version pinned in the URL")
                    self.assertIn('integrity="sha384-', tag)
                    self.assertIn('crossorigin="anonymous"', tag)

    def test_mediapipe_assets_use_the_pinned_version(self):
        for js in ("checkin_flow.js", "enrollment_flow.js", "mediapipe_liveness.js"):
            source = read(f"app/static/js/{js}")
            with self.subTest(js=js):
                self.assertNotIn("@mediapipe/face_mesh/${f}", source)

    def test_tailwind_is_a_prebuilt_local_stylesheet_last_in_head(self):
        base = read("app/templates/base.html")
        self.assertNotIn("cdn.tailwindcss.com", base)
        self.assertNotIn("cdn.tailwindcss.com", read("app/__init__.py"))
        self.assertNotIn("tailwind.config", base, "no Play CDN runtime")
        head = base[:base.index("</head>")]
        # Last shared stylesheet in <head>, where the runtime used to add its <style>:
        # same cascade. Only the theme sheets (pass.css, then the 2026-11-16 redesign.css
        # tokens) may follow it, on purpose.
        tail = head.rstrip().split("filename='css/tailwind.css') }}\">", 1)[1]
        tail = " ".join(line for line in tail.splitlines()
                        if "css/pass.css" not in line and "css/redesign.css" not in line)
        self.assertNotIn("<link", tail)
        self.assertNotIn("<style", tail)

    def test_every_template_class_is_in_the_built_css(self):
        # Fails when a template gains a Tailwind class and tools/tailwind was not
        # rebuilt (npm run build there). Classes may also come from the custom CSS.
        css = read("app/static/css/tailwind.css")
        custom = read("app/static/css/style.css") + read("app/static/css/modern.css") + read("app/static/css/pass.css")
        hooks = {"row-check", "todo"}   # JavaScript selectors/states, never styled
        escape = lambda t: re.sub(r'([:/.\[\]()%#!,=&@*+>~])', lambda m: chr(92) + m.group(1), t)
        missing = set()
        for page in TEMPLATES.rglob("*.html"):
            text = page.read_text(encoding="utf-8")
            custom += "".join(re.findall(r"<style[^>]*>(.*?)</style>", text, re.S))
            for attr in re.findall(r'class="([^"]*)"', text):
                for token in re.sub(r"\{\{.*?\}\}|\{%.*?%\}", " ", attr).split():
                    missing.add((token, page.name))
        missing = sorted((t, p) for t, p in missing if t not in hooks and "." + escape(t) not in css
                         and not re.search(r"\." + re.escape(t) + r"(?![\w-])", custom))
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()

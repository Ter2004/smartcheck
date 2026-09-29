"""Phase 3 front-end hardening, pinned at source level (no browser here)."""
import hashlib
import base64
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

    def test_tailwind_is_served_locally_and_unchanged(self):
        self.assertNotIn("cdn.tailwindcss.com", read("app/templates/base.html"))
        self.assertNotIn("cdn.tailwindcss.com", read("app/__init__.py"))
        digest = hashlib.sha384((ROOT / "app/static/vendor/tailwindcss-3.4.17.js").read_bytes()).digest()
        # Same bytes as https://cdn.tailwindcss.com/3.4.17 on 2026-09-29.
        self.assertEqual(base64.b64encode(digest).decode(),
                         "igm5BeiBt36UU4gqwWS7imYmelpTsZlQ45FZf+XBn9MuJbn4nQr7yx1yFydocC/K")


if __name__ == "__main__":
    unittest.main()

"""Every audit event_type the code writes is allowed by the audit_logs check (migration 20261027)."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class AuditEventTypeTests(unittest.TestCase):
    def test_code_event_types_are_in_the_check(self):
        used = set()
        for path in (ROOT / "app").rglob("*.py"):
            used |= set(re.findall(r'event_type="([a-z_]+)"', path.read_text(encoding="utf-8")))
        sql = (ROOT / "database" / "migrations" / "20261027_audit_event_types.sql").read_text(encoding="utf-8")
        check = sql.split("ADD CONSTRAINT", 1)[1].split(");", 1)[0]
        allowed = set(re.findall(r"'([a-z_]+)'", check))
        self.assertTrue(used, "no event types found in app/")
        self.assertEqual(used - allowed, set())


if __name__ == "__main__":
    unittest.main()

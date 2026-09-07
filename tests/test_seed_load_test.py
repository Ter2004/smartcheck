import importlib.util
import os
from pathlib import Path
from datetime import datetime
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('seed', Path(__file__).resolve().parents[1] / 'scripts/seed_load_test.py')
seed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seed)


class SeedTests(unittest.TestCase):
    def test_override_skips_lookup(self):
        with patch.object(seed, '_get_beacon_id') as lookup:
            self.assertEqual(seed._resolve_beacon_id(Mock(), 'manual-id'), 'manual-id')
            lookup.assert_not_called()

    def test_cli_overrides_environment(self):
        with patch.dict(os.environ, {'SEED_BEACON_ID': 'env-id'}):
            self.assertEqual(seed._parse_args([]).beacon_id, 'env-id')
            self.assertEqual(seed._parse_args(['--beacon-id', 'cli-id']).beacon_id, 'cli-id')

    def test_lookup_failure_has_manual_instructions(self):
        with patch.object(seed, '_get_beacon_id', side_effect=RuntimeError('Cloudflare 1101')):
            with self.assertRaisesRegex(seed.BeaconLookupError, '--beacon-id.*SEED_BEACON_ID'):
                seed._resolve_beacon_id(Mock())

    def test_missing_beacon_has_manual_instructions(self):
        sb = Mock()
        sb.table.return_value.select.return_value.ilike.return_value.execute.return_value.data = []
        with self.assertRaises(seed.BeaconLookupError):
            seed._resolve_beacon_id(sb)

    def test_today_is_computed_in_bangkok(self):
        sb, query = Mock(), Mock()
        sb.table.return_value = query
        for method in ['select', 'eq', 'gte', 'lt', 'insert']:
            getattr(query, method).return_value = query
        query.execute.side_effect = [Mock(data=[]), Mock(data=[{'id': 'session'}])]
        now = datetime(2026, 9, 7, 0, 15, tzinfo=seed.TZ_THAI)
        with patch.object(seed, 'datetime') as clock:
            clock.now.return_value = now
            seed._get_or_create_open_session(sb, 'course', 'beacon')
        row = query.insert.call_args.args[0]
        self.assertEqual(row['title'], 'LOADTEST101 Load Test 2026-09-07')
        self.assertEqual(row['start_time'], '2026-09-06T17:15:00+00:00')
        self.assertTrue(row['is_open'])
        query.gte.assert_called_with('start_time', '2026-09-06T17:00:00+00:00')
        query.lt.assert_called_with('start_time', '2026-09-07T17:00:00+00:00')

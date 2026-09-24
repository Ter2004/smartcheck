import math
from pathlib import Path
import tempfile
import unittest

from scripts.calibrate_thresholds import evaluate, metrics, read_pairs


class CalibrationTests(unittest.TestCase):
    def test_lfw_headers_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'pairs.txt'
            for header in ['1', '1 1']:
                path.write_text(header + '\nAlice 1 2\nAlice 1 Bob 1\n')
                pairs = read_pairs(path)
                self.assertEqual(pairs, [('Alice/Alice_0001.jpg', 'Alice/Alice_0002.jpg', True),
                                         ('Alice/Alice_0001.jpg', 'Bob/Bob_0001.jpg', False)])

    def test_rejects_bad_counts_and_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'pairs.txt'
            for content in ['2\nAlice 1 2\nAlice 1 Bob 1',
                            '1\n../Alice 1 2\nAlice 1 Bob 1']:
                path.write_text(content)
                with self.assertRaises(ValueError):
                    read_pairs(path)

    def test_rates_and_inclusive_threshold(self):
        result = metrics([(True, .8), (True, .6), (False, .8), (False, .2)], .8)
        self.assertEqual(result['false_accept_rate'], .5)
        self.assertEqual(result['false_reject_rate'], .5)
        self.assertIsNone(metrics([], .8)['false_accept_rate'])

    def test_extraction_is_cached_and_failures_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'a').write_bytes(b'a')
            calls = []
            def extract(data):
                calls.append(data)
                return [1.] + [0.] * 511
            scores, excluded, failures = evaluate(root, [('a', 'a', True), ('a', 'missing', False)],
                                                   extract, lambda a, b: 1.)
            self.assertEqual(len(calls), 1)
            self.assertEqual(scores, [(True, 1.)])
            self.assertEqual(excluded, {'impostor': 1})
            self.assertEqual(failures, {'FileNotFoundError': 1})

    def test_invalid_embeddings_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'a').write_bytes(b'a')
            for vector in ([math.nan] * 512, [0.] * 512, [1.] * 2):
                scores, excluded, _ = evaluate(root, [('a', 'a', True)],
                                                lambda data: vector, lambda a, b: 1.)
                self.assertEqual(scores, [])
                self.assertEqual(excluded, {'genuine': 1})


if __name__ == '__main__':
    unittest.main()

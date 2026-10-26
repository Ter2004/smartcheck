"""scripts/compare_models.py scoring on synthetic embeddings: the split mirrors enroll_v2 (3 frontal + 4 poses)."""
import os
import unittest

import numpy as np

from scripts import compare_models as cm


def unit(seed, base=None, noise=0.3):
    v = np.random.default_rng(seed).standard_normal(512).astype(np.float32)
    if base is not None:
        v = base + noise * v / np.linalg.norm(v)
    return v / np.linalg.norm(v)


NAMES = ["01_frontal_normal", "02_frontal_normal", "03_frontal_normal", "04_left_normal",
         "05_right_normal", "06_up_normal", "07_down_normal", "08_frontal_dark", "09_frontal_bright"]


def person(folder, seed):
    base = unit(seed)
    files = [os.path.join(folder, n + ".jpg") for n in NAMES]
    return files, {path: unit(seed * 100 + i, base) for i, path in enumerate(files)}


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        a, emb_a = person("A", 1)
        b, emb_b = person("B", 2)
        self.people = {"A": a, "B": b}
        self.embeddings = {**emb_a, **emb_b}

    def test_split_follows_production(self):
        rows, pairs, poses = cm.evaluate("arcface_clahe_off", self.people, self.embeddings)
        self.assertEqual(len(pairs), 2 * 3)                       # 3 frontal pairs per person
        self.assertEqual([p["pose"] for p in poses], list(cm.ENROLL_POSES) * 2)
        self.assertEqual(len(rows), 2 * 2 * 2)                    # 2 probes per person x 2 targets
        self.assertEqual({r["lighting"] for r in rows}, {"dark", "bright"})
        self.assertTrue(all(r["genuine"] == (r["probe_person"] == r["target_person"]) for r in rows))

    def test_scores_use_the_right_templates(self):
        rows, pairs, poses = cm.evaluate("arcface_clahe_off", self.people, self.embeddings)
        a = self.people["A"]
        e = self.embeddings
        frontal = np.stack([e[p] for p in a[:3]])
        self.assertAlmostEqual(pairs[0]["score"], round(float(e[a[0]] @ e[a[1]]), 4))
        self.assertAlmostEqual(poses[0]["score"], round(float((frontal @ e[a[3]]).max()), 4))
        # A probe equal to the "down" template scores 1.0: check-in takes the max over all 7.
        self.embeddings[a[8]] = e[a[6]]
        rows, _, _ = cm.evaluate("arcface_clahe_off", self.people, self.embeddings)
        probe = next(r for r in rows if r["probe_file"] == "09_frontal_bright.jpg" and r["genuine"])
        self.assertAlmostEqual(probe["score"], 1.0, places=4)

    def test_person_with_a_faceless_enrollment_image_is_skipped(self):
        del self.embeddings[self.people["B"][5]]                  # "up" had no face
        rows, pairs, poses = cm.evaluate("arcface_clahe_off", self.people, self.embeddings)
        self.assertEqual({p["person"] for p in pairs + poses}, {"A"})
        self.assertEqual({r["target_person"] for r in rows}, {"A"})

    def test_summary_has_both_enrollment_spreads(self):
        rows, pairs, poses = cm.evaluate("arcface_clahe_off", self.people, self.embeddings)
        s = cm.summarize(rows, pairs, poses, [10.0, 12.0], [], 18)
        self.assertEqual(s["enroll_frontal_pairwise"]["n"], 6)
        self.assertEqual(s["enroll_pose_vs_frontal"]["n"], 8)
        self.assertEqual(sorted(s["enroll_pose_vs_frontal_min_by_pose"]), sorted(cm.ENROLL_POSES))
        self.assertEqual(s["enroll_pose_vs_frontal"]["min"], min(p["score"] for p in poses))


class NameTests(unittest.TestCase):
    def test_misplaced_pose_words_are_reported(self):
        ok = {"A": [n + ".jpg" for n in NAMES]}
        self.assertEqual(cm.name_warnings(ok), [])
        swapped = {"A": [n + ".jpg" for n in NAMES[:3] + ["04_right_normal", "05_left_normal"] + NAMES[5:]]}
        self.assertEqual(len(cm.name_warnings(swapped)), 2)
        untagged = {"A": [f"{i:02d}_normal.jpg" for i in range(1, 10)]}   # no pose word: nothing to check
        self.assertEqual(cm.name_warnings(untagged), [])
        self.assertEqual(cm.lighting_tag("08_frontal_dark.jpg"), "dark")
        self.assertEqual(cm.pose_word("04_left_normal.jpg"), "left")


if __name__ == "__main__":
    unittest.main()

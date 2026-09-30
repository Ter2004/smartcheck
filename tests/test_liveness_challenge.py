import random
import unittest

import numpy as np
from app.services import liveness_challenge as lc

NOW = 1_000_000.0


def unit(seed):
    v = np.random.default_rng(seed).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


PERSON = unit(1)
OTHER = unit(2)


def near(vec, similarity):
    """A unit vector with the given cosine to vec."""
    noise = unit(99) - np.dot(unit(99), vec) * vec
    noise /= np.linalg.norm(noise)
    return similarity * vec + np.sqrt(1 - similarity ** 2) * noise


class FakeFrame:
    def __init__(self, yaw, embedding=PERSON, error=None, roll=0.0):
        self.yaw, self.embedding, self.error, self.roll = yaw, embedding, error, roll


def fake_analyze(frame):
    if frame.error:
        raise lc.FrameError(frame.error)
    return {"yaw": frame.yaw, "roll": frame.roll, "embedding": frame.embedding}


def challenge(actions=("turn_left", "turn_right")):
    return {"nonce": "n-1", "actions": list(actions), "issued_at": NOW}


def frames_for(actions, turned=0.35, tilted=20.0, base_roll=0.0):
    frames = []
    for a in actions:
        yaw = {"turn_left": turned, "turn_right": -turned}.get(a, 0.02)
        roll = base_roll + {"tilt_left": tilted, "tilt_right": -tilted}.get(a, 0.0)
        frames.append(FakeFrame(yaw, near(PERSON, 0.7), roll=roll))
    return frames


class ChallengeTests(unittest.TestCase):
    def check(self, ch=None, nonce="n-1", before=None, actions=None, after=None, now=NOW + 10):
        ch = ch or challenge()
        before = before or FakeFrame(0.02)
        actions = frames_for(ch["actions"]) if actions is None else actions
        after = after or FakeFrame(-0.03, near(PERSON, 0.95))
        return lc.verify(ch, nonce, before, actions, after, now=now, analyze=fake_analyze)

    def test_new_challenge_picks_distinct_gestures_from_actions(self):
        self.assertEqual(lc.ACTIONS, ("turn_left", "turn_right", "tilt_left", "tilt_right"))
        picks = [lc.new_challenge(now=NOW, rng=random.Random(s))["actions"] for s in range(200)]
        self.assertTrue(all(len(p) == lc.ENROLL_ACTIONS == 2 and len(set(p)) == 2 for p in picks))
        self.assertEqual({a for p in picks for a in p}, set(lc.ACTIONS))
        n = len(lc.ACTIONS)
        self.assertEqual(len({tuple(p) for p in picks}), n * (n - 1))  # every ordered pair
        single = {lc.new_challenge(now=NOW, rng=random.Random(s), count=lc.CHECKIN_ACTIONS)["actions"][0]
                  for s in range(200)}
        self.assertEqual(single, set(lc.ACTIONS))
        c = lc.new_challenge(now=NOW)
        self.assertGreaterEqual(len(c["nonce"]), 16)
        self.assertEqual(c["issued_at"], NOW)

    def test_yaw_sign_matches_browser_turn_left(self):
        lm = {"right_eye": (100, 100), "left_eye": (160, 100), "nose": (130, 130)}
        self.assertAlmostEqual(lc.yaw_ratio(lm), 0.0)
        lm["nose"] = (148, 130)  # nose toward image right
        self.assertAlmostEqual(lc.yaw_ratio(lm), 0.3)
        self.assertEqual(lc.direction(lc.yaw_ratio(lm)), "turn_left")
        mirrored = {"right_eye": (540, 100), "left_eye": (480, 100), "nose": (492, 130)}  # x -> 640-x
        self.assertEqual(lc.direction(lc.yaw_ratio(mirrored)), "turn_right")

    def test_roll_sign_and_tilt_does_not_read_as_turn(self):
        level = {"right_eye": (100, 100), "left_eye": (160, 100), "nose": (130, 130)}
        self.assertAlmostEqual(lc.roll_deg(level), 0.0)
        # Rotate the face 20 degrees so the image-right eye is lower (tilt_left).
        c, t = np.array([130.0, 100.0]), np.radians(20)
        rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        tilted = {k: tuple(rot @ (np.array(v, float) - c) + c) for k, v in level.items()}
        self.assertAlmostEqual(lc.roll_deg(tilted), 20.0, places=6)
        self.assertAlmostEqual(lc.yaw_ratio(tilted), 0.0, places=6)   # still frontal
        # The library's eye names do not matter: sorted by image x.
        swapped = {"right_eye": tilted["left_eye"], "left_eye": tilted["right_eye"], "nose": tilted["nose"]}
        self.assertAlmostEqual(lc.roll_deg(swapped), 20.0, places=6)

    def test_direction_bands(self):
        self.assertEqual(lc.direction(0.18), "frontal")
        self.assertEqual(lc.direction(0.19), "partial")
        self.assertEqual(lc.direction(0.20), "turn_left")
        self.assertEqual(lc.direction(-0.20), "turn_right")

    def test_passes_for_every_requested_pair(self):
        gestures = lc.ACTIONS + lc.TILT_ACTIONS   # tilts stay verified while not issued
        for first in gestures:
            for second in gestures:
                if first != second:
                    result = self.check(ch=challenge((first, second)))
                    self.assertTrue(result["passed"], (first, second, result))
                    self.assertEqual((len(result["yaws"]), len(result["rolls"])), (4, 4))

    def test_tilt_is_measured_from_the_before_frame(self):
        # Phone held at 15 degrees: the same absolute roll is not a gesture...
        ch = challenge(("tilt_left",))
        before = FakeFrame(0.02, roll=15.0)
        held = [FakeFrame(0.02, near(PERSON, 0.7), roll=15.0)]
        self.assertEqual(self.check(ch=ch, before=before, actions=held)["reason"],
                         "wrong_direction:action_1:level")
        # ...but tilting from there is.
        self.assertTrue(self.check(ch=ch, before=before,
                                   actions=frames_for(("tilt_left",), base_roll=15.0))["passed"])

    def test_tilt_threshold_and_direction(self):
        ch = challenge(("tilt_right",))
        small = frames_for(("tilt_right",), tilted=lc.TILT_MIN_DEG - 1)
        self.assertEqual(self.check(ch=ch, actions=small)["reason"], "wrong_direction:action_1:level")
        wrong = frames_for(("tilt_left",))
        self.assertEqual(self.check(ch=ch, actions=wrong)["reason"], "wrong_direction:action_1:tilt_left")
        self.assertTrue(self.check(ch=ch, actions=frames_for(("tilt_right",), tilted=lc.TILT_MIN_DEG))["passed"])

    def test_turning_instead_of_tilting_fails(self):
        ch = challenge(("tilt_left",))
        turned = [FakeFrame(lc.TILT_MAX_YAW + 0.1, near(PERSON, 0.7), roll=20.0)]
        self.assertEqual(self.check(ch=ch, actions=turned)["reason"], "wrong_direction:action_1:turned")

    def test_tilting_instead_of_turning_fails(self):
        ch = challenge(("turn_left",))
        tilted = [FakeFrame(0.02, near(PERSON, 0.7), roll=25.0)]
        self.assertEqual(self.check(ch=ch, actions=tilted)["reason"], "wrong_direction:action_1:frontal")

    def test_still_image_fails(self):
        still = FakeFrame(0.01)
        result = self.check(before=still, actions=[still, still], after=still)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "wrong_direction:action_1:frontal")

    def test_wrong_order_fails(self):
        result = self.check(ch=challenge(("turn_left", "turn_right")),
                            actions=frames_for(("turn_right", "turn_left")))
        self.assertEqual(result["reason"], "wrong_direction:action_1:turn_right")

    def test_partial_turn_fails(self):
        result = self.check(actions=frames_for(("turn_left", "turn_right"), turned=0.19))
        self.assertTrue(result["reason"].startswith("wrong_direction:action_1:partial"))

    def test_before_frame_must_be_frontal(self):
        self.assertEqual(self.check(before=FakeFrame(0.3))["reason"], "not_frontal:before")

    def test_turned_frame_from_another_person_fails(self):
        actions = frames_for(("turn_left", "turn_right"))
        actions[1] = FakeFrame(-0.35, near(OTHER, 0.9))
        self.assertEqual(self.check(actions=actions)["reason"], "identity_mismatch:action_2")

    def test_frontal_after_frame_needs_continuity_threshold(self):
        after = FakeFrame(0.0, near(PERSON, lc.FRONTAL_IDENTITY_MIN - 0.02))
        self.assertEqual(self.check(after=after)["reason"], "identity_mismatch:after")
        # Still turned after the last gesture: judged with the cross-pose threshold.
        after = FakeFrame(0.3, near(PERSON, lc.TURN_IDENTITY_MIN + 0.02))
        self.assertTrue(self.check(after=after)["passed"])

    def test_nonce_expiry_and_frame_count(self):
        self.assertEqual(self.check(nonce="other")["reason"], "nonce_mismatch")
        self.assertEqual(self.check(nonce=None)["reason"], "nonce_mismatch")
        self.assertEqual(self.check(now=NOW + lc.CHALLENGE_TTL_S + 1)["reason"], "expired")
        self.assertEqual(self.check(now=NOW - 5)["reason"], "expired")
        self.assertEqual(self.check(actions=frames_for(("turn_left",)))["reason"], "frame_count")
        self.assertEqual(lc.verify(None, "n-1", None, [], None, analyze=fake_analyze)["reason"], "no_challenge")

    def test_frame_errors_name_the_frame(self):
        actions = frames_for(("turn_left", "turn_right"))
        actions[0] = FakeFrame(0, error="no_face")
        self.assertEqual(self.check(actions=actions)["reason"], "no_face:action_1")
        self.assertEqual(self.check(before=FakeFrame(0, error="multiple_faces"))["reason"],
                         "multiple_faces:before")


if __name__ == "__main__":
    unittest.main()

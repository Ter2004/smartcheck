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
    def __init__(self, yaw, embedding=PERSON, error=None):
        self.yaw, self.embedding, self.error = yaw, embedding, error


def fake_analyze(frame):
    if frame.error:
        raise lc.FrameError(frame.error)
    return {"yaw": frame.yaw, "embedding": frame.embedding}


def challenge(actions=("turn_left", "turn_right")):
    return {"nonce": "n-1", "actions": list(actions), "issued_at": NOW}


def frames_for(actions, turned=0.35):
    return [FakeFrame(turned if a == "turn_left" else -turned, near(PERSON, 0.7)) for a in actions]


class ChallengeTests(unittest.TestCase):
    def check(self, ch=None, nonce="n-1", before=None, actions=None, after=None, now=NOW + 10):
        ch = ch or challenge()
        before = before or FakeFrame(0.02)
        actions = frames_for(ch["actions"]) if actions is None else actions
        after = after or FakeFrame(-0.03, near(PERSON, 0.95))
        return lc.verify(ch, nonce, before, actions, after, now=now, analyze=fake_analyze)

    def test_new_challenge_uses_both_turns_in_random_order(self):
        orders = {tuple(lc.new_challenge(now=NOW, rng=random.Random(s))["actions"]) for s in range(20)}
        self.assertEqual(orders, {("turn_left", "turn_right"), ("turn_right", "turn_left")})
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

    def test_direction_bands(self):
        self.assertEqual(lc.direction(0.18), "frontal")
        self.assertEqual(lc.direction(0.19), "partial")
        self.assertEqual(lc.direction(0.20), "turn_left")
        self.assertEqual(lc.direction(-0.20), "turn_right")

    def test_passes_for_requested_turns_in_order(self):
        for order in (("turn_left", "turn_right"), ("turn_right", "turn_left")):
            result = self.check(ch=challenge(order))
            self.assertTrue(result["passed"], result)
            self.assertEqual(len(result["yaws"]), 4)

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

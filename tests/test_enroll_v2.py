"""POST /student/api/enroll under FACE_ENGINE=arcface (v2), with the face model mocked per frame."""
import json
import random
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from flask import Flask

import app
from app.routes import student
from app.services import enroll_v2, face_engine, face_service, face_store, security_service
from scripts import report_turn_yaw

HEADERS = {"X-CSRF-Token": "csrf"}
CONFIG = {"FACE_ENGINE": "arcface", "FACE_CLAHE": "off", "ARCFACE_MATCH_THRESHOLD": "0.4",
          "ARCFACE_DUP_THRESHOLD": "0.5", "ENROLL_SELF_SIM_MIN": "0.6",   # test values, not production
          "EMBEDDING_INTEGRITY_SALT": "salt"}


def unit(seed, base=None, noise=0.15):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    if base is not None:
        v = base + noise * v / np.linalg.norm(v)
    return (v / np.linalg.norm(v)).astype(np.float32)


PERSON_A = unit(1)
PERSON_B = unit(2)
ORDER = ("up", "left", "down", "right")      # the order the mocked challenge issues
# Server angles per pose (yaw_deg + = turn_left; pitch_deg + = up) and what the page reports.
ANGLES = {"frontal": (2.0, 0.0), "left": (28.0, 0.0), "right": (-28.0, 0.0), "up": (2.0, 10.0), "down": (2.0, -10.0)}
CLIENT = {"frontal": (0.02, None), "left": (0.41, None), "right": (-0.41, None), "up": (0.02, 0.08), "down": (0.02, -0.08)}


def table_mock(data):
    q = Mock()
    for name in ("select", "eq", "order", "limit", "maybe_single", "update", "insert", "delete", "not_", "neq", "in_"):
        getattr(q, name).return_value = q
    q.execute.return_value = SimpleNamespace(data=data)
    return q


class EnrollV2RouteTests(unittest.TestCase):
    def setUp(self):
        self.web = Flask(__name__)
        self.web.config.update(SECRET_KEY="test", RATELIMIT_ENABLED=False, **CONFIG)
        app.limiter.init_app(self.web)
        self.web.register_blueprint(student.student_bp, url_prefix="/student")
        self.client = self.web.test_client()
        with self.client.session_transaction() as s:
            s.update(user_id="student-1", user_role="student", csrf_token="csrf",
                     consent_given_at="2026-10-19T00:00:00Z")
        self.enrollment = None                   # face_enrollments row seen by is_enrolled
        self.tables = {"consent_logs": table_mock([{"id": "consent-9", "consent_given": True}])}
        self.db = Mock()
        self.db.table.side_effect = lambda name: self.tables.setdefault(name, table_mock(self.enrollment))
        self.db.rpc.return_value.execute.return_value = SimpleNamespace(
            data=[{"allowed": True, "current_attempts": 1}])
        self.saved = []
        self.matches = []
        self.audits = []
        self.faces = {}                           # frame code -> analyze() result or exception
        self.codes = {}                           # image string -> frame code (its pixel value)
        for target in (
            patch.object(student, "supabase_admin", self.db),
            patch.object(face_service, "server_validate_frame", return_value={"valid": True, "reason": "passed", "metadata": {}}),
            patch.object(face_service, "_decode_image",
                         side_effect=lambda b64: np.full((480, 640, 3), self.codes[b64], np.uint8)),
            patch.object(face_engine, "analyze", side_effect=self.analyze),
            patch.object(enroll_v2, "quality", return_value=(None, {"blur": 50.0})),
            patch.object(face_service, "combined_spoof_score", return_value={"is_real": True}),
            patch.object(face_store, "is_enrolled", side_effect=lambda c, sid, mv: bool(self.enrollment)),
            patch.object(face_store, "match_duplicates", side_effect=lambda c, emb, mv, sid, limit=3: self.matches),
            patch.object(face_store, "complete_enrollment", side_effect=self.complete),
            patch.object(security_service, "log_audit_event", side_effect=lambda *a, **k: self.audits.append(k)),
        ):
            target.start()
            self.addCleanup(target.stop)

    # Each test frame decodes to a flat image whose pixel value is its code.
    def analyze(self, img):
        result = self.faces[int(img[0, 0, 0])]
        if isinstance(result, Exception):
            raise result
        return result

    def complete(self, client, student_id, model_version, consent_id, integrity_hash, thumbnail, templates):
        self.saved.append(dict(student_id=student_id, model_version=model_version, consent_id=consent_id,
                               integrity_hash=integrity_hash, templates=templates))
        return "enrollment-1"

    def face(self, name, yaw, emb, pitch=0.0, det=0.85):
        code = self.codes.setdefault(name, len(self.codes) + 1)
        self.faces[code] = {"embedding": emb, "det_score": det, "face_px": 300, "yaw_deg": yaw,
                            "pitch_deg": pitch, "roll_deg": 0.0, "eye_roll_deg": 0.0}
        return name

    def challenge(self, order=ORDER):
        with patch.object(enroll_v2.secrets.SystemRandom, "sample", return_value=list(order)):
            r = self.client.post("/student/api/enroll/challenge", headers=HEADERS)
        self.assertEqual(r.status_code, 200, r.json)
        self.assertEqual(r.json["order"], list(order))
        return r.json["challenge_id"]

    def frames(self, order=ORDER, angles=None, emb=None, n=None):
        """4 frontal frames, then 2 per pose in `order`, as the page captures them; overrides per pose."""
        angles = {**ANGLES, **(angles or {})}
        n = {"frontal": 4, **{pose: 2 for pose in enroll_v2.POSES}, **(n or {})}
        out, t = [], 500
        for k, pose in enumerate(["frontal"] + list(order)):
            for i in range(n[pose]):
                e = (emb or {}).get(pose)
                e = unit(10 + 10 * k + i, PERSON_A) if e is None else e
                yaw, pitch = angles[pose]
                out.append({"image": self.face(f"{pose}{i}", yaw, e, pitch), "t": t, "pose": pose,
                            "client_yaw": CLIENT[pose][0], "client_pitch": CLIENT[pose][1]})
                t += 200
            t += 1000
        return out

    def enroll(self, challenge_id, frames):
        return self.client.post("/student/api/enroll", json={"challenge_id": challenge_id, "frames": frames},
                                headers=HEADERS)

    def reset_retries(self):
        with self.client.session_transaction() as s:
            s.pop("enroll_retry", None)

    # ── scenarios ─────────────────────────────────────────────────────────────
    def test_five_poses_enroll_7_templates_in_challenge_order(self):
        r = self.enroll(self.challenge(), self.frames())
        self.assertEqual(r.status_code, 200, r.json)
        self.assertEqual(r.json["status"], "enrolled")
        saved = self.saved[0]
        self.assertEqual(saved["model_version"], "buffalo_l/w600k_r50")
        self.assertEqual(saved["consent_id"], "consent-9")
        self.assertEqual([t["pose"] for t in saved["templates"]], ["frontal"] * 3 + list(ORDER))
        expected = security_service.compute_template_integrity_hash(
            "student-1", "buffalo_l/w600k_r50", [t["embedding"] for t in saved["templates"]], "salt")
        self.assertEqual(saved["integrity_hash"], expected)
        self.db.storage.from_.assert_called_with("face-images")

    def test_wrong_turn_direction(self):
        r = self.enroll(self.challenge(), self.frames(angles={"left": (-28.0, 0.0)}))
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json["status"], "retry")
        self.assertEqual(r.json["fix"], "หันผิดทาง — กรุณาหันซ้ายตามลูกศร")
        self.assertEqual(self.saved, [])

    def test_up_and_down_need_the_right_pitch_direction(self):
        r = self.enroll(self.challenge(), self.frames(angles={"up": (2.0, -10.0)}))      # looked down instead
        self.assertEqual(r.json["fix"], "เงยหน้าอีกนิดตามลูกศร")
        r = self.enroll(self.challenge(), self.frames(angles={"down": (2.0, 10.0)}))     # looked up instead
        self.assertEqual(r.json["fix"], "ก้มหน้าอีกนิดตามลูกศร")
        self.assertEqual(self.saved, [])

    def test_up_down_with_a_turned_head_do_not_count(self):
        r = self.enroll(self.challenge(), self.frames(angles={"up": (28.0, 10.0)}))
        self.assertEqual(r.json["fix"], "เงยหน้าอีกนิดตามลูกศร")
        self.assertEqual(self.saved, [])

    def test_static_photo_has_no_poses(self):
        # The same still face in every frame: no pose reads as turned, raised or lowered.
        still = unit(10, PERSON_A)
        r = self.enroll(self.challenge(), self.frames(angles={p: (2.0, 0.0) for p in enroll_v2.POSES},
                                                      emb={p: still for p in ("frontal",) + enroll_v2.POSES}))
        self.assertEqual(r.json["status"], "retry")
        self.assertEqual(r.json["fix"], "เงยหน้าอีกนิดตามลูกศร")      # ORDER starts with up
        self.assertEqual(self.saved, [])

    def test_unknown_or_reused_challenge(self):
        self.assertEqual(self.enroll("not-the-challenge", self.frames()).json["fix"], enroll_v2.FIX["challenge"])
        cid = self.challenge()
        self.assertEqual(self.enroll(cid, self.frames()).status_code, 200)
        self.enrollment = None
        self.assertEqual(self.enroll(cid, self.frames()).json["fix"], enroll_v2.FIX["challenge"])
        self.assertEqual(len(self.saved), 1)

    def test_capture_order_must_follow_the_challenge(self):
        late_frontal = self.frames()
        late_frontal[0]["t"] = 60000                     # a frontal frame after the poses
        self.assertEqual(self.enroll(self.challenge(), late_frontal).json["fix"], enroll_v2.FIX["order"])
        self.reset_retries()
        swapped = ("left", "up", "down", "right")         # captured in another order than issued
        self.assertEqual(self.enroll(self.challenge(), self.frames(order=swapped)).json["fix"], enroll_v2.FIX["order"])
        self.reset_retries()
        no_down = self.frames(n={"down": 0})
        self.assertEqual(self.enroll(self.challenge(), no_down).json["fix"], enroll_v2.FIX["order"])
        self.assertEqual(self.saved, [])

    def test_person_swap_mid_capture(self):
        r = self.enroll(self.challenge(), self.frames(emb={"down": unit(30, PERSON_B)}))
        self.assertEqual(r.json["fix"], enroll_v2.FIX["inconsistent"])
        self.assertEqual(self.saved, [])

    def test_same_face_into_second_account_is_blocked_without_revealing_it(self):
        self.matches = [("student-OTHER", 0.93)]
        r = self.enroll(self.challenge(), self.frames())
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json["status"], "blocked")
        self.assertTrue(r.json["contact_staff"])
        self.assertEqual(r.json["message"], enroll_v2.FIX["duplicate"])
        self.assertNotIn("student-OTHER", json.dumps(r.json))
        self.assertEqual(self.audits[0]["event_type"], "face_duplicate_blocked")
        self.assertEqual(self.audits[0]["metadata"]["matched_student"], "student-OTHER")
        self.assertEqual(self.saved, [])

    def test_second_person_in_frame(self):
        frames = self.frames()
        self.faces[self.codes[frames[1]["image"]]] = face_engine.MultipleFacesError()
        r = self.enroll(self.challenge(), frames)
        self.assertEqual(r.json["fix"], enroll_v2.FIX["multiple_faces"])

    def test_overwrite_attempt(self):
        self.enrollment = {"status": "completed", "model_version": "buffalo_l/w600k_r50"}
        r = self.client.post("/student/api/enroll/challenge", headers=HEADERS)
        self.assertEqual(r.status_code, 409)
        r = self.enroll("x", self.frames())
        self.assertEqual(r.status_code, 409)
        # A race past the status check still meets the database trigger / RPC refusal.
        self.enrollment = None
        cid = self.challenge()
        with patch.object(face_store, "complete_enrollment", side_effect=Exception("already_enrolled")):
            self.assertEqual(self.enroll(cid, self.frames()).status_code, 409)

    def test_spoof_frames_fail(self):
        with patch.object(face_service, "combined_spoof_score", return_value={"is_real": False}):
            r = self.enroll(self.challenge(), self.frames())
        self.assertEqual(r.json["fix"], enroll_v2.FIX["spoof"])

    def test_three_failures_send_student_to_staff(self):
        for attempt in range(3):
            r = self.enroll(self.challenge(), self.frames(angles={"left": (-28.0, 0.0)}))
        self.assertTrue(r.json["contact_staff"])
        self.assertIn("ผู้ช่วยสอน", r.json["message"])

    def test_pose_angles_are_logged_for_the_report(self):
        with self.assertLogs(student._audit, "INFO") as logs:
            self.assertEqual(self.enroll(self.challenge(), self.frames()).status_code, 200)
        rows = report_turn_yaw.parse(logs.output)
        self.assertEqual(sorted({r["pose"] for r in rows}), sorted(enroll_v2.POSES))
        up = next(r for r in rows if r["pose"] == "up")
        self.assertEqual((up["client_pitch"], up["server_pitch"], up["seen"]), (0.08, 10.0, "frontal"))
        left = next(r for r in rows if r["pose"] == "left")
        self.assertEqual((left["client_yaw"], left["server_yaw"], left["seen"]), (0.41, 28.0, "turn_left"))

    def test_withdrawn_consent_blocks(self):
        self.tables["consent_logs"] = table_mock([{"id": "c", "consent_given": False}])
        r = self.enroll(self.challenge(), self.frames())
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.saved, [])

    def test_withdrawal_deletes_v2_enrollment(self):
        self.db.storage.from_.return_value.remove.return_value = None
        r = self.client.post("/student/api/withdraw-consent", headers=HEADERS)
        self.assertEqual(r.status_code, 200, r.json)
        self.tables["face_enrollments"].delete.assert_called_once()
        self.assertIn("face_templates", r.json["deleted"])


class ChallengeTests(unittest.TestCase):
    def test_orders_are_random_permutations_of_the_four_poses(self):
        challenges = [enroll_v2.new_challenge(now=0, rng=random.Random(seed)) for seed in range(60)]
        self.assertTrue(all(sorted(c["order"]) == sorted(enroll_v2.POSES) for c in challenges))
        self.assertGreater(len({tuple(c["order"]) for c in challenges}), 10)
        self.assertEqual(len({c["nonce"] for c in challenges}), 60)


class QualityGateTests(unittest.TestCase):
    def test_gates_on_real_frames(self):
        dark = np.full((480, 640, 3), 30, np.uint8)
        self.assertEqual(enroll_v2.quality(dark)[0], "too_dark")
        flat = np.full((480, 640, 3), 128, np.uint8)
        self.assertEqual(enroll_v2.quality(flat)[0], "blurry")
        backlit = np.full((480, 640, 3), 230, np.uint8)
        backlit[120:360, 160:480] = 60
        self.assertEqual(enroll_v2.quality(backlit)[0], "backlight")
        good = cv2.imread("test_images/real_05_sharp_face2_fit.jpg")
        self.assertIsNone(enroll_v2.quality(good)[0], enroll_v2.quality(good)[1])


if __name__ == "__main__":
    unittest.main()

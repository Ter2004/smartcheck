import ast
import base64
import inspect
import json
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from app.routes import student  # noqa: E402


def _undecorated(func):
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    return func


def _jpeg_data_url(image):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise RuntimeError("test JPEG encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


class _Result:
    def __init__(self, data):
        self.data = data


class _Query:
    def __init__(self, owner, table_name):
        self.owner = owner
        self.table_name = table_name

    def __getattr__(self, _name):
        return lambda *args, **kwargs: self

    @property
    def not_(self):
        return self

    def execute(self):
        if self.table_name == "consent_logs":
            return _Result([{"consent_given": True}])
        return _Result([])


class _StorageBucket:
    def upload(self, *args, **kwargs):
        return _Result({})


class _Storage:
    def from_(self, _name):
        return _StorageBucket()


class _Supabase:
    storage = _Storage()

    def table(self, name):
        return _Query(self, name)

    def rpc(self, *args, **kwargs):
        return _RpcQuery()


class _RpcQuery:
    def execute(self):
        return _Result([{"current_attempts": 1, "allowed": True}])


class SpoofCheckAuditPolicyTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.secret_key = "test-only"
        self.app.add_url_rule(
            "/probe", "probe", _undecorated(student.api_spoof_check), methods=["POST"]
        )
        self.app.config["ENROLL_FLOW_MODE"] = "classic"
        self.app.config["EMBEDDING_INTEGRITY_SALT"] = "test-salt"
        self.app.add_url_rule(
            "/enroll-probe", "enroll_probe", _undecorated(student.api_enroll), methods=["POST"]
        )
        self.client = self.app.test_client()
        with self.client.session_transaction() as sess:
            sess["user_id"] = "test-student"
            sess["consent_given_at"] = "2026-08-27T00:00:00+00:00"
            sess["liveness_embeddings"] = [[1.0, 0.0]]

        # Identical frames deliberately produce temporal variance 0. Validation
        # is mocked because these tests exercise route control flow, not JPEG quality.
        self.image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.payload = {"image": _jpeg_data_url(self.image)}

    def _enroll_patches(self, *, moire=None, texture=None, temporal=None):
        from app.services import face_service

        stack = ExitStack()
        combined = Mock(return_value={
            "is_real": True, "combined_score": 0.1, "layers": {}, "disagreements": []
        })
        audit_log = Mock()
        embedding = [1.0, 0.0]
        stack.enter_context(patch.object(student, "supabase_admin", _Supabase()))
        stack.enter_context(patch.object(student, "_log", audit_log))
        stack.enter_context(patch.object(face_service, "server_validate_frame", return_value={
            "valid": True, "reason": "passed", "metadata": {}
        }))
        stack.enter_context(patch.object(face_service, "_decode_image", return_value=self.image))
        stack.enter_context(patch.object(
            face_service, "detect_screen_moire",
            side_effect=moire if isinstance(moire, Exception) else None,
            return_value={"is_screen": False, "avg_score": 0.1},
        ))
        stack.enter_context(patch.object(
            face_service, "detect_screen_texture",
            side_effect=texture if isinstance(texture, Exception) else None,
            return_value=False,
        ))
        stack.enter_context(patch.object(
            face_service, "detect_static_image",
            side_effect=temporal if isinstance(temporal, Exception) else None,
            return_value=temporal if isinstance(temporal, dict) else {
                "is_static": True, "temporal_variance": 3.609
            },
        ))
        stack.enter_context(patch.object(face_service, "combined_spoof_score", combined))
        stack.enter_context(patch.object(
            face_service, "extract_embedding",
            return_value=(embedding, {"detector_crop": {"x": 10, "y": 20, "w": 100, "h": 120}}),
        ))
        stack.enter_context(patch.object(face_service, "check_embedding_consistency", return_value={
            "consistent": True,
            "outlier_indices": [],
            "pairwise_scores": [{"i": 0, "j": 1, "score": 1.0}],
            "average_similarities": [{"frame": 1, "average": 1.0}],
            "embedding_diagnostics": [{
                "frame": 1, "shape": [2], "dtype": "float64", "l2_norm": 1.0,
            }],
            "multi_outlier": False,
        }))
        stack.enter_context(patch.object(face_service, "max_similarity_multi", return_value=0.0))
        stack.enter_context(patch.object(student, "cosine_similarity", return_value=1.0))
        stack.enter_context(patch.object(student, "compute_embedding_integrity_hash", return_value="hash"))
        return stack, combined, audit_log

    def _enroll(self, ear_std=0.01):
        return self.client.post("/enroll-probe", json={
            "face_images": [self.payload["image"]] * 5,
            "baseline_ear": 0.25,
            "ear_std": ear_std,
        })

    def _route_patches(self, *, moire=None, texture=None, validation=None):
        from app.services import face_service

        stack = ExitStack()
        downstream = Mock(return_value={
            "is_real": True,
            "confidence": 0.99,
            "embedding": None,
            "message": "",
        })
        audit_log = Mock()
        stack.enter_context(patch.object(
            face_service, "server_validate_frame",
            return_value=validation or {"valid": True, "reason": "passed", "metadata": {}},
        ))
        stack.enter_context(patch.object(face_service, "_decode_image", return_value=self.image))
        stack.enter_context(patch.object(
            face_service, "detect_screen_moire",
            side_effect=moire if isinstance(moire, Exception) else None,
            return_value=moire if isinstance(moire, dict) else {"is_screen": False, "avg_score": 0.1},
        ))
        stack.enter_context(patch.object(
            face_service, "detect_screen_texture",
            side_effect=texture if isinstance(texture, Exception) else None,
            return_value=False if texture is None else texture,
        ))
        stack.enter_context(patch.object(face_service, "spoof_check_with_embedding", downstream))
        stack.enter_context(patch.object(student, "_log", audit_log))
        return stack, downstream, audit_log

    def test_low_temporal_variance_is_log_only_and_reaches_downstream(self):
        stack, downstream, audit_log = self._route_patches()
        with stack:
            responses = [self.client.post("/probe", json=self.payload) for _ in range(3)]

        self.assertTrue(all(response.status_code == 200 for response in responses))
        self.assertEqual(downstream.call_count, 3)
        temporal_logs = [call for call in audit_log.call_args_list if call.args[1] == "liveness_temporal"]
        self.assertEqual(len(temporal_logs), 1)
        self.assertEqual(temporal_logs[0].args[2], "static_log_only")
        details = temporal_logs[0].args[3]
        self.assertIn("variance=0.000", details)
        self.assertIn("frames=3", details)
        self.assertIn("reference_threshold=6.0", details)
        self.assertIn("decision=log_only", details)

    def test_audit_only_moire_and_texture_exceptions_do_not_terminate(self):
        for failing_layer in ("moire", "texture"):
            with self.subTest(layer=failing_layer):
                kwargs = {failing_layer: RuntimeError(f"{failing_layer} audit failed")}
                stack, downstream, audit_log = self._route_patches(**kwargs)
                with stack:
                    response = self.client.post("/probe", json=self.payload)
                self.assertEqual(response.status_code, 200)
                downstream.assert_called_once()
                error_logs = [
                    call for call in audit_log.call_args_list
                    if call.args[1] == f"liveness_{failing_layer}"
                ]
                self.assertEqual(error_logs[0].args[2], "error_log_only")
                self.assertIn("decision=log_only", error_logs[0].args[3])

    def test_final_and_checkin_temporal_are_log_only(self):
        checkin_source = (ROOT / "app/routes/api_checkin.py").read_text(encoding="utf-8")
        self.assertNotIn("if temporal_var < 4.0:", checkin_source)
        self.assertIn("TEMPORAL_VAR_THRESHOLD   = 4.0", (ROOT / "app/services/face_service.py").read_text(encoding="utf-8"))
        self.assertNotIn("frames_for_temporal", checkin_source)

        stack, combined, audit_log = self._enroll_patches()
        with stack:
            response = self._enroll()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "pending_verify")
        self.assertEqual(combined.call_count, 5)
        temporal_log = next(call for call in audit_log.call_args_list if call.args[1] == "temporal_var")
        self.assertEqual(temporal_log.args[2], "static_log_only")
        self.assertIn("variance=3.609", temporal_log.args[3])
        self.assertIn("reference_threshold=4.0", temporal_log.args[3])
        self.assertIn("decision=log_only", temporal_log.args[3])

        # No HTTP route supplies the optional temporal burst to the weighted layer.
        for path in (ROOT / "app/routes").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "combined_spoof_score":
                    self.assertFalse(any(kw.arg == "frames_for_temporal" for kw in node.keywords))
                    self.assertLessEqual(len(node.args), 1)

    def test_final_audit_exceptions_do_not_terminate(self):
        for layer in ("moire", "texture", "temporal"):
            with self.subTest(layer=layer):
                stack, combined, audit_log = self._enroll_patches(**{
                    layer: RuntimeError(f"{layer} failed")
                })
                with stack:
                    response = self._enroll()
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json()["status"], "pending_verify")
                self.assertEqual(combined.call_count, 5)
                step = {"moire": "moire_fft", "texture": "screen_texture", "temporal": "temporal_var"}[layer]
                error_log = next(call for call in audit_log.call_args_list if call.args[1] == step)
                self.assertEqual(error_log.args[2], "error_log_only")
                self.assertIn("decision=log_only", error_log.args[3])

    def test_ear_parse_is_non_blocking(self):
        for value in (None, "", "not-a-number", "NaN", "Infinity", float("-inf")):
            with self.subTest(value=value):
                stack, combined, audit_log = self._enroll_patches()
                with stack:
                    response = self._enroll(value)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json()["status"], "pending_verify")
                ear_log = next(call for call in audit_log.call_args_list if call.args[1] == "ear_std")
                self.assertEqual(ear_log.args[2], "absent")

        for value, expected_label in ((0.002, "low_but_pass"), (0.01, "pass")):
            with self.subTest(value=value):
                stack, _combined, audit_log = self._enroll_patches()
                with stack:
                    response = self._enroll(value)
                self.assertEqual(response.status_code, 200)
                ear_log = next(call for call in audit_log.call_args_list if call.args[1] == "ear_std")
                self.assertEqual(ear_log.args[2], expected_label)

    def test_out_of_scope_rules_are_unchanged(self):
        face_source = (ROOT / "app/services/face_service.py").read_text(encoding="utf-8")
        enroll_source = inspect.getsource(_undecorated(student.api_enroll))
        for expected in (
            "if size_kb < 3:", "if lap_var < 8:", "if min(ch_stds) < 2.0:",
            "if w > 1920 or h > 1080:", "if w < 160 or h < 120:",
        ):
            self.assertIn(expected, face_source)
        for expected in (
            "MIN_SPOOF_PASS = 4", "if spoof_pass_count < MIN_SPOOF_PASS:",
            "_consistency_threshold = 0.75 if _flow_mode == \"circular\" else 0.80",
            "if max_sim < CONTINUITY_THRESHOLD:", "if _sim >= DUPLICATE_THRESHOLD:",
        ):
            self.assertIn(expected, enroll_source)

    def test_consistency_diagnostics_are_logged_on_pass(self):
        stack, _combined, audit_log = self._enroll_patches()
        with stack:
            response = self._enroll()

        self.assertEqual(response.status_code, 200)
        diagnostic_log = next(
            call for call in audit_log.call_args_list
            if call.args[1] == "consistency_diagnostics"
        )
        self.assertEqual(diagnostic_log.args[2], "pass")
        details = json.loads(diagnostic_log.args[3])
        self.assertIn("pairwise_scores", details)
        self.assertIn("average_similarities", details)
        self.assertIn("embedding_diagnostics", details)
        self.assertEqual(len(details["detector_crops"]), 5)
        self.assertEqual(details["flagged_indices_zero_based"], [])
        self.assertEqual(details["flagged_frames"], [])
        self.assertEqual(details["classifier_reason"], "per_frame_average_below_threshold")

    def test_validation_rejection_logs_available_metadata(self):
        validation = {
            "valid": False,
            "reason": "frame_too_small",
            "metadata": {
                "size_kb": 2.4,
                "dimensions": "640x480",
                "laplacian_var": 7.5,
                "B_std": 12.1,
                "G_std": 13.2,
                "R_std": 14.3,
            },
        }
        stack, downstream, audit_log = self._route_patches(validation=validation)
        with stack:
            response = self.client.post("/probe", json=self.payload)

        self.assertEqual(response.status_code, 400)
        downstream.assert_not_called()
        details = audit_log.call_args.args[3]
        for expected in (
            "reason=frame_too_small", "size_kb=2.4", "dimensions=640x480",
            "laplacian_var=7.5", "B_std=12.1", "G_std=13.2", "R_std=14.3",
        ):
            self.assertIn(expected, details)


if __name__ == "__main__":
    unittest.main(verbosity=2)

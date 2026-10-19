import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
from app.services import face_service as service


class FaceConcurrencyTests(unittest.TestCase):
    def test_shared_cascade_is_serialized_across_crop_and_temporal_checks(self):
        active = peak = 0
        counter_lock = threading.Lock()
        barrier = threading.Barrier(8)

        def detect(*args, **kwargs):
            nonlocal active, peak
            with counter_lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(.01)
                return []
            finally:
                with counter_lock:
                    active -= 1

        def request(i):
            barrier.wait(timeout=5)
            image = np.zeros((80, 80, 3), dtype=np.uint8)
            if i % 2:
                return service._crop_face_for_antispoof(image)
            return service.detect_static_image([image, image, image])

        with patch.object(service, '_face_cascade') as cascade:
            cascade.detectMultiScale.side_effect = detect
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(request, range(8)))
        self.assertEqual(len(results), 8)
        self.assertEqual(peak, 1)

    def test_face_engine_inference_is_serialized(self):
        """Enrollment, check-in and anti-spoof share one InsightFace guard per process."""
        from types import SimpleNamespace
        from app.services import face_engine
        active = peak = 0
        counter_lock = threading.Lock()
        barrier = threading.Barrier(8)

        def native(value):
            nonlocal active, peak
            with counter_lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(.01)
                return value
            finally:
                with counter_lock:
                    active -= 1

        face = SimpleNamespace(bbox=np.array([10., 10., 60., 70.]), kps=np.array([[25., 30.], [45., 30.], [35., 40.], [28., 55.], [42., 55.]]),
                               det_score=.9, pose=np.array([0., 0., 0.]), normed_embedding=np.ones(512, np.float32) / np.sqrt(512))
        fake = SimpleNamespace(get=lambda img: native([face]),
                               det_model=SimpleNamespace(detect=lambda img, max_num=0, metric='default':
                                                         native((np.array([[10., 10., 60., 70., .9]]), None))))

        def request(i):
            barrier.wait(timeout=5)
            image = np.zeros((80, 80, 3), dtype=np.uint8)
            if i % 2:
                return face_engine.analyze(image, clahe=False)
            return face_engine.detect_main(image)

        with patch.object(face_engine, 'get_app', return_value=fake):
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(request, range(8)))
        self.assertEqual(len(results), 8)
        self.assertEqual(peak, 1)

    def test_failure_releases_guard_and_preserves_exception(self):
        from types import SimpleNamespace
        from app.services import face_engine
        failing = SimpleNamespace(get=lambda img: (_ for _ in ()).throw(RuntimeError('model failed')))
        with patch.object(face_engine, 'get_app', return_value=failing):
            with self.assertRaises(RuntimeError):
                face_engine.analyze(np.zeros((8, 8, 3), np.uint8), clahe=False)
        self.assertFalse(face_engine._infer_lock.locked())


if __name__ == '__main__':
    unittest.main()

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

    def test_detector_and_models_share_one_guard(self):
        import sys
        import types
        from deepface import DeepFace
        from deepface.modules import modeling
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

        face = {'score': .99, 'facial_area': [10, 10, 50, 50],
                'landmarks': {'left_eye': [40, 25], 'right_eye': [20, 25]}}
        retinaface = types.SimpleNamespace(RetinaFace=types.SimpleNamespace(
            detect_faces=lambda img, threshold=.9, allow_upscaling=True: native({'face_1': face})))

        class Fasnet:
            def analyze(self, img, facial_area):
                return native((True, .99))

        def request(i):
            barrier.wait(timeout=5)
            if i % 2:
                return service.extract_embedding('image')
            return service._run_fasnet_antispoof(np.zeros((64, 64, 3), dtype=np.uint8))

        with patch.dict(sys.modules, {'retinaface': retinaface}),              patch.object(DeepFace, 'represent', side_effect=lambda **kw: native([{'embedding': [1., 0.]}])),              patch.object(modeling, 'build_model', return_value=Fasnet()),              patch.object(service, '_decode_image', return_value=np.zeros((64, 64, 3), dtype=np.uint8)):
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(request, range(8)))
        self.assertEqual(len(results), 8)
        self.assertEqual(peak, 1, 'detector and models must not overlap across routes')

    def test_failure_releases_guard_and_preserves_exception(self):
        from deepface import DeepFace
        with patch.object(DeepFace, 'represent', side_effect=RuntimeError('model failed')):
            with self.assertRaises(RuntimeError):
                service._call_deepface('represent', img_path='test')
        with patch.object(DeepFace, 'represent', return_value='next request'):
            with ThreadPoolExecutor(max_workers=1) as pool:
                self.assertEqual(pool.submit(service._call_deepface, 'represent').result(timeout=3),
                                 'next request')


if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.mock import patch

import numpy as np

from app.services import face_service as service


class FaceDetectionRetryTests(unittest.TestCase):
    def test_missing_face_requests_new_frame_without_system_failure(self):
        with patch.object(service, '_decode_image', return_value=np.zeros((80, 80, 3), dtype=np.uint8)), \
             patch.object(service, '_run_fasnet_antispoof', side_effect=service.FaceNotDetectedError()), \
             patch.object(service, 'detect_screen_moire', return_value={'avg_score': 0, 'is_screen': False}), \
             patch.object(service, 'detect_screen_texture', return_value=False):
            result = service.spoof_check_with_embedding('test')
        self.assertTrue(result['retry_capture'])
        self.assertFalse(result['system_failure'])
        self.assertFalse(result['is_real'])
        self.assertIsNone(result['embedding'])

    def test_actual_model_failure_still_fails_closed(self):
        with patch.object(service, '_run_fasnet_antispoof', return_value=(None, None)), \
             patch.object(service, '_run_antispoof', return_value=(True, 1.0)):
            result = service.combined_spoof_score(np.zeros((80, 80, 3), dtype=np.uint8))
        self.assertTrue(service.is_system_failure(result))
        self.assertFalse(result.get('retry_capture', False))

    def test_only_known_no_face_error_is_retryable(self):
        from deepface import DeepFace
        with patch.object(DeepFace, 'extract_faces', side_effect=ValueError('Face could not be detected.')):
            with self.assertRaises(service.FaceNotDetectedError):
                service._run_fasnet_antispoof(np.zeros((80, 80, 3), dtype=np.uint8))
        with patch.object(DeepFace, 'extract_faces', side_effect=ValueError('invalid model shape')):
            self.assertEqual(service._run_fasnet_antispoof(np.zeros((80, 80, 3), dtype=np.uint8)), (None, None))

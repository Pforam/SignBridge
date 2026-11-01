"""
Unit tests for SignBridge MediaPipe Holistic Word Recognizer & Inference Pipeline.
"""

import unittest
import numpy as np
import os
import sys

# Ensure backend root is on sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from services.word_recognizer import WordRecognizer, synthesize_pose_from_anchors
from services.holistic_preprocessor import (
    extract_holistic_frame_features,
    preprocess_holistic_sequence,
    swap_holistic_sequence,
    COMBINED_FEATURE_DIMS,
    STATIC_FEATURE_DIMS,
)


class TestHolisticWordRecognizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recognizer = WordRecognizer()

    def test_01_model_loading(self):
        """Verify the holistic word model is loaded with proper metadata."""
        self.assertTrue(self.recognizer.is_available, "WordRecognizer should be available")
        self.assertEqual(self.recognizer.mode, 'holistic_gru', "Mode should be holistic_gru")
        self.assertGreaterEqual(len(self.recognizer.labels), 50, "Should have at least 50 word classes")
        self.assertIn("HELLO", self.recognizer.labels)
        self.assertIn("NAMASTE", self.recognizer.labels)
        self.assertIn("THANK_YOU", self.recognizer.labels)

    def test_02_preprocessor_feature_dimensions(self):
        """Verify static and temporal feature dimensions."""
        dummy_frame = np.random.randn(156).astype(np.float32)
        static_feats = extract_holistic_frame_features(dummy_frame)
        self.assertEqual(static_feats.shape, (STATIC_FEATURE_DIMS,), f"Static features should be {STATIC_FEATURE_DIMS}")

        dummy_seq = np.random.randn(30, 156).astype(np.float32)
        seq_feats = preprocess_holistic_sequence(dummy_seq)
        self.assertEqual(seq_feats.shape, (30, COMBINED_FEATURE_DIMS), f"Sequence features should be (30, {COMBINED_FEATURE_DIMS})")

    def test_03_prediction_with_156_features(self):
        """Verify inference succeeds with 30 frames of 156 features (hands + pose)."""
        seq = np.zeros((30, 156), dtype=np.float32)
        # Simulate active right hand motion
        for t in range(30):
            seq[t, 63:66] = [0.5 + t * 0.005, 0.5 - t * 0.005, 0.0]  # moving wrist
            seq[t, 66:126] = 0.5  # fingers
            seq[t, 126:156] = 0.3  # pose
        result = self.recognizer.predict(seq)
        self.assertIsNotNone(result, "Result should not be None")
        self.assertIn('word', result)
        self.assertIn('confidence', result)
        self.assertIn('top3', result)
        self.assertIn('mode', result)
        self.assertEqual(result['mode'], 'holistic_gru')

    def test_04_prediction_with_126_features_and_anchors(self):
        """Verify backward-compatible inference with 126 features and body_anchors."""
        seq = np.zeros((30, 126), dtype=np.float32)
        for t in range(30):
            seq[t, 0:3] = [0.4 + t * 0.004, 0.5, 0.0]
            seq[t, 3:63] = 0.4
        anchors = {
            'nose': {'x': 0.5, 'y': 0.2, 'z': 0.0},
            'leftShoulder': {'x': 0.6, 'y': 0.4, 'z': 0.0},
            'rightShoulder': {'x': 0.4, 'y': 0.4, 'z': 0.0},
            'leftElbow': {'x': 0.65, 'y': 0.6, 'z': 0.0},
            'rightElbow': {'x': 0.35, 'y': 0.6, 'z': 0.0},
            'leftWrist': {'x': 0.5, 'y': 0.5, 'z': 0.0},
            'rightWrist': {'x': 0.4, 'y': 0.7, 'z': 0.0},
        }
        result = self.recognizer.predict(seq, body_anchors=anchors)
        self.assertIsNotNone(result)
        self.assertIn('word', result)
        self.assertIn('confidence', result)

    def test_05_all_zeros_rejection(self):
        """Verify sequence with all zeros is properly rejected."""
        seq = np.zeros((30, 156), dtype=np.float32)
        result = self.recognizer.predict(seq)
        self.assertIsNotNone(result)
        self.assertEqual(result['word'], '?')
        self.assertEqual(result['confidence'], 0.0)

    def test_06_slot_swapping_invariance(self):
        """Verify slot-swapping operates without error and preserves shape."""
        seq = np.random.randn(30, 156).astype(np.float32)
        swapped = swap_holistic_sequence(seq)
        self.assertEqual(swapped.shape, (30, 156))
        # Left hand coords should match original right hand coords
        np.testing.assert_array_almost_equal(swapped[:, :63], seq[:, 63:126])
        np.testing.assert_array_almost_equal(swapped[:, 63:126], seq[:, :63])


if __name__ == '__main__':
    unittest.main()

"""
Sign-Bridge Flask API — ISL Gesture Translation Model Service

Loads trained classifiers that map MediaPipe hand landmark coordinate
sequences to ISL alphabet letters (A-Z).

Model loading priority:
  1. XGBoost (.pkl)         — highest accuracy, fastest CPU inference
  2. TensorFlow/Keras (.h5) — deep learning fallback
  3. Heuristic               — rule-based geometric matching

The model accepts a flat array of 126 landmark values:
  - Left hand:  21 landmarks x (x, y, z) -> flattened to indices 0-62
  - Right hand: 21 landmarks x (x, y, z) -> flattened to indices 63-125
  Total input: 126 features per frame

When no trained model file exists, falls back to a rule-based
heuristic classifier using the Mendeley ISL landmark descriptions.
"""
import os
import csv
import json
import logging
import pickle
import numpy as np

try:
    from .feature_extractor import extract_features, NUM_EXTRACTED_FEATURES, NUM_RAW_FEATURES
    from .gesture_config import (
        CONFIDENCE_THRESHOLD,
        CONFIDENCE_THRESHOLD_RELAXED,
        MARGIN_THRESHOLD,
        MARGIN_THRESHOLD_RELAXED,
    )
    from .gesture_pipeline import CentroidValidator
except ImportError:
    from services.feature_extractor import extract_features, NUM_EXTRACTED_FEATURES, NUM_RAW_FEATURES
    from services.gesture_config import (
        CONFIDENCE_THRESHOLD,
        CONFIDENCE_THRESHOLD_RELAXED,
        MARGIN_THRESHOLD,
        MARGIN_THRESHOLD_RELAXED,
    )
    from services.gesture_pipeline import CentroidValidator

logger = logging.getLogger(__name__)

# Path to the trained model weights files
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# ST-GCN Kinematic Graph model (.pt) — Priority 1 (Highest geometric fidelity)
STGCN_MODEL_PATH = os.path.join(MODEL_DIR, 'isl_stgcn_model.pt')
STGCN_META_PATH = os.path.join(MODEL_DIR, 'stgcn_training_meta.json')

# XGBoost model (created by train_model_xgb.py) — Priority 2
XGB_MODEL_PATH = os.path.join(MODEL_DIR, 'isl_xgboost_model.pkl')
XGB_META_PATH = os.path.join(MODEL_DIR, 'xgb_training_meta.json')

# Calibrated Hybrid Ensemble metadata
HYBRID_META_PATH = os.path.join(MODEL_DIR, 'hybrid_training_meta.json')

# Keras model (created by train_model.py) — Priority 3
MODEL_PATH = os.path.join(MODEL_DIR, 'isl_gesture_model.h5')
META_PATH = os.path.join(MODEL_DIR, 'training_meta.json')

# Path to the Mendeley reference CSV
MENDELEY_CSV = os.path.join(
    os.path.dirname(__file__), '..', '..', 'dataset',
    'Mendeley_ISL', 'extracted', 'ISL_Mendeley_Alphabets.csv'
)

# Default ISL reference postures for heuristic fallback (Mendeley ISL standard)
DEFAULT_ISL_POSTURES = {
    'A': {'left_hand_posture': 'flat palm', 'right_hand_posture': 'fist touch thumb', 'hand_system': 'two-hand'},
    'B': {'left_hand_posture': 'flat palm', 'right_hand_posture': 'fist fingers touch', 'hand_system': 'two-hand'},
    'C': {'left_hand_posture': 'c-shape', 'right_hand_posture': 'c-shape', 'hand_system': 'single-hand'},
    'D': {'left_hand_posture': 'index up', 'right_hand_posture': 'semicircle touch index', 'hand_system': 'two-hand'},
    'E': {'left_hand_posture': 'flat palm', 'right_hand_posture': 'touch index', 'hand_system': 'two-hand'},
    'F': {'left_hand_posture': 'cross', 'right_hand_posture': 'cross', 'hand_system': 'two-hand'},
    'G': {'left_hand_posture': 'fist', 'right_hand_posture': 'fist', 'hand_system': 'two-hand'},
    'H': {'left_hand_posture': 'flat palm', 'right_hand_posture': 'sweep palm', 'hand_system': 'two-hand'},
    'I': {'left_hand_posture': 'flat palm', 'right_hand_posture': 'touch middle', 'hand_system': 'two-hand'},
    'J': {'left_hand_posture': 'flat palm', 'right_hand_posture': 'trace j hook index palm', 'hand_system': 'two-hand'},
    'K': {'left_hand_posture': 'index up', 'right_hand_posture': 'v-shape touch', 'hand_system': 'two-hand'},
    'L': {'left_hand_posture': 'l-shape', 'right_hand_posture': 'flat palm base', 'hand_system': 'two-hand'},
    'M': {'left_hand_posture': 'flat palm', 'right_hand_posture': 'three fingers palm', 'hand_system': 'two-hand'},
    'N': {'left_hand_posture': 'flat palm', 'right_hand_posture': 'two fingers palm v-shape', 'hand_system': 'two-hand'},
    'O': {'left_hand_posture': 'flat palm', 'right_hand_posture': 'touch ring', 'hand_system': 'two-hand'},
    'P': {'left_hand_posture': 'index up', 'right_hand_posture': 'circle touch index', 'hand_system': 'two-hand'},
    'Q': {'left_hand_posture': 'flat palm', 'right_hand_posture': 'hook thumb', 'hand_system': 'two-hand'},
    'R': {'left_hand_posture': 'flat palm', 'right_hand_posture': 'curved index palm', 'hand_system': 'two-hand'},
    'S': {'left_hand_posture': 'flat palm', 'right_hand_posture': 'touch pinky', 'hand_system': 'two-hand'},
    'T': {'left_hand_posture': 'flat palm edge', 'right_hand_posture': 'index touch edge', 'hand_system': 'two-hand'},
    'U': {'left_hand_posture': 'flat palm', 'right_hand_posture': 'touch pinky tip', 'hand_system': 'two-hand'},
    'V': {'left_hand_posture': 'flat palm', 'right_hand_posture': 'v-shape', 'hand_system': 'two-hand'},
    'W': {'left_hand_posture': 'flat palm', 'right_hand_posture': 'w-peaks', 'hand_system': 'two-hand'},
    'X': {'left_hand_posture': 'index up', 'right_hand_posture': 'index cross', 'hand_system': 'two-hand'},
    'Y': {'left_hand_posture': 'flat palm', 'right_hand_posture': 'v-shape thumb index palm', 'hand_system': 'two-hand'},
    'Z': {'left_hand_posture': 'flat palm', 'right_hand_posture': 'bent palm edge', 'hand_system': 'two-hand'},
}

# ISL alphabet + digit labels (36 classes: A-Z then 0-9)
ISL_LABELS: list[str] = [str(c) for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789']

# Below this, a same-hand thumb-to-index distance counts as "touching"
# (in the same normalized units used by normalize_landmarks()).
THUMB_INDEX_TOUCH_THRESHOLD = 0.12

EXPECTED_INPUT_LEN = 126
POINTS_PER_HAND = 21


def normalize_landmarks(landmarks):
    points = np.asarray(landmarks, dtype=np.float32)
    if points.size != 126 or not np.isfinite(points).all():
        return None
    points = points.reshape(2, 21, 3)
    present = np.any(points != 0, axis=(1, 2))
    if not present.any():
        return None
    wrists = points[present, 0]
    anchor = wrists.mean(axis=0)
    hand_sizes = np.linalg.norm(points[present, 12] - points[present, 0], axis=1)
    scale = float(hand_sizes.mean())
    if not np.isfinite(scale) or scale < 1e-4:
        return None
    normalized = (points - anchor) / scale
    normalized[~present] = 0
    return normalized.reshape(-1)


def normalize_landmarks_batch(batch):
    """Vectorized normalize_landmarks for a (N, 126) float32 array.

    Mirrors normalize_landmarks() exactly but operates on all N frames
    in one numpy call — no Python loop, no per-frame overhead.

    Returns a (N, 126) float32 array.  Frames where both hands are absent
    are left as zeros (same contract as the scalar version).
    Returns None if no frame contains any hand data at all.
    """
    pts = np.asarray(batch, dtype=np.float32).reshape(-1, 2, 21, 3)   # (N, 2, 21, 3)
    N = pts.shape[0]

    # Which frames have at least one active hand
    present = np.any(pts != 0, axis=(2, 3))                            # (N, 2) bool

    if not present.any():
        return None

    # Anchor: mean wrist position across present hands per frame
    # Mask absent hands to 0 before summing so they don't skew the mean
    wrist_masked = pts[:, :, 0, :] * present[:, :, None]              # (N, 2, 3)
    n_present = present.sum(axis=1, keepdims=True).clip(min=1)         # (N, 1)
    anchor = wrist_masked.sum(axis=1) / n_present                      # (N, 3)

    # Scale: mean mid-finger-to-wrist distance across present hands
    hand_sizes = np.linalg.norm(
        pts[:, :, 12, :] - pts[:, :, 0, :], axis=-1                   # (N, 2)
    )
    scale = np.where(present, hand_sizes, 0).sum(axis=1) / n_present[:, 0]  # (N,)
    scale = scale.clip(min=1e-4)                                       # guard zero

    # Normalise: subtract anchor, divide by scale
    normed = (pts - anchor[:, None, None, :]) / scale[:, None, None, None]  # (N, 2, 21, 3)

    # Zero-out absent hand slots (keep same contract as scalar version)
    normed[~present] = 0

    return normed.reshape(N, 126).astype(np.float32)


class LandmarkValidationError(ValueError):
    """Raised when incoming landmark data doesn't match the expected shape."""


def validate_landmark_array(landmarks):
    """Coerce landmarks to a (126,) or (N, 126) float array, or raise."""
    if isinstance(landmarks, list) and len(landmarks) == 42 and len(landmarks) > 0 and isinstance(landmarks[0], dict):
        flat = []
        for pt in landmarks:
            flat.extend([float(pt.get('x', 0.0)), float(pt.get('y', 0.0)), float(pt.get('z', 0.0))])
        landmarks = flat

    try:
        arr = np.asarray(landmarks, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise LandmarkValidationError(f"landmarks must be numeric: {exc}") from exc

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != EXPECTED_INPUT_LEN:
        raise LandmarkValidationError(
            f"expected {EXPECTED_INPUT_LEN} values per frame "
            f"(21 landmarks x 3 coords x 2 hands), got shape {arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise LandmarkValidationError("landmarks contain NaN or infinite values")
    return arr


class TranslatorModel:
    """
    ISL Gesture-to-Text translation model.

    Supports four ML modes:
      1. Hybrid Ensemble mode — Combines ST-GCN graph logits + XGBoost trees (calibrated)
      2. ST-GCN mode          — loads PyTorch kinematic graph model (.pt)
      3. XGBoost mode         — loads trained XGBoost tree classifier (.pkl)
      4. Deep Learning mode   — loads a trained Keras .h5 model
      5. Heuristic mode       — rule-based matching using landmark geometry
    """

    HYBRID_XGB_WEIGHT = 0.65
    HYBRID_STGCN_WEIGHT = 0.35
    HYBRID_TEMPERATURE = 1.20

    def __init__(self):
        self.stgcn_model = None
        self.xgb_model = None
        self.model = None
        self.mode = 'heuristic'  # 'hybrid_ensemble' | 'stgcn' | 'xgboost' | 'deep_learning' | 'heuristic'
        self.mendeley_ref = {}
        self.metadata = {}
        self.centroid_validator = CentroidValidator()
        self._load_mendeley_ref()
        self._load()

    def _load(self):
        """Attempt to load trained models; fall back to heuristic.

        Loading priority:
          1. Hybrid Ensemble (ST-GCN + XGBoost) — Best calibration & geometric fidelity
          2. ST-GCN standalone (.pt)
          3. XGBoost standalone (.pkl)
          4. Keras (.h5)
          5. Heuristic — rule-based geometric matching
        """
        # Reset state on load/reload (ponytail: prevent stale state leakage)
        self.stgcn_model = None
        self.xgb_model = None
        self.model = None
        self.mode = 'heuristic'
        self.metadata = {}
        self.mendeley_ref = {}
        self._load_mendeley_ref()

        # Load ST-GCN if available
        if os.path.exists(STGCN_MODEL_PATH):
            try:
                import torch
                from models.st_gcn import STGCNHandClassifier

                if os.path.exists(STGCN_META_PATH):
                    with open(STGCN_META_PATH, 'r', encoding='utf-8') as f:
                        self.metadata = json.load(f)

                labels = self.metadata.get('labels', [c for c in ISL_LABELS if c.isalpha()])
                st_model = STGCNHandClassifier(num_classes=len(labels))
                checkpoint = torch.load(STGCN_MODEL_PATH, map_location='cpu', weights_only=True)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    st_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    st_model.load_state_dict(checkpoint)
                st_model.eval()
                self.stgcn_model = st_model
                logger.info(f"Loaded ST-GCN Kinematic Graph model from {STGCN_MODEL_PATH}")
            except Exception as e:
                logger.warning(f"Failed to load ST-GCN model: {e}")

        # Load XGBoost if available
        if os.path.exists(XGB_MODEL_PATH):
            try:
                with open(XGB_MODEL_PATH, 'rb') as f:
                    self.xgb_model = pickle.load(f)
                if not self.metadata and os.path.exists(XGB_META_PATH):
                    with open(XGB_META_PATH, 'r', encoding='utf-8') as f:
                        self.metadata = json.load(f)
                logger.info(f"Loaded XGBoost model from {XGB_MODEL_PATH}")
            except Exception as e:
                logger.warning(f"Failed to load XGBoost model: {e}")

        # Determine best operating mode and bind truthful metadata
        if self.stgcn_model is not None and self.xgb_model is not None:
            self.mode = 'hybrid_ensemble'
            self.model = self.stgcn_model
            if os.path.exists(HYBRID_META_PATH):
                with open(HYBRID_META_PATH, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            elif os.path.exists(XGB_META_PATH):
                with open(XGB_META_PATH, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            logger.info("Sign-Bridge Alphabet Engine initialized in CALIBRATED HYBRID ENSEMBLE mode (ST-GCN + XGBoost).")
            return
        elif self.stgcn_model is not None:
            self.mode = 'stgcn'
            self.model = self.stgcn_model
            if os.path.exists(STGCN_META_PATH):
                with open(STGCN_META_PATH, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            logger.info("Sign-Bridge Alphabet Engine initialized in ST-GCN mode.")
            return
        elif self.xgb_model is not None:
            self.mode = 'xgboost'
            self.model = self.xgb_model
            if os.path.exists(XGB_META_PATH):
                with open(XGB_META_PATH, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            logger.info("Sign-Bridge Alphabet Engine initialized in XGBoost mode.")
            return

        # Fallback to Keras model
        if os.path.exists(MODEL_PATH):
            try:
                import tensorflow as tf
                if os.path.exists(META_PATH):
                    with open(META_PATH, 'r', encoding='utf-8') as f:
                        self.metadata = json.load(f)
                self.model = tf.keras.models.load_model(MODEL_PATH)
                self.mode = 'deep_learning'
                logger.info(f"Loaded trained Keras model from {MODEL_PATH}")
                return
            except Exception as e:
                logger.warning(f"Failed to load Keras model: {e}. Falling back to heuristic mode.")

        # Fallback to Heuristic
        self.mode = 'heuristic'
        self._load_mendeley_ref()
        logger.info("Running in HEURISTIC mode (no trained model found).")

    def _load_mendeley_ref(self):
        """Parse the Mendeley CSV into a lookup dictionary, falling back to built-in ISL reference postures."""
        if not os.path.exists(MENDELEY_CSV):
            logger.info(f"Mendeley CSV not found at {MENDELEY_CSV}; loading {len(DEFAULT_ISL_POSTURES)} built-in reference postures.")
            self.mendeley_ref = dict(DEFAULT_ISL_POSTURES)
            return

        try:
            with open(MENDELEY_CSV, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    letter = row.get('letter', '').strip().upper()
                    if letter:
                        self.mendeley_ref[letter] = {
                            'hand_system': row.get('hand_system', ''),
                            'left_hand_posture': row.get('left_hand_posture', ''),
                            'right_hand_posture': row.get('right_hand_posture', ''),
                            'landmark_points': int(row.get('landmark_points', 42)),
                        }
            logger.info(f"Loaded {len(self.mendeley_ref)} Mendeley ISL letter references.")
        except Exception as e:
            logger.error(f"Error loading Mendeley CSV: {e}")
            self.mendeley_ref = dict(DEFAULT_ISL_POSTURES)

    def predict(self, landmarks, body_anchors=None):
        """
        Predict the ISL letter from a set of hand landmarks with optional Holistic body anchors.
        """
        if self.mode == 'hybrid_ensemble':
            res = self._predict_hybrid_ensemble(landmarks)
        elif self.mode == 'stgcn' and self.model is not None:
            res = self._predict_stgcn(landmarks)
        elif self.mode == 'xgboost' and self.model is not None:
            res = self._predict_xgb(landmarks)
        elif self.mode == 'deep_learning' and self.model is not None:
            res = self._predict_dl(landmarks)
        else:
            res = self._predict_heuristic(landmarks)

        if body_anchors and isinstance(res, dict):
            res['holistic_anchored'] = True
        return res

    def _predict_hybrid_ensemble(self, landmarks):
        """Run inference through the calibrated ST-GCN + XGBoost Hybrid Ensemble."""
        if self.stgcn_model is None or self.xgb_model is None:
            return self._predict_stgcn(landmarks) if self.stgcn_model else self._predict_xgb(landmarks)

        try:
            import torch
            arr = validate_landmark_array(landmarks)
            if not np.any(arr != 0):
                return self._invalid_input_result('hybrid_ensemble')

            normalized = [normalize_landmarks(row) for row in arr]
            if any(row is None for row in normalized):
                return self._invalid_input_result('hybrid_ensemble')
            arr_norm = np.asarray(normalized, dtype=np.float32)

            labels = self.metadata.get('labels', [c for c in ISL_LABELS if c.isalpha()])

            # Check single-hand status
            left_active = np.any(arr_norm[:, :63] != 0)
            right_active = np.any(arr_norm[:, 63:] != 0)
            is_single_hand = (left_active and not right_active) or (right_active and not left_active)

            # Adaptive weights: for single-hand gestures, ST-GCN receives an empty second-hand subgraph,
            # so XGBoost 208-D invariant features provide superior confidence calibration.
            xgb_weight = 0.85 if is_single_hand else self.HYBRID_XGB_WEIGHT
            stgcn_weight = 0.15 if is_single_hand else self.HYBRID_STGCN_WEIGHT
            temperature = 1.00 if is_single_hand else self.HYBRID_TEMPERATURE

            # 1. Primary orientation pass (both models evaluated on original input)
            feat_primary = extract_features(arr_norm)
            probs_xgb = self.xgb_model.predict_proba(feat_primary)[0]

            tensor_in = torch.tensor(arr_norm, dtype=torch.float32)
            with torch.no_grad():
                logits = self.stgcn_model(tensor_in)
                probs_stgcn = torch.softmax(logits / temperature, dim=1).numpy()[0]

            probs_primary = (xgb_weight * probs_xgb) + (stgcn_weight * probs_stgcn)
            conf_primary = float(np.max(probs_primary))

            # 2. Evaluate swapped pass if primary confidence is uncertain (< CONFIDENCE_THRESHOLD)
            if conf_primary < CONFIDENCE_THRESHOLD:
                swapped = np.zeros_like(arr_norm)
                swapped[:, :63] = arr_norm[:, 63:]
                swapped[:, 63:] = arr_norm[:, :63]
                feat_swapped = extract_features(swapped)
                probs_xgb_swapped = self.xgb_model.predict_proba(feat_swapped)[0]

                tensor_swapped = torch.tensor(swapped, dtype=torch.float32)
                with torch.no_grad():
                    logits_swapped = self.stgcn_model(tensor_swapped)
                    probs_stgcn_swapped = torch.softmax(logits_swapped / temperature, dim=1).numpy()[0]

                probs_swapped = (xgb_weight * probs_xgb_swapped) + (stgcn_weight * probs_stgcn_swapped)
                conf_swapped = float(np.max(probs_swapped))

                # Choose ONE consistent orientation for the entire ensemble
                if conf_swapped > conf_primary:
                    probs = probs_swapped
                else:
                    probs = probs_primary
            else:
                probs = probs_primary

            # 3. For contact gestures where hands overlap or one hand obscures another,
            # evaluate dominant sub-hands if confidence remains low (< 0.50)
            if left_active and right_active and np.max(probs) < 0.50:
                h_left_single = np.zeros_like(arr_norm)
                h_left_single[:, :63] = arr_norm[:, :63]
                f_l = extract_features(h_left_single)
                p_l = self.xgb_model.predict_proba(f_l)[0]
                if np.max(p_l) > np.max(probs):
                    probs = p_l

                h_right_single = np.zeros_like(arr_norm)
                h_right_single[:, 63:] = arr_norm[:, 63:]
                f_r = extract_features(h_right_single)
                p_r = self.xgb_model.predict_proba(f_r)[0]
                if np.max(p_r) > np.max(probs):
                    probs = p_r

            sorted_indices = np.argsort(probs)[::-1]
            top_idx = int(sorted_indices[0])
            confidence = float(probs[top_idx])
            second_conf = float(probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
            margin = float(confidence - second_conf)
            letter = labels[top_idx] if top_idx < len(labels) else '?'

            # Top 5 scores
            top_scores = {
                labels[i]: round(float(probs[i]), 4)
                for i in sorted_indices[:5] if i < len(labels)
            }

            # Out of dataset / centroid distance check
            norm_first = arr_norm[0] if arr_norm.ndim > 1 else arr_norm
            val_ok, dist, dist_thresh, r_reason = self.centroid_validator.validate(norm_first, letter)

            # Accept if meets standard threshold OR relaxed threshold with decisive margin
            is_accepted = bool(
                ((confidence >= CONFIDENCE_THRESHOLD and margin >= MARGIN_THRESHOLD) or
                 (confidence >= CONFIDENCE_THRESHOLD_RELAXED and margin >= MARGIN_THRESHOLD_RELAXED))
                and val_ok and letter != '?'
            )
            is_rejected = not is_accepted

            rejection_reason = None
            if is_rejected:
                if not val_ok:
                    rejection_reason = r_reason
                elif confidence < CONFIDENCE_THRESHOLD_RELAXED:
                    rejection_reason = 'low_confidence'
                elif margin < MARGIN_THRESHOLD:
                    rejection_reason = 'low_margin'
                elif confidence < CONFIDENCE_THRESHOLD and margin < MARGIN_THRESHOLD_RELAXED:
                    rejection_reason = 'low_confidence'
                else:
                    rejection_reason = 'unknown_gesture'

            return {
                'letter': '?' if is_rejected else letter,
                'confidence': round(confidence, 4),
                'margin': round(margin, 4),
                'centroid_distance': round(dist, 4),
                'mode': 'hybrid_ensemble',
                'rejected': is_rejected,
                'rejection_reason': rejection_reason,
                'all_scores': top_scores
            }
        except LandmarkValidationError as e:
            logger.warning(f"Ensemble prediction skipped — bad input: {e}")
            return self._invalid_input_result('hybrid_ensemble')
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            return self._predict_xgb(landmarks)

    def _predict_stgcn(self, landmarks):
        """Run inference through the trained PyTorch ST-GCN model."""
        model = self.stgcn_model or self.model
        if model is None or not hasattr(model, '__call__'):
            return self._predict_xgb(landmarks)

        try:
            import torch
            arr = validate_landmark_array(landmarks)

            if not np.any(arr != 0):
                return self._invalid_input_result('stgcn')

            # Normalization
            normalized = [normalize_landmarks(row) for row in arr]
            if any(row is None for row in normalized):
                return self._invalid_input_result('stgcn')
            arr = np.asarray(normalized, dtype=np.float32)

            labels = self.metadata.get('labels', [c for c in ISL_LABELS if c.isalpha()])

            # 1. Primary orientation pass
            tensor_in = torch.tensor(arr, dtype=torch.float32)
            with torch.no_grad():
                logits = model(tensor_in)
                probs = torch.softmax(logits, dim=1).numpy()[0]

            left_active = np.any(arr[:, :63] != 0)
            right_active = np.any(arr[:, 63:] != 0)
            is_single_hand = (left_active and not right_active) or (right_active and not left_active)

            # 2. Hand-swapped orientation pass only when single-hand or confidence is low (< 0.72)
            if is_single_hand or float(np.max(probs)) < 0.72:
                swapped = np.zeros_like(arr)
                swapped[:, :63] = arr[:, 63:]
                swapped[:, 63:] = arr[:, :63]
                tensor_swapped = torch.tensor(swapped, dtype=torch.float32)
                with torch.no_grad():
                    logits_swapped = model(tensor_swapped)
                    probs_swapped = torch.softmax(logits_swapped, dim=1).numpy()[0]

                if np.max(probs_swapped) > np.max(probs):
                    probs = probs_swapped

            sorted_indices = np.argsort(probs)[::-1]
            top_idx = int(sorted_indices[0])
            confidence = float(probs[top_idx])
            second_conf = float(probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
            margin = float(confidence - second_conf)
            letter = labels[top_idx] if top_idx < len(labels) else '?'

            # Top 5 scores
            top_scores = {
                labels[i]: round(float(probs[i]), 4)
                for i in sorted_indices[:5] if i < len(labels)
            }

            # Out of dataset / centroid distance check
            norm_first = arr[0] if arr.ndim > 1 else arr
            val_ok, dist, dist_thresh, r_reason = self.centroid_validator.validate(norm_first, letter)

            # Accept if meets standard threshold OR relaxed threshold with decisive margin
            is_accepted = bool(
                ((confidence >= CONFIDENCE_THRESHOLD and margin >= MARGIN_THRESHOLD) or
                 (confidence >= CONFIDENCE_THRESHOLD_RELAXED and margin >= MARGIN_THRESHOLD_RELAXED))
                and val_ok and letter != '?'
            )
            is_rejected = not is_accepted

            rejection_reason = None
            if is_rejected:
                if not val_ok:
                    rejection_reason = r_reason
                elif confidence < CONFIDENCE_THRESHOLD_RELAXED:
                    rejection_reason = 'low_confidence'
                elif margin < MARGIN_THRESHOLD:
                    rejection_reason = 'low_margin'
                elif confidence < CONFIDENCE_THRESHOLD and margin < MARGIN_THRESHOLD_RELAXED:
                    rejection_reason = 'low_confidence'
                else:
                    rejection_reason = 'unknown_gesture'

            return {
                'letter': '?' if is_rejected else letter,
                'confidence': round(confidence, 4),
                'margin': round(margin, 4),
                'centroid_distance': round(dist, 4),
                'mode': 'stgcn',
                'rejected': is_rejected,
                'rejection_reason': rejection_reason,
                'all_scores': top_scores
            }
        except LandmarkValidationError as e:
            logger.warning(f"ST-GCN prediction skipped — bad input: {e}")
            return self._invalid_input_result('stgcn')
        except Exception as e:
            logger.error(f"ST-GCN prediction failed: {e}")
            return self._predict_xgb(landmarks)

    def _predict_xgb(self, landmarks):
        """Run inference through the trained XGBoost model."""
        model = self.xgb_model or self.model
        if model is None or not hasattr(model, 'predict_proba'):
            return self._predict_heuristic(landmarks)

        try:
            arr = validate_landmark_array(landmarks)

            # Fast check: if all zeros
            if not np.any(arr != 0):
                return self._invalid_input_result('xgboost')

            # Apply same normalization as training pipeline unconditionally
            normalized = [normalize_landmarks(row) for row in arr]
            if any(row is None for row in normalized):
                return self._invalid_input_result('xgboost')
            arr = np.asarray(normalized, dtype=np.float32)

            labels = self.metadata.get('labels', ISL_LABELS)

            # Check if this is a single-hand gesture (only one hand slot is non-zero)
            left_active = np.any(arr[:, :63] != 0)
            right_active = np.any(arr[:, 63:] != 0)
            is_single_hand = (left_active and not right_active) or (right_active and not left_active)

            # Predict probabilities for primary placement using 176-D geometric features
            feat_primary = extract_features(arr)
            probs = model.predict_proba(feat_primary)[0]

            conf_primary = float(np.max(probs))

            # Only evaluate alternate hand slot arrangement if primary placement is uncertain (< CONFIDENCE_THRESHOLD)
            if conf_primary < CONFIDENCE_THRESHOLD:
                swapped = np.zeros_like(arr)
                swapped[:, :63] = arr[:, 63:]
                swapped[:, 63:] = arr[:, :63]
                feat_swapped = extract_features(swapped)
                probs_swapped = model.predict_proba(feat_swapped)[0]
                if np.max(probs_swapped) > np.max(probs):
                    probs = probs_swapped

            # For contact gestures where hands overlap, only evaluate dominant sub-hands if confidence remains low
            if left_active and right_active and np.max(probs) < 0.50:
                h_left_single = np.zeros_like(arr)
                h_left_single[:, :63] = arr[:, :63]
                f_l = extract_features(h_left_single)
                p_l = model.predict_proba(f_l)[0]
                if np.max(p_l) > np.max(probs):
                    probs = p_l

                h_right_single = np.zeros_like(arr)
                h_right_single[:, 63:] = arr[:, 63:]
                f_r = extract_features(h_right_single)
                p_r = model.predict_proba(f_r)[0]
                if np.max(p_r) > np.max(probs):
                    probs = p_r

            sorted_indices = np.argsort(probs)[::-1]
            top_idx = int(sorted_indices[0])
            confidence = float(probs[top_idx])
            second_conf = float(probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
            margin = float(confidence - second_conf)
            letter = labels[top_idx] if top_idx < len(labels) else '?'

            # Top 5 scores
            top_scores = {
                labels[i]: round(float(probs[i]), 4)
                for i in sorted_indices[:5] if i < len(labels)
            }

            # Out of dataset / centroid distance check
            norm_first = arr[0] if arr.ndim > 1 else arr
            val_ok, dist, dist_thresh, r_reason = self.centroid_validator.validate(norm_first, letter)

            # Accept if meets standard threshold OR relaxed threshold with decisive margin
            is_accepted = bool(
                ((confidence >= CONFIDENCE_THRESHOLD and margin >= MARGIN_THRESHOLD) or
                 (confidence >= CONFIDENCE_THRESHOLD_RELAXED and margin >= MARGIN_THRESHOLD_RELAXED))
                and val_ok and letter != '?'
            )
            is_rejected = not is_accepted

            rejection_reason = None
            if is_rejected:
                if not val_ok:
                    rejection_reason = r_reason
                elif confidence < CONFIDENCE_THRESHOLD_RELAXED:
                    rejection_reason = 'low_confidence'
                elif margin < MARGIN_THRESHOLD:
                    rejection_reason = 'low_margin'
                elif confidence < CONFIDENCE_THRESHOLD and margin < MARGIN_THRESHOLD_RELAXED:
                    rejection_reason = 'low_confidence'
                else:
                    rejection_reason = 'unknown_gesture'

            return {
                'letter': '?' if is_rejected else letter,
                'confidence': round(confidence, 4),
                'margin': round(margin, 4),
                'centroid_distance': round(dist, 4),
                'mode': 'xgboost',
                'rejected': is_rejected,
                'rejection_reason': rejection_reason,
                'all_scores': top_scores
            }
        except LandmarkValidationError as e:
            logger.warning(f"XGBoost prediction skipped — bad input: {e}")
            return self._invalid_input_result('xgboost')
        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            return self._predict_heuristic(landmarks)

    def _predict_dl(self, landmarks):
        """Run inference through the trained Keras model."""
        if self.model is None:
            return self._predict_heuristic(landmarks)

        model = self.model

        try:
            arr = validate_landmark_array(landmarks)

            if not np.any(arr != 0):
                return self._invalid_input_result('deep_learning')

            if self.metadata.get('preprocessing') == 'wrist_center_scale_v1':
                normalized = [normalize_landmarks(row) for row in arr]
                if any(row is None for row in normalized):
                    return self._invalid_input_result('deep_learning')
                arr = np.asarray(normalized, dtype=np.float32)

            labels = self.metadata.get('labels', ISL_LABELS)

            # Predict
            predictions = model.predict(arr, verbose=0)
            probs = predictions[0]
            temperature = float(self.metadata.get('confidence_temperature', 1.0))
            if temperature != 1.0:
                logits = np.log(np.clip(probs, 1e-7, 1.0)) / temperature
                logits -= logits.max()
                probs = np.exp(logits)
                probs /= probs.sum()

            sorted_indices = np.argsort(probs)[::-1]
            top_idx = int(sorted_indices[0])
            confidence = float(probs[top_idx])
            second_conf = float(probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
            margin = float(confidence - second_conf)
            letter = labels[top_idx] if top_idx < len(labels) else '?'

            # Top 5 scores
            top_scores = {
                labels[i]: round(float(probs[i]), 4)
                for i in sorted_indices[:5] if i < len(labels)
            }

            norm_first = arr[0] if arr.ndim > 1 else arr
            val_ok, dist, dist_thresh, r_reason = self.centroid_validator.validate(norm_first, letter)

            # Accept if meets standard threshold OR relaxed threshold with decisive margin
            is_accepted = bool(
                ((confidence >= CONFIDENCE_THRESHOLD and margin >= MARGIN_THRESHOLD) or
                 (confidence >= CONFIDENCE_THRESHOLD_RELAXED and margin >= MARGIN_THRESHOLD_RELAXED))
                and val_ok and letter != '?'
            )
            is_rejected = not is_accepted

            rejection_reason = None
            if is_rejected:
                if not val_ok:
                    rejection_reason = r_reason
                elif confidence < CONFIDENCE_THRESHOLD_RELAXED:
                    rejection_reason = 'low_confidence'
                elif margin < MARGIN_THRESHOLD:
                    rejection_reason = 'low_margin'
                elif confidence < CONFIDENCE_THRESHOLD and margin < MARGIN_THRESHOLD_RELAXED:
                    rejection_reason = 'low_confidence'
                else:
                    rejection_reason = 'unknown_gesture'

            return {
                'letter': '?' if is_rejected else letter,
                'confidence': round(confidence, 4),
                'margin': round(margin, 4),
                'centroid_distance': round(dist, 4),
                'mode': 'deep_learning',
                'rejected': is_rejected,
                'rejection_reason': rejection_reason,
                'all_scores': top_scores
            }
        except LandmarkValidationError as e:
            logger.warning(f"Keras prediction skipped — bad input: {e}")
            return self._invalid_input_result('deep_learning')
        except Exception as e:
            logger.error(f"DL prediction failed: {e}")
            return self._predict_heuristic(landmarks)

    @staticmethod
    def _invalid_input_result(mode):
        return {
            'letter': '?',
            'confidence': 0.0,
            'margin': 0.0,
            'centroid_distance': 999.0,
            'mode': mode,
            'rejected': True,
            'rejection_reason': 'invalid_input',
            'all_scores': {}
        }

    def _predict_heuristic(self, landmarks):
        """
        Rule-based heuristic prediction using geometric features
        extracted from 42 hand landmarks (21 per hand).
        """
        try:
            # Convert input to structured format
            points = self._parse_landmarks(landmarks)
            if points is None or len(points) < 42 or all(p.get('x', 0) == 0 and p.get('y', 0) == 0 and p.get('z', 0) == 0 for p in points):
                return {
                    'letter': '?',
                    'confidence': 0.0,
                    'mode': 'heuristic',
                    'rejected': True,
                    'all_scores': {}
                }

            if not self.mendeley_ref:
                return {
                    'letter': '?',
                    'confidence': 0.0,
                    'margin': 0.0,
                    'centroid_distance': 0.0,
                    'mode': 'heuristic',
                    'rejected': True,
                    'rejection_reason': 'no_reference_rules',
                    'all_scores': {}
                }

            # Extract geometric features
            features = self._extract_features(points)

            # Score each letter against the features
            scores = {}
            for letter in ISL_LABELS:
                scores[letter] = self._score_letter(letter, features)

            # Find best match
            sorted_letters = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            best_letter = sorted_letters[0][0] if sorted_letters else '?'
            best_score = sorted_letters[0][1] if sorted_letters else 0.0
            second_score = sorted_letters[1][1] if len(sorted_letters) > 1 else 0.0

            # Normalize confidence
            total = sum(scores.values()) or 1.0
            confidence = best_score / total
            second_conf = second_score / total
            margin = float(confidence - second_conf)

            # Top 5
            top_scores = {l: round(s / total, 4) for l, s in sorted_letters[:5]}

            is_rejected = bool(confidence < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD or best_letter == '?')
            rejection_reason = None
            if is_rejected:
                if confidence < CONFIDENCE_THRESHOLD:
                    rejection_reason = 'low_confidence'
                elif margin < MARGIN_THRESHOLD:
                    rejection_reason = 'low_margin'
                else:
                    rejection_reason = 'unknown_gesture'

            return {
                'letter': '?' if is_rejected else best_letter,
                'confidence': round(confidence, 4),
                'margin': round(margin, 4),
                'centroid_distance': 0.0,
                'mode': 'heuristic',
                'rejected': is_rejected,
                'rejection_reason': rejection_reason,
                'all_scores': top_scores
            }
        except Exception as e:
            logger.error(f"Heuristic prediction failed: {e}")
            return {
                'letter': '?',
                'confidence': 0.0,
                'mode': 'heuristic',
                'all_scores': {}
            }

    def _parse_landmarks(self, landmarks):
        """
        Parse landmarks into a list of 42 {x, y, z} dicts.
        Handles both flat arrays and structured input.
        """
        if isinstance(landmarks, list):
            if len(landmarks) == 42 and isinstance(landmarks[0], dict):
                return landmarks
            if len(landmarks) == 126:
                # Flat array: reshape to 42 x 3
                return [
                    {'x': landmarks[i*3], 'y': landmarks[i*3+1], 'z': landmarks[i*3+2]}
                    for i in range(42)
                ]
            if len(landmarks) >= 42:
                # Could be nested arrays [[x,y,z], ...]
                try:
                    return [
                        {'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2])}
                        for p in landmarks[:42]
                    ]
                except (TypeError, IndexError):
                    pass
        return None

    def _extract_features(self, points):
        """
        Extract geometric features from 42 landmark points.
        Points 0-20: Left hand, Points 21-41: Right hand.

        MediaPipe landmark indices per hand:
          0: Wrist
          4: Thumb tip
          8: Index tip
          12: Middle tip
          16: Ring tip
          20: Pinky tip
        """
        def dist(a, b):
            return np.sqrt(
                (a['x'] - b['x'])**2 +
                (a['y'] - b['y'])**2 +
                (a['z'] - b['z'])**2
            )

        def finger_extended(wrist, mcp, tip):
            """Check if a finger is extended (tip further from wrist than MCP)."""
            return dist(wrist, tip) > dist(wrist, mcp) * 1.1

        # Left hand landmarks (indices 0-20)
        lh = points[:21]
        # Right hand landmarks (indices 21-41)
        rh = points[21:42]

        # Reference scale per hand (wrist-to-middle-MCP), used to make the
        # touch and inter-hand checks scale-invariant instead of raw camera pixels.
        l_scale = dist(lh[0], lh[9]) or 1.0
        r_scale = dist(rh[0], rh[9]) or 1.0
        pair_scale = (l_scale + r_scale) / 2.0 if (l_scale > 0.01 and r_scale > 0.01) else max(l_scale, r_scale, 1.0)

        features = {
            # Left hand finger extensions
            'l_thumb_ext': finger_extended(lh[0], lh[2], lh[4]),
            'l_index_ext': finger_extended(lh[0], lh[5], lh[8]),
            'l_middle_ext': finger_extended(lh[0], lh[9], lh[12]),
            'l_ring_ext': finger_extended(lh[0], lh[13], lh[16]),
            'l_pinky_ext': finger_extended(lh[0], lh[17], lh[20]),

            # Right hand finger extensions
            'r_thumb_ext': finger_extended(rh[0], rh[2], rh[4]),
            'r_index_ext': finger_extended(rh[0], rh[5], rh[8]),
            'r_middle_ext': finger_extended(rh[0], rh[9], rh[12]),
            'r_ring_ext': finger_extended(rh[0], rh[13], rh[16]),
            'r_pinky_ext': finger_extended(rh[0], rh[17], rh[20]),

            # Inter-hand scale-normalized spatial relations
            'hands_dist': dist(lh[0], rh[0]) / pair_scale,  # Scale-invariant wrist-to-wrist
            'index_touch': dist(lh[8], rh[8]) / pair_scale,  # Scale-invariant index-to-index
            'thumb_touch': dist(lh[4], rh[4]) / pair_scale,  # Scale-invariant thumb-to-thumb
            'l_index_to_r_palm': dist(lh[8], rh[0]) / pair_scale,
            'r_index_to_l_palm': dist(rh[8], lh[0]) / pair_scale,

            # Same-hand thumb-to-index distance (scale-normalized)
            'l_index_thumb_touch': dist(lh[4], lh[8]) / l_scale,
            'r_index_thumb_touch': dist(rh[4], rh[8]) / r_scale,

            # Hand openness (average finger extension)
            'l_openness': sum([
                finger_extended(lh[0], lh[2], lh[4]),
                finger_extended(lh[0], lh[5], lh[8]),
                finger_extended(lh[0], lh[9], lh[12]),
                finger_extended(lh[0], lh[13], lh[16]),
                finger_extended(lh[0], lh[17], lh[20]),
            ]) / 5.0,
            'r_openness': sum([
                finger_extended(rh[0], rh[2], rh[4]),
                finger_extended(rh[0], rh[5], rh[8]),
                finger_extended(rh[0], rh[9], rh[12]),
                finger_extended(rh[0], rh[13], rh[16]),
                finger_extended(rh[0], rh[17], rh[20]),
            ]) / 5.0,
        }

        return features

    def _score_letter(self, letter, features):
        """
        Score how well the extracted features match a given ISL letter.
        Higher score = better match.
        """
        score = 0.0

        # Digits have no Mendeley heuristic reference — return zero score
        if letter.isdigit():
            return 0.0

        ref = self.mendeley_ref.get(letter, {})
        lh_posture = ref.get('left_hand_posture', '').lower()
        rh_posture = ref.get('right_hand_posture', '').lower()

        # Match left hand posture keywords
        if 'flat palm' in lh_posture:
            score += features['l_openness'] * 2.0
        elif 'fist' in lh_posture:
            score += (1.0 - features['l_openness']) * 2.0
        elif 'thumb up' in lh_posture:
            score += (1.0 if features['l_thumb_ext'] else 0.0) * 2.0
            score += (1.0 if not features['l_index_ext'] else 0.0)
        elif 'index up' in lh_posture:
            score += (1.0 if features['l_index_ext'] else 0.0) * 2.0
            score += (1.0 if not features['l_middle_ext'] else 0.0)
        elif 'l-shape' in lh_posture:
            score += (1.0 if features['l_thumb_ext'] and features['l_index_ext'] else 0.0) * 2.0
        elif 'cross' in lh_posture:
            is_isl_f = features['l_index_thumb_touch'] < THUMB_INDEX_TOUCH_THRESHOLD and features['l_middle_ext']
            score += 10.0 if is_isl_f else ((1.0 if features['l_index_ext'] or features['l_middle_ext'] else 0.0) * 2.0)

        # Match right hand posture keywords
        if 'touch index' in rh_posture:
            score += (2.0 if features['index_touch'] < 0.35 else 0.0)
        elif 'flat palm' in rh_posture or 'sweep palm' in rh_posture or 'base palm' in rh_posture:
            score += features['r_openness'] * 2.0
        elif 'fist' in rh_posture:
            score += (1.0 - features['r_openness']) * 2.0
        elif 'v-shape' in rh_posture:
            score += (1.0 if features['r_index_ext'] and features['r_middle_ext'] else 0.0) * 2.0
        elif 'w-peaks' in rh_posture:
            score += (1.0 if features['r_index_ext'] and features['r_middle_ext'] and features['r_ring_ext'] else 0.0) * 2.0
        elif 'cross' in rh_posture:
            # Was comparing the inter-hand index_touch distance here, which
            # measures left-index-to-right-index — not this hand's own
            # thumb-to-index touch. Fixed to use the same-hand feature.
            is_isl_f = features['r_index_thumb_touch'] < THUMB_INDEX_TOUCH_THRESHOLD and features['r_middle_ext']
            score += 10.0 if is_isl_f else ((1.0 if features['r_index_ext'] or features['r_middle_ext'] else 0.0) * 2.0)

        # Specific heuristic boosts for letters with distinctive finger patterns (J, N, W)
        if letter == 'J':
            # Right index pointing at or tracing center of left palm, or single-hand hook
            if features.get('r_index_ext') and not features.get('r_ring_ext') and not features.get('r_pinky_ext'):
                score += 2.0
                if features.get('l_openness', 0) > 0.4 and features.get('r_index_to_l_palm', 99) < 1.4:
                    score += 3.0
            elif features.get('l_index_ext') and not features.get('l_ring_ext') and not features.get('l_pinky_ext'):
                score += 2.0
                if features.get('r_openness', 0) > 0.4 and features.get('l_index_to_r_palm', 99) < 1.4:
                    score += 3.0

        elif letter == 'N':
            # Two fingers (index & middle) extended, ring & pinky folded, resting on opposite flat palm or held out
            r_two_fingers = features.get('r_index_ext') and features.get('r_middle_ext') and not features.get('r_ring_ext') and not features.get('r_pinky_ext')
            l_two_fingers = features.get('l_index_ext') and features.get('l_middle_ext') and not features.get('l_ring_ext') and not features.get('l_pinky_ext')
            if r_two_fingers:
                score += 2.5
                if features.get('l_openness', 0) > 0.4 and features.get('r_index_to_l_palm', 99) < 1.4:
                    score += 3.5
            elif l_two_fingers:
                score += 2.5
                if features.get('r_openness', 0) > 0.4 and features.get('l_index_to_r_palm', 99) < 1.4:
                    score += 3.5

        elif letter == 'W':
            # Three fingers (index, middle, ring) extended upright, pinky curled/held by thumb
            r_three_fingers = features.get('r_index_ext') and features.get('r_middle_ext') and features.get('r_ring_ext') and not features.get('r_pinky_ext')
            l_three_fingers = features.get('l_index_ext') and features.get('l_middle_ext') and features.get('l_ring_ext') and not features.get('l_pinky_ext')
            if r_three_fingers or l_three_fingers:
                score += 4.0

        return score

    def get_info(self):
        """Return truthful model metadata derived directly from active configuration."""
        if self.mode == 'hybrid_ensemble':
            model_path = f"{XGB_MODEL_PATH} + {STGCN_MODEL_PATH}"
        elif self.mode == 'xgboost':
            model_path = XGB_MODEL_PATH
        elif self.mode == 'stgcn':
            model_path = STGCN_MODEL_PATH
        else:
            model_path = MODEL_PATH

        labels = [c for c in ISL_LABELS if c.isalpha()] if self.mode in ('xgboost', 'hybrid_ensemble', 'stgcn') else ISL_LABELS
        metrics = self.metadata.get('metrics', {})

        return {
            'mode': self.mode,
            'model_path': model_path,
            'labels': labels,
            'num_classes': len(labels),
            'input_features': NUM_RAW_FEATURES,
            'estimator_features': NUM_EXTRACTED_FEATURES if self.mode in ('xgboost', 'hybrid_ensemble') else NUM_RAW_FEATURES,
            'feature_name': f'geometric_invariants_{NUM_EXTRACTED_FEATURES}d' if self.mode in ('xgboost', 'hybrid_ensemble') else 'wrist_center_scale_v1',
            'validation_accuracy': metrics.get('val_accuracy', self.metadata.get('val_accuracy')),
            'test_accuracy': metrics.get('test_accuracy'),
            'test_macro_f1': metrics.get('test_macro_f1'),
            'model_type': self.metadata.get('model_type', self.mode),
            'mendeley_letters': list(self.mendeley_ref.keys()),
        }
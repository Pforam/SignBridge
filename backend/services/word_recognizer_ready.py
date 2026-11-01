"""
Sign-Bridge Flask API — ISL Word Recognition Service

Loads a trained model that classifies temporal sequences of
30 MediaPipe landmark frames into whole ISL word labels (e.g., HELLO,
NAMASTE, THANK_YOU).

Supports PyTorch (.pt), Scikit-Learn/XGBoost (.pkl), and Keras (.h5) model artifacts.
"""

import os
import json
import logging
import pickle
import numpy as np

try:
    from .word_preprocessing import preprocess_word_sequence
except ImportError:
    from services.word_preprocessing import preprocess_word_sequence

try:
    from .gesture_config import CONFIDENCE_THRESHOLD_WORD, MARGIN_THRESHOLD_WORD
except ImportError:
    from services.gesture_config import CONFIDENCE_THRESHOLD_WORD, MARGIN_THRESHOLD_WORD

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
CNN_LSTM_MODEL_PATH_V2 = os.path.join(MODEL_DIR, 'isl_cnn_lstm_word_model_v2.pt')
CNN_LSTM_MODEL_PATH_LEGACY = os.path.join(MODEL_DIR, 'isl_cnn_lstm_word_model.pt')
# Prefer the v2 training artifact when present; otherwise retain the legacy model.
CNN_LSTM_MODEL_PATH = (
    CNN_LSTM_MODEL_PATH_V2
    if os.path.exists(CNN_LSTM_MODEL_PATH_V2)
    else CNN_LSTM_MODEL_PATH_LEGACY
)
CNN_LSTM_META_PATH = os.path.join(MODEL_DIR, 'cnn_lstm_training_meta.json')
PKL_MODEL_PATH = os.path.join(MODEL_DIR, 'isl_word_classifier.pkl')
PKL_META_PATH = os.path.join(MODEL_DIR, 'word_training_meta.json')
PT_MODEL_PATH = os.path.join(MODEL_DIR, 'isl_lstm_word_model.pt')
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'isl_lstm_word_model.h5')
LSTM_META_PATH = os.path.join(MODEL_DIR, 'lstm_training_meta.json')

SEQUENCE_LENGTH = 30
NUM_FEATURES = 126

# Gating & Confidence Thresholds (calibrated across live webcams)
MIN_MOTION_VELOCITY = 0.025
MAX_SINGLE_FRAME_JUMP = 0.15
CONFIDENCE_THRESHOLD_WORD = 0.50
MARGIN_THRESHOLD_WORD = 0.06
MAX_WORD_ENTROPY = 2.20


def compute_wrist_relative_motion(arr: np.ndarray, max_jump: float = MAX_SINGLE_FRAME_JUMP) -> float:
    """
    Computes motion velocity across sequence using wrist-relative per-hand deltas.
    Clips single-frame outlier jumps (> max_jump) so tracking glitches, slot-swaps,
    or sudden hand entries/exits cannot artificially trip the motion gate.
    """
    pts = arr.reshape(-1, 2, 21, 3)
    T = pts.shape[0]
    if T < 2:
        return 0.0

    present = np.any(pts != 0, axis=(2, 3))
    deltas = []

    for t in range(T - 1):
        for h in range(2):
            if present[t, h] and present[t + 1, h]:
                w_t = pts[t, h, 0]
                w_t1 = pts[t + 1, h, 0]
                rel_t = pts[t, h] - w_t
                rel_t1 = pts[t + 1, h] - w_t1
                d_rel = np.linalg.norm(rel_t1 - rel_t, axis=-1)
                d_w = float(np.linalg.norm(w_t1 - w_t))
                if d_w < max_jump:
                    valid = d_rel[d_rel < max_jump]
                    if len(valid) > 0:
                        deltas.append(float(np.mean(valid)) + d_w)

    return float(np.mean(deltas)) if deltas else 0.0


class WordRecognizer:
    """
    ISL whole-word gesture recognizer using temporal landmark sequences.

    Accepts a sliding window of 30 frames (each 126 landmark features)
    and returns a word prediction with confidence score.
    """

    def __init__(self):
        self.model = None
        self.mode = None  # 'sklearn' | 'pytorch' | 'keras'
        self.metadata = {}
        self.labels = []
        self.is_available = False
        self._load()

    def _load(self):
        """Attempt to load the trained word model.

        Loading priority:
          1. CNN-BiLSTM hybrid (.pt) — best accuracy for dynamic words
          2. Scikit-Learn/XGBoost sequence model (.pkl)
          3. Plain Bi-LSTM PyTorch (.pt)
          4. Keras Bi-LSTM (.h5)
        """
        # Reset state on load/reload (ponytail: prevent stale state leakage)
        self.model = None
        self.mode = None
        self.metadata = {}
        self.labels = []
        self.is_available = False

        # 1. Try the current CNN recurrent word model (.pt).
        # The training script may save either a CNN-BiLSTM or CNN-BiGRU model.
        if os.path.exists(CNN_LSTM_MODEL_PATH):
            try:
                import torch
                import torch.nn as nn

                checkpoint = torch.load(
                    CNN_LSTM_MODEL_PATH, map_location='cpu', weights_only=True
                )
                model_type = checkpoint.get('model_type', 'cnn_bilstm')

                if model_type in ('cnn_bilstm', 'cnn_bigru'):
                    if os.path.exists(CNN_LSTM_META_PATH):
                        with open(CNN_LSTM_META_PATH, 'r', encoding='utf-8') as f:
                            self.metadata = json.load(f)

                    self.labels = checkpoint.get('word_labels') or self.metadata.get('labels', [])
                    num_classes = len(self.labels)
                    if num_classes == 0:
                        raise ValueError('CNN word checkpoint contains no word labels')

                    cnn_channels = int(checkpoint.get('cnn_channels', 128 if model_type == 'cnn_bilstm' else 96))
                    hidden_dim = int(checkpoint.get('hidden_dim', 128 if model_type == 'cnn_bilstm' else 96))
                    input_dim = int(checkpoint.get('input_dim', 126))
                    if input_dim not in (126, 128, 252, 254):
                        raise ValueError(f'Unsupported CNN word model input_dim: {input_dim}')

                    state_dict = checkpoint.get('model_state_dict', {})
                    use_attn = bool(checkpoint.get('has_attention', False) or ('attn.0.weight' in state_dict))

                    class CNNRecurrentWordClassifier(nn.Module):
                        def __init__(self, in_dim, cnn_ch, hid, nc, recurrent_type, attn_enabled, dp=0.3):
                            super().__init__()
                            self.recurrent_type = recurrent_type
                            self.attn_enabled = attn_enabled
                            self.cnn = nn.Sequential(
                                nn.Conv1d(in_dim, 64, kernel_size=3, padding=1),
                                nn.BatchNorm1d(64),
                                nn.ReLU(),
                                nn.Conv1d(64, cnn_ch, kernel_size=3, padding=1),
                                nn.BatchNorm1d(cnn_ch),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                            )
                            recurrent_cls = nn.LSTM if recurrent_type == 'cnn_bilstm' else nn.GRU
                            recurrent_kwargs = {
                                'input_size': cnn_ch,
                                'hidden_size': hid,
                                'num_layers': 2,
                                'batch_first': True,
                                'bidirectional': True,
                                'dropout': dp,
                            }
                            if recurrent_type == 'cnn_bilstm':
                                self.lstm = recurrent_cls(**recurrent_kwargs)
                            else:
                                self.gru = recurrent_cls(**recurrent_kwargs)

                            if self.attn_enabled:
                                self.attn = nn.Module()
                                self.attn.attn = nn.Sequential(
                                    nn.Linear(hid * 2, 64),
                                    nn.Tanh(),
                                    nn.Linear(64, 1),
                                )
                            self.classifier = nn.Sequential(
                                nn.BatchNorm1d(hid * 2),
                                nn.Linear(hid * 2, 64),
                                nn.ReLU(),
                                nn.Dropout(dp),
                                nn.Linear(64, nc),
                            )

                        def forward(self, x):
                            x_cnn = self.cnn(x.permute(0, 2, 1))
                            x_seq = x_cnn.permute(0, 2, 1)
                            if self.recurrent_type == 'cnn_bilstm':
                                recurrent_out, _ = self.lstm(x_seq)
                            else:
                                recurrent_out, _ = self.gru(x_seq)
                            if self.attn_enabled:
                                weights = torch.softmax(
                                    self.attn.attn(recurrent_out), dim=1
                                )
                                pooled = torch.sum(recurrent_out * weights, dim=1)
                            else:
                                pooled = torch.mean(recurrent_out, dim=1)
                            return self.classifier(pooled)

                    model = CNNRecurrentWordClassifier(
                        in_dim=input_dim,
                        cnn_ch=cnn_channels,
                        hid=hidden_dim,
                        nc=num_classes,
                        recurrent_type=model_type,
                        attn_enabled=use_attn,
                    )
                    model.load_state_dict(state_dict)
                    model.eval()
                    self.model = model
                    self.mode = 'cnn_lstm' if model_type == 'cnn_bilstm' else 'cnn_gru'
                    self.input_dim = input_dim
                    self.is_available = True
                    logger.info(
                        f"Loaded {model_type} word model from {CNN_LSTM_MODEL_PATH} "
                        f"({num_classes} classes, in_dim={input_dim}, attn={use_attn})"
                    )
                    return
                else:
                    logger.warning(
                        f"Unsupported CNN word checkpoint model_type={model_type}; trying fallback models"
                    )
            except Exception as e:
                logger.warning(f"Failed to load CNN word model from {CNN_LSTM_MODEL_PATH}: {e}")

        # 2. Try Scikit-Learn/XGBoost sequence model (.pkl)
        if os.path.exists(PKL_MODEL_PATH):
            try:
                with open(PKL_MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                if os.path.exists(PKL_META_PATH):
                    with open(PKL_META_PATH, 'r', encoding='utf-8') as f:
                        self.metadata = json.load(f)
                self.labels = self.metadata.get('labels', [])
                self.mode = 'sklearn'
                self.is_available = True
                logger.info(f"Loaded Sequence Word Classifier from {PKL_MODEL_PATH} ({len(self.labels)} classes: {', '.join(self.labels[:10])})")
                return
            except Exception as e:
                logger.warning(f"Failed to load PKL word model: {e}")

        # 3. Try PyTorch model (.pt)
        if os.path.exists(PT_MODEL_PATH):
            try:
                import torch
                import torch.nn as nn

                class BiLSTMWordClassifier(nn.Module):
                    def __init__(self, input_dim=126, hidden_dim=128, num_classes=17, num_layers=2, dropout=0.3):
                        super().__init__()
                        self.lstm = nn.LSTM(
                            input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0
                        )
                        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)
                        self.fc1 = nn.Linear(hidden_dim * 2, 64)
                        self.relu = nn.ReLU()
                        self.dropout = nn.Dropout(dropout)
                        self.fc2 = nn.Linear(64, num_classes)
                        
                    def forward(self, x):
                        lstm_out, _ = self.lstm(x)
                        pooled = torch.mean(lstm_out, dim=1)
                        normed = self.batch_norm(pooled)
                        out = self.fc1(normed)
                        out = self.relu(out)
                        out = self.dropout(out)
                        return self.fc2(out)

                if os.path.exists(LSTM_META_PATH):
                    with open(LSTM_META_PATH, 'r', encoding='utf-8') as f:
                        self.metadata = json.load(f)

                checkpoint = torch.load(PT_MODEL_PATH, map_location='cpu', weights_only=True)
                self.labels = checkpoint.get('word_labels') or self.metadata.get('labels', [])
                num_classes = len(self.labels)

                model = BiLSTMWordClassifier(num_classes=num_classes)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                self.model = model
                self.mode = 'pytorch'
                self.is_available = True
                logger.info(f"Loaded PyTorch Bi-LSTM word model from {PT_MODEL_PATH}")
                return
            except Exception as e:
                logger.warning(f"Failed to load PyTorch LSTM word model: {e}")

        # 3. Try Keras model fallback (.h5)
        if os.path.exists(LSTM_MODEL_PATH):
            try:
                import tensorflow as tf
                if os.path.exists(LSTM_META_PATH):
                    with open(LSTM_META_PATH, 'r', encoding='utf-8') as f:
                        self.metadata = json.load(f)

                self.model = tf.keras.models.load_model(LSTM_MODEL_PATH)
                self.labels = self.metadata.get('labels', [])
                self.mode = 'keras'
                self.is_available = True
                logger.info(f"Loaded Keras Bi-LSTM word model from {LSTM_MODEL_PATH}")
                return
            except Exception as e:
                logger.warning(f"Failed to load Keras LSTM word model: {e}")

        logger.info("Word model not found. Word recognition is disabled.")

    def predict(self, frame_sequence, body_anchors=None):
        """
        Predict a word from a sequence of landmark frames with optional MediaPipe Holistic body anchors.

        Args:
            frame_sequence: list of 30 frames, each frame is a list/array
                            of 126 floats (42 landmarks x 3 coordinates).
            body_anchors: optional dict of upper-body relative anchors from MediaPipe Holistic.
        """
        if not self.is_available or self.model is None:
            return None

        try:
            from services.translator_model import normalize_landmarks_batch
            from services.word_preprocessing import preprocess_word_sequence
            from services.feature_extractor import transform_to_kinetic_invariants

            arr = np.asarray(frame_sequence, dtype=np.float32)

            if arr.ndim == 2:
                if arr.shape != (SEQUENCE_LENGTH, NUM_FEATURES):
                    if arr.shape[1] != NUM_FEATURES:
                        return None
                    arr = self._pad_or_trim(arr)
            elif arr.ndim == 3:
                arr = arr.reshape(-1, NUM_FEATURES)
                arr = self._pad_or_trim(arr)
            else:
                return None

            # Guard: If no valid hand coordinates exist across the sequence, return no prediction
            if not np.any(arr != 0):
                return {
                    'word': '?',
                    'confidence': 0.0,
                    'mode': self.mode,
                    'all_scores': {},
                }

            # Prepare the legacy 126-D representation only for model families
            # that were trained to consume normalized landmark coordinates.
            norm_arr = None
            if self.mode in ('sklearn', 'pytorch', 'keras') or getattr(self, 'input_dim', None) == 126:
                
                norm_arr = preprocess_word_sequence(
                    arr,
                    include_velocity=False,
                    include_mask=False
                )

                if norm_arr is None:
                    return {
                        'word': '?',
                        'confidence': 0.0,
                        'mode': self.mode,
                        'all_scores': {},
                    }

            # Motion velocity pre-gate: wrist-relative joint displacement velocity
            # Rejects static/resting hands while suppressing glitch/tracking jumps (> 0.15)
            motion_velocity = compute_wrist_relative_motion(arr, MAX_SINGLE_FRAME_JUMP)

            if motion_velocity < MIN_MOTION_VELOCITY:
                return {
                    'word': '?',
                    'confidence': 0.0,
                    'mode': self.mode,
                    'rejected': True,
                    'rejection_reason': 'static_hand_detected',
                    'motion_velocity': round(motion_velocity, 5),
                    'all_scores': {},
                }

            model = self.model

            norm_arr = preprocess_word_sequence(
                arr,
                include_velocity=False,
                include_mask=False
            )
            if self.mode in ('cnn_lstm', 'cnn_gru'):
                input_dim = int(getattr(self, 'input_dim', NUM_FEATURES))

                if input_dim == 254:
                    # Training pipeline: 126 normalized coordinates + 2 hand masks + 126 velocities.
                    input_data = preprocess_word_sequence(
                        arr, include_velocity=True, include_mask=True
                    )
                elif input_dim == 128:
                    # Training ablation: 126 normalized coordinates + 2 hand masks.
                    input_data = preprocess_word_sequence(
                        arr, include_velocity=False, include_mask=True
                    )
                elif input_dim == 252:
                    # Legacy kinetic-invariant model. Pass RAW coordinates because
                    # transform_to_kinetic_invariants performs its own normalization.
                    input_data = transform_to_kinetic_invariants(arr)
                elif input_dim == 126:
                    input_data = norm_arr
                else:
                    logger.error(f'Unsupported word model input dimension: {input_dim}')
                    return None

                input_data = np.asarray(input_data, dtype=np.float32).reshape(
                    1, SEQUENCE_LENGTH, input_dim
                )
                import torch
                tensor_in = torch.tensor(input_data, dtype=torch.float32)
                with torch.no_grad():
                    logits = model(tensor_in)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            
            elif self.mode == 'pytorch':
                input_data = norm_arr.reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)
                import torch
                tensor_in = torch.tensor(input_data, dtype=torch.float32)
                with torch.no_grad():
                    logits = model(tensor_in)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            elif self.mode == 'sklearn':
                flat_in = norm_arr.flatten().reshape(1, -1)
                probs = model.predict_proba(flat_in)[0]

            elif self.mode == 'keras':
                input_data = norm_arr.reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)
                probs = model.predict(input_data, verbose=0)[0]

            else:
                logger.error(f'Unsupported word recognizer mode: {self.mode}')
                return None

            # Multimodal Holistic anatomical re-weighting
            if body_anchors and isinstance(body_anchors, dict) and len(probs) == len(self.labels):
                l_mouth = float(body_anchors.get('l_hand_to_mouth', 999.0))
                r_mouth = float(body_anchors.get('r_hand_to_mouth', 999.0))
                min_mouth = min(l_mouth, r_mouth)

                l_chest = float(body_anchors.get('l_hand_to_chest', 999.0))
                r_chest = float(body_anchors.get('r_hand_to_chest', 999.0))
                min_chest = min(l_chest, r_chest)

                l_elev = float(body_anchors.get('l_elevation', 0.0))
                r_elev = float(body_anchors.get('r_elevation', 0.0))
                max_elev = max(l_elev, r_elev)
                min_elev = min(l_elev, r_elev)

                priors = np.ones(len(self.labels), dtype=np.float32)

                for idx, lbl in enumerate(self.labels):
                    # Facial / Mouth signs: THANK_YOU, WATER, FOOD
                    if lbl in ('THANK_YOU', 'WATER', 'FOOD', 'DOCTOR'):
                        if min_mouth < 0.40:
                            priors[idx] *= 1.45
                        elif min_mouth > 0.85:
                            priors[idx] *= 0.50

                    # Chest signs: NAMASTE, PLEASE, SORRY, ME, HEARING
                    elif lbl in ('NAMASTE', 'PLEASE', 'SORRY', 'ME', 'HEARING'):
                        if min_chest < 0.35 and max_elev > -0.15:
                            # Only mild prior boost when actively signing at chest height
                            priors[idx] *= 1.15
                        elif min_chest > 0.80:
                            priors[idx] *= 0.55

                    # High signs (above shoulder level): HELLO, BYE_BYE, INDIA
                    elif lbl in ('HELLO', 'BYE_BYE', 'INDIA'):
                        if max_elev > 0.05:
                            priors[idx] *= 1.35
                        elif max_elev < -0.25:
                            priors[idx] *= 0.60

                    # Lower body / waist signs: WASHROOM
                    elif lbl == 'WASHROOM':
                        if min_elev < -0.30:
                            priors[idx] *= 1.40
                        elif max_elev > 0.10:
                            priors[idx] *= 0.40

                probs = probs * priors
                sum_p = np.sum(probs)
                if sum_p > 1e-6:
                    probs = probs / sum_p

            top_idx = int(np.argmax(probs))
            confidence = float(probs[top_idx])
            word = self.labels[top_idx] if top_idx < len(self.labels) else '?'

            # Shannon entropy across class distribution
            entropy = -float(np.sum(probs * np.log(probs + 1e-9)))

            sorted_probs = np.sort(probs)[::-1]
            margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else float(sorted_probs[0])
            is_rejected = bool(
                word in ('NONE', 'IDLE', '?') or
                confidence < CONFIDENCE_THRESHOLD_WORD or
                margin < MARGIN_THRESHOLD_WORD or
                entropy > MAX_WORD_ENTROPY
            )
            rejection_reason = None
            if is_rejected:
                if word in ('NONE', 'IDLE'):
                    rejection_reason = 'idle_detected'
                elif confidence < CONFIDENCE_THRESHOLD_WORD:
                    rejection_reason = 'low_confidence'
                elif margin < MARGIN_THRESHOLD_WORD:
                    rejection_reason = 'low_margin'
                elif entropy > MAX_WORD_ENTROPY:
                    rejection_reason = 'high_entropy'
                else:
                    rejection_reason = 'unknown_gesture'

            sorted_indices = np.argsort(probs)[::-1][:5]
            top_scores = {
                self.labels[i]: round(float(probs[i]), 4)
                for i in sorted_indices if i < len(self.labels)
            }

            return {
                'word': '?' if is_rejected else word,
                'confidence': round(confidence, 4),
                'margin': round(margin, 4),
                'entropy': round(entropy, 4),
                'mode': self.mode,
                'rejected': is_rejected,
                'rejection_reason': rejection_reason,
                'motion_velocity': round(motion_velocity, 5),
                'all_scores': top_scores,
            }

        except Exception as e:
            logger.error(f"Word recognition prediction failed: {e}")
            return None

    def _pad_or_trim(self, frames):
        """Ensure frames array is exactly SEQUENCE_LENGTH frames long."""
        if len(frames) < SEQUENCE_LENGTH:
            if len(frames) > 0:
                last_frame = frames[-1:]
                pad = np.repeat(last_frame, SEQUENCE_LENGTH - len(frames), axis=0)
                return np.concatenate([frames, pad], axis=0)
            return np.zeros((SEQUENCE_LENGTH, NUM_FEATURES), dtype=np.float32)
        elif len(frames) > SEQUENCE_LENGTH:
            indices = np.linspace(0, len(frames) - 1, SEQUENCE_LENGTH, dtype=int)
            return frames[indices]
        return frames

    def get_info(self):
        """Return word recognizer metadata."""
        return {
            'available': self.is_available,
            'model_mode': self.mode,
            'labels': self.labels,
            'num_classes': len(self.labels),
            'sequence_length': SEQUENCE_LENGTH,
            'input_features': getattr(self, 'input_dim', NUM_FEATURES),
            'validation_accuracy': self.metadata.get('val_accuracy'),
        }
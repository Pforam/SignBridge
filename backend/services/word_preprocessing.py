"""
SignBridge — Centralized Word Landmark Preprocessing & Normalization.

Single source of truth for ISL sequence preprocessing across both training and inference.
Features provided per 30-frame sequence:
1. Per-hand wrist centering and palm scale normalization (126 dims)
2. Hand presence mask [left_present, right_present] (2 dims)
3. Inter-frame velocity deltas with missing-hand suppression (126 dims)
Combined shape: (30, 254) or (30, 128) without velocity.
"""

from typing import Tuple, Optional
import numpy as np

SEQUENCE_LENGTH = 30
RAW_FEATURES = 126
NUM_LANDMARKS_PER_HAND = 21
COORDS_PER_HAND = 63  # 21 * 3


def normalize_frame(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalizes a single 126-float landmark frame:
    - Centers each active hand on its wrist.
    - Scales each active hand by its palm length (wrist to middle MCP).
    - Returns normalized coordinates (126,) and presence mask (2,).
    """
    f = np.asarray(frame, dtype=np.float32).reshape(2, 21, 3)
    norm = np.zeros_like(f)
    mask = np.zeros(2, dtype=np.float32)

    for h in range(2):
        hand = f[h]
        # Check presence
        if np.any(np.abs(hand) > 1e-4):
            mask[h] = 1.0
            wrist = hand[0]  # Landmark 0: WRIST
            centered = hand - wrist

            # Scale reference: Euclidean distance from wrist (0) to middle finger MCP (9)
            palm_len = float(np.linalg.norm(centered[9]))
            scale = palm_len if palm_len > 1e-4 else 1.0

            norm[h] = centered / scale

    return norm.flatten(), mask


def preprocess_word_sequence(
    raw_sequences: np.ndarray,
    include_velocity: bool = True,
    include_mask: bool = True
) -> np.ndarray:
    """
    Preprocess raw landmark sequence(s) into normalized model input.
    Supports single sequence of shape (T, 126) or batch of shape (N, T, 126).

    Args:
        raw_sequences: array of shape (T, 126) or (N, T, 126).
        include_velocity: whether to append 126-dim inter-frame velocity deltas.
        include_mask: whether to append 2-dim hand presence mask.

    Returns:
        np.ndarray of shape (T, feature_dim) or (N, T, feature_dim).
        Feature dim is 254 (126 coords + 2 mask + 126 velocity) or 128 (no velocity).
    """
    arr = np.asarray(raw_sequences, dtype=np.float32)
    single = (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[1] != SEQUENCE_LENGTH and arr.shape[2] == 3))
    if single:
        if arr.ndim == 3:
            arr = arr.reshape(arr.shape[0], -1)
        arr = arr[None, ...]
    elif arr.ndim == 4:
        arr = arr.reshape(arr.shape[0], arr.shape[1], -1)

    N, T, D = arr.shape
    out_list = []

    for i in range(N):
        seq = arr[i]
        norm_frames = np.zeros((T, RAW_FEATURES), dtype=np.float32)
        masks = np.zeros((T, 2), dtype=np.float32)

        for t in range(T):
            norm_frames[t], masks[t] = normalize_frame(seq[t])

        features = [norm_frames]

        if include_mask:
            features.append(masks)

        if include_velocity:
            velocities = np.zeros((T, RAW_FEATURES), dtype=np.float32)
            v_h = velocities.reshape(T, 2, COORDS_PER_HAND)
            n_h = norm_frames.reshape(T, 2, COORDS_PER_HAND)

            for t in range(1, T):
                for h in range(2):
                    if masks[t - 1, h] > 0.5 and masks[t, h] > 0.5:
                        delta = n_h[t, h] - n_h[t - 1, h]
                        v_h[t, h] = np.clip(delta, -0.3, 0.3)

            features.append(velocities)

        out_list.append(np.concatenate(features, axis=-1))

    out_arr = np.array(out_list, dtype=np.float32)
    return out_arr[0] if single else out_arr


def compute_normalized_motion(raw_sequence: np.ndarray, max_jump: float = 0.25) -> float:
    """
    Computes mean joint motion across normalized landmarks.
    Used for the normalized motion pre-gate (> 0.015) to reject static gestures.
    """
    seq = np.asarray(raw_sequence, dtype=np.float32)
    if seq.ndim == 3:
        seq = seq.reshape(seq.shape[0], -1)
    T = seq.shape[0]
    if T < 2:
        return 0.0

    # Extract normalized frames and masks
    norm_frames = np.zeros((T, RAW_FEATURES), dtype=np.float32)
    masks = np.zeros((T, 2), dtype=np.float32)
    for t in range(T):
        norm_frames[t], masks[t] = normalize_frame(seq[t])

    n_h = norm_frames.reshape(T, 2, 21, 3)
    deltas = []

    for t in range(T - 1):
        for h in range(2):
            if masks[t, h] > 0.5 and masks[t + 1, h] > 0.5:
                diff = np.linalg.norm(n_h[t + 1, h] - n_h[t, h], axis=-1)
                valid = diff[diff < max_jump]
                if len(valid) > 0:
                    deltas.append(float(np.mean(valid)))

    return float(np.mean(deltas)) if deltas else 0.0

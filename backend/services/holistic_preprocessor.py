"""
SignBridge — Centralized MediaPipe Holistic Preprocessing & Feature Extraction.

Extracts a 328-dimensional normalized kinematic representation per frame (30 frames):
  - 126 dims: Wrist-centered, palm-scaled dual-hand coordinates
  - 2 dims: Hand presence binary masks [left_present, right_present]
  - 30 dims: Chest-centered, shoulder-normalized upper-body pose keypoints
  - 7 dims: Anatomical body-relative spatial anchors:
            [l_to_mouth, r_to_mouth, l_to_chest, r_to_chest, l_elevation, r_elevation, inter_wrist_dist]
  - 163 dims: Inter-frame kinematic velocity deltas

Total input shape per sequence: (30, 328)
"""

from typing import Tuple, Optional, Union
import numpy as np

SEQUENCE_LENGTH = 30
RAW_HOLISTIC_DIMS = 156
STATIC_FEATURE_DIMS = 165
DELTA_FEATURE_DIMS = 163
COMBINED_FEATURE_DIMS = 328  # 165 static + 163 deltas


def extract_holistic_frame_features(raw_frame: np.ndarray) -> np.ndarray:
    """
    Transforms a 156-dim raw frame (or 126-dim hand-only frame) into 165 static features.
    """
    f = np.asarray(raw_frame, dtype=np.float32)
    if f.size < 126:
        return np.zeros(STATIC_FEATURE_DIMS, dtype=np.float32)

    hand_feats = np.zeros(126, dtype=np.float32)
    mask = np.zeros(2, dtype=np.float32)

    # 1. Left Hand (indices 0..62)
    lh = f[:63].reshape(21, 3)
    if np.any(np.abs(lh) > 1e-4):
        mask[0] = 1.0
        wrist = lh[0].copy()
        centered = lh - wrist
        palm_len = float(np.linalg.norm(centered[9]))  # wrist to middle MCP
        scale = palm_len if palm_len > 1e-4 else 1.0
        hand_feats[:63] = (centered / scale).flatten()
        left_wrist_pos = wrist
    else:
        left_wrist_pos = None

    # 2. Right Hand (indices 63..125)
    rh = f[63:126].reshape(21, 3)
    if np.any(np.abs(rh) > 1e-4):
        mask[1] = 1.0
        wrist = rh[0].copy()
        centered = rh - wrist
        palm_len = float(np.linalg.norm(centered[9]))
        scale = palm_len if palm_len > 1e-4 else 1.0
        hand_feats[63:126] = (centered / scale).flatten()
        right_wrist_pos = wrist
    else:
        right_wrist_pos = None

    # 3. Upper-Body Pose (indices 126..155 if present)
    body_feats = np.zeros(30, dtype=np.float32)
    body_scalars = np.zeros(7, dtype=np.float32)

    if f.size >= 156 and np.any(np.abs(f[126:156]) > 1e-4):
        pose_raw = f[126:156].reshape(10, 3)
        # 0: nose, 1: l_sh, 2: r_sh, 3: l_el, 4: r_el, 5: l_wr, 6: r_wr, 7: l_idx, 8: r_idx, 9: mid_hip
        l_sh = pose_raw[1]
        r_sh = pose_raw[2]
        chest = (l_sh + r_sh) * 0.5
        sh_dist = float(np.linalg.norm(l_sh - r_sh))
        scale = sh_dist if sh_dist > 1e-4 else 1.0

        for i in range(10):
            body_feats[i * 3 : (i + 1) * 3] = (pose_raw[i] - chest) / scale

        nose = pose_raw[0]
        wl = left_wrist_pos if left_wrist_pos is not None else pose_raw[5]
        wr = right_wrist_pos if right_wrist_pos is not None else pose_raw[6]

        body_scalars[0] = float(np.linalg.norm(wl - nose)) / scale
        body_scalars[1] = float(np.linalg.norm(wr - nose)) / scale
        body_scalars[2] = float(np.linalg.norm(wl - chest)) / scale
        body_scalars[3] = float(np.linalg.norm(wr - chest)) / scale
        body_scalars[4] = (l_sh[1] - wl[1]) / scale  # elevation
        body_scalars[5] = (r_sh[1] - wr[1]) / scale

        if left_wrist_pos is not None and right_wrist_pos is not None:
            body_scalars[6] = float(np.linalg.norm(left_wrist_pos - right_wrist_pos)) / scale
        else:
            body_scalars[6] = float(np.linalg.norm(pose_raw[5] - pose_raw[6])) / scale

    return np.concatenate([hand_feats, mask, body_feats, body_scalars])


def preprocess_holistic_sequence(
    raw_sequence: np.ndarray,
    include_velocity: bool = True
) -> np.ndarray:
    """
    Transforms a sequence of shape (T, 156) or (N, T, 156) into (T, 328) or (N, T, 328).
    """
    arr = np.asarray(raw_sequence, dtype=np.float32)
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
        static_frames = np.zeros((T, STATIC_FEATURE_DIMS), dtype=np.float32)
        for t in range(T):
            static_frames[t] = extract_holistic_frame_features(seq[t])

        if not include_velocity:
            out_list.append(static_frames)
            continue

        # Continuous static features (all except 2 binary masks at indices 126, 127)
        cont_indices = np.concatenate([np.arange(0, 126), np.arange(128, STATIC_FEATURE_DIMS)])
        cont_features = static_frames[:, cont_indices]  # (T, 163)

        velocities = np.zeros_like(cont_features)
        velocities[1:] = cont_features[1:] - cont_features[:-1]

        # Suppress artificial jumps when hands transition present <-> absent
        masks = static_frames[:, 126:128]
        for t in range(1, T):
            # Left hand jumps
            if masks[t - 1, 0] < 0.5 or masks[t, 0] < 0.5:
                velocities[t, :63] = 0.0
            # Right hand jumps
            if masks[t - 1, 1] < 0.5 or masks[t, 1] < 0.5:
                velocities[t, 63:126] = 0.0

        combined = np.concatenate([static_frames, velocities], axis=-1)  # (T, 328)
        out_list.append(combined)

    res = np.stack(out_list, axis=0)
    return res[0] if single else res


def swap_holistic_sequence(raw_sequence: np.ndarray) -> np.ndarray:
    """
    Swaps left and right hand slots across raw 156-dim frame sequence for hand invariance.
    """
    swapped = raw_sequence.copy()
    if swapped.ndim == 2:
        # Swap hand coordinates
        swapped[:, :63] = raw_sequence[:, 63:126]
        swapped[:, 63:126] = raw_sequence[:, :63]
        if swapped.shape[1] >= 156:
            # Swap pose left/right keypoints:
            # l_sh (1) <-> r_sh (2)
            # l_el (3) <-> r_el (4)
            # l_wr (5) <-> r_wr (6)
            # l_idx (7) <-> r_idx (8)
            p_base = 126
            pairs = [(1, 2), (3, 4), (5, 6), (7, 8)]
            for p1, p2 in pairs:
                idx1 = p_base + p1 * 3
                idx2 = p_base + p2 * 3
                tmp = swapped[:, idx1 : idx1 + 3].copy()
                swapped[:, idx1 : idx1 + 3] = swapped[:, idx2 : idx2 + 3]
                swapped[:, idx2 : idx2 + 3] = tmp
    return swapped

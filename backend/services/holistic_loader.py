"""
SignBridge — MediaPipe Holistic Dataset Loader (landmark_output)

Streams and parses 266 MediaPipe Holistic video extractions from landmark_output/.
Extracts 156-dim raw coordinates per frame:
  - 126 floats: Dual hands (21 left * 3 + 21 right * 3)
  - 30 floats: Key upper-body pose landmarks (10 body points * 3)

Preserves video identity, signer session, and source path for leak-free partitioning.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Canonical label normalization mapping
LABEL_NORMALIZATION = {
    "ME,I": "ME",
    "ME_I": "ME",
    "I-AM-FINE": "IM_FINE",
    "I_AM_FINE": "IM_FINE",
    "BYE-BYE": "BYE_BYE",
    "GOOD-MORNING": "GOOD_MORNING",
    "GOOD-AFTERNOON": "GOOD_AFTERNOON",
    "GOOD-EVENING": "GOOD_EVENING",
    "GOOD-NIGHT": "GOOD_NIGHT",
    "HOW-ARE-YOU": "HOW_ARE_YOU",
    "I-DONT-UNDERSTAND": "I_DONT_UNDERSTAND",
    "MY-NAME-IS": "MY_NAME_IS",
    "NO-FEAR": "NO_FEAR",
    "FEAR-NOT": "FEAR_NOT",
    "THANK-YOU": "THANK_YOU",
    "THANK-YOU-VERY-MUCH": "THANK_YOU_VERY_MUCH",
    "INDIA": "INDIAN", # standardize to canonical folder label
}

KEY_POSE_INDICES = [0, 11, 12, 13, 14, 15, 16, 19, 20]  # nose, shoulders, elbows, wrists, index tips


def normalize_label(raw_label: str) -> str:
    """Standardize label names across datasets."""
    cleaned = raw_label.strip().upper().replace(" ", "_").replace("-", "_")
    return LABEL_NORMALIZATION.get(cleaned, cleaned)


def parse_holistic_frame(rec: Dict[str, Any]) -> np.ndarray:
    """
    Parses a single JSONL record into a 156-float raw feature vector:
    - 0..62: Left hand (21 * 3) or zeros if null
    - 63..125: Right hand (21 * 3) or zeros if null
    - 126..155: Upper-body pose points (10 points * 3) or zeros if null
    """
    frame_vec = np.zeros(156, dtype=np.float32)

    # Left Hand (21 points)
    lh = rec.get("left_hand")
    if lh and len(lh) == 21:
        for idx, pt in enumerate(lh):
            base = idx * 3
            frame_vec[base] = pt.get("x", 0.0)
            frame_vec[base + 1] = pt.get("y", 0.0)
            frame_vec[base + 2] = pt.get("z", 0.0)

    # Right Hand (21 points)
    rh = rec.get("right_hand")
    if rh and len(rh) == 21:
        for idx, pt in enumerate(rh):
            base = 63 + idx * 3
            frame_vec[base] = pt.get("x", 0.0)
            frame_vec[base + 1] = pt.get("y", 0.0)
            frame_vec[base + 2] = pt.get("z", 0.0)

    # Upper-Body Pose (10 key points * 3 = 30 floats)
    pose = rec.get("pose")
    if pose and len(pose) >= 25:
        p_dict = {i: (pose[i].get("x", 0.0), pose[i].get("y", 0.0), pose[i].get("z", 0.0)) for i in range(len(pose))}
        offset = 126
        for idx_i, ki in enumerate(KEY_POSE_INDICES):
            px, py, pz = p_dict.get(ki, (0.0, 0.0, 0.0))
            frame_vec[offset + idx_i * 3] = px
            frame_vec[offset + idx_i * 3 + 1] = py
            frame_vec[offset + idx_i * 3 + 2] = pz

        # Mid-hip waist anchor point (index 9 in pose slice)
        hx = (p_dict.get(23, (0.0, 0.0, 0.0))[0] + p_dict.get(24, (0.0, 0.0, 0.0))[0]) * 0.5
        hy = (p_dict.get(23, (0.0, 0.0, 0.0))[1] + p_dict.get(24, (0.0, 0.0, 0.0))[1]) * 0.5
        hz = (p_dict.get(23, (0.0, 0.0, 0.0))[2] + p_dict.get(24, (0.0, 0.0, 0.0))[2]) * 0.5
        frame_vec[offset + 27] = hx
        frame_vec[offset + 28] = hy
        frame_vec[offset + 29] = hz

    return frame_vec


def resample_sequence(frames: np.ndarray, target_length: int = 30) -> np.ndarray:
    """
    Linearly resamples a variable-length sequence (N, D) to (target_length, D).
    """
    n_frames = len(frames)
    if n_frames == target_length:
        return frames.astype(np.float32)
    if n_frames == 0:
        return np.zeros((target_length, frames.shape[1] if frames.ndim > 1 else 156), dtype=np.float32)

    indices = np.linspace(0, n_frames - 1, target_length)
    idx_floor = np.floor(indices).astype(int)
    idx_ceil = np.clip(idx_floor + 1, 0, n_frames - 1)
    weight = (indices - idx_floor)[:, None]

    resampled = (1.0 - weight) * frames[idx_floor] + weight * frames[idx_ceil]
    return resampled.astype(np.float32)


def load_holistic_dataset(
    dataset_dir: str = "d:/Sign-Bridge-/landmark_output",
    target_length: int = 30
) -> List[Dict[str, Any]]:
    """
    Loads all videos from landmark_output/ into structured sequence records.
    Each item in the returned list contains:
      - video_id: str (folder name)
      - label: str (normalized canonical sign label)
      - source_video: str
      - raw_frames: np.ndarray of shape (N, 156)
      - frames_30: np.ndarray of shape (30, 156)
      - frame_count: int
      - fps: float
      - coverage: dict
    """
    p = Path(dataset_dir)
    if not p.exists():
        raise FileNotFoundError(f"landmark_output directory not found at: {dataset_dir}")

    items = []
    subdirs = sorted([d for d in p.iterdir() if d.is_dir()])

    for d in subdirs:
        summary_path = d / "summary.json"
        jsonl_path = d / "landmarks.jsonl"
        if not summary_path.exists() or not jsonl_path.exists():
            continue

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        raw_label = summary.get("label", "")
        norm_lbl = normalize_label(raw_label)

        frames = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    rec = json.loads(line_str)
                    frames.append(parse_holistic_frame(rec))

        if not frames:
            continue

        frames_arr = np.array(frames, dtype=np.float32)
        frames_30 = resample_sequence(frames_arr, target_length=target_length)

        items.append({
            "video_id": d.name,
            "label": norm_lbl,
            "raw_label": raw_label,
            "source_video": summary.get("source_video", ""),
            "raw_frames": frames_arr,
            "frames_30": frames_30,
            "frame_count": len(frames),
            "fps": summary.get("fps_used", 30.0),
            "coverage": summary.get("detection_coverage", {}),
        })

    logger.info(f"Loaded {len(items)} holistic video sequences across {len(set(x['label'] for x in items))} classes.")
    return items

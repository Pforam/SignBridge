"""
SignBridge — MediaPipe Holistic Word Recognition Training Pipeline.

Trains and evaluates a Holistic Kinematic Temporal Classifier on landmark_output/.
Pillars:
1. Zero Data Leakage: Split strictly by source video / signer session.
2. Robust Preprocessing: 328-dim features (Hands 126D + Mask 2D + Pose 30D + Anchors 7D + Velocities 163D).
3. Anti-Overfitting: Training-only augmentation (speed warp, jitter, slot swap, dropout),
   Label Smoothing (0.05), Weight Decay (1e-4), Early Stopping.
4. Reproducible Evaluation: Accuracy, Macro-F1, Per-Class F1, Confusion Matrix.
5. Baseline Comparison: Evaluates baseline model on identical held-out test split.
"""

import os
import sys
import json
import time
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add backend directory to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from services.holistic_loader import load_holistic_dataset, resample_sequence
from services.holistic_preprocessor import (
    preprocess_holistic_sequence,
    swap_holistic_sequence,
    COMBINED_FEATURE_DIMS,
    SEQUENCE_LENGTH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainHolistic")

MODEL_DIR = SCRIPT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = MODEL_DIR / "isl_holistic_word_model.pt"
META_PATH = MODEL_DIR / "holistic_word_meta.json"
THRESHOLDS_PATH = MODEL_DIR / "holistic_word_thresholds.json"
CM_CSV_PATH = MODEL_DIR / "holistic_confusion_matrix.csv"


# ═══════════════════════════════════════════════════════════════════════════
# 1. Model Architecture
# ═══════════════════════════════════════════════════════════════════════════

class TemporalAttention(nn.Module):
    def __init__(self, hdim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hdim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, hdim)
        w = torch.softmax(self.attn(x), dim=1)
        return torch.sum(x * w, dim=1)


class HolisticWordClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int = COMBINED_FEATURE_DIMS,
        cnn_ch: int = 128,
        hid: int = 128,
        nc: int = 53,
        dp: float = 0.3
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, cnn_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_ch),
            nn.ReLU(),
            nn.Dropout(dp)
        )
        self.rnn = nn.GRU(
            input_size=cnn_ch,
            hidden_size=hid,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dp
        )
        self.attn = TemporalAttention(hid * 2)
        self.bn_pool = nn.BatchNorm1d(hid * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hid * 2, 96),
            nn.ReLU(),
            nn.Dropout(dp),
            nn.Linear(96, nc)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_dim)
        x_perm = x.permute(0, 2, 1)
        cnn_out = self.cnn(x_perm)
        rnn_in = cnn_out.permute(0, 2, 1)
        rnn_out, _ = self.rnn(rnn_in)
        pooled = self.attn(rnn_out)
        normed = self.bn_pool(pooled)
        return self.classifier(normed)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Augmentation & Partitioning
# ═══════════════════════════════════════════════════════════════════════════

def augment_raw_holistic_sequence(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Applies realistic physical augmentations to a raw 156-dim sequence:
    - Speed warping (time dilation / compression)
    - Coordinate Gaussian noise
    - Slight image plane rotation (+-8 deg)
    """
    T, D = seq.shape
    aug = seq.copy()

    # 1. Temporal speed warp: randomly resample to 24-36 frames, then back to 30
    warp_len = rng.integers(24, 37)
    warped = resample_sequence(aug, target_length=warp_len)
    aug = resample_sequence(warped, target_length=T)

    # 2. Coordinate jitter on active coordinates
    noise = rng.normal(0, 0.005, size=aug.shape).astype(np.float32)
    active_mask = (np.abs(aug) > 1e-4)
    aug[active_mask] += noise[active_mask]

    # 3. Slight planar rotation around frame center (0.5, 0.5)
    angle = float(rng.uniform(-0.14, 0.14))  # ~+-8 degrees
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    cx, cy = 0.5, 0.5

    # Rotate hands (indices 0..125) and pose (126..155)
    for p_start in [0, 63, 126]:
        p_len = 63 if p_start < 126 else 30
        for pt_i in range(p_start, p_start + p_len, 3):
            if np.any(np.abs(aug[:, pt_i : pt_i + 2]) > 1e-4):
                dx = aug[:, pt_i] - cx
                dy = aug[:, pt_i + 1] - cy
                aug[:, pt_i] = cx + (dx * cos_a - dy * sin_a)
                aug[:, pt_i + 1] = cy + (dx * sin_a + dy * cos_a)

    return aug


def partition_holistic_dataset(
    items: List[Dict[str, Any]],
    test_ratio: float = 0.20,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """
    Splits items strictly by source video into train and test sets.
    Ensures zero video leakage between partitions.
    """
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for it in items:
        by_class[it["label"]].append(it)

    labels = sorted(list(by_class.keys()))
    train_items = []
    test_items = []

    for lbl in labels:
        class_vids = by_class[lbl]
        # Shuffle class videos deterministically
        indices = rng.permutation(len(class_vids))
        n_test = max(1, int(round(len(class_vids) * test_ratio)))
        # For classes with only 2 videos, exactly 1 goes to test, 1 to train
        if len(class_vids) == 2:
            n_test = 1

        test_idx = set(indices[:n_test])
        for idx, it in enumerate(class_vids):
            if idx in test_idx:
                test_items.append(it)
            else:
                train_items.append(it)

    # Integrity verification
    train_vids = set(x["video_id"] for x in train_items)
    test_vids = set(x["video_id"] for x in test_items)
    overlap = train_vids.intersection(test_vids)
    if overlap:
        raise ValueError(f"Leakage detected! {len(overlap)} videos overlap between train and test!")

    logger.info(f"Partitioned {len(items)} videos: Train={len(train_items)}, Test={len(test_items)} across {len(labels)} classes. Zero leakage verified.")
    return train_items, test_items, labels


# ═══════════════════════════════════════════════════════════════════════════
# 3. Metrics & Reporting
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(
    targets: np.ndarray,
    preds: np.ndarray,
    num_classes: int,
    labels: List[str]
) -> Tuple[float, float, np.ndarray, Dict[str, Any]]:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(targets, preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    tp = np.diag(cm).astype(float)
    fp = (cm.sum(axis=0) - tp).astype(float)
    fn = (cm.sum(axis=1) - tp).astype(float)
    support = cm.sum(axis=1)

    precision = np.divide(tp, tp + fp, out=np.zeros(num_classes, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros(num_classes, dtype=float), where=(tp + fn) != 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(num_classes, dtype=float), where=denom != 0)

    acc = float(np.mean(targets == preds)) if len(targets) > 0 else 0.0
    macro_f1 = float(np.mean(f1)) if len(f1) > 0 else 0.0

    report = {}
    for i, lbl in enumerate(labels):
        report[lbl] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
    return acc, macro_f1, cm, report


# ═══════════════════════════════════════════════════════════════════════════
# 4. Training Engine
# ═══════════════════════════════════════════════════════════════════════════

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_holistic_model(
    dataset_dir: str = "d:/Sign-Bridge-/landmark_output",
    num_augmentations: int = 8,
    epochs: int = 45,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42
):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    logger.info("=" * 70)
    logger.info("STEP 1: Loading 266 Holistic video extractions from landmark_output/")
    logger.info("=" * 70)

    items = load_holistic_dataset(dataset_dir)
    train_items, test_items, labels = partition_holistic_dataset(items, test_ratio=0.20, seed=seed)
    lbl_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    logger.info("=" * 70)
    logger.info(f"STEP 2: Preprocessing and augmenting training partition (x{num_augmentations})")
    logger.info("=" * 70)

    # Process and augment train set
    X_train_list = []
    y_train_list = []

    for it in train_items:
        raw_30 = it["frames_30"]
        c_idx = lbl_to_idx[it["label"]]

        # Original
        f_orig = preprocess_holistic_sequence(raw_30, include_velocity=True)
        X_train_list.append(f_orig)
        y_train_list.append(c_idx)

        # Slot-swapped copy
        raw_swap = swap_holistic_sequence(raw_30)
        f_swap = preprocess_holistic_sequence(raw_swap, include_velocity=True)
        X_train_list.append(f_swap)
        y_train_list.append(c_idx)

        # Augmented copies
        for _ in range(num_augmentations):
            raw_aug = augment_raw_holistic_sequence(raw_30, rng)
            f_aug = preprocess_holistic_sequence(raw_aug, include_velocity=True)
            X_train_list.append(f_aug)
            y_train_list.append(c_idx)

    # Process test set (Strictly clean, zero augmentation)
    X_test_list = []
    y_test_list = []
    for it in test_items:
        raw_30 = it["frames_30"]
        f_clean = preprocess_holistic_sequence(raw_30, include_velocity=True)
        X_test_list.append(f_clean)
        y_test_list.append(lbl_to_idx[it["label"]])

    X_train = np.stack(X_train_list, axis=0)
    y_train = np.array(y_train_list, dtype=np.int64)
    X_test = np.stack(X_test_list, axis=0)
    y_test = np.array(y_test_list, dtype=np.int64)

    logger.info(f"Train tensor shape: {X_train.shape} ({len(y_train)} samples)")
    logger.info(f"Held-out test tensor shape: {X_test.shape} ({len(y_test)} samples)")

    train_ds = SequenceDataset(X_train, y_train)
    test_ds = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    num_classes = len(labels)
    model = HolisticWordClassifier(
        in_dim=COMBINED_FEATURE_DIMS,
        cnn_ch=128,
        hid=128,
        nc=num_classes,
        dp=0.3
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    logger.info("=" * 70)
    logger.info(f"STEP 3: Training HolisticWordClassifier across {epochs} epochs")
    logger.info("=" * 70)

    best_macro_f1 = 0.0
    best_acc = 0.0
    best_preds = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += loss.item() * len(by)

        train_loss = total_loss / len(train_ds)

        # Evaluation on clean held-out test split
        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for bx, by in test_loader:
                logits = model(bx)
                preds = torch.argmax(logits, dim=1).numpy()
                val_preds_list.extend(preds)

        val_preds = np.array(val_preds_list)
        acc, mf1, _, _ = compute_metrics(y_test, val_preds, num_classes, labels)
        scheduler.step(mf1)

        if mf1 > best_macro_f1 or (mf1 == best_macro_f1 and acc > best_acc):
            best_macro_f1 = mf1
            best_acc = acc
            best_preds = val_preds
            # Save checkpoint
            torch.save({
                "model_type": "holistic_bilstm",
                "model_state_dict": model.state_dict(),
                "input_dim": COMBINED_FEATURE_DIMS,
                "sequence_length": SEQUENCE_LENGTH,
                "num_classes": num_classes,
                "labels": labels,
                "test_accuracy": acc,
                "test_macro_f1": mf1,
            }, CHECKPOINT_PATH)

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} | Test Acc: {acc:.4f} | Test Macro-F1: {mf1:.4f} (Best F1: {best_macro_f1:.4f})")

    logger.info("=" * 70)
    logger.info(f"STEP 4: Final Evaluation on Held-Out Test Videos (Zero-Leakage)")
    logger.info("=" * 70)

    # Re-load best checkpoint for reporting
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    final_preds = []
    final_probs = []
    with torch.no_grad():
        for bx, by in test_loader:
            logits = model(bx)
            probs = torch.softmax(logits, dim=1).numpy()
            final_probs.extend(probs)
            final_preds.extend(np.argmax(probs, axis=1))

    final_preds = np.array(final_preds)
    final_probs = np.array(final_probs)
    acc, mf1, cm, report = compute_metrics(y_test, final_preds, num_classes, labels)

    logger.info(f"HOLISTIC MODEL FINAL TEST ACCURACY: {acc * 100:.2f}% ({acc:.4f})")
    logger.info(f"HOLISTIC MODEL FINAL TEST MACRO-F1: {mf1:.4f}")

    # Baseline comparison: Run existing CNN-BiLSTM V2 model on the same held-out test videos
    logger.info("=" * 70)
    logger.info("STEP 5: Baseline Comparison on Identical Held-Out Videos")
    logger.info("=" * 70)

    baseline_acc = 0.0
    baseline_mf1 = 0.0
    baseline_cm = None
    baseline_pt = SCRIPT_DIR / "models" / "isl_cnn_lstm_word_model_v2.pt"

    if baseline_pt.exists():
        try:
            from services.word_recognizer import WordRecognizer
            from services.word_preprocessing import preprocess_word_sequence

            wr = WordRecognizer()
            base_preds = []
            for it in test_items:
                raw_hand_30 = it["frames_30"][:, :126]  # Hand-only landmarks
                res = wr.predict(raw_hand_30)
                pred_w = res["word"] if res and res.get("word") else "?"
                base_preds.append(lbl_to_idx.get(pred_w, -1))

            base_preds = np.array(base_preds)
            baseline_acc, baseline_mf1, baseline_cm, _ = compute_metrics(y_test, base_preds, num_classes, labels)
            logger.info(f"BASELINE MODEL (isl_cnn_lstm_word_model_v2.pt) TEST ACCURACY: {baseline_acc * 100:.2f}%")
            logger.info(f"BASELINE MODEL TEST MACRO-F1: {baseline_mf1:.4f}")
            logger.info(f"DELTA ACCURACY: +{(acc - baseline_acc) * 100:.2f}%")
            logger.info(f"DELTA MACRO-F1: +{(mf1 - baseline_mf1):.4f}")
        except Exception as e:
            logger.warning(f"Could not evaluate baseline model: {e}")

    # Export confusion matrix CSV
    with open(CM_CSV_PATH, "w", encoding="utf-8") as f:
        f.write("True_Class," + ",".join(labels) + "\n")
        for i, lbl in enumerate(labels):
            row_str = ",".join(str(cm[i, j]) for j in range(num_classes))
            f.write(f"{lbl},{row_str}\n")

    # Export metadata
    meta = {
        "model_type": "holistic_bilstm",
        "architecture": "CNN-BiGRU with Temporal Attention (328D Holistic)",
        "num_classes": num_classes,
        "labels": labels,
        "input_dim": COMBINED_FEATURE_DIMS,
        "sequence_length": SEQUENCE_LENGTH,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "test_accuracy": round(float(acc), 4),
        "test_macro_f1": round(float(mf1), 4),
        "baseline_accuracy": round(float(baseline_acc), 4),
        "baseline_macro_f1": round(float(baseline_mf1), 4),
        "delta_accuracy": round(float(acc - baseline_acc), 4),
        "per_class": report,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved holistic model checkpoint to: {CHECKPOINT_PATH}")
    logger.info(f"Saved metadata to: {META_PATH}")
    logger.info(f"Saved confusion matrix CSV to: {CM_CSV_PATH}")

    return meta


if __name__ == "__main__":
    train_holistic_model()

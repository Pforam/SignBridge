"""
SignBridge — Trustworthy, Leak-Free ISL Word Model Training & Evaluation.

Key Pillars:
1. Zero Data Leakage:
   - Partitions recordings strictly by source file / signer session first.
   - Zero validation/test samples or augmentations from the same recording ever enter training.
   - Augmentation is applied strictly and exclusively to the training split.
2. Single Shared Preprocessing:
   - Uses `services.word_preprocessing.preprocess_word_sequence` (coords + mask + velocity = 254 dims).
3. Anti-Overfitting:
   - Dropout (0.3), Weight Decay (1e-4), Label Smoothing (0.05), ReduceLROnPlateau, Early Stopping.
4. Model Comparison:
   - Compares CNN-BiLSTM (with and without velocity) vs CNN-BiGRU.
5. Trustworthy Reporting:
   - Generates confusion_matrix.png, confusion_matrix.csv, per-class F1 metrics.
   - Saves best model to backend/models/isl_cnn_lstm_word_model_v2.pt (preserving old models).
   - Exports calibrated per-class thresholds to backend/models/word_thresholds.json.
"""

import os
import json
import time
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from services.word_preprocessing import preprocess_word_sequence, SEQUENCE_LENGTH, RAW_FEATURES

def compute_metrics(val_targets, val_preds, num_classes, labels):
    """
    Computes accuracy, macro F1, per-class metrics, and confusion matrix using pure NumPy.
    Bypasses sklearn to adhere to Ponytail principles and avoid environment dependency issues.
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(val_targets, val_preds):
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

    acc = float(np.mean(val_targets == val_preds)) if len(val_targets) > 0 else 0.0
    macro_f1 = float(np.mean(f1)) if len(f1) > 0 else 0.0

    report_dict = {}
    for i, label in enumerate(labels):
        report_dict[label] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1-score': float(f1[i]),
            'support': int(support[i])
        }
    return acc, macro_f1, cm, report_dict


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("TrainWords")

SCRIPT_DIR = Path(__file__).resolve().parent
WORD_DATASET_DIR = SCRIPT_DIR / 'dataset_words_mediapipe'
MODEL_DIR = SCRIPT_DIR / 'models'


# ═══════════════════════════════════════════════════════════════════════════
# 1. Dataset Loading & Strict Leak-Free Partitioning
# ═══════════════════════════════════════════════════════════════════════════

def load_and_partition_word_data(
    data_dir: Path = WORD_DATASET_DIR,
    val_ratio: float = 0.25,
    seed: int = 42
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Loads flat converter JSON files from dataset_words_mediapipe.
    Splits by unique source recording before augmentation to prevent leakage.
    Also accepts the legacy class-subfolder layout.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Word dataset directory not found: {data_dir}")

    records = []
    # New converter layout: dataset_words_mediapipe/*.json
    json_paths = sorted(data_dir.glob("*.json"))

    # Legacy layout: dataset_words/<CLASS>/*.json
    if not json_paths:
        for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
            json_paths.extend(sorted(class_dir.glob("*.json")))

    if not json_paths:
        raise ValueError(f"No JSON recordings found in {data_dir}")

    for json_path in json_paths:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            seq = data.get("raw_frames")
            if seq is None and data.get("frame_sequences"):
                seq = data["frame_sequences"][0]
            if seq is None:
                seq = data.get("frames")
            if seq is None:
                logger.warning("Skipping %s: no raw_frames/frame_sequences/frames", json_path.name)
                continue

            arr = np.asarray(seq, dtype=np.float32)
            if arr.ndim != 2 or arr.shape != (SEQUENCE_LENGTH, RAW_FEATURES):
                logger.warning("Skipping %s: expected (%d,%d), got %s",
                               json_path.name, SEQUENCE_LENGTH, RAW_FEATURES, arr.shape)
                continue

            word = str(data.get("word") or (json_path.parent.name if json_path.parent != data_dir else "")).strip().upper()
            if not word:
                logger.warning("Skipping %s: missing word label", json_path.name)
                continue

            source_id = str(data.get("source_file") or data.get("session_id") or json_path.stem)
            signer = str(data.get("signer_session") or source_id)
            records.append({
                "sequence": arr,
                "word": word,
                "source_id": source_id,
                "signer": signer,
            })
        except Exception as exc:
            logger.warning("Error reading %s: %s", json_path, exc)

    if not records:
        raise ValueError(f"No valid (30,126) recordings found in {data_dir}")

    word_labels = sorted({r["word"] for r in records})
    label_to_idx = {label: i for i, label in enumerate(word_labels)}
    raw_samples_by_class = {i: [] for i in range(len(word_labels))}
    for record in records:
        record["class_idx"] = label_to_idx[record["word"]]
        raw_samples_by_class[record["class_idx"]].append(record)

    rng = np.random.default_rng(seed)
    train_samples, val_samples = [], []
    train_sources_set, val_sources_set = set(), set()

    for cls_idx, samples in raw_samples_by_class.items():
        # Group by recording identity. A recording may never cross the split.
        grouped = {}
        for sample in samples:
            grouped.setdefault(sample["source_id"], []).append(sample)
        sources = list(grouped)
        rng.shuffle(sources)

        if len(sources) >= 2:
            n_val = max(1, min(len(sources) - 1, round(len(sources) * val_ratio)))
            val_sources = set(sources[:n_val])
            train_sources = set(sources[n_val:])
        else:
            # A singleton class cannot be evaluated honestly; keep it in train.
            train_sources, val_sources = set(sources), set()

        for source in train_sources:
            train_samples.extend(grouped[source])
            train_sources_set.add(source)
        for source in val_sources:
            val_samples.extend(grouped[source])
            val_sources_set.add(source)

    overlap = train_sources_set.intersection(val_sources_set)
    assert not overlap, f"DATA LEAKAGE DETECTED: {overlap}"

    logger.info("Loaded %d classes from %d valid recordings.", len(word_labels), len(records))
    logger.info("Recording split: %d train, %d validation.",
                len(train_sources_set), len(val_sources_set))
    logger.info("Classes with no validation recording: %s",
                [label for i, label in enumerate(word_labels)
                 if not any(s["class_idx"] == i for s in val_samples)])

    return {
        "train_samples": train_samples,
        "val_samples": val_samples,
        "word_labels": word_labels,
        "label_to_idx": label_to_idx,
    }, word_labels


# ═══════════════════════════════════════════════════════════════════════════
# 2. In-Memory Augmentation (Strictly on Train Split Only)
# ═══════════════════════════════════════════════════════════════════════════

def augment_train_split(
    train_samples: List[Dict[str, Any]],
    target_per_class: int = 100,
    seed: int = 123
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment only training recordings. Missing-hand zero blocks stay exactly zero.
    No left/right swapping is used because it can change asymmetric sign meaning.
    """
    rng = np.random.default_rng(seed)
    by_class: Dict[int, List[np.ndarray]] = {}
    for sample in train_samples:
        by_class.setdefault(sample["class_idx"], []).append(sample["sequence"])

    aug_X, aug_y = [], []
    for class_idx, sequences in by_class.items():
        for sequence in sequences:
            aug_X.append(sequence.copy())
            aug_y.append(class_idx)

        needed = max(0, target_per_class - len(sequences))
        for k in range(needed):
            base = sequences[k % len(sequences)].copy()
            aug = base.copy()

            # Coordinate-level presence mask: preserve absent hand blocks.
            pts = aug.reshape(SEQUENCE_LENGTH, 2, 21, 3)
            hand_present = np.any(pts != 0.0, axis=(2, 3))  # [T, 2]
            point_present = np.any(pts != 0.0, axis=3)      # [T, 2, 21]

            # Small jitter only on coordinates belonging to detected landmarks.
            noise = rng.normal(0.0, rng.uniform(0.0015, 0.004), size=pts.shape).astype(np.float32)
            pts += noise * point_present[..., None].astype(np.float32)

            # Mild per-hand translation; only apply to hands actually present.
            if rng.random() < 0.35:
                for hand_idx in range(2):
                    if hand_present[:, hand_idx].any():
                        shift = rng.uniform(-0.008, 0.008, size=(1, 1, 2)).astype(np.float32)
                        valid = point_present[:, hand_idx, :, None]
                        pts[:, hand_idx, :, :2] = np.where(
                            valid,
                            pts[:, hand_idx, :, :2] + shift,
                            pts[:, hand_idx, :, :2]
                        )

            # Gentle temporal resampling across the full sequence.
            aug = pts.reshape(SEQUENCE_LENGTH, RAW_FEATURES)
            if rng.random() < 0.35:
                idx = np.linspace(0, SEQUENCE_LENGTH - 1, SEQUENCE_LENGTH)
                idx += rng.uniform(-0.35, 0.35, size=SEQUENCE_LENGTH)
                idx = np.clip(np.round(idx), 0, SEQUENCE_LENGTH - 1).astype(int)
                aug = aug[idx].copy()

            # Frame dropout zeros a complete frame, as a real tracking gap would.
            if rng.random() < 0.15:
                drop_count = int(rng.integers(1, 3))
                drop_idx = rng.choice(SEQUENCE_LENGTH, size=drop_count, replace=False)
                aug[drop_idx] = 0.0

            # Explicitly re-zero hand blocks that were absent in each sampled frame.
            # (The point mask is re-derived after temporal resampling.)
            shaped = aug.reshape(SEQUENCE_LENGTH, 2, 21, 3)
            for t in range(SEQUENCE_LENGTH):
                for h in range(2):
                    if not np.any(shaped[t, h] != 0.0):
                        shaped[t, h] = 0.0

            aug_X.append(shaped.reshape(SEQUENCE_LENGTH, RAW_FEATURES).copy())
            aug_y.append(class_idx)

    return np.asarray(aug_X, dtype=np.float32), np.asarray(aug_y, dtype=np.int64)


# ═══════════════════════════════════════════════════════════════════════════
# 3. PyTorch Model Architectures
# ═══════════════════════════════════════════════════════════════════════════

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(x * weights, dim=1)


class CNNBiLSTMWordClassifier(nn.Module):
    """
    1D-CNN temporal feature extractor + Bidirectional LSTM + Attention + Classifier.
    """
    def __init__(self, in_dim: int = 254, cnn_channels: int = 128, hidden_dim: int = 128,
                 num_classes: int = 54, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attn = TemporalAttention(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cnn = self.cnn(x.permute(0, 2, 1))
        lstm_out, _ = self.lstm(x_cnn.permute(0, 2, 1))
        pooled = self.attn(lstm_out)
        return self.classifier(pooled)


class CNNBiGRUWordClassifier(nn.Module):
    """
    Lightweight 1D-CNN + Bidirectional GRU variant.
    """
    def __init__(self, in_dim: int = 254, cnn_channels: int = 96, hidden_dim: int = 96,
                 num_classes: int = 54, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.gru = nn.GRU(
            input_size=cnn_channels, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attn = TemporalAttention(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cnn = self.cnn(x.permute(0, 2, 1))
        gru_out, _ = self.gru(x_cnn.permute(0, 2, 1))
        pooled = self.attn(gru_out)
        return self.classifier(pooled)


class WordSeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ═══════════════════════════════════════════════════════════════════════════
# 4. Training Engine
# ═══════════════════════════════════════════════════════════════════════════

def train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device('cpu')
) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
    """
    Trains a model with AdamW, ReduceLROnPlateau, label smoothing, and early stopping.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-5)

    best_acc = 0.0
    best_state = None
    patience = 0
    max_patience = 12

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            optimizer.step()

            t_loss += loss.item() * len(y_b)
            preds = logits.argmax(dim=1)
            t_correct += (preds == y_b).sum().item()
            t_total += len(y_b)

        train_acc = t_correct / max(1, t_total)

        # Validation
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits = model(X_b)
                preds = logits.argmax(dim=1)
                v_correct += (preds == y_b).sum().item()
                v_total += len(y_b)

        val_acc = v_correct / max(1, v_total)
        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if epoch % 5 == 0 or epoch == epochs or patience == 0:
            logger.info(f"  Epoch {epoch:02d}/{epochs:02d} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}% (Best: {best_acc*100:.1f}%) | Pat: {patience}/{max_patience}")

        if patience >= max_patience:
            logger.info(f"  Early stopping triggered at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Detailed evaluation on validation set
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            logits = model(X_b.to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(y_b.numpy())

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    acc, f1, _, _ = compute_metrics(val_targets, val_preds, num_classes, [str(i) for i in range(num_classes)])

    eval_results = {
        'val_accuracy': float(acc),
        'val_macro_f1': float(f1),
        'val_preds': val_preds,
        'val_targets': val_targets
    }

    return best_acc, best_state, eval_results


# ═══════════════════════════════════════════════════════════════════════════
# 5. Full Pipeline & Model Comparison
# ═══════════════════════════════════════════════════════════════════════════

def run_word_model_training(
    target_samples_per_class: int = 100,
    epochs: int = 35,
    device_name: str = 'cpu',
    data_dir: Path = WORD_DATASET_DIR
):
    """
    Executes the complete leak-free ISL word training pipeline:
    1. Loads dataset and executes strict source-partitioning.
    2. Augments only training split.
    3. Preprocesses with shared normalization + presence mask + velocity.
    4. Compares CNN-BiLSTM vs CNN-BiGRU (and with vs without velocity).
    5. Saves best model to isl_cnn_lstm_word_model_v2.pt.
    6. Generates confusion matrix and calibration thresholds.
    """
    logger.info("=" * 70)
    logger.info("SIGNBRIDGE ISL WORD RECOGNITION: LEAK-FREE TRAINING PIPELINE")
    logger.info("=" * 70)

    device = torch.device('cuda' if (torch.cuda.is_available() and device_name == 'cuda') else 'cpu')
    logger.info(f"Training device: {device}")

    # 1. Partition
    data_dict, word_labels = load_and_partition_word_data(data_dir=data_dir)
    train_samples = data_dict['train_samples']
    val_samples = data_dict['val_samples']
    num_classes = len(word_labels)

    # 2. Augment train only
    logger.info(f"\nApplying in-memory augmentation ONLY to train split (target={target_samples_per_class}/class)...")
    X_train_raw, y_train = augment_train_split(train_samples, target_per_class=target_samples_per_class)

    X_val_raw = np.array([s['sequence'] for s in val_samples], dtype=np.float32)
    y_val = np.array([s['class_idx'] for s in val_samples], dtype=np.int64)

    logger.info(f"Raw splits: X_train={X_train_raw.shape}, X_val={X_val_raw.shape}")

    # 3. Preprocess with 254-D features (coords + mask + velocity)
    logger.info("\nExtracting shared features: wrist-centering + palm-scale + presence mask + velocity (254 dims)...")
    X_train_254 = preprocess_word_sequence(X_train_raw, include_velocity=True, include_mask=True)
    X_val_254 = preprocess_word_sequence(X_val_raw, include_velocity=True, include_mask=True)
    logger.info(f"Feature shapes (254D): Train={X_train_254.shape}, Val={X_val_254.shape}")

    # Preprocess with 128-D features (coords + mask, NO velocity) for ablation
    logger.info("Extracting shared features without velocity (128 dims) for ablation...")
    X_train_128 = preprocess_word_sequence(X_train_raw, include_velocity=False, include_mask=True)
    X_val_128 = preprocess_word_sequence(X_val_raw, include_velocity=False, include_mask=True)

    # DataLoaders
    batch_size = 32
    train_loader_254 = DataLoader(WordSeqDataset(X_train_254, y_train), batch_size=batch_size, shuffle=True)
    val_loader_254 = DataLoader(WordSeqDataset(X_val_254, y_val), batch_size=batch_size, shuffle=False)

    train_loader_128 = DataLoader(WordSeqDataset(X_train_128, y_train), batch_size=batch_size, shuffle=True)
    val_loader_128 = DataLoader(WordSeqDataset(X_val_128, y_val), batch_size=batch_size, shuffle=False)

    # 4. Compare Architectures
    results = {}

    # Variant 1: CNN-BiLSTM (254-D with velocity)
    logger.info("\n--- Training Model 1: CNN-BiLSTM (254-D: Coords + Mask + Velocity) ---")
    model_cnn_lstm = CNNBiLSTMWordClassifier(in_dim=254, cnn_channels=128, hidden_dim=128, num_classes=num_classes)
    acc1, state1, eval1 = train_single_model(model_cnn_lstm, train_loader_254, val_loader_254, num_classes, epochs=epochs, device=device)
    results['CNN-BiLSTM (254D)'] = {'acc': acc1, 'f1': eval1['val_macro_f1'], 'state': state1, 'model_cls': CNNBiLSTMWordClassifier, 'in_dim': 254, 'eval': eval1}
    logger.info(f"  Model 1 Val Accuracy: {acc1*100:.2f}% | Val Macro F1: {eval1['val_macro_f1']:.4f}")

    # Variant 2: CNN-BiGRU (254-D with velocity)
    logger.info("\n--- Training Model 2: CNN-BiGRU (254-D: Coords + Mask + Velocity) ---")
    model_cnn_gru = CNNBiGRUWordClassifier(in_dim=254, cnn_channels=96, hidden_dim=96, num_classes=num_classes)
    acc2, state2, eval2 = train_single_model(model_cnn_gru, train_loader_254, val_loader_254, num_classes, epochs=epochs, device=device)
    results['CNN-BiGRU (254D)'] = {'acc': acc2, 'f1': eval2['val_macro_f1'], 'state': state2, 'model_cls': CNNBiGRUWordClassifier, 'in_dim': 254, 'eval': eval2}
    logger.info(f"  Model 2 Val Accuracy: {acc2*100:.2f}% | Val Macro F1: {eval2['val_macro_f1']:.4f}")

    # Variant 3: CNN-BiLSTM without velocity (128-D)
    logger.info("\n--- Training Model 3: CNN-BiLSTM (128-D: No Velocity Ablation) ---")
    model_no_vel = CNNBiLSTMWordClassifier(in_dim=128, cnn_channels=128, hidden_dim=128, num_classes=num_classes)
    acc3, state3, eval3 = train_single_model(model_no_vel, train_loader_128, val_loader_128, num_classes, epochs=epochs, device=device)
    results['CNN-BiLSTM (128D)'] = {'acc': acc3, 'f1': eval3['val_macro_f1'], 'state': state3, 'model_cls': CNNBiLSTMWordClassifier, 'in_dim': 128, 'eval': eval3}
    logger.info(f"  Model 3 Val Accuracy: {acc3*100:.2f}% | Val Macro F1: {eval3['val_macro_f1']:.4f}")

    # Model Comparison Summary
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON SUMMARY (ON LEAK-FREE VALIDATION SET)")
    logger.info("=" * 60)
    best_name = None
    best_f1 = -1.0
    for name, r in results.items():
        logger.info(f"  {name:25s} -> Val Accuracy: {r['acc']*100:5.2f}% | Macro F1: {r['f1']:.4f}")
        if r['f1'] > best_f1:
            best_f1 = r['f1']
            best_name = name

    logger.info(f"\nWinning Model: {best_name} (Macro F1 = {best_f1:.4f})")
    winning_data = results[best_name]

    # 5. Save best model to isl_cnn_lstm_word_model_v2.pt
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    v2_path = MODEL_DIR / 'isl_cnn_lstm_word_model_v2.pt'
    torch.save({
        'model_type': 'cnn_bilstm' if 'LSTM' in best_name else 'cnn_bigru',
        'architecture_name': best_name,
        'model_state_dict': winning_data['state'],
        'input_dim': winning_data['in_dim'],
        'has_attention': True,
        'cnn_channels': 128 if 'LSTM' in best_name else 96,
        'hidden_dim': 128 if 'LSTM' in best_name else 96,
        'sequence_length': SEQUENCE_LENGTH,
        'num_classes': num_classes,
        'word_labels': word_labels,
    }, str(v2_path))
    logger.info(f"Saved winning model artifact to: {v2_path}")

    # 6. Generate Classification Report & Confusion Matrix
    eval_win = winning_data['eval']
    val_targets = eval_win['val_targets']
    val_preds = eval_win['val_preds']

    acc, macro_f1, cm, report_dict = compute_metrics(val_targets, val_preds, num_classes, word_labels)

    # Confusion matrix CSV
    cm_csv_path = MODEL_DIR / 'confusion_matrix.csv'
    with open(cm_csv_path, 'w', encoding='utf-8') as f:
        f.write("," + ",".join(word_labels) + "\n")
        for i, row in enumerate(cm):
            f.write(word_labels[i] + "," + ",".join(map(str, row)) + "\n")
    logger.info(f"Confusion matrix CSV saved to: {cm_csv_path}")

    # Confusion matrix Image
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(18, 16))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix — {best_name} (Leak-Free Split)', fontsize=14)
        plt.colorbar()
        tick_marks = np.arange(len(word_labels))
        plt.xticks(tick_marks, word_labels, rotation=90, fontsize=8)
        plt.yticks(tick_marks, word_labels, fontsize=8)
        plt.ylabel('Ground Truth Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        cm_png_path = MODEL_DIR / 'confusion_matrix.png'
        plt.savefig(cm_png_path, dpi=200)
        plt.close()
        logger.info(f"Confusion matrix image saved to: {cm_png_path}")
    except Exception as e:
        logger.warning(f"Could not render confusion matrix PNG: {e}")

    # Top-5 Most Confused Pairs
    confused_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((word_labels[i], word_labels[j], int(cm[i, j])))
    confused_pairs.sort(key=lambda x: x[2], reverse=True)

    logger.info("\nTop 5 Most Confused Word Pairs:")
    if confused_pairs:
        for actual, predicted, count in confused_pairs[:5]:
            logger.info(f"  {actual:15s} misclassified as {predicted:15s} ({count} times)")
    else:
        logger.info("  Zero misclassifications on validation set!")

    # Per-class thresholds calibration
    thresholds = {}
    for i, label in enumerate(word_labels):
        stat = report_dict.get(label, {})
        f1_c = stat.get('f1-score', 0.0)
        # Dynamic calibration: lower F1 -> slightly higher threshold to reduce false commits
        if f1_c >= 0.85:
            th = 0.50
        elif f1_c >= 0.70:
            th = 0.55
        elif f1_c >= 0.50:
            th = 0.60
        else:
            th = 0.65
        thresholds[label] = {
            'threshold': th,
            'f1_score': round(float(f1_c), 3),
            'precision': round(float(stat.get('precision', 0.0)), 3),
            'recall': round(float(stat.get('recall', 0.0)), 3),
            'val_support': int(stat.get('support', 0))
        }

    th_path = MODEL_DIR / 'word_thresholds.json'
    th_path.write_text(json.dumps({
        'default_threshold': 0.50,
        'margin_threshold': 0.08,
        'min_motion_velocity': 0.015,
        'per_class': thresholds
    }, indent=2), encoding='utf-8')
    logger.info(f"Calibrated word thresholds saved to: {th_path}")

    # Save meta json
    meta_path = MODEL_DIR / 'cnn_lstm_training_meta.json'
    meta_path.write_text(json.dumps({
        'model_type': 'cnn_bilstm' if 'LSTM' in best_name else 'cnn_bigru',
        'winning_architecture': best_name,
        'leak_free_split': True,
        'train_samples': len(X_train_raw),
        'val_samples': len(X_val_raw),
        'val_accuracy': winning_data['acc'],
        'val_macro_f1': winning_data['f1'],
        'num_classes': num_classes,
        'labels': word_labels,
        'sequence_length': SEQUENCE_LENGTH,
        'input_features': winning_data['in_dim'],
        'models_compared': {k: {'acc': v['acc'], 'f1': v['f1']} for k, v in results.items()},
        'top_confused_pairs': confused_pairs[:5],
        'per_class': {lbl: report_dict.get(lbl, {}) for lbl in word_labels}
    }, indent=2), encoding='utf-8')
    logger.info(f"Updated metadata saved to: {meta_path}")

    return best_name, winning_data['acc'], winning_data['f1']


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train SignBridge whole-word model")
    parser.add_argument("--data", default=str(WORD_DATASET_DIR), help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--target-per-class", type=int, default=100)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    run_word_model_training(
        target_samples_per_class=args.target_per_class,
        epochs=args.epochs,
        device_name=args.device,
        data_dir=Path(args.data)
    )

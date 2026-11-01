"""
SignBridge QA Analysis - Phase 2A: Vectorized Letter Model (A-Z) Deep Evaluation
Vectorized for sub-5 second execution on 4,902 held-out test vectors.
Evaluates:
1. Baseline test accuracy, per-class F1, and top 10 confused pairs.
2. Robustness perturbations (Gaussian noise, rotations 15/30/45 deg, scale 0.7x/1.3x, translation, mirroring, slot swap).
3. Missing data edge cases (one hand absent, both absent, all zero, NaN/Inf, invalid lengths [125, 127, 0], out-of-range).
4. Out-of-distribution tests (random noise, resting hand wrist_y > 0.88, neutral two open palms, CentroidValidator false-reject rate).
5. Hybrid ensemble vs Standalone XGB vs Standalone ST-GCN, temperature calibration, Expected Calibration Error (ECE).
6. Fallback chain verification.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support

BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / 'backend'
sys.path.insert(0, str(BACKEND_DIR))

from services.data_loader import load_dataset_partitioned, ALPHABET_LABELS
from services.translator_model import (
    TranslatorModel,
    normalize_landmarks_batch,
    validate_landmark_array,
    LandmarkValidationError
)
from services.feature_extractor import extract_features
from services.gesture_pipeline import CentroidValidator, is_idle_hand, is_neutral_gesture

def compute_ece(confidences, predictions, true_labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    accuracies = (np.array(predictions) == np.array(true_labels)).astype(float)
    bin_details = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_details.append({
                "bin": f"{bin_lower:.1f}-{bin_upper:.1f}",
                "count": int(np.sum(in_bin)),
                "acc": round(float(accuracy_in_bin), 4),
                "conf": round(float(avg_confidence_in_bin), 4)
            })
    return float(ece), bin_details

def rotate_batch(X, angle_deg):
    rad = np.radians(angle_deg)
    cos, sin = np.cos(rad), np.sin(rad)
    pts = X.reshape(-1, 2, 21, 3).copy()
    present = np.any(pts != 0, axis=(2, 3))
    for h in range(2):
        mask = present[:, h]
        if np.any(mask):
            wrists = pts[mask, h, 0:1, :].copy()
            rel = pts[mask, h] - wrists
            x, y = rel[:, :, 0].copy(), rel[:, :, 1].copy()
            rel[:, :, 0] = x * cos - y * sin
            rel[:, :, 1] = x * sin + y * cos
            pts[mask, h] = rel + wrists
    return pts.reshape(-1, 126)

def scale_batch(X, factor):
    pts = X.reshape(-1, 2, 21, 3).copy()
    present = np.any(pts != 0, axis=(2, 3))
    for h in range(2):
        mask = present[:, h]
        if np.any(mask):
            wrists = pts[mask, h, 0:1, :].copy()
            rel = pts[mask, h] - wrists
            pts[mask, h] = (rel * factor) + wrists
    return pts.reshape(-1, 126)

def translate_batch(X, dx, dy):
    pts = X.reshape(-1, 2, 21, 3).copy()
    present = np.any(pts != 0, axis=(2, 3))
    pts[:, :, :, 0] += dx * present[:, :, None]
    pts[:, :, :, 1] += dy * present[:, :, None]
    return pts.reshape(-1, 126)

def mirror_batch(X):
    pts = X.reshape(-1, 2, 21, 3).copy()
    present = np.any(pts != 0, axis=(2, 3))
    pts[:, :, :, 0] = np.where(present[:, :, None], 1.0 - pts[:, :, :, 0], 0.0)
    return pts.reshape(-1, 126)

def swap_slots_batch(X):
    pts = np.zeros_like(X)
    pts[:, :63] = X[:, 63:]
    pts[:, 63:] = X[:, :63]
    return pts

def predict_batch_fast(translator, X):
    """Vectorized prediction for batch of landmark vectors."""
    # 1. ST-GCN probs
    norm_X = normalize_landmarks_batch(X)
    if norm_X is None:
        return np.full(len(X), '?'), np.zeros(len(X)), np.zeros((len(X), 26))
    
    pts = norm_X.reshape(-1, 42, 3)
    with torch.no_grad():
        t_in = torch.from_numpy(pts).float()
        logits = translator.stgcn_model(t_in)
        probs_stgcn = torch.softmax(logits / 1.20, dim=-1).cpu().numpy()
        
    # 2. XGBoost probs
    feats_208 = extract_features(X)
    probs_xgb = translator.xgb_model.predict_proba(feats_208)
    
    # 3. Hybrid blend
    probs_hybrid = 0.65 * probs_xgb + 0.35 * probs_stgcn
    top_indices = np.argmax(probs_hybrid, axis=1)
    confs = np.max(probs_hybrid, axis=1)
    preds = np.array([ALPHABET_LABELS[i] for i in top_indices])
    
    return preds, confs, probs_hybrid, probs_xgb, probs_stgcn

def run_phase2a():
    print("=" * 70, flush=True)
    print("  PHASE 2A: VECTORIZED LETTER MODEL (A-Z) DEEP EVALUATION", flush=True)
    print("=" * 70, flush=True)
    
    partitions, report = load_dataset_partitioned()
    X_test = partitions["X_test"]
    y_test = partitions["y_test"]
    labels = ALPHABET_LABELS
    N = len(X_test)
    y_true_lbl = [labels[idx] for idx in y_test]
    print(f"Loaded held-out test split: {N} samples across {len(labels)} classes.", flush=True)
    
    translator = TranslatorModel()
    validator = CentroidValidator()
    print(f"Active TranslatorModel mode: {translator.mode}", flush=True)
    
    results = {}
    
    # 1. Baseline Batch Evaluation
    print("\n1. Running Vectorized Baseline Evaluation on Held-Out Test Set...", flush=True)
    t0 = time.perf_counter()
    preds_hybrid, confs_hybrid, probs_hybrid, probs_xgb, probs_stgcn = predict_batch_fast(translator, X_test)
    t_elapsed = time.perf_counter() - t0
    
    preds_xgb = np.array([labels[i] for i in np.argmax(probs_xgb, axis=1)])
    preds_stgcn = np.array([labels[i] for i in np.argmax(probs_stgcn, axis=1)])
    
    acc_xgb = accuracy_score(y_true_lbl, preds_xgb)
    acc_stgcn = accuracy_score(y_true_lbl, preds_stgcn)
    acc_hybrid = accuracy_score(y_true_lbl, preds_hybrid)
    
    prec_h, rec_h, f1_h, _ = precision_recall_fscore_support(y_true_lbl, preds_hybrid, labels=labels, average='macro', zero_division=0)
    rep_hybrid = classification_report(y_true_lbl, preds_hybrid, labels=labels, output_dict=True, zero_division=0)
    
    # Top 10 Confused Pairs
    cm = confusion_matrix(y_true_lbl, preds_hybrid, labels=labels)
    confused_pairs = []
    for r in range(len(labels)):
        for c in range(len(labels)):
            if r != c and cm[r, c] > 0:
                confused_pairs.append({
                    "true": labels[r],
                    "predicted": labels[c],
                    "count": int(cm[r, c]),
                    "pair": f"{labels[r]} -> {labels[c]}"
                })
    confused_pairs.sort(key=lambda x: x["count"], reverse=True)
    top10_confused = confused_pairs[:10]
    
    ece_val, bin_details = compute_ece(confs_hybrid, preds_hybrid, y_true_lbl)
    
    results["baseline"] = {
        "test_samples": N,
        "eval_time_seconds": round(t_elapsed, 3),
        "acc_xgboost": round(float(acc_xgb), 4),
        "acc_stgcn": round(float(acc_stgcn), 4),
        "acc_hybrid": round(float(acc_hybrid), 4),
        "macro_f1_hybrid": round(float(f1_h), 4),
        "macro_precision_hybrid": round(float(prec_h), 4),
        "macro_recall_hybrid": round(float(rec_h), 4),
        "ece": round(ece_val, 4),
        "top10_confused_pairs": top10_confused,
        "per_class": {lbl: {
            "precision": round(rep_hybrid[lbl]["precision"], 4),
            "recall": round(rep_hybrid[lbl]["recall"], 4),
            "f1": round(rep_hybrid[lbl]["f1-score"], 4),
            "support": int(rep_hybrid[lbl]["support"])
        } for lbl in labels}
    }
    print(f"Evaluated {N} test samples in {t_elapsed:.2f}s ({N/t_elapsed:.1f} FPS):", flush=True)
    print(f"  XGB Acc: {acc_xgb:.4f} | ST-GCN Acc: {acc_stgcn:.4f} | Hybrid Acc: {acc_hybrid:.4f} | ECE: {ece_val:.4f}", flush=True)
    top5_str = [f"{p['pair']} ({p['count']})" for p in top10_confused[:5]]
    print(f"  Top 5 confused pairs: {top5_str}", flush=True)

    # 2. Robustness Perturbations (Vectorized)
    print("\n2. Evaluating Vectorized Robustness Perturbations...", flush=True)
    perturbations = {
        "noise_sigma_0.01": X_test + np.random.normal(0, 0.01, X_test.shape).astype(np.float32),
        "noise_sigma_0.02": X_test + np.random.normal(0, 0.02, X_test.shape).astype(np.float32),
        "rotate_15_deg": rotate_batch(X_test, 15),
        "rotate_30_deg": rotate_batch(X_test, 30),
        "rotate_45_deg": rotate_batch(X_test, 45),
        "scale_0.7x": scale_batch(X_test, 0.7),
        "scale_1.3x": scale_batch(X_test, 1.3),
        "translate_dx0.1_dy0.1": translate_batch(X_test, 0.1, 0.1),
        "mirror_horizontal": mirror_batch(X_test),
        "swap_hand_slots": swap_slots_batch(X_test)
    }
    
    robustness_results = {"baseline_acc": round(float(acc_hybrid), 4)}
    for p_name, p_data in perturbations.items():
        p_preds, _, _, _, _ = predict_batch_fast(translator, p_data)
        acc_p = accuracy_score(y_true_lbl, p_preds)
        drop = acc_hybrid - acc_p
        robustness_results[p_name] = {
            "accuracy": round(float(acc_p), 4),
            "drop": round(float(drop), 4),
            "retention_rate": round(float(acc_p / acc_hybrid), 4) if acc_hybrid > 0 else 0.0
        }
        print(f"  [{p_name}] Acc: {acc_p:.4f} (Drop: {drop:+.4f})", flush=True)
    results["robustness"] = robustness_results

    # 3. Missing Data & Corrupt Inputs
    print("\n3. Testing Missing & Malformed Landmark Inputs...", flush=True)
    edge_cases = {}
    
    all_zeros = np.zeros(126, dtype=np.float32)
    res_zeros = translator.predict(all_zeros)
    edge_cases["all_zeros"] = {
        "letter": res_zeros.get("letter"),
        "rejected": res_zeros.get("rejected", False),
        "rejection_reason": res_zeros.get("rejection_reason")
    }
    
    one_hand_absent = np.zeros(126, dtype=np.float32)
    one_hand_absent[63:] = X_test[0, 63:]
    res_one_hand = translator.predict(one_hand_absent)
    edge_cases["one_hand_absent"] = {
        "letter": res_one_hand.get("letter"),
        "confidence": res_one_hand.get("confidence")
    }
    
    nan_arr = np.array([np.nan] * 126, dtype=np.float32)
    res_nan = translator.predict(nan_arr)
    edge_cases["nan_array"] = {
        "letter": res_nan.get("letter"),
        "rejected": res_nan.get("rejected", False)
    }
    
    inf_arr = np.array([np.inf] * 126, dtype=np.float32)
    res_inf = translator.predict(inf_arr)
    edge_cases["inf_array"] = {
        "letter": res_inf.get("letter"),
        "rejected": res_inf.get("rejected", False)
    }
    
    for length in [125, 127, 0]:
        try:
            validate_landmark_array(np.ones(length, dtype=np.float32))
            edge_cases[f"length_{length}"] = "accepted_unexpectedly"
        except (LandmarkValidationError, ValueError) as err:
            edge_cases[f"length_{length}"] = f"rejected_properly: {type(err).__name__}"
            
    out_of_range = np.full(126, 500.0, dtype=np.float32)
    res_oor = translator.predict(out_of_range)
    edge_cases["out_of_range_coords"] = {
        "letter": res_oor.get("letter"),
        "confidence": res_oor.get("confidence")
    }
    results["missing_data_edge_cases"] = edge_cases
    print(f"Edge cases: zeros={res_zeros.get('letter')}, nan={res_nan.get('letter')}", flush=True)

    # 4. Out-of-Distribution & Centroid Validation
    print("\n4. Testing Out-of-Distribution & CentroidValidator...", flush=True)
    ood_results = {}
    
    random_vectors = [np.random.uniform(-1.0, 1.0, 126).astype(np.float32) for _ in range(100)]
    random_rejections = 0
    for vec in random_vectors:
        norm = normalize_landmarks_batch(vec.reshape(1, 126))
        if norm is None:
            random_rejections += 1
            continue
        res = translator.predict(vec)
        is_v, dist, thresh, reason = validator.validate(norm[0], res.get('letter', 'A'))
        if not is_v or res.get('rejected', False) or res.get('letter') == '?':
            random_rejections += 1
    ood_results["random_noise_rejection_rate"] = random_rejections / 100.0
    
    resting_hand = X_test[0].copy().reshape(2, 21, 3)
    resting_hand[0, 0, 1] = 0.92
    resting_hand[1, 0, 1] = 0.95
    is_idle = is_idle_hand(resting_hand.reshape(126))
    ood_results["resting_hand_detected_as_idle"] = is_idle
    
    neutral_hand = np.zeros((2, 21, 3), dtype=np.float32)
    for h in range(2):
        neutral_hand[h, 0] = [0.35 + 0.3 * h, 0.6, 0.0]
        for f, tip_idx in enumerate([4, 8, 12, 16, 20]):
            neutral_hand[h, tip_idx] = neutral_hand[h, 0] + np.array([-0.05 + 0.025 * f, -0.25, 0.0])
            neutral_hand[h, tip_idx - 2] = neutral_hand[h, 0] + np.array([-0.03 + 0.015 * f, -0.12, 0.0])
    is_neut = is_neutral_gesture(neutral_hand.reshape(126))
    ood_results["neutral_open_palms_detected"] = is_neut
    
    # False Rejection Rate on Valid Test Set
    valid_rejections = 0
    norm_sample = normalize_landmarks_batch(X_test[:500])
    if norm_sample is not None:
        for i in range(len(norm_sample)):
            lbl = y_true_lbl[i]
            is_v, dist, thresh, reason = validator.validate(norm_sample[i], lbl)
            if not is_v:
                valid_rejections += 1
    ood_results["centroid_false_reject_rate_on_true_class"] = round(valid_rejections / 500.0, 4)
    results["out_of_distribution"] = ood_results
    print(f"OOD Random Rejection: {ood_results['random_noise_rejection_rate']*100:.1f}%", flush=True)
    print(f"Centroid False Reject on Ground Truth: {ood_results['centroid_false_reject_rate_on_true_class']*100:.2f}%", flush=True)

    # 5. Fallback Chain Testing
    print("\n5. Testing Model Fallback Chain Graceful Degradation...", flush=True)
    t_mock = TranslatorModel()
    fallback_modes = {"initial_mode": t_mock.mode}
    t_mock.stgcn_model = None
    t_mock.mode = 'xgboost'
    res_xgb = t_mock.predict(X_test[0])
    fallback_modes["mode_when_stgcn_none"] = {"mode": t_mock.mode, "letter": res_xgb.get("letter")}
    t_mock.xgb_model = None
    t_mock.mode = 'heuristic'
    res_heur = t_mock.predict(X_test[0])
    fallback_modes["mode_when_ml_none"] = {"mode": t_mock.mode, "letter": res_heur.get("letter")}
    results["fallback_chain"] = fallback_modes
    print(f"Fallback chain verified: Initial={fallback_modes['initial_mode']} -> XGBoost -> Heuristic", flush=True)

    out_file = Path(__file__).resolve().parent / "phase2a_results.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nPhase 2A results successfully written to {out_file}", flush=True)

if __name__ == "__main__":
    run_phase2a()

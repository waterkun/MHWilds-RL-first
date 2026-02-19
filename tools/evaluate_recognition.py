"""
Evaluate CRNN recognition quality with F1, ROC/AUC, and confusion matrix.

Loads labeled crops from data/reviewed/ (filename format: {idx}_{label}.png),
runs CRNN recognition, and sweeps confidence thresholds to compute
precision/recall/F1 curves and ROC/AUC.

Usage:
    python tools/evaluate_recognition.py --data-dir data/reviewed
    python tools/evaluate_recognition.py --data-dir data/reviewed --save-plots
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_labeled_samples(data_dir):
    """Load samples with valid digit labels from filename pattern {idx}_{label}.png."""
    samples = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.png'):
            continue
        parts = fname.rsplit('.', 1)[0].split('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            path = os.path.join(data_dir, fname)
            samples.append((path, parts[1]))
    return samples


def evaluate(samples, model_path, batch_size=64):
    """Run CRNN on all samples and return predictions with confidence info."""
    from models.crnn.recognizer import CRNNRecognizer
    recognizer = CRNNRecognizer(model_path)

    results = []  # list of (ground_truth, predicted_text, mean_conf)
    t0 = time.perf_counter()

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        images = []
        labels = []
        for path, label in batch:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            images.append(img)
            labels.append(label)

        if not images:
            continue

        preds = recognizer.recognize_batch_detailed(images)
        for label, (text, info) in zip(labels, preds):
            results.append((label, text, info['mean_conf']))

    elapsed = time.perf_counter() - t0
    return results, elapsed


def compute_threshold_metrics(results, thresholds):
    """
    At each confidence threshold, compute precision/recall/F1.

    A prediction is "accepted" if mean_conf >= threshold.
    - TP: accepted AND correct
    - FP: accepted AND wrong
    - FN: rejected (but ground truth exists, so we missed a correct prediction)
    """
    metrics = []
    for t in thresholds:
        tp = fp = fn = 0
        for gt, pred, conf in results:
            if conf >= t:
                if pred == gt:
                    tp += 1
                else:
                    fp += 1
            else:
                # Rejected - counts as FN (missed a real number)
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics.append({
            'threshold': round(t, 4),
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })
    return metrics


def compute_roc(results, thresholds):
    """
    Compute ROC curve (TPR vs FPR) treating correct predictions as positives.

    For each threshold:
    - True Positive: correct prediction accepted (conf >= t)
    - False Negative: correct prediction rejected (conf < t)
    - False Positive: wrong prediction accepted (conf >= t)
    - True Negative: wrong prediction rejected (conf < t)
    """
    # Separate correct and wrong predictions
    correct = [(conf, True) for gt, pred, conf in results if pred == gt]
    wrong = [(conf, False) for gt, pred, conf in results if pred != gt]

    n_pos = len(correct)  # total correct predictions (at threshold=0)
    n_neg = len(wrong)    # total wrong predictions (at threshold=0)

    roc_points = []
    for t in thresholds:
        tp = sum(1 for conf, _ in correct if conf >= t)
        fp = sum(1 for conf, _ in wrong if conf >= t)

        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        roc_points.append((fpr, tpr))

    # Compute AUC using trapezoidal rule (points are from high threshold to low)
    # Sort by FPR ascending for proper integration
    roc_sorted = sorted(roc_points, key=lambda p: p[0])
    auc = 0.0
    for i in range(1, len(roc_sorted)):
        dx = roc_sorted[i][0] - roc_sorted[i - 1][0]
        avg_y = (roc_sorted[i][1] + roc_sorted[i - 1][1]) / 2
        auc += dx * avg_y

    return roc_points, auc


def compute_confusion_matrix(results):
    """Per-digit confusion matrix (character-level)."""
    digits = list('0123456789')
    matrix = np.zeros((10, 10), dtype=int)  # [true_digit][pred_digit]

    for gt, pred, _ in results:
        for i in range(max(len(gt), len(pred))):
            gt_ch = gt[i] if i < len(gt) else None
            pr_ch = pred[i] if i < len(pred) else None
            if gt_ch is not None and pr_ch is not None:
                if gt_ch.isdigit() and pr_ch.isdigit():
                    matrix[int(gt_ch)][int(pr_ch)] += 1

    return matrix, digits


def compute_char_accuracy(results):
    """Character-level accuracy across all samples."""
    correct = 0
    total = 0
    for gt, pred, _ in results:
        for i in range(max(len(gt), len(pred))):
            if i < len(gt) and i < len(pred) and gt[i] == pred[i]:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


def print_summary(results, threshold_metrics, auc, elapsed):
    """Print a clean summary table."""
    total = len(results)
    exact_match = sum(1 for gt, pred, _ in results if gt == pred)
    char_acc = compute_char_accuracy(results)

    # Find best F1
    best = max(threshold_metrics, key=lambda m: m['f1'])

    print("\n" + "=" * 60)
    print("  CRNN Recognition Evaluation")
    print("=" * 60)
    print(f"  Samples:             {total}")
    print(f"  Exact-match acc:     {exact_match}/{total} = {exact_match / total:.4f}")
    print(f"  Character-level acc: {char_acc:.4f}")
    print(f"  Inference time:      {elapsed:.2f}s ({elapsed / total * 1000:.1f} ms/sample)")
    print("-" * 60)
    print(f"  Best F1:             {best['f1']:.4f}  (threshold={best['threshold']:.2f})")
    print(f"    Precision:         {best['precision']:.4f}")
    print(f"    Recall:            {best['recall']:.4f}")
    print(f"    TP={best['tp']}  FP={best['fp']}  FN={best['fn']}")
    print(f"  ROC AUC:             {auc:.4f}")
    print("=" * 60)


def print_confusion_matrix(matrix, digits):
    """Print per-digit confusion matrix."""
    print("\n  Confusion Matrix (rows=true, cols=pred):")
    print("      " + "  ".join(f"{d:>4}" for d in digits))
    for i, row in enumerate(matrix):
        if row.sum() > 0:
            print(f"  {digits[i]}:  " + "  ".join(f"{v:>4}" for v in row))
    print()


def save_plots(threshold_metrics, roc_points, auc, output_dir):
    """Save F1 curve and ROC curve plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [Warning] matplotlib not installed, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    # F1 / Precision / Recall vs threshold
    thresholds = [m['threshold'] for m in threshold_metrics]
    precisions = [m['precision'] for m in threshold_metrics]
    recalls = [m['recall'] for m in threshold_metrics]
    f1s = [m['f1'] for m in threshold_metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions, label='Precision', linewidth=1.5)
    ax.plot(thresholds, recalls, label='Recall', linewidth=1.5)
    ax.plot(thresholds, f1s, label='F1', linewidth=2)
    best = max(threshold_metrics, key=lambda m: m['f1'])
    ax.axvline(best['threshold'], color='gray', linestyle='--', alpha=0.5,
               label=f"Best F1={best['f1']:.3f} @ t={best['threshold']:.2f}")
    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('Score')
    ax.set_title('CRNN Recognition: Precision / Recall / F1')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'recognition_f1_curve.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/recognition_f1_curve.png")

    # ROC curve
    fprs = [p[0] for p in roc_points]
    tprs = [p[1] for p in roc_points]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fprs, tprs, linewidth=2, label=f'ROC (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('CRNN Recognition: ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'recognition_roc_curve.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/recognition_roc_curve.png")


def save_metrics_json(results, threshold_metrics, auc, confusion_matrix, output_dir):
    """Save metrics to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    total = len(results)
    exact_match = sum(1 for gt, pred, _ in results if gt == pred)
    best = max(threshold_metrics, key=lambda m: m['f1'])

    data = {
        'total_samples': total,
        'exact_match_accuracy': exact_match / total if total > 0 else 0.0,
        'char_accuracy': compute_char_accuracy(results),
        'best_f1': best['f1'],
        'best_f1_threshold': best['threshold'],
        'best_precision': best['precision'],
        'best_recall': best['recall'],
        'auc': auc,
        'confusion_matrix': confusion_matrix.tolist(),
    }

    path = os.path.join(output_dir, 'recognition_metrics.json')
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CRNN recognition (F1/ROC/AUC)")
    parser.add_argument("--data-dir", type=str, default="data/reviewed",
                        help="Directory with labeled samples ({idx}_{label}.png)")
    parser.add_argument("--model", type=str, default="models/crnn/damage_crnn_best.pth",
                        help="CRNN model checkpoint")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save F1 and ROC plots to logs/")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, args.data_dir)
    model_path = os.path.join(project_root, args.model)
    logs_dir = os.path.join(project_root, "logs")

    # Load samples
    samples = load_labeled_samples(data_dir)
    print(f"Loaded {len(samples)} labeled samples from {data_dir}")
    if not samples:
        print("No samples found!")
        return

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    # Run CRNN inference
    print("Running CRNN inference...")
    results, elapsed = evaluate(samples, model_path, args.batch_size)
    print(f"  {len(results)} predictions in {elapsed:.2f}s")

    # Sweep thresholds
    thresholds = np.arange(0.0, 1.001, 0.01).tolist()
    threshold_metrics = compute_threshold_metrics(results, thresholds)

    # ROC / AUC
    roc_points, auc = compute_roc(results, thresholds)

    # Confusion matrix
    matrix, digits = compute_confusion_matrix(results)

    # Print summary
    print_summary(results, threshold_metrics, auc, elapsed)
    print_confusion_matrix(matrix, digits)

    # Save outputs
    save_metrics_json(results, threshold_metrics, auc, matrix, logs_dir)

    if args.save_plots:
        save_plots(threshold_metrics, roc_points, auc, logs_dir)


if __name__ == "__main__":
    main()

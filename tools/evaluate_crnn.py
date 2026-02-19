"""
Evaluate CRNN vs Tesseract on a labeled dataset.

Compares:
- Sequence accuracy (exact match)
- Character-level accuracy
- Speed (ms per crop)

Usage:
    python tools/evaluate_crnn.py --data-dir data/raw_captures --model models/crnn/damage_crnn_best.pth
"""

import os
import sys
import time
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_labeled_samples(data_dir):
    """Load samples with valid digit labels."""
    samples = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.png'):
            continue
        parts = fname.rsplit('.', 1)[0].split('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            path = os.path.join(data_dir, fname)
            samples.append((path, parts[1]))
    return samples


def evaluate_crnn(samples, model_path):
    """Evaluate CRNN recognizer."""
    from models.crnn.recognizer import CRNNRecognizer
    recognizer = CRNNRecognizer(model_path)

    correct = 0
    char_correct = 0
    char_total = 0
    total = len(samples)
    times = []

    for path, label in samples:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        t0 = time.perf_counter()
        value, confidence = recognizer.recognize_single(img)
        elapsed = (time.perf_counter() - t0) * 1000  # ms
        times.append(elapsed)

        pred = str(value) if value is not None else ""

        if pred == label:
            correct += 1

        # Character-level accuracy
        for i in range(max(len(pred), len(label))):
            if i < len(pred) and i < len(label) and pred[i] == label[i]:
                char_correct += 1
            char_total += 1

    times.sort()
    return {
        'name': 'CRNN',
        'seq_accuracy': correct / total if total > 0 else 0.0,
        'char_accuracy': char_correct / char_total if char_total > 0 else 0.0,
        'correct': correct,
        'total': total,
        'p50_ms': times[len(times) // 2] if times else 0,
        'p95_ms': times[int(len(times) * 0.95)] if times else 0,
        'p99_ms': times[int(len(times) * 0.99)] if times else 0,
        'mean_ms': np.mean(times) if times else 0,
    }


def evaluate_tesseract(samples):
    """Evaluate Tesseract OCR."""
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = r'E:\Program Files\Tesseract-OCR\tesseract.exe'
    except ImportError:
        print("Tesseract not available, skipping")
        return None

    correct = 0
    char_correct = 0
    char_total = 0
    total = len(samples)
    times = []

    config = r'--psm 7 -c tessedit_char_whitelist=0123456789'

    for path, label in samples:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Preprocess like vision.py does
        scale = 3
        enlarged = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.bitwise_not(thresh)

        t0 = time.perf_counter()
        try:
            text = pytesseract.image_to_string(thresh, config=config)
            pred = ''.join(filter(str.isdigit, text))
        except Exception:
            pred = ""
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

        if pred == label:
            correct += 1

        for i in range(max(len(pred), len(label))):
            if i < len(pred) and i < len(label) and pred[i] == label[i]:
                char_correct += 1
            char_total += 1

    times.sort()
    return {
        'name': 'Tesseract',
        'seq_accuracy': correct / total if total > 0 else 0.0,
        'char_accuracy': char_correct / char_total if char_total > 0 else 0.0,
        'correct': correct,
        'total': total,
        'p50_ms': times[len(times) // 2] if times else 0,
        'p95_ms': times[int(len(times) * 0.95)] if times else 0,
        'p99_ms': times[int(len(times) * 0.99)] if times else 0,
        'mean_ms': np.mean(times) if times else 0,
    }


def print_results(results):
    """Pretty-print comparison table."""
    print("\n" + "=" * 70)
    print(f"{'Metric':<25} ", end="")
    for r in results:
        print(f"| {r['name']:>15} ", end="")
    print()
    print("-" * 70)

    metrics = [
        ('Seq Accuracy', 'seq_accuracy', '.4f'),
        ('Char Accuracy', 'char_accuracy', '.4f'),
        ('Correct / Total', None, None),
        ('Mean (ms)', 'mean_ms', '.2f'),
        ('P50 (ms)', 'p50_ms', '.2f'),
        ('P95 (ms)', 'p95_ms', '.2f'),
        ('P99 (ms)', 'p99_ms', '.2f'),
    ]

    for label, key, fmt in metrics:
        print(f"  {label:<23} ", end="")
        for r in results:
            if key is None:
                print(f"| {r['correct']:>6} / {r['total']:<6} ", end="")
            else:
                print(f"| {r[key]:>15{fmt}} ", end="")
        print()

    print("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate CRNN vs Tesseract")
    parser.add_argument("--data-dir", type=str, default="data/raw_captures",
                        help="Directory with labeled samples ({idx}_{label}.png)")
    parser.add_argument("--model", type=str, default="models/crnn/damage_crnn_best.pth",
                        help="CRNN model checkpoint")
    parser.add_argument("--syn-val", action="store_true",
                        help="Use synthetic validation set instead")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.syn_val:
        data_dir = os.path.join(project_root, "data/synthetic/val")
    else:
        data_dir = os.path.join(project_root, args.data_dir)

    model_path = os.path.join(project_root, args.model)

    samples = load_labeled_samples(data_dir)
    print(f"Loaded {len(samples)} labeled samples from {data_dir}")

    if not samples:
        print("No samples found!")
        return

    results = []

    # CRNN
    if os.path.exists(model_path):
        print("\nEvaluating CRNN...")
        crnn_results = evaluate_crnn(samples, model_path)
        results.append(crnn_results)
    else:
        print(f"CRNN model not found: {model_path}")

    # Tesseract
    print("\nEvaluating Tesseract...")
    tess_results = evaluate_tesseract(samples)
    if tess_results:
        results.append(tess_results)

    if results:
        print_results(results)


if __name__ == "__main__":
    main()

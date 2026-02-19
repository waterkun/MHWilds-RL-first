"""
Filter raw captures using trained CRNN model.

Uses the Phase A trained CRNN to recognize damage numbers from raw captures.
Only keeps images where CRNN detects digits with sufficient confidence.

Usage:
    python tools/filter_data.py
    python tools/filter_data.py --batch-size 10000
    python tools/filter_data.py --min-conf 0.8  (stricter)
"""

import os
import sys
import shutil
import time
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from utils.vision_utils import check_baseline_consistency


def is_digit_like(img_gray):
    """
    Structural heuristic checks to reject non-digit images.
    Returns (ok, reason) where ok=True means image looks like it could contain digits.
    """
    h, w = img_gray.shape

    # 1) Aspect ratio: damage numbers are wider than tall, but not extreme
    aspect = w / max(h, 1)
    if aspect < 0.3 or aspect > 8.0:
        return False, "bad_aspect"

    # 2) Image variance: blank/uniform images have very low variance
    variance = np.var(img_gray.astype(np.float32))
    if variance < 200:
        return False, "low_variance"

    # 3) Bright pixel ratio: digits are bright on dark, or dark on bright
    #    Too few or too many bright pixels = probably not digits
    bright_ratio = np.mean(img_gray > 128)
    if bright_ratio < 0.05 or bright_ratio > 0.95:
        return False, "bad_brightness"

    # 4) Edge density: real digits have edges; noise/blur doesn't
    edges = cv2.Canny(img_gray, 50, 150)
    edge_ratio = np.mean(edges > 0)
    if edge_ratio < 0.03:
        return False, "no_edges"
    if edge_ratio > 0.4:
        return False, "too_noisy"

    # 5) Contour check: digits form a reasonable number of connected components
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return False, "no_contours"
    if len(contours) > 15:
        return False, "too_many_contours"

    # 6) Digit-shaped contour check: at least one contour should be tall-ish
    #    (digits are taller than wide, or roughly square)
    has_digit_shape = False
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        cnt_area = cv2.contourArea(cnt)
        # Must be a meaningful size relative to image
        if cnt_area < (h * w * 0.05):
            continue
        # Digits are typically taller than wide, or roughly square
        cnt_aspect = cw / max(ch, 1)
        if 0.2 <= cnt_aspect <= 1.5:
            has_digit_shape = True
            break
    if not has_digit_shape:
        return False, "no_digit_shape"

    # 7) Baseline consistency: reject merged numbers (different vertical baselines)
    if not check_baseline_consistency(img_gray):
        return False, "baseline_mismatch"

    return True, "ok"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Filter raw captures using CRNN model")
    parser.add_argument("--input", type=str, default="data/raw_captures")
    parser.add_argument("--output", type=str, default="data/filtered")
    parser.add_argument("--model", type=str, default="models/crnn/damage_crnn_best.pth")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Process only N images (0 = all)")
    parser.add_argument("--infer-batch", type=int, default=64,
                        help="CRNN batch inference size")
    parser.add_argument("--min-conf", type=float, default=0.5,
                        help="Minimum average CRNN confidence to keep (0.0-1.0)")
    parser.add_argument("--min-digit-conf", type=float, default=0.0,
                        help="Minimum per-digit confidence (0.0-1.0). "
                             "Rejects if ANY digit is below this.")
    parser.add_argument("--max-blank-ratio", type=float, default=0.90,
                        help="Max blank ratio in CTC output (0.0-1.0). "
                             "Higher = more lenient.")
    parser.add_argument("--tta", action="store_true",
                        help="Enable test-time augmentation (shift consistency check). "
                             "Slower but much fewer false positives.")
    parser.add_argument("--min-digits", type=int, default=1,
                        help="Minimum digit count to keep. Set to 2 to reject all "
                             "single-digit detections (most are garbage).")
    parser.add_argument("--max-digits", type=int, default=6,
                        help="Max digits in a damage number")
    parser.add_argument("--max-value", type=int, default=9999,
                        help="Max plausible damage value. Numbers above this are "
                             "likely merged and will be rejected.")
    args = parser.parse_args()

    # Auto-set stricter defaults when min-conf is high
    if args.min_conf >= 0.9 and args.min_digit_conf == 0.0:
        args.min_digit_conf = 0.85
        print(f"Auto-set --min-digit-conf to {args.min_digit_conf} (high min-conf mode)")
    if args.min_conf >= 0.9 and args.min_digits == 1:
        args.min_digits = 2
        print(f"Auto-set --min-digits to {args.min_digits} (high min-conf mode, "
              f"single-digit detections are mostly garbage)")

    input_dir = os.path.join(PROJECT_ROOT, args.input)
    output_dir = os.path.join(PROJECT_ROOT, args.output)
    model_path = os.path.join(PROJECT_ROOT, args.model)
    os.makedirs(output_dir, exist_ok=True)

    # Load CRNN model
    print(f"Loading CRNN model from {model_path} ...")
    from models.crnn.recognizer import CRNNRecognizer
    recognizer = CRNNRecognizer(model_path)
    print(f"Model loaded on {recognizer.device}")

    # List all PNGs
    print(f"Scanning {input_dir} ...")
    all_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    total = len(all_files)
    print(f"Found {total} images")

    if args.batch_size > 0:
        all_files = all_files[:args.batch_size]
        print(f"Processing first {len(all_files)} images")

    print(f"Min avg confidence: {args.min_conf}")
    print(f"Min per-digit confidence: {args.min_digit_conf}")
    print(f"Max blank ratio: {args.max_blank_ratio}")
    print(f"Digit count range: {args.min_digits}-{args.max_digits}")
    print(f"TTA enabled: {args.tta}")
    print()

    kept = 0
    rejected = 0
    errors = 0
    start_time = time.time()

    # Process in batches for GPU efficiency
    batch_size = args.infer_batch
    for batch_start in range(0, len(all_files), batch_size):
        batch_files = all_files[batch_start:batch_start + batch_size]

        # Load images
        images = []
        valid_files = []
        for fname in batch_files:
            fpath = os.path.join(input_dir, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                errors += 1
                continue
            h, w = img.shape
            # Basic size filter (skip tiny/huge images)
            if h < 8 or w < 8 or h > 500 or w > 500:
                rejected += 1
                continue
            images.append(img)
            valid_files.append(fname)

        if not images:
            continue

        # Structural heuristic filter before CRNN
        heuristic_passed = []
        heuristic_files = []
        for img, fname in zip(images, valid_files):
            ok, reason = is_digit_like(img)
            if ok:
                heuristic_passed.append(img)
                heuristic_files.append(fname)
            else:
                rejected += 1

        if not heuristic_passed:
            continue

        # Batch CRNN inference with detailed metrics (or TTA)
        if args.tta:
            results = recognizer.recognize_batch_tta(heuristic_passed)
        else:
            results = recognizer.recognize_batch_detailed(heuristic_passed)

        for fname, (text, info) in zip(heuristic_files, results):
            # Skip empty detections
            if not text:
                rejected += 1
                continue

            mean_conf = info['mean_conf']
            min_conf = info['min_conf']
            blank_ratio = info['blank_ratio']

            # Filter 1: average confidence
            if mean_conf < args.min_conf:
                rejected += 1
                continue

            # Filter 2: per-digit minimum confidence
            if min_conf < args.min_digit_conf:
                rejected += 1
                continue

            # Filter 3: blank ratio (model mostly outputting blanks = uncertain)
            if blank_ratio > args.max_blank_ratio:
                rejected += 1
                continue

            # Filter 4: digit count
            if not (args.min_digits <= len(text) <= args.max_digits):
                rejected += 1
                continue

            # Filter 5: max plausible value
            try:
                if int(text) > args.max_value:
                    rejected += 1
                    continue
            except ValueError:
                rejected += 1
                continue

            idx = fname.split('_')[0]
            new_name = f"{idx}_{text}.png"
            shutil.copy2(
                os.path.join(input_dir, fname),
                os.path.join(output_dir, new_name)
            )
            kept += 1

        # Progress
        done = min(batch_start + batch_size, len(all_files))
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(all_files) - done) / rate if rate > 0 else 0
        print(f"  [{done}/{len(all_files)}] "
              f"kept={kept} rejected={rejected} errors={errors} "
              f"({rate:.0f} img/s, ETA {eta:.0f}s)")

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Total processed: {len(all_files)}")
    print(f"  Kept: {kept} ({100*kept/max(len(all_files),1):.1f}%)")
    print(f"  Rejected: {rejected}")
    print(f"  Errors: {errors}")
    print(f"  Output: {output_dir}")
    print(f"\nNext: python tools/fast_review.py --dir data/filtered")


if __name__ == "__main__":
    main()

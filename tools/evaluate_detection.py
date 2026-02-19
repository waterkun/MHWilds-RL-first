"""
Evaluate the full detection pipeline (color filter + contour + CRNN).

Two modes:
  --mode capture : Capture game frames, run detection, let user annotate ground truth
  --mode eval    : Load annotated frames and compute detection precision/recall/F1

Usage:
    python tools/evaluate_detection.py --mode capture
    python tools/evaluate_detection.py --mode eval
    python tools/evaluate_detection.py --mode eval --save-plots
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default ROI - adjust to your game resolution
DEFAULT_ROI = (0, 0, 1920, 1080)


def get_roi_from_config(project_root):
    """Try to load ROI from calibration config."""
    config_path = os.path.join(project_root, "config", "roi.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        if 'damage_roi' in cfg:
            r = cfg['damage_roi']
            return (r['x'], r['y'], r['w'], r['h'])
    return DEFAULT_ROI


# ──────────────────────────────────────────────
#  Phase A: Capture & Annotate
# ──────────────────────────────────────────────

def run_capture(data_dir, roi, window_name):
    """
    Capture game frames, run detection, and let user annotate.

    Controls:
      SPACE  - capture a frame and annotate
      Q      - quit

    Annotation per frame:
      Y - all detections correct, no misses (saves as-is)
      N - wrong: type the true count of visible damage numbers
      S - skip this frame
    """
    from utils.capture import WindowCapture
    from utils.vision import ImageProcessor

    os.makedirs(os.path.join(data_dir, "frames"), exist_ok=True)

    cap = WindowCapture(window_name)
    processor = ImageProcessor(crnn_model_path="models/crnn/damage_crnn_best.pth")

    frame_idx = _next_frame_idx(data_dir)
    print("\n" + "=" * 50)
    print("  Detection Evaluation - Capture Mode")
    print("=" * 50)
    print(f"  ROI: {roi}")
    print(f"  Output: {data_dir}")
    print(f"  Starting at frame index: {frame_idx}")
    print()
    print("  Controls:")
    print("    SPACE - capture frame")
    print("    Q     - quit")
    print("=" * 50 + "\n")

    while True:
        frame = cap.get_screenshot()
        preview = frame.copy()

        # Draw ROI rectangle
        x, y, w, h = roi
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show live preview (resized for display)
        disp = _resize_for_display(preview, max_w=1280)
        cv2.imshow("Detection Capture (SPACE=capture, Q=quit)", disp)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        if key != ord(' '):
            continue

        # Capture: run detection
        result = processor.detect_hit_signals(frame, roi, max_ocr=10, debug=True)
        total_damage, hit_count, mask, filtered_mask, ocr_details, rejected_stats = result

        # Draw detections on frame
        annotated = frame.copy()
        bboxes = []
        for det in ocr_details:
            bx, by, bw, bh = det['bbox']
            abs_x, abs_y = x + bx, y + by
            color = (0, 255, 0) if det['value'] is not None else (0, 0, 255)
            cv2.rectangle(annotated, (abs_x, abs_y), (abs_x + bw, abs_y + bh), color, 2)
            if det['value'] is not None:
                cv2.putText(annotated, str(det['value']), (abs_x, abs_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                bboxes.append({
                    'x': bx, 'y': by, 'w': bw, 'h': bh,
                    'value': det['value'],
                })

        # Show annotated frame
        disp = _resize_for_display(annotated, max_w=1280)
        cv2.imshow("Detections (Y=correct, N=wrong, S=skip)", disp)
        print(f"\n  Frame {frame_idx}: detected {hit_count} hits, total={total_damage}")
        print(f"  Press Y (all correct), N (has errors), S (skip)")

        while True:
            akey = cv2.waitKey(0) & 0xFF
            if akey == ord('y'):
                # Save frame + ground truth = detections
                _save_frame(data_dir, frame_idx, frame, bboxes, true_count=len(bboxes))
                print(f"  -> Saved frame {frame_idx} (GT = {len(bboxes)} detections)")
                frame_idx += 1
                break
            elif akey == ord('n'):
                # Ask for true count
                cv2.destroyWindow("Detections (Y=correct, N=wrong, S=skip)")
                true_count = _ask_true_count()
                if true_count is not None:
                    _save_frame(data_dir, frame_idx, frame, bboxes,
                                true_count=true_count)
                    print(f"  -> Saved frame {frame_idx} (detected={len(bboxes)}, true={true_count})")
                    frame_idx += 1
                break
            elif akey == ord('s'):
                print("  -> Skipped")
                break

    cv2.destroyAllWindows()
    print(f"\nDone. Captured frames saved to {data_dir}/frames/")


def _ask_true_count():
    """Ask user to type the true count of damage numbers in the terminal."""
    try:
        s = input("    Enter true count of visible damage numbers (or 'skip'): ").strip()
        if s.lower() == 'skip':
            return None
        return int(s)
    except (ValueError, EOFError):
        return None


def _next_frame_idx(data_dir):
    """Find the next available frame index."""
    frames_dir = os.path.join(data_dir, "frames")
    if not os.path.exists(frames_dir):
        return 0
    existing = []
    for f in os.listdir(frames_dir):
        if f.startswith("frame_") and f.endswith(".png"):
            try:
                idx = int(f.split("_")[1].split(".")[0])
                existing.append(idx)
            except ValueError:
                pass
    return max(existing) + 1 if existing else 0


def _save_frame(data_dir, idx, frame, detected_bboxes, true_count):
    """Save frame image and ground-truth annotation."""
    frames_dir = os.path.join(data_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Save frame image
    img_path = os.path.join(frames_dir, f"frame_{idx:04d}.png")
    cv2.imwrite(img_path, frame)

    # Save annotation
    gt = {
        'frame_idx': idx,
        'detected_bboxes': detected_bboxes,
        'detected_count': len(detected_bboxes),
        'true_count': true_count,
    }
    gt_path = os.path.join(frames_dir, f"frame_{idx:04d}_gt.json")
    with open(gt_path, 'w') as f:
        json.dump(gt, f, indent=2)


def _resize_for_display(img, max_w=1280):
    """Resize image if wider than max_w for comfortable display."""
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / w
        return cv2.resize(img, (max_w, int(h * scale)))
    return img


# ──────────────────────────────────────────────
#  Phase B: Evaluate
# ──────────────────────────────────────────────

def run_eval(data_dir, roi, save_plots=False):
    """
    Load annotated frames and evaluate detection pipeline.

    Two evaluation strategies:
    1. Count-based: compare detected count vs true_count (always available)
    2. IoU-based: if ground-truth bboxes are annotated, use IoU >= 0.5 matching
    """
    from utils.vision import ImageProcessor
    processor = ImageProcessor(crnn_model_path="models/crnn/damage_crnn_best.pth")

    frames_dir = os.path.join(data_dir, "frames")
    if not os.path.exists(frames_dir):
        print(f"No frames directory at {frames_dir}")
        return

    # Load all annotated frames
    annotations = []
    for f in sorted(os.listdir(frames_dir)):
        if f.endswith("_gt.json"):
            gt_path = os.path.join(frames_dir, f)
            img_name = f.replace("_gt.json", ".png")
            img_path = os.path.join(frames_dir, img_name)
            if os.path.exists(img_path):
                with open(gt_path, 'r') as fp:
                    gt = json.load(fp)
                annotations.append((img_path, gt))

    if not annotations:
        print("No annotated frames found!")
        return

    print(f"\n{'=' * 60}")
    print(f"  Detection Evaluation - {len(annotations)} annotated frames")
    print(f"{'=' * 60}")
    print(f"  ROI: {roi}\n")

    # Per-frame evaluation
    frame_results = []
    total_tp = total_fp = total_fn = 0
    count_errors = []  # (detected, true) pairs for count-based eval

    for img_path, gt in annotations:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  Warning: could not read {img_path}")
            continue

        # Run detection
        result = processor.detect_hit_signals(frame, roi, max_ocr=10, debug=True)
        _, hit_count, _, _, ocr_details, _ = result

        true_count = gt['true_count']
        detected_count = sum(1 for d in ocr_details if d['value'] is not None)

        # Count-based metrics
        count_errors.append((detected_count, true_count))

        # IoU-based matching if GT bboxes available with true_count matching detected
        gt_bboxes = gt.get('detected_bboxes', [])
        if gt_bboxes and gt['detected_count'] == true_count:
            # GT bboxes were confirmed correct - use IoU matching
            pred_bboxes = [d['bbox'] for d in ocr_details if d['value'] is not None]
            tp, fp, fn = _iou_match(pred_bboxes, gt_bboxes)
        else:
            # Fall back to count-based TP/FP/FN estimation
            tp = min(detected_count, true_count)
            fp = max(0, detected_count - true_count)
            fn = max(0, true_count - detected_count)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        fname = os.path.basename(img_path)
        frame_results.append({
            'frame': fname,
            'detected': detected_count,
            'true': true_count,
            'tp': tp, 'fp': fp, 'fn': fn,
        })

    # Overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Print per-frame table
    print(f"  {'Frame':<25} {'Det':>4} {'True':>5} {'TP':>4} {'FP':>4} {'FN':>4}")
    print(f"  {'-' * 50}")
    for r in frame_results:
        print(f"  {r['frame']:<25} {r['detected']:>4} {r['true']:>5} "
              f"{r['tp']:>4} {r['fp']:>4} {r['fn']:>4}")

    print(f"\n  {'=' * 50}")
    print(f"  Overall Detection Metrics")
    print(f"  {'=' * 50}")
    print(f"  Total TP:    {total_tp}")
    print(f"  Total FP:    {total_fp}")
    print(f"  Total FN:    {total_fn}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")
    print(f"  F1:          {f1:.4f}")

    # Count accuracy (how often detected_count == true_count)
    exact_count = sum(1 for d, t in count_errors if d == t)
    print(f"\n  Count accuracy: {exact_count}/{len(count_errors)} "
          f"= {exact_count / len(count_errors):.4f}")

    # Mean absolute count error
    mae = np.mean([abs(d - t) for d, t in count_errors])
    print(f"  Mean |det - true|: {mae:.2f}")
    print(f"  {'=' * 50}")

    # Save metrics
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    metrics = {
        'num_frames': len(frame_results),
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'count_accuracy': exact_count / len(count_errors) if count_errors else 0.0,
        'mean_abs_count_error': float(mae),
        'per_frame': frame_results,
    }

    metrics_path = os.path.join(logs_dir, "detection_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Saved: {metrics_path}")

    if save_plots:
        _save_detection_plots(frame_results, logs_dir)


def _iou_match(pred_bboxes, gt_bboxes, iou_threshold=0.5):
    """
    Match predicted bboxes to GT bboxes using IoU >= threshold.

    Returns: (tp, fp, fn)
    """
    if not gt_bboxes:
        return 0, len(pred_bboxes), 0
    if not pred_bboxes:
        return 0, 0, len(gt_bboxes)

    matched_gt = set()
    tp = 0

    for pred in pred_bboxes:
        px, py, pw, ph = _unpack_bbox(pred)
        best_iou = 0.0
        best_gt_idx = -1

        for gi, gt in enumerate(gt_bboxes):
            if gi in matched_gt:
                continue
            gx, gy, gw, gh = _unpack_bbox(gt)
            iou = _compute_iou(px, py, pw, ph, gx, gy, gw, gh)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gi

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)

    fp = len(pred_bboxes) - tp
    fn = len(gt_bboxes) - len(matched_gt)
    return tp, fp, fn


def _unpack_bbox(bbox):
    """Unpack bbox from either dict or tuple format."""
    if isinstance(bbox, dict):
        return bbox['x'], bbox['y'], bbox['w'], bbox['h']
    return bbox  # already (x, y, w, h)


def _compute_iou(x1, y1, w1, h1, x2, y2, w2, h2):
    """Compute Intersection over Union for two bboxes (x, y, w, h)."""
    # Convert to (left, top, right, bottom)
    l1, t1, r1, b1 = x1, y1, x1 + w1, y1 + h1
    l2, t2, r2, b2 = x2, y2, x2 + w2, y2 + h2

    inter_l = max(l1, l2)
    inter_t = max(t1, t2)
    inter_r = min(r1, r2)
    inter_b = min(b1, b2)

    inter_w = max(0, inter_r - inter_l)
    inter_h = max(0, inter_b - inter_t)
    intersection = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def _save_detection_plots(frame_results, output_dir):
    """Save per-frame detection bar chart."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [Warning] matplotlib not installed, skipping plots")
        return

    frames = [r['frame'] for r in frame_results]
    detected = [r['detected'] for r in frame_results]
    true = [r['true'] for r in frame_results]

    x = np.arange(len(frames))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(frames) * 0.4), 5))
    ax.bar(x - width / 2, detected, width, label='Detected', alpha=0.8)
    ax.bar(x + width / 2, true, width, label='True', alpha=0.8)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Count')
    ax.set_title('Detection: Detected vs True Count per Frame')
    ax.set_xticks(x)
    ax.set_xticklabels([f[-8:-4] for f in frames], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()

    path = os.path.join(output_dir, 'detection_counts.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate detection pipeline")
    parser.add_argument("--mode", type=str, required=True, choices=["capture", "eval"],
                        help="capture = annotate frames, eval = compute metrics")
    parser.add_argument("--data-dir", type=str, default="data/detection_eval",
                        help="Directory for frames and annotations")
    parser.add_argument("--roi", type=str, default=None,
                        help="ROI as x,y,w,h (e.g. 100,200,800,600)")
    parser.add_argument("--window", type=str, default="Monster Hunter Wilds",
                        help="Game window name (capture mode)")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save detection plots to logs/")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, args.data_dir)

    if args.roi:
        roi = tuple(int(v) for v in args.roi.split(','))
    else:
        roi = get_roi_from_config(project_root)

    if args.mode == "capture":
        run_capture(data_dir, roi, args.window)
    elif args.mode == "eval":
        run_eval(data_dir, roi, save_plots=args.save_plots)


if __name__ == "__main__":
    main()

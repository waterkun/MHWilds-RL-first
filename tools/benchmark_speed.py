"""
Speed benchmark: CRNN vs Tesseract latency comparison.

Generates synthetic test crops and measures p50/p95/p99 latency.

Usage:
    python tools/benchmark_speed.py --model models/crnn/damage_crnn_best.pth
"""

import os
import sys
import time
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_test_crops(num=200):
    """Generate random grayscale test crops with varying sizes."""
    crops = []
    for _ in range(num):
        # Simulate typical damage number crop sizes
        h = np.random.randint(15, 60)
        w = np.random.randint(20, 120)
        img = np.random.randint(0, 256, (h, w), dtype=np.uint8)
        crops.append(img)
    return crops


def benchmark_crnn(crops, model_path, warmup=10):
    """Benchmark CRNN single and batch inference."""
    from models.crnn.recognizer import CRNNRecognizer
    recognizer = CRNNRecognizer(model_path)

    # Warmup
    for i in range(min(warmup, len(crops))):
        recognizer.recognize_single(crops[i])

    # Single inference
    single_times = []
    for crop in crops:
        t0 = time.perf_counter()
        recognizer.recognize_single(crop)
        elapsed = (time.perf_counter() - t0) * 1000
        single_times.append(elapsed)

    # Batch inference (batch size 5, simulating typical frame)
    batch_times = []
    for i in range(0, len(crops) - 4, 5):
        batch = crops[i:i+5]
        t0 = time.perf_counter()
        recognizer.recognize_batch(batch)
        elapsed = (time.perf_counter() - t0) * 1000
        batch_times.append(elapsed / len(batch))  # per-crop time

    single_times.sort()
    batch_times.sort()

    return {
        'name': 'CRNN (single)',
        'times': single_times,
    }, {
        'name': 'CRNN (batch=5)',
        'times': batch_times,
    }


def benchmark_tesseract(crops, warmup=3):
    """Benchmark Tesseract OCR."""
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = r'E:\Program Files\Tesseract-OCR\tesseract.exe'
    except ImportError:
        print("Tesseract not available")
        return None

    config = r'--psm 7 -c tessedit_char_whitelist=0123456789'

    # Warmup
    for i in range(min(warmup, len(crops))):
        try:
            enlarged = cv2.resize(crops[i], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            _, thresh = cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            pytesseract.image_to_string(thresh, config=config)
        except Exception:
            pass

    times = []
    # Only test a subset for Tesseract (it's much slower)
    test_crops = crops[:min(50, len(crops))]

    for crop in test_crops:
        enlarged = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.bitwise_not(thresh)

        t0 = time.perf_counter()
        try:
            pytesseract.image_to_string(thresh, config=config)
        except Exception:
            pass
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

    times.sort()
    return {
        'name': 'Tesseract',
        'times': times,
    }


def print_benchmark(results):
    """Print benchmark results table."""
    print("\n" + "=" * 70)
    print(f"{'Method':<20} | {'Count':>6} | {'Mean':>8} | {'P50':>8} | {'P95':>8} | {'P99':>8}")
    print("-" * 70)

    for r in results:
        times = r['times']
        n = len(times)
        if n == 0:
            continue
        mean = np.mean(times)
        p50 = times[n // 2]
        p95 = times[int(n * 0.95)]
        p99 = times[int(n * 0.99)]
        print(f"  {r['name']:<18} | {n:>6} | {mean:>6.2f}ms | {p50:>6.2f}ms | {p95:>6.2f}ms | {p99:>6.2f}ms")

    print("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CRNN vs Tesseract speed benchmark")
    parser.add_argument("--model", type=str, default="models/crnn/damage_crnn_best.pth",
                        help="CRNN model checkpoint")
    parser.add_argument("--num", type=int, default=200,
                        help="Number of test crops (default: 200)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, args.model)

    print("Generating test crops...")
    crops = generate_test_crops(args.num)
    print(f"Generated {len(crops)} test crops")

    results = []

    # CRNN
    if os.path.exists(model_path):
        print("\nBenchmarking CRNN...")
        single, batch = benchmark_crnn(crops, model_path)
        results.append(single)
        results.append(batch)
    else:
        print(f"CRNN model not found: {model_path}")

    # Tesseract
    print("\nBenchmarking Tesseract...")
    tess = benchmark_tesseract(crops)
    if tess:
        results.append(tess)

    if results:
        print_benchmark(results)

        # Speedup calculation
        crnn_results = [r for r in results if 'CRNN' in r['name']]
        tess_results = [r for r in results if 'Tesseract' in r['name']]
        if crnn_results and tess_results:
            crnn_mean = np.mean(crnn_results[0]['times'])
            tess_mean = np.mean(tess_results[0]['times'])
            print(f"\nSpeedup: {tess_mean / crnn_mean:.1f}x faster than Tesseract")


if __name__ == "__main__":
    main()

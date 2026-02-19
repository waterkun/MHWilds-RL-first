"""
Auto-collect damage number crops from live game footage.

Reuses WindowCapture + ImageProcessor pipelines.
Triggered by controller attack inputs (XInput Y / RT).
Saves grayscale crops + Tesseract pseudo-labels to data/raw_captures/.

Usage:
    python tools/collect_data.py
    - Press Y or RT on controller to attack
    - Damage crops are automatically saved after each attack
    - Press ';' to quit
"""

import os
import sys
import time
import cv2
import numpy as np
import ctypes

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.capture import WindowCapture
from utils.vision import ImageProcessor
from utils.vision_utils import check_baseline_consistency, try_horizontal_split

# ============ XInput (reused from calibrate_roi.py) ============

class XINPUT_GAMEPAD(ctypes.Structure):
    _fields_ = [
        ("wButtons", ctypes.c_ushort),
        ("bLeftTrigger", ctypes.c_ubyte),
        ("bRightTrigger", ctypes.c_ubyte),
        ("sThumbLX", ctypes.c_short),
        ("sThumbLY", ctypes.c_short),
        ("sThumbRX", ctypes.c_short),
        ("sThumbRY", ctypes.c_short),
    ]

class XINPUT_STATE(ctypes.Structure):
    _fields_ = [
        ("dwPacketNumber", ctypes.c_ulong),
        ("Gamepad", XINPUT_GAMEPAD),
    ]

XINPUT_BUTTON_Y = 0x8000
RT_THRESHOLD = 100

def _load_xinput():
    for lib in ["xinput1_4", "xinput1_3", "xinput9_1_0"]:
        try:
            return ctypes.windll.LoadLibrary(lib)
        except OSError:
            continue
    return None

_xinput_dll = _load_xinput()

def get_gamepad_state(controller_id=0):
    if _xinput_dll is None:
        return None
    state = XINPUT_STATE()
    res = _xinput_dll.XInputGetState(controller_id, ctypes.byref(state))
    if res == 0:
        return state.Gamepad.wButtons, state.Gamepad.bRightTrigger
    return None


def main():
    import keyboard
    import argparse

    parser = argparse.ArgumentParser(description="Collect damage number crops from game")
    parser.add_argument("--output", type=str, default="data/raw_captures",
                        help="Output directory")
    parser.add_argument("--roi", type=str, default="924,25,1534,1398",
                        help="Damage ROI as x,y,w,h")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay after attack before capture (seconds)")
    parser.add_argument("--window-sec", type=float, default=1.5,
                        help="Detection window duration (seconds)")
    parser.add_argument("--controller", type=int, default=1,
                        help="XInput controller ID (0-3)")
    parser.add_argument("--crnn-model", type=str,
                        default="models/crnn/damage_crnn_best.pth",
                        help="CRNN model for real-time filtering (set 'none' to disable)")
    parser.add_argument("--min-conf", type=float, default=0.90,
                        help="Minimum CRNN confidence to save crop")
    parser.add_argument("--min-digits", type=int, default=2,
                        help="Minimum digit count to save (rejects single-digit garbage)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, args.output)
    os.makedirs(output_dir, exist_ok=True)

    roi = tuple(int(x) for x in args.roi.split(','))
    assert len(roi) == 4, "ROI must be x,y,w,h"

    cap = WindowCapture("Monster Hunter Wilds")
    processor = ImageProcessor()

    # Load CRNN model for real-time filtering
    crnn_recognizer = None
    if args.crnn_model and args.crnn_model.lower() != 'none':
        model_path = os.path.join(project_root, args.crnn_model)
        if os.path.exists(model_path):
            from models.crnn.recognizer import CRNNRecognizer
            crnn_recognizer = CRNNRecognizer(model_path)
            print(f"CRNN filter loaded: {model_path} (device: {crnn_recognizer.device})")
            print(f"  Min confidence: {args.min_conf}, Min digits: {args.min_digits}")
        else:
            print(f"Warning: CRNN model not found at {model_path}, saving all crops")

    print("=" * 50)
    print("Damage Number Data Collector")
    print(f"Output: {output_dir}")
    print(f"ROI: {roi}")
    print(f"Delay: {args.delay}s, Window: {args.window_sec}s")
    print(f"CRNN filter: {'ON' if crnn_recognizer else 'OFF'}")
    if _xinput_dll:
        print("Controller: XInput detected")
    else:
        print("Controller: NOT detected (keyboard fallback)")
    print("Press ';' to quit")
    print("=" * 50)

    # Track existing samples
    existing = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
    sample_idx = existing
    print(f"Starting from index {sample_idx} ({existing} existing files)")

    _prev_y = False
    _prev_rt = False

    while True:
        if keyboard.is_pressed(';'):
            break

        # Check controller input
        attack_triggered = False
        pad = get_gamepad_state(args.controller)
        if pad is not None:
            buttons, rt_val = pad
            y_now = bool(buttons & XINPUT_BUTTON_Y)
            rt_now = rt_val > RT_THRESHOLD

            if y_now and not _prev_y:
                print(f"  [DEBUG] Controller Y pressed (buttons=0x{buttons:04X})")
                attack_triggered = True
            if rt_now and not _prev_rt:
                print(f"  [DEBUG] Controller RT pressed (rt_val={rt_val})")
                attack_triggered = True

            _prev_y = y_now
            _prev_rt = rt_now

        # Keyboard 'f' always works (not just fallback)
        if not attack_triggered and keyboard.is_pressed('f'):
            print("  [DEBUG] Keyboard 'F' pressed")
            attack_triggered = True
            time.sleep(0.3)  # debounce

        if not attack_triggered:
            time.sleep(0.016)  # ~60fps polling
            continue

        print(f"  [DEBUG] Attack detected! Waiting {args.delay}s...")

        # Wait for damage numbers to appear
        time.sleep(args.delay)

        # Capture frames in detection window
        x, y, w, h = roi
        t_start = time.time()
        t_end = t_start + args.window_sec
        crops_saved = 0

        while time.time() < t_end:
            frame = cap.get_screenshot()
            crop = frame[y:y+h, x:x+w]

            # Use color mask + contour pipeline to find damage regions
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            # Yellow/orange (weakness/crit)
            mask_yellow = cv2.inRange(hsv, np.array([15, 60, 180]), np.array([35, 255, 255]))
            # White (normal damage)
            mask_white = cv2.inRange(hsv, np.array([0, 0, 220]), np.array([180, 30, 255]))
            mask = cv2.bitwise_or(mask_yellow, mask_white)

            # Dilate only vertically (connects broken strokes) not horizontally
            # (prevents merging adjacent side-by-side numbers)
            kernel = np.ones((2, 1), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            roi_area = w * h
            area_max = max(5000, int(roi_area * 0.05))
            area_min = max(50, int(roi_area * 0.0001))

            # --- Spatial proximity grouping ---
            # Merge contours that belong to the same damage number:
            # vertically overlapping + horizontally close
            rects = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 5:  # drop tiny noise before grouping
                    continue
                rects.append(cv2.boundingRect(cnt))

            # Group rects by proximity
            used = [False] * len(rects)
            groups = []
            for i, (bx, by, bw, bh) in enumerate(rects):
                if used[i]:
                    continue
                group = [i]
                used[i] = True
                # BFS to find nearby rects belonging to same number
                queue = [i]
                while queue:
                    ci = queue.pop(0)
                    cx, cy, cw, ch = rects[ci]
                    c_center_y = cy + ch / 2
                    for j, (jx, jy, jw, jh) in enumerate(rects):
                        if used[j]:
                            continue
                        j_center_y = jy + jh / 2
                        # Vertical overlap: centers within 30% of avg height
                        avg_h = (ch + jh) / 2
                        if abs(c_center_y - j_center_y) > avg_h * 0.3:
                            continue
                        # Horizontal gap: < 1.5x average char width
                        avg_w = (cw + jw) / 2
                        gap = max(0, max(jx - (cx + cw), cx - (jx + jw)))
                        if gap > avg_w * 1.5:
                            continue
                        used[j] = True
                        group.append(j)
                        queue.append(j)
                groups.append(group)

            # Build merged bounding boxes from groups
            merged_rects = []
            for group in groups:
                xs = [rects[i][0] for i in group]
                ys = [rects[i][1] for i in group]
                x2s = [rects[i][0] + rects[i][2] for i in group]
                y2s = [rects[i][1] + rects[i][3] for i in group]
                merged_rects.append((min(xs), min(ys),
                                     max(x2s) - min(xs), max(y2s) - min(ys)))

            # Collect candidate crops from this frame
            frame_crops = []      # grayscale crops for CRNN
            for (bx, by, bw, bh) in merged_rects:
                area = bw * bh
                if area < area_min or area > area_max:
                    continue
                if bh < 12:
                    continue

                aspect = bw / bh
                if aspect < 0.2 or aspect > 4.5:
                    continue

                # Extract crop
                pad_px = 5
                cx1 = max(0, bx - pad_px)
                cy1 = max(0, by - pad_px)
                cx2 = min(w, bx + bw + pad_px)
                cy2 = min(h, by + bh + pad_px)
                if (cx2 - cx1) < 8 or (cy2 - cy1) < 8:
                    continue

                digit_crop = gray[cy1:cy2, cx1:cx2]

                # Reject likely merged numbers by baseline consistency
                if not check_baseline_consistency(digit_crop):
                    continue

                # Try splitting wide crops (merged side-by-side numbers)
                sub_crops = try_horizontal_split(digit_crop)
                for sc in sub_crops:
                    frame_crops.append(sc)

            # Validate and save crops
            if frame_crops and crnn_recognizer:
                # Batch CRNN inference for all crops in this frame
                results = crnn_recognizer.recognize_batch(frame_crops)
                for digit_crop, (value, confidence) in zip(frame_crops, results):
                    if value is None or confidence < args.min_conf:
                        continue
                    label = str(value)
                    if len(label) < args.min_digits:
                        continue
                    filename = f"{sample_idx:06d}_{label}.png"
                    cv2.imwrite(os.path.join(output_dir, filename), digit_crop)
                    sample_idx += 1
                    crops_saved += 1
            elif frame_crops:
                # No CRNN - save all with "unknown" label (fallback)
                for digit_crop in frame_crops:
                    filename = f"{sample_idx:06d}_unknown.png"
                    cv2.imwrite(os.path.join(output_dir, filename), digit_crop)
                    sample_idx += 1
                    crops_saved += 1

            time.sleep(0.033)  # ~30fps capture

        if crops_saved > 0:
            print(f"  Saved {crops_saved} crops (total: {sample_idx})")
        else:
            print(f"  [DEBUG] No damage numbers detected in {args.window_sec}s window. "
                  f"Contours found in last frame: {len(contours) if 'contours' in dir() else '?'}")

    print(f"\nCollection complete. Total samples: {sample_idx}")


if __name__ == "__main__":
    main()

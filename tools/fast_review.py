"""
Fast image reviewer for filtered/raw captures.

Shows images rapidly. Single-key controls:
    Space / Right arrow  = SKIP (move to next without action)
    D / Delete           = DELETE current image from disk
    Enter                = KEEP (move to reviewed folder with label)
    0-9 then Enter       = KEEP with typed label
    B / Left arrow       = BACK to previous image (undo last action)
    A                    = Auto-skip 50 images
    X                    = Auto-DELETE next 50 images (with preview)
    Q                    = Quit and save

For filtered data (--dir data/filtered), labels are read from filenames.
No OCR is run - the tool is lightweight and fast.

Usage:
    python tools/fast_review.py --dir data/filtered
    python tools/fast_review.py --dir data/raw_captures
    python tools/fast_review.py --start 0 --dir data/filtered
"""

import os
import sys
import cv2
import numpy as np
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def label_from_filename(fname):
    """Extract label from filename like '00123_4567.png' -> '4567'."""
    name = os.path.splitext(fname)[0]
    parts = name.split('_', 1)
    if len(parts) >= 2:
        label = parts[1]
        if label.isdigit():
            return label
    return ""


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fast image reviewer")
    parser.add_argument("--dir", type=str, default="data/raw_captures")
    parser.add_argument("--output", type=str, default="data/reviewed")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--target", type=int, default=1000,
                        help="Stop after keeping this many images")
    args = parser.parse_args()

    input_dir = os.path.join(PROJECT_ROOT, args.dir)
    output_dir = os.path.join(PROJECT_ROOT, args.output)
    os.makedirs(output_dir, exist_ok=True)

    # Load file list
    print(f"Scanning {input_dir} ...")
    all_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    total = len(all_files)
    print(f"Found {total} images")

    # Resume progress
    progress_file = os.path.join(output_dir, "_progress.txt")
    if args.start is not None:
        pos = args.start
    elif os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            try:
                pos = int(f.read().strip())
            except ValueError:
                pos = 0
    else:
        pos = 0

    kept_count = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
    print(f"Starting at index {pos}, already kept {kept_count} images")
    print(f"Target: {args.target} images")
    print()
    print("Controls:")
    print("  Space/Right = SKIP (next)")
    print("  D/Delete    = DELETE this image from disk")
    print("  Enter       = KEEP (move to reviewed)")
    print("  0-9 + Enter = KEEP with typed label")
    print("  B/Left      = BACK to last image (undo & redo)")
    print("  A           = Skip 50 images")
    print("  X           = DELETE next 50 images")
    print("  Q           = Quit")
    print()

    win_name = "Fast Review"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 800, 600)

    input_buffer = ""
    kept_this_session = 0
    deleted_this_session = 0
    history = []  # stack of (pos, saved_filepath_or_None) for undo

    while pos < total and (kept_count < args.target):
        fname = all_files[pos]
        fpath = os.path.join(input_dir, fname)

        # File may have been deleted
        if not os.path.exists(fpath):
            pos += 1
            continue

        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            pos += 1
            continue

        h, w = img.shape

        # Get label from filename (no OCR)
        file_label = label_from_filename(fname)

        # Enlarge for display
        scale = max(1, min(8, 400 // max(h, 1)))
        display = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_NEAREST)
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

        # Info bar
        bar_h = 100
        bar_w = max(display.shape[1], 600)
        info_bar = np.zeros((bar_h, bar_w, 3), dtype=np.uint8)

        # Pad display to match bar width if needed
        if display.shape[1] < bar_w:
            pad = np.zeros((display.shape[0], bar_w - display.shape[1], 3), dtype=np.uint8)
            display = np.hstack([display, pad])

        cv2.putText(info_bar, f"[{pos+1}/{total}] {fname}  ({w}x{h}px)",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(info_bar, f"Label: {file_label if file_label else '(none)'}  |  Kept: {kept_count}  Del: {deleted_this_session}",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        if input_buffer:
            cv2.putText(info_bar, f"Typing: {input_buffer}_",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        else:
            cv2.putText(info_bar, "Space=SKIP D=DEL Enter=KEEP B=BACK 0-9=type Q=quit",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        combined = np.vstack([display, info_bar])
        cv2.imshow(win_name, combined)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('b') or key == ord('B') or key == 81:
            # B or Left arrow = go back to previous image, undo last action
            if history:
                prev_pos, saved_path = history.pop()
                # If last action saved a file, delete it to undo
                if saved_path and os.path.exists(saved_path):
                    os.remove(saved_path)
                    kept_count -= 1
                    kept_this_session -= 1
                    print(f"  UNDO: removed {os.path.basename(saved_path)}, back to [{prev_pos+1}/{total}]")
                else:
                    print(f"  BACK to [{prev_pos+1}/{total}]")
                pos = prev_pos
            else:
                print("  (no history to go back)")
            input_buffer = ""
        elif key == ord(' ') or key == 83:
            # Space, Right arrow = skip
            history.append((pos, None))
            pos += 1
            input_buffer = ""
        elif key == ord('d') or key == ord('D') or key == 0:
            # D or Delete = delete from disk
            history.append((pos, None))
            try:
                os.remove(fpath)
                deleted_this_session += 1
            except OSError:
                pass
            pos += 1
            input_buffer = ""
        elif key == ord('a') or key == ord('A'):
            # Auto-skip 50
            history.append((pos, None))
            pos += 50
            input_buffer = ""
            print(f"  Skipped to {pos}")
        elif key == ord('x') or key == ord('X'):
            # Auto-delete next 50
            history.append((pos, None))
            count = 0
            for i in range(pos, min(pos + 50, total)):
                fp = os.path.join(input_dir, all_files[i])
                if os.path.exists(fp):
                    try:
                        os.remove(fp)
                        count += 1
                    except OSError:
                        pass
            deleted_this_session += count
            pos += 50
            input_buffer = ""
            print(f"  Deleted {count} images, moved to {pos}")
        elif key == 13:
            # Enter = keep
            if input_buffer:
                label = input_buffer
            elif file_label:
                label = file_label
            else:
                label = "unknown"
            idx_str = fname.split('_')[0]
            new_name = f"{idx_str}_{label}.png"
            saved_path = os.path.join(output_dir, new_name)
            shutil.copy2(fpath, saved_path)
            history.append((pos, saved_path))
            kept_count += 1
            kept_this_session += 1
            print(f"  KEPT [{kept_count}]: {new_name}")
            pos += 1
            input_buffer = ""
        elif key == 8:
            # Backspace
            input_buffer = input_buffer[:-1]
        elif chr(key).isdigit():
            input_buffer += chr(key)
        else:
            # Any other key = skip
            history.append((pos, None))
            pos += 1
            input_buffer = ""

        # Save progress
        with open(progress_file, 'w') as f:
            f.write(str(pos))

    cv2.destroyAllWindows()
    print(f"\nDone! Kept {kept_this_session}, deleted {deleted_this_session} this session")
    print(f"Total kept: {kept_count}")


if __name__ == "__main__":
    main()

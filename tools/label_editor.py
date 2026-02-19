"""
Label correction tool for damage number crops.

Opens an OpenCV window showing each image enlarged with its current label.
Keyboard controls:
    - Type digits (0-9) to set new label, then Enter to confirm
    - Enter alone to accept current label
    - 'd' to mark as delete (bad crop)
    - 'u' to undo last action
    - 'q' to quit and save progress

Reads from data/raw_captures/, renames files to {index}_{corrected_label}.png
Saves progress to data/raw_captures/_progress.txt
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_samples(data_dir):
    """Load all PNG samples from directory, sorted by index."""
    samples = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.png'):
            continue
        path = os.path.join(data_dir, fname)
        parts = fname.rsplit('.', 1)[0].split('_', 1)
        if len(parts) == 2:
            idx_str, label = parts
            samples.append({
                'path': path,
                'filename': fname,
                'index': idx_str,
                'label': label,
                'original_label': label,
            })
    return samples


def save_progress(data_dir, current_pos):
    """Save current position for resuming later."""
    progress_file = os.path.join(data_dir, "_progress.txt")
    with open(progress_file, 'w') as f:
        f.write(str(current_pos))


def load_progress(data_dir):
    """Load saved position."""
    progress_file = os.path.join(data_dir, "_progress.txt")
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            try:
                return int(f.read().strip())
            except ValueError:
                pass
    return 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Label editor for damage number crops")
    parser.add_argument("--dir", type=str, default="data/raw_captures",
                        help="Data directory")
    parser.add_argument("--start", type=int, default=None,
                        help="Start from specific index (overrides saved progress)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, args.dir)

    if not os.path.isdir(data_dir):
        print(f"Directory not found: {data_dir}")
        return

    samples = load_samples(data_dir)
    if not samples:
        print("No PNG files found in directory")
        return

    print(f"Loaded {len(samples)} samples from {data_dir}")

    # Resume from saved progress
    pos = args.start if args.start is not None else load_progress(data_dir)
    pos = min(pos, len(samples) - 1)

    stats = {'accepted': 0, 'corrected': 0, 'deleted': 0}
    undo_stack = []

    win_name = "Label Editor"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 600, 400)

    input_buffer = ""
    delete_dir = os.path.join(data_dir, "_deleted")

    while pos < len(samples):
        sample = samples[pos]
        if not os.path.exists(sample['path']):
            pos += 1
            continue

        img = cv2.imread(sample['path'], cv2.IMREAD_GRAYSCALE)
        if img is None:
            pos += 1
            continue

        # Enlarge for display
        scale = max(4, 200 // max(img.shape[0], 1))
        display = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

        # Draw info
        info_h = 120
        info_bar = np.zeros((info_h, display.shape[1], 3), dtype=np.uint8)
        cv2.putText(info_bar, f"[{pos+1}/{len(samples)}] {sample['filename']}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(info_bar, f"Current label: {sample['label']}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(info_bar, f"Type digits + Enter = relabel | Enter = accept | d = delete | u = undo",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        if input_buffer:
            cv2.putText(info_bar, f"New label: {input_buffer}_",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        combined = np.vstack([display, info_bar])
        cv2.imshow(win_name, combined)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('d'):
            # Mark for deletion
            os.makedirs(delete_dir, exist_ok=True)
            new_path = os.path.join(delete_dir, sample['filename'])
            os.rename(sample['path'], new_path)
            undo_stack.append(('delete', pos, sample['path'], new_path))
            stats['deleted'] += 1
            print(f"  Deleted: {sample['filename']}")
            pos += 1
            input_buffer = ""
        elif key == ord('u'):
            # Undo
            if undo_stack:
                action = undo_stack.pop()
                if action[0] == 'delete':
                    _, old_pos, orig_path, del_path = action
                    if os.path.exists(del_path):
                        os.rename(del_path, orig_path)
                    pos = old_pos
                    stats['deleted'] -= 1
                    print(f"  Undo delete")
                elif action[0] == 'rename':
                    _, old_pos, orig_path, new_path = action
                    if os.path.exists(new_path):
                        os.rename(new_path, orig_path)
                        samples[old_pos]['path'] = orig_path
                    pos = old_pos
                    stats['corrected'] -= 1
                    print(f"  Undo rename")
            input_buffer = ""
        elif key == 13:  # Enter
            if input_buffer:
                # Relabel
                new_label = input_buffer
                new_filename = f"{sample['index']}_{new_label}.png"
                new_path = os.path.join(data_dir, new_filename)
                old_path = sample['path']
                os.rename(old_path, new_path)
                sample['path'] = new_path
                sample['filename'] = new_filename
                sample['label'] = new_label
                undo_stack.append(('rename', pos, old_path, new_path))
                stats['corrected'] += 1
                print(f"  Relabeled: {new_label}")
            else:
                # Accept
                stats['accepted'] += 1
            pos += 1
            input_buffer = ""
        elif key == 8:  # Backspace
            input_buffer = input_buffer[:-1]
        elif chr(key).isdigit():
            input_buffer += chr(key)

        save_progress(data_dir, pos)

    cv2.destroyAllWindows()
    save_progress(data_dir, pos)

    print(f"\nDone! Accepted: {stats['accepted']}, "
          f"Corrected: {stats['corrected']}, Deleted: {stats['deleted']}")


if __name__ == "__main__":
    main()

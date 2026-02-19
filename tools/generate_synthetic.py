"""
Synthetic damage number generator for CRNN training.

Generates grayscale images of 1-4 digit numbers with game-like augmentations:
- Random fonts (bold, italic variants)
- Rotation +/-5 degrees
- Scale variation
- Gaussian blur
- Salt & pepper noise
- Morphological dilation/erosion

Output: data/synthetic/train/ and data/synthetic/val/
Each image is saved as {index}_{label}.png
"""

import os
import sys
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_system_fonts():
    """Collect available monospace/bold system fonts suitable for digit rendering."""
    candidates = [
        "arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf",
        "calibri.ttf", "calibrib.ttf",
        "consola.ttf", "consolab.ttf",
        "cour.ttf", "courbd.ttf",
        "impact.ttf",
        "verdana.ttf", "verdanab.ttf",
        "tahoma.ttf", "tahomabd.ttf",
        "trebuc.ttf", "trebucbd.ttf",
        "segoeui.ttf", "segoeuib.ttf",
    ]

    font_dirs = []
    if sys.platform == "win32":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        font_dirs.append(os.path.join(windir, "Fonts"))
    else:
        font_dirs.extend(["/usr/share/fonts", "/usr/local/share/fonts",
                          os.path.expanduser("~/.fonts")])

    found = []
    for d in font_dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower() in [c.lower() for c in candidates]:
                    found.append(os.path.join(root, f))

    if not found:
        found.append(None)  # Will use PIL default font
    return found


def render_digit_image(label, font_path, font_size, img_height=32):
    """
    Render a digit string onto a grayscale image.

    Returns:
        img: PIL Image (mode 'L'), height=img_height, width=proportional
        or None if rendering fails
    """
    try:
        if font_path is not None:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except (IOError, OSError):
        font = ImageFont.load_default()

    # Measure text size
    dummy = Image.new('L', (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    if tw <= 0 or th <= 0:
        return None

    # Create image with padding
    pad_x = random.randint(2, 8)
    pad_y = random.randint(2, 6)
    canvas_w = tw + pad_x * 2
    canvas_h = th + pad_y * 2

    # Random background brightness (dark, simulating game background)
    bg = random.randint(0, 40)
    img = Image.new('L', (canvas_w, canvas_h), bg)
    draw = ImageDraw.Draw(img)

    # Random foreground brightness (bright, simulating damage numbers)
    fg = random.randint(180, 255)
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), label, fill=fg, font=font)

    # Resize to target height, keep aspect ratio
    new_w = max(4, int(canvas_w * img_height / canvas_h))
    img = img.resize((new_w, img_height), Image.BILINEAR)

    return img


def augment_image(img):
    """Apply random augmentations to a PIL grayscale image."""
    # Random rotation (+/- 5 degrees)
    if random.random() < 0.5:
        angle = random.uniform(-5, 5)
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False,
                         fillcolor=random.randint(0, 30))

    # Random scale (0.8x - 1.2x), then resize back
    if random.random() < 0.4:
        w, h = img.size
        scale = random.uniform(0.8, 1.2)
        new_w = max(4, int(w * scale))
        new_h = max(8, int(h * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        # Resize back to original height
        final_w = max(4, int(new_w * h / new_h))
        img = img.resize((final_w, h), Image.BILINEAR)

    # Gaussian blur
    if random.random() < 0.3:
        radius = random.choice([0.5, 1.0, 1.5])
        img = img.filter(ImageFilter.GaussianBlur(radius))

    # Salt & pepper noise
    if random.random() < 0.3:
        arr = np.array(img)
        noise_ratio = random.uniform(0.01, 0.05)
        num_noise = int(arr.size * noise_ratio)
        # Salt
        coords = [np.random.randint(0, max(1, d), num_noise // 2) for d in arr.shape]
        arr[coords[0], coords[1]] = 255
        # Pepper
        coords = [np.random.randint(0, max(1, d), num_noise // 2) for d in arr.shape]
        arr[coords[0], coords[1]] = 0
        img = Image.fromarray(arr)

    # Morphological operations (simulate thick/thin strokes)
    if random.random() < 0.2:
        arr = np.array(img)
        import cv2
        kernel = np.ones((2, 2), np.uint8)
        if random.random() < 0.5:
            arr = cv2.dilate(arr, kernel, iterations=1)
        else:
            arr = cv2.erode(arr, kernel, iterations=1)
        img = Image.fromarray(arr)

    return img


def generate_label():
    """Generate a random damage number label (1-4 digits)."""
    num_digits = random.choices([1, 2, 3, 4], weights=[10, 40, 35, 15])[0]

    if num_digits == 1:
        return str(random.randint(1, 9))
    elif num_digits == 2:
        return str(random.randint(10, 99))
    elif num_digits == 3:
        return str(random.randint(100, 999))
    else:
        return str(random.randint(1000, 9999))


def generate_dataset(output_dir, num_samples, split_ratio=0.9):
    """Generate synthetic dataset with train/val split."""
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    fonts = get_system_fonts()
    print(f"Found {len(fonts)} font(s)")

    num_train = int(num_samples * split_ratio)
    num_val = num_samples - num_train
    print(f"Generating {num_train} train + {num_val} val = {num_samples} total")

    for split, count, out_dir in [("train", num_train, train_dir),
                                   ("val", num_val, val_dir)]:
        generated = 0
        attempts = 0
        while generated < count:
            attempts += 1
            if attempts > count * 3:
                print(f"Warning: too many failed attempts for {split}, stopping at {generated}")
                break

            label = generate_label()
            font_path = random.choice(fonts)
            font_size = random.randint(20, 48)

            img = render_digit_image(label, font_path, font_size, img_height=32)
            if img is None:
                continue

            img = augment_image(img)

            # Save as {index}_{label}.png
            filename = f"{generated:06d}_{label}.png"
            img.save(os.path.join(out_dir, filename))

            generated += 1
            if generated % 5000 == 0:
                print(f"  [{split}] {generated}/{count}")

        print(f"  [{split}] Done: {generated} images")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic damage number images")
    parser.add_argument("--output", type=str, default="data/synthetic",
                        help="Output directory (default: data/synthetic)")
    parser.add_argument("--num", type=int, default=50000,
                        help="Number of images to generate (default: 50000)")
    parser.add_argument("--split", type=float, default=0.9,
                        help="Train/val split ratio (default: 0.9)")
    args = parser.parse_args()

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.output
    )
    generate_dataset(output_dir, args.num, args.split)
    print(f"\nDataset saved to: {output_dir}")


if __name__ == "__main__":
    main()

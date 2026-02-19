"""
CRNN training script for damage number recognition.

Features:
- CTC Loss + Adam optimizer + ReduceLROnPlateau
- Custom Dataset that parses labels from filenames ({index}_{label}.png)
- Variable-width collate function with padding
- TensorBoard logging
- Model checkpointing (best val accuracy)
- Phase A (synthetic) and Phase B (fine-tune with real data) support
"""

import os
import sys
import argparse
import time
import glob

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.crnn.architecture import DamageCRNN


# ============ Dataset ============

class DamageNumberDataset(Dataset):
    """
    Loads images from a directory where filenames are {index}_{label}.png
    Returns grayscale images normalized to height=32, float32 in [0,1].
    """

    TARGET_HEIGHT = 32

    def __init__(self, img_dir):
        self.samples = []  # list of (path, label_str)

        if not os.path.isdir(img_dir):
            print(f"Warning: directory not found: {img_dir}")
            return

        for fname in os.listdir(img_dir):
            if not fname.endswith('.png'):
                continue
            # Parse label from filename: {index}_{label}.png
            parts = fname.rsplit('.', 1)[0].split('_', 1)
            if len(parts) == 2:
                label = parts[1]
                if label.isdigit():
                    self.samples.append((os.path.join(img_dir, fname), label))

        print(f"  Loaded {len(self.samples)} samples from {img_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # Load grayscale
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Return a minimal valid sample
            img = np.zeros((self.TARGET_HEIGHT, 4), dtype=np.uint8)
            label = "0"

        h, w = img.shape
        new_w = max(4, int(w * self.TARGET_HEIGHT / h))
        img = cv2.resize(img, (new_w, self.TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1], add channel dim: (1, 32, W)
        tensor = img.astype(np.float32) / 255.0
        tensor = tensor[np.newaxis, :, :]

        # Encode label: digit '0' -> index 1, '1' -> index 2, ..., '9' -> index 10
        target = [int(c) + 1 for c in label]

        return torch.from_numpy(tensor), torch.tensor(target, dtype=torch.long), len(target)


class MixedDataset(Dataset):
    """Mix multiple datasets with specified ratios for fine-tuning."""

    def __init__(self, datasets, weights=None):
        """
        Args:
            datasets: list of DamageNumberDataset
            weights: list of float (sampling weights per dataset)
        """
        self.datasets = datasets
        self.cumulative_sizes = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative_sizes.append(total)
        self.total = total

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        for i, cs in enumerate(self.cumulative_sizes):
            if idx < cs:
                if i == 0:
                    return self.datasets[i][idx]
                return self.datasets[i][idx - self.cumulative_sizes[i - 1]]
        return self.datasets[-1][idx - self.cumulative_sizes[-2]]


def collate_fn(batch):
    """
    Custom collate for variable-width images.
    Pads all images to the max width in the batch.

    Returns:
        images: (B, 1, 32, max_W) float tensor
        targets: concatenated target tensor (for CTC loss)
        target_lengths: (B,) tensor
        input_lengths: (B,) tensor  (T = max_W // 4 for each sample)
    """
    images, targets, target_lengths = zip(*batch)

    # Find max width
    max_w = max(img.shape[2] for img in images)

    # Pad images
    padded = []
    for img in images:
        w = img.shape[2]
        if w < max_w:
            pad = torch.zeros(1, 32, max_w - w)
            img = torch.cat([img, pad], dim=2)
        padded.append(img)

    images_tensor = torch.stack(padded, dim=0)  # (B, 1, 32, max_W)

    # CTC loss expects concatenated targets
    targets_concat = torch.cat(targets)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    # Input lengths: T = max_W // 4 for all (since we padded to same width)
    input_lengths = torch.full((len(images),), max_w // 4, dtype=torch.long)

    return images_tensor, targets_concat, target_lengths, input_lengths


# ============ Training ============

def levenshtein_distance(s1, s2):
    """Calculates the Levenshtein distance between two lists/strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def compute_accuracy(model, dataloader, device, max_batches=None):
    """
    Compute sequence-level accuracy (exact match) and Character Error Rate (CER).
    Returns: (accuracy, avg_cer)
    """
    model.eval()
    correct = 0
    total = 0
    total_cer = 0.0
    total_chars = 0

    with torch.no_grad():
        for batch_idx, (images, targets, target_lengths, input_lengths) in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            images = images.to(device)
            log_probs = model(images)  # (T, B, C)

            # Decode predictions
            preds = log_probs.argmax(dim=2)  # (T, B)
            preds = preds.permute(1, 0).cpu().numpy()  # (B, T)

            # Reconstruct target labels
            offset = 0
            for i in range(len(target_lengths)):
                tlen = target_lengths[i].item()
                target_seq = targets[offset:offset + tlen].numpy()
                offset += tlen

                # Greedy decode prediction
                pred_seq = preds[i]
                decoded = []
                prev = -1
                for p in pred_seq:
                    if p != prev:
                        if p != 0:  # skip blank
                            decoded.append(p)
                        prev = p

                # Compare
                if list(decoded) == list(target_seq):
                    correct += 1
                total += 1
                
                # Calculate CER
                dist = levenshtein_distance(decoded, target_seq)
                total_cer += dist
                total_chars += len(target_seq)

    accuracy = correct / total if total > 0 else 0.0
    avg_cer = total_cer / total_chars if total_chars > 0 else 0.0
    return accuracy, avg_cer


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Resolve paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ---- Dataset ----
    if args.phase == 'A':
        # Phase A: synthetic only
        train_dir = os.path.join(project_root, args.train_dir)
        val_dir = os.path.join(project_root, args.val_dir)
        print(f"Phase A: synthetic pre-training")
        train_dataset = DamageNumberDataset(train_dir)
        val_dataset = DamageNumberDataset(val_dir)
    else:
        # Phase B: mix real (30%) + synthetic (70%)
        syn_train_dir = os.path.join(project_root, "data/synthetic/train")
        real_train_dir = os.path.join(project_root, args.train_dir)
        val_dir = os.path.join(project_root, args.val_dir)
        print(f"Phase B: fine-tuning (real + synthetic)")
        syn_dataset = DamageNumberDataset(syn_train_dir)
        real_dataset = DamageNumberDataset(real_train_dir)
        train_dataset = MixedDataset([real_dataset, syn_dataset])
        val_dataset = DamageNumberDataset(val_dir)

    if len(train_dataset) == 0:
        print("Error: no training data found!")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers,
                              collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers,
                            collate_fn=collate_fn, pin_memory=True)

    # ---- Model ----
    model = DamageCRNN().to(device)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Resumed from checkpoint: {args.resume}")
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded weights from: {args.resume}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ---- Optimizer & Scheduler ----
    lr = args.lr if args.lr else (1e-3 if args.phase == 'A' else 1e-4)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                      factor=0.5, patience=5)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # ---- Logging ----
    log_dir = os.path.join(project_root, "logs", f"crnn_phase{args.phase}")
    writer = SummaryWriter(log_dir)
    save_dir = os.path.join(project_root, "models", "crnn")
    os.makedirs(save_dir, exist_ok=True)
    best_acc = 0.0

    # ---- Training Loop ----
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        t0 = time.time()

        for images, targets, target_lengths, input_lengths in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            log_probs = model(images)  # (T, B, C)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.time() - t0

        # Validation
        val_acc, val_cer = compute_accuracy(model, val_loader, device)
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Val CER: {val_cer:.4f} | "
              f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s")

        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("CER/val", val_cer, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(save_dir, "damage_crnn_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'loss': avg_loss,
            }, save_path)
            print(f"  >> Saved best model (acc={val_acc:.4f}) to {save_path}")

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(save_dir, f"damage_crnn_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'loss': avg_loss,
            }, ckpt_path)

    writer.close()
    print(f"\nTraining complete. Best val accuracy: {best_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train CRNN damage number recognizer")
    parser.add_argument("--phase", type=str, default="A", choices=["A", "B"],
                        help="Training phase: A=synthetic, B=fine-tune with real data")
    parser.add_argument("--train-dir", type=str, default="data/synthetic/train",
                        help="Training data directory")
    parser.add_argument("--val-dir", type=str, default="data/synthetic/val",
                        help="Validation data directory")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs (default: 100 for A, 50 for B)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: 1e-3 for A, 1e-4 for B)")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    args = parser.parse_args()

    if args.phase == 'B' and args.epochs == 100:
        args.epochs = 50  # Default for phase B

    train(args)


if __name__ == "__main__":
    main()

"""
CRNNRecognizer: inference wrapper for the DamageCRNN model.

Handles:
- Image preprocessing (height normalization to 32px, [0,1] scaling)
- Batch inference with variable-width padding
- CTC greedy decoding with confidence scores
"""

import numpy as np
import torch
import cv2

from models.crnn.architecture import DamageCRNN


class CRNNRecognizer:
    """Wraps DamageCRNN for production inference."""

    TARGET_HEIGHT = 32
    BLANK = 0  # CTC blank index

    def __init__(self, model_path, device=None):
        """
        Args:
            model_path: path to .pth checkpoint
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.model = DamageCRNN()
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _clean_artifacts(img):
        """
        Remove small bright artifacts (stars, arrows, sparkles) from grayscale crop.

        Uses morphological opening with a small kernel to erase small bright spots
        while preserving larger digit strokes.

        Args:
            img: grayscale numpy array (H, W), uint8

        Returns:
            cleaned: grayscale numpy array, same shape
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    def preprocess(self, img):
        """
        Preprocess a single grayscale crop for CRNN input.

        Args:
            img: grayscale numpy array (H, W), uint8

        Returns:
            tensor: (1, 32, W') float32 in [0, 1], W' proportionally scaled
        """
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return None

        # Clean small artifacts (stars, arrows) before recognition
        img = self._clean_artifacts(img)

        # Resize height to TARGET_HEIGHT, keep aspect ratio
        new_w = max(1, int(w * self.TARGET_HEIGHT / h))
        # Minimum width to produce at least 1 timestep (W/4 >= 1)
        new_w = max(new_w, 4)
        resized = cv2.resize(img, (new_w, self.TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1]
        tensor = resized.astype(np.float32) / 255.0
        # Add channel dim: (32, W') -> (1, 32, W')
        tensor = tensor[np.newaxis, :, :]
        return tensor

    def _pad_batch(self, tensors):
        """
        Pad variable-width tensors to the same width for batched inference.

        Args:
            tensors: list of (1, 32, W_i) numpy arrays

        Returns:
            batch: (B, 1, 32, max_W) torch tensor
            widths: list of original widths
        """
        widths = [t.shape[2] for t in tensors]
        max_w = max(widths)

        batch = np.zeros((len(tensors), 1, self.TARGET_HEIGHT, max_w), dtype=np.float32)
        for i, t in enumerate(tensors):
            batch[i, :, :, :t.shape[2]] = t

        return torch.from_numpy(batch).to(self.device), widths

    def _ctc_greedy_decode(self, log_probs):
        """
        Greedy CTC decoding for a single sequence.

        Args:
            log_probs: (T, num_classes) numpy array

        Returns:
            text: decoded digit string
            confidence: average probability of non-blank predictions
        """
        preds = np.argmax(log_probs, axis=1)  # (T,)
        probs = np.exp(np.max(log_probs, axis=1))  # (T,)

        decoded = []
        confidences = []
        prev = -1
        for t in range(len(preds)):
            if preds[t] != prev:
                if preds[t] != self.BLANK:
                    decoded.append(str(preds[t] - 1))
                    confidences.append(float(probs[t]))
                prev = preds[t]

        text = ''.join(decoded)
        confidence = float(np.mean(confidences)) if confidences else 0.0
        return text, confidence

    def _ctc_greedy_decode_detailed(self, log_probs):
        """
        Greedy CTC decoding with detailed quality metrics.

        Returns:
            text: decoded digit string
            info: dict with mean_conf, min_conf, blank_ratio, digit_timesteps
        """
        T = log_probs.shape[0]
        preds = np.argmax(log_probs, axis=1)
        probs = np.exp(np.max(log_probs, axis=1))

        decoded = []
        confidences = []
        prev = -1
        blank_count = 0
        for t in range(T):
            if preds[t] == self.BLANK:
                blank_count += 1
            if preds[t] != prev:
                if preds[t] != self.BLANK:
                    decoded.append(str(preds[t] - 1))
                    confidences.append(float(probs[t]))
                prev = preds[t]

        text = ''.join(decoded)
        info = {
            'mean_conf': float(np.mean(confidences)) if confidences else 0.0,
            'min_conf': float(min(confidences)) if confidences else 0.0,
            'blank_ratio': blank_count / T if T > 0 else 1.0,
            'digit_timesteps': len(confidences),
            'per_digit_conf': confidences,
        }
        return text, info

    def recognize_single(self, img):
        """
        Recognize digits from a single grayscale crop.

        Args:
            img: grayscale numpy array (H, W), uint8

        Returns:
            value: int or None
            confidence: float (0-1)
        """
        tensor = self.preprocess(img)
        if tensor is None:
            return None, 0.0

        batch = torch.from_numpy(tensor).unsqueeze(0).to(self.device)  # (1, 1, 32, W)

        with torch.no_grad():
            log_probs = self.model(batch)  # (T, 1, C)

        log_probs_np = log_probs[:, 0, :].cpu().numpy()  # (T, C)
        text, confidence = self._ctc_greedy_decode(log_probs_np)

        if text:
            return int(text), confidence
        return None, 0.0

    def _run_batch_inference(self, images):
        """
        Run batch inference and return raw per-image log_probs.

        Returns:
            list of (T_i, C) numpy arrays, one per image.
            Returns None for images that failed preprocessing.
        """
        if not images:
            return [None] * len(images)

        tensors = []
        valid_indices = []
        for i, img in enumerate(images):
            t = self.preprocess(img)
            if t is not None:
                tensors.append(t)
                valid_indices.append(i)

        if not tensors:
            return [None] * len(images)

        batch, widths = self._pad_batch(tensors)

        with torch.no_grad():
            log_probs = self.model(batch)

        results = [None] * len(images)
        for batch_idx, orig_idx in enumerate(valid_indices):
            actual_t = widths[batch_idx] // 4
            lp = log_probs[:actual_t, batch_idx, :].cpu().numpy()
            results[orig_idx] = lp

        return results

    def recognize_batch_detailed(self, images):
        """
        Batch recognize with detailed quality metrics per image.

        Returns:
            list of (text: str, info: dict) where info contains:
              mean_conf, min_conf, blank_ratio, digit_timesteps, per_digit_conf
        """
        raw = self._run_batch_inference(images)
        results = []
        for lp in raw:
            if lp is None:
                results.append(("", {
                    'mean_conf': 0.0, 'min_conf': 0.0,
                    'blank_ratio': 1.0, 'digit_timesteps': 0,
                    'per_digit_conf': [],
                }))
            else:
                text, info = self._ctc_greedy_decode_detailed(lp)
                results.append((text, info))
        return results

    def recognize_batch_tta(self, images, shifts=(-2, 0, 2)):
        """
        Test-time augmentation: run inference on shifted versions of each image.
        Only returns a result if all shifts agree on the same text.

        Args:
            images: list of grayscale numpy arrays
            shifts: horizontal pixel shifts to test

        Returns:
            list of (text: str, info: dict) - text="" if TTA disagrees
        """
        # Build all augmented versions
        all_augmented = []  # flat list
        img_count = len(images)
        n_shifts = len(shifts)

        for img in images:
            h, w = img.shape
            for dx in shifts:
                M = np.float32([[1, 0, dx], [0, 1, 0]])
                shifted = cv2.warpAffine(img, M, (w, h),
                                         borderMode=cv2.BORDER_REPLICATE)
                all_augmented.append(shifted)

        # Single batch inference for all augmented images
        all_results = self.recognize_batch_detailed(all_augmented)

        # Group by original image and check agreement
        final = []
        for i in range(img_count):
            group = all_results[i * n_shifts: (i + 1) * n_shifts]
            texts = [r[0] for r in group]
            infos = [r[1] for r in group]

            # All shifts must produce the same non-empty text
            if texts[0] and all(t == texts[0] for t in texts):
                # Use the info from the unshifted (middle) version
                mid = n_shifts // 2
                final.append((texts[mid], infos[mid]))
            else:
                final.append(("", {
                    'mean_conf': 0.0, 'min_conf': 0.0,
                    'blank_ratio': 1.0, 'digit_timesteps': 0,
                    'per_digit_conf': [],
                }))

        return final

    def recognize_batch(self, images):
        """
        Batch recognize digits from multiple grayscale crops.

        Args:
            images: list of grayscale numpy arrays (H_i, W_i), uint8

        Returns:
            results: list of (value: int|None, confidence: float)
        """
        if not images:
            return []

        # Preprocess all images
        tensors = []
        valid_indices = []
        for i, img in enumerate(images):
            t = self.preprocess(img)
            if t is not None:
                tensors.append(t)
                valid_indices.append(i)

        if not tensors:
            return [(None, 0.0)] * len(images)

        # Pad and batch
        batch, widths = self._pad_batch(tensors)

        # Forward pass
        with torch.no_grad():
            log_probs = self.model(batch)  # (T, B, C)

        # Decode each sequence
        results = [(None, 0.0)] * len(images)
        for batch_idx, orig_idx in enumerate(valid_indices):
            # Only use timesteps corresponding to actual width (not padding)
            actual_t = widths[batch_idx] // 4
            lp = log_probs[:actual_t, batch_idx, :].cpu().numpy()
            text, confidence = self._ctc_greedy_decode(lp)
            if text:
                results[orig_idx] = (int(text), confidence)

        return results

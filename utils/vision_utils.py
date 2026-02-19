"""
Utility functions for damage number image analysis.

Provides merge detection helpers for the data collection pipeline.
"""

import cv2
import numpy as np


def check_baseline_consistency(gray_crop, threshold=0.20):
    """
    Check if a crop contains merged numbers by comparing vertical
    center-of-mass of left half vs right half.

    Args:
        gray_crop: grayscale numpy array
        threshold: max allowed difference as fraction of height

    Returns:
        bool: True if consistent (likely single number), False if likely merged
    """
    h, w = gray_crop.shape[:2]
    if w < 8 or h < 8:
        return True

    mid = w // 2
    left_half = gray_crop[:, :mid]
    right_half = gray_crop[:, mid:]

    left_com = _vertical_center_of_mass(left_half)
    right_com = _vertical_center_of_mass(right_half)

    if left_com is None or right_com is None:
        return True

    diff = abs(left_com - right_com) / h
    return diff <= threshold


def try_horizontal_split(gray_crop, max_aspect=1.8, _depth=0):
    """
    Attempt to split a wide crop into separate numbers using vertical
    projection profile.  Splits recursively to handle 3+ merged numbers.

    Args:
        gray_crop: grayscale numpy array
        max_aspect: only attempt split if width > max_aspect * height
        _depth: internal recursion depth guard

    Returns:
        list of grayscale crops (1 if no split, 2+ if split succeeded)
    """
    h, w = gray_crop.shape[:2]
    if h == 0 or w / max(h, 1) <= max_aspect:
        return [gray_crop]
    if _depth > 4:
        return [gray_crop]

    # Vertical projection: sum brightness per column
    _, binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    projection = np.sum(binary, axis=0).astype(np.float32)

    # Find gap columns (projection below threshold) in the middle 80%
    margin = w // 10
    search_start = max(margin, 4)
    search_end = min(w - margin, w - 4)
    if search_end <= search_start:
        return [gray_crop]

    avg_proj = np.mean(projection)
    gap_thresh = avg_proj * 0.3

    # Find all gap columns in the search region
    gap_cols = []
    for c in range(search_start, search_end):
        if projection[c] <= gap_thresh:
            gap_cols.append(c)

    if not gap_cols:
        return [gray_crop]

    # Group consecutive gap columns into gap regions, pick center of best gap
    gap_groups = []
    start = gap_cols[0]
    for i in range(1, len(gap_cols)):
        if gap_cols[i] != gap_cols[i - 1] + 1:
            gap_groups.append((start, gap_cols[i - 1]))
            start = gap_cols[i]
    gap_groups.append((start, gap_cols[-1]))

    # Pick the widest gap (most likely a real separator)
    best_gap = max(gap_groups, key=lambda g: g[1] - g[0])
    split_col = (best_gap[0] + best_gap[1]) // 2

    left = gray_crop[:, :split_col]
    right = gray_crop[:, split_col:]

    # Reject tiny splits
    if left.shape[1] < 6 or right.shape[1] < 6:
        return [gray_crop]

    # Recurse on each half in case there are 3+ merged numbers
    return try_horizontal_split(left, max_aspect, _depth + 1) + \
           try_horizontal_split(right, max_aspect, _depth + 1)


def _vertical_center_of_mass(gray_region):
    """Compute the vertical center of mass of bright pixels in a region."""
    h, w = gray_region.shape[:2]
    if h == 0 or w == 0:
        return None

    _, binary = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    total = np.sum(binary)
    if total == 0:
        return None

    rows = np.arange(h).reshape(-1, 1)
    weighted = np.sum(rows * binary)
    return weighted / total

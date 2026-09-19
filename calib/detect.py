"""Find the calibration square in a scene-camera frame.

The square is a small, bright, roughly square patch that appears on an otherwise black screen.
Two things separate it from everything else in the room:
  * it *appeared* - so we subtract a frame taken with a black screen just before (static windows,
    lights and furniture cancel out);
  * it is compact, filled and square-ish - people moving or a shifting edge give thin/elongated blobs.
"""
import cv2
import numpy as np


def _appeared(rgb, base_rgb):
    """Per-pixel brightening (0..255): how much brighter any channel got versus the baseline."""
    d = rgb.astype(np.int16) - base_rgb.astype(np.int16)
    return np.clip(d.max(axis=2), 0, 255).astype(np.uint8)


def find_square(rgb, base_rgb=None, min_contrast=35.0):
    """rgb, base_rgb: HxWx3 uint8. Returns (cx, cy, score, (x, y, w, h)) or None."""
    v = _appeared(rgb, base_rgb) if base_rgb is not None else rgb.max(axis=2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))   # larger than the square (~50 px)
    th = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, k).astype(np.float32)
    peak = float(th.max())
    if peak < min_contrast:
        return None
    mask = (th > max(min_contrast, 0.45 * peak)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (700 <= area <= 12000 and 28 <= w <= 140 and 28 <= h <= 140):
            continue
        fill = area / float(w * h)
        aspect = w / float(h)
        if fill < 0.6 or not (0.65 <= aspect <= 1.55):
            continue
        strength = float(th[lab == i].mean())
        score = strength * fill
        if best is None or score > best[2]:
            best = (float(cent[i][0]), float(cent[i][1]), score, (int(x), int(y), int(w), int(h)))
    return best

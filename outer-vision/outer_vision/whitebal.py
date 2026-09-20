"""Automatic white balance for the detector.

The QNX scene camera has a strong green/yellow cast: a white wall comes out BGR ~(142, 198, 177), which is a weakly
saturated yellow-green and gets matched as "yellow". The gains are estimated from the neutral (low-saturation, mid-bright)
pixels of the picture and smoothed over time, and applied only to the frame the detector sees.
"""
from __future__ import annotations

import cv2
import numpy as np


class AutoWB:
    def __init__(self, ema: float = 0.08, min_frac: float = 0.03, lo: float = 0.65, hi: float = 1.6):
        self.ema, self.min_frac, self.lo, self.hi = ema, min_frac, lo, hi
        self.gains = np.ones(3, np.float32)        # B, G, R
        self.seeded = False

    def _estimate(self, frame: np.ndarray):
        small = frame[::4, ::4]
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        px = small.reshape(-1, 3).astype(np.float32)
        m = (hsv[:, 1] < 90) & (hsv[:, 2] > 70) & (hsv[:, 2] < 245)
        if m.mean() < self.min_frac:
            return None                              # not enough neutral surface to judge (keep the last gains)
        med = np.median(px[m], axis=0)
        return np.clip(med.mean() / np.maximum(med, 1.0), self.lo, self.hi).astype(np.float32)

    def update(self, frame: np.ndarray):
        g = self._estimate(frame)
        if g is None:
            return
        if not self.seeded:
            self.gains, self.seeded = g, True
        else:
            self.gains = (1 - self.ema) * self.gains + self.ema * g

    def apply(self, frame: np.ndarray) -> np.ndarray:
        self.update(frame)
        return np.clip(frame.astype(np.float32) * self.gains, 0, 255).astype(np.uint8)

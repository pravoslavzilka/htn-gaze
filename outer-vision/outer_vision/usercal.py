"""Per-user gaze correction: an affine map on the normalized gaze point, fitted from a short calibration
(the wearer looks at coloured squares whose scene-camera positions are known).

The file is re-read whenever it changes, so a new calibration takes effect without restarting the pipeline.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np


def fit_affine(raw: np.ndarray, target: np.ndarray, ridge: float = 1e-3):
    """raw, target: (n, 2) normalized points. Returns A (2x3) with target ~ A @ [x, y, 1]."""
    X = np.hstack([raw, np.ones((len(raw), 1))])
    # ridge toward the identity map so a few noisy points cannot produce a wild transform
    I = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    R = ridge * np.eye(3)
    R[2, 2] = 0.0
    A = np.linalg.solve(X.T @ X + R, X.T @ target + R @ I.T).T
    return A


def fit(raw, target, frame_w, frame_h):
    """Fit plus honest numbers: error before, after, and leave-one-out (all in scene-camera pixels)."""
    raw, target = np.asarray(raw, float), np.asarray(target, float)
    n = len(raw)
    scale = np.array([frame_w, frame_h])
    before = np.linalg.norm((raw - target) * scale, axis=1)
    A = fit_affine(raw, target)
    pred = np.hstack([raw, np.ones((n, 1))]) @ A.T
    after = np.linalg.norm((pred - target) * scale, axis=1)
    loo = []
    for i in range(n):
        m = np.arange(n) != i
        Ai = fit_affine(raw[m], target[m])
        p = np.hstack([raw[i:i + 1], [[1.0]]]) @ Ai.T
        loo.append(np.linalg.norm((p[0] - target[i]) * scale))
    return {"A": A.tolist(), "n": int(n), "frame_w": int(frame_w), "frame_h": int(frame_h),
            "before_px": float(before.mean()), "after_px": float(after.mean()), "loo_px": float(np.mean(loo)),
            "created": time.time()}


class UserCal:
    def __init__(self, path):
        self.path = path
        self.A = None
        self._mtime = None
        self.info = None

    def _reload(self):
        try:
            m = os.path.getmtime(self.path)
        except OSError:
            self.A, self._mtime, self.info = None, None, None
            return
        if m == self._mtime:
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
            self.A = np.array(d["A"], float)
            self.info = d
            self._mtime = m
        except (OSError, ValueError, KeyError):
            self.A = None

    def apply(self, g):
        """g: (x, y) normalized or None."""
        self._reload()
        if g is None or self.A is None:
            return g
        v = self.A @ np.array([g[0], g[1], 1.0])
        return (float(min(1.0, max(0.0, v[0]))), float(min(1.0, max(0.0, v[1]))))


class ManualOffset:
    """A hand-set shift of the gaze point (fractions of the picture width / height), applied after the calibration.
    The file is re-read whenever it changes, so the control page can move the pointer live."""

    def __init__(self, path):
        self.path = path
        self.dx = self.dy = 0.0
        self._mtime = None

    def _reload(self):
        try:
            m = os.path.getmtime(self.path)
        except OSError:
            self.dx = self.dy = 0.0
            self._mtime = None
            return
        if m == self._mtime:
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
            self.dx, self.dy, self._mtime = float(d.get("dx", 0.0)), float(d.get("dy", 0.0)), m
        except (OSError, ValueError):
            pass

    def apply(self, g):
        self._reload()
        if g is None or (self.dx == 0.0 and self.dy == 0.0):
            return g
        return (float(min(1.0, max(0.0, g[0] + self.dx))), float(min(1.0, max(0.0, g[1] + self.dy))))

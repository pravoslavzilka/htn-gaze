"""Learned shape classifier (round / square / cylinder / triangle / reject) on object crops.

Runs the ONNX export through OpenCV DNN (no extra dependency); ~0.3 ms per crop on a laptop.
Input: Nx3xSxS float RGB in 0..255 (the model divides by 255 itself).
If the model is missing, the detector falls back to contour rules.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


class ShapeNet:
    backend = "onnx"

    def __init__(self, model_dir="models/shape"):
        d = Path(model_dir)
        meta = json.loads((d / "labels.json").read_text())
        self.labels, self.size = meta["labels"], meta["input_size"]
        self.net = cv2.dnn.readNetFromONNX(str(d / "shape.onnx"))

    def probs(self, crops: list) -> np.ndarray:
        """crops: list of BGR uint8 SxS images -> (N, n_labels) softmax probabilities."""
        if not crops:
            return np.zeros((0, len(self.labels)), np.float32)
        self.net.setInput(cv2.dnn.blobFromImages(crops, 1.0, (self.size, self.size), swapRB=True))
        logits = self.net.forward().reshape(len(crops), -1)
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)


def load(cfg: dict):
    """Returns a ShapeNet, or None if disabled/missing (detector then uses rules)."""
    c = cfg.get("shape_net", {})
    if not c.get("enabled", True):
        return None
    try:
        return ShapeNet(c.get("model_dir", "models/shape"))
    except (FileNotFoundError, cv2.error, OSError) as e:
        print(f"[shape_net] disabled ({e}); using contour rules", flush=True)
        return None

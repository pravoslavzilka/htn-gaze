"""Object detection: colour prototypes find WHERE (pixel-exact outlines), a small net says WHAT shape.

1. Every pixel's (hue, saturation) is mapped through a lookup table to the nearest registered colour
   prototype (or to nothing: the grey table, shadows, glare). One gather, ~1-2 ms at 640x480.
2. Per colour: clean the mask, take outer contours = candidate objects.
3. Each candidate's crop goes through the shape net (ONNX), which says round/square/cylinder/triangle,
   or "reject" for hands, scraps, pens. Without a model, contour-geometry rules decide the shape.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

NONE = 255  # LUT value for "no colour"


@dataclass
class Detection:
    color: str
    shape: str               # "round" | "square" | "cylinder" | "triangle"
    shape_conf: float        # net probability, or 1.0/0.5 for rule match/fallback
    contour: np.ndarray      # Nx1x2 int32, process-resolution pixels
    cx: float
    cy: float
    area: float
    bbox: tuple              # x, y, w, h
    partial: bool            # touches the frame edge -> shape/size unreliable
    features: dict


def crop_for(img, bbox, out=64, pad=0.18, jitter=None):
    """Square crop around bbox with context padding, resized to out x out. Shared by training + runtime."""
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * (1 + 2 * pad)
    if jitter is not None:   # training-time augmentation: imperfect boxes
        side *= jitter.uniform(0.9, 1.15)
        cx += jitter.uniform(-0.06, 0.06) * side
        cy += jitter.uniform(-0.06, 0.06) * side
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    s = max(2, int(round(side)))
    H, W = img.shape[:2]
    pad_img = cv2.copyMakeBorder(img, s, s, s, s, cv2.BORDER_REPLICATE)
    crop = pad_img[y0 + s:y0 + 2 * s, x0 + s:x0 + 2 * s]
    return cv2.resize(crop, (out, out), interpolation=cv2.INTER_AREA)


def build_color_lut(colors: dict, m: dict) -> tuple:
    """LUT[h, s] -> colour index (NONE if unsaturated or too far from every prototype)."""
    names = list(colors)
    h = np.arange(180, dtype=np.float32)[:, None]
    s = np.arange(256, dtype=np.float32)[None, :]
    best = np.full((180, 256), np.inf, np.float32)
    lut = np.full((180, 256), NONE, np.uint8)
    for i, n in enumerate(names):
        ph, ps = colors[n]["hsv"][0], colors[n]["hsv"][1]
        dh = np.abs(h - ph)
        dh = np.minimum(dh, 180 - dh)          # hue is circular
        d = np.sqrt((dh / m["hue_tol"]) ** 2 + ((s - ps) / m["sat_tol"]) ** 2)
        better = d < best
        best[better] = d[better]
        lut[better] = i
    lut[best > m["max_dist"]] = NONE
    lut[:, : m["s_min"]] = NONE
    return names, lut


def shape_features(contour: np.ndarray, poly_eps: float) -> dict:
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = max(cv2.contourArea(hull), 1e-6)
    hull_perim = cv2.arcLength(hull, True)
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    long_side, short_side = max(rw, rh), max(min(rw, rh), 1e-6)
    (_, _), r_enc = cv2.minEnclosingCircle(contour)
    vertices = len(cv2.approxPolyDP(hull, poly_eps * hull_perim, True))
    return {
        "aspect": long_side / short_side,
        "rectfill": area / max(rw * rh, 1e-6),
        "circlefill": area / max(math.pi * r_enc * r_enc, 1e-6),
        "solidity": area / hull_area,
        "vertices": vertices,
    }


def classify_shape(f: dict, p: dict) -> tuple:
    """Rule fallback when no shape net is available.

    Triangle is tested before square, because the square rule's "few vertices and solid" branch would
    otherwise swallow it. A triangle is the shape that fills least of both the circle around it and the
    rect around it; a cylinder can have a similar circlefill but nearly fills its rect, which is what
    triangle_max_rectfill separates.
    """
    if f["circlefill"] >= p["round_min_circlefill"] and f["aspect"] < p["cylinder_min_aspect"]:
        return "round", 1.0
    if f["circlefill"] <= p["triangle_max_circlefill"] and f["rectfill"] <= p["triangle_max_rectfill"]:
        return "triangle", 1.0
    if f["aspect"] < p["square_max_aspect"] and (
        f["rectfill"] >= p["square_min_rectfill"]
        or (f["vertices"] <= p["square_max_vertices"] and f["solidity"] >= p["square_min_solidity"])
    ):
        return "square", 1.0
    if f["aspect"] >= p["cylinder_min_aspect"]:
        return "cylinder", 1.0
    return ("square", 0.5) if f["vertices"] <= p["square_max_vertices"] else ("round", 0.5)


class Detector:
    def __init__(self, cfg: dict, shape_net=None):
        self.cfg = cfg
        self.p = cfg["detector"]
        self.colors = cfg["colors"]
        self.names, self.lut = build_color_lut(self.colors, cfg["color_match"])
        self.v_min = cfg["color_match"]["v_min"]
        k = max(1, int(self.p["morph"]))
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self.net = shape_net
        self.reject_min = cfg.get("shape_net", {}).get("reject_min_prob", 0.6)
        self.rejected = 0   # count on the last frame, for the HUD

    def label_map(self, frame: np.ndarray) -> np.ndarray:
        b = int(self.p["blur"])
        if b > 1:
            frame = cv2.GaussianBlur(frame, (b | 1, b | 1), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = self.lut[hsv[:, :, 0], hsv[:, :, 1]]
        lab[hsv[:, :, 2] < self.v_min] = NONE
        return lab

    def masks(self, frame: np.ndarray) -> dict:
        lab = self.label_map(frame)
        present = np.bincount(lab.ravel(), minlength=256)
        out = {}
        min_px = self.p["min_area_frac"] * lab.size * 0.5
        for i, name in enumerate(self.names):
            if present[i] < min_px:
                continue   # skip colours that can't possibly form an object this frame
            m = (lab == i).astype(np.uint8) * 255
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.kernel)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.kernel)
            out[name] = m
        return out

    def detect(self, frame: np.ndarray) -> list:
        h, w = frame.shape[:2]
        frame_area = float(h * w)
        lo, hi = self.p["min_area_frac"] * frame_area, self.p["max_area_frac"] * frame_area
        bm = self.p["border_margin"]
        cands = []
        for name, mask in self.masks(frame).items():
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if not lo <= area <= hi:
                    continue
                m = cv2.moments(c)
                x, y, bw, bh = cv2.boundingRect(c)
                feats = shape_features(c, self.p["poly_eps"])
                shape, conf = classify_shape(feats, self.p)
                cands.append(Detection(
                    color=name, shape=shape, shape_conf=conf, contour=c,
                    cx=m["m10"] / m["m00"], cy=m["m01"] / m["m00"], area=area, bbox=(x, y, bw, bh),
                    partial=x <= bm or y <= bm or x + bw >= w - bm or y + bh >= h - bm, features=feats,
                ))
        self.rejected = 0
        if self.net is None or not cands:
            return cands
        probs = self.net.probs([crop_for(frame, d.bbox, self.net.size) for d in cands])
        labels = self.net.labels
        rej = labels.index("reject") if "reject" in labels else -1
        keep = []
        for d, p in zip(cands, probs):
            d.features["net"] = {l: round(float(v), 3) for l, v in zip(labels, p)}
            if rej >= 0 and p[rej] >= self.reject_min:
                self.rejected += 1
                continue
            if rej >= 0:
                p = p.copy()
                p[rej] = -1
            k = int(np.argmax(p))
            d.shape, d.shape_conf = labels[k], float(p[k])
            keep.append(d)
        return keep

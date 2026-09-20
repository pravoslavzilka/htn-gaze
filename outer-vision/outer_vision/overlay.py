"""Debug drawing. Never feeds back into the pipeline."""
from __future__ import annotations

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale=0.45, color=(255, 255, 255)):
    (tw, _), _ = cv2.getTextSize(s, FONT, scale, 1)
    org = (int(max(2, min(org[0], img.shape[1] - tw - 4))), int(org[1]))   # keep labels inside the frame
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, 1, cv2.LINE_AA)


def draw(frame, tracks, depth, cfg, gaze_px, selector, hud_lines, flash_id=None, markers=(), show_features=False,
         voices=None, next_id=None, menu_focus=None, caption=None, t=0.0):
    """voices: id -> {note, instrument}; next_id: lesson's next object; menu_focus: object the open blink
    menu is about; caption: (status, text) from the blink menu / Maestro."""
    img = frame.copy()
    voices = voices or {}
    t_now = t
    for t in tracks:
        col = tuple(cfg["colors"].get(t.color, {}).get("bgr", (255, 255, 255)))
        is_target = t.id == selector.target_id
        thick = 3 if is_target else 1
        if t.id == flash_id:
            overlay = img.copy()
            cv2.drawContours(overlay, [t.det.contour], -1, (255, 255, 255), -1)
            img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
        cv2.drawContours(img, [t.det.contour], -1, col, thick, cv2.LINE_AA)
        x, y, w, h = t.det.bbox
        dist, vol = depth.get(t.id, (None, None))
        label = f"#{t.id} {t.color} {t.shape}"
        if show_features or t.det.shape_conf < 0.8:
            label += f" {t.det.shape_conf:.0%}"
        v = voices.get(t.id)
        if v:
            label += f"  {v['note']} {v['instrument']}"
        if dist is not None:
            label += f" {dist:.0f}cm v{vol:.2f}"
        if t.det.partial:
            label += " (edge)"
        _text(img, label, (x, max(12, y - 6)), color=col)
        if show_features:
            f = t.det.features
            _text(img, f"cf{f['circlefill']:.2f} rf{f['rectfill']:.2f} ar{f['aspect']:.2f} v{f['vertices']} so{f['solidity']:.2f}",
                  (x, y + h + 14), 0.38)
            if "net" in f:
                _text(img, " ".join(f"{k[:3]}{v:.2f}" for k, v in f["net"].items()), (x, y + h + 28), 0.38)
        if t.id == next_id:   # lesson: pulse the object to look at next
            r = int(max(w, h) * 0.75) + 10 + int(4 * np.sin(t_now * 6))
            cv2.circle(img, (int(t.cx), int(t.cy)), r, (255, 255, 255), 2, cv2.LINE_AA)
            _text(img, "next", (int(t.cx) - 14, int(t.cy) - r - 4), 0.5)
        if t.id == menu_focus:   # the object the blink menu is about
            cv2.rectangle(img, (x - 6, y - 6), (x + w + 6, y + h + 6), (255, 0, 255), 2, cv2.LINE_AA)
        if is_target and selector.progress > 0:
            c = (int(t.cx), int(t.cy))
            r = int(max(w, h) * 0.6) + 6
            done = selector.locked
            cv2.ellipse(img, c, (r, r), -90, 0, 360 * selector.progress,
                        (0, 255, 0) if done else (0, 255, 255), 3, cv2.LINE_AA)
    for m in markers:
        pts = np.array(m["corners"], np.int32)
        cv2.polylines(img, [pts], True, (255, 0, 255), 2)
        _text(img, f"marker {m['id']}", tuple(pts[0]), color=(255, 0, 255))
    if gaze_px is not None:
        g = (int(gaze_px[0]), int(gaze_px[1]))
        # big pointer: translucent disc, white ring, cross hair, so it reads from across the room
        disc = img.copy()
        cv2.circle(disc, g, 26, (0, 0, 255), -1, cv2.LINE_AA)
        img = cv2.addWeighted(disc, 0.35, img, 0.65, 0)
        cv2.circle(img, g, 26, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(img, g, 26, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(img, (g[0] - 40, g[1]), (g[0] + 40, g[1]), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(img, (g[0], g[1] - 40), (g[0], g[1] + 40), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(img, g, 3, (0, 0, 255), -1, cv2.LINE_AA)
    else:
        _text(img, "NO GAZE  (eyes not detected)", (8, img.shape[0] - 8), 0.5, (0, 0, 255))
    for i, s in enumerate(hud_lines):
        _text(img, s, (8, 18 + 18 * i))
    if caption and (caption[0] != "idle" or caption[1]):
        status, text = caption
        H, W = img.shape[:2]
        bar = img.copy()
        cv2.rectangle(bar, (0, H - 46), (W, H), (0, 0, 0), -1)
        img = cv2.addWeighted(bar, 0.55, img, 0.45, 0)
        dot = {"menu": (255, 0, 255), "thinking": (0, 200, 255), "speaking": (0, 220, 0)}.get(status, (160, 160, 160))
        cv2.circle(img, (16, H - 23), 7, dot, -1, cv2.LINE_AA)
        _text(img, status.upper(), (30, H - 28), 0.45, dot)
        _text(img, text[:100], (30, H - 10), 0.45)
    return img


def mask_view(label_map: np.ndarray, names: list, cfg: dict) -> np.ndarray:
    """Every pixel painted with the colour prototype it was assigned to (black = none)."""
    palette = np.zeros((256, 3), np.uint8)
    for i, n in enumerate(names):
        palette[i] = cfg["colors"][n].get("bgr", (255, 255, 255))
    return palette[label_map]

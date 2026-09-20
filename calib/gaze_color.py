"""Name the colour under the gaze point: median of a small patch of the scene picture, classified in HSV."""
import cv2
import numpy as np


def name_color(bgr_patch):
    """-> (name, (r, g, b)) for a BGR patch. Uses the median so a thin edge or highlight doesn't decide it."""
    med = np.median(bgr_patch.reshape(-1, 3), axis=0).astype(np.uint8)
    h, s, v = (int(x) for x in cv2.cvtColor(med.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0])
    rgb = (int(med[2]), int(med[1]), int(med[0]))
    if v < 50:
        return "black", rgb
    if s < 40:
        return ("white" if v > 170 else "gray"), rgb
    if h < 8 or h >= 170:
        name = "red" if s > 110 or v < 200 else "pink"
    elif h < 20:
        name = "orange" if v > 120 else "brown"
    elif h < 35:
        name = "yellow"
    elif h < 85:
        name = "green"
    elif h < 100:
        name = "cyan"
    elif h < 130:
        name = "blue"
    elif h < 160:
        name = "purple"
    else:
        name = "pink"
    return name, rgb


def color_at(img_bgr, x, y, radius=14):
    h, w = img_bgr.shape[:2]
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    return name_color(img_bgr[y0:y1, x0:x1])

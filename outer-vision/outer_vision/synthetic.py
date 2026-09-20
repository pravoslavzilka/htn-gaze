"""Synthetic table scenes + gaze: tests, demos without hardware, and bootstrap training data for the shape net.

Geometry: orthographic camera looking down at `pitch` degrees below horizontal.
World X = right, Y = away from the user (along the table), Z = up. Image:
    px = X,   py = -(Y*sin(pitch) + Z*cos(pitch))
so a ground circle becomes an ellipse with axes (r, r*sin(pitch)) and height moves things up the image.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

# Display/scene colours, BGR. These must be the colours registered in config.json -> "colors", or the
# detector won't find them. Rainbow order = the C major pentatonic scale (C D E G A).
BGR = {
    "red": (30, 30, 200), "yellow": (30, 200, 220), "green": (50, 170, 50),
    "blue": (200, 90, 30), "purple": (150, 50, 120),
}
SHAPES = ("round", "square", "cylinder", "triangle")

# (color, shape, x, y, size_px) for the default demo scene: one object per colour = all five notes,
# and all four shapes = all four default instruments.
DEFAULT_OBJECTS = [
    ("red", "round", 150, 330, 46),         # C4 marimba
    ("blue", "square", 330, 360, 48),       # G4 piano
    ("green", "cylinder", 500, 320, 40),    # E4 flute
    ("yellow", "triangle", 230, 170, 40),   # D4 bell
    ("purple", "round", 430, 165, 32),      # A4 marimba
]


def _shade(c, k):
    return tuple(int(min(255, max(0, v * k))) for v in c)


def _proj(pts, pitch):
    """pts: Nx3 world (relative to object base centre) -> Nx2 image offsets."""
    sp, cp = math.sin(pitch), math.cos(pitch)
    return np.stack([pts[:, 0], -(pts[:, 1] * sp + pts[:, 2] * cp)], 1)


def table(h, w, rng, base=None):
    base = rng.uniform(105, 160) if base is None else base
    tilt = rng.uniform(-25, 25)
    grad = np.linspace(base + tilt, base - tilt, h, dtype=np.float32)[:, None]
    img = np.repeat(np.repeat(grad, w, 1)[:, :, None], 3, 2)
    img += rng.normal(0, 3.5, img.shape)
    img[:, :, :] *= rng.uniform(0.96, 1.04, 3)  # slight white-balance cast, still ~unsaturated
    return img


def draw_object(img, mask, color, shape, x, y, size, pitch=math.radians(50), yaw=0.0, aspect=1.6, light=1.0):
    """Draws one object with its base centre at (x, y). `mask` (uint8) receives the object's silhouette."""
    c = _shade(color, light)
    s = float(size)
    sp = math.sin(pitch)

    def poly(pts, col):
        p = np.round(pts + [x, y]).astype(np.int32)
        cv2.fillPoly(img, [p], col, cv2.LINE_AA)
        cv2.fillPoly(mask, [p], 255)

    # contact shadow (not part of the object)
    cv2.ellipse(img, (int(x + s * 0.15), int(y + s * 0.1)), (int(s * 0.75), max(2, int(s * 0.75 * sp * 0.6))),
                0, 0, 360, (80, 80, 80), -1, cv2.LINE_AA)
    if shape == "round":
        r = s / 2
        cy = y - r * math.cos(pitch)                   # sphere centre is r above the table
        cv2.circle(img, (int(x), int(cy)), int(r), _shade(c, 0.7), -1, cv2.LINE_AA)
        cv2.circle(img, (int(x - r * 0.15), int(cy - r * 0.15)), int(r * 0.75), c, -1, cv2.LINE_AA)
        cv2.circle(mask, (int(x), int(cy)), int(r), 255, -1)
    elif shape == "cylinder":
        r, hgt = s / 2, s * aspect
        t = np.linspace(0, 2 * math.pi, 48, endpoint=False)
        ring = np.stack([r * np.cos(t), r * np.sin(t), np.zeros_like(t)], 1)
        bottom = _proj(ring, pitch)
        top = _proj(ring + [0, 0, hgt], pitch)
        hull = cv2.convexHull(np.concatenate([bottom, top]).astype(np.float32))[:, 0]
        poly(hull, _shade(c, 0.8))       # body silhouette
        poly(top, _shade(c, 1.15))       # lit top face
    elif shape == "triangle":
        # A triangular prism standing on its base, triangular faces toward and away from the camera
        # (a foam wedge, a folded card, a Toblerone on end). The silhouette is what matters: a triangle.
        # Taller than it is wide: a triangle covers only ~0.3 of its bounding box against ~0.79 for a ball,
        # so a squat one falls under the detector's min_area_frac before any other shape does.
        a, hgt, depth = s / 2, s * 1.25, s * 0.25
        face = np.array([[-a, 0.0, 0.0], [a, 0.0, 0.0], [0.0, 0.0, hgt]])
        front, back = face + [0, -depth / 2, 0], face + [0, depth / 2, 0]
        pf, pb = _proj(front, pitch), _proj(back, pitch)
        hull = cv2.convexHull(np.concatenate([pf, pb]).astype(np.float32))[:, 0]
        poly(hull, _shade(c, 0.75))      # body silhouette (the sliver of the far face and the top edge)
        poly(pf, c)                      # lit face pointing at the camera
    elif shape == "square":
        a = s / 2
        base = np.array([[-a, -a], [a, -a], [a, a], [-a, a]])
        rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        b = base @ rot.T
        bot = np.hstack([b, np.zeros((4, 1))])
        top = bot + [0, 0, 2 * a]
        view = np.array([0, -math.cos(pitch), math.sin(pitch)])  # towards the camera
        faces = []
        for i in range(4):
            j = (i + 1) % 4
            n = np.cross(bot[j] - bot[i], top[i] - bot[i])
            n /= np.linalg.norm(n)
            if n @ view > 0:
                faces.append((n @ view, np.array([bot[i], bot[j], top[j], top[i]])))
        hull = cv2.convexHull(np.concatenate([_proj(bot, pitch), _proj(top, pitch)]).astype(np.float32))[:, 0]
        poly(hull, c)
        for k, f in sorted(faces, key=lambda q: q[0]):
            poly(_proj(f, pitch), _shade(c, 0.55 + 0.35 * k))
        poly(_proj(top, pitch), _shade(c, 1.2))


def draw_distractor(img, mask, rng, x, y, size):
    """Things that must be REJECTED: hands, paper scraps, pens. Drawn into `mask` like an object."""
    kind = rng.integers(3)
    s = float(size)
    if kind == 0:   # hand: skin-tone palm + fingers
        skin = (int(rng.uniform(90, 150)), int(rng.uniform(130, 185)), int(rng.uniform(190, 245)))
        ang = rng.uniform(0, 2 * math.pi)
        cv2.ellipse(img, (int(x), int(y)), (int(s * 0.5), int(s * 0.42)), math.degrees(ang), 0, 360, skin, -1, cv2.LINE_AA)
        cv2.ellipse(mask, (int(x), int(y)), (int(s * 0.5), int(s * 0.42)), math.degrees(ang), 0, 360, 255, -1)
        for k in range(int(rng.integers(3, 6))):
            a = ang + (k - 2) * 0.35
            p0 = (int(x + math.cos(a) * s * 0.35), int(y + math.sin(a) * s * 0.35))
            L = s * rng.uniform(0.5, 0.8)
            p1 = (int(p0[0] + math.cos(a) * L), int(p0[1] + math.sin(a) * L))
            th = max(3, int(s * 0.14))
            cv2.line(img, p0, p1, skin, th, cv2.LINE_AA)
            cv2.line(mask, p0, p1, 255, th)
    elif kind == 1:  # irregular scrap in a random saturated colour
        col = tuple(int(v) for v in cv2.cvtColor(np.uint8([[[rng.integers(180), rng.integers(150, 255), rng.integers(120, 240)]]]), cv2.COLOR_HSV2BGR)[0, 0])
        n = int(rng.integers(5, 10))
        ang = np.sort(rng.uniform(0, 2 * math.pi, n))
        rad = s * rng.uniform(0.2, 0.7, n)
        pts = np.stack([x + rad * np.cos(ang), y + rad * np.sin(ang) * rng.uniform(0.4, 1.0)], 1).astype(np.int32)
        cv2.fillPoly(img, [pts], col, cv2.LINE_AA)
        cv2.fillPoly(mask, [pts], 255)
    else:            # pen / strip: long and thin
        col = tuple(int(v) for v in cv2.cvtColor(np.uint8([[[rng.integers(180), rng.integers(150, 255), rng.integers(120, 240)]]]), cv2.COLOR_HSV2BGR)[0, 0])
        a = rng.uniform(0, math.pi)
        L = s * rng.uniform(1.2, 2.2)
        p0 = (int(x - math.cos(a) * L / 2), int(y - math.sin(a) * L / 2))
        p1 = (int(x + math.cos(a) * L / 2), int(y + math.sin(a) * L / 2))
        th = max(3, int(s * rng.uniform(0.1, 0.2)))
        cv2.line(img, p0, p1, col, th, cv2.LINE_AA)
        cv2.line(mask, p0, p1, 255, th)


def render(frame_idx: int, objects=DEFAULT_OBJECTS, w=640, h=480, seed=0, head_motion=True, pitch_deg=50):
    rng = np.random.default_rng(seed + frame_idx)
    img = table(h, w, np.random.default_rng(seed), base=130)
    img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    mask = np.zeros((h, w), np.uint8)
    t = frame_idx / 30.0
    dx = 12 * math.sin(t * 0.9) if head_motion else 0.0   # slow head sway
    dy = 6 * math.sin(t * 1.3) if head_motion else 0.0
    for i, (color, shape, x, y, s) in enumerate(objects):
        draw_object(img, mask, BGR[color], shape, x + dx, y + dy, s,
                    pitch=math.radians(pitch_deg), yaw=0.5 + i * 0.4)
    return cv2.GaussianBlur(img, (3, 3), 0)


def object_center(obj, pitch_deg=50):
    """Where an object's silhouette centre roughly lands (for fake gaze / tests)."""
    color, shape, x, y, s = obj
    up = {"round": 0.5, "square": 0.5, "cylinder": 0.8, "triangle": 0.5}[shape] * s * math.cos(math.radians(pitch_deg))
    return x, y - up


def gaze(frame_idx: int, objects=DEFAULT_OBJECTS, w=640, h=480, dwell_frames=30, move_frames=6, seed=0):
    """Visits each object in turn (1 s fixation, 0.2 s saccade), with jitter and a 100 ms blink.

    Returns normalized (x, y) or None during a blink.
    """
    period = dwell_frames + move_frames
    i, k = divmod(frame_idx, period)
    a, b = object_center(objects[i % len(objects)]), object_center(objects[(i + 1) % len(objects)])
    t = frame_idx / 30.0
    dx, dy = 12 * math.sin(t * 0.9), 6 * math.sin(t * 1.3)
    if k < dwell_frames:
        x, y = a
        if k in (15, 16, 17):   # blink mid-fixation: must NOT reset the dwell
            return None
    else:
        u = (k - dwell_frames) / move_frames
        x, y = a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u
    rng = np.random.default_rng(seed + 10_000 + frame_idx)
    x, y = x + dx + rng.normal(0, 4), y + dy + rng.normal(0, 4)
    return (x / w, y / h)


def random_crop_sample(rng, label: str, out=64):
    """One training crop (BGR uint8, out x out) for the shape classifier, label in SHAPES or 'reject'.

    The object gets a random hue so the net learns shape, not colour.
    """
    W, H = 200, 260   # tall enough for the tallest cylinder at the steepest angle
    img = table(H, W, rng)
    mask = np.zeros((H, W), np.uint8)
    size = rng.uniform(34, 70)
    x, y = W / 2 + rng.uniform(-6, 6), H * 0.75 + rng.uniform(-6, 6)
    if label == "reject":
        draw_distractor(img, mask, rng, x, H / 2, size)
    else:
        hsv = np.uint8([[[rng.integers(180), rng.integers(140, 256), rng.integers(110, 245)]]])
        col = tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
        draw_object(img, mask, col, label, x, y, size, pitch=math.radians(rng.uniform(25, 80)),
                    yaw=rng.uniform(0, math.pi / 2), aspect=rng.uniform(1.3, 2.2), light=rng.uniform(0.75, 1.15))
    img = np.clip(img * rng.uniform(0.8, 1.15) + rng.normal(0, rng.uniform(1, 6), img.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        img = cv2.GaussianBlur(img, (3, 3) if rng.random() < 0.7 else (5, 5), 0)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return random_crop_sample(rng, label, out)
    bbox = (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
    from .detector import crop_for   # same cropping as at runtime
    return crop_for(img, bbox, out, jitter=rng)

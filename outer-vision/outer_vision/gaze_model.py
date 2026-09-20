"""Coaxial fixed-rig gaze model: the eye camera's pupil pixels -> a point in the scene camera's image.

This is a port of `gui/src/geometry.js` from the eye-tracking repo (htn-gaze, branch `pupil-in-eye`),
which is what the EyeMelody GUI uses live. Keeping the two in step matters: the board reports the same
numbers to both, so if this file drifts, the overlay the eye team looks at and the notes this repo plays
stop agreeing about where the user is looking. `selfcheck()` mirrors that file's `selfCheck()`.

Physical setup (millimetres), both cameras on one optical axis looking opposite ways:

    world  <---  scene camera  |  eye camera  --->  eyes
                   (outer)         (inner)
                                  80 mm

Per eye, in the eye-camera frame (origin at the lens, +Z toward the face, +X right in the image, +Y down):
  1. the eye-corner midpoint is the cornea, on the plane z = D
  2. the eyeball is a sphere of radius R centred D+R behind that cornea
  3. the pupil pixel is a ray; take the sphere hit nearer the camera
  4. optical axis = pupil - sphere centre, then kappa (visual axis is ~5 deg temporal of it)
A 180 deg rotation about Y takes that into the scene-camera frame, where the ray is intersected with the
plane z = scene_depth_mm and projected through the scene camera's pinhole.

Detection stays on the board (MediaPipe face mesh + iris, and the dark-pupil fit in `pi/pupil.c`); nothing
here looks at pixels. Input is the JSON from the board's `GET /api/state`.
"""
from __future__ import annotations

import math

# Mirrors DEFAULT_RIG in gui/src/geometry.js, plus the GUI's "Z" key (zero_*_deg) as config.
RIG_DEFAULTS = {
    "eye_distance_mm": 80.0,      # eye camera to the corneas, measured on the rig
    "baseline_mm": 25.0,          # scene camera behind the eye camera (two stacked Camera Module 3 boards)
    "eyeball_radius_mm": 12.0,    # adult mean
    "hfov_deg": 66.0,             # Camera Module 3 standard lens
    "scene_depth_mm": 1500.0,     # the plane the gaze ray is intersected with
    "kappa_deg": 5.0,             # visual axis vs pupillary axis
    "flip_x": False,              # a module mounted inverted (the GUI's X / Y keys)
    "flip_y": False,
    "scene_w": 960,               # scene stream size, for the projection
    "scene_h": 540,
    "zero_yaw_deg": 0.0,          # residual aim, measured with tools/qnx_bridge.py --zero
    "zero_pitch_deg": 0.0,
    "fixed_center": None,         # optional {"left": [x, y], "right": [x, y]}, normalized eye-camera coords: hold the cornea
                                  # position constant (rigid rig) instead of using the noisy per-frame corner midpoint
}

MIN_OPEN_FOR_GAZE = 0.08   # an eye contributes to the gaze average only above this lid openness


def rig_from(cfg: dict | None = None) -> dict:
    """Rig defaults with `cfg` merged over them, so a partial config.json block is enough."""
    return {**RIG_DEFAULTS, **(cfg or {})}


# ---------------------------------------------------------------- small vector helpers
def _norm(v):
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) or 1.0
    return (v[0] / n, v[1] / n, v[2] / n)


def intrinsics(w: float, h: float, hfov_deg: float) -> dict:
    """Pinhole from the horizontal field of view; the vertical one follows the streamed aspect ratio."""
    hfov = math.radians(hfov_deg)
    fx = w / 2 / math.tan(hfov / 2)
    vfov = 2 * math.atan(math.tan(hfov / 2) * (h / w))
    fy = h / 2 / math.tan(vfov / 2)
    return {"fx": fx, "fy": fy, "cx": w / 2, "cy": h / 2}


def _pixel_to_dir(px, py, k):
    x = (px - k["cx"]) / k["fx"]
    y = (py - k["cy"]) / k["fy"]
    n = math.hypot(math.hypot(x, y), 1.0) or 1.0
    return (x / n, y / n, 1.0 / n)


def _ray_plane_z(origin, d, z):
    if abs(d[2]) < 1e-9:
        return None
    t = (z - origin[2]) / d[2]
    if t < 1e-6:
        return None
    return (origin[0] + t * d[0], origin[1] + t * d[1], origin[2] + t * d[2])


def _ray_sphere(origin, d, center, radius):
    """Nearest hit of a unit-direction ray with a sphere, or None."""
    ox, oy, oz = origin[0] - center[0], origin[1] - center[1], origin[2] - center[2]
    b = 2 * (ox * d[0] + oy * d[1] + oz * d[2])
    c = ox * ox + oy * oy + oz * oz - radius * radius
    disc = b * b - 4 * c
    if disc < 0:
        return None
    s = math.sqrt(disc)
    t1, t2 = (-b - s) / 2, (-b + s) / 2
    t = t1 if t1 > 1e-6 else t2
    if t < 1e-6:
        return None
    return (origin[0] + t * d[0], origin[1] + t * d[1], origin[2] + t * d[2])


def _yaw_pitch(d):
    return math.atan2(d[0], d[2]), math.atan2(d[1], math.hypot(d[0], d[2]))


def _from_yaw_pitch(yaw, pitch):
    cp = math.cos(pitch)
    return (math.sin(yaw) * cp, math.sin(pitch), math.cos(yaw) * cp)


# ---------------------------------------------------------------- per-eye
def eye_open(eye: dict) -> float:
    """Lid openness: the height of the eyelid outline over its width. ~0.3 open, ~0.05 closed.

    The absolute value depends on the wearer and on where the camera sits, so the closed/open thresholds
    are config (`qnx.lid`); `tools/qnx_bridge.py --probe` prints live values to set them from.
    """
    lid = eye.get("lid") or []
    if len(lid) < 4:
        return 1.0
    xs, ys = lid[0::2], lid[1::2]
    width = max(max(xs) - min(xs), 1e-6)
    return (max(ys) - min(ys)) / width


def _pupil_px(eye, w, h):
    """The dark-pupil fit from pupil.c when it succeeded, else MediaPipe's iris centre."""
    src = eye.get("pupil") if eye.get("pupil_ok") and isinstance(eye.get("pupil"), list) else eye.get("iris")
    if not isinstance(src, list) or len(src) < 2:
        return None
    return (src[0] * w, src[1] * h)


def _to_scene_dir(g, rig):
    """Eye-camera optical axis -> scene-camera frame: 180 deg about Y (back-to-back, same "up")."""
    x, y, z = -g[0], g[1], -g[2]
    if rig["flip_x"]:
        x = -x
    if rig["flip_y"]:
        y = -y
    return _norm((x, y, z))


def _to_scene_point(p, rig):
    x, y, z = -p[0], p[1], -(p[2] + rig["baseline_mm"])
    if rig["flip_x"]:
        x = -x
    if rig["flip_y"]:
        y = -y
    return (x, y, z)


def _eye_gaze(eye, side, w, h, k, rig):
    center = ((rig.get("fixed_center") or {}).get(side)) or eye.get("center")
    if not isinstance(center, list) or len(center) < 2:
        return None
    d, r = rig["eye_distance_mm"], rig["eyeball_radius_mm"]

    cornea = _ray_plane_z((0.0, 0.0, 0.0), _pixel_to_dir(center[0] * w, center[1] * h, k), d)
    if cornea is None:
        return None
    sphere_c = (cornea[0], cornea[1], d + r)

    pup = _pupil_px(eye, w, h)
    if pup is None:
        return None
    hit = _ray_sphere((0.0, 0.0, 0.0), _pixel_to_dir(pup[0], pup[1], k), sphere_c, r)
    if hit is None:
        return None

    g_eye = _norm((hit[0] - sphere_c[0], hit[1] - sphere_c[1], hit[2] - sphere_c[2]))
    yaw, pitch = _yaw_pitch(_to_scene_dir(g_eye, rig))
    kappa = math.radians(rig["kappa_deg"] or 0.0)
    signed = -kappa if side == "left" else kappa       # visual axis is temporal of the pupillary axis
    return {
        "dir": _from_yaw_pitch(yaw + signed, pitch),
        "origin": _to_scene_point(sphere_c, rig),
        "open": eye_open(eye),
        "pupil_ok": bool(eye.get("pupil_ok")),
        "pupil_score": float(eye.get("pupil_score") or 0.0),
    }


# ---------------------------------------------------------------- whole state
def gaze_from_state(state: dict, rig: dict, min_open: float = MIN_OPEN_FOR_GAZE) -> dict:
    """One `GET /api/state` body -> where the wearer is looking, in the scene image.

    Returns `x`, `y` NORMALIZED to the scene image (0..1, origin top-left: what INTERFACE.md wants), and
    `ok` False when no eye was open enough or the ray missed the scene plane. `on_image` is False when the
    wearer is looking outside the scene camera's field of view, in which case x/y are clamped to the edge.
    """
    w = float(state.get("width") or rig["scene_w"])
    h = float(state.get("height") or rig["scene_h"])
    sw = float(rig["scene_w"] or w)
    sh = float(rig["scene_h"] or h)
    k = intrinsics(w, h, rig["hfov_deg"])
    ks = intrinsics(sw, sh, rig["hfov_deg"])

    eyes_in = state.get("eyes") or {}
    out = {"ok": False, "x": None, "y": None, "on_image": False, "yaw_deg": 0.0, "pitch_deg": 0.0,
           "n": int(state.get("n") or 0), "score": float(state.get("score") or 0.0),
           "camera_fps": float(state.get("camera_fps") or 0.0),
           "infer_fps": float(state.get("infer_fps") or 0.0),
           "frame_id": int(state.get("frame_id") or 0), "eyes": {"left": None, "right": None}}

    dirs, origins = [], []
    for side in ("left", "right"):
        e = eyes_in.get(side)
        if not isinstance(e, dict):
            continue
        g = _eye_gaze(e, side, w, h, k, rig)
        out["eyes"][side] = g
        if g is not None and g["open"] >= min_open:
            dirs.append(g["dir"])
            origins.append(g["origin"])
    if not dirs:
        return out

    d = _norm((sum(v[0] for v in dirs), sum(v[1] for v in dirs), sum(v[2] for v in dirs)))
    origin = tuple(sum(o[i] for o in origins) / len(origins) for i in range(3))

    yaw, pitch = _yaw_pitch(d)
    yaw -= math.radians(rig["zero_yaw_deg"] or 0.0)
    pitch -= math.radians(rig["zero_pitch_deg"] or 0.0)

    hit = _ray_plane_z(origin, _from_yaw_pitch(yaw, pitch), rig["scene_depth_mm"])
    if hit is None or hit[2] <= 1e-3:
        return out
    px = ks["cx"] + ks["fx"] * hit[0] / hit[2]
    py = ks["cy"] + ks["fy"] * hit[1] / hit[2]

    out["on_image"] = 0 <= px < sw and 0 <= py < sh
    out["ok"] = True
    out["x"] = min(1.0, max(0.0, px / sw))
    out["y"] = min(1.0, max(0.0, py / sh))
    out["yaw_deg"] = math.degrees(yaw)
    out["pitch_deg"] = math.degrees(pitch)
    return out


class Smoother:
    """Median of the last `median` points, then an EMA. Same filter as the eye repo's useGaze.js, so the
    GUI's reticle and this repo's gaze point move together rather than one lagging the other."""

    def __init__(self, median: int = 5, ema: float = 0.45):
        self.n = max(1, int(median))
        self.w = float(ema)
        self.hist: list = []
        self.value = None

    def reset(self):
        self.hist.clear()
        self.value = None

    def push(self, x: float, y: float):
        self.hist.append((x, y))
        if len(self.hist) > self.n:
            self.hist.pop(0)
        mid = len(self.hist) // 2
        med = (sorted(p[0] for p in self.hist)[mid], sorted(p[1] for p in self.hist)[mid])
        self.value = med if self.value is None else (
            (1 - self.w) * self.value[0] + self.w * med[0],
            (1 - self.w) * self.value[1] + self.w * med[1])
        return self.value


def selfcheck() -> list:
    """Mirrors selfCheck() in gui/src/geometry.js: looking straight ahead must hit the image centre."""
    rig = rig_from({"kappa_deg": 0.0, "baseline_mm": 25.0})
    lid = [0.4, 0.45, 0.6, 0.45, 0.6, 0.55, 0.4, 0.55]
    eye = {"center": [0.5, 0.5], "iris": [0.5, 0.5], "pupil": [0.5, 0.5], "pupil_ok": 1, "lid": lid}
    g = gaze_from_state({"width": 960, "height": 540, "n": 2, "eyes": {"left": eye, "right": eye}}, rig)
    fails = []
    if not g["ok"]:
        fails.append("straight-ahead should succeed")
        return fails
    if abs(g["x"] - 0.5) > 0.01 or abs(g["y"] - 0.5) > 0.015:
        fails.append(f"straight-ahead should hit the image centre, got ({g['x']:.3f}, {g['y']:.3f})")
    if abs(g["yaw_deg"]) > 0.5 or abs(g["pitch_deg"]) > 0.5:
        fails.append(f"straight-ahead angles should be ~0, got yaw {g['yaw_deg']:.2f} pitch {g['pitch_deg']:.2f}")
    return fails

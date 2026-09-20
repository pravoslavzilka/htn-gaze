"""Python port of the coaxial geometric gaze model from the pupil-in-eye branch (gui/src/geometry.js).

Eye camera at eyeDistanceMm from the eyes, scene camera on the same axis baselineMm behind it, both looking opposite ways.
Per eye: the eye-corner midpoint is the cornea on the plane z = D; the eyeball is a sphere of radius R behind it; the iris
(or pupil) pixel is a ray hitting that sphere; optical axis = pupil - sphere centre; the gaze ray is projected onto a plane
sceneDepthMm in front of the scene camera. No calibration squares are involved.
"""
import math
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Rig:
    eyeDistanceMm: float = 80.0
    baselineMm: float = 25.0
    eyeballRadiusMm: float = 12.0
    hfovDeg: float = 66.0
    sceneDepthMm: float = 1500.0
    kappaDeg: float = 5.0
    flipX: bool = False
    flipY: bool = False
    sceneW: int = 960
    sceneH: int = 540
    yaw0Deg: float = 0.0     # residual yaw / pitch offset (what the GUI's Z key measures)
    pitch0Deg: float = 0.0


def intrinsics(w, h, hfov_deg):
    hfov = math.radians(hfov_deg)
    fx = w / 2 / math.tan(hfov / 2)
    vfov = 2 * math.atan(math.tan(hfov / 2) * (h / w))
    fy = h / 2 / math.tan(vfov / 2)
    return fx, fy, w / 2, h / 2


def _pixel_dir(px, py, K):
    fx, fy, cx, cy = K
    x, y = (px - cx) / fx, (py - cy) / fy
    n = math.sqrt(x * x + y * y + 1) or 1
    return np.array([x / n, y / n, 1 / n])


def _ray_plane_z(o, d, z):
    if abs(d[2]) < 1e-9:
        return None
    t = (z - o[2]) / d[2]
    return None if t < 1e-6 else o + t * d


def _ray_sphere(o, d, c, r):
    oc = o - c
    b = 2 * float(oc @ d)
    cc = float(oc @ oc) - r * r
    disc = b * b - 4 * cc
    if disc < 0:
        return None
    s = math.sqrt(disc)
    t1, t2 = (-b - s) / 2, (-b + s) / 2
    t = t1 if t1 > 1e-6 else t2
    return None if t < 1e-6 else o + t * d


def _norm(v):
    n = float(np.linalg.norm(v)) or 1.0
    return v / n


def _yaw_pitch(d):
    return math.atan2(d[0], d[2]), math.atan2(d[1], math.hypot(d[0], d[2]))


def _from_yaw_pitch(yaw, pitch):
    cp = math.cos(pitch)
    return np.array([math.sin(yaw) * cp, math.sin(pitch), math.cos(yaw) * cp])


def lid_open(eye):
    lid = eye.get("lid") or []
    if len(lid) < 4:
        return 1.0
    xs, ys = lid[0::2], lid[1::2]
    return (max(ys) - min(ys)) / max(max(xs) - min(xs), 1e-6)


def _eye_gaze(eye, side, w, h, K, rig):
    D, R = rig.eyeDistanceMm, rig.eyeballRadiusMm
    center, iris = eye.get("center"), eye.get("iris")
    if not center or not iris:
        return None
    cornea = _ray_plane_z(np.zeros(3), _pixel_dir(center[0] * w, center[1] * h, K), D)
    if cornea is None:
        return None
    C = np.array([cornea[0], cornea[1], D + R])
    pupil = _ray_sphere(np.zeros(3), _pixel_dir(iris[0] * w, iris[1] * h, K), C, R)
    if pupil is None:
        return None
    g = _norm(pupil - C)
    x, y, z = -g[0], g[1], -g[2]
    if rig.flipX:
        x = -x
    if rig.flipY:
        y = -y
    yaw, pitch = _yaw_pitch(_norm(np.array([x, y, z])))
    kappa = math.radians(rig.kappaDeg)
    yaw += -kappa if side == "left" else kappa
    d = _from_yaw_pitch(yaw, pitch)
    ox, oy, oz = -C[0], C[1], -(C[2] + rig.baselineMm)
    if rig.flipX:
        ox = -ox
    if rig.flipY:
        oy = -oy
    return d, np.array([ox, oy, oz])


def gaze_from_state(state, rig=Rig()):
    """Returns (x, y, yaw_deg, pitch_deg) in scene-camera pixels, or None when no usable eye."""
    w, h = state.get("width") or rig.sceneW, state.get("height") or rig.sceneH
    K = intrinsics(w, h, rig.hfovDeg)
    Ks = intrinsics(rig.sceneW, rig.sceneH, rig.hfovDeg)
    dirs, origins = [], []
    for side in ("left", "right"):
        e = (state.get("eyes") or {}).get(side)
        if not e:
            continue
        g = _eye_gaze(e, side, w, h, K, rig)
        if g is not None and lid_open(e) >= 0.08:
            dirs.append(g[0])
            origins.append(g[1])
    if not dirs:
        return None
    d = _norm(np.sum(dirs, axis=0))
    o = np.mean(origins, axis=0)
    yaw, pitch = _yaw_pitch(d)
    yaw -= math.radians(rig.yaw0Deg)
    pitch -= math.radians(rig.pitch0Deg)
    hit = _ray_plane_z(o, _from_yaw_pitch(yaw, pitch), rig.sceneDepthMm)
    if hit is None or hit[2] <= 1e-3:
        return None
    fx, fy, cx, cy = Ks
    x = cx + fx * hit[0] / hit[2]
    y = cy + fy * hit[1] / hit[2]
    return (max(-rig.sceneW * 0.15, min(rig.sceneW * 1.15, x)), max(-rig.sceneH * 0.15, min(rig.sceneH * 1.15, y)),
            math.degrees(yaw), math.degrees(pitch))


if __name__ == "__main__":   # port check against the JavaScript on a live state: prints the Python result
    import json
    import sys
    import urllib.request

    st = json.loads(urllib.request.urlopen(sys.argv[1] + "/api/state", timeout=3).read())
    json.dump(st, open("state_for_port_check.json", "w"))
    print(json.dumps(gaze_from_state(st)))

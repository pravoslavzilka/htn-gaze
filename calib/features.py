"""Pupil features from full-resolution eye crops.

The board serves, at /api/eyepack, a JPEG mosaic [left eye | right eye] cut from the full-resolution
camera frame and (in the X-Pack header) the MediaPipe landmarks the crops were centred on. Here we

  1. refine the iris centre inside each crop to sub-pixel accuracy (circle fit on the iris edge,
     using only the left/right arcs so the eyelids do not disturb it),
  2. measure it relative to a slowly-updating reference of the eye corners (median over a few seconds),
     which removes the frame-to-frame jitter of the corner landmarks,
  3. flag blinks / failed fits so they can be left out of the calibration.

Coordinates are in the "virtual" frame: the full-resolution frame with the same rotation as the RGB stream.
"""
import io
import json
import sys
import time
import urllib.request
from collections import deque

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------- fetching


def fetch_pack(base, timeout=3):
    """Return (pack dict, mosaic RGB uint8 array) from <base>/api/eyepack, or (None, None)."""
    try:
        r = urllib.request.urlopen(base + "/api/eyepack", timeout=timeout)
        pack = json.loads(r.headers["X-Pack"])
        mosaic = np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB"))
        return pack, mosaic
    except Exception:  # noqa: BLE001
        return None, None


# ---------------------------------------------------------------- iris refinement
_THETAS = np.deg2rad(np.r_[np.linspace(-40, 40, 11), np.linspace(140, 220, 11)])
_COS, _SIN = np.cos(_THETAS), np.sin(_THETAS)


def _sample(img, xs, ys):
    """Bilinear samples of img at float coordinates xs, ys (any shape)."""
    shape = xs.shape
    n = xs.size
    w2 = 1024
    pad = (-n) % w2
    mx = np.pad(xs.reshape(-1), (0, pad)).astype(np.float32).reshape(-1, w2)
    my = np.pad(ys.reshape(-1), (0, pad)).astype(np.float32).reshape(-1, w2)
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(-1)[:n].reshape(shape)


def refine_iris(gray, cx0, cy0, r0, search=0.55, step=2.5):
    """Circle fit around a predicted iris (cx0, cy0, radius r0), all in crop pixels.

    Score of a circle = mean outward image gradient along its left and right arcs (an iris is darker
    than the white of the eye, so the gradient points outward). Returns (cx, cy, r, z) where z is how far
    the best circle stands out from all other candidates (scale-free, so soft or low-contrast images work)."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), max(1.5, r0 / 14.0))
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    R = search * r0
    offs = np.arange(-R, R + 0.01, step)
    rs = r0 * np.linspace(0.6, 1.4, 9)
    X = cx0 + offs[None, None, :, None] + rs[:, None, None, None] * _COS[None, None, None, :]
    Y = cy0 + offs[None, :, None, None] + rs[:, None, None, None] * _SIN[None, None, None, :]
    X, Y = np.broadcast_arrays(X, Y)
    proj = _sample(gx, X, Y) * _COS + _sample(gy, X, Y) * _SIN            # (nr, ny, nx, nt)
    score = proj.mean(axis=3)
    ir, iy, ix = np.unravel_index(int(np.argmax(score)), score.shape)
    best = float(score[ir, iy, ix])
    # sub-pixel: quadratic interpolation of the score around the maximum along each axis
    def peak(vm, v0, vp, d):
        den = vm - 2 * v0 + vp
        return 0.0 if abs(den) < 1e-9 else float(np.clip(0.5 * (vm - vp) / den, -1, 1)) * d

    dx = dy = 0.0
    if 0 < ix < len(offs) - 1:
        dx = peak(score[ir, iy, ix - 1], score[ir, iy, ix], score[ir, iy, ix + 1], step)
    if 0 < iy < len(offs) - 1:
        dy = peak(score[ir, iy - 1, ix], score[ir, iy, ix], score[ir, iy + 1, ix], step)
    z = (best - float(score.mean())) / (float(score.std()) + 1e-9)
    return cx0 + offs[ix] + dx, cy0 + offs[iy] + dy, float(rs[ir]), float(z)


# ---------------------------------------------------------------- per-eye measurement
def measure_eye(pack, i, crop_rgb):
    """Landmarks + refined iris for eye i (0 = left, 1 = right), in virtual-frame pixels."""
    e = pack["eyes"][i]
    vw, vh = pack["vw"], pack["vh"]
    iris_mp = np.array([e["iris"][0] * vw, e["iris"][1] * vh])
    ring = np.array(e["ring"]).reshape(-1, 2) * [vw, vh]
    r0 = float(np.mean(np.linalg.norm(ring - iris_mp, axis=1)))
    corner = np.array([e["c"][0] * vw, e["c"][1] * vh])
    width = float(e["w"] * vw)
    lid = np.array(e["lid"]).reshape(-1, 2) * [vw, vh]
    openness = float((lid[:, 1].max() - lid[:, 1].min()) / max(width, 1e-6))
    m = {"iris_mp": iris_mp, "r_mp": r0, "corner": corner, "width": width, "open": openness,
         "iris": iris_mp.copy(), "r": r0, "fit_score": 0.0, "fit_ok": False}
    ch, cw = crop_rgb.shape[:2]
    px, py = iris_mp[0] - e["x0"], iris_mp[1] - e["y0"]
    if r0 > 6 and 0 < px < cw and 0 < py < ch:
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        fx, fy, fr, sc = refine_iris(gray, px, py, r0)
        moved = float(np.hypot(fx - px, fy - py))
        m["fit_score"] = sc
        # a real iris fit stays near MediaPipe's guess and keeps a plausible radius; a fit that jumps far or grows
        # is usually latching onto an eyelid edge (eye nearly closed, or looking down)
        # the fitted iris must sit inside the eye opening and keep a plausible radius. (It may move well away from
        # MediaPipe's guess: that guess is often off by half an iris radius or more.)
        fxv, fyv = fx + e["x0"], fy + e["y0"]
        inside = bool(lid[:, 0].min() - 0.1 * width <= fxv <= lid[:, 0].max() + 0.1 * width and
                      lid[:, 1].min() - 0.25 * width <= fyv <= lid[:, 1].max() + 0.25 * width)
        m["fit_ok"] = bool(sc > 2.8 and moved < 0.9 * r0 and 0.65 * r0 <= fr <= 1.35 * r0 and inside)
        if m["fit_ok"]:
            m["iris"] = np.array([fx + e["x0"], fy + e["y0"]])
            m["r"] = fr
    return m


class EyeTracker:
    """Turns eye packs into features, keeping a short history for the corner reference."""

    def __init__(self, window_s=8.0):
        self.window_s = window_s
        self.hist = [deque(), deque()]      # (t, corner_x, corner_y, width, openness)

    def update(self, pack, mosaic, t=None):
        t = time.time() if t is None else t
        cw = pack["cw"]
        res = {"fid": pack["fid"], "eyes": []}
        for i in range(2):
            crop = mosaic[:, i * cw:(i + 1) * cw]
            m = measure_eye(pack, i, crop)
            h = self.hist[i]
            h.append((t, m["corner"][0], m["corner"][1], m["width"], m["open"]))
            while h and t - h[0][0] > self.window_s:
                h.popleft()
            arr = np.array(h)
            ref_c = np.median(arr[:, 1:3], axis=0)
            ref_w = float(np.median(arr[:, 3]))
            ref_open = float(np.median(arr[:, 4]))
            m["ref_center"], m["ref_width"] = ref_c, ref_w
            m["off_ref"] = (m["iris"] - ref_c) / ref_w          # refined iris vs steady eye reference
            m["off_mp"] = (m["iris_mp"] - ref_c) / ref_w        # raw MediaPipe iris vs the same reference
            m["blink"] = bool(m["open"] < 0.6 * ref_open or m["open"] < 0.08)
            # A failed circle fit is not a reason to drop the sample: with the lids covering most of the iris there is
            # little edge to fit, and MediaPipe's own estimate (already stored in m["iris"]) is used instead.
            m["ok"] = bool(not m["blink"])
            res["eyes"].append(m)
        res["ok"] = all(m["ok"] for m in res["eyes"])
        return res


# ---------------------------------------------------------------- feature vectors
KINDS = ("ref_avg", "ref_lr", "ref_open", "mp_avg")


def feature_vector(kind, res):
    L, R = res["eyes"]
    if kind == "ref_avg":
        return (L["off_ref"] + R["off_ref"]) / 2
    if kind == "ref_lr":
        return np.concatenate([L["off_ref"], R["off_ref"]])
    if kind == "ref_open":
        return np.concatenate([(L["off_ref"] + R["off_ref"]) / 2, [(L["open"] / L["ref_width"] + R["open"] / R["ref_width"]) / 2]])
    if kind == "mp_avg":
        return (L["off_mp"] + R["off_mp"]) / 2
    raise ValueError(kind)


def flat(res):
    """Compact, JSON-friendly copy of a tracker result (what calibration saves per sample)."""
    out = {"fid": res["fid"], "ok": res["ok"], "eyes": []}
    for m in res["eyes"]:
        out["eyes"].append({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items()})
    return out


def unflat(d):
    res = {"fid": d["fid"], "ok": d["ok"], "eyes": []}
    for m in d["eyes"]:
        res["eyes"].append({k: (np.array(v) if isinstance(v, list) else v) for k, v in m.items()})
    return res


# ---------------------------------------------------------------- self-check
def _annotate(pack, mosaic, res, path):
    cw = pack["cw"]
    img = cv2.cvtColor(mosaic.copy(), cv2.COLOR_RGB2BGR)
    for i, m in enumerate(res["eyes"]):
        e = pack["eyes"][i]
        ox = i * cw - e["x0"]
        oy = -e["y0"]
        cv2.circle(img, (int(m["iris_mp"][0] + ox), int(m["iris_mp"][1] + oy)), int(m["r_mp"]), (255, 128, 0), 2)   # MediaPipe: blue
        cv2.drawMarker(img, (int(m["iris_mp"][0] + ox), int(m["iris_mp"][1] + oy)), (255, 128, 0), cv2.MARKER_CROSS, 14, 1)
        col = (0, 255, 0) if m["fit_ok"] else (0, 0, 255)
        cv2.circle(img, (int(round(m["iris"][0] + ox)), int(round(m["iris"][1] + oy))), int(m["r"]), col, 2)      # refined: green
        cv2.drawMarker(img, (int(round(m["iris"][0] + ox)), int(round(m["iris"][1] + oy))), col, cv2.MARKER_CROSS, 14, 1)
        cv2.putText(img, f"fit {m['fit_score']:.1f} open {m['open']:.2f}" + (" BLINK" if m["blink"] else ""),
                    (i * cw + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(path, img)


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "http://169.254.96.94:8080"
    out = sys.argv[2] if len(sys.argv) > 2 else "features_check.png"
    tr = EyeTracker()
    last = None
    t0 = time.time()
    n = 0
    while time.time() - t0 < 4:
        pack, mosaic = fetch_pack(base)
        if pack is None:
            time.sleep(0.2)
            continue
        last = (pack, mosaic, tr.update(pack, mosaic))
        n += 1
        time.sleep(0.05)
    if last is None:
        print("no eye pack (no face detected, or the board does not serve /api/eyepack yet)")
        sys.exit(1)
    pack, mosaic, res = last
    Image.fromarray(mosaic).save(out.replace(".png", "_raw.png"))
    json.dump(pack, open(out.replace(".png", "_pack.json"), "w"))
    print(f"packs: {n}   crop {pack['cw']}x{pack['ch']}   virtual frame {pack['vw']}x{pack['vh']}")
    for name, m in zip(("left", "right"), res["eyes"]):
        print(f"{name:5s}: iris r={m['r']:.1f}px  mp->refined shift {np.linalg.norm(m['iris'] - m['iris_mp']):.2f}px  "
              f"fit {m['fit_score']:.1f} ok={m['fit_ok']}  open={m['open']:.2f} blink={m['blink']}  off_ref={np.round(m['off_ref'], 4)}")
    _annotate(pack, mosaic, res, out)
    print("saved", out)

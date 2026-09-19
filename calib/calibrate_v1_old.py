"""Gaze calibration: coloured squares on this laptop's screen.

For every square the script records
  * pupil features from the eye camera  (GET <eye>/api/state)
  * where the square appears in the scene camera (GET <scene>/api/frame.jpg), found by
    subtracting a frame taken with a black screen just before the square appears,
then fits a mapping  pupil features -> scene-camera pixel  and reports its error on
held-out squares.

  python calibrate.py [--eye URL] [--scene URL] [--hold SECONDS] [--no-browser]
"""
import argparse
import http.server
import io
import json
import os
import random
import threading
import time
import urllib.request
import webbrowser
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw

from detect import find_square

HERE = os.path.dirname(os.path.abspath(__file__))
COLORS = ["#ff2020", "#20ff20", "#3060ff", "#ff20ff", "#20ffff", "#ffff20", "#ff8c00"]
CAL_GRID = [(x, y) for y in (0.12, 0.31, 0.5, 0.69, 0.88) for x in (0.1, 0.3, 0.5, 0.7, 0.9)]   # 5x5 = 25
VAL_PTS = [(0.2, 0.21), (0.4, 0.4), (0.6, 0.21), (0.8, 0.4), (0.5, 0.6),
           (0.2, 0.6), (0.4, 0.79), (0.6, 0.6), (0.8, 0.79)]   # 9 held-out squares, not on the grid
SQUARE = 0.16           # square side as a fraction of the screen's short side
SETTLE_S = 1.3          # ignore this long after a square appears (the eye is still moving)
BLANK_S = 0.5           # black screen between squares (a clear 'new target' cue for the eye)
TICK_S = 0.2
SCENE_HFOV_DEG = 66.0   # Camera Module 3 (standard lens); used only to quote errors in degrees

_lock = threading.Lock()
_state = {"phase": "idle", "x": 0.5, "y": 0.5, "color": "#fff", "size": SQUARE}
_started = threading.Event()
_seen = threading.Event()


def set_state(**kw):
    with _lock:
        _state.update(kw)


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path.startswith("/state"):
            with _lock:
                body = json.dumps(_state).encode()
            _seen.set()
            self._send(body, "application/json")
        else:
            with open(os.path.join(HERE, "calib.html"), "rb") as f:
                self._send(f.read(), "text/html; charset=utf-8")

    def do_POST(self):
        if self.path == "/start":
            _started.set()
        self._send(b"ok", "text/plain")

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def http_get(url, timeout=4):
    return urllib.request.urlopen(url, timeout=timeout).read()


def fetch_eye(base):
    """Return (features dict or None, frame_id)."""
    try:
        s = json.loads(http_get(base + "/api/state"))
    except Exception:  # noqa: BLE001
        return None, -1
    eyes = s.get("eyes") or {}
    if s.get("n", 0) < 2 or "left" not in eyes or "right" not in eyes:
        return None, s.get("frame_id", -1)
    f = {}
    for side in ("left", "right"):
        e = eyes[side]
        f[side + "_noff"] = e["noff"]
        f[side + "_iris"] = e["iris"]
    return f, s.get("frame_id", -1)


def fetch_scene(base):
    try:
        return Image.open(io.BytesIO(http_get(base + "/api/frame.jpg"))).convert("RGB")
    except Exception:  # noqa: BLE001
        return None


def gray(img):
    return np.asarray(img, dtype=np.float32).mean(axis=2)


def box_blur(a, k=9):
    pad = k // 2
    p = np.pad(a, pad, mode="edge")
    c = np.cumsum(np.cumsum(p, 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def find_bright(diff, min_peak=25.0):
    """Centroid of the region that got brighter. Returns (x, y, area) or None."""
    b = box_blur(diff, 9)
    peak = float(b.max())
    if peak < min_peak:
        return None
    m = b > 0.5 * peak
    area = int(m.sum())
    if area < 80 or area > 0.6 * m.size:
        return None
    ys, xs = np.nonzero(m)
    w = np.clip(b[m] - 0.5 * peak, 1e-3, None)
    return float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum()), area


def baseline_from(frames):
    return np.median(np.stack([gray(f) for f in frames]), axis=0)


def preflight(scene_url, out_dir):
    """Show a square near each screen corner and check the scene camera sees all four, not cut off."""
    print("Checking that the scene camera sees the whole screen ...", flush=True)
    ok = True
    for name, (x, y) in (("top-left", (0.06, 0.08)), ("top-right", (0.94, 0.08)),
                         ("bottom-left", (0.06, 0.92)), ("bottom-right", (0.94, 0.92))):
        set_state(phase="blank")
        time.sleep(0.7)
        base = fetch_scene(scene_url)
        set_state(phase="show", x=x, y=y, color="#20ff20")
        time.sleep(1.0)
        img = fetch_scene(scene_url)
        if img is None or base is None:
            print(f"  cannot reach the scene camera at {scene_url} - is the board on, and is the streamer running?")
            set_state(phase="blank")
            return False
        r = find_square(np.asarray(img), np.asarray(base))
        h, w = img.size[1], img.size[0]
        dbg = img.copy()
        if r is not None:
            bx, by, bw, bh = r[3]
            ImageDraw.Draw(dbg).rectangle([bx, by, bx + bw, by + bh], outline=(255, 0, 0), width=3)
        dbg.save(os.path.join(out_dir, f"preflight_{name}.jpg"), quality=80)
        if r is None:
            print(f"  {name:13s}: square NOT found - that corner is out of view (or the room is too bright)")
            ok = False
            continue
        bx, by, bw, bh = r[3]
        cut = bx <= 1 or by <= 1 or bx + bw >= w - 1 or by + bh >= h - 1
        print(f"  {name:13s}: found at ({r[0]:.0f},{r[1]:.0f})" + ("  -- touches the image edge, re-aim" if cut else ""))
        ok = ok and not cut
    set_state(phase="blank")
    if not ok:
        print("  Re-aim the scene camera so the WHOLE screen is well inside the image, then run again.")
    return ok


def collect(eye_url, scene_url, targets, hold, out_dir):
    pool = ThreadPoolExecutor(max_workers=2)
    records = []
    for i, (kind, (tx, ty)) in enumerate(targets):
        color = COLORS[i % len(COLORS)]
        # 1. black screen between squares
        set_state(phase="blank")
        time.sleep(BLANK_S)
        base_img = fetch_scene(scene_url)
        base = np.asarray(base_img) if base_img is not None else None
        # 2. show the square
        set_state(phase="show", x=tx, y=ty, color=color)
        t0 = time.time()
        samples, last_id = [], -1
        while time.time() - t0 < SETTLE_S + hold:
            tick = time.time()
            fe = pool.submit(fetch_eye, eye_url)
            fs = pool.submit(fetch_scene, scene_url)
            feats, fid = fe.result()
            img = fs.result()
            if time.time() - t0 >= SETTLE_S and img is not None:
                det = find_square(np.asarray(img), base)
                fresh = fid != last_id
                last_id = fid
                samples.append({"t": time.time() - t0, "feats": feats if fresh else None, "det": det, "img": img})
            time.sleep(max(0.0, TICK_S - (time.time() - tick)))
        dets = [s["det"] for s in samples if s["det"]]
        feats = [s["feats"] for s in samples if s["feats"]]
        rec = {"kind": kind, "screen": [tx, ty], "color": color, "n_samples": len(samples),
               "n_det": len(dets), "n_feat": len(feats)}
        if len(dets) >= 3 and len(feats) >= 4:
            rec["scene_xy"] = [float(np.median([d[0] for d in dets])), float(np.median([d[1] for d in dets]))]
            rec["feats"] = {k: np.median(np.array([f[k] for f in feats]), axis=0).tolist() for k in feats[0]}
            ok = "ok"
        else:
            ok = "FAILED"
        records.append(rec)
        print(f"  [{i + 1}/{len(targets)}] {kind:5s} screen=({tx:.2f},{ty:.2f})  eye frames {len(feats)}/{len(samples)}"
              f"  square seen {len(dets)}/{len(samples)}  {ok}", flush=True)
        # debug picture: the last scene frame with the detected centre marked
        if samples and "scene_xy" in rec:
            im = samples[-1]["img"].copy()
            d = ImageDraw.Draw(im)
            x, y = rec["scene_xy"]
            d.line([x - 15, y, x + 15, y], fill=(255, 0, 0), width=2)
            d.line([x, y - 15, x, y + 15], fill=(255, 0, 0), width=2)
            im.save(os.path.join(out_dir, f"target_{i + 1:02d}.jpg"), quality=80)
    set_state(phase="done")
    pool.shutdown()
    return records


# ---------------------------------------------------------------- fitting
def feat_vector(f, kind):
    ln, rn, li, ri = (np.array(f[k]) for k in ("left_noff", "right_noff", "left_iris", "right_iris"))
    if kind == "noff_avg":
        return (ln + rn) / 2
    if kind == "iris_avg":
        return (li + ri) / 2
    if kind == "noff_lr":
        return np.concatenate([ln, rn])
    if kind == "noff_iris":
        return np.concatenate([(ln + rn) / 2, (li + ri) / 2])
    raise ValueError(kind)


def design(z, degree):
    z = np.atleast_2d(z)
    cols = [np.ones(len(z))] + [z[:, i] for i in range(z.shape[1])]
    if degree == 2:
        d = z.shape[1]
        cols += [z[:, i] * z[:, j] for i in range(d) for j in range(i, d)]
    return np.stack(cols, axis=1)


def fit(F, Y, degree, lam=1e-3):
    mu, sd = F.mean(0), F.std(0) + 1e-9
    A = design((F - mu) / sd, degree)
    reg = lam * np.eye(A.shape[1])
    reg[0, 0] = 0
    W = np.linalg.solve(A.T @ A + reg, A.T @ Y)
    return {"mu": mu, "sd": sd, "W": W, "degree": degree}


def predict(m, F):
    return design((np.atleast_2d(F) - m["mu"]) / m["sd"], m["degree"]) @ m["W"]


def evaluate(recs, kind, degree):
    cal = [r for r in recs if r["kind"] == "cal" and "scene_xy" in r]
    val = [r for r in recs if r["kind"] == "val" and "scene_xy" in r]
    if len(cal) < 6:
        return None
    F = np.stack([feat_vector(r["feats"], kind) for r in cal])
    Y = np.array([r["scene_xy"] for r in cal])
    loo = []
    for i in range(len(cal)):
        keep = [j for j in range(len(cal)) if j != i]
        m = fit(F[keep], Y[keep], degree)
        loo.append(np.linalg.norm(predict(m, F[i]) - Y[i]))
    model = fit(F, Y, degree)
    ve = []
    for r in val:
        p = predict(model, feat_vector(r["feats"], kind))[0]
        ve.append(np.linalg.norm(p - np.array(r["scene_xy"])))
    return {"kind": kind, "degree": degree, "loo": float(np.mean(loo)),
            "val": float(np.mean(ve)) if ve else None, "model": model, "n_cal": len(cal), "n_val": len(val)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eye", default="http://10.37.108.103:8080")
    ap.add_argument("--scene", default="http://10.37.108.103:8081")
    ap.add_argument("--hold", type=float, default=2.5, help="seconds recorded per square after the settle time")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    out_dir = os.path.join(HERE, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{a.port}/"
    print("Calibration page:", url)
    if not a.no_browser:
        webbrowser.open(url)
    print("Waiting for you to click Start in the browser (put it in full screen on the screen the scene camera sees) ...")
    _started.wait()

    if not preflight(a.scene, out_dir):
        print("Fix the camera view and run again.")
        return
    order = CAL_GRID[:]
    random.shuffle(order)
    targets = [("cal", p) for p in order] + [("val", p) for p in VAL_PTS]
    print(f"Recording {len(targets)} squares (~{len(targets) * (BLANK_S + SETTLE_S + a.hold):.0f} s). Look at the centre of each.")
    recs = collect(a.eye, a.scene, targets, a.hold, out_dir)
    with open(os.path.join(out_dir, "samples.json"), "w") as f:
        json.dump(recs, f, indent=1)

    scene_w = 960.0
    ppd = scene_w / SCENE_HFOV_DEG
    print("\nModel comparison (mean error in scene-camera pixels; ~{:.0f} px = 1 degree):".format(ppd))
    print(f"  {'features':10s} deg  {'leave-one-out':>13s}  {'held-out squares':>16s}")
    best = None
    for kind in ("noff_avg", "iris_avg", "noff_lr", "noff_iris"):
        for degree in (1, 2):
            if degree == 2 and kind not in ("noff_avg", "iris_avg"):
                continue
            r = evaluate(recs, kind, degree)
            if r is None:
                print("  not enough good squares to fit (need >= 6 calibration squares).")
                return
            v = "-" if r["val"] is None else f"{r['val']:.1f} px / {r['val'] / ppd:.1f} deg"
            print(f"  {kind:10s} {degree:>3d}  {r['loo']:8.1f} px      {v:>16s}")
            score = r["val"] if r["val"] is not None else r["loo"]
            if best is None or score < best[0]:
                best = (score, r)
    r = best[1]
    print(f"\nBest: {r['kind']} degree {r['degree']}  ->  {best[0]:.1f} px (~{best[0] / ppd:.1f} deg)")
    m = r["model"]
    with open(os.path.join(out_dir, "gaze_model.json"), "w") as f:
        json.dump({"features": r["kind"], "degree": r["degree"], "mu": m["mu"].tolist(), "sd": m["sd"].tolist(),
                   "W": m["W"].tolist(), "scene_size": [960, 540], "error_px": best[0]}, f, indent=1)
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()

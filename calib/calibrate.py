"""Gaze calibration: coloured squares on this laptop's screen.

For every square the script records, about 10 times a second,
  * the pupil position from the eye camera's full-resolution eye crops (features.py: refined iris centre,
    measured against a steady reference of the eye corners, blinks flagged),
  * where the square appears in the scene camera (detect.py: a compact bright square that just appeared),
then fits a robust mapping  pupil features -> scene-camera pixel  on every usable sample (fitlib.py) and
reports its error on squares that were held out of the fit.

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
from features import EyeTracker, fetch_pack, flat
from fitlib import save_model, select_model

HERE = os.path.dirname(os.path.abspath(__file__))
COLORS = ["#ff2020", "#20ff20", "#3060ff", "#ff20ff", "#20ffff", "#ffff20", "#ff8c00"]
CAL_GRID = [(x, y) for y in (0.12, 0.31, 0.5, 0.69, 0.88) for x in (0.1, 0.3, 0.5, 0.7, 0.9)]   # 5x5 = 25
VAL_PTS = [(0.2, 0.21), (0.4, 0.4), (0.6, 0.21), (0.8, 0.4), (0.5, 0.6),
           (0.2, 0.6), (0.4, 0.79), (0.6, 0.6), (0.8, 0.79)]   # 9 held-out squares, not on the grid
SQUARE = 0.16           # square side as a fraction of the screen's short side
SETTLE_S = 1.3          # ignore this long after a square appears (the eye is still moving)
BLANK_S = 0.5           # black screen between squares (a clear 'new target' cue for the eye)
TICK_S = 0.1
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


def fetch_scene(base):
    try:
        return Image.open(io.BytesIO(http_get(base + "/api/frame.jpg"))).convert("RGB")
    except Exception:  # noqa: BLE001
        return None


def preflight(scene_url, eye_url, out_dir):
    """Check the eye camera sees a face, and that the scene camera sees all four screen corners uncut."""
    print("Checking the eye camera ...", flush=True)
    pack, _ = fetch_pack(eye_url)
    if pack is None:
        print("  no eye pack from the board: no face in front of the eye camera, or the streamer is not running.")
        return False
    print(f"  ok: eye crops {pack['cw']}x{pack['ch']} from a {pack['vw']}x{pack['vh']} frame")
    tr, good, seen = EyeTracker(), 0, 0
    t_end = time.time() + 3.0
    while time.time() < t_end:
        pk, mo = fetch_pack(eye_url)
        if pk is not None:
            seen += 1
            good += int(tr.update(pk, mo)["ok"])
        time.sleep(0.1)
    rate = good / max(seen, 1)
    print(f"  iris measurable on {100 * rate:.0f}% of {seen} frames (both eyes open, iris visible, sharp enough)")
    if rate < 0.25:
        print("  Too few usable frames. Look straight ahead with your eyes open, keep the glasses off, and make sure the "
              "eye camera is sharp with both eyes in the middle of its picture. Then run again.")
        return False
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
    pool = ThreadPoolExecutor(max_workers=3)
    tracker = EyeTracker()
    records = []

    def eye_tick():
        pack, mosaic = fetch_pack(eye_url)
        return None if pack is None else tracker.update(pack, mosaic)

    for i, (kind, (tx, ty)) in enumerate(targets):
        color = COLORS[i % len(COLORS)]
        # 1. black screen between squares; keep the tracker's eye-corner reference warm
        set_state(phase="blank")
        t_end = time.time() + BLANK_S
        while time.time() < t_end:
            tick = time.time()
            eye_tick()
            time.sleep(max(0.0, TICK_S - (time.time() - tick)))
        base_img = fetch_scene(scene_url)
        base = np.asarray(base_img) if base_img is not None else None
        # 2. show the square
        set_state(phase="show", x=tx, y=ty, color=color)
        t0 = time.time()
        samples, last_img = [], None
        while time.time() - t0 < SETTLE_S + hold:
            tick = time.time()
            fs = pool.submit(fetch_scene, scene_url)
            res = eye_tick()
            img = fs.result()
            if time.time() - t0 >= SETTLE_S and img is not None and res is not None:
                det = find_square(np.asarray(img), base)
                samples.append({"t": round(time.time() - t0, 3), "res": flat(res),
                                "det": None if det is None else [det[0], det[1]]})
                last_img = img
            time.sleep(max(0.0, TICK_S - (time.time() - tick)))
        dets = [s["det"] for s in samples if s["det"] is not None]
        n_eye_ok = sum(1 for s in samples if s["res"]["ok"])
        n_ok = sum(1 for s in samples if s["res"]["ok"] and s["det"] is not None)
        rec = {"kind": kind, "screen": [tx, ty], "color": color, "n_samples": len(samples), "n_det": len(dets),
               "n_eye_ok": n_eye_ok, "n_ok": n_ok, "samples": samples}
        if len(dets) >= 3:
            rec["scene_xy"] = [float(np.median([d[0] for d in dets])), float(np.median([d[1] for d in dets]))]
        status = "ok" if n_ok >= 5 else "FAILED"
        records.append(rec)
        print(f"  [{i + 1}/{len(targets)}] {kind:5s} screen=({tx:.2f},{ty:.2f})  usable {n_ok}/{len(samples)}"
              f"  (eyes ok {n_eye_ok}, square seen {len(dets)})  {status}", flush=True)
        if last_img is not None and "scene_xy" in rec:
            im = last_img.copy()
            d = ImageDraw.Draw(im)
            x, y = rec["scene_xy"]
            d.line([x - 15, y, x + 15, y], fill=(255, 0, 0), width=2)
            d.line([x, y - 15, x, y + 15], fill=(255, 0, 0), width=2)
            im.save(os.path.join(out_dir, f"target_{i + 1:02d}.jpg"), quality=80)
    set_state(phase="done")
    pool.shutdown()
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eye", default="http://169.254.96.94:8080")
    ap.add_argument("--scene", default="http://169.254.96.94:8081")
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

    if not preflight(a.scene, a.eye, out_dir):
        print("Fix the camera view and run again.")
        return
    order = CAL_GRID[:]
    random.shuffle(order)
    targets = [("cal", p) for p in order] + [("val", p) for p in VAL_PTS]
    print(f"Recording {len(targets)} squares (~{len(targets) * (BLANK_S + SETTLE_S + a.hold):.0f} s). "
          "Keep your head still and look at the centre of each.")
    recs = collect(a.eye, a.scene, targets, a.hold, out_dir)
    with open(os.path.join(out_dir, "samples.json"), "w") as f:
        json.dump(recs, f)

    best = select_model(recs, ppd=960.0 / SCENE_HFOV_DEG)
    if best is None:
        print("Not enough good squares to fit a model.")
        return
    r = best[1]
    print(f"\nBest: {r['kind']} degree {r['degree']}  ->  {best[0]:.1f} px (~{best[0] / (960.0 / SCENE_HFOV_DEG):.1f} deg)")
    save_model(os.path.join(out_dir, "gaze_model.json"), best)
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Coloured blocks on the laptop screen instead of coloured objects on a table.

The scene camera looks at this screen; run.py finds the blocks like any coloured object, and the block you look at
plays its note (colour -> pitch) from assets/samples. A lit screen does not look like a printed object to a camera, so the
colours have to be measured once through the camera:

  python tools/screen_colors.py show                      # just show the blocks (full screen it, F11)
  python tools/screen_colors.py tune --host 169.254.96.94 # shows one block at a time, measures how the camera sees it,
                                                          # writes config.laptop.json (colours + the QNX rig)

`tune` needs the scene camera (port 8081 on the board) to see the whole screen. Afterwards:
  python tools/synth.py &
  python run.py --source qnx:HOST --gaze qnx:HOST --config config.laptop.json --offline --no-menu
"""
import argparse
import sys
import http.server
import json
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from outer_vision.whitebal import AutoWB  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# name -> (centre x, centre y) as fractions of the screen; blocks are squares, height fraction SIZE
LAYOUT = {"red": (0.20, 0.28), "yellow": (0.50, 0.28), "green": (0.80, 0.28), "blue": (0.35, 0.66), "purple": (0.65, 0.66)}
SIZE = 0.24
# RGB shown on screen (a little below full brightness so the camera does not clip them to white)
RGB = {"red": (150, 12, 12), "yellow": (100, 88, 4), "green": (10, 105, 30), "blue": (25, 45, 165), "purple": (105, 25, 140)}

PAGE = """<!doctype html><meta charset=utf-8><title>Screen colours</title>
<style>html,body{margin:0;height:100%;background:#000;overflow:hidden;cursor:none}canvas{position:fixed;inset:0;width:100%;height:100%}</style>
<canvas id=c></canvas><script>
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
function rs(){cv.width=innerWidth*devicePixelRatio;cv.height=innerHeight*devicePixelRatio}addEventListener('resize',rs);rs();
addEventListener('click',()=>document.documentElement.requestFullscreen&&document.documentElement.requestFullscreen());
async function tick(){try{const s=await (await fetch('/state',{cache:'no-store'})).json();
 ctx.fillStyle='#000';ctx.fillRect(0,0,cv.width,cv.height);
 for(const p of s.patches){if(!s.visible.includes(p.name))continue;const side=s.size*cv.height;
  ctx.fillStyle='rgb('+p.rgb.join(',')+')';ctx.fillRect(p.x*cv.width-side/2,p.y*cv.height-side/2,side,side);}
}catch(e){}setTimeout(tick,50)}tick();
</script>"""

_state = {"visible": list(LAYOUT)}
_lock = threading.Lock()


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/state"):
            with _lock:
                body = json.dumps({"visible": _state["visible"], "size": SIZE,
                                   "patches": [{"name": n, "x": xy[0], "y": xy[1], "rgb": RGB[n]} for n, xy in LAYOUT.items()]}).encode()
            ctype = "application/json"
        else:
            body, ctype = PAGE.encode(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _open_fullscreen(url):
    """Own fullscreen browser window (so it is in front of the terminal; F11 or Alt+F4 closes it)."""
    import os
    import shutil
    import subprocess
    cands = [shutil.which("msedge"), shutil.which("chrome"),
             r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
             r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
             r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
    for c in cands:
        if c and os.path.exists(c):
            subprocess.Popen([c, "--new-window", "--start-fullscreen", "--app=" + url])
            return
    webbrowser.open(url)


def serve(port):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/"
    _open_fullscreen(url)
    return url


def show_only(names):
    with _lock:
        _state["visible"] = list(names)


def grab(scene_url):
    raw = urllib.request.urlopen(scene_url + "/api/frame.jpg", timeout=4).read()
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def measure_all(scene_url):
    """One frame with all blocks up: the 5 largest saturated, bright blobs, named by their known layout position
    (top row left to right: red, yellow, green; bottom row: blue on the left, purple on the right)."""
    show_only(list(LAYOUT))
    time.sleep(3.0)
    img = np.median(np.stack([grab(scene_url) for _ in range(5)]), axis=0).astype(np.uint8)
    img = AutoWB(ema=1.0).apply(img)      # the detector sees the picture white-balanced, so measure it that way
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = (((hsv[..., 1] > 100) & (hsv[..., 2] > 70)) | ((hsv[..., 1] > 40) & (hsv[..., 2] > 170))).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    def squareish(i):
        x, y, w, h, area = stats[i]
        return area >= 400 and 0.7 < w / h < 1.4 and area / (w * h) > 0.8
    comps = sorted([i for i in range(1, n) if squareish(i)], key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:len(LAYOUT)]
    if len(comps) < len(LAYOUT):
        print("  blobs:", [tuple(int(v) for v in stats[i][:4]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 400])
        print(f"  only {len(comps)} of {len(LAYOUT)} blocks found: the whole screen must be in the scene camera's view")
        return {}
    top = sorted([i for i in comps if cent[i][1] < np.mean([cent[j][1] for j in comps])], key=lambda i: cent[i][0])
    bot = sorted([i for i in comps if i not in top], key=lambda i: cent[i][0])
    if len(top) != 3 or len(bot) != 2:
        print("  blocks not in the expected 3 + 2 layout; is the screen upright in the picture?")
        return {}
    out = {}
    for name, i in zip(["red", "yellow", "green", "blue", "purple"], top + bot):
        x, y, w, h, area = stats[i]
        inner = img[y + int(0.25 * h):y + int(0.75 * h), x + int(0.25 * w):x + int(0.75 * w)].reshape(-1, 3)
        bgr = np.median(inner, axis=0).astype(np.uint8)
        hh = cv2.cvtColor(bgr.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
        out[name] = {"bgr": [int(v) for v in bgr], "hsv": [int(v) for v in hh], "bbox": [int(x), int(y), int(w), int(h)]}
    return out


def tune(host, scene_port, eye_port, port, out_name):
    scene_url = f"http://{host}:{scene_port}"
    print("colour page:", serve(port), "  (put it in full screen: click on it or F11)", flush=True)
    input_wait = 6
    print(f"tuning starts in {input_wait} s ...", flush=True)
    time.sleep(input_wait)
    results = measure_all(scene_url)
    for name, m in results.items():
        print(f"  {name:7s}: camera sees hsv={m['hsv']} bgr={m['bgr']}  block {m['bbox'][2]}x{m['bbox'][3]} px")
    show_only(list(LAYOUT))
    if len(results) < len(LAYOUT):
        print("too few colours measured; not writing a config")
        return
    cfg = json.loads((ROOT / "config.json").read_text())
    cfg["colors"] = {n: {"hsv": m["hsv"], "bgr": m["bgr"]} for n, m in results.items()}
    # big surfaces only (blocks / balloons), not small coloured things in the background
    cfg["detector"]["min_area_frac"], cfg["detector"]["max_area_frac"] = 0.012, 0.6
    cfg["white_balance"] = {"enabled": True}
    cfg["color_match"].update({"s_min": 110, "max_dist": 1.3, "hue_tol": 13})   # weakly saturated surfaces (walls, tables) are not colours
    cfg["selector"].update({"select_radius_frac": 0.04, "best_guess_radius_frac": 0.08})   # the nearest balloon within a few px
    cfg.setdefault("qnx", {})["host"] = host
    cfg["qnx"]["eye_port"], cfg["qnx"]["scene_port"] = eye_port, scene_port
    fit = ROOT / "rig_fit.json"
    if fit.exists():
        r = json.loads(fit.read_text())
        cfg["qnx"]["rig"].update({
            "eye_distance_mm": r["eye_distance_mm"], "scene_depth_mm": r["scene_depth_mm"], "kappa_deg": r["kappa_deg"],
            "zero_yaw_deg": r["zero_yaw_deg"], "zero_pitch_deg": r["zero_pitch_deg"], "fixed_center": r["fixed_center"]})
        print("qnx rig: loaded the fitted rig from rig_fit.json")
    (ROOT / out_name).write_text(json.dumps(cfg, indent=2))
    print("wrote", ROOT / out_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["show", "tune"])
    ap.add_argument("--host", default="169.254.96.94")
    ap.add_argument("--scene-port", type=int, default=8081)
    ap.add_argument("--eye-port", type=int, default=8080)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--out", default="config.laptop.json")
    a = ap.parse_args()
    if a.mode == "show":
        print("colour page:", serve(a.port))
        while True:
            time.sleep(1)
    tune(a.host, a.scene_port, a.eye_port, a.port, a.out)


if __name__ == "__main__":
    main()

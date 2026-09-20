"""Live gaze: draw where the user is looking on the scene camera's picture.

  python gaze_live.py [--model run_xxx/gaze_model.json] [--eye URL] [--scene URL] [--port 8766]
Then open http://127.0.0.1:8766/ . The dot is the estimate; the thin circle is the calibration error.
"""
import argparse
import glob
import http.server
import io
import json
import os
import threading
import time
import urllib.request
from collections import deque

import cv2
import numpy as np
from PIL import Image

from features import EyeTracker, fetch_pack, feature_vector
from fitlib import design, load_model
from gaze_color import color_at

HERE = os.path.dirname(os.path.abspath(__file__))
_lock = threading.Lock()
_gaze = {"p": None, "t": 0.0}          # smoothed gaze point in scene pixels
_jpeg = {"b": b"", "id": 0}
_look = {"valid": False}               # gaze point (0..1) and the colour under it, served at /gaze


def get(url, timeout=3):
    return urllib.request.urlopen(url, timeout=timeout).read()


def predict(m, res):
    z = (feature_vector(m["kind"], res) - m["mu"]) / m["sd"]
    return (design(z, m["degree"]) @ m["W"])[0]


def eye_loop(m, eye_url):
    hist = deque(maxlen=5)
    tracker = EyeTracker()
    ema = None
    last_fid = -1
    while True:
        pack, mosaic = fetch_pack(eye_url)
        if pack is None:
            time.sleep(0.1)
            continue
        if pack["fid"] == last_fid:
            time.sleep(0.03)
            continue
        last_fid = pack["fid"]
        res = tracker.update(pack, mosaic)
        if not res["ok"]:                     # blink or failed iris fit: keep the last estimate
            time.sleep(0.03)
            continue
        hist.append(predict(m, res))
        med = np.median(np.array(hist), axis=0)          # robust to a bad frame
        ema = med if ema is None else 0.6 * ema + 0.4 * med
        with _lock:
            _gaze["p"] = ema.copy()
            _gaze["t"] = time.time()
        time.sleep(0.03)


def scene_loop(m, scene_url):
    n = 0
    while True:
        try:
            frame = np.asarray(Image.open(io.BytesIO(get(scene_url + "/api/frame.jpg"))).convert("RGB"))
        except Exception:  # noqa: BLE001
            time.sleep(0.2)
            continue
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        with _lock:
            p, t = (_gaze["p"].copy() if _gaze["p"] is not None else None), _gaze["t"]
        age = time.time() - t
        if p is not None and age < 1.5:
            x, y = int(np.clip(p[0], 0, w - 1)), int(np.clip(p[1], 0, h - 1))
            name, rgb = color_at(img, x, y)          # read the colour BEFORE the overlay is drawn on it
            with _lock:
                _look.update(valid=True, x=x / w, y=y / h, w=w, h=h, color=name, rgb=rgb, stale=age > 0.4, t=time.time())
            stale = age > 0.4          # eyes briefly lost (blink / glance): show the last estimate in yellow
            col = (0, 220, 255) if stale else (0, 255, 0)
            cv2.circle(img, (x, y), int(m["err"]), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(img, (x, y), 14, col, 3, cv2.LINE_AA)
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1, cv2.LINE_AA)
        else:
            with _lock:
                _look["valid"] = False
            cv2.putText(img, "no eyes detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with _lock:
                _jpeg["b"] = buf.tobytes()
                _jpeg["id"] += 1


PAGE = b"""<!doctype html><meta charset=utf-8><title>Live gaze</title>
<style>:root{color-scheme:dark}body{margin:0;background:#111;color:#ccc;font:14px system-ui}
img{width:100%;max-width:1280px;display:block;margin:0 auto}p{text-align:center}</style>
<p>Green dot = where you are looking. White circle = expected error.</p>
<img src="/stream.mjpg" alt="live gaze">"""


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last = -1
            try:
                while True:
                    with _lock:
                        b, i = _jpeg["b"], _jpeg["id"]
                    if i == last or not b:
                        time.sleep(0.02)
                        continue
                    last = i
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(b))
                    self.wfile.write(b + b"\r\n")
            except Exception:  # noqa: BLE001
                return
        elif self.path.startswith("/gaze"):        # what the user is looking at right now (used by laptop/omni_voice.py)
            with _lock:
                b = json.dumps({**_look, "age": round(time.time() - _look.get("t", 0), 2)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif self.path.startswith("/frame"):
            with _lock:
                b = _jpeg["b"]
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--eye", default="http://169.254.96.94:8080")
    ap.add_argument("--scene", default="http://169.254.96.94:8081")
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    path = a.model or sorted(glob.glob(os.path.join(HERE, "run_*", "gaze_model.json")))[-1]
    m = load_model(path)
    print("model:", path, "features:", m["kind"], "degree:", m["degree"], "error ~%.0f px" % m["err"], flush=True)
    threading.Thread(target=eye_loop, args=(m, a.eye), daemon=True).start()
    threading.Thread(target=scene_loop, args=(m, a.scene), daemon=True).start()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"open http://127.0.0.1:{a.port}/", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

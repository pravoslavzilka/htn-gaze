"""Stand-in for calib/gaze_live.py: same /gaze and /frame endpoints, a synthetic scene, no glasses needed.

  python fake_gaze_app.py [--color red|green|blue|yellow] [--port 8766]
The gaze sits on a patch of the chosen colour. The colour is measured with the app's real classifier.
"""
import argparse
import http.server
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calib"))
from gaze_color import color_at  # noqa: E402

BGR = {"red": (40, 40, 220), "green": (60, 190, 60), "blue": (200, 90, 30), "yellow": (40, 220, 240)}
ap = argparse.ArgumentParser()
ap.add_argument("--color", default="red", choices=BGR)
ap.add_argument("--port", type=int, default=8766)
a = ap.parse_args()

img = np.full((540, 960, 3), 110, np.uint8)
cv2.circle(img, (480, 270), 90, BGR[a.color], -1)
name, rgb = color_at(img, 480, 270)
cv2.circle(img, (480, 270), 14, (0, 255, 0), 3)
JPEG = cv2.imencode(".jpg", img)[1].tobytes()


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *x):
        pass

    def do_GET(self):
        if self.path.startswith("/gaze"):
            body, ct = json.dumps({"valid": True, "x": 0.5, "y": 0.5, "w": 960, "h": 540, "color": name,
                                   "rgb": rgb, "stale": False, "age": 0.05}).encode(), "application/json"
        else:
            body, ct = JPEG, "image/jpeg"
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


print(f"fake gaze app on :{a.port}, looking at {name} {rgb}", flush=True)
http.server.ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()

#!/usr/bin/env python3
"""Run ON THE PI (Raspberry Pi OS): serve a camera as MJPEG so the Mac can process it.

  python3 tools/pi_camera_server.py --camera 0 --port 8081           # world camera
  # Mac:  python run.py --source http://<pi>.local:8081/stream --gaze udp

Needs picamera2 (preinstalled on Pi OS) + opencv. Latency over Wi-Fi is typically 50-120 ms;
use 5 GHz or Ethernet/USB-gadget networking for the demo.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from outer_vision.io import MjpegServer, PicamSource  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--camera", type=int, default=0)
ap.add_argument("--port", type=int, default=8081)
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--quality", type=int, default=80)
args = ap.parse_args()

cam = PicamSource(f"picam:{args.camera}", args.width, args.height, args.fps)
srv = MjpegServer(args.port, quality=args.quality, max_fps=args.fps)
print(f"serving camera {args.camera} at http://0.0.0.0:{args.port}/stream", flush=True)
n, t0 = 0, time.monotonic()
while True:
    frame, _, _ = cam.read()
    srv.publish(frame)
    n += 1
    if n % 300 == 0:
        print(f"{n / (time.monotonic() - t0):.1f} fps", flush=True)

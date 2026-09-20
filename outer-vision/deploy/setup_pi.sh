#!/usr/bin/env bash
# Run ON THE PI (Raspberry Pi OS) from the repo folder. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> system"
. /etc/os-release && echo "$PRETTY_NAME, $(python3 --version), $(uname -m)"
python3 -c "import picamera2" 2>/dev/null || {
  echo "picamera2 missing: sudo apt install -y python3-picamera2 (needs internet), then re-run" >&2; exit 1; }

echo "==> python env (.venv, sharing the system picamera2/numpy)"
[ -d .venv ] || python3 -m venv --system-site-packages .venv
if [ -d deploy/wheels ]; then
  .venv/bin/pip install -q --no-index --find-links deploy/wheels --no-deps "opencv-python-headless<4.11"
else
  # --no-deps: keep the system numpy that picamera2 was built against
  .venv/bin/pip install -q --no-deps "opencv-python-headless<4.11"
fi
.venv/bin/python -c "import cv2, numpy; print('opencv', cv2.__version__, '| numpy', numpy.__version__, '| aruco', hasattr(cv2.aruco, 'ArucoDetector'))"

echo "==> cameras"
if command -v rpicam-hello >/dev/null; then rpicam-hello --list-cameras 2>&1 | grep -E "^[0-9]|Available|No cameras" || true; fi
.venv/bin/python - <<'PY'
import sys, time
sys.path.insert(0, ".")
from outer_vision.io import PicamSource
import cv2
try:
    cam = PicamSource("picam:0", 1280, 720, 30)
except Exception as e:
    print(f"camera 0 FAILED: {e}"); sys.exit(0)
for _ in range(10):
    f, _, _ = cam.read()
t = time.monotonic()
for _ in range(30):
    f, _, _ = cam.read()
fps = 30 / (time.monotonic() - t)
cam.close()
import os; os.makedirs("recordings", exist_ok=True)
cv2.imwrite("recordings/pi_camera_check.jpg", f)
print(f"camera 0 OK: {f.shape[1]}x{f.shape[0]} at {fps:.0f} fps -> recordings/pi_camera_check.jpg")
PY

echo "==> tests + speed"
.venv/bin/python -m unittest discover tests 2>&1 | tail -3
.venv/bin/python run.py --source synthetic --gaze synthetic --headless --max-frames 200 --snapshot-every 199 > /dev/null
.venv/bin/python - <<'PY'
import sys, time
sys.path.insert(0, ".")
from outer_vision import config, shape_net, synthetic
from outer_vision.detector import Detector
cfg = config.load("config.json")
det = Detector(cfg, shape_net.load(cfg))
img = synthetic.render(0)
for _ in range(5): det.detect(img)
t = time.perf_counter()
for _ in range(50): det.detect(img)
print(f"pipeline: {(time.perf_counter() - t) / 50 * 1000:.1f} ms/frame on this Pi (6 objects, shape net on)")
PY
echo "==> ready. Examples:"
echo "  .venv/bin/python run.py --source picam --gaze udp --headless --stream 8080 --send 192.168.2.1:5006"
echo "  .venv/bin/python tools/pi_camera_server.py --camera 0     # stream the camera to the laptop instead"

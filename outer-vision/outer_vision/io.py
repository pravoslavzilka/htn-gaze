"""Frame sources, gaze inputs, event output, recording, calibration-marker detection."""
from __future__ import annotations

import collections
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from . import gaze_model, synthetic
from .blink import BlinkDetector


# ---------------------------------------------------------------- frame sources
def scene_url(host: str, port=8081) -> str:
    """The QNX board's scene (world) camera: camera_streamer --no-infer serves MJPEG at /stream.mjpg."""
    return f"http://{host}:{port}/stream.mjpg"


def state_url(host: str, port=8080) -> str:
    """The QNX board's eye camera: face mesh, iris and the dark-pupil fit, as JSON."""
    return f"http://{host}:{port}/api/state"


class CameraSource:
    """Live camera: int index, or any URL/string cv2.VideoCapture accepts, e.g. the QNX board's scene
    stream http://<board>:8081/stream.mjpg, or http://<pi>:8081/stream from tools/pi_camera_server.py."""
    live = True

    def __init__(self, spec, width=640, height=480, fps=30):
        self.cap = cv2.VideoCapture(int(spec) if str(spec).isdigit() else spec)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # newest frame, not a queue of stale ones
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera {spec!r} (macOS: allow camera access for your terminal in System Settings > Privacy & Security > Camera; try index 1 for an iPhone/USB camera)")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or fps
        self.idx = -1

    def read(self):
        ok, frame = self.cap.read()
        self.idx += 1
        return (frame if ok else None), time.monotonic(), self.idx


class VideoSource:
    """Recorded video. Time comes from the frame index so dwell timing matches the original run."""
    live = False

    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open video {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.idx = -1

    def read(self):
        ok, frame = self.cap.read()
        self.idx += 1
        return (frame if ok else None), self.idx / self.fps, self.idx


class SyntheticSource:
    live = False
    fps = 30.0

    def __init__(self, n_frames=None):
        self.n, self.idx = n_frames, -1

    def read(self):
        self.idx += 1
        if self.n is not None and self.idx >= self.n:
            return None, self.idx / self.fps, self.idx
        return synthetic.render(self.idx), self.idx / self.fps, self.idx


class PicamSource:
    """Raspberry Pi camera via picamera2 (Pi OS). spec: "picam" or "picam:<index>"."""
    live = True

    def __init__(self, spec: str, width=1280, height=720, fps=30):
        from picamera2 import Picamera2   # only exists on the Pi
        idx = int(spec.split(":", 1)[1]) if ":" in spec else 0
        self.cam = Picamera2(idx)
        cfg = self.cam.create_video_configuration(main={"size": (width, height), "format": "RGB888"},
                                                  controls={"FrameRate": fps})
        self.cam.configure(cfg)
        self.cam.start()
        self.fps, self.idx = fps, -1

    def read(self):
        frame = self.cam.capture_array()   # "RGB888" in picamera2 is BGR byte order, i.e. OpenCV-ready
        self.idx += 1
        return frame, time.monotonic(), self.idx

    def close(self):
        self.cam.stop()


def open_source(spec: str, cfg: dict | None = None):
    """spec: "synthetic" | "qnx[:host]" | "picam[:N]" | camera index | video file | any URL."""
    if spec == "synthetic":
        return SyntheticSource()
    if spec == "qnx" or spec.startswith("qnx:"):
        q = (cfg or {}).get("qnx", {})
        host = spec.split(":", 1)[1] if ":" in spec else q.get("host", "192.168.2.2")
        return CameraSource(scene_url(host, q.get("scene_port", 8081)))
    if spec.startswith("picam"):
        return PicamSource(spec)
    if Path(spec).is_file():
        return VideoSource(spec)
    return CameraSource(spec)


# ---------------------------------------------------------------- gaze inputs
# All gaze is normalized world-camera coordinates: x, y in [0, 1], origin top-left. See INTERFACE.md.
# Every source also answers eyes_closed() (freeze dwell) and gestures() (deliberate blinks, see blink.py).
class NoGaze:
    name = "none"

    def get(self, frame_idx):
        return None

    def eyes_closed(self, frame_idx):
        return False

    def gestures(self, frame_idx):
        return []


class MouseGaze(NoGaze):
    """Mouse position over the debug window stands in for gaze (blinks: keys 1/2/3/x in run.py)."""
    name = "mouse"

    def __init__(self):
        self.pos = None

    def on_mouse(self, event, x, y, flags, frame_wh):
        w, h = frame_wh
        self.pos = (x / w, y / h)

    def get(self, frame_idx):
        return self.pos


class UdpGaze(NoGaze):
    """Listens for gaze JSON from the inner-camera process. Stale samples count as no gaze.
    Eye state (left_closed / right_closed, else valid:false = both closed) feeds a BlinkDetector here, at
    the tracker's full sample rate, so blink timing doesn't depend on the camera frame rate."""
    name = "udp"

    def __init__(self, port: int, max_age_s: float, blink_cfg: dict):
        self.max_age = max_age_s
        self.latest = None       # (x, y, recv_monotonic)
        self.count = 0
        self.blink = BlinkDetector(blink_cfg)
        self._gestures = collections.deque()
        self._last_sample = 0.0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            data, _ = self.sock.recvfrom(4096)
            try:
                g = json.loads(data)
                now = time.monotonic()
                valid = bool(g.get("valid", True))
                if valid:
                    self.latest = (float(g["x"]), float(g["y"]), now)
                else:
                    self.latest = None
                lc, rc = g.get("left_closed"), g.get("right_closed")
                if lc is None and rc is None:
                    lc = rc = not valid           # no per-eye state: invalid gaze = both eyes closed
                elif lc is None or rc is None:
                    lc = rc = bool(lc if rc is None else rc)   # one eye tracked: can't tell a wink from a blink
                self._gestures.extend(self.blink.feed(now, bool(lc), bool(rc)))
                self._last_sample = now
                self.count += 1
            except (ValueError, KeyError, TypeError):
                pass

    def get(self, frame_idx):
        g = self.latest
        if g is None or time.monotonic() - g[2] > self.max_age:
            return None
        return g[0], g[1]

    def age_ms(self):
        g = self.latest
        return None if g is None else round((time.monotonic() - g[2]) * 1000, 1)

    def eyes_closed(self, frame_idx):
        return self.blink.closed and time.monotonic() - self._last_sample < self.max_age

    def gestures(self, frame_idx):
        out = []
        while self._gestures:
            out.append(self._gestures.popleft())
        return out


class QnxGaze(NoGaze):
    """Gaze straight from the QNX board, with no UDP hop: poll GET /api/state on the eye camera, run the
    coaxial rig model (gaze_model.py), and turn eyelid openness into the per-eye state blinks are made of.

    The board (htn-gaze, branch pupil-in-eye) runs MediaPipe Face Mesh and the dark-pupil fit in C on
    QNX 8.0 and reports, per eye, the eyelid outline, the iris ring, the MediaPipe iris centre and the
    dark-pupil centre. `state.n` is 0 or 2: either a face was found and both eyes are reported, or nothing
    is. The model prefers the dark pupil and falls back to the iris centre, exactly as the eye team's GUI
    does, so both views of the rig agree about where the wearer is looking.

    Eye state, which is the user's ONLY way to give commands, needs care here:
      * Openness has hysteresis (closed_below / open_above), so an eye hovering at the threshold can't
        chatter and split one long blink into several short ones.
      * Samples are fed to the BlinkDetector only when the board's landmarks actually CHANGED. If the
        inference thread stalls while the camera keeps running, the repeated frame would otherwise look
        like eyes held shut, and a stalled board would open menus by itself. A gap longer than
        blink.max_sample_gap_s makes the detector discard the closure in progress, which is what we want.
      * n == 0 (no face: looked away, or the tracker lost the eyes) feeds nothing at all, rather than
        guessing "both closed": a lost face is not a command.
    """
    name = "qnx"

    def __init__(self, host: str, cfg: dict):
        q = cfg.get("qnx", {})
        self.url = state_url(host, q.get("eye_port", 8080))
        self.host = host
        self.rig = gaze_model.rig_from(q.get("rig"))
        self.lid = {"closed_below": 0.15, "open_above": 0.20, "min_open_for_gaze": gaze_model.MIN_OPEN_FOR_GAZE,
                    **q.get("lid", {})}
        sm = {"median": 5, "ema": 0.45, **q.get("smooth", {})}
        self.smooth = gaze_model.Smoother(sm["median"], sm["ema"])
        self.max_age = cfg["selector"]["gaze_max_age_s"]
        self.period = 1.0 / max(1.0, float(q.get("poll_hz", 30.0)))
        self.timeout = float(q.get("timeout_s", 0.5))
        self.blink = BlinkDetector(cfg["blink"])
        self._gestures = collections.deque()
        self.latest = None            # (x, y, recv_monotonic)
        self._last_sample = 0.0
        self._closed = {"left": False, "right": False}
        self._fingerprint = None      # last landmark values seen, to spot a stalled inference thread
        self.count = 0                # fresh board samples seen; bumps once per new inference result
        self.last_open = {"left": None, "right": None}   # lid openness, for tools/qnx_bridge.py --probe
        self.last_yaw_deg = self.last_pitch_deg = 0.0    # aim, for tools/qnx_bridge.py --zero
        self.stats = {"connected": False, "error": None, "n": 0, "camera_fps": 0.0, "infer_fps": 0.0,
                      "on_image": False, "pupil_ok": 0}
        threading.Thread(target=self._loop, daemon=True).start()

    # -------------------------------------------------------------- polling
    def _fetch(self):
        with urllib.request.urlopen(self.url, timeout=self.timeout) as r:
            return json.loads(r.read())

    def _loop(self):
        while True:
            t0 = time.monotonic()
            try:
                self._handle(self._fetch(), time.monotonic())
                self.stats["connected"], self.stats["error"] = True, None
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
                self.stats["connected"] = False
                self.stats["error"] = str(e)[:120]
                self.latest = None
                self.smooth.reset()
            time.sleep(max(0.0, self.period - (time.monotonic() - t0)))

    def _handle(self, state: dict, now: float):
        eyes = state.get("eyes") or {}
        self.stats.update(n=int(state.get("n") or 0), camera_fps=float(state.get("camera_fps") or 0.0),
                          infer_fps=float(state.get("infer_fps") or 0.0))
        if not eyes:                              # no face: no gaze, and nothing to say about the eyelids
            self.latest = None
            self.smooth.reset()
            self.stats["on_image"] = False
            return

        fresh = self._is_fresh(eyes)
        _m0 = time.perf_counter()
        g = gaze_model.gaze_from_state(state, self.rig, self.lid["min_open_for_gaze"])
        self.stats["model_ms"] = round((time.perf_counter() - _m0) * 1000, 2)   # pupil/iris -> gaze model, on this laptop
        self.stats["on_image"] = bool(g["on_image"])
        self.stats["pupil_ok"] = sum(1 for s in ("left", "right")
                                     if (g["eyes"].get(s) or {}).get("pupil_ok"))
        if g["ok"]:
            self.latest = (*self.smooth.push(g.get("x_free", g["x"]), g.get("y_free", g["y"])), now)   # un-clamped: run.py clamps after the corrections
            self.last_yaw_deg, self.last_pitch_deg = g["yaw_deg"], g["pitch_deg"]
        else:
            self.latest = None
            self.smooth.reset()

        if fresh:
            for side in ("left", "right"):
                e = eyes.get(side)
                if isinstance(e, dict):
                    self.last_open[side] = gaze_model.eye_open(e)
                    self._closed[side] = self._eye_closed(side, self.last_open[side])
            self._gestures.extend(self.blink.feed(now, self._closed["left"], self._closed["right"]))
            self._last_sample = now
            self.count += 1

    def _is_fresh(self, eyes: dict) -> bool:
        """Did the board's inference actually produce a new result since the last poll?"""
        fp = tuple(tuple(eyes.get(s, {}).get(k) or ()) for s in ("left", "right")
                   for k in ("iris", "pupil", "lid"))
        if fp == self._fingerprint:
            return False
        self._fingerprint = fp
        return True

    def _eye_closed(self, side: str, open_ratio: float) -> bool:
        if open_ratio < self.lid["closed_below"]:
            return True
        if open_ratio > self.lid["open_above"]:
            return False
        return self._closed[side]                 # in between: keep the last decision (hysteresis)

    # -------------------------------------------------------------- gaze source interface
    def get(self, frame_idx):
        g = self.latest
        if g is None or time.monotonic() - g[2] > self.max_age:
            return None
        return g[0], g[1]

    def age_ms(self):
        g = self.latest
        return None if g is None else round((time.monotonic() - g[2]) * 1000, 1)

    def eyes_closed(self, frame_idx):
        return self.blink.closed and time.monotonic() - self._last_sample < self.max_age

    def gestures(self, frame_idx):
        out = []
        while self._gestures:
            out.append(self._gestures.popleft())
        return out

    @property
    def closed(self) -> dict:
        """Per-eye lid state right now, after hysteresis: what INTERFACE.md calls left_closed/right_closed."""
        return dict(self._closed)

    def health(self):
        return dict(self.stats)


class ReplayGaze(NoGaze):
    """Gaze, eye state and blink gestures from a recording's log.jsonl, matched by frame index."""
    name = "replay"

    def __init__(self, log_path):
        self.by_frame, self.closed, self.gest = {}, set(), {}
        with open(log_path) as f:
            for line in f:
                r = json.loads(line)
                if "frame" in r:
                    self.by_frame[r["frame"]] = r.get("gaze")
                    if r.get("eyes_closed"):
                        self.closed.add(r["frame"])
                    if r.get("gestures"):
                        self.gest[r["frame"]] = r["gestures"]

    def eyes_closed(self, frame_idx):
        return frame_idx in self.closed

    def gestures(self, frame_idx):
        return self.gest.get(frame_idx, [])

    def get(self, frame_idx):
        g = self.by_frame.get(frame_idx)
        return tuple(g) if g else None


class SyntheticGaze(NoGaze):
    name = "synthetic"

    def get(self, frame_idx):
        return synthetic.gaze(frame_idx)


def open_gaze(spec: str, cfg: dict):
    """spec: "none" | "mouse" | "udp" | "qnx[:host]" | "synthetic" | "replay:<log.jsonl>"."""
    if spec == "none":
        return NoGaze()
    if spec == "mouse":
        return MouseGaze()
    if spec == "udp":
        return UdpGaze(cfg["gaze_udp_port"], cfg["selector"]["gaze_max_age_s"], cfg["blink"])
    if spec == "qnx" or spec.startswith("qnx:"):
        host = spec.split(":", 1)[1] if ":" in spec else cfg.get("qnx", {}).get("host", "192.168.2.2")
        return QnxGaze(host, cfg)
    if spec == "synthetic":
        return SyntheticGaze()
    if spec.startswith("replay:"):
        return ReplayGaze(spec.split(":", 1)[1])
    raise ValueError(f"unknown gaze source {spec!r}")


# ---------------------------------------------------------------- output
class Publisher:
    """Fire-and-forget UDP JSON to the game/audio process."""

    def __init__(self, host: str, port: int):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, msg: dict):
        try:
            self.sock.sendto(json.dumps(msg, separators=(",", ":")).encode(), self.addr)
        except OSError:
            pass  # receiver not up yet; never let output stall the vision loop


class Recorder:
    """Raw (un-annotated) world frames + per-frame JSON log, so runs can be replayed through new code."""

    def __init__(self, root="recordings", fps=30.0, cfg=None):
        self.dir = Path(root) / time.strftime("%Y%m%d-%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fps, self.writer = fps, None
        self.log = open(self.dir / "log.jsonl", "w")
        self.frame = 0
        if cfg is not None:
            (self.dir / "config.json").write_text(json.dumps(cfg, indent=2))

    def write(self, frame, record: dict):
        if self.writer is None:
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(str(self.dir / "world.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        self.writer.write(frame)
        # Re-index from 0 so the log lines up with world.mp4 when replayed.
        self.log.write(json.dumps({**record, "frame": self.frame}, separators=(",", ":")) + "\n")
        self.frame += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
        self.log.close()


class MjpegServer:
    """Debug overlay as an MJPEG stream: open http://<pi>:<port>/ on the Mac. Stdlib only."""

    PAGE = (b"<html><head><title>outer-vision</title></head><body style='margin:0;background:#111'>"
            b"<img src='/stream' style='width:100%;height:auto'></body></html>")

    def __init__(self, port: int, quality=70, max_fps=15):
        self.quality, self.min_dt = quality, 1.0 / max_fps
        self._jpeg, self._last = None, 0.0
        self._cond = threading.Condition()
        srv = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.split("?")[0] != "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(srv.PAGE)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with srv._cond:
                            srv._cond.wait(timeout=2.0)
                            jpg = srv._jpeg
                        if jpg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def wants_frame(self) -> bool:
        return time.monotonic() - self._last >= self.min_dt

    def publish(self, img):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            with self._cond:
                self._jpeg, self._last = buf.tobytes(), time.monotonic()
                self._cond.notify_all()


class CalibMarker:
    """Finds ArUco markers so the inner-camera team can pair pupil positions with world positions."""

    def __init__(self, dictionary="DICT_4X4_50"):
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
        self.det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())

    def detect(self, frame):
        h, w = frame.shape[:2]
        corners, ids, _ = self.det.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if ids is None:
            return []
        return [
            {"id": int(i), "x": float(c[0][:, 0].mean() / w), "y": float(c[0][:, 1].mean() / h), "corners": c[0].tolist()}
            for c, i in zip(corners, ids.flatten())
        ]

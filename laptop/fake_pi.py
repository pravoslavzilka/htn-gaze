"""Stand-in for the Pi: sends synthetic packets so the laptop side can be built and tested without hardware.

  python fake_pi.py --run demo1 --fps 30 --seconds 20 [--host 127.0.0.1] [--slow-stage scene] [--loss 0.02]
Synthetic numbers only: never quote them as results.
"""
import argparse
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sender"))
from gaze_sender import GazeSender  # noqa: E402

OBJECTS = ["laptop", "cup", "phone", "book", None]

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=9999)
ap.add_argument("--run", default="demo1")
ap.add_argument("--fps", type=float, default=30)
ap.add_argument("--seconds", type=float, default=20)
ap.add_argument("--loss", type=float, default=0.0, help="fraction of packets silently not sent")
ap.add_argument("--slow-stage", default=None, help="add ~35 ms to this stage (cap|pupil|gaze|scene|fix)")
a = ap.parse_args()

tx = GazeSender(a.host, a.port, a.run)
t_end, i = time.time() + a.seconds, 0
while time.time() < t_end:
    tick = time.perf_counter()
    ph = i / a.fps
    obj = OBJECTS[int(ph / 3) % len(OBJECTS)]
    blink = i % 150 == 149
    with tx.frame(send=random.random() >= a.loss) as fr:
        base = {"cap": 6, "pupil": 4, "gaze": 0.3, "scene": 40, "fix": 0.5}
        if a.slow_stage:
            base[a.slow_stage] += 35
        for k, v in base.items():
            fr.pkt[f"{k}_ms"] = round(max(0.1, random.gauss(v, v * 0.1)), 2)
        fr.set(conf=0.05 if blink else round(random.uniform(0.8, 0.98), 2),
               gx=round(0.5 + 0.3 * math.sin(ph / 2), 3), gy=round(0.5 + 0.2 * math.cos(ph / 3), 3),
               obj=obj, obj_conf=round(random.uniform(0.6, 0.95), 2) if obj else None)
        if blink:
            fr.event("blink")
    i += 1
    time.sleep(max(0, 1 / a.fps - (time.perf_counter() - tick)))
print(f"sent {i} frames for run {a.run}")

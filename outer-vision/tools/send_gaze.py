#!/usr/bin/env python3
"""Reference gaze sender for the inner-camera team: shows the exact UDP message run.py expects.

  python tools/send_gaze.py --demo                    # sweeps a fake gaze point in a circle at 60 Hz
  python tools/send_gaze.py --at 0.4 0.6 --blink both # look at a point, then do a long blink (opens the menu)
  python tools/send_gaze.py --at 0.4 0.6 --blink left # ...a long LEFT wink (answers "left" in the menu)
  python tools/send_gaze.py --at 0.4 0.6 --blink double
"""
import argparse
import json
import math
import socket
import time

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=5005)
ap.add_argument("--demo", action="store_true")
ap.add_argument("--at", nargs=2, type=float, metavar=("X", "Y"), help="hold gaze at this normalized point")
ap.add_argument("--blink", choices=["left", "right", "both", "double"], help="with --at: gesture after 1 s")
args = ap.parse_args()
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send(x, y, left_closed=False, right_closed=False, conf=1.0):
    """x, y: normalized WORLD-camera coords in [0,1], origin top-left.
    left_closed/right_closed: per-eye lid state (send both if you can; they drive the blink menu).
    valid: false whenever the gaze point is unusable (both eyes closed, pupil lost)."""
    valid = not (left_closed and right_closed)
    msg = {"x": x, "y": y, "valid": valid, "left_closed": left_closed, "right_closed": right_closed,
           "conf": conf, "t": time.time()}
    sock.sendto(json.dumps(msg).encode(), (args.host, args.port))


def hold(x, y, secs, lc=False, rc=False):
    end = time.time() + secs
    while time.time() < end:
        send(x, y, lc, rc)
        time.sleep(1 / 60)


if args.demo:
    t0 = time.time()
    while True:
        a = (time.time() - t0) * 0.5
        send(0.5 + 0.3 * math.cos(a), 0.5 + 0.3 * math.sin(a))
        time.sleep(1 / 60)
elif args.at:
    x, y = args.at
    hold(x, y, 1.0)
    if args.blink == "double":
        for _ in range(2):
            hold(x, y, 0.12, True, True)
            hold(x, y, 0.25)
    elif args.blink:
        hold(x, y, 0.9, args.blink in ("left", "both"), args.blink in ("right", "both"))
    hold(x, y, 1.0)

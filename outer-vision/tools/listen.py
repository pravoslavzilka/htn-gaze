#!/usr/bin/env python3
"""Print events from run.py. Reference receiver for whoever builds the game/audio side.

  python tools/listen.py            # lock events only
  python tools/listen.py --state    # also the per-frame state stream
"""
import argparse
import json
import socket

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=5006)
ap.add_argument("--state", action="store_true")
args = ap.parse_args()
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", args.port))
print(f"listening on udp :{args.port}")
while True:
    msg = json.loads(sock.recv(65536))
    if msg["type"] == "lock":
        o = msg["object"]
        print(f"LOCK  #{o['id']:<3} {o['color']:<7} {o['shape']:<9} vol={o['volume']}  "
              f"dist={o['distance_cm']}  best_guess={msg['best_guess']}")
    elif args.state:
        print(f"state target={msg['target']} dwell={msg['dwell']:.2f} gaze={msg['gaze']} objs={len(msg['objects'])}")

#!/usr/bin/env python3
"""Print-ready ArUco marker for gaze calibration (look at it; run.py --calib-marker reports where it is).

  python tools/make_marker.py --id 0      # -> recordings/marker_0.png, print ~5 cm wide
"""
import argparse
from pathlib import Path

import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--id", type=int, default=0)
ap.add_argument("--px", type=int, default=600)
args = ap.parse_args()
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
img = cv2.aruco.generateImageMarker(d, args.id, args.px)
img = cv2.copyMakeBorder(img, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)  # quiet zone is required
Path("recordings").mkdir(exist_ok=True)
out = f"recordings/marker_{args.id}.png"
cv2.imwrite(out, img)
print(out)

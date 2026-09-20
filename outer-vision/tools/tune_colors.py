#!/usr/bin/env python3
"""Register each object's colour under the real lighting, then save to config.json.

  python tools/tune_colors.py --source 0

Press 1-N to pick a colour slot (N = however many colours are registered, 5 by default), then
click that object: its median HSV becomes the prototype.
Right half = every pixel painted with the colour it is assigned to (black = ignored).
Keys: 1-N slot | [ ] shrink/grow max_dist | s save | q quit
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from outer_vision import config, overlay  # noqa: E402
from outer_vision.detector import Detector  # noqa: E402
from outer_vision.io import open_source  # noqa: E402

WIN = "tune-colors"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()
    cfg = config.load(args.config)
    cfg["shape_net"]["enabled"] = False
    names = list(cfg["colors"])
    src = open_source(args.source, cfg)
    st = {"sel": 0, "frame": None, "det": Detector(cfg)}

    def on_click(event, x, y, *_):
        if event != cv2.EVENT_LBUTTONDOWN or st["frame"] is None:
            return
        f = st["frame"]
        x %= f.shape[1]
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        patch = hsv[max(0, y - 5):y + 6, max(0, x - 5):x + 6].reshape(-1, 3).astype(int)
        # circular median for hue (red straddles 0/180)
        hue = patch[:, 0]
        if hue.max() - hue.min() > 90:
            hue = np.where(hue < 90, hue + 180, hue)
        h = int(np.median(hue)) % 180
        s, v = (int(np.median(patch[:, 1])), int(np.median(patch[:, 2])))
        name = names[st["sel"]]
        cfg["colors"][name]["hsv"] = [h, s, v]
        cfg["colors"][name]["bgr"] = [int(c) for c in f[y, x]]
        st["det"] = Detector(cfg)
        print(f"{name} <- HSV ({h}, {s}, {v})", flush=True)
        if s < cfg["color_match"]["s_min"]:
            print(f"  warning: saturation {s} < s_min {cfg['color_match']['s_min']}; this object will be ignored", flush=True)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_click)
    while True:
        frame, _, _ = src.read()
        if frame is None:
            break
        scale = cfg["process_width"] / frame.shape[1]
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
        st["frame"] = frame
        det = st["det"]
        vis = np.hstack([frame, overlay.mask_view(det.label_map(frame), det.names, cfg)])
        slots = "  ".join(f"[{i + 1}]{n}" + ("*" if i == st["sel"] else "") for i, n in enumerate(names))
        for i, line in enumerate([slots, f"max_dist {cfg['color_match']['max_dist']:.2f}  [ ] adjust | s save | q quit"]):
            cv2.putText(vis, line, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, line, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WIN, vis)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            config.save(cfg, args.config)
            print(f"saved {args.config}", flush=True)
        if k in (ord("["), ord("]")):
            cfg["color_match"]["max_dist"] = max(0.3, cfg["color_match"]["max_dist"] + (0.1 if k == ord("]") else -0.1))
            st["det"] = Detector(cfg)
        if ord("1") <= k <= ord("9") and k - ord("1") < len(names):
            st["sel"] = k - ord("1")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

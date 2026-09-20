#!/usr/bin/env python3
"""Collect real training crops for the shape net. Two ways to get labels:

  SESSION label: only objects of ONE shape on the table, and move your head around.
    python tools/collect.py --source 0 --label cylinder        # ~2-3 min per shape, varied angles/distances
    python tools/collect.py --source 0 --label reject          # empty table: wave hands, pens, paper over it
  OMNI label: the real mixed table (plus hands, pens...), labelled afterwards in batches by OMNI.
    python tools/collect.py --source 0 --label auto            # -> data/unlabeled/
    python tools/omni_label.py                                 # -> data/real/<label>/
  then: .venv-train/bin/python tools/train_shape.py --real data/real

Works on recordings too (--source recordings/<ts>/world.mp4), so you can record once and label later.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from outer_vision import config  # noqa: E402
from outer_vision.detector import Detector, crop_for  # noqa: E402
from outer_vision.io import open_source  # noqa: E402
from outer_vision.tracker import Tracker  # noqa: E402

LABELS = ("round", "square", "cylinder", "triangle", "reject", "auto")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--label", required=True, choices=LABELS)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--out", default=None, help="default data/real (or data/unlabeled for --label auto)")
    ap.add_argument("--every", type=float, default=0.25, help="seconds between saves of the same object")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()
    cfg = config.load(args.config)
    cfg["shape_net"]["enabled"] = False   # rules/net are irrelevant here: every candidate gets the session label
    det, trk = Detector(cfg), Tracker(cfg)
    src = open_source(args.source, cfg)
    out = Path(args.out or "data/real") / args.label if args.label != "auto" else Path(args.out or "data/unlabeled")
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%H%M%S")
    last_saved, n = {}, 0
    while True:
        frame, t, idx = src.read()
        if frame is None:
            break
        scale = cfg["process_width"] / frame.shape[1]
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        tracks = trk.update(det.detect(frame), t, frame.shape[1])
        vis = frame.copy()
        for tr in tracks:
            if tr.det.partial:
                continue
            if t - last_saved.get(tr.id, -1e9) >= args.every:
                cv2.imwrite(str(out / f"{stamp}_{idx:06d}_{tr.id}.png"), crop_for(frame, tr.det.bbox, 96))
                last_saved[tr.id] = t
                n += 1
            cv2.drawContours(vis, [tr.det.contour], -1, (0, 255, 0), 2)
        if not args.headless:
            cv2.putText(vis, f"label={args.label}  saved={n}  (q to stop)", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("collect", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    print(f"saved {n} crops to {out}")


if __name__ == "__main__":
    main()

"""Measure eye-camera sharpness at several target distances.

Usage: python focus_test.py [seconds_per_step] [distances_cm ...]
Default: 10 s per step at 30 20 15 10 7 5 cm. Hold a page with text at each distance in turn,
switching every `seconds_per_step` seconds, starting when the script prints GO.
"""
import io
import os
import sys
import time
import urllib.request

import numpy as np
from PIL import Image

URL = "http://10.37.108.103:8080/api/frame.jpg"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "focus_out")


def sharpness(jpeg: bytes) -> float:
    """Variance of a Laplacian on the central crop (higher = sharper)."""
    g = np.asarray(Image.open(io.BytesIO(jpeg)).convert("L"), dtype=np.float32)
    h, w = g.shape
    g = g[h // 4 : 3 * h // 4, w // 8 : 7 * w // 8]
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def grab() -> bytes:
    return urllib.request.urlopen(URL, timeout=5).read()


def main() -> None:
    per = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    dists = [float(x) for x in sys.argv[2:]] or [30, 20, 15, 10, 7, 5]
    os.makedirs(OUT, exist_ok=True)
    print(f"Starting in 5 s. Hold text at {dists[0]:g} cm first; move to the next distance every {per:g} s.", flush=True)
    time.sleep(5)
    print("GO", flush=True)
    results = []
    for d in dists:
        print(f"-> {d:g} cm", flush=True)
        t_end = time.time() + per
        best = (-1.0, b"")
        scores = []
        while time.time() < t_end:
            try:
                j = grab()
            except Exception as e:  # noqa: BLE001
                print("grab failed:", e)
                continue
            s = sharpness(j)
            scores.append(s)
            if s > best[0]:
                best = (s, j)
            time.sleep(0.25)
        # skip the first third of the window: the user is still moving into position
        settled = scores[len(scores) // 3 :] or scores
        results.append((d, float(np.median(settled)), max(scores) if scores else 0.0))
        with open(os.path.join(OUT, f"best_{d:g}cm.jpg"), "wb") as f:
            f.write(best[1])
    print("\ndistance_cm  median_sharpness  peak_sharpness")
    for d, med, peak in results:
        print(f"{d:10g}  {med:16.1f}  {peak:14.1f}")


if __name__ == "__main__":
    main()

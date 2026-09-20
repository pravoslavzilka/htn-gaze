"""Measure detection quality of an eye-camera streamer over a fixed window.

  python ab_measure.py <base_url> <seconds> <label>

Reports how often both eyes are found, how fast the face model updates, and how steady the iris (and, on the
pupil-in-eye branch, the dark pupil) position is while the wearer keeps still. Jitter is the standard deviation of
successive differences divided by sqrt(2), so slow drifts of the eye do not count as noise. Pixels are in the 960x540 stream.
"""
import json
import sys
import time
import urllib.request

import numpy as np


def get(url, timeout=3):
    return urllib.request.urlopen(url, timeout=timeout).read()


def main():
    base, seconds, label = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    rows, last = [], None
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            s = json.loads(get(base + "/api/state"))
        except Exception:  # noqa: BLE001
            time.sleep(0.1)
            continue
        if s["frame_id"] == last:
            time.sleep(0.02)
            continue
        last = s["frame_id"]
        row = {"n": s["n"], "infer": s["infer_fps"], "W": s["width"], "H": s["height"]}
        for side in ("left", "right"):
            e = (s.get("eyes") or {}).get(side)
            if e:
                row[side] = {"iris": e["iris"], "pupil": e.get("pupil"), "pupil_ok": e.get("pupil_ok"), "pr": e.get("pupil_r"),
                             "lid": e["lid"]}
        rows.append(row)
        time.sleep(0.05)
    n = len(rows)
    both = [r for r in rows if r["n"] == 2 and "left" in r]
    print(f"== {label}: {n} frames in {seconds:.0f} s | both eyes found {100 * len(both) / max(n, 1):.0f}% "
          f"| face model {np.median([r['infer'] for r in rows]):.1f} fps (median)")
    if len(both) < 10:
        print("   too few frames with both eyes to measure steadiness")
        return
    W, H = both[0]["W"], both[0]["H"]

    def jitter(seq):
        a = np.array(seq, dtype=float)
        # only count real updates (the face model runs slower than the camera, so many frames repeat the last value)
        d = np.diff(a, axis=0)
        d = d[np.any(d != 0, axis=1)]
        return (d.std(axis=0) / np.sqrt(2)) if len(d) > 5 else np.array([np.nan, np.nan])

    for side in ("left", "right"):
        iris = [(r[side]["iris"][0] * W, r[side]["iris"][1] * H) for r in both]
        j = jitter(iris)
        lid = np.array([r[side]["lid"] for r in both]).reshape(len(both), -1, 2)
        eye_w = float(np.median((lid[:, :, 0].max(1) - lid[:, :, 0].min(1)) * W))
        eye_h = float(np.median((lid[:, :, 1].max(1) - lid[:, :, 1].min(1)) * H))
        print(f"   {side:5s}: eye {eye_w:.0f}x{eye_h:.0f} px | iris jitter x {j[0]:.2f} y {j[1]:.2f} px | iris spread (std) "
              f"x {np.std([p[0] for p in iris]):.1f} y {np.std([p[1] for p in iris]):.1f} px")
        if both[0][side].get("pupil") is not None:
            ok = [r for r in both if r[side].get("pupil_ok")]
            print(f"          dark-pupil fit ok on {100 * len(ok) / len(both):.0f}% of frames", end="")
            if len(ok) > 10:
                pup = [(r[side]["pupil"][0] * W, r[side]["pupil"][1] * H) for r in ok]
                iri = [(r[side]["iris"][0] * W, r[side]["iris"][1] * H) for r in ok]
                jp = jitter(pup)
                off = np.array(pup) - np.array(iri)
                print(f" | pupil jitter x {jp[0]:.2f} y {jp[1]:.2f} px | pupil-minus-iris mean ({off[:, 0].mean():+.1f}, {off[:, 1].mean():+.1f}) px, "
                      f"std ({off[:, 0].std():.1f}, {off[:, 1].std():.1f}) px | pupil radius {np.median([r[side]['pr'] for r in ok]) * W:.1f} px")
            else:
                print()


if __name__ == "__main__":
    main()

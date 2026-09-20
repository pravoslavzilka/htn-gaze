"""Record calibration squares together with the raw eye landmarks (/api/state) so any gaze model can be scored offline.

  python collect_geom.py [--eye URL] [--scene URL] [--hold SECONDS]

Same page, corner check and scene-camera square detection as calibrate.py. Output: run_<timestamp>/geom_samples.json
"""
import argparse
import http.server
import json
import os
import random
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import calibrate as C
from detect import find_square


def fetch_state(base):
    try:
        s = json.loads(C.http_get(base + "/api/state"))
    except Exception:  # noqa: BLE001
        return None
    eyes = {}
    for side in ("left", "right"):
        e = (s.get("eyes") or {}).get(side)
        if e:
            eyes[side] = {"center": e["center"], "iris": e["iris"], "lid": e["lid"]}
    return {"width": s["width"], "height": s["height"], "n": s["n"], "frame_id": s["frame_id"], "eyes": eyes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eye", default="http://169.254.96.94:8080")
    ap.add_argument("--scene", default="http://169.254.96.94:8081")
    ap.add_argument("--hold", type=float, default=3.0)
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()

    out_dir = os.path.join(C.HERE, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), C.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print("Calibration page:", f"http://127.0.0.1:{a.port}/")
    webbrowser.open(f"http://127.0.0.1:{a.port}/")
    print("Waiting for you to click Start (full screen, on the screen the scene camera sees) ...", flush=True)
    C._started.wait()
    if not C.preflight(a.scene, a.eye, out_dir, min_corners=3):
        print("Fix the camera view and run again.")
        return
    order = C.CAL_GRID[:]
    random.shuffle(order)
    targets = [("cal", p) for p in order] + [("val", p) for p in C.VAL_PTS]
    print(f"Recording {len(targets)} squares (~{len(targets) * (C.BLANK_S + C.SETTLE_S + a.hold):.0f} s). "
          "Head still, eyes open, look at the centre cross of each.", flush=True)
    pool = ThreadPoolExecutor(max_workers=3)
    recs = []
    for i, (kind, (tx, ty)) in enumerate(targets):
        C.set_state(phase="blank")
        t_end = time.time() + C.BLANK_S
        while time.time() < t_end:
            time.sleep(0.05)
        base_img = C.fetch_scene(a.scene)
        base = np.asarray(base_img) if base_img is not None else None
        C.set_state(phase="show", x=tx, y=ty, color=C.COLORS[i % len(C.COLORS)])
        t0, samples, last_fid = time.time(), [], -1
        while time.time() - t0 < C.SETTLE_S + a.hold:
            tick = time.time()
            fs = pool.submit(C.fetch_scene, a.scene)
            st = fetch_state(a.eye)
            img = fs.result()
            if time.time() - t0 >= C.SETTLE_S and img is not None and st is not None and st["frame_id"] != last_fid:
                last_fid = st["frame_id"]
                det = find_square(np.asarray(img), base)
                samples.append({"state": st, "det": None if det is None else [det[0], det[1]]})
            time.sleep(max(0.0, C.TICK_S - (time.time() - tick)))
        dets = [s["det"] for s in samples if s["det"] is not None]
        n_eyes = sum(1 for s in samples if s["state"]["n"] == 2)
        rec = {"kind": kind, "screen": [tx, ty], "samples": samples}
        if len(dets) >= 3:
            rec["scene_xy"] = [float(np.median([d[0] for d in dets])), float(np.median([d[1] for d in dets]))]
        recs.append(rec)
        print(f"  [{i + 1}/{len(targets)}] {kind:4s} screen=({tx:.2f},{ty:.2f})  samples {len(samples)}  eyes found {n_eyes}  "
              f"square seen {len(dets)}", flush=True)
    C.set_state(phase="done")
    with open(os.path.join(out_dir, "geom_samples.json"), "w") as f:
        json.dump(recs, f)
    print("Saved:", out_dir, flush=True)


if __name__ == "__main__":
    main()

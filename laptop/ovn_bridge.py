"""Turns the gaze app's status file into the per-frame UDP packets the receiver expects.

The teammate's run.py writes ovn_status.json ~7 times a second (what the gaze is on, where, whether the eyes are
closed). This sends one packet per new status, exactly like the Pi would, so the same receiver -> Tiger path is used.

  python ovn_bridge.py --run test1 [--host 127.0.0.1] [--port 9999]

What is measured (run.py with the timing patch, --status-file): the five stage timings (cap = frame read+resize,
pupil = pupil/iris -> gaze model on this laptop, gaze = calibration mapping, scene = colour detect + track,
fix = fixation/dwell select), confidence (0 = no gaze, 0.5 = iris fallback, 1 = dark-pupil fit on both eyes), gaze point, object looked at, blinks.
Not measured: the on-board (QNX) pupil-fit time; the board only reports its inference rate. An older run.py without
the patch sends no timings (stored as NULL, never invented) and a binary conf.
"""
import argparse
import json
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sender"))
from gaze_sender import GazeSender  # noqa: E402


def packet(st, f):
    w, h = st.get("frame_w") or 1, st.get("frame_h") or 1
    gp, tgt, closed = st.get("gaze_px"), st.get("target"), bool(st.get("eyes_closed"))
    stage, trk = st.get("stage_ms") or {}, st.get("tracker")
    if trk is not None:
        # 0 = no usable gaze (eyes closed / no face / no gaze point) = tracking LOST.
        # 0.5 = gaze from the iris-centre fallback (both eyes found, no dark-pupil fit); up to 1.0 with dark-pupil fits.
        have = gp is not None and not closed and bool(trk.get("n"))
        conf = 0.5 + 0.5 * trk.get("pupil_ok", 0) / 2 if have else 0.0
    else:                                            # older app build without a tracker: 1 if it has a gaze point
        conf = 1.0 if (gp is not None and not closed) else 0.0
    ev = (["blink"] if closed else []) + (["camera_error"] if trk is not None and not trk.get("connected", True) else [])
    return {"f": f, "t": st["t"], **{f"{k}_ms": stage.get(k) for k in ("cap", "pupil", "gaze", "scene", "fix")},
            "conf": conf, "gx": round(gp[0] / w, 4) if gp else None, "gy": round(gp[1] / h, 4) if gp else None,
            "obj": f"{tgt['color']} {tgt['shape']}" if tgt else None, "obj_conf": None, "ev": ev}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="auto", help="run id, or 'auto' = follow the dashboard's current run (New run button)")
    ap.add_argument("--dashboard", default="http://127.0.0.1:8800")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--status", default=str(Path(tempfile.gettempdir()) / "ovn_status.json"))
    a = ap.parse_args()
    run = "live" if a.run == "auto" else a.run
    tx, path, last_t, f, waiting, last_poll = GazeSender(a.host, a.port, run), Path(a.status), None, 0, False, 0.0
    print(f"bridge: {path} -> udp {a.host}:{a.port} as run {run!r}" + (" (following the dashboard)" if a.run == "auto" else ""), flush=True)
    while True:
        if a.run == "auto" and time.time() - last_poll > 1.0:          # a new run starts from frame 0
            last_poll = time.time()
            try:
                new = json.loads(urllib.request.urlopen(a.dashboard + "/api/current_run", timeout=0.5).read())["run"]
            except Exception:
                new = run                                                # dashboard down: keep the current run
            if new != run:
                run, f, tx = new, 0, GazeSender(a.host, a.port, new)
                print(f"bridge: new run {run!r}", flush=True)
        try:
            st = json.loads(path.read_text(encoding="utf-8"))
            if time.time() - st["t"] > 3:
                raise ValueError("stale")
        except (OSError, ValueError, KeyError):
            if not waiting:
                print("waiting for the gaze app status file...", flush=True)
            waiting = True
            time.sleep(0.5)
            continue
        waiting = False
        if st["t"] != last_t:                       # a new status write = a new frame
            last_t = st["t"]
            tx.send({**packet(st, f)})
            f += 1
        time.sleep(0.02)


if __name__ == "__main__":
    main()

"""Turns the gaze app's status file into the per-frame UDP packets the receiver expects.

The teammate's run.py writes ovn_status.json ~7 times a second (what the gaze is on, where, whether the eyes are
closed). This sends one packet per new status, exactly like the Pi would, so the same receiver -> Tiger path is used.

  python ovn_bridge.py --run test1 [--host 127.0.0.1] [--port 9999]

What is measured (run.py with the timing patch, --status-file): the five stage timings (cap = frame read+resize,
pupil = pupil/iris -> gaze model on this laptop, gaze = calibration mapping, scene = colour detect + track,
fix = fixation/dwell select), pupil confidence (usable pupil fits / 2 eyes), gaze point, object looked at, blinks.
Not measured: the on-board (QNX) pupil-fit time; the board only reports its inference rate. An older run.py without
the patch sends no timings (stored as NULL, never invented) and a binary conf.
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sender"))
from gaze_sender import GazeSender  # noqa: E402


def packet(st, f):
    w, h = st.get("frame_w") or 1, st.get("frame_h") or 1
    gp, tgt, closed = st.get("gaze_px"), st.get("target"), bool(st.get("eyes_closed"))
    stage, trk = st.get("stage_ms") or {}, st.get("tracker")
    if trk is not None:                              # real: how many of the two eyes have a usable pupil fit
        conf = 0.0 if closed or not trk.get("n") else trk.get("pupil_ok", 0) / 2
    else:                                            # older app build without a tracker: 1 if it has a gaze point
        conf = 1.0 if (gp is not None and not closed) else 0.0
    ev = (["blink"] if closed else []) + (["camera_error"] if trk is not None and not trk.get("connected", True) else [])
    return {"f": f, "t": st["t"], **{f"{k}_ms": stage.get(k) for k in ("cap", "pupil", "gaze", "scene", "fix")},
            "conf": conf, "gx": round(gp[0] / w, 4) if gp else None, "gy": round(gp[1] / h, 4) if gp else None,
            "obj": f"{tgt['color']} {tgt['shape']}" if tgt else None, "obj_conf": None, "ev": ev}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="live")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--status", default=str(Path(tempfile.gettempdir()) / "ovn_status.json"))
    a = ap.parse_args()
    tx, path, last_t, f, waiting = GazeSender(a.host, a.port, a.run), Path(a.status), None, 0, False
    print(f"bridge: {path} -> udp {a.host}:{a.port} as run {a.run!r}", flush=True)
    while True:
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

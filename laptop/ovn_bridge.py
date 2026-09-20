"""Turns the gaze app's status file into the per-frame UDP packets the receiver expects.

The teammate's run.py writes ovn_status.json ~7 times a second (what the gaze is on, where, whether the eyes are
closed). This sends one packet per new status, exactly like the Pi would, so the same receiver -> Tiger path is used.

  python ovn_bridge.py --run test1 [--host 127.0.0.1] [--port 9999]

What is REAL here: time, object looked at (colour + shape), gaze point, blink (eyes closed), packet cadence.
What is NOT available from the app and is therefore left EMPTY (NULL in Tiger, never faked): the five stage timings.
conf is a stand-in: 1.0 while the app has a gaze point, 0.0 while it has none (eyes closed / no pupil). It is not a
pupil-fit confidence. Latency panels stay empty until the real Pi pipeline sends timings.
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
    have_gaze = gp is not None and not closed
    return {"f": f, "t": st["t"], "cap_ms": None, "pupil_ms": None, "gaze_ms": None, "scene_ms": None, "fix_ms": None,
            "conf": 1.0 if have_gaze else 0.0,
            "gx": round(gp[0] / w, 4) if gp else None, "gy": round(gp[1] / h, 4) if gp else None,
            "obj": f"{tgt['color']} {tgt['shape']}" if tgt else None, "obj_conf": None,
            "ev": ["blink"] if closed else []}


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

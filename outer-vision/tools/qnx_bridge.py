#!/usr/bin/env python3
"""Bridge the QNX eye tracker onto the UDP gaze protocol in INTERFACE.md.

The board (htn-gaze repo, branch `pupil-in-eye`) runs MediaPipe Face Mesh and a dark-pupil fit in C on
QNX 8.0 and answers HTTP polls; this repo's documented input is one UDP JSON datagram per sample. This
tool is the adapter: poll -> coaxial rig model -> {"x","y","valid","left_closed","right_closed"} on :5005.

`run.py --gaze qnx` does the same thing in-process and needs no bridge. Use this one when something ELSE
also wants the gaze (the game, tools/listen.py, a second machine), or to keep the UDP path exercised.

  python tools/qnx_bridge.py                                 # board from config.json -> 127.0.0.1:5005
  python tools/qnx_bridge.py --board 192.168.2.2 --host 192.168.2.1
  python tools/qnx_bridge.py --probe                         # print live gaze + eyelid openness, send nothing
  python tools/qnx_bridge.py --zero                          # look at the centre of the scene: prints rig zeros

Only samples the board has actually recomputed are sent. A stalled inference thread therefore shows up as
a gap in the stream, which the blink detector treats as a dropout, instead of looking like held-shut eyes.
"""
import argparse
import json
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from outer_vision import config  # noqa: E402
from outer_vision.io import QnxGaze  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--board", default=None, help="QNX board IP (default: config.json -> qnx.host)")
    ap.add_argument("--host", default="127.0.0.1", help="where run.py is listening")
    ap.add_argument("--port", type=int, default=None, help="default: config.json -> gaze_udp_port")
    ap.add_argument("--probe", action="store_true", help="print gaze and eyelid openness instead of sending")
    ap.add_argument("--zero", type=float, nargs="?", const=3.0, default=None, metavar="SECONDS",
                    help="stare at the centre of the scene camera's view for this long; prints the "
                         "zero_yaw_deg / zero_pitch_deg to put in config.json")
    args = ap.parse_args()

    cfg = config.load(args.config)
    board = args.board or cfg["qnx"]["host"]
    port = args.port or cfg["gaze_udp_port"]
    if args.zero is not None:                       # measure the residual aim, so don't subtract it yet
        cfg["qnx"]["rig"]["zero_yaw_deg"] = cfg["qnx"]["rig"]["zero_pitch_deg"] = 0.0

    gaze = QnxGaze(board, cfg)
    print(f"[qnx] polling http://{board}:{cfg['qnx']['eye_port']}/api/state at {cfg['qnx']['poll_hz']:.0f} Hz",
          flush=True)

    if args.zero is not None:
        return zero(gaze, args.zero)
    if args.probe:
        return probe(gaze)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[qnx] -> {args.host}:{port}   (Ctrl-C to stop)", flush=True)
    seen, sent, last_report = -1, 0, time.monotonic()
    while True:
        if gaze.count != seen:                      # one datagram per fresh board sample, never per poll
            seen = gaze.count
            g = gaze.get(0)
            c = gaze.closed
            sock.sendto(json.dumps({
                "x": None if g is None else round(g[0], 4),
                "y": None if g is None else round(g[1], 4),
                "valid": g is not None,
                "left_closed": c["left"], "right_closed": c["right"],
                "conf": round(gaze.health()["pupil_ok"] / 2, 2),   # both eyes' dark-pupil fits succeeded
                "t": time.time(),
            }, separators=(",", ":")).encode(), (args.host, port))
            sent += 1
        now = time.monotonic()
        if now - last_report >= 5.0:
            h = gaze.health()
            print(f"[qnx] {'up' if h['connected'] else 'DOWN: ' + str(h['error'])}  "
                  f"camera {h['camera_fps']:.0f} fps, infer {h['infer_fps']:.0f} fps, eyes {h['n']}, "
                  f"pupil fits {h['pupil_ok']}/2  |  sent {sent / (now - last_report):.0f}/s", flush=True)
            sent, last_report = 0, now
        time.sleep(0.002)


def probe(gaze: QnxGaze):
    """Live numbers for setting qnx.lid thresholds: hold your eyes open, then closed, and watch `open`."""
    print("  gaze          eyes  open L/R      closed L/R   pupil  board", flush=True)
    while True:
        g, h, c = gaze.get(0), gaze.health(), gaze.closed
        opens = gaze.last_open
        pos = "   --,--   " if g is None else f" {g[0]:.3f},{g[1]:.3f} "
        ol = "  -  " if opens["left"] is None else f"{opens['left']:.3f}"
        orr = "  -  " if opens["right"] is None else f"{opens['right']:.3f}"
        print(f"\r {pos}  n={h['n']}  {ol} {orr}   "
              f"{'C' if c['left'] else '.'} {'C' if c['right'] else '.'}        "
              f"{h['pupil_ok']}/2   {'up' if h['connected'] else 'DOWN'}  {'(off image)' if not h['on_image'] else '           '}",
              end="", flush=True)
        time.sleep(0.1)


def zero(gaze: QnxGaze, seconds: float):
    """Average the residual yaw/pitch while the wearer stares at the centre of the scene camera's view."""
    print(f"[qnx] look at the CENTRE of the scene camera's view and hold still for {seconds:.0f}s...", flush=True)
    rig = gaze.rig
    end, yaws, pitches = time.monotonic() + seconds, [], []
    seen = -1
    while time.monotonic() < end:
        if gaze.count != seen:
            seen = gaze.count
            g = gaze.get(0)
            if g is not None:
                yaws.append(gaze.last_yaw_deg)
                pitches.append(gaze.last_pitch_deg)
        time.sleep(0.005)
    if len(yaws) < 10:
        print(f"[qnx] only {len(yaws)} samples; is the board up and are your eyes open?", flush=True)
        return 1
    yaws.sort()
    pitches.sort()
    y, p = yaws[len(yaws) // 2], pitches[len(pitches) // 2]
    print(f"\n[qnx] {len(yaws)} samples. Put this in config.json -> qnx.rig:\n"
          f'    "zero_yaw_deg": {y + rig["zero_yaw_deg"]:.2f},\n'
          f'    "zero_pitch_deg": {p + rig["zero_pitch_deg"]:.2f},', flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        print()

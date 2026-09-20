"""Pi-side UDP packet sender. Stdlib only: no DB driver, no Sentry, no cloud calls.

Fire-and-forget: send() never blocks on the laptop and never raises, so a dead receiver cannot stall
the vision loop.

    tx = GazeSender("192.168.1.50", run="demo1")
    with tx.frame() as fr:                     # frame counter and t are handled for you
        with fr.stage("cap"):   ...            # each stage timed with time.perf_counter()
        with fr.stage("pupil"): ...
        fr.set(conf=0.9, gx=0.6, gy=0.4, obj="laptop", obj_conf=0.8)
        fr.event("blink")
"""
import contextlib
import json
import socket
import time

import pi_profiling

STAGES = ("cap", "pupil", "gaze", "scene", "fix")


class Frame:
    def __init__(self, f):
        self.pkt = {"f": f, "t": round(time.monotonic(), 3), **{f"{s}_ms": 0.0 for s in STAGES},
                    "conf": 0.0, "gx": None, "gy": None, "obj": None, "obj_conf": None, "ev": []}

    @contextlib.contextmanager
    def stage(self, name):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.pkt[f"{name}_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    def set(self, **kw):
        self.pkt.update(kw)

    def event(self, name):
        self.pkt["ev"].append(name)


class GazeSender:
    def __init__(self, host, port=9999, run="run"):
        self.addr, self.run, self.f = (host, int(port)), run, 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.prof = pi_profiling.start(run)      # None unless PROFILE=1 (then sentry_sdk is imported, otherwise never)

    @contextlib.contextmanager
    def frame(self, send=True):
        """send=False still advances the frame counter (used by tests to simulate packet loss)."""
        fr = Frame(self.f)
        self.f += 1
        with (self.prof.frame() if self.prof else contextlib.nullcontext()):
            try:
                yield fr
            finally:
                if send:
                    self.send(fr.pkt)

    def close(self):
        if self.prof:
            self.prof.close()

    def send(self, pkt):
        pkt = {"run": self.run, **pkt}
        try:
            self.sock.sendto(json.dumps(pkt, separators=(",", ":")).encode(), self.addr)
        except OSError:
            pass  # buffer full / network down: drop the packet, keep the pipeline moving

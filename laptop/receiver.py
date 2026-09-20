"""UDP receiver: one packet per Pi frame in, Tiger rows and derived health metrics out.

Derived here (the Pi sends none of these): receive time, tracking_lost, packet loss, FPS.
Tolerates a Pi that disconnects, restarts (frame counter resets) or sends garbage.
"""
import collections
import json
import logging
import socket
import threading
import time
from datetime import datetime, timezone

import config

log = logging.getLogger("receiver")
MS_FIELDS = ("cap_ms", "pupil_ms", "gaze_ms", "scene_ms", "fix_ms")


class RunStats:
    def __init__(self):
        self.last_f = None
        self.received = self.lost = self.tracking_lost = 0
        self.ts = collections.deque(maxlen=60)          # Pi monotonic t of recent packets
        self.last_rx = time.monotonic()

    def fps(self):
        if len(self.ts) > 1 and self.ts[-1] > self.ts[0]:
            return (len(self.ts) - 1) / (self.ts[-1] - self.ts[0])
        return None


class Receiver:
    def __init__(self, sink=None, on_frame=None, lost_conf=None):
        self.sink, self.on_frame = sink, on_frame       # sink.add(row); on_frame(pkt, derived) for Sentry etc.
        self.lost_conf = config.TRACKING_LOST_CONF if lost_conf is None else lost_conf
        self.runs = {}
        self.lock = threading.Lock()
        self.bad_packets = 0
        self._stop = threading.Event()

    def handle(self, data, now=None):
        """Process one datagram. Returns the derived dict, or None if the packet was unusable."""
        try:
            p = json.loads(data)
            run, f = str(p["run"]), int(p["f"])
        except (ValueError, KeyError, TypeError):
            self.bad_packets += 1
            return None
        wall = now if now is not None else time.time()
        with self.lock:
            s = self.runs.setdefault(run, RunStats())
            gap = 0
            if s.last_f is not None:
                if f > s.last_f:
                    gap = f - s.last_f - 1
                elif s.last_f - f > 100:               # counter went back a lot: Pi restarted
                    s.ts.clear()
                else:
                    return None                        # duplicate / reordered: ignore
            s.last_f, s.last_rx = f, time.monotonic()
            s.received += 1
            s.lost += gap
            if isinstance(p.get("t"), (int, float)):
                s.ts.append(float(p["t"]))
            conf = p.get("conf")
            lost = conf is None or conf < self.lost_conf
            s.tracking_lost += lost
            fps = s.fps()
        total = sum(float(p.get(k) or 0) for k in MS_FIELDS)
        d = {"run": run, "frame": f, "rx": wall, "total_ms": total, "tracking_lost": lost,
             "lost_packets": gap, "fps": fps}
        if self.sink:
            self.sink.add((datetime.fromtimestamp(wall, timezone.utc), run, f,
                           *(p.get(k) for k in MS_FIELDS), conf, p.get("gx"), p.get("gy"),
                           p.get("obj"), p.get("obj_conf")))
        if self.on_frame:
            try:
                self.on_frame(p, d)
            except Exception:
                log.exception("on_frame hook failed")
        return d

    def snapshot(self):
        with self.lock:
            out = {}
            for run, s in self.runs.items():
                n = s.received + s.lost
                out[run] = {"received": s.received, "lost": s.lost, "loss_rate": s.lost / n if n else 0,
                            "tracking_lost_rate": s.tracking_lost / s.received if s.received else 0,
                            "fps": s.fps(), "age_s": time.monotonic() - s.last_rx}
            return out

    def reset(self):
        """Forget all runs (demo reset between back-to-back runs)."""
        with self.lock:
            self.runs.clear()

    def serve(self, host=None, port=None):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        sock.bind((host or config.UDP_HOST, port or config.UDP_PORT))
        sock.settimeout(0.5)
        log.info("listening on udp %s", sock.getsockname())
        while not self._stop.is_set():
            try:
                data, _ = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError as e:                       # e.g. Windows ICMP reset when the Pi vanishes
                log.debug("recv: %s", e)
                continue
            self.handle(data)

    def stop(self):
        self._stop.set()


def _report(rx, tiger):
    while True:
        time.sleep(5)
        for run, s in rx.snapshot().items():
            log.info("run=%s rx=%d lost=%d fps=%s trk_lost=%.0f%% | tiger ok=%d buffered=%d fail=%d",
                     run, s["received"], s["lost"], s["fps"] and round(s["fps"], 1),
                     s["tracking_lost_rate"] * 100, tiger.inserted, len(tiger.buf), tiger.failures)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from tiger import TigerWriter
    tiger = TigerWriter(config.TIGER_DSN, config.FLUSH_SECONDS, config.MAX_BUFFER_ROWS).start()
    rx = Receiver(sink=tiger)
    threading.Thread(target=_report, args=(rx, tiger), daemon=True).start()
    try:
        rx.serve()
    except KeyboardInterrupt:
        pass
    finally:
        rx.stop()
        tiger.stop()


if __name__ == "__main__":
    main()

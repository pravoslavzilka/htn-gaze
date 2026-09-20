import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from receiver import Receiver  # noqa: E402
from tiger import TigerWriter  # noqa: E402


def pkt(f, t=None, conf=0.9, run="r"):
    return json.dumps({"run": run, "f": f, "t": f / 30 if t is None else t, "cap_ms": 5, "pupil_ms": 4,
                       "gaze_ms": 1, "scene_ms": 40, "fix_ms": 0.5, "conf": conf, "gx": .5, "gy": .5,
                       "obj": "cup", "obj_conf": .8}).encode()


class Sink:
    def __init__(self):
        self.rows = []

    def add(self, r):
        self.rows.append(r)


class ReceiverTests(unittest.TestCase):
    def test_derived_fields(self):
        s = Sink()
        rx = Receiver(sink=s, lost_conf=0.5)
        for f in range(31):
            d = rx.handle(pkt(f))
        self.assertAlmostEqual(d["fps"], 30, places=3)
        self.assertAlmostEqual(d["total_ms"], 50.5)
        self.assertEqual(len(s.rows), 31)
        self.assertTrue(rx.handle(pkt(31, conf=0.1))["tracking_lost"])

    def test_packet_loss_and_restart(self):
        rx = Receiver()
        for f in (0, 1, 5, 6):
            rx.handle(pkt(f))
        self.assertEqual(rx.snapshot()["r"]["lost"], 3)
        self.assertIsNone(rx.handle(pkt(6)))            # duplicate ignored
        for f in (500, 501):
            rx.handle(pkt(f))
        self.assertIsNotNone(rx.handle(pkt(0)))         # counter reset = Pi restart, not loss
        self.assertEqual(rx.snapshot()["r"]["lost"], 3 + 493)

    def test_garbage_is_survived(self):
        rx = Receiver()
        for junk in (b"", b"\xff\xfe", b"{}", b"[1]", b'{"run":"a","f":"x"}'):
            self.assertIsNone(rx.handle(junk))
        self.assertEqual(rx.bad_packets, 5)

    def test_hook_failure_does_not_break(self):
        def boom(p, d):
            raise RuntimeError("x")
        self.assertIsNotNone(Receiver(on_frame=boom).handle(pkt(0)))


class TigerTests(unittest.TestCase):
    def make(self, fail):
        class Cur:
            def __init__(s, c): s.c = c
            def __enter__(s): return s
            def __exit__(s, *a): pass
            def executemany(s, q, rows):
                if s.c.fail:
                    raise ConnectionError("down")
                s.c.rows += rows

        class Conn:
            rows = []
            def __init__(s): s.fail = fail[0]
            def cursor(s): return Cur(s)
            def close(s): pass
        conn = Conn()
        return conn, TigerWriter("dsn", connect=lambda: conn, max_rows=5)

    def test_outage_buffers_then_recovers(self):
        fail = [True]
        conn, w = self.make(fail)
        for i in range(8):
            w.add((i,))
        self.assertEqual(w.dropped, 3)                  # bounded buffer
        w.flush()                                       # outage: no crash, rows kept
        self.assertEqual((w.failures, len(w.buf)), (1, 5))
        conn.fail, w.next_try = False, 0
        w.flush()
        self.assertEqual((len(conn.rows), len(w.buf), w.inserted), (5, 0, 5))

    def test_disabled_without_dsn(self):
        w = TigerWriter(None).start()
        w.add((1,))
        self.assertEqual(len(w.buf), 0)


if __name__ == "__main__":
    unittest.main()

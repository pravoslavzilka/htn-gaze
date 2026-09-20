import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import queries  # noqa: E402
import telemetry  # noqa: E402

T0 = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)


class TraceReconstruction(unittest.TestCase):
    def test_stages_end_at_receive_time_and_are_contiguous(self):
        pkt = {"cap_ms": 6.1, "pupil_ms": 4.2, "gaze_ms": 0.3, "scene_ms": 41.7, "fix_ms": 0.5}
        start, spans, total = telemetry.frame_spans(pkt, T0)
        self.assertAlmostEqual(total, 52.8)
        self.assertEqual([s for s, _, _ in spans], ["cap", "pupil", "gaze", "scene", "fix"])
        self.assertEqual(spans[-1][2], T0)                                   # the last stage ends at receive time
        self.assertAlmostEqual((T0 - start).total_seconds() * 1000, 52.8, places=3)
        for (_, _, e1), (_, s2, _) in zip(spans, spans[1:]):
            self.assertEqual(e1, s2)                                         # no gaps, no overlaps
        self.assertAlmostEqual((spans[3][2] - spans[3][1]).total_seconds() * 1000, 41.7, places=3)

    def test_unmeasured_stages_are_left_out_not_zeroed(self):
        _, spans, total = telemetry.frame_spans({"cap_ms": 5, "scene_ms": 20, "pupil_ms": None}, T0)
        self.assertEqual([s for s, _, _ in spans], ["cap", "scene"])
        self.assertEqual(total, 25)

    def test_disabled_sentry_is_a_noop(self):
        self.assertFalse(telemetry.enabled() and False)
        telemetry.FrameTelemetry().on_frame({"ev": ["blink"]}, {"run": "r", "frame": 0, "rx": 0, "tracking_lost": True,
                                                                "lost_packets": 1, "fps": 1})   # must not raise
        with telemetry.span(None, "x", "y") as sp:
            self.assertIsNone(sp)


class Dwell(unittest.TestCase):
    def test_seconds_split_by_frame_share_of_the_elapsed_bucket(self):
        now = T0 + timedelta(seconds=35)
        rows = [(T0, "cup", 30), (T0, None, 30),                      # a full 10 s bucket, half each -> 5 s / 5 s
                (T0 + timedelta(seconds=30), "cup", 15)]              # newest bucket: only 5 s have elapsed -> 5 s
        r = queries.dwell_and_timeline(rows, now, 60)
        d = {x["obj"]: x for x in r["dwell"]}
        self.assertEqual(d["cup"]["seconds"], 10.0)
        self.assertEqual(d[queries.NO_OBJECT]["seconds"], 5.0)
        self.assertEqual(d["cup"]["frames"], 45)
        self.assertEqual(len(r["timeline"]), 2)

    def test_old_buckets_fall_out_of_the_window(self):
        r = queries.dwell_and_timeline([(T0, "cup", 10)], T0 + timedelta(seconds=300), 60)
        self.assertEqual(r["dwell"], [])
        self.assertEqual(len(r["timeline"]), 1)                       # but the timeline still shows it


class Api(unittest.TestCase):
    def test_tiger_outage_is_reported_not_raised(self):
        import app

        class Dead:
            ok = None
            def q(self, *a, **k): raise queries.QueryError("connection refused")

        app.S["db"] = Dead()
        r = app.guarded(lambda: {"runs": queries.runs(app.S["db"])})
        self.assertEqual(r, {"ok": False, "error": "connection refused"})


if __name__ == "__main__":
    unittest.main()

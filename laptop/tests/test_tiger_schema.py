import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ovn_bridge  # noqa: E402
from tiger import TigerWriter, schema_statements  # noqa: E402


class SchemaTests(unittest.TestCase):
    def test_statements_are_split_and_the_aggregate_is_real_time(self):
        st = schema_statements()
        self.assertEqual(len(st), 11)                            # incl. the sound_library table, trigger function and trigger
        cagg = [s for s in st if "timescaledb.continuous" in s]
        self.assertEqual(len(cagg), 1)                       # alone: cannot share a transaction with anything else
        self.assertIn("materialized_only = false", cagg[0])
        self.assertIn("time_bucket('10 seconds'", cagg[0])
        self.assertTrue(all(not s.startswith("--") for s in st))

    def test_omni_rows_buffered_and_flushed(self):
        rows = []

        class Cur:
            def __enter__(s): return s
            def __exit__(s, *a): pass
            def executemany(s, q, r): rows.append((q.split()[2], list(r)))

        class Conn:
            def cursor(s): return Cur()

        w = TigerWriter("dsn", connect=lambda: Conn())
        w.add_omni(("t", "r", "red round", "what is this", "red", 12.0))
        w.flush()
        self.assertEqual(rows[0][0], "omni_interactions")
        self.assertEqual(w.omni_inserted, 1)


class BridgeTests(unittest.TestCase):
    ST = {"t": 5.0, "frame_w": 640, "frame_h": 360, "gaze_px": [320, 180], "eyes_closed": False,
          "target": {"color": "red", "shape": "square"}}

    def test_old_app_build_has_no_timings_and_they_stay_null(self):
        p = ovn_bridge.packet(self.ST, 3)
        self.assertEqual((p["obj"], p["gx"], p["gy"], p["conf"], p["ev"]), ("red square", 0.5, 0.5, 1.0, []))
        self.assertIsNone(p["scene_ms"])
        p = ovn_bridge.packet({**self.ST, "gaze_px": None, "target": None, "eyes_closed": True}, 4)
        self.assertEqual((p["obj"], p["conf"], p["ev"]), (None, 0.0, ["blink"]))

    def test_patched_app_reports_real_timings_and_pupil_confidence(self):
        st = {**self.ST, "stage_ms": {"cap": 6.1, "pupil": 0.4, "gaze": 0.3, "scene": 41.7, "fix": 0.5},
              "tracker": {"connected": True, "n": 2, "pupil_ok": 1}}
        p = ovn_bridge.packet(st, 9)
        self.assertEqual((p["cap_ms"], p["scene_ms"], p["conf"]), (6.1, 41.7, 0.75))    # gaze + one dark-pupil fit
        st["tracker"] = {"connected": True, "n": 2, "pupil_ok": 0}
        self.assertEqual(ovn_bridge.packet(st, 11)["conf"], 0.5)                        # iris fallback: tracked, not lost
        self.assertEqual(ovn_bridge.packet({**st, "gaze_px": None}, 12)["conf"], 0.0)   # no gaze at all: lost
        st["tracker"] = {"connected": False, "n": 0, "pupil_ok": 2}
        p = ovn_bridge.packet(st, 10)
        self.assertEqual((p["conf"], p["ev"]), (0.0, ["camera_error"]))


class Drift(unittest.TestCase):
    @staticmethod
    def feed(det, dists, t0=100.0, dt=0.15, w=640):
        fired = []
        for i, d in enumerate(dists):
            fired.append(det.update({"t": t0 + i * dt, "frame_w": w, "eyes_closed": False, "target": {"dist_px": d}}))
        return fired

    def test_sustained_miss_fires_once_then_is_quiet(self):
        det = ovn_bridge.DriftDetector()
        fired = self.feed(det, [70] * 60)                       # 9 s of the gaze landing ~70 px off its object
        self.assertEqual(sum(fired), 1)

    def test_good_tracking_never_fires_and_neither_do_sparse_samples(self):
        self.assertEqual(sum(self.feed(ovn_bridge.DriftDetector(), [3, 10, 0, 25, 8] * 20)), 0)
        self.assertEqual(sum(self.feed(ovn_bridge.DriftDetector(), [90] * 5)), 0)          # too few samples to say

    def test_one_outlier_burst_is_not_drift_and_the_median_decides(self):
        self.assertEqual(sum(self.feed(ovn_bridge.DriftDetector(), ([5] * 8 + [200] * 3) * 6)), 0)

    def test_threshold_scales_with_frame_width(self):
        self.assertEqual(sum(self.feed(ovn_bridge.DriftDetector(), [70] * 30, w=1280)), 0)  # 70 px at 1280 wide = 35 px at 640

    def test_the_event_lands_in_the_packet(self):
        self.assertIn("calib_drift", ovn_bridge.packet(BridgeTests.ST, 1, drift=True)["ev"])
        self.assertNotIn("calib_drift", ovn_bridge.packet(BridgeTests.ST, 1)["ev"])


if __name__ == "__main__":
    unittest.main()

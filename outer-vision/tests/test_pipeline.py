"""Run: .venv/bin/python -m unittest discover tests -v"""
import math
import random
import unittest

import threading
import time
from pathlib import Path

import cv2
import numpy as np

from outer_vision import config, shape_net, synthetic
from outer_vision.detector import NONE, Detector, build_color_lut
from outer_vision.io import CameraSource, MjpegServer
from outer_vision.selector import Selector
from outer_vision.tracker import Tracker, estimate_depth, reference_sizes

CFG = config.load(None)


# A triangle fills only ~0.3 of its bounding box against ~0.79 for a ball, so at the same nominal size it
# is the first shape to fall under detector.min_area_frac. Below ~34 px it really is a handful of pixels,
# and not detecting it is correct; give triangles that floor rather than detecting around it.
MIN_SIZE = {"triangle": 34}


def random_scene(seed, n=5):
    """n=5, not 6: with the spacing rule below, six objects only fit in the 480x290 placement area for a
    lucky draw of sizes, so the rejection sampler used to spin for seconds on unlucky seeds."""
    rnd = random.Random(seed)
    objs = []
    for _ in range(4000):         # rejection sampling: bounded so a bad constraint fails loudly, not slowly
        shape = rnd.choice(synthetic.SHAPES)
        o = (rnd.choice(list(synthetic.BGR)), shape, rnd.randint(80, 560), rnd.randint(150, 440),
             max(rnd.randint(26, 56), MIN_SIZE.get(shape, 0)))
        if all(math.hypot(o[2] - p[2], o[3] - p[3]) > 1.6 * (o[4] + p[4]) + 30 for p in objs):
            objs.append(o)
            if len(objs) == n:
                return objs
    raise AssertionError(f"could not place {n} objects for seed {seed}: the spacing rule is too tight")


def scene_accuracy(det, n_scenes=50):
    total = correct = missed = 0
    for seed in range(n_scenes):
        objs = random_scene(seed)
        img = synthetic.render(seed, objs, seed=seed, head_motion=False)
        dets = det.detect(img)
        for o in objs:
            color, shape, _, _, s = o
            x, y = synthetic.object_center(o)
            total += 1
            cands = [d for d in dets if d.color == color and math.hypot(d.cx - x, d.cy - y) < s * 1.2]
            if not cands:
                missed += 1
                continue
            correct += min(cands, key=lambda d: math.hypot(d.cx - x, d.cy - y)).shape == shape
    return total, correct, missed


class TestDetector(unittest.TestCase):
    def test_rules_random_scenes(self):
        cfg = config.load(None)
        total, correct, missed = scene_accuracy(Detector(cfg))
        print(f"\n  rules: shape accuracy {correct}/{total}, missed {missed}")
        self.assertEqual(missed, 0)
        self.assertGreaterEqual(correct / total, 0.85)

    @unittest.skipUnless(Path("models/shape/labels.json").exists(), "no trained shape model")
    def test_net_random_scenes(self):
        cfg = config.load(None)
        total, correct, missed = scene_accuracy(Detector(cfg, shape_net.load(cfg)))
        print(f"\n  shape net: shape accuracy {correct}/{total}, missed {missed}")
        self.assertEqual(missed, 0)
        self.assertGreaterEqual(correct / total, 0.97)

    @unittest.skipUnless(Path("models/shape/labels.json").exists(), "no trained shape model")
    def test_net_rejects_hands_and_scraps(self):
        net = shape_net.load(config.load(None))
        rng = np.random.default_rng(123)
        crops = [synthetic.random_crop_sample(rng, "reject") for _ in range(200)]
        rej = net.labels.index("reject")
        rate = float((net.probs(crops).argmax(1) == rej).mean())
        print(f"\n  distractor reject rate {rate:.1%}")
        self.assertGreaterEqual(rate, 0.95)

    def test_grey_table_is_empty(self):
        img = synthetic.render(0, objects=[])
        self.assertEqual(Detector(config.load(None)).detect(img), [])

    def test_color_lut(self):
        cfg = config.load(None)
        names, lut = build_color_lut(cfg["colors"], cfg["color_match"])
        for i, n in enumerate(names):
            h, s, _ = cfg["colors"][n]["hsv"]
            self.assertEqual(lut[h, s], i, n)             # each prototype maps to itself
        self.assertTrue((lut[:, :cfg["color_match"]["s_min"]] == NONE).all())   # grey is never a colour
        self.assertEqual(lut[178, 200], names.index("red"))                     # hue wraps around 180


class TestNetworkCamera(unittest.TestCase):
    def test_mjpeg_roundtrip(self):
        """Pi -> Mac path: MjpegServer on one side, cv2.VideoCapture(URL) on the other."""
        srv = MjpegServer(8765, quality=90, max_fps=100)
        ref = synthetic.render(0)
        stop = threading.Event()

        def feed():
            while not stop.is_set():
                srv.publish(ref)
                time.sleep(0.02)

        threading.Thread(target=feed, daemon=True).start()
        try:
            src = CameraSource("http://127.0.0.1:8765/stream")
            frame, _, _ = src.read()
            self.assertEqual(frame.shape, ref.shape)
            self.assertLess(np.abs(frame.astype(int) - ref.astype(int)).mean(), 3)   # JPEG loss only
        finally:
            stop.set()
            srv.httpd.shutdown()


def track_at(tid, x, y, r=20):
    """A fake confirmed track with a circular outline."""
    c = cv2.ellipse2Poly((x, y), (r, r), 0, 0, 360, 10).reshape(-1, 1, 2).astype(np.int32)

    class D:
        contour, area, partial, bbox = c, math.pi * r * r, False, (x - r, y - r, 2 * r, 2 * r)

    class T:
        pass

    t = T()
    t.id, t.cx, t.cy, t.det, t.color, t.shape = tid, x, y, D, "red", "round"
    return t


class TestSelector(unittest.TestCase):
    W = 640

    def run_seq(self, seq, tracks, dt=1 / 30):
        """seq: list of gaze points (or None) per frame. Returns [(frame, id)] of locks."""
        s, out = Selector(CFG), []
        for i, g in enumerate(seq):
            for e in s.update(tracks, g, i * dt, self.W):
                out.append((i, e["id"]))
        return out

    def test_locks_once_after_dwell(self):
        a = track_at(1, 100, 100)
        locks = self.run_seq([(100, 100)] * 60, [a])
        self.assertEqual(locks, [(15, 1)])   # 0.5 s at 30 fps, exactly once

    def test_blink_does_not_reset(self):
        a = track_at(1, 100, 100)
        seq = [(100, 100)] * 8 + [None] * 3 + [(100, 100)] * 20   # 100 ms blink
        self.assertEqual(self.run_seq(seq, [a]), [(15, 1)])

    def test_look_away_rearms(self):
        a = track_at(1, 100, 100)
        seq = [(100, 100)] * 20 + [(400, 400)] * 10 + [(100, 100)] * 20
        self.assertEqual([i for _, i in self.run_seq(seq, [a])], [1, 1])

    def test_switch_latency_not_padded_by_grace(self):
        a, b = track_at(1, 100, 100), track_at(2, 300, 100)
        seq = [(100, 100)] * 20 + [(300, 100)] * 30
        locks = self.run_seq(seq, [a, b])
        self.assertEqual(locks[1], (35, 2))   # 15 frames after arriving at b

    def test_best_guess_and_nothing(self):
        a = track_at(1, 100, 100, r=20)
        near = self.run_seq([(100, 150)] * 20, [a])        # 30 px off the edge -> best guess
        far = self.run_seq([(500, 400)] * 20, [a])
        self.assertEqual(near, [(15, 1)])
        self.assertEqual(far, [])

    def test_edge_flicker_is_sticky(self):
        a, b = track_at(1, 100, 100), track_at(2, 150, 100)   # outlines 10 px apart
        seq = [(125, 100), (126, 100)] * 15                     # gaze wobbling in the gap
        self.assertEqual(len(self.run_seq(seq, [a, b])), 1)


class TestTrackerDepth(unittest.TestCase):
    def test_ids_stable_under_head_motion_and_depth(self):
        det, trk = Detector(CFG), Tracker(CFG)
        cfg = config.load(None)
        ids = None
        n = len(synthetic.DEFAULT_OBJECTS)
        for i in range(60):
            tracks = trk.update(det.detect(synthetic.render(i)), i / 30, 640)
            self.assertEqual(len(tracks) if i >= 5 else n, n)
            if i == 10:
                ids = sorted(t.id for t in tracks)
                cfg["depth"]["ref_distance_cm"] = 60.0
                cfg["depth"]["ref_size"] = reference_sizes(tracks, 640)
        self.assertEqual(sorted(t.id for t in tracks), ids)
        for t in tracks:
            dist, vol = estimate_depth(t, cfg, 640)
            self.assertAlmostEqual(dist, 60.0, delta=6.0)


if __name__ == "__main__":
    unittest.main()

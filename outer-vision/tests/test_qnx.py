"""The QNX eye-tracker link: the gaze model, and QnxGaze against a stand-in for the board.

Run: .venv/bin/python -m unittest discover tests -v

FakeBoard serves the same JSON shape as `camera_streamer`'s GET /api/state (see pi/http_server.c in the
htn-gaze repo, branch pupil-in-eye), so the whole path -- poll, rig model, eyelid state, blink gestures --
is exercised with no board, no cameras and no network.
"""
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from outer_vision import config, gaze_model
from outer_vision.io import QnxGaze, scene_url, state_url

CFG = config.load(None)

# Expected gaze for a given (left pupil, right pupil), computed by running the ORIGINAL
# gui/src/geometry.js under node and normalizing by the 960x540 scene size. If a change here breaks these,
# this repo and the eye team's live GUI have stopped agreeing about where the wearer is looking.
GOLDEN = {
    (0.500, 0.500, 0.500, 0.500): (0.5, 0.5),
    (0.530, 0.500, 0.530, 0.500): (0.2754892713875209, 0.5),
    (0.470, 0.500, 0.470, 0.500): (0.7245107286124791, 0.5),
    (0.500, 0.470, 0.500, 0.470): (0.5, 0.280867374573521),
    (0.500, 0.530, 0.500, 0.530): (0.5, 0.7191326254264785),
    (0.545, 0.520, 0.510, 0.490): (0.29081811089184806, 0.5393070201463467),
    (0.440, 0.440, 0.460, 0.470): (0.9309862146906426, 0.11529153994765967),
    (0.620, 0.580, 0.580, 0.610): (0.0, 1.0),
    (0.500, 0.500, 0.560, 0.440): (0.24925149081282189, 0.2566915423705137),
    (0.505, 0.498, 0.495, 0.502): (0.5, 0.5),
    (0.580, 0.470, 0.520, 0.530): (0.04493641551356375, 0.4937993810083364),
    (0.490, 0.610, 0.530, 0.570): (0.42050481112290883, 1.0),
    (0.500, 0.360, 0.500, 0.640): (0.5, 0.5000000000000003),
}

LID_OPEN = [0.40, 0.45, 0.50, 0.435, 0.60, 0.45, 0.60, 0.55, 0.50, 0.565, 0.40, 0.55,
            0.45, 0.44, 0.55, 0.44]                      # openness ~0.325: eye open
LID_SHUT = [0.40, 0.497, 0.50, 0.495, 0.60, 0.497, 0.60, 0.503, 0.50, 0.505, 0.40, 0.503,
            0.45, 0.496, 0.55, 0.496]                    # openness ~0.05: eye closed


def eye(px, py, lid=None, pupil_ok=1):
    return {"lid": list(lid if lid is not None else LID_OPEN), "iris": [px, py],
            "ring": [px, py] * 4, "center": [0.5, 0.5], "off": [0.0, 0.0], "noff": [0.0, 0.0],
            "pupil": [px, py], "pupil_r": 0.01, "pupil_ok": pupil_ok, "pupil_score": 0.8,
            "poff": [0.0, 0.0], "npoff": [0.0, 0.0]}


def board_state(left=None, right=None, n=2, frame_id=1):
    return {"source": "qnx", "qnx_connected": True, "camera_fps": 30.0, "infer_fps": 20.0,
            "link_fps": 0, "width": 960, "height": 540, "frame_id": frame_id, "jpeg_bytes": 1,
            "n": n, "score": 0.95, "engine": "mediapipe-facemesh-v2+pupil",
            "eyes": {} if n == 0 else {"left": left or eye(0.5, 0.5), "right": right or eye(0.5, 0.5)}}


class FakeBoard:
    """Stand-in for camera_streamer's HTTP server: serves whatever state you last set."""

    def __init__(self):
        self.state = board_state()
        self.hits = 0
        srv = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                srv.hits += 1
                body = json.dumps(srv.state).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


def wait_for(pred, timeout=4.0, step=0.01):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = pred()
        if v:
            return v
        time.sleep(step)
    return pred()


class TestGazeModel(unittest.TestCase):
    def test_selfcheck(self):
        self.assertEqual(gaze_model.selfcheck(), [])

    def test_matches_the_eye_repos_javascript(self):
        rig = gaze_model.rig_from()
        for (lx, ly, rx, ry), (ex, ey) in GOLDEN.items():
            g = gaze_model.gaze_from_state(
                board_state(eye(lx, ly), eye(rx, ry)), rig)
            self.assertTrue(g["ok"], (lx, ly, rx, ry))
            self.assertAlmostEqual(g["x"], ex, places=7, msg=f"x for {(lx, ly, rx, ry)}")
            self.assertAlmostEqual(g["y"], ey, places=7, msg=f"y for {(lx, ly, rx, ry)}")

    def test_prefers_the_dark_pupil_over_the_iris(self):
        """pupil.c's fit is used when it succeeded, and MediaPipe's iris centre when it didn't."""
        rig = gaze_model.rig_from()
        e = eye(0.5, 0.5)
        e["iris"], e["pupil"], e["pupil_ok"] = [0.50, 0.50], [0.53, 0.50], 1
        with_pupil = gaze_model.gaze_from_state(board_state(e, dict(e)), rig)
        e2 = dict(e, pupil_ok=0)
        with_iris = gaze_model.gaze_from_state(board_state(e2, dict(e2)), rig)
        self.assertAlmostEqual(with_pupil["x"], GOLDEN[(0.530, 0.500, 0.530, 0.500)][0], places=7)
        self.assertAlmostEqual(with_iris["x"], 0.5, places=7)

    def test_lid_openness(self):
        self.assertGreater(gaze_model.eye_open(eye(0.5, 0.5, LID_OPEN)), 0.25)
        self.assertLess(gaze_model.eye_open(eye(0.5, 0.5, LID_SHUT)), 0.10)

    def test_a_shut_eye_does_not_drag_the_gaze(self):
        """A closed eye's pupil is wherever the lids pushed it. It must not influence the gaze point."""
        rig = gaze_model.rig_from()
        a = gaze_model.gaze_from_state(board_state(eye(0.53, 0.5), eye(0.47, 0.5, LID_SHUT)), rig)
        b = gaze_model.gaze_from_state(board_state(eye(0.53, 0.5), eye(0.58, 0.44, LID_SHUT)), rig)
        self.assertTrue(a["ok"] and b["ok"])
        self.assertAlmostEqual(a["x"], b["x"], places=9)
        self.assertAlmostEqual(a["y"], b["y"], places=9)
        # ...and it is not simply the two-eye answer: with one eye, kappa no longer cancels.
        both = gaze_model.gaze_from_state(board_state(eye(0.53, 0.5), eye(0.53, 0.5)), rig)
        self.assertNotAlmostEqual(a["x"], both["x"], places=3)

    def test_no_face_is_not_a_gaze(self):
        self.assertFalse(gaze_model.gaze_from_state(board_state(n=0), gaze_model.rig_from())["ok"])

    def test_off_image_is_flagged_and_clamped(self):
        g = gaze_model.gaze_from_state(board_state(eye(0.62, 0.58), eye(0.58, 0.61)), gaze_model.rig_from())
        self.assertTrue(g["ok"])
        self.assertFalse(g["on_image"])
        self.assertTrue(0.0 <= g["x"] <= 1.0 and 0.0 <= g["y"] <= 1.0)

    def test_zeroing_removes_a_residual_aim(self):
        raw = gaze_model.gaze_from_state(board_state(eye(0.53, 0.5), eye(0.53, 0.5)), gaze_model.rig_from())
        rig = gaze_model.rig_from({"zero_yaw_deg": raw["yaw_deg"], "zero_pitch_deg": raw["pitch_deg"]})
        zeroed = gaze_model.gaze_from_state(board_state(eye(0.53, 0.5), eye(0.53, 0.5)), rig)
        self.assertAlmostEqual(zeroed["x"], 0.5, places=6)
        self.assertAlmostEqual(zeroed["y"], 0.5, places=6)

    def test_smoother_follows_but_rejects_a_single_spike(self):
        s = gaze_model.Smoother(median=5, ema=0.45)
        for _ in range(8):
            s.push(0.5, 0.5)
        self.assertAlmostEqual(s.value[0], 0.5, places=6)
        s.push(0.9, 0.9)                       # one bad frame must not move the point far
        self.assertLess(abs(s.value[0] - 0.5), 0.02)
        for _ in range(10):
            s.push(0.8, 0.8)                   # a real move gets there
        self.assertLess(abs(s.value[0] - 0.8), 0.02)


class TestUrls(unittest.TestCase):
    def test_board_endpoints(self):
        self.assertEqual(state_url("192.168.2.2"), "http://192.168.2.2:8080/api/state")
        self.assertEqual(scene_url("192.168.2.2"), "http://192.168.2.2:8081/stream.mjpg")


class TestQnxGaze(unittest.TestCase):
    def setUp(self):
        self.board = FakeBoard()
        cfg = config.load(None)
        cfg["qnx"] = {**cfg["qnx"], "eye_port": self.board.port, "poll_hz": 120.0}
        self.cfg = cfg
        self.gaze = QnxGaze("127.0.0.1", cfg)
        self.assertTrue(wait_for(lambda: self.gaze.health()["connected"]), "never reached the fake board")

    def tearDown(self):
        self.board.close()

    def set(self, **kw):
        """Publish a new board state; frame_id and the landmarks both change, so it counts as fresh."""
        self.board.state = board_state(frame_id=self.board.state["frame_id"] + 1, **kw)

    def test_gaze_follows_the_pupil(self):
        self.set(left=eye(0.53, 0.5), right=eye(0.53, 0.5))
        g = wait_for(lambda: self.gaze.get(0) if (self.gaze.get(0) or (1, 1))[0] < 0.45 else None)
        self.assertIsNotNone(g, "gaze never moved left for a pupil offset to the right in the eye image")
        self.assertLess(g[0], 0.45)
        self.assertLess(self.gaze.age_ms(), 500)

    def test_long_left_wink_is_a_gesture(self):
        """The one thing the user can actually command with: a long wink of a single eye."""
        self.gaze.gestures(0)
        deadline = time.monotonic() + CFG["blink"]["long_min_s"] + 0.35
        i = 0
        while time.monotonic() < deadline:      # left eye shut, right eye open, landmarks moving
            i += 1
            self.set(left=eye(0.5 + i * 1e-4, 0.5, LID_SHUT), right=eye(0.5 + i * 1e-4, 0.5))
            time.sleep(0.02)
        self.set(left=eye(0.5, 0.5), right=eye(0.5, 0.5))          # both open again -> closure ends
        gs = wait_for(lambda: self.gaze.gestures(0) or None, timeout=2.0)
        self.assertTrue(gs, "no gesture from a long left wink")
        self.assertEqual(gs[0]["kind"], "long")
        self.assertEqual(gs[0]["side"], "left")

    def test_eyes_closed_freezes_dwell(self):
        self.set(left=eye(0.5, 0.5, LID_SHUT), right=eye(0.5, 0.5, LID_SHUT))
        self.assertTrue(wait_for(lambda: self.gaze.eyes_closed(0)), "closed eyes not reported")
        self.set(left=eye(0.5, 0.5), right=eye(0.5, 0.5))
        self.assertTrue(wait_for(lambda: not self.gaze.eyes_closed(0)), "eyes stayed closed after opening")

    def test_a_stalled_board_cannot_open_a_menu(self):
        """If inference freezes with the eyes shut, the repeated frame must NOT look like a long blink."""
        self.gaze.gestures(0)
        frozen = board_state(eye(0.5, 0.5, LID_SHUT), eye(0.5, 0.5, LID_SHUT), frame_id=99)
        self.board.state = frozen
        time.sleep(0.2)
        before = self.gaze.count
        # Camera frames keep advancing; the landmarks do not. This is the observed failure mode.
        end = time.monotonic() + CFG["blink"]["long_min_s"] + 0.4
        while time.monotonic() < end:
            frozen = dict(frozen, frame_id=frozen["frame_id"] + 1)
            self.board.state = frozen
            time.sleep(0.02)
        self.set(left=eye(0.5, 0.5), right=eye(0.5, 0.5))
        time.sleep(0.2)
        self.assertLessEqual(self.gaze.count - before, 1, "stale landmarks were counted as new samples")
        self.assertEqual(self.gaze.gestures(0), [], "a frozen board produced a blink gesture")

    def test_lost_face_is_not_a_command(self):
        """n == 0 means the tracker lost the face. That is not 'both eyes closed'."""
        self.gaze.gestures(0)
        self.board.state = board_state(n=0)
        self.assertTrue(wait_for(lambda: self.gaze.get(0) is None), "gaze survived a lost face")
        self.assertFalse(self.gaze.eyes_closed(0), "a lost face was read as closed eyes")
        time.sleep(CFG["blink"]["long_min_s"] + 0.3)
        self.set(left=eye(0.5, 0.5), right=eye(0.5, 0.5))
        time.sleep(0.2)
        self.assertEqual(self.gaze.gestures(0), [], "a lost face produced a blink gesture")

    def test_board_down_reports_and_recovers(self):
        self.board.close()
        self.assertTrue(wait_for(lambda: not self.gaze.health()["connected"]), "outage not reported")
        self.assertIsNone(self.gaze.get(0))
        self.assertIsNone(self.gaze.age_ms())

    def test_hysteresis_holds_a_borderline_eye_steady(self):
        lid = self.cfg["qnx"]["lid"]
        self.assertLess(lid["closed_below"], lid["open_above"], "thresholds must leave a dead band")
        self.gaze._closed["left"] = True
        mid = (lid["closed_below"] + lid["open_above"]) / 2
        self.assertTrue(self.gaze._eye_closed("left", mid), "a borderline eye flipped open")
        self.gaze._closed["left"] = False
        self.assertFalse(self.gaze._eye_closed("left", mid), "a borderline eye flipped closed")


if __name__ == "__main__":
    unittest.main()

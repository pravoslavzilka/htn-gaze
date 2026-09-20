import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import omni_voice as o  # noqa: E402


def status(**kw):
    st = {"t": time.time(), "frame_w": 640, "frame_h": 360, "gaze_px": [320, 180], "target": None, "last_lock": None,
          "seeing": [{"color": "red", "shape": "round"}]}
    st.update(kw)
    return st


class OvnAppTests(unittest.TestCase):
    def app(self, st):
        f = Path(tempfile.mkdtemp()) / "s.json"
        f.write_text(json.dumps(st))
        return o.OvnApp(f, "http://127.0.0.1:1/x")

    def test_target_is_what_is_looked_at(self):
        d = self.app(status(target={"id": 1, "color": "blue", "shape": "round", "note": "G4", "instrument": "marimba"})).look()
        self.assertEqual((d["color"], d["source"], d["x"], d["y"]), ("blue", "looking_at", 0.5, 0.5))

    def test_recent_lock_counts_but_old_one_does_not(self):
        ll = {"color": "red", "shape": "round", "note": "C4", "instrument": "marimba", "at": time.time() - 1}
        self.assertEqual(self.app(status(last_lock=ll)).look()["source"], "just_played")
        self.assertIsNone(self.app(status(last_lock={**ll, "at": time.time() - 30})).look())

    def test_stale_or_missing_status(self):
        self.assertIsNone(self.app(status(t=time.time() - 60, target={"color": "red"})).look())
        self.assertIsNone(o.OvnApp(Path(tempfile.gettempdir()) / "nope_missing.json", "x").look())

    def test_note_recent(self):
        a = self.app(status(last_lock={"color": "red", "at": time.time() - 0.3}))
        a.look()
        self.assertTrue(a.note_recent())


class NoiseTests(unittest.TestCase):
    def test_echo_and_placeholders(self):
        v = o.Voice(None)
        v.recent_says.append("I can't see the color without your gaze data. Could you describe it or point again?")
        self.assertTrue(v.is_noise(""))
        self.assertTrue(v.is_noise("The wearer's request is in the audio."))
        self.assertTrue(v.is_noise("Could you describe it or point again?"))          # its own words coming back
        self.assertFalse(v.is_noise("Omni, could you describe it or point again?"))    # addressed by name = real
        self.assertFalse(v.is_noise("make it longer"))

    def test_sleep_phrases(self):
        for t in ("Omni go to sleep", "okay good night", "that's all", "stop listening"):
            self.assertTrue(o.SLEEP_RE.search(t), t)
        self.assertFalse(o.SLEEP_RE.search("what color is this"))
        self.assertTrue(o.NAME_RE.search("Omni, hello"))


class CropTests(unittest.TestCase):
    @staticmethod
    def jpg(w=640, h=360):
        import cv2
        import numpy as np
        return cv2.imencode(".jpg", np.zeros((h, w, 3), np.uint8))[1].tobytes()

    @staticmethod
    def size(jpeg):
        import cv2
        import numpy as np
        h, w = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape[:2]
        return w, h

    def test_crop_is_a_close_up_with_the_same_aspect_ratio(self):
        w, h = self.size(o.crop_around(self.jpg(), 0.5, 0.5))
        self.assertEqual((w, h), (192, 108))                      # 30% of 640 wide

    def test_crop_stays_inside_the_frame_at_the_edges(self):
        for x, y in ((0.0, 0.0), (1.0, 1.0), (0.99, 0.01)):
            self.assertEqual(self.size(o.crop_around(self.jpg(), x, y)), (192, 108))

    def test_no_gaze_or_no_picture_means_no_crop(self):
        self.assertIsNone(o.crop_around(self.jpg(), None, 0.5))
        self.assertIsNone(o.crop_around(None, 0.5, 0.5))
        self.assertIsNone(o.crop_around(b"not a jpeg", 0.5, 0.5))


if __name__ == "__main__":
    unittest.main()

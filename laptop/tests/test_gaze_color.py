import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "calib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaze_color import name_color  # noqa: E402
from omni_voice import summarize  # noqa: E402


def patch(bgr):
    return np.full((9, 9, 3), bgr, np.uint8)


class ColorTests(unittest.TestCase):
    def test_names(self):
        for bgr, want in [((40, 40, 220), "red"), ((60, 190, 60), "green"), ((200, 90, 30), "blue"),
                          ((40, 220, 240), "yellow"), ((30, 30, 30), "black"), ((240, 240, 240), "white"),
                          ((128, 128, 128), "gray"), ((20, 130, 245), "orange"), ((200, 60, 150), "purple")]:
            self.assertEqual(name_color(patch(bgr))[0], want, bgr)

    def test_median_ignores_highlight(self):
        p = patch((40, 40, 220))
        p[0, :3] = (255, 255, 255)
        self.assertEqual(name_color(p)[0], "red")

    def test_summarize_majority(self):
        s = [{"color": c, "rgb": [1, 2, 3], "x": .5, "y": .5} for c in ("red", "blue", "red")]
        r = summarize(s)
        self.assertEqual((r["color"], r["share"]), ("red", 0.67))
        self.assertIsNone(summarize([]))


if __name__ == "__main__":
    unittest.main()

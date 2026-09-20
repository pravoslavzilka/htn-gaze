"""Deliberate blinks from the eye tracker's per-sample eye state. Pure logic (no I/O), so it's unit-testable.

A closure is any stretch where at least one eye is closed. When both eyes open again it is classified:
  shorter than short_max_s, both eyes      -> natural blink (ignored, but two in a row = "double")
  long_min_s .. long_max_s                 -> "long" gesture with side left | right | both
  anything else                            -> ignored (half-blinks, resting the eyes, tracker dropouts)
Side = which eye was closed for most of the closure, so a squinting other eye doesn't flip it.
"""
from __future__ import annotations


class BlinkDetector:
    def __init__(self, p: dict):
        self.p = p
        self.start = None           # closure start time
        self.n = self.n_left = self.n_right = 0
        self.last_t = None
        self.last_short_end = -1e9  # end of the previous natural blink (for double blinks)
        self.closed = False         # at least one eye closed right now

    def closed_for(self, now: float) -> float:
        return 0.0 if self.start is None else now - self.start

    def feed(self, t: float, left_closed: bool, right_closed: bool) -> list:
        """One eye-state sample. Returns a list of gesture dicts (usually empty)."""
        out = []
        if self.start is not None and self.last_t is not None and t - self.last_t > self.p["max_sample_gap_s"]:
            self.start = None       # tracker went quiet mid-closure: can't tell what happened, drop it
        self.last_t = t
        any_closed = left_closed or right_closed
        self.closed = any_closed
        if any_closed:
            if self.start is None:
                self.start, self.n, self.n_left, self.n_right = t, 0, 0, 0
            self.n += 1
            self.n_left += left_closed
            self.n_right += right_closed
            return out
        if self.start is None:
            return out
        dur = t - self.start
        fl, fr = self.n_left / self.n, self.n_right / self.n
        self.start = None
        hi, lo = self.p["side_frac"], 1 - self.p["side_frac"]
        side = ("both" if fl >= hi and fr >= hi else "left" if fl >= hi and fr <= lo
                else "right" if fr >= hi and fl <= lo else None)
        if dur <= self.p["short_max_s"]:
            if side == "both":
                if t - dur - self.last_short_end <= self.p["double_gap_s"]:
                    out.append({"kind": "double", "t": t})
                    self.last_short_end = -1e9
                else:
                    self.last_short_end = t
        elif self.p["long_min_s"] <= dur <= self.p["long_max_s"] and side is not None:
            out.append({"kind": "long", "side": side, "t": t, "duration": round(dur, 2)})
        return out

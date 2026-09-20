"""Gaze point -> target object -> dwell -> one lock event per visit."""
from __future__ import annotations

import cv2


class Selector:
    def __init__(self, cfg: dict):
        self.p = cfg["selector"]
        self.target_id = None       # object currently being dwelt on
        self.dwell_start = 0.0
        self.last_on = -1e9         # last time gaze was on the target
        self.locked = False         # already fired for this visit
        self.best_guess = False
        self.progress = 0.0
        self.pending_id = None      # what gaze is on while the grace period for the old target runs
        self.pending_since = 0.0
        self.held_since = None      # eyes closed: dwell frozen, target kept

    def hold(self, now: float):
        """Eyes are closed (blink or blink gesture): freeze dwell and keep the target instead of letting the
        grace period drop it. Dwell resumes where it was when the eyes open."""
        if self.held_since is None:
            self.held_since = now

    def _candidate(self, tracks: list, gaze_px, frame_w: int):
        """Closest object outline to the gaze point; returns (track, is_best_guess) or (None, False)."""
        if gaze_px is None or not tracks:
            return None, False
        gx, gy = gaze_px
        dists = {}
        for t in tracks:
            # pointPolygonTest: +inside, -outside; clamp inside to 0 so "on the object" ties at 0.
            dists[t.id] = max(0.0, -cv2.pointPolygonTest(t.det.contour, (float(gx), float(gy)), True))
        best = min(tracks, key=lambda t: dists[t.id])
        # Stickiness: keep the current target unless another is clearly closer (stops edge flicker).
        cur = next((t for t in tracks if t.id == self.target_id), None)
        if cur is not None and dists[cur.id] - dists[best.id] < self.p["switch_margin_frac"] * frame_w:
            best = cur
        d = dists[best.id]
        if d <= self.p["select_radius_frac"] * frame_w:
            return best, False
        if d <= self.p["best_guess_radius_frac"] * frame_w:
            return best, True
        return None, False

    def update(self, tracks: list, gaze_px, now: float, frame_w: int) -> list:
        """Returns a list of events (currently only lock events) produced on this frame."""
        if self.held_since is not None:
            d = now - self.held_since
            self.dwell_start += d
            self.last_on += d
            self.pending_since += d
            self.held_since = None
        cand, guess = self._candidate(tracks, gaze_px, frame_w)
        events = []
        if cand is not None and cand.id == self.target_id:
            self.last_on = now
            self.best_guess = guess
            self.pending_id = None
        else:
            cid = cand.id if cand is not None else None
            if cid != self.pending_id:
                self.pending_id, self.pending_since = cid, now
            if now - self.last_on > self.p["grace_s"] or self.target_id not in {t.id for t in tracks}:
                # Gaze really moved on (not a blink/jitter) -> new visit, re-arm the trigger.
                # Dwell counts from when gaze arrived, so the grace period adds no latency.
                self.target_id = cid
                self.dwell_start = self.pending_since
                self.last_on = now
                self.locked = False
                self.best_guess = guess
                self.pending_id = None
        if self.target_id is None:
            self.progress = 0.0
            return events
        self.progress = min(1.0, (now - self.dwell_start) / max(self.p["dwell_s"], 1e-6))
        if not self.locked and self.progress >= 1.0:
            self.locked = True
            events.append({"type": "lock", "id": self.target_id, "best_guess": self.best_guess})
        return events

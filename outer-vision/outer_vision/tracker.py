"""Frame-to-frame association so objects keep stable IDs and shape labels stop flickering."""
from __future__ import annotations

import math
from collections import Counter, deque


class Track:
    def __init__(self, tid: int, det, now: float, votes: int):
        self.id = tid
        self.color = det.color
        self.cx, self.cy = det.cx, det.cy
        self.det = det                      # latest raw detection (contour, bbox, area, ...)
        self.shape_votes = deque(maxlen=votes)
        self.shape_votes.append(det.shape)
        self.hits = 1
        self.last_seen = now

    @property
    def shape(self) -> str:
        return Counter(self.shape_votes).most_common(1)[0][0]


class Tracker:
    def __init__(self, cfg: dict):
        self.p = cfg["tracker"]
        self.tracks: list = []
        self._next_id = 1

    def update(self, dets: list, now: float, frame_w: int) -> list:
        max_jump = self.p["max_jump_frac"] * frame_w
        a = self.p["smooth"]
        # Greedy nearest-pair matching, same colour only. With ~5 objects this is plenty.
        pairs = sorted(
            (math.hypot(t.cx - d.cx, t.cy - d.cy), ti, di)
            for ti, t in enumerate(self.tracks)
            for di, d in enumerate(dets)
            if t.color == d.color
        )
        used_t, used_d = set(), set()
        for dist, ti, di in pairs:
            if dist > max_jump or ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t, d = self.tracks[ti], dets[di]
            t.cx = a * d.cx + (1 - a) * t.cx
            t.cy = a * d.cy + (1 - a) * t.cy
            t.det = d
            t.shape_votes.append(d.shape)
            t.hits += 1
            t.last_seen = now
        for di, d in enumerate(dets):
            if di not in used_d:
                self.tracks.append(Track(self._next_id, d, now, self.p["shape_votes"]))
                self._next_id += 1
        self.tracks = [t for t in self.tracks if now - t.last_seen <= self.p["max_missing_s"]]
        return self.confirmed()

    def confirmed(self) -> list:
        return [t for t in self.tracks if t.hits >= self.p["confirm_hits"]]


def estimate_depth(track, cfg: dict, frame_w: int) -> tuple:
    """(distance_cm, volume) from apparent size vs. a reference capture; (None, None) if uncalibrated.

    Pinhole: size_on_image ∝ 1/distance, so distance = ref_distance * ref_size / size.
    Measures distance from the camera, which stands in for "farther back on the table".
    """
    d = cfg["depth"]
    ref = d["ref_size"].get(f"{track.color}/{track.shape}") or d["ref_size"].get(track.shape)
    if not d["ref_distance_cm"] or not ref or track.det.partial:
        return None, None
    size = math.sqrt(track.det.area) / frame_w
    dist = d["ref_distance_cm"] * ref / max(size, 1e-6)
    t = (dist - d["near_cm"]) / max(d["far_cm"] - d["near_cm"], 1e-6)
    vol = 1.0 - min(max(t, 0.0), 1.0) * (1.0 - d["min_volume"])
    return dist, vol


def reference_sizes(tracks: list, frame_w: int) -> dict:
    """Current apparent sizes, keyed both per colour/shape and per shape (median) for fallback."""
    out, by_shape = {}, {}
    for t in tracks:
        if t.det.partial:
            continue
        s = math.sqrt(t.det.area) / frame_w
        out[f"{t.color}/{t.shape}"] = s
        by_shape.setdefault(t.shape, []).append(s)
    for shape, sizes in by_shape.items():
        out[shape] = sorted(sizes)[len(sizes) // 2]
    return out

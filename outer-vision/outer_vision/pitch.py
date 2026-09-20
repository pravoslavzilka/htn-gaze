"""Small audio helpers for generated samples: trim, pitch estimate, repitch (numpy only)."""
from __future__ import annotations

import numpy as np


def trim(x: np.ndarray, sr: int, keep_tail=False, thresh=0.02) -> np.ndarray:
    """Cut leading (and, for speech, trailing) silence, normalise to 0.9 peak, short fade-out. int16 in/out."""
    f = x.astype(np.float32)
    peak = np.abs(f).max()
    if peak == 0:
        return x
    loud = np.flatnonzero(np.abs(f) > thresh * peak)
    start = max(0, loud[0] - int(0.005 * sr))
    end = min(len(f), loud[-1] + int(0.05 * sr)) if keep_tail else len(f)
    f = f[start:end] / peak * 0.9
    n = min(len(f), int(0.05 * sr))
    f[len(f) - n:] *= np.linspace(1, 0, n)
    return (f * 32767).astype(np.int16)


def estimate_f0(x: np.ndarray, sr: int, fmin=60.0, fmax=1200.0):
    """Fundamental of a single note (autocorrelation just after the attack), or None if unpitched."""
    f = x.astype(np.float32)
    onset = int(np.argmax(np.abs(f) > 0.1 * np.abs(f).max()))
    seg = f[onset + int(0.04 * sr): onset + int(0.29 * sr)]
    if len(seg) < int(sr / fmin) * 3:
        return None
    seg = (seg - seg.mean()) * np.hanning(len(seg))
    spec = np.fft.rfft(seg, 2 * len(seg))
    ac = np.fft.irfft(np.abs(spec) ** 2)[:len(seg)]
    if ac[0] <= 0:
        return None
    ac /= ac[0]
    lo, hi = int(sr / fmax), int(sr / fmin)
    best = lo + int(np.argmax(ac[lo:hi]))
    if ac[best] < 0.5:
        return None
    # Octave errors: prefer the shortest lag that's nearly as periodic (a lag of 2 periods also peaks).
    for lag in range(lo, best):
        if ac[lag] >= 0.9 * ac[best] and ac[lag] >= ac[lag - 1] and ac[lag] >= ac[lag + 1]:
            best = lag
            break
    a, b, c = ac[best - 1], ac[best], ac[best + 1]
    shift = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    return float(sr / (best + shift))


def repitch(x: np.ndarray, ratio: float) -> np.ndarray:
    """Play back `ratio` times faster (pitch x ratio, shorter). float32 in/out."""
    pos = np.arange(0, len(x) - 1, ratio)
    return np.interp(pos, np.arange(len(x)), x).astype(np.float32)

"""Speech output: a streaming PCM player (sounddevice/PortAudio) and WAV helpers. There is no microphone
input anywhere: the user controls everything with gaze and blinks."""
from __future__ import annotations

import collections
import io
import threading
import time
import wave

import numpy as np


def to_wav(pcm16: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16.astype("<i2").tobytes())
    return buf.getvalue()


def read_wav(path) -> tuple:
    """-> (int16 mono samples, sample rate)."""
    with wave.open(str(path), "rb") as w:
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        x = np.frombuffer(w.readframes(n), "<i2")
    return (x.reshape(-1, ch)[:, 0].copy() if ch > 1 else x), sr


class Player:
    """Streams PCM16 mono chunks as they arrive (low time-to-first-sound). stop() cuts playback."""

    def __init__(self, sr=24000):
        import sounddevice as sd
        self.sr = sr
        self._buf = collections.deque()
        self._lock = threading.Lock()
        self._pending = np.zeros(0, np.int16)
        self.last_audio = 0.0
        self.stream = sd.OutputStream(samplerate=sr, channels=1, dtype="int16", callback=self._cb)
        self.stream.start()

    def feed(self, pcm_bytes: bytes):
        with self._lock:
            self._buf.append(np.frombuffer(pcm_bytes, "<i2"))

    def stop(self):
        with self._lock:
            self._buf.clear()
            self._pending = np.zeros(0, np.int16)

    @property
    def playing(self):
        return bool(self._buf) or len(self._pending) > 0 or time.monotonic() - self.last_audio < 0.25

    def _cb(self, out, frames, t, status):
        with self._lock:
            while len(self._pending) < frames and self._buf:
                self._pending = np.concatenate([self._pending, self._buf.popleft()])
            n = min(frames, len(self._pending))
            out[:n, 0] = self._pending[:n]
            out[n:, 0] = 0
            self._pending = self._pending[n:]
            if n:
                self.last_audio = time.monotonic()

"""Audio plumbing: microphone capture (push-to-talk), a streaming PCM speaker, WAV helpers.

Player and the WAV helpers are adapted from the teammate's Outer-Vision-Network repo (outer_vision/audio.py).
"""
import collections
import io
import threading
import time
import wave

import numpy as np

MIC_SR = 16000


def to_wav(pcm16, sr):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.asarray(pcm16, "<i2").tobytes())
    return buf.getvalue()


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2")
    return (x.reshape(-1, ch)[:, 0].copy() if ch > 1 else x), sr


def record_until_enter(sr=MIC_SR, max_s=15.0):
    """Push-to-talk: recording starts now, stops on Enter (or after max_s). Returns int16 mono samples."""
    import sounddevice as sd
    chunks = []
    with sd.InputStream(samplerate=sr, channels=1, dtype="int16", callback=lambda d, f, t, s: chunks.append(d.copy())):
        stop = threading.Event()
        threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
        stop.wait(max_s)
    return np.concatenate(chunks)[:, 0] if chunks else np.zeros(0, np.int16)


class Player:
    """Streams PCM16 mono chunks as they arrive (low time-to-first-sound)."""

    def __init__(self, sr=24000):
        import sounddevice as sd
        self.sr = sr
        self._buf = collections.deque()
        self._lock = threading.Lock()
        self._pending = np.zeros(0, np.int16)
        self.last_audio = 0.0
        self.stream = sd.OutputStream(samplerate=sr, channels=1, dtype="int16", callback=self._cb)
        self.stream.start()

    def feed(self, pcm_bytes):
        with self._lock:
            self._buf.append(np.frombuffer(pcm_bytes[:len(pcm_bytes) - len(pcm_bytes) % 2], "<i2"))

    @property
    def playing(self):
        return bool(self._buf) or len(self._pending) > 0 or time.monotonic() - self.last_audio < 0.25

    def wait(self):
        time.sleep(0.05)
        while self.playing:
            time.sleep(0.05)

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

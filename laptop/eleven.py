"""ElevenLabs: speech (TTS) and generated sound effects, as raw PCM16 mono at 24 kHz.

Adapted from the teammate's Outer-Vision-Network repo (outer_vision/eleven.py). pcm_44100 needs a Pro plan,
so everything is 24 kHz. Key from env ELEVENLABS_API_KEY.
"""
import json
import os
import urllib.error
import urllib.request

import numpy as np

API = "https://api.elevenlabs.io/v1"
SR = 24000
DEFAULT_VOICE = "21m00Tcm4TlvDq8ikWAM"
TTS_MODEL = "eleven_flash_v2_5"


class ElevenError(RuntimeError):
    pass


class Eleven:
    def __init__(self, key=None, voice_id=None, timeout=45):
        self.key = key if key is not None else os.environ.get("ELEVENLABS_API_KEY", "")
        self.voice_id = voice_id or os.environ.get("ELEVENLABS_VOICE_ID", DEFAULT_VOICE)
        self.timeout = timeout

    def _open(self, path, body):
        req = urllib.request.Request(f"{API}{path}?output_format=pcm_{SR}", json.dumps(body).encode(), method="POST",
                                     headers={"xi-api-key": self.key, "Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            raise ElevenError(f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            raise ElevenError(f"network: {e}") from None

    def tts_stream(self, text, chunk=4096):
        """Yield PCM16 24 kHz as it is generated."""
        with self._open(f"/text-to-speech/{self.voice_id}/stream", {"text": text, "model_id": TTS_MODEL}) as r:
            while True:
                try:
                    buf = r.read(chunk)
                except (urllib.error.URLError, TimeoutError, OSError) as e:
                    raise ElevenError(f"network mid-stream: {e}") from None
                if not buf:
                    return
                yield buf

    def sfx(self, prompt, seconds, influence=0.7):
        """Generated sound as mono PCM16 24 kHz bytes."""
        with self._open("/sound-generation", {"text": prompt, "duration_seconds": seconds,
                                              "prompt_influence": influence}) as r:
            b = r.read()
        x = np.frombuffer(b[:len(b) - len(b) % 4], "<i2")
        # The sound API returns interleaved STEREO PCM (measured by the teammate): mix down or it plays an octave low.
        if abs(len(x) / 2 / SR - seconds) < abs(len(x) / SR - seconds):
            x = x.reshape(-1, 2).astype(np.int32).mean(1).astype(np.int16)
        return x.tobytes()

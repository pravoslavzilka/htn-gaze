"""ElevenLabs: Maestro's voice, the menu prompt clips, and the generated instrument samples.

Everything is requested as raw PCM16 mono at 24 kHz (pcm_44100 needs a Pro plan; 24 kHz also matches the
speech Player). Key from env ELEVENLABS_API_KEY.

Two ways in, both used live:
  * tools/gen_audio.py renders the instrument samples and the fixed menu prompts ahead of time, so the
    menu answers instantly and works with no network at all.
  * tts_stream() speaks whatever OMNI decided to say, streaming the audio as it arrives so the first
    syllable starts long before the sentence is finished. This is Maestro's normal voice.
One voice for the whole instrument: the prompt clips come from the same voice_id as the live speech.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import numpy as np

API = "https://api.elevenlabs.io/v1"
SR = 24000

# Maestro asks for a tone; ElevenLabs takes voice_settings. Lower stability and higher style mean more
# expressive delivery, higher stability a steadier read. Anything unknown falls back to "calm".
TONES = {
    "cheerful":    {"stability": 0.35, "style": 0.55},
    "playful":     {"stability": 0.30, "style": 0.65},
    "encouraging": {"stability": 0.45, "style": 0.45},
    "calm":        {"stability": 0.60, "style": 0.25},
}


class ElevenError(RuntimeError):
    pass


class Eleven:
    def __init__(self, cfg: dict, key=None, timeout=30):
        e = cfg["eleven"]
        self.key = key if key is not None else os.environ.get("ELEVENLABS_API_KEY", "")
        self.voice_id = os.environ.get("ELEVENLABS_VOICE_ID", e["voice_id"])
        self.tts_model = e["tts_model"]
        self.timeout = timeout

    @property
    def ok(self) -> bool:
        return bool(self.key)

    def _open(self, path: str, body: dict):
        req = urllib.request.Request(f"{API}{path}?output_format=pcm_{SR}", json.dumps(body).encode(), method="POST",
                                     headers={"xi-api-key": self.key, "Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            raise ElevenError(f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            raise ElevenError(f"network: {e}") from None

    def _post(self, path: str, body: dict) -> bytes:
        with self._open(path, body) as r:
            return r.read()

    def _body(self, text: str, tone=None) -> dict:
        body = {"text": text, "model_id": self.tts_model}
        if tone is not None:
            body["voice_settings"] = {**TONES.get(str(tone).lower(), TONES["calm"]),
                                      "similarity_boost": 0.75, "use_speaker_boost": True}
        return body

    def tts(self, text: str, tone=None) -> bytes:
        """The whole utterance as PCM16 24 kHz."""
        return self._post(f"/text-to-speech/{self.voice_id}", self._body(text, tone))

    def tts_stream(self, text: str, tone=None, chunk=4096):
        """Yield PCM16 24 kHz as it is generated, so speaking starts before the sentence is finished.

        The network can fail halfway; callers get whatever arrived and then an ElevenError, so they can
        decide whether a partial sentence is better than none (omni.py keeps it and moves on).
        """
        with self._open(f"/text-to-speech/{self.voice_id}/stream", self._body(text, tone)) as r:
            while True:
                try:
                    buf = r.read(chunk)
                except (urllib.error.URLError, TimeoutError, OSError) as e:
                    raise ElevenError(f"network mid-stream: {e}") from None
                if not buf:
                    return
                yield buf

    def sfx(self, prompt: str, seconds: float, influence: float = 0.7) -> bytes:
        """Sound effect as mono PCM16 24 kHz."""
        b = self._post("/sound-generation", {"text": prompt, "duration_seconds": seconds,
                                             "prompt_influence": influence})
        x = np.frombuffer(b[:len(b) - len(b) % 4], "<i2")
        # The sound effects API returns interleaved STEREO PCM (measured 2026-09-19: twice the samples for
        # the requested duration). Read as mono it plays at half speed, an octave low. Mix it down.
        if abs(len(x) / 2 / SR - seconds) < abs(len(x) / SR - seconds):
            x = x.reshape(-1, 2).astype(np.int32).mean(1).astype(np.int16)
        return x.tobytes()

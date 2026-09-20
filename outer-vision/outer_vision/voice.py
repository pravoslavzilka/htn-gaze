"""Maestro's voice: the fixed blink-menu lines, and speaking whatever OMNI decided to say.

Prompts play from pre-generated clips (assets/voice/<key>.wav, made by tools/gen_audio.py with ElevenLabs),
so the menu answers instantly and with no network. Free text goes to live ElevenLabs TTS, streamed so the
first syllable starts early, and falls back to macOS `say` / espeak if ElevenLabs can't be reached: the
user is never left without an answer. Clips and live speech use the same voice_id, so it is one voice.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from .audio import read_wav
from .eleven import SR, ElevenError
from .menu import PROMPTS


def local_say(text: str):
    for cmd in (["say", text], ["espeak", text]):
        if shutil.which(cmd[0]):
            subprocess.run(cmd, check=False)
            return


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x
    n = int(len(x) * sr_to / sr_from)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x.astype(np.float32)).astype(np.int16)


class Voice:
    def __init__(self, player=None, eleven=None, clip_dir="assets/voice", log=print):
        self.player, self.eleven, self.log = player, eleven, log
        self.clips = {}
        for key in PROMPTS:
            p = Path(clip_dir) / f"{key}.wav"
            if p.exists():
                x, sr = read_wav(p)
                self.clips[key] = resample(x, sr, SR)
        missing = sorted(set(PROMPTS) - set(self.clips))
        if missing:
            self.log(f"[voice] no clips for {missing}; run tools/gen_audio.py (using live/local speech meanwhile)")

    @property
    def can_speak(self) -> bool:
        """True when live ElevenLabs speech is actually available (key present and a speaker to play it on).
        Callers check this before say(), because say() falls back to local TTS rather than failing, and a
        caller that treated that as a failure would end up saying the same sentence twice."""
        return self.player is not None and self.eleven is not None and self.eleven.ok

    def prompt(self, key: str):
        """Speak a menu prompt now, cutting off anything still playing. Never blocks."""
        clip = self.clips.get(key)
        if clip is not None and self.player is not None:
            self.player.stop()
            self.player.feed(clip.tobytes())
        else:
            threading.Thread(target=self.say, args=(PROMPTS[key],), daemon=True).start()

    def say(self, text: str, tone=None, cut=False):
        """Speak free text in Maestro's voice; blocks until done. Returns the monotonic time the first
        audio reached the speaker, or None if it fell back to local TTS (which is not measurable here).

        `cut=True` stops whatever is playing at the moment the first chunk arrives, so the "One moment"
        clip is replaced by the real answer rather than queued behind it.
        """
        if self.player is not None and self.eleven is not None and self.eleven.ok:
            first = None
            try:
                for pcm in self.eleven.tts_stream(text, tone):
                    if first is None:
                        first = time.monotonic()
                        if cut:
                            self.player.stop()
                    self.player.feed(pcm)
            except ElevenError as e:
                if first is None:
                    self.log(f"[voice] ElevenLabs failed ({e}); using local speech")
                else:
                    self.log(f"[voice] ElevenLabs cut out ({e}); saying what arrived")
            if first is not None:
                while self.player.playing:
                    time.sleep(0.05)
                return first
        local_say(text)
        return None

"""The sound library: sounds OMNI creates are stored in Tiger (table sound_library) and can be played back by name.

PROTECTED colours are a hard no: OMNI can not create, replace or delete a sound for them. Enforced three times:
  1. the OMNI prompt tells the model,
  2. this module refuses (is_protected / ProtectedError) before any ElevenLabs credit is spent,
  3. a Postgres trigger on sound_library rejects the write even from another client (schema.sql).

Every sound is also cached as a WAV in laptop/sounds/, so the library keeps working (and nothing is lost) if Tiger is down;
unsynced cache files are uploaded to Tiger the next time it is reachable (sync_local).
Audio is raw mono PCM16 at 24 kHz.
"""
import json
import logging
import re
import time
from pathlib import Path

import numpy as np

from queries import QueryError
from voice_io import read_wav, to_wav

log = logging.getLogger("library")
SR = 24000
PROTECTED = frozenset({"blue", "green", "yellow", "red", "orange"})


class ProtectedError(RuntimeError):
    pass


def safe_name(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s or "").lower()).strip("_") or "sound"


def is_protected(name):
    """True for a protected colour, also inside a compound name ("dark red", "blue_green"): a hard no means hard."""
    return any(tok in PROTECTED for tok in safe_name(name).split("_"))


class Library:
    def __init__(self, db, cache_dir):
        self.db, self.dir = db, Path(cache_dir)
        self._names = (0.0, [])

    # ------------------------------------------------------------------ local cache
    def _index(self):
        p = self.dir / "index.json"
        try:
            return json.loads(p.read_text()) if p.exists() else {}
        except ValueError:
            return {}

    def _cache_write(self, name, pcm, prompt, seconds):
        self.dir.mkdir(exist_ok=True)
        (self.dir / f"{name}.wav").write_bytes(to_wav(np.frombuffer(pcm, "<i2"), SR))
        idx = self._index()
        idx[name] = {"file": f"{name}.wav", "prompt": prompt, "seconds": seconds}
        (self.dir / "index.json").write_text(json.dumps(idx, indent=1))

    # ------------------------------------------------------------------ write
    def save(self, name, pcm, prompt, seconds, created_by="omni"):
        """Store a sound. Raises ProtectedError for a protected colour. -> {"name", "tiger": bool}."""
        n = safe_name(name)
        if is_protected(n):
            raise ProtectedError(n)
        self._cache_write(n, pcm, prompt, seconds)
        stored = False
        if self.db is not None:
            try:
                self.db.execute(
                    """insert into sound_library (name, prompt, seconds, sample_rate, audio, created_by)
                       values (%s, %s, %s, %s, %s, %s)
                       on conflict (name) do update set prompt = excluded.prompt, seconds = excluded.seconds,
                         audio = excluded.audio, updated_at = now(), created_by = excluded.created_by""",
                    (n, prompt, float(seconds), SR, pcm, created_by))
                stored = True
            except QueryError as e:
                log.warning("sound '%s' is saved locally but not in Tiger yet: %s", n, e)
        self._names = (0.0, [])
        return {"name": n, "tiger": stored}

    def sync_local(self):
        """Upload cached sounds that Tiger does not have yet. Protected colours are skipped, never imported."""
        if self.db is None:
            return 0
        try:
            have = {r[0] for r in self.db.q("select name from sound_library")}
        except QueryError:
            return 0
        n = 0
        for name, meta in self._index().items():
            if name in have or is_protected(name):
                continue
            f = self.dir / meta.get("file", f"{name}.wav")
            if not f.exists():
                continue
            samples, sr = read_wav(f)
            if sr != SR:
                continue
            try:
                self.save(name, samples.astype("<i2").tobytes(), meta.get("prompt"), meta.get("seconds") or 0, "sync")
                n += 1
            except (ProtectedError, QueryError):
                pass
        return n

    # ------------------------------------------------------------------ read
    def get(self, name):
        """-> (pcm bytes, sample rate, "tiger"|"cache") or None. Protected colours are not in the library."""
        n = safe_name(name)
        if is_protected(n):
            return None
        if self.db is not None:
            try:
                rows = self.db.q("select audio, sample_rate from sound_library where name = %s", (n,))
                if rows:
                    try:
                        self.db.execute("update sound_library set plays = plays + 1 where name = %s", (n,))
                    except QueryError:
                        pass
                    return bytes(rows[0][0]), rows[0][1], "tiger"
            except QueryError as e:
                log.warning("Tiger unreachable, trying the local cache: %s", e)
        f = self.dir / f"{n}.wav"
        if f.exists():
            samples, sr = read_wav(f)
            return samples.astype("<i2").tobytes(), sr, "cache"
        return None

    def names(self, ttl=5.0):
        """Sound names available to play (Tiger + cache), excluding protected colours. Cached for a few seconds."""
        at, cached = self._names
        if cached and time.time() - at < ttl:
            return cached
        found = set(k for k in self._index() if not is_protected(k))
        if self.db is not None:
            try:
                found |= {r[0] for r in self.db.q("select name from sound_library order by name")}
            except QueryError:
                pass
        out = sorted(n for n in found if not is_protected(n))
        self._names = (time.time(), out)
        return out

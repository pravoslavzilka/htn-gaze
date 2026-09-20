"""Voice loop: you talk, OMNI understands, ElevenLabs makes the sound.

    "OMNI, look at this color and create a sound for it"
      -> OMNI hears the audio (+ the scene picture, if a camera is given) and answers JSON
      -> speaks "On it."  while ElevenLabs generates the sound in the background
      -> saves sounds/<color>.wav, registers it in sounds/index.json, speaks "It was added.", plays it

  python omni_voice.py                        # always listening: say "OMNI, ..."
  python omni_voice.py --ptt                  # push-to-talk instead: Enter starts, Enter stops
  python omni_voice.py --app http://127.0.0.1:8766       # gaze app (calib/gaze_live.py) that reads the colour
  python omni_voice.py --text "create a sound for the color red"     # no mic (testing)
  python omni_voice.py --wav question.wav                            # a recorded question

Any failure (OMNI, ElevenLabs, network) is logged and spoken/printed; the loop keeps running.
"""
import argparse
from datetime import datetime, timezone
import collections
import difflib
import base64
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import config
import library as soundlib
import queries
import telemetry
from eleven import SR, Eleven, ElevenError
from voice_io import Listener, Player, read_wav, record_until_enter, to_wav

log = logging.getLogger("omni_voice")
HERE = Path(__file__).parent
SOUNDS = HERE / "sounds"
LOG = HERE / "omni_interactions.jsonl"

BASE_URL = config.env("OMNI_BASE_URL", "https://yibuapi.com/v1")
MODEL = config.env("OMNI_MODEL", "qwen3.5-omni-flash")

SYSTEM_PROMPT = """You are OMNI, the voice assistant of a head-mounted eye-tracking wearable. The wearer talks
to you; they may address you as "OMNI". You get their spoken request (audio or text) and LOOK: JSON from the
gaze app with what they were looking at while speaking: colour, shape, and the musical note/instrument the app
maps to it; source "looking_at" (gaze is on it now) or "just_played" (they looked at it and it played its note);
x,y is the gaze point (0..1); on_table lists what the camera sees. The front camera picture may be attached, with
the gaze pointer drawn on it. "this", "this color", "that", "it" mean the LOOK object. Trust LOOK over your own
reading of the picture; if LOOK is unavailable, use a colour they named or the picture.

You can ask ElevenLabs to generate a sound for a colour. Write a short sound-effect description that suits the
colour (red = warm, bold, brassy; blue = calm, watery, soft chime; yellow = bright, sparkling; green = organic,
wooden marimba; ...).

SOUND LIBRARY: every sound you create is saved to a library and can be played back later. Each request ends with
LIBRARY: the names available. When they ask to play, hear or replay a tone/sound from the library, reply with the action
{"type": "play_sound", "name": "<the colour/name they said>"} and say "Playing the <name> tone." If they ask what sounds
exist, answer from LIBRARY with no action.
LOCKS AND AVAILABILITY ARE DECIDED BY THE SYSTEM, NOT BY YOU: always emit the action the wearer asked for (create_sound
for a colour, play_sound for a name), whatever the colour. The system refuses locked colours and missing sounds and speaks
the result itself. Never refuse, never say a colour is protected, never say a sound is missing.

WAKE WORD: the microphone is always on and hears everything, including other people and TV. Each request ends
with STATE: asleep or STATE: awake.
- asleep: only act when the speech is addressed to you by name: it contains "OMNI" (also accept how it may be
  transcribed: "Omni", "Omnie", "Ohm knee", "Hey Omni", "Okay Omni").
- awake: the wearer is mid-conversation with you, so follow-ups need no name ("make it longer", "and blue?").
  Act on speech unless it is clearly meant for another person.
Reply {"heard": "<text>", "wake": false} and nothing else ONLY when (asleep and the name is missing) or (awake and
it is clearly meant for someone else). Never use wake false for any other reason: if they addressed you, answer,
even a general question, briefly (max 2 sentences).
If the wearer tells you to stop, go to sleep, be quiet or that they are done, reply with the normal JSON
object below, with "say": "Going to sleep." and "actions": [{"type": "sleep"}] (works in either state).

Reply with ONLY one JSON object, no prose, no code fences:
{"heard": "<what they said>", "wake": true, "say": "<spoken reply>",
 "actions": [{"type": "create_sound", "color": "<one lowercase colour word>",
              "prompt": "<sound effect description, max 20 words, one short musical note or sound>",
              "seconds": <1-3>}]}
Use create_sound ONLY when the wearer explicitly asks you to create/make/generate a sound (or tone) for it.
Questions about the colour, note or what they see get NO actions, only an answer in "say".
When you use create_sound, "say" MUST be exactly "On it." (the wearer hears "It was added." separately when done).
For anything else (a question about the colour or what they see), use no actions and put a short answer
(max 2 sentences) in "say". If you can't tell what they mean, ask one short question in "say"."""


SLEEP_RE = re.compile(r"go(ing)? to sleep|good ?night|that'?s all|stop listening|be quiet|shut ?down|never ?mind", re.I)
NAME_RE = re.compile(r"\bomni|omnie|ohm ?knee|\bamni", re.I)


class OmniError(RuntimeError):
    pass


def omni_text(messages, timeout=40):
    """OpenAI-compatible streaming chat call (adapted from the teammate's OmniClient); returns the text."""
    body = {"model": MODEL, "messages": messages, "stream": True, "modalities": ["text"]}
    req = urllib.request.Request(BASE_URL.rstrip("/") + "/chat/completions", json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {config.env('OMNI_API_KEY', '')}"})
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise OmniError(f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise OmniError(f"network: {e}") from None
    out = []
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                for ch in json.loads(data).get("choices") or []:
                    out.append((ch.get("delta") or {}).get("content") or "")
            except ValueError:
                continue
    return "".join(out)


def parse_reply(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise OmniError(f"no JSON in reply: {text[:200]!r}")
    r = json.loads(m.group(0))
    if "type" in r and "actions" not in r:            # the model sometimes returns just the action
        r = {"heard": "", "wake": True, "actions": [r]}
        if r["actions"][0].get("type") == "sleep":
            r["say"] = "Going to sleep."
    r["say"] = str(r.get("say") or "")
    r["actions"] = [a for a in r.get("actions") or [] if isinstance(a, dict)]
    return r


class GazeApp:
    """Client for calib/gaze_live.py: /gaze (point + colour under it) and /frame (annotated scene picture)."""

    def __init__(self, url):
        self.url = url.rstrip("/") if url else None

    def _get(self, path, timeout=2):
        return urllib.request.urlopen(self.url + path, timeout=timeout).read()

    def look(self):
        """Current gaze reading, or None if the app is down or the eyes are not detected."""
        if not self.url:
            return None
        try:
            d = json.loads(self._get("/gaze"))
        except Exception as e:
            if not getattr(self, "down", False):      # once per outage, not once per poll
                log.warning("gaze app unavailable (%s)", e)
            self.down = True
            return None
        if getattr(self, "down", False):
            log.info("gaze app is back")
        self.down = False
        return d if d.get("valid") and d.get("age", 9) < 1.5 else None

    def note_recent(self, within=1.2):
        return False

    def frame(self):
        try:
            return self._get("/frame") if self.url else None
        except Exception as e:
            log.warning("scene frame unavailable (%s)", e)
            return None


class OvnApp:
    """Client for the teammate's Outer-Vision-Network run.py: the JSON it writes with --status-file (what it sees,
    which object the gaze is on, the last note it played) and its MJPEG overlay stream for the picture."""

    def __init__(self, status_path, stream_url):
        self.path, self.stream_url = Path(status_path), stream_url
        self.last_lock_at = 0.0
        self._warned = None

    def _warn(self, key, msg):
        if self._warned != key:                    # once per problem, not once per poll
            log.warning(msg)
        self._warned = key

    def look(self):
        try:
            st = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._warn("nofile", f"no status from the gaze app at {self.path} (is run.py started with --status-file?)")
            return None
        now = time.time()
        if now - st.get("t", 0) > 3:
            self._warn("stale", "gaze app status is stale (run.py stopped?)")
            return None
        self._warned = None
        ll = st.get("last_lock") or {}
        self.last_lock_at = ll.get("at", self.last_lock_at)
        tgt = st.get("target")
        if tgt:
            src, o = "looking_at", tgt
        elif ll and now - ll.get("at", 0) < 3.0:
            src, o = "just_played", ll
        else:
            return None
        w, h = st.get("frame_w") or 1, st.get("frame_h") or 1
        gp = st.get("gaze_px")
        return {"valid": True, "source": src, "color": o["color"], "shape": o.get("shape"), "note": o.get("note"),
                "instrument": o.get("instrument"), "x": gp[0] / w if gp else None, "y": gp[1] / h if gp else None,
                "on_table": [f"{t['color']} {t['shape']}" for t in st.get("seeing", [])]}

    def note_recent(self, within=1.2):
        """True while the instrument is (probably) still sounding its last note."""
        return time.time() - self.last_lock_at < within

    def frame(self):
        """One JPEG cut out of the MJPEG overlay stream."""
        try:
            with urllib.request.urlopen(self.stream_url, timeout=3) as r:
                buf = b""
                while len(buf) < 3_000_000:
                    buf += r.read(8192)
                    i = buf.find(b"\xff\xd8")
                    j = buf.find(b"\xff\xd9", i + 2) if i >= 0 else -1
                    if j > 0:
                        return buf[i:j + 2]
        except Exception as e:
            log.warning("front picture unavailable (%s)", e)
        return None


class GazeSampler(threading.Thread):
    """Polls /gaze while the wearer speaks; the colour they looked at most is what "this color" means."""

    def __init__(self, app, every=0.2):
        super().__init__(daemon=True)
        self.app, self.every, self.samples, self._halt = app, every, [], threading.Event()

    def run(self):
        while not self._halt.is_set():
            d = self.app.look()
            if d:
                self.samples.append(d)
            self._halt.wait(self.every)

    def finish(self):
        self._halt.set()
        self.join(3)
        return summarize(self.samples)


class GazeTrail(threading.Thread):
    """Keeps the last few seconds of gaze readings so an utterance can be matched to what was looked at."""

    def __init__(self, app, every=0.2):
        super().__init__(daemon=True)
        import collections
        self.app, self.every, self.buf = app, every, collections.deque(maxlen=150)

    def run(self):
        while True:
            d = self.app.look()
            if d:
                self.buf.append((time.time(), d))
            time.sleep(self.every)

    def window(self, t0, t1):
        return summarize([d for t, d in list(self.buf) if t0 <= t <= t1 + 0.3])


def summarize(samples):
    """Most-looked-at colour over the samples -> {"color", "shape", "note", ..., "share"} or None."""
    if not samples:
        return None
    import collections
    top, n = collections.Counter(s["color"] for s in samples).most_common(1)[0]
    last = [s for s in samples if s["color"] == top][-1]
    out = dict(last)
    out.pop("valid", None)
    out.pop("age", None)
    for k in ("x", "y"):
        if out.get(k) is not None:
            out[k] = round(out[k], 3)
    out["share"] = round(n / len(samples), 2)
    return {k: v for k, v in out.items() if v is not None}


def safe_name(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_") or "sound"


class Voice:
    def __init__(self, player=None, eleven=None, sink=None, run="live", library=None):
        self.player, self.eleven = player, eleven or Eleven()
        self.library = library or soundlib.Library(None, SOUNDS)
        self.sink, self.run = sink, run              # sink.add_omni(row): Tiger's omni_interactions
        self.speak_lock = threading.Lock()
        self.working = False                    # true from request start until the answer (and new sound) finished
        self.awake_s = config.env("AWAKE_SECONDS", 10.0, float)
        self.awake_until = 0.0                  # after a wake, follow-ups need no "OMNI" until this time
        self.recent_says = collections.deque(maxlen=6)   # what OMNI itself just said, to recognise its own echo

    @property
    def awake(self):
        return time.monotonic() < self.awake_until

    def sleep(self):
        self.awake_until = 0.0

    def busy(self):
        return self.working or (self.player is not None and self.player.playing)

    def say(self, text, parent=None):
        """Speak with ElevenLabs; if that fails, print it. Never raises."""
        print(f"OMNI: {text}", flush=True)
        if text and not SLEEP_RE.search(text):
            self.recent_says.append(text)
        if not text or self.player is None:
            return
        with self.speak_lock:
            try:
                with telemetry.span(parent, "elevenlabs.tts", "text-to-speech", chars=len(text)):
                    for chunk in self.eleven.tts_stream(text):
                        self.player.feed(chunk)
                    self.player.wait()
            except ElevenError as e:
                log.error("speech failed: %s", e)

    def is_noise(self, heard):
        """Empty/placeholder transcripts, or OMNI's own words coming back through the microphone."""
        h = re.sub(r"[^a-z0-9 ]", "", heard.lower()).strip()
        if not h or h == "the wearers request is in the audio":
            return True
        if NAME_RE.search(heard):
            return False                                   # they addressed OMNI by name: a real request
        for say in self.recent_says:
            sv = re.sub(r"[^a-z0-9 ]", "", say.lower()).strip()
            if difflib.SequenceMatcher(None, h, sv).ratio() > 0.8 or (len(h) > 12 and h in sv):
                return True
        return False

    def _store(self, look, heard, say, ms):
        """One row per real voice request in Tiger. Never raises."""
        if self.sink is None:
            return
        try:
            obj = f"{look['color']} {look['shape']}" if look and look.get("shape") else (look or {}).get("color")
            self.sink.add_omni((datetime.now(timezone.utc), self.run, obj, heard, say, float(ms)))
        except Exception:
            log.exception("could not queue the interaction for Tiger")

    def create_sound(self, action):
        color = safe_name(action.get("color"))
        seconds = min(3.0, max(0.5, float(action.get("seconds") or 1.5)))
        prompt = str(action.get("prompt") or f"a short musical note that sounds like the colour {color}")[:300]
        if soundlib.is_protected(color):                 # belt and braces: never spend a credit on a protected colour
            raise soundlib.ProtectedError(color)
        t0 = time.monotonic()
        pcm = self.eleven.sfx(prompt, seconds)
        with telemetry.span(getattr(self, "_tx", None), "library.write", "save to library", sound=color) as sp:
            saved = self.library.save(color, pcm, prompt, seconds)        # Tiger + local cache
            if sp is not None:
                sp.set_data("in_tiger", saved["tiger"])
        self.library_last = saved
        return saved["name"], pcm, round((time.monotonic() - t0) * 1000)

    def handle(self, audio=None, text=None, scene=None, look=None):
        """One request. audio: int16 mono 16 kHz samples, or text: str. Returns the parsed reply or None."""
        self.working = True
        try:
            with telemetry.transaction("omni_request", "omni.request") as tx:
                return self._handle(audio, text, scene, look, tx)
        finally:
            self.working = False

    def _handle(self, audio, text, scene, look, tx=None):
        t0 = time.monotonic()
        user = []
        if scene:
            user.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(scene).decode()}})
        if audio is not None:
            user.append({"type": "input_audio", "input_audio": {"data": "data:;base64," + base64.b64encode(
                to_wav(audio, 16000)).decode(), "format": "wav"}})
        request = text or "The wearer's request is in the audio."
        user.append({"type": "text", "text": request + "\nLOOK: " + (json.dumps(look) if look else "unavailable")
                     + "\nSTATE: " + ("awake" if self.awake else "asleep")
                     + "\nLIBRARY: " + json.dumps(self.library.names())})
        try:
            with telemetry.span(tx, "gen_ai.chat", f"chat {MODEL}", **{
                    "gen_ai.system": "omni", "gen_ai.request.model": MODEL, "gen_ai.operation.name": "chat",
                    "has_audio": audio is not None, "has_image": bool(scene)}):
                raw = omni_text([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}])
            reply = parse_reply(raw)
        except (OmniError, ValueError) as e:
            log.error("OMNI failed: %s", e)         # also becomes a Sentry issue, linked to this trace
            self.say("Sorry, I couldn't reach OMNI.", tx)
            return None
        omni_ms = round((time.monotonic() - t0) * 1000)
        heard = str(reply.get("heard") or "")
        if self.is_noise(heard):
            print(f"(noise/echo, ignored: {heard!r})", flush=True)
            telemetry.drop(tx)                        # not a real request: keep it out of Sentry
            return None
        if SLEEP_RE.search(heard) and (self.awake or NAME_RE.search(heard)):   # deterministic, not up to the model
            print(f"heard: {heard!r}  ({omni_ms} ms)", flush=True)
            self._store(look, heard, "Going to sleep.", omni_ms)
            self.say("Going to sleep.", tx)
            self.sleep()
            return reply
        if reply.get("wake") is False:
            print(f"(not for OMNI, ignored: {reply.get('heard')!r})", flush=True)
            telemetry.drop(tx)
            return None
        print(f"heard: {reply.get('heard')!r}  ({omni_ms} ms)", flush=True)
        record = {"time": time.time(), "heard": reply.get("heard"), "say": reply["say"], "omni_ms": omni_ms,
                  "actions": reply["actions"], "results": []}
        jobs = []
        go_sleep = any(a.get("type") == "sleep" for a in reply["actions"])
        if tx is not None:
            tx.set_tag("actions", ",".join(a.get("type", "?") for a in reply["actions"]) or "none")
            tx.set_data("object", (look or {}).get("color"))
            tx.set_data("understand_ms", omni_ms)
        refused = None
        for a in reply["actions"]:
            if go_sleep:
                break
            kind = a.get("type")
            if kind == "create_sound":
                if soundlib.is_protected(a.get("color")):     # a hard no, checked here whatever the model said
                    refused = safe_name(a.get("color"))
                    continue
                jobs.append(self._start_job(a, record, tx))
            elif kind == "play_sound":
                nm = safe_name(a.get("name") or a.get("color"))
                if soundlib.is_protected(nm):
                    refused = None
                    reply["say"] = f"{nm} is a built-in colour, so it is not in the library. Look at the {nm} object to hear its tone."
                    record["results"].append({"play": nm, "built_in": True})
                elif nm not in self.library.names(ttl=0):
                    reply["say"] = f"I don't have a sound for {nm} yet."
                    record["results"].append({"play": nm, "found": False})
                else:
                    reply["say"] = f"Playing the {nm} tone."
                    jobs.append(self._start_play(a, record, tx))
        if refused:
            reply["say"] = f"Sorry, {refused} is a protected colour, so I can't change its sound."
            record["results"].append({"refused": refused})
            print(f"(refused: {refused} is protected)", flush=True)
        self.say(reply["say"] or ("On it." if jobs else ""), tx)
        for th in jobs:
            th.join()
        LOG.open("a", encoding="utf-8").write(json.dumps(record) + "\n")
        self._store(look, heard, reply["say"], omni_ms)
        if go_sleep:
            self.sleep()
        else:
            self.awake_until = time.monotonic() + self.awake_s   # the 10 s start once OMNI has finished talking
        return reply

    def _start_job(self, action, record, tx=None):
        def run():          # runs on its own thread, so it attaches to the request's transaction explicitly
            try:
                self._tx = tx
                with telemetry.span(tx, "elevenlabs.sound_generation", "sound effect", color=str(action.get("color"))):
                    color, pcm, ms = self.create_sound(action)
            except soundlib.ProtectedError as e:
                self.say(f"Sorry, {e} is a protected colour, so I can't change its sound.", tx)
                return
            except (ElevenError, ValueError, OSError) as e:
                log.error("sound generation failed: %s", e)
                record["results"].append({"error": str(e)[:200]})
                self.say("Sorry, I couldn't create that sound.", tx)
                return
            record["results"].append({"color": color, "eleven_ms": ms})
            self.say(f"It was added. That's the sound for {color}.", tx)
            if self.player:
                self.player.feed(pcm)
                self.player.wait()
        th = threading.Thread(target=run, daemon=True)
        th.start()
        return th


def _play_from_library(self, action, record, tx=None):
    """Play a stored tone by name (from Tiger, or the local cache if Tiger is down). Runs on its own thread."""
    def run():
        name = safe_name(action.get("name") or action.get("color"))
        if soundlib.is_protected(name):
            self.say(f"{name} is a built-in colour. Look at the object to hear its tone.", tx)
            return
        with telemetry.span(tx, "library.play", "play from library", sound=name):
            got = self.library.get(name)
        if got is None:
            record["results"].append({"play": name, "found": False})
            self.say(f"I don't have a sound for {name} yet.", tx)
            return
        pcm, sr, source = got
        record["results"].append({"play": name, "source": source})
        if self.player is None:
            print(f"(would play '{name}': {len(pcm) / 2 / sr:.1f}s from {source})", flush=True)
            return
        with self.speak_lock:                      # wait until OMNI has finished saying "Playing ..."
            self.player.feed(pcm)
            self.player.wait()
    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th


Voice._start_play = _play_from_library


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", help="send this text instead of recording (testing)")
    ap.add_argument("--wav", help="send a recorded question instead of using the mic")
    ap.add_argument("--status", default=str(Path(tempfile.gettempdir()) / "ovn_status.json"),
                    help="status JSON written by Outer-Vision-Network run.py --status-file")
    ap.add_argument("--stream", default="http://127.0.0.1:8790/stream", help="run.py --stream MJPEG URL (front picture)")
    ap.add_argument("--app", default="", help="use calib/gaze_live.py instead, e.g. http://127.0.0.1:8766")
    ap.add_argument("--ptt", action="store_true", help="push-to-talk (Enter) instead of always listening for 'OMNI'")
    ap.add_argument("--run", default=config.env("RUN_ID", "live"), help="run id stored with each interaction")
    ap.add_argument("--no-tiger", action="store_true", help="don't store interactions in Tiger")
    ap.add_argument("--no-audio-out", action="store_true", help="don't play sound (print only)")
    a = ap.parse_args()
    for k in ("OMNI_API_KEY", "ELEVENLABS_API_KEY"):
        if not config.env(k):
            sys.exit(f"{k} missing: put it in laptop/.env")
    app = GazeApp(a.app) if a.app else OvnApp(a.status, a.stream)
    if app.look() is None:
        log.info("not looking at any object right now (or the gaze app is not running)")
    telemetry.init("omni_voice")
    sink = None
    if config.TIGER_DSN and not a.no_tiger:
        from tiger import TigerWriter
        sink = TigerWriter(config.TIGER_DSN).start()
    db = queries.Db(config.TIGER_DSN) if config.TIGER_DSN and not a.no_tiger else None
    library = soundlib.Library(db, SOUNDS)
    threading.Thread(target=lambda: log.info("sound library: %d local sound(s) synced to Tiger", library.sync_local()),
                     daemon=True).start()
    voice = Voice(None if a.no_audio_out else Player(SR), sink=sink, run=a.run, library=library)
    if a.text or a.wav:
        audio = read_wav(a.wav)[0] if a.wav else None
        voice.handle(audio=audio, text=a.text, scene=app.frame(), look=summarize([app.look()] if app.look() else []))
        if sink:
            sink.stop()                              # flush before exiting
        telemetry.flush()
        return
    if a.ptt:
        print("Push-to-talk: press Enter, speak, press Enter again. Ctrl+C quits.")
        while True:
            try:
                input("\n[Enter] to talk > ")
                print("listening... (Enter to stop)", flush=True)
                sampler = GazeSampler(app)
                sampler.start()
                audio = record_until_enter()
                look = sampler.finish()
                print(f"looking at: {look}", flush=True)
                if len(audio) < 8000:
                    print("too short, try again")
                    continue
                voice.handle(audio=audio, scene=app.frame(), look=look)
            except (KeyboardInterrupt, EOFError):
                break
            except Exception:
                log.exception("request failed; continuing")
        return
    trail = GazeTrail(app)
    trail.start()

    def announce_sleep():
        was = False
        while True:
            if was and not voice.awake:
                print('(asleep: say "OMNI" to wake me)', flush=True)
            was = voice.awake
            time.sleep(0.3)
    threading.Thread(target=announce_sleep, daemon=True).start()
    print(f'Listening. Say "OMNI, ..."  ({voice.awake_s:.0f} s after my last answer I go back to sleep; Ctrl+C quits)', flush=True)
    while True:
        try:
            for audio, t0, t1 in Listener(voice.busy, app.note_recent).utterances():
                look = trail.window(t0, t1)
                print(f"[{len(audio) / 16000:.1f}s of speech] looking at: {look}", flush=True)
                try:
                    voice.handle(audio=audio, scene=app.frame(), look=look)
                except Exception:
                    log.exception("request failed; continuing")
        except KeyboardInterrupt:
            break
        except Exception:
            log.exception("microphone stream failed; restarting in 2 s")
            time.sleep(2)


if __name__ == "__main__":
    main()

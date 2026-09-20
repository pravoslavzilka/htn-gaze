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
import base64
import json
import logging
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import config
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
gaze tracker with what they were looking at while speaking: the colour name and RGB measured under their gaze
point, and where (x,y in 0..1). The scene camera picture may be attached, with a green dot at the gaze point.
"this", "this color", "that" mean the LOOK colour. Trust the measured LOOK colour over your own reading of the
picture; if LOOK is missing, use a colour they named or the picture.

You can ask ElevenLabs to generate a sound for a colour. Write a short sound-effect description that suits the
colour (red = warm, bold, brassy; blue = calm, watery, soft chime; yellow = bright, sparkling; green = organic,
wooden marimba; ...).

WAKE WORD: the microphone is always on and hears everything, including other people and TV. Only act when the
speech is addressed to you by name: it contains "OMNI" (also accept how it may be transcribed: "Omni", "Omnie",
"Ohm knee", "Hey Omni", "Okay Omni"). If it is not addressed to OMNI, reply {"heard": "<text>", "wake": false}
and nothing else.

Reply with ONLY one JSON object, no prose, no code fences:
{"heard": "<what they said>", "wake": true, "say": "<spoken reply>",
 "actions": [{"type": "create_sound", "color": "<one lowercase colour word>",
              "prompt": "<sound effect description, max 20 words, one short musical note or sound>",
              "seconds": <1-3>}]}
When you use create_sound, "say" MUST be exactly "On it." (the wearer hears "It was added." separately when done).
For anything else (a question about the colour or what they see), use no actions and put a short answer
(max 2 sentences) in "say". If you can't tell what they mean, ask one short question in "say"."""


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

    def frame(self):
        try:
            return self._get("/frame") if self.url else None
        except Exception as e:
            log.warning("scene frame unavailable (%s)", e)
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
    """Most-looked-at colour over the samples -> {"color", "rgb", "x", "y", "share"} or None."""
    if not samples:
        return None
    import collections
    top, n = collections.Counter(s["color"] for s in samples).most_common(1)[0]
    last = [s for s in samples if s["color"] == top][-1]
    return {"color": top, "rgb": last["rgb"], "x": round(last["x"], 3), "y": round(last["y"], 3),
            "share": round(n / len(samples), 2)}


def safe_name(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_") or "sound"


class Voice:
    def __init__(self, player=None, eleven=None):
        self.player, self.eleven = player, eleven or Eleven()
        self.speak_lock = threading.Lock()
        self.working = False                    # true from request start until the answer (and new sound) finished

    def busy(self):
        return self.working or (self.player is not None and self.player.playing)

    def say(self, text):
        """Speak with ElevenLabs; if that fails, print it. Never raises."""
        print(f"OMNI: {text}", flush=True)
        if not text or self.player is None:
            return
        with self.speak_lock:
            try:
                for chunk in self.eleven.tts_stream(text):
                    self.player.feed(chunk)
                self.player.wait()
            except ElevenError as e:
                log.error("speech failed: %s", e)

    def create_sound(self, action):
        color = safe_name(action.get("color"))
        seconds = min(3.0, max(0.5, float(action.get("seconds") or 1.5)))
        prompt = str(action.get("prompt") or f"a short musical note that sounds like the colour {color}")[:300]
        t0 = time.monotonic()
        pcm = self.eleven.sfx(prompt, seconds)
        SOUNDS.mkdir(exist_ok=True)
        path = SOUNDS / f"{color}.wav"
        path.write_bytes(to_wav(__import__("numpy").frombuffer(pcm, "<i2"), SR))
        idx_path = SOUNDS / "index.json"
        idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
        idx[color] = {"file": path.name, "prompt": prompt, "seconds": seconds}
        idx_path.write_text(json.dumps(idx, indent=1))
        return color, pcm, round((time.monotonic() - t0) * 1000)

    def handle(self, audio=None, text=None, scene=None, look=None):
        """One request. audio: int16 mono 16 kHz samples, or text: str. Returns the parsed reply or None."""
        self.working = True
        try:
            return self._handle(audio, text, scene, look)
        finally:
            self.working = False

    def _handle(self, audio, text, scene, look):
        t0 = time.monotonic()
        user = []
        if scene:
            user.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(scene).decode()}})
        if audio is not None:
            user.append({"type": "input_audio", "input_audio": {"data": "data:;base64," + base64.b64encode(
                to_wav(audio, 16000)).decode(), "format": "wav"}})
        request = text or "The wearer's request is in the audio."
        user.append({"type": "text", "text": request + "\nLOOK: " + (json.dumps(look) if look else "unavailable")})
        try:
            reply = parse_reply(omni_text([{"role": "system", "content": SYSTEM_PROMPT},
                                           {"role": "user", "content": user}]))
        except (OmniError, ValueError) as e:
            log.error("OMNI failed: %s", e)
            self.say("Sorry, I couldn't reach OMNI.")
            return None
        omni_ms = round((time.monotonic() - t0) * 1000)
        if reply.get("wake") is False:
            print(f"(not for OMNI, ignored: {reply.get('heard')!r})", flush=True)
            return None
        print(f"heard: {reply.get('heard')!r}  ({omni_ms} ms)", flush=True)
        record = {"time": time.time(), "heard": reply.get("heard"), "say": reply["say"], "omni_ms": omni_ms,
                  "actions": reply["actions"], "results": []}
        jobs = []
        for a in reply["actions"]:
            if a.get("type") == "create_sound":
                jobs.append(self._start_job(a, record))
        self.say(reply["say"] or ("On it." if jobs else ""))
        for th in jobs:
            th.join()
        LOG.open("a", encoding="utf-8").write(json.dumps(record) + "\n")
        return reply

    def _start_job(self, action, record):
        def run():
            try:
                color, pcm, ms = self.create_sound(action)
            except (ElevenError, ValueError, OSError) as e:
                log.error("sound generation failed: %s", e)
                record["results"].append({"error": str(e)[:200]})
                self.say("Sorry, I couldn't create that sound.")
                return
            record["results"].append({"color": color, "eleven_ms": ms})
            self.say(f"It was added. That's the sound for {color}.")
            if self.player:
                self.player.feed(pcm)
                self.player.wait()
        th = threading.Thread(target=run, daemon=True)
        th.start()
        return th


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", help="send this text instead of recording (testing)")
    ap.add_argument("--wav", help="send a recorded question instead of using the mic")
    ap.add_argument("--app", default="http://127.0.0.1:8766", help="gaze app (calib/gaze_live.py) base URL; '' to disable")
    ap.add_argument("--ptt", action="store_true", help="push-to-talk (Enter) instead of always listening for 'OMNI'")
    ap.add_argument("--no-audio-out", action="store_true", help="don't play sound (print only)")
    a = ap.parse_args()
    for k in ("OMNI_API_KEY", "ELEVENLABS_API_KEY"):
        if not config.env(k):
            sys.exit(f"{k} missing: put it in laptop/.env")
    app = GazeApp(a.app)
    if a.app and app.look() is None:
        log.warning("gaze app at %s not reporting a gaze yet; say a colour name if it stays that way", a.app)
    voice = Voice(None if a.no_audio_out else Player(SR))
    if a.text or a.wav:
        audio = read_wav(a.wav)[0] if a.wav else None
        voice.handle(audio=audio, text=a.text, scene=app.frame(), look=summarize([app.look()] if app.look() else []))
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
    print('Listening. Say "OMNI, ..."  (Ctrl+C quits)', flush=True)
    while True:
        try:
            for audio, t0, t1 in Listener(voice.busy).utterances():
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

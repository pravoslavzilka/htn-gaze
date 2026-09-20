"""Voice loop: you talk, OMNI understands, ElevenLabs makes the sound.

    "OMNI, tell ElevenLabs to create a sound for this color"
      -> OMNI hears the audio (+ the scene picture, if a camera is given) and answers JSON
      -> speaks "On it."  while ElevenLabs generates the sound in the background
      -> saves sounds/<color>.wav, registers it in sounds/index.json, speaks "It was added.", plays it

  python omni_voice.py                        # push-to-talk on the default mic: Enter starts, Enter stops
  python omni_voice.py --scene http://169.254.96.94:8081/api/frame.jpg --gaze 0.62,0.40
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
from voice_io import Player, read_wav, record_until_enter, to_wav

log = logging.getLogger("omni_voice")
HERE = Path(__file__).parent
SOUNDS = HERE / "sounds"
LOG = HERE / "omni_interactions.jsonl"

BASE_URL = config.env("OMNI_BASE_URL", "https://yibuapi.com/v1")
MODEL = config.env("OMNI_MODEL", "qwen3.5-omni-flash")

SYSTEM_PROMPT = """You are OMNI, the voice assistant of a head-mounted eye-tracking wearable. The wearer talks
to you; they may address you as "OMNI". You get their spoken request (audio or text), optionally the scene
camera picture with a white circle where they are looking (GAZE), and JSON with what is known.

You can ask ElevenLabs to generate a sound for a colour. When the wearer asks for a sound for "this color",
"that", or names a colour, work out WHICH colour: the colour of the object under the white circle, or the colour
they named. Then write a short sound-effect description that suits that colour (red = warm, bold, brassy;
blue = calm, watery, soft chime; yellow = bright, sparkling; green = organic, wooden marimba; ...).

Reply with ONLY one JSON object, no prose, no code fences:
{"heard": "<what they said>", "say": "On it.",
 "actions": [{"type": "create_sound", "color": "<one lowercase colour word>",
              "prompt": "<sound effect description, max 20 words, one short musical note or sound>",
              "seconds": <1-3>}]}
When you use create_sound, "say" MUST be exactly "On it." (the wearer hears "It was added." separately when done).
If you can't tell the colour, or the request is something else, use no actions and ask one short question in "say"."""


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


def fetch_scene(url, gaze):
    """Scene JPEG from the camera URL, with the gaze point drawn on it. None if unavailable."""
    if not url:
        return None
    try:
        jpg = urllib.request.urlopen(url, timeout=3).read()
        if gaze:
            import cv2
            import numpy as np
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            c = (int(gaze[0] * w), int(gaze[1] * h))
            cv2.circle(img, c, 18, (255, 255, 255), 3)
            jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        return jpg
    except Exception as e:
        log.warning("scene camera unavailable (%s); continuing without a picture", e)
        return None


def safe_name(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_") or "sound"


class Voice:
    def __init__(self, player=None, eleven=None):
        self.player, self.eleven = player, eleven or Eleven()
        self.speak_lock = threading.Lock()

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

    def handle(self, audio=None, text=None, scene=None):
        """One request. audio: int16 mono 16 kHz samples, or text: str. Returns the parsed reply or None."""
        t0 = time.monotonic()
        user = []
        if scene:
            user.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(scene).decode()}})
        if audio is not None:
            user.append({"type": "input_audio", "input_audio": {"data": "data:;base64," + base64.b64encode(
                to_wav(audio, 16000)).decode(), "format": "wav"}})
        user.append({"type": "text", "text": text or "The wearer's request is in the audio."})
        try:
            reply = parse_reply(omni_text([{"role": "system", "content": SYSTEM_PROMPT},
                                           {"role": "user", "content": user}]))
        except (OmniError, ValueError) as e:
            log.error("OMNI failed: %s", e)
            self.say("Sorry, I couldn't reach OMNI.")
            return None
        omni_ms = round((time.monotonic() - t0) * 1000)
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
    ap.add_argument("--scene", help="scene camera JPEG URL, e.g. http://<board>:8081/api/frame.jpg")
    ap.add_argument("--gaze", help="gaze point as x,y in 0..1 to draw on the scene picture")
    ap.add_argument("--no-audio-out", action="store_true", help="don't play sound (print only)")
    a = ap.parse_args()
    for k in ("OMNI_API_KEY", "ELEVENLABS_API_KEY"):
        if not config.env(k):
            sys.exit(f"{k} missing: put it in laptop/.env")
    gaze = tuple(float(v) for v in a.gaze.split(",")) if a.gaze else None
    voice = Voice(None if a.no_audio_out else Player(SR))
    if a.text or a.wav:
        audio = read_wav(a.wav)[0] if a.wav else None
        voice.handle(audio=audio, text=a.text, scene=fetch_scene(a.scene, gaze))
        return
    print("Push-to-talk: press Enter, speak, press Enter again. Ctrl+C quits.")
    while True:
        try:
            input("\n[Enter] to talk > ")
            print("listening... (Enter to stop)", flush=True)
            audio = record_until_enter()
            if len(audio) < 8000:
                print("too short, try again")
                continue
            voice.handle(audio=audio, scene=fetch_scene(a.scene, gaze))
        except (KeyboardInterrupt, EOFError):
            break
        except Exception:
            log.exception("request failed; continuing")


if __name__ == "__main__":
    main()

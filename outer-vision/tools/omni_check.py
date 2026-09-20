#!/usr/bin/env python3
"""One-shot check of both live APIs before the demo: sends the synthetic table + a blink command, prints
OMNI's decision and whether it fits the command, then speaks the reply through the SAME voice path the
demo uses (config.json -> omni.speak_with, ElevenLabs by default) and reports which engine actually spoke.

  python tools/omni_check.py                          # change_instrument on object #3, the green cylinder
  python tools/omni_check.py teach
  python tools/omni_check.py faster --no-voice
Commands: change_instrument | change_note | faster | slower | teach.
Needs OMNI_API_KEY, and ELEVENLABS_API_KEY unless speak_with is "omni" or "none" (env or .env).
"""
import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from outer_vision import config, omni, synthetic  # noqa: E402
from outer_vision.eleven import Eleven  # noqa: E402
from outer_vision.music import Music  # noqa: E402
from outer_vision.selector import Selector  # noqa: E402
from outer_vision.voice import Voice  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("command", nargs="?", default="change_instrument", choices=sorted(omni.ALLOWED))
ap.add_argument("--target", type=int, default=3, help="object id for change_instrument / change_note")
ap.add_argument("--no-voice", action="store_true")
args = ap.parse_args()

config.load_env(ROOT / ".env")
cfg = config.load(str(ROOT / "config.json"))
o = cfg["omni"]
if not os.environ.get("OMNI_API_KEY"):
    sys.exit("set OMNI_API_KEY (env or .env) first")
client = omni.OmniClient(os.environ.get("OMNI_BASE_URL", o["base_url"]), os.environ["OMNI_API_KEY"],
                         os.environ.get("OMNI_MODEL", o["model"]), os.environ.get("OMNI_VOICE", o["voice"]),
                         timeout=o["timeout_s"])
music, sel = Music(cfg), Selector(cfg)
objs = {i + 1: {"id": i + 1, "color": c, "shape": s, **music.voice_of(c, s)}
        for i, (c, s, *_) in enumerate(synthetic.DEFAULT_OBJECTS)}
command = {"command": args.command}
if args.command in ("change_instrument", "change_note"):
    command["target"] = args.target
jpg = cv2.imencode(".jpg", synthetic.render(0))[1].tobytes()
scene = omni.build_scene(objs, sel.p["dwell_s"], music, [{"note": "C4", "best_guess": False, "ago_s": 3.1},
                                                         {"note": "E4", "best_guess": False, "ago_s": 2.2}])
msgs = [{"role": "system", "content": omni.DECIDE_PROMPT.replace("AVAILABLE_INSTRUMENTS", "/".join(music.available))},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()}},
            {"type": "text", "text": "COMMAND " + json.dumps(command)},
            {"type": "text", "text": "SCENE " + json.dumps(scene)}]}]
print(f"{client.model} via {client.url}\ncommand: {command}")
t0 = time.monotonic()
text = "".join(d for k, d in client.stream(msgs) if k == "text")
print(f"decide: {time.monotonic() - t0:.2f}s\n{text}\n")
reply = omni.parse_reply(text)
for a in reply["actions"]:
    ok = omni.fits(command, a, objs, music, sel.p["dwell_s"])
    print("action:", a, "| fits command:", ok, "->", music.apply(a, objs, sel) if ok else "(would fall back)")
if not args.no_voice:
    # Speak through the real routing (omni.Assistant._speak), so this checks the demo's voice path and
    # not a second one that only this script uses.
    from outer_vision.audio import Player
    p = Player()
    voice = Voice(p, Eleven(cfg), log=print)
    if o["speak_with"] not in ("omni", "none") and not voice.eleven.ok:
        print("warning: speak_with is", repr(o["speak_with"]), "but ELEVENLABS_API_KEY is not set; "
              "the demo will fall back to OMNI's voice")
    assistant = omni.Assistant(cfg, music, lambda: {}, p, log=print, client=client, voice=voice)
    t1 = time.monotonic()
    first = assistant._speak(reply["say"], reply.get("tone", "calm"))
    print(f"voice: spoke_with={assistant.spoke_with} (config says {o['speak_with']!r}), "
          f"first audio after {None if first is None else round(first - t1, 2)}s")

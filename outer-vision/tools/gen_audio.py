#!/usr/bin/env python3
"""Generate the demo's audio with ElevenLabs, ahead of time (so the demo never waits on venue Wi-Fi):

  assets/samples/<instrument>.wav   one real-sounding note per instrument (sound effects API); tools/synth.py
                                    measures its pitch and repitches it to every note
  assets/samples/fanfare.wav        lesson-complete chime
  assets/voice/<prompt>.wav         Maestro's blink-menu lines (text to speech)

  python tools/gen_audio.py              # only what's missing
  python tools/gen_audio.py --force      # regenerate everything (e.g. after changing a prompt or the voice)
  python tools/gen_audio.py --only piano,drum
Needs ELEVENLABS_API_KEY (env or .env).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from outer_vision import config  # noqa: E402
from outer_vision.audio import to_wav  # noqa: E402
from outer_vision.eleven import SR, Eleven, ElevenError  # noqa: E402
from outer_vision.menu import PROMPTS  # noqa: E402
from outer_vision.pitch import estimate_f0, trim  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = {   # instrument -> (prompt, seconds). "middle C" so the repitch distance stays small.
    "piano": ("A single grand piano note, middle C, struck once and left to ring, dry studio recording, no reverb, no other sounds", 2.0),
    "marimba": ("A single wooden marimba note, middle C, one soft mallet hit, dry studio recording, no other sounds", 1.5),
    "flute": ("A single sustained concert flute note, middle C, gentle breathy tone, dry studio recording, no melody", 2.0),
    "strings": ("A single sustained cello note, middle C, legato bow, warm, dry studio recording, no melody", 2.0),
    "bell": ("A single tubular bell note, middle C, one strike ringing out, dry studio recording", 2.5),
    "drum": ("A single deep tom drum hit, one stroke, dry studio recording, no other sounds", 1.0),
    "synth": ("A single warm analog synthesizer pluck note, middle C, dry, no effects, no melody", 1.5),
    "fanfare": ("A short cheerful success chime, three rising bell notes, bright, celebratory, clean", 1.5),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated sample or prompt names")
    args = ap.parse_args()
    config.load_env(ROOT / ".env")
    el = Eleven(config.load(str(ROOT / "config.json")))
    if not el.ok:
        sys.exit("set ELEVENLABS_API_KEY (env or .env) first")
    only = set(filter(None, args.only.split(",")))
    sdir, vdir = ROOT / "assets/samples", ROOT / "assets/voice"
    sdir.mkdir(parents=True, exist_ok=True)
    vdir.mkdir(parents=True, exist_ok=True)
    meta_path = sdir / "samples.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    failed = []

    for name, (prompt, secs) in SAMPLES.items():
        out = sdir / f"{name}.wav"
        if (only and name not in only) or (out.exists() and not args.force):
            continue
        try:
            x = trim(np.frombuffer(el.sfx(prompt, secs), "<i2"), SR)
        except ElevenError as e:
            failed.append(name)
            print(f"  {name}: FAILED {e}")
            continue
        out.write_bytes(to_wav(x, SR))
        f0 = None if name in ("drum", "fanfare") else estimate_f0(x, SR)
        meta[name] = {"f0": f0, "prompt": prompt}
        print(f"  {name}: {len(x) / SR:.2f}s  f0={'n/a' if f0 is None else f'{f0:.1f} Hz'} (middle C = 261.6)")

    for key, text in PROMPTS.items():
        out = vdir / f"{key}.wav"
        if (only and key not in only) or (out.exists() and not args.force):
            continue
        try:
            x = trim(np.frombuffer(el.tts(text), "<i2"), SR, keep_tail=True)
        except ElevenError as e:
            failed.append(key)
            print(f"  {key}: FAILED {e}")
            continue
        out.write_bytes(to_wav(x, SR))
        print(f"  {key}: {len(x) / SR:.2f}s  {text!r}")

    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print("done" if not failed else f"done, failed: {failed}")


if __name__ == "__main__":
    main()

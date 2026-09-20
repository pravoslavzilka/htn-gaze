"""What each object plays, runtime overrides (set from the blink menu), and guided lessons.

Mapping is keyed by "color/shape" (stable across tracker ID changes). Every change goes through
apply(), which validates it, so a bad model reply can never put the instrument in a broken state.
"""
from __future__ import annotations

import copy
import re

NOTE_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d)$")
SEMI = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def midi(note: str):
    m = NOTE_RE.match(note.strip())
    if not m:
        return None
    n = SEMI[m.group(1).upper()] + {"#": 1, "b": -1, "": 0}[m.group(2)]
    return n + 12 * (int(m.group(3)) + 1)


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SCALE_MIDI = [60, 62, 64, 67, 69]                       # C major pentatonic, C4 D4 E4 G4 A4 (the table)
SONGS = {   # fallback lessons when OMNI is unreachable. Every note must be in SCALE_MIDI, or the song
            # can't be played on the default table: no F and no B, which rules out Twinkle and Ode to Joy.
    "Mary Had a Little Lamb": ["E4", "D4", "C4", "D4", "E4", "E4", "E4", "D4", "D4", "D4", "E4", "G4", "G4"],
    "Hot Cross Buns": ["E4", "D4", "C4", "E4", "D4", "C4", "C4", "C4", "D4", "D4", "E4", "D4", "C4"],
    "Old MacDonald": ["C4", "C4", "C4", "G4", "A4", "A4", "G4", "E4", "E4", "D4", "D4", "C4"],
    "Jingle Bells": ["E4", "E4", "E4", "E4", "E4", "E4", "E4", "G4", "C4", "D4", "E4"],
}


def key_of(color, shape):
    return f"{color}/{shape}"


class Music:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        m = cfg["music"]
        self.notes = dict(m["notes"])                  # colour -> note name
        self.instruments = dict(m["instruments"])      # shape -> instrument
        self.available = list(m["available_instruments"])
        self.overrides = {}                             # "color/shape" -> {"note"?, "instrument"?}
        self.lesson = None                              # {"title", "notes", "index"}

    # ------------------------------------------------------------------ mapping
    def voice_of(self, color, shape) -> dict:
        o = self.overrides.get(key_of(color, shape), {})
        note = o.get("note", self.notes.get(color))
        return {"note": note, "midi": midi(note) if note else None,
                "instrument": o.get("instrument", self.instruments.get(shape, "piano"))}

    def snapshot(self, objects: list) -> list:
        """objects: output-JSON dicts from run.py; returns them annotated with note/instrument."""
        return [{**o, **self.voice_of(o["color"], o["shape"])} for o in objects]

    # ------------------------------------------------------------------ actions (from Maestro / the blink menu)
    def apply(self, action: dict, objects: dict, selector=None) -> str:
        """Apply one validated action. objects: id -> object dict. Returns a short log line."""
        kind = action.get("type")

        def resolve(target):
            if target in (None, "all"):
                return list(objects.values()) if target == "all" else []
            try:
                return [objects[int(target)]]
            except (ValueError, KeyError, TypeError):
                return []

        if kind == "set_instrument":
            inst = str(action.get("instrument", "")).lower()
            if inst not in self.available:
                return f"ignored: unknown instrument {inst!r}"
            objs = resolve(action.get("target"))
            for o in objs:
                self.overrides.setdefault(key_of(o["color"], o["shape"]), {})["instrument"] = inst
            return f"instrument {inst} -> {[o['id'] for o in objs]}" if objs else "ignored: no such object"
        if kind == "set_note":
            note = str(action.get("note", ""))
            if midi(note) is None:
                return f"ignored: bad note {note!r}"
            objs = resolve(action.get("target"))
            for o in objs:
                self.overrides.setdefault(key_of(o["color"], o["shape"]), {})["note"] = note
            return f"note {note} -> {[o['id'] for o in objs]}" if objs else "ignored: no such object"
        if kind == "swap_notes":
            a, b = resolve(action.get("a")), resolve(action.get("b"))
            if not a or not b:
                return "ignored: no such object"
            va, vb = self.voice_of(a[0]["color"], a[0]["shape"]), self.voice_of(b[0]["color"], b[0]["shape"])
            self.overrides.setdefault(key_of(a[0]["color"], a[0]["shape"]), {})["note"] = vb["note"]
            self.overrides.setdefault(key_of(b[0]["color"], b[0]["shape"]), {})["note"] = va["note"]
            return f"swapped {a[0]['id']} <-> {b[0]['id']}"
        if kind == "start_lesson":
            notes = [str(n) for n in action.get("notes", []) if midi(str(n)) is not None]
            playable = {self.voice_of(o["color"], o["shape"])["midi"] for o in objects.values()}
            missing = sorted({n for n in notes if midi(n) not in playable})
            notes = [n for n in notes if midi(n) in playable]
            if not notes:
                return "ignored: none of the lesson's notes are on the table"
            self.lesson = {"title": str(action.get("title", "lesson"))[:60], "notes": notes, "index": 0}
            return f"lesson {self.lesson['title']!r}: {len(notes)} notes" + (f", skipped {missing}" if missing else "")
        if kind == "stop_lesson":
            self.lesson = None
            return "lesson stopped"
        if kind == "set_dwell" and selector is not None:
            try:
                s = min(1.5, max(0.2, float(action.get("seconds"))))   # clamp: never unusably fast/slow
            except (TypeError, ValueError):
                return "ignored: bad dwell"
            selector.p["dwell_s"] = s
            return f"dwell {s:.2f}s"
        if kind == "reset":
            self.overrides.clear()
            self.lesson = None
            return "mapping reset"
        return f"ignored: unknown action {kind!r}"

    # ------------------------------------------------------------------ lessons
    def expected_note(self):
        if not self.lesson or self.lesson["index"] >= len(self.lesson["notes"]):
            return None
        return self.lesson["notes"][self.lesson["index"]]

    def on_lock(self, obj: dict):
        """Advance the lesson on a correct note. Returns lesson feedback dict or None."""
        exp = self.expected_note()
        if exp is None:
            return None
        ok = obj.get("note") is not None and midi(obj["note"]) == midi(exp)   # F#4 == Gb4
        if ok:
            self.lesson["index"] += 1
        done = self.lesson["index"] >= len(self.lesson["notes"])
        fb = {"title": self.lesson["title"], "correct": ok, "expected": exp,
              "index": self.lesson["index"], "total": len(self.lesson["notes"]), "done": done}
        if done:
            self.lesson = None
        return fb

    # ------------------------------------------------------------------ offline defaults
    def default_action(self, command: dict, objects: dict, dwell_s: float):
        """What a blink command does when OMNI can't be reached: (action, spoken line). Always valid."""
        kind, target = command.get("command"), command.get("target")
        o = objects.get(target)
        if kind == "change_instrument" and o is not None:
            cur = self.voice_of(o["color"], o["shape"])["instrument"]
            inst = self.available[(self.available.index(cur) + 1) % len(self.available)] if cur in self.available else self.available[0]
            return {"type": "set_instrument", "target": target, "instrument": inst}, f"That one plays {inst} now."
        if kind == "change_note" and o is not None:
            cur = midi(self.voice_of(o["color"], o["shape"])["note"] or "C4")
            nxt = next((m for m in SCALE_MIDI if m > cur), SCALE_MIDI[0])      # next C-major note, wraps
            note = NOTE_NAMES[nxt % 12] + str(nxt // 12 - 1)
            return {"type": "set_note", "target": target, "note": note}, f"That one plays {note} now."
        if kind == "faster":
            s = max(0.2, round(dwell_s * 0.75, 2))
            return {"type": "set_dwell", "seconds": s}, "A bit faster now."
        if kind == "slower":
            s = min(1.5, round(dwell_s * 1.33, 2))
            return {"type": "set_dwell", "seconds": s}, "A bit slower now."
        if kind == "teach":
            playable = {self.voice_of(v["color"], v["shape"])["midi"] for v in objects.values()}
            title, notes = max(SONGS.items(), key=lambda kv: sum(midi(n) in playable for n in kv[1]) / len(kv[1]))
            return {"type": "start_lesson", "title": title, "notes": notes}, f"Let's learn {title}. Follow the circle."
        return None, "Sorry, I couldn't do that."

    def state(self):
        return {
            "overrides": copy.deepcopy(self.overrides),
            "lesson": None if not self.lesson else {**self.lesson, "next": self.expected_note()},
        }

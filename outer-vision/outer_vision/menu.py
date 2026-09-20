"""Blink menu: the only way to reconfigure the instrument (no microphone, no hands).

  long blink on an object   -> "Left eye: new instrument. Right eye: new note. Both eyes: swap it."
  long blink on empty table -> "Left eye: slower. Right eye: faster. Both eyes: teach me a song."
                               (during a lesson, both eyes = stop the lesson)
  answer with a long blink of the matching eye(s); a double blink cancels; silence closes the menu.

Pure logic: speaking, OMNI requests and local actions go out through callbacks, so it's unit-testable.
Notes are not played while the menu is open (see `active`).
"""
from __future__ import annotations

PROMPTS = {
    "menu_object": "Left eye, new instrument. Right eye, new note. Both eyes, swap it.",
    "menu_space": "Left eye, slower. Right eye, faster. Both eyes, teach me a song.",
    "menu_space_lesson": "Left eye, slower. Right eye, faster. Both eyes, stop the lesson.",
    "swap_pick": "Look at the other one, and blink.",
    "swapped": "Swapped.",
    "cancelled": "Okay, never mind.",
    "lesson_stopped": "Lesson stopped.",
    "thinking": "One moment.",
}

OPTIONS = {   # shown on the overlay / state stream
    "object": {"left": "new instrument", "right": "new note", "both": "swap"},
    "space": {"left": "slower", "right": "faster", "both": "teach a song"},
    "space_lesson": {"left": "slower", "right": "faster", "both": "stop lesson"},
    "swap_pick": {"left": "swap with this", "right": "swap with this", "both": "swap with this"},
}


class Menu:
    def __init__(self, cfg: dict, say, request, local):
        """say(prompt_key); request(command dict) -> OMNI; local(action dict) -> Music.apply."""
        self.p = cfg["menu"]
        self.say, self.request, self.local = say, request, local
        self.state = None           # None | "object" | "space" | "space_lesson" | "swap_pick"
        self.focus = None           # object id the menu is about
        self.opened_at = 0.0

    @property
    def active(self) -> bool:
        return self.state is not None

    def options(self):
        return OPTIONS.get(self.state)

    def _open(self, state, focus, now, prompt):
        self.state, self.focus, self.opened_at = state, focus, now
        self.say(prompt)

    def _close(self, prompt=None):
        self.state = self.focus = None
        if prompt:
            self.say(prompt)

    def on_gesture(self, g: dict, focus, lesson_active: bool, now: float):
        """g: gesture from BlinkDetector; focus: object id under gaze (None = empty table)."""
        if g["kind"] == "double":
            if self.active:
                self._close("cancelled")
            return
        side = g["side"]
        if self.state is None:
            if focus is not None:
                self._open("object", focus, now, "menu_object")
            elif lesson_active:
                self._open("space_lesson", None, now, "menu_space_lesson")
            else:
                self._open("space", None, now, "menu_space")
        elif self.state == "object":
            target = self.focus
            if side == "both":
                self._open("swap_pick", target, now, "swap_pick")
                return
            self._close("thinking")
            self.request({"command": "change_instrument" if side == "left" else "change_note", "target": target})
        elif self.state in ("space", "space_lesson"):
            if side == "left":
                self._close("thinking")
                self.request({"command": "slower"})
            elif side == "right":
                self._close("thinking")
                self.request({"command": "faster"})
            elif self.state == "space_lesson":
                self.local({"type": "stop_lesson"})
                self._close("lesson_stopped")
            else:
                self._close("thinking")
                self.request({"command": "teach"})
        elif self.state == "swap_pick":
            if focus is None or focus == self.focus:
                self.say("swap_pick")            # still waiting for the other object
                self.opened_at = now
                return
            r = self.local({"type": "swap_notes", "a": self.focus, "b": focus})
            self._close("cancelled" if str(r).startswith("ignored") else "swapped")

    def tick(self, now: float):
        if self.active and now - self.opened_at > self.p["timeout_s"]:
            self._close("cancelled")

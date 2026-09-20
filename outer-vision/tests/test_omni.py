"""Maestro (blink menu -> OMNI -> speech) and music logic, against a mock OpenAI-compatible OMNI server
(no key, no network, no audio device)."""
import base64
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from outer_vision import config, omni
from outer_vision.blink import BlinkDetector
from outer_vision.menu import Menu
from outer_vision.music import Music, midi
from outer_vision.selector import Selector
from tests.test_pipeline import track_at

CFG = config.load(None)
OBJECTS = {                                                # the default pentatonic table
    1: {"id": 1, "color": "red", "shape": "round"},         # C4 marimba
    4: {"id": 4, "color": "green", "shape": "cylinder"},    # E4 flute
    5: {"id": 5, "color": "blue", "shape": "square"},       # G4 piano
    7: {"id": 7, "color": "yellow", "shape": "triangle"},   # D4 bell
}


class MockOmni:
    """Streams SSE like Qwen-Omni: text deltas; audio deltas when modalities include audio."""

    def __init__(self, reply, fail_text=False, fail_audio=False):
        self.reply, self.fail_text, self.fail_audio, self.requests = reply, fail_text, fail_audio, []
        self.error_chunk = None          # like the live gateway: HTTP 200, then data: {"error": ...}
        mock = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                mock.requests.append(body)
                audio = "audio" in body.get("modalities", [])
                if (audio and mock.fail_audio) or (not audio and mock.fail_text):
                    self.send_response(500)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                if mock.error_chunk:
                    self.wfile.write(b"data: " + json.dumps({"error": mock.error_chunk}).encode() + b"\n\ndata: [DONE]\n\n")
                    return
                if audio:
                    pcm = (np.sin(np.arange(2400) / 10) * 8000).astype("<i2").tobytes()
                    chunks = [{"audio": {"data": base64.b64encode(pcm).decode()}}] * 3
                else:
                    r = mock.reply.pop(0) if isinstance(mock.reply, list) else mock.reply   # list = one per request
                    txt = r if isinstance(r, str) else "```json\n" + json.dumps(r) + "\n```"
                    chunks = [{"content": txt[i:i + 20]} for i in range(0, len(txt), 20)]
                for c in chunks:
                    self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": c}]}).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1"


class FakePlayer:
    def __init__(self):
        self.data, self.playing, self.stops = b"", False, 0

    def feed(self, b):
        self.data += b

    def stop(self):
        self.stops += 1


class FakeVoice:
    """Stands in for outer_vision.voice.Voice. can_speak False = no ElevenLabs key or no speaker."""

    def __init__(self, can_speak=True):
        self.said, self.tones, self.cuts, self.can_speak = [], [], [], can_speak

    def say(self, text, tone=None, cut=False):
        self.said.append(text)
        self.tones.append(tone)
        self.cuts.append(cut)
        return None if not self.can_speak else 1234.5   # monotonic time of first audio


def make_assistant(reply, fail_text=False, fail_audio=False, key="test-key", speak_with="eleven",
                   can_speak=True):
    mock = MockOmni(reply, fail_text, fail_audio)
    cfg = config.load(None)
    cfg["omni"]["speak_with"] = speak_with
    music, player, voice, published = Music(cfg), FakePlayer(), FakeVoice(can_speak), []
    sel = Selector(config.load(None))
    ctx = lambda: {"jpeg": b"\xff\xd8fakejpeg", "selector": sel, "recent": [{"note": "C4", "ago_s": 1.2}],  # noqa: E731
                   "objects": {k: {**v, **music.voice_of(v["color"], v["shape"])} for k, v in OBJECTS.items()}}
    client = omni.OmniClient(mock.url, key, "qwen3.5-omni-flash", "Serena")
    a = omni.Assistant(cfg, music, ctx, player, published.append, log=lambda m: None, client=client, voice=voice)
    return a, mock, music, player, voice, published, sel


class TestAssistant(unittest.TestCase):
    def test_omni_picks_the_instrument_for_the_blinked_object(self):
        reply = {"say": "That tall one sounds like strings now.", "tone": "cheerful",
                 "actions": [{"type": "set_instrument", "target": 4, "instrument": "strings"}]}
        a, mock, music, player, voice, pub, _ = make_assistant(reply)
        a.handle({"command": "change_instrument", "target": 4})
        decide = mock.requests[0]
        parts = [p for p in decide["messages"][-1]["content"]]
        self.assertEqual([p["type"] for p in parts], ["image_url", "text", "text"])
        self.assertNotIn("input_audio", json.dumps(decide))                            # no microphone anywhere
        self.assertEqual(json.loads(parts[1]["text"][len("COMMAND "):]),
                         {"command": "change_instrument", "target": 4, "reply_with_action_type": "set_instrument"})
        scene = json.loads(parts[2]["text"][len("SCENE "):])
        self.assertEqual((scene["dwell_s"], scene["recent"][0]["note"]), (0.5, "C4"))
        self.assertEqual(scene["notes_on_table"], ["C4", "D4", "E4", "G4"])            # pentatonic, in order
        self.assertEqual(decide["modalities"], ["text"])
        self.assertEqual(music.voice_of("green", "cylinder")["instrument"], "strings")
        self.assertEqual(music.voice_of("red", "round")["instrument"], "marimba")     # untouched
        self.assertEqual((pub[0]["type"], pub[0]["source"]), ("assistant", "omni"))
        # Spoken by ElevenLabs, in the tone OMNI asked for, cutting off the "One moment" clip.
        self.assertEqual(len(mock.requests), 1)                                        # no second OMNI call
        self.assertEqual(voice.said, ["That tall one sounds like strings now."])
        self.assertEqual((voice.tones, voice.cuts), (["cheerful"], [True]))
        self.assertEqual(a.last_metrics["voice"], "eleven")

    def test_speak_with_omni_uses_the_models_own_voice(self):
        reply = {"say": "Strings it is.", "tone": "cheerful",
                 "actions": [{"type": "set_instrument", "target": 4, "instrument": "strings"}]}
        a, mock, music, player, voice, pub, _ = make_assistant(reply, speak_with="omni")
        a.handle({"command": "change_instrument", "target": 4})
        self.assertEqual(mock.requests[1]["modalities"], ["text", "audio"])
        self.assertEqual(len(player.data), 3 * 4800)
        self.assertEqual(voice.said, [])
        self.assertEqual(a.last_metrics["voice"], "omni")

    def test_no_elevenlabs_key_falls_back_to_omnis_voice(self):
        reply = {"say": "Strings it is.", "actions": [{"type": "set_instrument", "target": 4, "instrument": "strings"}]}
        a, mock, music, player, voice, pub, _ = make_assistant(reply, can_speak=False)
        a.handle({"command": "change_instrument", "target": 4})
        self.assertEqual(voice.said, [])                       # never even attempted
        self.assertEqual(mock.requests[1]["modalities"], ["text", "audio"])
        self.assertEqual(a.last_metrics["voice"], "omni")

    def test_no_voice_at_all_falls_back_to_local_tts(self):
        """Every engine down: the user still hears the answer, and hears it exactly once."""
        reply = {"say": "Strings it is.", "actions": [{"type": "set_instrument", "target": 4, "instrument": "strings"}]}
        a, mock, music, player, voice, pub, _ = make_assistant(reply, fail_audio=True, can_speak=False)
        spoken = []
        import outer_vision.voice as vmod
        real, vmod.local_say = vmod.local_say, spoken.append
        try:
            a.handle({"command": "change_instrument", "target": 4})
        finally:
            vmod.local_say = real
        self.assertEqual(spoken, ["Strings it is."])
        self.assertEqual(voice.said, [])
        self.assertEqual(a.last_metrics["voice"], "local")

    def test_action_that_does_not_fit_the_command_falls_back(self):
        # asked for "faster" but OMNI changed an instrument and slowed down: neither may be applied
        reply = {"say": "Slower!", "actions": [{"type": "set_instrument", "target": 1, "instrument": "drum"},
                                               {"type": "set_dwell", "seconds": 0.9}]}
        a, mock, music, player, voice, pub, sel = make_assistant(reply)
        a.handle({"command": "faster"})
        self.assertEqual(music.overrides, {})
        self.assertAlmostEqual(sel.p["dwell_s"], 0.38)                                # offline default: x0.75
        self.assertEqual(pub[0]["source"], "fallback")
        self.assertEqual(voice.said, ["A bit faster now."])                          # our own line, our own voice
        self.assertEqual(len(mock.requests), 2)                                       # one retry, no speech call
        self.assertIn("faster means fewer seconds", mock.requests[1]["messages"][-1]["content"])

    def test_retry_with_the_reason_fixes_a_bad_reply(self):
        # Twinkle needs F4 and A4; a pentatonic table has neither, so the first reply can't be played.
        bad = {"say": "Twinkle!", "actions": [{"type": "start_lesson", "title": "Twinkle Twinkle",
                                               "notes": ["C4", "C4", "G4", "G4", "A4", "A4", "G4", "F4"]}]}
        good = {"say": "Hot Cross Buns!", "actions": [{"type": "start_lesson", "title": "Hot Cross Buns",
                                                      "notes": ["E4", "D4", "C4", "E4", "D4", "C4"]}]}
        a, mock, music, player, voice, pub, _ = make_assistant([bad, good])
        a.handle({"command": "teach"})
        told = mock.requests[1]["messages"][-1]["content"]
        self.assertIn("'A4'", told)                                                   # told exactly what was wrong
        self.assertIn("'F4'", told)
        self.assertEqual((pub[0]["source"], music.lesson["title"]), ("omni", "Hot Cross Buns"))

    def test_omni_down_still_changes_the_note(self):
        a, mock, music, player, voice, pub, _ = make_assistant({}, fail_text=True)
        a.handle({"command": "change_note", "target": 1})
        self.assertEqual(music.voice_of("red", "round")["note"], "D4")               # next note up the pentatonic
        self.assertEqual(voice.said, ["That one plays D4 now."])

    def test_no_key_skips_the_network(self):
        a, mock, music, *_ = make_assistant({}, key="")
        a.handle({"command": "teach"})
        self.assertEqual(mock.requests, [])
        self.assertIsNotNone(music.lesson)

    def test_error_inside_a_200_stream_falls_back(self):
        a, mock, music, player, voice, pub, _ = make_assistant({})
        mock.error_chunk = {"message": "<400> InternalError.Algo.InvalidParameter: Voice 'Cherry' is not supported."}
        a.handle({"command": "change_instrument", "target": 1})
        self.assertEqual(pub[0]["source"], "fallback")
        self.assertEqual(music.voice_of("red", "round")["instrument"], "flute")      # default: next instrument
        self.assertEqual(voice.said, ["That one plays flute now."])

    def test_omni_voice_failure_still_speaks(self):
        """speak_with omni, but the audio call fails: ElevenLabs picks the sentence up."""
        reply = {"say": "Hello there", "actions": [{"type": "set_dwell", "seconds": 0.8}]}
        a, _, _, _, voice, _, sel = make_assistant(reply, fail_audio=True, speak_with="omni")
        a.handle({"command": "slower"})
        self.assertEqual(sel.p["dwell_s"], 0.8)
        self.assertEqual(voice.said, ["Hello there"])
        self.assertEqual(a.last_metrics["voice"], "eleven")


class TestMusic(unittest.TestCase):
    def objs(self, music):
        return {k: {**v, **music.voice_of(v["color"], v["shape"])} for k, v in OBJECTS.items()}

    def test_bad_actions_are_ignored(self):
        music, sel = Music(CFG), Selector(config.load(None))
        objs = self.objs(music)
        self.assertTrue(music.apply({"type": "set_instrument", "target": 4, "instrument": "kazoo"}, objs).startswith("ignored"))
        self.assertTrue(music.apply({"type": "set_instrument", "target": 99, "instrument": "drum"}, objs).startswith("ignored"))
        self.assertTrue(music.apply({"type": "rm -rf"}, objs).startswith("ignored"))
        music.apply({"type": "set_dwell", "seconds": 0.01}, objs, sel)
        self.assertEqual(sel.p["dwell_s"], 0.2)                                      # clamped
        self.assertEqual(music.overrides, {})

    def test_lesson_uses_only_notes_on_table_and_matches_enharmonics(self):
        music = Music(CFG)
        objs = self.objs(music)                                                      # C4, E4, G4, D4
        music.apply({"type": "set_note", "target": 4, "note": "F#4"}, objs)           # table: C4 F#4 G4 D4
        objs = self.objs(music)
        r = music.apply({"type": "start_lesson", "title": "test", "notes": ["C4", "C4", "A4", "Gb4", "G4"]}, objs)
        self.assertIn("skipped ['A4']", r)                                            # A4 is not on this table
        seq = [objs[1], objs[5], objs[1], objs[4], objs[5]]     # right, wrong, right, right (F#4 = Gb4), right
        fbs = [music.on_lock(o) for o in seq]
        self.assertEqual([f["correct"] for f in fbs], [True, False, True, True, True])
        self.assertTrue(fbs[-1]["done"])
        self.assertIsNone(music.lesson)

    def test_default_teach_picks_the_most_playable_song(self):
        music = Music(CFG)
        objs = {i: {"id": i, "color": c, "shape": "round"} for i, c in enumerate(["red", "yellow", "green"])}  # C D E
        action, say = music.default_action({"command": "teach"}, objs, 0.5)
        self.assertEqual(action["title"], "Hot Cross Buns")     # the only fallback song that needs just C D E
        self.assertTrue(set(action["notes"]) <= {"C4", "D4", "E4"})

    def test_parse_reply_and_midi(self):
        r = omni.parse_reply('Sure!\n```json\n{"say": "hi", "actions": "oops"}\n```')
        self.assertEqual((r["say"], r["actions"], r["tone"]), ("hi", [], "calm"))
        # seen from the live model: one action as an object, typed with the command name
        r = omni.parse_reply('{"say": "ok", "actions": {"type": "change_instrument", "target": 4, "instrument": "bell"}}')
        self.assertEqual(r["actions"], [{"type": "set_instrument", "target": 4, "instrument": "bell"}])
        self.assertEqual((midi("C4"), midi("A4"), midi("F#4"), midi("Bb3"), midi("H2")), (60, 69, 66, 58, None))


class TestBlink(unittest.TestCase):
    def run_eyes(self, segments, hz=60):
        """segments: (seconds, left_closed, right_closed) -> gestures."""
        det, t, out = BlinkDetector(CFG["blink"]), 0.0, []
        for secs, lc, rc in segments:
            for _ in range(int(round(secs * hz))):
                out += det.feed(t, lc, rc)
                t += 1 / hz
        return out

    def test_natural_blink_is_ignored(self):
        self.assertEqual(self.run_eyes([(1, 0, 0), (0.15, 1, 1), (1, 0, 0)]), [])

    def test_long_blinks_and_winks(self):
        for lc, rc, side in [(1, 1, "both"), (1, 0, "left"), (0, 1, "right")]:
            g = self.run_eyes([(0.5, 0, 0), (0.9, lc, rc), (0.2, 0, 0)])
            self.assertEqual([(x["kind"], x["side"]) for x in g], [("long", side)])

    def test_wink_survives_a_brief_squint_of_the_other_eye(self):
        g = self.run_eyes([(0.5, 0, 0), (0.1, 1, 1), (0.7, 1, 0), (0.2, 0, 0)])
        self.assertEqual([x["side"] for x in g], ["left"])

    def test_double_blink(self):
        g = self.run_eyes([(0.5, 0, 0), (0.12, 1, 1), (0.25, 0, 0), (0.12, 1, 1), (0.5, 0, 0)])
        self.assertEqual([x["kind"] for x in g], ["double"])

    def test_resting_eyes_and_half_blinks_are_ignored(self):
        self.assertEqual(self.run_eyes([(0.5, 0, 0), (3.0, 1, 1), (0.5, 0, 0)]), [])
        self.assertEqual(self.run_eyes([(0.5, 0, 0), (0.5, 1, 1), (0.5, 0, 0)]), [])

    def test_tracker_dropout_mid_closure_is_dropped(self):
        det = BlinkDetector(CFG["blink"])
        det.feed(0.0, True, True)
        det.feed(0.1, True, True)
        self.assertEqual(det.feed(0.9, False, False), [])     # 0.8 s gap with no samples


class TestMenu(unittest.TestCase):
    def setUp(self):
        self.said, self.requests, self.local = [], [], []
        self.m = Menu(CFG, self.said.append, self.requests.append, lambda a: self.local.append(a) or "ok")

    def blink(self, side, focus, lesson=False, now=0.0):
        self.m.on_gesture({"kind": "long", "side": side}, focus, lesson, now)

    def test_object_menu_left_asks_omni_for_an_instrument(self):
        self.blink("both", 4)
        self.assertEqual((self.m.state, self.said), ("object", ["menu_object"]))
        self.blink("left", None)                        # answer while looking elsewhere: still about object 4
        self.assertEqual(self.requests, [{"command": "change_instrument", "target": 4}])
        self.assertFalse(self.m.active)

    def test_space_menu(self):
        self.blink("left", None)
        self.blink("right", None)
        self.blink("both", None)
        self.blink("both", None)
        self.assertEqual(self.requests, [{"command": "faster"}, {"command": "teach"}])

    def test_stop_lesson_is_local(self):
        self.blink("both", None, lesson=True)
        self.assertEqual(self.said, ["menu_space_lesson"])
        self.blink("both", None, lesson=True)
        self.assertEqual((self.local, self.requests), ([{"type": "stop_lesson"}], []))

    def test_swap_needs_a_second_object(self):
        self.blink("both", 1)
        self.blink("both", 1)                           # -> swap: now pick the other one
        self.blink("left", 1)                           # same object: keep waiting
        self.blink("right", None)                       # nothing: keep waiting
        self.assertEqual(self.m.state, "swap_pick")
        self.blink("right", 5)
        self.assertEqual(self.local, [{"type": "swap_notes", "a": 1, "b": 5}])
        self.assertEqual(self.said[-1], "swapped")

    def test_double_blink_cancels_and_timeout_closes(self):
        self.blink("both", 4)
        self.m.on_gesture({"kind": "double"}, 4, False, 1.0)
        self.assertFalse(self.m.active)
        self.blink("both", 4, now=10.0)
        self.m.tick(10.0 + CFG["menu"]["timeout_s"] + 0.1)
        self.assertFalse(self.m.active)
        self.assertEqual(self.said.count("cancelled"), 2)
        self.assertEqual(self.requests, [])


class TestSelectorHold(unittest.TestCase):
    def test_closed_eyes_freeze_dwell_and_keep_target(self):
        s, a = Selector(config.load(None)), track_at(1, 100, 100)
        dt, t, locks = 1 / 30, 0.0, []
        for _ in range(10):                             # 0.33 s of dwell
            locks += s.update([a], (100, 100), t, 640)
            t += dt
        for _ in range(30):                             # 1 s long blink: no gaze, no lock, target kept
            s.hold(t)
            t += dt
        self.assertEqual((locks, s.target_id), ([], 1))
        n = 0
        while not locks:
            locks += s.update([a], (100, 100), t, 640)
            t += dt
            n += 1
        self.assertLessEqual(abs(n - 6), 1)            # only the remaining ~0.17 s of dwell


if __name__ == "__main__":
    unittest.main()

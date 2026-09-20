"""Config: built-in defaults, optionally overridden by a JSON file (deep-merged)."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

DEFAULTS = {
    # Frames are resized to this width before processing. Every *_frac value below is a
    # fraction of this width, so tuning survives resolution changes.
    "process_width": 640,
    # Colour prototypes, OpenCV HSV (H 0-180, S/V 0-255): each pixel goes to the NEAREST prototype.
    # Register real values per object with tools/tune_colors.py (click the object). Five well-separated
    # hues, in rainbow order = the C major pentatonic scale (see "music"). Five rather than eight because
    # neighbouring hues (red/orange, blue/cyan, purple/pink) are what the colour LUT confuses under venue
    # lighting, and because a pentatonic scale has no semitone clashes: any order of looks sounds musical.
    # "bgr" is only the debug-overlay colour.
    "colors": {
        "red":    {"hsv": [0, 200, 170],   "bgr": [40, 40, 230]},
        "yellow": {"hsv": [27, 200, 210],  "bgr": [40, 220, 240]},
        "green":  {"hsv": [62, 170, 150],  "bgr": [60, 200, 60]},
        "blue":   {"hsv": [110, 200, 170], "bgr": [230, 120, 40]},
        "purple": {"hsv": [135, 150, 140], "bgr": [180, 60, 150]},
    },
    "color_match": {
        "s_min": 90,        # below this saturation = table/glare/shadow, never an object
        "v_min": 60,        # below this brightness = shadow/black
        "hue_tol": 8,       # hue units that count as distance 1.0
        "sat_tol": 90,      # saturation units that count as distance 1.0
        "max_dist": 1.7,    # pixels farther than this from every prototype are ignored
    },
    "shape_net": {
        "enabled": True,
        "model_dir": "models/shape",
        "reject_min_prob": 0.6,     # drop a candidate (hand, scrap, pen) above this "reject" probability
    },
    "detector": {
        "blur": 5,                    # Gaussian kernel (odd) before HSV; 0 disables
        "morph": 5,                   # open/close kernel size in px
        "min_area_frac": 0.0015,      # contour area / frame area
        "max_area_frac": 0.25,
        "poly_eps": 0.02,             # approxPolyDP epsilon, fraction of hull perimeter
        "round_min_circlefill": 0.86, # area / min-enclosing-circle area (circle~1, hexagon .83, square .64)
        "square_max_aspect": 1.5,
        "square_min_rectfill": 0.90,  # area / min-area-rect area (square~1, circle .785)
        "square_max_vertices": 6,
        "square_min_solidity": 0.90,
        "cylinder_min_aspect": 1.25,
        # A triangle inscribed in its circumcircle fills ~0.41 of it, far below a circle (1.0), a square
        # (~0.72) or a cylinder (>=0.62), and it fills little of its bounding rect. Vertex count is NOT a
        # usable signal: morphology rounds the apex, so real triangles come out with 4-5 hull vertices.
        "triangle_max_circlefill": 0.62,
        "triangle_max_rectfill": 0.81,
        "border_margin": 3,           # px; contours touching the frame edge are flagged partial
    },
    "tracker": {
        "max_jump_frac": 0.15,        # max centroid move between frames to count as same object
        "max_missing_s": 0.5,         # drop a track after this long unseen
        "confirm_hits": 3,            # frames seen before a track is reported
        "smooth": 0.5,                # EMA weight on the new centroid (1 = no smoothing)
        "shape_votes": 9,             # majority vote window for shape label
    },
    "selector": {
        "dwell_s": 0.5,               # look this long -> lock (= play note once)
        "grace_s": 0.15,              # gaze may leave/blink this long without resetting dwell
        "select_radius_frac": 0.03,   # gaze within this distance of an outline = confident hit
        "best_guess_radius_frac": 0.12,  # beyond select radius but within this = best guess
        "switch_margin_frac": 0.02,   # a new object must be this much closer to steal the target
        "gaze_max_age_s": 0.2,        # older gaze samples are treated as invalid
    },
    "depth": {
        # Filled by pressing 'c' in run.py with objects at a known distance (--ref-distance).
        "ref_distance_cm": None,
        "ref_size": {},               # "color/shape" or "shape" -> sqrt(area)/width at ref distance
        "near_cm": 40.0,              # volume 1.0 at/inside this
        "far_cm": 100.0,              # volume min_volume at/beyond this
        "min_volume": 0.2,
    },
    "music": {
        # C major pentatonic: C D E G A. No semitones, so no two objects can clash.
        "notes": {"red": "C4", "yellow": "D4", "green": "E4", "blue": "G4", "purple": "A4"},
        "instruments": {"round": "marimba", "square": "piano", "cylinder": "flute", "triangle": "bell"},
        "available_instruments": ["piano", "marimba", "flute", "strings", "bell", "drum", "synth"],
    },
    "omni": {
        # OpenAI-compatible endpoint serving an OMNI model. Key from env OMNI_API_KEY, never from this file.
        "base_url": "https://yibuapi.com/v1",
        "model": "qwen3.5-omni-flash",
        "voice": "Serena",
        # Who speaks Maestro's replies. "eleven": ElevenLabs, the same voice as the menu prompt clips,
        # streamed (default). "omni": the model reads its own reply back, one round trip fewer but a
        # second voice in the demo. "none": captions only. Whichever is chosen, the other engine and then
        # local TTS stand behind it. ("fallback" is the old name for "eleven" and still works.)
        "speak_with": "eleven",
        "timeout_s": 10.0,             # after this the command falls back to the offline default
        "retries": 1,                  # a reply that doesn't fit the command is sent back once with the reason
        "history_turns": 6,
    },
    "blink": {
        # Deliberate blinks from the eye tracker's eye state (INTERFACE.md). Natural blinks are ~0.1-0.4 s.
        "short_max_s": 0.4,            # shorter = natural blink (two of them = double blink = cancel)
        "long_min_s": 0.6,             # long blink / wink = select
        "long_max_s": 2.0,             # longer = resting the eyes, ignored
        "double_gap_s": 0.7,           # max gap between the two blinks of a double blink
        "side_frac": 0.7,              # an eye counts as closed if closed for this share of the closure
        "max_sample_gap_s": 0.25,      # tracker silent this long mid-closure -> drop the closure
    },
    "menu": {
        "timeout_s": 8.0,              # menu closes if nothing is chosen
    },
    "eleven": {
        # ElevenLabs, key from env ELEVENLABS_API_KEY. Used by tools/gen_audio.py (samples, prompts) and as
        # live fallback speech. voice_id: premade "Rachel"; override with env ELEVENLABS_VOICE_ID.
        "voice_id": "21m00Tcm4TlvDq8ikWAM",
        "tts_model": "eleven_flash_v2_5",
    },
    "qnx": {
        # The eye-tracking half: a Raspberry Pi 5 running QNX 8.0 with two Camera Module 3 cameras
        # (htn-gaze repo, branch pupil-in-eye). `camera_streamer` is C/C++ on the board and serves the eye
        # camera with MediaPipe Face Mesh + the dark-pupil fit on :8080, and the scene camera as video
        # only on :8081. run.py polls the first and reads the second; nothing of this repo runs on QNX.
        "host": "192.168.2.2",
        "eye_port": 8080,              # GET /api/state, GET /api/frame.jpg
        "scene_port": 8081,            # GET /stream.mjpg  (camera_streamer --no-infer)
        "poll_hz": 30.0,               # the board infers at 10-30 fps; polling faster just repeats frames
        "timeout_s": 0.5,
        # Fixed-rig geometry, in millimetres/degrees. Mirrors DEFAULT_RIG in the eye repo's
        # gui/src/geometry.js: keep the two the same or the GUI's reticle and our notes disagree.
        "rig": {
            "eye_distance_mm": 80.0,   # eye camera to the corneas
            "baseline_mm": 25.0,       # scene camera behind the eye camera
            "eyeball_radius_mm": 12.0,
            "hfov_deg": 66.0,          # Camera Module 3 standard lens
            "scene_depth_mm": 1500.0,  # plane the gaze ray is intersected with
            "kappa_deg": 5.0,          # visual axis vs pupillary axis
            "flip_x": False,           # set if a camera is mounted inverted (GUI keys X / Y)
            "flip_y": False,
            "scene_w": 960,            # start_streamers.sh runs both cameras at 960x540
            "scene_h": 540,
            "zero_yaw_deg": 0.0,       # residual aim; measure with tools/qnx_bridge.py --zero
            "zero_pitch_deg": 0.0,
        },
        # Eyelid openness (height/width of the eyelid outline) -> eye closed. Depends on the wearer and on
        # where the eye camera sits: read live values with tools/qnx_bridge.py --probe. Two thresholds, so
        # an eye resting near the boundary can't chatter and break one long blink into several short ones.
        "lid": {
            "closed_below": 0.15,
            "open_above": 0.20,
            "min_open_for_gaze": 0.08,  # below this an eye is left out of the two-eye gaze average
        },
        "smooth": {"median": 5, "ema": 0.45},   # same filter as the eye repo's gui/src/useGaze.js
    },
    "gaze_udp_port": 5005,
    "events": {"host": "127.0.0.1", "port": 5006},
    "calib_marker": {"dictionary": "DICT_4X4_50"},
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "colors":
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)  # colours replace wholesale so removed colours stay removed
    return out


def load_env(path=".env") -> None:
    """KEY=value lines into os.environ (existing variables win). Keeps API keys out of config.json and git."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip().removeprefix("export ").strip(), v.strip().strip("'\""))


def load(path: str | None) -> dict:
    if path and Path(path).exists():
        with open(path) as f:
            user = json.load(f)
        cols = user.get("colors", {})
        if any(isinstance(c.get("hsv", [None])[0], list) for c in cols.values()):
            print(f"[config] {path}: v0 colour ranges found; using v1 colour prototypes instead", flush=True)
            user.pop("colors")
        return _merge(DEFAULTS, user)
    return copy.deepcopy(DEFAULTS)


def save(cfg: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

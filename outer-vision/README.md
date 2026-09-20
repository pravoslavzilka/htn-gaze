# Outer Vision

**An instrument you play with your eyes.** Look at a coloured object on the table and it plays a note:
**colour → pitch, shape → instrument, farther → quieter**. Reconfigure it with **blinks**: a long blink
opens a spoken menu from **Maestro** (Huawei's OMNI model, speaking in an ElevenLabs voice), and a long
wink of the left eye, the right eye, or both picks an option. OMNI looks at the table and decides the
details (which instrument, which note, which song, how much faster). No hands, no voice: built for people
with ALS or paralysis, fun for anyone.

This repo is the **world-camera ("outer") half**: objects, gaze target, notes, blink menu, Maestro. The
eye half is a **Raspberry Pi 5 running QNX 8.0** ([htn-gaze](https://github.com/pranaycv/htn-gaze), branch
`pupil-in-eye`): a C/C++ `camera_streamer` runs MediaPipe Face Mesh and a dark-pupil fit on the board and
serves the eye camera on `:8080` and the scene camera on `:8081`. Nothing in this repo runs on QNX;
`run.py` runs on the laptop and reads both (`--source qnx --gaze qnx`). Any other tracker can send gaze
over UDP instead (INTERFACE.md).

- **Setting it up, running it, testing it, training the net, saving your work: [docs/SETUP.md](docs/SETUP.md)** — start here
- Interfaces (gaze + eye state in, events out): **[INTERFACE.md](INTERFACE.md)**
- Shortcuts and their proper versions: **[COMPROMISES.md](COMPROMISES.md)** · Decisions, prizes: **[SPEC.md](SPEC.md)**

## How it works
```
QNX board :8081 scene camera ─MJPEG─► run.py on the laptop
QNX board :8080 eye camera   ─JSON──► pupil + eyelids ─► rig model ─► gaze + blinks
  0. gaze     /api/state → coaxial rig model (gaze_model.py) → a point in the scene image  WHERE I LOOK
  1. colour   every pixel → nearest registered colour (lookup table, ~1 ms)                 WHERE
  2. shape    crop → CNN (ONNX, OpenCV) → round/square/cylinder/triangle/reject(hands)      WHAT
  3. track    stable IDs, shape vote, size → distance → volume
  4. select   gaze → nearest outline → 0.5 s dwell → LOCK = note (colour/shape → note/instrument)
  5. blinks   eyelid openness → long blink / left wink / right wink / double blink (closed eyes freeze dwell)
  6. Maestro  blink menu → command → OMNI (camera frame + scene + recent notes) → one validated action
              → spoken by ElevenLabs. Offline defaults if OMNI is unreachable.
  ─► UDP events → tools/synth.py (ElevenLabs-generated instrument samples) / game   ─► MJPEG overlay (--stream)
```

### The blink menu
| Look at… and long-blink (0.6–2 s) | Maestro says | Left eye | Right eye | Both eyes |
|---|---|---|---|---|
| an object | *"Left eye, new instrument. Right eye, new note. Both eyes, swap it."* | OMNI picks a new instrument | OMNI picks a new note | look at another object + blink → swap notes |
| empty table | *"Left eye, slower. Right eye, faster. Both eyes, teach me a song."* | OMNI lengthens the look-to-play time | OMNI shortens it | OMNI picks a song from the notes on the table |

A **double blink** cancels; the menu also closes after 8 s. Natural blinks (<0.4 s) are ignored. No
notes play while a menu is open, and the object you answered on won't play until you look away and back.
Winks need per-eye lid state: the QNX eye camera sees both eyes and reports both, so left/right work. A
tracker that only reports one eye makes every long blink count as "both" (INTERFACE.md).

Why OMNI: the user can only give a coarse command ("new note for this one"), so the model has to fill
in the rest from what it **sees** (the table, the object's look, the other notes) and what the player has
been doing, then **say** what it did. Every reply is checked against the command: a "faster" command
can only make the look-to-play time shorter, never change an instrument.

**ElevenLabs speaks.** OMNI decides and writes the line; ElevenLabs says it, streamed so the first
syllable starts before the sentence is finished, in the tone OMNI asked for. That is the same voice as the
pre-generated menu prompts, so the instrument has one voice throughout, and it still works when OMNI is
the thing that is down. `config.json` → `omni.speak_with`: `"eleven"` (default), `"omni"` (the model reads
its own reply, one round trip fewer but a second voice) or `"none"`. Each falls back to the next, then to
local TTS, so a blink is never answered by silence.

## Setup
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover tests -v     # no hardware or API keys needed
cat > .env <<'EOF'                                  # gitignored; env variables override it
OMNI_API_KEY=...                                    # Huawei OMNI Live credits (yibuapi)
ELEVENLABS_API_KEY=...
EOF
.venv/bin/python tools/gen_audio.py                # ElevenLabs: instrument samples + menu voice clips (once)
.venv/bin/python tools/omni_check.py teach         # verifies the OMNI key, model, reply format and voice
```

## Run
```bash
.venv/bin/python tools/synth.py &                                        # sound for the notes
.venv/bin/python run.py --source synthetic --gaze synthetic --realtime   # no hardware
.venv/bin/python run.py --source 0 --gaze mouse                          # webcam, mouse = gaze, keys = blinks
.venv/bin/python run.py --source qnx --gaze qnx                          # the QNX rig (board IP from config.json)
.venv/bin/python run.py --source qnx:192.168.2.2 --gaze qnx --stream 8080   # ...and the overlay in a browser
.venv/bin/python run.py --source 0 --gaze udp                            # any tracker sending UDP (INTERFACE.md)
```
Keys: `1`/`2`/`3` long blink left/right/both · `x` double blink · `q` quit · `m` colour view · `f` features
· `r` record · `c` depth ref · `p` pause · `s` snapshot.
Flags: `--offline` (never call OMNI; built-in defaults), `--no-audio` (menu shown on the overlay only), `--no-menu` (ignore blinks: colours play notes only).
Simulate the eye tracker: `tools/send_gaze.py --at 0.4 0.6 --blink both` (then `--blink left`, …).

## On the laptop screen, or with balloons (colour blocks and big coloured surfaces play tones)
This is the setup we used at Hack the North: the QNX rig looks at a laptop screen showing coloured blocks (or at coloured
balloons), and the block you look at plays its note from `assets/samples`. No blink menu, no API keys needed.

```bash
start_colour_tones.bat                              # Windows: opens the control page, or:
.venv/Scripts/python tools/launcher.py --host 169.254.96.94      # the board's address; page at http://127.0.0.1:8780/
```

**Control page** (`tools/launcher.py`): **Run** starts the colour-block window, the tone player (`tools/synth.py`) and the gaze
pipeline; **Stop** stops them. The page shows the front camera with the pointer, the eye camera with its tracking, what the
camera **sees** (colour, shape, note, share of the picture), what you are **looking at** (and how many pixels away the pointer
is), and what is **playing** (note, instrument and the colour). **Record** saves both cameras side by side with a text bar to
`recordings/*.mp4` (git-ignored).

First use, in this order:
1. **Re-tune colours** (`tools/screen_colors.py tune`): shows the five blocks, measures how the scene camera sees them and
   writes `config.laptop.json` (colour prototypes, white balance on, big-surface detection, pointing radius, and the QNX rig from
   `rig_fit.json`). The block window must be in front and the scene camera must see the whole screen. Do it again when the
   lighting or the camera aim changes.
2. **Run**, then **Calibrate for me**: 12 coloured squares appear flush with the screen edges, one at a time (about 46 s). Look at
   the centre of each. Each square is found in the scene picture by frame difference against a black baseline, so it does not
   depend on the colour classifier. An affine correction from your raw gaze to the squares is saved to `user_calibration.json`
   and is picked up live. The page shows the error before and after and on held-out points. Keep your head still and the
   headset unmoved; a new mounting needs a new calibration. Run/Stop are ignored while it runs.
3. **Manual offset**: if the pointer is still shifted, drag on the front picture (or use the arrow buttons or sliders). It is
   applied after the calibration, live, and saved to `manual_offset.json` (git-ignored). A new calibration clears it.

What it does:
- **Nearest surface wins:** the surface nearest the pointer plays after a 0.5 s hold, if it is on it or within about 4% of the
  picture width (about 8% counts as "near, not on it"). Only big surfaces count (1.2% to 60% of the picture), so small
  coloured things in the background are ignored.
- **Colours through a green cast:** the scene camera has a green/yellow cast (a white wall reads yellow-green), so the detector
  sees an automatically white-balanced picture (`outer_vision/whitebal.py`). White, grey and black are never colours
  (`color_match.s_min`). Orange balloons read as red.
- **Gaze:** the geometric model from `outer_vision/gaze_model.py` with a fixed eye centre, then the user calibration, then
  the manual offset, then a clamp to the picture. The calibration is fitted on the un-clamped gaze (`x_free`, `y_free`).

Command-line pieces (used by the control page): `run.py --no-menu --status-file F --user-cal user_calibration.json
--manual-offset manual_offset.json --stream 8790 --headless --config config.laptop.json`. `--status-file` writes what it sees /
looks at / played as JSON (about 7 times a second).

Accuracy, honestly: about 4 degrees (roughly 60 px in a 640 px picture) after a good calibration, so a block or balloon has to
be much bigger than that to be picked reliably. The lower screen is hard: the lids cover the iris when looking down. The
rig geometry was fitted for one mounting; a very different mounting can push the raw gaze far off the picture.

## Demo script (≈90 s)
1. Look at red, yellow, blue → C, D, G play (marimba, bell, piano).
2. Look at the green cylinder, long blink → Maestro reads the object menu. Wink left → OMNI replies with something like *"That tall
   one sounds like strings now, to go with the piano."* It now plays strings.
3. Look at the empty table, long blink, both eyes → OMNI picks a song from the notes on the table. A
   "next" ring guides your eyes and a fanfare plays at the end.
4. Long blink on the table, wink left → something like *"I'll give you a little more time on each note."* The instrument
   adapts to the player.

## On a real table (≈20 min, redo when the lighting changes)
1. **Colours:** `tools/tune_colors.py --source 0`: press 1–5, click the object, `s` to save.
2. **Shape data**, either way (or both):
   - *OMNI labels (recommended):* the real mixed table, plus hands and pens waved over it, ~3 min of
     moving your head: `tools/collect.py --source 0 --label auto`, then `tools/omni_label.py`. OMNI labels
     4×4 sheets of crops twice, shuffled, and keeps only the crops both passes agree on. The rest go to
     `data/review/` for a human look.
   - *Session labels:* one shape on the table at a time: `tools/collect.py --source 0 --label round`
     (then `square`, `cylinder`, `triangle`, `reject`).
3. **Train** (Mac): `.venv-train/bin/python tools/train_shape.py --real data/real`. The shipped
   `models/shape` is an ImageNet-pretrained MobileNetV3-small at 96 px trained on synthetic renders only
   (`arch` and the scores are recorded in `models/shape/labels.json`); `--arch tiny` is the small 64 px
   CNN, about 5x faster and 7x smaller with the same synthetic accuracy. Train both into separate `--out`
   folders and point `config.json` → `shape_net.model_dir` at the one with the higher `real_val_acc` (it
   holds out whole object tracks, so near-duplicate crops can't inflate it). Training env: `uv venv --python 3.12 .venv-train && uv pip install --python .venv-train/bin/python -r requirements-train.txt`.
   The ONNX copy is checked against torch before it's written.
4. **Depth:** all objects at one known distance, `run.py --ref-distance 60`, press `c`.
5. **Record sessions** (`r`), including blinks, and replay them to tune without wearing the rig:
   `run.py --source recordings/<ts>/world.mp4 --gaze replay:recordings/<ts>/log.jsonl`

## The rig: a Raspberry Pi 5 running QNX 8.0
*Step-by-step version, with every command: **[docs/SETUP.md](docs/SETUP.md)**.*

Both cameras are on the board and **the board runs QNX, not Pi OS**. Everything on it is the C/C++
`camera_streamer` from the eye repo ([htn-gaze](https://github.com/pranaycv/htn-gaze), branch
`pupil-in-eye`), which had to port MediaPipe to run there at all: prebuilt TFLite libraries for QNX
aarch64 and the face/iris models, plus a dark-pupil fit in `pi/pupil.c`. **Do not try to run this repo's
Python on the board** — there is no picamera2 and no CPython environment for it. `run.py` runs on the
laptop and reads the board over HTTP.

Build and start it from the eye repo (`pi/README.md` there):
```sh
scp -r ./pi/* qnxuser@<board>:gazecomp/           # from the eye repo, on the laptop
ssh qnxuser@<board> 'sh ~/gazecomp/scripts/build.sh'
ssh qnxuser@<board> 'slay -f -9 camera_streamer; mv -f ~/gazecomp/camera_streamer.new ~/gazecomp/camera_streamer'
ssh qnxuser@<board> 'sh ~/gazecomp/scripts/start_streamers.sh'    # eye :8080, scene :8081, 960x540
```
Then, on the laptop:
```bash
.venv/bin/python tools/find_board.py             # where is the board, and why isn't it answering?
.venv/bin/python tools/qnx_bridge.py --probe      # is the board up? live gaze + eyelid openness
.venv/bin/python run.py --source qnx --gaze qnx   # scene camera + gaze, in one process
```
Set the board's address once in `config.json` → `qnx.host` (or pass `--source qnx:<ip> --gaze qnx:<ip>`).
On a direct Ethernet cable the Mac needs an address on it too (System Settings → Network → the
USB/Thunderbolt LAN adapter → Details → TCP/IP → Manually: IP `192.168.2.1`, mask `255.255.255.0`, no
router). Check which address the board actually has: the eye repo's `start_streamers.sh` aliases
`192.168.127.94`, while this repo defaults to `192.168.2.2`.

Things worth knowing, all from the eye repo's notes: the Camera Module 3 lens **cannot be focused**
through QNX, so place the eye camera where the picture is sharp; the streamers do **not** survive a
reboot; `slay` without `-f -9` can leave a process holding the camera, after which `camera_open` fails
with error 16; and the board has gone offline under load, which points at the power supply (use a proper
5 V / 5 A one). The Pi 5 has no audio jack either way: Maestro's voice and the notes play on the laptop.

### Setting the rig up
1. **Check the link:** `tools/qnx_bridge.py --probe` prints gaze, per-eye lid openness and whether the
   board is answering. Nothing else works until `n=2` and the openness numbers move when you blink.
2. **Eyelid thresholds:** watch `open L/R` with your eyes open, then shut, and put values either side of
   the gap into `config.json` → `qnx.lid` (`closed_below` / `open_above`). Blinks are the only commands
   the user has, so this is worth a minute.
3. **Aim:** look at the centre of the scene camera's view and run `tools/qnx_bridge.py --zero`; paste the
   printed `zero_yaw_deg` / `zero_pitch_deg` into `config.json` → `qnx.rig`.
4. If the gaze point moves the wrong way, a camera is mounted inverted: set `qnx.rig.flip_x` / `flip_y`.

The rig geometry (`qnx.rig`) mirrors `gui/src/geometry.js` in the eye repo, and
`tests/test_qnx.py` pins our port against values generated by that file. If the eye team change the rig,
change both.

### Other world cameras (Pi OS)
`deploy/deploy_pi.sh` and `tools/pi_camera_server.py` are the older **Raspberry Pi OS** path, for a board
that is not running QNX. They still work, and are useful for a spare Pi as a plain world camera:
```bash
ssh-copy-id <user>@<pi> && deploy/deploy_pi.sh <user>@<pi>
ssh <user>@<pi> 'cd outer-vision && .venv/bin/python tools/pi_camera_server.py --camera 0'
.venv/bin/python run.py --source http://<pi>:8081/stream --gaze udp
```

## Objects
Matte, single saturated colour, 6–10 cm, ≥10 cm apart, on a grey table.

**Five colours, C major pentatonic:** red C4, yellow D4, green E4, blue G4, purple A4. Five rather than
eight because the pairs that used to sit next to each other (red/orange, blue/cyan, purple/pink) are
exactly the ones the colour lookup confuses under venue lighting, and because a pentatonic scale has no
semitones: whatever order you look in, it sounds like music. Register the real hues with
`tools/tune_colors.py` (below) — the defaults are a starting point, not a promise.

| Shape → default instrument | Good | Avoid |
|---|---|---|
| round → marimba | foam/stress balls, ball-pit balls | tennis balls, shiny ornaments |
| square → piano | wooden/foam cubes, Post-it pad, paper-wrapped box | Rubik's cube, printed faces |
| cylinder → flute | paper-wrapped can/tube, plastic cup; taller than wide | bare metal cans |
| triangle → bell | foam wedge, folded card, triangular block; taller than wide | flat triangles lying face-down |

Make the triangle a little **bigger** than the others: a triangle covers about 0.3 of its bounding box
where a ball covers 0.79, so it is the first shape to fall under the detector's minimum area as it gets
further away.

## Layout
```
outer_vision/detector.py   colour LUT → contours → shape (net or rules)
outer_vision/shape_net.py  ONNX shape CNN via OpenCV DNN
outer_vision/tracker.py    IDs, shape voting, size → distance → volume
outer_vision/gaze_model.py QNX rig: pupil + eyelids → a point in the scene image (port of the eye repo's geometry.js)
outer_vision/selector.py   gaze → target → dwell → one lock per look (held while the eyes are closed)
outer_vision/blink.py      eye state → long blink / wink / double blink
outer_vision/menu.py       the blink menu (prompts, options, swap, cancel, timeout)
outer_vision/music.py      colour/shape → note/instrument, overrides, lessons, offline defaults
outer_vision/omni.py       Maestro: OMNI client (streaming), decide → check → act → speak
outer_vision/voice.py      menu prompt clips and live speech (ElevenLabs, then local TTS)
outer_vision/eleven.py     ElevenLabs client (streaming text to speech, sound effects), PCM 24 kHz
outer_vision/audio.py      streaming speech player (output only: there is no microphone)
outer_vision/pitch.py      trim / pitch estimate / repitch for generated samples
outer_vision/telemetry.py  optional Sentry traces/logs
outer_vision/io.py         sources (webcam, video, synthetic, qnx, picam, MJPEG URL), gaze in (qnx poll or
                           UDP) + blinks, UDP out, MJPEG, recorder, ArUco
deploy/                    Raspberry Pi OS setup: deploy_pi.sh (Mac → Pi), setup_pi.sh (on the Pi)
tools/                     find_board (locate the QNX board and say why it isn't answering),
                           qnx_bridge (QNX board → UDP gaze, --probe, --zero), synth, gen_audio, omni_check,
                           omni_label, collect, train_shape, tune_colors, pi_camera_server, listen,
                           send_gaze, make_marker
```

The Solana marketplace and eye-signed wallet live on the **`solana-marketplace`** branch, not here.

# Setup, run, test, train — the whole thing, in order

Everything you need to go from a cold laptop to a working demo, and what to do to keep the work.

If you only have five minutes before a demo, do **[§9 The five-minute pre-demo check](#9-the-five-minute-pre-demo-check)**.

---

## 0. What runs where

There are two computers. Being clear about which is which saves most of the confusion.

```
   THE BOARD (Raspberry Pi 5, QNX 8.0)              THE LAPTOP (macOS)
   ┌──────────────────────────────┐                 ┌──────────────────────────┐
   │ camera_streamer  (C/C++)     │                 │ run.py       (Python)    │
   │  eye camera   :8080 ─────────┼── /api/state ──►│  gaze, blinks, objects   │
   │  scene camera :8081 ─────────┼── /stream.mjpg ►│  notes, menu, Maestro    │
   └──────────────────────────────┘                 │ tools/synth.py  (sound)  │
                                                    └──────────────────────────┘
```

| | The board | The laptop |
|---|---|---|
| OS | QNX 8.0 | macOS |
| Code | the **eye repo** ([htn-gaze](https://github.com/pranaycv/htn-gaze), branch `pupil-in-eye`), folder `pi/` | **this repo** |
| Language | C / C++ | Python |
| Does | runs the two cameras, MediaPipe face mesh, iris, dark-pupil fit | everything else |
| Sound | none (the Pi 5 has no audio jack) | all of it |

**Nothing in this repo runs on the board.** There is no `picamera2` and no Python environment for it under
QNX. Do not try `deploy/deploy_pi.sh` on the QNX board — that script is for a *different*, Raspberry Pi OS
board used as a plain world camera (see [§12](#12-the-old-raspberry-pi-os-path)).

---

## 1. Laptop setup (once)

```bash
cd ~/Outer-Vision-Network
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Check it works with no hardware and no API keys at all:

```bash
.venv/bin/python -m unittest discover tests -v
```

You should see **61 tests, OK**, in about 8 seconds. If that passes, the laptop side is fine.

### API keys

Put them in `.env` in the repo root. This file is gitignored — it will never be committed.

```bash
cat > .env <<'EOF'
OMNI_API_KEY=...
ELEVENLABS_API_KEY=...
EOF
```

| Key | What breaks without it |
|---|---|
| `OMNI_API_KEY` | Maestro can't decide anything; blink commands fall back to fixed built-in defaults (still works, just dumber) |
| `ELEVENLABS_API_KEY` | Maestro's replies fall back to OMNI's voice, then to the Mac's `say`. The pre-recorded menu prompts still work — they're already in `assets/voice/` |

### Generate the audio (once, ~2 minutes, costs ElevenLabs credits)

Already done and committed — `assets/samples/` and `assets/voice/` are in git. Only run this if you
changed the menu wording or want different instrument sounds:

```bash
.venv/bin/python tools/gen_audio.py
```

Redo one sound only:

```bash
.venv/bin/python tools/gen_audio.py --force --only bell
```

Listen to all of them (this also doubles as the speaker check — see [§2a](#2a-first-check-you-can-actually-hear-something)):

```bash
.venv/bin/python tools/synth.py --test
```

---

## 2. Run it with no hardware

Do this before you touch the board. It proves the laptop half works.

> **Some commands here are meant to keep running.** `tools/synth.py` and `run.py` are not scripts that
> finish — they start, print a line or two, and then sit there until you press Ctrl-C. **A terminal that
> looks frozen is a terminal that is working.** Give each one its own tab.

### 2a. First, check you can actually hear something

```bash
.venv/bin/python tools/synth.py --test
```

This plays three notes on each of the seven instruments and then exits on its own, in about 12 seconds.
**You should hear it right away.**

Heard nothing?

- Check **System Settings → Sound → Output**: the right device, and the volume up.
- Other apps (Zoom, a DAW, another copy of this) can hold the audio device. Quit them.
- If it printed `no speaker: ...`, that's the real error — the device couldn't be opened at all.

Don't go further until you can hear this. Everything below assumes the speaker works.

### 2b. Start the sound server — then leave it alone

```bash
.venv/bin/python tools/synth.py
```

**This one does not exit, and it does not make any sound by itself.** It binds UDP port 5006 and waits for
`run.py` to tell it which notes to play. It should print:

```
samples: ['bell', 'drum', 'fanfare', 'flute', 'marimba', 'piano', 'strings', 'synth']
synth listening on udp :5006  --  ready, and SILENT until run.py plays a note.
```

If that is all you see, it is working correctly. **Leave this terminal open** and open a new one.

### 2c. Now start the instrument, in a second terminal

```bash
.venv/bin/python run.py --source synthetic --gaze synthetic --realtime
```

Now you get sound: a window opens with five outlined objects and a white gaze circle that moves between
them, and a note plays each time it settles on one. Back in the first terminal, a line appears per note:

```
  C4 marimba  vol 1.00
  G4 piano    vol 1.00
```

If the window appears and notes print in terminal 1 but you hear nothing, the problem is the speaker, not
this repo — go back to 2a.

Use your **mouse as the eyes** and the keyboard as blinks:

```bash
.venv/bin/python run.py --source 0 --gaze mouse
```

| Key | What it does |
|---|---|
| `1` / `2` / `3` | long blink: left eye / right eye / both eyes |
| `x` | double blink (cancel) |
| `q` | quit |
| `m` | show what colour every pixel was assigned to |
| `f` | show shape features and net probabilities |
| `r` | start / stop recording |
| `c` | save the depth reference (see [§6](#6-tune-for-the-real-table)) |
| `p` | pause |
| `s` | save a snapshot |

Useful flags: `--offline` (never call OMNI), `--no-audio` (captions only, no speaking),
`--headless` (no window), `--stream 8080` (watch the overlay in a browser at `http://localhost:8080/`).

---

## 3. The board: network

The board and the laptop must be on the same network with addresses that can see each other.

### If you use a direct Ethernet cable

Give the Mac an address on that cable: **System Settings → Network → your USB/Thunderbolt LAN adapter →
Details → TCP/IP → Configure IPv4: Manually**

- IP address `192.168.2.1`
- Subnet mask `255.255.255.0`
- Router: leave empty

### Find the board's address

**Don't guess, and don't just ping `192.168.2.2`.** The board's boot config takes its IP from **DHCP**,
and on a direct cable there is no DHCP server — so it gives up and assigns itself a **link-local
`169.254.x.x`** address instead. It is then on a different subnet from your Mac, and every ping times out
even though the board is sitting right there on the end of the cable.

One command finds it wherever it landed, and tells you what to do about it:

```bash
.venv/bin/python tools/find_board.py
```

It checks the ARP table for a Raspberry Pi MAC address, asks mDNS, and tries the addresses both repos
hard-code (`192.168.2.2` here, `192.168.127.94` in the eye repo). Then it says which of four things is
true and prints the fix.

**If it found the cameras**, it prints the exact `config.json` line to paste:

```json
"qnx": { "host": "169.254.96.94", ... }
```

**If it found a Raspberry Pi but nothing answers**, read the two reasons it prints. They are the two that
actually happen:

- **Wrong subnet.** The board picked a `169.254.x.x` address and your Mac is on `192.168.2.1/24`, so the
  board physically cannot reply to you. Give the Mac an address on the board's subnet too (temporary —
  it disappears on reboot, and it does not disturb your existing `192.168.2.1`):

  ```bash
  sudo ifconfig en7 alias 169.254.96.1 255.255.0.0
  ```

  `find_board.py` prints this line with the right interface and numbers already filled in.

- **The board is wedged.** An Ethernet link that shows `status: active` only proves the *PHY chip* has
  power. It stays up when the board itself has hung or browned out, so a live link is **not** evidence
  that the board is running. This is nearly always power — see below.

**If it found nothing at all**, the cable, the adapter, or the board's power is the problem.

### Read the lights on the board first

Before debugging the network at all, look at the Pi. The lights tell you whether it is even booting, and
that decides whether the problem is the network or the board.

| What you see | What it means | What to do |
|---|---|---|
| **Green** light (flickering) | it found a bootable card and is running | the board is fine — debug the network |
| Green + amber at the **Ethernet socket** | link is up and passing traffic | the cable and adapter are fine |
| **Red only, no green** | it has power but is **not booting** | it is not a network problem. See below |
| No light at all | no power reaching it | cable, charger, or the socket |

**Red with no green means the board never started.** The network can look alive for a while anyway — the
Ethernet PHY holds link on its own power — so you can chase a "network problem" that does not exist.
Two things cause it, in this order:

1. **Not enough power** (much the most common — see below).
2. **The SD card.** The bootloader could not find something to boot: card not seated, or the QNX image on
   it is damaged. Reseat the card. If it is still red with a known-good power supply, suspect the card.

Fastest way to tell them apart: **unplug both cameras and power it up.** That roughly halves the draw. If
it goes green without cameras and red with them, it is power.

### ⚠️ Power is the most common cause

The Pi 5 with two cameras needs a real **5 V / 5 A (27 W)** USB-C supply — in practice, the official
Raspberry Pi 27 W PSU. Note that most USB-C PD chargers do **not** offer the 5 V / 5 A profile: a generic
laptop charger negotiates 5 V / 3 A, which boots a bare Pi 5 but leaves little headroom once two cameras
are attached.

**Do not power the board from the docking station or the laptop.** A dock's downstream port is the worst
case — it shares a budget with everything else plugged into it. Run the Ethernet cable through the dock
if you like (that is only data, and it works fine), but power must come from its own charger.

Under-powered, the board will often boot far enough to bring up Ethernet and then stop responding — which
looks exactly like a network problem and isn't one. Both repos record it dropping off the network with
normal temperature and free memory just beforehand.

Power it from a proper charger, then watch for it to come back:

```bash
.venv/bin/python tools/find_board.py --watch
```

### Checking the Mac's side

Your Mac needs an address on the cable. Check what it has:

```bash
ifconfig en7 | grep -E "status|inet "
```

`status: active` with an `inet` line means the adapter is up and addressed. To set it by hand:
**System Settings → Network → your USB/Thunderbolt LAN adapter → Details → TCP/IP → Configure IPv4:
Manually**, IP `192.168.2.1`, mask `255.255.255.0`, no router.

---

## 4. The board: build and start the cameras

All of this is the **eye repo's** code, run on the board. Commands are from its `pi/README.md`.

> **You need [§3](#3-the-board-network) working first.** `scp` and `ssh` below cannot work until
> `tools/find_board.py` can reach the board. Replace `192.168.2.2` in every command with whatever
> address it reported.

From the laptop, in a clone of the **eye repo**, copy the board code over:

```bash
scp -r ./pi/* qnxuser@192.168.2.2:gazecomp/
```

Build it on the board (takes a couple of minutes):

```bash
ssh qnxuser@192.168.2.2 'sh ~/gazecomp/scripts/build.sh'
```

Swap the new binary in:

```bash
ssh qnxuser@192.168.2.2 'slay -f -9 camera_streamer; mv -f ~/gazecomp/camera_streamer.new ~/gazecomp/camera_streamer'
```

Start both cameras (eye on `:8080`, scene on `:8081`, both 960x540):

```bash
ssh qnxuser@192.168.2.2 'sh ~/gazecomp/scripts/start_streamers.sh'
```

⚠️ **The streamers do not survive a reboot.** After every power cycle, run the start command again.

⚠️ **Always kill with `slay -f -9`.** A plain `slay` can leave a process holding the camera, and the next
`camera_open` then fails with error 16. If that happens, `slay -f -9 camera_streamer` and start again.

Check it from the laptop — this should print a wall of JSON:

```bash
curl -s http://192.168.2.2:8080/api/state | head -c 400
```

Look for `"n":2` (it found a face and both eyes). `"n":0` means it can't see your eyes yet.

### Known QNX limits (not bugs you can fix)

- **The lens cannot be focused.** QNX drives the sensor but not the lens motor. Physically place the eye
  camera at the distance where the picture is sharp.
- **Power.** The board has dropped off the network under load. Use a proper **5 V / 5 A (27 W)** supply,
  not a laptop USB-C port or a dock.
- A camera not detected at boot leaves its `/dev/sensor/cameraN` missing — reseat the ribbon **with the
  board off**.

---

## 5. Aim the rig (do this once per wearer, ~3 minutes)

This is the step people skip and then wonder why nothing works. Do it in order.

### 5a. Is the board alive and seeing eyes?

```bash
.venv/bin/python tools/qnx_bridge.py --probe
```

A live line updates in place. You want:

- `n=2` — it sees a face and both eyes
- `open L/R` numbers that **drop when you close your eyes** and rise when you open them
- `pupil 2/2` — both dark-pupil fits succeeded
- `up` — the board is answering

If `n=0`, fix the camera position or lighting before going further. Nothing else will work.

### 5b. Set the eyelid thresholds

Still in `--probe`. Write down the `open L/R` numbers with your eyes **open**, then with them **shut**.
There will be a gap. Put two values inside that gap into `config.json`:

```json
"qnx": {
  "lid": { "closed_below": 0.15, "open_above": 0.20, "min_open_for_gaze": 0.08 }
}
```

- `closed_below` — below this, the eye counts as shut. Put it just above your "shut" reading.
- `open_above` — above this, the eye counts as open. Put it just below your "open" reading.
- Leave a gap between the two. That gap is what stops one long blink from being chopped into several
  short ones.

**Blinks are the only way the user can give a command.** A minute here is worth it.

### 5c. Zero the aim

Look at the **centre of what the scene camera sees** and hold still:

```bash
.venv/bin/python tools/qnx_bridge.py --zero
```

It prints two numbers. Paste them into `config.json` → `qnx.rig`:

```json
"zero_yaw_deg": 1.23,
"zero_pitch_deg": -0.45
```

### 5d. If the gaze point moves the wrong way

A camera is mounted inverted. In `config.json` → `qnx.rig`, flip the axis that's wrong:

```json
"flip_x": true,
"flip_y": false
```

`flip_x` fixes left/right being mirrored; `flip_y` fixes up/down.

---

## 6. Tune for the real table

Redo this whenever the lighting changes. It matters more than anything else on this page.

### 6a. Colours (~5 minutes)

```bash
.venv/bin/python tools/tune_colors.py --source qnx
```

For each of the five objects: press its number (`1`–`5`), then **click on the object** in the video. Its
real colour under this light becomes the prototype. The right half of the window shows what the detector
thinks each pixel is — you want solid blobs on your objects and black everywhere else.

- `[` and `]` shrink and grow how permissive the matching is
- `s` saves to `config.json`
- `q` quits

The five slots, in order, are red · yellow · green · blue · purple = **C4 D4 E4 G4 A4**.

### 6b. Depth, for "farther away = quieter" (~1 minute)

Put every object at one known distance from the camera — say 60 cm — then:

```bash
.venv/bin/python run.py --source qnx --gaze qnx --ref-distance 60
```

Press **`c`**. It measures each object's apparent size at that distance and writes it to `config.json`.
Until you do this, `volume` is `null` and everything plays at full loudness.

---

## 7. Run the real thing

Two terminals. **Both stay running** until you stop them — see the note in [§2](#2-run-it-with-no-hardware).

Terminal 1 — sound. Prints two lines and then waits, silently, for notes:

```bash
.venv/bin/python tools/synth.py
```

Terminal 2 — the instrument. This is what makes the sound happen:

```bash
.venv/bin/python run.py --source qnx --gaze qnx
```

To let other people watch on their own screens, serve the overlay:

```bash
.venv/bin/python run.py --source qnx --gaze qnx --stream 8080
```

Then open `http://localhost:8080/` (or `http://<laptop-ip>:8080/` from another machine).

**Read the top-left HUD.** It tells you whether the board is alive:

```
30.0 fps  proc 12.3 ms  shape:onnx  gaze:qnx  board:up 22fps eyes:2 pupil:2  objs:5 ...
```

`board:DOWN`, `eyes:0` or `pupil:0` means look at the board, not at this repo.

---

## 8. Test every feature

Each row has a no-hardware version, so you can test the logic without the rig.

| Feature | With the rig | Without hardware |
|---|---|---|
| **Everything at once** | `run.py --source qnx --gaze qnx` | `run.py --source synthetic --gaze synthetic --realtime` |
| **A note plays** | look at an object for 0.5 s | mouse over an object |
| **Colour → pitch** | the five objects give C4 D4 E4 G4 A4 | `--source synthetic` shows all five |
| **Shape → instrument** | round marimba, square piano, cylinder flute, triangle bell | same |
| **Blink menu opens** | long blink (0.6–2 s) at an object | press `3` |
| **Left / right winks** | long wink of one eye | press `1` / `2` |
| **Cancel** | double blink | press `x` |
| **Maestro decides** | wink to answer the menu | press `3` then `1` |
| **Offline fallback** | — | add `--offline`: commands still work, with fixed answers |
| **Song lesson** | long blink at the **empty table**, then both eyes | press `3` then `3` |
| **Events for the game** | `tools/listen.py` in another terminal | same |

Watch the events the game receives:

```bash
.venv/bin/python tools/listen.py
```

Include the per-frame state, not just notes and commands:

```bash
.venv/bin/python tools/listen.py --state
```

Check both live APIs before a demo — decision **and** the voice the demo actually uses:

```bash
.venv/bin/python tools/omni_check.py teach
```

Fake an eye tracker over UDP, to test the blink menu without the rig:

```bash
.venv/bin/python tools/send_gaze.py --at 0.36 0.33 --blink both
```

Run the whole test suite (nothing external needed):

```bash
.venv/bin/python -m unittest discover tests -v
```

### Record a session and replay it

Recording is the best way to tune blinks without wearing the rig. Press **`r`** while running, do the
thing, press `r` again. It saves the video *and* every gaze sample and blink.

Replay it through the current code as many times as you like:

```bash
.venv/bin/python run.py --source recordings/<timestamp>/world.mp4 --gaze replay:recordings/<timestamp>/log.jsonl
```

---

## 9. The five-minute pre-demo check

```bash
.venv/bin/python -m unittest discover tests
```

```bash
.venv/bin/python tools/find_board.py
```

```bash
ssh qnxuser@192.168.2.2 'sh ~/gazecomp/scripts/start_streamers.sh'
```

```bash
.venv/bin/python tools/qnx_bridge.py --probe
```

```bash
.venv/bin/python tools/omni_check.py teach
```

```bash
.venv/bin/python tools/synth.py --test
```

That last one is the speaker check and exits by itself. Then run it for real ([§7](#7-run-the-real-thing))
— `tools/synth.py` with no flags in one terminal, `run.py` in another — and play one note of each colour.

---

## 10. Train the shape neural network

The shipped model is trained on **synthetic renders only**. It has never seen a real object. Training on
real crops of your actual objects is the single biggest accuracy win available.

Three stages: **collect → label → train**.

### Stage 1: collect crops

The training environment is separate (PyTorch is not needed to *run* the model, only to train it). Set it
up once. It already exists on Ali's laptop — check first:

```bash
.venv-train/bin/python -c "import torch, torchvision, onnx; print('training env ready')"
```

If that fails, create it. With `uv` (fast):

```bash
uv venv --python 3.12 .venv-train && uv pip install --python .venv-train/bin/python -r requirements-train.txt
```

Without `uv`, plain Python works too (slower, needs Python 3.12):

```bash
python3.12 -m venv .venv-train && .venv-train/bin/pip install -r requirements-train.txt
```

Now collect. **Option A (recommended)** — put the real, mixed table in front of the camera, wave your
hands and a pen over it, and move your head around for about 3 minutes:

```bash
.venv/bin/python tools/collect.py --source qnx --label auto
```

That saves unlabelled crops to `data/unlabeled/`.

**Option B** — one shape on the table at a time, and tell it which:

```bash
.venv/bin/python tools/collect.py --source qnx --label round
```

Repeat with `--label square`, `--label cylinder`, `--label triangle`, and `--label reject` (an empty table
with hands, pens and paper waving over it). Those go straight into `data/real/<label>/`.

### Stage 2: label them with OMNI (Option A only)

```bash
.venv/bin/python tools/omni_label.py
```

OMNI labels the crops in 4×4 sheets, **twice, shuffled**, and keeps only the ones both passes agree on.
Disagreements go to `data/review/` for you to look at by eye. Try a small batch first:

```bash
.venv/bin/python tools/omni_label.py --limit 64 --dry-run
```

### Stage 3: train

Train the accurate model:

```bash
.venv-train/bin/python tools/train_shape.py --real data/real --arch mobilenet --out models/shape_mnv3
```

Train the small fast one, for comparison:

```bash
.venv-train/bin/python tools/train_shape.py --real data/real --arch tiny --out models/shape_tiny
```

Each prints a line per epoch and finishes with a summary. The number that matters is **`real_val_acc`** —
accuracy on real crops it never trained on. Ignore `synthetic_val_acc`; it is always ~1.0 and means
nothing about the real table.

Validation holds out **whole object tracks**, not random crops, because crops of the same object a quarter
of a second apart are near-duplicates and would otherwise inflate the score.

Compare them:

```bash
cat models/shape_mnv3/labels.json models/shape_tiny/labels.json
```

### Stage 4: ship the winner

Point `config.json` at whichever has the higher `real_val_acc`:

```json
"shape_net": { "enabled": true, "model_dir": "models/shape_mnv3", "reject_min_prob": 0.6 }
```

Confirm it loads — the startup line should say `[shape] onnx`, not `[shape] rules`:

```bash
.venv/bin/python run.py --source synthetic --gaze synthetic --headless --max-frames 30
```

Then check it against the objective tests:

```bash
.venv/bin/python -m unittest discover tests
```

If you'd rather replace the default model in place, copy both files over `models/shape/`:

```bash
cp models/shape_mnv3/shape.onnx models/shape_mnv3/labels.json models/shape/
```

---

## 11. Saving your progress

**Read this before you close the laptop.** Some of what you produce is saved by git and some is not.

| What you made | Where | Saved by git? |
|---|---|---|
| Colour tuning, depth reference, board IP, lid thresholds | `config.json` | ✅ **yes** — commit it |
| Trained model | `models/shape*/` | ✅ **yes** — commit it |
| Generated sounds and menu voice | `assets/` | ✅ **yes** — already committed |
| Code and docs | everywhere | ✅ yes |
| **Collected training crops** | `data/` | ❌ **NO — ignored** |
| **Recorded sessions** | `recordings/` | ❌ **NO — ignored** |
| API keys | `.env` | ❌ no, deliberately — never commit keys |

### Commit and push the things that are saved

```bash
git status
```

```bash
git add config.json models/
```

```bash
git commit -m "Tuned colours and retrained shape net on real crops"
```

```bash
git push origin main
```

### Back up the things git ignores

`data/` and `recordings/` are ignored because they're large and full of pictures of people's faces. If you
spent 3 minutes collecting crops, **that work only exists on this laptop.** Copy it somewhere:

```bash
tar czf ~/Desktop/outer-vision-data-$(date +%Y%m%d).tgz data recordings
```

⚠️ Those files contain **pictures of the wearer's face and eyes**. Don't put them anywhere public.

### One thing to never commit

The `marketplace/`, `wallet/` and `songs/` folders may be sitting in your working tree from the
`solana-marketplace` branch. Two of them hold **real private keys**
(`wallet/session_key.json`, `marketplace/.keys/marketplace.json`). `.gitignore` on `main` blocks all
three, plus `*session_key.json`, `*keypair.json` and any `.keys/` folder. Check before a big commit:

```bash
git status --short
```

If a key file ever shows up in that list, stop and fix `.gitignore` — do not commit it.

---

## 12. The old Raspberry Pi OS path

`deploy/deploy_pi.sh`, `deploy/setup_pi.sh` and `tools/pi_camera_server.py` are for a **Raspberry Pi OS**
board — a spare Pi used as a plain world camera. They do **not** apply to the QNX board.

```bash
ssh-copy-id user@<pi>
```

```bash
deploy/deploy_pi.sh user@<pi>
```

```bash
ssh user@<pi> 'cd outer-vision && .venv/bin/python tools/pi_camera_server.py --camera 0'
```

```bash
.venv/bin/python run.py --source http://<pi>:8081/stream --gaze udp
```

The historical debugging log for that path is in [PI_DEBUGGING.md](PI_DEBUGGING.md).

---

## 13. Troubleshooting

| Symptom | Most likely cause | Do this |
|---|---|---|
| **Red light on the board, no green** | it has power but is not booting — not a network fault | 27 W supply; reseat the SD card; try it with the cameras unplugged |
| Ethernet socket lights went out, `status: inactive` | the board stopped driving the link — it is off or hung | check the board's lights before anything else |
| Can't ping the board at all | it is probably on a link-local `169.254.x.x` address, not `192.168.2.2` | `tools/find_board.py` — it finds it and prints the fix |
| Ping times out but the Ethernet link is "active" | an active link only means the PHY has power; the board itself can be wedged | repower from a 5 V / 5 A supply, then `tools/find_board.py --watch` |
| Board answers nothing, and no SSH either | under-powered, or it never finished booting | proper 27 W supply; serial console if it stays dead |
| `board:DOWN` in the HUD | streamers aren't running (they die on reboot) | `ssh qnxuser@<board> 'sh ~/gazecomp/scripts/start_streamers.sh'` |
| `camera_open` fails with error 16 | an old process still holds the camera | `slay -f -9 camera_streamer`, then start again |
| `eyes:0` / `n=0` | camera can't see the eyes, or it's too dark | reposition the eye camera; more light |
| Eye picture is blurry | the lens can't be focused under QNX | physically move the camera to the sharp distance |
| `pupil:0` | dark-pupil fit failing; falls back to the iris | more light on the eye; check focus |
| Gaze is mirrored | a camera is mounted inverted | set `qnx.rig.flip_x` / `flip_y` |
| Gaze is offset by a constant | aim not zeroed | `tools/qnx_bridge.py --zero` |
| Blinks never register | lid thresholds wrong for this wearer | `tools/qnx_bridge.py --probe`, redo §5b |
| One long blink acts like several | no gap between the two lid thresholds | widen `closed_below` → `open_above` |
| Menus open by themselves | usually a stalled board | check `infer_fps` in `--probe`; restart the streamers |
| Board keeps dropping off the network | power | use a 5 V / 5 A supply |
| Objects not detected | colours not tuned for this light | `tools/tune_colors.py`, §6a |
| A triangle isn't detected but others are | a triangle covers ~0.3 of its box vs ~0.79 for a ball | use a bigger triangle, or move it closer |
| `[shape] rules` at startup | model missing or `model_dir` wrong | check `config.json` → `shape_net.model_dir` |
| Everything plays at full volume | depth never calibrated | §6b, press `c` |
| Maestro says the same thing every time | no `OMNI_API_KEY`, so it's using offline defaults | check `.env`; `tools/omni_check.py` |
| Maestro speaks in a different voice than the menu | no `ELEVENLABS_API_KEY`, falling back to OMNI's voice | check `.env` |
| `tools/synth.py` prints two lines then "hangs" | that is correct — it is a server waiting for notes | leave it running; start `run.py` in another terminal |
| No sound, but notes print in the synth terminal | the Mac's output device or volume | System Settings → Sound → Output; quit apps holding the device |
| No sound and nothing prints in the synth terminal | `run.py` isn't sending, or is sending elsewhere | check `run.py` is running; both must agree on port 5006 (`--send host:port`) |
| `no speaker: ...` on startup | the audio device couldn't be opened | another app is holding it, or no output device is selected |
| No sound at all | `tools/synth.py` isn't running | start it; test the speaker with `tools/synth.py --test` |

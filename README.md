# Hack the North 2026: head-mounted eye tracking on QNX

A head-mounted rig with two Raspberry Pi Camera Module 3 cameras on a **Raspberry Pi 5 running QNX 8.0**:

- an **eye camera** looking at the wearer's eyes, and
- a **scene camera** looking forward.

The Pi streams both cameras over the LAN and runs MediaPipe face mesh (TFLite) on the eye camera. A laptop calibrates a mapping from
pupil position to scene-camera pixel (by looking at coloured squares on the laptop screen) and then shows where the wearer is looking on the
scene camera's picture.

```
eye camera   -> Pi (QNX): face mesh, iris landmarks, full-resolution eye crops --\
                                                                                  >-- HTTP --> laptop: calibrate, map, show gaze dot
scene camera -> Pi (QNX): video only --------------------------------------------/
```

| Folder | What |
|---|---|
| [`pi/`](pi/README.md) | Everything that runs on the board: C/C++ camera streamer, models, build and start scripts, SD-card config |
| [`calib/`](calib/README.md) | Laptop tools: calibration session, pupil features, robust fit, live gaze view, camera viewers |
| [`outer-vision/`](outer-vision/README.md) | The instrument: what you look at on the laptop screen (coloured blocks) or with balloons plays a tone. Control page, per-user calibration, manual offset, white balance |
| [`experiments/`](experiments) | Side experiments that are not part of the main pipeline (a Luxonis OAK-1 UVC attempt, a focus-sharpness test) |

## Quick start

1. Copy `pi/` to `~/gazecomp` on the board, build with `scripts/build.sh`, start with `scripts/start_streamers.sh` (see `pi/README.md`).
2. On the laptop: `pip install -r calib/requirements.txt`, open `calib/view_fast.html` to check both cameras, then `python calib/calibrate.py`.
3. Live pointing: `python calib/gaze_live.py`.
4. Tones from where you look (blocks or balloons): `cd outer-vision`, `pip install -r requirements.txt`, then `python tools/launcher.py --host <board ip>` and open http://127.0.0.1:8780/ (see `outer-vision/README.md`).

## Status

- Working: both cameras streaming at about 30 fps; face mesh and iris landmarks at 10-30 fps; scene-camera square detection; calibration
  and live gaze view. Best measured pointing error in early runs: about 3.6 to 3.9 degrees on held-out squares (assumed 66 degree field of view).
- Implemented but not yet validated end to end: full-resolution eye crops with an iris circle fit and a per-sample robust fit.
- Limits we hit: the lens of the Camera Module 3 cannot be focused through QNX here (fixed focus); iris visibility drops when the wearer looks
  down at a camera mounted below the eye; the Pi went offline several times, apparently from power (see `pi/README.md`).
- The OAK-1 camera was tried and dropped: this unit has no onboard flash, so it cannot run standalone as a USB webcam.

## Privacy

Calibration runs save pictures of the wearer's face and eyes. `run_*` folders and check images are git-ignored; do not commit them.

# Laptop side: calibration and gaze pointing

Python tools that talk to the QNX board (`../pi`). They calibrate a mapping from **pupil position in the eye camera**
to **pixel position in the scene camera**, then draw a live "where you are looking" dot on the scene camera's picture.

## Setup

```sh
python -m venv .venv && .venv/Scripts/activate      # or any Python 3.10+ environment
pip install -r requirements.txt                     # numpy, pillow, opencv-python-headless
```

The scripts default to the board's wired link-local address `169.254.96.94` (eye camera on port 8080, scene camera on 8081).
Pass `--eye URL --scene URL` to use another address. Start the streamers on the board first (`../pi/scripts/start_streamers.sh`).

## Files

| File | What |
|---|---|
| `calibrate.py` | Calibration session. Serves a full-screen page (`calib.html`) that shows coloured squares one at a time; for each square it records the pupil features and where the square appears in the scene camera, then fits and evaluates a model |
| `detect.py` | Finds the calibration square in a scene-camera frame: a compact, filled, square blob that *appeared* compared with a black-screen frame taken just before |
| `features.py` | Pupil features from the full-resolution eye crops (`/api/eyepack`): iris circle fit, a steady eye-corner reference, blink flags |
| `fitlib.py` | Robust (Huber) ridge fit on every usable sample; model choice by leave-one-square-out cross-validation |
| `gaze_live.py` | Live pointing view at `http://127.0.0.1:8766/` using a model from `calibrate.py` (v2 format) |
| `calib.html` | The page shown during calibration; ends with a large **END** |
| `view.html`, `view_fast.html` | Side-by-side viewers for the two cameras (`view_fast` polls single frames, so it does not lag) |
| `run_showcase.sh` | Runs a calibration, then starts the live view if a model was produced |
| `calibrate_v1_old.py`, `gaze_live_v1.py` | First-generation pipeline (features from `/api/state`); needed to use models saved by early runs |

## Running

```sh
python calibrate.py --hold 4.5          # opens the page; click Start, then look at the cross in each square
python gaze_live.py                     # uses the newest run_*/gaze_model.json
```

A run first checks the eye camera (iris measurable on enough frames) and that the scene camera sees **all four screen corners
uncut**, then shows 25 calibration squares and 9 held-out squares. Output goes to `run_<timestamp>/` (`samples.json`,
`gaze_model.json`, annotated pictures). Those folders contain pictures of the user and are git-ignored.

Requirements for a good run: rigid rig (both cameras fixed relative to the head), head still, eyes open and gaze roughly level
(screen near eye height), eye camera sharp with both eyes well inside its picture, whole screen well inside the scene picture.

## How it works

1. **Target position (scene camera):** square found by subtracting a black-screen frame and looking for a compact square blob (`detect.py`).
2. **Pupil position (eye camera):** the board cuts a 512x384 crop around each eye from the *full-resolution* frame. `features.py` fits the iris
   circle on its left/right edge arcs (eyelids excluded) and measures the iris against a median of the eye-corner landmarks over the last 8 s.
   If the fit is not confident (for example the lids cover the iris), MediaPipe's own iris estimate is used instead. Blinks are dropped.
3. **Mapping:** `fitlib.py` fits linear or degree-2 polynomials on all samples with Huber weights and compares feature sets by
   leave-one-square-out cross-validation and on held-out squares.

## Results so far (be careful how you quote these)

- With the first pipeline (MediaPipe iris offset from the eye corners, one median per square), the best runs gave
  **about 53 px (3.6 degrees)** and **57 px (3.9 degrees)** mean error on 9 held-out squares in the scene camera picture.
  Degrees assume the scene camera's horizontal field of view is 66 degrees (standard Camera Module 3), which we did not measure.
- The second pipeline (full-resolution crops, iris circle fit, per-sample robust fit) is implemented and passes synthetic tests
  (iris centre recovered to under 1 px on synthetic sharp eyes), but **has no end-to-end calibration result yet**. On real, half-closed
  eyes the circle fit was accepted on only a small share of frames, so most samples fall back to MediaPipe's estimate.
- Accuracy depends on the eyes being open and the iris visible. Looking down at a laptop from a camera below the eye covers the iris with the lid.

## Known limits

- The mapping is exact only at the calibration distance (the scene camera is a few centimetres from the eye) and only while the rig does not move.
- The scene-camera field of view is assumed; error in degrees is approximate.
- Defaults assume a Windows machine (`run_showcase.sh` opens the browser with PowerShell).

# Challenges and how we solved them

The first four are the team's own write-up (kept as written). The rest were collected from the repo's READMEs, the Devpost draft and the
commit history. Each says what was solved and what was not.

## Written by the team

We had planned to place the camera on the side of a face, but winking needs to be seen from both sides, so we had to move the camera to the front using a rig and take an accurate snapshot.

Black, white, brown and grey are impossible. They're the table and its shadows. We ended up restricting to 5 saturated colours matched by nearest prototype.

We also realized that a hand reaching over the table can be seen as a blob of colour. We solved this by adding an explicit "reject" class to the CNN and training on real footage of hands and pens waved over the table.

Volume depends on distance, which we estimate from apparent size and one reference capture per object type. It drifts if the player leans, and we know the proper fix is intersecting the gaze ray with the table plane.

## Hardware and QNX

- **The camera lens cannot be focused.** QNX drives the IMX708 sensor but not its lens motor (`camera_set_manual_focus_step` returns error 103).
  *Solved by working around it:* the eye camera is placed at the distance where the picture is sharp. Not fixed.
- **CPU was the bottleneck.** Both cameras run the ISP on the CPU and the display driver used about one core. Copying the full-resolution frame on
  every frame dropped the face model to about 4 fps.
  *Solved:* removed the per-frame copy; the face model is back to about 10-30 fps and the cameras stream at about 30 fps.
- **A camera handle of 0 looked like an error.** `camera_open` can legitimately return 0, which the header calls invalid.
  *Solved:* validity is tracked with a separate flag.
- **A stale process held the camera.** A plain `slay` left one running and the next `camera_open` failed with error 16.
  *Solved:* always `slay -f -9`, documented in the start scripts.
- **A camera missing at boot.** A ribbon that is not detected leaves `/dev/sensor/cameraN` missing. *Solved:* reseat with the board off.
- **The board went offline several times** with normal temperature and free memory. We suspect the power supply and use a 5 V / 5 A supply
  and `health.sh` logging. This is our inference, not confirmed.
- **No SDP on the board.** *Solved:* a minimal `camera_api.h`, prebuilt TFLite libraries, and a `build.sh` that compiles on the board.
- **The Luxonis OAK-1 was dropped.** This unit has no onboard flash, so it cannot run as a standalone USB webcam.

## Eye tracking

- **Calibration needs to find the target.** *Solved:* show a black frame, then a coloured square, and detect the compact filled square blob that
  appeared (`detect.py`). The run also checks that the scene camera sees all four screen corners.
- **Noisy pupil position.** *Solved:* full-resolution eye crops from the board, an iris circle fit, an eye-corner reference averaged over 8 s,
  and a Huber-weighted ridge fit chosen by leave-one-square-out cross-validation. Best held-out error about 3.6 to 3.9 degrees (assumed 66 degree field of view).
- **Eyelids cover the iris when looking down** at a laptop from a camera below the eye. Not solved: the lower part of the screen had no usable data.
- **The circle fit is only confident on a small share of frames** with half-closed eyes, so most samples fall back to MediaPipe's iris estimate.
  Partly solved by the fallback, with the fit's confidence reported as 0.5 for the fallback.
- **The pre-built geometric gaze model was 11 to 15 degrees off.** We ported it to Python and scored it offline. With the eye centre held fixed
  and 5 fitted rig parameters it reached about 4.1 degrees, the same as the polynomial calibration. The fitted depth and eye distance hit the search
  limits, so it is an empirical mapping, not a physical one.
- **A camera too close for Face Mesh** ("not a face" when the face is clipped). We built a Hough-circle iris finder in seven rounds:
  - A vertical and sideways prior kept it off clothing and cables.
  - Windowed tracking raised the share of frames near the typical position from about 1% to 68%.
  - Once it slipped onto a wrong round spot it could not recover, so it now searches the whole upper frame and switches only after 3 wins in a row.
  - The purple glasses frame and the nose-bridge shadow were winning, so we added darkness, contrast and colour tests with thresholds taken from logged
    candidate statistics. Result: spread 8-9 px (was 41-109 px), both eyes reported on 86% of frames.
  - Limit: the thresholds were fitted to one wearer and one lighting setup.
- **The rig must not move.** The mapping is exact only at the calibration distance and while the rig is rigid. Not solved.

## Voice assistant (OMNI)

- **The always-on microphone heard OMNI's own voice and the instrument's notes**, which kept the assistant awake.
  *Solved:* echo and noise filters, and muting new clips while a note plays.
- **Hands-free but not trigger-happy.** *Solved:* wake word "OMNI", a 10 second follow-up window without the name, "go to sleep" to end it,
  and a retry for the wake word.
- **OMNI must know what "this" is.** *Solved:* each request carries the scene picture with the gaze pointer, a close-up crop around the gaze point,
  and the colour, shape and note the gaze app measured under the gaze.
- **Letting the model decide what is protected made it refuse harmless requests.**
  *Solved:* the model only proposes actions; code decides. Blue, green, yellow, red and orange sounds cannot be created, replaced or deleted
  (application check plus a database trigger).
- **The gaze app runs on the laptop, not the glasses.** Honest limit: the wearable is tethered.
- **Outages must not stop the loop.** *Solved:* OMNI, ElevenLabs and Tiger failures are logged and the loop keeps running, with a one-time outage warning.

## Data and telemetry

- **Duplicate rows from the database.** After five repeats the driver used a prepared statement and TimescaleDB's generic plan for
  `ORDER BY time DESC LIMIT $1` returned 8 rows from a 5-row table. *Solved:* auto-prepare is off for reads.
- **A database outage stalled the dashboard.** *Solved:* a circuit breaker on reads.
- **Per-frame inserts would be too slow.** *Solved:* the receiver batches about once a second, with a bounded buffer and retry.
- **Live dashboard missing the newest rows.** *Solved:* the continuous aggregate `gaze_10s` runs in real-time mode with a refresh policy.
- **Tracking loss was overstated.** The iris-centre fallback was counted as lost tracking. *Solved:* confidence is 0 = no gaze, 0.5 = iris fallback,
  up to 1 = dark-pupil fit.
- **Stage timings were not real.** *Solved:* the bridge now uses measured stage timings and pupil confidence.
- **A long `cap` span looks like a bottleneck but is not.** It is mostly waiting for the next camera frame (about 33 ms at 30 fps).
- **Sentry privacy.** Sentry trace headers are never sent to OMNI, ElevenLabs or Tiger, and the OMNI transcript is masked in Session Replay.
- **Only interesting frames should be traced.** *Solved:* sample 1 in N frames plus rate-limited anomalous frames. Profiling is behind `PROFILE=1`.

## Known limits

- "Frames" in Tiger are the gaze app's status updates (about 6 per second), not camera frames.
- The board reports only an inference rate, so the "pupil" stage is the laptop-side gaze-model time.
- `calib_drift` is a heuristic, not a measured calibration error.
- The scene camera's field of view is assumed, so errors in degrees are approximate.

# Close-up eye path (experimental, NOT validated)

Attempt to make eye tracking work when the eye camera is so close that the whole face does not fit in the frame
(Face Mesh then says "not a face"). Built on the `pupil-in-eye` branch's `pi/face_lm.cpp`.

- `face_lm.cpp`: branch code plus a fallback when the mesh rejects the face: a circular Hough transform finds the two
  irises on a down-scaled gray image, then each eye is cropped and run through the MediaPipe iris model (`iris_landmark.tflite`).
- `patch_closeup.py`, `patch_hough.py`: the two patches that produce it from the branch file (first version used BlazeFace
  eye keypoints, which turned out to be about 150 px wrong when the face is clipped by the frame).

Status: compiles and runs on the Pi. On the one view we tried (face larger than the frame, eyes half closed behind tinted
glasses lenses) the irises are not visible and the finder locks onto other round things (clothing edge, cable, lens glare),
so it reports false eyes. It needs validation on a view where the irises are visible, and a stricter check (iris darker than
its surroundings, eye contour around it) before it can be trusted. Lid indices of the iris model's contour output are still
the placeholder 0..15 and were not identified.

## Update (after three Hough rounds)
Round 2 added a vertical prior (eyes in the upper 45% of the frame, env `GAZE_EYE_YMAX`) and a dark-disc check;
round 3 gave every edge an equal vote, narrowed the radii to 20-40 px and added a sideways prior (`GAZE_EYE_XMIN/XMAX`).
On a view where the eyes are visible (glasses lenses removed, eyes half-open) the overlay lands on both real eyes on many
frames, but over 20 s both irises were within 40 px of their typical position on only ~1% of frames: the finder jumps
between the real eyes and other round things. Next steps that were not done: temporal tracking (accept a new pair only near
the previous one), a stricter first-lock test, and identifying the eyelid indices of the iris model's contour output.

## Update: temporal tracking (round 4, `patch_hough4.py`)
First lock needs the same iris pair on 3 consecutive frames; then each iris is searched only within ~48 px of its last
position, positions are smoothed, and the lock is dropped after 6 lost frames. Measured on the same view as before
(20 s, `consistency.py`): frames with both irises within 40 px of their typical position went from about 1% to 68%.
Independent check with the laptop-side circle fit on full-resolution crops was mixed (the circle fit is itself only confident on
11-22% of frames at this focus): median gap 7 px (left, 11% of frames) and 31 px (right, 22%), and one snapshot had the left
estimate on the eyebrow. Accuracy of the iris centre is therefore NOT established; the eye-contour (lid) indices of the iris model
are still a placeholder and they also place the full-resolution eye crops.

## Update: rounds 5-7 (recovery from a slipped lock, and thresholds from data)
- Round 4's windowed tracking could not recover once it slipped onto a wrong round spot. Round 5 searches the whole upper frame
  every frame, prefers the current lock by 25%, and switches to a different pair only after it wins 3 frames in a row.
- Round 6 added a darkness test and a blue/purple test (the wearer's purple glasses frame was winning).
- Round 7 set the thresholds from logged candidate statistics (`patch_dbg2.py`, env `GAZE_DBG=1`): irises were dark (inner gray 40-67),
  contrasty (inner/ring 0.65-0.74) and neutral in colour (blue-minus-red -12..-16); the nose-bridge shadow, which had the highest edge score,
  was brighter (92), low contrast (0.88) and strongly red (-36). Defaults now: inner gray <= 85, inner/ring <= 0.78,
  blue-minus-red in [-26, 14] (env `GAZE_MAX_IRIS_GRAY`, `GAZE_MAX_RATIO`, `GAZE_MIN_BMR`, `GAZE_MAX_BMR`).
- Result on one wearer (purple glasses frame, brown eyes, this lighting), eyes open, 18 s: iris position spread 8-9 px (was 41-109 px),
  99% of reported frames near the typical position, both eyes reported 86% (nothing reported with eyes shut).
  The thresholds were fitted to that wearer and scene; other eye colours and lighting are untested.
- `process.c`: the branch's dark-pupil fit is off unless `GAZE_PUPIL` is set.

# v0 compromises → future editions

Everything we knowingly cut for the hackathon, and what "done properly" looks like.

| # | Area | v0 compromise | Future edition |
|---|---|---|---|
| 1 | Audio / portability | Sound on the laptop; the board's cameras and our overlay streamed as MJPEG over Ethernet/Wi-Fi (~50–120 ms on Wi-Fi). The Pi 5 has no audio jack, and under QNX there is no Python on the board to run this half on | USB speaker on the wearable; port the world-camera half to C/C++ so it runs on the board too |
| 2 | Distance | Gaze calibrated at one fixed working distance | Variable depth: intersect the gaze ray with the table plane, or add a depth/stereo camera |
| 3 | Depth for volume | Distance from apparent size, needs one reference capture per object type (`c` key) | Table-plane geometry or a learned monocular depth model; no reference capture |
| 4 | "Farther back on the table" | Approximated as distance from the camera, so it shifts if the user leans | Position on the table (the table itself as the reference frame) |
| 5 | Colours | **v5:** 5 saturated colours via nearest-prototype matching, cut from 8 because neighbouring hues (red/orange, blue/cyan, purple/pink) are what the LUT confuses under venue light; black, white, brown and grey impossible (they're the table/shadows) | Learned colour/material recognition |
| 6 | Pitches / instruments | **v5:** 5 colours = C major pentatonic (C D E G A), so nothing on the table can clash, at the cost of F and B: songs needing them (Twinkle, Ode to Joy) can't be played. 4 shapes = 4 instruments | Position → octave, chords, a mode switch for other scales |
| 7 | Detection method | **v1:** colour finds objects; a CNN (ONNX) classifies shape and rejects hands. Still needs saturated objects on a grey table | Full learned detector for arbitrary objects in any setting |
| 8 | Shapes | CNN trained on **synthetic renders only** — no real crops have been collected yet, so `real_val_acc` in `models/shape/labels.json` is `null` and the 100% figures are all synthetic. Rules as fallback. Real crops can be labelled by OMNI offline (two shuffled passes must agree); the shipped model is a pretrained MobileNetV3-small at 96 px | Human-checked labels from many venues and lighting conditions; an active-learning loop that sends only the crops the model is unsure about |
| 8b | Triangle | A triangle covers ~0.3 of its bounding box against ~0.79 for a ball, so it is the first shape to drop under `min_area_frac`: it has to be a bit bigger, or a bit closer, than the others | Per-shape minimum area |
| 9 | Motion | Still objects only; the tracker assumes small movement between frames | Motion model / proper multi-object tracker |
| 10 | Lighting | Camera auto-exposure and auto white balance; indoor only | Locked exposure/WB and auto colour recalibration |
| 11 | Eye | One eye camera, but it sees **both** eyes (MediaPipe reports a left and a right eye, or neither), so left/right winks work. Eye closed/open comes from the height/width of the eyelid outline against two fixed thresholds, not from a per-user model | Per-user blink calibration; convergence depth from the two eyes |
| 12 | Calibration | QNX rig: a fixed geometric model from measured camera spacing plus a one-off aim zero — no per-session calibration, but also no correction for a rig that shifts on the head mid-demo. Other trackers can still use the ArUco marker path | Marker-free calibration off the objects themselves, plus continuous drift correction |
| 13 | Glasses | Users who wear glasses are not supported | Eye-camera placement / IR that works through or around lenses |
| 14 | Musicality | One note per look, one note at a time, 0.5 s dwell, so no real rhythm | Onset on fixation (~150 ms), sustained notes, tempo/quantisation, chords |
| 15 | Game | Free play only | Song mode with guidance and scoring |
| 16 | Hands | **v1:** CNN "reject" class drops hands, pens and paper (100% on synthetic; unproven on real) | Train reject class on real hands |
| 17 | Edges | Objects cut off by the frame edge are flagged `partial` and get no depth | Track through the edge; wider lens |
| 18 | Sync | Eye and world cameras are unsynchronised; gaze is paired with the newest frame (≤ 0.2 s old). Gaze is also *polled* at 30 Hz over HTTP rather than pushed, so a sample can be up to one poll old | Hardware trigger or timestamp alignment; a push channel from the board |
| 19 | Transport | UDP JSON with no delivery guarantee (`--gaze udp`), or HTTP polling with no back-pressure (`--gaze qnx`) | A reliable channel for lock events (they're rare) |
| 20 | Language | Python + OpenCV | Fine for a laptop; port hot paths if it moves onto the wearable |
| 21 | Markers | ArUco marker only for calibration, to stay "works anywhere" | None at all |
| 22 | QNX | The board **does** run QNX 8.0, but only the eye half: MediaPipe had to be ported there (prebuilt TFLite libraries for QNX aarch64, the models, and a hand-written dark-pupil fit in C). This repo stays Python on the laptop, because there is no CPython/OpenCV environment for it on the board. Two known QNX limits we live with: the Camera Module 3 lens can't be focused through the sensor framework, and the streamers don't survive a reboot | The world-camera half in C/C++ on the board too, so the whole instrument is one device |
| 23 | Maestro latency | Two calls (OMNI decides, then ElevenLabs speaks) so the voice matches the *checked* action: ~1–3 s; a pre-generated "One moment" covers the gap and is cut off the moment real audio arrives | One streaming call with tool-calling, or the OMNI realtime API |
| 24 | Commands | Blink menu with 3 options per menu (left / right / both), two menus picked by where you look; fixed blink thresholds for everyone | Per-user blink calibration; deeper menus or dwell-scanning for more commands |
| 25 | Privacy | No microphone at all; the user asks nothing aloud, so every command is one of six blinks. A downscaled frame + scene JSON leave the device only when the user blinks a command; faces aren't blurred yet. The eye camera's pictures never leave the board | On-device blur of faces/people before upload |
| 26 | Assistant memory | Last 6 commands only, lost on restart | Per-user profile (preferred dwell, instruments, songs learned) |
| 27 | Reliability | Health numbers are reported and a stalled eye tracker can't fire a command (repeated landmarks aren't counted as new samples), but nothing restarts a stalled process | Supervisor process + heartbeat |
| 28 | Voices | One voice throughout: the menu clips and the live replies are the same ElevenLabs `voice_id`. Tone is a coarse `voice_settings` preset per mood, not real prosody direction, and the fallbacks (OMNI's own voice, then local TTS) do sound different | Prosody markup, or a cloned voice with per-sentence direction |
| 29 | Instrument sound | One ElevenLabs sample per instrument, repitched by resampling (notes far from middle C sound shorter/brighter or longer/duller) | A sample per octave, or a proper sampler with time-stretching |

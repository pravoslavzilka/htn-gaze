# Outer Vision — Spec (living)

## Goal
Head-mounted gaze tracker. Where you look on a table selects an object; the object plays a note.
- Colour → pitch · Shape → instrument · Distance → volume (farther = quieter)

## Decided
| Area | Decision |
|---|---|
| Rig | Head-mounted, coaxial: 1 eye camera (sees both eyes) and 1 scene camera back to back, rigid to the head, 80 mm from the eyes |
| Compute | Raspberry Pi 5 on **QNX 8.0** (direct Ethernet) runs the eye + scene camera streamers in C/C++; the laptop runs this repo and reads both over HTTP |
| Targets | ~5 still, wallet-sized objects, no overlap, clean table, indoor controlled light |
| Shapes / colours | round, square, cylinder, triangle / red, yellow, green, blue, purple = C major pentatonic (black/white dropped: shadows, glare) |
| Detection | Classical CV (colour threshold + contour shape), no VLM in the hot path |
| Selection | Continuous gaze stream; "lock" after 0.5 s dwell (tunable); low confidence → best guess |
| Calibration | Fixed-rig geometry plus a one-off aim zero; no per-user, per-session calibration, and no hand or voice input during use |
| Commands | Blinks only (2026-09-19): long blink opens a spoken menu, left / right / both-eye winks pick, double blink cancels. No microphone |
| Voice | ElevenLabs speaks everything (2026-09-19): the pre-generated menu prompts and OMNI's live replies use one `voice_id`. OMNI's own voice and local TTS stand behind it |
| Latency | 100–200 ms target |
| Output | Location only (no naming). Live debug overlay. Record sessions (compressed) |
| Language | Python |
| Success | Passes reliably in a live demo |

| Team | 4 people, 30 h. This repo = OUTER vision only; gaze tracking is a teammate's |
| Mapping | colour → pitch, shape → instrument, farther back → quieter; note plays once per look |
| Mode | Free play |
| Framing | Assistive instrument for people who can't use their hands (ALS, paralysis): gaze is their only input, so it must not fail |
| v3 outer vision | Colour LUT (8 colours) + ShapeCNN (ONNX/OpenCV) with rules fallback; picamera2 source; MJPEG camera/overlay streams; health in state. QNX removed |
| v4 (OMNI merged) | Blink menu → Maestro on OMNI (decides details from the camera view, checked against the command, offline defaults); ElevenLabs instrument samples + menu voice; lessons; optional Sentry. Shape labels from OMNI offline, pretrained MobileNetV3-small |
| v5 | **QNX rig connected**: `--gaze qnx` / `--source qnx` read the board's pupil + eyelid JSON and scene MJPEG directly, via a Python port of the eye team's coaxial rig model, pinned to their JavaScript by test. 5 colours on a pentatonic scale; triangle as a 4th shape (bell); ElevenLabs speaks every line |
| Rig | Cameras on glasses; Pi worn on the body; minimal markers (market as "works anywhere") |

## TODO / later
- [ ] **Variable distance** (v0 = one fixed distance). Needed for "farther = quieter"
- [ ] Multi-user calibration robustness; glasses wearers
- [ ] Moving targets
- [ ] Servo output (shelved in favour of music game)

## Hardware
- Raspberry Pi 5 running **QNX 8.0** on a direct Ethernet cable to the Mac; 2× Camera Module 3 (CSI units 4 = eye, 3 = scene), both at 960x540
- Check the board's address before a demo: this repo defaults to `192.168.2.2`, the eye repo's start script aliases `192.168.127.94`
- Pi 5 has 2 CSI ports and **no 3.5 mm audio jack**, so sound plays on the laptop (or a USB speaker)
- The eye camera sees both eyes, so left/right winks work
- QNX known limits: the Camera Module 3 lens can't be focused through the sensor framework; the streamers don't survive a reboot; use a 5 V / 5 A supply (the board has dropped off the network under load)
- RDK X5 dropped

## Prize targets (re-checked 2026-09-19)
| Track | Status |
|---|---|
| Finalist | primary: playful + assistive |
| Huawei OMNI Live | **built**, key in hand: Maestro decides instrument / note / dwell / song from the camera view + gaze focus + play history; OMNI also labels real shape crops offline. Its reply is spoken by ElevenLabs (its own voice is the fallback) |
| ElevenLabs | **built**: instrument samples (sound effects API) repitched per note, menu voice prompts (TTS), and every live reply streamed from `/text-to-speech/{id}/stream` with a tone preset. One voice throughout |
| QNX | **built**: the eye half runs on QNX 8.0 (MediaPipe ported to TFLite for QNX aarch64 plus a dark-pupil fit in C); this repo reads it over HTTP and converts with a tested port of their rig model |
| Solana ($5k) + Badge Hack ($2.5k) | built on the **`solana-marketplace`** branch, deliberately kept off `main` until the core demo is solid |
| Sentry | hooks built (`SENTRY_DSN`): frame/stage traces, lock + gesture logs, profiling. Judged on how the data changed the project, so actually use it to find and fix something |
| LeLamp / Bracket Bot | only if their hardware is free: a lamp that spotlights the object you're looking at (the original servo idea) |
| OpenAI / Baseten / Backboard | not a fit unless we route a model call through them |

## Open questions

See COMPROMISES.md for every v0 shortcut.

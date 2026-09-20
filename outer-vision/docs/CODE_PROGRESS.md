# Outer Vision: code progress

*As of 2026-09-19, ~22:30. Covers the outer (world-camera) vision + Maestro, and the link to the QNX eye rig.*

## What it is
A head-mounted, eye-played instrument. The world camera finds coloured objects on a table, gaze (from
the QNX eye rig, polled over HTTP, or from any tracker over UDP) picks one, and after a 0.5 s dwell it
fires a **lock**, which is a note:
**colour → pitch, shape → instrument, farther → quieter.** The user reconfigures it with **blinks only**:
a long blink opens a spoken menu, a left / right / both-eye wink picks, and **Maestro** (Huawei OMNI)
decides the details from the camera view; **ElevenLabs** says it.

```
QNX board :8080 ─► pupil + eyelids JSON ─► rig model (gaze_model.py) ─► gaze point + per-eye lid state
QNX board :8081 ─► scene MJPEG ─┐
       ─► colour lookup (5 colours, pentatonic, ~1 ms)   WHERE: pixel-exact outlines
       ─► shape CNN (ONNX via OpenCV) or rules  WHAT: round / square / cylinder / triangle / reject (hands, pens)
       ─► tracker (stable IDs, shape vote, size → distance → volume)
       ─► selector (gaze → nearest outline → dwell → one lock per look; frozen while eyes are closed)
eye lids ─► blink detector (long / left wink / right wink / double) ─► blink menu ─► OMNI ─► checked action
       ─► ElevenLabs streaming speech
       ─► UDP events (:5006) ─► synth (ElevenLabs samples)   ─► MJPEG overlay (--stream)   ─► recordings for replay
```

## Branches
| Branch | State | Tests |
|---|---|---|
| `main` | v5: QNX rig connected, 5 pentatonic colours, triangle instrument, ElevenLabs voice | 61 pass |
| `solana-marketplace` | Solana marketplace, eye-signed wallet, song minting. **Deliberately not on `main`** | n/a |
| `OMNI-branch` | merged into `main`; can be deleted | n/a |

## Version history
| Version | Commit | What changed |
|---|---|---|
| v0 | `7373e65` | Colour-range detection, contour shape rules, tracker, dwell selector, UDP events, overlay, record/replay, depth from size, ArUco calibration, synthetic scene + tests |
| v2 | `5971950` | 8 colours (nearest prototype); shape CNN with "reject" class; auto-labelled data + training; QNX camera bridge; MJPEG overlay; health stats |
| v3 | `5f324cf` | QNX removed; shape net on ONNX/OpenCV; `picam` source; Pi camera MJPEG server; Pi deploy scripts |
| merge | `3d5fcb4` | `OMNI-branch` merged: Maestro on OMNI, note/instrument mapping, lessons, `tools/synth.py`, Sentry |
| v4 | `ba60e5a` | **No microphone.** Blink detection (`blink.py`), blink menu (`menu.py`), OMNI decides from command + frame + scene + recent notes, actions checked against the command, offline defaults; selector freezes during blinks; ElevenLabs samples + menu prompts; lessons match enharmonics |
| v4.1 | `1e5d9df` | Shapes: `collect.py --label auto` → `omni_label.py` (OMNI labels 4×4 sheets, 2 shuffled passes must agree) → `train_shape.py --arch mobilenet` (pretrained MobileNetV3-small, camera-style augmentation); docs |
| v5 | this change | **QNX rig connected** (`gaze_model.py`, `--gaze qnx`, `--source qnx`, `tools/qnx_bridge.py`); 8 colours → 5 on a C major pentatonic; **triangle** as a 4th shape → bell, shape net retrained to 5 classes; **ElevenLabs speaks every line** (streaming, tone presets), OMNI's own voice demoted to fallback; docs corrected across the board; Solana leftovers gitignored off `main` |

## Blink menu (what the user can do)
| Long blink while looking at… | Left wink | Right wink | Both eyes |
|---|---|---|---|
| an object | OMNI picks a new instrument | OMNI picks a new note | look at another object + blink → swap notes |
| empty table | OMNI makes it slower | OMNI makes it faster | OMNI picks and teaches a song (during a lesson: stop it) |

Double blink = cancel, menu times out after 8 s, no notes while it's open. Keyboard stand-ins: `1`/`2`/`3`/`x`.

## Verified
*Everything below is synthetic or against a stand-in server unless it says "live".*
- Synthetic scenes: **250/250** shapes with the CNN and with the rules, now including triangles;
  **100%** of distractors rejected; grey table never detected.
- **QNX link, against a fake board** (`tests/test_qnx.py`, serves the same JSON as `pi/http_server.c`):
  gaze follows the pupil; a long left wink becomes a `long/left` gesture; closed eyes freeze dwell; a
  **stalled inference thread cannot open a menu** (repeated landmarks aren't counted as new samples) and
  a lost face isn't read as "both eyes closed"; an outage is reported and recovered from.
- **Rig model pinned to the eye team's JavaScript**: 13 pupil positions run through their
  `gui/src/geometry.js` under node and through our port agree to 2e-16; the golden values are in the test.
- Dwell: locks once after 0.5 s; blinks don't reset it; a 1 s closed-eye hold freezes dwell and keeps the target.
- Blinks: natural blinks ignored; long both / left / right classified; a brief squint of the other eye
  doesn't flip a wink; double blink; 3 s eye rest and 0.5 s half-blinks ignored; tracker dropout discarded.
- Menu: every path (instrument, note, swap incl. "same object" and "nothing", faster, slower, teach, stop
  lesson, cancel, timeout).
- Maestro against a mock OMNI server: no audio sent; command + scene (dwell, recent notes) reach the model;
  an action that doesn't fit the command is rejected and the offline default runs; OMNI down → default +
  fallback speech; no key → no network call.
- Voice routing: ElevenLabs speaks by default and is given the tone OMNI asked for; `speak_with: omni`
  uses the model's own voice; no ElevenLabs key falls through to OMNI's voice; everything down falls to
  local TTS, and the line is spoken **exactly once** in every case.
- **End-to-end over real UDP** (`run.py` + a fake eye tracker): long blink on an object → object menu →
  left wink → instrument changed; `gesture`, `assistant`, `music` events published.
- ElevenLabs sampler: pitch estimate within 1% on harmonic notes, no octave error with a strong 2nd
  harmonic; a C4 sample repitched to A4 measures 440.0 Hz at the mixer rate.
- MobileNetV3-small exports to ONNX and runs in the runtime OpenCV 4.10 (max diff vs torch ~1e-6),
  ~1 ms/crop on the Mac vs 0.2 ms for the tiny CNN.

- **Live APIs (2026-09-19 ~13:00).** ElevenLabs: 8 instrument samples + 8 menu prompts generated (1.7 MB, in
  `assets/`). The sound effects API returns *stereo* PCM (read as mono it was an octave low and half speed),
  now mixed down; piano/flute/synth measure 261–263 Hz (middle C). OMNI (`qwen3.5-omni-flash`, voice `Serena`;
  `Cherry` is not supported): decide 1.5–3.3 s (one outlier 7.4 s), first voice audio ~1 s after that.
  Live quirks handled: actions sent as an object instead of a list, command names used as action types, errors
  inside an HTTP 200 stream. 15 of 17 live commands decided by OMNI; the rest used the offline defaults.
  "teach" picks real songs (Twinkle Twinkle), and an unplayable song is sent back once with the missing notes.
  Note: that run was on the old 8-colour table. On the pentatonic table Twinkle is no longer playable (it
  needs F), and the prompt now tells OMNI so; that has not been re-run live.
- **Live end to end:** `run.py` + a fake eye tracker + `synth.py`: long blink → spoken menu → left wink →
  OMNI switched the object to piano and said why (first audio 3.3 s after the wink). That was with OMNI's
  own voice, before ElevenLabs became the default speaker.

## Not verified yet
- **Anything against the real QNX board.** The whole link is tested against a stand-in that serves the same
  JSON. Unknown until it is plugged in: the actual eyelid openness numbers (so `qnx.lid` thresholds are
  guesses), the residual aim (`--zero` has never been run), whether the rig's flip_x/flip_y are right, the
  real poll latency, and whether the board's address is `192.168.2.2` or `192.168.127.94`.
- **Real blinks** from a real wearer, and whether left/right winks separate cleanly in practice.
- **Real objects under real lighting**, including a real triangle, and whether MobileNet beats the tiny CNN
  on them. Synthetic tests are saturated (both score 100%), so this can only be decided on real crops.
- **ElevenLabs streaming TTS live**: `tts_stream` and the tone presets have not been run against the real
  API; only the non-streaming `tts` has.
- Speaker in the live loop.

## Next steps
1. **Plug in the board**, in this order: `tools/qnx_bridge.py --probe` (is it up, does `n=2`, do the lid
   numbers move when you blink) → set `qnx.lid` thresholds → `tools/qnx_bridge.py --zero` → `run.py
   --source qnx --gaze qnx`. If the gaze point moves the wrong way, set `qnx.rig.flip_x` / `flip_y`.
2. Listen to the samples (`tools/synth.py --test`); regenerate any you don't like with
   `tools/gen_audio.py --force --only <name>`. Check the live ElevenLabs reply voice with
   `tools/omni_check.py teach`.
3. Real shapes: `tools/collect.py --label auto` on the real table (≈3 min, with hands) → `tools/omni_label.py`
   → train both: `train_shape.py --real data/real --arch tiny --out models/shape_tiny` and
   `--arch mobilenet --out models/shape_mnv3`. Ship the one with the higher `real_val_acc` (validation holds
   out whole object tracks, so near-duplicate crops can't inflate it) by pointing `config.json` →
   `shape_net.model_dir` at it. Check `data/review/` for crops OMNI wasn't sure about.
4. Tune `blink` thresholds on the wearer's real blinks (record with `r`, blinks replay from the log).
5. Solana: the branch is ready, merge it only once the core demo is solid.

## Prize tracks
| Track | Status |
|---|---|
| Finalist | Primary target (playful + assistive: blinks are the only input) |
| Huawei OMNI Live | Built and run live: Maestro decides from the camera view + gaze focus + play history; OMNI also labels shape data |
| ElevenLabs | Built: generated instrument samples, menu voice, and now every live reply (streaming, tone presets). Samples + prompts run live; streaming replies not yet |
| Solana ($5k) / Badge Hack ($2.5k) | Built on the `solana-marketplace` branch, kept off `main` |
| Sentry | Hooks built; only worth it if the traces are used to fix something |
| QNX | The eye half runs on it; this repo talks to it over HTTP and converts with a tested port of their rig model |

## Where things are
| Path | Purpose |
|---|---|
| `run.py` | Main loop (sources, gaze + blinks, detection, dwell, menu, Maestro, events, overlay, recording) |
| `outer_vision/` | `detector`, `shape_net`, `tracker`, `gaze_model`, `selector`, `blink`, `menu`, `music`, `omni`, `voice`, `eleven`, `audio`, `pitch`, `io`, `overlay`, `synthetic`, `telemetry`, `config` |
| `tools/` | `find_board`, `qnx_bridge`, `synth`, `gen_audio`, `omni_check`, `omni_label`, `collect`, `train_shape`, `tune_colors`, `pi_camera_server`, `listen`, `send_gaze`, `make_marker` |
| `deploy/` | Raspberry Pi **OS** only (a spare board as a plain world camera): `deploy_pi.sh`, `setup_pi.sh`. The QNX board is built from the eye repo's `pi/` |
| `README.md` · `INTERFACE.md` · `COMPROMISES.md` · `SPEC.md` | Usage · message formats · known shortcuts · decisions |

# Devpost draft: sponsor sections

Everything marked **[TODO]** must be filled with something we actually measured or captured. Do not publish a claim that has no
number or screenshot behind it. Facts about what the software does are written as plain statements because they are in the code
and were exercised in tests; results are placeholders on purpose.

---

## Project one-liner
Glasses that know what you are looking at: two cameras (one on the eye, one on the scene), a gaze pipeline, and an OMNI voice
assistant you talk to hands-free ("OMNI, look at this colour and make a sound for it").

## How it fits together
```
eye camera + scene camera (QNX Raspberry Pi 5)
   -> gaze app on the laptop: pupil -> gaze point in the scene picture, colour/shape objects, which object is fixated
   -> per-frame packets (UDP JSON) -> receiver -> Tiger Data (hypertable, continuous aggregate)  +  Sentry (traces, logs)
   -> OMNI voice loop: microphone audio + scene picture + close-up crop of the gaze point -> OMNI -> spoken answer + actions
   -> dashboard (live view, attention, OMNI history, system health, run comparison) with Sentry Session Replay
```

---

## Huawei OMNI Live
**What the wearer can do (end to end, running):** say "OMNI, look at this colour and create a sound for it."
1. **Speech / audio:** the microphone is always listening. Speech is cut out with a voice-activity detector and sent to the
   OMNI model as audio. A wake word ("OMNI") and a 10-second follow-up window keep it hands-free; "OMNI, go to sleep" ends it.
2. **Vision / video:** the request carries the scene camera picture with the gaze pointer, a close-up crop around the gaze point,
   and what the gaze app measured under the gaze (colour, shape, note).
3. **Language:** OMNI resolves "this" to the object being looked at and answers with a JSON plan (what to say, which action).
   The system executes the action: ElevenLabs generates the sound, it is saved to a library in Tiger Data, and the reply is
   spoken back ("On it." ... "It was added.").
Then: "OMNI, play the pink tone" plays it back from the library.

- Model and API: `qwen3.5-omni-flash` through the OMNI Live OpenAI-compatible endpoint (streaming chat completions), called from Python.
- Honest scope: the glasses are the wearable; the app that talks to OMNI runs on a tethered laptop, not on the glasses' own compute.
- Safety rule in code: blue, green, yellow, red and orange sounds cannot be created, replaced or deleted (application check plus a
  database trigger). The model only proposes actions; the code decides.
- Reliability details: OMNI's own voice is ignored when it comes back through the microphone; OMNI/ElevenLabs/Tiger failures are
  logged and the loop keeps running.
- **[TODO]** typical OMNI understand latency: ____ ms (median of ____ requests; from `omni_interactions`).
- **[TODO]** demo video link: ____ . **[SCREENSHOT]** OMNI history panel showing a real exchange.

## MLH Tiger Data
- **Hypertables:** `gaze_frames` (one row per frame: stage timings, confidence, gaze point, object) and `omni_interactions`.
- **Continuous aggregate `gaze_10s`:** 10-second buckets per run and object with frame count (dwell), average/max latency and
  tracking-loss rate, in **real-time mode** (`materialized_only = false`) so the live dashboard includes the newest rows,
  with an automatic refresh policy. The attention panels read this aggregate, not the raw table.
- **Batching:** the receiver inserts about once a second, never per frame, and buffers (bounded) with retry if Tiger is unreachable.
- **`sound_library` table** holds the sounds OMNI creates (audio as bytea) with a **trigger** that hard-blocks the five protected colours from any client.
- **Reads under load:** a circuit breaker keeps the dashboard responsive when the database is down.
- A real bug we found and fixed: after five repeats the driver switched to a prepared statement and TimescaleDB's generic plan for
  `ORDER BY time DESC LIMIT $1` returned **duplicate rows** (8 rows from a 5-row table). Auto-prepare is now off for reads.
- **[TODO]** rows stored: ____ over ____ runs. Insert rate: ____ rows/s. Aggregate query time vs raw-table query time for the same
  60 s window: ____ ms vs ____ ms (measure both, run each a few times).
- **[SCREENSHOT]** dashboard attention panels; the `gaze_10s` definition.

## Sentry
Products used beyond error monitoring:
- **Tracing:** each sampled frame becomes a `gaze_pipeline` transaction rebuilt from the durations the pipeline reports, with child
  spans `cap`, `pupil`, `gaze`, `scene`, `fix` ending at the receive time. Sampling is 1 in N frames **plus** anomalous frames
  (rate-limited), so the interesting moments are always captured. Every voice request is an `omni_request` transaction with spans for the
  OMNI call, ElevenLabs speech and sound generation.
- **Logs:** structured `tracking_lost` / `tracking_recovered` (on transitions), `packet_loss` and Pi events (`blink`, `camera_error`,
  `calib_drift`) with `run` and `frame`, emitted inside the trace where one exists.
- **Profiling:** a `PROFILE=1` switch (off by default: the SDK is not even imported) wraps a few seconds of frames in one transaction so
  the continuous profiler attaches a profile; for test runs only.
- **Session Replay** and browser tracing on the dashboard (the OMNI transcript panel is masked).
- Privacy hygiene: Sentry trace headers are never sent to OMNI, ElevenLabs or Tiger.
- **How Sentry data changed what we built. [TODO: only fill this with something that really happened]**
  Slow stage found: ____ (trace link ____ ). What we changed: ____ . Second run, same conditions: p95 ____ ms -> ____ ms
  (run ids ____ and ____ on the dashboard's compare panel).
  Note for the writer: in this pipeline `cap` is mostly waiting for the next camera frame (cap + processing is about one 33 ms frame
  period at 30 fps), so a long `cap` span is not compute; check that before calling it a bottleneck.
- **[SCREENSHOT]** trace waterfall of a `gaze_pipeline` transaction; the Logs view filtered on `tracking_lost`; a profile; a Replay.
- **[SCREENSHOT]** the three Sentry dashboard widgets (p95 by span op, tracking_lost logs over time, OMNI latency).

---

## Things we got wrong and fixed (good "what we learned" material)
- We first scored "iris-centre fallback" as lost tracking, overstating tracking loss; the scale is now 0 = no gaze, 0.5 = iris fallback, up to 1 = dark-pupil fit.
- The always-on microphone heard OMNI's own voice and the instrument's notes, which kept the assistant awake; echo, noise and note-muting fixes.
- Letting the model decide what is "protected" made it refuse harmless requests, so decisions moved into code.

## Honest limitations
- The QNX board reports only an inference rate, so the "pupil" stage is the laptop-side gaze-model time, not the on-board pupil fit.
- "Frames" in Tiger are the gaze app's status updates (about 6 per second), not camera frames.
- `calib_drift` is a heuristic (gaze consistently landing away from the object it selects), not a measured calibration error.

## Built with
Python, FastAPI, Chart.js, PostgreSQL/TimescaleDB (Tiger Data), Sentry (Python + browser SDKs), Huawei OMNI Live (qwen3.5-omni-flash), ElevenLabs, OpenCV, QNX on Raspberry Pi 5.

# Sponsor descriptions (two per sponsor)

Each sponsor has a **short** description (about 100 words, for a submission form field) and a **long** one (for the Devpost
body, mapped to the judging criteria). Numbers were measured on 2026-09-20 against our real Tiger database and real APIs;
re-run the named command before submitting if the data has grown. **[FILL]** marks something we have not observed yet: leave it
out or fill it with a real result, never a guess.

---

# 1. Sentry, "Best use of Sentry"

## Short
We use Sentry well beyond errors: **Tracing, Logs, Profiling and Session Replay**. Our gaze pipeline reports only stage durations,
so we *rebuild* a `gaze_pipeline` trace per sampled frame (spans cap, pupil, gaze, scene, fix, anchored to the receive time) and
always capture anomalous frames. Tracking loss, packet loss and Pi events are structured logs attached to the frame's trace. Every
OMNI voice request is a trace with spans for the model call and ElevenLabs. The dashboard runs Replay, and a `PROFILE=1` switch (SDK
not even imported when off) profiles test runs. **[FILL: the slow stage we found in a trace, and the second run that proved the fix.]**

## Long
**What we built.** The system is a head-mounted gaze tracker plus a voice assistant, so "system health" means frame latency,
tracking quality and the voice round trip. We instrumented all three with Sentry products beyond error monitoring.

**Tracing, with a twist.** The pipeline is a set of stages (capture, pupil, gaze mapping, scene detection, fixation) whose
durations arrive as numbers in a UDP packet; nothing runs "inside" the receiver. So the receiver *reconstructs* a transaction
`gaze_pipeline` with five child spans laid end to end and anchored so the transaction ends at the laptop's receive time (using
`start_timestamp` / `finish(end_timestamp=)`, checked against sentry-sdk 2.69). Sampling is one frame in N **plus** any frame
with a warning-level anomaly, rate-limited to one per second, so the interesting moments are never sampled away. A custom
`traces_sampler` drops the dashboard's own one-second HTTP polling, which would otherwise drown the real traces.

**Logs that point somewhere.** Tracking loss is logged on *transitions* (`tracking_lost`, `tracking_recovered`), not per frame;
`packet_loss` and each Pi event (`blink`, `camera_error`, `calib_drift`) carry `run` and `frame` attributes and are emitted inside
the frame's trace when it is traced, so a log line links straight to its waterfall.

**OMNI and voice observability.** Each voice request is an `omni_request` transaction with spans for the OMNI model call
(`gen_ai.chat`, model name and modality flags as data), ElevenLabs speech, and ElevenLabs sound generation, including the work that
runs on a background thread. Clips that turn out to be noise or echo are dropped from Sentry rather than sent.

**Profiling and Replay.** `PROFILE=1` (test runs only) initialises the SDK with the newer continuous profiler
(`profile_session_sample_rate`, `profile_lifecycle="trace"`) and wraps a few seconds of frames in one `gaze_chunk` transaction so
profiles attach; with the flag unset the SDK is never imported (there is a test for that). The dashboard runs the browser SDK with
Session Replay and tracing; the panel that shows what people said to OMNI is masked.

**Care about the integration.** We turned off trace-header propagation after SDK debug output showed `sentry-trace`/`baggage`
headers being attached to our calls to third-party APIs, and switched a deprecated `set_measurement` call to `set_data`.

**What we verified.** Ingest accepted our events (HTTP 200 with an event id); transactions, logs and a `profile_chunk` were sent
without errors; the dashboard reports Replay and tracing active; 42 automated tests cover the trace reconstruction maths, logging
transitions, the profiling switch and the tooling. **We have not yet confirmed how each item looks in the Sentry UI.**

**How observability shaped the build.** **[FILL, only with something real:]** *Trace of frame ____ showed the `____` span at
p95 ____ ms. We changed ____. Run `____` vs `____` on the dashboard's compare panel: p95 ____ ms -> ____ ms.* One interpretation
lesson we can state truthfully today: in this pipeline `cap` is mostly waiting for the next camera frame (cap + processing is about
one 33 ms frame period at 30 fps), so a long `cap` span is not a compute bottleneck.

**Screenshots to attach [FILL]:** a `gaze_pipeline` waterfall; the Logs view filtered on `tracking_lost`; a `gaze_chunk` profile;
a dashboard Replay; the Sentry dashboard widgets (p95 by span op, tracking_lost logs over time, OMNI latency).

---

# 2. MLH Tiger Data

## Short
Tiger Data is the **product-data backbone** of our gaze tracker. Every frame is batch-inserted (about once a second) into a
`gaze_frames` **hypertable**; a **real-time continuous aggregate** in 10-second buckets per run and object powers the dashboard's
attention panels (measured 1.1 ms vs 9.3 ms on the raw table at 25k rows). Voice interactions and a **sound library** sit in the
same PostgreSQL database, with a database **trigger** that locks five protected sounds. **Compression** measured at 84% smaller on
our real rows. Reads survive an outage with a circuit breaker.

## Long
**Why a time-series database fits.** A gaze pipeline emits a constant stream of small measurements (stage timings, confidence, gaze
point, the object being looked at). We needed to keep it all, ask "what was looked at over the last minute?", and compare runs
before and after a change, without running two databases.

**Hypertables and batching.** Frames go into the `gaze_frames` hypertable (time, run, frame id, five stage timings, confidence,
gaze point, object). The receiver never writes per frame: it buffers and flushes about once a second on its own thread, retries with
backoff if Tiger is unreachable, and keeps a bounded buffer. `omni_interactions` (what the wearer asked OMNI, the answer, latency)
is a second hypertable, so product questions can be answered next to the sensor data.

**Continuous aggregates for instant dashboards.** `gaze_10s` groups by 10-second bucket, run and object: frame count (dwell),
average and maximum total latency, tracking-loss rate. It runs in **real-time mode** (`materialized_only = false`), so the newest
seconds appear on the dashboard before they are materialized, and a refresh policy runs every 30 seconds (83 runs, 0 failures when we
checked). The dwell and timeline panels read only this aggregate. Measured server-side on about 25,000 raw rows: the same attention
query takes **9.3 ms on the raw table and 1.1 ms on the aggregate (8.6x)**; the dashboard's end-to-end API calls take 28 to 50 ms, mostly
network round trip to the cloud.

**Compression.** Compression is enabled (segment by run, order by time) with a policy. We measured it on a copy of our real rows so the
live table was untouched: **24,696 rows went from 3,080 kB to 480 kB, 84.4% smaller** (82 to 85% across the settings we tried). That is
below a round "90%" because our data is mostly noisy floating-point timings; `python laptop/measure_compression.py` prints the current
figure and it should improve as the table grows. **[FILL after the event: rows stored ____, size before/after ____.]**

**One database for relational and metric data.** The same PostgreSQL holds the frame stream, OMNI interactions, and a relational
`sound_library` (name, prompt, audio as `bytea`, play count). A **`BEFORE INSERT OR UPDATE OR DELETE` trigger** rejects any write for the
five protected colours, from any client, so a business rule lives in the database and not only in our Python. All of it is plain SQL.

**Engineering lessons.** The dashboard exposed a real bug: after five repeats the driver switched to a prepared statement and
TimescaleDB's generic plan for `ORDER BY time DESC LIMIT $1` returned **duplicate rows** (8 rows from a 5-row table); we turned off
auto-prepare for reads. A circuit breaker returns in about 2 ms while Tiger is down, so one-second polling can never pile up; live
counters keep working and frames are buffered for retry.

**Screenshots to attach [FILL]:** attention panels; the `gaze_10s` definition; `measure_compression.py` output; `tiger_report.py` output.

---

# 3. ElevenLabs

## Short
ElevenLabs gives the glasses a voice **and** an instrument. The assistant speaks every reply through streaming text-to-speech
(first audio in about 170 to 210 ms). Then it goes further: when the wearer says "OMNI, make a sound for pink," OMNI writes the
sound description itself and ElevenLabs' sound-effects model generates it (about 1.5 s for 2 s of audio); it is saved to a library
and played back on request. The team also pre-generated every instrument sample and menu prompt with ElevenLabs. The whole loop is
hands-free and autonomous.

## Long
**A fully autonomous audio loop.** The wearer says "OMNI, look at this colour and create a sound for it." With no button press,
OMNI hears the request, works out which colour is being looked at, writes a sound-effect prompt suited to that colour, and answers
"On it." in an ElevenLabs voice **while** ElevenLabs generates the sound in the background; then it says "It was added" and plays the
new sound. Later, "OMNI, play the pink tone" replays it from the library. Nobody records, edits or chooses audio.

**Text-to-speech as the assistant's voice.** Every reply is streamed from the text-to-speech API (`eleven_flash_v2_5`, raw PCM at
24 kHz) into the speaker as chunks arrive, so speech starts before the sentence is finished: **first audio in 169 to 209 ms** over
three measurements. Speech uses a lock so replies never talk over each other, and the microphone is muted while the assistant speaks
so it never hears itself.

**Sound generation as a creative tool.** The sound-effects API is used two ways.
1. *Live:* OMNI generates a per-colour sound on demand (2.0 s of audio in about 1.5 s). We handle a real quirk of the API: it returns
   interleaved **stereo** PCM, which plays an octave low if read as mono, so it is mixed down.
2. *Ahead of time:* the team's instrument app uses ElevenLabs-generated notes for every instrument (nine samples) and its menu voice
   lines (eight prompts), pre-rendered so the demo never depends on venue Wi-Fi.

**Guardrails and reliability.** Five colours are protected: the code refuses before any ElevenLabs credit is spent, and a database
trigger enforces it too. If ElevenLabs is unreachable, the failure is logged, the assistant says so, and the loop keeps running.
Sounds are stored in Tiger Data and cached locally, so playback works even if the network is down.

**What we did not build.** The live assistant uses one consistent voice; it does not vary emotion or style per reply (our teammate's
module has tone presets, but they are not wired into the live loop). **[FILL if added: the tone control and a before/after clip.]**

**Assets to attach [FILL]:** a short demo clip of the create-then-play flow; the library listing from `python laptop/tiger_report.py`.

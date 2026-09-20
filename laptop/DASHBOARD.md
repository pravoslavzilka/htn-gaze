# Laptop side: dashboard, Tiger, Sentry, OMNI

```
gaze app (Outer-Vision-Network run.py) --status file--> ovn_bridge.py --UDP :9999--> app.py (receiver + FastAPI)
                                                                                      |-> Tiger  (batched inserts, reads)
                                                                                      |-> Sentry (traces, logs)
omni_voice.py (mic -> OMNI -> ElevenLabs) ----------------------------------------> Tiger (omni_interactions) + Sentry
browser  http://127.0.0.1:8800  <-- /api/*  and the gaze app's MJPEG stream (:8790)
```

## Run

```
pip install -r requirements.txt          # secrets go in laptop/.env (see .env.example), never in git
python app.py                            # dashboard + UDP receiver on :9999   -> http://127.0.0.1:8800
python ovn_bridge.py                     # gaze app status -> UDP; follows the dashboard's "New run" button
python omni_voice.py --run live          # say "OMNI ..."; 10 s awake window; "OMNI, go to sleep"
python tiger_report.py                   # the same numbers in the terminal
python -m unittest discover tests
```

The gaze app must be started with `--status-file` (the launcher does) and `--stream 8790` for the live picture.

## Demo reset (seconds)
- **New run**: names a new run; the bridge switches to it at frame 0. Nothing is deleted.
- **Reset**: clears the live counters; OK on the prompt also deletes the selected run from Tiger.
- The dashboard and bridge tolerate the gaze app or Pi disappearing and coming back. Tiger, Sentry, OMNI or ElevenLabs
  being down is logged and skipped: frames buffer (50k rows) and are retried with backoff.

## What is measured, and what is not
- Real: the five stage timings from `run.py` (cap = frame read + resize, pupil = pupil/iris -> gaze model on this laptop,
  gaze = calibration mapping, scene = colour detect + track, fix = fixation select), the object looked at, the gaze
  point, blinks, and confidence (0 = no gaze, 0.5 = iris fallback, up to 1 with dark-pupil fits).
- Not measured: the QNX board's own pupil-fit time (it only reports an inference rate). The Python Pi pipeline from the
  original brief does not exist; `sender/gaze_sender.py` is ready for it.
- Frames are the app's status writes (about 6 per second), not camera frames.
- Tracking-loss thresholds: 0.5 (`TRACKING_LOST_CONF`); the Tiger aggregate has the same 0.5 baked in.

## Division of roles
- **Tiger** = product data: what was looked at, dwell, attention, OMNI interactions (`gaze_frames`, `gaze_10s`, `omni_interactions`).
- **Sentry** = system health: stage latency, tracking loss, errors, OMNI/ElevenLabs call performance, dashboard replays.

## Sound library (OMNI writes and plays tones)
- "OMNI, create a sound for pink" -> OMNI asks ElevenLabs, saves the sound to Tiger (`sound_library`: name, prompt, seconds,
  audio as PCM16 24 kHz) and to `laptop/sounds/` as a cache, says "It was added", and plays it.
- "OMNI, play the pink tone" -> plays it from Tiger (from the cache if Tiger is down). "What sounds do you have?" is answered
  from the library list. Cached sounds Tiger does not have yet are uploaded at startup.
- **Hard no: blue, green, yellow, red, orange** can never be created, replaced or deleted. Enforced three times: the code refuses
  before any ElevenLabs credit is spent (also for compound names like "dark red"), a Postgres trigger on `sound_library` rejects
  the write from any client, and the model is only told to emit the action (the code decides and speaks the refusal). Playing
  a protected colour says it is built in: look at the object to hear the instrument's own tone.
- Old red/green files in `laptop/sounds/` predate the lock. They are left untouched and are not served or imported.

## Sentry
- **Tracing**: `gaze_pipeline` transactions are rebuilt from the packet durations for 1 in `TRACE_EVERY_N` frames (30) and
  for anomalies (at most one per second), with child spans `cap pupil gaze scene fix`, ending at the receive time.
  Every voice request is an `omni_request` transaction with `gen_ai.chat` (the OMNI call), `elevenlabs.tts` and
  `elevenlabs.sound_generation` spans. Noise/echo clips are dropped, not sent.
- **Logs**: `tracking_lost` / `tracking_recovered` (on transitions), `packet_loss`, and `pi_event_<name>`, all with `run`
  and `frame` attributes, emitted inside the frame's trace when it is traced.
- **Replay + browser tracing** on the dashboard. The OMNI history panel is masked in replays (`data-sentry-mask`).
- Sentry trace headers are NOT sent to OMNI / ElevenLabs / Tiger (`trace_propagation_targets=[]`).
- Optional: `SENTRY_AUTH_TOKEN` (read-only), `SENTRY_ORG_SLUG`, `SENTRY_PROJECT_SLUG` enable the issues list and org-specific
  links; without them the buttons open the project pages.

### Widgets to create in Sentry's own Dashboards (Dashboards -> Create -> Add widget)
Field names differ slightly between Sentry UI versions; if a field is missing, pick the closest in the query builder.
1. **Pipeline p95 by stage** - Dataset: *Spans*. Query: `transaction:gaze_pipeline span.op:[cap,pupil,gaze,scene,fix]`.
   Y-axis: `p95(span.duration)`. Group by: `span.op`. Display: bar or line. (Add `p50` for a second series.)
2. **Tracking lost over time** - Dataset: *Logs*. Query: `message:tracking_lost`. Y-axis: `count()`. Group by: `run`. Display: line, 1m interval.
3. **OMNI call latency** - Dataset: *Spans*. Query: `span.op:gen_ai.chat`. Y-axis: `p50(span.duration)` and `p95(span.duration)`. Display: line.
   Companion widget: same with `span.op:elevenlabs.sound_generation`, and `transaction:omni_request` for the whole request.
4. (Optional) **Pi events and packet loss** - Dataset: *Logs*. Query: `message:pi_event_* OR message:packet_loss`. Y: `count()`. Group by: `message`.

Do not claim a Sentry finding until it has been observed in these widgets: find a slow stage in traces, fix it, and verify
with a second run (compare runs on the dashboard).

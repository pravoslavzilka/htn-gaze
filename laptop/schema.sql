-- Run statement by statement (tiger.py does this): CREATE MATERIALIZED VIEW ... continuous cannot run
-- inside a transaction block, so this file must not be sent as one multi-statement string.
-- Everything is idempotent.

CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS gaze_frames (
  time TIMESTAMPTZ NOT NULL, run TEXT, frame_id INT,
  cap_ms REAL, pupil_ms REAL, gaze_ms REAL, scene_ms REAL, fix_ms REAL,
  conf REAL, gx REAL, gy REAL, obj TEXT, obj_conf REAL);

SELECT create_hypertable('gaze_frames', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS gaze_frames_run_time ON gaze_frames (run, time DESC);

-- What the wearer asked OMNI, what it answered, and how long the understand step took.
CREATE TABLE IF NOT EXISTS omni_interactions (
  time TIMESTAMPTZ NOT NULL, run TEXT, object TEXT, transcript TEXT, answer TEXT, latency_ms REAL);

SELECT create_hypertable('omni_interactions', 'time', if_not_exists => TRUE);

-- 10-second buckets per run and object. Frame count = dwell (frames / fps = seconds). Total latency is the sum of the
-- stages that were measured, and NULL for frames with no timings at all (scene_ms is the marker), so untimed frames never
-- drag the average to zero.
-- Tracking loss uses conf < 0.5 (the receiver's TRACKING_LOST_CONF default); change both together.
-- materialized_only = false: queries also read the not-yet-materialized newest rows, so the live demo is current.
CREATE MATERIALIZED VIEW IF NOT EXISTS gaze_10s
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT time_bucket('10 seconds', time) AS bucket, run, obj,
       count(*)::int AS frames,
       avg(CASE WHEN scene_ms IS NOT NULL THEN coalesce(cap_ms,0) + coalesce(pupil_ms,0) + coalesce(gaze_ms,0) + scene_ms + coalesce(fix_ms,0) END)::real AS avg_total_ms,
       max(CASE WHEN scene_ms IS NOT NULL THEN coalesce(cap_ms,0) + coalesce(pupil_ms,0) + coalesce(gaze_ms,0) + scene_ms + coalesce(fix_ms,0) END)::real AS max_total_ms,
       avg((conf IS NULL OR conf < 0.5)::int)::real AS tracking_loss_rate
FROM gaze_frames
GROUP BY bucket, run, obj
WITH NO DATA;

SELECT add_continuous_aggregate_policy('gaze_10s',
  start_offset => INTERVAL '1 hour', end_offset => INTERVAL '10 seconds',
  schedule_interval => INTERVAL '30 seconds', if_not_exists => TRUE);

-- Sounds OMNI created, playable by name. audio = raw mono PCM16 at sample_rate.
CREATE TABLE IF NOT EXISTS sound_library (
  name TEXT PRIMARY KEY, prompt TEXT, seconds REAL, sample_rate INT NOT NULL DEFAULT 24000, audio BYTEA NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  created_by TEXT DEFAULT 'omni', plays INT NOT NULL DEFAULT 0);

-- HARD NO: these colours can never be created, replaced or deleted in the library, by anyone (also enforced in code).
CREATE OR REPLACE FUNCTION protect_builtin_colors() RETURNS trigger AS $$
BEGIN
  IF lower(coalesce(NEW.name, OLD.name)) = ANY (ARRAY['blue', 'green', 'yellow', 'red', 'orange']) THEN
    RAISE EXCEPTION 'sound "%" is a protected colour and cannot be created, changed or deleted', coalesce(NEW.name, OLD.name);
  END IF;
  IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
  RETURN NEW;
END
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER protect_builtin_colors_trg
BEFORE INSERT OR UPDATE OR DELETE ON sound_library
FOR EACH ROW EXECUTE FUNCTION protect_builtin_colors();

CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS gaze_frames (
  time TIMESTAMPTZ NOT NULL, run TEXT, frame_id INT,
  cap_ms REAL, pupil_ms REAL, gaze_ms REAL, scene_ms REAL, fix_ms REAL,
  conf REAL, gx REAL, gy REAL, obj TEXT, obj_conf REAL);
SELECT create_hypertable('gaze_frames', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS gaze_frames_run_time ON gaze_frames (run, time DESC);

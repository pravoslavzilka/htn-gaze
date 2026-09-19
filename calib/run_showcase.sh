#!/bin/bash
# calibrate -> (if a new model was produced) start the live gaze view and open it
cd "$(dirname "$0")"
touch .showcase_marker   # anything newer than this is output of this run
# use the local virtual environment if there is one (../oak-venv), otherwise whatever "python" is on PATH,
# or set PY=/path/to/python yourself
if [ -x "../oak-venv/Scripts/python.exe" ]; then PY="${PY:-../oak-venv/Scripts/python.exe}"; else PY="${PY:-python}"; fi
$PY -u calibrate.py --hold 4.5 --eye http://169.254.96.94:8080 --scene http://169.254.96.94:8081 > calib_run.log 2>&1
if [ -n "$(find run_* -newer .showcase_marker -name gaze_model.json 2>/dev/null)" ]; then
  echo "calibration done - starting live gaze" >> calib_run.log
  powershell -NoProfile -Command "Start-Sleep 3; Start-Process 'http://127.0.0.1:8766/'" &
  $PY -u gaze_live.py > gaze_live.log 2>&1
else
  echo "no new model produced - live gaze not started" >> calib_run.log
fi

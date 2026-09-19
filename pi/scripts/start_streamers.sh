#!/bin/sh
# Start both camera streamers on the QNX board (they do NOT survive a reboot).
#   eye camera   : CSI unit 4, port 8080, face mesh + iris landmarks, image rotated 180 deg
#   scene camera : CSI unit 3, port 8081, video only (--no-infer), image rotated 180 deg
# Adjust --unit / --rotate to your mounting. Units 3 and 4 are the two CSI ports on a Raspberry Pi 5.
cd "$HOME/gazecomp" || exit 1
export LD_LIBRARY_PATH="$HOME/gazecomp/third_party/tflite/lib"

slay -f -9 camera_streamer 2>/dev/null   # the old process can survive a plain slay
sleep 3

nohup ./camera_streamer --http 8080 --width 960 --height 540 --unit 4 --rotate 180 \
    >/tmp/stream_u4.log 2>&1 </dev/null &
sleep 3   # start the second one after the first has opened its camera
nohup ./camera_streamer --http 8081 --width 960 --height 540 --unit 3 --rotate 180 --no-infer \
    >/tmp/stream_u3.log 2>&1 </dev/null &

sleep 8
grep -E "Camera unit|failed" /tmp/stream_u4.log /tmp/stream_u3.log

# optional: log temperature and free memory every 2 s to ~/health.log
# nohup ./scripts/health.sh >/dev/null 2>&1 </dev/null &

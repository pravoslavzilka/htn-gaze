#!/bin/sh
# Build camera_streamer on the QNX board (no SDP needed; the image ships clang++).
# Run from anywhere; expects the repo contents copied to ~/gazecomp (see README).
set -e
cd "$HOME/gazecomp"
TFLIB="$HOME/gazecomp/third_party/tflite/lib"
g++ -O3 -Wall -Iinclude -Ithird_party/tflite/include \
  -o camera_streamer.new \
  camera_streamer.c jpeg_enc.c http_server.c process.c face_lm.cpp \
  -lcamapi -lsocket -lm /usr/lib/libturbojpeg.so.0 \
  -L"$TFLIB" -Wl,-rpath-link,"$TFLIB" -Wl,-rpath,"$TFLIB" \
  -ltensorflow-lite -lXNNPACK -lcpuinfo -lpthreadpool -lkleidiai \
  -lfarmhash -lfft2d_fftsg -lfft2d_fftsg2d -leight_bit_int_gemm \
  "$TFLIB"/libabsl_*.so.2508.0.0
echo "build ok: camera_streamer.new"
# Swap it in only after the running streamers are stopped (they hold the camera):
#   slay camera_streamer; mv -f camera_streamer.new camera_streamer

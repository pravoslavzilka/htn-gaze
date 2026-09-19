#!/bin/sh
set -e
cd "$HOME/gazecomp"
echo "building MediaPipe TFLite camera_streamer in $(pwd)"
TFLIB="$HOME/gazecomp/third_party/tflite/lib"
g++ -O3 -Wall -Iinclude -Ithird_party/tflite/include \
  -o camera_streamer.new \
  camera_streamer.c jpeg_enc.c http_server.c process.c face_lm.cpp \
  -lcamapi -lsocket -lm /usr/lib/libturbojpeg.so.0 \
  -L"$TFLIB" \
  -Wl,-rpath-link,"$TFLIB" \
  -Wl,-rpath,"$TFLIB" \
  -ltensorflow-lite -lXNNPACK -lcpuinfo -lpthreadpool -lkleidiai \
  -lfarmhash -lfft2d_fftsg -lfft2d_fftsg2d -leight_bit_int_gemm \
  $TFLIB/libabsl_*.so.2508.0.0
echo "build ok"
slay camera_streamer || true
sleep 2
mv -f camera_streamer.new camera_streamer
chmod +x camera_streamer
export LD_LIBRARY_PATH="$HOME/gazecomp/third_party/tflite/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
nohup ./camera_streamer --http 8080 --width 960 --height 540 --unit 4 \
  >/tmp/gazecomp_stream.log 2>&1 </dev/null &
echo "started pid $!"
sleep 1
tail -n 30 /tmp/gazecomp_stream.log || true

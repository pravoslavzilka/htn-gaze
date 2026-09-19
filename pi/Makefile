# On-device QNX gcc/g++ (no SDP required):
#   make            (from ~/gazecomp on the board)
#
# Host SDP cross-compile:
#   source ~/qnx800/qnxsdp-env.sh && make

ARTIFACT := camera_streamer
CSRCS    := camera_streamer.c jpeg_enc.c http_server.c process.c
CPPSRCS  := face_lm.cpp
TFL      := third_party/tflite

CFLAGS   += -O2 -Wall -Wextra -Iinclude
CXXFLAGS += -O3 -Wall -Iinclude -I$(TFL)/include
LIBS     += -lcamapi -lsocket -lm /usr/lib/libturbojpeg.so.0 \
            -L$(TFL)/lib -Wl,-rpath-link,$(TFL)/lib -Wl,-rpath,$(TFL)/lib \
            -ltensorflow-lite -lXNNPACK -lcpuinfo -lpthreadpool -lkleidiai \
            -lfarmhash -lfft2d_fftsg -lfft2d_fftsg2d -leight_bit_int_gemm \
            $(wildcard $(TFL)/lib/libabsl_*.so.*)
CC       ?= gcc
CXX      ?= g++

.PHONY: all clean

all: $(ARTIFACT)

$(ARTIFACT): $(CSRCS) $(CPPSRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(CSRCS) $(CPPSRCS) $(LIBS)

clean:
	rm -f $(ARTIFACT)

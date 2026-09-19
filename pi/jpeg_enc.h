#ifndef GAZECOMP_JPEG_ENC_H
#define GAZECOMP_JPEG_ENC_H

#include <stdint.h>
#include <stddef.h>

/* Baseline JPEG encoder (YCbCr 4:2:0). Returns bytes written, 0 on failure. */
size_t jpeg_encode_rgb(const uint8_t *rgb, int width, int height, int quality,
                       uint8_t *out, size_t out_cap);

#endif

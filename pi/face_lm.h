#ifndef GAZECOMP_FACE_LM_H
#define GAZECOMP_FACE_LM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EYE_LID_N 16
#define EYE_IRIS_N 4

typedef struct {
    float lid_xy[EYE_LID_N * 2];
    float iris_ring[EYE_IRIS_N * 2];
    float iris_x, iris_y;
    float cx, cy;
    float dx, dy;
    float ndx, ndy;
    float width;
} EyeMeas;

typedef struct {
    const uint8_t *rgb;
    int rgb_w, rgb_h;
    const uint8_t *y;
    const uint8_t *uv;
    int native_w, native_h, stride;
} FrameSrc;

/* Official MediaPipe Face Landmarker (BlazeFace + Face Mesh v2). Overlay uses
 * eyelid / iris indices only. Returns 2 on success, 0 if no face/eyes. */
int face_eyes_from_frame(const FrameSrc *src, EyeMeas *left, EyeMeas *right, float *score_out);

#ifdef __cplusplus
}
#endif

#endif

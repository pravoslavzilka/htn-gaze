#ifndef GAZECOMP_APP_STATE_H
#define GAZECOMP_APP_STATE_H

#include "face_lm.h"

#include <pthread.h>
#include <stdint.h>

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int seq;
    uint32_t frame_id;
    uint32_t camera_fps_x100;
    uint32_t infer_fps_x100;
    uint16_t width;
    uint16_t height;
    uint32_t rgb_bytes;
    uint8_t *rgb;
    uint8_t *nv12;
    uint32_t nv12_bytes;
    uint32_t nv12_cap;
    uint16_t native_w;
    uint16_t native_h;
    uint32_t y_stride;
    uint8_t *jpeg;
    uint32_t jpeg_len;
    uint32_t jpeg_cap;
    int n_eyes;
    float lm_score;
    EyeMeas left;
    EyeMeas right;
    /* full-resolution crops around each eye (upright, same orientation as the RGB stream) */
    uint8_t *crop[2];             /* RGB24, crop_w * crop_h each */
    int crop_w, crop_h;
    int crop_x0[2], crop_y0[2];   /* crop origin in the virtual full-res, rotated frame */
    int crop_vw, crop_vh;         /* size of that virtual frame */
    int crop_valid;
    uint32_t crop_frame_id;
    EyeMeas crop_meas[2];         /* landmarks the crops were centred on */
} AppState;

extern AppState g_app;
extern volatile int g_running;

int http_server_run(const char *bind_ip, int port);
int camera_focus_ctl(int mode, int step, char *out, size_t cap);
void *process_loop(void *arg);
void *infer_loop(void *arg);

#endif

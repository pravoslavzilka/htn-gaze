#include "app_state.h"
#include "face_lm.h"
#include "jpeg_enc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <math.h>

static void put_px(uint8_t *rgb, int w, int h, int x, int y, uint8_t r, uint8_t g, uint8_t b)
{
    if (x < 0 || y < 0 || x >= w || y >= h) {
        return;
    }
    uint8_t *p = rgb + ((size_t)y * (size_t)w + (size_t)x) * 3u;
    p[0] = r;
    p[1] = g;
    p[2] = b;
}

/* Ring of the given radius and thickness centred on (cx, cy). */
static void draw_ring(uint8_t *rgb, int w, int h, float cx, float cy, float radius, float thick,
                      uint8_t r, uint8_t g, uint8_t b)
{
    int x0 = (int)(cx - radius - thick) - 1, x1 = (int)(cx + radius + thick) + 1;
    int y0 = (int)(cy - radius - thick) - 1, y1 = (int)(cy + radius + thick) + 1;
    for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
            float d = sqrtf(((float)x - cx) * ((float)x - cx) + ((float)y - cy) * ((float)y - cy));
            if (fabsf(d - radius) <= thick * 0.5f) {
                put_px(rgb, w, h, x, y, r, g, b);
            }
        }
    }
}

static void draw_dot(uint8_t *rgb, int w, int h, float cx, float cy, int rad, uint8_t r, uint8_t g, uint8_t b)
{
    for (int dy = -rad; dy <= rad; dy++) {
        for (int dx = -rad; dx <= rad; dx++) {
            if (dx * dx + dy * dy <= rad * rad) {
                put_px(rgb, w, h, (int)cx + dx, (int)cy + dy, r, g, b);
            }
        }
    }
}

/* Marks a detected pupil: ring sized from the 4 iris-ring landmarks, crosshair and centre dot. */
static void draw_pupil(uint8_t *rgb, int w, int h, const EyeMeas *e)
{
    float cx = e->iris_x * (float)w;
    float cy = e->iris_y * (float)h;
    float rsum = 0.f;
    float radius;
    int arm;

    for (int i = 0; i < EYE_IRIS_N; i++) {
        float rx = e->iris_ring[i * 2] * (float)w - cx;
        float ry = e->iris_ring[i * 2 + 1] * (float)h - cy;
        rsum += sqrtf(rx * rx + ry * ry);
    }
    radius = rsum / (float)EYE_IRIS_N;
    if (radius < 5.f) {
        radius = 5.f;
    }
    arm = (int)(radius * 1.8f) + 3;

    draw_ring(rgb, w, h, cx, cy, radius + 3.f, 2.5f, 0, 255, 0);
    for (int i = -arm; i <= arm; i++) {
        if (i > -(int)radius - 1 && i < (int)radius + 1) {
            continue;
        }
        put_px(rgb, w, h, (int)cx + i, (int)cy, 0, 255, 0);
        put_px(rgb, w, h, (int)cx, (int)cy + i, 0, 255, 0);
    }
    draw_dot(rgb, w, h, cx, cy, 2, 255, 40, 40);
}

static int wait_seq(int *last_seq, uint16_t *w, uint16_t *h, uint32_t *nbytes, uint32_t *fps_cam,
                    uint32_t *frame_id, uint8_t **rgb, uint32_t *rgb_cap)
{
    struct timespec ts;
    pthread_mutex_lock(&g_app.mu);
    while (g_running && g_app.seq == *last_seq) {
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 40000000L;
        if (ts.tv_nsec >= 1000000000L) {
            ts.tv_sec += 1;
            ts.tv_nsec -= 1000000000L;
        }
        pthread_cond_timedwait(&g_app.cv, &g_app.mu, &ts);
    }
    if (!g_running) {
        pthread_mutex_unlock(&g_app.mu);
        return 0;
    }
    *last_seq = g_app.seq;
    *w = g_app.width;
    *h = g_app.height;
    *nbytes = g_app.rgb_bytes;
    *fps_cam = g_app.camera_fps_x100;
    *frame_id = g_app.frame_id;
    if (*rgb == NULL || *rgb_cap != *nbytes) {
        free(*rgb);
        *rgb = (uint8_t *)malloc(*nbytes);
        *rgb_cap = *nbytes;
    }
    if (*rgb == NULL) {
        pthread_mutex_unlock(&g_app.mu);
        return -1;
    }
    memcpy(*rgb, g_app.rgb, *nbytes);
    pthread_mutex_unlock(&g_app.mu);
    return 1;
}

void *process_loop(void *arg)
{
    int last_seq = -1;
    uint8_t *rgb = NULL;
    uint8_t *jbuf = NULL;
    uint32_t rgb_cap = 0;
    uint32_t jcap = 0;

    (void)arg;
    while (g_running) {
        uint16_t w, h;
        uint32_t nbytes, fps_cam, frame_id;
        size_t jn;
        int wr = wait_seq(&last_seq, &w, &h, &nbytes, &fps_cam, &frame_id, &rgb, &rgb_cap);
        if (wr == 0) {
            break;
        }
        if (wr < 0) {
            continue;
        }
        if (jbuf == NULL || jcap < nbytes / 2 + 65536) {
            free(jbuf);
            jcap = nbytes / 2 + 65536;
            jbuf = (uint8_t *)malloc(jcap);
        }
        if (jbuf == NULL) {
            continue;
        }
        {
            EyeMeas l, r;
            int n;
            pthread_mutex_lock(&g_app.mu);
            n = g_app.n_eyes;
            l = g_app.left;
            r = g_app.right;
            pthread_mutex_unlock(&g_app.mu);
            if (n > 0) {
                draw_pupil(rgb, (int)w, (int)h, &l);
                draw_pupil(rgb, (int)w, (int)h, &r);
            }
        }
        jn = jpeg_encode_rgb(rgb, (int)w, (int)h, 55, jbuf, jcap);
        pthread_mutex_lock(&g_app.mu);
        if (g_app.jpeg == NULL || g_app.jpeg_cap < jcap) {
            free(g_app.jpeg);
            g_app.jpeg = (uint8_t *)malloc(jcap);
            g_app.jpeg_cap = jcap;
        }
        if (g_app.jpeg && jn > 0) {
            memcpy(g_app.jpeg, jbuf, jn);
            g_app.jpeg_len = (uint32_t)jn;
        }
        g_app.camera_fps_x100 = fps_cam;
        g_app.frame_id = frame_id;
        pthread_mutex_unlock(&g_app.mu);
    }
    free(rgb);
    free(jbuf);
    return NULL;
}

void *infer_loop(void *arg)
{
    int last_seq = -1;
    uint8_t *rgb = NULL;
    uint32_t rgb_cap = 0;
    uint64_t last_us = 0;
    double fps_ema = 0.0;
    int fps_n = 0;

    (void)arg;
    while (g_running) {
        uint16_t w, h;
        uint32_t nbytes, fps_cam, frame_id;
        int n_eyes;
        float score = 0.f;
        EyeMeas left, right;
        FrameSrc src;
        struct timespec now;
        int wr;

        memset(&left, 0, sizeof(left));
        memset(&right, 0, sizeof(right));
        wr = wait_seq(&last_seq, &w, &h, &nbytes, &fps_cam, &frame_id, &rgb, &rgb_cap);
        if (wr == 0) {
            break;
        }
        if (wr < 0) {
            continue;
        }

        memset(&src, 0, sizeof(src));
        src.rgb = rgb;
        src.rgb_w = (int)w;
        src.rgb_h = (int)h;
        n_eyes = face_eyes_from_frame(&src, &left, &right, &score);
        clock_gettime(CLOCK_MONOTONIC, &now);
        {
            uint64_t t = (uint64_t)now.tv_sec * 1000000ull + (uint64_t)now.tv_nsec / 1000ull;
            if (last_us > 0 && t > last_us) {
                double inst = 1000000.0 / (double)(t - last_us);
                fps_ema = (fps_n == 0) ? inst : (0.25 * inst + 0.75 * fps_ema);
                fps_n += 1;
            }
            last_us = t;
        }
        pthread_mutex_lock(&g_app.mu);
        g_app.n_eyes = n_eyes;
        g_app.lm_score = score;
        if (n_eyes > 0) {
            g_app.left = left;
            g_app.right = right;
        }
        g_app.infer_fps_x100 = (uint32_t)(fps_ema * 100.0 + 0.5);
        pthread_mutex_unlock(&g_app.mu);
    }
    free(rgb);
    return NULL;
}

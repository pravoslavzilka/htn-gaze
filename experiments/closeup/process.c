#include "app_state.h"
#include "face_lm.h"
#include "jpeg_enc.h"
#include "pupil.h"

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

static void draw_line(uint8_t *rgb, int w, int h, int x0, int y0, int x1, int y1,
                     uint8_t r, uint8_t g, uint8_t b, int thick)
{
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    int t = thick < 1 ? 1 : thick;
    for (;;) {
        int ox, oy;
        for (oy = -t / 2; oy <= t / 2; oy++) {
            for (ox = -t / 2; ox <= t / 2; ox++) {
                put_px(rgb, w, h, x0 + ox, y0 + oy, r, g, b);
            }
        }
        if (x0 == x1 && y0 == y1) {
            break;
        }
        {
            int e2 = 2 * err;
            if (e2 >= dy) {
                err += dy;
                x0 += sx;
            }
            if (e2 <= dx) {
                err += dx;
                y0 += sy;
            }
        }
    }
}

/* MediaPipe eyelid contour (16 points) plus iris ring. Yellow outline = mesh
 * thought this was an eye; if it sits on a brow, glasses rim, or nothing, that is a false lock. */
static void draw_eye_outline(uint8_t *rgb, int w, int h, const EyeMeas *e)
{
    int i, x[EYE_LID_N], y[EYE_LID_N];
    int rx[EYE_IRIS_N], ry[EYE_IRIS_N];
    for (i = 0; i < EYE_LID_N; i++) {
        x[i] = (int)(e->lid_xy[i * 2] * (float)w + 0.5f);
        y[i] = (int)(e->lid_xy[i * 2 + 1] * (float)h + 0.5f);
    }
    for (i = 0; i < EYE_LID_N; i++) {
        int j = (i + 1) % EYE_LID_N;
        draw_line(rgb, w, h, x[i], y[i], x[j], y[j], 255, 220, 40, 2);
    }
    for (i = 0; i < EYE_LID_N; i++) {
        draw_dot(rgb, w, h, (float)x[i], (float)y[i], 2, 255, 255, 255);
    }
    for (i = 0; i < EYE_IRIS_N; i++) {
        rx[i] = (int)(e->iris_ring[i * 2] * (float)w + 0.5f);
        ry[i] = (int)(e->iris_ring[i * 2 + 1] * (float)h + 0.5f);
    }
    for (i = 0; i < EYE_IRIS_N; i++) {
        int j = (i + 1) % EYE_IRIS_N;
        draw_line(rgb, w, h, rx[i], ry[i], rx[j], ry[j], 0, 180, 220, 2);
    }
}

/* Marks a detected pupil: ring sized from the 4 iris-ring landmarks, crosshair and centre dot.
 * When the full-res dark-pupil fit succeeded, that is the green mark; MediaPipe iris stays cyan. */
static void draw_pupil(uint8_t *rgb, int w, int h, const EyeMeas *e)
{
    float cx = e->iris_x * (float)w;
    float cy = e->iris_y * (float)h;
    float rsum = 0.f;
    float radius;
    int arm;

    draw_eye_outline(rgb, w, h, e);

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

    draw_ring(rgb, w, h, cx, cy, radius + 3.f, 1.5f, 0, 180, 220);
    if (e->pupil_ok) {
        float px = e->pupil_x * (float)w;
        float py = e->pupil_y * (float)h;
        float pr = e->pupil_r * (float)w;
        int parm;
        if (pr < 3.f) {
            pr = 3.f;
        }
        parm = (int)(pr * 1.8f) + 2;
        draw_ring(rgb, w, h, px, py, pr, 2.5f, 0, 255, 0);
        for (int i = -parm; i <= parm; i++) {
            if (i > -(int)pr - 1 && i < (int)pr + 1) {
                continue;
            }
            put_px(rgb, w, h, (int)px + i, (int)py, 0, 255, 0);
            put_px(rgb, w, h, (int)px, (int)py + i, 0, 255, 0);
        }
        draw_dot(rgb, w, h, px, py, 2, 255, 40, 40);
        return;
    }
    for (int i = -arm; i <= arm; i++) {
        if (i > -(int)radius - 1 && i < (int)radius + 1) {
            continue;
        }
        put_px(rgb, w, h, (int)cx + i, (int)cy, 0, 180, 220);
        put_px(rgb, w, h, (int)cx, (int)cy + i, 0, 180, 220);
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

static void publish_annotated_jpeg(const uint8_t *rgb, int w, int h, uint32_t fps_cam,
                                   uint32_t frame_id)
{
    static uint8_t *jbuf;
    static uint32_t jcap;
    size_t jn;
    uint32_t nbytes = (uint32_t)w * (uint32_t)h * 3u;
    if (jbuf == NULL || jcap < nbytes / 2 + 65536) {
        free(jbuf);
        jcap = nbytes / 2 + 65536;
        jbuf = (uint8_t *)malloc(jcap);
    }
    if (jbuf == NULL) {
        return;
    }
    jn = jpeg_encode_rgb(rgb, w, h, 55, jbuf, jcap);
    pthread_mutex_lock(&g_app.mu);
    if (g_app.jpeg == NULL || g_app.jpeg_cap < jcap) {
        free(g_app.jpeg);
        g_app.jpeg = (uint8_t *)malloc(jcap);
        g_app.jpeg_cap = jcap;
    }
    if (g_app.jpeg && jn > 0) {
        memcpy(g_app.jpeg, jbuf, jn);
        g_app.jpeg_len = (uint32_t)jn;
        g_app.jpeg_seq += 1;
    }
    g_app.camera_fps_x100 = fps_cam;
    g_app.frame_id = frame_id;
    pthread_mutex_unlock(&g_app.mu);
}

void *process_loop(void *arg)
{
    int last_seq = -1;
    uint8_t *rgb = NULL;
    uint8_t *jbuf = NULL;
    uint32_t rgb_cap = 0;
    uint32_t jcap = 0;
    int from_infer;

    (void)arg;
    pthread_mutex_lock(&g_app.mu);
    from_infer = g_app.jpeg_from_infer;
    pthread_mutex_unlock(&g_app.mu);
    if (from_infer) {
        /* Eye camera: infer_loop encodes the same RGB it ran the mesh on. */
        return NULL;
    }
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
            g_app.jpeg_seq += 1;
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

        /* Dark pupil on the full-res crops (camera thread only copies RGB). */
        if (n_eyes > 0) {
            static uint8_t *copy[2];
            static size_t copy_cap;
            EyeMeas meas[2];
            int x0[2], y0[2], cw, ch, vw, vh, valid, i, have[2] = {0, 0};
            size_t bytes;
            pthread_mutex_lock(&g_app.mu);
            valid = g_app.crop_valid;
            cw = g_app.crop_w;
            ch = g_app.crop_h;
            vw = g_app.crop_vw;
            vh = g_app.crop_vh;
            bytes = (size_t)cw * (size_t)ch * 3u;
            if (bytes > copy_cap) {
                for (i = 0; i < 2; i++) {
                    free(copy[i]);
                    copy[i] = (uint8_t *)malloc(bytes);
                }
                copy_cap = (copy[0] && copy[1]) ? bytes : 0;
            }
            for (i = 0; i < 2; i++) {
                x0[i] = g_app.crop_x0[i];
                y0[i] = g_app.crop_y0[i];
                meas[i] = (i == 0) ? g_app.left : g_app.right;
                if (valid && g_app.crop[i] && copy[i] && bytes > 0 && bytes <= copy_cap) {
                    memcpy(copy[i], g_app.crop[i], bytes);
                    have[i] = 1;
                }
            }
            pthread_mutex_unlock(&g_app.mu);
            for (i = 0; i < 2; i++) {
                PupilMeas p;
                static int use_pupil = -1;   /* the small dark-pupil fit is ignored unless GAZE_PUPIL is set */
                if (use_pupil < 0) {
                    use_pupil = getenv("GAZE_PUPIL") != NULL;
                }
                memset(&p, 0, sizeof(p));
                if (have[i] && use_pupil &&
                    pupil_from_crop(copy[i], cw, ch, &meas[i], x0[i], y0[i], vw, vh, &p) == 0) {
                    pupil_apply(&meas[i], &p);
                } else {
                    meas[i].pupil_ok = 0;
                    meas[i].pupil_x = meas[i].pupil_y = meas[i].pupil_r = 0.f;
                    meas[i].pupil_score = 0.f;
                    meas[i].pdx = meas[i].pdy = meas[i].npdx = meas[i].npdy = 0.f;
                }
            }
            pthread_mutex_lock(&g_app.mu);
            g_app.left = meas[0];
            g_app.right = meas[1];
            g_app.crop_meas[0] = meas[0];
            g_app.crop_meas[1] = meas[1];
            pthread_mutex_unlock(&g_app.mu);
            left = meas[0];
            right = meas[1];
        }
        /* Draw on this RGB copy (the one Face Mesh just ran on), then JPEG it.
         * process_loop must not overlay a newer camera frame with these landmarks. */
        if (n_eyes > 0) {
            draw_pupil(rgb, (int)w, (int)h, &left);
            draw_pupil(rgb, (int)w, (int)h, &right);
        }
        publish_annotated_jpeg(rgb, (int)w, (int)h, fps_cam, frame_id);
    }
    free(rgb);
    return NULL;
}

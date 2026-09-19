/*
 * GazeComp QNX camera streamer
 *
 * Captures frames from the Raspberry Pi Camera Module 3 via libcamapi,
 * downscales them to RGB24, and serves a length-prefixed TCP stream to
 * the host GUI (default listen 0.0.0.0:9000).
 *
 * Protocol (little-endian, packed 36-byte header + payload):
 *   magic u32 'GZCM' | version u16 | flags u16 | width u16 | height u16
 *   format u16 (1=RGB24) | reserved u16 | frame_id u32 | timestamp_us u64
 *   capture_fps_x100 u32 | payload_bytes u32
 *   payload: width*height*3 RGB bytes
 */

#include "app_state.h"

#include <arpa/inet.h>
#include <camera/camera_api.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define GZCM_MAGIC 0x4D435A47u /* 'GZCM' little-endian */
#define GZCM_VERSION 1
#define GZCM_FMT_RGB24 1

#define DEFAULT_PORT 8080
#define DEFAULT_OUT_W 960
#define DEFAULT_OUT_H 540

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;
    uint16_t version;
    uint16_t flags;
    uint16_t width;
    uint16_t height;
    uint16_t format;
    uint16_t reserved;
    uint32_t frame_id;
    uint64_t timestamp_us;
    uint32_t capture_fps_x100;
    uint32_t payload_bytes;
} gzcm_header_t;
#pragma pack(pop)

volatile int g_running = 1;
AppState g_app;
static uint32_t g_out_w = DEFAULT_OUT_W;
static uint32_t g_out_h = DEFAULT_OUT_H;
static camera_unit_t g_unit = CAMERA_UNIT_4;

static void on_signal(int sig)
{
    (void)sig;
    g_running = 0;
}

static uint64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000ull;
}

static uint8_t clamp_u8(int v)
{
    if (v < 0) {
        return 0;
    }
    if (v > 255) {
        return 255;
    }
    return (uint8_t)v;
}

/* Clockwise rotation (0/90/180/270) applied to the displayed image. For 90/270 the
 * destination is portrait: dst_w maps onto src_h and dst_h maps onto src_w. */
static int g_rot = 270;

static void convert_nv12_nn(const uint8_t *y_plane, uint32_t y_stride,
                            const uint8_t *uv_plane, uint32_t uv_stride,
                            uint32_t src_w, uint32_t src_h,
                            uint8_t *dst, uint32_t dst_w, uint32_t dst_h)
{
    for (uint32_t dy = 0; dy < dst_h; dy++) {
        uint8_t *drow = dst + (size_t)dy * dst_w * 3u;
        for (uint32_t dx = 0; dx < dst_w; dx++) {
            uint32_t sx, sy;
            switch (g_rot) {
            case 90:
                sx = dy * src_w / dst_h;
                sy = src_h - 1u - dx * src_h / dst_w;
                break;
            case 180:
                sx = src_w - 1u - dx * src_w / dst_w;
                sy = src_h - 1u - dy * src_h / dst_h;
                break;
            case 270:
                sx = src_w - 1u - dy * src_w / dst_h;
                sy = dx * src_h / dst_w;
                break;
            default:
                sx = dx * src_w / dst_w;
                sy = dy * src_h / dst_h;
                break;
            }
            const uint8_t *yrow = y_plane + (size_t)sy * y_stride;
            const uint8_t *uvrow = uv_plane + (size_t)(sy / 2u) * uv_stride;
            int y = yrow[sx];
            uint32_t uvx = sx & ~1u;
            int u = (int)uvrow[uvx] - 128;
            int v = (int)uvrow[uvx + 1u] - 128;
            int r = y + ((1436 * v) >> 10);
            int g = y - ((352 * u + 731 * v) >> 10);
            int b = y + ((1815 * u) >> 10);
            drow[dx * 3u + 0u] = clamp_u8(r);
            drow[dx * 3u + 1u] = clamp_u8(g);
            drow[dx * 3u + 2u] = clamp_u8(b);
        }
    }
}

static void convert_bayer14_nn(const uint8_t *src, uint32_t src_w, uint32_t src_h,
                               uint32_t stride, uint8_t *dst, uint32_t dst_w, uint32_t dst_h)
{
    for (uint32_t dy = 0; dy < dst_h; dy++) {
        uint32_t sy = (dy * src_h / dst_h) & ~1u;
        if (sy + 1u >= src_h) {
            sy = src_h - 2u;
        }
        const uint8_t *r0 = src + (size_t)sy * stride;
        const uint8_t *r1 = src + (size_t)(sy + 1u) * stride;
        uint8_t *drow = dst + (size_t)dy * dst_w * 3u;
        for (uint32_t dx = 0; dx < dst_w; dx++) {
            uint32_t sx = (dx * src_w / dst_w) & ~1u;
            if (sx + 1u >= src_w) {
                sx = src_w - 2u;
            }
            uint16_t R = (uint16_t)r0[sx * 2u] | ((uint16_t)r0[sx * 2u + 1u] << 8);
            uint16_t G0 = (uint16_t)r0[(sx + 1u) * 2u] | ((uint16_t)r0[(sx + 1u) * 2u + 1u] << 8);
            uint16_t G1 = (uint16_t)r1[sx * 2u] | ((uint16_t)r1[sx * 2u + 1u] << 8);
            uint16_t B = (uint16_t)r1[(sx + 1u) * 2u] | ((uint16_t)r1[(sx + 1u) * 2u + 1u] << 8);
            drow[dx * 3u + 0u] = (uint8_t)(R >> 6);
            drow[dx * 3u + 1u] = (uint8_t)(((unsigned)G0 + (unsigned)G1) >> 7);
            drow[dx * 3u + 2u] = (uint8_t)(B >> 6);
        }
    }
}

static int parse_geom(const uint8_t *raw, uint32_t *height, uint32_t *width, uint32_t *stride)
{
    const uint32_t *u = (const uint32_t *)raw;
    for (int i = 0; i < 7; i++) {
        uint32_t h = u[i];
        uint32_t w = u[i + 1];
        uint32_t s = u[i + 2];
        if (h >= 64u && h <= 4096u && w >= 64u && w <= 4096u && s >= w && s <= w * 4u) {
            *height = h;
            *width = w;
            *stride = s;
            return 1;
        }
    }
    return 0;
}

static void convert_bgrx_nn(const uint8_t *src, uint32_t src_w, uint32_t src_h,
                            uint32_t stride, int bpp, int is_bgr,
                            uint8_t *dst, uint32_t dst_w, uint32_t dst_h)
{
    for (uint32_t dy = 0; dy < dst_h; dy++) {
        uint32_t sy = dy * src_h / dst_h;
        const uint8_t *srow = src + (size_t)sy * stride;
        uint8_t *drow = dst + (size_t)dy * dst_w * 3u;
        for (uint32_t dx = 0; dx < dst_w; dx++) {
            uint32_t sx = dx * src_w / dst_w;
            const uint8_t *p = srow + (size_t)sx * (size_t)bpp;
            if (is_bgr) {
                drow[dx * 3u + 0u] = p[2];
                drow[dx * 3u + 1u] = p[1];
                drow[dx * 3u + 2u] = p[0];
            } else {
                drow[dx * 3u + 0u] = p[0];
                drow[dx * 3u + 1u] = p[1];
                drow[dx * 3u + 2u] = p[2];
            }
        }
    }
}

static void publish_rgb(const uint8_t *rgb, uint32_t nbytes, uint32_t fps_x100,
                        const uint8_t *nv12, uint32_t native_w, uint32_t native_h, uint32_t stride)
{
    uint32_t nv12_bytes = 0;
    pthread_mutex_lock(&g_app.mu);
    memcpy(g_app.rgb, rgb, nbytes);
    g_app.rgb_bytes = nbytes;
    g_app.width = (uint16_t)g_out_w;
    g_app.height = (uint16_t)g_out_h;
    if (nv12 && native_w > 0 && native_h > 0 && stride > 0) {
        nv12_bytes = stride * native_h + stride * ((native_h + 1u) / 2u);
        if (g_app.nv12 == NULL || g_app.nv12_cap < nv12_bytes) {
            free(g_app.nv12);
            g_app.nv12 = (uint8_t *)malloc(nv12_bytes);
            g_app.nv12_cap = g_app.nv12 ? nv12_bytes : 0;
        }
        if (g_app.nv12) {
            memcpy(g_app.nv12, nv12, nv12_bytes);
            g_app.nv12_bytes = nv12_bytes;
            g_app.native_w = (uint16_t)native_w;
            g_app.native_h = (uint16_t)native_h;
            g_app.y_stride = stride;
        }
    }
    g_app.camera_fps_x100 = fps_x100;
    g_app.frame_id += 1u;
    g_app.seq += 1;
    pthread_cond_broadcast(&g_app.cv);
    pthread_mutex_unlock(&g_app.mu);
}

/* ---- full-resolution eye crops -------------------------------------------------------------
 * The RGB stream is downscaled (2304x1296 -> 960x540). For fine pupil measurement we also cut a
 * fixed-size window around each eye straight out of the full-resolution NV12 frame. Coordinates are
 * in a "virtual" frame: the full-resolution frame after the same rotation as the RGB stream. */
static int g_crop_w = 512, g_crop_h = 384;

static void convert_nv12_crop(const uint8_t *yp, uint32_t ys, const uint8_t *uvp, uint32_t uvs,
                              uint32_t sw, uint32_t sh, int vx0, int vy0, int cw, int ch, uint8_t *dst)
{
    for (int j = 0; j < ch; j++) {
        uint8_t *drow = dst + (size_t)j * (size_t)cw * 3u;
        for (int i = 0; i < cw; i++) {
            uint32_t vx = (uint32_t)(vx0 + i), vy = (uint32_t)(vy0 + j), sx, sy;
            switch (g_rot) {
            case 90:
                sx = vy;
                sy = sh - 1u - vx;
                break;
            case 180:
                sx = sw - 1u - vx;
                sy = sh - 1u - vy;
                break;
            case 270:
                sx = sw - 1u - vy;
                sy = vx;
                break;
            default:
                sx = vx;
                sy = vy;
                break;
            }
            const uint8_t *yrow = yp + (size_t)sy * ys;
            const uint8_t *uvrow = uvp + (size_t)(sy / 2u) * uvs;
            int y = yrow[sx];
            uint32_t uvx = sx & ~1u;
            int u = (int)uvrow[uvx] - 128;
            int v = (int)uvrow[uvx + 1u] - 128;
            drow[i * 3 + 0] = clamp_u8(y + ((1436 * v) >> 10));
            drow[i * 3 + 1] = clamp_u8(y - ((352 * u + 731 * v) >> 10));
            drow[i * 3 + 2] = clamp_u8(y + ((1815 * u) >> 10));
        }
    }
}

static void extract_eye_crops(const uint8_t *yp, uint32_t ys, const uint8_t *uvp, uint32_t uvs,
                              uint32_t sw, uint32_t sh)
{
    static uint8_t *tmp[2] = {NULL, NULL};
    EyeMeas m[2];
    int n, x0[2], y0[2], i;
    int vw = (g_rot == 90 || g_rot == 270) ? (int)sh : (int)sw;
    int vh = (g_rot == 90 || g_rot == 270) ? (int)sw : (int)sh;
    size_t bytes = (size_t)g_crop_w * (size_t)g_crop_h * 3u;

    if (g_crop_w > vw || g_crop_h > vh) {
        return;
    }
    pthread_mutex_lock(&g_app.mu);
    n = g_app.n_eyes;
    m[0] = g_app.left;
    m[1] = g_app.right;
    pthread_mutex_unlock(&g_app.mu);
    if (n < 1) {
        return;
    }
    for (i = 0; i < 2; i++) {
        int cx = (int)(m[i].cx * (float)vw) - g_crop_w / 2;
        int cy = (int)(m[i].cy * (float)vh) - g_crop_h / 2;
        x0[i] = cx < 0 ? 0 : (cx > vw - g_crop_w ? vw - g_crop_w : cx);
        y0[i] = cy < 0 ? 0 : (cy > vh - g_crop_h ? vh - g_crop_h : cy);
        if (tmp[i] == NULL) {
            tmp[i] = (uint8_t *)malloc(bytes);
            if (tmp[i] == NULL) {
                return;
            }
        }
        convert_nv12_crop(yp, ys, uvp, uvs, sw, sh, x0[i], y0[i], g_crop_w, g_crop_h, tmp[i]);
    }
    pthread_mutex_lock(&g_app.mu);
    for (i = 0; i < 2; i++) {
        if (g_app.crop[i] == NULL || g_app.crop_w != g_crop_w || g_app.crop_h != g_crop_h) {
            free(g_app.crop[i]);
            g_app.crop[i] = (uint8_t *)malloc(bytes);
        }
        if (g_app.crop[i] == NULL) {
            pthread_mutex_unlock(&g_app.mu);
            return;
        }
        memcpy(g_app.crop[i], tmp[i], bytes);
        g_app.crop_x0[i] = x0[i];
        g_app.crop_y0[i] = y0[i];
        g_app.crop_meas[i] = m[i];
    }
    g_app.crop_w = g_crop_w;
    g_app.crop_h = g_crop_h;
    g_app.crop_vw = vw;
    g_app.crop_vh = vh;
    g_app.crop_frame_id = g_app.frame_id;
    g_app.crop_valid = 1;
    pthread_mutex_unlock(&g_app.mu);
}

static void process_camera_data(camera_handle_t handle, camera_buffer_t *buffer, void *arg)
{
    static uint8_t *scratch = NULL;
    static uint32_t scratch_bytes = 0;
    static uint64_t last_us = 0;
    static double fps_ema = 0.0;
    static int fps_n = 0;
    uint32_t src_w = 0;
    uint32_t src_h = 0;
    uint32_t nbytes;
    uint64_t t;
    double inst;
    uint32_t fps_x100;

    (void)handle;
    (void)arg;
    if (!g_running || buffer == NULL || buffer->framebuf == NULL) {
        return;
    }

    nbytes = g_out_w * g_out_h * 3u;
    if (scratch == NULL || scratch_bytes != nbytes) {
        free(scratch);
        scratch = (uint8_t *)malloc(nbytes);
        scratch_bytes = nbytes;
        if (scratch == NULL) {
            return;
        }
    }

    {
        static int logged = 0;
        if (!logged) {
            const uint32_t *dwords = (const uint32_t *)buffer->framedesc.raw;
            fprintf(stderr, "First frame: type=%d buf=%p desc=%u %u %u %u %u %u %u %u %u\n",
                    (int)buffer->frametype, (void *)buffer->framebuf,
                    dwords[0], dwords[1], dwords[2], dwords[3], dwords[4],
                    dwords[5], dwords[6], dwords[7], dwords[8]);
            logged = 1;
        }
    }

    {
        uint32_t stride = 0;
        if (!parse_geom(buffer->framedesc.raw, &src_h, &src_w, &stride) ||
            buffer->framebuf == NULL) {
            static int warned = 0;
            if (!warned) {
                fprintf(stderr, "Unsupported frametype %d (no geometry)\n",
                        (int)buffer->frametype);
                warned = 1;
            }
            return;
        }
        if (stride == src_w * 4u) {
            int is_bgr = (buffer->frametype == CAMERA_FRAMETYPE_BGR8888) ? 1 : 0;
            convert_bgrx_nn(buffer->framebuf, src_w, src_h, stride, 4, is_bgr,
                            scratch, g_out_w, g_out_h);
        } else if (stride == src_w * 3u) {
            convert_bgrx_nn(buffer->framebuf, src_w, src_h, stride, 3, 0,
                            scratch, g_out_w, g_out_h);
        } else if (stride == src_w * 2u ||
                   buffer->frametype == CAMERA_FRAMETYPE_BAYER14_RGGB_PADLO16) {
            convert_bayer14_nn(buffer->framebuf, src_w, src_h, stride,
                               scratch, g_out_w, g_out_h);
        } else {
            const uint8_t *y_plane = buffer->framebuf;
            const uint8_t *uv_plane = y_plane + (size_t)stride * src_h;
            convert_nv12_nn(y_plane, stride, uv_plane, stride,
                            src_w, src_h, scratch, g_out_w, g_out_h);
            t = now_us();
            if (last_us > 0 && t > last_us) {
                inst = 1000000.0 / (double)(t - last_us);
                fps_ema = (fps_n == 0) ? inst : (0.2 * inst + 0.8 * fps_ema);
                fps_n += 1;
            }
            last_us = t;
            fps_x100 = (uint32_t)(fps_ema * 100.0 + 0.5);
            publish_rgb(scratch, nbytes, fps_x100, NULL, 0, 0, 0); /* the full-res NV12 copy was never read */
            extract_eye_crops(y_plane, stride, uv_plane, stride, src_w, src_h);
            return;
        }
    }

    t = now_us();
    if (last_us > 0 && t > last_us) {
        inst = 1000000.0 / (double)(t - last_us);
        if (fps_n == 0) {
            fps_ema = inst;
        } else {
            fps_ema = 0.2 * inst + 0.8 * fps_ema;
        }
        fps_n += 1;
    }
    last_us = t;
    fps_x100 = (uint32_t)(fps_ema * 100.0 + 0.5);

    publish_rgb(scratch, nbytes, fps_x100, NULL, 0, 0, 0);
}

static void status_callback(camera_handle_t handle, camera_devstatus_t devstatus,
                            uint16_t extra, void *arg)
{
    (void)handle;
    (void)extra;
    (void)arg;
    if (devstatus == CAMERA_STATUS_PHYSICAL_REMOVAL) {
        fprintf(stderr, "Camera physically removed\n");
    }
}

static int write_full(int fd, const void *buf, size_t n)
{
    const uint8_t *p = (const uint8_t *)buf;
    while (n > 0) {
        ssize_t k = send(fd, p, n, 0);
        if (k < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        if (k == 0) {
            return -1;
        }
        p += (size_t)k;
        n -= (size_t)k;
    }
    return 0;
}

static int serve_client(int client)
{
    int last_seq = -1;
    int flag = 1;
    uint8_t *copy;
    uint32_t copy_bytes;

    setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
    copy_bytes = g_out_w * g_out_h * 3u;
    copy = (uint8_t *)malloc(copy_bytes);
    if (copy == NULL) {
        return -1;
    }

    printf("Client connected — streaming %ux%u RGB24\n", g_out_w, g_out_h);
    fflush(stdout);

    while (g_running) {
        gzcm_header_t hdr;
        uint32_t frame_id;
        uint32_t fps_x100;
        uint16_t width;
        uint16_t height;
        uint32_t nbytes;

        pthread_mutex_lock(&g_app.mu);
        while (g_running && g_app.seq == last_seq) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += 100000000L;
            if (ts.tv_nsec >= 1000000000L) {
                ts.tv_sec += 1;
                ts.tv_nsec -= 1000000000L;
            }
            pthread_cond_timedwait(&g_app.cv, &g_app.mu, &ts);
        }
        if (!g_running) {
            pthread_mutex_unlock(&g_app.mu);
            break;
        }
        last_seq = g_app.seq;
        frame_id = g_app.frame_id;
        fps_x100 = g_app.camera_fps_x100;
        width = g_app.width;
        height = g_app.height;
        nbytes = g_app.rgb_bytes;
        memcpy(copy, g_app.rgb, nbytes);
        pthread_mutex_unlock(&g_app.mu);

        memset(&hdr, 0, sizeof(hdr));
        hdr.magic = GZCM_MAGIC;
        hdr.version = GZCM_VERSION;
        hdr.width = width;
        hdr.height = height;
        hdr.format = GZCM_FMT_RGB24;
        hdr.frame_id = frame_id;
        hdr.timestamp_us = now_us();
        hdr.capture_fps_x100 = fps_x100;
        hdr.payload_bytes = nbytes;

        if (write_full(client, &hdr, sizeof(hdr)) < 0 ||
            write_full(client, copy, nbytes) < 0) {
            fprintf(stderr, "Client disconnected\n");
            free(copy);
            return -1;
        }
    }

    free(copy);
    return 0;
}

static int run_listen(const char *bind_ip, int port)
{
    int sock;
    int opt = 1;
    struct sockaddr_in addr;

    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("socket");
        return -1;
    }
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (bind_ip == NULL || strcmp(bind_ip, "0.0.0.0") == 0) {
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (inet_pton(AF_INET, bind_ip, &addr.sin_addr) != 1) {
        fprintf(stderr, "Invalid bind address %s\n", bind_ip);
        close(sock);
        return -1;
    }

    if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(sock);
        return -1;
    }
    if (listen(sock, 1) < 0) {
        perror("listen");
        close(sock);
        return -1;
    }

    printf("Listening on %s:%d\n", bind_ip ? bind_ip : "0.0.0.0", port);
    fflush(stdout);

    while (g_running) {
        struct sockaddr_in peer;
        socklen_t plen = sizeof(peer);
        int client = accept(sock, (struct sockaddr *)&peer, &plen);
        if (client < 0) {
            if (errno == EINTR) {
                continue;
            }
            perror("accept");
            break;
        }
        serve_client(client);
        close(client);
    }

    close(sock);
    return 0;
}

/* Focus control. The on-target camera_api.h is minimal, so declare the libcamapi calls here.
 * Enum arguments are passed as int (same ABI on aarch64). */
extern "C" {
camera_error_t camera_get_focus_modes(camera_handle_t handle, uint32_t numasked, uint32_t *numsupported,
                                      int *modes);
camera_error_t camera_set_focus_mode(camera_handle_t handle, int mode);
camera_error_t camera_get_focus_mode(camera_handle_t handle, int *mode);
camera_error_t camera_set_manual_focus_step(camera_handle_t handle, int32_t step);
camera_error_t camera_get_manual_focus_step(camera_handle_t handle, int32_t *a, int32_t *b);
}

static camera_handle_t g_cam = CAMERA_HANDLE_INVALID;
static int g_cam_ok = 0;
static int g_focus_mode = -1;
static int g_focus_step = -1;
static int g_vf_w = 0, g_vf_h = 0, g_vf_fps = 0; /* requested viewfinder size / rate; 0 = camera default */

/* mode/step < 0 means "leave unchanged". Writes a JSON status object into out. */
int camera_focus_ctl(int mode, int step, char *out, size_t cap)
{
    int e_mode = 0, e_step = 0, cur = -1, n, i, len;
    uint32_t nsup = 0;
    int modes[16];
    int32_t a = -1, b = -1;

    if (!g_cam_ok) { /* the real handle value can be 0, which the header calls INVALID */
        return snprintf(out, cap, "{\"error\":\"no camera\"}");
    }
    memset(modes, 0, sizeof(modes));
    if (mode >= 0) {
        e_mode = camera_set_focus_mode(g_cam, mode);
    }
    if (step >= 0) {
        e_step = camera_set_manual_focus_step(g_cam, step);
    }
    int e_modes = camera_get_focus_modes(g_cam, 16, &nsup, modes);
    int e_cur = camera_get_focus_mode(g_cam, &cur);
    int e_get = camera_get_manual_focus_step(g_cam, &a, &b);

    n = (int)(nsup > 16 ? 16 : nsup);
    len = snprintf(out, cap, "{\"set_mode_err\":%d,\"set_step_err\":%d,\"modes_err\":%d,\"mode_err\":%d,"
                             "\"step_err\":%d,\"mode\":%d,\"step_a\":%d,\"step_b\":%d,\"supported\":[",
                   e_mode, e_step, e_modes, e_cur, e_get, cur, (int)a, (int)b);
    for (i = 0; i < n && len < (int)cap - 16; i++) {
        len += snprintf(out + len, cap - (size_t)len, "%s%d", i ? "," : "", modes[i]);
    }
    len += snprintf(out + len, cap - (size_t)len, "]}");
    return len;
}

static int start_camera(camera_handle_t *handle)
{
    int err;

    err = camera_open(g_unit, CAMERA_MODE_RO | CAMERA_MODE_PWRITE, handle);
    if (err != CAMERA_EOK) {
        fprintf(stderr, "camera_open failed: %d\n", err);
        return err;
    }

    err = camera_set_vf_property(*handle, CAMERA_IMGPROP_CREATEWINDOW, 0);
    fprintf(stderr, "set CREATEWINDOW=0 -> %d\n", err);
    err = camera_set_vf_property(*handle, CAMERA_IMGPROP_FORMAT, CAMERA_FRAMETYPE_NV12);
    fprintf(stderr, "set FORMAT=NV12 -> %d\n", err);

    err = camera_start_viewfinder(*handle, &process_camera_data, &status_callback, NULL);
    if (err != CAMERA_EOK) {
        fprintf(stderr, "camera_start_viewfinder failed: %d\n", err);
        camera_close(*handle);
        *handle = CAMERA_HANDLE_INVALID;
        return err;
    }

    printf("Camera unit %d viewfinder started\n", (int)g_unit);
    fflush(stdout);
    g_cam = *handle;
    g_cam_ok = 1;
    if (g_focus_mode >= 0 || g_focus_step >= 0) {
        char js[512];
        camera_focus_ctl(g_focus_mode, g_focus_step, js, sizeof(js));
        fprintf(stderr, "focus: %s\n", js);
    }
    return CAMERA_EOK;
}

static void usage(const char *argv0)
{
    fprintf(stderr,
            "Usage: %s [--http PORT] [--listen PORT] [--bind IP] [--width W] [--height H] [--unit N] [--rotate 0|90|180|270]\n"
            "  HTTP JPEG + landmarks on PORT (default 8080). Optional GZCM RGB on --listen.\n",
            argv0);
}

int main(int argc, char **argv)
{
    int http_port = DEFAULT_PORT;
    int gzcm_port = 0;
    const char *bind_ip = "0.0.0.0";
    camera_handle_t handle = CAMERA_HANDLE_INVALID;
    uint32_t nbytes;
    int opt;
    pthread_t proc_th;
    pthread_t infer_th;
    int no_infer = 0; /* scene camera: stream only, skip the face-mesh model */

    static const struct option longopts[] = {
        {"http", required_argument, NULL, 'p'},
        {"listen", required_argument, NULL, 'l'},
        {"bind", required_argument, NULL, 'b'},
        {"width", required_argument, NULL, 'w'},
        {"height", required_argument, NULL, 'H'},
        {"unit", required_argument, NULL, 'u'},
        {"rotate", required_argument, NULL, 'r'},
        {"focus-mode", required_argument, NULL, 'm'},
        {"focus-step", required_argument, NULL, 's'},
        {"no-infer", no_argument, NULL, 'n'},
        {"vf-width", required_argument, NULL, 'W'},
        {"vf-height", required_argument, NULL, 'V'},
        {"vf-fps", required_argument, NULL, 'F'},
        {"crop-w", required_argument, NULL, 'C'},
        {"crop-h", required_argument, NULL, 'D'},
        {"help", no_argument, NULL, 1},
        {NULL, 0, NULL, 0},
    };

    while ((opt = getopt_long(argc, argv, "p:l:b:w:H:u:r:m:s:nW:V:F:C:D:", longopts, NULL)) != -1) {
        switch (opt) {
        case 'n':
            no_infer = 1;
            break;
        case 'W':
            g_vf_w = atoi(optarg);
            break;
        case 'V':
            g_vf_h = atoi(optarg);
            break;
        case 'F':
            g_vf_fps = atoi(optarg);
            break;
        case 'C':
            g_crop_w = atoi(optarg);
            break;
        case 'D':
            g_crop_h = atoi(optarg);
            break;
        case 'm':
            g_focus_mode = atoi(optarg);
            break;
        case 's':
            g_focus_step = atoi(optarg);
            break;
        case 'p':
            http_port = atoi(optarg);
            break;
        case 'l':
            gzcm_port = atoi(optarg);
            break;
        case 'b':
            bind_ip = optarg;
            break;
        case 'w':
            g_out_w = (uint32_t)atoi(optarg);
            break;
        case 'H':
            g_out_h = (uint32_t)atoi(optarg);
            break;
        case 'u':
            g_unit = (camera_unit_t)atoi(optarg);
            break;
        case 'r':
            g_rot = atoi(optarg);
            if (g_rot != 0 && g_rot != 90 && g_rot != 180 && g_rot != 270) {
                fprintf(stderr, "--rotate must be 0, 90, 180 or 270\n");
                return EXIT_FAILURE;
            }
            break;
        case 1:
            usage(argv[0]);
            return EXIT_SUCCESS;
        default:
            usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    /* --width/--height describe the sensor-orientation frame; 90/270 produce portrait output. */
    if (g_rot == 90 || g_rot == 270) {
        uint32_t tmp = g_out_w;
        g_out_w = g_out_h;
        g_out_h = tmp;
    }

    if (g_out_w < 16 || g_out_h < 16 || g_out_w > 4096 || g_out_h > 4096) {
        fprintf(stderr, "Invalid output size %ux%u\n", g_out_w, g_out_h);
        return EXIT_FAILURE;
    }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN);

    nbytes = g_out_w * g_out_h * 3u;
    memset(&g_app, 0, sizeof(g_app));
    pthread_mutex_init(&g_app.mu, NULL);
    pthread_cond_init(&g_app.cv, NULL);
    g_app.rgb = (uint8_t *)calloc(1, nbytes);
    if (g_app.rgb == NULL) {
        fprintf(stderr, "Out of memory\n");
        return EXIT_FAILURE;
    }
    g_app.rgb_bytes = nbytes;
    g_app.width = (uint16_t)g_out_w;
    g_app.height = (uint16_t)g_out_h;

    if (start_camera(&handle) != CAMERA_EOK) {
        free(g_app.rgb);
        return EXIT_FAILURE;
    }

    if (pthread_create(&proc_th, NULL, process_loop, NULL) != 0) {
        fprintf(stderr, "failed to start process thread\n");
        camera_stop_viewfinder(handle);
        camera_close(handle);
        free(g_app.rgb);
        return EXIT_FAILURE;
    }
    if (!no_infer && pthread_create(&infer_th, NULL, infer_loop, NULL) != 0) {
        fprintf(stderr, "failed to start infer thread\n");
        g_running = 0;
        pthread_cond_broadcast(&g_app.cv);
        pthread_join(proc_th, NULL);
        camera_stop_viewfinder(handle);
        camera_close(handle);
        free(g_app.rgb);
        return EXIT_FAILURE;
    }

    (void)gzcm_port;
    http_server_run(bind_ip, http_port);
    g_running = 0;
    pthread_cond_broadcast(&g_app.cv);
    pthread_join(proc_th, NULL);
    if (!no_infer) {
        pthread_join(infer_th, NULL);
    }

    camera_stop_viewfinder(handle);
    camera_close(handle);
    pthread_mutex_destroy(&g_app.mu);
    pthread_cond_destroy(&g_app.cv);
    free(g_app.rgb);
    free(g_app.jpeg);
    return EXIT_SUCCESS;
}

#include "app_state.h"
#include "jpeg_enc.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

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

static int write_str(int fd, const char *s)
{
    return write_full(fd, s, strlen(s));
}

static void cors_headers(char *dst, size_t cap, const char *ctype, size_t body_len, int extra)
{
    snprintf(dst, cap,
             "HTTP/1.1 200 OK\r\n"
             "Access-Control-Allow-Origin: *\r\n"
             "Access-Control-Allow-Methods: GET, OPTIONS\r\n"
             "Access-Control-Allow-Headers: *\r\n"
             "Cache-Control: no-store\r\n"
             "Content-Type: %s\r\n"
             "%s"
             "Content-Length: %zu\r\n"
             "Connection: close\r\n"
             "\r\n",
             ctype, extra ? "X-Accel-Buffering: no\r\n" : "", body_len);
}

static int write_eye(char *json, size_t cap, int pos, const char *name, const EyeMeas *e, int first)
{
    int i;
    pos += snprintf(json + pos, cap - (size_t)pos, "%s\"%s\":{\"lid\":[", first ? "" : ",", name);
    for (i = 0; i < EYE_LID_N * 2 && pos + 16 < (int)cap; i++) {
        pos += snprintf(json + pos, cap - (size_t)pos, "%s%.4f", i ? "," : "", e->lid_xy[i]);
    }
    pos += snprintf(json + pos, cap - (size_t)pos,
                    "],\"iris\":[%.4f,%.4f],\"ring\":[", e->iris_x, e->iris_y);
    for (i = 0; i < EYE_IRIS_N * 2 && pos + 16 < (int)cap; i++) {
        pos += snprintf(json + pos, cap - (size_t)pos, "%s%.4f", i ? "," : "", e->iris_ring[i]);
    }
    pos += snprintf(json + pos, cap - (size_t)pos,
                    "],\"center\":[%.4f,%.4f],\"off\":[%.4f,%.4f],\"noff\":[%.4f,%.4f]}",
                    e->cx, e->cy, e->dx, e->dy, e->ndx, e->ndy);
    return pos;
}

static int build_state_json(char *json, size_t cap, int n_eyes, float score, uint32_t cam, uint32_t inf,
                            uint32_t fid, uint16_t w, uint16_t h, uint32_t jlen, const EyeMeas *left,
                            const EyeMeas *right)
{
    int pos;
    pos = snprintf(json, cap,
                   "{\"source\":\"qnx\",\"qnx_connected\":true,\"qnx_host\":\"192.168.2.2\","
                   "\"camera_fps\":%.2f,\"infer_fps\":%.2f,\"link_fps\":0,"
                   "\"width\":%u,\"height\":%u,\"frame_id\":%u,\"jpeg_bytes\":%u,"
                   "\"n\":%d,\"score\":%.3f,\"engine\":\"mediapipe-facemesh-v2\",\"eyes\":{",
                   cam / 100.0, inf / 100.0, (unsigned)w, (unsigned)h, fid, jlen, n_eyes, score);
    if (n_eyes > 0 && left && right && pos + 64 < (int)cap) {
        pos = write_eye(json, cap, pos, "left", left, 1);
        pos = write_eye(json, cap, pos, "right", right, 0);
    }
    if (pos + 3 < (int)cap) {
        memcpy(json + pos, "}}", 3);
        pos += 2;
    }
    return pos;
}

static void snapshot_state(int *n_eyes, float *score, uint32_t *cam, uint32_t *inf, uint32_t *fid,
                           uint16_t *w, uint16_t *h, uint32_t *jlen, EyeMeas *left, EyeMeas *right,
                           uint8_t **jpeg_copy)
{
    *jpeg_copy = NULL;
    pthread_mutex_lock(&g_app.mu);
    *n_eyes = g_app.n_eyes;
    *score = g_app.lm_score;
    *cam = g_app.camera_fps_x100;
    *inf = g_app.infer_fps_x100;
    *fid = g_app.frame_id;
    *w = g_app.width;
    *h = g_app.height;
    *jlen = g_app.jpeg_len;
    if (*n_eyes > 0) {
        *left = g_app.left;
        *right = g_app.right;
    } else {
        memset(left, 0, sizeof(*left));
        memset(right, 0, sizeof(*right));
    }
    if (*jlen > 0 && g_app.jpeg != NULL) {
        *jpeg_copy = (uint8_t *)malloc(*jlen);
        if (*jpeg_copy) {
            memcpy(*jpeg_copy, g_app.jpeg, *jlen);
        }
    }
    pthread_mutex_unlock(&g_app.mu);
}

static void send_state(int fd)
{
    char json[8192];
    char hdr[512];
    int n_eyes, pos;
    uint32_t cam, inf, fid, jlen;
    uint16_t w, h;
    float score;
    EyeMeas left, right;
    uint8_t *jpeg_copy;

    snapshot_state(&n_eyes, &score, &cam, &inf, &fid, &w, &h, &jlen, &left, &right, &jpeg_copy);
    free(jpeg_copy);
    pos = build_state_json(json, sizeof(json), n_eyes, score, cam, inf, fid, w, h, jlen, &left,
                           &right);
    cors_headers(hdr, sizeof(hdr), "application/json", (size_t)pos, 0);
    write_str(fd, hdr);
    write_full(fd, json, (size_t)pos);
}

static void send_jpeg(int fd)
{
    char json[8192];
    char hdr[9216];
    int n_eyes, pos;
    uint32_t cam, inf, fid, jlen;
    uint16_t w, h;
    float score;
    EyeMeas left, right;
    uint8_t *copy = NULL;

    snapshot_state(&n_eyes, &score, &cam, &inf, &fid, &w, &h, &jlen, &left, &right, &copy);
    pos = build_state_json(json, sizeof(json), n_eyes, score, cam, inf, fid, w, h, jlen, &left,
                           &right);
    if (copy == NULL || jlen == 0) {
        const char *body = "no frame";
        cors_headers(hdr, sizeof(hdr), "text/plain", strlen(body), 0);
        write_str(fd, hdr);
        write_str(fd, body);
        return;
    }
    snprintf(hdr, sizeof(hdr),
             "HTTP/1.1 200 OK\r\n"
             "Access-Control-Allow-Origin: *\r\n"
             "Access-Control-Allow-Methods: GET, OPTIONS\r\n"
             "Access-Control-Allow-Headers: *\r\n"
             "Access-Control-Expose-Headers: X-Gazecomp-State\r\n"
             "Cache-Control: no-store\r\n"
             "Content-Type: image/jpeg\r\n"
             "X-Gazecomp-State: %s\r\n"
             "Content-Length: %u\r\n"
             "Connection: close\r\n"
             "\r\n",
             json, jlen);
    write_str(fd, hdr);
    write_full(fd, copy, jlen);
    free(copy);
}

static void send_mjpeg(int fd)
{
    int last = -1;
    const char *hdr =
        "HTTP/1.1 200 OK\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Cache-Control: no-store\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
        "Connection: close\r\n"
        "\r\n";
    write_str(fd, hdr);
    while (g_running) {
        uint8_t *copy = NULL;
        uint32_t n = 0;
        char part[160];
        pthread_mutex_lock(&g_app.mu);
        while (g_running && g_app.seq == last) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += 30000000L;
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
        last = g_app.seq;
        n = g_app.jpeg_len;
        if (n > 0 && g_app.jpeg) {
            copy = (uint8_t *)malloc(n);
            if (copy) {
                memcpy(copy, g_app.jpeg, n);
            }
        }
        pthread_mutex_unlock(&g_app.mu);
        if (copy == NULL || n == 0) {
            free(copy);
            continue;
        }
        snprintf(part, sizeof(part),
                 "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", n);
        if (write_str(fd, part) < 0 || write_full(fd, copy, n) < 0 || write_str(fd, "\r\n") < 0) {
            free(copy);
            break;
        }
        free(copy);
    }
}

/* GET /api/eyepack : JPEG mosaic [left eye | right eye] at full resolution, plus the landmarks the crops
 * were centred on (JSON in the X-Pack header) so a client can refine the pupil position. */
static void send_eyepack(int fd)
{
    int cw, ch, vw, vh, valid, x0[2], y0[2], i, k;
    uint32_t fid, cur;
    EyeMeas m[2];
    uint8_t *mosaic, *jbuf;
    size_t jcap, jn;
    char pack[4096], hdr[4400];
    int len;

    pthread_mutex_lock(&g_app.mu);
    valid = g_app.crop_valid;
    cw = g_app.crop_w;
    ch = g_app.crop_h;
    vw = g_app.crop_vw;
    vh = g_app.crop_vh;
    fid = g_app.crop_frame_id;
    cur = g_app.frame_id;
    for (i = 0; i < 2; i++) {
        x0[i] = g_app.crop_x0[i];
        y0[i] = g_app.crop_y0[i];
        m[i] = g_app.crop_meas[i];
    }
    if (!valid || cw <= 0 || ch <= 0 || g_app.crop[0] == NULL || g_app.crop[1] == NULL) {
        pthread_mutex_unlock(&g_app.mu);
        write_str(fd, "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        return;
    }
    mosaic = (uint8_t *)malloc((size_t)cw * 2u * (size_t)ch * 3u);
    if (mosaic == NULL) {
        pthread_mutex_unlock(&g_app.mu);
        return;
    }
    for (int row = 0; row < ch; row++) {
        memcpy(mosaic + (size_t)row * (size_t)cw * 6u, g_app.crop[0] + (size_t)row * (size_t)cw * 3u, (size_t)cw * 3u);
        memcpy(mosaic + (size_t)row * (size_t)cw * 6u + (size_t)cw * 3u, g_app.crop[1] + (size_t)row * (size_t)cw * 3u,
               (size_t)cw * 3u);
    }
    pthread_mutex_unlock(&g_app.mu);

    len = snprintf(pack, sizeof(pack), "{\"fid\":%u,\"cur\":%u,\"vw\":%d,\"vh\":%d,\"cw\":%d,\"ch\":%d,\"eyes\":[",
                   fid, cur, vw, vh, cw, ch);
    for (i = 0; i < 2; i++) {
        len += snprintf(pack + len, sizeof(pack) - (size_t)len,
                        "%s{\"x0\":%d,\"y0\":%d,\"c\":[%.5f,%.5f],\"w\":%.5f,\"iris\":[%.5f,%.5f],\"ring\":[",
                        i ? "," : "", x0[i], y0[i], m[i].cx, m[i].cy, m[i].width, m[i].iris_x, m[i].iris_y);
        for (k = 0; k < EYE_IRIS_N * 2; k++) {
            len += snprintf(pack + len, sizeof(pack) - (size_t)len, "%s%.5f", k ? "," : "", m[i].iris_ring[k]);
        }
        len += snprintf(pack + len, sizeof(pack) - (size_t)len, "],\"lid\":[");
        for (k = 0; k < EYE_LID_N * 2; k++) {
            len += snprintf(pack + len, sizeof(pack) - (size_t)len, "%s%.5f", k ? "," : "", m[i].lid_xy[k]);
        }
        len += snprintf(pack + len, sizeof(pack) - (size_t)len, "]}");
    }
    snprintf(pack + len, sizeof(pack) - (size_t)len, "]}");

    jcap = (size_t)cw * 2u * (size_t)ch + 65536u;
    jbuf = (uint8_t *)malloc(jcap);
    if (jbuf == NULL) {
        free(mosaic);
        return;
    }
    jn = jpeg_encode_rgb(mosaic, cw * 2, ch, 92, jbuf, jcap);
    free(mosaic);
    if (jn == 0) {
        free(jbuf);
        return;
    }
    snprintf(hdr, sizeof(hdr),
             "HTTP/1.1 200 OK\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Expose-Headers: X-Pack\r\n"
             "Cache-Control: no-store\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\nX-Pack: %s\r\n"
             "Connection: close\r\n\r\n",
             jn, pack);
    write_str(fd, hdr);
    write_full(fd, jbuf, jn);
    free(jbuf);
}

static void handle_client(int fd)
{
    char req[1024];
    ssize_t n;
    int got = 0;
    int flag = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
    while (got < (int)sizeof(req) - 1) {
        n = recv(fd, req + got, sizeof(req) - 1 - (size_t)got, 0);
        if (n <= 0) {
            return;
        }
        got += (int)n;
        req[got] = 0;
        if (strstr(req, "\r\n\r\n")) {
            break;
        }
    }
    if (strncmp(req, "OPTIONS ", 8) == 0) {
        write_str(fd, "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\n"
                      "Access-Control-Allow-Methods: GET, OPTIONS\r\n"
                      "Access-Control-Allow-Headers: *\r\nContent-Length: 0\r\n\r\n");
        return;
    }
    if (strncmp(req, "GET /api/eyepack", 16) == 0) {
        send_eyepack(fd);
        return;
    }
    if (strncmp(req, "GET /api/focus", 14) == 0) {
        /* /api/focus[?mode=N][&step=M]  -- query or change camera focus */
        char js[512], hdr[320];
        int mode = -1, step = -1;
        const char *q = strstr(req, "mode=");
        if (q) {
            mode = atoi(q + 5);
        }
        q = strstr(req, "step=");
        if (q) {
            step = atoi(q + 5);
        }
        size_t jl = (size_t)camera_focus_ctl(mode, step, js, sizeof(js));
        cors_headers(hdr, sizeof(hdr), "application/json", jl, 0);
        write_str(fd, hdr);
        write_full(fd, js, jl);
        return;
    }
    if (strstr(req, "GET /api/state") || strstr(req, "GET /state")) {
        send_state(fd);
        return;
    }
    if (strstr(req, "GET /api/frame") || strstr(req, "GET /frame.jpg") ||
        strstr(req, "GET /api/jpeg")) {
        send_jpeg(fd);
        return;
    }
    if (strstr(req, "GET /stream") || strstr(req, "GET /mjpeg")) {
        send_mjpeg(fd);
        return;
    }
    {
        const char *body = "GazeComp QNX\n/api/state /api/frame.jpg /stream.mjpg\n";
        char hdr[320];
        cors_headers(hdr, sizeof(hdr), "text/plain", strlen(body), 0);
        write_str(fd, hdr);
        write_str(fd, body);
    }
}

static void *client_thread(void *arg)
{
    int fd = (int)(intptr_t)arg;
    handle_client(fd);
    close(fd);
    return NULL;
}

int http_server_run(const char *bind_ip, int port)
{
    int sock;
    int opt = 1;
    struct sockaddr_in addr;

    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("http socket");
        return -1;
    }
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (bind_ip == NULL || strcmp(bind_ip, "0.0.0.0") == 0) {
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (inet_pton(AF_INET, bind_ip, &addr.sin_addr) != 1) {
        close(sock);
        return -1;
    }
    if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("http bind");
        close(sock);
        return -1;
    }
    if (listen(sock, 8) < 0) {
        perror("http listen");
        close(sock);
        return -1;
    }
    printf("HTTP on %s:%d  /api/state  /api/frame.jpg  /stream.mjpg\n",
           bind_ip ? bind_ip : "0.0.0.0", port);
    fflush(stdout);
    while (g_running) {
        pthread_t th;
        int client = accept(sock, NULL, NULL);
        if (client < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }
        if (pthread_create(&th, NULL, client_thread, (void *)(intptr_t)client) != 0) {
            handle_client(client);
            close(client);
        } else {
            pthread_detach(th);
        }
    }
    close(sock);
    return 0;
}

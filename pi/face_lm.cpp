#include "face_lm.h"

#include "tensorflow/lite/c/c_api.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { kDetSize = 128 };
enum { kLmSize = 256 };
enum { kNumAnchors = 896 };
enum { kNumCoords = 16 };
enum { kNumKeypoints = 6 };
enum { kMeshPts = 478 };
enum { kMeshFloats = kMeshPts * 3 };
static const float kDetScoreMin = 0.30f;
static const float kFaceFlagMin = 0.30f;

/* Official MediaPipe Face Mesh eye contours + iris centers. */
static const int kLeftLid[EYE_LID_N] = {33, 7, 163, 144, 145, 153, 154, 155,
                                        133, 173, 157, 158, 159, 160, 161, 246};
static const int kRightLid[EYE_LID_N] = {362, 382, 381, 380, 374, 373, 390, 249,
                                         263, 466, 388, 387, 386, 385, 384, 398};
enum { kLeftOuter = 33, kLeftInner = 133, kLeftIris = 468, kLeftIrisRing = 469 };
enum { kRightOuter = 362, kRightInner = 263, kRightIris = 473, kRightIrisRing = 474 };

static const int kStrides[] = {8, 16, 16, 16};

typedef struct {
    TfLiteModel *model;
    TfLiteInterpreterOptions *opts;
    TfLiteInterpreter *interp;
} TfliteNet;

static TfliteNet g_det, g_lm;
static float g_anchors[kNumAnchors][4];
static float *g_det_in, *g_lm_in;
static float *g_reg, *g_cls, *g_mesh, *g_flag;
static int g_ready;
static int g_fail_logged;

static float clampf(float v, float lo, float hi)
{
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

static float sigmoidf_fast(float x)
{
    if (x < -12.f) {
        return 0.f;
    }
    if (x > 12.f) {
        return 1.f;
    }
    return 1.f / (1.f + expf(-x));
}

static void build_anchors(void)
{
    int i, y, x, a, n = 0;
    for (i = 0; i < 4; i++) {
        int fm = (kDetSize + kStrides[i] - 1) / kStrides[i];
        for (y = 0; y < fm; y++) {
            for (x = 0; x < fm; x++) {
                float cx = ((float)x + 0.5f) / (float)fm;
                float cy = ((float)y + 0.5f) / (float)fm;
                for (a = 0; a < 2 && n < kNumAnchors; a++, n++) {
                    g_anchors[n][0] = cx;
                    g_anchors[n][1] = cy;
                    g_anchors[n][2] = 1.f;
                    g_anchors[n][3] = 1.f;
                }
            }
        }
    }
}

static int file_ok(const char *p)
{
    FILE *f = fopen(p, "rb");
    if (!f) {
        return 0;
    }
    fclose(f);
    return 1;
}

static const char *first_existing(const char *const *paths)
{
    int i;
    for (i = 0; paths[i]; i++) {
        if (file_ok(paths[i])) {
            return paths[i];
        }
    }
    return NULL;
}

static void net_free(TfliteNet *n)
{
    if (n->interp) {
        TfLiteInterpreterDelete(n->interp);
    }
    if (n->opts) {
        TfLiteInterpreterOptionsDelete(n->opts);
    }
    if (n->model) {
        TfLiteModelDelete(n->model);
    }
    memset(n, 0, sizeof(*n));
}

static int net_load(TfliteNet *n, const char *path, int threads)
{
    n->model = TfLiteModelCreateFromFile(path);
    if (!n->model) {
        fprintf(stderr, "gazecomp: TfLiteModelCreateFromFile failed: %s\n", path);
        return -1;
    }
    n->opts = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(n->opts, threads);
    n->interp = TfLiteInterpreterCreate(n->model, n->opts);
    if (!n->interp) {
        fprintf(stderr, "gazecomp: TfLiteInterpreterCreate failed: %s\n", path);
        net_free(n);
        return -1;
    }
    if (TfLiteInterpreterAllocateTensors(n->interp) != kTfLiteOk) {
        fprintf(stderr, "gazecomp: AllocateTensors failed: %s\n", path);
        net_free(n);
        return -1;
    }
    return 0;
}

static int ensure_ready(void)
{
    static const char *det_paths[] = {
        "models/face_detector.tflite",
        "/data/home/qnxuser/gazecomp/models/face_detector.tflite",
        NULL,
    };
    static const char *lm_paths[] = {
        "models/face_landmarks_detector.tflite",
        "/data/home/qnxuser/gazecomp/models/face_landmarks_detector.tflite",
        NULL,
    };
    const char *det_path, *lm_path;
    if (g_ready) {
        return 1;
    }
    if (g_fail_logged) {
        return 0;
    }
    det_path = first_existing(det_paths);
    lm_path = first_existing(lm_paths);
    if (!det_path || !lm_path) {
        fprintf(stderr, "gazecomp: MediaPipe models missing\n");
        g_fail_logged = 1;
        return 0;
    }
    build_anchors();
    if (net_load(&g_det, det_path, 2) != 0 || net_load(&g_lm, lm_path, 2) != 0) {
        net_free(&g_det);
        net_free(&g_lm);
        g_fail_logged = 1;
        return 0;
    }
    g_det_in = (float *)malloc((size_t)kDetSize * kDetSize * 3u * sizeof(float));
    g_lm_in = (float *)malloc((size_t)kLmSize * kLmSize * 3u * sizeof(float));
    g_reg = (float *)malloc((size_t)kNumAnchors * kNumCoords * sizeof(float));
    g_cls = (float *)malloc((size_t)kNumAnchors * sizeof(float));
    g_mesh = (float *)malloc((size_t)kMeshFloats * sizeof(float));
    g_flag = (float *)malloc(sizeof(float));
    if (!g_det_in || !g_lm_in || !g_reg || !g_cls || !g_mesh || !g_flag) {
        fprintf(stderr, "gazecomp: out of memory for MediaPipe tensors\n");
        g_fail_logged = 1;
        return 0;
    }
    {
        int i, nout;
        fprintf(stderr, "gazecomp: Face Mesh v2 ready det=%s lm=%s\n", det_path, lm_path);
        nout = TfLiteInterpreterGetOutputTensorCount(g_det.interp);
        for (i = 0; i < nout; i++) {
            const TfLiteTensor *t = TfLiteInterpreterGetOutputTensor(g_det.interp, i);
            fprintf(stderr, "gazecomp: det out[%d] %s bytes=%u\n", i, t ? TfLiteTensorName(t) : "?",
                    t ? (unsigned)TfLiteTensorByteSize(t) : 0);
        }
        nout = TfLiteInterpreterGetOutputTensorCount(g_lm.interp);
        for (i = 0; i < nout; i++) {
            const TfLiteTensor *t = TfLiteInterpreterGetOutputTensor(g_lm.interp, i);
            fprintf(stderr, "gazecomp: lm out[%d] %s bytes=%u\n", i, t ? TfLiteTensorName(t) : "?",
                    t ? (unsigned)TfLiteTensorByteSize(t) : 0);
        }
    }
    g_ready = 1;
    return 1;
}

static void nv12_rgb8(const uint8_t *y_plane, const uint8_t *uv_plane, int stride, int w, int h,
                      int sx, int sy, uint8_t *rgb)
{
    int y, u, v, r, g, b;
    const uint8_t *uv;
    if (sx < 0) {
        sx = 0;
    }
    if (sy < 0) {
        sy = 0;
    }
    if (sx >= w) {
        sx = w - 1;
    }
    if (sy >= h) {
        sy = h - 1;
    }
    y = y_plane[(size_t)sy * (size_t)stride + (size_t)sx];
    uv = uv_plane + (size_t)(sy / 2) * (size_t)stride + (size_t)(sx & ~1);
    u = (int)uv[0] - 128;
    v = (int)uv[1] - 128;
    r = y + ((1436 * v) >> 10);
    g = y - ((352 * u + 731 * v) >> 10);
    b = y + ((1815 * u) >> 10);
    rgb[0] = (uint8_t)(r < 0 ? 0 : r > 255 ? 255 : r);
    rgb[1] = (uint8_t)(g < 0 ? 0 : g > 255 ? 255 : g);
    rgb[2] = (uint8_t)(b < 0 ? 0 : b > 255 ? 255 : b);
}

static void src_rgb01(const FrameSrc *src, float px, float py, float *out)
{
    int x0, y0, x1, y1, c;
    float fx, fy, wx, wy;
    int sw, sh;
    const uint8_t *base = NULL;
    uint8_t a[3], b[3], cc[3], d[3];

    if (src->y && src->uv && src->native_w > 0) {
        sw = src->native_w;
        sh = src->native_h;
        if (px < 0.f || py < 0.f || px >= (float)sw || py >= (float)sh) {
            out[0] = out[1] = out[2] = 0.f;
            return;
        }
        x0 = (int)px;
        y0 = (int)py;
        x1 = x0 + 1 < sw ? x0 + 1 : sw - 1;
        y1 = y0 + 1 < sh ? y0 + 1 : sh - 1;
        fx = px - (float)x0;
        fy = py - (float)y0;
        wx = 1.f - fx;
        wy = 1.f - fy;
        nv12_rgb8(src->y, src->uv, src->stride, sw, sh, x0, y0, a);
        nv12_rgb8(src->y, src->uv, src->stride, sw, sh, x1, y0, b);
        nv12_rgb8(src->y, src->uv, src->stride, sw, sh, x0, y1, cc);
        nv12_rgb8(src->y, src->uv, src->stride, sw, sh, x1, y1, d);
        for (c = 0; c < 3; c++) {
            out[c] = (wy * (wx * a[c] + fx * b[c]) + fy * (wx * cc[c] + fx * d[c])) / 255.f;
        }
        return;
    }

    sw = src->rgb_w;
    sh = src->rgb_h;
    base = src->rgb;
    if (!base || px < 0.f || py < 0.f || px >= (float)sw || py >= (float)sh) {
        out[0] = out[1] = out[2] = 0.f;
        return;
    }
    x0 = (int)px;
    y0 = (int)py;
    x1 = x0 + 1 < sw ? x0 + 1 : sw - 1;
    y1 = y0 + 1 < sh ? y0 + 1 : sh - 1;
    fx = px - (float)x0;
    fy = py - (float)y0;
    wx = 1.f - fx;
    wy = 1.f - fy;
    {
        const uint8_t *pa = base + ((size_t)y0 * (size_t)sw + (size_t)x0) * 3u;
        const uint8_t *pb = base + ((size_t)y0 * (size_t)sw + (size_t)x1) * 3u;
        const uint8_t *pc = base + ((size_t)y1 * (size_t)sw + (size_t)x0) * 3u;
        const uint8_t *pd = base + ((size_t)y1 * (size_t)sw + (size_t)x1) * 3u;
        for (c = 0; c < 3; c++) {
            out[c] = (wy * (wx * pa[c] + fx * pb[c]) + fy * (wx * pc[c] + fx * pd[c])) / 255.f;
        }
    }
}

static void letterbox_neg1_rgb(const uint8_t *rgb, int w, int h, float *dst, float *scale,
                               float *pad_x, float *pad_y)
{
    int y, x;
    float s = fminf(kDetSize / (float)w, kDetSize / (float)h);
    *scale = s;
    *pad_x = (kDetSize - (float)w * s) * 0.5f;
    *pad_y = (kDetSize - (float)h * s) * 0.5f;
    for (y = 0; y < kDetSize; y++) {
        for (x = 0; x < kDetSize; x++) {
            float sx = ((float)x + 0.5f - *pad_x) / s;
            float sy = ((float)y + 0.5f - *pad_y) / s;
            float *p = dst + ((size_t)y * kDetSize + (size_t)x) * 3u;
            if (sx < 0.f || sy < 0.f || sx >= (float)w || sy >= (float)h) {
                p[0] = p[1] = p[2] = -1.f;
                continue;
            }
            {
                int x0 = (int)sx, y0 = (int)sy;
                int x1 = x0 + 1 < w ? x0 + 1 : w - 1;
                int y1 = y0 + 1 < h ? y0 + 1 : h - 1;
                float fx = sx - (float)x0, fy = sy - (float)y0;
                float wx = 1.f - fx, wy = 1.f - fy;
                const uint8_t *a = rgb + ((size_t)y0 * (size_t)w + (size_t)x0) * 3u;
                const uint8_t *b = rgb + ((size_t)y0 * (size_t)w + (size_t)x1) * 3u;
                const uint8_t *c = rgb + ((size_t)y1 * (size_t)w + (size_t)x0) * 3u;
                const uint8_t *d = rgb + ((size_t)y1 * (size_t)w + (size_t)x1) * 3u;
                int k;
                for (k = 0; k < 3; k++) {
                    p[k] = (wy * (wx * a[k] + fx * b[k]) + fy * (wx * c[k] + fx * d[k])) / 127.5f -
                           1.f;
                }
            }
        }
    }
}

static void sample_rotated(const FrameSrc *src, float cx, float cy, float size, float rot, int flip,
                           int out_n, float *dst)
{
    float ca = cosf(rot);
    float sa = sinf(rot);
    int y, x;
    for (y = 0; y < out_n; y++) {
        for (x = 0; x < out_n; x++) {
            float u = ((float)x + 0.5f) / (float)out_n - 0.5f;
            float v = ((float)y + 0.5f) / (float)out_n - 0.5f;
            float px, py, rgb[3];
            if (flip) {
                u = -u;
            }
            px = ca * u * size - sa * v * size + cx;
            py = sa * u * size + ca * v * size + cy;
            src_rgb01(src, px, py, rgb);
            memcpy(dst + ((size_t)y * (size_t)out_n + (size_t)x) * 3u, rgb, sizeof(rgb));
        }
    }
}

static void project_norm(float nx, float ny, float cx, float cy, float size, float rot, int flip,
                         float *ox, float *oy)
{
    float u = nx - 0.5f;
    float v = ny - 0.5f;
    float ca = cosf(rot);
    float sa = sinf(rot);
    if (flip) {
        u = -u;
    }
    *ox = ca * u * size - sa * v * size + cx;
    *oy = sa * u * size + ca * v * size + cy;
}

typedef struct {
    float score;
    float xmin, ymin, xmax, ymax;
    float kp[12];
} Det;

static int decode_dets(float scale, float pad_x, float pad_y, int img_w, int img_h, Det *out,
                       int max_out)
{
    const float x_scale = (float)kDetSize;
    int i, n = 0;
    for (i = 0; i < kNumAnchors; i++) {
        float score = sigmoidf_fast(g_cls[i]);
        float ax, ay, dx, dy, dw, dh, xc, yc, bw, bh, xmin, ymin, xmax, ymax;
        int k;
        if (score < kDetScoreMin) {
            continue;
        }
        ax = g_anchors[i][0];
        ay = g_anchors[i][1];
        dx = g_reg[i * kNumCoords + 0];
        dy = g_reg[i * kNumCoords + 1];
        dw = g_reg[i * kNumCoords + 2];
        dh = g_reg[i * kNumCoords + 3];
        xc = dx / x_scale * g_anchors[i][2] + ax;
        yc = dy / x_scale * g_anchors[i][3] + ay;
        bw = dw / x_scale * g_anchors[i][2];
        bh = dh / x_scale * g_anchors[i][3];
        xmin = ((xc - bw * 0.5f) * kDetSize - pad_x) / scale;
        ymin = ((yc - bh * 0.5f) * kDetSize - pad_y) / scale;
        xmax = ((xc + bw * 0.5f) * kDetSize - pad_x) / scale;
        ymax = ((yc + bh * 0.5f) * kDetSize - pad_y) / scale;
        if (xmax <= 0.f || ymax <= 0.f || xmin >= (float)img_w || ymin >= (float)img_h) {
            continue;
        }
        if (n >= max_out) {
            break;
        }
        out[n].score = score;
        out[n].xmin = xmin;
        out[n].ymin = ymin;
        out[n].xmax = xmax;
        out[n].ymax = ymax;
        for (k = 0; k < kNumKeypoints; k++) {
            float kx = g_reg[i * kNumCoords + 4 + k * 2] / x_scale * g_anchors[i][2] + ax;
            float ky = g_reg[i * kNumCoords + 4 + k * 2 + 1] / x_scale * g_anchors[i][3] + ay;
            out[n].kp[k * 2] = (kx * kDetSize - pad_x) / scale;
            out[n].kp[k * 2 + 1] = (ky * kDetSize - pad_y) / scale;
        }
        n++;
    }
    return n;
}

static float iou(const Det *a, const Det *b)
{
    float ix1 = fmaxf(a->xmin, b->xmin);
    float iy1 = fmaxf(a->ymin, b->ymin);
    float ix2 = fminf(a->xmax, b->xmax);
    float iy2 = fminf(a->ymax, b->ymax);
    float iw = fmaxf(0.f, ix2 - ix1);
    float ih = fmaxf(0.f, iy2 - iy1);
    float inter = iw * ih;
    float aa = fmaxf(0.f, a->xmax - a->xmin) * fmaxf(0.f, a->ymax - a->ymin);
    float ba = fmaxf(0.f, b->xmax - b->xmin) * fmaxf(0.f, b->ymax - b->ymin);
    float u = aa + ba - inter;
    return u > 1e-6f ? inter / u : 0.f;
}

/* MediaPipe FaceDetection min_suppression_threshold is IoU 0.3. Keep every
 * survivor so a slightly weaker true-face box can still feed Face Mesh. */
static int nms_ranked(Det *dets, int n, int *order, int max_keep)
{
    int i, j, nkeep = 0;
    char suppressed[96];
    if (n <= 0) {
        return 0;
    }
    if (n > 96) {
        n = 96;
    }
    memset(suppressed, 0, (size_t)n);
    for (i = 0; i < n; i++) {
        if (suppressed[i]) {
            continue;
        }
        for (j = i + 1; j < n; j++) {
            if (!suppressed[j] && iou(&dets[i], &dets[j]) > 0.3f) {
                if (dets[j].score > dets[i].score) {
                    suppressed[i] = 1;
                    break;
                }
                suppressed[j] = 1;
            }
        }
        if (!suppressed[i] && nkeep < max_keep) {
            order[nkeep++] = i;
        }
    }
    for (i = 0; i < nkeep; i++) {
        int best = i;
        for (j = i + 1; j < nkeep; j++) {
            if (dets[order[j]].score > dets[order[best]].score) {
                best = j;
            }
        }
        if (best != i) {
            int tmp = order[i];
            order[i] = order[best];
            order[best] = tmp;
        }
    }
    return nkeep;
}

static int invoke_copy(TfliteNet *n, float *in, int in_bytes, float **outs, const int *out_bytes,
                       int nout)
{
    TfLiteTensor *tin = TfLiteInterpreterGetInputTensor(n->interp, 0);
    int i;
    if (!tin) {
        return -1;
    }
    if (TfLiteTensorCopyFromBuffer(tin, in, (size_t)in_bytes) != kTfLiteOk) {
        return -1;
    }
    if (TfLiteInterpreterInvoke(n->interp) != kTfLiteOk) {
        return -1;
    }
    for (i = 0; i < nout; i++) {
        const TfLiteTensor *t = TfLiteInterpreterGetOutputTensor(n->interp, i);
        if (!t || TfLiteTensorCopyToBuffer(t, outs[i], (size_t)out_bytes[i]) != kTfLiteOk) {
            return -1;
        }
    }
    return 0;
}

static float mesh_inv(void)
{
    float mx = 0.f;
    int i;
    for (i = 0; i < 24; i++) {
        mx = fmaxf(mx, fmaxf(fabsf(g_mesh[i * 3]), fabsf(g_mesh[i * 3 + 1])));
    }
    return mx > 2.f ? 1.f / (float)kLmSize : 1.f;
}

static void mesh_xy(int idx, float inv, float cx, float cy, float size, float rot, float *x,
                    float *y)
{
    float nx = g_mesh[idx * 3] * inv;
    float ny = g_mesh[idx * 3 + 1] * inv;
    project_norm(nx, ny, cx, cy, size, rot, 0, x, y);
}

static void fill_eye(EyeMeas *e, const int *lid, int outer, int inner, int iris_i, int iris_ring0,
                     float inv, float cx, float cy, float size, float rot, float src_w, float src_h)
{
    int i;
    float ox, oy, ix, iy, irx, iry, w;
    for (i = 0; i < EYE_LID_N; i++) {
        float x, y;
        mesh_xy(lid[i], inv, cx, cy, size, rot, &x, &y);
        e->lid_xy[i * 2] = clampf(x / src_w, 0.f, 1.f);
        e->lid_xy[i * 2 + 1] = clampf(y / src_h, 0.f, 1.f);
    }
    for (i = 0; i < EYE_IRIS_N; i++) {
        float x, y;
        mesh_xy(iris_ring0 + i, inv, cx, cy, size, rot, &x, &y);
        e->iris_ring[i * 2] = clampf(x / src_w, 0.f, 1.f);
        e->iris_ring[i * 2 + 1] = clampf(y / src_h, 0.f, 1.f);
    }
    mesh_xy(outer, inv, cx, cy, size, rot, &ox, &oy);
    mesh_xy(inner, inv, cx, cy, size, rot, &ix, &iy);
    mesh_xy(iris_i, inv, cx, cy, size, rot, &irx, &iry);
    e->cx = clampf(0.5f * (ox + ix) / src_w, 0.f, 1.f);
    e->cy = clampf(0.5f * (oy + iy) / src_h, 0.f, 1.f);
    e->iris_x = clampf(irx / src_w, 0.f, 1.f);
    e->iris_y = clampf(iry / src_h, 0.f, 1.f);
    w = sqrtf((ix - ox) * (ix - ox) + (iy - oy) * (iy - oy));
    e->width = w / src_w;
    e->dx = e->iris_x - e->cx;
    e->dy = e->iris_y - e->cy;
    if (e->width > 1e-5f) {
        e->ndx = e->dx / e->width;
        e->ndy = e->dy / e->width;
    } else {
        e->ndx = e->ndy = 0.f;
    }
}

int face_eyes_from_frame(const FrameSrc *src, EyeMeas *left, EyeMeas *right, float *score_out)
{
    float scale, pad_x, pad_y, *det_outs[2], *lm_outs[2];
    int det_bytes[2], lm_bytes[2], ndet, nkeep, k, order[8];
    Det dets[96];
    float src_w, src_h, best_flag = 0.f;
    FrameSrc crop_src;
    static EyeMeas hold_l, hold_r;
    static float hold_s;
    static int hold_left;

    if (score_out) {
        *score_out = 0.f;
    }
    if (!src || !src->rgb || src->rgb_w < 16 || src->rgb_h < 16 || !left || !right) {
        return 0;
    }
    if (!ensure_ready()) {
        return 0;
    }

    letterbox_neg1_rgb(src->rgb, src->rgb_w, src->rgb_h, g_det_in, &scale, &pad_x, &pad_y);
    det_outs[0] = g_reg;
    det_outs[1] = g_cls;
    det_bytes[0] = kNumAnchors * kNumCoords * (int)sizeof(float);
    det_bytes[1] = kNumAnchors * (int)sizeof(float);
    if (invoke_copy(&g_det, g_det_in, kDetSize * kDetSize * 3 * (int)sizeof(float), det_outs,
                    det_bytes, 2) != 0) {
        return 0;
    }
    ndet = decode_dets(scale, pad_x, pad_y, src->rgb_w, src->rgb_h, dets, 96);
    {
        static int dbg;
        if (dbg < 12) {
            float mx = g_cls[0];
            int j;
            for (j = 1; j < kNumAnchors; j++) {
                if (g_cls[j] > mx) {
                    mx = g_cls[j];
                }
            }
            fprintf(stderr, "gazecomp: det max_logit=%.3f score=%.3f ndet=%d\n", mx,
                    sigmoidf_fast(mx), ndet);
            dbg++;
        }
    }
    nkeep = nms_ranked(dets, ndet, order, 8);
    if (nkeep <= 0) {
        if (hold_left > 0) {
            hold_left--;
            *left = hold_l;
            *right = hold_r;
            if (score_out) {
                *score_out = hold_s;
            }
            return 2;
        }
        return 0;
    }

    /* Same path that previously produced live 478-pt mesh: crop from the 960×540 RGB
     * preview, not native NV12 (that path dropped detections / zeroed n). */
    crop_src = *src;
    crop_src.y = NULL;
    crop_src.uv = NULL;
    crop_src.native_w = 0;
    crop_src.native_h = 0;
    src_w = (float)src->rgb_w;
    src_h = (float)src->rgb_h;
    lm_outs[0] = g_mesh;
    lm_outs[1] = g_flag;
    lm_bytes[0] = kMeshFloats * (int)sizeof(float);
    lm_bytes[1] = (int)sizeof(float);

    for (k = 0; k < nkeep; k++) {
        const Det *d = &dets[order[k]];
        float cx = 0.5f * (d->xmin + d->xmax);
        float cy = 0.5f * (d->ymin + d->ymax);
        float bw = d->xmax - d->xmin;
        float bh = d->ymax - d->ymin;
        float side = fmaxf(bw, bh) * 1.5f;
        float rot, flag, inv;
        if (side < 8.f) {
            continue;
        }
        rot = atan2f(d->kp[3] - d->kp[1], d->kp[2] - d->kp[0]);
        sample_rotated(&crop_src, cx, cy, side, rot, 0, kLmSize, g_lm_in);
        if (invoke_copy(&g_lm, g_lm_in, kLmSize * kLmSize * 3 * (int)sizeof(float), lm_outs,
                        lm_bytes, 2) != 0) {
            continue;
        }
        /* Official Face Mesh v2: TensorsToFloatsCalculator activation SIGMOID, then
         * min_face_presence_confidence (we use 0.3 so 0.2–0.4 BlazeFace locks show). */
        flag = sigmoidf_fast(g_flag[0]);
        if (flag > best_flag) {
            best_flag = flag;
        }
        {
            static int mdbg;
            float extra = 0.f;
            const TfLiteTensor *t2 = TfLiteInterpreterGetOutputTensor(g_lm.interp, 2);
            if (t2 && TfLiteTensorByteSize(t2) == sizeof(float)) {
                TfLiteTensorCopyToBuffer(t2, &extra, sizeof(extra));
            }
            if (mdbg < 16) {
                fprintf(stderr,
                        "gazecomp: mesh det=%.3f flag_raw=%.3f flag_sig=%.3f id2=%.3f id2_sig=%.3f side=%.1f\n",
                        d->score, g_flag[0], flag, extra, sigmoidf_fast(extra), side);
                mdbg++;
            }
        }
        if (flag < kFaceFlagMin) {
            continue;
        }
        inv = mesh_inv();
        fill_eye(left, kLeftLid, kLeftOuter, kLeftInner, kLeftIris, kLeftIrisRing, inv, cx, cy,
                 side, rot, src_w, src_h);
        fill_eye(right, kRightLid, kRightOuter, kRightInner, kRightIris, kRightIrisRing, inv, cx,
                 cy, side, rot, src_w, src_h);
        if (score_out) {
            *score_out = fminf(1.f, fmaxf(d->score, flag));
        }
        hold_l = *left;
        hold_r = *right;
        hold_s = score_out ? *score_out : flag;
        hold_left = 12;
        {
            static int once;
            if (!once) {
                fprintf(stderr,
                        "gazecomp: Face Mesh v2 eyes %dx%d L_noff=%.3f,%.3f R_noff=%.3f,%.3f w=%.4f,%.4f\n",
                        src->rgb_w, src->rgb_h, left->ndx, left->ndy, right->ndx, right->ndy,
                        left->width, right->width);
                once = 1;
            }
        }
        return 2;
    }

    if (score_out) {
        *score_out = best_flag;
    }
    if (hold_left > 0) {
        hold_left--;
        *left = hold_l;
        *right = hold_r;
        if (score_out) {
            *score_out = hold_s;
        }
        return 2;
    }
    return 0;
}

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
static const float kDetScoreMin = 0.30f; /* close-up + glasses often scores 0.3–0.5 */
static const float kFaceFlagMin = 0.30f;
/* Near-face camera: a real eye fills tens of pixels. Specks on a brow, nostril
 * or a person in the background are not possible at this working distance. */
static const float kMinFaceFrac = 0.20f;  /* det box vs min(frame w,h) */
static const float kMinEyeWFrac = 0.050f; /* canthus width vs frame width */

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
    if (net_load(&g_det, det_path, 2) != 0 || net_load(&g_lm, lm_path, 4) != 0) {
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

/* Map a 256×256 tensor onto the image ROI the same way MediaPipe's
 * LandmarkProjectionCalculator does: x uses rw, y uses rh, rotation from the eyes. */
static void sample_rotated(const FrameSrc *src, float cx, float cy, float rw, float rh, float rot,
                           int flip, int out_n, float *dst)
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
            px = ca * u * rw - sa * v * rh + cx;
            py = sa * u * rw + ca * v * rh + cy;
            src_rgb01(src, px, py, rgb);
            memcpy(dst + ((size_t)y * (size_t)out_n + (size_t)x) * 3u, rgb, sizeof(rgb));
        }
    }
}

static void project_norm(float nx, float ny, float cx, float cy, float rw, float rh, float rot,
                         int flip, float *ox, float *oy)
{
    float u = nx - 0.5f;
    float v = ny - 0.5f;
    float ca = cosf(rot);
    float sa = sinf(rot);
    if (flip) {
        u = -u;
    }
    *ox = ca * u * rw - sa * v * rh + cx;
    *oy = sa * u * rw + ca * v * rh + cy;
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

/* Greedy NMS: score-sort first (anchor order is not score order), IoU 0.3. */
static float lid_span_x(const EyeMeas *e)
{
    int i;
    float lo = e->lid_xy[0], hi = e->lid_xy[0];
    for (i = 1; i < EYE_LID_N; i++) {
        float x = e->lid_xy[i * 2];
        if (x < lo) {
            lo = x;
        }
        if (x > hi) {
            hi = x;
        }
    }
    return hi - lo;
}

/* Camera sits centimetres from the face. Drop meshes whose eyes are tiny,
 * wildly mismatched, or farther apart than one head. */
static int eyes_plausible(const EyeMeas *l, const EyeMeas *r, float src_w, float src_h)
{
    float wL, wR, spanL, spanR, iod, ratio, minw;
    float lx, ly, rx, ry;
    (void)src_h;
    if (!l || !r || src_w < 8.f) {
        return 0;
    }
    wL = l->width * src_w;
    wR = r->width * src_w;
    minw = kMinEyeWFrac * src_w;
    if (wL < minw || wR < minw) {
        return 0;
    }
    spanL = lid_span_x(l) * src_w;
    spanR = lid_span_x(r) * src_w;
    if (spanL < minw || spanR < minw) {
        return 0;
    }
    ratio = wL > wR ? wL / wR : wR / wL;
    if (ratio > 2.2f) {
        return 0;
    }
    lx = l->cx * src_w;
    ly = l->cy * src_h;
    rx = r->cx * src_w;
    ry = r->cy * src_h;
    iod = sqrtf((lx - rx) * (lx - rx) + (ly - ry) * (ly - ry));
    if (iod < 1.6f * fmaxf(wL, wR)) {
        return 0; /* two lids on the same speck */
    }
    if (iod > 0.70f * src_w) {
        return 0; /* two different people / background faces */
    }
    /* Clamped to the frame edge is almost always a bad projection, not an eye. */
    if (l->iris_y < 0.02f || l->iris_y > 0.98f || r->iris_y < 0.02f || r->iris_y > 0.98f) {
        return 0;
    }
    if (l->iris_x < 0.02f || l->iris_x > 0.98f || r->iris_x < 0.02f || r->iris_x > 0.98f) {
        return 0;
    }
    return 1;
}

static int nms_ranked(Det *dets, int n, int *order, int max_keep)
{
    int idx[96];
    int i, j, nkeep = 0;
    char suppressed[96];
    if (n <= 0) {
        return 0;
    }
    if (n > 96) {
        n = 96;
    }
    for (i = 0; i < n; i++) {
        idx[i] = i;
    }
    for (i = 0; i < n; i++) {
        int best = i;
        for (j = i + 1; j < n; j++) {
            if (dets[idx[j]].score > dets[idx[best]].score) {
                best = j;
            }
        }
        if (best != i) {
            int tmp = idx[i];
            idx[i] = idx[best];
            idx[best] = tmp;
        }
    }
    memset(suppressed, 0, (size_t)n);
    for (i = 0; i < n && nkeep < max_keep; i++) {
        int a = idx[i];
        if (suppressed[a]) {
            continue;
        }
        order[nkeep++] = a;
        for (j = i + 1; j < n; j++) {
            int b = idx[j];
            if (!suppressed[b] && iou(&dets[a], &dets[b]) > 0.3f) {
                suppressed[b] = 1;
            }
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

static void mesh_xy(int idx, float inv, float cx, float cy, float rw, float rh, float rot, float *x,
                    float *y)
{
    float nx = g_mesh[idx * 3] * inv;
    float ny = g_mesh[idx * 3 + 1] * inv;
    project_norm(nx, ny, cx, cy, rw, rh, rot, 0, x, y);
}

static void fill_eye(EyeMeas *e, const int *lid, int outer, int inner, int iris_i, int iris_ring0,
                     float inv, float cx, float cy, float rw, float rh, float rot, float src_w,
                     float src_h)
{
    int i;
    float ox, oy, ix, iy, irx, iry, w;
    for (i = 0; i < EYE_LID_N; i++) {
        float x, y;
        mesh_xy(lid[i], inv, cx, cy, rw, rh, rot, &x, &y);
        e->lid_xy[i * 2] = clampf(x / src_w, 0.f, 1.f);
        e->lid_xy[i * 2 + 1] = clampf(y / src_h, 0.f, 1.f);
    }
    for (i = 0; i < EYE_IRIS_N; i++) {
        float x, y;
        mesh_xy(iris_ring0 + i, inv, cx, cy, rw, rh, rot, &x, &y);
        e->iris_ring[i * 2] = clampf(x / src_w, 0.f, 1.f);
        e->iris_ring[i * 2 + 1] = clampf(y / src_h, 0.f, 1.f);
    }
    mesh_xy(outer, inv, cx, cy, rw, rh, rot, &ox, &oy);
    mesh_xy(inner, inv, cx, cy, rw, rh, rot, &ix, &iy);
    mesh_xy(iris_i, inv, cx, cy, rw, rh, rot, &irx, &iry);
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

/* ---- close-up eye path ---------------------------------------------------------------------------
 * When the camera is centimetres from the face the whole face does not fit in the frame and the Face Mesh
 * network says "not a face". BlazeFace still finds the face and gives the two eye positions, so we crop each
 * eye and run the small MediaPipe iris model (64x64 input) on it directly. */
enum { kIrisSize = 64, kIrisPts = 5 };
static TfliteNet g_iris;
static float *g_iris_in;
static float g_iris_contour[71 * 3 + 64];
static float g_iris_pts[kIrisPts * 3 + 8];
static int g_iris_ok, g_iris_tried, g_iris_out_contour = -1, g_iris_out_pts = -1;
static int g_iris_contour_bytes, g_iris_pts_bytes;
/* indices into the model's contour output that outline the eye (16 points) */
static int g_iris_lid[EYE_LID_N] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

static void iris_init(void)
{
    static const char *paths[] = {
        "models/iris_landmark.tflite",
        "/data/home/qnxuser/gazecomp_pupil/models/iris_landmark.tflite",
        "/data/home/qnxuser/gazecomp/models/iris_landmark.tflite",
        NULL,
    };
    const char *p;
    int i, nout;
    if (g_iris_tried) {
        return;
    }
    g_iris_tried = 1;
    p = first_existing(paths);
    if (!p || net_load(&g_iris, p, 2) != 0) {
        fprintf(stderr, "gazecomp: iris model not available; close-up path off\n");
        return;
    }
    {
        const TfLiteTensor *tin = TfLiteInterpreterGetInputTensor(g_iris.interp, 0);
        int nd = tin ? TfLiteTensorNumDims(tin) : 0;
        fprintf(stderr, "gazecomp: iris input type=%d ndims=%d dims=%d,%d,%d,%d bytes=%u\n",
                tin ? (int)TfLiteTensorType(tin) : -1, nd, nd > 0 ? TfLiteTensorDim(tin, 0) : 0,
                nd > 1 ? TfLiteTensorDim(tin, 1) : 0, nd > 2 ? TfLiteTensorDim(tin, 2) : 0,
                nd > 3 ? TfLiteTensorDim(tin, 3) : 0, tin ? (unsigned)TfLiteTensorByteSize(tin) : 0);
    }
    nout = TfLiteInterpreterGetOutputTensorCount(g_iris.interp);
    for (i = 0; i < nout; i++) {
        const TfLiteTensor *t = TfLiteInterpreterGetOutputTensor(g_iris.interp, i);
        int bytes = t ? (int)TfLiteTensorByteSize(t) : 0;
        fprintf(stderr, "gazecomp: iris out[%d] %s bytes=%d\n", i, t ? TfLiteTensorName(t) : "?", bytes);
        if (bytes == kIrisPts * 3 * (int)sizeof(float)) {
            g_iris_out_pts = i;
            g_iris_pts_bytes = bytes;
        } else if (bytes >= 71 * 3 * (int)sizeof(float) && bytes <= (int)sizeof(g_iris_contour)) {
            g_iris_out_contour = i;
            g_iris_contour_bytes = bytes;
        }
    }
    g_iris_in = (float *)malloc((size_t)kIrisSize * kIrisSize * 3u * sizeof(float));
    if (g_iris_in && g_iris_out_pts >= 0 && g_iris_out_contour >= 0) {
        g_iris_ok = 1;
        fprintf(stderr, "gazecomp: close-up iris path ready (%s)\n", p);
    } else {
        fprintf(stderr, "gazecomp: iris model outputs not recognised; close-up path off\n");
    }
}

static void iris_pt(const float *arr, int idx, float inv, float cx, float cy, float side, float rot, int flip,
                    float *x, float *y)
{
    project_norm(arr[idx * 3] * inv, arr[idx * 3 + 1] * inv, cx, cy, side, side, rot, flip, x, y);
}

static int run_iris_eye(const FrameSrc *src, float cx, float cy, float side, float rot, int flip, EyeMeas *e,
                        float src_w, float src_h, int dbg_idx)
{
    TfLiteTensor *tin = TfLiteInterpreterGetInputTensor(g_iris.interp, 0);
    const TfLiteTensor *t;
    float inv, mx = 0.f, ox, oy, ix, iy, irx, iry, lx[EYE_LID_N], ly[EYE_LID_N], w;
    int i, imin = 0, imax = 0;
    if (!tin) {
        return 0;
    }
    sample_rotated(src, cx, cy, side, side, rot, flip, kIrisSize, g_iris_in);
    if (TfLiteTensorCopyFromBuffer(tin, g_iris_in, (size_t)kIrisSize * kIrisSize * 3u * sizeof(float)) != kTfLiteOk) {
        return 0;
    }
    if (TfLiteInterpreterInvoke(g_iris.interp) != kTfLiteOk) {
        return 0;
    }
    t = TfLiteInterpreterGetOutputTensor(g_iris.interp, g_iris_out_contour);
    if (!t || TfLiteTensorCopyToBuffer(t, g_iris_contour, (size_t)g_iris_contour_bytes) != kTfLiteOk) {
        return 0;
    }
    t = TfLiteInterpreterGetOutputTensor(g_iris.interp, g_iris_out_pts);
    if (!t || TfLiteTensorCopyToBuffer(t, g_iris_pts, (size_t)g_iris_pts_bytes) != kTfLiteOk) {
        return 0;
    }
    for (i = 0; i < kIrisPts; i++) {
        mx = fmaxf(mx, fmaxf(fabsf(g_iris_pts[i * 3]), fabsf(g_iris_pts[i * 3 + 1])));
    }
    inv = mx > 2.f ? 1.f / (float)kIrisSize : 1.f;   /* pixels of the 64x64 input, or already normalised */

    if (getenv("GAZE_DUMP")) {
        static int dumped;
        if (dumped < 8) {
            char fn[64];
            FILE *f;
            snprintf(fn, sizeof(fn), "/tmp/eye_dbg_%d.ppm", dumped);
            f = fopen(fn, "wb");
            if (f) {
                fprintf(f, "P6\n%d %d\n255\n", kIrisSize, kIrisSize);
                for (i = 0; i < kIrisSize * kIrisSize * 3; i++) {
                    fputc((int)(clampf(g_iris_in[i], 0.f, 1.f) * 255.f + 0.5f), f);
                }
                fclose(f);
            }
            snprintf(fn, sizeof(fn), "/tmp/eye_dbg_%d.txt", dumped);
            f = fopen(fn, "w");
            if (f) {
                fprintf(f, "eye %d flip %d inv %.5f\n", dbg_idx, flip, inv);
                for (i = 0; i < kIrisPts; i++) {
                    fprintf(f, "iris %d %.3f %.3f %.3f\n", i, g_iris_pts[i * 3], g_iris_pts[i * 3 + 1], g_iris_pts[i * 3 + 2]);
                }
                for (i = 0; i < g_iris_contour_bytes / 12; i++) {
                    fprintf(f, "c %d %.3f %.3f %.3f\n", i, g_iris_contour[i * 3], g_iris_contour[i * 3 + 1],
                            g_iris_contour[i * 3 + 2]);
                }
                fclose(f);
            }
            dumped++;
        }
    }

    iris_pt(g_iris_pts, 0, inv, cx, cy, side, rot, flip, &irx, &iry);
    for (i = 0; i < EYE_IRIS_N; i++) {
        float x, y;
        iris_pt(g_iris_pts, 1 + i, inv, cx, cy, side, rot, flip, &x, &y);
        e->iris_ring[i * 2] = clampf(x / src_w, 0.f, 1.f);
        e->iris_ring[i * 2 + 1] = clampf(y / src_h, 0.f, 1.f);
    }
    for (i = 0; i < EYE_LID_N; i++) {
        iris_pt(g_iris_contour, g_iris_lid[i], inv, cx, cy, side, rot, flip, &lx[i], &ly[i]);
        e->lid_xy[i * 2] = clampf(lx[i] / src_w, 0.f, 1.f);
        e->lid_xy[i * 2 + 1] = clampf(ly[i] / src_h, 0.f, 1.f);
        if (lx[i] < lx[imin]) {
            imin = i;
        }
        if (lx[i] > lx[imax]) {
            imax = i;
        }
    }
    ox = lx[imin];
    oy = ly[imin];
    ix = lx[imax];
    iy = ly[imax];
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
    /* the iris must land inside the crop, near its middle, or the crop missed the eye */
    {
        float ddx = irx - cx, ddy = iry - cy;
        if (sqrtf(ddx * ddx + ddy * ddy) > 0.30f * side) {
            return 0;
        }
    }
    return 1;
}

/* Iris finder for the close-up camera: a circular Hough transform on a 4x down-scaled gray image. An iris is a
 * dark disc on a lighter eye, so image gradients on its rim point away from the centre; every edge pixel votes for
 * the centres at a range of radii. Two strong, similar-sized peaks a plausible distance apart are the two irises. */
enum { kHS = 4, kHR0 = 5, kHNR = 6 };   /* radii 5..10 small px = 20..40 px in the 960 frame */
typedef struct {
    float x, y, r, s;
} HPeak;

/* An iris is darker than the ring of eye around it. Bounds are in small (down-scaled) pixels. */
static float g_dd[3];   /* last candidate's inner gray, inner blue-minus-red, ring gray (debug) */

static int dark_disc(const uint8_t *gray, const int16_t *bmr, int gw, int gh, int bx, int by, float rr)
{
    float in_sum = 0.f, out_sum = 0.f, in_bmr = 0.f;
    int in_n = 0, out_n = 0, xx, yy;
    for (yy = by - (int)(1.8f * rr); yy <= by + (int)(1.8f * rr); yy++) {
        for (xx = bx - (int)(1.8f * rr); xx <= bx + (int)(1.8f * rr); xx++) {
            float d2, gv;
            if (xx < 0 || xx >= gw || yy < 0 || yy >= gh) {
                continue;
            }
            d2 = sqrtf((float)((xx - bx) * (xx - bx) + (yy - by) * (yy - by)));
            gv = (float)gray[(size_t)yy * (size_t)gw + (size_t)xx];
            if (d2 < 0.6f * rr) {
                in_sum += gv;
                in_bmr += (float)bmr[(size_t)yy * (size_t)gw + (size_t)xx];
                in_n++;
            } else if (d2 > 1.3f * rr && d2 < 1.8f * rr) {
                out_sum += gv;
                out_n++;
            }
        }
    }
    if (in_n < 3 || out_n < 3) {
        return 0;
    }
    g_dd[0] = in_sum / (float)in_n;
    g_dd[1] = in_bmr / (float)in_n;
    g_dd[2] = out_sum / (float)out_n;
    {
        static float max_gray = -1.f, max_bmr = 0.f, min_bmr = 0.f, ratio = 0.f;
        if (max_gray < 0.f) {
            const char *a = getenv("GAZE_MAX_IRIS_GRAY"), *b = getenv("GAZE_MAX_BMR"), *c = getenv("GAZE_MIN_BMR"),
                       *d = getenv("GAZE_MAX_RATIO");
            max_gray = a ? (float)atof(a) : 85.f;
            max_bmr = b ? (float)atof(b) : 14.f;
            min_bmr = c ? (float)atof(c) : -26.f;
            ratio = d ? (float)atof(d) : 0.78f;
        }
        if (in_sum / (float)in_n > max_gray || in_bmr / (float)in_n > max_bmr || in_bmr / (float)in_n < min_bmr) {
            return 0;   /* too bright, blue/purple like a plastic frame, or skin-red like the nose bridge */
        }
        return in_sum / (float)in_n <= ratio * (out_sum / (float)out_n);
    }
}

static void suppress_around(float *sm, int gw, int gh, int bx, int by, int sr)
{
    int r, x, y;
    for (r = 0; r < kHNR; r++) {
        float *o = sm + (size_t)r * (size_t)gh * (size_t)gw;
        for (y = by - sr; y <= by + sr; y++) {
            for (x = bx - sr; x <= bx + sr; x++) {
                if (x >= 0 && x < gw && y >= 0 && y < gh) {
                    o[(size_t)y * (size_t)gw + (size_t)x] = 0.f;
                }
            }
        }
    }
}

/* Search the whole (prior-limited) frame for the best plausible pair of irises; out[0] is the image-left eye.
 * prev (optional) is the current lock: a pair near it gets a 25% preference so the choice does not flicker. */
static int hough_irises(const uint8_t *rgb, int w, int h, const HPeak *prev, HPeak *out)
{
    static uint8_t *gray;
    static int16_t *bmr;
    static float *acc, *sm;
    static size_t gcap, acap;
    int gw = w / kHS, gh = h / kHS, x, y, r, i, j, np = 0, k, ylim, xlo, xhi;
    HPeak pk[12];
    size_t nacc = (size_t)kHNR * (size_t)gw * (size_t)gh;
    float first = 0.f;
    if (gw < 32 || gh < 32 || !rgb) {
        return 0;
    }
    {
        static float ymax_frac = -1.f, xmin_frac = -1.f, xmax_frac = -1.f;
        if (ymax_frac < 0.f) {
            const char *e = getenv("GAZE_EYE_YMAX"), *a = getenv("GAZE_EYE_XMIN"), *b = getenv("GAZE_EYE_XMAX");
            ymax_frac = e ? (float)atof(e) : 0.45f;
            xmin_frac = a ? (float)atof(a) : 0.12f;
            xmax_frac = b ? (float)atof(b) : 0.88f;
        }
        ylim = (int)(ymax_frac * (float)gh);
        if (ylim < 8) {
            ylim = 8;
        }
        if (ylim > gh) {
            ylim = gh;
        }
        xlo = (int)(xmin_frac * (float)gw);
        xhi = (int)(xmax_frac * (float)gw);
    }
    if (gcap < (size_t)gw * (size_t)gh) {
        free(gray);
        free(bmr);
        gray = (uint8_t *)malloc((size_t)gw * (size_t)gh);
        bmr = (int16_t *)malloc((size_t)gw * (size_t)gh * sizeof(int16_t));
        gcap = (gray && bmr) ? (size_t)gw * (size_t)gh : 0;
    }
    if (acap < nacc) {
        free(acc);
        free(sm);
        acc = (float *)malloc(nacc * sizeof(float));
        sm = (float *)malloc(nacc * sizeof(float));
        acap = (acc && sm) ? nacc : 0;
    }
    if (!gray || !bmr || !acc || !sm || acap < nacc || gcap < (size_t)gw * (size_t)gh) {
        return 0;
    }
    for (y = 0; y < gh; y++) {
        for (x = 0; x < gw; x++) {
            int sum = 0, bsum = 0, dx, dy;
            for (dy = 0; dy < kHS; dy++) {
                for (dx = 0; dx < kHS; dx++) {
                    const uint8_t *p = rgb + ((size_t)(y * kHS + dy) * (size_t)w + (size_t)(x * kHS + dx)) * 3u;
                    sum += (77 * p[0] + 150 * p[1] + 29 * p[2]) >> 8;
                    bsum += (int)p[2] - (int)p[0];
                }
            }
            gray[(size_t)y * (size_t)gw + (size_t)x] = (uint8_t)(sum / (kHS * kHS));
            bmr[(size_t)y * (size_t)gw + (size_t)x] = (int16_t)(bsum / (kHS * kHS));
        }
    }
    memset(acc, 0, nacc * sizeof(float));
    for (y = 1; y < gh - 1; y++) {
        for (x = 1; x < gw - 1; x++) {
            const uint8_t *g = gray + (size_t)y * (size_t)gw + (size_t)x;
            int gx = (g[-gw + 1] + 2 * g[1] + g[gw + 1]) - (g[-gw - 1] + 2 * g[-1] + g[gw - 1]);
            int gy = (g[gw - 1] + 2 * g[gw] + g[gw + 1]) - (g[-gw - 1] + 2 * g[-gw] + g[-gw + 1]);
            float mag = sqrtf((float)(gx * gx + gy * gy)), nx, ny;
            if (mag < 32.f) {
                continue;
            }
            nx = (float)gx / mag;
            ny = (float)gy / mag;
            for (r = 0; r < kHNR; r++) {
                float rr = (float)(kHR0 + r);
                int ix = (int)((float)x - nx * rr + 0.5f), iy = (int)((float)y - ny * rr + 0.5f);
                if (ix >= 0 && ix < gw && iy >= 0 && iy < gh) {
                    acc[((size_t)r * (size_t)gh + (size_t)iy) * (size_t)gw + (size_t)ix] += 1.f;
                }
            }
        }
    }
    for (r = 0; r < kHNR; r++) {
        const float *a = acc + (size_t)r * (size_t)gh * (size_t)gw;
        float *o = sm + (size_t)r * (size_t)gh * (size_t)gw;
        memset(o, 0, (size_t)gh * (size_t)gw * sizeof(float));
        for (y = 1; y < gh - 1; y++) {
            for (x = 1; x < gw - 1; x++) {
                const float *p = a + (size_t)y * (size_t)gw + (size_t)x;
                o[(size_t)y * (size_t)gw + (size_t)x] = p[-gw - 1] + p[-gw] + p[-gw + 1] + p[-1] + p[0] + p[1] +
                                                          p[gw - 1] + p[gw] + p[gw + 1];
            }
        }
    }

    /* ---- acquisition: best plausible pair anywhere in the prior region ---- */
    for (k = 0; k < 12; k++) {
        float best = 0.f;
        int bx = 0, by = 0, br = 0;
        for (r = 0; r < kHNR; r++) {
            const float *o = sm + (size_t)r * (size_t)gh * (size_t)gw;
            for (y = 0; y < ylim; y++) {
                for (x = xlo; x < xhi; x++) {
                    float v = o[(size_t)y * (size_t)gw + (size_t)x];
                    if (v > best) {
                        best = v;
                        bx = x;
                        by = y;
                        br = r;
                    }
                }
            }
        }
        if (best <= 0.f || (k > 0 && best < 0.35f * first)) {
            break;
        }
        if (k == 0) {
            first = best;
        }
        {
            int pass;
            g_dd[0] = g_dd[1] = g_dd[2] = -1.f;
            pass = dark_disc(gray, bmr, gw, gh, bx, by, (float)(kHR0 + br));
            if (getenv("GAZE_DBG")) {
                static int dl;
                if (dl < 900) {
                    fprintf(stderr, "gazecomp: cand k=%d x=%.0f y=%.0f r=%.0f s=%.0f ig=%.0f ib=%.0f og=%.0f pass=%d\n", k,
                            ((float)bx + 0.5f) * (float)kHS, ((float)by + 0.5f) * (float)kHS, (float)(kHR0 + br) * (float)kHS, best,
                            g_dd[0], g_dd[1], g_dd[2], pass);
                    dl++;
                }
            }
            if (!pass) {
                suppress_around(sm, gw, gh, bx, by, 2 * (kHR0 + br));
                continue;
            }
        }
        pk[np].x = ((float)bx + 0.5f) * (float)kHS;
        pk[np].y = ((float)by + 0.5f) * (float)kHS;
        pk[np].r = (float)(kHR0 + br) * (float)kHS;
        pk[np].s = best;
        np++;
        suppress_around(sm, gw, gh, bx, by, 2 * (kHR0 + br));
    }
    {
        float best = 0.f;
        int bi = -1, bj = -1;
        for (i = 0; i < np; i++) {
            for (j = i + 1; j < np; j++) {
                float rm = 0.5f * (pk[i].r + pk[j].r);
                float dx = fabsf(pk[i].x - pk[j].x), dy = fabsf(pk[i].y - pk[j].y);
                if (dx < 6.f * rm || dx > 17.f * rm || dy > 4.f * rm || fabsf(pk[i].r - pk[j].r) > 0.4f * rm) {
                    continue;
                }
                {
                    float sc = pk[i].s + pk[j].s;
                    if (prev) {   /* prefer the pair that is already locked on (25%) */
                        const HPeak *lft = pk[i].x < pk[j].x ? &pk[i] : &pk[j];
                        const HPeak *rgt = pk[i].x < pk[j].x ? &pk[j] : &pk[i];
                        if (hypotf(lft->x - prev[0].x, lft->y - prev[0].y) < 60.f &&
                            hypotf(rgt->x - prev[1].x, rgt->y - prev[1].y) < 60.f) {
                            sc *= 1.25f;
                        }
                    }
                    if (sc > best) {
                        best = sc;
                        bi = i;
                        bj = j;
                    }
                }
            }
        }
        if (bi < 0) {
            return 0;
        }
        out[0] = pk[bi];
        out[1] = pk[bj];
        if (out[0].x > out[1].x) {
            HPeak t = out[0];
            out[0] = out[1];
            out[1] = t;
        }
    }
    return 1;
}

/* Close-up path. The whole upper frame is searched every frame (so a slipped lock can always recover); the current
 * lock is preferred, and a different pair replaces it only after winning on 3 frames in a row. A first lock needs the
 * same pair on 3 consecutive frames. Positions are smoothed; a lock that finds nothing for 6 frames is dropped. */
static int closeup_eyes(const FrameSrc *src, const Det *d, float src_w, float src_h, EyeMeas *left, EyeMeas *right)
{
    static int flip_mode = -1, have_lock, lost, confirm_n, jump_n;
    static HPeak lock[2], cand[2];
    HPeak q[2];
    float rm, rot, side;
    int i, got;
    (void)d;
    if (flip_mode < 0) {
        const char *e = getenv("GAZE_IRIS_FLIP");
        flip_mode = e ? atoi(e) : 1;
    }
    got = hough_irises(src->rgb, src->rgb_w, src->rgb_h, have_lock ? lock : NULL, q);
    if (have_lock) {
        if (got) {
            int near_lock = hypotf(q[0].x - lock[0].x, q[0].y - lock[0].y) < 60.f &&
                            hypotf(q[1].x - lock[1].x, q[1].y - lock[1].y) < 60.f;
            if (near_lock) {
                for (i = 0; i < 2; i++) {   /* smooth: 0.6 new + 0.4 old */
                    lock[i].x = 0.4f * lock[i].x + 0.6f * q[i].x;
                    lock[i].y = 0.4f * lock[i].y + 0.6f * q[i].y;
                    lock[i].r = 0.4f * lock[i].r + 0.6f * q[i].r;
                    lock[i].s = q[i].s;
                }
                lost = 0;
                jump_n = 0;
            } else {   /* a different pair is winning: switch only if it persists */
                if (jump_n > 0 && hypotf(q[0].x - cand[0].x, q[0].y - cand[0].y) < 45.f &&
                    hypotf(q[1].x - cand[1].x, q[1].y - cand[1].y) < 45.f) {
                    jump_n++;
                } else {
                    jump_n = 1;
                }
                cand[0] = q[0];
                cand[1] = q[1];
                if (jump_n >= 3) {
                    lock[0] = q[0];
                    lock[1] = q[1];
                    jump_n = 0;
                    lost = 0;
                }
            }
        } else {
            lost++;
            jump_n = 0;
            if (lost > 6) {
                have_lock = 0;
                confirm_n = 0;
                return 0;
            }
        }
    } else {
        if (got) {
            if (confirm_n > 0 && hypotf(q[0].x - cand[0].x, q[0].y - cand[0].y) < 45.f &&
                hypotf(q[1].x - cand[1].x, q[1].y - cand[1].y) < 45.f) {
                confirm_n++;
            } else {
                confirm_n = 1;
            }
            cand[0] = q[0];
            cand[1] = q[1];
            if (confirm_n >= 3) {
                lock[0] = q[0];
                lock[1] = q[1];
                have_lock = 1;
                lost = 0;
                jump_n = 0;
                confirm_n = 0;
            }
        } else {
            confirm_n = 0;
        }
        if (!have_lock) {
            return 0;
        }
    }
    rm = 0.5f * (lock[0].r + lock[1].r);
    rot = atan2f(lock[1].y - lock[0].y, lock[1].x - lock[0].x);
    side = 9.5f * rm;
    {
        static int calls;
        if ((calls++ % 15) == 0 && calls < 900) {
            fprintf(stderr, "gazecomp: closeup track L=(%.0f,%.0f r=%.0f) R=(%.0f,%.0f r=%.0f) lost=%d jump=%d side=%.0f\n",
                    lock[0].x, lock[0].y, lock[0].r, lock[1].x, lock[1].y, lock[1].r, lost, jump_n, side);
        }
    }
    if (!run_iris_eye(src, lock[0].x, lock[0].y, side, rot, (flip_mode & 1) ? 1 : 0, left, src_w, src_h, 0) ||
        !run_iris_eye(src, lock[1].x, lock[1].y, side, rot, (flip_mode & 2) ? 1 : 0, right, src_w, src_h, 1)) {
        lost++;
        if (lost > 6) {
            have_lock = 0;
            confirm_n = 0;
        }
        return 0;
    }
    return 1;
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
    nkeep = nms_ranked(dets, ndet, order, 1);
    if (nkeep <= 0) {
        iris_init();
        if (g_iris_ok && closeup_eyes(src, NULL, (float)src->rgb_w, (float)src->rgb_h, left, right)) {
            if (score_out) {
                *score_out = 0.6f;
            }
            hold_l = *left;
            hold_r = *right;
            hold_s = 0.6f;
            hold_left = 3;
            return 2;
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
        float bw = d->xmax - d->xmin;
        float bh = d->ymax - d->ymin;
        /* Centre on the detector's two eye keypoints (0,1), not the box
         * centre — the box centre sits on the nose and pulls the mesh down
         * onto the cheeks. */
        float cx = 0.5f * (d->kp[0] + d->kp[2]);
        float cy = 0.5f * (d->kp[1] + d->kp[3]);
        float rw = bw * 1.5f;
        float rh = bh * 1.5f;
        float rot, flag, inv;
        {
            float min_side = fminf(src_w, src_h) * kMinFaceFrac;
            if (bw < min_side || bh < min_side) {
                continue; /* too small to be this camera's wearer */
            }
        }
        if (rw < 8.f || rh < 8.f) {
            continue;
        }
        if (cx < d->xmin || cx > d->xmax || cy < d->ymin || cy > d->ymax) {
            cx = 0.5f * (d->xmin + d->xmax);
            cy = 0.5f * (d->ymin + d->ymax);
        }
        rot = atan2f(d->kp[3] - d->kp[1], d->kp[2] - d->kp[0]);
        sample_rotated(&crop_src, cx, cy, rw, rh, rot, 0, kLmSize, g_lm_in);
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
                        "gazecomp: mesh det=%.3f flag_raw=%.3f flag_sig=%.3f id2=%.3f id2_sig=%.3f rw=%.1f rh=%.1f\n",
                        d->score, g_flag[0], flag, extra, sigmoidf_fast(extra), rw, rh);
                mdbg++;
            }
        }
        if (flag < kFaceFlagMin) {
            continue;
        }
        inv = mesh_inv();
        fill_eye(left, kLeftLid, kLeftOuter, kLeftInner, kLeftIris, kLeftIrisRing, inv, cx, cy, rw,
                 rh, rot, src_w, src_h);
        fill_eye(right, kRightLid, kRightOuter, kRightInner, kRightIris, kRightIrisRing, inv, cx, cy,
                 rw, rh, rot, src_w, src_h);
        if (!eyes_plausible(left, right, src_w, src_h)) {
            continue;
        }
        if (score_out) {
            *score_out = fminf(1.f, fmaxf(d->score, flag));
        }
        hold_l = *left;
        hold_r = *right;
        hold_s = score_out ? *score_out : flag;
        hold_left = 3;
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

    /* The mesh rejected the face (typically: camera very close, face larger than the frame). BlazeFace still
     * found it, so try the close-up path on its two eye keypoints. */
    iris_init();
    if (g_iris_ok && closeup_eyes(&crop_src, NULL, src_w, src_h, left, right)) {
        if (score_out) {
            *score_out = 0.6f;
        }
        hold_l = *left;
        hold_r = *right;
        hold_s = 0.6f;
        hold_left = 3;
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

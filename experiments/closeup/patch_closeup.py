"""Add a close-up eye path to the pupil-in-eye branch's face_lm.cpp.

When the face is too big for Face Mesh (camera centimetres from the face), BlazeFace still finds it and gives two
eye keypoints. Crop each eye and run the MediaPipe iris model directly."""
import shutil
import sys

path = sys.argv[1]
shutil.copyfile(path, path + ".branch")
s = open(path, encoding="utf-8").read()
assert "closeup_eyes" not in s

block = r'''
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
        if (sqrtf(ddx * ddx + ddy * ddy) > 0.45f * side) {
            return 0;
        }
    }
    return 1;
}

/* BlazeFace eye keypoints -> two eye crops -> iris model. Returns 1 on success. */
static int closeup_eyes(const FrameSrc *src, const Det *d, float src_w, float src_h, EyeMeas *left, EyeMeas *right)
{
    static int flip_mode = -1;
    float ax = d->kp[0], ay = d->kp[1], bx = d->kp[2], by = d->kp[3], iod, side, rot;
    if (flip_mode < 0) {
        const char *e = getenv("GAZE_IRIS_FLIP");
        flip_mode = e ? atoi(e) : 1;
    }
    if (ax > bx) {   /* keep the eye on the image's left first */
        float tx = ax, ty = ay;
        ax = bx;
        ay = by;
        bx = tx;
        by = ty;
    }
    iod = sqrtf((bx - ax) * (bx - ax) + (by - ay) * (by - ay));
    if (iod < 0.10f * src_w) {
        return 0;
    }
    side = 0.75f * iod;
    rot = atan2f(by - ay, bx - ax);
    if (!run_iris_eye(src, ax, ay, side, rot, (flip_mode & 1) ? 1 : 0, left, src_w, src_h, 0)) {
        return 0;
    }
    if (!run_iris_eye(src, bx, by, side, rot, (flip_mode & 2) ? 1 : 0, right, src_w, src_h, 1)) {
        return 0;
    }
    return 1;
}

int face_eyes_from_frame('''
assert "\nint face_eyes_from_frame(" in s
s = s.replace("\nint face_eyes_from_frame(", block, 1)

old = '''    if (score_out) {
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
}'''
new = '''    /* The mesh rejected the face (typically: camera very close, face larger than the frame). BlazeFace still
     * found it, so try the close-up path on its two eye keypoints. */
    if (nkeep > 0) {
        iris_init();
        if (g_iris_ok && dets[order[0]].score >= 0.5f &&
            closeup_eyes(&crop_src, &dets[order[0]], src_w, src_h, left, right)) {
            if (score_out) {
                *score_out = dets[order[0]].score;
            }
            hold_l = *left;
            hold_r = *right;
            hold_s = dets[order[0]].score;
            hold_left = 3;
            return 2;
        }
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
}'''
assert old in s
s = s.replace(old, new, 1)
open(path, "w", encoding="utf-8").write(s)
print("patched", path)

"""Replace the BlazeFace-keypoint close-up path with a Hough iris finder."""
import sys

path = sys.argv[1]
s = open(path, encoding="utf-8").read()
assert "hough_irises" not in s

start = s.index("/* BlazeFace eye keypoints -> two eye crops -> iris model.")
end = s.index("\nint face_eyes_from_frame(")
new_block = r'''/* Iris finder for the close-up camera: a circular Hough transform on a 4x down-scaled gray image. An iris is a
 * dark disc on a lighter eye, so image gradients on its rim point away from the centre; every edge pixel votes for
 * the centres at a range of radii. Two strong, similar-sized peaks a plausible distance apart are the two irises. */
enum { kHS = 4, kHR0 = 7, kHNR = 11 };   /* radii 7..17 small px = 28..68 px in the 960 frame */
typedef struct {
    float x, y, r, s;
} HPeak;

static int hough_irises(const uint8_t *rgb, int w, int h, HPeak *out)
{
    static uint8_t *gray;
    static float *acc, *sm;
    static size_t gcap, acap;
    int gw = w / kHS, gh = h / kHS, x, y, r, i, j, np = 0, k;
    HPeak pk[12];
    size_t nacc = (size_t)kHNR * (size_t)gw * (size_t)gh;
    float first = 0.f;
    if (gw < 32 || gh < 32 || !rgb) {
        return 0;
    }
    if (gcap < (size_t)gw * (size_t)gh) {
        free(gray);
        gray = (uint8_t *)malloc((size_t)gw * (size_t)gh);
        gcap = gray ? (size_t)gw * (size_t)gh : 0;
    }
    if (acap < nacc) {
        free(acc);
        free(sm);
        acc = (float *)malloc(nacc * sizeof(float));
        sm = (float *)malloc(nacc * sizeof(float));
        acap = (acc && sm) ? nacc : 0;
    }
    if (!gray || !acc || !sm || acap < nacc) {
        return 0;
    }
    for (y = 0; y < gh; y++) {
        for (x = 0; x < gw; x++) {
            int sum = 0, dx, dy;
            for (dy = 0; dy < kHS; dy++) {
                for (dx = 0; dx < kHS; dx++) {
                    const uint8_t *p = rgb + ((size_t)(y * kHS + dy) * (size_t)w + (size_t)(x * kHS + dx)) * 3u;
                    sum += (77 * p[0] + 150 * p[1] + 29 * p[2]) >> 8;
                }
            }
            gray[(size_t)y * (size_t)gw + (size_t)x] = (uint8_t)(sum / (kHS * kHS));
        }
    }
    memset(acc, 0, nacc * sizeof(float));
    for (y = 1; y < gh - 1; y++) {
        for (x = 1; x < gw - 1; x++) {
            const uint8_t *g = gray + (size_t)y * (size_t)gw + (size_t)x;
            int gx = (g[-gw + 1] + 2 * g[1] + g[gw + 1]) - (g[-gw - 1] + 2 * g[-1] + g[gw - 1]);
            int gy = (g[gw - 1] + 2 * g[gw] + g[gw + 1]) - (g[-gw - 1] + 2 * g[-gw] + g[-gw + 1]);
            float mag = sqrtf((float)(gx * gx + gy * gy)), nx, ny;
            if (mag < 48.f) {
                continue;
            }
            nx = (float)gx / mag;
            ny = (float)gy / mag;
            for (r = 0; r < kHNR; r++) {
                float rr = (float)(kHR0 + r);
                int ix = (int)((float)x - nx * rr + 0.5f), iy = (int)((float)y - ny * rr + 0.5f);
                if (ix >= 0 && ix < gw && iy >= 0 && iy < gh) {
                    acc[((size_t)r * (size_t)gh + (size_t)iy) * (size_t)gw + (size_t)ix] += mag;
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
    for (k = 0; k < 12; k++) {
        float best = 0.f;
        int bx = 0, by = 0, br = 0;
        for (r = 0; r < kHNR; r++) {
            const float *o = sm + (size_t)r * (size_t)gh * (size_t)gw;
            for (y = 0; y < gh; y++) {
                for (x = 0; x < gw; x++) {
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
        pk[np].x = ((float)bx + 0.5f) * (float)kHS;
        pk[np].y = ((float)by + 0.5f) * (float)kHS;
        pk[np].r = (float)(kHR0 + br) * (float)kHS;
        pk[np].s = best;
        np++;
        {   /* suppress everything within twice this radius, at every radius */
            int sr = 2 * (kHR0 + br);
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
    }
    {   /* best plausible pair: similar radius, level, spaced about 8..15 iris radii apart */
        float best = 0.f;
        int bi = -1, bj = -1;
        for (i = 0; i < np; i++) {
            for (j = i + 1; j < np; j++) {
                float rm = 0.5f * (pk[i].r + pk[j].r);
                float dx = fabsf(pk[i].x - pk[j].x), dy = fabsf(pk[i].y - pk[j].y);
                if (dx < 6.f * rm || dx > 17.f * rm || dy > 4.f * rm || fabsf(pk[i].r - pk[j].r) > 0.4f * rm) {
                    continue;
                }
                if (pk[i].s + pk[j].s > best) {
                    best = pk[i].s + pk[j].s;
                    bi = i;
                    bj = j;
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

/* Close-up path: find the two irises with the Hough transform, then crop each eye and run the iris model. */
static int closeup_eyes(const FrameSrc *src, const Det *d, float src_w, float src_h, EyeMeas *left, EyeMeas *right)
{
    static int flip_mode = -1;
    HPeak p[2];
    float rm, rot, side;
    (void)d;
    if (flip_mode < 0) {
        const char *e = getenv("GAZE_IRIS_FLIP");
        flip_mode = e ? atoi(e) : 1;
    }
    if (!hough_irises(src->rgb, src->rgb_w, src->rgb_h, p)) {
        return 0;
    }
    rm = 0.5f * (p[0].r + p[1].r);
    rot = atan2f(p[1].y - p[0].y, p[1].x - p[0].x);
    side = 9.5f * rm;
    {
        static int calls;
        if ((calls++ % 15) == 0 && calls < 600) {
            fprintf(stderr, "gazecomp: closeup hough L=(%.0f,%.0f r=%.0f s=%.0f) R=(%.0f,%.0f r=%.0f s=%.0f) side=%.0f\n",
                    p[0].x, p[0].y, p[0].r, p[0].s, p[1].x, p[1].y, p[1].r, p[1].s, side);
        }
    }
    if (!run_iris_eye(src, p[0].x, p[0].y, side, rot, (flip_mode & 1) ? 1 : 0, left, src_w, src_h, 0)) {
        return 0;
    }
    if (!run_iris_eye(src, p[1].x, p[1].y, side, rot, (flip_mode & 2) ? 1 : 0, right, src_w, src_h, 1)) {
        return 0;
    }
    return 1;
}
'''
s = s[:start] + new_block + s[end:]

# the iris must land near the middle of a crop that was already centred on it
assert "0.45f * side" in s
s = s.replace("0.45f * side", "0.30f * side", 1)

# call site 1: after the mesh loop (was gated on a BlazeFace detection)
old = '''    if (nkeep > 0) {
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
    }'''
new = '''    iris_init();
    if (g_iris_ok && closeup_eyes(&crop_src, NULL, src_w, src_h, left, right)) {
        if (score_out) {
            *score_out = 0.6f;
        }
        hold_l = *left;
        hold_r = *right;
        hold_s = 0.6f;
        hold_left = 3;
        return 2;
    }'''
assert old in s
s = s.replace(old, new, 1)

# call site 2: BlazeFace found nothing at all
old = '''    nkeep = nms_ranked(dets, ndet, order, 1);
    if (nkeep <= 0) {
        if (hold_left > 0) {'''
new = '''    nkeep = nms_ranked(dets, ndet, order, 1);
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
        if (hold_left > 0) {'''
assert old in s
s = s.replace(old, new, 1)
open(path, "w", encoding="utf-8").write(s)
print("patched", path)

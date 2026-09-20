"""Hough iris finder, round 4: temporal tracking and a confirmed first lock."""
import shutil
import sys

path = sys.argv[1]
shutil.copyfile(path, path + ".hough3")
s = open(path, encoding="utf-8").read()
assert "dark_disc(" not in s

start = s.index("static int hough_irises(")
end = s.index("\nint face_eyes_from_frame(")

new_block = r'''/* An iris is darker than the ring of eye around it. Bounds are in small (down-scaled) pixels. */
static int dark_disc(const uint8_t *gray, int gw, int gh, int bx, int by, float rr)
{
    float in_sum = 0.f, out_sum = 0.f;
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
                in_n++;
            } else if (d2 > 1.3f * rr && d2 < 1.8f * rr) {
                out_sum += gv;
                out_n++;
            }
        }
    }
    return in_n >= 3 && out_n >= 3 && in_sum / (float)in_n <= 0.92f * (out_sum / (float)out_n);
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

/* prev == NULL: search the whole (prior-limited) frame for the best plausible pair of irises.
 * prev != NULL: track - look for each iris only near where it was last seen. out[0] is the image-left eye. */
static int hough_irises(const uint8_t *rgb, int w, int h, const HPeak *prev, HPeak *out)
{
    static uint8_t *gray;
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

    if (prev) {   /* ---- tracking: look only near where each iris was last seen ---- */
        for (i = 0; i < 2; i++) {
            int pcx = (int)(prev[i].x / (float)kHS), pcy = (int)(prev[i].y / (float)kHS), win = 12;
            float best = 0.f;
            int bx = -1, by = -1, br = 0;
            for (r = 0; r < kHNR; r++) {
                const float *o = sm + (size_t)r * (size_t)gh * (size_t)gw;
                for (y = pcy - win; y <= pcy + win; y++) {
                    for (x = pcx - win; x <= pcx + win; x++) {
                        float v;
                        if (x < 1 || x >= gw - 1 || y < 1 || y >= gh - 1) {
                            continue;
                        }
                        v = o[(size_t)y * (size_t)gw + (size_t)x];
                        if (v > best) {
                            best = v;
                            bx = x;
                            by = y;
                            br = r;
                        }
                    }
                }
            }
            if (bx < 0 || best < 10.f || !dark_disc(gray, gw, gh, bx, by, (float)(kHR0 + br))) {
                return 0;
            }
            out[i].x = ((float)bx + 0.5f) * (float)kHS;
            out[i].y = ((float)by + 0.5f) * (float)kHS;
            out[i].r = (float)(kHR0 + br) * (float)kHS;
            out[i].s = best;
        }
        return 1;
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
        if (!dark_disc(gray, gw, gh, bx, by, (float)(kHR0 + br))) {
            suppress_around(sm, gw, gh, bx, by, 2 * (kHR0 + br));
            continue;
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

/* Close-up path with temporal tracking. A first lock needs the same pair on 3 consecutive frames; after that each
 * iris is searched only near its last position, positions are smoothed, and a lost lock is dropped after a few frames. */
static int closeup_eyes(const FrameSrc *src, const Det *d, float src_w, float src_h, EyeMeas *left, EyeMeas *right)
{
    static int flip_mode = -1, have_lock, lost, confirm_n;
    static HPeak lock[2], cand[2];
    HPeak q[2];
    float rm, rot, side;
    int i;
    (void)d;
    if (flip_mode < 0) {
        const char *e = getenv("GAZE_IRIS_FLIP");
        flip_mode = e ? atoi(e) : 1;
    }
    if (have_lock) {
        if (hough_irises(src->rgb, src->rgb_w, src->rgb_h, lock, q)) {
            for (i = 0; i < 2; i++) {   /* smooth: 0.6 new + 0.4 old */
                lock[i].x = 0.4f * lock[i].x + 0.6f * q[i].x;
                lock[i].y = 0.4f * lock[i].y + 0.6f * q[i].y;
                lock[i].r = 0.4f * lock[i].r + 0.6f * q[i].r;
                lock[i].s = q[i].s;
            }
            lost = 0;
        } else {
            lost++;
            if (lost > 6) {   /* lost for too long: drop the lock and start over */
                have_lock = 0;
                confirm_n = 0;
                return 0;
            }
        }
    } else {
        if (hough_irises(src->rgb, src->rgb_w, src->rgb_h, NULL, q)) {
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
            fprintf(stderr, "gazecomp: closeup track L=(%.0f,%.0f r=%.0f) R=(%.0f,%.0f r=%.0f) lost=%d side=%.0f\n",
                    lock[0].x, lock[0].y, lock[0].r, lock[1].x, lock[1].y, lock[1].r, lost, side);
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
'''
s = s[:start] + new_block + s[end:]
open(path, "w", encoding="utf-8").write(s)
print("patched", path)

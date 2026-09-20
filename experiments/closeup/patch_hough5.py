"""Round 5: no more windowed tracking (it could not recover once it slipped). Search the whole prior region every frame,
give the current lock a 25% preference, and only switch to a different pair after it wins 3 frames in a row.
Also: the dark-pupil fit in process.c is off unless GAZE_PUPIL is set."""
import shutil
import sys

face_path, proc_path = sys.argv[1], sys.argv[2]

# ------------------------------------------------------------------ face_lm.cpp
s = open(face_path, encoding="utf-8").read()
shutil.copyfile(face_path, face_path + ".hough4")
assert "jump_n" not in s

# 1. drop the windowed tracking block
a = s.index("    if (prev) {   /* ---- tracking: look only near where each iris was last seen ---- */")
b = s.index("    /* ---- acquisition: best plausible pair anywhere in the prior region ---- */")
s = s[:a] + s[b:]

# 2. continuity preference in the pair choice
old = """                if (pk[i].s + pk[j].s > best) {
                    best = pk[i].s + pk[j].s;
                    bi = i;
                    bj = j;
                }"""
new = """                {
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
                }"""
assert old in s
s = s.replace(old, new, 1)
s = s.replace("/* prev == NULL: search the whole (prior-limited) frame for the best plausible pair of irises.\n * prev != NULL: track - look for each iris only near where it was last seen. out[0] is the image-left eye. */",
              "/* Search the whole (prior-limited) frame for the best plausible pair of irises; out[0] is the image-left eye.\n * prev (optional) is the current lock: a pair near it gets a 25% preference so the choice does not flicker. */")

# 3. new closeup_eyes control logic
start = s.index("/* Close-up path with temporal tracking.")
end = s.index("\nint face_eyes_from_frame(")
new_fn = r'''/* Close-up path. The whole upper frame is searched every frame (so a slipped lock can always recover); the current
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
'''
s = s[:start] + new_fn + s[end:]
open(face_path, "w", encoding="utf-8").write(s)
print("face_lm.cpp patched")

# ------------------------------------------------------------------ process.c: dark pupil off by default
p = open(proc_path, encoding="utf-8").read()
shutil.copyfile(proc_path, proc_path + ".branch")
assert "GAZE_PUPIL" not in p
old = """            for (i = 0; i < 2; i++) {
                PupilMeas p;
                memset(&p, 0, sizeof(p));
                if (have[i] &&
                    pupil_from_crop("""
new = """            for (i = 0; i < 2; i++) {
                PupilMeas p;
                static int use_pupil = -1;   /* the small dark-pupil fit is ignored unless GAZE_PUPIL is set */
                if (use_pupil < 0) {
                    use_pupil = getenv("GAZE_PUPIL") != NULL;
                }
                memset(&p, 0, sizeof(p));
                if (have[i] && use_pupil &&
                    pupil_from_crop("""
assert old in p
p = p.replace(old, new, 1)
if "#include <stdlib.h>" not in p:
    p = p.replace('#include "pupil.h"', '#include "pupil.h"\n#include <stdlib.h>', 1)
open(proc_path, "w", encoding="utf-8").write(p)
print("process.c patched")

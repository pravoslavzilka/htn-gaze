"""Tighten the Hough iris finder: smaller radii, a vertical prior for this mount, and a dark-centre check."""
import shutil
import sys

path = sys.argv[1]
shutil.copyfile(path, path + ".hough1")
s = open(path, encoding="utf-8").read()
assert "GAZE_EYE_YMAX" not in s

old = "enum { kHS = 4, kHR0 = 7, kHNR = 11 };   /* radii 7..17 small px = 28..68 px in the 960 frame */"
new = "enum { kHS = 4, kHR0 = 5, kHNR = 11 };   /* radii 5..15 small px = 20..60 px in the 960 frame */"
assert old in s
s = s.replace(old, new, 1)

# vertical prior: on this rig the eyes sit in the upper part of the frame
old = "    float first = 0.f;\n    if (gw < 32 || gh < 32 || !rgb) {\n        return 0;\n    }\n"
new = ("    float first = 0.f;\n    int ylim;\n"
       "    if (gw < 32 || gh < 32 || !rgb) {\n        return 0;\n    }\n"
       "    {\n        static float ymax_frac = -1.f;\n"
       "        if (ymax_frac < 0.f) {\n            const char *e = getenv(\"GAZE_EYE_YMAX\");\n"
       "            ymax_frac = e ? (float)atof(e) : 0.45f;\n        }\n"
       "        ylim = (int)(ymax_frac * (float)gh);\n        if (ylim < 8) {\n            ylim = 8;\n        }\n        if (ylim > gh) {\n            ylim = gh;\n        }\n    }\n")
assert old in s
s = s.replace(old, new, 1)

old = "            for (y = 0; y < gh; y++) {\n                for (x = 0; x < gw; x++) {\n                    float v = o[(size_t)y * (size_t)gw + (size_t)x];"
new = "            for (y = 0; y < ylim; y++) {\n                for (x = 0; x < gw; x++) {\n                    float v = o[(size_t)y * (size_t)gw + (size_t)x];"
assert old in s
s = s.replace(old, new, 1)

# dark-centre check: an iris is darker than the ring of eye around it
old = "        if (best <= 0.f || (k > 0 && best < 0.35f * first)) {\n            break;\n        }\n"
new = old
old_rec = ("        pk[np].x = ((float)bx + 0.5f) * (float)kHS;\n"
           "        pk[np].y = ((float)by + 0.5f) * (float)kHS;\n"
           "        pk[np].r = (float)(kHR0 + br) * (float)kHS;\n"
           "        pk[np].s = best;\n"
           "        np++;\n")
new_rec = ("        {   /* mean brightness inside the disc versus a ring around it */\n"
           "            float rr = (float)(kHR0 + br), in_sum = 0.f, out_sum = 0.f;\n"
           "            int in_n = 0, out_n = 0, xx, yy;\n"
           "            for (yy = by - (int)(1.8f * rr); yy <= by + (int)(1.8f * rr); yy++) {\n"
           "                for (xx = bx - (int)(1.8f * rr); xx <= bx + (int)(1.8f * rr); xx++) {\n"
           "                    float d2, gv;\n"
           "                    if (xx < 0 || xx >= gw || yy < 0 || yy >= gh) {\n                        continue;\n                    }\n"
           "                    d2 = sqrtf((float)((xx - bx) * (xx - bx) + (yy - by) * (yy - by)));\n"
           "                    gv = (float)gray[(size_t)yy * (size_t)gw + (size_t)xx];\n"
           "                    if (d2 < 0.6f * rr) {\n                        in_sum += gv;\n                        in_n++;\n"
           "                    } else if (d2 > 1.3f * rr && d2 < 1.8f * rr) {\n                        out_sum += gv;\n                        out_n++;\n                    }\n"
           "                }\n            }\n"
           "            if (in_n < 3 || out_n < 3 || in_sum / (float)in_n > 0.92f * (out_sum / (float)out_n)) {\n"
           "                /* not a dark disc: forget this peak but keep looking */\n"
           "                int sr2 = 2 * (kHR0 + br), r2;\n"
           "                for (r2 = 0; r2 < kHNR; r2++) {\n"
           "                    float *o2 = sm + (size_t)r2 * (size_t)gh * (size_t)gw;\n"
           "                    for (yy = by - sr2; yy <= by + sr2; yy++) {\n"
           "                        for (xx = bx - sr2; xx <= bx + sr2; xx++) {\n"
           "                            if (xx >= 0 && xx < gw && yy >= 0 && yy < gh) {\n                                o2[(size_t)yy * (size_t)gw + (size_t)xx] = 0.f;\n                            }\n"
           "                        }\n                    }\n                }\n"
           "                continue;\n            }\n        }\n") + old_rec
assert old_rec in s
s = s.replace(old_rec, new_rec, 1)
open(path, "w", encoding="utf-8").write(s)
print("patched", path)

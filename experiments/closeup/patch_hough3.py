"""Hough iris finder, round 3: equal votes per edge, narrower radii, sideways prior."""
import shutil
import sys

path = sys.argv[1]
shutil.copyfile(path, path + ".hough2")
s = open(path, encoding="utf-8").read()
assert "GAZE_EYE_XMIN" not in s

# radii 5..10 small px (20..40 px in the 960 frame) matches irises seen on this mount
old = "enum { kHS = 4, kHR0 = 5, kHNR = 11 };"
new = "enum { kHS = 4, kHR0 = 5, kHNR = 6 };"
assert old in s
s = s.replace(old, new, 1)
s = s.replace("/* radii 5..15 small px = 20..60 px in the 960 frame */", "/* radii 5..10 small px = 20..40 px in the 960 frame */")

# every edge pixel votes with weight 1 (a strong purple rim must not outvote a soft iris edge)
old = "                    acc[((size_t)r * (size_t)gh + (size_t)iy) * (size_t)gw + (size_t)ix] += mag;"
new = "                    acc[((size_t)r * (size_t)gh + (size_t)iy) * (size_t)gw + (size_t)ix] += 1.f;"
assert old in s
s = s.replace(old, new, 1)
# a softer edge threshold now that weights are equal (about 4 gray levels per pixel of gradient)
old = "            if (mag < 48.f) {\n                continue;\n            }"
new = "            if (mag < 32.f) {\n                continue;\n            }"
assert old in s
s = s.replace(old, new, 1)

# sideways prior: skip the outer parts of the frame when picking peaks
old = "        int bx = 0, by = 0, br = 0;\n        for (r = 0; r < kHNR; r++) {"
new = ("        int bx = 0, by = 0, br = 0;\n"
       "        static float xmin_frac = -1.f, xmax_frac = -1.f;\n"
       "        int xlo, xhi;\n"
       "        if (xmin_frac < 0.f) {\n"
       "            const char *a = getenv(\"GAZE_EYE_XMIN\"), *b = getenv(\"GAZE_EYE_XMAX\");\n"
       "            xmin_frac = a ? (float)atof(a) : 0.12f;\n"
       "            xmax_frac = b ? (float)atof(b) : 0.88f;\n"
       "        }\n"
       "        xlo = (int)(xmin_frac * (float)gw);\n"
       "        xhi = (int)(xmax_frac * (float)gw);\n"
       "        for (r = 0; r < kHNR; r++) {")
assert old in s
s = s.replace(old, new, 1)
old = "            for (y = 0; y < ylim; y++) {\n                for (x = 0; x < gw; x++) {\n                    float v = o[(size_t)y * (size_t)gw + (size_t)x];"
new = "            for (y = 0; y < ylim; y++) {\n                for (x = xlo; x < xhi; x++) {\n                    float v = o[(size_t)y * (size_t)gw + (size_t)x];"
assert old in s
s = s.replace(old, new, 1)
open(path, "w", encoding="utf-8").write(s)
print("patched", path)

"""Round 6: the dark-disc check also requires (a) the disc to be dark in absolute terms and (b) not blue/purple-dominant
(the wearer's purple glasses frame produced strong round edges that beat the real irises)."""
import shutil
import sys

path = sys.argv[1]
shutil.copyfile(path, path + ".hough5")
s = open(path, encoding="utf-8").read()
assert "bmr" not in s

# dark_disc: add the colour + absolute darkness tests
old = "static int dark_disc(const uint8_t *gray, int gw, int gh, int bx, int by, float rr)\n{\n    float in_sum = 0.f, out_sum = 0.f;"
new = "static int dark_disc(const uint8_t *gray, const int16_t *bmr, int gw, int gh, int bx, int by, float rr)\n{\n    float in_sum = 0.f, out_sum = 0.f, in_bmr = 0.f;"
assert old in s
s = s.replace(old, new, 1)
old = "            if (d2 < 0.6f * rr) {\n                in_sum += gv;\n                in_n++;\n"
new = "            if (d2 < 0.6f * rr) {\n                in_sum += gv;\n                in_bmr += (float)bmr[(size_t)yy * (size_t)gw + (size_t)xx];\n                in_n++;\n"
assert old in s
s = s.replace(old, new, 1)
old = "    return in_n >= 3 && out_n >= 3 && in_sum / (float)in_n <= 0.92f * (out_sum / (float)out_n);\n}"
new = ("    if (in_n < 3 || out_n < 3) {\n        return 0;\n    }\n"
       "    {\n        static float max_gray = -1.f, max_bmr = 0.f;\n"
       "        if (max_gray < 0.f) {\n            const char *a = getenv(\"GAZE_MAX_IRIS_GRAY\"), *b = getenv(\"GAZE_MAX_BMR\");\n"
       "            max_gray = a ? (float)atof(a) : 105.f;\n            max_bmr = b ? (float)atof(b) : 14.f;\n        }\n"
       "        if (in_sum / (float)in_n > max_gray || in_bmr / (float)in_n > max_bmr) {\n            return 0;   /* too bright, or blue/purple like a plastic frame */\n        }\n    }\n"
       "    return in_sum / (float)in_n <= 0.92f * (out_sum / (float)out_n);\n}")
assert old in s
s = s.replace(old, new, 1)

# storage + fill of the blue-minus-red image
old = "    static uint8_t *gray;\n    static float *acc, *sm;\n    static size_t gcap, acap;\n    int gw = w / kHS, gh = h / kHS, x, y, r, i, j, np = 0, k, ylim, xlo, xhi;"
new = "    static uint8_t *gray;\n    static int16_t *bmr;\n    static float *acc, *sm;\n    static size_t gcap, acap;\n    int gw = w / kHS, gh = h / kHS, x, y, r, i, j, np = 0, k, ylim, xlo, xhi;"
assert old in s
s = s.replace(old, new, 1)
old = "        free(gray);\n        gray = (uint8_t *)malloc((size_t)gw * (size_t)gh);\n        gcap = gray ? (size_t)gw * (size_t)gh : 0;"
new = ("        free(gray);\n        free(bmr);\n        gray = (uint8_t *)malloc((size_t)gw * (size_t)gh);\n"
       "        bmr = (int16_t *)malloc((size_t)gw * (size_t)gh * sizeof(int16_t));\n"
       "        gcap = (gray && bmr) ? (size_t)gw * (size_t)gh : 0;")
assert old in s
s = s.replace(old, new, 1)
old = "    if (!gray || !acc || !sm || acap < nacc) {"
new = "    if (!gray || !bmr || !acc || !sm || acap < nacc || gcap < (size_t)gw * (size_t)gh) {"
assert old in s
s = s.replace(old, new, 1)
old = ("            int sum = 0, dx, dy;\n            for (dy = 0; dy < kHS; dy++) {\n                for (dx = 0; dx < kHS; dx++) {\n"
       "                    const uint8_t *p = rgb + ((size_t)(y * kHS + dy) * (size_t)w + (size_t)(x * kHS + dx)) * 3u;\n"
       "                    sum += (77 * p[0] + 150 * p[1] + 29 * p[2]) >> 8;\n                }\n            }\n"
       "            gray[(size_t)y * (size_t)gw + (size_t)x] = (uint8_t)(sum / (kHS * kHS));")
new = ("            int sum = 0, bsum = 0, dx, dy;\n            for (dy = 0; dy < kHS; dy++) {\n                for (dx = 0; dx < kHS; dx++) {\n"
       "                    const uint8_t *p = rgb + ((size_t)(y * kHS + dy) * (size_t)w + (size_t)(x * kHS + dx)) * 3u;\n"
       "                    sum += (77 * p[0] + 150 * p[1] + 29 * p[2]) >> 8;\n                    bsum += (int)p[2] - (int)p[0];\n                }\n            }\n"
       "            gray[(size_t)y * (size_t)gw + (size_t)x] = (uint8_t)(sum / (kHS * kHS));\n"
       "            bmr[(size_t)y * (size_t)gw + (size_t)x] = (int16_t)(bsum / (kHS * kHS));")
assert old in s
s = s.replace(old, new, 1)

# call sites of dark_disc
n = s.count("dark_disc(gray, gw, gh,")
assert n >= 1, n
s = s.replace("dark_disc(gray, gw, gh,", "dark_disc(gray, bmr, gw, gh,")
open(path, "w", encoding="utf-8").write(s)
print("patched", path, "call sites:", n)

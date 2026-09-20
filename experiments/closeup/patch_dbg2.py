"""Debug: log every iris candidate with its brightness / colour statistics (env GAZE_DBG=1)."""
import shutil
import sys

path = sys.argv[1]
shutil.copyfile(path, path + ".pre_dbg2")
s = open(path, encoding="utf-8").read()
assert "cand k=" not in s

old = "static int dark_disc(const uint8_t *gray, const int16_t *bmr, int gw, int gh, int bx, int by, float rr)\n{\n    float in_sum = 0.f, out_sum = 0.f, in_bmr = 0.f;"
new = "static float g_dd[3];   /* last candidate's inner gray, inner blue-minus-red, ring gray (debug) */\n\n" + old
assert old in s
s = s.replace(old, new, 1)

old = "    if (in_n < 3 || out_n < 3) {\n        return 0;\n    }\n    {\n        static float max_gray = -1.f, max_bmr = 0.f;"
new = ("    if (in_n < 3 || out_n < 3) {\n        return 0;\n    }\n"
       "    g_dd[0] = in_sum / (float)in_n;\n    g_dd[1] = in_bmr / (float)in_n;\n    g_dd[2] = out_sum / (float)out_n;\n"
       "    {\n        static float max_gray = -1.f, max_bmr = 0.f;")
assert old in s
s = s.replace(old, new, 1)

old = "        if (!dark_disc(gray, bmr, gw, gh, bx, by, (float)(kHR0 + br))) {\n            suppress_around(sm, gw, gh, bx, by, 2 * (kHR0 + br));\n            continue;\n        }"
new = ("        {\n            int pass;\n            g_dd[0] = g_dd[1] = g_dd[2] = -1.f;\n"
       "            pass = dark_disc(gray, bmr, gw, gh, bx, by, (float)(kHR0 + br));\n"
       "            if (getenv(\"GAZE_DBG\")) {\n                static int dl;\n                if (dl < 900) {\n                    fprintf(stderr, \"gazecomp: cand k=%d x=%.0f y=%.0f r=%.0f s=%.0f ig=%.0f ib=%.0f og=%.0f pass=%d\\n\", k,\n"
       "                            ((float)bx + 0.5f) * (float)kHS, ((float)by + 0.5f) * (float)kHS, (float)(kHR0 + br) * (float)kHS, best,\n"
       "                            g_dd[0], g_dd[1], g_dd[2], pass);\n                    dl++;\n                }\n            }\n"
       "            if (!pass) {\n                suppress_around(sm, gw, gh, bx, by, 2 * (kHR0 + br));\n                continue;\n            }\n        }")
assert old in s
s = s.replace(old, new, 1)
open(path, "w", encoding="utf-8").write(s)
print("patched", path)

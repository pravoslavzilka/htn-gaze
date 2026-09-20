"""Round 7: thresholds taken from measured candidate statistics (irises: dark, contrasty, neutral colour; the nose-bridge
shadow and skin: brighter, low contrast, strongly red)."""
import shutil
import sys

path = sys.argv[1]
shutil.copyfile(path, path + ".hough6")
s = open(path, encoding="utf-8").read()
assert "GAZE_MIN_BMR" not in s

old = ("        static float max_gray = -1.f, max_bmr = 0.f;\n"
       "        if (max_gray < 0.f) {\n"
       "            const char *a = getenv(\"GAZE_MAX_IRIS_GRAY\"), *b = getenv(\"GAZE_MAX_BMR\");\n"
       "            max_gray = a ? (float)atof(a) : 105.f;\n            max_bmr = b ? (float)atof(b) : 14.f;\n        }\n"
       "        if (in_sum / (float)in_n > max_gray || in_bmr / (float)in_n > max_bmr) {\n"
       "            return 0;   /* too bright, or blue/purple like a plastic frame */\n        }\n    }\n"
       "    return in_sum / (float)in_n <= 0.92f * (out_sum / (float)out_n);\n}")
new = ("        static float max_gray = -1.f, max_bmr = 0.f, min_bmr = 0.f, ratio = 0.f;\n"
       "        if (max_gray < 0.f) {\n"
       "            const char *a = getenv(\"GAZE_MAX_IRIS_GRAY\"), *b = getenv(\"GAZE_MAX_BMR\"), *c = getenv(\"GAZE_MIN_BMR\"),\n"
       "                       *d = getenv(\"GAZE_MAX_RATIO\");\n"
       "            max_gray = a ? (float)atof(a) : 85.f;\n            max_bmr = b ? (float)atof(b) : 14.f;\n"
       "            min_bmr = c ? (float)atof(c) : -26.f;\n            ratio = d ? (float)atof(d) : 0.78f;\n        }\n"
       "        if (in_sum / (float)in_n > max_gray || in_bmr / (float)in_n > max_bmr || in_bmr / (float)in_n < min_bmr) {\n"
       "            return 0;   /* too bright, blue/purple like a plastic frame, or skin-red like the nose bridge */\n        }\n"
       "        return in_sum / (float)in_n <= ratio * (out_sum / (float)out_n);\n    }\n}")
assert old in s
s = s.replace(old, new, 1)
open(path, "w", encoding="utf-8").write(s)
print("patched", path)

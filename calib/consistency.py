import json, sys, time, urllib.request, numpy as np
base, secs = sys.argv[1], float(sys.argv[2]); W, H = 960, 540
rows = []; t0 = time.time(); last = None
while time.time() - t0 < secs:
    try: s = json.loads(urllib.request.urlopen(base + "/api/state", timeout=3).read())
    except Exception: time.sleep(0.1); continue
    if s["frame_id"] == last: time.sleep(0.02); continue
    last = s["frame_id"]; e = s.get("eyes") or {}
    rows.append((s["n"], s["infer_fps"], e.get("left", {}).get("iris"), e.get("right", {}).get("iris")))
    time.sleep(0.05)
n = len(rows); both = [r for r in rows if r[0] == 2 and r[2] and r[3]]
print(f"frames {n}, both eyes reported {100*len(both)/max(n,1):.0f}%, face model {np.median([r[1] for r in rows]):.1f} fps")
if len(both) > 10:
    L = np.array([[r[2][0]*W, r[2][1]*H] for r in both]); R = np.array([[r[3][0]*W, r[3][1]*H] for r in both])
    mL, mR = np.median(L, 0), np.median(R, 0)
    print("left iris  median (%.0f,%.0f) std x %.1f y %.1f" % (*mL, *L.std(0)))
    print("right iris median (%.0f,%.0f) std x %.1f y %.1f" % (*mR, *R.std(0)))
    near = (np.linalg.norm(L-mL, axis=1) < 40) & (np.linalg.norm(R-mR, axis=1) < 40)
    print("frames with BOTH irises within 40 px of their median: %.0f%% of reported frames = %.0f%% of all frames" % (100*near.mean(), 100*near.sum()/n))

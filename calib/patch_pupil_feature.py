"""One-off patch: add the pupil-in-eye branch's dark-pupil measurement as extra feature kinds.
Applied once; safe to delete afterwards."""
import re

# ---------------- features.py
p = "features.py"
s = open(p, encoding="utf-8").read()
if "pupil_avg" not in s:
    old = "    return m\n\n\nclass EyeTracker:"
    new = ('    # dark-pupil fit done on the board by the pupil-in-eye branch (absent on the older build)\n'
           '    if "pupil" in e:\n'
           '        m["pupil_ok"] = bool(e.get("pupil_ok", 0))\n'
           '        m["pupil"] = np.array([e["pupil"][0] * vw, e["pupil"][1] * vh])\n'
           '        m["pupil_r"] = float(e.get("pupil_r", 0.0) * vw)\n'
           '    else:\n'
           '        m["pupil_ok"] = False\n'
           '        m["pupil"] = iris_mp.copy()\n'
           '        m["pupil_r"] = 0.0\n'
           '    return m\n\n\nclass EyeTracker:')
    assert old in s
    s = s.replace(old, new, 1)
    old = '            m["off_mp"] = (m["iris_mp"] - ref_c) / ref_w        # raw MediaPipe iris vs the same reference\n'
    assert old in s
    s = s.replace(old, old + '            m["off_pupil"] = (m["pupil"] - ref_c) / ref_w       # board-side dark pupil vs the same reference\n', 1)
    old = 'KINDS = ("ref_avg", "ref_lr", "ref_open", "mp_avg")'
    assert old in s
    s = s.replace(old, 'KINDS = ("ref_avg", "ref_lr", "ref_open", "mp_avg", "pupil_avg", "pupil_lr")\n\n\n'
                       'def usable(kind, res):\n'
                       '    """A sample is usable for a feature kind if the eyes are not blinking and, for the pupil kinds, the\n'
                       '    board-side pupil fit succeeded on both eyes."""\n'
                       '    if not res["ok"]:\n'
                       '        return False\n'
                       '    return all(e.get("pupil_ok") for e in res["eyes"]) if kind.startswith("pupil") else True')
    old = '    if kind == "mp_avg":\n        return (L["off_mp"] + R["off_mp"]) / 2\n'
    assert old in s
    s = s.replace(old, old + '    if kind == "pupil_avg":\n        return (L["off_pupil"] + R["off_pupil"]) / 2\n'
                             '    if kind == "pupil_lr":\n        return np.concatenate([L["off_pupil"], R["off_pupil"]])\n', 1)
    open(p, "w", encoding="utf-8").write(s)
    print("features.py patched")

# ---------------- fitlib.py
p = "fitlib.py"
s = open(p, encoding="utf-8").read()
if "unflat, usable" not in s:
    s = s.replace("from features import KINDS, feature_vector, unflat", "from features import KINDS, feature_vector, unflat, usable")
    old = ('    for s in rec["samples"]:\n'
           '        if s["det"] is None or not s["res"]["ok"]:\n'
           '            continue\n'
           '        F.append(feature_vector(kind, unflat(s["res"])))\n')
    new = ('    for s in rec["samples"]:\n'
           '        res = unflat(s["res"])\n'
           '        if s["det"] is None or not usable(kind, res):\n'
           '            continue\n'
           '        F.append(feature_vector(kind, res))\n')
    assert old in s
    s = s.replace(old, new, 1)
    old = '    cal = [r for r in recs if r["kind"] == "cal" and r.get("scene_xy") and r["n_ok"] >= 5]\n    val = [r for r in recs if r["kind"] == "val" and r.get("scene_xy") and r["n_ok"] >= 5]\n    if len(cal) < 8:\n        return None\n'
    new = ('    cal = [r for r in recs if r["kind"] == "cal" and r.get("scene_xy") and len(_samples(r, kind)[0]) >= 5]\n'
           '    val = [r for r in recs if r["kind"] == "val" and r.get("scene_xy") and len(_samples(r, kind)[0]) >= 5]\n'
           '    if len(cal) < 8:\n        return {"kind": kind, "degree": degree, "skipped": True, "n_cal": len(cal), "n_val": len(val)}\n')
    assert old in s
    s = s.replace(old, new, 1)
    old = ('            if degree == 2 and kind == "ref_lr":\n                continue\n'
           '            r = evaluate(recs, kind, degree)\n'
           '            if r is None:\n'
           '                log("  not enough good squares to fit.")\n'
           '                return None\n')
    new = ('            if degree == 2 and kind in ("ref_lr", "pupil_lr"):\n                continue\n'
           '            r = evaluate(recs, kind, degree)\n'
           '            if r.get("skipped"):\n'
           '                log(f"  {kind:10s} {degree:>3d}  skipped: only {r[\'n_cal\']} calibration squares have usable samples for this feature")\n'
           '                continue\n')
    assert old in s
    s = s.replace(old, new, 1)
    old = '    return best\n\n\ndef save_model'
    assert old in s
    open(p, "w", encoding="utf-8").write(s)
    print("fitlib.py patched")

# ---------------- calibrate.py: show how many samples had a board-side pupil fit
p = "calibrate.py"
s = open(p, encoding="utf-8").read()
if "pupil fit" not in s:
    old = '        n_ok = sum(1 for s in samples if s["res"]["ok"] and s["det"] is not None)\n'
    assert old in s
    s = s.replace(old, old + '        n_pup = sum(1 for s in samples if s["res"]["ok"] and s["det"] is not None and all(e.get("pupil_ok") for e in s["res"]["eyes"]))\n', 1)
    old = '              f"  (eyes ok {n_eye_ok}, square seen {len(dets)})  {status}", flush=True)'
    assert old in s
    s = s.replace(old, '              f"  (eyes ok {n_eye_ok}, pupil fit {n_pup}, square seen {len(dets)})  {status}", flush=True)', 1)
    s = s.replace('"n_eye_ok": n_eye_ok, "n_ok": n_ok,', '"n_eye_ok": n_eye_ok, "n_ok": n_ok, "n_pupil": n_pup,', 1)
    open(p, "w", encoding="utf-8").write(s)
    print("calibrate.py patched")

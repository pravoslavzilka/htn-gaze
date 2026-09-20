"""Score the geometric gaze model against calibration squares recorded by collect_geom.py.

  python eval_geom.py [run_dir]

For every square the eyes' landmarks are reduced to a per-square median state; the model's predicted scene pixel is compared
with where the square really appeared in the scene camera. Compared on the held-out ("val") squares:
  1. the model exactly as shipped (no calibration)
  2. best axis flips only
  3. + a look-at-the-centre offset (what the GUI's Z key measures; 2 numbers)
  4. + fitted rig parameters (yaw/pitch offsets, reticle depth, eye distance, kappa) - fitted on the "cal" squares
  5. for reference: our polynomial calibration on the iris-offset features
"""
import glob
import itertools
import json
import math
import os
import sys
from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares

from gaze_geom import Rig, gaze_from_state, lid_open

PPD = 960.0 / 66.0


def median_state(samples, need_both=True):
    """One synthetic state whose landmarks are the per-coordinate medians over the usable samples of a square."""
    use = [s["state"] for s in samples if s["state"]["n"] == 2 and all(k in s["state"]["eyes"] for k in ("left", "right"))]
    use = [s for s in use if all(lid_open(s["eyes"][k]) >= 0.08 for k in ("left", "right"))]
    if len(use) < 5:
        return None, len(use)
    out = {"width": use[0]["width"], "height": use[0]["height"], "n": 2, "eyes": {}}
    for side in ("left", "right"):
        out["eyes"][side] = {k: np.median(np.array([u["eyes"][side][k] for u in use]), axis=0).tolist()
                             for k in ("center", "iris", "lid")}
    return out, len(use)


def load(run_dir):
    recs = json.load(open(os.path.join(run_dir, "geom_samples.json")))
    sq = []
    for r in recs:
        if "scene_xy" not in r:
            continue
        st, n = median_state(r["samples"])
        if st is None:
            continue
        sq.append({"kind": r["kind"], "screen": r["screen"], "xy": np.array(r["scene_xy"]), "state": st, "n": n})
    return sq


def predict(sq, rig):
    out = []
    for s in sq:
        g = gaze_from_state(s["state"], rig)
        out.append([np.nan, np.nan] if g is None else [g[0], g[1]])
    return np.array(out)


def errors(pred, truth):
    d = pred - truth
    ok = ~np.isnan(d).any(axis=1)
    e = np.linalg.norm(d[ok], axis=1)
    return {"n": int(ok.sum()), "mean": float(e.mean()) if ok.any() else float("nan"),
            "median": float(np.median(e)) if ok.any() else float("nan"),
            "bias": d[ok].mean(axis=0) if ok.any() else np.array([np.nan, np.nan])}


def fit_params(cal, rig0, names, bounds):
    truth = np.array([s["xy"] for s in cal])

    def resid(v):
        rig = replace(rig0, **dict(zip(names, v)))
        p = predict(cal, rig)
        r = (p - truth)
        r[np.isnan(r)] = 400.0
        return r.reshape(-1)

    x0 = np.array([getattr(rig0, n) for n in names], dtype=float)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    res = least_squares(resid, x0, bounds=(lo, hi), loss="soft_l1", f_scale=30.0, x_scale=np.maximum(np.abs(x0), 1.0) * 0.2 + 1.0)
    return replace(rig0, **dict(zip(names, res.x))), res


def poly_features(state):
    f = []
    for side in ("left", "right"):
        e = state["eyes"][side]
        lid = np.array(e["lid"]).reshape(-1, 2)
        w = max(lid[:, 0].max() - lid[:, 0].min(), 1e-6)
        f.append([(e["iris"][0] - e["center"][0]) / w, (e["iris"][1] - e["center"][1]) / w])
    return np.mean(f, axis=0)


def poly_fit_eval(cal, val, degree):
    def design(F):
        F = np.atleast_2d(F)
        cols = [np.ones(len(F)), F[:, 0], F[:, 1]]
        if degree == 2:
            cols += [F[:, 0] ** 2, F[:, 0] * F[:, 1], F[:, 1] ** 2]
        return np.stack(cols, axis=1)
    Fc = np.array([poly_features(s["state"]) for s in cal]); Yc = np.array([s["xy"] for s in cal])
    mu, sd = Fc.mean(0), Fc.std(0) + 1e-9
    A = design((Fc - mu) / sd); reg = 1e-3 * np.eye(A.shape[1]); reg[0, 0] = 0
    W = np.linalg.solve(A.T @ A + reg, A.T @ Yc)
    Fv = np.array([poly_features(s["state"]) for s in val]); Yv = np.array([s["xy"] for s in val])
    return errors(design((Fv - mu) / sd) @ W, Yv), errors(A @ W, Yc)


def _best_flips(train):
    Y = np.array([s["xy"] for s in train]); best = None
    for fx, fy in itertools.product((False, True), repeat=2):
        r = replace(Rig(), flipX=fx, flipY=fy); e = errors(predict(train, r), Y)
        if best is None or e["mean"] < best[0]:
            best = (e["mean"], r)
    return best[1]


def loo_eval(sq):
    """Leave-one-out: each square is scored by a model fitted on all the others."""
    names = ["yaw0Deg", "pitch0Deg", "sceneDepthMm", "eyeDistanceMm", "kappaDeg"]
    bounds = [(-40, 40), (-40, 40), (250, 4000), (30, 250), (-12, 12)]
    rows = {"as shipped": [], "best flips": [], "centre offset": [], "fitted rig": [], "poly deg 1": [], "poly deg 2": []}
    fitted = []
    for i, held in enumerate(sq):
        train = [s for j, s in enumerate(sq) if j != i]
        truth = held["xy"]
        rows["as shipped"].append(predict([held], Rig())[0] - truth)
        rf = _best_flips(train); rows["best flips"].append(predict([held], rf)[0] - truth)
        r3, _ = fit_params(train, rf, ["yaw0Deg", "pitch0Deg"], [(-40, 40), (-40, 40)])
        rows["centre offset"].append(predict([held], r3)[0] - truth)
        r4, _ = fit_params(train, r3, names, bounds); fitted.append([getattr(r4, n) for n in names])
        rows["fitted rig"].append(predict([held], r4)[0] - truth)
        for deg in (1, 2):
            ev, _ = poly_fit_eval(train, [held], deg)
            # poly_fit_eval returns errors only; recompute the signed residual for the bias
            rows[f"poly deg {deg}"].append(np.array([np.nan, np.nan]) if ev["n"] == 0 else _poly_resid(train, held, deg))
    return rows, np.array(fitted), names


def _poly_resid(train, held, degree):
    def design(F):
        F = np.atleast_2d(F); cols = [np.ones(len(F)), F[:, 0], F[:, 1]]
        if degree == 2:
            cols += [F[:, 0] ** 2, F[:, 0] * F[:, 1], F[:, 1] ** 2]
        return np.stack(cols, axis=1)
    Ft = np.array([poly_features(s["state"]) for s in train]); Yt = np.array([s["xy"] for s in train])
    mu, sd = Ft.mean(0), Ft.std(0) + 1e-9
    A = design((Ft - mu) / sd); reg = 1e-3 * np.eye(A.shape[1]); reg[0, 0] = 0
    W = np.linalg.solve(A.T @ A + reg, A.T @ Yt)
    return (design((poly_features(held["state"]) - mu) / sd) @ W)[0] - held["xy"]


def report_loo(sq):
    print(f"Only {len(sq)} squares had both usable eyes and a detected square, too few for a separate held-out set.")
    print("Scoring by leave-one-out: each square is predicted by a model fitted on the other squares.\n")
    rows, fitted, names = loo_eval(sq)
    for k, v in rows.items():
        d = np.array(v); e = np.linalg.norm(d, axis=1)
        print(f"  {k:14s} {np.nanmean(e):6.1f} px = {np.nanmean(e) / PPD:4.1f} deg   (median {np.nanmedian(e):5.1f} px, bias x {np.nanmean(d[:,0]):+.0f} y {np.nanmean(d[:,1]):+.0f}, n={int((~np.isnan(e)).sum())})")
    print("\n  fitted rig parameters across the leave-one-out fits (median [min..max]):")
    for j, n in enumerate(names):
        print(f"     {n:14s} {np.median(fitted[:, j]):8.1f}  [{fitted[:, j].min():.1f} .. {fitted[:, j].max():.1f}]")
    e4 = np.linalg.norm(np.array(rows["fitted rig"]), axis=1)
    print("\n  per square (screen x,y -> error of the fitted-rig model):")
    for s, e in zip(sq, e4):
        print(f"     ({s['screen'][0]:.2f},{s['screen'][1]:.2f}) samples {s['n']:>2d}  error {e:5.1f} px")


def fmt(e):
    return f"{e['mean']:6.1f} px = {e['mean'] / PPD:4.1f} deg   (median {e['median']:5.1f} px, n={e['n']}, bias x {e['bias'][0]:+.0f} y {e['bias'][1]:+.0f})"


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_*", "geom_samples.json")))[-1].rsplit(os.sep, 1)[0]
    sq = load(run)
    cal = [s for s in sq if s["kind"] == "cal"]; val = [s for s in sq if s["kind"] == "val"]
    print(f"run: {run}\nusable squares: {len(cal)} calibration, {len(val)} held-out (need >= 10 / >= 4)")
    if len(cal) + len(val) < 10:
        print("not enough usable squares")
        return
    if len(cal) < 10 or len(val) < 4:
        report_loo(cal + val)
        return
    Yc = np.array([s["xy"] for s in cal]); Yv = np.array([s["xy"] for s in val])
    print(f"median samples per square: {int(np.median([s['n'] for s in sq]))}")
    print("\nHeld-out squares (never used in any fit):")

    r0 = Rig()
    print("  1. geometry as shipped, no calibration        ", fmt(errors(predict(val, r0), Yv)))

    best = None
    for fx, fy in itertools.product((False, True), repeat=2):
        r = replace(r0, flipX=fx, flipY=fy)
        e = errors(predict(cal, r), Yc)
        if best is None or e["mean"] < best[0]:
            best = (e["mean"], r)
    rf = best[1]
    print(f"  2. + best axis flips (flipX={rf.flipX}, flipY={rf.flipY})".ljust(50), fmt(errors(predict(val, rf), Yv)))

    r3, _ = fit_params(cal, rf, ["yaw0Deg", "pitch0Deg"], [(-40, 40), (-40, 40)])
    print(f"  3. + centre offset (yaw {r3.yaw0Deg:+.1f}, pitch {r3.pitch0Deg:+.1f} deg)".ljust(50), fmt(errors(predict(val, r3), Yv)))

    names = ["yaw0Deg", "pitch0Deg", "sceneDepthMm", "eyeDistanceMm", "kappaDeg"]
    bounds = [(-40, 40), (-40, 40), (250, 4000), (30, 250), (-12, 12)]
    r4, res = fit_params(cal, r3, names, bounds)
    e4v = errors(predict(val, r4), Yv); e4c = errors(predict(cal, r4), Yc)
    print("  4. + fitted rig parameters".ljust(50), fmt(e4v))
    print("     fitted: " + ", ".join(f"{n}={getattr(r4, n):.1f}" for n in names) + f"   (fit error on the calibration squares: {e4c['mean']:.1f} px)")

    for deg in (1, 2):
        ev, ec = poly_fit_eval(cal, val, deg)
        print(f"  5. reference: polynomial calibration degree {deg}".ljust(50), fmt(ev))

    print("\nPer held-out square, error in px (model 3 = centre offset | model 4 = fitted):")
    p3, p4 = predict(val, r3), predict(val, r4)
    for i, s in enumerate(val):
        print(f"   screen ({s['screen'][0]:.2f},{s['screen'][1]:.2f})  truth ({s['xy'][0]:5.0f},{s['xy'][1]:5.0f})  "
              f"m3 ({p3[i][0]:5.0f},{p3[i][1]:5.0f}) err {np.linalg.norm(p3[i] - s['xy']):5.1f} | m4 ({p4[i][0]:5.0f},{p4[i][1]:5.0f}) err {np.linalg.norm(p4[i] - s['xy']):5.1f}")
    json.dump({"rig_fitted": {n: getattr(r4, n) for n in names + ["flipX", "flipY"]}}, open(os.path.join(run, "geom_fit.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

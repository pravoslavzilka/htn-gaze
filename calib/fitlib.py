"""Fit pupil features -> scene-camera pixel.

Improvements over the first version:
  * fits on every recorded sample (hundreds of points) instead of one median per square,
  * leaves out blinks / failed iris fits (flagged by features.EyeTracker),
  * robust regression (Huber, iteratively re-weighted) so a few bad samples cannot bend the fit,
  * chooses the feature set / polynomial degree by leave-one-square-out cross-validation.
"""
import json

import numpy as np

from features import KINDS, feature_vector, unflat, usable


def design(z, degree):
    z = np.atleast_2d(z)
    cols = [np.ones(len(z))] + [z[:, i] for i in range(z.shape[1])]
    if degree == 2:
        d = z.shape[1]
        cols += [z[:, i] * z[:, j] for i in range(d) for j in range(i, d)]
    return np.stack(cols, axis=1)


def fit(F, Y, degree, lam=1e-2, iters=8):
    """Huber-weighted ridge regression. F: (n, d) features, Y: (n, 2) targets."""
    mu, sd = F.mean(0), F.std(0) + 1e-9
    A = design((F - mu) / sd, degree)
    reg = lam * np.eye(A.shape[1])
    reg[0, 0] = 0
    w = np.ones(len(A))
    for _ in range(iters):
        Aw = A * w[:, None]
        W = np.linalg.solve(A.T @ Aw + reg, Aw.T @ Y)
        e = np.linalg.norm(Y - A @ W, axis=1)
        scale = 1.4826 * np.median(e) + 1e-6
        c = 1.345 * scale
        w = np.where(e <= c, 1.0, c / np.maximum(e, 1e-9))
    return {"mu": mu, "sd": sd, "W": W, "degree": degree}


def predict(m, F):
    return design((np.atleast_2d(F) - m["mu"]) / m["sd"], m["degree"]) @ m["W"]


def _samples(rec, kind):
    """Feature matrix and per-sample targets for the usable samples of one square."""
    F, Y = [], []
    for s in rec["samples"]:
        res = unflat(s["res"])
        if s["det"] is None or not usable(kind, res):
            continue
        F.append(feature_vector(kind, res))
        Y.append(s["det"])
    if not F:
        return np.zeros((0, 1)), np.zeros((0, 2))
    return np.array(F).reshape(len(F), -1), np.array(Y).reshape(len(Y), 2)


def _square_error(m, kind, rec):
    F, _ = _samples(rec, kind)
    if len(F) < 3:
        return None
    p = np.median(predict(m, F), axis=0)
    return float(np.linalg.norm(p - np.array(rec["scene_xy"])))


def evaluate(recs, kind, degree):
    cal = [r for r in recs if r["kind"] == "cal" and r.get("scene_xy") and len(_samples(r, kind)[0]) >= 5]
    val = [r for r in recs if r["kind"] == "val" and r.get("scene_xy") and len(_samples(r, kind)[0]) >= 5]
    if len(cal) < 8:
        return {"kind": kind, "degree": degree, "skipped": True, "n_cal": len(cal), "n_val": len(val)}
    data = [_samples(r, kind) for r in cal]
    loo = []
    for i, r in enumerate(cal):
        keep = [j for j in range(len(cal)) if j != i]
        F = np.concatenate([data[j][0] for j in keep])
        Y = np.concatenate([data[j][1] for j in keep])
        e = _square_error(fit(F, Y, degree), kind, r)
        if e is not None:
            loo.append(e)
    model = fit(np.concatenate([d[0] for d in data]), np.concatenate([d[1] for d in data]), degree)
    ve = [e for e in (_square_error(model, kind, r) for r in val) if e is not None]
    return {"kind": kind, "degree": degree, "loo": float(np.mean(loo)), "val": float(np.mean(ve)) if ve else None,
            "model": model, "n_cal": len(cal), "n_val": len(val),
            "n_samples": int(sum(len(d[0]) for d in data))}


def select_model(recs, ppd=14.5, log=print):
    """Try every feature set and degree; return the best result dict (or None)."""
    log(f"\nModel comparison (mean error in scene-camera pixels; ~{ppd:.0f} px = 1 degree):")
    log(f"  {'features':10s} deg  {'samples':>7s}  {'leave-one-square-out':>21s}  {'held-out squares':>18s}")
    best = None
    for kind in KINDS:
        for degree in (1, 2):
            if degree == 2 and kind in ("ref_lr", "pupil_lr"):
                continue
            r = evaluate(recs, kind, degree)
            if r.get("skipped"):
                log(f"  {kind:10s} {degree:>3d}  skipped: only {r['n_cal']} calibration squares have usable samples for this feature")
                continue
            v = "-" if r["val"] is None else f"{r['val']:.1f} px / {r['val'] / ppd:.1f} deg"
            log(f"  {kind:10s} {degree:>3d}  {r['n_samples']:>7d}  {r['loo']:>12.1f} px        {v:>18s}")
            score = r["val"] if r["val"] is not None else r["loo"]
            if best is None or score < best[0]:
                best = (score, r)
    return best


def save_model(path, best):
    score, r = best
    m = r["model"]
    with open(path, "w") as f:
        json.dump({"version": 2, "features": r["kind"], "degree": r["degree"], "mu": m["mu"].tolist(),
                   "sd": m["sd"].tolist(), "W": m["W"].tolist(), "scene_size": [960, 540], "error_px": score}, f, indent=1)


def load_model(path):
    m = json.load(open(path))
    if m.get("version") != 2:
        raise ValueError(f"{path} is an old-format model; run the calibration again")
    return {"kind": m["features"], "degree": m["degree"], "mu": np.array(m["mu"]), "sd": np.array(m["sd"]),
            "W": np.array(m["W"]), "err": float(m.get("error_px", 50.0))}

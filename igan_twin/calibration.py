"""Estimate population parameters against published trial read-outs.

The objective stacks weighted residuals for every :class:`~igan_twin.trials.Target`
of every trial with ``calibrate=True`` plus a weak prior pulling each
parameter towards its literature default (SD 1.5 on the transformed scale),
which keeps weakly identified parameters finite. ``scipy.optimize.least_squares``
returns the fit and a Gauss-Newton covariance that is used for parameter
uncertainty.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares

from .parameters import CALIBRATED, CALIBRATED_PATH, ModelParams
from .trials import TRIALS, Target, run_trial

PRIOR_SD = 1.5


def _to_x(params: ModelParams, names) -> np.ndarray:
    x = []
    for k in names:
        tr, _, _ = CALIBRATED[k]
        v = getattr(params, k)
        x.append(np.log(v) if tr == "log" else np.log(v / (1 - v)))
    return np.array(x)


def _from_x(x, base: ModelParams, names) -> ModelParams:
    vals = {}
    for xi, k in zip(x, names):
        tr, lo, hi = CALIBRATED[k]
        v = float(np.exp(xi)) if tr == "log" else float(1 / (1 + np.exp(-xi)))
        vals[k] = float(np.clip(v, lo, hi))
    return replace(base, **vals)


def _bounds(names):
    lo, hi = [], []
    for k in names:
        tr, a, b = CALIBRATED[k]
        f = np.log if tr == "log" else (lambda v: np.log(v / (1 - v)))
        lo.append(f(a))
        hi.append(f(b))
    return np.array(lo), np.array(hi)


def residual(tgt: Target, pred: float) -> float:
    if tgt.kind in ("upcr_ratio", "biomarker_ratio"):
        return (np.log(pred) - np.log(tgt.value)) / tgt.sd
    return (pred - tgt.value) / tgt.sd


def evaluate(params: ModelParams, trials=None, n: int = 80, seed: int = 1):
    """Return list of (trial, target, predicted, residual)."""
    trials = trials or [k for k, t in TRIALS.items() if t.calibrate]
    rows = []
    for key in trials:
        res = run_trial(TRIALS[key], params, n=n, seed=seed, record_every=TRIALS[key].primary_day / 4)
        for tgt in TRIALS[key].targets:
            pred = res.predicted(tgt)
            rows.append((key, tgt, pred, residual(tgt, pred)))
    return rows


def calibrate(base: ModelParams | None = None, n: int = 80, seed: int = 1, max_nfev: int = 400, verbose: bool = True):
    base = base or ModelParams()
    names = list(CALIBRATED)
    x0 = _to_x(base, names)
    lo, hi = _bounds(names)
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
    t0 = time.time()
    calls = {"n": 0}

    def fun(x):
        p = _from_x(x, base, names)
        r = [row[3] for row in evaluate(p, n=n, seed=seed)]
        prior = (x - _to_x(base, names)) / PRIOR_SD
        calls["n"] += 1
        out = np.concatenate([r, prior])
        if verbose and calls["n"] % 13 == 1:
            print(f"  eval {calls['n']:4d}  SSE={np.sum(np.square(r)):8.3f}  t={time.time() - t0:6.1f}s", flush=True)
        return out

    sol = least_squares(fun, x0, bounds=(lo, hi), diff_step=1e-3, max_nfev=max_nfev, x_scale="jac")
    fitted = _from_x(sol.x, base, names)
    # Gauss-Newton covariance on transformed scale
    J = sol.jac
    try:
        cov = np.linalg.pinv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = np.full((len(names), len(names)), np.nan)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return fitted, {"names": names, "x": sol.x.tolist(), "se_transformed": se.tolist(), "cov": cov.tolist(),
                    "cost": float(sol.cost), "nfev": int(sol.nfev), "status": int(sol.status)}


def summarise(params: ModelParams, n: int = 150, seed: int = 7):
    """Fit table on a fresh, larger virtual population (out-of-sample seed)."""
    table = []
    for key, tgt, pred, r in evaluate(params, n=n, seed=seed):
        table.append({"trial": key, "kind": tgt.kind, "arm": tgt.arm, "day": tgt.day, "observed": tgt.value,
                      "predicted": pred, "z": r, "note": tgt.note})
    return table


def save(params: ModelParams, info: dict, table: list, path=CALIBRATED_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"params": params.calibrated_dict(), "fit": info, "targets": table}
    path.write_text(json.dumps(payload, indent=2))
    return path

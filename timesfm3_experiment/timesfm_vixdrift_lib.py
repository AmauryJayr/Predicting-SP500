"""The VIX on top of the best drift layer (follow-up to timesfm_vix_lib.py, which put the VIX on the raw score).

The baseline D0 is the drift layer chosen per horizon in the drift study (timesfm_drift_choices.json):
  h=1  stack-4: logistic on [logit score z, median forecast return, q90-q10 spread], C=0.03, fit once
  h=5  platt-6: offset-only recalibration of z, last 252 known labels, refit every 21 rows

Four variants change D0 only through the VIX (all choices inherited from the earlier studies, nothing new is tuned):
  D1  VIX-triggered refit   refit when 1[VIX rank >= cut] flips (cut and min gap of the V1 study)
  D2  VIX-regime selection  experts {D0, team LR, always-up}; lowest Brier on past resolved same-regime rows
  D3  VIX terms             VIX rank and z x (rank - 0.5) added to the layer
  D4  VIX removed           D0 applied to the forecasts made without the ^VIX covariate

Every frame is [y, proba, thr, pred] indexed by date. Labels are used only if label_end <= decision date
(`Inp.train_rows`, asserted).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit

import timesfm_drift_lib as dl
import timesfm_lib as tl
import timesfm_vix_lib as vl
from common import fit_model, make_estimator

NOVIX_CACHE = tl.RES_DIR / "timesfm_vix_cache" / "novix"
RIDGE = 1.0                         # D3 h=5: L2 weight on the two VIX coefficients (as C=1 of dl.fit_cal "platt")
WF_EVERY = 63                       # walk-forward refit schedule of the D0 expert history (as the V2 study)


def make_inp(h, upto=None, novix=False):
    """`dl.Inp`, optionally on the no-VIX forecast cache (the cache is only needed while the object is built)."""
    if not novix:
        return dl.Inp(h, upto=upto)
    tl.CACHE = NOVIX_CACHE
    try:
        return dl.Inp(h, upto=upto)
    finally:
        tl.CACHE = tl.RES_DIR / "timesfm_cache"


def d0_family(h):
    return "stack" if h == 1 else "platt"


def run_d0(inp, spec, period):
    return dl.run(inp, d0_family(inp.h), spec, period)


# ---------------------------------------------------------------------------
# h=1: the stack layer on rows [s, e) with a fit schedule: once / every R rows / on a VIX trigger
# ---------------------------------------------------------------------------
def stack_range(inp, s, e, spec, extra=False, every=None, cut=None, min_gap=vl.MIN_GAP):
    X = [inp.z] + ([inp.med, inp.spr] if spec["feats"] == "zms" else [])
    if extra:
        rk = vl.rank_of(inp)
        X += [rk, inp.z * (rk - 0.5)]
    X = np.column_stack(X)
    est, y = make_estimator("LogReg", {"C": spec["C"]}), inp.ds.y
    hi = None if cut is None else vl.rank_of(inp) >= cut
    proba, thr = np.empty(e - s), np.empty(e - s)
    m, last, thr_c, fires, co = None, None, None, 0, []
    for t in range(s, e):
        if m is None:
            due = True
        elif cut is not None:
            due = bool(hi[t] != hi[t - 1] and t - last >= min_gap)
        elif every is not None:
            due = (t - s) % every == 0
        else:
            due = False
        if due:
            if m is not None:
                fires += 1
            tr = inp.train_rows(t, None)
            m, last, thr_c = fit_model(est, X[tr], y[tr]), t, float(y[tr].mean())
            co.append(m[-1].coef_[0] / m[0].scale_)
        proba[t - s], thr[t - s] = m.predict_proba(X[t:t + 1])[0, 1], thr_c
    co = np.array(co)
    return proba, thr, {"n_refits": fires, "mean_slope": float(co[:, 0].mean()),
                        "neg_slope_share": float((co[:, 0] < 0).mean()),
                        **({"mean_coef_rank": float(co[:, -2].mean()), "mean_coef_interaction": float(co[:, -1].mean())} if extra else {})}


# ---------------------------------------------------------------------------
# h=5: offset layer, optionally with VIX regressors (slope on z fixed at 1)
# ---------------------------------------------------------------------------
def fit_offset_vix(z, rk, y, ridge=RIDGE):
    """Maximum likelihood of logit p = z + a + b1 rk + b2 z (rk - 0.5); ridge on b1, b2 only (a is free)."""
    F = np.column_stack([np.ones_like(z), rk, z * (rk - 0.5)])
    P = np.diag([0.0, ridge, ridge])
    beta = np.array([brentq(lambda b: (expit(z + b) - y).mean(), -10, 10), 0.0, 0.0])
    for _ in range(50):
        p = expit(z + F @ beta)
        g = F.T @ (p - y) + P @ beta
        Hm = (F * (p * (1 - p))[:, None]).T @ F + P
        step = np.linalg.solve(Hm, g)
        beta = beta - step
        if np.abs(step).max() < 1e-10:
            break
    return beta


def offset_vix_range(inp, s, e, N=252, R=21):
    z, y, rk = inp.z, inp.ds.y, vl.rank_of(inp)
    proba, thr = np.empty(e - s), np.empty(e - s)
    cur, thr_c, co, refits = None, None, [], 0
    for i, t in enumerate(range(s, e)):
        if cur is None or i % R == 0:
            idx = inp.train_rows(t, N)
            if len(np.unique(y[idx])) == 2:
                if cur is not None:
                    refits += 1
                cur, thr_c = fit_offset_vix(z[idx], rk[idx], y[idx]), float(y[idx].mean())
                co.append(cur)
        proba[i] = expit(z[t] + cur[0] + cur[1] * rk[t] + cur[2] * z[t] * (rk[t] - 0.5))
        thr[i] = thr_c
    co = np.array(co)
    return proba, thr, {"n_refits": refits, "mean_slope": 1.0, "neg_slope_share": 0.0,
                        "mean_coef_rank": float(co[:, 1].mean()), "mean_coef_interaction": float(co[:, 2].mean())}


# ---------------------------------------------------------------------------
# D1 and D3 through one entry point
# ---------------------------------------------------------------------------
def run_d1(inp, spec, vix, period):
    s, e = inp.span(period)
    if inp.h == 1:
        p, t, a = stack_range(inp, s, e, spec, cut=vix["recal"]["cut"])
    else:
        p, t, a = vl.recal_range(inp, s, e, spec["cal"], spec["N"], cut=vix["recal"]["cut"])
    out = inp.frame(s, e, p, t)
    out.attrs.update(a)
    return out


def run_d3(inp, spec, period):
    s, e = inp.span(period)
    if inp.h == 1:
        p, t, a = stack_range(inp, s, e, spec, extra=True)
    else:
        p, t, a = offset_vix_range(inp, s, e, spec["N"], spec["R"])
    out = inp.frame(s, e, p, t)
    out.attrs.update(a)
    return out


# ---------------------------------------------------------------------------
# D2: selection among {D0, team LR, always-up}
# ---------------------------------------------------------------------------
def d0_history(inp, spec, s0, e):
    """Walk-forward D0 on rows [s0, e): same layer, expanding refit every 63 rows (h=1) or the D0 schedule (h=5)."""
    if inp.h == 1:
        p, t, _ = stack_range(inp, s0, e, spec, every=WF_EVERY)
    else:
        p, t, _ = vl.recal_range(inp, s0, e, spec["cal"], spec["N"], every=spec["R"])
    return p, t


def expert_table(inp, spec, periods, d0_test=None):
    """Experts d0 / lr / up on [s0, end of the last period).

    d0 : walk-forward D0 (see `d0_history`) up to the start of TEST; on TEST rows the D0 test frame itself
         (`d0_test`), so the expert is exactly the baseline there;
    lr : the team's fixed LR, trained once at the start of each period (as in the V2 study);
    up : always-up with the base rate of the labels known at t.
    """
    ds = inp.ds
    s0 = vl.first_expert_row(inp)
    e0 = inp.span(periods[-1])[1]
    cut = inp.span("TEST")[0] if d0_test is not None else e0
    p, t = d0_history(inp, spec, s0, cut)
    if d0_test is not None:
        assert d0_test.index.equals(ds.index[cut:e0])
        p, t = np.concatenate([p, d0_test["proba"].to_numpy()]), np.concatenate([t, d0_test["thr"].to_numpy()])
    lr = pd.concat([inp.fixed_lr(q) for q in periods])
    k0 = inp.span(periods[0])[0]
    assert lr.index.equals(ds.index[k0:e0]) and k0 <= s0
    plr, tlr = lr["proba"].to_numpy()[s0 - k0:], lr["thr"].to_numpy()[s0 - k0:]
    rate = np.array([ds.y[:ds.n_known(t_)].mean() for t_ in range(s0, e0)])
    return {"s0": s0, "e0": e0, "P": np.column_stack([p, plr, rate]), "T": np.column_stack([t, tlr, rate])}


def run_d2(inp, X, vix, period, regime=True):
    return vl.run_select(inp, X, vix["select"], period, use_regime=regime)

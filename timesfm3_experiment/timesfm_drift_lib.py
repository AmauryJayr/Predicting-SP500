"""Drift-aware outer layers on top of the frozen, zero-shot TimesFM-3 score (no model call, no retraining).

TimesFM itself is never refitted: the cached forecasts of the official setup (variant `multi`, context 256)
are read through `timesfm_lib`, and four families of methods act on the daily score:

  platt   recalibration of the score (slope + offset, or offset only) on a trailing window, refit on a schedule
  stack   walk-forward logistic regression on TimesFM-derived features (fixed / expanding / rolling)
  hedge   Hedge ensemble over {TimesFM calibrated, the team's fixed LR, base rate}
  detect  Page-Hinkley / ADWIN on TimesFM's own log loss; when it fires, the calibration is refitted

The logic mirrors `experiments/methods.py` (run_platt, run_wf, run_hedge, run_detect), whose functions need a
feature matrix `ds.X(...)` and so cannot take a precomputed score column as is; the detectors, the loss and the
protocol (purging with `Dataset.n_known`, threshold = positive rate of the training window, Hedge weights from
delayed discounted log loss) are imported or copied unchanged. Every frame is [y, proba, thr, pred] indexed by date.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

import timesfm_lib as tl
from tf_paths import RES_DIR
from common import Dataset, fit_model, make_estimator
from methods import ADWIN, PageHinkley, logloss, run_spec

VARIANT, CTX = "multi", 256            # the official TimesFM setup of the study
FIRST = "2013-01-01"                   # the cached forecasts start here: no TimesFM score exists before
MIN_TRAIN = 250
LR_SPEC = {"feats": "core", "kind": "wf", "model": "LogReg", "norm": None, "params": {"C": 0.003}, "refit": None}


class Inp:
    """Dataset(h) plus the TimesFM score z = logit(P(up)) and two extra features on every row from FIRST on.

    `upto` blanks every score after that date (used by `select`, so no 2019+ score can enter a selection run).
    """

    def __init__(self, h: int, upto: str | None = None):
        self.h, self.ds = h, Dataset(h)
        ds = self.ds
        self.p0 = ds.pos(FIRST)
        dates = ds.index[self.p0:]
        if upto is not None:
            dates = dates[dates <= pd.Timestamp(upto)]
        q, _ = tl.aligned(VARIANT, CTX, dates)
        sc = tl.scores(VARIANT, CTX, h, dates).to_numpy()
        pn = tl.panel()
        anchor = pd.Series(pn["spy"], index=pn["dates"]).reindex(dates).to_numpy()
        n = ds.n
        self.z, self.med, self.spr = (np.full(n, np.nan) for _ in range(3))
        k = self.p0 + len(dates)
        self.z[self.p0:k] = tl.logit(sc)
        self.med[self.p0:k] = q[:, h - 1, 4] - anchor              # median forecast log return over h days
        self.spr[self.p0:k] = q[:, h - 1, 8] - q[:, h - 1, 0]      # q90 - q10 at step h
        self._lr = {}

    def span(self, period):
        s, e = self.ds.period_positions(period)
        assert np.isfinite(self.z[s:e]).all(), f"no TimesFM score for some {period} rows"
        return s, e

    def frame(self, s, e, proba, thr):
        ds = self.ds
        d = pd.DataFrame({"y": ds.y[s:e], "proba": np.asarray(proba, float), "thr": np.asarray(thr, float)},
                         index=ds.index[s:e])
        d["pred"] = (d["proba"] >= d["thr"]).astype(int)
        return d

    def static(self, period):
        """The official static TimesFM frame: offset-only calibration fitted once on the purged earlier rows."""
        return tl.pred_frame(self.ds, VARIANT, CTX, self.h, period)

    def fixed_lr(self, period):
        """The team's trained-once LR (C=0.003, 23 core features)."""
        if period not in self._lr:
            self._lr[period] = run_spec(self.ds, LR_SPEC, period)
        return self._lr[period]

    def train_rows(self, t, N):
        """Rows with label_end <= date[t] (purged), restricted to the last N and to rows with a TimesFM score."""
        ds = self.ds
        nk = ds.n_known(t)
        idx = np.arange(max(self.p0, nk - N) if N else self.p0, nk)
        assert len(idx) >= MIN_TRAIN and ds.label_end[idx].max() <= ds.dates[t]
        return idx


# ---------------------------------------------------------------------------
# calibrators on the logit score
# ---------------------------------------------------------------------------
def fit_cal(z, y, cal):
    """(intercept, slope) of logit(proba) = a + b z.  'platt': LogisticRegression(C=1) as in methods.run_platt
    (the slope may turn negative = flip the direction); 'offset': slope fixed at 1, ML offset (the official rule)."""
    if len(np.unique(y)) < 2:
        return None
    if cal == "platt":
        m = LogisticRegression(C=1.0).fit(z[:, None], y)
        return float(m.intercept_[0]), float(m.coef_[0, 0])
    return float(brentq(lambda b: (expit(z + b) - y).mean(), -10, 10)), 1.0


# ---------------------------------------------------------------------------
# 1. recalibration on a trailing window (mirrors methods.run_platt)
# ---------------------------------------------------------------------------
def run_platt(inp, spec, period):
    s, e = inp.span(period)
    N, R, cal = spec["N"], spec["R"], spec["cal"]
    z, y = inp.z, inp.ds.y
    proba, thr, slopes = np.empty(e - s), np.empty(e - s), []
    cur = None
    for i, t in enumerate(range(s, e)):
        if cur is None or i % R == 0:
            idx = inp.train_rows(t, N)
            c = fit_cal(z[idx], y[idx], cal)
            if c is not None:
                cur, thr_c = c, float(y[idx].mean())
                slopes.append(c[1])
        proba[i] = expit(cur[0] + cur[1] * z[t])
        thr[i] = thr_c
    out = inp.frame(s, e, proba, thr)
    out.attrs.update(n_refits=len(slopes), neg_slope_share=float(np.mean(np.array(slopes) < 0)),
                     mean_slope=float(np.mean(slopes)))
    return out


# ---------------------------------------------------------------------------
# 2. walk-forward stacking (mirrors common.walk_forward on a TimesFM-derived feature matrix)
# ---------------------------------------------------------------------------
def run_stack(inp, spec, period):
    s, e = inp.span(period)
    X = np.column_stack([inp.z] + ([inp.med, inp.spr] if spec["feats"] == "zms" else []))
    est = make_estimator("LogReg", {"C": spec["C"]})
    y = inp.ds.y
    R = None if spec["mode"] == "fixed" else spec["R"]
    N = spec.get("W") if spec["mode"] == "rolling" else None
    blocks = [(s, np.arange(s, e))] if R is None else [(b, np.arange(b, min(b + R, e))) for b in range(s, e, R)]
    proba, thr, slopes = np.empty(e - s), np.empty(e - s), []
    for t0, rows in blocks:
        tr = inp.train_rows(t0, N)
        m = fit_model(est, X[tr], y[tr])
        proba[rows - s] = m.predict_proba(X[rows])[:, 1]
        thr[rows - s] = y[tr].mean()
        slopes.append(float(m[-1].coef_[0, 0] / m[0].scale_[0]))       # slope on the raw logit score
    out = inp.frame(s, e, proba, thr)
    out.attrs.update(n_refits=len(blocks) - 1, neg_slope_share=float(np.mean(np.array(slopes) < 0)),
                     mean_slope=float(np.mean(slopes)))
    return out


# ---------------------------------------------------------------------------
# 3. Hedge over {TimesFM calibrated, fixed LR, base rate} (mirrors methods.run_hedge, fixed experts)
# ---------------------------------------------------------------------------
def run_hedge(inp, spec, period):
    s, e = inp.span(period)
    ds, y = inp.ds, inp.ds.y
    a, g = inp.static(period), inp.fixed_lr(period)
    rate = tl.base_rate(ds, period)
    P = np.column_stack([a["proba"], g["proba"], np.full(e - s, rate)])
    T = np.column_stack([a["thr"], g["thr"], np.full(e - s, rate)])
    k = P.shape[1]
    L = logloss(y[s:e, None], P)
    eta, hl = spec["eta"], spec["loss_hl"]
    proba, thr, W = np.empty(e - s), np.empty(e - s), np.empty((e - s, k))
    for t in range(s, e):
        nk = min(ds.n_known(t), e)
        idx = np.arange(s, nk)
        if len(idx) == 0:
            w = np.full(k, 1 / k)
        else:
            disc = 0.5 ** ((t - idx) / hl)
            avg = (disc[:, None] * L[idx - s]).sum(0) / disc.sum()
            w = np.exp(-eta * (avg - avg.min())); w /= w.sum()
        proba[t - s], thr[t - s], W[t - s] = P[t - s] @ w, T[t - s] @ w, w
    out = inp.frame(s, e, proba, thr)
    out.attrs.update(mean_w_timesfm=float(W[:, 0].mean()), mean_w_lr=float(W[:, 1].mean()),
                     mean_w_rate=float(W[:, 2].mean()), n_refits=0)
    return out


# ---------------------------------------------------------------------------
# 4. drift-detector-triggered recalibration (mirrors methods.run_detect, calibrator instead of a model)
# ---------------------------------------------------------------------------
def run_detect(inp, spec, period):
    s, e = inp.span(period)
    ds, y, z = inp.ds, inp.ds.y, inp.z
    W, min_gap, stream = spec.get("W"), spec.get("min_gap", 21), spec["stream"]
    det = PageHinkley(spec["delta"], spec["lam"]) if spec["detector"] == "ph" else ADWIN(spec["delta"])

    def fit_at(t):
        idx = inp.train_rows(t, W)
        return fit_cal(z[idx], y[idx], "platt"), float(y[idx].mean())

    cal, thr0 = fit_at(s)
    last_fit, fed = s, ds.n_known(s)
    proba, th = np.empty(e - s), np.empty(e - s)
    slopes = [cal[1]]
    n_fires = 0
    for t in range(s, e):
        nk = ds.n_known(t)
        fire = False
        for r in range(max(fed, s), nk):                # outcomes of our own past predictions that are now known
            p = proba[r - s]
            x = logloss(y[r], p) if stream == "logloss" else float((p >= th[r - s]) != y[r])
            fire |= det.update(x)
        fed = max(fed, nk)
        if fire and t - last_fit >= min_gap:
            cal, thr0 = fit_at(t)
            last_fit = t
            det.reset()
            n_fires += 1
            slopes.append(cal[1])
        proba[t - s] = expit(cal[0] + cal[1] * z[t])
        th[t - s] = thr0
    out = inp.frame(s, e, proba, th)
    out.attrs.update(n_refits=n_fires, neg_slope_share=float(np.mean(np.array(slopes) < 0)),
                     mean_slope=float(np.mean(slopes)))
    return out


RUN = {"platt": run_platt, "stack": run_stack, "hedge": run_hedge, "detect": run_detect}


def run(inp, family, spec, period):
    return RUN[family](inp, spec, period)


# ---------------------------------------------------------------------------
# the (small) grid, fixed before any selection run
# ---------------------------------------------------------------------------
def grid():
    G = {"platt": [], "stack": [], "hedge": [], "detect": []}
    for cal in ("platt", "offset"):
        for N in (252, 504, None):                      # trailing window; None = expanding from the first score
            for R in (21, 63):
                G["platt"].append({"cal": cal, "N": N, "R": R})
    for feats in ("z", "zms"):                          # z = logit score; zms = z + median return + q90-q10 spread
        G["stack"].append({"feats": feats, "C": 0.03, "mode": "fixed"})
        G["stack"].append({"feats": feats, "C": 0.03, "mode": "expanding", "R": 21})
        for W in (252, 504):
            G["stack"].append({"feats": feats, "C": 0.03, "mode": "rolling", "W": W, "R": 21})
    for eta in (20, 100, 500):
        for hl in (63, 252):
            G["hedge"].append({"eta": eta, "loss_hl": hl})
    for det, delta, lam, stream in (("ph", 0.005, 0.5, "logloss"), ("ph", 0.005, 2.0, "logloss"),
                                    ("adwin", 0.002, None, "error"), ("adwin", 0.05, None, "error")):
        for W in (504, None):
            G["detect"].append({"detector": det, "delta": delta, "lam": lam, "stream": stream, "W": W})
    return G

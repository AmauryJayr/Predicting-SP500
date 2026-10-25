"""Drift-aware methods and their matched non-adaptive counterparts.

Every method is described by a plain dict ("spec"); `run_spec(ds, spec, period)`
returns a prediction frame (y, proba, thr, pred) on that evaluation period.

Spec keys common to all kinds:
  kind   : 'wf' | 'prior' | 'platt' | 'hedge' | 'detect' | 'online' | 'regime'
  model  : 'LogReg' | 'GBDT'  (+ 'params')
  feats  : 'all' | 'core'
  norm   : None | 'z252' | 'z504' | 'z1260' | 'rank252' | 'rank504'  (trailing, causal)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler

from common import (WARMUP, Dataset, fit_model, make_estimator, to_pred, walk_forward, weighted_rate,
                    decay_weights)

EPS = 1e-6


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logloss(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def spec_key(spec: dict) -> str:
    return json.dumps(spec, sort_keys=True, default=str)


def _est(spec):
    return make_estimator(spec["model"], spec.get("params"))


def _X(ds, spec):
    return ds.X(spec.get("feats", "all"), spec.get("norm"))


# ---------------------------------------------------------------------------
# 'wf': plain walk-forward (fixed / expanding / rolling), optional decay weights
# ---------------------------------------------------------------------------
def run_wf(ds, spec, start, end, warm=False):
    s0 = start - WARMUP if warm else start
    R = spec.get("refit")
    df = walk_forward(ds, _X(ds, spec), _est(spec), s0 if R else start, end, R,
                      window=spec.get("window"), half_life=spec.get("half_life"))
    return df


# ---------------------------------------------------------------------------
# 'prior': label-shift (prior-drift) correction on top of a base walk-forward
#   p' = sigmoid(logit p + logit(pi_recent) - logit(pi_train)), pi_recent = trailing
#   positive rate of the last N *known* labels at each decision date.
#   thr_mode 'recent' -> threshold pi_recent ; 'train' -> threshold pi_train
# ---------------------------------------------------------------------------
def run_prior(ds, spec, start, end):
    base = run_wf(ds, spec["base"] | {"kind": "wf"}, start, end)
    N = spec["N"]
    out = base.copy()
    for t in base.index:
        nk = ds.n_known(t)
        pr = np.clip(ds.y[max(0, nk - N):nk].mean(), 0.05, 0.95)
        pt = base.at[t, "thr"]
        out.at[t, "proba"] = sigmoid(logit(base.at[t, "proba"]) + logit(pr) - logit(pt))
        out.at[t, "thr"] = pr if spec["thr_mode"] == "recent" else pt
    return out


# ---------------------------------------------------------------------------
# 'platt': recalibration of a walk-forward base on its own out-of-sample predictions
#   mode 'recent': calibrator refit every `every` rows on the last N known OOS rows
#   mode 'static': calibrator fit once, at the start of the evaluation period
#   threshold = positive rate in the calibrator's training window
# ---------------------------------------------------------------------------
def _fit_platt(z, y):
    if len(np.unique(y)) < 2:
        return None
    return LogisticRegression(C=1.0).fit(z[:, None], y)


def run_platt(ds, spec, start, end):
    base = run_wf(ds, spec["base"] | {"kind": "wf"}, start, end, warm=True)
    z = logit(base["proba"].to_numpy())
    pos0 = base.index[0]
    N, every, mode = spec["N"], spec.get("every", 5), spec["mode"]
    out = base.loc[start:end - 1].copy()
    cal, thr = None, None
    for i, t in enumerate(range(start, end)):
        if cal is None or (mode == "recent" and i % every == 0):
            nk = ds.n_known(t)
            lo = max(pos0, nk - N)
            idx = np.arange(lo, nk) - pos0
            assert ds.label_end[idx + pos0].max() <= ds.dates[t]
            c = _fit_platt(z[idx], ds.y[idx + pos0])
            if c is not None:
                cal, thr = c, ds.y[idx + pos0].mean()
        out.at[t, "proba"] = cal.predict_proba(np.array([[z[t - pos0]]]))[0, 1]
        out.at[t, "thr"] = thr
    return out


# ---------------------------------------------------------------------------
# 'hedge': exponentially weighted expert ensemble with delayed, discounted log-loss
#   experts: list of base wf specs. adaptive=False -> uniform weights (counterpart)
# ---------------------------------------------------------------------------
def run_hedge(ds, spec, start, end):
    experts = spec["experts"]
    fixed = spec.get("fixed_experts", False)
    P, T = [], []
    for e in experts:
        e = dict(e)
        if fixed:
            e["refit"] = None  # trained once (keeps its window length)
            e.pop("half_life", None)  # and no time-decay weighting
        df = run_wf(ds, e | {"kind": "wf"}, start, end, warm=not fixed)
        P.append(df["proba"]); T.append(df["thr"])
    P = pd.concat(P, axis=1).to_numpy(); T = pd.concat(T, axis=1).to_numpy()
    pos0 = start if fixed else start - WARMUP
    k = P.shape[1]
    out = pd.DataFrame(index=np.arange(start, end), columns=["proba", "thr"], dtype=float)
    if not spec.get("adaptive", True):
        w = np.full(k, 1 / k)
        out["proba"] = P[start - pos0:] @ w
        out["thr"] = T[start - pos0:] @ w
        return out
    eta, hl = spec["eta"], spec["loss_hl"]
    L = logloss(ds.y[pos0:end, None], P)  # per-row expert losses (only used once known)
    for t in range(start, end):
        nk = min(ds.n_known(t), end)
        idx = np.arange(pos0, nk)
        if len(idx) == 0:
            w = np.full(k, 1 / k)
        else:
            disc = 0.5 ** ((t - idx) / hl)
            avg = (disc[:, None] * L[idx - pos0]).sum(0) / disc.sum()
            a = -eta * (avg - avg.min())
            w = np.exp(a); w /= w.sum()
        out.at[t, "proba"] = P[t - pos0] @ w
        out.at[t, "thr"] = T[t - pos0] @ w
    return out


# ---------------------------------------------------------------------------
# 'detect': drift-detection-triggered retraining on the delayed loss stream
# ---------------------------------------------------------------------------
class PageHinkley:
    """Detects an increase in the mean of a stream (here: log-loss)."""

    def __init__(self, delta, lam):
        self.delta, self.lam = delta, lam
        self.reset()

    def reset(self):
        self.n, self.mean, self.m, self.M = 0, 0.0, 0.0, 0.0

    def update(self, x):
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.m += x - self.mean - self.delta
        self.M = min(self.M, self.m)
        return self.m - self.M > self.lam


class ADWIN:
    """Simplified ADWIN (Bifet & Gavalda 2007): exhaustive cut search on a bounded window."""

    def __init__(self, delta, max_len=1000, min_side=30, step=5):
        self.delta, self.max_len, self.min_side, self.step = delta, max_len, min_side, step
        self.w = []

    def reset(self):
        self.w = []

    def update(self, x):
        self.w.append(float(x))
        if len(self.w) > self.max_len:
            self.w.pop(0)
        n = len(self.w)
        if n < 2 * self.min_side:
            return False
        a = np.asarray(self.w)
        cs = np.cumsum(a)
        for c in range(self.min_side, n - self.min_side + 1, self.step):
            n0, n1 = c, n - c
            mu0, mu1 = cs[c - 1] / n0, (cs[-1] - cs[c - 1]) / n1
            m = 1 / (1 / n0 + 1 / n1)
            eps = np.sqrt(np.log(4 * n / self.delta) / (2 * m))
            if mu1 - mu0 > eps:  # only increases in error matter
                self.w = self.w[c:]
                return True
        return False


def run_detect(ds, spec, start, end):
    X, y, est = _X(ds, spec), ds.y, _est(spec)
    W, min_gap = spec.get("window"), spec.get("min_gap", 21)
    det = PageHinkley(spec["delta"], spec["lam"]) if spec["detector"] == "ph" else ADWIN(spec["delta"])
    stream = spec.get("stream", "logloss")

    def fit_at(t):
        nk = ds.n_known(t)
        tr = np.arange(nk) if W is None else np.arange(max(0, nk - W), nk)
        return fit_model(est, X[tr], y[tr]), y[tr].mean()

    model, thr = fit_at(start)
    last_fit, fed = start, ds.n_known(start)
    proba = np.empty(end - start); th = np.empty(end - start)
    n_fires = 0
    for t in range(start, end):
        nk = ds.n_known(t)
        fire = False
        for s in range(max(fed, start), nk):  # newly realised outcomes of our own past predictions
            p = proba[s - start]
            x = logloss(y[s], p) if stream == "logloss" else float((p >= th[s - start]) != y[s])
            fire |= det.update(x)
        fed = max(fed, nk)
        if fire and t - last_fit >= min_gap:
            model, thr = fit_at(t)
            last_fit = t
            det.reset()
            n_fires += 1
        proba[t - start] = model.predict_proba(X[t:t + 1])[0, 1]
        th[t - start] = thr
    out = pd.DataFrame({"proba": proba, "thr": th}, index=np.arange(start, end))
    out.attrs["n_refits"] = n_fires
    return out


# ---------------------------------------------------------------------------
# 'online': SGD logistic regression updated as delayed labels arrive
# ---------------------------------------------------------------------------
def run_online(ds, spec, start, end):
    X, y = _X(ds, spec), ds.y
    nk0 = ds.n_known(start)
    sc = StandardScaler().fit(X[:nk0])
    Z = np.clip(sc.transform(X), -6, 6)
    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=spec["alpha"], learning_rate="constant",
                        eta0=spec["eta0"], random_state=42)
    rng = np.random.default_rng(42)
    for _ in range(spec.get("epochs", 5)):
        perm = rng.permutation(nk0)
        clf.partial_fit(Z[:nk0][perm], y[:nk0][perm], classes=np.array([0, 1]))
    adapt = spec.get("adaptive", True)
    proba = np.empty(end - start); th = np.empty(end - start)
    fed = nk0
    for t in range(start, end):
        nk = ds.n_known(t)
        if adapt and nk > fed:
            clf.partial_fit(Z[fed:nk], y[fed:nk])
            fed = nk
        proba[t - start] = clf.predict_proba(Z[t:t + 1])[0, 1]
        th[t - start] = y[:fed].mean() if adapt else y[:nk0].mean()
    return pd.DataFrame({"proba": proba, "thr": th}, index=np.arange(start, end))


# ---------------------------------------------------------------------------
# 'regime': separate models per volatility regime (regime cut fixed at fit time)
# ---------------------------------------------------------------------------
def run_regime(ds, spec, start, end):
    X, y, est = _X(ds, spec), ds.y, _est(spec)
    var = ds.data[spec["regime_var"]].to_numpy()
    R = spec.get("refit")
    blocks = [(start, np.arange(start, end))] if R is None else \
        [(b, np.arange(b, min(b + R, end))) for b in range(start, end, R)]
    proba = np.empty(end - start); th = np.empty(end - start)
    for t0, rows in blocks:
        nk = ds.n_known(t0)
        tr = np.arange(nk)
        cut = np.median(var[tr]) if spec["regime_var"] == "vix_level" else 0.5
        for hi in (False, True):
            trm = tr[(var[tr] >= cut) == hi]
            rr = rows[(var[rows] >= cut) == hi]
            if len(rr) == 0:
                continue
            m = fit_model(est, X[trm], y[trm])
            proba[rr - start] = m.predict_proba(X[rr])[:, 1]
            th[rr - start] = y[trm].mean()
    return pd.DataFrame({"proba": proba, "thr": th}, index=np.arange(start, end))


RUNNERS = {"wf": lambda ds, s, a, b: run_wf(ds, s, a, b), "prior": run_prior, "platt": run_platt,
           "hedge": run_hedge, "detect": run_detect, "online": run_online, "regime": run_regime}


def run_spec(ds: Dataset, spec: dict, period: str) -> pd.DataFrame:
    start, end = ds.period_positions(period)
    df = RUNNERS[spec["kind"]](ds, spec, start, end)
    pred = to_pred(ds, df, start, end)
    pred.attrs.update(df.attrs)
    return pred

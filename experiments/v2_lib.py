"""v2: drift-aware learning for targets with a real, persistent signal (volatility, cross-asset correlation).

Reusable, notebook-importable library. Everything here is causal:
  * features at row t use prices up to the close of t;
  * the target of row t is computed over sessions t+1..t+h and is "known" at the close of t+h (label_end);
  * a model fitted at decision row t0 uses only rows with label_end <= date[t0] (n_known prefix, asserted);
  * every online mechanism (bias correction, recalibration, RLS, expert weights, drift detectors) consumes the
    outcome of row s only once label_end[s] <= date[t] (delayed feedback).

Targets (built by `build_targets`):
  spy_vol   h=5 : log annualised realised vol of SPY close-to-close returns over t+1..t+5
  tlt_vol   h=5 : same for TLT (optional third target: bond-vol regime changed in 2022)
  spy_tlt_corr h=21 : realised correlation of SPY and TLT daily log returns over t+1..t+21

Spec = plain dict:  {"model": "lin"|"hgb", "feats": <feature-set name>, "method": <kind>, ...params}
  method 'fixed'  : one fit at the period start (the matched trained-once counterpart)
  method 'refit'  : refit every R rows; window W (None = expanding); optional decay half-life hl
  method 'rls'    : recursive least squares with forgetting factor lam (linear only), updated daily
  method 'detect' : corrected drift-triggered retraining (ADWIN2 / PH2 from audit_fixed_methods, fix of B1)
  method 'ens'    : recent-error-weighted ensemble over refitting experts (Hedge on delayed squared error)
  method 'bias'   : fixed model + trailing mean of the last N known residuals (intercept recalibration)
  method 'recal'  : fixed model + trailing OLS y ~ a + b*yhat on the last N known rows (level+scale)
  method 'const'  : training-mean forecast (long-run mean baseline), fixed at period start
  method 'raw'    : non-fitted baseline: prediction = one feature column (persistence, VIX-implied)
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "experiments" / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
STUDY_START, STUDY_END = "2008-01-01", "2025-12-31"
PERIODS = {
    "F1": ("2013-01-01", "2014-12-31"),
    "F2": ("2015-01-01", "2016-12-31"),
    "VAL": ("2017-01-01", "2018-12-31"),
    "TEST": ("2019-01-01", "2025-12-31"),
}
FOLDS = ["F1", "F2", "VAL"]
WARMUP = 252  # out-of-sample warm-up for ensemble experts
ANN = 252.0


# ---------------------------------------------------------------------------
# Targets and features
# ---------------------------------------------------------------------------
def _lr(c):
    return np.log(c).diff()


def _logvol(r, w):
    """log annualised RMS of returns over the trailing w rows (ending at t, inclusive)."""
    return 0.5 * np.log(ANN * (r ** 2).rolling(w).mean().clip(lower=1e-10))


def _fwd_logvol(r, h):
    """log annualised RMS over t+1..t+h."""
    return 0.5 * np.log(ANN * (r ** 2).rolling(h).mean().shift(-h).clip(lower=1e-10))


def _parkinson_log(d, w):
    hl2 = np.log(d["High"] / d["Low"]) ** 2 / (4 * np.log(2))
    return 0.5 * np.log(ANN * hl2.rolling(w).mean().clip(lower=1e-10))


def vol_frame(px, tk):
    d = px[tk]
    r = _lr(d["Close"])
    f = pd.DataFrame(index=d.index)
    f["lpk1"] = _parkinson_log(d, 1)  # daily (range-based) vol, the HAR daily lag
    f["lpk5"] = _parkinson_log(d, 5)
    f["lrv5"] = _logvol(r, 5)
    f["lrv20"] = _logvol(r, 20)
    f["lrv22"] = _logvol(r, 22)
    f["lrv63"] = _logvol(r, 63)
    f["lsemi5"] = 0.5 * np.log(ANN * ((r.clip(upper=0)) ** 2).rolling(5).mean().clip(lower=1e-8))
    f["ret1"] = r
    f["ret5"] = np.log(d["Close"]).diff(5)
    f["ret20"] = np.log(d["Close"]).diff(20)
    f["dist_high"] = d["Close"] / d["Close"].rolling(252).max() - 1
    return f, r


class Target:
    """Container for one prediction problem. Arrays are aligned to `self.index` (rows)."""

    def __init__(self, name, h, y, ref, F: pd.DataFrame, feature_sets: dict, kind: str, cal):
        df = F.copy()
        df["__y"] = y
        df["__ref"] = ref
        label_end = pd.Series(cal, index=cal).shift(-h)
        df["__le"] = label_end
        df = df.loc[STUDY_START:STUDY_END]
        needed = sorted({c for v in feature_sets.values() for c in v})
        ok = df[needed].notna().all(axis=1) & df["__y"].notna() & df["__ref"].notna() & df["__le"].notna()
        df = df[ok]
        self.name, self.h, self.kind = name, h, kind
        self.index = df.index
        self.dates = df.index.to_numpy()
        self.label_end = df["__le"].to_numpy()
        self.y = df["__y"].to_numpy(float)
        self.ref = df["__ref"].to_numpy(float)  # classification reference (trailing RV20 or 0)
        self.ycls = (self.y > self.ref).astype(int)
        self.F = df[needed]
        self.feature_sets = feature_sets
        self.n = len(df)
        self._X = {}
        assert np.all(np.diff(self.label_end.astype("datetime64[ns]").astype(np.int64)) >= 0)

    def X(self, fs):
        if fs not in self._X:
            self._X[fs] = self.F[self.feature_sets[fs]].to_numpy(float)
        return self._X[fs]

    def col(self, c):
        return self.F[c].to_numpy(float)

    def n_known(self, t):
        return int(np.searchsorted(self.label_end, self.dates[t], side="right"))

    def period(self, p):
        a, b = PERIODS[p]
        s = int(np.searchsorted(self.dates, np.datetime64(pd.Timestamp(a))))
        e = int(np.searchsorted(self.dates, np.datetime64(pd.Timestamp(b)), side="right"))
        return s, e


def build_targets(px=None) -> dict:
    if px is None:
        px = pd.read_pickle(ROOT / "data" / "aligned_prices.pkl")
    cal = px["SPY"].index
    lvix = np.log(px["^VIX"]["Close"] / 100)
    out = {}

    # --- SPY next-week realised vol -------------------------------------------------------
    f, r = vol_frame(px, "SPY")
    ft, rt = vol_frame(px, "TLT")
    f["lvix"] = lvix
    f["vix_chg5"] = lvix.diff(5)
    f["tlt_lrv22"] = ft["lrv22"]
    har = ["lpk1", "lrv5", "lrv22"]
    fsets = {"har": har, "harx": har + ["lvix"], "vix": ["lvix"],
             "full": har + ["lvix", "lpk5", "lrv63", "lsemi5", "ret1", "ret5", "ret20", "vix_chg5", "dist_high",
                            "tlt_lrv22"],
             "p5": ["lrv5"], "p20": ["lrv20"]}
    out["spy_vol"] = Target("spy_vol", 5, _fwd_logvol(r, 5), f["lrv20"], f, fsets, "logvol", cal)

    # --- TLT next-week realised vol (optional third target) -----------------------------------
    g = ft.copy()
    g["lvix"] = lvix
    g["spy_lrv22"] = f["lrv22"]
    g["ief_lrv22"] = _logvol(_lr(px["IEF"]["Close"]), 22)
    gsets = {"har": har, "harx": har + ["lvix", "spy_lrv22"],
             "full": har + ["lvix", "spy_lrv22", "lpk5", "lrv63", "lsemi5", "ret1", "ret5", "ret20", "dist_high",
                            "ief_lrv22"],
             "p5": ["lrv5"], "p20": ["lrv20"]}
    out["tlt_vol"] = Target("tlt_vol", 5, _fwd_logvol(rt, 5), g["lrv20"], g, gsets, "logvol", cal)

    # --- SPY-TLT next-month realised correlation -------------------------------------------
    h = 21
    rs, rb = _lr(px["SPY"]["Close"]), _lr(px["TLT"]["Close"])
    c = pd.DataFrame(index=cal)
    for w in (21, 63, 126, 252):
        c[f"c{w}"] = rs.rolling(w).corr(rb)
    c["lvix"] = lvix
    c["spy_lrv22"] = f["lrv22"]
    c["tlt_lrv22"] = ft["lrv22"]
    c["spy_ret63"] = np.log(px["SPY"]["Close"]).diff(63)
    c["tlt_ret63"] = np.log(px["TLT"]["Close"]).diff(63)
    c["tlt_ret252"] = np.log(px["TLT"]["Close"]).diff(252)
    c["tlt_ief_20"] = np.log(px["TLT"]["Close"]).diff(20) - np.log(px["IEF"]["Close"]).diff(20)
    c["gld_ret63"] = np.log(px["GLD"]["Close"]).diff(63)
    fwd_corr = rs.rolling(h).corr(rb).shift(-h)
    csets = {"hc": ["c21", "c63", "c252"],
             "full": ["c21", "c63", "c126", "c252", "lvix", "spy_lrv22", "tlt_lrv22", "spy_ret63", "tlt_ret63",
                      "tlt_ret252", "tlt_ief_20", "gld_ret63"],
             "p21": ["c21"], "p63": ["c63"]}
    zero = pd.Series(0.0, index=cal)
    out["spy_tlt_corr"] = Target("spy_tlt_corr", h, fwd_corr, zero, c, csets, "corr", cal)
    return out


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def make_model(name):
    if name == "lin":
        return Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])
    if name == "hgb":  # hyper-parameters fixed a priori (not tuned)
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                             min_samples_leaf=50, l2_regularization=1.0, early_stopping=False,
                                             random_state=SEED)
    raise ValueError(name)


def decay_w(n, hl):
    if hl is None:
        return None
    age = np.arange(n)[::-1].astype(float)
    w = 0.5 ** (age / hl)
    return w / w.mean()


def fit(tg: Target, spec, t0, lo=0, W=None, hl=None):
    """Fit spec's model at decision row t0 on known rows [max(lo, nk-W), nk)."""
    nk = tg.n_known(t0)
    a = lo if W is None else max(lo, nk - W)
    tr = np.arange(a, nk)
    assert len(tr) >= 120, (tg.name, t0, len(tr))
    assert tg.label_end[tr].max() <= tg.dates[t0]  # purging
    X = tg.X(spec["feats"])
    m = clone(make_model(spec["model"]))
    w = decay_w(len(tr), hl)
    if w is None:
        m.fit(X[tr], tg.y[tr])
    elif spec["model"] == "lin":
        m.fit(X[tr], tg.y[tr], m__sample_weight=w)
    else:
        m.fit(X[tr], tg.y[tr], sample_weight=w)
    return m


def _clip(tg, p):
    return np.clip(p, -0.99, 0.99) if tg.kind == "corr" else p


# ---------------------------------------------------------------------------
# Methods. Each returns predictions for rows [start, end) as a numpy array.
# ---------------------------------------------------------------------------
def m_fixed(tg, spec, start, end, all_rows=False):
    m = fit(tg, spec, start)
    X = tg.X(spec["feats"])
    if all_rows:  # predictions on [0, end) (in-sample before start; used by bias/recal on known rows)
        return _clip(tg, m.predict(X[:end]))
    return _clip(tg, m.predict(X[start:end]))


def m_refit(tg, spec, start, end):
    R, W, hl = spec["R"], spec.get("W"), spec.get("hl")
    X = tg.X(spec["feats"])
    out = np.empty(end - start)
    for b in range(start, end, R):
        e = min(b + R, end)
        out[b - start:e - start] = fit(tg, spec, b, W=W, hl=hl).predict(X[b:e])
    return _clip(tg, out)


def m_const(tg, spec, start, end):
    nk = tg.n_known(start)
    return np.full(end - start, tg.y[:nk].mean())


def m_raw(tg, spec, start, end):
    p = tg.col(spec["col"])[start:end]
    return p.copy()


def m_rls(tg, spec, start, end):
    """RLS with exponential forgetting on standardized features + intercept. Runs sequentially over all
    known history from row 0 (equivalent to exponentially weighted least squares) and then updates online as
    each label becomes known. Ridge-like prior through the initial P."""
    lam = spec["lam"]
    X = tg.X(spec["feats"])
    nk0 = tg.n_known(start)
    sc = StandardScaler().fit(X[:nk0])
    Z = np.column_stack([np.ones(len(X)), np.clip(sc.transform(X), -8, 8)])
    d = Z.shape[1]
    theta = np.zeros(d)
    theta[0] = tg.y[:nk0].mean()
    P = np.eye(d) * 10.0
    y = tg.y

    def upd(theta, P, s):
        z = Z[s]
        Pz = P @ z
        k = Pz / (lam + z @ Pz)
        theta = theta + k * (y[s] - z @ theta)
        P = (P - np.outer(k, Pz)) / lam
        P = 0.5 * (P + P.T)
        return theta, P

    for s in range(nk0):
        theta, P = upd(theta, P, s)
    fed = nk0
    out = np.empty(end - start)
    for t in range(start, end):
        nk = tg.n_known(t)
        for s in range(fed, nk):
            theta, P = upd(theta, P, s)
        fed = max(fed, nk)
        out[t - start] = Z[t] @ theta
    return _clip(tg, out)


def m_bias(tg, spec, start, end):
    base = m_fixed(tg, spec, start, end, all_rows=True)
    N = spec["N"]
    res = tg.y[:end] - base
    cs = np.concatenate([[0.0], np.cumsum(res)])
    out = np.empty(end - start)
    for t in range(start, end):
        nk = tg.n_known(t)
        a = max(0, nk - N)
        out[t - start] = base[t] + (cs[nk] - cs[a]) / (nk - a)
    return _clip(tg, out)


def m_recal(tg, spec, start, end):
    base = m_fixed(tg, spec, start, end, all_rows=True)
    N = spec["N"]
    out = np.empty(end - start)
    for t in range(start, end):
        nk = tg.n_known(t)
        a = max(0, nk - N)
        xb, yb = base[a:nk], tg.y[a:nk]
        vx = xb.var()
        b = np.clip(np.cov(xb, yb, bias=True)[0, 1] / vx, 0.0, 2.0) if vx > 1e-12 else 1.0
        out[t - start] = yb.mean() + b * (base[t] - xb.mean())
    return _clip(tg, out)


ENS_EXPERTS = [dict(R=21, W=None), dict(R=21, W=1008), dict(R=21, W=504), dict(R=21, W=252), dict(R=21, hl=126)]


def m_ens(tg, spec, start, end, cache=None):
    """Hedge over refitting experts; weight_k ∝ exp(-eta * (L_k / min L - 1)), L_k = discounted mean squared
    error (half-life `hl`) over rows whose labels are known. Experts are run from start-WARMUP so the first test
    weights come from genuinely out-of-sample errors."""
    s0 = max(start - WARMUP, 0)
    P = []
    for e in ENS_EXPERTS:
        sp = {"model": spec["model"], "feats": spec["feats"], "method": "refit", **e}
        key = (json.dumps(sp, sort_keys=True), s0, end)
        if cache is not None and key in cache:
            P.append(cache[key])
        else:
            p = m_refit(tg, sp, s0, end)
            if cache is not None:
                cache[key] = p
            P.append(p)
    P = np.column_stack(P)
    L = (tg.y[s0:end, None] - P) ** 2
    eta, hl = spec["eta"], spec["hl"]
    out = np.empty(end - start)
    k = P.shape[1]
    for t in range(start, end):
        nk = min(tg.n_known(t), end)
        idx = np.arange(s0, nk)
        if len(idx) < 21:
            w = np.full(k, 1 / k)
        else:
            disc = 0.5 ** ((t - idx) / hl)
            avg = (disc[:, None] * L[idx - s0]).sum(0) / disc.sum()
            a = -eta * (avg / avg.min() - 1)
            w = np.exp(a)
            w /= w.sum()
        out[t - start] = P[t - s0] @ w
    return _clip(tg, out)


# --- corrected detectors (B1 fix; copied from audit_fixed_methods.py, unchanged logic) ---------------
class ADWIN2:
    def __init__(self, delta, max_len=1000, min_side=30, step=5):
        self.delta, self.max_len, self.min_side, self.step = delta, max_len, min_side, step
        self.w, self.rows = [], []

    def update(self, x, row):
        self.w.append(float(x)); self.rows.append(row)
        if len(self.w) > self.max_len:
            self.w.pop(0); self.rows.pop(0)
        n = len(self.w)
        if n < 2 * self.min_side:
            return None
        cs = np.cumsum(self.w)
        for c in range(self.min_side, n - self.min_side + 1, self.step):
            n0, n1 = c, n - c
            mu0, mu1 = cs[c - 1] / n0, (cs[-1] - cs[c - 1]) / n1
            m = 1 / (1 / n0 + 1 / n1)
            if mu1 - mu0 > np.sqrt(np.log(4 * n / self.delta) / (2 * m)):
                cut_row = self.rows[c]
                self.w, self.rows = self.w[c:], self.rows[c:]
                return cut_row
        return None


class PH2:
    def __init__(self, delta, lam):
        self.delta, self.lam = delta, lam
        self.n, self.mean, self.m, self.M, self.arg = 0, 0.0, 0.0, 0.0, None

    def update(self, x, row):
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.m += x - self.mean - self.delta
        if self.m <= self.M:
            self.M, self.arg = self.m, row
        if self.m - self.M > self.lam:
            cut = self.arg if self.arg is not None else row
            self.n, self.mean, self.m, self.M, self.arg = 0, 0.0, 0.0, 0.0, None
            return cut
        return None


def m_detect(tg, spec, start, end, Wmax=1260, R=21, min_gap=21):
    """Drift-triggered retraining with the corrected logic (audit B1): on detection, refit on rows after the
    estimated change point (at least `min_rows` = 252 most recent known rows), then keep refitting every R rows
    on the growing post-change window until it reaches Wmax rows. Before any detection it IS the fixed model.
    Stream = |residual| / (4 * in-sample residual sd), clipped to [0, 1] (ADWIN's bound needs [0,1])."""
    X = tg.X(spec["feats"])
    min_rows = 252
    det = ADWIN2(spec["delta"]) if spec["detector"] == "adwin" else PH2(0.005, spec["lam"])
    model = fit(tg, spec, start)
    nk0 = tg.n_known(start)
    scale = 4 * np.std(tg.y[:nk0] - model.predict(X[:nk0])) + 1e-12

    def fit_from(t, lo):
        nk = tg.n_known(t)
        lo = max(0, min(lo, nk - min_rows), nk - Wmax)
        return fit(tg, spec, t, lo=lo), nk - lo

    out = np.empty(end - start)
    regime_lo, last_fit, growing, fires = 0, start, False, 0
    fed = start
    for t in range(start, end):
        nk = tg.n_known(t)
        cut = None
        for s in range(max(fed, start), nk):
            x = min(1.0, abs(tg.y[s] - out[s - start]) / scale)
            c = det.update(x, s)
            cut = c if c is not None else cut
        fed = max(fed, nk)
        if cut is not None and t - last_fit >= min_gap:
            regime_lo, growing = cut, True
            model, ntr = fit_from(t, regime_lo)
            last_fit, fires = t, fires + 1
        elif growing and t - last_fit >= R:
            model, ntr = fit_from(t, regime_lo)
            last_fit = t
            growing = ntr < Wmax
        out[t - start] = _clip(tg, model.predict(X[t:t + 1]))[0]
    m_detect.last_fires = fires
    return out


METHODS = {"fixed": m_fixed, "refit": m_refit, "const": m_const, "raw": m_raw, "rls": m_rls, "bias": m_bias,
           "recal": m_recal, "ens": m_ens, "detect": m_detect}


def run(tg: Target, spec: dict, period: str, cache=None):
    s, e = tg.period(period)
    if spec["method"] == "ens":
        return m_ens(tg, spec, s, e, cache=cache)
    return METHODS[spec["method"]](tg, spec, s, e)


def counterpart(spec):
    """The same model and features trained once at the period start."""
    return {"model": spec["model"], "feats": spec["feats"], "method": "fixed"}


def key(spec):
    return json.dumps(spec, sort_keys=True)


def label(spec):
    m = spec["method"]
    if m == "raw":
        return f"raw:{spec['col']}"
    if m == "const":
        return "longrun_mean"
    extra = ",".join(f"{k}={v}" for k, v in sorted(spec.items()) if k not in ("model", "feats", "method"))
    return f"{spec['model']}/{spec['feats']}/{m}" + (f"({extra})" if extra else "")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def auc(y, s):
    r = rankdata(s)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def qlike(y, p):
    """QLIKE on variance; y, p are log-vol, variance = exp(2*.)."""
    ratio = np.exp(2 * y) / np.exp(2 * p)
    return ratio - np.log(ratio) - 1


def metrics(tg: Target, period: str, p: np.ndarray) -> dict:
    s, e = tg.period(period)
    return metrics_rows(tg, np.arange(s, e), p)


def metrics_rows(tg, rows, p):
    y, ref, yc = tg.y[rows], tg.ref[rows], tg.ycls[rows]
    se = (y - p) ** 2
    d = {"n": len(rows), "mse": se.mean(), "r2": 1 - se.sum() / ((y - y.mean()) ** 2).sum(),
         "bias": (p - y).mean()}
    if tg.kind == "logvol":
        d["qlike"] = qlike(y, p).mean()
    sc = p - ref
    yh = (sc > 0).astype(int)
    if 0 < yc.sum() < len(yc):
        d["ba"] = 0.5 * (yh[yc == 1].mean() + (1 - yh[yc == 0]).mean())
        d["auc"] = auc(yc, sc)
    else:
        d["ba"] = d["auc"] = np.nan
    return d


# ---------------------------------------------------------------------------
# Paired moving-block bootstrap (positive delta = A better)
# ---------------------------------------------------------------------------
def block_len(h):
    return max(20, 3 * h)


def _block_idx(n, L, n_boot, rng):
    k = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, (n_boot, k))
    return (starts[:, :, None] + np.arange(L)).reshape(n_boot, -1)[:, :n]


def paired_boot(tg: Target, rows, pa, pb, n_boot=2000, seed=SEED, L=None):
    L = L or block_len(tg.h)
    rng = np.random.default_rng(seed)
    y, ref, yc = tg.y[rows], tg.ref[rows], tg.ycls[rows]
    I = _block_idx(len(rows), L, n_boot, rng)
    out = {}
    la, lb = (y - pa) ** 2, (y - pb) ** 2
    # MSE improvement (B - A); R2 improvement has the same sign/p (= dMSE / var(y) per draw)
    D = lb[I].mean(1) - la[I].mean(1)
    pt = lb.mean() - la.mean()
    out.update(_summ("mse", pt, D))
    yv = y[I]
    D2 = D / yv.var(1)
    out.update(_summ("r2", pt / y.var(), D2))
    if tg.kind == "logvol":
        qa, qb = qlike(y, pa), qlike(y, pb)
        out.update(_summ("qlike", qb.mean() - qa.mean(), qb[I].mean(1) - qa[I].mean(1)))
    sa, sb = pa - ref, pb - ref
    ha, hb = (sa > 0).astype(float), (sb > 0).astype(float)

    def ba(yc_, h_):
        pos = yc_ == 1
        return 0.5 * ((h_ * pos).sum(1) / pos.sum(1) + ((1 - h_) * ~pos).sum(1) / (~pos).sum(1))

    ycI = yc[I]
    if 0 < yc.sum() < len(yc):
        Dba = ba(ycI, ha[I]) - ba(ycI, hb[I])
        ptba = ba(yc[None], ha[None])[0] - ba(yc[None], hb[None])[0]
        out.update(_summ("ba", ptba, Dba))
        Dauc = np.array([auc(ycI[i], sa[I[i]]) - auc(ycI[i], sb[I[i]]) for i in range(n_boot)])
        out.update(_summ("auc", auc(yc, sa) - auc(yc, sb), Dauc))
    return out


def _summ(m, pt, D):
    D = D[np.isfinite(D)]
    lo, hi = np.percentile(D, [2.5, 97.5])
    return {f"d_{m}": pt, f"d_{m}_lo": lo, f"d_{m}_hi": hi, f"p_{m}": float(np.mean(D <= 0))}


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(1.0, run)
    return adj


# ---------------------------------------------------------------------------
# Configuration grid and baselines
# ---------------------------------------------------------------------------
MODEL_CLASSES = {  # target -> [(model, feature set)] ; linear = HAR-style, plus one tree model
    "spy_vol": [("lin", "harx"), ("hgb", "full")],
    "tlt_vol": [("lin", "harx"), ("hgb", "full")],
    "spy_tlt_corr": [("lin", "hc"), ("hgb", "full")],
}


def baselines(target):
    if target == "spy_vol":
        return {"persist_5": {"method": "raw", "col": "lrv5"}, "persist_20": {"method": "raw", "col": "lrv20"},
                "vix_raw": {"method": "raw", "col": "lvix"},
                "vix_lin_fixed": {"model": "lin", "feats": "vix", "method": "fixed"},
                "har_fixed": {"model": "lin", "feats": "har", "method": "fixed"},
                "harx_fixed": {"model": "lin", "feats": "harx", "method": "fixed"}}
    if target == "tlt_vol":
        return {"persist_5": {"method": "raw", "col": "lrv5"}, "persist_20": {"method": "raw", "col": "lrv20"},
                "har_fixed": {"model": "lin", "feats": "har", "method": "fixed"},
                "harx_fixed": {"model": "lin", "feats": "harx", "method": "fixed"}}
    if target == "spy_tlt_corr":
        return {"persist_21": {"method": "raw", "col": "c21"}, "persist_63": {"method": "raw", "col": "c63"},
                "longrun_mean": {"method": "const"},
                "hc_fixed": {"model": "lin", "feats": "hc", "method": "fixed"}}
    raise ValueError(target)


def grid(target):
    specs = []
    for model, fs in MODEL_CLASSES[target]:
        b = {"model": model, "feats": fs}
        specs.append(b | {"method": "fixed"})
        for R in ([5, 21, 63, 252] if model == "lin" else [21, 63, 252]):
            specs.append(b | {"method": "refit", "R": R, "W": None})
        for W in (252, 504, 1008, 1512):
            specs.append(b | {"method": "refit", "R": 21, "W": W})
        for hl in (126, 252, 504, 1008):
            specs.append(b | {"method": "refit", "R": 21, "W": None, "hl": hl})
        if model == "lin":
            for lam in (0.99, 0.995, 0.998, 0.999):
                specs.append(b | {"method": "rls", "lam": lam})
        for d in (0.002, 0.05):
            specs.append(b | {"method": "detect", "detector": "adwin", "delta": d})
        for lam in (1.0, 3.0):
            specs.append(b | {"method": "detect", "detector": "ph", "lam": lam})
        for hl in (63, 252):
            for eta in (2.0, 10.0):
                specs.append(b | {"method": "ens", "hl": hl, "eta": eta})
        for N in (21, 63, 126, 252):
            specs.append(b | {"method": "bias", "N": N})
        for N in (252, 504):
            specs.append(b | {"method": "recal", "N": N})
    return specs


def family(spec):
    m = spec["method"]
    if m == "refit":
        if spec.get("hl"):
            return "decay"
        return "rolling" if spec.get("W") else "expanding"
    return m

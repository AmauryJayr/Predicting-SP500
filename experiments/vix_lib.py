"""VIX as a drift / regime indicator: three uses suggested by the literature.

M1  VIX-triggered retraining (label-free drift trigger)
    Refit the base model whenever the VIX regime flips (1-year percentile rank crosses a cut, either
    direction), at least `min_gap` sessions apart. The VIX is known at each close, so unlike ADWIN /
    Page-Hinkley the trigger does not wait for delayed labels.
M2  Regime-conditional model selection
    Experts produce out-of-sample walk-forward predictions. At each decision date, use the expert with
    the lowest Brier score on past *resolved* predictions made in the same VIX regime (lookback L).
M3  Volatility-managed exposure (Moreira & Muir 2017)
    Hold SPY with weight w_t = min(cap, target_var / var_hat_t), decided at the close of t and held from
    the open of t+1 to the open of t+2. Uses the VIX for risk, not for direction.

Leakage: regime variables and variance estimates use closes <= t; models fit only on rows whose
label_end <= date t; expert scores use only predictions whose label is resolved by t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import DATA_DIR, Dataset, fit_model, make_estimator, walk_forward

BASE = ("LogReg", {"C": 0.003}, "core")            # the base model of the earlier drift study
EXPERTS = {
    "always_up": None,
    "lr_core": ("LogReg", {"C": 0.003}, "core"),
    "lr_all": ("LogReg", {"C": 0.003}, "all"),
    "gbdt_core": ("GBDT", {"max_iter": 150, "learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 50}, "core"),
}
EXPERT_START = "2010-01-01"                         # out-of-sample expert history starts here
EXPERT_REFIT = 63


def vix_rank(ds: Dataset) -> np.ndarray:
    return ds.data["vix_pct_rank_252d"].to_numpy()


def regime_of(rank: np.ndarray, split: str) -> np.ndarray:
    if split == "median":
        return (rank >= 0.5).astype(int)
    if split == "tercile":
        return np.digitize(rank, [1 / 3, 2 / 3])
    raise ValueError(split)


def _frame(ds, start, end, proba, thr, attrs=None):
    d = pd.DataFrame({"y": ds.y[start:end], "proba": proba, "thr": thr}, index=ds.index[start:end])
    d["pred"] = (d["proba"] >= d["thr"]).astype(int)
    d.attrs.update(attrs or {})
    return d


# ---------------------------------------------------------------- baselines
def fixed_base(ds, start, end):
    name, params, feats = BASE
    df = walk_forward(ds, ds.X(feats), make_estimator(name, params), start, end, None)
    return _frame(ds, start, end, df["proba"].to_numpy(), df["thr"].to_numpy())


def scheduled_base(ds, start, end, window, every=63):
    name, params, feats = BASE
    df = walk_forward(ds, ds.X(feats), make_estimator(name, params), start, end, every, window=window)
    return _frame(ds, start, end, df["proba"].to_numpy(), df["thr"].to_numpy())


def always_up(ds, start, end):
    t = np.array([ds.y[:ds.n_known(i)].mean() for i in range(start, end)])
    return _frame(ds, start, end, t, t)


# ---------------------------------------------------------------- M1
def vix_trigger(ds, start, end, cut, window, min_gap=21):
    name, params, feats = BASE
    X, y, est = ds.X(feats), ds.y, make_estimator(name, params)
    hi = vix_rank(ds) >= cut

    def fit(t0):
        tr = np.arange(ds.n_known(t0))
        if window is not None:
            tr = tr[-window:]
        assert ds.label_end[tr].max() <= ds.dates[t0]
        return fit_model(est, X[tr], y[tr]), y[tr].mean()

    m, thr0 = fit(start)
    last, fires = start, 0
    proba, thr = np.empty(end - start), np.empty(end - start)
    for t in range(start, end):
        if t > start and hi[t] != hi[t - 1] and t - last >= min_gap:
            m, thr0 = fit(t)
            last, fires = t, fires + 1
        proba[t - start] = m.predict_proba(X[t:t + 1])[0, 1]
        thr[t - start] = thr0
    return _frame(ds, start, end, proba, thr, {"n_fires": fires})


# ---------------------------------------------------------------- M2
def expert_predictions(ds) -> dict:
    """Out-of-sample walk-forward predictions of every expert from EXPERT_START to the end of the data."""
    s = ds.pos(EXPERT_START)
    out = {}
    for k, spec in EXPERTS.items():
        if spec is None:
            out[k] = always_up(ds, s, ds.n)
        else:
            name, params, feats = spec
            df = walk_forward(ds, ds.X(feats), make_estimator(name, params), s, ds.n, EXPERT_REFIT)
            out[k] = _frame(ds, s, ds.n, df["proba"].to_numpy(), df["thr"].to_numpy())
    return out


def regime_select(ds, E, start, end, split, lookback, use_regime=True, min_rows=30):
    s0 = ds.pos(EXPERT_START)
    keys = list(E)
    P = np.column_stack([E[k]["proba"].to_numpy() for k in keys])      # rows = positions s0..n-1
    T = np.column_stack([E[k]["thr"].to_numpy() for k in keys])
    sq = (P - ds.y[s0:, None]) ** 2
    reg = regime_of(vix_rank(ds), split)
    proba, thr, chosen = np.empty(end - start), np.empty(end - start), []
    for t in range(start, end):
        nk = ds.n_known(t)                                              # resolved rows: positions < nk
        lo = s0 if lookback is None else max(s0, nk - lookback)
        rows = np.arange(lo, nk)
        if use_regime:
            same = rows[reg[rows] == reg[t]]
            rows = same if len(same) >= min_rows else rows
        j = int(np.argmin(sq[rows - s0].mean(0)))
        proba[t - start], thr[t - start] = P[t - s0, j], T[t - s0, j]
        chosen.append(keys[j])
    return _frame(ds, start, end, proba, thr, {"chosen": pd.Series(chosen).value_counts().to_dict()})


# ---------------------------------------------------------------- M3
def spy_daily():
    px = pd.read_pickle(DATA_DIR / "aligned_prices.pkl")
    spy, vix = px["SPY"], px["^VIX"]["Close"]
    o = spy["Open"]
    r_next = (o.shift(-2) / o.shift(-1) - 1)                            # open t+1 -> open t+2, decided at close t
    rv = np.log(spy["Close"]).diff().rolling(20).var()                  # daily variance, closes <= t
    var_hat = {"vix": (vix / 100) ** 2 / 252, "rv20": rv}
    return pd.DataFrame({"r": r_next, **{f"v_{k}": v for k, v in var_hat.items()}}).dropna()


def scale_constant(hist, signal, cap):
    """c such that min(cap, c / var_hat) * r has the same volatility as r on `hist` (Moreira & Muir), by bisection."""
    v, r, target = hist[f"v_{signal}"], hist["r"], hist["r"].std()
    lo, hi = 1e-8, 1.0
    for _ in range(100):
        c = (lo + hi) / 2
        lo, hi = (c, hi) if (np.minimum(cap, c / v) * r).std() < target else (lo, c)
    return c


def vol_managed(D, start, end, signal, cap, cost_bps):
    """Daily returns of the managed and buy-and-hold strategies on [start, end] (dates)."""
    c = scale_constant(D.loc[:pd.Timestamp(start) - pd.Timedelta(days=1)], signal, cap)
    d = D.loc[start:end]
    w = np.minimum(cap, c / d[f"v_{signal}"])
    turn = w.diff().abs().fillna(w.iloc[0])
    managed = w * d["r"] - cost_bps / 1e4 * turn
    bh = d["r"].copy()
    bh.iloc[0] -= cost_bps / 1e4
    return pd.DataFrame({"managed": managed, "bh": bh, "w": w})


def sharpe(r):
    return float(r.mean() / r.std() * np.sqrt(252))


def strat_stats(r):
    eq = (1 + r).cumprod()
    return {"ann_return": float(eq.iloc[-1] ** (252 / len(r)) - 1), "ann_vol": float(r.std() * np.sqrt(252)),
            "sharpe": sharpe(r), "max_drawdown": float((eq / eq.cummax() - 1).min())}


def sharpe_bootstrap(a, b, n_boot=2000, L=20, seed=42):
    """Sharpe(a) - Sharpe(b), paired moving-block bootstrap on daily returns."""
    a, b = a.to_numpy(), b.to_numpy()
    rng = np.random.default_rng(seed)
    n = len(a)
    f = lambda x: x.mean() / x.std() * np.sqrt(252)
    point = f(a) - f(b)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        st = rng.integers(0, n - L + 1, int(np.ceil(n / L)))
        idx = (st[:, None] + np.arange(L)).ravel()[:n]
        draws[i] = f(a[idx]) - f(b[idx])
    return {"d_sharpe": point, "lo": np.percentile(draws, 2.5), "hi": np.percentile(draws, 97.5),
            "p": float(np.mean(draws <= 0))}

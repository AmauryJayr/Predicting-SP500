"""Three ways to use the VIX on top of the frozen, zero-shot TimesFM-3 score, and a volatility-managed strategy.

Mirrors the team's VIX study (`experiments/vix_lib.py` on master) on the TimesFM score instead of the logistic
regression. TimesFM is never refitted; the cached forecasts of the official setup (variant `multi`, context 256)
are read through `timesfm_drift_lib.Inp`. The VIX regime variable is `vix_pct_rank_252d` (the 1-year percentile
rank of the VIX close), known at the close of day t, so no regime value looks ahead.

  V1  recal_vix   refit the calibration layer on the score only when the VIX regime flips (rank crosses a cut)
  V2  select      each day take the expert (static TimesFM, fixed LR, always-up) with the lowest Brier on past
                  resolved predictions made in the same VIX regime
  V3  stack_vix   logistic layer on [logit score, VIX rank, score x (rank - 0.5)]
  V5  vol_managed volatility-managed SPY exposure with three variance signals (TimesFM spread, VIX, rv20)

Every frame is [y, proba, thr, pred] indexed by date, as everywhere in the project. Labels are used only if
label_end <= decision date (`Dataset.n_known`, via `Inp.train_rows`, asserted).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

import timesfm_drift_lib as dl
import timesfm_lib as tl
from common import DATA_DIR, fit_model, make_estimator
from tf_paths import RES_DIR

MIN_GAP = 21                      # sessions between two trigger refits (the team's value)
SCHED_EVERY = 63                  # scheduled refits of the isolating comparison (the team's value)
MIN_ROWS = 30                     # fewer same-regime rows than this -> use all rows (the team's value)


# ---------------------------------------------------------------------------
# VIX regime (known at close t)
# ---------------------------------------------------------------------------
def rank_of(inp) -> np.ndarray:
    r = inp.ds.data["vix_pct_rank_252d"].to_numpy(dtype=float)
    assert np.isfinite(r[inp.p0:]).all()
    return r


def regime_of(rank, split):
    if split == "median":
        return (rank >= 0.5).astype(int)
    if split == "tercile":
        return np.digitize(rank, [1 / 3, 2 / 3])
    raise ValueError(split)


# ---------------------------------------------------------------------------
# V1: recalibration of the score, on a schedule or on a VIX trigger (shared loop)
# ---------------------------------------------------------------------------
def recal_range(inp, s, e, cal, N, every=None, cut=None, min_gap=MIN_GAP):
    """Calibration layer on the logit score over rows [s, e). `every`: refit every R rows (scheduled, as
    `dl.run_platt`); `cut`: refit when 1[rank >= cut] flips, at least `min_gap` rows after the last refit."""
    z, y = inp.z, inp.ds.y
    hi = None if cut is None else rank_of(inp) >= cut
    proba, thr = np.empty(e - s), np.empty(e - s)
    cur, thr_c, last, fires, slopes, fit_rows = None, None, None, 0, [], []
    for i, t in enumerate(range(s, e)):
        if cut is None:
            due = cur is None or i % every == 0
        else:
            due = cur is None or (hi[t] != hi[t - 1] and t - last >= min_gap)
        if due:
            idx = inp.train_rows(t, N)
            c = dl.fit_cal(z[idx], y[idx], cal)
            if c is not None:
                if cur is not None:
                    fires += 1
                cur, thr_c, last = c, float(y[idx].mean()), t
                slopes.append(c[1])
                fit_rows.append(t)
        proba[i], thr[i] = expit(cur[0] + cur[1] * z[t]), thr_c
    return proba, thr, {"n_refits": fires, "neg_slope_share": float(np.mean(np.array(slopes) < 0)),
                        "mean_slope": float(np.mean(slopes)), "fit_rows": fit_rows}


def run_recal(inp, spec, period, scheduled=False):
    s, e = inp.span(period)
    if scheduled:
        p, t, a = recal_range(inp, s, e, spec["cal"], spec["N"], every=SCHED_EVERY)
    else:
        p, t, a = recal_range(inp, s, e, spec["cal"], spec["N"], cut=spec["cut"])
    out = inp.frame(s, e, p, t)
    out.attrs.update(a)
    return out


# ---------------------------------------------------------------------------
# V2: expert selection (experts: TimesFM, fixed LR, always-up)
# ---------------------------------------------------------------------------
EXPERTS = ("tfm", "lr", "up")


def first_expert_row(inp):
    """First row with at least MIN_TRAIN known TimesFM-scored labels (when the walk-forward TimesFM expert starts)."""
    t = inp.p0
    while inp.ds.n_known(t) - inp.p0 < dl.MIN_TRAIN:
        t += 1
    return t


def expert_table(inp, periods):
    """Out-of-sample probabilities and thresholds of the three experts on rows [s0, end of the last period).

    tfm: offset-only calibration of the score on all known rows, refit every 63 rows (walk-forward, expanding);
    lr : the team's fixed LR, trained once at the start of each period F1, F2, VAL, TEST (frames of `Inp.fixed_lr`);
    up : always-up, threshold = base rate of the labels known at t.
    """
    ds = inp.ds
    s0 = first_expert_row(inp)
    e0 = inp.span(periods[-1])[1]
    ptf, ttf, _ = recal_range(inp, s0, e0, "offset", None, every=SCHED_EVERY)
    lr = pd.concat([inp.fixed_lr(p) for p in periods])
    assert lr.index.equals(ds.index[inp.span(periods[0])[0]:e0])
    k0 = inp.span(periods[0])[0]
    assert k0 <= s0
    plr, tlr = lr["proba"].to_numpy()[s0 - k0:], lr["thr"].to_numpy()[s0 - k0:]
    rate = np.array([ds.y[:ds.n_known(t)].mean() for t in range(s0, e0)])
    P = np.column_stack([ptf, plr, rate])
    T = np.column_stack([ttf, tlr, rate])
    return {"s0": s0, "e0": e0, "P": P, "T": T}


def run_select(inp, X, spec, period, use_regime=True):
    """Expert with the lowest mean Brier on resolved same-regime predictions (lookback L rows, None = all)."""
    s, e = inp.span(period)
    s0, P, T = X["s0"], X["P"], X["T"]
    y = inp.ds.y
    sq = (P - y[s0:X["e0"], None]) ** 2
    reg = regime_of(rank_of(inp), spec["split"])
    proba, thr, chosen = np.empty(e - s), np.empty(e - s), []
    for t in range(s, e):
        nk = inp.ds.n_known(t)
        lo = s0 if spec["lookback"] is None else max(s0, nk - spec["lookback"])
        rows = np.arange(lo, nk)
        rows = rows[np.isfinite(sq[rows - s0]).all(axis=1)]            # the LR column starts at F1
        if use_regime:
            same = rows[reg[rows] == reg[t]]
            rows = same if len(same) >= MIN_ROWS else rows
        j = int(np.argmin(sq[rows - s0].mean(0)))
        proba[t - s], thr[t - s] = P[t - s0, j], T[t - s0, j]
        chosen.append(EXPERTS[j])
    out = inp.frame(s, e, proba, thr)
    out.attrs.update(chosen=pd.Series(chosen).value_counts().to_dict(), n_refits=0)
    return out


# ---------------------------------------------------------------------------
# V3: stacking with VIX terms (the no-VIX twin is dl.run_stack with feats "z")
# ---------------------------------------------------------------------------
def run_stack_vix(inp, spec, period):
    s, e = inp.span(period)
    rk = rank_of(inp)
    X = np.column_stack([inp.z, rk, inp.z * (rk - 0.5)])
    est = make_estimator("LogReg", {"C": spec["C"]})
    y = inp.ds.y
    R = None if spec["mode"] == "fixed" else spec["R"]
    N = spec.get("W") if spec["mode"] == "rolling" else None
    blocks = [(s, np.arange(s, e))] if R is None else [(b, np.arange(b, min(b + R, e))) for b in range(s, e, R)]
    proba, thr, co = np.empty(e - s), np.empty(e - s), []
    for t0, rows in blocks:
        tr = inp.train_rows(t0, N)
        m = fit_model(est, X[tr], y[tr])
        proba[rows - s] = m.predict_proba(X[rows])[:, 1]
        thr[rows - s] = y[tr].mean()
        co.append(m[-1].coef_[0] / m[0].scale_)                        # coefficients in raw units
    co = np.array(co)
    out = inp.frame(s, e, proba, thr)
    out.attrs.update(n_refits=len(blocks) - 1, mean_slope=float(co[:, 0].mean()),
                     neg_slope_share=float((co[:, 0] < 0).mean()), mean_coef_rank=float(co[:, 1].mean()),
                     mean_coef_interaction=float(co[:, 2].mean()))
    return out


def run(inp, family, spec, period):
    if family == "recal":
        return run_recal(inp, spec, period)
    if family == "select":
        return run_select(inp, inp._X, spec, period)
    if family == "stack":
        return run_stack_vix(inp, spec, period)
    raise ValueError(family)


def isolating(inp, family, spec, period):
    """The comparison that removes the VIX: scheduled refits / regime-blind selection / stack without VIX terms."""
    if family == "recal":
        return run_recal(inp, spec, period, scheduled=True)
    if family == "select":
        return run_select(inp, inp._X, spec, period, use_regime=False)
    return dl.run_stack(inp, {"feats": "z", **{k: v for k, v in spec.items()}}, period)


def grid():
    G = {"recal": [], "select": [], "stack": []}
    for cal in ("platt", "offset"):
        for N in (252, 504):
            for cut in (0.8, 0.9):
                G["recal"].append({"cal": cal, "N": N, "cut": cut})
    for split in ("median", "tercile"):
        for L in (504, None):
            G["select"].append({"split": split, "lookback": L})
    for C in (0.03, 0.3):
        G["stack"].append({"C": C, "mode": "fixed"})
        G["stack"].append({"C": C, "mode": "expanding", "R": 21})
        for W in (252, 504):
            G["stack"].append({"C": C, "mode": "rolling", "W": W, "R": 21})
    return G


# ---------------------------------------------------------------------------
# V5: volatility-managed exposure (Moreira & Muir 2017; logic of the team's vix_lib.vol_managed)
# ---------------------------------------------------------------------------
SIGNALS = ("tfm_spread", "vix", "rv20")
HIST_START = "2013-01-01"         # first date with a TimesFM forecast; the scale constant uses only rows from here on


def tfm_spread_series() -> pd.Series:
    """TimesFM's predicted spread at step 1: q90 - q10 of the log close, from the official cache (past-only)."""
    d, q, _ = tl.read_cache(dl.VARIANT, dl.CTX)
    return pd.Series(q[:, 0, 8] - q[:, 0, 0], index=d)


def daily_frame() -> pd.DataFrame:
    """Next-day return r (open t+1 -> open t+2, decided at close t) and the three variance estimates at close t."""
    px = pd.read_pickle(DATA_DIR / "aligned_prices.pkl")
    spy, vix = px["SPY"], px["^VIX"]["Close"]
    o = spy["Open"]
    sp = tfm_spread_series().reindex(spy.index)
    sigma = sp / (2 * tl.Z90)                                     # normal approximation: q90 - q10 = 2 * 1.2816 sigma
    D = pd.DataFrame({"r": o.shift(-2) / o.shift(-1) - 1,
                      "v_tfm_spread": sigma ** 2,
                      "v_vix": (vix / 100) ** 2 / 252,
                      "v_rv20": np.log(spy["Close"]).diff().rolling(20).var()})
    return D.loc[HIST_START:].dropna()


def scale_constant(hist, signal, cap):
    """c such that min(cap, c / var_hat) * r has the volatility of r on `hist`, by bisection."""
    v, r, target = hist[f"v_{signal}"], hist["r"], hist["r"].std()
    lo, hi = 1e-10, 1.0
    for _ in range(100):
        c = (lo + hi) / 2
        lo, hi = (c, hi) if (np.minimum(cap, c / v) * r).std() < target else (lo, c)
    return c


def vol_managed(D, start, end, signal, cap, cost_bps):
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
    """Sharpe(a) - Sharpe(b), paired moving-block bootstrap on daily returns; one-sided p = P(draw <= 0)."""
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    n = len(a)
    f = lambda x: x.mean() / x.std() * np.sqrt(252)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        st = rng.integers(0, n - L + 1, int(np.ceil(n / L)))
        idx = (st[:, None] + np.arange(L)).ravel()[:n]
        draws[i] = f(a[idx]) - f(b[idx])
    return {"d_sharpe": f(a) - f(b), "lo": float(np.percentile(draws, 2.5)), "hi": float(np.percentile(draws, 97.5)),
            "p": float(np.mean(draws <= 0))}

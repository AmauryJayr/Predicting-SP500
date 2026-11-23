"""ROOT CAUSE, experiment B: simple NON-transformer forecasters through the identical pipeline
   forecast (mean path + residual-volatility normal quantiles) -> official P(up) read-out -> AUC,
on the same 1,759 test rows, using only the last 256 log-closes up to and including day t.
statsmodels is not installed in this env, so AR / ETS are small own implementations (least squares / scipy).
Writes results/timesfm_rootcause_baselines.csv (AUC pooled + per year, bootstrap CI) and the cached scores.
Run:  python timesfm_rootcause_baselines.py     (about 1-2 min)
"""
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

import timesfm_lib as tl
from tf_paths import RES_DIR
from timesfm_rootcause_lib import test_dates, anchor, score_from_q, evaluate, HS

L, H = 256, 5
Z = norm.ppf(tl.QLEVELS)
T = np.arange(-(L - 1), 1) / L                      # same time axis as the model's own detrending line


def q_from(mu, sd):
    """mu, sd: arrays (H,) -> (H, 9) normal quantiles."""
    return mu[:, None] + sd[:, None] * Z[None, :]


def line_fit(x):
    tm = T.mean()
    m = ((T - tm) * (x - x.mean())).sum() / ((T - tm) ** 2).sum()
    c = x.mean() - m * tm                          # line value at the last point (t = 0)
    return m, c, x - (m * T + c)


def f_rwd(x):                                       # 1. random walk with drift (endpoint drift of the window)
    r = np.diff(x); k = np.arange(1, H + 1)
    return q_from(x[-1] + k * r.mean(), r.std(ddof=1) * np.sqrt(k))


def f_line_slope(x):                                # 2a. the detrending line's SLOPE only, anchored at today's close
    m, c, res = line_fit(x); r = np.diff(x); k = np.arange(1, H + 1)
    return q_from(x[-1] + k * m / L, r.std(ddof=1) * np.sqrt(k))


def f_line_anchored(x):                             # 2b. the whole line extrapolated (anchored on the fitted line, not on today)
    m, c, res = line_fit(x); k = np.arange(1, H + 1)
    return q_from(c + k * m / L, np.full(H, res.std(ddof=2)))


def f_ar5(x, p=5):                                  # 3a. AR(p) on daily log returns, OLS with intercept, iterated
    r = np.diff(x)
    X = np.column_stack([np.ones(len(r) - p)] + [r[p - j - 1:len(r) - j - 1] for j in range(p)])
    b, *_ = np.linalg.lstsq(X, r[p:], rcond=None)
    s = (r[p:] - X @ b).std(ddof=p + 1)
    hist = list(r[-p:][::-1]); fr = []
    for _ in range(H):
        v = b[0] + np.dot(b[1:], hist[:p]); fr.append(v); hist.insert(0, v)
    psi = [1.0]
    for j in range(1, H):
        psi.append(sum(b[1 + i] * psi[j - 1 - i] for i in range(min(p, j))))
    cum = np.cumsum(psi)                            # cumulative-return error weights
    var = np.array([s ** 2 * (cum[:h][::-1] ** 2).sum() for h in range(1, H + 1)])
    return q_from(x[-1] + np.cumsum(fr), np.sqrt(var))


def f_ar1_level(x):                                 # 3b. AR(1) on log-price LEVELS (mean-reverts to the window mean)
    X = np.column_stack([np.ones(L - 1), x[:-1]])
    b, *_ = np.linalg.lstsq(X, x[1:], rcond=None)
    s = (x[1:] - X @ b).std(ddof=2); phi = b[1]
    mu, v = x[-1], []
    for _ in range(H):
        mu = b[0] + phi * mu; v.append(mu)
    k = np.arange(1, H + 1)
    var = np.array([s ** 2 * sum(phi ** (2 * i) for i in range(h)) for h in k])
    return q_from(np.array(v), np.sqrt(var))


def f_line_ar1(x):                                  # 3c. line + AR(1) on the residual around it ("detrend, then a simple model")
    m, c, res = line_fit(x)
    X = np.column_stack([np.ones(L - 1), res[:-1]])
    b, *_ = np.linalg.lstsq(X, res[1:], rcond=None)
    s = (res[1:] - X @ b).std(ddof=2); phi = b[1]
    e, v = res[-1], []
    for _ in range(H):
        e = b[0] + phi * e; v.append(e)
    k = np.arange(1, H + 1)
    var = np.array([s ** 2 * sum(phi ** (2 * i) for i in range(h)) for h in k])
    return q_from(c + k * m / L + np.array(v), np.sqrt(var))


def _ets_filter(par, x):
    al, be, ph = par
    l, b = x[0], x[1] - x[0]; sse = 0.0
    for v in x[1:]:
        f = l + ph * b; e = v - f; sse += e * e
        ln = f + al * e; b = ph * b + be * e; l = ln
    return sse, l, b


def f_ets(x):                                       # 3d. ETS(A,Ad,N) = Holt damped trend on log price (own scipy fit)
    xs = (x - x[0]) * 100                           # scale for the optimiser
    best = None
    for p0 in ((0.3, 0.05, 0.9), (0.8, 0.1, 0.98)):
        r = minimize(lambda p: _ets_filter(p, xs)[0], p0, method="L-BFGS-B",
                     bounds=[(0.01, 0.99), (0.0, 0.5), (0.8, 0.98)])
        if best is None or r.fun < best.fun:
            best = r
    sse, l, b = _ets_filter(best.x, xs)
    ph = best.x[2]; k = np.arange(1, H + 1)
    damp = np.cumsum(ph ** k)
    s = np.sqrt(sse / (len(xs) - 1)) / 100
    return q_from(x[0] + (l + damp * b) / 100 * 1.0, s * np.sqrt(k))


MODELS = {"random_walk_drift": f_rwd, "line_slope_only": f_line_slope, "line_anchored_extrap": f_line_anchored,
          "AR5_returns": f_ar5, "AR1_levels": f_ar1_level, "line_plus_AR1_resid": f_line_ar1, "ETS_damped_trend": f_ets}


def main():
    dates = test_dates()
    p = tl.panel()
    pos = p["dates"].get_indexer(dates)
    x = p["spy"]
    a = anchor(dates)
    sc = {nm: {h: np.zeros(len(dates)) for h in HS} for nm in MODELS}
    for nm, f in MODELS.items():
        t0 = time.time()
        qs = np.stack([f(x[i - L + 1:i + 1]) for i in pos])
        assert np.isfinite(qs).all()
        for h in HS:
            sc[nm][h] = score_from_q(qs, a, h)
        print(f"{nm}: {time.time() - t0:.1f}s", flush=True)
    qo, _ = tl.aligned("multi", 256, dates)
    scores = {"official_TimesFM": {h: score_from_q(qo, a, h) for h in HS}, **sc}
    df = evaluate(scores, dates)
    df.to_csv(RES_DIR / "timesfm_rootcause_baselines.csv", index=False)
    pd.DataFrame({f"{nm}_h{h}": v[h] for nm, v in scores.items() for h in HS}, index=dates).to_csv(
        RES_DIR / "timesfm_rootcause_baseline_scores.csv.gz")
    with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
        print(df[["model", "h", "auc_pooled", "ci_lo", "ci_hi"] + [f"auc_{y}" for y in (2019, 2020, 2021, 2022, 2023, 2024, 2025)]].to_string(index=False))


if __name__ == "__main__":
    main()

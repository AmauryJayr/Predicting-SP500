"""EXPLORATORY (post-test, h=5 only): is TimesFM-3's bet on a turn what made 2020, 2022 and 2025 bad?
Splits the forecast into the part explained by SPY's past moves ("turn part") and the rest, and scores each
part per year and over the falls / recoveries of timesfm_explore_years.STRETCHES. Uses saved forecasts only.

Usage:  python timesfm_explore_split.py  -> results/timesfm_split_years.csv
                                            results/timesfm_split_stretches.csv

  official  the pre-registered tfm.h5 TEST score (multi, ctx 256); turn part = linear fit of P(up) on r20.
            Also a 95% moving-block bootstrap interval of each year's AUC (20-day blocks, 1000 draws).
  price     SPY-only forecast, ctx 256, detrending as shipped (results/timesfm_cache_detrend/default_price_ctx256.npz);
            d5 = median forecast of logC[t+5] - logC[t].
            turn part = fit of d5 on the moves over days 1 / 2-5 / 6-10 / 11-20 back, each also times vr and vr^2
            (vr = 20-day vol / 255-day vol), plus vr.  `long` adds days 21-60 and 61-255 back the same way.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import timesfm_lib as tl
from tf_paths import RES_DIR
from common import DATA_DIR
from timesfm_explore_years import frame, STRETCHES, POOLS
from timesfm_verify_detrend import labels

SEGS = [(0, 1), (1, 5), (5, 10), (10, 20)]
LONG = SEGS + [(20, 60), (60, 255)]


def split(t, target, cols):
    """Least-squares fit of `target` on cols (+ intercept) -> (fitted, rest, R2)."""
    X = np.column_stack([np.ones(len(t))] + [t[c] for c in cols])
    fit = X @ np.linalg.lstsq(X, t[target], rcond=None)[0]
    rest = t[target] - fit
    return fit, rest, 1 - rest.var() / t[target].var()


def block_ci(y, s, rng, B=1000, b=20):
    n = len(y)
    k = int(np.ceil(n / b))
    out = []
    for _ in range(B):
        st = rng.integers(0, n - b, k)
        i = (st[:, None] + np.arange(b)).ravel()[:n]
        if 0 < y[i].sum() < n:
            out.append(roc_auc_score(y[i], s[i]))
    return np.percentile(out, [2.5, 97.5])


def price_frame(ls):
    z = np.load(RES_DIR / "timesfm_cache_detrend" / "default_price_ctx256.npz", allow_pickle=True)
    d = pd.DatetimeIndex(z["dates"])
    q = z["q"]
    anc = ls.reindex(d).to_numpy()
    t = pd.DataFrame({"pup": [tl.p_up(q[i], "price", 5, anc[i]) for i in range(len(d))],
                      "d5": q[:, 4, 4] - anc}, index=d)
    r1 = ls.diff()
    t["vr"] = (r1.rolling(20).std() / r1.rolling(255).std()).reindex(d)
    for a, b in LONG:
        c = f"s{a + 1}_{b}"
        t[c] = (ls.shift(a) - ls.shift(b)).reindex(d)
        t[c + "_x"] = t[c] * t["vr"]
        t[c + "_x2"] = t[c] * t["vr"] ** 2
    y = labels(5)
    t = t.loc[y.index]
    t["y"] = y
    return t


def cols_for(segs):
    return [f"s{a + 1}_{b}{e}" for a, b in segs for e in ("", "_x", "_x2")] + ["vr"]


def main():
    off, close = frame()
    ls = np.log(close)
    off["turn"], off["rest"], r2_off = split(off, "proba", ["r20"])
    pr = price_frame(ls)
    pr["turn"], pr["rest"], r2_pr = split(pr, "d5", cols_for(SEGS))
    _, pr["rest_long"], r2_long = split(pr, "d5", cols_for(LONG))
    print(f"R2 of the turn part: official {r2_off:.2f}, price {r2_pr:.2f}, price long {r2_long:.2f}")

    rng = np.random.default_rng(0)
    rows = []
    for yr, g in off.groupby(off.index.year):
        lo, hi = block_ci(g["y"].to_numpy(), g["proba"].to_numpy(), rng)
        p = pr[pr.index.year == yr]
        rows.append({"year": yr, "n": len(g), "official_auc": roc_auc_score(g["y"], g["proba"]), "official_ci_lo": lo,
                     "official_ci_hi": hi, "official_turn_auc": roc_auc_score(g["y"], g["turn"]),
                     "official_rest_auc": roc_auc_score(g["y"], g["rest"]),
                     "price_auc": roc_auc_score(p["y"], p["pup"]), "price_turn_auc": roc_auc_score(p["y"], p["turn"]),
                     "price_rest_auc": roc_auc_score(p["y"], p["rest"]),
                     "price_rest_long_auc": roc_auc_score(p["y"], p["rest_long"])})
    rows.append({"year": "all", "n": len(off), "official_auc": roc_auc_score(off["y"], off["proba"]),
                 "official_turn_auc": roc_auc_score(off["y"], off["turn"]),
                 "official_rest_auc": roc_auc_score(off["y"], off["rest"]),
                 "price_auc": roc_auc_score(pr["y"], pr["pup"]), "price_turn_auc": roc_auc_score(pr["y"], pr["turn"]),
                 "price_rest_auc": roc_auc_score(pr["y"], pr["rest"]),
                 "price_rest_long_auc": roc_auc_score(pr["y"], pr["rest_long"]),
                 "r2_official": r2_off, "r2_price": r2_pr, "r2_price_long": r2_long})
    Y = pd.DataFrame(rows)
    Y.to_csv(RES_DIR / "timesfm_split_years.csv", index=False)

    srows = []
    spans = [(lab, kind, [(a, e)]) for lab, kind, a, e, _ in STRETCHES]
    span = {lab: sp[0] for lab, _, sp in spans}
    spans += [(lab, kind, [span[m] for m in members]) for lab, kind, members in POOLS]
    for lab, kind, sp in spans:
        g = pd.concat([pr.loc[a:e] for a, e in sp])
        srows.append({"stretch": lab, "kind": kind, "n": len(g), "up_rate": g["y"].mean(),
                      "turn_points_up": (g["turn"] > 0).mean(), "rest_pushes_up": (g["rest"] > 0).mean(),
                      "whole_points_up": (g["d5"] > 0).mean()})
    S = pd.DataFrame(srows)
    S.to_csv(RES_DIR / "timesfm_split_stretches.csv", index=False)
    pd.set_option("display.width", 250)
    print(Y.round(3).to_string(index=False))
    print(S.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

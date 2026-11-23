"""Shared helpers for the root-cause study (labels, AUC pooled / per year, moving-block bootstrap)."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import timesfm_lib as tl
from timesfm_verify_detrend import test_dates, labels   # noqa: F401  (re-exported)

HS = (1, 5)
YEARS = list(range(2019, 2026))
BAD = (2020, 2022, 2025)
BLOCK, NBOOT, SEED = 20, 1000, 7


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    n1 = int((y == 1).sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def anchor(dates):
    p = tl.panel()
    return pd.Series(p["spy"], index=p["dates"]).reindex(dates).to_numpy()


def score_from_q(q, a, h):
    """The official pipeline's P(up): step-h CDF interpolated through the 9 quantiles, clipped to [0.05, 0.95]."""
    return np.array([tl.p_up(q[i], "multi", h, a[i]) for i in range(len(q))])


def boot_idx(n, rng):
    """Moving-block bootstrap indices (block BLOCK) for n consecutive days; one (NBOOT, n) matrix, reusable."""
    nb = -(-n // BLOCK)
    st = rng.integers(0, n - BLOCK + 1, size=(NBOOT, nb))
    return (st[:, :, None] + np.arange(BLOCK)).reshape(NBOOT, -1)[:, :n]


def evaluate(scores: dict, dates, label_extra=None):
    """scores: name -> {h: array over `dates` (the 1759 test days)}.  Returns a long DataFrame with pooled and
    per-year AUC, pooled bootstrap CI and (for every score) the paired difference to the first score."""
    rows = []
    first = next(iter(scores))
    for h in HS:
        y = labels(h)
        idx = np.asarray(dates.get_indexer(y.index))
        yy = y.to_numpy().astype(int)
        yr = y.index.year.to_numpy()
        B = boot_idx(len(yy), np.random.default_rng(SEED + h))
        ref = np.asarray(scores[first][h], float)[idx]
        ref_b = np.array([auc(yy[b], ref[b]) for b in B])
        for nm, d in scores.items():
            s = np.asarray(d[h], float)[idx]
            a_b = np.array([auc(yy[b], s[b]) for b in B])
            row = dict(model=nm, h=h, n=len(yy), auc_pooled=auc(yy, s),
                       ci_lo=np.nanpercentile(a_b, 2.5), ci_hi=np.nanpercentile(a_b, 97.5),
                       d_vs_chance_lo=np.nanpercentile(a_b - 0.5, 2.5), d_vs_chance_hi=np.nanpercentile(a_b - 0.5, 97.5),
                       d_vs_first=auc(yy, s) - auc(yy, ref),
                       d_vs_first_lo=np.nanpercentile(a_b - ref_b, 2.5), d_vs_first_hi=np.nanpercentile(a_b - ref_b, 97.5))
            for y_ in YEARS:
                m = yr == y_
                row[f"auc_{y_}"] = auc(yy[m], s[m])
            rows.append(row)
    return pd.DataFrame(rows)

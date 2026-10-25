"""AUDIT helpers: paired moving-block bootstrap with configurable block length (same scheme and sign
conventions as common.paired_bootstrap: positive = A better, Brier sign flipped, one-sided p = P(draw <= 0)),
plus a rank-based AUC for speed. Verified against common.paired_bootstrap in audit_robustness.py."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def auc(y, p):
    r = rankdata(p)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def scores(y, p, yhat):
    ba = 0.5 * (yhat[y == 1].mean() + (1 - yhat[y == 0]).mean())
    return np.array([ba, auc(y, p), np.mean((p - y) ** 2)])


def block_indices(n, L, rng):
    starts = rng.integers(0, n - L + 1, int(np.ceil(n / L)))
    return (starts[:, None] + np.arange(L)).ravel()[:n]


def boot(a: pd.DataFrame, b: pd.DataFrame, L=20, n_boot=2000, seed=42):
    assert a.index.equals(b.index)
    y = a["y"].to_numpy(); pa = a["proba"].to_numpy(); ya = a["pred"].to_numpy()
    pb = b["proba"].to_numpy(); yb = b["pred"].to_numpy()
    sign = np.array([1, 1, -1])
    point = sign * (scores(y, pa, ya) - scores(y, pb, yb))
    rng = np.random.default_rng(seed)
    D = np.empty((n_boot, 3))
    for i in range(n_boot):
        idx = block_indices(len(y), L, rng)
        D[i] = sign * (scores(y[idx], pa[idx], ya[idx]) - scores(y[idx], pb[idx], yb[idx]))
    out = {}
    for j, m in enumerate(["BA", "AUC", "Brier"]):
        out[f"d{m}"] = point[j]
        out[f"d{m}_lo"], out[f"d{m}_hi"] = np.percentile(D[:, j], [2.5, 97.5])
        out[f"p{m}"] = float(np.mean(D[:, j] <= 0))
    return out


def chance_like(p: pd.DataFrame, rate: float) -> pd.DataFrame:
    return pd.DataFrame({"y": p["y"], "proba": rate, "thr": rate, "pred": 1}, index=p.index)


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(1, run)
    return adj

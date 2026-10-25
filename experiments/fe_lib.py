"""Feature-engineering / feature-selection study: dataset, selectors and walk-forward runner.

Leakage rules (same as common.py):
  * a model fitted at decision row t0 sees only rows with label_end <= date[t0];
  * every selector is fitted on that same training set only (it never sees the
    evaluation rows), and re-fitted at every refit in the 'refit' scheme;
  * all features are trailing computations (fe_features.truncation_test).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (CORE_FEATURES, DATA_DIR, PERIODS, SEED, STUDY_END, STUDY_START,
                    block_indices, block_len, metrics, paired_bootstrap)
from fe_features import build_all, load_extra

# Chosen on economic grounds before any result of this study was seen.
CORE_PLUS = CORE_FEATURES + ["vs_vix_vix3m_ratio", "vs_skew_z60", "cr_hyg_minus_ief_ret20", "rt_slope_10y_3m",
                             "rt_10y_chg20", "br_rsp_minus_spy_ret20", "st_volscaled_ret5",
                             "st_volscaled_ret20", "gl_usd_ret20", "st_variance_ratio_5"]


class FEDataset:
    def __init__(self, horizon: int):
        self.h = horizon
        px = pd.read_pickle(DATA_DIR / "aligned_prices.pkl")
        cal = px["SPY"].index
        ex = load_extra(cal)
        F, fam = build_all(px, ex)
        spy = px["SPY"]
        fwd = spy["Close"].shift(-horizon) / spy["Open"].shift(-1) - 1
        y = (fwd > 0).astype(float).where(fwd.notna())
        label_end = pd.Series(cal, index=cal).shift(-horizon)
        d = F.copy()
        d["target"], d["label_end"] = y, label_end
        d = d.loc[STUDY_START:STUDY_END]
        d = d[~(d[F.columns].isna().any(axis=1) | d["target"].isna())]
        self.F = d[F.columns].astype(float)
        self.cols = list(F.columns)
        self.base_cols = [c for c in self.cols if not any(c in v for k, v in fam.items()
                                                          if k in ["volsurface", "rates", "credit", "global",
                                                                   "breadth", "structure", "calendar2"])]
        self.fam = fam
        self.y = d["target"].astype(int).to_numpy()
        self.dates = d.index.to_numpy()
        self.index = d.index
        self.label_end = d["label_end"].to_numpy()
        self.years = d.index.year.to_numpy()
        self.n = len(d)

    def n_known(self, t: int) -> int:
        return int(np.searchsorted(self.label_end, self.dates[t], side="right"))

    def period(self, name: str):
        a, b = PERIODS[name]
        return (int(np.searchsorted(self.dates, np.datetime64(pd.Timestamp(a)))),
                int(np.searchsorted(self.dates, np.datetime64(pd.Timestamp(b)), side="right")))


# ---------------------------------------------------------------------------
# Selectors: fit(ds, train_rows) -> transform(rows) -> ndarray
# ---------------------------------------------------------------------------
class ColumnSet:
    def __init__(self, ds, cols):
        self.cols = list(cols)
        self.ix = [ds.cols.index(c) for c in self.cols]
        self.X = ds.F.to_numpy()

    def __call__(self, rows):
        return self.X[np.ix_(rows, self.ix)]


def _spearman_with_target(X, y):
    Xr = pd.DataFrame(X).rank().to_numpy()
    yr = pd.Series(y).rank().to_numpy()
    Xc, yc = Xr - Xr.mean(0), yr - yr.mean()
    return (Xc * yc[:, None]).sum(0) / (np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum()) + 1e-12)


def sel_l1_stability(ds, tr, cols, thr=0.6, B=30, C=0.2):
    """Stability selection: L1-logistic on B contiguous-block half-samples; keep features chosen >= thr."""
    rng = np.random.default_rng(SEED)
    X = ds.F[cols].to_numpy()[tr]
    y = ds.y[tr]
    X = StandardScaler().fit_transform(X)
    L = max(20, 3 * ds.h)
    counts = np.zeros(len(cols))
    for _ in range(B):
        idx = np.sort(np.unique(block_indices(len(tr), L, rng)[: len(tr) // 2]))
        m = LogisticRegression(penalty="l1", C=C, solver="liblinear", max_iter=2000).fit(X[idx], y[idx])
        counts += (np.abs(m.coef_.ravel()) > 1e-8)
    freq = counts / B
    keep = [c for c, f in zip(cols, freq) if f >= thr]
    if len(keep) < 3:
        keep = [cols[i] for i in np.argsort(-freq)[:10]]
    return keep


def sel_mi(ds, tr, cols, k):
    X = ds.F[cols].to_numpy()[tr]
    mi = mutual_info_classif(X, ds.y[tr], n_neighbors=5, random_state=SEED)
    return [cols[i] for i in np.argsort(-mi)[:k]]


def sel_cluster(ds, tr, cols, rho):
    """De-duplicate: cluster features with |Spearman| >= rho; keep the most target-correlated member."""
    X = ds.F[cols].to_numpy()[tr]
    C = np.abs(spearmanr(X).correlation)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)
    D = squareform(1 - C, checks=False)
    lab = fcluster(linkage(D, method="average"), t=1 - rho, criterion="distance")
    tc = np.abs(_spearman_with_target(X, ds.y[tr]))
    keep = []
    for g in np.unique(lab):
        m = np.flatnonzero(lab == g)
        keep.append(cols[m[np.argmax(tc[m])]])
    return keep


def sel_stable(ds, tr, cols, k):
    """Drift-aware filter: rank features by the t-stat of their per-year Spearman correlation with the target
    (sign consistency over years), so features whose relationship flips are dropped."""
    X = ds.F[cols].to_numpy()[tr]
    y, yrs = ds.y[tr], ds.years[tr]
    per_year = []
    for yr in np.unique(yrs):
        m = yrs == yr
        if m.sum() >= 120:
            per_year.append(_spearman_with_target(X[m], y[m]))
    P = np.vstack(per_year)
    t = P.mean(0) / (P.std(0, ddof=1) / np.sqrt(len(P)) + 1e-12)
    return [cols[i] for i in np.argsort(-np.abs(t))[:k]]


def sel_perm(ds, tr, cols, k):
    """GBDT permutation importance (AUC) on the last 20% of the training window (purged gap of h rows)."""
    n = len(tr)
    cut = int(n * 0.8)
    a, b = tr[: cut - ds.h], tr[cut:]
    X = ds.F[cols].to_numpy()
    m = HistGradientBoostingClassifier(learning_rate=0.05, max_depth=2, max_iter=200, min_samples_leaf=100,
                                       l2_regularization=1.0, random_state=SEED).fit(X[a], ds.y[a])
    pi = permutation_importance(m, X[b], ds.y[b], scoring="roc_auc", n_repeats=5, random_state=SEED)
    return [cols[i] for i in np.argsort(-pi.importances_mean)[:k]]


def _inner_auc(ds, tr, cols, C=0.1):
    n = len(tr)
    cut = n - 504
    a, b = tr[: cut - ds.h], tr[cut:]
    X = ds.F[cols].to_numpy()
    m = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=C, max_iter=2000))]).fit(X[a], ds.y[a])
    return roc_auc_score(ds.y[b], m.predict_proba(X[b])[:, 1])


def sel_family_forward(ds, tr, cols, max_fam=5, min_gain=0.002):
    """Greedy forward selection over feature families, scored by AUC on the last 2 years of the training window."""
    fams = {k: [c for c in v if c in cols] for k, v in ds.fam.items()}
    fams = {k: v for k, v in fams.items() if v}
    chosen, best = [], 0.5
    while len(chosen) < max_fam:
        cand = {k: _inner_auc(ds, tr, sum((fams[c] for c in chosen), []) + v)
                for k, v in fams.items() if k not in chosen}
        k, s = max(cand.items(), key=lambda kv: kv[1])
        if s < best + min_gain:
            break
        chosen.append(k); best = s
    if not chosen:
        return list(CORE_FEATURES)
    return sum((fams[c] for c in chosen), [])


class PCATransform:
    def __init__(self, ds, tr, cols, n):
        self.cs = ColumnSet(ds, cols)
        self.sc = StandardScaler().fit(self.cs(tr))
        self.pca = PCA(n_components=n, random_state=SEED).fit(self.sc.transform(self.cs(tr)))
        self.cols = [f"pc{i}" for i in range(n)]

    def __call__(self, rows):
        return self.pca.transform(self.sc.transform(self.cs(rows)))


SELECTORS = {
    "base122": lambda ds, tr: ColumnSet(ds, ds.base_cols),
    "all188": lambda ds, tr: ColumnSet(ds, ds.cols),
    "core23": lambda ds, tr: ColumnSet(ds, CORE_FEATURES),
    "coreplus33": lambda ds, tr: ColumnSet(ds, CORE_PLUS),
    "l1stab_0.6": lambda ds, tr: ColumnSet(ds, sel_l1_stability(ds, tr, ds.cols, 0.6)),
    "l1stab_0.8": lambda ds, tr: ColumnSet(ds, sel_l1_stability(ds, tr, ds.cols, 0.8)),
    "mi_10": lambda ds, tr: ColumnSet(ds, sel_mi(ds, tr, ds.cols, 10)),
    "mi_25": lambda ds, tr: ColumnSet(ds, sel_mi(ds, tr, ds.cols, 25)),
    "mi_50": lambda ds, tr: ColumnSet(ds, sel_mi(ds, tr, ds.cols, 50)),
    "cluster_0.7": lambda ds, tr: ColumnSet(ds, sel_cluster(ds, tr, ds.cols, 0.7)),
    "cluster_0.85": lambda ds, tr: ColumnSet(ds, sel_cluster(ds, tr, ds.cols, 0.85)),
    "stable_10": lambda ds, tr: ColumnSet(ds, sel_stable(ds, tr, ds.cols, 10)),
    "stable_25": lambda ds, tr: ColumnSet(ds, sel_stable(ds, tr, ds.cols, 25)),
    "perm_10": lambda ds, tr: ColumnSet(ds, sel_perm(ds, tr, ds.cols, 10)),
    "perm_25": lambda ds, tr: ColumnSet(ds, sel_perm(ds, tr, ds.cols, 25)),
    "pca_5": lambda ds, tr: PCATransform(ds, tr, ds.cols, 5),
    "pca_15": lambda ds, tr: PCATransform(ds, tr, ds.cols, 15),
    "family_fwd": lambda ds, tr: ColumnSet(ds, sel_family_forward(ds, tr, ds.cols)),
}

MODELS = {
    "LR_C0.01": lambda: Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=0.01, max_iter=3000))]),
    "LR_C0.1": lambda: Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=0.1, max_iter=3000))]),
    "GBDT": lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_depth=2, max_iter=200,
                                                   min_samples_leaf=100, l2_regularization=1.0,
                                                   random_state=SEED),
}


def run(ds: FEDataset, selector: str, model: str, scheme: str, start: int, end: int, refit_every: int = 63):
    """Walk-forward predictions for rows [start, end).

    scheme 'fixed': select features + fit once on everything known at `start`.
    scheme 'refit': re-select features AND re-fit every `refit_every` rows (expanding window).
    Returns (prediction DataFrame, list of selected-feature lists per fit).
    """
    if scheme == "fixed":
        blocks = [(start, np.arange(start, end))]
    else:
        blocks = [(b, np.arange(b, min(b + refit_every, end))) for b in range(start, end, refit_every)]
    proba, thr, chosen = np.empty(end - start), np.empty(end - start), []
    for t0, rows in blocks:
        tr = np.arange(ds.n_known(t0))
        assert ds.label_end[tr].max() <= ds.dates[t0]
        tf = SELECTORS[selector](ds, tr)
        m = MODELS[model]().fit(tf(tr), ds.y[tr])
        proba[rows - start] = m.predict_proba(tf(rows))[:, 1]
        thr[rows - start] = ds.y[tr].mean()
        chosen.append(tf.cols)
    pred = pd.DataFrame({"y": ds.y[start:end], "proba": proba, "thr": thr}, index=ds.index[start:end])
    pred["pred"] = (pred["proba"] >= pred["thr"]).astype(int)
    return pred, chosen


def bootstrap_vs_chance(pred: pd.DataFrame, h: int, n_boot: int = 2000, seed: int = SEED) -> dict:
    """Block-bootstrap CI for balanced accuracy - 0.5 and AUC - 0.5 (one-sided p = P(draw <= 0))."""
    rng = np.random.default_rng(seed)
    y, p, yh = pred["y"].to_numpy(), pred["proba"].to_numpy(), pred["pred"].to_numpy()
    L = block_len(h)

    def sc(i):
        yy, pp, hh = y[i], p[i], yh[i]
        return np.array([0.5 * (hh[yy == 1].mean() + (1 - hh[yy == 0]).mean()) - 0.5, roc_auc_score(yy, pp) - 0.5])
    point = sc(np.arange(len(y)))
    draws = np.array([sc(block_indices(len(y), L, rng)) for _ in range(n_boot)])
    out = {}
    for j, m in enumerate(["balanced_acc", "roc_auc"]):
        out[f"d_{m}"], out[f"d_{m}_lo"], out[f"d_{m}_hi"] = point[j], *np.percentile(draws[:, j], [2.5, 97.5])
        out[f"p_{m}"] = float(np.mean(draws[:, j] <= 0))
    return out


__all__ = ["FEDataset", "SELECTORS", "MODELS", "run", "metrics", "paired_bootstrap", "bootstrap_vs_chance",
           "CORE_PLUS"]

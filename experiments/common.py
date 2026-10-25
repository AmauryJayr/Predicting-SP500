"""Shared, leakage-free infrastructure for the drift-aware experiments.

Feature engineering, target construction, metrics and the paired moving-block
bootstrap are copied from the notebook builder (build_notebooks.py) without
logic changes. The walk-forward engine is generalised to support
  * per-sample weights (e.g. exponential time decay),
  * alternative (causal) feature matrices, e.g. trailing-window normalisation,
  * arbitrary refit schedules, including data-dependent triggers,
  * a warm-up stretch before the evaluation period so that online mechanisms
    (recalibration, expert weighting, drift detectors) start with history made
    of genuinely out-of-sample predictions.

Leakage rules enforced here:
  * a model fitted at decision date t0 uses only rows with label_end <= t0;
  * any post-hoc mechanism at date t uses only realised outcomes of rows with
    label_end <= t (see `known_upto`);
  * feature transforms are trailing-window (past-only) computations.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EXP_DIR = ROOT / "experiments"
RES_DIR = EXP_DIR / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
STUDY_START, STUDY_END = "2008-01-01", "2025-12-31"
TRAIN_END, VAL_END = "2016-12-31", "2018-12-31"

EQUITY = ["SPY", "QQQ", "IWM"]
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
BONDS = ["TLT", "IEF"]
COMMODITIES = ["GLD"]

# Evaluation periods. F1/F2 are extra walk-forward validation folds inside the
# training era (used only for more robust model selection); VAL is the
# official 2017-2018 validation period; TEST is touched once.
PERIODS = {
    "F1": ("2013-01-01", "2014-12-31"),
    "F2": ("2015-01-01", "2016-12-31"),
    "VAL": ("2017-01-01", "2018-12-31"),
    "TEST": ("2019-01-01", "2025-12-31"),
}
SELECTION_FOLDS = ["F1", "F2", "VAL"]
WARMUP = 252  # sessions of out-of-sample history produced before each evaluation period


# ---------------------------------------------------------------------------
# Feature engineering (verbatim logic from the builder)
# ---------------------------------------------------------------------------
def log_ret(s, k=1):
    return np.log(s).diff(k)


def zscore(s, w):
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def spy_features(d):
    c, o, h, l, v = d["Close"], d["Open"], d["High"], d["Low"], d["Volume"]
    r = log_ret(c)
    f = {}
    for k in [1, 2, 3, 5, 10, 20, 60, 120, 250]:
        f[f"spy_ret_{k}d"] = log_ret(c, k)
    f["spy_ret_1d_lag1"] = r.shift(1)
    f["spy_ret_1d_lag2"] = r.shift(2)
    f["spy_intraday_ret"] = np.log(c / o)
    f["spy_overnight_ret"] = np.log(o / c.shift(1))
    for w in [5, 10, 20, 60]:
        f[f"spy_vol_{w}d"] = r.rolling(w).std() * np.sqrt(252)
    f["spy_vol_ratio_5_60"] = f["spy_vol_5d"] / f["spy_vol_60d"]
    f["spy_parkinson_vol_20d"] = np.sqrt((np.log(h / l) ** 2).rolling(20).mean() / (4 * np.log(2)) * 252)
    f["spy_range_pct"] = (h - l) / c
    f["spy_range_z20"] = zscore(f["spy_range_pct"], 20)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["spy_atr14_pct"] = tr.rolling(14).mean() / c
    f["spy_close_location"] = ((c - l) / (h - l)).where(h > l, 0.5)
    for w in [10, 20, 50, 200]:
        f[f"spy_px_vs_sma{w}"] = c / c.rolling(w).mean() - 1
    f["spy_sma50_vs_sma200"] = c.rolling(50).mean() / c.rolling(200).mean() - 1
    f["spy_bollinger_z20"] = zscore(c, 20)
    f["spy_rsi14"] = rsi(c)
    f["spy_dist_252d_high"] = c / c.rolling(252).max() - 1
    f["spy_dist_252d_low"] = c / c.rolling(252).min() - 1
    f["spy_skew_60d"] = r.rolling(60).skew()
    f["spy_up_frac_10d"] = (r > 0).astype(float).rolling(10).mean()
    lv = np.log(v.replace(0, np.nan))
    f["spy_volume_z20"] = zscore(lv, 20)
    f["spy_volume_ratio_5_60"] = v.rolling(5).mean() / v.rolling(60).mean()
    f["spy_volume_chg_1d"] = lv.diff()
    return pd.DataFrame(f)


def calendar_features(idx):
    f = pd.DataFrame(index=idx)
    for k in range(5):
        f[f"cal_dow_{k}"] = (idx.dayofweek == k).astype(float)
    f["cal_month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    f["cal_month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    return f


def cross_features(px):
    f = {}
    spy_c = px["SPY"]["Close"]
    for t in EQUITY[1:] + SECTORS + BONDS + COMMODITIES:
        c, n = px[t]["Close"], t.lower()
        for k in [1, 5, 20]:
            f[f"{n}_ret_{k}d"] = log_ret(c, k)
        f[f"{n}_px_vs_sma50"] = c / c.rolling(50).mean() - 1
    rel = lambda a, b, k: log_ret(px[a]["Close"], k) - log_ret(px[b]["Close"], k)
    for k in [5, 20]:
        f[f"qqq_minus_spy_ret_{k}d"] = rel("QQQ", "SPY", k)
        f[f"iwm_minus_spy_ret_{k}d"] = rel("IWM", "SPY", k)
        f[f"tlt_minus_ief_ret_{k}d"] = rel("TLT", "IEF", k)
    f["gld_minus_spy_ret_20d"] = rel("GLD", "SPY", 20)
    f["xly_minus_xlp_ret_20d"] = rel("XLY", "XLP", 20)
    sec_c = pd.concat({t: px[t]["Close"] for t in SECTORS}, axis=1)
    sec_r1, sec_r5 = np.log(sec_c).diff(), np.log(sec_c).diff(5)
    f["sector_dispersion_1d"] = sec_r1.std(axis=1)
    f["sector_dispersion_5d"] = sec_r5.std(axis=1)
    f["sector_breadth_up_1d"] = (sec_r1 > 0).mean(axis=1).where(sec_r1.notna().all(axis=1))
    sma50 = sec_c.rolling(50).mean()
    f["sector_breadth_above_sma50"] = (sec_c > sma50).mean(axis=1).where(sma50.notna().all(axis=1))
    f["corr_spy_tlt_60d"] = log_ret(spy_c).rolling(60).corr(log_ret(px["TLT"]["Close"]))
    f["tlt_vol_20d"] = log_ret(px["TLT"]["Close"]).rolling(20).std() * np.sqrt(252)
    vix = px["^VIX"]
    vc = vix["Close"]
    f["vix_level"] = vc
    f["vix_log_chg_1d"] = log_ret(vc, 1)
    f["vix_log_chg_5d"] = log_ret(vc, 5)
    f["vix_z60"] = zscore(vc, 60)
    f["vix_vs_sma20"] = vc / vc.rolling(20).mean() - 1
    f["vix_pct_rank_252d"] = vc.rolling(252).rank(pct=True)
    f["vix_minus_rv20"] = vc / 100 - log_ret(spy_c).rolling(20).std() * np.sqrt(252)
    f["vix_range_pct"] = (vix["High"] - vix["Low"]) / vc
    return pd.DataFrame(f)


def build_features(px):
    F = pd.concat([spy_features(px["SPY"]), calendar_features(px["SPY"].index), cross_features(px)], axis=1)
    return F.replace([np.inf, -np.inf], np.nan)


# A small, a-priori "core" subset (chosen by economic intuition before any
# result was seen): trend, short-term reversal, volatility, VIX, breadth, bonds.
CORE_FEATURES = [
    "spy_ret_1d", "spy_ret_5d", "spy_ret_20d", "spy_ret_60d", "spy_ret_250d",
    "spy_overnight_ret", "spy_intraday_ret", "spy_vol_20d", "spy_vol_ratio_5_60",
    "spy_px_vs_sma50", "spy_px_vs_sma200", "spy_rsi14", "spy_dist_252d_high",
    "vix_level", "vix_z60", "vix_minus_rv20", "vix_log_chg_5d",
    "sector_breadth_above_sma50", "sector_dispersion_5d", "corr_spy_tlt_60d",
    "tlt_ret_20d", "iwm_minus_spy_ret_20d", "xly_minus_xlp_ret_20d",
]


def causal_normalize(F: pd.DataFrame, window: int, kind: str = "z") -> pd.DataFrame:
    """Trailing-window normalisation (uses rows <= t only). Calendar dummies are left untouched."""
    out = F.copy()
    cols = [c for c in F.columns if not c.startswith("cal_")]
    mp = max(window // 2, 20)
    if kind == "z":
        mu = F[cols].rolling(window, min_periods=mp).mean()
        sd = F[cols].rolling(window, min_periods=mp).std()
        z = (F[cols] - mu) / sd.replace(0, np.nan)
        out[cols] = z.clip(-5, 5)
    elif kind == "rank":
        out[cols] = F[cols].rolling(window, min_periods=mp).rank(pct=True) - 0.5
    else:
        raise ValueError(kind)
    return out.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class Dataset:
    """Modelling dataset for one horizon. Feature matrices are aligned to `self.index`."""

    def __init__(self, horizon: int):
        self.h = horizon
        px = pd.read_pickle(DATA_DIR / "aligned_prices.pkl")
        calendar = px["SPY"].index
        F = build_features(px)
        spy = px["SPY"]
        entry = spy["Open"].shift(-1)
        exit_ = spy["Close"].shift(-horizon)
        fwd_ret = exit_ / entry - 1
        target = (fwd_ret > 0).astype(float).where(fwd_ret.notna())
        label_end = pd.Series(calendar, index=calendar).shift(-horizon)

        self.F_full = F  # full history from 2006 (needed for trailing transforms)
        data = F.copy()
        data["target"] = target
        data["fwd_ret"] = fwd_ret
        data["label_end"] = label_end
        data = data.loc[STUDY_START:STUDY_END]
        self.all_features = list(F.columns)
        miss = data[self.all_features].isna().any(axis=1) | data["target"].isna()
        data = data[~miss].copy()
        data["target"] = data["target"].astype(int)
        self.data = data
        self.index = data.index
        self.dates = data.index.to_numpy()
        self.label_end = data["label_end"].to_numpy()
        self.y = data["target"].to_numpy()
        self.n = len(data)
        self._X_cache = {}

    # feature matrices -------------------------------------------------------
    def X(self, feats: str = "all", norm: str | None = None) -> np.ndarray:
        """feats in {'all', 'core'}; norm in {None, 'z252', 'z504', 'z1260', 'rank252', 'rank504', ...}."""
        key = (feats, norm)
        if key not in self._X_cache:
            cols = self.all_features if feats == "all" else CORE_FEATURES
            F = self.F_full[cols]
            if norm is not None:
                kind = "z" if norm.startswith("z") else "rank"
                w = int(norm.lstrip("zrank"))
                F = causal_normalize(F, w, kind)
            M = F.reindex(self.index).to_numpy(dtype=float)
            M = np.nan_to_num(M, nan=0.0)  # only affects early trailing-window warm-up
            self._X_cache[key] = M
        return self._X_cache[key]

    def pos(self, date) -> int:
        """First row position with date >= `date`."""
        return int(np.searchsorted(self.dates, np.datetime64(pd.Timestamp(date))))

    def period_positions(self, period: str):
        a, b = PERIODS[period]
        s, e = self.pos(a), int(np.searchsorted(self.dates, np.datetime64(pd.Timestamp(b)), side="right"))
        return s, e  # [s, e)

    def known_upto(self, t_pos: int) -> np.ndarray:
        """Row positions whose label is known at the close of row t_pos (purging rule)."""
        return np.flatnonzero(self.label_end <= self.dates[t_pos])

    def n_known(self, t_pos: int) -> int:
        # label_end is monotone non-decreasing in row order, so the known rows form a prefix
        return int(np.searchsorted(self.label_end, self.dates[t_pos], side="right"))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def make_estimator(name: str, params: dict | None = None):
    params = dict(params or {})
    if name == "LogReg":
        return Pipeline([("scale", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000, random_state=SEED, **params))])
    if name == "GBDT":
        return HistGradientBoostingClassifier(early_stopping=False, random_state=SEED, **params)
    raise ValueError(name)


def fit_model(est, X, y, w=None):
    m = clone(est)
    if w is None:
        return m.fit(X, y)
    if isinstance(m, Pipeline):
        return m.fit(X, y, clf__sample_weight=w)
    return m.fit(X, y, sample_weight=w)


def decay_weights(n_rows: int, half_life: float | None) -> np.ndarray | None:
    """Exponential time-decay weights for the n most recent known rows (oldest first), mean 1."""
    if half_life is None:
        return None
    age = np.arange(n_rows)[::-1].astype(float)
    w = 0.5 ** (age / half_life)
    return w / w.mean()


def weighted_rate(y, w):
    return float(y.mean()) if w is None else float(np.average(y, weights=w))


# ---------------------------------------------------------------------------
# Generic walk-forward engine
# ---------------------------------------------------------------------------
def walk_forward(ds: Dataset, X: np.ndarray, est, start: int, end: int, refit_every: int | None,
                 window: int | None = None, half_life: float | None = None, fixed_fit_pos: int | None = None,
                 min_train: int = 250):
    """Predict rows [start, end).

    refit_every=None -> 'fixed': one model fitted at `fixed_fit_pos` (default `start`).
    Otherwise refit every `refit_every` rows, block-wise, at the first row of each block.
    Training set at fit row t0: rows with label_end <= date[t0] (prefix), optionally the last `window`.
    Threshold = (weighted) positive rate of that training set.
    Returns DataFrame indexed by position with proba, thr, fit_pos.
    """
    y = ds.y
    if refit_every is None:
        fits = [(fixed_fit_pos if fixed_fit_pos is not None else start, np.arange(start, end))]
    else:
        fits = [(b, np.arange(b, min(b + refit_every, end))) for b in range(start, end, refit_every)]
    proba = np.empty(end - start)
    thr = np.empty(end - start)
    fitp = np.empty(end - start, dtype=int)
    for t0, rows in fits:
        nk = ds.n_known(t0)
        tr = np.arange(nk)
        if window is not None:
            tr = tr[-window:]
        assert len(tr) >= min_train
        assert ds.label_end[tr].max() <= ds.dates[t0]  # purging check
        w = decay_weights(len(tr), half_life)
        m = fit_model(est, X[tr], y[tr], w)
        proba[rows - start] = m.predict_proba(X[rows])[:, 1]
        thr[rows - start] = weighted_rate(y[tr], w)
        fitp[rows - start] = t0
    return pd.DataFrame({"proba": proba, "thr": thr, "fit_pos": fitp}, index=np.arange(start, end))


def to_pred(ds: Dataset, df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Restrict a positional prediction frame to [start, end) and attach y / pred / dates."""
    d = df.loc[start:end - 1].copy()
    d["y"] = ds.y[d.index]
    d["pred"] = (d["proba"] >= d["thr"]).astype(int)
    d.index = ds.index[d.index]
    return d[["y", "proba", "thr", "pred"]]


# ---------------------------------------------------------------------------
# Metrics and paired block bootstrap (builder logic)
# ---------------------------------------------------------------------------
def metrics(pred: pd.DataFrame) -> dict:
    y, p, yhat = pred["y"], pred["proba"], pred["pred"]
    return {"balanced_acc": balanced_accuracy_score(y, yhat),
            "roc_auc": roc_auc_score(y, p) if y.nunique() > 1 else np.nan,
            "brier": brier_score_loss(y, p),
            "accuracy": accuracy_score(y, yhat),
            "pred_up_rate": float(yhat.mean()),
            "n": len(y)}


def block_len(h: int) -> int:
    return max(20, 3 * h)


def block_indices(n, L, rng):
    starts = rng.integers(0, n - L + 1, int(np.ceil(n / L)))
    return (starts[:, None] + np.arange(L)).ravel()[:n]


def fast_scores(y, p, yhat):
    ba = 0.5 * (yhat[y == 1].mean() + (1 - yhat[y == 0]).mean())
    return np.array([ba, roc_auc_score(y, p), np.mean((p - y) ** 2)])


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, h: int, n_boot: int = 2000, seed: int = SEED) -> dict:
    """Improvement of A over B (positive = A better; Brier sign flipped). One-sided p = P(draw <= 0)."""
    assert a.index.equals(b.index) and (a["y"].to_numpy() == b["y"].to_numpy()).all()
    L = block_len(h)
    rng = np.random.default_rng(seed)
    y, pa, ya, pb, yb = (a["y"].to_numpy(), a["proba"].to_numpy(), a["pred"].to_numpy(),
                         b["proba"].to_numpy(), b["pred"].to_numpy())
    sign = np.array([1, 1, -1])
    point = sign * (fast_scores(y, pa, ya) - fast_scores(y, pb, yb))
    draws = np.empty((n_boot, 3))
    for i in range(n_boot):
        idx = block_indices(len(y), L, rng)
        draws[i] = sign * (fast_scores(y[idx], pa[idx], ya[idx]) - fast_scores(y[idx], pb[idx], yb[idx]))
    out = {}
    for j, m in enumerate(["balanced_acc", "roc_auc", "brier"]):
        out[f"d_{m}"] = point[j]
        out[f"d_{m}_lo"] = np.percentile(draws[:, j], 2.5)
        out[f"d_{m}_hi"] = np.percentile(draws[:, j], 97.5)
        out[f"p_{m}"] = float(np.mean(draws[:, j] <= 0))
    return out

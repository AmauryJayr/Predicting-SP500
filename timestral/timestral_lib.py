"""Zero-shot Mistral Timestral-209M forecasts of SPY, turned into a P(up) direction score.

Mirrors timesfm3_experiment/timesfm_lib.py. One forecast per decision date t (horizon 5, context ends
at Close[t], nothing after t), so the same call serves h=1 (step 1) and h=5 (step 5). Forecasts are
cached per (variant, ctx) under results/timestral_cache/ keyed by date, so any later run -- including
the one-shot test run -- only computes the dates it is missing. The forecaster is frozen (no fitting,
no fine-tuning): the only thing fitted on labels is `pred_frame`'s one-parameter offset calibrator,
which is given an explicitly purged row range. Scores are an approximation of the project's label,
which compares Close[t+h] with Open[t+1] rather than with Close[t]; selection therefore uses AUC,
which is calibration-free.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import norm

from ts_paths import RES_DIR
from common import BONDS, COMMODITIES, DATA_DIR, EQUITY, PERIODS, SECTORS, SEED

REPO = "mistralai/Timestral-209M"
REVISION = "9874d252ab0aae16f7f7c999f98f98d6ad7d8365"
PKG_COMMIT = "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"
VARIANTS = ["price", "ret", "multi"]
CTXS = [256, 512, 1024, 2048]
HORIZONS = [1, 5]
H_MAX = 5                                                        # one 5-step forecast serves both horizons
QLEVELS = np.array([0.01, 0.05] + [round(0.10 + 0.05 * i, 2) for i in range(17)] + [0.95, 0.99])
NQ = len(QLEVELS)                                                # 21; quantiles[..., i] is the QLEVELS[i] quantile
assert NQ == 21
I10, I50, I90 = (int(np.argmin(np.abs(QLEVELS - x))) for x in (0.1, 0.5, 0.9))
CLIP = (0.01, 0.99)                                              # outermost levels: score clip outside q01..q99
Z90 = float(norm.ppf(0.9))                                       # 1.2816, for the q90-q10 sigma estimate
COVS = EQUITY[1:] + SECTORS + BONDS + COMMODITIES + ["^VIX"]     # the 15 past-only covariates of `multi`
FIELDS = ["Open", "High", "Low", "Close", "Volume"]
CACHE = RES_DIR / "timestral_cache"
CACHE.mkdir(parents=True, exist_ok=True)

RUN = {"batch": 8, "threads": 16}                                # runtime knobs: they change speed, not results
_MEM: dict = {}


# ---------------------------------------------------------------------------
# Price panel -> model inputs
# ---------------------------------------------------------------------------
def panel(px: dict | None = None) -> dict:
    """Log-Close panel {'dates', 'spy', 'cov'}; `cov` is (15, n) in COVS order."""
    if px is None:
        if "panel" not in _MEM:
            _MEM["panel"] = panel(pd.read_pickle(DATA_DIR / "aligned_prices.pkl"))
        return _MEM["panel"]
    return {"dates": px["SPY"].index,
            "spy": np.log(px["SPY"]["Close"].to_numpy(dtype=float)),
            "cov": np.log(np.stack([px[t]["Close"].to_numpy(dtype=float) for t in COVS]))}


def series(variant: str, p: dict | None = None):
    """(dates, target, covariates); row i of the target is fully known at the close of dates[i]."""
    p = p or panel()
    if variant == "price":
        return p["dates"], p["spy"], None
    if variant == "ret":
        return p["dates"][1:], np.diff(p["spy"]), None           # daily close-to-close log returns
    if variant == "multi":
        return p["dates"], p["spy"], p["cov"]                    # same target as `price` + 15 covariates
    raise ValueError(variant)


def contexts(variant: str, ctx: int, dates, p: dict | None = None) -> list:
    """Context windows ending at each date; shorter than `ctx` only where the history is too short."""
    d, tgt, cov = series(variant, p)
    pos = d.get_indexer(pd.DatetimeIndex(dates))
    assert (pos >= 0).all(), "decision date not on the series calendar"
    out = []
    for q in pos:
        lo = max(0, q + 1 - ctx)
        out.append((np.ascontiguousarray(tgt[lo:q + 1], dtype=np.float32),
                    None if cov is None else np.ascontiguousarray(cov[:, lo:q + 1], dtype=np.float32)))
    return out


def grid_dates(a: str, b: str) -> pd.DatetimeIndex:
    """SPY trading days in [a, b] -- a superset of the Dataset(h) rows of those periods."""
    d = panel()["dates"]
    return d[(d >= pd.Timestamp(a)) & (d <= pd.Timestamp(b))]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_model(batch: int | None = None, threads: int | None = None):
    """Frozen forecaster on CPU, vendor defaults, pinned weights revision."""
    import torch
    set_runtime(None, batch, threads)
    torch.manual_seed(SEED)
    from timestral import TimestralForecaster
    m = TimestralForecaster.from_pretrained(REPO, device="cpu", revision=REVISION)
    m.model.eval()
    return m


def set_runtime(m, batch: int | None = None, threads: int | None = None):
    """Change the inference batch size / thread count (speed only)."""
    import torch
    if threads:
        RUN["threads"] = int(threads)
    if batch:
        RUN["batch"] = int(batch)
    torch.set_num_threads(RUN["threads"])
    return {"batch": RUN["batch"], "torch_threads": int(torch.get_num_threads())}


def _batch_tensors(cs: list):
    """Stack ragged contexts into (B,1,T) [+ (B,15,T)], left-padded with NaN (= missing) to the longest."""
    T = max(len(c[0]) for c in cs)
    x = np.full((len(cs), 1, T), np.nan, dtype=np.float32)
    cov = None if cs[0][1] is None else np.full((len(cs), cs[0][1].shape[0], T), np.nan, dtype=np.float32)
    for i, (t, c) in enumerate(cs):
        x[i, 0, T - len(t):] = t
        if cov is not None:
            cov[i, :, T - len(t):] = c
    return x, cov


def forecast(m, variant: str, ctx: int, dates, p: dict | None = None, with_flags: bool = False):
    """(n, 5, 21) step-by-quantile forecasts. Vendor defaults: symmetric averaging and clip_nonnegative on."""
    import torch
    cs = contexts(variant, ctx, dates, p)
    outq, nuns = [], []
    torch.manual_seed(SEED)
    for i in range(0, len(cs), RUN["batch"]):
        x, cov = _batch_tensors(cs[i:i + RUN["batch"]])
        with torch.inference_mode():
            f = m.forecast(x, prediction_length=H_MAX, past_covariates=cov,
                           symmetric_averaging=True, clip_nonnegative=True)
        assert tuple(f.quantile_levels) == tuple(float(v) for v in QLEVELS), f.quantile_levels
        outq.append(f.quantiles[:, 0].double().numpy())
    q = np.concatenate(outq)
    assert q.shape == (len(cs), H_MAX, NQ), q.shape
    assert np.isfinite(q).all()
    uns = (np.diff(q, axis=-1) < -1e-12).any(axis=(1, 2))      # forecasts whose quantiles were not monotone
    q = np.sort(q, axis=-1)
    return (q, uns) if with_flags else q


# ---------------------------------------------------------------------------
# Forecast cache: {variant}_ctx{ctx}.npz, keyed by date
# ---------------------------------------------------------------------------
def cache_file(variant: str, ctx: int):
    return CACHE / f"{variant}_ctx{ctx}.npz"


def read_cache(variant: str, ctx: int):
    f = cache_file(variant, ctx)
    if not f.exists():
        return pd.DatetimeIndex([]), np.zeros((0, H_MAX, NQ)), np.zeros(0, dtype=int), np.zeros(0, dtype=bool)
    z = np.load(f)
    return pd.DatetimeIndex(z["dates"]), z["q"], z["ctx_used"], z["unsorted"]


def write_cache(variant: str, ctx: int, dates: pd.DatetimeIndex, q, used, uns):
    o = np.argsort(dates.to_numpy())
    np.savez_compressed(cache_file(variant, ctx), dates=dates.to_numpy()[o], q=q[o], ctx_used=used[o],
                        unsorted=uns[o])


def ensure(m, variant: str, ctx: int, dates, chunk: int = 128, log=print, p: dict | None = None):
    """Compute the forecasts for `dates` that are not cached yet, writing after every chunk."""
    have, q, used, uns = read_cache(variant, ctx)
    todo = pd.DatetimeIndex(dates).difference(have)
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        t0 = time.time()
        qc, uc_ = forecast(m, variant, ctx, part, p, with_flags=True)
        uc = np.array([len(c[0]) for c in contexts(variant, ctx, part, p)], dtype=int)
        have, q, used = have.append(part), np.concatenate([q, qc]), np.concatenate([used, uc])
        uns = np.concatenate([uns, uc_])
        write_cache(variant, ctx, have, q, used, uns)            # written per chunk: the run is resumable
        log(f"{variant} ctx{ctx}: {i + len(part)}/{len(todo)} new, "
            f"{(time.time() - t0) / len(part):.3f}s/forecast", flush=True)
    return have, q, used


def aligned(variant: str, ctx: int, dates):
    """Cached (n, 5, 21) quantiles and context lengths in the order of `dates`."""
    have, q, used, _ = read_cache(variant, ctx)
    i = have.get_indexer(pd.DatetimeIndex(dates))
    assert (i >= 0).all(), f"{int((i < 0).sum())} dates missing from {cache_file(variant, ctx).name}"
    return q[i], used[i]


# ---------------------------------------------------------------------------
# Quantiles -> raw P(up) score
# ---------------------------------------------------------------------------
def p_up(qrow: np.ndarray, variant: str, h: int, anchor: float) -> float:
    """P(up) at step h: the step-h CDF, linearly interpolated through (q_i, QLEVELS[i]).

    Outside q01..q99 the score is clipped to [0.01, 0.99]. `ret` at h=5 cannot be read off one
    step, so it uses a normal approximation of the 5-day return sum.
    """
    if variant == "ret" and h > 1:
        mu = float(qrow[:h, I50].sum())                          # sum of the step medians
        sd = float(np.sqrt((((qrow[:h, I90] - qrow[:h, I10]) / (2 * Z90)) ** 2).sum()))
        return 0.5 if sd <= 0 else float(norm.cdf(mu / sd))
    thr = 0.0 if variant == "ret" else anchor                    # ret: r[t+1] > 0; levels: logC[t+h] > logC[t]
    return float(1.0 - np.interp(thr, qrow[h - 1], QLEVELS, left=CLIP[0], right=CLIP[1]))


def scores(variant: str, ctx: int, h: int, dates, p: dict | None = None) -> pd.Series:
    """Raw P(up) scores for `dates`, read out of the cached forecasts."""
    dates = pd.DatetimeIndex(dates)
    q, _ = aligned(variant, ctx, dates)
    pp = p or panel()
    a = pd.Series(pp["spy"], index=pp["dates"]).reindex(dates).to_numpy()   # logC[t], the level anchor
    s = np.array([p_up(q[i], variant, h, a[i]) for i in range(len(dates))])
    return pd.Series(np.clip(s, *CLIP), index=dates, name="score")


# ---------------------------------------------------------------------------
# Prediction frames (project convention: DataFrame[y, proba, thr, pred] indexed by date)
# ---------------------------------------------------------------------------
def logit(s, lo: float = 0.01):
    s = np.clip(np.asarray(s, dtype=float), lo, 1 - lo)
    return np.log(s / (1 - s))


def base_rate(ds, period: str) -> float:
    """`run_test.chance` threshold rule: positive rate of the labels known at the period start."""
    s, _ = ds.period_positions(period)
    return float(ds.y[:ds.n_known(s)].mean())


def cal_rows(ds, period: str) -> np.ndarray:
    """Calibration rows: selection-fold (F1..VAL) rows whose label is known at the period start."""
    s, _ = ds.period_positions(period)
    a, b = ds.pos(PERIODS["F1"][0]), ds.n_known(s)
    assert b > a and ds.label_end[b - 1] <= ds.dates[s]          # purging check
    return np.arange(a, b)


def raw_frame(ds, variant: str, ctx: int, h: int, period: str) -> pd.DataFrame:
    """Uncalibrated frame for validation: proba = the raw score, thr = the chance base rate."""
    s, e = ds.period_positions(period)
    r = base_rate(ds, period)
    sc = scores(variant, ctx, h, ds.index[s:e]).to_numpy()
    return pd.DataFrame({"y": ds.y[s:e], "proba": sc, "thr": r, "pred": (sc >= r).astype(int)},
                        index=ds.index[s:e])


def calibrator(ds, variant: str, ctx: int, h: int, period: str):
    """Offset b of logit(proba) = logit(s) + b, fitted once on the purged calibration rows of `period`.

    The slope is fixed at 1, so the calibration can never reverse Timestral's own ranking (AUC is the raw
    score's); b is the logistic MLE, i.e. the root of mean(expit(z + b) - y) = 0.
    """
    tr = cal_rows(ds, period)
    z, y = logit(scores(variant, ctx, h, ds.index[tr])), ds.y[tr]
    return brentq(lambda b: (expit(z + b) - y).mean(), -10, 10), tr


def apply_cal(b: float, sc) -> np.ndarray:
    return expit(logit(sc) + b)


def pred_frame(ds, variant: str, ctx: int, h: int, period: str) -> pd.DataFrame:
    """Calibrated frame on exactly the `period` rows: offset fitted once on the purged earlier fold rows."""
    s, e = ds.period_positions(period)
    b, tr = calibrator(ds, variant, ctx, h, period)
    proba = apply_cal(b, scores(variant, ctx, h, ds.index[s:e]).to_numpy())
    r = base_rate(ds, period)
    out = pd.DataFrame({"y": ds.y[s:e], "proba": proba, "thr": r, "pred": (proba >= r).astype(int)},
                       index=ds.index[s:e])
    out.attrs["n_cal"] = len(tr)
    out.attrs["cal_offset"] = float(b)
    return out


def chance_frame(y, index, rate: float) -> pd.DataFrame:
    """`run_test.chance` on an arbitrary index (used for the 2026 check)."""
    return pd.DataFrame({"y": np.asarray(y, dtype=int), "proba": rate, "thr": rate, "pred": 1}, index=index)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def truncation_check(m, variant: str, ctx: int, dates) -> list:
    """Forecasts for date t must not move when every price series is cut after t (leakage self-check)."""
    full = forecast(m, variant, ctx, dates)
    px = pd.read_pickle(DATA_DIR / "aligned_prices.pkl")
    out = []
    for i, t in enumerate(pd.DatetimeIndex(dates)):
        cut = panel({k: v.loc[:t] for k, v in px.items()})
        out.append(float(np.abs(forecast(m, variant, ctx, [t], cut)[0] - full[i]).max()))
    return out


# ---------------------------------------------------------------------------
# Fresh data for the 2026 out-of-sample check
# ---------------------------------------------------------------------------
def fresh_prices(start: str = "2025-06-01", end=None) -> dict:
    """SPY + the 15 covariates from yfinance, aligned to SPY's calendar as the notebooks do."""
    import yfinance as yf
    tick = ["SPY"] + COVS
    end = pd.Timestamp(end or pd.Timestamp.today().normalize())
    raw = yf.download(tick, start=start, end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                      auto_adjust=True, progress=False, group_by="column", threads=True)
    raw.columns.names = ["field", "ticker"]
    px = {t: raw.xs(t, axis=1, level="ticker")[FIELDS].copy() for t in tick}
    calendar = px["SPY"].dropna(subset=["Close"]).index
    return {t: d.reindex(calendar).ffill(limit=3) for t, d in px.items()}


def spliced_panel(fresh: dict):
    """The stored 2006-2025 log-Close panel continued with fresh yfinance sessions.

    yfinance back-adjusts with every later dividend, so the fresh download is re-based on the stored
    panel at the last shared session (an additive shift in logs; log returns are untouched).
    Returns (panel, join_date, max |log-return mismatch| over the overlap).
    """
    old = panel()
    new = panel(fresh)
    join = old["dates"][-1]
    assert join in new["dates"], "fresh download does not reach the end of the stored panel"
    jo, jn = len(old["dates"]) - 1, int(new["dates"].get_loc(join))
    ov = new["dates"][(new["dates"] >= old["dates"][0]) & (new["dates"] <= join)]
    io, inn = old["dates"].get_indexer(ov), new["dates"].get_indexer(ov)
    mism = max(float(np.abs(np.diff(old["spy"][io]) - np.diff(new["spy"][inn])).max()),
               float(np.abs(np.diff(old["cov"][:, io], axis=1) - np.diff(new["cov"][:, inn], axis=1)).max()))
    tail = slice(jn + 1, None)
    out = {"dates": old["dates"].append(new["dates"][tail]),
           "spy": np.concatenate([old["spy"], new["spy"][tail] + (old["spy"][jo] - new["spy"][jn])]),
           "cov": np.concatenate([old["cov"], new["cov"][:, tail] +
                                  (old["cov"][:, jo] - new["cov"][:, jn])[:, None]], axis=1)}
    return out, join, mism


def fresh_labels(fresh: dict, h: int) -> pd.DataFrame:
    """Labels built exactly as in Dataset: y = 1[Close[t+h] / Open[t+1] - 1 > 0]."""
    spy = fresh["SPY"]
    fwd = spy["Close"].shift(-h) / spy["Open"].shift(-1) - 1
    return pd.DataFrame({"fwd_ret": fwd, "y": (fwd > 0).astype(float).where(fwd.notna())}).dropna()

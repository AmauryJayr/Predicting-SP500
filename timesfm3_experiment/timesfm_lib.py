"""Zero-shot Google TimesFM-3.0 forecasts of SPY, turned into a P(up) direction score.

One forecast per decision date t (horizon 5, context ends at Close[t], nothing after t), so the
same call serves h=1 (step 1) and h=5 (step 5). Forecasts are cached per (variant, ctx) under
results/timesfm_cache/ keyed by date, so any later run -- including the one-shot test run -- only
computes the dates it is missing. The forecaster is frozen (no fitting, no fine-tuning): the only
thing fitted on labels is `pred_frame`'s one-parameter offset calibrator, which is given an explicitly
purged row range. Scores are an approximation of the project's label, which compares Close[t+h] with Open[t+1]
rather than with Close[t]; selection therefore uses AUC, which is calibration-free.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import norm

from common import BONDS, COMMODITIES, DATA_DIR, EQUITY, PERIODS, RES_DIR, SECTORS, SEED

CKPT = "google/timesfm-3.0-pytorch"
VARIANTS = ["price", "ret", "multi"]
CTXS = [256, 512, 1024, 2048]
HORIZONS = [1, 5]
H_MAX = 5                                                        # one 5-step forecast serves both horizons
NQ = 9
QLEVELS = np.arange(1, 10) / 10.0                                # quantiles[..., i] is the 0.1*(i+1) quantile
Z90 = float(norm.ppf(0.9))                                       # 1.2816, for the q90-q10 sigma estimate
COVS = EQUITY[1:] + SECTORS + BONDS + COMMODITIES + ["^VIX"]     # the 15 past-only covariates of `multi`
FIELDS = ["Open", "High", "Low", "Close", "Volume"]
CACHE = RES_DIR / "timesfm_cache"
CACHE.mkdir(parents=True, exist_ok=True)

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
    assert (pos >= 0).all(), "decision date not on the SPY calendar"
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
def load_model(per_core_batch_size: int = 8, threads: int | None = None):
    """Frozen forecaster on CPU. TimesFM3Forecaster, not the Evaluator (whose defaults differ)."""
    import torch
    if threads:
        torch.set_num_threads(int(threads))
    torch.manual_seed(SEED)
    from timesfm3 import TimesFM3Forecaster
    m = TimesFM3Forecaster.from_pretrained(CKPT, device="cpu", per_core_batch_size=per_core_batch_size)
    m.model.eval()
    return m


def set_runtime(m, per_core_batch_size: int | None = None, threads: int | None = None):
    """Change the inference batch size / thread count of an already loaded model."""
    import torch
    if threads:
        torch.set_num_threads(int(threads))
    if per_core_batch_size:
        m.config.per_core_batch_size = int(per_core_batch_size)
    return {"per_core_batch_size": m.config.per_core_batch_size, "torch_threads": int(torch.get_num_threads())}


def forecast(m, variant: str, ctx: int, dates, p: dict | None = None) -> np.ndarray:
    """(n, 5, 9) step-by-quantile forecasts. make_positive / symmetric averaging / znorm stay off."""
    import torch
    cs = contexts(variant, ctx, dates, p)
    tg = [c[0] for c in cs]
    co = [c[1] for c in cs] if cs and cs[0][1] is not None else None
    torch.manual_seed(SEED)
    with torch.no_grad():
        out = list(m.predict_batch(contexts=tg, horizon=H_MAX, past_only_covariates=co,
                                   return_quantiles=True, make_positive=False,
                                   use_symmetric_averaging=False, use_znorm=False, sort_quantiles=True))
    q = np.stack([o.quantiles for o in out]).astype(np.float64)
    assert q.shape == (len(cs), H_MAX, NQ), q.shape
    assert np.isfinite(q).all()
    return q


# ---------------------------------------------------------------------------
# Forecast cache: {variant}_ctx{ctx}.npz, keyed by date
# ---------------------------------------------------------------------------
def cache_file(variant: str, ctx: int):
    return CACHE / f"{variant}_ctx{ctx}.npz"


def read_cache(variant: str, ctx: int):
    f = cache_file(variant, ctx)
    if not f.exists():
        return pd.DatetimeIndex([]), np.zeros((0, H_MAX, NQ)), np.zeros(0, dtype=int)
    z = np.load(f)
    return pd.DatetimeIndex(z["dates"]), z["q"], z["ctx_used"]


def write_cache(variant: str, ctx: int, dates: pd.DatetimeIndex, q, used):
    o = np.argsort(dates.to_numpy())
    np.savez_compressed(cache_file(variant, ctx), dates=dates.to_numpy()[o], q=q[o], ctx_used=used[o])


def ensure(m, variant: str, ctx: int, dates, chunk: int = 128, log=print, p: dict | None = None):
    """Compute the forecasts for `dates` that are not cached yet, writing after every chunk."""
    have, q, used = read_cache(variant, ctx)
    todo = pd.DatetimeIndex(dates).difference(have)
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        t0 = time.time()
        qc = forecast(m, variant, ctx, part, p)
        uc = np.array([len(c[0]) for c in contexts(variant, ctx, part, p)], dtype=int)
        have, q, used = have.append(part), np.concatenate([q, qc]), np.concatenate([used, uc])
        write_cache(variant, ctx, have, q, used)                 # written per chunk: the run is resumable
        log(f"{variant} ctx{ctx}: {i + len(part)}/{len(todo)} new, "
            f"{(time.time() - t0) / len(part):.3f}s/forecast", flush=True)
    return have, q, used


def aligned(variant: str, ctx: int, dates):
    """Cached (n, 5, 9) quantiles and context lengths in the order of `dates`."""
    have, q, used = read_cache(variant, ctx)
    i = have.get_indexer(pd.DatetimeIndex(dates))
    assert (i >= 0).all(), f"{int((i < 0).sum())} dates missing from {cache_file(variant, ctx).name}"
    return q[i], used[i]


# ---------------------------------------------------------------------------
# Quantiles -> raw P(up) score
# ---------------------------------------------------------------------------
def p_up(qrow: np.ndarray, variant: str, h: int, anchor: float) -> float:
    """P(up) at step h: the step-h CDF, linearly interpolated through (q_i, 0.1(i+1)).

    Outside q10..q90 the score is clipped to [0.05, 0.95]. `ret` at h=5 cannot be read off one
    step, so it uses a normal approximation of the 5-day return sum.
    """
    if variant == "ret" and h > 1:
        mu = float(qrow[:h, 4].sum())                            # sum of the step medians
        sd = float(np.sqrt((((qrow[:h, NQ - 1] - qrow[:h, 0]) / (2 * Z90)) ** 2).sum()))
        return 0.5 if sd <= 0 else float(norm.cdf(mu / sd))
    thr = 0.0 if variant == "ret" else anchor                    # ret: r[t+1] > 0; levels: logC[t+h] > logC[t]
    return float(1.0 - np.interp(thr, qrow[h - 1], QLEVELS, left=0.05, right=0.95))


def scores(variant: str, ctx: int, h: int, dates, p: dict | None = None) -> pd.Series:
    """Raw P(up) scores for `dates`, read out of the cached forecasts."""
    dates = pd.DatetimeIndex(dates)
    q, _ = aligned(variant, ctx, dates)
    pp = p or panel()
    a = pd.Series(pp["spy"], index=pp["dates"]).reindex(dates).to_numpy()   # logC[t], the level anchor
    return pd.Series([p_up(q[i], variant, h, a[i]) for i in range(len(dates))], index=dates, name="score")


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

    The slope is fixed at 1, so the calibration can never reverse TimesFM's own ranking (AUC is the raw
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

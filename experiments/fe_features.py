"""Extended feature engineering for the feature-engineering / feature-selection study.

Adds ~60 causal features on top of the 122 baseline features (common.build_features),
grouped in families so that family-level selection is possible:

  volsurface : VIX term structure (VIX/VIX3M), vol-of-vol (VVIX), tail-risk (SKEW)
  rates      : 10y / 5y / 3m Treasury yields, curve slope and their changes
  credit     : high-yield & investment-grade credit vs Treasuries, inflation breakevens (TIP)
  global     : emerging / developed ex-US equities, US dollar, oil, broad commodities
  breadth    : equal-weight vs cap-weight S&P 500
  structure  : volatility-scaled returns, variance ratio, autocorrelation, drawdown,
               illiquidity, return-volume correlation, streaks, overnight/intraday decomposition
  calendar2  : turn-of-month and option-expiration week

Every feature at row t uses data up to and including the close of t only
(verified by `truncation_test`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from common import DATA_DIR, build_features, log_ret, zscore

EXTRA_TICKERS = ["^VIX3M", "^VVIX", "^SKEW", "^TNX", "^FVX", "^IRX", "HYG", "LQD", "TIP",
                 "EEM", "EFA", "UUP", "USO", "DBC", "RSP"]
EXTRA_CACHE = DATA_DIR / "extra_prices.pkl"
FFILL_LIMIT = 3


def load_extra(calendar: pd.DatetimeIndex, force: bool = False) -> dict:
    """Download (once) and align the extra tickers to SPY's calendar; forward-fill <= 3 sessions."""
    if EXTRA_CACHE.exists() and not force:
        raw = pd.read_pickle(EXTRA_CACHE)
    else:
        raw = yf.download(EXTRA_TICKERS, start="2005-01-01", end="2026-01-01", auto_adjust=True,
                          progress=False, group_by="column", threads=True)
        raw.columns.names = ["field", "ticker"]
        raw.to_pickle(EXTRA_CACHE)
    raw.columns.names = ["field", "ticker"]
    out = {}
    for t in EXTRA_TICKERS:
        d = raw.xs(t, axis=1, level="ticker")[["Open", "High", "Low", "Close", "Volume"]]
        out[t] = d.reindex(calendar).ffill(limit=FFILL_LIMIT)
    return out


def extra_features(px: dict, ex: dict) -> tuple[pd.DataFrame, dict]:
    spy = px["SPY"]
    c, o, h, l, v = spy["Close"], spy["Open"], spy["High"], spy["Low"], spy["Volume"]
    r = log_ret(c)
    vix = px["^VIX"]["Close"]
    fam: dict[str, dict] = {k: {} for k in
                            ["volsurface", "rates", "credit", "global", "breadth", "structure", "calendar2"]}

    # --- volatility surface -------------------------------------------------------------
    v3m, vvix, skew = ex["^VIX3M"]["Close"], ex["^VVIX"]["Close"], ex["^SKEW"]["Close"]
    f = fam["volsurface"]
    f["vs_vix_vix3m_ratio"] = np.log(vix / v3m)                # >0 = inverted term structure (stress)
    f["vs_vix_vix3m_ratio_chg5"] = f["vs_vix_vix3m_ratio"].diff(5)
    f["vs_vix_vix3m_ratio_z60"] = zscore(f["vs_vix_vix3m_ratio"], 60)
    f["vs_vix3m_log_chg5"] = log_ret(v3m, 5)
    f["vs_vvix_level"] = vvix
    f["vs_vvix_z60"] = zscore(vvix, 60)
    f["vs_vvix_over_vix"] = np.log(vvix / vix)
    f["vs_skew_level"] = skew
    f["vs_skew_z60"] = zscore(skew, 60)
    f["vs_skew_chg5"] = skew.diff(5)

    # --- rates ---------------------------------------------------------------------------
    tnx, fvx, irx = ex["^TNX"]["Close"], ex["^FVX"]["Close"], ex["^IRX"]["Close"]
    f = fam["rates"]
    f["rt_10y_level"] = tnx
    f["rt_10y_chg5"] = tnx.diff(5)
    f["rt_10y_chg20"] = tnx.diff(20)
    f["rt_3m_chg20"] = irx.diff(20)
    f["rt_slope_10y_3m"] = tnx - irx
    f["rt_slope_10y_3m_chg20"] = (tnx - irx).diff(20)
    f["rt_slope_10y_5y"] = tnx - fvx
    f["rt_10y_z60"] = zscore(tnx, 60)
    f["rt_10y_vol20"] = tnx.diff().rolling(20).std()

    # --- credit / inflation ------------------------------------------------------------------
    hyg, lqd, tip, ief = ex["HYG"]["Close"], ex["LQD"]["Close"], ex["TIP"]["Close"], px["IEF"]["Close"]
    f = fam["credit"]
    for k in [5, 20]:
        f[f"cr_hyg_minus_ief_ret{k}"] = log_ret(hyg, k) - log_ret(ief, k)
        f[f"cr_lqd_minus_ief_ret{k}"] = log_ret(lqd, k) - log_ret(ief, k)
        f[f"cr_tip_minus_ief_ret{k}"] = log_ret(tip, k) - log_ret(ief, k)
    f["cr_hyg_px_vs_sma50"] = hyg / hyg.rolling(50).mean() - 1
    f["cr_hyg_ret1"] = log_ret(hyg)
    f["cr_hyg_vol20"] = log_ret(hyg).rolling(20).std() * np.sqrt(252)
    f["cr_hyg_spy_corr60"] = log_ret(hyg).rolling(60).corr(r)

    # --- global / macro ------------------------------------------------------------------------
    f = fam["global"]
    for t, n in [("EEM", "eem"), ("EFA", "efa")]:
        for k in [5, 20]:
            f[f"gl_{n}_minus_spy_ret{k}"] = log_ret(ex[t]["Close"], k) - log_ret(c, k)
    f["gl_eem_ret1"] = log_ret(ex["EEM"]["Close"])
    for t, n in [("UUP", "usd"), ("USO", "oil"), ("DBC", "cmdty")]:
        for k in [5, 20]:
            f[f"gl_{n}_ret{k}"] = log_ret(ex[t]["Close"], k)
    f["gl_usd_px_vs_sma50"] = ex["UUP"]["Close"] / ex["UUP"]["Close"].rolling(50).mean() - 1

    # --- breadth -----------------------------------------------------------------------------
    rsp = ex["RSP"]["Close"]
    f = fam["breadth"]
    for k in [1, 5, 20, 60]:
        f[f"br_rsp_minus_spy_ret{k}"] = log_ret(rsp, k) - log_ret(c, k)

    # --- return structure ------------------------------------------------------------------------
    f = fam["structure"]
    vol20 = r.rolling(20).std()
    for k in [1, 5, 20, 60]:
        f[f"st_volscaled_ret{k}"] = log_ret(c, k) / (vol20 * np.sqrt(k))
    var1 = r.rolling(60).var()
    var5 = log_ret(c, 5).rolling(60).var()
    f["st_variance_ratio_5"] = var5 / (5 * var1)
    f["st_autocorr1_60"] = r.rolling(60).corr(r.shift(1))
    f["st_drawdown_20"] = c / c.rolling(20).max() - 1
    f["st_runup_20"] = c / c.rolling(20).min() - 1
    dollar_vol = (c * v).replace(0, np.nan)
    illiq = (r.abs() / dollar_vol) * 1e9
    f["st_amihud_z60"] = zscore(illiq.rolling(5).mean(), 60)
    f["st_ret_volume_corr20"] = r.rolling(20).corr(np.log(v.replace(0, np.nan)).diff())
    up = (r > 0).astype(int)
    grp = (up != up.shift()).cumsum()
    streak = up.groupby(grp).cumcount() + 1
    f["st_streak_signed"] = np.where(up == 1, streak, -streak).astype(float)
    on = np.log(o / c.shift(1))
    intra = np.log(c / o)
    f["st_overnight_cum20"] = on.rolling(20).sum()
    f["st_intraday_cum20"] = intra.rolling(20).sum()
    f["st_overnight_minus_intraday20"] = f["st_overnight_cum20"] - f["st_intraday_cum20"]
    f["st_downside_vol_ratio60"] = (r.clip(upper=0).pow(2).rolling(60).mean()
                                    / r.pow(2).rolling(60).mean())
    f["st_kurt_60"] = r.rolling(60).kurt()
    f["st_dist_20d_high"] = c / h.rolling(20).max() - 1
    f["st_dist_20d_low"] = c / l.rolling(20).min() - 1

    # --- calendar (only uses the date itself and the published exchange calendar up to t) ---------
    idx = spy.index
    f = fam["calendar2"]
    dom = pd.Series(idx.day, index=idx)
    f["cal2_turn_of_month"] = ((dom >= 26) | (dom <= 3)).astype(float).to_numpy()
    # OpEx week = the Mon..Fri week whose Friday is the month's third Friday (pure date arithmetic,
    # so it does not depend on which sessions are in the data)
    week_fri = idx + pd.to_timedelta(4 - idx.weekday, unit="D")
    f["cal2_opex_week"] = ((week_fri.day >= 15) & (week_fri.day <= 21)).astype(float)
    f["cal2_days_into_month"] = dom.to_numpy() / 31.0

    families = {}
    frames = []
    for k, d in fam.items():
        df = pd.DataFrame(d, index=idx)
        families[k] = list(df.columns)
        frames.append(df)
    E = pd.concat(frames, axis=1).replace([np.inf, -np.inf], np.nan)
    return E, families


def baseline_families(cols: list[str]) -> dict:
    fam = {"spy_returns": [], "spy_vol_range": [], "spy_trend": [], "spy_volume": [], "calendar": [],
           "xasset_returns": [], "xasset_relative": [], "sector_breadth": [], "vix": []}
    for c in cols:
        if c.startswith("cal_"):
            fam["calendar"].append(c)
        elif c.startswith("spy_volume"):
            fam["spy_volume"].append(c)
        elif c.startswith("spy_") and ("_ret" in c):
            fam["spy_returns"].append(c)
        elif c.startswith("spy_") and any(s in c for s in ["vol", "range", "atr", "parkinson", "skew", "close_loc"]):
            fam["spy_vol_range"].append(c)
        elif c.startswith("spy_"):
            fam["spy_trend"].append(c)
        elif c.startswith("vix_"):
            fam["vix"].append(c)
        elif c.startswith("sector_") or c == "corr_spy_tlt_60d" or c == "tlt_vol_20d":
            fam["sector_breadth"].append(c)
        elif "minus" in c:
            fam["xasset_relative"].append(c)
        else:
            fam["xasset_returns"].append(c)
    return fam


def build_all(px: dict, ex: dict) -> tuple[pd.DataFrame, dict]:
    B = build_features(px)
    E, efam = extra_features(px, ex)
    fam = {**baseline_families(list(B.columns)), **efam}
    return pd.concat([B, E], axis=1), fam


def truncation_test(px: dict, ex: dict, n_dates: int = 8, seed: int = 0) -> float:
    full, _ = build_all(px, ex)
    rng = np.random.default_rng(seed)
    dates = rng.choice(full.loc["2008-06-01":"2025-06-30"].index, n_dates, replace=False)
    worst = 0.0
    for d in sorted(dates):
        tr, _ = build_all({k: v.loc[:d] for k, v in px.items()}, {k: v.loc[:d] for k, v in ex.items()})
        a, b = tr.loc[d], full.loc[d]
        assert (a.isna() == b.isna()).all(), f"NaN pattern differs at {d}"
        diff = (a - b).abs()[~a.isna()]
        if diff.max() > 1e-9:
            print("  mismatch at", d.date(), diff[diff > 1e-9].to_dict())
        worst = max(worst, float(diff.max()))
    return worst


if __name__ == "__main__":
    px = pd.read_pickle(DATA_DIR / "aligned_prices.pkl")
    ex = load_extra(px["SPY"].index)
    F, fam = build_all(px, ex)
    print({k: len(v) for k, v in fam.items()}, "total", F.shape[1])
    s = F.loc["2008-01-01":"2025-12-31"]
    nan_cols = s.isna().sum()
    print("columns with NaN in study window:\n", nan_cols[nan_cols > 0].sort_values(ascending=False).head(20))
    raw_missing = {t: int(d["Close"].loc["2008-01-01":].isna().sum()) for t, d in ex.items()}
    print("extra tickers: sessions still missing after ffill(<=3):", raw_missing)
    print("truncation test max |diff| =", truncation_test(px, ex))

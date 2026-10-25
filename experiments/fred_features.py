"""Non-Yahoo features from FRED (Federal Reserve Bank of St. Louis), made point-in-time.

Raw files: data/fred/<SERIES>.csv, downloaded from https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>.

Each observation is only used once it would have been PUBLISHED (conservative release lags):
  daily market series (H.15 rates, Moody's spread, TIPS breakevens) -> 1 session after the observation date
  DTWEXBGS (H.10 dollar index, published weekly on Mondays)         -> 6 sessions
  STLFSI4 (week ending Friday, released the following Thursday)     -> 7 calendar days
  ICSA (initial claims, week ending Saturday, released Thursday)    -> 6 calendar days
  USEPUINDXD (daily news-based Economic Policy Uncertainty)         -> 2 sessions
Known limitation: FRED serves the latest vintage. Claims and EPU are revised slightly after first release;
the series with heavy revisions (NFCI, WEI) were deliberately excluded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import DATA_DIR, zscore

FRED_DIR = DATA_DIR / "fred"
DAILY_LAG_SESSIONS = {"BAA10Y": 1, "T10Y2Y": 1, "T10Y3M": 1, "DFII10": 1, "T10YIE": 1,
                      "DTWEXBGS": 6, "USEPUINDXD": 2}
WEEKLY_LAG_DAYS = {"STLFSI4": 7, "ICSA": 6}


def read_series(sid: str) -> pd.Series:
    d = pd.read_csv(FRED_DIR / f"{sid}.csv", na_values=".")
    d.columns = ["date", sid]
    d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date")[sid].astype(float).dropna()


def point_in_time(raw: dict, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Value of each series known at the close of each SPY session."""
    out = {}
    for sid, lag in DAILY_LAG_SESSIONS.items():
        s = raw[sid]
        # value observed on/before session t is published `lag` sessions later
        on_cal = s.reindex(calendar.union(s.index)).ffill().reindex(calendar)
        out[sid] = on_cal.shift(lag)
    for sid, days in WEEKLY_LAG_DAYS.items():
        s = raw[sid].copy()
        s.index = s.index + pd.Timedelta(days=days)          # availability date
        out[sid] = s.reindex(calendar.union(s.index)).ffill().reindex(calendar)
    return pd.DataFrame(out, index=calendar)


def fred_features(raw: dict, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    P = point_in_time(raw, calendar)
    f = {}
    for sid, n in [("BAA10Y", "credit_baa"), ("T10Y2Y", "curve_10y2y"), ("T10Y3M", "curve_10y3m"),
                   ("DFII10", "real_10y"), ("T10YIE", "breakeven_10y")]:
        s = P[sid]
        f[f"fred_{n}_level"] = s
        f[f"fred_{n}_chg20"] = s.diff(20)
    f["fred_credit_baa_chg5"] = P["BAA10Y"].diff(5)
    f["fred_credit_baa_z60"] = zscore(P["BAA10Y"], 60)
    usd = np.log(P["DTWEXBGS"])
    f["fred_usd_broad_ret20"] = usd.diff(20)
    f["fred_usd_broad_z60"] = zscore(usd, 60)
    f["fred_stress_level"] = P["STLFSI4"]
    f["fred_stress_chg20"] = P["STLFSI4"].diff(20)
    lc = np.log(P["ICSA"])
    f["fred_claims_vs_1y"] = lc - lc.rolling(252).mean()
    f["fred_claims_chg20"] = lc.diff(20)
    epu = np.log(P["USEPUINDXD"].clip(lower=1))
    epu7 = epu.rolling(5).mean()
    f["fred_epu_5d"] = epu7
    f["fred_epu_z60"] = zscore(epu7, 60)
    f["fred_epu_chg20"] = epu7.diff(20)
    return pd.DataFrame(f, index=calendar).replace([np.inf, -np.inf], np.nan)


def load_fred(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    raw = {sid: read_series(sid) for sid in [*DAILY_LAG_SESSIONS, *WEEKLY_LAG_DAYS]}
    return fred_features(raw, calendar)


def truncation_test(calendar: pd.DatetimeIndex, n_dates: int = 25, seed: int = 0) -> float:
    """Recompute with every raw series cut at observation date d: values at d must not change."""
    raw = {sid: read_series(sid) for sid in [*DAILY_LAG_SESSIONS, *WEEKLY_LAG_DAYS]}
    full = fred_features(raw, calendar)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for d in sorted(rng.choice(calendar[(calendar >= "2008-06-01") & (calendar <= "2025-06-30")], n_dates, replace=False)):
        cut = fred_features({k: v.loc[:d] for k, v in raw.items()}, calendar)
        a, b = cut.loc[d], full.loc[d]
        assert (a.isna() == b.isna()).all(), d
        worst = max(worst, float((a - b).abs().max()))
    return worst


if __name__ == "__main__":
    cal = pd.read_pickle(DATA_DIR / "aligned_prices.pkl")["SPY"].index
    F = load_fred(cal)
    s = F.loc["2008-01-01":"2025-12-31"]
    print(F.shape[1], "FRED features; NaN rows in study window:", int(s.isna().any(axis=1).sum()))
    print("truncation test max |diff| =", truncation_test(cal))

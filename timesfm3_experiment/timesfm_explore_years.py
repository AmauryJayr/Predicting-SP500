"""EXPLORATORY (post-test, h=5 only): what the TimesFM score tracks, year by year.

Usage:  python timesfm_explore_years.py  -> results/timesfm_exploratory_years.csv
                                            results/timesfm_exploratory_stretches.csv

The pre-registered `tfm.h5` TEST frame is scored per calendar year against two trailing-trend
readings of SPY -- bounce20 (-r20, mean reversion of the 20-session log change) and keep256
(+r256, continuation of the 256-session one) -- and then over the 2020 / 2022 / 2025 falls and the
recoveries that followed them. Nothing here changes a pre-registered result.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tf_paths import RES_DIR
from common import DATA_DIR

KEY = ("tfm.h5", "timesfm")
YEARS = RES_DIR / "timesfm_exploratory_years.csv"
STRETCHES_CSV = RES_DIR / "timesfm_exploratory_stretches.csv"

# (label, kind, start, end, ref): rows are start <= date <= end, move = Close[end] / Close[ref] - 1
STRETCHES = [("2020 fall", "fall", "2020-02-20", "2020-03-23", "2020-02-19"),
             ("2020 recovery", "recovery", "2020-03-24", "2020-08-31", "2020-03-23"),
             ("2022 fall 1", "fall", "2022-01-04", "2022-03-08", "2022-01-03"),
             ("2022 rebound 1", "recovery", "2022-03-09", "2022-03-29", "2022-03-08"),
             ("2022 fall 2", "fall", "2022-03-30", "2022-06-16", "2022-03-29"),
             ("2022 rebound 2", "recovery", "2022-06-17", "2022-08-16", "2022-06-16"),
             ("2022 fall 3", "fall", "2022-08-17", "2022-10-12", "2022-08-16"),
             ("2022 rebound 3", "recovery", "2022-10-13", "2022-11-30", "2022-10-12"),
             ("2025 fall", "fall", "2025-02-20", "2025-04-08", "2025-02-19"),
             ("2025 recovery", "recovery", "2025-04-09", "2025-09-30", "2025-04-08")]
POOLS = [("2022 falls (all)", "fall", ["2022 fall 1", "2022 fall 2", "2022 fall 3"]),
         ("2022 rebounds (all)", "recovery", ["2022 rebound 1", "2022 rebound 2", "2022 rebound 3"])]


def frame():
    """The tfm.h5 TEST frame plus the two trailing log changes of SPY on its dates, and Close."""
    p = pd.read_pickle(RES_DIR / "timesfm_test_predictions.pkl")[KEY]
    close = pd.read_pickle(DATA_DIR / "aligned_prices.pkl")["SPY"]["Close"].astype(float)
    ls = np.log(close)
    t = p.assign(r20=ls.diff(20).reindex(p.index), r256=ls.diff(256).reindex(p.index))
    assert t.notna().all().all(), "missing trailing window on a prediction date"
    return t, close


def block(g) -> dict:
    """Calls, hit rate and the three rankings on one set of rows."""
    return {"n": len(g), "up_rate": g["y"].mean(), "mean_proba": g["proba"].mean(),
            "called_up": g["pred"].mean(), "accuracy": (g["pred"] == g["y"]).mean(),
            "auc_timesfm": roc_auc_score(g["y"], g["proba"]),
            "auc_bounce20": roc_auc_score(g["y"], -g["r20"]),
            "auc_keep256": roc_auc_score(g["y"], g["r256"]),
            "corr_r20": g["proba"].corr(g["r20"]), "corr_r256": g["proba"].corr(g["r256"])}


def terciles(g) -> dict:
    """Up rate of the rows the score likes most / least (its own terciles)."""
    lo, hi = g["proba"].quantile(1 / 3), g["proba"].quantile(2 / 3)
    return {"top_third_up": g["y"][g["proba"] >= hi].mean(),
            "bottom_third_up": g["y"][g["proba"] <= lo].mean()}


def years(t) -> pd.DataFrame:
    rows = [{"year": str(y), **block(g), **terciles(g)} for y, g in t.groupby(t.index.year)]
    return pd.DataFrame(rows + [{"year": "all", **block(t)}])


def stretches(t, close) -> pd.DataFrame:
    parts, rows = {}, []
    for lab, kind, a, b, ref in STRETCHES:
        g = t.loc[a:b]
        parts[lab] = g
        rows.append({"label": lab, "kind": kind, "start": a, "end": b, "ref": ref, "n": len(g),
                     "move": close.loc[b] / close.loc[ref] - 1, "mean_proba": g["proba"].mean(),
                     "called_up": g["pred"].mean(), "up_rate": g["y"].mean()})
    for lab, kind, legs in POOLS:
        g = pd.concat([parts[x] for x in legs])
        rows.append({"label": lab, "kind": kind, "start": legs[0], "end": legs[-1], "ref": "",
                     "n": len(g), "move": np.nan, "mean_proba": g["proba"].mean(),
                     "called_up": g["pred"].mean(), "up_rate": g["y"].mean()})
    return pd.DataFrame(rows)


def main():
    t, close = frame()
    Y, S = years(t), stretches(t, close)
    Y.to_csv(YEARS, index=False)
    S.to_csv(STRETCHES_CSV, index=False)
    with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
        print(f"tfm.h5 TEST rows {t.index[0].date()} .. {t.index[-1].date()}, thr {t['thr'].iloc[0]:.3f}\n")
        print(Y.to_string(index=False))
        y = Y[Y.year != "all"]
        print(f"\nacross the 7 years: corr(auc_timesfm, auc_bounce20) = "
              f"{y['auc_timesfm'].corr(y['auc_bounce20']):.3f}, corr(auc_timesfm, auc_keep256) = "
              f"{y['auc_timesfm'].corr(y['auc_keep256']):.3f}")
        print(f"\n{S.to_string(index=False)}")


if __name__ == "__main__":
    main()

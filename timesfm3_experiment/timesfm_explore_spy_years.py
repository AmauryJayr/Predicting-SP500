"""DESCRIPTIVE: what SPY did in each test year (2019-2025), for the "In short" section at the top of the report.

Usage:  python timesfm_explore_spy_years.py  -> results/timesfm_spy_years.csv

Per calendar year, from the SPY Close in aligned_prices.pkl:
  ret          Close at the last day of the year / Close at the last day of the year before - 1
  max_fall     deepest fall from a running high, measured from the previous year-end close; with its peak and trough dates
  recovered    first date after the trough when Close is back at or above that peak (blank if not within the year)
  vol          annualised sd of daily log returns
  big_days     days with |daily move| > 2%
  up5          share of the year's 5-day test windows that went up (Dataset(5) labels)
  auc_h1/h5    TimesFM-3 test AUC (results/timesfm_test_by_year.csv is the source in the report; recomputed here)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tf_paths import RES_DIR
from common import DATA_DIR


def main():
    c = pd.read_pickle(DATA_DIR / "aligned_prices.pkl")["SPY"]["Close"].astype(float)
    r = np.log(c).diff()
    preds = pd.read_pickle(RES_DIR / "timesfm_test_predictions.pkl")
    rows = []
    for yr in range(2019, 2026):
        prev = c[c.index.year == yr - 1].iloc[-1]
        cy = c[c.index.year == yr]
        path = pd.concat([pd.Series([prev], index=[cy.index[0] - pd.Timedelta(days=1)]), cy])
        dd = path / path.cummax() - 1
        trough = dd.idxmin()
        peak = path.loc[:trough].idxmax()
        back = cy.loc[trough:][cy.loc[trough:] >= path[peak]]
        row = {"year": yr, "ret": cy.iloc[-1] / prev - 1, "max_fall": dd.min(), "peak": peak.date(),
               "trough": trough.date(), "recovered": back.index[0].date() if len(back) else "",
               "vol": r[r.index.year == yr].std() * np.sqrt(252), "big_days": int((r[r.index.year == yr].abs() > 0.02).sum())}
        for h in (1, 5):
            p = preds[(f"tfm.h{h}", "timesfm")]
            p = p[p.index.year == yr]
            if h == 5:
                row["up5"] = p["y"].mean()
            row[f"auc_h{h}"] = roc_auc_score(p["y"], p["proba"])
        rows.append(row)
    R = pd.DataFrame(rows)
    R.to_csv(RES_DIR / "timesfm_spy_years.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

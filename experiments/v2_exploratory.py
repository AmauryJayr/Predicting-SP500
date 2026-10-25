"""EXPLORATORY (run after the one-shot test; not preregistered).
  (a) mechanism of the spy_tlt_corr.5 gain: coefficients of the linear HAR-corr model (c21, c63, c252) when refit
      on an expanding window at each year start, vs the trained-once fit; implied long-run mean a / (1 - sum b).
  (b) block-length sensitivity of the corr comparisons (L = 21, 42, 63, 126, 252).
Outputs: results/v2_expl_corr_coefficients.csv, results/v2_expl_blocklen.csv
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import v2_lib as L


def coefficients():
    tg = L.build_targets()["spy_tlt_corr"]
    sp = {"model": "lin", "feats": "hc", "method": "fixed"}
    rows = []
    for yr in range(2013, 2026):
        t0 = int(np.searchsorted(tg.dates, np.datetime64(f"{yr}-01-01")))
        m = L.fit(tg, sp, t0)
        sc, rg = m.named_steps["sc"], m.named_steps["m"]
        b = rg.coef_ / sc.scale_
        a = rg.intercept_ - (b * sc.mean_).sum()
        rows.append(dict(fit_date=str(tg.index[t0].date()), intercept=a, b_c21=b[0], b_c63=b[1], b_c252=b[2],
                         sum_b=b.sum(), implied_longrun_mean=a / (1 - b.sum()),
                         mean_target_known=tg.y[:tg.n_known(t0)].mean()))
    return pd.DataFrame(rows)


def blocklen():
    P = pd.read_pickle(L.RES_DIR / "v2_test_predictions.pkl")["preds"]
    tg = L.build_targets()["spy_tlt_corr"]
    s, e = tg.period("TEST"); rows = np.arange(s, e)
    out = []
    pairs = [("spy_tlt_corr.5", "fixed:lin/hc/fixed"), ("spy_tlt_corr.5", "base:persist_63"),
             ("spy_tlt_corr.2", "fixed:hgb/full/fixed"), ("spy_tlt_corr.2", "base:hc_fixed"),
             ("spy_tlt_corr.2", "base:persist_63")]
    for a, b in pairs:
        for Lb in (21, 42, 63, 126, 252):
            r = L.paired_boot(tg, rows, P[("spy_tlt_corr", a)], P[("spy_tlt_corr", b)], L=Lb)
            out.append(dict(a=a, b=b, block=Lb, **{k: r[k] for k in ["d_mse", "d_mse_lo", "d_mse_hi", "p_mse",
                                                                       "d_ba", "d_ba_lo", "d_ba_hi"]}))
    return pd.DataFrame(out)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    C = coefficients(); C.to_csv(L.RES_DIR / "v2_expl_corr_coefficients.csv", index=False)
    print(C.round(3).to_string(index=False))
    B = blocklen(); B.to_csv(L.RES_DIR / "v2_expl_blocklen.csv", index=False)
    print(B.round(4).to_string(index=False))

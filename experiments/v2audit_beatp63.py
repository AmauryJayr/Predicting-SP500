"""V2 PESSIMIST AUDIT, EXPLORATORY (post-test; everything here was designed after seeing the 2019-2025 results).
 (1) Is there a sensible adaptive model that beats 63-day persistence on the SPY-TLT correlation target?
     Candidates are evaluated on VAL (2017-18, pre-test) AND TEST so the reader can see whether validation would
     have pointed to them.
 (2) Mechanism of the refit gain: coefficient path, prediction-difference decomposition into intercept vs slope
     changes, bias^2 / variance decomposition by year, hybrid models.
Outputs results/v2audit_beatp63_*.csv
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy import stats
import v2_lib as L
from v2audit_robust import boot, dm

pd.set_option("display.width", 250)
L.PERIODS["ALL"] = ("2013-01-01", "2025-12-31")


def combos(tg, s, e, comps, N_inv=252, N_ols=504):
    """comps: dict name -> predictions on [s, e). Causal combinations using only rows whose label is known."""
    y = tg.y
    a, b = comps["p63"], comps["r21"]
    out = {k: np.empty(e - s) for k in ["avg50", "invmse252", "stack504", "hedge5"]}
    names = list(comps)
    M = np.column_stack([comps[k] for k in names])
    for t in range(s, e):
        nk = min(tg.n_known(t), e)
        i = t - s
        out["avg50"][i] = 0.5 * (a[i] + b[i])
        lo = max(s, nk - N_inv)
        if nk - s < 63:
            out["invmse252"][i] = out["avg50"][i]; out["stack504"][i] = out["avg50"][i]
            out["hedge5"][i] = M[i].mean(); continue
        ea = ((y[lo:nk] - a[lo - s:nk - s]) ** 2).mean(); eb = ((y[lo:nk] - b[lo - s:nk - s]) ** 2).mean()
        w = (1 / ea) / (1 / ea + 1 / eb)
        out["invmse252"][i] = w * a[i] + (1 - w) * b[i]
        lo2 = max(s, nk - N_ols)
        Xs = np.column_stack([np.ones(nk - lo2), a[lo2 - s:nk - s], b[lo2 - s:nk - s]])
        coef = np.linalg.lstsq(Xs, y[lo2:nk], rcond=None)[0]
        out["stack504"][i] = np.clip(coef @ [1, a[i], b[i]], -0.99, 0.99)
        idx = np.arange(s, nk); disc = 0.5 ** ((t - idx) / 63)
        Lk = (disc[:, None] * (y[idx, None] - M[idx - s]) ** 2).sum(0) / disc.sum()
        ww = np.exp(-2 * (Lk / Lk.min() - 1)); ww /= ww.sum()
        out["hedge5"][i] = M[i] @ ww
    return out


def candidates(tg):
    s, e = tg.period("ALL")
    comps = {"p63": L.run(tg, {"method": "raw", "col": "c63"}, "ALL"),
             "p21": L.run(tg, {"method": "raw", "col": "c21"}, "ALL"),
             "r21": L.run(tg, {"model": "lin", "feats": "hc", "method": "refit", "R": 21, "W": None}, "ALL"),
             "r21_w504": L.run(tg, {"model": "lin", "feats": "hc", "method": "refit", "R": 21, "W": 504}, "ALL"),
             "p63_ar_refit": L.run(tg, {"model": "lin", "feats": "p63", "method": "refit", "R": 21, "W": None}, "ALL")}
    C = dict(comps)
    C.update(combos(tg, s, e, {k: comps[k] for k in ["p63", "p21", "r21", "r21_w504", "p63_ar_refit"]}))
    # rolling-504 AR shrinkage of c63 toward its recent mean (refit 21)
    C["p63_ar_w504"] = L.run(tg, {"model": "lin", "feats": "p63", "method": "refit", "R": 21, "W": 504}, "ALL")
    return s, e, C


def mechanism(tg):
    s, e = tg.period("TEST"); rows = np.arange(s, e); dates = tg.index[rows]; y = tg.y[rows]
    X = tg.X("hc")
    sp = {"model": "lin", "feats": "hc", "method": "fixed"}

    def coef(t0):
        m = L.fit(tg, sp, t0); sc, rg = m.named_steps["sc"], m.named_steps["m"]
        b = rg.coef_ / sc.scale_
        return rg.intercept_ - (b * sc.mean_).sum(), b

    a0, b0 = coef(s)
    path, hyb_int, hyb_slope, pr = [], np.empty(len(rows)), np.empty(len(rows)), np.empty(len(rows))
    for bk in range(s, e, 21):
        a, b = coef(bk)
        ee = min(bk + 21, e)
        xx = X[bk:ee]
        pr[bk - s:ee - s] = a + xx @ b
        # hybrid 1: fixed slopes, intercept re-estimated so the in-sample mean matches the refit (level-only update)
        nk = tg.n_known(bk)
        a_lvl = tg.y[:nk].mean() - X[:nk].mean(0) @ b0
        hyb_int[bk - s:ee - s] = a_lvl + xx @ b0
        # hybrid 2: refit slopes, but intercept chosen so the long-run mean stays at the 2019 value
        mu0 = a0 / (1 - b0.sum())
        hyb_slope[bk - s:ee - s] = mu0 * (1 - b.sum()) + xx @ b
        path.append(dict(fit_date=str(tg.index[bk].date()), intercept=a, b_c21=b[0], b_c63=b[1], b_c252=b[2],
                         sum_b=b.sum(), implied_mean=a / (1 - b.sum()),
                         d_intercept_part=a - a0, d_slope_part_mean=float((xx @ (b - b0)).mean())))
    pf = a0 + X[s:e] @ b0
    clipf = lambda p: np.clip(p, -0.99, 0.99)
    pr, pf, hyb_int, hyb_slope = map(clipf, (pr, pf, hyb_int, hyb_slope))
    p63 = tg.col("c63")[s:e]
    P = pd.DataFrame(path)
    dec = []
    for yr in list(sorted(set(dates.year))) + ["2019-21", "2022-25", "all"]:
        m = (dates.year == yr) if isinstance(yr, (int, np.integer)) else \
            (dates.year < 2022) if yr == "2019-21" else (dates.year >= 2022) if yr == "2022-25" else np.ones(len(rows), bool)
        for nm, p in (("fixed", pf), ("refit21", pr), ("fixed_slopes+level_update", hyb_int),
                      ("refit_slopes+fixed_mean", hyb_slope), ("persist63", p63)):
            err = p[m] - y[m]
            dec.append(dict(period=str(yr), model=nm, mse=(err ** 2).mean(), bias=err.mean(), bias2=err.mean() ** 2,
                            err_var=err.var(), corr_with_y=np.corrcoef(p[m], y[m])[0, 1],
                            pred_sd=p[m].std(), y_sd=y[m].std()))
    return P, pd.DataFrame(dec)


def main():
    tg = L.build_targets()["spy_tlt_corr"]
    s, e, C = candidates(tg)
    out = []
    for per in ["VAL", "TEST"]:
        ps, pe = tg.period(per)
        rows = np.arange(ps, pe); sl = slice(ps - s, pe - s)
        y = tg.y[rows]; p63 = C["p63"][sl]
        for nm, p in C.items():
            q = p[sl]
            b = boot(tg, rows, q, p63)
            d = dm(y - q, y - p63, 42)
            out.append(dict(period=per, model=nm, mse=((y - q) ** 2).mean(), rel_vs_p63=1 - ((y - q) ** 2).mean() / ((y - p63) ** 2).mean(),
                            d_p63=b["d_mse"], lo=b["d_mse_lo"], hi=b["d_mse_hi"], p_boot=b["p_mse"], p_dm42=d["p_hln"],
                            ba=L.metrics_rows(tg, rows, q)["ba"]))
    R = pd.DataFrame(out); R.to_csv(L.RES_DIR / "v2audit_beatp63_candidates.csv", index=False)
    print("(1) candidates vs persistence-63 (exploratory; d>0 = candidate better)\n", R.round(4).to_string(index=False))
    # subperiods for the best few on TEST
    ps, pe = tg.period("TEST"); rows = np.arange(ps, pe); dates = tg.index[rows]; sl = slice(ps - s, pe - s)
    sub = []
    for nm in C:
        for lab, m in (("2019-21", dates.year < 2022), ("2022-25", dates.year >= 2022)):
            b = boot(tg, rows[m], C[nm][sl][m], C["p63"][sl][m])
            sub.append(dict(model=nm, sub=lab, d_p63=b["d_mse"], lo=b["d_mse_lo"], hi=b["d_mse_hi"]))
    S = pd.DataFrame(sub); S.to_csv(L.RES_DIR / "v2audit_beatp63_subperiods.csv", index=False)
    print(S.round(4).to_string(index=False))

    P, D = mechanism(tg)
    P.to_csv(L.RES_DIR / "v2audit_beatp63_coefpath.csv", index=False)
    D.to_csv(L.RES_DIR / "v2audit_beatp63_decomp.csv", index=False)
    print("\n(2) coefficient path (every 4th refit)\n", P.iloc[::4].round(3).to_string(index=False))
    print("\n    bias/variance decomposition\n",
          D[D.period.isin(["2019-21", "2022-25", "all", "2022", "2023"])].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

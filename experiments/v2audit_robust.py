"""V2 PESSIMIST AUDIT: robustness of the spy_tlt_corr.5 win (lin/hc expanding refit 21 vs trained once).
All post-test, exploratory. Outputs results/v2audit_robust_*.csv and stdout.
 A. reproduce .5 and fixed predictions from scratch, compare with the pickled test predictions
 B. block length 21/63/126/252 (moving block) for .5 vs fixed and vs persist-63
 C. the whole linear grid on TEST (refit freq 5/21/63/252, rolling, decay, RLS, detectors, ensembles, bias, recal)
    vs fixed and vs persist-63
 D. alternative test start dates (2015 ... 2023, end 2025-12) and alternative end dates
 E. non-overlapping monthly evaluation (every 21st row, all 21 phase offsets) + effective sample size
 F. Diebold-Mariano with Newey-West HAC (lags 20/42/63/126) + Harvey-Leybourne-Newbold correction
 G. leave-one-year-out (evaluation) and single-year contributions
 H. Holm over all 36 confirmatory MSE tests (3 targets) instead of per target
"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
import v2_lib as L

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
FX = {"model": "lin", "feats": "hc", "method": "fixed"}
R21 = {"model": "lin", "feats": "hc", "method": "refit", "R": 21, "W": None}
P63 = {"method": "raw", "col": "c63"}


def boot(tg, rows, pa, pb, Lb=63, n=2000):
    r = L.paired_boot(tg, rows, pa, pb, n_boot=n, L=Lb)
    return {k: r[k] for k in ["d_mse", "d_mse_lo", "d_mse_hi", "p_mse", "d_ba"]}


def nw_var(d, lag):
    d = d - d.mean(); n = len(d)
    v = d @ d / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (d[k:] @ d[:-k]) / n
    return v


def dm(ea, eb, lag, h=21):
    """DM test of H0: E[eb^2 - ea^2] <= 0 (positive = a better). HAC NW Bartlett; HLN small-sample factor."""
    d = eb ** 2 - ea ** 2; n = len(d)
    v = nw_var(d, lag)
    t = d.mean() / np.sqrt(v / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return dict(d=d.mean(), t_dm=t, p_dm=1 - stats.norm.cdf(t), t_hln=t * hln, p_hln=1 - stats.t.cdf(t * hln, n - 1))


def ess(d, maxlag=126):
    d = d - d.mean(); n = len(d); g0 = d @ d / n
    rho = [(d[k:] @ d[:-k]) / n / g0 for k in range(1, maxlag + 1)]
    s = 1 + 2 * sum((1 - k / (maxlag + 1)) * r for k, r in enumerate(rho, 1))
    return n / s


def lin_grid_job(spec, period="TEST"):
    tg = L.build_targets()["spy_tlt_corr"]
    p = L.run(tg, spec, period, cache={})
    fires = getattr(L.m_detect, "last_fires", None) if spec["method"] == "detect" else None
    return L.key(spec), p, fires


def main():
    T = L.build_targets(); tg = T["spy_tlt_corr"]
    s, e = tg.period("TEST"); rows = np.arange(s, e); dates = tg.index[rows]; y = tg.y[rows]
    PK = pd.read_pickle(L.RES_DIR / "v2_test_predictions.pkl")["preds"]
    p5, pf, p63 = L.run(tg, R21, "TEST"), L.run(tg, FX, "TEST"), L.run(tg, P63, "TEST")
    print("A. reproduce: max|.5 - pickle| =", np.abs(p5 - PK[("spy_tlt_corr", "spy_tlt_corr.5")]).max(),
          " max|fixed - pickle| =", np.abs(pf - PK[("spy_tlt_corr", "fixed:lin/hc/fixed")]).max())
    mse = lambda p, m=slice(None): ((y[m] - p[m]) ** 2).mean()
    print(f"   MSE fixed {mse(pf):.4f}  .5 {mse(p5):.4f}  p63 {mse(p63):.4f}")

    # B. block lengths
    out = []
    for Lb in (21, 63, 126, 252):
        for nm, pb in (("fixed", pf), ("persist_63", p63)):
            out.append(dict(vs=nm, block=Lb, **boot(tg, rows, p5, pb, Lb)))
    B = pd.DataFrame(out); B.to_csv(L.RES_DIR / "v2audit_robust_blocklen.csv", index=False)
    print("\nB. block length\n", B.round(4).to_string(index=False))

    # C. full linear grid on TEST
    specs = [sp for sp in L.grid("spy_tlt_corr") if sp["model"] == "lin" and sp["method"] != "fixed"]
    specs += [{"model": "lin", "feats": "hc", "method": "refit", "R": R, "W": W} for R in (5, 63, 252)
              for W in (504, 1008)]
    res = Parallel(n_jobs=15)(delayed(lin_grid_job)(sp) for sp in specs)
    val = pd.read_csv(L.RES_DIR / "v2_validation_log.csv")
    val = val[val.target == "spy_tlt_corr"].pivot_table(index="key", columns="fold", values="rel_vs_fixed")
    out = []
    m22 = np.asarray(dates >= "2022-01-01")
    for k, p, fires in res:
        sp = json.loads(k)
        a = boot(tg, rows, p, pf); b = boot(tg, rows, p, p63)
        out.append(dict(label=L.label(sp), family=L.family(sp), fires=fires, mse=mse(p),
                        rel_vs_fixed=1 - mse(p) / mse(pf), d_fixed=a["d_mse"], lo_fixed=a["d_mse_lo"],
                        hi_fixed=a["d_mse_hi"], p_fixed=a["p_mse"], d_p63=b["d_mse"], lo_p63=b["d_mse_lo"],
                        hi_p63=b["d_mse_hi"], p_p63=b["p_mse"],
                        d_fixed_1921=mse(pf, ~m22) - mse(p, ~m22), d_fixed_2225=mse(pf, m22) - mse(p, m22),
                        val_rel_VAL=val.loc[k, "VAL"] if k in val.index else np.nan,
                        val_rel_mean=val.loc[k].mean() if k in val.index else np.nan))
    C = pd.DataFrame(out).sort_values("mse"); C.to_csv(L.RES_DIR / "v2audit_robust_lingrid.csv", index=False)
    print("\nC. linear grid on TEST (block 63)\n", C.round(4).to_string(index=False))
    sig = C[C.lo_fixed > 0]
    print(f"   {len(sig)}/{len(C)} linear configs have CI vs fixed excluding 0; "
          f"{(C.d_fixed > 0).sum()}/{len(C)} have positive point delta; "
          f"{(C.lo_p63 > 0).sum()} beat persist-63 with CI excluding 0; best d_p63 = {C.d_p63.max():.4f}")
    print("   Spearman(val mean rel, test rel) over linear configs:",
          stats.spearmanr(C.val_rel_mean, C.rel_vs_fixed, nan_policy="omit"))

    # D. alternative start / end dates
    out = []
    for a, b in [("2015-01-01", "2025-12-31"), ("2017-01-01", "2025-12-31"), ("2019-01-01", "2025-12-31"),
                 ("2020-01-01", "2025-12-31"), ("2021-01-01", "2025-12-31"), ("2022-01-01", "2025-12-31"),
                 ("2022-07-01", "2025-12-31"), ("2023-01-01", "2025-12-31"), ("2019-01-01", "2021-12-31"),
                 ("2019-01-01", "2022-12-31"), ("2019-01-01", "2023-12-31"), ("2015-01-01", "2021-12-31")]:
        L.PERIODS["ALT"] = (a, b)
        s2, e2 = tg.period("ALT"); r2 = np.arange(s2, e2)
        q5, qf, q63 = L.run(tg, R21, "ALT"), L.run(tg, FX, "ALT"), L.run(tg, P63, "ALT")
        x, z = boot(tg, r2, q5, qf), boot(tg, r2, q5, q63)
        yy = tg.y[r2]
        out.append(dict(start=a, end=b, n=len(r2), mse_fixed=((yy - qf) ** 2).mean(), mse_r21=((yy - q5) ** 2).mean(),
                        mse_p63=((yy - q63) ** 2).mean(), rel=1 - ((yy - q5) ** 2).mean() / ((yy - qf) ** 2).mean(),
                        d_fixed=x["d_mse"], lo=x["d_mse_lo"], hi=x["d_mse_hi"], p=x["p_mse"],
                        d_p63=z["d_mse"], lo_p63=z["d_mse_lo"], hi_p63=z["d_mse_hi"]))
    D = pd.DataFrame(out); D.to_csv(L.RES_DIR / "v2audit_robust_startdates.csv", index=False)
    print("\nD. alternative evaluation windows (fixed model trained once at window start)\n", D.round(4).to_string(index=False))

    # E. non-overlapping monthly evaluation
    ea, eb, ep = y - p5, y - pf, y - p63
    d = eb ** 2 - ea ** 2
    out = []
    for off in range(21):
        idx = np.arange(off, len(rows), 21)
        dd = d[idx]; n = len(dd)
        t = dd.mean() / (dd.std(ddof=1) / np.sqrt(n))
        dp = ep[idx] ** 2 - ea[idx] ** 2
        tp = dp.mean() / (dp.std(ddof=1) / np.sqrt(n))
        out.append(dict(offset=off, n=n, d_fixed=dd.mean(), t_fixed=t, p_fixed=1 - stats.t.cdf(t, n - 1),
                        t_fixed_nw3=dd.mean() / np.sqrt(nw_var(dd, 3) / n),
                        d_p63=dp.mean(), t_p63=tp, p_p63=1 - stats.t.cdf(tp, n - 1)))
    E = pd.DataFrame(out); E.to_csv(L.RES_DIR / "v2audit_robust_monthly.csv", index=False)
    print("\nE. non-overlapping monthly (21 phase offsets)\n", E.round(4).to_string(index=False))
    print(f"   offsets with one-sided p<0.05 vs fixed: {(E.p_fixed < 0.05).sum()}/21; median t {E.t_fixed.median():.2f}"
          f"; min t {E.t_fixed.min():.2f}; median t with NW(3) {E.t_fixed_nw3.median():.2f}")
    print(f"   ESS of daily loss differential (Bartlett, 126 lags): vs fixed {ess(d):.1f} of {len(d)};"
          f" vs p63 {ess(ep**2-ea**2):.1f}")

    # F. DM-HAC
    out = []
    for nm, eo in (("fixed", eb), ("persist_63", ep)):
        for lag in (20, 42, 63, 126):
            out.append(dict(vs=nm, lag=lag, **dm(ea, eo, lag)))
    F = pd.DataFrame(out); F.to_csv(L.RES_DIR / "v2audit_robust_dm.csv", index=False)
    print("\nF. Diebold-Mariano (HAC Newey-West)\n", F.round(4).to_string(index=False))

    # G. leave-one-year-out
    out = []
    yrs = np.asarray(dates.year)
    for yr in sorted(set(yrs)):
        m = yrs != yr
        x = boot(tg, rows[m], p5[m], pf[m]); z = dm(ea[m], eb[m], 42)
        mo = yrs == yr
        out.append(dict(dropped=yr, n=int(m.sum()), d_fixed=x["d_mse"], lo=x["d_mse_lo"], hi=x["d_mse_hi"],
                        p_boot=x["p_mse"], p_dm42=z["p_hln"], year_only_d=d[mo].mean(),
                        year_share_of_total=d[mo].sum() / d.sum()))
    for nm, mm in (("drop 2022-2025 (=2019-21)", yrs < 2022), ("drop 2023-2025", yrs < 2023),
                   ("drop 2023+2024", ~np.isin(yrs, [2023, 2024])), ("drop 2022+2023", ~np.isin(yrs, [2022, 2023]))):
        x = boot(tg, rows[mm], p5[mm], pf[mm])
        out.append(dict(dropped=nm, n=int(mm.sum()), d_fixed=x["d_mse"], lo=x["d_mse_lo"], hi=x["d_mse_hi"],
                        p_boot=x["p_mse"], p_dm42=dm(ea[mm], eb[mm], 42)["p_hln"]))
    G = pd.DataFrame(out); G.to_csv(L.RES_DIR / "v2audit_robust_loyo.csv", index=False)
    print("\nG. leave-one-year-out\n", G.round(4).to_string(index=False))

    # H. Holm across all 36 confirmatory MSE tests and across the 18 unique ones
    Bt = pd.read_csv(L.RES_DIR / "v2_test_bootstrap.csv")
    c = Bt[Bt.confirmatory].copy()
    c["holm36"] = L.holm(c.p_mse.to_numpy())
    u = c[c.comparison == "vs_fixed"].copy()
    u["holm18_unique"] = L.holm(u.p_mse.to_numpy())
    print("\nH. global Holm\n", c[c.id == "spy_tlt_corr.5"][["id", "comparison", "p_mse", "holm_mse", "holm36"]].to_string(index=False))
    print(u[u.id.str.startswith("spy_tlt")][["id", "p_mse", "holm18_unique"]].to_string(index=False))
    r = L.paired_boot(tg, rows, p5, pf, n_boot=20000, seed=7, L=63)
    print(f"   20000-draw bootstrap .5 vs fixed: p={r['p_mse']:.5f} CI [{r['d_mse_lo']:.4f}, {r['d_mse_hi']:.4f}]")
    pd.to_pickle({"p5": p5, "pf": pf, "p63": p63, "grid": {k: p for k, p, _ in res}},
                 L.RES_DIR / "v2audit_robust_preds.pkl")


if __name__ == "__main__":
    main()

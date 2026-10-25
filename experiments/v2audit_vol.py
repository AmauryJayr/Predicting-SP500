"""V2 PESSIMIST AUDIT: were the volatility drift methods sabotaged? (post-test; exploratory)
 (1) synthetic level shift: SPY / TLT next-week log-RV target + delta for every row dated >= 2021-01-01, features
     untouched (so the features cannot carry the shift). Every drift method must recover; the fixed model cannot.
 (2) the real data: fixed HAR-X bias by year on TEST (was there a level shift to correct at all?)
 (3) QLIKE with a log-normal variance correction exp(2*yhat + 2*s2), s2 = causal trailing variance of known
     residuals (252 rows; first rows use the training residual variance). Does any ranking change?
 (4) detector fire counts for the linear vol models on TEST.
Outputs results/v2audit_vol_*.csv
"""
import os, sys, copy, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import v2_lib as L

pd.set_option("display.width", 250)
B = {"model": "lin", "feats": "harx"}
METHODS = {"fixed": {"method": "fixed"}, "bias126": {"method": "bias", "N": 126}, "recal252": {"method": "recal", "N": 252},
           "rls0.998": {"method": "rls", "lam": 0.998}, "rls0.999": {"method": "rls", "lam": 0.999},
           "expand_R21": {"method": "refit", "R": 21, "W": None}, "roll504": {"method": "refit", "R": 21, "W": 504},
           "decay252": {"method": "refit", "R": 21, "W": None, "hl": 252},
           "hedge": {"method": "ens", "eta": 2.0, "hl": 63},
           "adwin0.002": {"method": "detect", "detector": "adwin", "delta": 0.002},
           "ph3": {"method": "detect", "detector": "ph", "lam": 3.0}}
SHIFT = "2021-01-01"


def synth_job(tname, delta, mname):
    tg = copy.deepcopy(L.build_targets()[tname])
    m = tg.dates >= np.datetime64(SHIFT)
    tg.y = tg.y.copy(); tg.y[m] += delta
    p = L.run(tg, B | METHODS[mname], "TEST", cache={})
    fires = getattr(L.m_detect, "last_fires", None) if METHODS[mname]["method"] == "detect" else None
    s, e = tg.period("TEST"); y = tg.y[s:e]; d = tg.index[s:e]
    post = np.asarray(d >= SHIFT)
    err = p - y
    r = dict(target=tname, delta=delta, method=mname, fires=fires, mse_pre=(err[~post] ** 2).mean(),
             mse_post=(err[post] ** 2).mean(), bias_post=err[post].mean())
    yrs = np.asarray(d.year)
    for yr in range(2021, 2026):
        r[f"bias_{yr}"] = err[yrs == yr].mean()
    # first-6-months bias after the shift
    r["bias_first126"] = err[post][:126].mean()
    return r


def main():
    jobs = [(t, dl, m) for t in ("spy_vol", "tlt_vol") for dl in (0.0, 0.3) for m in METHODS]
    R = pd.DataFrame(Parallel(n_jobs=15)(delayed(synth_job)(*j) for j in jobs))
    fx = R[R.method == "fixed"].set_index(["target", "delta"])["mse_post"]
    R["rel_post_vs_fixed"] = 1 - R.mse_post / R.set_index(["target", "delta"]).index.map(fx).to_numpy()
    R.to_csv(L.RES_DIR / "v2audit_vol_synth.csv", index=False)
    print("(1) synthetic +delta level shift of the target from", SHIFT, "(features untouched)\n",
          R.round(4).to_string(index=False))

    # (2) real-data bias by year
    Y = pd.read_csv(L.RES_DIR / "v2_test_by_year.csv")
    Yb = Y[Y.model.isin(["fixed:lin/harx/fixed", "spy_tlt_corr.5", "fixed:lin/hc/fixed"])].pivot_table(
        index=["target", "model"], columns="year", values="bias")
    Yb.to_csv(L.RES_DIR / "v2audit_vol_realbias.csv")
    print("\n(2) real TEST bias (pred - y) by year\n", Yb.round(3).to_string())

    # (3) QLIKE with log-normal correction
    T = L.build_targets()
    PK = pd.read_pickle(L.RES_DIR / "v2_test_predictions.pkl")["preds"]
    out = []
    for tname in ("spy_vol", "tlt_vol"):
        tg = T[tname]; s, e = tg.period("TEST"); rows = np.arange(s, e); y = tg.y[rows]
        m0 = L.fit(tg, B | {"method": "fixed"}, s); nk0 = tg.n_known(s)
        s2_0 = np.var(tg.y[:nk0] - m0.predict(tg.X("harx")[:nk0]))
        names = [k[1] for k in PK if k[0] == tname]
        corr = {}
        for n in names:
            p = PK[(tname, n)]
            s2 = np.empty(len(rows))
            for i, t in enumerate(rows):
                nk = tg.n_known(t); k = nk - s  # known test residuals: rows s..nk-1
                s2[i] = np.var(y[max(0, k - 252):k] - p[max(0, k - 252):k]) if k >= 63 else s2_0
            corr[n] = p + s2  # log-vol shift so that exp(2*(p+s2)) = exp(2p + 2 s2)
        fxn = "fixed:lin/harx/fixed"
        for n in names:
            q_raw = L.qlike(y, PK[(tname, n)]).mean(); q_c = L.qlike(y, corr[n]).mean()
            bt = L.paired_boot(tg, rows, corr[n], corr[fxn], n_boot=1000)
            out.append(dict(target=tname, model=n, qlike_raw=q_raw, qlike_lognormal=q_c, d_qlike_c_vs_fixed=bt["d_qlike"],
                            lo=bt["d_qlike_lo"], hi=bt["d_qlike_hi"], p=bt["p_qlike"]))
    Q = pd.DataFrame(out)
    Q["rank_raw"] = Q.groupby("target").qlike_raw.rank(); Q["rank_lognormal"] = Q.groupby("target").qlike_lognormal.rank()
    Q.to_csv(L.RES_DIR / "v2audit_vol_qlike.csv", index=False)
    print("\n(3) QLIKE raw vs log-normal-corrected (d>0 = model better than fixed HAR-X, corrected)\n",
          Q.sort_values(["target", "qlike_lognormal"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()

"""V2 PESSIMIST AUDIT: leakage checks the optimist skipped.
 (1) target/label construction re-derived from raw prices for 200 random rows (all 3 targets);
 (2) truncation/corruption test extended to the TREE refit / rolling / decay / recal / ensemble / PH-detector
     methods on VAL (cutoff 2017-09-29), all targets;
 (3) a second truncation test inside the TEST period (cutoff 2022-06-30) for every spy_tlt_corr shortlist entry,
     the linear ensemble/RLS/bias/recal, and the counterparts.
Corruption: every label with label_end > cutoff and every feature row with date > cutoff -> N(0,5) noise.
Predictions for rows with date <= cutoff must be bit-identical.
Output: results/v2audit_leakage_truncation.csv, results/v2audit_leakage_labels.csv
"""
import os, sys, copy
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import v2_lib as L


def corrupt(tg, cut):
    bad = copy.deepcopy(tg)
    rng = np.random.default_rng(1)
    mlab = bad.label_end > cut
    bad.y = bad.y.copy(); bad.y[mlab] = rng.normal(0, 5, mlab.sum())
    mrow = bad.dates > cut
    F = bad.F.copy(); F.loc[mrow] = rng.normal(0, 5, (mrow.sum(), F.shape[1])); bad.F = F; bad._X = {}
    return bad


def job(tname, spec, period, cut):
    tg = L.build_targets()[tname]
    cut = np.datetime64(cut)
    bad = corrupt(tg, cut)
    a, b = L.run(tg, spec, period, cache={}), L.run(bad, spec, period, cache={})
    s, e = tg.period(period)
    keep = tg.dates[s:e] <= cut
    return dict(target=tname, period=period, cutoff=str(cut), label=L.label(spec), n_before=int(keep.sum()),
                max_diff_before=float(np.abs(a[keep] - b[keep]).max()),
                mean_diff_after=float(np.abs(a[~keep] - b[~keep]).mean()))


def labels_check():
    px = pd.read_pickle(L.ROOT / "data" / "aligned_prices.pkl")
    T = L.build_targets(px)
    cal = px["SPY"].index
    r = {k: np.log(px[k]["Close"]).diff().to_numpy() for k in ("SPY", "TLT")}
    rng = np.random.default_rng(0)
    out = []
    for tname, tg in T.items():
        for i in rng.choice(tg.n, 200, replace=False):
            k = cal.get_loc(tg.index[i])
            h = tg.h
            if tname == "spy_tlt_corr":
                yy = np.corrcoef(r["SPY"][k + 1:k + h + 1], r["TLT"][k + 1:k + h + 1])[0, 1]
                feat = np.corrcoef(r["SPY"][k - 20:k + 1], r["TLT"][k - 20:k + 1])[0, 1]; fname = "c21"
            else:
                rr = r["SPY" if tname == "spy_vol" else "TLT"]
                yy = 0.5 * np.log(252 * np.mean(rr[k + 1:k + h + 1] ** 2))
                feat = 0.5 * np.log(252 * np.mean(rr[k - 21:k + 1] ** 2)); fname = "lrv22"
            out.append(dict(target=tname, row=int(i), y_err=abs(yy - tg.y[i]),
                            label_end_ok=bool(pd.Timestamp(tg.label_end[i]) == cal[k + h]),
                            feat_err=abs(feat - tg.F[fname].iloc[i])))
    return pd.DataFrame(out)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    LB = labels_check(); LB.to_csv(L.RES_DIR / "v2audit_leakage_labels.csv", index=False)
    print(LB.groupby("target").agg(max_y_err=("y_err", "max"), all_le_ok=("label_end_ok", "all"),
                                   max_feat_err=("feat_err", "max")))
    jobs = []
    for tname in ["spy_vol", "tlt_vol", "spy_tlt_corr"]:
        mc = [m for m in L.MODEL_CLASSES[tname] if m[0] == "hgb"][0]
        b = {"model": mc[0], "feats": mc[1]}
        specs = [b | {"method": "refit", "R": R, "W": None} for R in (21, 63, 252)]
        specs += [b | {"method": "refit", "R": 21, "W": W} for W in (252, 1008)]
        specs += [b | {"method": "refit", "R": 21, "W": None, "hl": 126}]
        specs += [b | {"method": "recal", "N": N} for N in (252, 504)]
        specs += [b | {"method": "ens", "hl": 63, "eta": 2.0}]
        specs += [b | {"method": "detect", "detector": "ph", "lam": 1.0}]
        jobs += [(tname, sp, "VAL", "2017-09-29") for sp in specs]
    # inside the test period, corr target
    SL = __import__("json").loads((L.RES_DIR / "v2_shortlist.json").read_text())
    specs = [e["spec"] for e in SL["spy_tlt_corr"]["entries"]]
    specs += [{"model": "lin", "feats": "hc", "method": m} | p for m, p in
              [("fixed", {}), ("bias", {"N": 252}), ("recal", {"N": 504}), ("rls", {"lam": 0.995}),
               ("ens", {"hl": 63, "eta": 2.0}), ("detect", {"detector": "adwin", "delta": 0.002}),
               ("refit", {"R": 5, "W": None}), ("refit", {"R": 63, "W": None})]]
    specs += [{"model": "hgb", "feats": "full", "method": "fixed"}, {"method": "raw", "col": "c63"}]
    jobs += [("spy_tlt_corr", sp, "TEST", "2022-06-30") for sp in specs]
    jobs.sort(key=lambda j: 0 if j[1].get("method") == "ens" else 1)
    R = pd.DataFrame(Parallel(n_jobs=15, verbose=2)(delayed(job)(*j) for j in jobs))
    R.to_csv(L.RES_DIR / "v2audit_leakage_truncation.csv", index=False)
    print(R.to_string(index=False))
    print("runs:", len(R), "all identical before cutoff:", bool((R.max_diff_before == 0).all()),
          "all differ after:", bool((R.mean_diff_after > 0).all()))

"""v2 pre-test checks (run on validation folds only, no test data).
  1. truncation test: corrupt every label with label_end > cutoff and every feature row after cutoff; predictions
     for rows <= cutoff must be bit-identical (no look-ahead in any method, including online/delayed-feedback ones).
  2. bootstrap sanity: paired_boot(A, A) gives 0; sign convention (better model -> positive delta).
Run: python v2_checks.py -> results/v2_checks_truncation.csv
"""
import os, sys, copy
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import v2_lib as L

CUT = np.datetime64("2017-09-29")


def truncation(fold="VAL"):
    T = L.build_targets()
    out = []
    for tname, tg in T.items():
        specs = [s for s in L.grid(tname) if s["model"] == "lin" or s["method"] in ("fixed", "bias", "detect")]
        specs = [s for s in specs if s["method"] != "ens" or s["model"] == "lin"] + list(L.baselines(tname).values())
        bad = copy.deepcopy(tg)
        rng = np.random.default_rng(0)
        mlab = bad.label_end > CUT
        bad.y = bad.y.copy(); bad.y[mlab] = rng.normal(0, 5, mlab.sum())
        mrow = bad.dates > CUT
        F = bad.F.copy(); F.loc[mrow] = rng.normal(0, 5, (mrow.sum(), F.shape[1])); bad.F = F; bad._X = {}
        s, e = tg.period(fold)
        keep = tg.dates[s:e] <= CUT
        for sp in specs:
            a, b = L.run(tg, sp, fold), L.run(bad, sp, fold)
            out.append(dict(target=tname, label=L.label(sp), max_diff_before=float(np.abs(a[keep] - b[keep]).max()),
                            mean_diff_after=float(np.abs(a[~keep] - b[~keep]).mean())))
    return pd.DataFrame(out)


if __name__ == "__main__":
    D = truncation()
    D.to_csv(L.RES_DIR / "v2_checks_truncation.csv", index=False)
    print(D.groupby("target")[["max_diff_before", "mean_diff_after"]].agg(["max", "min"]))
    print("runs:", len(D), "all identical before cutoff:", bool((D.max_diff_before == 0).all()))
    tg = L.build_targets()["spy_vol"]; s, e = tg.period("VAL"); rows = np.arange(s, e)
    a = L.run(tg, {"model": "lin", "feats": "harx", "method": "fixed"}, "VAL")
    b = L.run(tg, {"method": "raw", "col": "lvix"}, "VAL")
    r0 = L.paired_boot(tg, rows, a, a, n_boot=200); r1 = L.paired_boot(tg, rows, a, b, n_boot=500)
    print("self:", {k: round(v, 4) for k, v in r0.items() if k.startswith("d_")})
    print("harx vs vix_raw:", {k: round(v, 4) for k, v in r1.items()})

"""AUDIT (pessimist): stress test of the one positive EXPLORATORY finding:
trained-once LR (C=0.003) on the 23 'core' features, h=5, test BA 0.536 vs chance.

S1 hyperparameter stability: C grid
S2 feature perturbations: drop-one (23 variants) and 200 random 18-of-23 subsets; all 122 features
S3 threshold choice: training base rate (default) / 0.5 / median of in-sample fitted probabilities
S4 retraining the same model: expanding refits every 21 / 63 / 252 sessions
S5 sub-periods / per-year, block lengths 20-120 vs chance
S6 trivial rules (no fitting): SMA200 trend, SMA50 trend, 20d momentum, 5d / 1d reversal, VIX-high, constant-up
S7 the same model on the validation folds and at h=1
Outputs: results/audit_corelr_*.csv
"""
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import common
from common import RES_DIR, CORE_FEATURES, Dataset, make_estimator, metrics, to_pred, walk_forward
from audit_lib import boot, chance_like, holm

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)


def fixed_lr(ds, X, C=0.003, period="TEST", refit=None, thr_mode="rate"):
    s, e = ds.period_positions(period)
    df = walk_forward(ds, X, make_estimator("LogReg", {"C": C}), s, e, refit)
    if thr_mode != "rate":
        nk = ds.n_known(s)
        m = make_estimator("LogReg", {"C": C}).fit(X[:nk], ds.y[:nk])
        df["thr"] = 0.5 if thr_mode == "half" else np.median(m.predict_proba(X[:nk])[:, 1])
    return to_pred(ds, df, s, e)


def rule(ds, signal, period="TEST"):
    s, e = ds.period_positions(period)
    sig_all = np.asarray(signal, float)
    sd = sig_all[:s].std()  # scale from pre-period data only; monotone map, so AUC/BA are unaffected
    sig = sig_all[s:e]
    return pd.DataFrame({"y": ds.y[s:e], "proba": 1 / (1 + np.exp(-sig / sd)), "thr": 0.5,
                         "pred": (sig > 0).astype(int)}, index=ds.index[s:e])


def lvl(p):
    m = metrics(p)
    return dict(BA=m["balanced_acc"], AUC=m["roc_auc"], pred_up=m["pred_up_rate"])


def main():
    ds = Dataset(5)
    Xc = ds.X("core")
    base = fixed_lr(ds, Xc)
    s, e = ds.period_positions("TEST")
    rate = ds.y[:ds.n_known(s)].mean()
    ch = chance_like(base, rate)
    print("base:", lvl(base), "test up-rate", ds.y[s:e].mean().round(3), "train rate", round(rate, 3))
    rows = []

    def add(group, name, p, L=20):
        b = boot(p, ch, L=L, n_boot=1000)
        rows.append(dict(group=group, variant=name, **lvl(p), dBA_vs_chance=b["dBA"], dBA_lo=b["dBA_lo"],
                         dBA_hi=b["dBA_hi"], pBA=b["pBA"], dAUC_lo=b["dAUC_lo"], dAUC_hi=b["dAUC_hi"], pAUC=b["pAUC"]))

    add("base", "LR core C=0.003", base)
    for C in [1e-4, 3e-4, 1e-3, 0.01, 0.03, 0.1, 1.0]:
        add("C", f"C={C}", fixed_lr(ds, Xc, C))
    for j, f in enumerate(CORE_FEATURES):
        add("drop1", f"-{f}", fixed_lr(ds, np.delete(Xc, j, axis=1)))
    rng = np.random.default_rng(0)
    for k in range(200):
        cols = np.sort(rng.choice(len(CORE_FEATURES), 18, replace=False))
        add("sub18", f"subset{k}", fixed_lr(ds, Xc[:, cols]))
    add("feats", "all 122 features", fixed_lr(ds, ds.X("all")))
    add("thr", "thr=0.5", fixed_lr(ds, Xc, thr_mode="half"))
    add("thr", "thr=train median", fixed_lr(ds, Xc, thr_mode="median"))
    for R in (21, 63, 252):
        add("refit", f"expanding refit {R}", fixed_lr(ds, Xc, refit=R))
    D = ds.data
    rules = {"SMA200 trend": D["spy_px_vs_sma200"], "SMA50 trend": D["spy_px_vs_sma50"],
             "20d momentum": D["spy_ret_20d"], "5d reversal": -D["spy_ret_5d"], "1d reversal": -D["spy_ret_1d"],
             "VIX above 60d mean": D["vix_z60"], "50/200 golden cross": D["spy_sma50_vs_sma200"]}
    R_ = {}
    for n, sig in rules.items():
        R_[n] = rule(ds, sig.to_numpy())
        add("rule", n, R_[n])
    T = pd.DataFrame(rows)
    T.to_csv(RES_DIR / "audit_corelr_variants.csv", index=False)
    print(T[T.group != "sub18"].round(3).to_string(index=False))
    for g in ("drop1", "sub18", "C"):
        x = T[T.group == g]
        print(f"{g}: n={len(x)} BA median {x.BA.median():.3f} [min {x.BA.min():.3f}, max {x.BA.max():.3f}], "
              f"share BA>0.52 {np.mean(x.BA > 0.52):.2f}, share pBA<0.05 {np.mean(x.pBA < 0.05):.2f}; "
              f"AUC median {x.AUC.median():.3f}")

    # block lengths / sub-periods / years vs chance
    rows = []
    for L in (20, 40, 60, 120):
        rows.append(dict(what=f"2019-2025 L={L}", **boot(base, ch, L=L)))
    for a, b in [("2019", "2021"), ("2022", "2025")]:
        rows.append(dict(what=f"{a}-{b} L=20", **boot(base.loc[a:b], ch.loc[a:b], L=20)))
    for y in range(2019, 2026):
        g = base.loc[str(y)]
        rows.append(dict(what=f"year {y}", dBA=lvl(g)["BA"] - 0.5, dAUC=lvl(g)["AUC"] - 0.5,
                         SMA200_BA=lvl(R_["SMA200 trend"].loc[str(y)])["BA"]))
    for n in ("SMA200 trend", "5d reversal", "VIX above 60d mean"):
        rows.append(dict(what=f"LR core vs rule: {n}", **boot(base, R_[n], L=20)))
    B = pd.DataFrame(rows)
    B.to_csv(RES_DIR / "audit_corelr_blocks_periods.csv", index=False)
    print(B.round(3).to_string(index=False))

    # validation folds and h=1
    rows = []
    for per in ("F1", "F2", "VAL"):
        p = fixed_lr(ds, Xc, period=per)
        rows.append(dict(h=5, period=per, **lvl(p)))
        for n, sig in rules.items():
            rows.append(dict(h=5, period=per, rule=n, **lvl(rule(ds, sig.to_numpy(), per))))
    d1 = Dataset(1)
    for per in ("F1", "F2", "VAL", "TEST"):
        rows.append(dict(h=1, period=per, **lvl(fixed_lr(d1, d1.X("core"), period=per))))
    V = pd.DataFrame(rows)
    V.to_csv(RES_DIR / "audit_corelr_folds.csv", index=False)
    print(V.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

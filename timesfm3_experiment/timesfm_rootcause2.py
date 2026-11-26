"""EXPLORATORY, post-test: what does the team's simple LR see that TimesFM-3 does not?  (TIMESFM_ROOT_CAUSE.md, section 8)

Plan (hashed before any test number): timesfm_rootcause2_plan.md, results/timesfm_rootcause2_lock.json.
Cached forecasts and the team's code only (no TimesFM / Chronos call).
Usage (from timesfm3_experiment/):  PYTHONDONTWRITEBYTECODE=1 python timesfm_rootcause2.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import timesfm_drift_lib as dl
from tf_paths import RES_DIR
from common import CORE_FEATURES, SEED, block_indices, block_len, fit_model, make_estimator, metrics, paired_bootstrap

OUT = lambda n: RES_DIR / f"timesfm_rootcause2_{n}.csv"
C0 = dl.LR_SPEC["params"]["C"]
NB = 2000

GROUPS = {
    "SPY short-term": ["spy_ret_1d", "spy_ret_5d", "spy_ret_20d", "spy_rsi14"],
    "SPY long-term/trend": ["spy_ret_60d", "spy_ret_250d", "spy_px_vs_sma50", "spy_px_vs_sma200", "spy_dist_252d_high"],
    "volatility": ["spy_vol_20d", "spy_vol_ratio_5_60"],
    "VIX": ["vix_level", "vix_z60", "vix_minus_rv20", "vix_log_chg_5d"],
    "cross-asset": ["sector_breadth_above_sma50", "sector_dispersion_5d", "corr_spy_tlt_60d", "tlt_ret_20d",
                    "iwm_minus_spy_ret_20d", "xly_minus_xlp_ret_20d"],
    "Open-based": ["spy_overnight_ret", "spy_intraday_ret"],
}
assert sorted(sum(GROUPS.values(), [])) == sorted(CORE_FEATURES)
CLOSE_ONLY = GROUPS["SPY short-term"] + GROUPS["SPY long-term/trend"] + GROUPS["volatility"]
SUBSETS = {"(a) all 23 core": CORE_FEATURES,
           "(b) SPY Close-only (11)": CLOSE_ONLY,
           "(c) all SPY incl. Open (13)": CLOSE_ONLY + GROUPS["Open-based"],
           "(d) VIX group (4)": GROUPS["VIX"],
           "(e) cross-asset group (6)": GROUPS["cross-asset"]}
# what each feature needs (read from experiments/common.py spy_features / cross_features)
NEEDS = {"spy_ret_1d": "Close, 2 days", "spy_ret_5d": "Close, 6 days", "spy_ret_20d": "Close, 21 days",
         "spy_ret_60d": "Close, 61 days", "spy_ret_250d": "Close, 251 days", "spy_vol_20d": "Close, 21 days",
         "spy_vol_ratio_5_60": "Close, 61 days", "spy_px_vs_sma50": "Close, 50 days", "spy_px_vs_sma200": "Close, 200 days",
         "spy_rsi14": "Close; exponentially weighted from the start of the full history (approx. 256-day window only)",
         "spy_dist_252d_high": "Close, 252 days", "spy_overnight_ret": "Open (not in TimesFM input)",
         "spy_intraday_ret": "Open (not in TimesFM input)"}


def save(df, n):
    df.to_csv(OUT(n), index=False, float_format="%.5f")


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def p2(p_one):
    return float(min(1.0, max(2 * min(p_one, 1 - p_one), 1 / (NB + 1))))


def auc_boot(f, h):
    """AUC with moving-block bootstrap CI and two-sided p vs chance (0.5)."""
    y, p = f["y"].to_numpy(), f["proba"].to_numpy()
    rng = np.random.default_rng(SEED); L = block_len(h)
    d = np.empty(NB)
    for i in range(NB):
        k = block_indices(len(y), L, rng); d[i] = roc_auc_score(y[k], p[k])
    a = roc_auc_score(y, p)
    return {"auc": a, "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5),
            "p_two_sided": float(max(2 * min((d <= 0.5).mean(), (d >= 0.5).mean()), 1 / (NB + 1)))}


def pb(a, b, h):
    r = paired_bootstrap(a, b, h, n_boot=NB)
    return {"d_auc": r["d_roc_auc"], "lo": r["d_roc_auc_lo"], "hi": r["d_roc_auc_hi"], "p_two_sided": p2(r["p_roc_auc"])}


def auc(y, x):
    return float(roc_auc_score(y, x)) if len(np.unique(y)) > 1 else np.nan


def fit_frame(inp, X, tr, s, e, C=C0):
    m = fit_model(make_estimator("LogReg", {"C": C}), X[tr], inp.ds.y[tr])
    return inp.frame(s, e, m.predict_proba(X[s:e])[:, 1], np.full(e - s, inp.ds.y[tr].mean())), m


def sub_X(ds, cols):
    Xc = ds.X("core")
    return Xc[:, [CORE_FEATURES.index(c) for c in cols]]


# ----------------------------------------------------------------------------------------------
INP = {1: dl.Inp(1), 5: dl.Inp(5)}
feat_rows = [{"feature": c, "group": g, "close_only_qualifies": c in CLOSE_ONLY, "needs": NEEDS.get(c, "other assets")}
             for g, cs in GROUPS.items() for c in cs]
save(pd.DataFrame(feat_rows), "features")

# step 0: reproduce ---------------------------------------------------------------------------
repro, FULL, ST = [], {}, {}
for h, inp in INP.items():
    ds = inp.ds; s, e = ds.period_positions("TEST"); tr = np.arange(ds.n_known(s))
    ref = inp.fixed_lr("TEST")
    FULL[h], _ = fit_frame(inp, ds.X("core"), tr, s, e)
    assert np.allclose(FULL[h]["proba"], ref["proba"]) and FULL[h].index.equals(ref.index)
    ST[h] = inp.static("TEST")
    repro.append({"h": h, "n_test": e - s, "n_train": len(tr), "train_first": str(ds.index[tr[0]].date()),
                  "train_last_label_end": str(pd.Timestamp(ds.label_end[tr].max()).date()),
                  "fixed_lr_auc_run_spec": metrics(ref)["roc_auc"], "subset_path_auc_all23": metrics(FULL[h])["roc_auc"],
                  "timesfm_static_auc": metrics(ST[h])["roc_auc"]})
    assert round(repro[-1]["fixed_lr_auc_run_spec"], 3) == {1: 0.508, 5: 0.536}[h]
save(pd.DataFrame(repro), "repro")
print(pd.DataFrame(repro).to_string())

# task 1: ceiling --------------------------------------------------------------------------------
rows, SF = [], {}
for h, inp in INP.items():
    ds = inp.ds; s, e = ds.period_positions("TEST"); tr = np.arange(ds.n_known(s))
    for nm, cols in SUBSETS.items():
        f, _ = fit_frame(inp, sub_X(ds, cols), tr, s, e); SF[(h, nm)] = f
        rows.append({"h": h, "subset": nm, "n_features": len(cols), "test": "AUC vs chance 0.5", **auc_boot(f, h)})
    b = pb(FULL[h], ST[h], h)
    rows.append({"h": h, "subset": "(a) all 23 core vs TimesFM static", "n_features": 23, "test": "paired diff (LR - TimesFM)",
                 "auc": metrics(FULL[h])["roc_auc"], "d_auc": b["d_auc"], "lo": b["lo"], "hi": b["hi"], "p_two_sided": b["p_two_sided"]})
t1 = pd.DataFrame(rows); t1["p_holm_12"] = holm(t1["p_two_sided"])
save(t1, "subsets"); print(t1.to_string())

# task 2: ablation, coefficients, per year ---------------------------------------------------------
ab, co, yr = [], [], []
for h, inp in INP.items():
    ds = inp.ds; s, e = ds.period_positions("TEST"); tr = np.arange(ds.n_known(s))
    for g, gc in GROUPS.items():
        keep = [c for c in CORE_FEATURES if c not in gc]
        f, _ = fit_frame(inp, sub_X(ds, keep), tr, s, e)
        b = pb(FULL[h], f, h)                       # full minus dropped: positive = the group helps
        ab.append({"h": h, "dropped_group": g, "n_left": len(keep), "auc_without": metrics(f)["roc_auc"],
                   "auc_full": metrics(FULL[h])["roc_auc"], "full_minus_without": b["d_auc"], "lo": b["lo"], "hi": b["hi"],
                   "p_two_sided_uncorrected": b["p_two_sided"]})
    m = fit_model(make_estimator("LogReg", {"C": C0}), ds.X("core")[tr], ds.y[tr])
    for c, v in zip(CORE_FEATURES, m[-1].coef_[0]):
        co.append({"h": h, "feature": c, "group": [g for g, cs in GROUPS.items() if c in cs][0], "coef_standardised": v})
    years = FULL[h].index.year
    for Y in range(2019, 2026):
        k = years == Y
        yr.append({"h": h, "year": Y, "n": int(k.sum()), "auc_lr": auc(FULL[h]["y"][k], FULL[h]["proba"][k]),
                   "auc_timesfm_static": auc(ST[h]["y"][k], ST[h]["proba"][k])})
    k = np.ones(len(years), bool)
    yr.append({"h": h, "year": "2019-2025", "n": len(years), "auc_lr": auc(FULL[h]["y"], FULL[h]["proba"]),
               "auc_timesfm_static": auc(ST[h]["y"], ST[h]["proba"])})
ab = pd.DataFrame(ab); co = pd.DataFrame(co); yr = pd.DataFrame(yr)
save(ab, "ablation"); save(co, "coefs"); save(yr, "by_year")
print(ab.to_string()); print(yr.to_string())
for h in (1, 5):
    print(h, co[co.h == h].sort_values("coef_standardised", key=abs, ascending=False).head(8).to_string())

# task 3: does TimesFM add? ---------------------------------------------------------------------------
add, addco = [], []
for h, inp in INP.items():
    ds = inp.ds; s, e = ds.period_positions("TEST"); tr = inp.train_rows(s, None)
    Xc = ds.X("core"); Xt = np.column_stack([inp.z, inp.med, inp.spr]); Xct = np.column_stack([Xc, Xt])
    assert np.isfinite(Xt[tr]).all() and np.isfinite(Xt[s:e]).all()
    for C in (C0, 0.03):
        fb, _ = fit_frame(inp, Xc, tr, s, e, C); fa, ma = fit_frame(inp, Xct, tr, s, e, C)
        ft, _ = fit_frame(inp, Xt, tr, s, e, C)
        b = pb(fa, fb, h)
        add.append({"h": h, "C": C, "train_rows": f"{ds.index[tr[0]].date()}..{ds.index[tr[-1]].date()} n={len(tr)}",
                    "auc_core_only": metrics(fb)["roc_auc"], "auc_core_plus_timesfm": metrics(fa)["roc_auc"],
                    "auc_timesfm3_features_only": metrics(ft)["roc_auc"], "diff_plus_minus_core": b["d_auc"],
                    "lo": b["lo"], "hi": b["hi"], "p_two_sided": b["p_two_sided"],
                    "auc_core_trained_all_pre2019": metrics(FULL[h])["roc_auc"]})
        if C == C0:
            for nm, v in zip(CORE_FEATURES + ["tf_logit_score", "tf_median_ret", "tf_q90_q10"], ma[-1].coef_[0]):
                addco.append({"h": h, "feature": nm, "coef_standardised": v})
add = pd.DataFrame(add); mask = add["C"] == C0
add.loc[mask, "p_holm_2"] = holm(add.loc[mask, "p_two_sided"]); save(add, "add_timesfm"); save(pd.DataFrame(addco), "add_timesfm_coefs")
print(add.T.to_string())

# task 4: same bet? ------------------------------------------------------------------------------------
cr = []
for h, inp in INP.items():
    ds = inp.ds; s, e = ds.period_positions("TEST"); tr = np.arange(ds.n_known(s)); Xc = ds.X("core")
    m = fit_model(make_estimator("LogReg", {"C": C0}), Xc[tr], ds.y[tr]); coef = m[-1].coef_[0]
    lrs = m.decision_function(Xc)
    for per, sl in (("2019-2025", slice(s, e)), ("2013-2018 (LR in sample)", slice(inp.p0, s))):
        z = inp.z[sl]
        cr.append({"h": h, "period": per, "feature": "LR score", "rho_timesfm": spearmanr(z, lrs[sl])[0], "rho_lr": np.nan,
                   "coef_lr_std": np.nan, "opposite": ""})
        for j, c in enumerate(CORE_FEATURES):
            rt, rl = spearmanr(z, Xc[sl, j])[0], spearmanr(lrs[sl], Xc[sl, j])[0]
            cr.append({"h": h, "period": per, "feature": c, "rho_timesfm": rt, "rho_lr": rl, "coef_lr_std": coef[j],
                       "opposite": "OPPOSITE" if (np.sign(rt) != np.sign(rl) and min(abs(rt), abs(rl)) >= 0.10) else ""})
cr = pd.DataFrame(cr); save(cr, "corr"); print(cr[(cr.period == "2019-2025") & ((cr.opposite != "") | (cr.feature == "LR score"))].to_string())
print("done")

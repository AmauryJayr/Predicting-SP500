"""EXPLORATORY, post-test: why did the drift layers raise TimesFM-3's test AUC?  (TIMESFM_DRIFT.md, section 7)

Nothing here changes a choice of timesfm_drift_study.py. Uses only the cached forecasts (no model call).
  h=1  stack-4 (LR C=0.03 on [logit score, median return, q90-q10 spread], fitted once): is the spread a volatility proxy?
  h=5  platt-6 (offset-only recalibration on the last 252 known labels, refit every 21 rows): exact pair decomposition,
       what the offset tracks, and a no-TimesFM base-rate benchmark.

Usage (from timesfm3_experiment/):  PYTHONDONTWRITEBYTECODE=1 python timesfm_drift_why.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import json

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

import timesfm_drift_lib as dl
from tf_paths import RES_DIR
from common import DATA_DIR, fit_model, make_estimator, metrics, paired_bootstrap

CH = json.loads((RES_DIR / "timesfm_drift_choices.json").read_text())["chosen"]
S1, P5 = CH["h1.stack"]["spec"], CH["h5.platt"]["spec"]
OUT = lambda n: RES_DIR / f"timesfm_drift_why_{n}.csv"
YEARS = list(range(2019, 2026))
NCMP = 0                                                   # running count of comparisons (reported in the doc)


def auc(y, x):
    ok = np.isfinite(x)
    return float(roc_auc_score(y[ok], x[ok])) if len(np.unique(y[ok])) > 1 else np.nan


def save(df, name):
    df.to_csv(OUT(name), index=False, float_format="%.5f")
    print(f"\n== {name}\n{df.to_string(index=False)}", flush=True)


# =====================================================================================================
# 0. reproduce
# =====================================================================================================
inp1, inp5 = dl.Inp(1), dl.Inp(5)
pred = pd.read_pickle(RES_DIR / "timesfm_drift_test_predictions.pkl")
repro = []
for h, inp, fam, spec, key in ((1, inp1, "stack", S1, "stack"), (5, inp5, "platt", P5, "platt")):
    s, e = inp.ds.period_positions("TEST")
    st, ch = inp.static("TEST"), dl.run(inp, fam, spec, "TEST")
    for nm, f in (("static", st), (CH[f"h{h}.{fam}"]["cfg"], ch)):
        saved = pred[(h, "static" if nm == "static" else key)]
        assert np.allclose(f["proba"], saved["proba"]) and f.index.equals(saved.index), (h, nm)
        repro.append({"h": h, "method": nm, "n": len(f), "auc": metrics(f)["roc_auc"], "matches_saved_pickle": True})
save(pd.DataFrame(repro), "repro")
ST1, SK = inp1.static("TEST"), dl.run(inp1, "stack", S1, "TEST")
ST5, PL = inp5.static("TEST"), dl.run(inp5, "platt", P5, "TEST")

# =====================================================================================================
# h = 1
# =====================================================================================================
ds, y1 = inp1.ds, inp1.ds.y
s1, e1 = ds.period_positions("TEST")
p0 = inp1.p0
# label definition, confirmed from common.Dataset: target = Close[t+h]/Open[t+1] - 1 > 0
px = pd.read_pickle(DATA_DIR / "aligned_prices.pkl")
spy = px["SPY"]
chk = ((spy["Close"].shift(-1) / spy["Open"].shift(-1) - 1) > 0).reindex(ds.index).astype(int).to_numpy()
assert (chk[:-1] == y1[:-1]).all()
vix = px["^VIX"]["Close"].reindex(ds.index).to_numpy(float)
rv20 = np.log(spy["Close"]).diff().rolling(20).std().reindex(ds.index).to_numpy(float)
spr_series = pd.Series(inp1.spr)
spr_rel = (spr_series / spr_series.rolling(252, min_periods=60).mean()).to_numpy()   # spread vs its own past level
FEAT = {"score_logit": inp1.z, "median_return": inp1.med, "spread_q90q10": inp1.spr,
        "spread_over_trailing252_mean": spr_rel, "VIX": vix, "rv20": rv20}
# NB: the spread is already the log-price quantile gap (= relative to price), so 'spread scaled by price' is the same
# series; the scaled variant reported is the spread divided by its own trailing-252 mean.
tr = inp1.train_rows(s1, None)                              # stack-4 fit window (fixed, fit once)
fit_s, fit_e = ds.index[tr[0]], ds.index[tr[-1]]
tests = slice(s1, e1)
fitwin = slice(p0, s1)

# --- 1. coefficients ---
X = np.column_stack([inp1.z, inp1.med, inp1.spr])
m = fit_model(make_estimator("LogReg", {"C": S1["C"]}), X[tr], y1[tr])
sc = m[0]
coef_std, raw = m[-1].coef_[0], m[-1].coef_[0] / sc.scale_
rows = [{"feature": f, "coef_standardised": c, "coef_per_raw_unit": r, "train_mean": mu, "train_sd": sd}
        for f, c, r, mu, sd in zip(["score_logit", "median_return", "spread_q90q10"], coef_std, raw, sc.mean_, sc.scale_)]
rows.append({"feature": "intercept", "coef_standardised": float(m[-1].intercept_[0]), "coef_per_raw_unit": np.nan,
             "train_mean": np.nan, "train_sd": np.nan})
rows.append({"feature": f"fit window {fit_s.date()} .. {fit_e.date()} n={len(tr)} up_rate={y1[tr].mean():.4f}",
             "coef_standardised": np.nan, "coef_per_raw_unit": np.nan, "train_mean": np.nan, "train_sd": np.nan})
save(pd.DataFrame(rows), "h1_coefs")
# does stack-4 reproduce from this refit?
assert np.allclose(m.predict_proba(X[s1:e1])[:, 1], SK["proba"])

# --- 2. single feature AUCs ---
sf = []
for nm, v in FEAT.items():
    sf.append({"feature": nm, "auc_2013_2018": auc(y1[fitwin], v[fitwin]), "auc_test_2019_2025": auc(y1[tests], v[tests])})
sf.append({"feature": "stack-4 (fixed layer)", "auc_2013_2018": np.nan, "auc_test_2019_2025": metrics(SK)["roc_auc"]})
sf.append({"feature": "static TimesFM (official)", "auc_2013_2018": np.nan, "auc_test_2019_2025": metrics(ST1)["roc_auc"]})
SF = pd.DataFrame(sf)
save(SF, "h1_single_auc")
NCMP += len(FEAT) * 2

yr = ds.index.year.to_numpy()
fy = []
for nm in list(FEAT) + ["stack4", "static"]:
    for Y in YEARS:
        k = np.flatnonzero(yr == Y); k = k[(k >= s1) & (k < e1)]
        if nm == "stack4":
            a = auc(y1[k], SK["proba"].to_numpy()[k - s1])
        elif nm == "static":
            a = auc(y1[k], ST1["proba"].to_numpy()[k - s1])
        else:
            a = auc(y1[k], FEAT[nm][k])
        fy.append({"feature": nm, "year": Y, "auc": a, "n": len(k), "up_rate": y1[k].mean()})
FY = pd.DataFrame(fy)
save(FY.pivot(index="feature", columns="year", values="auc").reset_index(), "h1_single_auc_by_year")
NCMP += 7 * 8

# --- 3. is spread a vol proxy? ---
cor = []
for per, sl in (("2013-2018", fitwin), ("2019-2025", tests)):
    for other in ("VIX", "rv20", "score_logit", "median_return"):
        ok = np.isfinite(FEAT[other][sl]) & np.isfinite(inp1.spr[sl])
        cor.append({"period": per, "pair": f"spread vs {other}", "spearman": spearmanr(inp1.spr[sl][ok], FEAT[other][sl][ok])[0]})
for Y in YEARS:
    k = np.flatnonzero(yr == Y); k = k[(k >= s1) & (k < e1)]
    for other in ("VIX", "rv20"):
        cor.append({"period": str(Y), "pair": f"spread vs {other}", "spearman": spearmanr(inp1.spr[k], FEAT[other][k])[0]})
save(pd.DataFrame(cor), "h1_spread_correlations")


def fixed_lr(cols, C=S1["C"]):
    Xc = np.column_stack(cols)
    mm = fit_model(make_estimator("LogReg", {"C": C}), Xc[tr], y1[tr])
    return inp1.frame(s1, e1, mm.predict_proba(Xc[s1:e1])[:, 1], np.full(e1 - s1, y1[tr].mean())), mm


F = {}
F["LR[spread]"], mS = fixed_lr([inp1.spr])
F["LR[VIX]"], mV = fixed_lr([vix])
F["LR[rv20]"], mR = fixed_lr([rv20])
F["LR[spread+VIX]"], _ = fixed_lr([inp1.spr, vix])
F["LR[spread+rv20]"], _ = fixed_lr([inp1.spr, rv20])
F["LR[score,median,spread]=stack-4"] = SK
lr = []
for nm, f in F.items():
    lr.append({"model": nm, "test_auc": metrics(f)["roc_auc"]})
for nm, mm in (("LR[spread]", mS), ("LR[VIX]", mV), ("LR[rv20]", mR)):
    lr[[r["model"] for r in lr].index(nm)]["std_coef"] = float(mm[-1].coef_[0, 0])
# residualised spread: OLS fitted on the stack-4 fit window, applied to test (levels and logs)
for nm, v in (("VIX", vix), ("rv20", rv20)):
    for lg in (False, True):
        a, b = (np.log(inp1.spr), np.log(v)) if lg else (inp1.spr, v)
        ols = LinearRegression().fit(b[tr][:, None], a[tr])
        res = a - ols.predict(b[:, None])
        lr.append({"model": f"spread residualised on {'log ' if lg else ''}{nm}", "test_auc": auc(y1[tests], res[tests]),
                   "auc_2013_2018": auc(y1[fitwin], res[fitwin]), "r2_on_fit_window": float(ols.score(b[tr][:, None], a[tr]))})
save(pd.DataFrame(lr), "h1_lr_compare")
NCMP += len(lr)

bt = []
for nm in ("LR[rv20]", "LR[VIX]", "LR[spread]", "LR[spread+VIX]"):
    b = paired_bootstrap(SK, F[nm], 1)                       # same call as timesfm_drift_study._boot
    bt.append({"A": "stack-4", "B": nm, "d_auc": b["d_roc_auc"], "lo": b["d_roc_auc_lo"], "hi": b["d_roc_auc_hi"], "p_one_sided": b["p_roc_auc"]})
b = paired_bootstrap(F["LR[spread]"], F["LR[rv20]"], 1)
bt.append({"A": "LR[spread]", "B": "LR[rv20]", "d_auc": b["d_roc_auc"], "lo": b["d_roc_auc_lo"], "hi": b["d_roc_auc_hi"], "p_one_sided": b["p_roc_auc"]})
b = paired_bootstrap(F["LR[spread]"], ST1, 1)
bt.append({"A": "LR[spread]", "B": "static", "d_auc": b["d_roc_auc"], "lo": b["d_roc_auc_lo"], "hi": b["d_roc_auc_hi"], "p_one_sided": b["p_roc_auc"]})
save(pd.DataFrame(bt), "h1_bootstrap")
NCMP += len(bt)

# --- 4. up-rate by quintile ---
qr, qy = [], []
for nm in ("spread_q90q10", "VIX", "rv20"):
    v = FEAT[nm]
    cut_fit = np.nanquantile(v[tr], [.2, .4, .6, .8])         # cut points from the stack-4 fit window
    for per, sl in (("2013-2018", fitwin), ("2019-2025", tests)):
        x, yy = v[sl], y1[sl]
        own = np.digitize(x, np.nanquantile(x, [.2, .4, .6, .8])) + 1
        fix = np.digitize(x, cut_fit) + 1
        for q in range(1, 6):
            qr.append({"feature": nm, "period": per, "quintile": q, "n_own": int((own == q).sum()), "up_rate_own_quintile": yy[own == q].mean(),
                       "n_fixed": int((fix == q).sum()), "up_rate_fixed_cuts": yy[fix == q].mean() if (fix == q).any() else np.nan})
    # per year: up-rate above vs below the fit-window median, and top vs bottom fit-window quintile
    med_fit = np.nanmedian(v[tr])
    for Y in YEARS:
        k = np.flatnonzero(yr == Y); k = k[(k >= s1) & (k < e1)]
        hi, lo = k[v[k] > med_fit], k[v[k] <= med_fit]
        t5, b5 = k[v[k] > cut_fit[3]], k[v[k] <= cut_fit[0]]
        qy.append({"feature": nm, "year": Y, "n_above_fitmedian": len(hi), "up_above": y1[hi].mean() if len(hi) else np.nan,
                   "n_below": len(lo), "up_below": y1[lo].mean() if len(lo) else np.nan,
                   "diff_above_minus_below": (y1[hi].mean() - y1[lo].mean()) if len(hi) and len(lo) else np.nan,
                   "n_top_q": len(t5), "up_top_q": y1[t5].mean() if len(t5) else np.nan,
                   "n_bottom_q": len(b5), "up_bottom_q": y1[b5].mean() if len(b5) else np.nan})
save(pd.DataFrame(qr), "h1_quintiles")
QY = pd.DataFrame(qy)
save(QY, "h1_year_uprate")

# robustness: test AUC without the two stress years
rb = []
for nm in ("spread_q90q10", "VIX", "rv20", "median_return", "score_logit"):
    for lab, ex in (("all", []), ("excl 2020", [2020]), ("excl 2022", [2022]), ("excl 2020+2022", [2020, 2022])):
        k = np.array([i for i in range(s1, e1) if yr[i] not in ex])
        rb.append({"feature": nm, "sample": lab, "n": len(k), "auc": auc(y1[k], FEAT[nm][k])})
for lab, ex in (("all", []), ("excl 2020", [2020]), ("excl 2022", [2022]), ("excl 2020+2022", [2020, 2022])):
    k = np.array([i - s1 for i in range(s1, e1) if yr[i] not in ex])
    rb.append({"feature": "stack-4", "sample": lab, "n": len(k), "auc": auc(SK["y"].to_numpy()[k], SK["proba"].to_numpy()[k])})
    rb.append({"feature": "static", "sample": lab, "n": len(k), "auc": auc(ST1["y"].to_numpy()[k], ST1["proba"].to_numpy()[k])})
save(pd.DataFrame(rb), "h1_robust_years")
NCMP += len(rb)

# =====================================================================================================
# h = 5
# =====================================================================================================
ds5, y5, z5 = inp5.ds, inp5.ds.y, inp5.z
s5, e5 = ds5.period_positions("TEST")
n5 = e5 - s5
R, N = P5["R"], P5["N"]
blk = np.arange(n5) // R                                    # run_platt refits at i % R == 0 (checked below)
nb = blk.max() + 1
offs, upr, mp, ztr = [], [], [], []
for b in range(nb):
    t = s5 + b * R
    idx = inp5.train_rows(t, N)
    off, slope = dl.fit_cal(z5[idx], y5[idx], P5["cal"])
    assert slope == 1.0
    offs.append(off); upr.append(y5[idx].mean()); mp.append(expit(z5[idx]).mean()); ztr.append(z5[idx].mean())
offs, upr, mp, ztr = map(np.array, (offs, upr, mp, ztr))
off_row = offs[blk]
assert np.allclose(expit(z5[s5:e5] + off_row), PL["proba"].to_numpy())                      # blocks/offsets = run_platt's
assert np.allclose(expit(z5[s5:e5] + ST5.attrs["cal_offset"]), ST5["proba"].to_numpy())
yt = y5[s5:e5]
ps, pp = ST5["proba"].to_numpy(), PL["proba"].to_numpy()
pos, neg = np.flatnonzero(yt == 1), np.flatnonzero(yt == 0)


def conc(sig):
    """matrix (n_pos x n_neg): 1 if positive scored higher than negative, 0.5 on ties."""
    d = sig[pos][:, None] - sig[neg][None, :]
    return (d > 0) + 0.5 * (d == 0)


def decomp(sig):
    M = conc(sig)
    same = blk[pos][:, None] == blk[neg][None, :]
    return M, same


Cs, same = decomp(ps)
Cp, _ = decomp(pp)
npair = Cs.size
rows = []
for nm, M in (("static", Cs), ("platt-6", Cp)):
    rows.append({"method": nm, "auc_all": M.sum() / npair,
                 "auc_within_block": M[same].sum() / same.sum(), "auc_between_block": M[~same].sum() / (~same).sum(),
                 "share_pairs_within": same.sum() / npair, "n_pairs": npair})
dec = pd.DataFrame(rows)
tot_gain = (Cp.sum() - Cs.sum()) / npair
dec["gain_from_within_pairs_(auc pts)"] = [np.nan, (Cp[same].sum() - Cs[same].sum()) / npair]
dec["gain_from_between_pairs_(auc pts)"] = [np.nan, (Cp[~same].sum() - Cs[~same].sum()) / npair]
dec["total_gain"] = [np.nan, tot_gain]
assert abs(auc(yt, ps) - dec.auc_all[0]) < 1e-9 and abs(auc(yt, pp) - dec.auc_all[1]) < 1e-9
assert np.array_equal(Cs[same], Cp[same])                   # within-block ordering identical, pair by pair
save(dec, "h5_decomposition")

# --- 6. what the offset tracks ---
bl = pd.DataFrame({"block": np.arange(nb), "start": ds5.index[s5 + np.arange(nb) * R], "offset": offs, "trailing252_up_rate": upr,
                   "trailing252_mean_timesfm_pup": mp, "trailing252_mean_logit": ztr,
                   "next_block_up_rate_realised": [yt[blk == b].mean() for b in range(nb)],
                   "block_mean_logit": [z5[s5:e5][blk == b].mean() for b in range(nb)], "n": np.bincount(blk)})
save(bl, "h5_blocks")
cr = []
for a, b_ in (("offset", "trailing252_up_rate"), ("offset", "trailing252_mean_timesfm_pup"),
              ("offset", "trailing252_mean_logit"), ("trailing252_up_rate", "trailing252_mean_timesfm_pup"),
              ("offset", "next_block_up_rate_realised"), ("trailing252_up_rate", "next_block_up_rate_realised"),
              ("trailing252_mean_timesfm_pup", "next_block_up_rate_realised"), ("block_mean_logit", "next_block_up_rate_realised")):
    cr.append({"a": a, "b": b_, "pearson": pearsonr(bl[a], bl[b_])[0], "spearman": spearmanr(bl[a], bl[b_])[0], "n_blocks": nb})
d_ = bl.trailing252_up_rate - bl.trailing252_mean_timesfm_pup
cr.append({"a": "offset", "b": "trailing up-rate minus mean TimesFM P(up)", "pearson": pearsonr(bl.offset, d_)[0],
           "spearman": spearmanr(bl.offset, d_)[0], "n_blocks": nb})
save(pd.DataFrame(cr), "h5_offset_correlations")
NCMP += len(cr)

# signals on test
daily_up = np.array([y5[inp5.train_rows(t, N)].mean() for t in range(s5, e5)])             # refreshed every row, purged
block_up = upr[blk]
sig = {"static TimesFM": ps, "platt-6 (TimesFM + offset)": pp, "offset alone (block-constant)": off_row,
       "trailing-252 up-rate, refreshed every 21 rows": block_up, "trailing-252 up-rate, refreshed daily": daily_up,
       "trailing-252 mean TimesFM P(up), 21-row refresh": mp[blk], "TimesFM logit z alone": z5[s5:e5],
       "constant offset: block mean of z (z blockwise demeaned)": z5[s5:e5] - bl.block_mean_logit.to_numpy()[blk]}
z_tr21 = ztr[blk]                                                                        # trailing-252 mean of z, refreshed per block
z_trd = np.array([z5[inp5.train_rows(t, N)].mean() for t in range(s5, e5)])             # same, refreshed daily
sig["z minus trailing-252 mean z (21-row refresh; no labels used)"] = z5[s5:e5] - z_tr21
sig["z minus trailing-252 mean z (daily; no labels used)"] = z5[s5:e5] - z_trd
sig["minus trailing-252 mean z alone (sign chosen after seeing test)"] = -z_tr21
sg = pd.DataFrame([{"signal": k, "test_auc": auc(yt, v)} for k, v in sig.items()])
save(sg, "h5_signals")
NCMP += len(sg)


def frame(v):
    v = np.asarray(v, float)
    return pd.DataFrame({"y": yt, "proba": v, "thr": v.mean(), "pred": (v >= v.mean()).astype(int)}, index=PL.index)


bt5 = []
for nm, B in (("static", ST5), ("z minus trailing-252 mean z (21-row refresh)", frame(z5[s5:e5] - z_tr21)), ("trailing-252 up-rate, 21-row refresh", frame(block_up)),
              ("trailing-252 up-rate, daily", frame(daily_up)), ("offset alone", frame(off_row))):
    b = paired_bootstrap(PL, B, 5)
    bt5.append({"A": "platt-6", "B": nm, "d_auc": b["d_roc_auc"], "lo": b["d_roc_auc_lo"], "hi": b["d_roc_auc_hi"], "p_one_sided": b["p_roc_auc"]})
save(pd.DataFrame(bt5), "h5_bootstrap")
NCMP += len(bt5)

# --- 7. per year ---
yr5 = ds5.index.year.to_numpy()[s5:e5]
py = []
Dg = Cp - Cs
for Y in YEARS:
    k = np.flatnonzero(yr5 == Y)
    ip = np.flatnonzero(yr5[pos] == Y)                                   # positives in year Y
    py.append({"year": Y, "n": len(k), "up_rate": yt[k].mean(), "auc_static": auc(yt[k], ps[k]), "auc_platt6": auc(yt[k], pp[k]),
               "d_auc_within_year": auc(yt[k], pp[k]) - auc(yt[k], ps[k]), "auc_baserate_21": auc(yt[k], block_up[k]),
               "mean_offset": off_row[k].mean(), "offset_sd_within_year": off_row[k].std(),
               "share_of_total_gain_(pos in year)": Dg[ip].sum() / npair / tot_gain})
PY = pd.DataFrame(py)
sy = yr5[pos][:, None] == yr5[neg][None, :]
cls = {"same block": same, "same year, different block": sy & ~same, "different years": ~sy}
save(pd.DataFrame([{"pair_class": k, "share_of_pairs": m_.sum() / npair, "auc_static": Cs[m_].mean(), "auc_platt6": Cp[m_].mean(),
                    "gain_auc_pts_(share_weighted)": (Cp[m_].sum() - Cs[m_].sum()) / npair} for k, m_ in cls.items()]), "h5_pair_classes")
save(PY, "h5_by_year")
NCMP += len(py) * 3
ex = []
for lab, drop in (("all years", []), ("excl 2022", [2022]), ("excl 2020", [2020]), ("excl 2022+2024", [2022, 2024])):
    k = np.flatnonzero(~np.isin(yr5, drop))
    ex.append({"sample": lab, "n": len(k), "auc_static": auc(yt[k], ps[k]), "auc_platt6": auc(yt[k], pp[k]),
               "d_auc": auc(yt[k], pp[k]) - auc(yt[k], ps[k]), "auc_trailing_up_rate_21": auc(yt[k], block_up[k])})
save(pd.DataFrame(ex), "h5_robust_years")
NCMP += 12
print(f"\nnumber of comparisons / AUC figures computed in this exploratory script: about {NCMP}")

"""ROOT CAUSE, experiment A (step 2 of 2): exact decomposition of the official forecast, read from the cache that
timesfm_rootcause_capture.py wrote.  Per test day and step h (SPY series, all in log-price units):
    Q_h,q   official quantiles           = N_h,q + T_h
    N_h,q   the network's own quantiles  (after reverse RevIN; the detrended-series level)
    T_h     detrending line add-back     = apply * (c + m*h/L)       c = line value today, m = slope per L days
    r_t     today's close as the network sees it = a - apply*c       (a = log close today)
  => official median change  Qmed_h - a  =  (Nmed_h - r_t)  +  apply*m*h/L     (EXACT, checked below)
                                           network part        line-slope part
Scores (all through the official read-out, 9 quantiles interpolated + clip [0.05, 0.95]):
    official        P(Q_h > a)                                             (== the cached official score)
    network_only    P(N_h > r_t)         the network's call with the line's slope removed
    line_slope      P(a + apply*m*h/L + (N - Nmed)_h > a)  the line's slope alone, with the network's spread but no network drift
    (point versions: sign/size of Nmed-r_t, and of apply*m*h/L, also scored)
    revin_free      -z_last: how far today is from the window mean in window-sigma units (no network at all)
Writes results/timesfm_rootcause_decompose*.csv.   Run:  python timesfm_rootcause_decompose.py  (about 1 min)
"""
import numpy as np
import pandas as pd

import timesfm_lib as tl
from tf_paths import RES_DIR
from timesfm_rootcause_capture import load, CTX
from timesfm_rootcause_lib import test_dates, anchor, score_from_q, evaluate, labels, auc, HS, YEARS
from timesfm_verify_detrend import stack_ctx, detrend_np, aligned2

H = 5
dates = test_dates()
z = load()
assert (pd.DatetimeIndex(z["dates"]) == dates).all(), "capture incomplete"
Q, N, RM, RS = z["Q"], np.sort(z["N"], -1), z["RM"], z["RS"]
a = anchor(dates)
X = stack_ctx(dates)
m, c, app_np, res = detrend_np(X)
app = np.abs((Q - N).mean(-1)[:, 0]) > 1e-9              # the model's own decision (float32): 1 borderline day differs from the float64 replica
win = np.where(app[:, None], res[:, 0], X[:, 0])   # the series the network was shown (SPY)
m, c, app_np = m[:, 0], c[:, 0], app_np[:, 0]
k = np.arange(1, H + 1)
Tmod = (Q - N).mean(-1)                                   # what the model actually added back, per step
Tnp = np.where(app[:, None], c[:, None] + m[:, None] * k[None, :] / CTX, 0.0)
r_t = a - app * c
slope = np.where(app[:, None], m[:, None] * k[None, :] / CTX, 0.0)

# ------------------------------------------------------------------- reproduction / exactness checks
qo, _ = tl.aligned("multi", CTX, dates)
chk = {
    "n_days": len(dates), "applied_share": float(app.mean()),
    "max_abs_Qcapture_vs_official_cache": float(np.abs(Q - qo).max()),
    "max_abs_Qcapture_vs_Qout": float(np.abs(Q - z["Qout"]).max()),
    "max_abs_addback_model_vs_numpy_line": float(np.abs(Tmod - Tnp).max()),
    "max_abs_addback_spread_across_quantiles": float(np.abs((Q - N) - Tmod[..., None]).max()),
    "max_abs_N_plus_T_minus_Q": float(np.abs(N + Tnp[..., None] - Q).max()),
    "max_abs_median_identity(Qmed-a = (Nmed-r_t)+slope)": float(np.abs((Q[..., 4] - a[:, None]) - ((N[..., 4] - r_t[:, None]) + slope)).max()),
    "apply_float64_replica_vs_model_mismatch_days (borderline std ratio 0.50009)": int((app != app_np).sum()),
}
Zl = (r_t - RM) / RS                                      # today's close in RevIN (window mean/std) units
chk["max_abs_RevIN_mean_vs_window_mean"] = float(np.abs(RM - win.mean(-1)).max())
chk["max_rel_RevIN_std_vs_window_std"] = float((np.abs(RS - win.std(-1)) / win.std(-1)).max())
pd.DataFrame([chk]).T.rename(columns={0: "value"}).to_csv(RES_DIR / "timesfm_rootcause_repro.csv")
print(pd.DataFrame([chk]).T.to_string(header=False))

# ------------------------------------------------------------------- scores
sc = {"official": {}, "network_only": {}, "line_slope": {}, "network_point": {}, "line_slope_point": {},
      "official_point": {}, "revin_free_(-z_last)": {}}
for h in HS:
    sc["official"][h] = score_from_q(Q, a, h)
    sc["network_only"][h] = np.array([tl.p_up(N[i], "multi", h, r_t[i]) for i in range(len(a))])
    Ql = a[:, None, None] + slope[..., None] + (N - N[..., 4:5])
    sc["line_slope"][h] = score_from_q(Ql, a, h)
    sc["network_point"][h] = N[:, h - 1, 4] - r_t
    sc["line_slope_point"][h] = slope[:, h - 1]
    sc["official_point"][h] = Q[:, h - 1, 4] - a
    sc["revin_free_(-z_last)"][h] = -Zl
df = evaluate(sc, dates)
df.to_csv(RES_DIR / "timesfm_rootcause_components.csv", index=False)

# ------------------------------------------------------------------- applied / not applied subsets
rows = []
for h in HS:
    y = labels(h)
    ix = dates.get_indexer(y.index)
    yy = y.to_numpy().astype(int)
    for sub, msk in (("detrend applied", app[ix]), ("detrend not applied", ~app[ix]), ("all days", np.ones(len(ix), bool))):
        for nm in ("official", "network_only", "line_slope", "revin_free_(-z_last)"):
            s = sc[nm][h][ix]
            row = dict(h=h, subset=sub, score=nm, n=int(msk.sum()), auc=auc(yy[msk], s[msk]))
            for yr in (2020, 2022, 2025):
                mm = msk & (y.index.year == yr)
                row[f"auc_{yr}"] = auc(yy[mm], s[mm]) if mm.sum() > 20 and 0 < yy[mm].mean() < 1 else np.nan
                row[f"n_{yr}"] = int(mm.sum())
            rows.append(row)
pd.DataFrame(rows).to_csv(RES_DIR / "timesfm_rootcause_subsets.csv", index=False)

# ------------------------------------------------------------------- sizes (log-price units and RevIN units)
rows = []
for h in HS:
    nc = N[:, h - 1, 4] - r_t
    for sub, msk in (("applied", app), ("not applied", ~app)):
        rows.append(dict(h=h, subset=sub, n=int(msk.sum()),
                         med_abs_official_change=np.median(np.abs(Q[msk, h - 1, 4] - a[msk])),
                         med_abs_network_change=np.median(np.abs(nc[msk])),
                         med_abs_line_slope=np.median(np.abs(slope[msk, h - 1])),
                         mean_network_change=nc[msk].mean(), mean_line_slope=slope[msk, h - 1].mean(),
                         med_RevIN_mean=np.median(RM[msk]), med_RevIN_std=np.median(RS[msk]),
                         med_abs_zlast=np.median(np.abs(Zl[msk])),
                         med_abs_slope_in_sigma=np.median(np.abs(slope[msk, h - 1] / RS[msk])),
                         med_abs_network_change_in_sigma=np.median(np.abs(nc[msk] / RS[msk])),
                         corr_network_change_vs_zlast=np.corrcoef(nc[msk], Zl[msk])[0, 1],
                         beta_netchange_sigma_on_zlast=np.polyfit(Zl[msk], nc[msk] / RS[msk], 1)[0],
                         corr_network_change_vs_line_slope=np.corrcoef(nc[msk], slope[msk, h - 1])[0, 1] if msk.sum() > 3 and slope[msk, h - 1].std() > 0 else np.nan,
                         frac_official_up=(Q[msk, h - 1, 4] > a[msk]).mean(), frac_network_up=(nc[msk] > 0).mean(),
                         frac_slope_up=(slope[msk, h - 1] > 0).mean()))
pd.DataFrame(rows).to_csv(RES_DIR / "timesfm_rootcause_sizes.csv", index=False)

# ------------------------------------------------------------------- RevIN check: scale/location cancel in P(up)
Zq = (N - RM[:, None, None]) / RS[:, None, None]
alt = np.array([tl.p_up(Zq[i], "multi", 5, Zl[i]) for i in range(len(a))])
print("RevIN cancellation: max |P(up) network_only in log-price units - same in RevIN units| =",
      np.abs(alt - sc["network_only"][5]).max())

with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
    print(df[["model", "h", "auc_pooled", "ci_lo", "ci_hi", "d_vs_first", "d_vs_first_lo", "d_vs_first_hi"] + [f"auc_{y}" for y in YEARS]].to_string(index=False))
    print(pd.DataFrame(rows).T.to_string())
    print(pd.read_csv(RES_DIR / "timesfm_rootcause_subsets.csv").to_string(index=False))

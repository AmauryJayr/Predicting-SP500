"""EXPLORATORY (post-test, h=5, price input, ctx 256): what does the "rest" of TimesFM-3's forecast react to?

rest = d5 - turn part, as in timesfm_explore_split.py (d5 = median 5-day log-move forecast; turn part = OLS of d5 on SPY's
log moves over days 1 / 2-5 / 6-10 / 11-20 back, each also times vr and vr^2, plus vr; fitted once on the REAL contexts of
the same detrending mode over the test rows).  Every condition is scored with those real-data coefficients applied to the
condition's OWN context (features recomputed from the modified context), anchor = the modified context's last log price.

Subcommands (python timesfm_verify_rest.py <cmd> ...), all outputs results/timesfm_rest_*.csv:
  local              Q1: re-score every existing trigger / cause cache (no model run), contexts are replayed from the
                     original seeds and checked against the saved anchors     -> timesfm_rest_local.csv
  diag               Q2: leave-one-year-out HistGradientBoosting of the rest on past-return features (CORRELATIONAL only)
                                                                              -> timesfm_rest_diag.csv, timesfm_rest_diag_stretch.csv
  run <set>          Q3: new model runs (default detrending, price, ctx 256, all test dates); resumable, cache
                     results/timesfm_rest_cache/{set-condition}_s{seed}.npz
  synth              Q3: controlled synthetic test of the long-trend term (drift sweep, both detrending modes)
                                                                              -> timesfm_rest_synth.csv
  report             Q3: score the new caches like Q1                          -> timesfm_rest_new.csv
Seeds: replay uses the original BASE_SEEDs of timesfm_verify_cause / _trigger; new runs use BASE_SEED below (+ condition code, seed).
"""
import re
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import timesfm_lib as tl
import timesfm_verify_cause as vc
import timesfm_verify_trigger as vt
from tf_paths import RES_DIR
from timesfm_explore_split import SEGS, LONG, cols_for
from timesfm_explore_years import STRETCHES, POOLS
from timesfm_verify_detrend import test_dates, labels, CTX

CACHE = RES_DIR / "timesfm_rest_cache"
CACHE.mkdir(parents=True, exist_ok=True)
BASE_SEED = 20261102
YEARS = list(range(2019, 2026))
TURN_COLS = cols_for(SEGS)
LONG_COLS = cols_for(LONG)
KEY_SPANS = {"r22reb": ["2022 rebound 1", "2022 rebound 2", "2022 rebound 3"], "r25rec": ["2025 recovery"],
             "f20": ["2020 fall"], "f25": ["2025 fall"]}


def out(name):
    return RES_DIR / f"timesfm_rest_{name}.csv"


# ------------------------------------------------------------------------------------------------ data
def setup():
    p = tl.panel()
    dates = test_dates()
    pos = p["dates"].get_indexer(dates)
    assert (pos >= CTX - 1).all()
    return p["spy"], dates, pos


def real_ctx():
    spy, dates, pos = setup()
    return np.stack([spy[k - CTX + 1:k + 1] for k in pos])


def replay(fam, cond, L, seed):
    """Rebuild the exact contexts (n, 256) of an existing cache file from its original seeds."""
    spy, dates, pos = setup()
    r_all = np.diff(spy)
    sig = vc.ewma_sigma(r_all)
    if fam == "cause":
        rng = np.random.default_rng([vc.BASE_SEED, vc.CONDS.index(cond), seed])
    else:
        rng = np.random.default_rng([vt.BASE_SEED, vt.CODE[cond], L, seed])
    xs = []
    for k in pos:
        lo = k - (CTX - 1)
        r, s = r_all[lo:k], sig[lo:k]
        rr = vc.transform(cond, r, s, rng) if fam == "cause" else vt.build(cond, L, r, rng, s)
        xs.append(np.concatenate([[spy[lo]], spy[lo] + np.cumsum(rr)]))
    return np.stack(xs)


def feats(xs):
    """Turn-part features of each context row (xs: (n, 256) log prices, last = anchor)."""
    r1 = np.diff(xs, axis=1)
    f = {"vr": r1[:, -20:].std(1, ddof=1) / r1.std(1, ddof=1)}
    for a, b in LONG:
        c = f"s{a + 1}_{b}"
        f[c] = xs[:, -1 - a] - xs[:, -1 - b]
        f[c + "_x"] = f[c] * f["vr"]
        f[c + "_x2"] = f[c] * f["vr"] ** 2
    return pd.DataFrame(f)


def frame_of(q, xs, dates, y):
    """d5, P(up) and turn features from forecast quantiles q and the (possibly modified) contexts xs."""
    a = xs[:, -1]
    t = feats(xs)
    t.index = pd.DatetimeIndex(dates)
    t["d5"] = q[:, 4, 4] - a
    t["pup"] = [tl.p_up(q[i], "price", 5, a[i]) for i in range(len(q))]
    t = t.loc[y.index]
    t["y"] = y
    return t


def fit_beta(t, cols=TURN_COLS):
    X = np.column_stack([np.ones(len(t))] + [t[c] for c in cols])
    return np.linalg.lstsq(X, t["d5"].to_numpy(), rcond=None)[0]


def turn_of(t, beta, cols=TURN_COLS):
    return np.column_stack([np.ones(len(t))] + [t[c] for c in cols]) @ beta


def spans():
    d = {lab: (a, e) for lab, _, a, e, _ in STRETCHES}
    return {k: [d[m] for m in v] for k, v in KEY_SPANS.items()}


SPANS = spans()


def score(t, beta):
    """One flat dict of per-year / all-years AUCs and rest>0 shares for a condition frame."""
    t = t.copy()
    t["turn"] = turn_of(t, beta)
    t["rest"] = t["d5"] - t["turn"]
    o = {}
    for yr in YEARS + ["all"]:
        g = t if yr == "all" else t[t.index.year == yr]
        for nm, col in (("pup", "pup"), ("turn", "turn"), ("rest", "rest")):
            o[f"{nm}_auc_{yr}"] = roc_auc_score(g["y"], g[col])
    for k, sp in SPANS.items():
        g = pd.concat([t.loc[a:e] for a, e in sp])
        o[f"rest_pos_{k}"] = (g["rest"] > 0).mean()
        o[f"restc_pos_{k}"] = (g["rest"] > t["rest"].median()).mean()      # centred: vs the condition's own median rest
        o[f"restz_{k}"] = (g["rest"].mean() - t["rest"].mean()) / t["rest"].std()
        o[f"turn_pos_{k}"] = (g["turn"] > 0).mean()
        o[f"d5_pos_{k}"] = (g["d5"] > 0).mean()
        o[f"up_{k}"] = g["y"].mean()
    # same, but with the turn part refitted on this condition's own contexts (own coefficients)
    t["rest_own"] = t["d5"] - turn_of(t, fit_beta(t))
    for yr in YEARS + ["all"]:
        g = t if yr == "all" else t[t.index.year == yr]
        o[f"rest_own_auc_{yr}"] = roc_auc_score(g["y"], g["rest_own"])
    for k, sp in SPANS.items():
        g = pd.concat([t.loc[a:e] for a, e in sp])
        o[f"restc_own_pos_{k}"] = (g["rest_own"] > t["rest_own"].median()).mean()
    o["r2_own_fit"] = 1 - t["rest_own"].var() / t["d5"].var()
    o["rest_pos_all"] = (t["rest"] > 0).mean()
    o["rest_sd"] = t["rest"].std()
    o["rest_mean"] = t["rest"].mean()
    o["r2_turn_in_cond"] = 1 - (t["d5"] - t["turn"]).var() / t["d5"].var()
    return o


def real_ref(mode):
    f = RES_DIR / "timesfm_cache_detrend" / f"{mode}_price_ctx256.npz"
    z = np.load(f, allow_pickle=True)
    y = labels(5)
    t = frame_of(z["q"], real_ctx(), pd.DatetimeIndex(z["dates"]), y)
    return t, fit_beta(t)


# ------------------------------------------------------------------------------------------------ Q1
FILE_RE = {"cause": re.compile(r"^(default|off)_test_([A-Za-z0-9]+)_s(\d+)\.npz$"),
           "trigger": re.compile(r"^(default|off)_test_([A-Za-z0-9]+)_L(\d+)_s(\d+)\.npz$")}


def list_caches():
    items = {}
    for fam, d in (("cause", vc.CACHE), ("trigger", vt.CACHE)):
        for f in sorted(d.glob("*_test_*.npz")):
            m = FILE_RE[fam].match(f.name)
            if not m:
                continue
            g = m.groups()
            mode, cond = g[0], g[1]
            L, seed = (0, int(g[2])) if fam == "cause" else (int(g[2]), int(g[3]))
            items.setdefault((fam, mode, cond, L), []).append((seed, f))
    return items


def cmd_local():
    y = labels(5)
    refs = {m: real_ref(m) for m in ("default", "off")}
    betas = {m: refs[m][1] for m in refs}
    rows = []
    for m in refs:
        r = score(refs[m][0], betas[m])
        rows.append({"family": "real", "mode": m, "cond": "real", "L": 0, "nseeds": 1, "max_anchor_err": 0.0, **r})
    for (fam, mode, cond, L), lst in list_caches().items():
        per, errs = [], []
        for seed, f in lst:
            z = np.load(f, allow_pickle=True)
            xs = replay(fam, cond, L, seed)
            err = np.abs(xs[:, -1] - (z["q"][:, 4, 4] - z["s"][:, 1])).max()
            errs.append(err)
            t = frame_of(z["q"], xs, pd.DatetimeIndex(z["dates"]), y)
            per.append(score(t, betas[mode]))
        avg = pd.DataFrame(per).mean().to_dict()
        rows.append({"family": fam, "mode": mode, "cond": cond, "L": L, "nseeds": len(lst),
                     "max_anchor_err": max(errs), **avg})
        print(f"{fam}/{mode}/{cond}/L{L}: seeds {len(lst)} anchor err {max(errs):.1e}", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(out("local"), index=False)
    show(D)


SHOW = ["family", "mode", "cond", "L", "rest_auc_2020", "rest_auc_2022", "rest_auc_2025", "rest_auc_all",
        "turn_auc_2020", "turn_auc_2025", "pup_auc_all", "rest_pos_r22reb", "rest_pos_r25rec", "restc_pos_r22reb", "restc_pos_r25rec", "restc_pos_f20", "restc_pos_f25"]


def show(D):
    pd.set_option("display.width", 250)
    print(D[SHOW].round(3).to_string(index=False))


# ------------------------------------------------------------------------------------------------ Q2
def diag_features(X):
    """X: (n, 256) real contexts.  Returns dict of feature blocks (a), (b), (c)."""
    r = np.diff(X, axis=1)                                   # (n, 255), last = day 1 back
    vr = r[:, -20:].std(1, ddof=1) / r.std(1, ddof=1)
    A = np.column_stack([r[:, -20:], vr])
    B = np.column_stack([r[:, 5 * i:5 * i + 5].sum(1) for i in range(51)])   # 51 five-day sums, oldest first
    hi, lo = X.max(1), X.min(1)
    C = np.column_stack([
        X[:, -1] - hi, (CTX - 1) - X.argmax(1), (CTX - 1) - X.argmin(1), (X[:, -1] - lo) / (hi - lo),
        X[:, -1] - X.mean(1), X[:, -1] - X[:, -61], X[:, -1] - X[:, -121], vr,
        r[:, -10:].std(1, ddof=1) / r.std(1, ddof=1), r[:, -60:].std(1, ddof=1) / r.std(1, ddof=1)])
    return {"a_last20_vr": A, "b_5daysums": B, "c_handcrafted": C, "abc_all": np.column_stack([A, B, C])}


def cmd_diag():
    from sklearn.ensemble import HistGradientBoostingRegressor
    y = labels(5)
    t, beta = real_ref("default")
    t["turn"] = turn_of(t, beta)
    t["rest"] = t["d5"] - t["turn"]
    full = real_ctx()
    dts = pd.DatetimeIndex(test_dates())
    X = full[dts.get_indexer(t.index)]
    yrs = t.index.year.to_numpy()
    FB = diag_features(X)
    rows, srows = [], []
    base = {nm: {} for nm in FB}
    for nm, F in FB.items():
        oof = np.zeros(len(t))
        for yr in YEARS:
            tr, te = yrs != yr, yrs == yr
            mdl = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=30,
                                                random_state=0)
            mdl.fit(F[tr], t["rest"].to_numpy()[tr])
            oof[te] = mdl.predict(F[te])
        t["oof_" + nm] = oof
        for yr in YEARS + ["all"]:
            msk = np.ones(len(t), bool) if yr == "all" else yrs == yr
            g = t[msk]
            rows.append({"features": nm, "year": yr, "n": int(msk.sum()),
                         "oof_r2": 1 - ((g["rest"] - g["oof_" + nm]) ** 2).sum() / ((g["rest"] - g["rest"].mean()) ** 2).sum(),
                         "corr_with_rest": np.corrcoef(g["rest"], g["oof_" + nm])[0, 1],
                         "auc_oof_pred": roc_auc_score(g["y"], g["oof_" + nm]), "auc_rest": roc_auc_score(g["y"], g["rest"])})
        # rest-AUC of the part NOT predicted
        t["res_" + nm] = t["rest"] - t["oof_" + nm]
    for k, sp in SPANS.items():
        g = pd.concat([t.loc[a:e] for a, e in sp])
        row = {"span": k, "n": len(g), "up_rate": g["y"].mean(), "rest_pos": (g["rest"] > 0).mean()}
        for nm in FB:
            row["pos_" + nm] = (g["oof_" + nm] > 0).mean()
        srows.append(row)
    # in-sample Spearman of the rest with each handcrafted feature (pooled and per year), to pick Q3 interventions
    names = ["drawdown_from_high", "days_since_high", "days_since_low", "pos_in_range", "dist_from_mean", "move60",
             "move120", "vr20_255", "vr10_255", "vr60_255"]
    crow = []
    for j, nm in enumerate(names):
        f = pd.Series(FB["c_handcrafted"][:, j], index=t.index)
        o = {"feature": nm, "all": f.corr(t["rest"], method="spearman")}
        o.update({str(yr): f[t.index.year == yr].corr(t["rest"][t.index.year == yr], method="spearman") for yr in YEARS})
        o["turn_all"] = f.corr(t["turn"], method="spearman")
        crow.append(o)
    pd.DataFrame(crow).to_csv(out("diag_corr"), index=False)
    print(pd.DataFrame(crow).round(2).to_string(index=False))
    D, S = pd.DataFrame(rows), pd.DataFrame(srows)
    D.to_csv(out("diag"), index=False)
    S.to_csv(out("diag_stretch"), index=False)
    pd.set_option("display.width", 250)
    print(D.pivot(index="year", columns="features", values="oof_r2").round(3).to_string())
    print(D.pivot(index="year", columns="features", values="auc_oof_pred").assign(rest=D[D.features == "abc_all"].set_index("year")["auc_rest"]).round(3).to_string())
    print(S.round(2).to_string(index=False))


# ------------------------------------------------------------------------------------------------ Q3
# New conditions (default detrending, price input, ctx 256, all test dates).  x = real 256-point log-price context.
#   w{a}_{b}   returns of days a..b back replaced by a Gaussian bridge (same sd, same sum -> same last price); rest real
#   lindetr    the OLS line slope of the whole context (as in the model's own detrend, t=arange(-255,1)/256) removed;
#              t=0 at the last point, so the last price is unchanged
#   tiltflip   slope of that line reversed (x - 2*m*t): same shape around the line, opposite long trend
# Conditions listed here are run in the order given; each is cached per seed, so a run can be resumed or extended.
WINDOWS = {"w21_60": (20, 60), "w61_120": (60, 120), "w121_190": (120, 190), "w191_255": (190, 255)}
#   bg*        wider bridged blocks: days 21-255, 21-120, 121-255 back (same bridge rule as w*)
BGW = {"bg21_255": (20, 255), "bg21_120": (20, 120), "bg121_255": (120, 255)}
WINDOWS.update(BGW)
NEW = {c: i + 1 for i, c in enumerate(["w21_60", "w61_120", "w121_190", "w191_255", "lindetr", "tiltflip"] + list(BGW))}
NSTOCH = set(WINDOWS)
SETS = {"win": ["w21_60", "w61_120", "w121_190", "w191_255"], "trend": ["lindetr", "tiltflip"], "bg": list(BGW)}


def new_ctx(cond, x, rng):
    """x: (256,) real log prices -> modified context (last price unchanged)."""
    if cond in WINDOWS:
        a, b = WINDOWS[cond]
        r = np.diff(x)
        w = r[255 - b:255 - a]
        g = rng.normal(0.0, w.std(), len(w))
        r = r.copy()
        r[255 - b:255 - a] = g - g.mean() + w.mean()
        return np.concatenate([[x[0]], x[0] + np.cumsum(r)])
    t = np.arange(-(CTX - 1), 1) / CTX
    tm = t.mean()
    m = ((t - tm) * (x - x.mean())).sum() / ((t - tm) ** 2).sum()
    if cond == "lindetr":
        return x - m * t
    if cond == "tiltflip":
        return x - 2 * m * t
    raise ValueError(cond)


def new_contexts(cond, seed):
    X = real_ctx()
    rng = np.random.default_rng([BASE_SEED, NEW[cond], seed])
    return np.stack([new_ctx(cond, x, rng) for x in X])


def cmd_run(which):
    from timesfm_verify_detrend import get_model, set_mode
    m = get_model()
    set_mode(m, "default")
    dates = test_dates()
    conds = SETS.get(which, [which])
    for cond in conds:
        for seed in ((0, 1) if cond in NSTOCH else (0,)):
            f = CACHE / f"default_test_{cond}_s{seed}.npz"
            if f.exists():
                continue
            t0 = time.time()
            xs = new_contexts(cond, seed)
            assert xs.shape[1] == CTX
            q = vc.forecast_paths(m, xs)
            np.savez_compressed(f, dates=dates.to_numpy(), q=q, s=vc.summarise(q, xs), seed=seed, base_seed=BASE_SEED)
            print(f"default/test/{cond}/s{seed}: {time.time() - t0:.0f}s", flush=True)


def cmd_synth(n=200):
    """Controlled synthetic test of the long-trend term: iid Gaussian returns (sd 0.01) + a constant drift giving a total
    255-day move M; same noise draws for every M and both detrending modes.  Reports mean d5 and the model's own
    line add-back (m*5/256 where the detrend rule applies)."""
    from timesfm_verify_detrend import get_model, set_mode, detrend_np
    m = get_model()
    eps = np.stack([np.random.default_rng([BASE_SEED, 900, s]).normal(0, 0.01, 255) for s in range(n)])
    rows = []
    for mode in ("default", "off"):
        set_mode(m, mode)
        for M in (-0.4, -0.2, 0.0, 0.2, 0.4):
            r = eps - eps.mean(1, keepdims=True) + M / 255
            xs = np.concatenate([np.zeros((n, 1)), np.cumsum(r, 1)], 1)
            q = vc.forecast_paths(m, xs)
            d5 = q[:, 4, 4] - xs[:, -1]
            sl, _, ap, _ = detrend_np(xs)
            rows.append({"mode": mode, "M": M, "n": n, "mean_d5": d5.mean(), "se_d5": d5.std() / np.sqrt(n),
                         "apply_share": ap.mean(), "mean_addback": (sl * 5 / CTX * ap).mean(),
                         "mean_d5_minus_addback": (d5 - sl * 5 / CTX * ap).mean(), "mean_slope_x5_over_L": (sl * 5 / CTX).mean()})
    D = pd.DataFrame(rows)
    D.to_csv(out("synth"), index=False)
    print(D.round(5).to_string(index=False))


def cmd_report():
    y = labels(5)
    tref, beta = real_ref("default")
    conds = {}
    for f in sorted(CACHE.glob("default_test_*_s*.npz")):
        mm = re.match(r"default_test_(.+)_s(\d+)\.npz", f.name)
        conds.setdefault(mm.group(1), []).append((int(mm.group(2)), f))
    rows = [{"cond": "real", "nseeds": 1, "max_anchor_err": 0.0, **score(tref, beta)}]
    for cond, lst in conds.items():
        per, errs = [], []
        for seed, f in lst:
            z = np.load(f, allow_pickle=True)
            xs = new_contexts(cond, seed)
            errs.append(np.abs(xs[:, -1] - (z["q"][:, 4, 4] - z["s"][:, 1])).max())
            per.append(score(frame_of(z["q"], xs, pd.DatetimeIndex(z["dates"]), y), beta))
        rows.append({"cond": cond, "nseeds": len(lst), "max_anchor_err": max(errs), **pd.DataFrame(per).mean().to_dict()})
    D = pd.DataFrame(rows)
    D.to_csv(out("new"), index=False)
    pd.set_option("display.width", 250)
    c = ["cond", "nseeds", "max_anchor_err", "rest_auc_2020", "rest_auc_2022", "rest_auc_2025", "rest_auc_all",
         "rest_own_auc_2020", "rest_own_auc_2025", "r2_own_fit", "pup_auc_all", "restc_pos_r22reb", "restc_pos_r25rec",
         "restc_pos_f20", "restc_pos_f25"]
    print(D[c].round(3).to_string(index=False))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "local":
        cmd_local()
    elif cmd == "diag":
        cmd_diag()
    elif cmd == "run":
        cmd_run(sys.argv[2])
    elif cmd == "synth":
        cmd_synth()
    elif cmd == "report":
        cmd_report()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()

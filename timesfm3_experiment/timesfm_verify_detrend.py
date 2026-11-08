"""VERIFICATION (post-test, exploratory): is TimesFM-3's "bet on a turn" behaviour on SPY caused by the
model's built-in linear detrending, or by the network itself?  Nothing here touches a pre-registered file.

Subcommands (python timesfm_verify_detrend.py <cmd> ...), all write results/timesfm_detrend_*.csv:
  repro               recompute 64 test dates with default settings, compare with the official cache
  validate            check the numpy detrend (m, c, apply, trend add-back) against the model's own tensors
  stats               how often detrend applies; decomposition of the official forecasts (steps 3 and 4)
  run <off|always>    full-test forecasts with detrend switched off / forced on, resumable, own cache dir
  ablate              AUC / per-year corr / fall-rebound table for official vs off vs always (step 5)
  synth               synthetic controlled test (step 6)
"""
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import timesfm_lib as tl
from common import DATA_DIR, RES_DIR, Dataset
from timesfm_explore_years import STRETCHES

CTX, VARIANT, H5 = 256, "multi", 5
CACHE2 = RES_DIR / "timesfm_cache_detrend"
CACHE2.mkdir(parents=True, exist_ok=True)
THR = 0.5                                   # model default linear_detrending_threshold
PCBS, THREADS = 16, 8


def out(name):
    return RES_DIR / f"timesfm_detrend_{name}.csv"


# ---------------------------------------------------------------------------------------------- data
def test_dates():
    ds = Dataset(1)
    s, e = ds.period_positions("TEST")
    return ds.index[s:e]


def labels(h):
    ds = Dataset(h)
    s, e = ds.period_positions("TEST")
    return pd.Series(ds.y[s:e], index=ds.index[s:e])


def ret20(dates):
    ls = np.log(pd.read_pickle(DATA_DIR / "aligned_prices.pkl")["SPY"]["Close"].astype(float))
    return ls.diff(20).reindex(dates)


def official_q(dates):
    q, used = tl.aligned(VARIANT, CTX, dates)
    assert (used == CTX).all()
    return q


def anchor(dates):
    p = tl.panel()
    return pd.Series(p["spy"], index=p["dates"]).reindex(dates).to_numpy()


def score_arr(q, a, h):
    return np.array([tl.p_up(q[i], VARIANT, h, a[i]) for i in range(len(q))])


def stack_ctx(dates):
    cs = tl.contexts(VARIANT, CTX, dates)
    return np.stack([np.vstack([c[0][None, :], c[1]]).astype(np.float64) for c in cs])   # (n, 16, 256)


# ------------------------------------------------------------------- numpy replica of model.py:472-511
def detrend_np(x, thr=THR):
    """x (..., L).  OLS line on t = arange(-(L-1), 1)/L (model.py:473-475), apply rule (model.py:508)."""
    L = x.shape[-1]
    t = np.arange(-(L - 1), 1) / L
    tm = t.mean()
    m = ((t - tm) * (x - x.mean(-1, keepdims=True))).sum(-1) / ((t - tm) ** 2).sum()
    c = x.mean(-1) - m * tm                                  # value of the line at the last point (t = 0)
    res = x - (m[..., None] * t + c[..., None])
    apply = res.std(-1) < thr * x.std(-1)
    return m, c, apply, res


# ---------------------------------------------------------------------------------------------- model
def get_model():
    m = tl.load_model(PCBS, THREADS)
    tl.set_runtime(m, PCBS, THREADS)
    return m


def set_mode(m, mode):
    m.model.use_linear_detrending = mode != "off"
    m.model.linear_detrending_threshold = 1e9 if mode == "always" else THR


def cache_path(mode, variant=VARIANT):
    return CACHE2 / f"{mode}_{variant}_ctx{CTX}.npz"


def read2(mode, variant=VARIANT):
    f = cache_path(mode, variant)
    if not f.exists():
        return pd.DatetimeIndex([]), np.zeros((0, H5, tl.NQ))
    z = np.load(f)
    return pd.DatetimeIndex(z["dates"]), z["q"]


def aligned2(mode, dates):
    if mode == "official":
        return official_q(dates)
    d, q = (read2(mode[:-6], "price") if mode.endswith("_price") else read2(mode))
    i = d.get_indexer(pd.DatetimeIndex(dates))
    assert (i >= 0).all(), f"{int((i < 0).sum())} dates missing for {mode}"
    return q[i]


# ----------------------------------------------------------------------------------------------- repro
def cmd_repro(n=64):
    dates = test_dates()
    pick = dates[np.linspace(0, len(dates) - 1, n).astype(int)]
    m = get_model()
    set_mode(m, "default")
    assert m.model.use_linear_detrending and m.model.linear_detrending_threshold == 0.5
    t0 = time.time()
    q = tl.forecast(m, VARIANT, CTX, pick)
    d = np.abs(q - official_q(pick))
    print(f"repro {n} dates, {time.time() - t0:.1f}s: max abs diff {d.max():.3e}, mean {d.mean():.3e}, "
          f"max |q| scale {np.abs(q).max():.2f}")
    # toggling the runtime attribute changes outputs?
    set_mode(m, "off")
    q_off = tl.forecast(m, VARIANT, CTX, pick[:16])
    set_mode(m, "default")
    print(f"toggle check (16 dates): max |q_off - q_default| = {np.abs(q_off - q[:16]).max():.3f}; "
          f"median |delta| = {np.median(np.abs(q_off - q[:16])):.4f}")
    pd.DataFrame([{"n": n, "max_abs_diff": d.max(), "mean_abs_diff": d.mean(),
                   "toggle_max_abs_change": np.abs(q_off - q[:16]).max()}]).to_csv(out("repro"), index=False)


# ------------------------------------------------------------------------------------------- validate
def cmd_validate(n=48):
    """Wrap model.decode / model.forward to capture the tensors that the model actually uses."""
    import math
    import torch
    from timesfm3.torch import util
    dates = test_dates()
    pick = dates[np.linspace(0, len(dates) - 1, n).astype(int)]
    X = stack_ctx(pick)                                       # float64 copies of the float32 inputs
    m = get_model()
    set_mode(m, "default")
    mod = m.model
    assert mod.use_stitching
    cap = {"vals": [], "hor": [], "stitched": [], "dec": []}
    f0, d0 = mod.forward, mod.decode

    def fwd(inputs, **kw):
        o = f0(inputs, **kw)
        cap["vals"].append(inputs["values"].clone())          # BEFORE forward: detrended context + horizon
        cap["last_logits"] = o["logits"]
        return o

    def dec(*a, **kw):
        cap["last_logits"] = None
        res = d0(*a, **kw)
        lg = cap["last_logits"]
        H = res.shape[2]
        ip = mod.input_patch_len
        # recompute the stitch exactly as model.py:613-624
        ext = mod._stitching_extract_len
        ov = ext - ip
        nfp = max(math.ceil((H - ov) / ip), 1)
        n_ctx_p = lg.shape[2] - (nfp + mod.rolls - 1)
        idx = torch.arange(nfp) + (n_ctx_p - 1)
        hl = util.stitch_patches(lg[:, :, idx, :ext, :], ip)[:, :, :H, :]
        cap["stitched"].append(hl.clone())
        cap["dec"].append(res.clone())
        cap["nctx"] = n_ctx_p
        return res

    mod.forward, mod.decode = fwd, dec
    q = tl.forecast(m, VARIANT, CTX, pick)
    mod.forward, mod.decode = f0, d0
    assert np.abs(q - official_q(pick)).max() < 1e-4
    vals = torch.cat(cap["vals"]).numpy().astype(np.float64)  # (n, 16, n_patches, p)
    nctx = cap["nctx"] * mod.input_patch_len
    assert nctx == CTX, nctx
    seen = vals.reshape(vals.shape[0], vals.shape[1], -1)[:, :, :CTX]
    mm, cc, app, res = detrend_np(X)
    exp = np.where(app[..., None], res, X)
    d_in = np.abs(seen - exp)
    # classify what the model fed in: closer to residual or to raw?
    to_res = np.abs(seen - res).max(-1) < np.abs(seen - X).max(-1)
    mism = int((to_res != app).sum())
    print(f"validate {n} dates x 16 variates: apply(numpy) vs model-fed-detrended mismatches {mism}/{app.size}; "
          f"max |model input - numpy expected| = {d_in.max():.2e} (input scale {np.abs(X).max():.1f}); "
          f"applied share {app.mean():.3f}")
    # trend add-back: decode output - stitched logits  must equal m*k/L + c where applied
    dec_o = torch.cat(cap["dec"]).numpy().astype(np.float64)[:, :, :H5, :]
    st = torch.cat(cap["stitched"]).numpy().astype(np.float64)[:, :, :H5, :]
    k = np.arange(1, H5 + 1) / CTX
    trend = np.where(app[..., None], mm[..., None] * k + cc[..., None], 0.0)
    d_tr = np.abs((dec_o - st) - trend[..., None]).max()
    print(f"trend add-back: max |(decode - stitched logits) - (m*k/L + c)| = {d_tr:.2e} "
          f"(trend scale {np.abs(trend).max():.2f})")
    pd.DataFrame([{"n_dates": n, "n_series": app.size, "apply_mismatch": mism,
                   "max_input_diff": d_in.max(), "max_trend_diff": d_tr, "applied_share": app.mean()}]
                 ).to_csv(out("validate"), index=False)


# ------------------------------------------------------------------------------------------- stats
def r2(y, X):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(x) for x in X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    return 1 - res.var() / y.var(), b


def cmd_stats():
    dates = test_dates()
    X = stack_ctx(dates)
    mm, cc, app, res = detrend_np(X)
    q = official_q(dates)
    a = anchor(dates)
    r20 = ret20(dates).to_numpy()
    yr = dates.year.to_numpy()
    rows = []
    # step 3
    for y in sorted(set(yr)):
        s = yr == y
        rows.append({"table": "apply_share", "group": str(y), "spy": app[s, 0].mean(), "covariates": app[s, 1:].mean()})
    rows.append({"table": "apply_share", "group": "all", "spy": app[:, 0].mean(), "covariates": app[:, 1:].mean()})
    # step 4
    for h in (5, 1):
        sc = score_arr(q, a, h)
        D = q[:, h - 1, 4] - a                                # median forecast change
        line_t = cc[:, 0]                                     # line at the last observed point
        r_t = a - line_t                                      # logC[t] - line(t)
        slope = mm[:, 0] * h / CTX
        f = q[:, h - 1, 4] - (line_t + slope)                 # residual forecast after reverse RevIN
        dev = a - X[:, 0, :].mean(-1)                         # deviation from the 256-day mean
        for name, s in (("applied", app[:, 0]), ("not_applied", ~app[:, 0])):
            n = int(s.sum())
            if n < 10:
                continue
            row = {"table": "decomp", "h": h, "subset": name, "n": n, "share_of_days": n / len(dates),
                   "med_delta": np.median(D[s]), "mean_delta": D[s].mean(),
                   "corr_score_r20": np.corrcoef(sc[s], r20[s])[0, 1],
                   "corr_delta_r20": np.corrcoef(D[s], r20[s])[0, 1],
                   "corr_score_dev256": np.corrcoef(sc[s], dev[s])[0, 1],
                   "R2_delta_r20": r2(D[s], [r20[s]])[0], "R2_score_r20": r2(sc[s], [r20[s]])[0],
                   "R2_delta_dev256": r2(D[s], [dev[s]])[0], "R2_score_dev256": r2(sc[s], [dev[s]])[0],
                   "mean_p_up": sc[s].mean(), "frac_score_gt_half": (sc[s] > 0.5).mean()}
            row.update({"R2_delta_dev_r20": r2(D[s], [dev[s], r20[s]])[0], "R2_score_dev_r20": r2(sc[s], [dev[s], r20[s]])[0]})
            if name == "applied":
                row.update({"R2_delta_r_r20": r2(D[s], [r_t[s], r20[s]])[0], "R2_score_r_r20": r2(sc[s], [r_t[s], r20[s]])[0],
                            "R2_delta_r_slope_r20": r2(D[s], [r_t[s], slope[s], r20[s]])[0],
                            "R2_score_r_slope_r20": r2(sc[s], [r_t[s], slope[s], r20[s]])[0]})
                R2f, bf = r2(f[s], [r_t[s]])
                row.update({"beta_f_on_r": bf[1], "R2_f_on_r": R2f, "intercept_f": bf[0],
                            "med_slope_part": np.median(slope[s]), "med_gap_part": np.median(-r_t[s]),
                            "med_f": np.median(f[s]), "std_slope_part": slope[s].std(),
                            "std_gap_part": r_t[s].std(), "std_f": f[s].std(), "std_delta": D[s].std(),
                            "R2_delta_r": r2(D[s], [r_t[s]])[0], "R2_delta_slope": r2(D[s], [slope[s]])[0],
                            "R2_delta_r_slope": r2(D[s], [r_t[s], slope[s]])[0],
                            "R2_score_r": r2(sc[s], [r_t[s]])[0], "R2_score_slope": r2(sc[s], [slope[s]])[0],
                            "R2_score_r_slope": r2(sc[s], [r_t[s], slope[s]])[0],
                            "corr_r_r20": np.corrcoef(r_t[s], r20[s])[0, 1],
                            "corr_score_r": np.corrcoef(sc[s], r_t[s])[0, 1],
                            "corr_f_r": np.corrcoef(f[s], r_t[s])[0, 1]})
            rows.append(row)
    pd.DataFrame(rows).to_csv(out("stats"), index=False)
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format,
                           "display.max_columns", 60):
        print(pd.DataFrame(rows).loc[lambda d: d.table == "apply_share", ["group", "spy", "covariates"]].to_string(index=False))
        print(pd.DataFrame(rows).loc[lambda d: d.table == "decomp"].dropna(axis=1, how="all").T.to_string())


# ---------------------------------------------------------------------------------------------- run
def cmd_run(mode, variant=VARIANT, chunk=64):
    assert mode in ("off", "always", "default")
    dates = test_dates()
    have, q = read2(mode, variant)
    todo = dates.difference(have)
    m = get_model()
    set_mode(m, mode)
    print(f"{mode}: {len(have)} cached, {len(todo)} to do; use_linear_detrending={m.model.use_linear_detrending} "
          f"thr={m.model.linear_detrending_threshold}", flush=True)
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        t0 = time.time()
        qc = tl.forecast(m, variant, CTX, part)
        have, q = have.append(part), np.concatenate([q, qc])
        o = np.argsort(have.to_numpy())
        np.savez_compressed(cache_path(mode, variant), dates=have.to_numpy()[o], q=q[o])
        print(f"{mode}: {i + len(part)}/{len(todo)}, {(time.time() - t0) / len(part):.3f}s/forecast", flush=True)


# ------------------------------------------------------------------------------------------- ablate
def cmd_ablate():
    dates = test_dates()
    a = anchor(dates)
    r20 = ret20(dates)
    modes = ["official", "off", "always", "default_price", "off_price"]
    modes = [x for x in modes if x == "official" or
             (cache_path(x[:-6] if x.endswith("_price") else x, "price" if x.endswith("_price") else VARIANT).exists()
              and len(read2(x[:-6], "price")[0] if x.endswith("_price") else read2(x)[0]) == len(dates))]
    Q = {k: aligned2(k, dates) for k in modes}
    S = {k: {h: pd.Series(score_arr(Q[k], a, h), index=dates) for h in (1, 5)} for k in modes}
    y = {h: labels(h) for h in (1, 5)}
    rows = []
    for k in modes:
        for h in (1, 5):
            yy = y[h]
            sc = S[k][h].reindex(yy.index)
            rows.append({"table": "auc", "mode": k, "h": h, "group": "all", "n": len(yy),
                         "auc": roc_auc_score(yy, sc), "corr_r20": sc.corr(r20.reindex(yy.index)),
                         "mean_p_up": sc.mean()})
        yy, sc = y[5], S[k][5].reindex(y[5].index)
        r = r20.reindex(yy.index)
        for yr in sorted(set(yy.index.year)):
            s = yy.index.year == yr
            rows.append({"table": "year_h5", "mode": k, "h": 5, "group": str(yr), "n": int(s.sum()),
                         "auc": roc_auc_score(yy[s], sc[s]), "corr_r20": sc[s].corr(r[s]),
                         "mean_p_up": sc[s].mean(), "frac_ge_half": (sc[s] >= 0.5).mean()})
        for lab, kind, s0, e0, ref in STRETCHES:
            g = sc.loc[s0:e0]
            rows.append({"table": "stretch_h5", "mode": k, "h": 5, "group": lab, "kind": kind, "n": len(g),
                         "mean_p_up": g.mean(), "frac_ge_half": (g >= 0.5).mean(), "frac_gt_half": (g > 0.5).mean()})
        for lab, kind, legs in (("2022 falls (all)", "fall", ["2022 fall 1", "2022 fall 2", "2022 fall 3"]),
                                ("2022 rebounds (all)", "recovery", ["2022 rebound 1", "2022 rebound 2", "2022 rebound 3"])):
            g = pd.concat([sc.loc[s0:e0] for l2, _, s0, e0, _ in STRETCHES if l2 in legs])
            rows.append({"table": "stretch_h5", "mode": k, "h": 5, "group": lab, "kind": kind, "n": len(g),
                         "mean_p_up": g.mean(), "frac_ge_half": (g >= 0.5).mean(), "frac_gt_half": (g > 0.5).mean()})
    # official calibrated 'called up' rule, for reference
    pk = pd.read_pickle(RES_DIR / "timesfm_test_predictions.pkl")[("tfm.h5", "timesfm")]
    for lab, kind, s0, e0, ref in STRETCHES:
        rows.append({"table": "stretch_h5", "mode": "official_calibrated_rule", "h": 5, "group": lab, "kind": kind,
                     "n": len(pk.loc[s0:e0]), "mean_p_up": pk.loc[s0:e0, "proba"].mean(),
                     "frac_ge_half": pk.loc[s0:e0, "pred"].mean()})
    df = pd.DataFrame(rows)
    df.to_csv(out("ablation"), index=False)
    # also: how far apart are the forecasts themselves
    for k in modes[1:]:
        d = Q[k][:, 4, 4] - Q["official"][:, 4, 4]
        print(f"{k} vs official: step-5 median differs by median |{np.median(np.abs(d)):.4f}|, "
              f"corr(score5 off, score5 official) = {S[k][5].corr(S['official'][5]):.3f}")
    with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format, "display.max_rows", 200):
        print(df[df.table == "auc"].to_string(index=False))
        yr = df[df.table == "year_h5"].pivot(index="group", columns="mode", values=["auc", "corr_r20"])
        print(yr.to_string())
        st = df[df.table == "stretch_h5"].pivot_table(index="group", columns="mode", values=["mean_p_up", "frac_ge_half"], sort=False)
        print(st.to_string())


# --------------------------------------------------------------------------------------------- synth
def synth_series(kind, seed, L=CTX, drift=6e-4):
    r = np.random.default_rng(seed)
    t = np.arange(L)
    base = 5.0
    if kind == "rw":
        return base + np.cumsum(r.normal(0, 0.01, L)), None
    noise_sd, ramp = kind
    x = base + drift * t + r.normal(0, noise_sd, L)
    j = np.arange(1, 21) / 20
    x[-20:] += ramp * j
    return x, None


def cmd_synth(n_seeds=20):
    m = get_model()
    cases = [("trend", sd, rp) for sd in (0.002, 0.008) for rp in (0.0, -0.05, -0.10, 0.05, 0.10)] + [("rw", None, None)]
    rows = []
    for mode in ("default", "off", "always"):
        set_mode(m, mode)
        for case in cases:
            kind = "rw" if case[0] == "rw" else (case[1], case[2])
            xs = [synth_series(kind, s)[0].astype(np.float32) for s in range(n_seeds)]
            _, _, app, _ = detrend_np(np.stack(xs).astype(np.float64))
            if mode == "always":
                app = np.ones_like(app)
            elif mode == "off":
                app = np.zeros_like(app)
            qq = tl.forecast_target_only(m, xs) if hasattr(tl, "forecast_target_only") else _fc(m, xs)
            last = np.array([x[-1] for x in xs], dtype=np.float64)
            d5 = qq[:, 4, 4] - last
            p5 = np.array([tl.p_up(qq[i], "price", 5, last[i]) for i in range(n_seeds)])
            c20 = np.array([x[-1] - x[-21] for x in xs])
            rows.append({"mode": mode, "case": (f"trend sd={case[1]} ramp={case[2]:+.2f}" if case[0] == "trend" else "random walk"),
                         "n_seeds": n_seeds, "chg20_median": np.median(c20), "applied_share": app.mean(),
                         "med_step5_change": np.median(d5), "med_P_up": np.median(p5),
                         "min_P_up": p5.min(), "max_P_up": p5.max()})
    # many random walks (daily sd 1%, with and without drift): does step-5 forecast relate to the last-20-day move?
    rw = []
    for drift in (0.0, 4e-4):
        xs = []
        for sd in range(400):
            r = np.random.default_rng(1000 + sd)
            xs.append((5.0 + np.cumsum(r.normal(drift, 0.01, CTX))).astype(np.float32))
        c20 = np.array([x[-1] - x[-21] for x in xs])
        last = np.array([x[-1] for x in xs], dtype=np.float64)
        _, _, app0, _ = detrend_np(np.stack(xs).astype(np.float64))
        for mode in ("default", "off", "always"):
            set_mode(m, mode)
            qq = _fc(m, xs)
            d5 = qq[:, 4, 4] - last
            p5 = np.array([tl.p_up(qq[i], "price", 5, last[i]) for i in range(len(xs))])
            ap = {"default": app0, "off": np.zeros_like(app0), "always": np.ones_like(app0)}[mode]
            rw.append({"drift": drift, "mode": mode, "n": len(xs), "applied_share": ap.mean(),
                       "corr_delta5_chg20": np.corrcoef(d5, c20)[0, 1], "corr_pup_chg20": np.corrcoef(p5, c20)[0, 1],
                       "med_delta5": np.median(d5), "mean_pup": p5.mean()})
    pd.DataFrame(rw).to_csv(out("synthetic_rwcorr"), index=False)
    print(pd.DataFrame(rw).to_string(index=False))
    df = pd.DataFrame(rows)
    df.to_csv(out("synthetic"), index=False)
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format, "display.max_rows", 100):
        print(df.to_string(index=False))


def _fc(m, xs):
    import torch
    torch.manual_seed(0)
    with torch.no_grad():
        o = list(m.predict_batch(contexts=list(xs), horizon=H5, return_quantiles=True, make_positive=False,
                                 use_symmetric_averaging=False, use_znorm=False, sort_quantiles=True))
    return np.stack([x.quantiles for x in o]).astype(np.float64)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"repro": cmd_repro, "validate": cmd_validate, "stats": cmd_stats, "run": cmd_run, "ablate": cmd_ablate,
     "synth": cmd_synth}[cmd](*args)

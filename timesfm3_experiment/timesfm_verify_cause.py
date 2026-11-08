"""VERIFICATION (post-test, exploratory): which property of the real SPY input makes zero-shot TimesFM-3 forecast a
short-term reversal?  Controlled transforms of each real 256-point log-price context (variant `price`, ctx 256).

Subcommands (python timesfm_verify_cause.py <cmd> ...):
  run <off|default> <test|early>   forecast all conditions for every decision date (resumable, own cache dir)
  synth                            condition 7: vol-burst synthetic (20 seeds), both detrend modes
  cond                             condition 8: real-SPY terciles + EWMA-gap regressions
  report                           metrics table (conditions 1-6) -> results/timesfm_cause_metrics.csv
Outputs: results/timesfm_cause_*.csv ; cache: results/timesfm_cause_cache/
"""
import sys
import time

import numpy as np
import pandas as pd

import timesfm_lib as tl
from timesfm_verify_detrend import get_model, set_mode, test_dates, CTX, H5
from tf_paths import RES_DIR

CACHE = RES_DIR / "timesfm_cause_cache"
CACHE.mkdir(parents=True, exist_ok=True)
BASE_SEED = 20261031
LAM = 0.94
SEEDS = [0, 1, 2]
# (name, stochastic?)  names parsed in transform()
CONDS = ["real", "iid", "block5", "block20", "block60", "gauss", "volstd", "mirror", "reverse",
         "signflip", "gaussvol", "zshuf"]      # added after the first results: keep the |r| / vol path, destroy signs
STOCH = {"iid", "block5", "block20", "block60", "gauss", "signflip", "gaussvol", "zshuf"}


def out(name):
    return RES_DIR / f"timesfm_cause_{name}.csv"


def dates_for(period):
    if period == "test":
        return test_dates()
    d = tl.panel()["dates"]
    return d[(d >= "2013-01-01") & (d <= "2018-12-31")]


def ewma_sigma(r_all):
    """causal EWMA vol: sigma[k] uses returns r_all[:k] only (sigma[k] is the vol known before return k)."""
    s2 = np.empty(len(r_all))
    v = np.var(r_all[:60])
    for k in range(len(r_all)):
        s2[k] = v                                   # known before return k
        v = LAM * v + (1 - LAM) * r_all[k] ** 2
    return np.sqrt(s2)


def transform(cond, r, sig, rng):
    """r: 255 real returns of the window, sig: their causal EWMA sigmas."""
    if cond == "real":
        return r
    if cond == "iid":
        return rng.permutation(r)
    if cond.startswith("block"):
        b = int(cond[5:])
        blocks = [r[i:i + b] for i in range(0, len(r), b)]
        return np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
    if cond == "gauss":
        return rng.normal(r.mean(), r.std(), len(r))
    if cond == "volstd":
        z = r / sig
        return z * (r.std() / z.std())              # rescale so the window sd equals the real window sd
    if cond == "signflip":                          # keep every |r| in place (vol clustering), random signs
        return r * rng.choice([-1.0, 1.0], len(r))
    if cond == "gaussvol":                          # Gaussian noise shaped by the real causal vol path
        e = s_ = sig * rng.normal(size=len(r))
        return e * (r.std() / np.sqrt(np.mean(sig ** 2))) + r.mean()
    if cond == "zshuf":                             # shuffle the vol-standardised returns, keep the real vol path
        z = rng.permutation(r / sig)
        e = z * sig
        return e * (r.std() / e.std())
    if cond == "mirror":
        return -r
    if cond == "reverse":
        return r[::-1]
    raise ValueError(cond)


def forecast_paths(m, xs, chunk=256):
    import torch
    res = []
    for i in range(0, len(xs), chunk):
        torch.manual_seed(tl.SEED)
        with torch.no_grad():
            o = list(m.predict_batch(contexts=[x.astype(np.float32) for x in xs[i:i + chunk]], horizon=H5,
                                     return_quantiles=True, make_positive=False, use_symmetric_averaging=False,
                                     use_znorm=False, sort_quantiles=True))
        res.append(np.stack([x.quantiles for x in o]).astype(np.float64))
    return np.concatenate(res)


def summarise(q, x):
    """per path: d1, d5 (median change), r20 of the constructed path, p5 = P(up at step 5)."""
    a = x[:, -1]
    d1 = q[:, 0, 4] - a
    d5 = q[:, 4, 4] - a
    r20 = x[:, -1] - x[:, -21]
    p5 = np.array([tl.p_up(q[i], "price", 5, a[i]) for i in range(len(q))])
    p1 = np.array([tl.p_up(q[i], "price", 1, a[i]) for i in range(len(q))])
    return np.stack([d1, d5, r20, p5, p1], 1)


def cmd_run(mode, period):
    dates = dates_for(period)
    p = tl.panel()
    spy = p["spy"]
    pos = p["dates"].get_indexer(dates)
    assert (pos >= CTX - 1).all()
    r_all = np.diff(spy)                            # r_all[k] = spy[k+1]-spy[k]
    sig = ewma_sigma(r_all)
    m = get_model()
    set_mode(m, mode)
    print(f"{mode}/{period}: n={len(dates)} detrending={m.model.use_linear_detrending}", flush=True)
    for ci, cond in enumerate(CONDS):
        for seed in (SEEDS if cond in STOCH else [0]):
            f = CACHE / f"{mode}_{period}_{cond}_s{seed}.npz"
            if f.exists():
                continue
            rng = np.random.default_rng([BASE_SEED, ci, seed])
            t0 = time.time()
            xs = []
            for k in pos:
                lo = k - (CTX - 1)                  # window = spy[lo..k]; returns r_all[lo..k-1]
                r = r_all[lo:k]
                s = sig[lo:k]
                rr = transform(cond, r, s, rng)
                xs.append(np.concatenate([[spy[lo]], spy[lo] + np.cumsum(rr)]))
            xs = np.stack(xs)
            assert xs.shape[1] == CTX
            q = forecast_paths(m, xs)
            np.savez_compressed(f, dates=dates.to_numpy(), q=q, s=summarise(q, xs), seed=seed, base_seed=BASE_SEED)
            print(f"{mode}/{period}/{cond}/s{seed}: {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------------------------- synth (7)
def cmd_synth(nseed=20):
    m = get_model()
    rows = []
    for mode in ("off", "default"):
        set_mode(m, mode)
        for move in (-0.05, 0.0, 0.05):
            for k in (0.5, 1, 2, 4):
                xs = []
                for s in range(nseed):
                    g = np.random.default_rng([BASE_SEED, 777, s])
                    z = g.normal(size=CTX - 1)      # common random numbers across k and move
                    r = 4e-4 + 0.01 * z
                    r[-20:] = 4e-4 + move / 20 + 0.01 * k * z[-20:]
                    xs.append(np.concatenate([[5.0], 5.0 + np.cumsum(r)]))
                xs = np.stack(xs)
                sm = summarise(forecast_paths(m, xs), xs)
                rows.append(dict(mode=mode, move=move, k=k, n_seeds=nseed, base_seed=BASE_SEED,
                                 med_d5=np.median(sm[:, 1]), med_p5=np.median(sm[:, 3]), med_r20=np.median(sm[:, 2]),
                                 frac_d5_follow=np.mean(np.sign(sm[:, 1]) == np.sign(move)) if move else np.nan,
                                 med_d5_minus_drift=np.median(sm[:, 1]) - 5 * 4e-4))
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out("synth"), index=False)


def cmd_synth2(nseed=300):
    """Higher-power version of 7: per k, nseed paths with the 20-step move drawn U(-8%, 8%); corr/slope of d5 on r20."""
    m = get_model()
    rows = []
    for mode in ("off", "default"):
        set_mode(m, mode)
        for k in (0.5, 1, 2, 4):
            xs = []
            for s in range(nseed):
                g = np.random.default_rng([BASE_SEED, 888, s])
                z = g.normal(size=CTX - 1)
                move = g.uniform(-0.08, 0.08)
                r = 4e-4 + 0.01 * z
                r[-20:] = 4e-4 + move / 20 + 0.01 * k * z[-20:]
                xs.append(np.concatenate([[5.0], 5.0 + np.cumsum(r)]))
            xs = np.stack(xs)
            sm = summarise(forecast_paths(m, xs), xs)
            rows.append(dict(mode=mode, k=k, n_paths=nseed, base_seed=BASE_SEED,
                             corr_d5_r20=np.corrcoef(sm[:, 1], sm[:, 2])[0, 1], slope_d5_r20=np.polyfit(sm[:, 2], sm[:, 1], 1)[0],
                             corr_p5_r20=np.corrcoef(sm[:, 3], sm[:, 2])[0, 1], mean_d5=sm[:, 1].mean(), sd_r20=sm[:, 2].std()))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out("synth2"), index=False)


# ---------------------------------------------------------------------------------------------- metrics
def metr(sm):
    d5, r20, p5 = sm[:, 1], sm[:, 2], sm[:, 3]
    c = np.corrcoef(d5, r20)[0, 1]
    slope = np.polyfit(r20, d5, 1)[0]
    cp = np.corrcoef(p5, r20)[0, 1]
    return c, slope, cp, np.mean(sm[:, 1]) , np.std(d5)


def cmd_report():
    rows = []
    for mode in ("off", "default"):
        for period in ("test", "early"):
            for cond in CONDS:
                fs = sorted(CACHE.glob(f"{mode}_{period}_{cond}_s*.npz"))
                if not fs:
                    continue
                ms = np.array([metr(np.load(f)["s"]) for f in fs])
                row = dict(mode=mode, period=period, cond=cond, n_seeds=len(fs),
                           seeds=",".join(str(int(np.load(f)["seed"])) for f in fs), base_seed=BASE_SEED)
                for j, nm in enumerate(["corr_d5_r20", "slope_d5_r20", "corr_p5_r20", "mean_d5", "sd_d5"]):
                    row[nm] = ms[:, j].mean()
                    row[nm + "_min"] = ms[:, j].min()
                    row[nm + "_max"] = ms[:, j].max()
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out("metrics"), index=False)
    with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
        print(df[["mode", "period", "cond", "n_seeds", "corr_d5_r20", "corr_d5_r20_min", "corr_d5_r20_max",
                  "slope_d5_r20", "corr_p5_r20", "mean_d5"]].to_string(index=False))


def cmd_segments():
    """Which part of the past drives d5?  OLS of d5 on non-overlapping segment returns (real forecasts, test)."""
    p = tl.panel()
    spy = p["spy"]
    rows = []
    segs = [("r1", 0, 1), ("r2_5", 1, 5), ("r6_10", 5, 10), ("r11_20", 10, 20), ("r21_60", 20, 60), ("r61_255", 60, 255)]
    for mode in ("off", "default"):
        f0 = CACHE / f"{mode}_test_real_s0.npz"
        if not f0.exists():
            continue
        z = np.load(f0)
        pos = p["dates"].get_indexer(pd.DatetimeIndex(z["dates"]))
        d5 = z["s"][:, 1]
        X = np.stack([np.array([spy[k - a] - spy[k - b] for _, a, b in segs]) for k in pos])
        # segment (a,b]: price at k-a minus price at k-b; (0,1) is today's return
        A = np.column_stack([np.ones(len(d5)), X])
        beta, *_ = np.linalg.lstsq(A, d5, rcond=None)
        r2_all = 1 - (d5 - A @ beta).var() / d5.var()
        sd = X.std(0)
        for j, (nm, _, _) in enumerate(segs):
            rows.append(dict(mode=mode, seg=nm, beta=beta[j + 1], beta_x_sd=beta[j + 1] * sd[j], univ_corr=np.corrcoef(X[:, j], d5)[0, 1],
                             r2_all=r2_all))
    df = pd.DataFrame(rows)
    df.to_csv(out("segments"), index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(df.to_string(index=False))


# ---------------------------------------------------------------------------------------------- cond 8
def r2(y, X):
    X = np.column_stack([np.ones(len(y))] + list(X))
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    return 1 - res.var() / y.var()


def cmd_cond():
    p = tl.panel()
    spy = p["spy"]
    rows = []
    for mode in ("off", "default"):
        for period in ("test", "early"):
            f = CACHE / f"{mode}_{period}_real_s0.npz"
            if not f.exists():
                continue
            z = np.load(f)
            dates = pd.DatetimeIndex(z["dates"])
            sm = z["s"]
            d5, r20 = sm[:, 1], sm[:, 2]
            pos = p["dates"].get_indexer(dates)
            r_all = np.diff(spy)
            ratio = np.array([r_all[k - 20:k].std() / r_all[k - (CTX - 1):k].std() for k in pos])
            terc = pd.qcut(ratio, 3, labels=False)
            for t in range(3):
                i = terc == t
                rows.append(dict(kind="tercile", mode=mode, period=period, grp=f"T{t + 1}", n=int(i.sum()),
                                 lo=ratio[i].min(), hi=ratio[i].max(), corr=np.corrcoef(d5[i], r20[i])[0, 1],
                                 slope=np.polyfit(r20[i], d5[i], 1)[0], r2=np.nan))
            rows.append(dict(kind="tercile", mode=mode, period=period, grp="ALL", n=len(d5), lo=ratio.min(), hi=ratio.max(),
                             corr=np.corrcoef(d5, r20)[0, 1], slope=np.polyfit(r20, d5, 1)[0], r2=np.nan))
            rows.append(dict(kind="corr_ratio_r20abs", mode=mode, period=period, grp="", n=len(d5),
                             corr=np.corrcoef(ratio, np.abs(r20))[0, 1]))
            gaps = {}
            for hl in (5, 10, 20, 60):
                e = pd.Series(spy).ewm(halflife=hl, adjust=False).mean().to_numpy()
                gaps[hl] = e[pos] - spy[pos]
            rows.append(dict(kind="r2", mode=mode, period=period, grp="r20", n=len(d5), r2=r2(d5, [r20])))
            for hl, g in gaps.items():
                rows.append(dict(kind="r2", mode=mode, period=period, grp=f"gap{hl}", n=len(d5), r2=r2(d5, [g]),
                                 corr=np.corrcoef(d5, g)[0, 1]))
                rows.append(dict(kind="r2", mode=mode, period=period, grp=f"r20+gap{hl}", n=len(d5), r2=r2(d5, [r20, g])))
            rows.append(dict(kind="r2", mode=mode, period=period, grp="r20+all4gaps", n=len(d5),
                             r2=r2(d5, [r20] + list(gaps.values()))))
    # extra: per real-ratio tercile for transforms that keep the last-20 |r| in place (signflip, mirror, gaussvol, zshuf use
    # the real vol path; volstd approx), and slope split by sign of r20 for the real forecasts.
    for mode in ("off", "default"):
        f0 = CACHE / f"{mode}_test_real_s0.npz"
        if not f0.exists():
            continue
        dates = pd.DatetimeIndex(np.load(f0)["dates"])
        pos = p["dates"].get_indexer(dates)
        r_all = np.diff(spy)
        ratio = np.array([r_all[k - 20:k].std() / r_all[k - (CTX - 1):k].std() for k in pos])
        terc = pd.qcut(ratio, 3, labels=False)
        for cond in ("real", "mirror", "signflip", "volstd", "gaussvol", "zshuf"):
            fs = sorted(CACHE.glob(f"{mode}_test_{cond}_s*.npz"))
            for t in range(3):
                i = terc == t
                cs = [np.corrcoef(np.load(g)["s"][i, 1], np.load(g)["s"][i, 2])[0, 1] for g in fs]
                sl = [np.polyfit(np.load(g)["s"][i, 2], np.load(g)["s"][i, 1], 1)[0] for g in fs]
                if fs:
                    rows.append(dict(kind="tercile_by_real_ratio", mode=mode, period="test", grp=f"{cond}/T{t + 1}", n=int(i.sum()),
                                     corr=np.mean(cs), slope=np.mean(sl)))
        sm = np.load(f0)["s"]
        for nm, i in (("r20>0", sm[:, 2] > 0), ("r20<0", sm[:, 2] < 0)):
            rows.append(dict(kind="sign_split", mode=mode, period="test", grp=nm, n=int(i.sum()),
                             corr=np.corrcoef(sm[i, 1], sm[i, 2])[0, 1], slope=np.polyfit(sm[i, 2], sm[i, 1], 1)[0]))
    df = pd.DataFrame(rows)
    df.to_csv(out("conditional"), index=False)
    with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False))


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"run": cmd_run, "synth": cmd_synth, "synth2": cmd_synth2, "cond": cmd_cond, "segments": cmd_segments, "report": cmd_report}[cmd](*args)

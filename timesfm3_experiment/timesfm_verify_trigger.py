"""TRIGGER localisation for zero-shot TimesFM-3 short-term reversal (variant `price`, ctx 256).
Transplant experiments: background (first 255-L returns) vs recent stretch (last L returns).

  run <off|default> <test|early> <key|lsweep>   Stage 1 forecasts (resumable, cache results/timesfm_trigger_cache/)
  report1                                        Stage 1 table -> results/timesfm_trigger_stage1.csv
  shape                                          Stage 2a synthetic shape test -> timesfm_trigger_shape_synth.csv
  realshape                                      Stage 2b shape features on real test data -> timesfm_trigger_realshape*.csv
Conditions: A real/real; B gauss bg + real stretch; C real bg + gauss-bridge stretch; D gauss bg + gauss-bridge stretch;
            E real bg + within-stretch shuffle; F real bg + time-reversed stretch.
"""
import sys
import time

import numpy as np
import pandas as pd

import timesfm_lib as tl
from timesfm_verify_detrend import get_model, set_mode, CTX, H5, test_dates
from timesfm_verify_cause import forecast_paths, summarise, dates_for, ewma_sigma
from tf_paths import RES_DIR

CACHE = RES_DIR / "timesfm_trigger_cache"
CACHE.mkdir(parents=True, exist_ok=True)
BASE_SEED = 20261101
SEEDS = [0, 1, 2]
CODE = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "Giid": 6, "Gb5": 7, "Gb20": 8, "Grev": 9, "Gmir": 10, "Gsf": 11,
        "Ggv": 12, "Gnear40": 13, "Gnear100": 14, "Gold40": 15, "Gold100": 16}
# Stage 2 (background dissection): real stretch (L=20) kept; only the background (first 235 returns) is modified.
#   Giid: bg returns permuted | Gb5/Gb20: bg block-shuffled | Grev: bg time-reversed | Gmir: bg mirrored about its mean
#   Gsf: bg random signs on (r-mean), keeps |r| path | Ggv: Gaussian x real causal vol path | Gnear40/100: real last N bg returns,
#   Gaussian before | Gold40/100: Gaussian last N bg returns, real before


def out(name):
    return RES_DIR / f"timesfm_trigger_{name}.csv"


def build(cond, L, r, rng, sig=None):
    """r: 255 real returns -> constructed 255 returns."""
    nb = len(r) - L
    bg, st = r[:nb], r[nb:]
    gbg = lambda: rng.normal(bg.mean(), bg.std(), nb)

    def bridge():
        g = rng.normal(0.0, st.std(), L)
        return g - g.mean() + st.mean()          # same sd as drawn from st.std(), sum == real stretch sum exactly
    if cond == "A":
        return r
    if cond == "B":
        return np.concatenate([gbg(), st])
    if cond == "C":
        return np.concatenate([bg, bridge()])
    if cond == "D":
        return np.concatenate([gbg(), bridge()])
    if cond == "E":
        return np.concatenate([bg, rng.permutation(st)])
    if cond == "F":
        return np.concatenate([bg, st[::-1]])
    if cond.startswith("G"):
        mu = bg.mean()
        if cond == "Giid":
            nbg = rng.permutation(bg)
        elif cond in ("Gb5", "Gb20"):
            b = int(cond[2:])
            blocks = [bg[i:i + b] for i in range(0, nb, b)]
            nbg = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
        elif cond == "Grev":
            nbg = bg[::-1]
        elif cond == "Gmir":
            nbg = 2 * mu - bg
        elif cond == "Gsf":
            nbg = (bg - mu) * rng.choice([-1.0, 1.0], nb) + mu
        elif cond == "Ggv":
            sb = sig[:nb]
            nbg = sb * rng.normal(size=nb) * (bg.std() / np.sqrt(np.mean(sb ** 2))) + mu
        elif cond.startswith("Gnear"):
            N = int(cond[5:])
            nbg = np.concatenate([gbg()[:nb - N], bg[nb - N:]])
        elif cond.startswith("Gold"):
            N = int(cond[4:])
            nbg = np.concatenate([bg[:nb - N], gbg()[nb - N:]])
        else:
            raise ValueError(cond)
        return np.concatenate([nbg, st])
    raise ValueError(cond)


STOCH = {"B", "C", "D", "E", "Giid", "Gb5", "Gb20", "Gsf", "Ggv", "Gnear40", "Gnear100", "Gold40", "Gold100"}


def cmd_run(mode, period, which):
    dates = dates_for(period)
    p = tl.panel()
    spy = p["spy"]
    pos = p["dates"].get_indexer(dates)
    assert (pos >= CTX - 1).all()
    r_all = np.diff(spy)
    sigall = ewma_sigma(r_all)
    m = get_model()
    set_mode(m, mode)
    if which == "bg":
        jobs = [(c, 20) for c in ("Giid", "Gb5", "Gb20", "Grev", "Gmir", "Gsf", "Ggv", "Gnear40", "Gold40", "Gnear100", "Gold100")]
    elif which == "key":
        jobs = [(c, 20) for c in "ABCDEF"]
    elif which == "keyAD":
        jobs = [(c, 20) for c in "ABCD"]
    elif which == "lsweep":
        jobs = [(c, L) for L in (5, 10, 40, 60) for c in "BC"]
    else:
        raise ValueError(which)
    for cond, L in jobs:
        for seed in (SEEDS if cond in STOCH else [0]):
            f = CACHE / f"{mode}_{period}_{cond}_L{L}_s{seed}.npz"
            if f.exists():
                continue
            rng = np.random.default_rng([BASE_SEED, CODE[cond], L, seed])
            t0 = time.time()
            xs = []
            for k in pos:
                lo = k - (CTX - 1)
                rr = build(cond, L, r_all[lo:k], rng, sigall[lo:k])
                xs.append(np.concatenate([[spy[lo]], spy[lo] + np.cumsum(rr)]))
            xs = np.stack(xs)
            assert xs.shape[1] == CTX
            q = forecast_paths(m, xs)
            np.savez_compressed(f, dates=dates.to_numpy(), q=q, s=summarise(q, xs), seed=seed, base_seed=BASE_SEED)
            print(f"{mode}/{period}/{cond}/L{L}/s{seed}: {time.time() - t0:.0f}s", flush=True)


def real_ratio(dates):
    p = tl.panel()
    spy = p["spy"]
    pos = p["dates"].get_indexer(pd.DatetimeIndex(dates))
    r_all = np.diff(spy)
    return np.array([r_all[k - 20:k].std() / r_all[k - (CTX - 1):k].std() for k in pos])


def mets(s, top):
    d5, r20, p5 = s[:, 1], s[:, 2], s[:, 3]
    return (np.corrcoef(d5, r20)[0, 1], np.polyfit(r20, d5, 1)[0], np.corrcoef(p5, r20)[0, 1],
            np.polyfit(r20[top], d5[top], 1)[0], np.corrcoef(d5[top], r20[top])[0, 1])


NAMES = ["corr_d5_r20", "slope_d5_r20", "corr_p5_r20", "slope_top", "corr_top"]


def cmd_report1():
    rows = []
    for f in sorted(CACHE.glob("*_L*_s*.npz")):
        pass
    keys = sorted({tuple(f.stem.split("_")[:4]) for f in CACHE.glob("*_L*_s*.npz")})
    for mode, period, cond, L in keys:
        fs = sorted(CACHE.glob(f"{mode}_{period}_{cond}_{L}_s*.npz"))
        dates = np.load(fs[0])["dates"]
        ratio = real_ratio(dates)
        top = ratio >= np.quantile(ratio, 2 / 3)
        ms = np.array([mets(np.load(f)["s"], top) for f in fs])
        row = dict(mode=mode, period=period, cond=cond, L=int(L[1:]), n_seeds=len(fs), n=len(dates), ntop=int(top.sum()),
                   base_seed=BASE_SEED)
        for j, nm in enumerate(NAMES):
            row[nm] = ms[:, j].mean()
            row[nm + "_min"] = ms[:, j].min()
            row[nm + "_max"] = ms[:, j].max()
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out("stage1"), index=False)
    with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
        print(df[["mode", "period", "cond", "L", "n_seeds", "corr_d5_r20", "corr_d5_r20_min", "corr_d5_r20_max",
                  "slope_d5_r20", "corr_p5_r20", "slope_top", "slope_top_min", "slope_top_max"]].to_string(index=False))


# ------------------------------------------------------------------------------------------ Stage 2a
def shape_returns(shape, M, L=20):
    """deterministic extra daily increments (length L) summing to M."""
    w = np.zeros(L)
    if shape == "jump_last":
        w[-1] = 1
    elif shape == "jump_d5":
        w[-5] = 1                                   # 5th day from the end
    elif shape == "jump_d15":
        w[-15] = 1                                  # 15th day from the end
    elif shape == "spread5":
        w[-5:] = 1 / 5
    elif shape == "spread20":
        w[:] = 1 / L
    elif shape == "accel":                          # convex: weight grows exponentially, e-folding 3 days
        w = np.exp(np.arange(L) / 3.0)
        w /= w.sum()
    elif shape == "decel":                          # mirror of accel: most of the move early in the stretch
        w = np.exp(np.arange(L)[::-1] / 3.0)
        w /= w.sum()
    else:
        raise ValueError(shape)
    return M * w


SHAPES = ["jump_last", "jump_d5", "jump_d15", "spread5", "spread20", "accel", "decel"]


def cmd_shape(nseed=300, modes="off"):
    nseed = int(nseed)
    m = get_model()
    rows = []
    for mode in modes.split(","):
        set_mode(m, mode)
        for shape in SHAPES:
            for k in (0.0, 1.0, 2.0):
                xs = []
                for s in range(nseed):
                    g = np.random.default_rng([BASE_SEED, 999, s])   # common random numbers across cells
                    z = g.normal(size=CTX - 1)
                    M = g.uniform(-0.08, 0.08)
                    r = 4e-4 + 0.01 * z
                    r[-20:] = 4e-4 + 0.01 * k * z[-20:] + shape_returns(shape, M)
                    xs.append(np.concatenate([[5.0], 5.0 + np.cumsum(r)]))
                xs = np.stack(xs)
                sm = summarise(forecast_paths(m, xs), xs)
                rows.append(dict(mode=mode, shape=shape, noise_k=k, n_paths=nseed, base_seed=BASE_SEED,
                                 corr_d5_r20=np.corrcoef(sm[:, 1], sm[:, 2])[0, 1],
                                 slope_d5_r20=np.polyfit(sm[:, 2], sm[:, 1], 1)[0],
                                 corr_p5_r20=np.corrcoef(sm[:, 3], sm[:, 2])[0, 1], sd_r20=sm[:, 2].std()))
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out("shape_synth"), index=False)


# ------------------------------------------------------------------------------------------ Stage 2b
def nw_se(X, res, lag=25):
    """Newey-West (Bartlett, lag 25 trading days) standard errors for OLS; windows overlap so naive SEs are far too small."""
    n = len(res)
    U = X * res[:, None]
    S = U.T @ U / n
    for l in range(1, lag + 1):
        G = U[l:].T @ U[:-l] / n
        S += (1 - l / (lag + 1)) * (G + G.T)
    XtXi = np.linalg.inv(X.T @ X / n)
    return np.sqrt(np.diag(XtXi @ S @ XtXi / n))


def features(r20arr):
    """r20arr: (n,20) last-20 returns. All features signed by direction of r20 and scaled by total variation TV=sum|r|."""
    tot = r20arr.sum(1)
    sg = np.sign(tot)
    sg[sg == 0] = 1
    tv = np.abs(r20arr).sum(1)
    c = np.cumsum(r20arr, 1) * sg[:, None]          # cumulative path in direction of the move
    c0 = np.concatenate([np.zeros((len(c), 1)), c], 1)
    final = c[:, -1]
    peak_giveback = (c0.max(1) - final) / tv       # how much of the peak move was already given back at the end
    # max run-up in direction of move: max_{i<j}(c_j-c_i)
    runmin = np.minimum.accumulate(c0, 1)
    runup = (c0 - runmin).max(1) / tv
    # max adverse drawdown inside the stretch (against direction): max_{i<j}(c_i-c_j)
    runmax = np.maximum.accumulate(c0, 1)
    drawdown = (runmax - c0).max(1) / tv
    f = dict(
        last5_share=sg * r20arr[:, -5:].sum(1) / tv,               # signed move in last 5 days (in direction of r20)
        maxday_share=np.abs(r20arr).max(1) / tv,                    # largest single day / total variation
        runup=runup,
        drawdown=drawdown,
        peak_giveback=peak_giveback,
        curvature=sg * (r20arr[:, 10:].sum(1) - r20arr[:, :10].sum(1)) / tv,   # 2nd-half minus 1st-half in direction
        efficiency=np.abs(tot) / tv,
    )
    return f, tot, tv


def cmd_realshape():
    p = tl.panel()
    spy = p["spy"]
    z = np.load(CACHE / "off_test_A_L20_s0.npz")
    dates = pd.DatetimeIndex(z["dates"])
    s = z["s"]
    d5, r20 = s[:, 1], s[:, 2]
    pos = p["dates"].get_indexer(dates)
    r_all = np.diff(spy)
    R = np.stack([r_all[k - 20:k] for k in pos])
    assert np.allclose(R.sum(1), r20)
    f, tot, tv = features(R)
    ratio = real_ratio(dates)
    top = ratio >= np.quantile(ratio, 2 / 3)
    sg = np.sign(r20)
    pbn = -sg * d5 / tv                              # normalised pull-back, independent of r20 sign (>0 = reversal)
    rows = []
    f2 = dict(f)
    f2["vol_ratio"] = ratio
    f2["abs_r20"] = np.abs(r20)
    for name, x in f2.items():
        for grp, idx in (("all", np.ones(len(d5), bool)), ("top_vol", top)):
            xi = x[idx]
            terc = pd.qcut(xi, 3, labels=False, duplicates="drop")
            for t in sorted(set(terc)):
                j = terc == t
                rows.append(dict(feature=name, group=grp, tercile=f"T{t + 1}", n=int(j.sum()), f_lo=xi[j].min(), f_hi=xi[j].max(),
                                 mean_pbn=pbn[idx][j].mean(), slope_d5_r20=np.polyfit(r20[idx][j], d5[idx][j], 1)[0],
                                 corr_d5_r20=np.corrcoef(r20[idx][j], d5[idx][j])[0, 1], mean_abs_r20=np.abs(r20[idx][j]).mean()))
            # interaction regression: d5 = a + b r20 + c*zf*r20 (+ main effect of zf); report c and its HAC-free t
            zf = (xi - xi.mean()) / xi.std()
            y, rr = d5[idx], r20[idx]
            X = np.column_stack([np.ones(len(y)), rr, zf, zf * rr])
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            res = y - X @ b
            se = nw_se(X, res)
            rows.append(dict(feature=name, group=grp, tercile="interaction", n=len(y), slope_d5_r20=b[1], mean_pbn=b[3],
                             corr_d5_r20=b[3] / se[3]))   # mean_pbn col = interaction coef; corr col = Newey-West t
    df = pd.DataFrame(rows)
    df.to_csv(out("realshape"), index=False)
    # multivariate: all features z-scored interacting with r20, in top-vol and all
    rows2 = []
    names = list(f.keys())
    for grp, idx in (("all", np.ones(len(d5), bool)), ("top_vol", top)):
        Z = np.column_stack([(f[n][idx] - f[n][idx].mean()) / f[n][idx].std() for n in names])
        y, rr = d5[idx], r20[idx]
        X = np.column_stack([np.ones(len(y)), rr, Z, Z * rr[:, None]])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ b
        se = nw_se(X, res)
        for j, n in enumerate(names):
            rows2.append(dict(group=grp, feature=n, base_slope=b[1], interaction=b[2 + len(names) + j],
                              t=b[2 + len(names) + j] / se[2 + len(names) + j]))
    d2 = pd.DataFrame(rows2)
    d2.to_csv(out("realshape_multi"), index=False)
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format, "display.max_rows", 200):
        print(df.to_string(index=False))
        print(d2.to_string(index=False))


# ------------------------------------------------------------------------------------------ Stage 2 (background, real data)
def bg_features(B):
    """B: (n,235) background returns (first 235 of the 255)."""
    mu = B.mean(1, keepdims=True)
    sd = B.std(1)
    z = (B - mu) / sd[:, None]
    A = np.abs(B - mu)
    f = dict(
        bg_sd=sd,
        bg_kurt=(z ** 4).mean(1),
        bg_maxabs_z=np.abs(z).max(1),
        bg_ac1=(z[:, 1:] * z[:, :-1]).mean(1),
        bg_volclust=np.array([np.corrcoef(a[1:], a[:-1])[0, 1] for a in A]),     # lag-1 autocorr of |r|
        bg_mean=mu[:, 0] / sd,                                                    # drift per unit vol
        bg_sd_recent60_over_all=B[:, -60:].std(1) / sd,
        bg_ret_recent60=B[:, -60:].sum(1) / sd,
    )
    O = B[:, :-40]                                   # old part of the background: returns 61..255 days before the end
    mo = O.mean(1, keepdims=True)
    so = O.std(1)
    zo = (O - mo) / so[:, None]
    f.update(old_sd=so, old_kurt=(zo ** 4).mean(1), old_mean=mo[:, 0] / so, old_maxabs_z=np.abs(zo).max(1))
    f["old_sd_over_recent_sd"] = so / B[:, -40:].std(1)
    return f


def cmd_realbg():
    p = tl.panel()
    spy = p["spy"]
    z = np.load(CACHE / "off_test_A_L20_s0.npz")
    dates = pd.DatetimeIndex(z["dates"])
    d5, r20 = z["s"][:, 1], z["s"][:, 2]
    pos = p["dates"].get_indexer(dates)
    r_all = np.diff(spy)
    B = np.stack([r_all[k - 255:k - 20] for k in pos])
    f = bg_features(B)
    ratio = real_ratio(dates)
    f["vol_ratio_last20_over_256"] = ratio
    # position of today's price inside the window's own range (0 = window low, 1 = window high)
    P = np.stack([spy[k - 255:k + 1] for k in pos])
    f["price_pos_in_range"] = (P[:, -1] - P.min(1)) / (P.max(1) - P.min(1))
    top = ratio >= np.quantile(ratio, 2 / 3)
    rows = []
    for grp, idx in (("all", np.ones(len(d5), bool)), ("top_vol", top)):
        for name, x in f.items():
            xi = x[idx]
            y, rr = d5[idx], r20[idx]
            terc = pd.qcut(xi, 3, labels=False, duplicates="drop")
            sl = [np.polyfit(rr[terc == t], y[terc == t], 1)[0] for t in sorted(set(terc))]
            zf = (xi - xi.mean()) / xi.std()
            X = np.column_stack([np.ones(len(y)), rr, zf, zf * rr])
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            res = y - X @ b
            se = nw_se(X, res)
            rows.append(dict(group=grp, feature=name, n=len(y), slope_T1=sl[0], slope_T2=sl[1], slope_T3=sl[2],
                             base_slope=b[1], interaction=b[3], t=b[3] / se[3], corr_feat_absr20=np.corrcoef(xi, np.abs(rr))[0, 1]))
    df = pd.DataFrame(rows)
    df.to_csv(out("realbg"), index=False)
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
        print(df.to_string(index=False))


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"run": cmd_run, "report1": cmd_report1, "shape": cmd_shape, "realshape": cmd_realshape, "realbg": cmd_realbg}[cmd](*args)

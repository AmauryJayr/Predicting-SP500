"""Generate every new forecast of the TimesFM-3 behaviour study (plan: timesfm_behaviour_plan.md, locked).
  python timesfm_behaviour_run.py repro | e1 [i/n] | e3 | e4 | e5 [limit]
Resumable; caches in results/timesfm_behaviour_cache/ (git-ignored). Threads: env BEH_THREADS (default 8)."""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import timesfm_lib as tl
import timesfm_behaviour_sim as S
from timesfm_verify_detrend import test_dates

THREADS = int(os.environ.get("BEH_THREADS", 8))
NSUB, N3, N4, N5 = 800, 3000, 4000, 1000
AUC_TARGETS = [0.52, 0.55, 0.60, 0.70, 0.80, 0.90]
MOVES = [-5, -3, -1, 0, 1, 3, 5]
GROUPS = {"equity": ["QQQ", "IWM"], "sectors": ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"],
          "bonds": ["TLT", "IEF"], "gold": ["GLD"], "vix": ["^VIX"]}
VARIANTS = ([f"drop_{c}" for c in tl.COVS] + [f"dropgroup_{g}" for g in GROUPS] + ["scr_each", "scr_block"])


def model():
    m = tl.load_model(16, THREADS)
    tl.set_runtime(m, 16, THREADS)
    return m


def sub_dates():
    td = test_dates()
    rng = np.random.default_rng(S.SEED_BEH)
    return td[np.sort(rng.choice(len(td), size=NSUB, replace=False))]


def pos_of(dates):
    return tl.panel()["dates"].get_indexer(pd.DatetimeIndex(dates))


def e1_inputs(variant, dates):
    p = tl.panel()
    pos = pos_of(dates)
    last_pre = int(p["dates"].get_indexer([pd.Timestamp("2018-12-31")])[0]) if pd.Timestamp("2018-12-31") in p["dates"] else \
        int((p["dates"] < pd.Timestamp("2019-01-01")).sum() - 1)
    tg, co = [], []
    for d, q in zip(dates, pos):
        sl = slice(q - S.CTX + 1, q + 1)
        tg.append(p["spy"][sl])
        C = p["cov"][:, sl].copy()
        if variant.startswith("drop_"):
            keep = [i for i, c in enumerate(tl.COVS) if c != variant[5:]]
            C = C[keep]
        elif variant.startswith("dropgroup_"):
            bad = set(GROUPS[variant[10:]])
            C = C[[i for i, c in enumerate(tl.COVS) if c not in bad]]
        else:
            rng = np.random.default_rng([S.SEED_BEH, int(pd.Timestamp(d).strftime("%Y%m%d"))])
            hi = min(last_pre, q - S.CTX)
            tp = rng.integers(S.CTX - 1, hi + 1, size=len(tl.COVS)) if variant == "scr_each" else \
                np.full(len(tl.COVS), int(rng.integers(S.CTX - 1, hi + 1)))
            for j, t in enumerate(tp):
                w = p["cov"][j, t - S.CTX + 1:t + 1]
                C[j] = w - w[-1] + p["cov"][j, q]
        co.append(C)
    return tg, co


def cmd_repro():
    m = model()
    rng = np.random.default_rng([S.SEED_BEH, 1])
    td = test_dates()
    rows = []
    d = td[np.sort(rng.choice(len(td), 30, replace=False))]
    t0 = time.time()
    p = tl.panel(); pos = pos_of(d)
    tg = [p["spy"][q - 255:q + 1] for q in pos]
    co = [p["cov"][:, q - 255:q + 1] for q in pos]
    q_multi = S.fc(m, tg, co)
    t1 = time.time()
    q_uni = S.fc(m, tg)
    t2 = time.time()
    a, _ = tl.aligned("multi", 256, d)
    z = np.load(tl.RES_DIR / "timesfm_ctrl_cache" / "tf_L_TEST.npz")
    i = pd.DatetimeIndex(z["dates"]).get_indexer(d)
    rows.append(dict(setting="multi (SPY + 15 side series) vs cache multi_ctx256", n=len(d),
                     max_abs_diff=float(np.abs(a - q_multi).max()), sec_per_forecast=(t1 - t0) / len(d)))
    rows.append(dict(setting="SPY only vs cache tf_L_TEST", n=len(d),
                     max_abs_diff=float(np.abs(z["q"][i] - q_uni).max()), sec_per_forecast=(t2 - t1) / len(d)))
    df = pd.DataFrame(rows)
    df.to_csv(tl.RES_DIR / "timesfm_behaviour_repro.csv", index=False)
    print(df.to_string())


def cmd_e1(part="0/1"):
    i, n = map(int, part.split("/"))
    m = model()
    dates = sub_dates()
    pd.Series(dates).to_csv(tl.RES_DIR / "timesfm_behaviour_e1_dates.csv", index=False, header=["date"])
    for v in VARIANTS[i::n]:
        def make(a, b, v=v):
            tg, co = e1_inputs(v, dates[a:b])
            p = tl.panel(); pos = pos_of(dates[a:b])
            return dict(q=S.fc(m, [p["spy"][q - 255:q + 1] for q in pos], co))
        S.run_cached(S.CACHE / f"e1_{v}.npz", len(dates), make, chunk=100)


# ------------------------------------------------------------------------------------------ E3
def e3_rho(target_auc, n=2_000_000):
    rng = np.random.default_rng([S.SEED_BEH, 3, 0])
    zx, ze = rng.standard_normal(n), rng.standard_normal(n)

    def auc(rho):
        y = (rho * zx + np.sqrt(1 - rho ** 2) * ze) > 0
        r = rankdata(zx)
        n1 = y.sum(); n0 = n - n1
        return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    lo, hi = 0.0, 0.99
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if auc(mid) < target_auc else (lo, mid)
    return (lo + hi) / 2, auc((lo + hi) / 2)


def e3_data(world):
    rng = np.random.default_rng([S.SEED_BEH, 3, 1])
    return rng.standard_normal((N3, 256)), rng.standard_normal((N3, 256))


def cmd_e3():
    world = S.World()
    m = model()
    zx, ze = e3_data(world)
    levels = [("0.50", 0.0, 0.5)] + [(f"{t:.2f}",) + e3_rho(t) for t in AUC_TARGETS]
    meta = []
    sig = world.sigma
    xl = S.to_levels(sig * zx[:, 1:])                       # (W, 256): x_0..x_255
    for name, rho, ref_auc in levels:
        ry = sig * (rho * zx + np.sqrt(1 - rho ** 2) * ze)  # ry[:,k-1] = r_y[k], k=1..256
        yl = S.to_levels(ry[:, :255])                        # y_0..y_255
        meta.append(dict(level=name, rho=rho, ref_auc=ref_auc))
        for kind in ("with", "without"):
            def make(a, b, kind=kind, yl=yl):
                return dict(q=S.fc(m, list(yl[a:b]), list(xl[a:b, None, :]) if kind == "with" else None))
            S.run_cached(S.CACHE / f"e3_{name}_{kind}.npz", N3, make, chunk=250)
        np.savez(S.CACHE / f"e3_{name}_truth.npz", fut=ry[:, 255], anchor=yl[:, -1], oracle=zx[:, 255],
                 ols=_ols_scores(ry, zx))
    pd.DataFrame(meta).to_csv(tl.RES_DIR / "timesfm_behaviour_e3_levels.csv", index=False)


def _ols_scores(ry, zx):
    """In-window OLS of r_y[k] on r_x[k-1] (k=1..255), score = slope * r_x[255]. Standardised units cancel (sigma)."""
    y = ry[:, :255]                                         # r_y[1..255]
    x = zx[:, :255]                                         # z_x[0..254]  = lag of r_y
    xm, ym = x.mean(1, keepdims=True), y.mean(1, keepdims=True)
    slope = ((x - xm) * (y - ym)).sum(1) / ((x - xm) ** 2).sum(1)
    return slope * zx[:, 255]


# ------------------------------------------------------------------------------------------ E4 / E5
def cmd_e4():
    world = S.World()
    m = model()
    print("SPY-like:", world.mu, world.sigma, world.g, flush=True)
    pd.Series({"mu": world.mu, "sigma": world.sigma, **{f"garch_{k}": v for k, v in world.g.items()}}).to_csv(
        tl.RES_DIR / "timesfm_behaviour_world.csv", header=["value"])
    for fam in "ABC":
        rng = np.random.default_rng([S.SEED_BEH, 4, ord(fam)])
        lv = S.to_levels(S.sim_returns(world, fam, N4, 255, rng))   # (N4, 256)
        np.save(S.CACHE / f"e4_{fam}_levels.npy", lv)

        def make(a, b, lv=lv):
            with S.Capture(m) as c:
                q = S.fc(m, list(lv[a:b]))
                t = c.take()
            assert np.abs(np.sort(t["Q"], -1) - q).max() < 1e-4, np.abs(np.sort(t["Q"], -1) - q).max()
            return dict(q=q, N=t["N"], Q=t["Q"])
        S.run_cached(S.CACHE / f"e4_{fam}.npz", N4, make, chunk=250)


def e5_backgrounds(world):
    rng = np.random.default_rng([S.SEED_BEH, 5])
    rw = S.sim_returns(world, "A", N5, 255, rng)
    p = tl.panel()
    last_pre = int((p["dates"] < pd.Timestamp("2019-01-01")).sum() - 1)
    ends = rng.integers(255, last_pre + 1, size=N5)
    real = np.stack([np.diff(p["spy"][q - 255:q + 1]) for q in ends])
    return {"rw": rw, "real": real}


def cmd_e5(limit=0):
    world = S.World()
    m = model()
    bgs = e5_backgrounds(world)
    for bg, ret in bgs.items():
        for k in (5, 20):
            for mv in MOVES:
                r = ret.copy()
                tail = r[:, -k:]
                r[:, -k:] = tail - tail.mean(1, keepdims=True) + np.log1p(mv / 100.0) / k
                lv = S.to_levels(r)

                def make(a, b, lv=lv):
                    return dict(q=S.fc(m, list(lv[a:b])))
                S.run_cached(S.CACHE / f"e5_{bg}_k{k}_m{mv:+d}.npz", N5, make, chunk=250)


if __name__ == "__main__":
    c = sys.argv[1]
    {"repro": cmd_repro, "e1": cmd_e1, "e3": cmd_e3, "e4": cmd_e4, "e5": cmd_e5}[c](*sys.argv[2:])

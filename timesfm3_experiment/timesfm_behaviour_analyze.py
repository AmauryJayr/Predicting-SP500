"""Analysis of the TimesFM-3 behaviour study (plan: timesfm_behaviour_plan.md). python timesfm_behaviour_analyze.py e1|e2|e3|e4|e5|e6
Writes results/timesfm_behaviour_*.csv and figures/timesfm_behaviour_reversal.png."""
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

import timesfm_lib as tl
import timesfm_behaviour_sim as S
import timesfm_behaviour_run as R
from tf_paths import RES_DIR, TF_DIR
from common import CORE_FEATURES, Dataset
from timesfm_rootcause_lib import auc, labels, boot_idx, evaluate, HS
from timesfm_verify_detrend import test_dates

OUT = lambda n: RES_DIR / f"timesfm_behaviour_{n}.csv"
CROSS6 = ["sector_breadth_above_sma50", "sector_dispersion_5d", "corr_spy_tlt_60d", "tlt_ret_20d",
          "iwm_minus_spy_ret_20d", "xly_minus_xlp_ret_20d"]


def anchors(dates):
    p = tl.panel()
    return pd.Series(p["spy"], index=p["dates"]).reindex(pd.DatetimeIndex(dates)).to_numpy()


def P(q, a, h):
    return S.p_up_levels(q, a, h)


def cached_q(name, dates):
    z = np.load(RES_DIR / name)
    i = pd.DatetimeIndex(z["dates"]).get_indexer(pd.DatetimeIndex(dates))
    assert (i >= 0).all()
    return z["q"][i]


def spy_only_full(dates):
    return cached_q("timesfm_ctrl_cache/tf_L_TEST.npz", dates), cached_q("timesfm_cache/multi_ctx256.npz", dates)


# ------------------------------------------------------------------------------------------ E1
def e1():
    dates = pd.DatetimeIndex(pd.read_csv(OUT("e1_dates"), parse_dates=["date"])["date"])
    a = anchors(dates)
    q_spy, q_full = spy_only_full(dates)
    var = {"SPY only (no side series)": q_spy}
    for v in R.VARIANTS:
        z = np.load(S.CACHE / f"e1_{v}.npz")
        assert int(z["done"]) == len(dates), v
        var[v] = z["q"]
    rows = []
    for nm, q in var.items():
        for h in HS:
            pf, pv = P(q_full, a, h), P(q, a, h)
            y = labels(h).reindex(dates).to_numpy()
            ok = ~np.isnan(y)
            rows.append(dict(variant=nm, h=h, n=len(dates), mean_abs_dP=np.abs(pv - pf).mean(),
                             spearman_with_full=spearmanr(pv, pf)[0],
                             pct_calls_flipped=100 * ((pv > .5) != (pf > .5)).mean(),
                             auc=auc(y[ok].astype(int), pv[ok]), auc_full=auc(y[ok].astype(int), pf[ok])))
    df = pd.DataFrame(rows)
    df.to_csv(OUT("e1"), index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(df.pivot(index="variant", columns="h", values=["mean_abs_dP", "spearman_with_full", "pct_calls_flipped", "auc"]).round(3).sort_values(("mean_abs_dP", 5), ascending=False).to_string())


# ------------------------------------------------------------------------------------------ E2
def e2():
    dates = test_dates()
    a = anchors(dates)
    q_spy, q_full = spy_only_full(dates)
    p = tl.panel()
    pos = p["dates"].get_indexer(dates)
    feats = {}
    for k in (5, 20):
        spy_r = p["spy"][pos] - p["spy"][pos - k]
        feats[f"(added) SPY own {k}d return"] = spy_r
        for j, c in enumerate(tl.COVS):
            feats[f"{c} {k}d return minus SPY"] = (p["cov"][j, pos] - p["cov"][j, pos - k]) - spy_r
    for h in HS:
        ds = Dataset(h)
        s, e = ds.period_positions("TEST")
        assert (ds.index[s:e] == dates).all()
        X = pd.DataFrame(ds.X("core")[s:e], columns=CORE_FEATURES, index=dates)
        for c in CROSS6:
            feats[f"[feature] {c}"] = X[c].to_numpy()
        break
    rng = np.random.default_rng(7)
    B = boot_idx(len(dates), rng)
    rows, sc = [], {}
    for h in HS:
        ps, pf = P(q_spy, a, h), P(q_full, a, h)
        eff = pf - ps
        y = labels(h).reindex(dates).to_numpy()
        ok = ~np.isnan(y)
        sc[h] = (ps, pf)
        rows.append(dict(h=h, item="[effect] mean / sd / share>0", rho=eff.mean(), lo=eff.std(), hi=(eff > 0).mean(), rho_label=np.nan))
        rho_lab = spearmanr(eff[ok], y[ok])[0]
        rows.append(dict(h=h, item="[effect] vs label (up=1)", rho=rho_lab, lo=np.nan, hi=np.nan, rho_label=np.nan))
        rr = rankdata(eff)
        for nm, f in feats.items():
            rf = rankdata(f)
            r0 = np.corrcoef(rr, rf)[0, 1]
            bs = np.array([np.corrcoef(rr[b], rf[b])[0, 1] for b in B])      # rank correlation of the original ranks, block resampled
            lab = np.corrcoef(rankdata(y[ok]), rankdata(f[ok]))[0, 1]
            rows.append(dict(h=h, item=nm, rho=r0, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), rho_label=lab))
    df = pd.DataFrame(rows)
    df.to_csv(OUT("e2_effect_corr"), index=False, float_format="%.4f")
    # helps or hurts: AUC of full vs SPY-only with the paired block bootstrap
    ev = evaluate({"spy_only": {h: sc[h][0] for h in HS}, "full": {h: sc[h][1] for h in HS}}, dates)
    ev.to_csv(OUT("e2_auc"), index=False, float_format="%.4f")
    print(ev[["model", "h", "auc_pooled", "ci_lo", "ci_hi", "d_vs_first", "d_vs_first_lo", "d_vs_first_hi"]].round(3).to_string())
    for h in HS:
        d = df[df.h == h]
        d = d[~d.item.str.startswith("[effect]")].copy()
        d["abs"] = d.rho.abs()
        print(f"--- h={h}: effect row(s)")
        print(df[(df.h == h) & df.item.str.startswith("[effect]")].round(3).to_string())
        print(d.sort_values("abs", ascending=False).head(12)[["item", "rho", "lo", "hi", "rho_label"]].round(2).to_string())


# ------------------------------------------------------------------------------------------ E3
def e3():
    lv = pd.read_csv(OUT("e3_levels"), dtype={"level": str})
    rows = []
    rng = np.random.default_rng(11)
    B = rng.integers(0, R.N3, size=(1000, R.N3))
    for _, r in lv.iterrows():
        nm = r.level
        t = np.load(S.CACHE / f"e3_{nm}_truth.npz")
        y = (t["fut"] > 0).astype(int)
        sc = {"oracle (true x today)": t["oracle"], "in-window OLS": t["ols"]}
        pw = P(np.load(S.CACHE / f"e3_{nm}_with.npz")["q"], t["anchor"], 1)
        pn = P(np.load(S.CACHE / f"e3_{nm}_without.npz")["q"], t["anchor"], 1)
        sc["TimesFM with x"], sc["TimesFM y only"] = pw, pn
        a = {k: auc(y, v) for k, v in sc.items()}
        ab = {k: np.array([auc(y[b], v[b]) for b in B]) for k, v in sc.items()}
        d = ab["TimesFM with x"] - ab["TimesFM y only"]
        rows.append(dict(level=nm, ref_auc=r.ref_auc, rho=r.rho, **{f"auc {k}": v for k, v in a.items()},
                         with_minus_without=a["TimesFM with x"] - a["TimesFM y only"],
                         dlo=np.percentile(d, 2.5), dhi=np.percentile(d, 97.5),
                         ci_with_lo=np.percentile(ab["TimesFM with x"], 2.5), ci_with_hi=np.percentile(ab["TimesFM with x"], 97.5),
                         spearman_dP_with_oracle=spearmanr(pw - pn, t["oracle"])[0],
                         spearman_ols_with_oracle=spearmanr(t["ols"], t["oracle"])[0]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT("e3"), index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print(df.round(3).T.to_string())


# ------------------------------------------------------------------------------------------ E4 / E6
def trailing(lv):
    """lv (n, 256) log prices -> dict of measures."""
    last = lv[:, -1]
    return {"last 1-day return": last - lv[:, -2], "last 5-day return": last - lv[:, -6],
            "last 20-day return": last - lv[:, -21], "price minus 20-day average": last - lv[:, -20:].mean(1),
            "price minus 256-day average": last - lv.mean(1)}


def e4():
    rows = []
    sets = {}
    for fam, nm in (("A", "random walk (iid normal)"), ("B", "random walk (GARCH, volatility clusters)"),
                    ("C", "random walk (shuffled real SPY days)")):
        sets[nm] = (np.load(S.CACHE / f"e4_{fam}_levels.npy"), np.load(S.CACHE / f"e4_{fam}.npz")["q"])
    # real SPY: test days (SPY-only cached) and all cached price_ctx256 dates
    p = tl.panel()
    z = np.load(RES_DIR / "timesfm_cache/price_ctx256.npz")
    cd = pd.DatetimeIndex(z["dates"])
    for nm, dates in (("real SPY, test days 2019-2025", test_dates()), ("real SPY, all cached days", cd)):
        if "test" in nm:
            q = cached_q("timesfm_ctrl_cache/tf_L_TEST.npz", dates)
        else:
            q = z["q"]
        pos = p["dates"].get_indexer(dates)
        ok = pos >= 255
        lv = np.stack([p["spy"][k - 255:k + 1] for k in pos[ok]])
        sets[nm] = (lv, q[ok])
    for nm, (lv, q) in sets.items():
        tr = trailing(lv)
        for h in HS:
            pu = P(q, lv[:, -1], h)
            for k, v in tr.items():
                rows.append(dict(source=nm, h=h, measure=k, n=len(lv), spearman=spearmanr(pu, v)[0],
                                 mean_pup=pu.mean(), pct_up_call=100 * (pu > .5).mean()))
    df = pd.DataFrame(rows)
    df.to_csv(OUT("e4"), index=False, float_format="%.4f")
    print(df.pivot_table(index=["source", "measure"], columns="h", values="spearman", sort=False).round(3).to_string())
    print(df.groupby(["source", "h"])[["mean_pup", "pct_up_call"]].first().round(3).to_string())


def e6():
    rows = []
    for fam, nm in (("A", "iid normal"), ("B", "GARCH"), ("C", "shuffled real days")):
        lv = np.load(S.CACHE / f"e4_{fam}_levels.npy")
        z = np.load(S.CACHE / f"e4_{fam}.npz")
        N, Q = np.sort(z["N"], -1), np.sort(z["Q"], -1)
        a = lv[:, -1]
        T = (Q - N).mean(-1)                                  # (n, 5) add-back per step
        resid = np.abs((Q - N) - T[..., None]).max()
        app = np.abs(T[:, 0]) > 1e-9
        s = T[:, 1] - T[:, 0]
        c = T[:, 0] - s
        r_t = a - app * c
        tr = trailing(lv)
        for h in HS:
            k = h - 1
            offs = (a + app * s * h)[:, None]
            parts = {"official": P(z["q"], a, h),
                     "network only": np.array([tl.p_up(N[i, k], "price", h, r_t[i]) if False else
                                               float(1 - np.interp(r_t[i], N[i, k], tl.QLEVELS, left=0.05, right=0.95))
                                               for i in range(len(a))]),
                     "trend-line slope only": np.array([float(1 - np.interp(a[i], N[i, k] - N[i, k, 4] + offs[i, 0], tl.QLEVELS, left=0.05, right=0.95))
                                                        for i in range(len(a))])}
            parts["official"] = np.array([float(1 - np.interp(a[i], Q[i, k], tl.QLEVELS, left=0.05, right=0.95)) for i in range(len(a))])
            for pn, pu in parts.items():
                for mk, v in tr.items():
                    rows.append(dict(family=nm, h=h, part=pn, measure=mk, spearman=spearmanr(pu, v)[0], mean_pup=pu.mean(),
                                     share_line_applied=app.mean(), max_resid_QminusN_const=resid))
    df = pd.DataFrame(rows)
    df.to_csv(OUT("e6"), index=False, float_format="%.4f")
    print(df.pivot_table(index=["family", "h", "part"], columns="measure", values="spearman", sort=False).round(2).to_string())
    print(df.groupby("family")[["share_line_applied", "max_resid_QminusN_const"]].first())


# ------------------------------------------------------------------------------------------ E5
def e5():
    rows = []
    for bg in ("rw", "real"):
        for k in (5, 20):
            for mv in R.MOVES:
                q = np.load(S.CACHE / f"e5_{bg}_k{k}_m{mv:+d}.npz")["q"]
                # anchor = last level of the rebuilt path; recompute from the stored returns
    world = S.World()
    bgs = R.e5_backgrounds(world)
    for bg, ret in bgs.items():
        for k in (5, 20):
            for mv in R.MOVES:
                r = ret.copy()
                tail = r[:, -k:]
                r[:, -k:] = tail - tail.mean(1, keepdims=True) + np.log1p(mv / 100.0) / k
                lv = S.to_levels(r)
                q = np.load(S.CACHE / f"e5_{bg}_k{k}_m{mv:+d}.npz")["q"]
                for h in HS:
                    pu = P(q, lv[:, -1], h)
                    rows.append(dict(background=bg, last_days=k, move_pct=mv, h=h, mean_pup=pu.mean(), se=pu.std(ddof=1) / np.sqrt(len(pu)),
                                     pct_up_call=100 * (pu > .5).mean(), n=len(pu)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT("e5"), index=False, float_format="%.4f")
    print(df.pivot_table(index=["background", "last_days", "h"], columns="move_pct", values="mean_pup").round(3).to_string())
    plot_e5(df)


def plot_e5(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    col = {1: "#0072B2", 5: "#D55E00"}
    lab = {1: "chance of up TOMORROW", 5: "chance of up in 5 DAYS"}
    for ax, k in zip(axs, (5, 20)):
        for h in (1, 5):
            for bg, ls, al in (("rw", "-", 1.0), ("real", ":", 0.9)):
                d = df[(df.background == bg) & (df.last_days == k) & (df.h == h)].sort_values("move_pct")
                ax.errorbar(d.move_pct, 100 * d.mean_pup, yerr=100 * 1.96 * d.se, color=col[h], ls=ls, lw=2 if bg == "rw" else 1.6,
                            marker="o" if bg == "rw" else "s", ms=5, alpha=al, capsize=2,
                            label=f"{lab[h]} ({'random-walk charts' if bg == 'rw' else 'real SPY history'})")
        ax.axhline(50, color="0.5", lw=1)
        ax.set_title(f"After a move over the last {k} days", loc="left", fontsize=12)
        ax.set_xlabel(f"How much the price moved in the last {k} days (%)")
        ax.set_xticks(R.MOVES)
    axs[0].set_ylabel("TimesFM's chance of up (%)  [average over 1,000 charts]")
    axs[0].legend(fontsize=8, loc="upper right", frameon=False)
    fig.suptitle("TimesFM-3 chance of up vs the last move: the bigger the rise, the lower the chance (lines show 95% range)", x=0.01, ha="left", fontsize=12)
    fig.text(0.01, 0.005, "Same chart with only the last days replaced by the move shown; the older price history is unchanged. 50% = coin flip.", fontsize=8, color="0.3")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out = TF_DIR / "figures" / "timesfm_behaviour_reversal.png"
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5, "e6": e6}[sys.argv[1]]()

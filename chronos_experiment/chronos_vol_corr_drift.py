#!/usr/bin/env python
"""
Chronos-2 extensions: the parts of research question 2 not covered by chronos_study.py.

A. Volatility and correlation targets (the proposal's fallback targets "with clearer regime shifts").
   Same targets, rows, periods, comparators and bootstrap as the teammates' v2 study (experiments/v2_lib.py):
     spy_vol       h = 5   log annualised realised vol of SPY over t+1..t+5
     tlt_vol       h = 5   same for TLT
     spy_tlt_corr  h = 21  realised SPY-TLT return correlation over t+1..t+21 (flipped sign in 2022)
   Question: does zero-shot Chronos-2, which only re-reads recent history, match or beat
     (i) the trained-once HAR model, (ii) the same HAR refit every month, (iii) persistence?
   Plus a descriptive drift analysis: 2019-21 vs 2022-25, and rolling 252-day MSE around the 2022 flip.

B. Drift breakdown of the existing direction results (no new forecasts): Chronos vs the trained-once logistic
   regression and chance, by sub-period and by year.

Run from the project root, in this order:
    python chronos_experiment/chronos_vol_corr_drift.py smoke
    python chronos_experiment/chronos_vol_corr_drift.py validate
    python chronos_experiment/chronos_vol_corr_drift.py select      # freezes one config per target (+ SHA-256)
    python chronos_experiment/chronos_vol_corr_drift.py test        # one-shot 2019-2025 run; refuses to run twice
    python chronos_experiment/chronos_vol_corr_drift.py direction-drift

How a forecast is made (no look-ahead)
--------------------------------------
Each target is the future value of a series that is fully known at the close of t:
    vol:  v[t] = log annualised RMS of returns over t-4..t   ->  v[t+5]  = next-week realised vol
    corr: c[t] = SPY-TLT correlation over t-20..t            ->  c[t+21] = next-month correlation
Chronos-2 gets that series (and optional covariates) up to the close of t, forecasts h steps, and the step-h
forecast mean is the prediction. The identity is asserted against v2_lib's targets for every row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chronos_study as CS  # noqa: E402  (also puts experiments/ on sys.path)
import v2_lib as L  # noqa: E402
from common import metrics as dir_metrics, paired_bootstrap as dir_boot  # noqa: E402

OUT = HERE / "results_ext"
OUT.mkdir(parents=True, exist_ok=True)
VAL_LOG = OUT / "ext_validation_log.csv"
SHORTLIST = OUT / "ext_shortlist.json"
HASHFILE = OUT / "ext_shortlist.sha256"
V2_SHORTLIST = L.RES_DIR / "v2_shortlist.json"

TARGETS = ["spy_vol", "tlt_vol", "spy_tlt_corr"]
VARIANTS = {"uni": "target series only", "cov": "target series + the HAR-type features as past covariates"}
CONTEXTS = [256, 512, 1024, 2048]
FOLDS = ["F1", "F2", "VAL"]
SUBPERIODS = {"2019-2021": ("2019-01-01", "2021-12-31"), "2022-2025": ("2022-01-01", "2025-12-31")}
PERSIST = {"spy_vol": ["persist_5", "persist_20"], "tlt_vol": ["persist_5", "persist_20"],
           "spy_tlt_corr": ["persist_21", "persist_63"]}


# ---------------------------------------------------------------------------
# Data: Chronos inputs aligned with v2_lib's Target rows
# ---------------------------------------------------------------------------
class ExtData:
    def __init__(self, name: str, targets: dict, px: pd.DataFrame):
        self.name = name
        self.tg = targets[name]
        self.h = self.tg.h
        cal = px["SPY"].index
        lvix = np.log(px["^VIX"]["Close"] / 100)
        fs, rs = L.vol_frame(px, "SPY")
        ft, rt = L.vol_frame(px, "TLT")
        if name == "spy_vol":
            series = fs["lrv5"]
            cov = {"lpk1": fs["lpk1"], "lrv22": fs["lrv22"], "lvix": lvix}
        elif name == "tlt_vol":
            series = ft["lrv5"]
            cov = {"lpk1": ft["lpk1"], "lrv22": ft["lrv22"], "lvix": lvix, "spy_lrv22": fs["lrv22"]}
        else:
            r_s, r_b = L._lr(px["SPY"]["Close"]), L._lr(px["TLT"]["Close"])
            series = r_s.rolling(21).corr(r_b)
            cov = {"c63": r_s.rolling(63).corr(r_b), "c252": r_s.rolling(252).corr(r_b), "lvix": lvix,
                   "spy_lrv22": fs["lrv22"], "tlt_lrv22": ft["lrv22"]}
        # the series' value h steps ahead must be exactly the v2 target
        chk = series.shift(-self.h).reindex(self.tg.index).to_numpy(float)
        assert np.allclose(chk, self.tg.y, atol=1e-8), f"{name}: series does not reproduce the v2 target"
        frame = pd.concat([series.rename("__y"), pd.DataFrame(cov)], axis=1).reindex(cal)
        self.first = int(np.flatnonzero(frame.notna().all(axis=1).to_numpy())[0])
        frame = frame.ffill()
        self.y_series = frame["__y"].to_numpy(float)
        self.cov = {c: frame[c].to_numpy(float) for c in cov}
        self.cal_pos = cal.get_indexer(self.tg.index)
        assert (self.cal_pos >= 0).all()

    def make_input(self, t: int, Lctx: int, variant: str) -> dict:
        """Context ending at calendar position t (decision day's close). Nothing after t is used."""
        sl = slice(max(t - Lctx + 1, self.first), t + 1)
        d = {"target": self.y_series[sl]}
        if variant == "cov":
            d["past_covariates"] = {c: v[sl] for c, v in self.cov.items()}
        return d


def build_all():
    px = pd.read_pickle(CS.DATA_DIR / "aligned_prices.pkl")
    T = L.build_targets(px)
    return {n: ExtData(n, T, px) for n in TARGETS}


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------
def _chunk(pipe, chunk, h):
    import torch
    try:
        q, mean = pipe.predict_quantiles(chunk, prediction_length=h, quantile_levels=CS.QLEVELS)
        return list(q), list(mean)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(chunk) == 1:
            raise
        CS.BATCH = max(1, len(chunk) // 2)
        print(f"  GPU out of memory -> batch size now {CS.BATCH}")
        qs, ms = [], []
        for j in range(0, len(chunk), CS.BATCH):
            a, b = _chunk(pipe, chunk[j:j + CS.BATCH], h)
            qs += a
            ms += b
        return qs, ms


def forecast(pipe, inputs, h):
    """Step-h forecast mean and quantiles of the target variate, shapes (n,) and (n, 21)."""
    means, quants = [], []
    i = 0
    while i < len(inputs):
        b = CS.BATCH
        qs, ms = _chunk(pipe, inputs[i:i + b], h)
        i += b
        for q, m in zip(qs, ms):
            q, m = CS._np(q), CS._np(m)
            if q.ndim == 3:
                q = q[0]
            if m.ndim == 2:
                m = m[0]
            assert q.shape == (h, len(CS.QLEVELS)) and m.shape == (h,), f"unexpected shapes {q.shape} {m.shape}"
            means.append(m[h - 1])
            quants.append(q[h - 1])
    return np.array(means), np.vstack(quants)


def run_period(pipe, ed: ExtData, variant: str, Lctx: int, period: str) -> np.ndarray:
    """Chronos predictions for every v2 Target row of `period` (cached on disk)."""
    f = OUT / f"pred_{ed.name}_{variant}_L{Lctx}_{period}.pkl"
    if f.exists():
        return pd.read_pickle(f)["pred"].to_numpy()
    s, e = ed.tg.period(period)
    rows = np.arange(s, e)
    t0 = time.time()
    mean, Q = forecast(pipe, [ed.make_input(int(ed.cal_pos[r]), Lctx, variant) for r in rows], ed.h)
    pred = L._clip(ed.tg, mean)
    df = pd.DataFrame({"y": ed.tg.y[rows], "pred": pred}, index=ed.tg.index[rows])
    for j, tq in enumerate(CS.QLEVELS):
        df[f"q{tq:.2f}"] = Q[:, j]
    pd.to_pickle(df, f)
    print(f"  {ed.name:12s} {variant:3s} L={Lctx:4d} {period:4s}: {len(rows)} forecasts in {time.time() - t0:.0f}s")
    return pred


def v2_shortlist():
    return json.loads(V2_SHORTLIST.read_text())


def comparators(name: str) -> dict:
    """Trained-once HAR (v2 best baseline), the same HAR refit monthly (v2 default slot .5), persistence."""
    info = v2_shortlist()[name]
    refit = [e for e in info["entries"] if e["id"] == f"{name}.5"][0]
    comps = {"har_fixed": info["baselines"][info["best_baseline"]], "har_refit21": refit["spec"]}
    for p in PERSIST[name]:
        comps[p] = info["baselines"][p]
    return comps


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_smoke(args):
    pipe = CS.load_pipeline()
    E = build_all()
    per = []
    for n, ed in E.items():
        s, _ = ed.tg.period("VAL")
        for v in VARIANTS:
            inp = [ed.make_input(int(ed.cal_pos[r]), 1024, v) for r in range(s, s + 8)]
            t0 = time.time()
            mean, Q = forecast(pipe, inp, ed.h)
            dt = (time.time() - t0) / len(inp)
            per.append(dt)
            print(f"{n:12s} {v:3s}: forecasts {np.round(mean, 3)}  actual {np.round(ed.tg.y[s:s + 8], 3)}  "
                  f"{dt:.2f}s each")
    n_val = len(TARGETS) * len(VARIANTS) * len(CONTEXTS) * 1500
    print(f"\nRough runtime: validate ~{n_val * np.mean(per) / 60:.0f} min, test ~"
          f"{3 * 1750 * np.mean(per) / 60:.0f} min. Smoke test passed.")


def cmd_validate(args):
    pipe = CS.load_pipeline()
    E = build_all()
    rows = []
    for n in args.targets:
        ed = E[n]
        comps = comparators(n)
        for fold in FOLDS:
            s, e = ed.tg.period(fold)
            for cname, spec in comps.items():
                p = L.run(ed.tg, spec, fold)
                rows.append({"target": n, "model": cname, "variant": "", "L": 0, "fold": fold,
                             **L.metrics(ed.tg, fold, p)})
            for v in args.variants:
                for Lctx in args.contexts:
                    p = run_period(pipe, ed, v, Lctx, fold)
                    rows.append({"target": n, "model": "chronos", "variant": v, "L": Lctx, "fold": fold,
                                 **L.metrics(ed.tg, fold, p)})
    log = pd.DataFrame(rows)
    if VAL_LOG.exists():  # keep rows from earlier partial runs
        old = pd.read_csv(VAL_LOG).fillna({"variant": ""})
        log = pd.concat([old, log]).drop_duplicates(["target", "model", "variant", "L", "fold"], keep="last")
    log.to_csv(VAL_LOG, index=False)
    ch = log[log.model == "chronos"]
    print(ch.pivot_table(index=["target", "variant", "L"], columns="fold", values="mse").round(4).to_string())
    print(log[log.model != "chronos"].pivot_table(index=["target", "model"], columns="fold", values="mse")
          .round(4).to_string())


def cmd_select(args):
    log = pd.read_csv(VAL_LOG).fillna({"variant": ""})
    sel, tables = {}, []
    for n in TARGETS:
        d = log[log.target == n]
        bb = d[d.model == "har_fixed"].set_index("fold")["mse"]
        ch = d[d.model == "chronos"].copy()
        ch["skill"] = 1 - ch["mse"] / ch["fold"].map(bb)
        tab = ch.groupby(["variant", "L"]).agg(mean_skill=("skill", "mean"), n_folds=("fold", "nunique"),
                                                mean_mse=("mse", "mean")).reset_index()
        val = ch[ch.fold == "VAL"].set_index(["variant", "L"])["skill"].rename("val_skill")
        tab = tab.join(val, on=["variant", "L"])
        # Rule (v2 style, written before validation): beats the trained-once HAR on MSE on VAL and on average
        # over the 3 folds. Rank by mean skill vs the trained-once HAR. Fallback: best mean skill, marked.
        tab["eligible"] = (tab.n_folds == 3) & (tab.val_skill > 0) & (tab.mean_skill > 0)
        tab["target"] = n
        tables.append(tab)
        pool = tab[tab.eligible] if tab.eligible.any() else tab
        best = pool.sort_values("mean_skill", ascending=False).iloc[0]
        # persistence comparator chosen a priori by lower mean validation MSE
        pm = d[d.model.isin(PERSIST[n])].groupby("model")["mse"].mean()
        sel[n] = {"variant": best.variant, "L": int(best.L), "eligible": bool(best.eligible),
                  "val_mean_skill_vs_har_fixed": round(float(best.mean_skill), 4),
                  "val_skill_VAL": round(float(best.val_skill), 4), "persistence": str(pm.idxmin())}
    pd.concat(tables).sort_values(["target", "mean_skill"], ascending=[True, False]) \
        .to_csv(OUT / "ext_selection_table.csv", index=False)
    spec = {"model": CS.MODEL_ID, "selected": sel,
            "confirmatory": "per target: chronos vs har_fixed, vs har_refit21, vs persistence; MSE primary; "
                            "Holm over the 3 comparisons per target and metric",
            "win": "Holm p_mse < 0.05 vs har_fixed AND MSE reduction >= 5% (the v2 'large, real' bar)"}
    SHORTLIST.write_text(json.dumps(spec, indent=2, sort_keys=True))
    HASHFILE.write_text(hashlib.sha256(SHORTLIST.read_bytes()).hexdigest())
    print(json.dumps(sel, indent=2))
    print(f"\nFrozen: {SHORTLIST}\nSHA-256: {HASHFILE.read_text()}\nCommit it before running `test`.")


def cmd_test(args):
    assert SHORTLIST.exists(), "run `select` first"
    assert hashlib.sha256(SHORTLIST.read_bytes()).hexdigest() == HASHFILE.read_text().strip(), \
        "shortlist changed after it was frozen"
    out_c = OUT / "ext_test_comparisons.csv"
    if out_c.exists() and not args.force:
        sys.exit("The test has already been run once. Use --force only if it crashed before finishing.")
    spec = json.loads(SHORTLIST.read_text())["selected"]
    pipe = CS.load_pipeline()
    E = build_all()
    v2p = None
    v2f = L.RES_DIR / "v2_test_predictions.pkl"
    if v2f.exists():
        v2p = pd.read_pickle(v2f)["preds"]
    levels, comps, subs, years, rolls = [], [], [], [], []
    for n in TARGETS:
        ed, sel = E[n], spec[n]
        tg = ed.tg
        s, e = tg.period("TEST")
        rows = np.arange(s, e)
        dates = tg.index[rows]
        P = {"chronos": run_period(pipe, ed, sel["variant"], sel["L"], "TEST")}
        cspecs = comparators(n)
        for cname in ["har_fixed", "har_refit21", sel["persistence"]]:
            P[cname] = L.run(tg, cspecs[cname], "TEST")
        if v2p is not None:  # sanity: our comparators reproduce the teammates' saved test predictions
            info = v2_shortlist()[n]
            for cname, key in [("har_fixed", "base:" + info["best_baseline"]), ("har_refit21", f"{n}.5")]:
                if (n, key) in v2p:
                    assert np.allclose(P[cname], v2p[(n, key)], atol=1e-8), f"{n} {cname} differs from v2"
        for m, p in P.items():
            levels.append({"target": n, "model": m, **L.metrics_rows(tg, rows, p)})
            for yr in sorted(set(dates.year)):
                k = dates.year == yr
                years.append({"target": n, "model": m, "year": yr, **L.metrics_rows(tg, rows[k], p[k])})
        hc = []
        for other in ["har_fixed", "har_refit21", sel["persistence"]]:
            r = L.paired_boot(tg, rows, P["chronos"], P[other])
            hc.append({"target": n, "comparison": f"chronos vs {other}", "other": other,
                       "rel_mse_reduction": r["d_mse"] / np.mean((tg.y[rows] - P[other]) ** 2), **r})
            for sp, (a, b) in SUBPERIODS.items():
                k = (dates >= a) & (dates <= b)
                rr = L.paired_boot(tg, rows[k], P["chronos"][k], P[other][k])
                subs.append({"target": n, "comparison": f"chronos vs {other}", "subperiod": sp, **rr})
        hc = pd.DataFrame(hc)
        for m in ["mse", "r2", "qlike", "ba", "auc"]:
            if f"p_{m}" in hc and hc[f"p_{m}"].notna().any():
                hc[f"holm_{m}"] = L.holm(hc[f"p_{m}"].to_numpy())
        bb = hc[hc.other == "har_fixed"].iloc[0]
        hc["large_real_win"] = (hc.other == "har_fixed") & (bb.holm_mse < 0.05) & (bb.rel_mse_reduction >= 0.05)
        comps.append(hc)
        R = pd.DataFrame({m: (tg.y[rows] - p) ** 2 for m, p in P.items()}, index=dates).rolling(252).mean()
        R["target"] = n
        rolls.append(R.dropna(subset=["chronos"]))
    pd.DataFrame(levels).to_csv(OUT / "ext_test_levels.csv", index=False)
    pd.DataFrame(years).to_csv(OUT / "ext_test_by_year.csv", index=False)
    pd.DataFrame(subs).to_csv(OUT / "ext_test_subperiods.csv", index=False)
    rolls_df = pd.concat(rolls)
    rolls_df.to_csv(OUT / "ext_test_rolling252_mse.csv")
    comps = pd.concat(comps)
    comps.to_csv(out_c, index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(levels).round(4).to_string())
    print(comps[["target", "comparison", "d_mse", "d_mse_lo", "d_mse_hi", "rel_mse_reduction", "holm_mse",
                 "d_auc", "large_real_win"]].round(4).to_string())
    plot_rolling(rolls_df, spec)


def plot_rolling(R: pd.DataFrame, spec: dict):
    """Rolling 252-day MSE on the test period: Chronos vs trained-once HAR, monthly-refit HAR, persistence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    style = {"chronos": ("#2a78d6", "-", "Chronos-2 (zero-shot)"),
             "har_fixed": ("#eb6834", "--", "HAR, trained once"),
             "har_refit21": ("#1baf7a", "-.", "HAR, refit monthly")}
    pcol = ("#eda100", ":", "persistence")
    titles = {"spy_vol": "SPY next-week volatility (log RV)", "tlt_vol": "TLT next-week volatility (log RV)",
              "spy_tlt_corr": "SPY–TLT next-month correlation"}
    fig, axes = plt.subplots(len(TARGETS), 1, figsize=(10, 3.1 * len(TARGETS)), sharex=True)
    for ax, n in zip(axes, TARGETS):
        d = R[R.target == n].drop(columns="target").dropna(axis=1, how="all")
        ends = []
        for col in d.columns:
            c, ls, lab = style.get(col, (*pcol[:2], f"{pcol[2]} ({col.replace('persist_', '')}-day)"))
            ax.plot(d.index, d[col], color=c, ls=ls, lw=2, label=lab)
            ends.append([d[col].iloc[-1], lab])
        lo_y, hi_y = ax.get_ylim()
        gap = 0.09 * (hi_y - lo_y)
        ends.sort()
        for k in range(1, len(ends)):
            ends[k][0] = max(ends[k][0], ends[k - 1][0] + gap)
        for yv, lab in ends:
            ax.annotate(lab, (d.index[-1], yv), xytext=(8, 0), textcoords="offset points", fontsize=8,
                        color=ink, va="center", annotation_clip=False)
        ax.axvline(pd.Timestamp("2022-01-01"), color=ink2, lw=1, ls=(0, (4, 3)))
        ax.text(pd.Timestamp("2022-01-20"), hi_y, "2022", color=ink2, fontsize=8, va="top")
        sel = spec[n]
        ax.set_title(f"{titles[n]}  ·  Chronos config: {sel['variant']}, context {sel['L']}", loc="left",
                     fontsize=10, color=ink)
        ax.set_ylabel("rolling 252-day MSE", color=ink2, fontsize=9)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for sname in ("top", "right"):
            ax.spines[sname].set_visible(False)
        for sname in ("left", "bottom"):
            ax.spines[sname].set_color(grid)
        ax.tick_params(colors=ink2, labelsize=8)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, ls=ls, lw=2) for c, ls, _ in style.values()] + \
              [Line2D([], [], color=pcol[0], ls=pcol[1], lw=2)]
    labels = [lab for *_, lab in style.values()] + ["persistence (window chosen on validation)"]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0))
    fig.suptitle("Does Chronos-2 adapt without retraining? Rolling test-period error (lower is better)",
                 x=0.01, ha="left", fontsize=12, color=ink)
    fig.subplots_adjust(left=0.08, right=0.80, top=0.93, bottom=0.07, hspace=0.35)
    out = OUT / "ext_rolling_mse.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    print(f"figure saved: {out}")


def cmd_direction_drift(args):
    """Descriptive drift breakdown of the preregistered direction picks (no new forecasts)."""
    spec = json.loads(CS.SHORTLIST.read_text())["selected"]
    rows, years = [], []
    for hk, sel in spec.items():
        h = int(hk)
        f = CS.OUT / f"pred_h{h}_{sel['variant']}_L{sel['L']}_TEST_s1.pkl"
        assert f.exists(), f"missing {f}; run chronos_study.py test first"
        c = pd.read_pickle(f)
        lr = CS.notebook_logreg_fixed(h)
        idx = c.index.intersection(lr.index)
        c, lr = c.loc[idx], lr.loc[idx]
        chance = c.copy()
        chance["proba"], chance["pred"] = c.thr, 1
        for sp, (a, b) in {"2019-2025": ("2019-01-01", "2025-12-31"), **SUBPERIODS}.items():
            k = (idx >= a) & (idx <= b)
            for other, od in [("chance", chance), ("logreg_fixed", lr)]:
                r = dir_boot(c[k], od[k], h)
                rows.append({"h": h, "subperiod": sp, "comparison": f"chronos vs {other}",
                             "chronos_auc": dir_metrics(c[k])["roc_auc"], "other_auc": dir_metrics(od[k])["roc_auc"],
                             **r})
        for name, d in [("chronos", c), ("logreg_fixed", lr)]:
            for yr, g in d.groupby(d.index.year):
                years.append({"h": h, "model": name, "year": yr, **dir_metrics(g)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "direction_drift_subperiods.csv", index=False)
    Y = pd.DataFrame(years)
    Y.to_csv(OUT / "direction_drift_by_year.csv", index=False)
    pd.set_option("display.width", 200)
    print(t[["h", "subperiod", "comparison", "chronos_auc", "other_auc", "d_roc_auc", "d_roc_auc_lo",
             "d_roc_auc_hi", "p_roc_auc"]].round(3).to_string())
    plot_direction_years(Y)


def plot_direction_years(Y: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    style = {"chronos": ("#2a78d6", "o", "Chronos-2 (preregistered pick)"),
             "logreg_fixed": ("#eb6834", "s", "logistic regression, trained once")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, h in zip(axes, [1, 5]):
        ax.axhline(0.5, color=ink2, lw=1, ls=(0, (4, 3)))
        ax.axvspan(2021.5, 2022.5, color=grid, alpha=0.6, lw=0)
        d = Y[Y.h == h]
        for m, (c, mk, lab) in style.items():
            g = d[d.model == m].sort_values("year")
            ax.plot(g.year, g.roc_auc, color=c, marker=mk, ms=8, lw=2, mec="#fcfcfb", mew=2, label=lab)
        ax.set_title(f"{h}-day horizon", loc="left", fontsize=11, color=ink)
        ax.set_xticks(sorted(d.year.unique()))
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for sname in ("top", "right"):
            ax.spines[sname].set_visible(False)
        for sname in ("left", "bottom"):
            ax.spines[sname].set_color(grid)
        ax.tick_params(colors=ink2, labelsize=8)
        ax.text(2022, ax.get_ylim()[1], "2022", ha="center", va="top", fontsize=8, color=ink2)
    axes[0].set_ylabel("ROC-AUC within year", color=ink2, fontsize=9)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("SPY direction: AUC by test year (dashed line = chance)", x=0.01, ha="left", fontsize=12,
                 color=ink)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.18, wspace=0.12)
    out = OUT / "direction_auc_by_year.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    print(f"figure saved: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=64, help="forecasts per GPU batch (put before the command)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    v = sub.add_parser("validate")
    v.add_argument("--targets", nargs="+", default=TARGETS, choices=TARGETS)
    v.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    v.add_argument("--contexts", type=int, nargs="+", default=CONTEXTS)
    sub.add_parser("select")
    t = sub.add_parser("test")
    t.add_argument("--force", action="store_true")
    sub.add_parser("direction-drift")
    args = ap.parse_args()
    CS.BATCH = args.batch
    {"smoke": cmd_smoke, "validate": cmd_validate, "select": cmd_select, "test": cmd_test,
     "direction-drift": cmd_direction_drift}[args.cmd](args)


if __name__ == "__main__":
    main()

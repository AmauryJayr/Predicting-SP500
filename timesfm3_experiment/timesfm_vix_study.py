"""Does the VIX help the frozen, zero-shot TimesFM-3 predict SPY direction?

Usage:  python timesfm_vix_study.py select    -> choices (selection folds only) + pre-registration (hash-locked)
        python timesfm_vix_study.py test      -> ONE-SHOT 2019-2025 run; refuses if the choices hash differs
        python timesfm_vix_study.py explore   -> EXPLORATORY (after the test): static TimesFM by VIX tercile
        python timesfm_vix_study.py report    -> prints the tables and draws slides/fig/vix_methods.png
        python timesfm_vix_study.py example   -> numbers of the worked example in TIMESFM_VIX.md (h=1, one test day)

TimesFM is never refitted. V1-V3 act on its cached daily score (see timesfm_vix_lib.py); V4 reads the forecasts of
the ablation runs made by timesfm_vix_ablation.py (a new cache folder); V5 is a volatility-managed strategy.
Nothing in this file calls the model.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
import hashlib
import io
import json
import re
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import timesfm_drift_lib as dl
import timesfm_lib as tl
import timesfm_vix_lib as vl
from tf_paths import RES_DIR, TF_DIR
from common import PERIODS, metrics, paired_bootstrap
from run_test import chance, holm

HORIZONS = (1, 5)
FOLDS = ("F2", "VAL")
SEL_END = "2018-12-31"
FAMILIES = ("recal", "select", "stack")
FAM_NAME = {"recal": "V1 VIX-triggered recalibration", "select": "V2 VIX-regime selection", "stack": "V3 VIX-conditioned stacking"}
ISO_NAME = {"recal": "scheduled refits (every 63 rows, same window)", "select": "regime-blind selection (same lookback)",
            "stack": "same stack without the VIX terms"}
COST, CAPS = 5, (1.5, 2.0)
CHOICES = RES_DIR / "timesfm_vix_choices.json"
PREREG = TF_DIR / "timesfm_vix_preregistration.md"
VLOG = RES_DIR / "timesfm_vix_validation_log.csv"
STAMPS = RES_DIR / "timesfm_vix_timestamps.csv"
ABL_LOG = RES_DIR / "timesfm_vix_ablation_run.log"
TEST_FILES = [RES_DIR / f"timesfm_vix_test_{x}" for x in
              ("levels.csv", "comparisons.csv", "by_year.csv", "diagnostics.csv", "predictions.pkl", "v4.csv",
               "strategy.csv")]
OLD_PREDS = RES_DIR / "timesfm_test_predictions.pkl"
OLD_KEY = {1: "tfm.h1", 5: "tfm.h5"}
N_ROWS = {1: 1759, 5: 1755}
V4_RUNS = {"novix": ("multi", "SPY + 14 covariates, no ^VIX"), "vixonly": ("multi", "SPY + ^VIX only"),
           "uni": ("price", "SPY alone")}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stamp(event, utc=None, **kw):
    row = pd.DataFrame([{"event": event, "utc": utc or now(), "sha256": kw.get("sha256", "")}])
    row.to_csv(STAMPS, mode="a", header=not STAMPS.exists(), index=False)


def v4_frame(ds, name, h, period):
    """The official frame pipeline (`tl.pred_frame`) applied to the ablation cache `name`."""
    variant = V4_RUNS[name][0]
    tl.CACHE = tl.RES_DIR / "timesfm_vix_cache" / name
    try:
        return tl.pred_frame(ds, variant, 256, h, period)
    finally:
        tl.CACHE = tl.RES_DIR / "timesfm_cache"


def check_ablation_caches():
    """All ablation caches complete and on exactly the official dates (and the official cache is untouched)."""
    off, _, _ = tl.read_cache("multi", 256)
    for name, (variant, _) in V4_RUNS.items():
        tl.CACHE = tl.RES_DIR / "timesfm_vix_cache" / name
        try:
            d, q, _ = tl.read_cache(variant, 256)
        finally:
            tl.CACHE = tl.RES_DIR / "timesfm_cache"
        need = off if name != "uni" else off[off >= pd.Timestamp("2019-01-01")]
        assert len(d) >= len(need) and set(need) <= set(d), f"ablation cache {name} incomplete"
        assert np.isfinite(q).all()


# ---------------------------------------------------------------------------
# select: configs chosen on the selection folds only
# ---------------------------------------------------------------------------
def select(*args):
    assert not PREREG.exists() and not CHOICES.exists(), "already locked; the choices are final"
    assert not any(f.exists() for f in TEST_FILES), "test outputs exist: selection must come first"
    check_ablation_caches()
    log = ABL_LOG.read_text().strip().splitlines()
    assert re.match(r"\d{4}-\d\d-\d\dT", log[0]) and re.match(r"\d{4}-\d\d-\d\dT", log[-1]), "ablation run unfinished"
    stamp("ablation_inference_start", utc=log[0])
    stamp("ablation_inference_end", utc=log[-1])
    stamp("select_start")
    G = vl.grid()
    rows, info = [], {}
    for h in HORIZONS:
        inp = dl.Inp(h, upto=SEL_END)                   # no TimesFM score after 2018-12-31 is even loaded
        inp._X = vl.expert_table(inp, ["F1", "F2", "VAL"])
        for fold in FOLDS:
            st, lr = metrics(inp.static(fold)), metrics(inp.fixed_lr(fold))
            rows.append({"horizon": h, "family": "static", "cfg": "static", "fold": fold, "spec": "", **st})
            rows.append({"horizon": h, "family": "fixed_lr", "cfg": "fixed_lr", "fold": fold, "spec": "", **lr})
            for fam in FAMILIES:
                for i, spec in enumerate(G[fam]):
                    f = vl.run(inp, fam, spec, fold)
                    m = metrics(f)
                    rows.append({"horizon": h, "family": fam, "cfg": f"{fam}-{i}", "fold": fold,
                                 "spec": json.dumps(spec, sort_keys=True),
                                 "d_ba": m["balanced_acc"] - st["balanced_acc"], "d_auc": m["roc_auc"] - st["roc_auc"],
                                 "d_brier": st["brier"] - m["brier"], "n_refits": f.attrs.get("n_refits", np.nan),
                                 "neg_slope_share": f.attrs.get("neg_slope_share", np.nan), **m})
            # V4 on the folds (descriptive, no choice depends on it): official vs the ablation runs
            ds = inp.ds
            for name in V4_RUNS:
                m = metrics(v4_frame(ds, name, h, fold))
                rows.append({"horizon": h, "family": "v4", "cfg": name, "fold": fold, "spec": V4_RUNS[name][1],
                             "d_ba": m["balanced_acc"] - st["balanced_acc"], "d_auc": m["roc_auc"] - st["roc_auc"], **m})
    V = pd.DataFrame(rows)

    # V5: cap per variance signal, on F2 and VAL only (Sharpe of managed minus buy-and-hold at 5 bps)
    D = vl.daily_frame()
    srows = []
    for sig in vl.SIGNALS:
        for cap in CAPS:
            for fold in FOLDS:
                a, b = PERIODS[fold]
                r = vl.vol_managed(D, a, b, sig, cap, COST)
                srows.append({"family": "v5", "cfg": f"{sig}-{cap}", "signal": sig, "cap": cap, "fold": fold,
                              "sharpe_managed": vl.sharpe(r["managed"]), "sharpe_bh": vl.sharpe(r["bh"]),
                              "d_sharpe": vl.sharpe(r["managed"]) - vl.sharpe(r["bh"]), "mean_weight": float(r["w"].mean())})
    S = pd.DataFrame(srows)
    pd.concat([V, S], ignore_index=True).to_csv(VLOG, index=False, float_format="%.5f")

    choices = {"rule": "V1-V3, per horizon and family: eligible if delta BA > 0 and delta AUC > 0 vs the static TimesFM "
                       "frame in BOTH folds F2 and VAL; best = highest mean(delta BA) + mean(delta AUC); if none is "
                       "eligible the best-ranked config is still taken and flagged. V5: per signal, the cap with the "
                       "highest mean over F2 and VAL of Sharpe(managed) - Sharpe(buy-and-hold) at 5 bps. V4 has no "
                       "tunable choice.",
               "folds": list(FOLDS), "variant": dl.VARIANT, "ctx": dl.CTX, "chosen": {}, "v5": {}}
    for h in HORIZONS:
        for fam in FAMILIES:
            g = V[(V.horizon == h) & (V.family == fam)]
            t = g.groupby("cfg").agg(d_ba=("d_ba", "mean"), d_auc=("d_auc", "mean"),
                                     ok=("d_ba", lambda s: bool((s > 0).all())),
                                     ok_auc=("d_auc", lambda s: bool((s > 0).all())),
                                     spec=("spec", "first")).reset_index()
            t["eligible"] = t["ok"] & t["ok_auc"]
            t["score"] = t["d_ba"] + t["d_auc"]
            t = t.sort_values(["eligible", "score", "cfg"], ascending=[False, False, True])
            r = t.iloc[0]
            choices["chosen"][f"h{h}.{fam}"] = {"horizon": h, "family": fam, "cfg": r["cfg"],
                                                "spec": json.loads(r["spec"]), "eligible": bool(r["eligible"]),
                                                "mean_d_ba": round(float(r["d_ba"]), 4),
                                                "mean_d_auc": round(float(r["d_auc"]), 4),
                                                "n_eligible": int(t["eligible"].sum()), "n_configs": len(t)}
    for sig in vl.SIGNALS:
        m = S[S.signal == sig].groupby("cap").d_sharpe.mean()
        choices["v5"][sig] = {"cap": float(m.idxmax()), "mean_d_sharpe_F2_VAL": {str(k): round(float(v), 4) for k, v in m.items()}}
    CHOICES.write_text(json.dumps(choices, indent=2, sort_keys=True))
    digest = hashlib.sha256(CHOICES.read_bytes()).hexdigest()
    stamp("choices_written", sha256=digest)
    PREREG.write_text(prereg_text(V, choices, digest), encoding="utf-8")
    stamp("prereg_locked", sha256=digest)
    print(PREREG.read_text(encoding="utf-8"))


def spec_str(fam, s):
    if fam == "recal":
        return (f"{'Platt (slope + offset)' if s['cal'] == 'platt' else 'offset only'} on the last {s['N']} known "
                f"labels; refit when the VIX rank crosses {s['cut']} (min gap {vl.MIN_GAP} rows)")
    if fam == "select":
        return f"{s['split']} split of the VIX rank, lookback {s['lookback'] or 'all since the first expert row'}"
    w = {"fixed": "fixed (fit once)", "expanding": f"expanding, refit every {s.get('R')}",
         "rolling": f"rolling {s.get('W')}, refit every {s.get('R')}"}[s["mode"]]
    return f"LR C={s['C']} on [logit score, VIX rank, score x (rank - 0.5)]; {w}"


def prereg_text(V, ch, digest):
    L = ["# Does the VIX help TimesFM-3? Pre-registration", "",
         f"Choices `results/timesfm_vix_choices.json`  SHA-256: `{digest}`", "",
         f"Locked at (UTC): `{now()}`. `timesfm_vix_study.py test` refuses to run if the hash differs, and the "
         "timestamps of every step are in `results/timesfm_vix_timestamps.csv` "
         "(`ablation_inference_*` < `select_start` < `choices_written` < `prereg_locked` < `test_start`). Written after all "
         "selection runs and **before any 2019-2025 metric of a VIX method was computed**: `select` loads no TimesFM score "
         f"after {SEL_END} (`Inp(h, upto=...)`) and evaluates only the folds {', '.join(FOLDS)}. The V4 ablation "
         "forecasts for 2019-2025 were generated before the lock (model inference only, scores and metrics were not "
         "looked at); V4 appears in the selection log only on F2 and VAL, descriptively.", "",
         "## What is tested", "",
         "The question: does the VIX help TimesFM-3, frozen and zero-shot, predict SPY direction? It mirrors the team's "
         "VIX study (`experiments/vix_study.py` on master, applied to the logistic regression) on the TimesFM score "
         f"z = logit(P(up)) of the official setup (variant `{dl.VARIANT}`, context {dl.CTX}). The regime variable is "
         "`vix_pct_rank_252d`, the 1-year percentile rank of the VIX close, known at the close of day t. Every "
         "label-fitted step uses only labels with `label_end <= decision date` (`Dataset.n_known`, asserted); "
         "thr = positive rate of the training window, pred = proba >= thr.", "",
         "1. **V1 VIX-triggered recalibration** (`recal`): the calibration layer on z (Platt, or offset only as the "
         "official frame) is refit on the last N known labels only when 1[rank >= cut] flips, at least 21 rows apart. "
         "Isolating comparison: the same layer on scheduled refits every 63 rows with the same window.",
         "2. **V2 VIX-regime expert selection** (`select`): experts {TimesFM (offset-only calibration, expanding, refit "
         "every 63 rows), the team's fixed LR (C=0.003, 23 core features, trained once per period F1, F2, VAL, TEST), "
         "always-up}. Each day, take the expert with the lowest Brier on past resolved predictions made in the same VIX "
         "regime (median or tercile split of the rank; lookback 504 rows or all; at least 30 same-regime rows, else all "
         "rows). Isolating comparison: the same selection ignoring the regime. The expert history starts at the first "
         f"row with {dl.MIN_TRAIN} known scored labels (about Jan 2014).",
         "3. **V3 VIX-conditioned stacking** (`stack`): logistic regression (standardised, C 0.03 or 0.3) on "
         "[z, VIX rank, z x (rank - 0.5)], fixed, expanding (refit 21) or rolling 252 / 504 (refit 21). Isolating "
         "comparison: the same stack on z alone (`timesfm_drift_lib.run_stack`, feats z).",
         "4. **V4 input ablation** (no outer layer): the official `multi` forecasts (SPY + 15 covariates) minus the same "
         "run with ^VIX removed (14 covariates), context 256, default model flags, batch 16 / 8 threads, over 2013-2025 "
         "(new cache `results/timesfm_vix_cache/novix/`; the existing caches are only read). Reproduction check before "
         "use: the official `multi` re-run on 64 random 2019-2025 dates and `price` on 16 dates matches the existing "
         "caches exactly (max abs diff 0; `results/timesfm_vix_repro.csv`). Both runs get exactly the official calibration "
         "(offset fitted once on 2013-2018, threshold = base rate). Secondary: SPY + ^VIX only versus SPY alone (the "
         "`price` variant; its 2019-2025 forecasts are new, its 2013-2018 forecasts are the existing ones).",
         "5. **V5 volatility-managed exposure** (secondary, the team's M3): weight w_t = min(cap, c / var_hat_t) decided at "
         "the close of t, held open t+1 to open t+2; c scales the managed return to the buy-and-hold volatility on all "
         f"data from {vl.HIST_START} to the day before the period (the first TimesFM forecast is on that date, so every "
         "signal uses the same history). Variance signals: the squared TimesFM spread at h=1 "
         "((q90 - q10) / (2 x 1.2816))^2, the VIX ((VIX/100)^2 / 252) and 20-day realised variance. 5 bps per unit of "
         "turnover (and 0 bps). The cap, 1.5 or 2.0, is chosen per signal on F2 and VAL only (below).", "",
         "## Selection rule (selection folds only)", "",
         f"Folds {', '.join(FOLDS)} (F2 2015-16, VAL 2017-18), as in the drift study: the cached TimesFM forecasts start on "
         "2013-01-02, so the team's fold F1 has no earlier scores to calibrate on. The counterpart of every config in a "
         "fold is the *static official* TimesFM frame of that fold. Rule per horizon and family (V1-V3): eligible if "
         "delta BA > 0 and delta AUC > 0 in both folds; take the eligible config with the highest mean(delta BA) + "
         "mean(delta AUC); if none is eligible the best-ranked config is taken anyway and flagged (`eligible` below). "
         "Grid: V1 8 configs (Platt / offset x window 252, 504 x cut 0.8, 0.9), V2 4 (median / tercile x lookback 504 / "
         "all), V3 8 (C 0.03, 0.3 x fixed, expanding, rolling 252, rolling 504). All configs are in "
         "`results/timesfm_vix_validation_log.csv`.", "",
         "## Chosen configs", "",
         "| h | method | config | spec | eligible | mean dBA | mean dAUC | eligible / configs |", "|---|---|---|---|---|---|---|---|"]
    for k, c in ch["chosen"].items():
        L.append(f"| {c['horizon']} | {FAM_NAME[c['family']]} | {c['cfg']} | {spec_str(c['family'], c['spec'])} | "
                 f"{c['eligible']} | {c['mean_d_ba']:+.4f} | {c['mean_d_auc']:+.4f} | {c['n_eligible']} / {c['n_configs']} |")
    L += ["", "V5 caps (mean over F2 and VAL of Sharpe(managed) - Sharpe(buy-and-hold) at 5 bps): " +
          "; ".join(f"{s}: cap {v['cap']} ({', '.join(f'{k}: {x:+.3f}' for k, x in v['mean_d_sharpe_F2_VAL'].items())})"
                     for s, v in ch["v5"].items()) + ".", "",
          "## Test (2019-2025, run once)", "",
          "Exactly the TimesFM study's rows: `Dataset(h)` TEST rows, 1,759 for h=1 and 1,755 for h=5 (asserted; the "
          "static frame is checked equal to the saved `results/timesfm_test_predictions.pkl`, and so is the fixed LR). "
          "The test period has no effect on any choice; no config is changed after the numbers are seen, anything later "
          "is exploratory and labelled so.", "",
          "**Metrics:** ROC-AUC, balanced accuracy (BA), Brier. **Inference:** paired moving-block bootstrap "
          "(`common.paired_bootstrap`, 2000 draws, block length max(20, 3h), seed 42) of method minus counterpart, 95% "
          "percentile CI, one-sided p = P(delta* <= 0). Holm within each horizon.", "",
          "**Primary family** (V1-V3): each of the 3 methods versus (a) the **static official TimesFM** frame and (b) the "
          "team's **fixed LR**, on BA and AUC: 12 tests per horizon, Holm over those 12. **Isolating family** (V1-V3): "
          "each method versus its VIX-free twin (scheduled refits / regime-blind selection / stack without VIX terms), BA "
          "and AUC: 6 tests per horizon, Holm over those 6. **V4**: official `multi` minus no-VIX on BA and AUC, Holm "
          "over those 2 per horizon (positive = the VIX in the input helps); secondary: SPY + VIX only minus SPY alone, "
          "Holm over 2. **V5**: Sharpe(managed) - Sharpe(buy-and-hold) at 5 bps for each signal (Holm over the 3 signals) "
          "and the TimesFM-spread signal minus the VIX signal, and minus rv20, directly (paired moving-block bootstrap of "
          "the Sharpe difference, block 20, 2000 draws); reported at 5 and 0 bps, max drawdown included. Descriptive only "
          "(no Holm): each method versus chance.", "",
          "## Win rule", "",
          "* A VIX method (V1-V3) **helps TimesFM** at a horizon if its Holm-adjusted p < 0.05 against the static TimesFM "
          "frame on AUC or on BA, with a non-negative point estimate on the other.",
          "* It is **useful** only if, in addition, (i) it is not significantly worse than the fixed LR (the 95% CI of "
          "method minus fixed LR does not lie entirely below 0 on AUC or on BA) and (ii) the isolating comparison shows "
          "the gain comes from the VIX: Holm-adjusted p < 0.05 against the VIX-free twin on AUC or on BA, with a "
          "positive point estimate on the metric concerned.",
          "* **The VIX in the input helps** (V4) if Holm-adjusted p < 0.05 on AUC or on BA for official minus no-VIX.",
          "* V5 is secondary and has no win rule; it is reported as the Sharpe table. Both horizons are reported, win or "
          "not. If no config is eligible on selection, the least-bad one is used and flagged (see the table).", "",
          "## Caveats", "",
          "* **The test period is not a pristine hold-out for this question.** TimesFM's 2019-2025 results were already "
          "known (AUC 0.482 at h=1, 0.459 at h=5), so were the drift study's results and the team's VIX study's results "
          "on the logistic regression, and the team's fixed LR is known to be weakest in high-VIX periods (h=5 AUC 0.489 "
          "in high VIX vs 0.536 in mid VIX). The question was asked knowing this. This pre-registration locks the "
          "*configs* before their own test metrics exist; it cannot undo that knowledge.",
          "* Possible pretraining leakage of TimesFM-3.0 on SPY's public history (as in the TimesFM study); it is shared "
          "by every method and by the static frame.",
          "* Only two selection folds (F2, VAL) and a small grid; selection on two folds is noisy, and for some families "
          "no config may be eligible (flagged above).",
          "* The TimesFM expert of V2 is the walk-forward offset-calibrated score, not the once-fitted static frame, "
          "so that its out-of-sample history exists from 2014; V2 is compared with the official static frame and, "
          "descriptively, with that expert alone.",
          "* V4 changes the model input: the no-VIX forecasts are a different forecast series, so the comparison shows "
          "what the covariate does inside the model, not a general statement about the VIX.",
          "* The score approximates the project's label (Open[t+1] to Close[t+h]); the overnight gap is not modelled.",
          "* The VIX methods cannot add information TimesFM does not contain, except through the VIX itself and, for "
          "V2, the other experts."]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# test: one shot
# ---------------------------------------------------------------------------
def _boot(a, b, h):
    return paired_bootstrap(a, b, h)


def test(*args):
    assert not any(f.exists() for f in TEST_FILES), "the test was already run once; it is not repeated"
    txt = PREREG.read_text(encoding="utf-8")
    want = re.search(r"SHA-256: `([0-9a-f]{64})`", txt).group(1)
    assert hashlib.sha256(CHOICES.read_bytes()).hexdigest() == want, "choices changed after pre-registration"
    assert CHOICES.stat().st_mtime <= PREREG.stat().st_mtime, "choices written after the pre-registration"
    st = pd.read_csv(STAMPS)
    assert list(st.event[:5]) == ["ablation_inference_start", "ablation_inference_end", "select_start",
                                  "choices_written", "prereg_locked"]
    CH = json.loads(CHOICES.read_text())
    ch = CH["chosen"]
    stamp("test_start", sha256=want)
    print(f"one-shot test, choices {want[:12]}...", flush=True)
    old = pd.read_pickle(OLD_PREDS)

    P, lvl, comps, years, diag, v4 = {}, [], [], [], [], []
    for h in HORIZONS:
        inp = dl.Inp(h)
        ds = inp.ds
        inp._X = vl.expert_table(inp, ["F1", "F2", "VAL", "TEST"])
        s, e = ds.period_positions("TEST")
        assert e - s == N_ROWS[h], (h, e - s)
        a_old, g_old = old[(OLD_KEY[h], "timesfm")], old[(OLD_KEY[h], "fixed_lr")]
        static = inp.static("TEST")
        assert static.index.equals(ds.index[s:e]) and a_old.index.equals(static.index)
        assert np.allclose(static["proba"], a_old["proba"]) and (static["pred"].to_numpy() == a_old["pred"].to_numpy()).all()
        assert g_old.index.equals(static.index) and (g_old["y"].to_numpy() == ds.y[s:e]).all()
        assert np.allclose(inp.fixed_lr("TEST")["proba"], g_old["proba"])
        frames = {"static": static, "fixed_lr": g_old, "chance": chance(ds)}
        iso = {}
        for fam in FAMILIES:
            c = ch[f"h{h}.{fam}"]
            frames[fam] = vl.run(inp, fam, c["spec"], "TEST")
            iso[fam] = vl.isolating(inp, fam, c["spec"], "TEST")
            frames[f"{fam}_iso"] = iso[fam]
            for f in (frames[fam], iso[fam]):
                assert f.index.equals(static.index) and (f["y"].to_numpy() == static["y"].to_numpy()).all()
            diag.append({"horizon": h, "family": fam, "cfg": c["cfg"], **frames[fam].attrs,
                         **{f"iso_{k}": v for k, v in iso[fam].attrs.items() if k in ("n_refits", "chosen")}})
        X = inp._X                                                   # the TimesFM expert alone (descriptive reference)
        k = s - X["s0"]
        frames["tfm_expert"] = inp.frame(s, e, X["P"][k:k + e - s, 0], X["T"][k:k + e - s, 0])
        # V4 frames
        for name in V4_RUNS:
            frames[f"v4_{name}"] = v4_frame(ds, name, h, "TEST")
            assert frames[f"v4_{name}"].index.equals(static.index)
        P[h] = frames
        for role, f in frames.items():
            m = metrics(f)
            lvl.append({"horizon": h, "role": role, "cfg": ch.get(f"h{h}.{role}", {}).get("cfg", ""), **m})
            for yr, gg in f.groupby(f.index.year):
                mm = metrics(gg)
                years.append({"horizon": h, "role": role, "year": int(yr), **{k_: mm[k_] for k_ in ("balanced_acc", "roc_auc", "brier", "n")}})
        jobs = []
        for fam in FAMILIES:
            jobs += [(fam, fam, "static", "primary"), (fam, fam, "fixed_lr", "primary"), (fam, fam, f"{fam}_iso", "isolating"),
                     (fam, fam, "chance", "descriptive")]
        jobs += [("select", "select", "tfm_expert", "descriptive")]
        jobs += [("v4", "static", "v4_novix", "v4"), ("v4_secondary", "v4_vixonly", "v4_uni", "v4_secondary")]
        res = Parallel(n_jobs=4)(delayed(_boot)(frames[a], frames[b], h) for _, a, b, _ in jobs)
        for (fam, a, b, famtype), r in zip(jobs, res):
            comps.append({"horizon": h, "method": fam, "a": a, "b": b,
                          "cfg": ch.get(f"h{h}.{fam}", {}).get("cfg", ""), "family": famtype, **r})
        print(f"h={h} done", flush=True)

    C = pd.DataFrame(comps)
    for h in HORIZONS:
        for famtype in ("primary", "isolating", "v4", "v4_secondary"):
            sel = (C.horizon == h) & (C.family == famtype)
            adj = holm(np.concatenate([C.loc[sel, "p_balanced_acc"], C.loc[sel, "p_roc_auc"]]))
            n = int(sel.sum())
            C.loc[sel, "holm_balanced_acc"], C.loc[sel, "holm_roc_auc"] = adj[:n], adj[n:]

    # V5
    D = vl.daily_frame()
    a, b = PERIODS["TEST"]
    srows, R = [], {}
    for cost in (COST, 0):
        for sig in vl.SIGNALS:
            cap = CH["v5"][sig]["cap"]
            r = vl.vol_managed(D, a, b, sig, cap, cost)
            R[(cost, sig)] = r
            for kind in ("managed", "bh"):
                srows.append({"cost_bps": cost, "signal": sig, "cap": cap, "strategy": kind, "n": len(r),
                              **vl.strat_stats(r[kind]), "mean_weight": float(r["w"].mean()) if kind == "managed" else 1.0})
            srows.append({"cost_bps": cost, "signal": sig, "cap": cap, "strategy": "managed - bh", "n": len(r),
                          **vl.sharpe_bootstrap(r["managed"], r["bh"])})
        for x, y_ in (("tfm_spread", "vix"), ("tfm_spread", "rv20"), ("vix", "rv20")):
            srows.append({"cost_bps": cost, "signal": f"{x} - {y_}", "strategy": "managed - managed", "n": len(R[(cost, x)]),
                          **vl.sharpe_bootstrap(R[(cost, x)]["managed"], R[(cost, y_)]["managed"])})
    S = pd.DataFrame(srows)
    for cost in (COST, 0):
        m = (S.cost_bps == cost) & (S.strategy == "managed - bh")
        S.loc[m, "holm_p"] = holm(S.loc[m, "p"].to_numpy())
    L, Y, Dg = pd.DataFrame(lvl), pd.DataFrame(years), pd.DataFrame(diag)
    L.to_csv(RES_DIR / "timesfm_vix_test_levels.csv", index=False, float_format="%.5f")
    C.to_csv(RES_DIR / "timesfm_vix_test_comparisons.csv", index=False, float_format="%.5f")
    Y.to_csv(RES_DIR / "timesfm_vix_test_by_year.csv", index=False, float_format="%.5f")
    Dg.to_csv(RES_DIR / "timesfm_vix_test_diagnostics.csv", index=False, float_format="%.5f")
    L[L.role.str.startswith("v4_") | (L.role == "static")].to_csv(RES_DIR / "timesfm_vix_test_v4.csv", index=False, float_format="%.5f")
    S.to_csv(RES_DIR / "timesfm_vix_test_strategy.csv", index=False, float_format="%.5f")
    pd.to_pickle({(h, k): f for h in P for k, f in P[h].items()}, RES_DIR / "timesfm_vix_test_predictions.pkl")
    stamp("test_end")
    report()


# ---------------------------------------------------------------------------
# explore: EXPLORATORY, after the test
# ---------------------------------------------------------------------------
def team_regime_rows(h):
    out = subprocess.run(["git", "show", f"origin/master:results/h{h}/metrics_by_vix_regime.csv"], cwd=TF_DIR.parent,
                         capture_output=True, text=True, check=True).stdout
    T = pd.read_csv(io.StringIO(out))
    return T[(T.model == "LogReg") & (T.scheme == "fixed")]


def explore(*args):
    from sklearn.metrics import roc_auc_score
    P = pd.read_pickle(RES_DIR / "timesfm_vix_test_predictions.pkl")
    rows = []
    for h in HORIZONS:
        ds = dl.Inp(h).ds
        lv = ds.data["vix_level"]
        tr = lv[lv.index < pd.Timestamp(PERIODS["TEST"][0])]
        q1, q2 = tr.quantile([1 / 3, 2 / 3])                 # terciles of the VIX level before 2019 (as the team's notebook)
        te = P[(h, "static")].index
        reg = pd.cut(lv.loc[te], [-np.inf, q1, q2, np.inf], labels=["low VIX", "mid VIX", "high VIX"])
        T = team_regime_rows(h).set_index("regime")
        for name, key in (("TimesFM static", "static"), ("fixed LR (ours, reused frame)", "fixed_lr"),
                          ("V1 VIX recalibration", "recal"), ("V2 VIX selection", "select"), ("V3 VIX stacking", "stack"),
                          ("TimesFM without VIX input", "v4_novix")):
            f = P[(h, key)]
            for r in ("low VIX", "mid VIX", "high VIX"):
                g = f[(reg == r).to_numpy()]
                rows.append({"horizon": h, "regime": r, "model": name, "n": len(g),
                             "roc_auc": roc_auc_score(g["y"], g["proba"]),
                             "balanced_acc": metrics(g)["balanced_acc"], "vix_cuts": f"{q1:.2f}/{q2:.2f}"})
        for r in ("low VIX", "mid VIX", "high VIX"):
            rows.append({"horizon": h, "regime": r, "model": "LogReg fixed (team table on master)",
                         "n": int(T.loc[r, "n"]), "roc_auc": float(T.loc[r, "roc_auc"]),
                         "balanced_acc": float(T.loc[r, "balanced_acc"]), "vix_cuts": ""})
    E = pd.DataFrame(rows)
    E.to_csv(RES_DIR / "timesfm_vix_exploratory_by_regime.csv", index=False, float_format="%.5f")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(E.pivot_table(index=["horizon", "model"], columns="regime", values=["roc_auc", "n"], sort=False).to_string())


# ---------------------------------------------------------------------------
# example: one test day, written out (illustration only, nothing is chosen from it)
# ---------------------------------------------------------------------------
def example(*args):
    day, h = pd.Timestamp("2020-03-09"), 1
    ch = json.loads(CHOICES.read_text())["chosen"]
    inp = dl.Inp(h)
    ds = inp.ds
    inp._X = vl.expert_table(inp, ["F1", "F2", "VAL", "TEST"])
    t = int(ds.index.get_loc(day))
    rk = vl.rank_of(inp)
    out = {"day": str(day.date()), "h": h, "vix_rank_today": float(rk[t]), "vix_close": float(ds.data["vix_level"].iloc[t]),
           "tfm_score_raw": float(1 / (1 + np.exp(-inp.z[t]))), "logit_score": float(inp.z[t])}
    # V1: refits of the chosen config over 2019-2020 up to the day
    c1 = ch[f"h{h}.recal"]["spec"]
    s, e = ds.period_positions("TEST")
    _, _, a1 = vl.recal_range(inp, s, t + 1, c1["cal"], c1["N"], cut=c1["cut"])
    fits = [str(ds.index[i].date()) for i in a1["fit_rows"]]
    out["v1_cut"], out["v1_fit_dates_in_test_up_to_day"] = c1["cut"], fits
    t_fit = a1["fit_rows"][-1]
    idx_now = inp.train_rows(t_fit, c1["N"])
    out["v1_offset_in_use"] = dl.fit_cal(inp.z[idx_now], ds.y[idx_now], c1["cal"])[0]
    idx_s = inp.train_rows(s, c1["N"])
    out["v1_offset_at_test_start"] = dl.fit_cal(inp.z[idx_s], ds.y[idx_s], c1["cal"])[0]
    # V2: same-regime Brier of the experts on resolved rows (regime of the chosen split, lookback as chosen)
    c2 = ch[f"h{h}.select"]["spec"]
    X = inp._X
    reg = vl.regime_of(rk, c2["split"])
    nk = ds.n_known(t)
    lo = X["s0"] if c2["lookback"] is None else max(X["s0"], nk - c2["lookback"])
    rows = np.arange(lo, nk)
    same = rows[reg[rows] == reg[t]]
    sq = (X["P"] - ds.y[X["s0"]:X["e0"], None]) ** 2
    out["v2"] = {"split": c2["split"], "regime_today": int(reg[t]), "n_same_regime_rows": int(len(same)),
                 "brier_same_regime": dict(zip(vl.EXPERTS, map(float, sq[same - X["s0"]].mean(0)))),
                 "brier_all_rows": dict(zip(vl.EXPERTS, map(float, sq[rows - X["s0"]].mean(0))))}
    # V3: the chosen stack, fitted as in the test (fixed at the period start)
    c3 = ch[f"h{h}.stack"]["spec"]
    Xs = np.column_stack([inp.z, rk, inp.z * (rk - 0.5)])
    from common import fit_model, make_estimator
    tr = inp.train_rows(s, None)
    m = fit_model(make_estimator("LogReg", {"C": c3["C"]}), Xs[tr], ds.y[tr])
    raw = m[-1].coef_[0] / m[0].scale_
    b0 = float(m[-1].intercept_[0] - (raw * m[0].mean_).sum())
    out["v3"] = {"cfg": c3, "intercept": b0, "coef_z": float(raw[0]), "coef_rank": float(raw[1]), "coef_z_x_rank": float(raw[2]),
                 "proba_today": float(m.predict_proba(Xs[t:t + 1])[0, 1])}
    (RES_DIR / "timesfm_vix_example.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


# ---------------------------------------------------------------------------
# report: tables + figure
# ---------------------------------------------------------------------------
def figure(L):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["Arimo", "DejaVu Sans"], "font.size": 13})
    INK, MUTE, GRID, BLUE, VIOLET = "#1f1f1e", "#52514e", "#e6e5e1", "#2a78d6", "#6250d6"
    names = [("static", "Static\nTimesFM"), ("recal", "VIX\nrecalib."), ("select", "VIX\nselection"),
             ("stack", "VIX\nstacking"), ("v4_novix", "No VIX\nin input"), ("fixed_lr", "Fixed\nLR")]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.set_facecolor("white")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#bdbcb8")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=MUTE, length=0)
    w = 0.36
    for j, (h, col, lab) in enumerate([(1, BLUE, "h=1 (tomorrow)"), (5, VIOLET, "h=5 (next 5 days)")]):
        v = [float(L[(L.horizon == h) & (L.role == r)].roc_auc.iloc[0]) for r, _ in names]
        xs = np.arange(len(names)) + (j - 0.5) * w
        ax.bar(xs, [a - 0.5 for a in v], w * 0.94, bottom=0.5, color=col, label=lab, zorder=3)
        for x, a in zip(xs, v):
            ax.text(x, a + (0.004 if a >= 0.5 else -0.004), f"{a:.3f}".lstrip("0"), ha="center",
                    va="bottom" if a >= 0.5 else "top", fontsize=11, color=INK, fontweight="bold")
    ax.axhline(0.5, color=INK, lw=1.4, ls=(0, (5, 3)), zorder=4)
    ax.text(-0.46, 0.5, "coin flip", ha="left", va="bottom", fontsize=12, color=INK)
    ax.set_xticks(range(len(names))); ax.set_xticklabels([n for _, n in names], fontsize=12, color=INK)
    allv = L[L.role.isin([r for r, _ in names])].roc_auc
    lo = min(0.44, allv.min() - 0.012); hi = max(0.56, allv.max() + 0.015)
    ax.set_ylim(lo, hi)
    ax.set_yticks([t for t in (0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56) if lo <= t <= hi])
    ax.set_ylabel("Test AUC 2019-2025", color=MUTE)
    ax.legend(loc="upper left", frameon=False, fontsize=12, labelcolor=INK, borderaxespad=0, ncol=2)
    fig.tight_layout()
    out = TF_DIR / "slides" / "fig" / "vix_methods.png"
    fig.savefig(out, dpi=220, facecolor="white")
    plt.close(fig)
    return out


def report(*args):
    L = pd.read_csv(RES_DIR / "timesfm_vix_test_levels.csv")
    C = pd.read_csv(RES_DIR / "timesfm_vix_test_comparisons.csv")
    Y = pd.read_csv(RES_DIR / "timesfm_vix_test_by_year.csv")
    D = pd.read_csv(RES_DIR / "timesfm_vix_test_diagnostics.csv")
    S = pd.read_csv(RES_DIR / "timesfm_vix_test_strategy.csv")
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.4f}".format)
    print(L[["horizon", "role", "cfg", "n", "roc_auc", "balanced_acc", "brier", "pred_up_rate"]].to_string(index=False))
    cols = ["horizon", "method", "a", "b", "family", "d_roc_auc", "d_roc_auc_lo", "d_roc_auc_hi", "p_roc_auc", "holm_roc_auc",
            "d_balanced_acc", "d_balanced_acc_lo", "d_balanced_acc_hi", "p_balanced_acc", "holm_balanced_acc"]
    print(C[cols].to_string(index=False))
    print(D.to_string(index=False))
    print(S.to_string(index=False))
    for h in HORIZONS:
        print(f"-- per-year AUC (exploratory), h={h}")
        print(Y[Y.horizon == h].pivot_table(index="role", columns="year", values="roc_auc").to_string())
    print(figure(L))


if __name__ == "__main__":
    {"select": select, "test": test, "explore": explore, "example": example, "report": report}[sys.argv[1]](*sys.argv[2:])

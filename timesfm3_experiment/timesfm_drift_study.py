"""Do the team's drift-aware methods help the frozen, zero-shot TimesFM-3 score on SPY direction?

Usage:  python timesfm_drift_study.py select   -> choices (selection folds only) + pre-registration (locked)
        python timesfm_drift_study.py test     -> ONE-SHOT 2019-2025 run; refuses if the choices hash differs
        python timesfm_drift_study.py report   -> prints the tables and draws slides/fig/drift_methods.png

TimesFM is never refitted. The drift methods (recalibration, stacking, Hedge, detector-triggered recalibration)
act on top of its cached daily score; see timesfm_drift_lib.py. No model call is made anywhere.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
import hashlib
import json
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import timesfm_drift_lib as dl
from tf_paths import RES_DIR, TF_DIR
from common import metrics, paired_bootstrap
from run_test import chance, holm

HORIZONS = (1, 5)
FOLDS = ("F2", "VAL")                       # F1 has no earlier TimesFM score to recalibrate on (cache starts 2013-01-02)
SEL_END = "2018-12-31"
FAMILIES = ("platt", "stack", "hedge", "detect")
FAM_NAME = {"platt": "recalibration", "stack": "stacking", "hedge": "Hedge", "detect": "detector"}
CHOICES = RES_DIR / "timesfm_drift_choices.json"
PREREG = TF_DIR / "timesfm_drift_preregistration.md"
VLOG = RES_DIR / "timesfm_drift_validation_log.csv"
STAMPS = RES_DIR / "timesfm_drift_timestamps.csv"
TEST_FILES = [RES_DIR / f"timesfm_drift_test_{x}" for x in
              ("levels.csv", "comparisons.csv", "by_year.csv", "diagnostics.csv", "predictions.pkl")]
OLD_PREDS = RES_DIR / "timesfm_test_predictions.pkl"           # the TimesFM study's saved TEST frames
OLD_KEY = {1: "tfm.h1", 5: "tfm.h5"}
N_ROWS = {1: 1759, 5: 1755}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stamp(event, **kw):
    row = pd.DataFrame([{"event": event, "utc": now(), "sha256": kw.get("sha256", "")}])
    row.to_csv(STAMPS, mode="a", header=not STAMPS.exists(), index=False)


# ---------------------------------------------------------------------------
# select: configs chosen on the selection folds only
# ---------------------------------------------------------------------------
def select(*args):
    assert not PREREG.exists() and not CHOICES.exists(), "already locked; remove nothing, the choices are final"
    assert not any(f.exists() for f in TEST_FILES), "test outputs exist: selection must come first"
    stamp("select_start")
    G = dl.grid()
    rows = []
    for h in HORIZONS:
        inp = dl.Inp(h, upto=SEL_END)                   # no score after 2018-12-31 is even loaded
        for fold in FOLDS:
            st = metrics(inp.static(fold))
            lr = metrics(inp.fixed_lr(fold))
            rows.append({"horizon": h, "family": "static", "cfg": "static", "fold": fold, "spec": "", **st})
            rows.append({"horizon": h, "family": "fixed_lr", "cfg": "fixed_lr", "fold": fold, "spec": "", **lr})
            for fam in FAMILIES:
                for i, spec in enumerate(G[fam]):
                    f = dl.run(inp, fam, spec, fold)
                    m = metrics(f)
                    rows.append({"horizon": h, "family": fam, "cfg": f"{fam}-{i}", "fold": fold,
                                 "spec": json.dumps(spec, sort_keys=True),
                                 "d_ba": m["balanced_acc"] - st["balanced_acc"], "d_auc": m["roc_auc"] - st["roc_auc"],
                                 "d_brier": st["brier"] - m["brier"], "n_refits": f.attrs.get("n_refits", np.nan),
                                 "neg_slope_share": f.attrs.get("neg_slope_share", np.nan), **m})
    V = pd.DataFrame(rows)
    V.to_csv(VLOG, index=False, float_format="%.5f")

    choices = {"rule": "per horizon and family: eligible if delta BA > 0 and delta AUC > 0 vs the static TimesFM frame "
                       "in BOTH folds F2 and VAL; best = highest mean(delta BA) + mean(delta AUC); if none is eligible "
                       "the best-ranked config is still taken and flagged",
               "folds": list(FOLDS), "variant": dl.VARIANT, "ctx": dl.CTX, "chosen": {}}
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
    CHOICES.write_text(json.dumps(choices, indent=2, sort_keys=True))
    digest = hashlib.sha256(CHOICES.read_bytes()).hexdigest()
    stamp("choices_written", sha256=digest)
    PREREG.write_text(prereg_text(V, choices, digest), encoding="utf-8")
    stamp("prereg_locked", sha256=digest)
    print(PREREG.read_text(encoding="utf-8"))


def spec_str(fam, s):
    if fam == "platt":
        return f"{s['cal']}, window {s['N'] or 'expanding'}, refit every {s['R']} rows"
    if fam == "stack":
        f = "logit score" if s["feats"] == "z" else "logit score + median return + q90-q10 spread"
        w = {"fixed": "fixed (fit once)", "expanding": f"expanding, refit every {s.get('R')}",
             "rolling": f"rolling {s.get('W')}, refit every {s.get('R')}"}[s["mode"]]
        return f"LR C={s['C']} on {f}; {w}"
    if fam == "hedge":
        return f"eta={s['eta']}, loss half-life {s['loss_hl']}"
    d = "Page-Hinkley" if s["detector"] == "ph" else "ADWIN"
    p = f"delta={s['delta']}, lambda={s['lam']}" if s["detector"] == "ph" else f"delta={s['delta']}"
    return f"{d} ({p}) on {'log loss' if s['stream'] == 'logloss' else '0/1 error'}; Platt refit window {s['W'] or 'expanding'}"


def prereg_text(V, ch, digest):
    L = ["# Drift methods on frozen TimesFM-3: pre-registration", "",
         f"Choices `results/timesfm_drift_choices.json`  SHA-256: `{digest}`", "",
         f"Locked at (UTC): `{now()}`. `timesfm_drift_study.py test` refuses to run if the hash differs, and the "
         "timestamps of every step are in `results/timesfm_drift_timestamps.csv` "
         "(`select_start` < `choices_written` < `prereg_locked` < `test_start`). Written after all selection runs and "
         "**before any 2019-2025 metric of a drift method was computed**: `select` loads no TimesFM score after "
         f"{SEL_END} (`Inp(h, upto=...)`) and evaluates only the folds {', '.join(FOLDS)}.", "",
         "## What is tested", "",
         "TimesFM-3.0 stays frozen and zero-shot (no retraining, no fine-tuning, no model call: the cached forecasts "
         f"of the official setup, variant `{dl.VARIANT}`, context {dl.CTX}, are read). Four families of outer layers "
         "act on its daily P(up) score z = logit(score):", "",
         "1. **recalibration** (`platt`): logistic recalibration of z (Platt: slope + offset, the slope may go negative "
         "= flip the direction; or offset only) on the last N known labels, expanding or rolling, refit every R rows;",
         "2. **stacking** (`stack`): logistic regression (C=0.03, standardised) on TimesFM-derived features, fixed (fit "
         "once at the period start), expanding or rolling, as in the project's walk-forward experiment;",
         "3. **Hedge** (`hedge`): exponentially weighted ensemble over three fixed experts {TimesFM calibrated (official "
         "static frame), the team's fixed LR, base rate}; weights from delayed, discounted log loss "
         "(`methods.run_hedge` logic); the thresholds are weighted the same way;",
         "4. **detector** (`detect`): Page-Hinkley or ADWIN on TimesFM's own delayed log loss / error; when it fires "
         "(at least 21 rows since the last refit) the Platt calibration is refit on a recent window "
         "(`methods.run_detect` logic).", "",
         "Every label-fitted step uses only labels with `label_end <= decision date` (`Dataset.n_known`, asserted), "
         "thr = positive rate of the training window, pred = proba >= thr. The logic mirrors `experiments/methods.py`; "
         "its functions need a feature matrix, so they are re-implemented on the score column in "
         "`timesfm_drift_lib.py` (detectors, loss and Dataset are imported unchanged).", "",
         "## Selection rule (selection folds only)", "",
         f"Folds {', '.join(FOLDS)} (F2 2015-16, VAL 2017-18). **Deviation from the team's three folds:** the cached "
         "TimesFM forecasts start on 2013-01-02, so F1 (2013-14) has no earlier scores to recalibrate on and no static "
         "counterpart; adding pre-2013 forecasts would need new model calls, which this study does not make. The "
         "counterpart of every config in a fold is the *static official* TimesFM frame of that fold (offset fitted once "
         "on the purged rows before the fold, threshold = their positive rate). Rule per horizon and family: eligible "
         "if delta BA > 0 and delta AUC > 0 in both folds; take the eligible config with the highest mean(delta BA) + "
         "mean(delta AUC); if none is eligible the best-ranked config is taken anyway and flagged (`eligible` below). "
         "Grid (34 configs per horizon): recalibration 12 (Platt / offset x window 252, 504, expanding x refit 21, 63), "
         "stacking 8 (2 feature sets x fixed, expanding, rolling 252, rolling 504), Hedge 6 (eta 20, 100, 500 x loss "
         "half-life 63, 252), detector 8 (PH lambda 0.5 / 2, ADWIN delta 0.002 / 0.05 x window 504 / expanding). "
         "All configs are logged in `results/timesfm_drift_validation_log.csv`.", "",
         "## Chosen configs", "",
         "| h | family | config | spec | eligible | mean dBA | mean dAUC | eligible / configs |", "|---|---|---|---|---|---|---|---|"]
    for k, c in ch["chosen"].items():
        L.append(f"| {c['horizon']} | {FAM_NAME[c['family']]} | {c['cfg']} | {spec_str(c['family'], c['spec'])} | "
                 f"{c['eligible']} | {c['mean_d_ba']:+.4f} | {c['mean_d_auc']:+.4f} | {c['n_eligible']} / {c['n_configs']} |")
    L += ["", "## Test (2019-2025, run once)", "",
          "Exactly the TimesFM study's rows: `Dataset(h)` TEST rows, 1,759 for h=1 and 1,755 for h=5 (asserted, and the "
          "static frame is checked equal to the saved `results/timesfm_test_predictions.pkl`). The test period has no "
          "effect on any choice; no config is changed after the numbers are seen, anything later is exploratory.", "",
          "**Primary comparisons**, for each of the 4 chosen methods and each horizon: "
          "(a) vs the **static official TimesFM** frame (offset fitted once on 2013-2018, never updated); "
          "(b) vs the project's **fixed LR** (`LogReg C=0.003`, 23 core features, fitted once; frames reused from "
          "`results/timesfm_test_predictions.pkl`). Descriptive only (not in the Holm family): each method vs chance.", "",
          "**Metrics:** ROC-AUC, balanced accuracy (BA), Brier. **Inference:** paired moving-block bootstrap "
          "(`common.paired_bootstrap`, 2000 draws, block length max(20, 3h), seed 42) of method minus counterpart, 95% "
          "percentile CI, one-sided p = P(delta* <= 0). **Holm** within each horizon over the 16 tests "
          "{4 methods} x {2 counterparts} x {BA, AUC}.", "",
          "A method **wins** only if its Holm-adjusted p < 0.05 against **both** counterparts on BA or on AUC, with a "
          "non-negative point estimate on the other of the two. Both horizons are reported, win or not. Exploratory "
          "(marked as such): per-year AUC 2019-2025 of the chosen methods.", "",
          "## Caveats", "",
          "* **TimesFM's 2019-2025 results were already seen** (the TimesFM study: AUC 0.482 at h=1, 0.459 at h=5), as were "
          "the fixed LR's. This pre-registration locks the *drift methods'* configs before their own test metrics exist, "
          "but the test period is not a pristine hold-out for the project: the choice of asking whether drift handling "
          "could rescue TimesFM was made knowing that TimesFM's static score did not work on it (cf. the similar caveat "
          "of the VIX study).",
          "* Possible pretraining leakage of TimesFM-3.0 on SPY's public history, as in the TimesFM study; it does not "
          "affect the comparison of the drift methods with the static score, which share it.",
          "* Only two selection folds (F2, VAL) and a small grid; selection on two folds is noisy, and for most families "
          "no config may be eligible (flagged above).",
          "* The score approximates the project's label (Open[t+1] to Close[t+h]); the overnight gap is not modelled.",
          "* The drift layers cannot add information TimesFM does not contain, except through the other experts "
          "(Hedge: the fixed LR, the base rate) and the two extra stacking features."]
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
    assert list(st.event[:3]) == ["select_start", "choices_written", "prereg_locked"]
    ch = json.loads(CHOICES.read_text())["chosen"]
    stamp("test_start", sha256=want)
    print(f"one-shot test, choices {want[:12]}...", flush=True)
    old = pd.read_pickle(OLD_PREDS)

    P, lvl, comps, years, diag = {}, [], [], [], []
    for h in HORIZONS:
        inp = dl.Inp(h)
        ds = inp.ds
        s, e = ds.period_positions("TEST")
        assert e - s == N_ROWS[h], (h, e - s)
        a_old, g_old = old[(OLD_KEY[h], "timesfm")], old[(OLD_KEY[h], "fixed_lr")]
        static = inp.static("TEST")
        assert static.index.equals(ds.index[s:e]) and a_old.index.equals(static.index)           # same rows as the TimesFM study
        assert np.allclose(static["proba"], a_old["proba"]) and (static["pred"].to_numpy() == a_old["pred"].to_numpy()).all()
        assert g_old.index.equals(static.index) and (g_old["y"].to_numpy() == ds.y[s:e]).all()
        assert np.allclose(inp.fixed_lr("TEST")["proba"], g_old["proba"])
        frames = {"static": static, "fixed_lr": g_old, "chance": chance(ds)}
        for fam in FAMILIES:
            c = ch[f"h{h}.{fam}"]
            frames[fam] = dl.run(inp, fam, c["spec"], "TEST")
            assert frames[fam].index.equals(static.index) and (frames[fam]["y"].to_numpy() == static["y"].to_numpy()).all()
            diag.append({"horizon": h, "family": fam, "cfg": c["cfg"], **frames[fam].attrs})
        P[h] = frames
        for role, f in frames.items():
            m = metrics(f)
            lvl.append({"horizon": h, "role": role, "cfg": ch.get(f"h{h}.{role}", {}).get("cfg", ""), **m})
            for yr, gg in f.groupby(f.index.year):
                mm = metrics(gg)
                years.append({"horizon": h, "role": role, "year": int(yr), **{k: mm[k] for k in ("balanced_acc", "roc_auc", "brier", "n")}})
        jobs = [(fam, ref) for fam in FAMILIES for ref in ("static", "fixed_lr", "chance")]
        res = Parallel(n_jobs=4)(delayed(_boot)(frames[f], frames[r], h) for f, r in jobs)
        for (f, r), b in zip(jobs, res):
            comps.append({"horizon": h, "method": f, "cfg": ch[f"h{h}.{f}"]["cfg"], "vs": r, **b})
        print(f"h={h} done", flush=True)

    C = pd.DataFrame(comps)
    for h in HORIZONS:
        sel = (C.horizon == h) & C.vs.isin(["static", "fixed_lr"])
        adj = holm(np.concatenate([C.loc[sel, "p_balanced_acc"], C.loc[sel, "p_roc_auc"]]))
        n = int(sel.sum())
        C.loc[sel, "holm_balanced_acc"], C.loc[sel, "holm_roc_auc"] = adj[:n], adj[n:]
    # Hedge weights are not part of `attrs` printing below; diagnostics are saved as they are
    L, Y, D = pd.DataFrame(lvl), pd.DataFrame(years), pd.DataFrame(diag)
    L.to_csv(RES_DIR / "timesfm_drift_test_levels.csv", index=False, float_format="%.5f")
    C.to_csv(RES_DIR / "timesfm_drift_test_comparisons.csv", index=False, float_format="%.5f")
    Y.to_csv(RES_DIR / "timesfm_drift_test_by_year.csv", index=False, float_format="%.5f")
    D.to_csv(RES_DIR / "timesfm_drift_test_diagnostics.csv", index=False, float_format="%.5f")
    pd.to_pickle({(h, k): f for h in P for k, f in P[h].items()}, RES_DIR / "timesfm_drift_test_predictions.pkl")
    stamp("test_end")
    report()


# ---------------------------------------------------------------------------
# report: tables + figure
# ---------------------------------------------------------------------------
def figure(L):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    plt.rcParams.update({"font.family": ["Arimo", "DejaVu Sans"], "font.size": 13})
    INK, MUTE, GRID, BLUE, VIOLET = "#1f1f1e", "#52514e", "#e6e5e1", "#2a78d6", "#6250d6"
    names = [("static", "Static\nTimesFM"), ("platt", "Recalib-\nration"), ("stack", "Stacking"),
             ("hedge", "Hedge"), ("detect", "Detector"), ("fixed_lr", "Fixed\nLR")]
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
    lo = min(0.44, L.roc_auc.min() - 0.012); hi = max(0.56, L.roc_auc.max() + 0.015)
    ax.set_ylim(lo, hi)
    ax.set_yticks([t for t in (0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56) if lo <= t <= hi])
    ax.set_ylabel("Test AUC 2019-2025", color=MUTE)
    ax.legend(loc="upper left", frameon=False, fontsize=12, labelcolor=INK, borderaxespad=0, ncol=2)
    fig.tight_layout()
    out = TF_DIR / "slides" / "fig" / "drift_methods.png"
    fig.savefig(out, dpi=220, facecolor="white")
    plt.close(fig)
    return out


def report(*args):
    L = pd.read_csv(RES_DIR / "timesfm_drift_test_levels.csv")
    C = pd.read_csv(RES_DIR / "timesfm_drift_test_comparisons.csv")
    Y = pd.read_csv(RES_DIR / "timesfm_drift_test_by_year.csv")
    D = pd.read_csv(RES_DIR / "timesfm_drift_test_diagnostics.csv")
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.4f}".format)
    print(L[["horizon", "role", "cfg", "n", "roc_auc", "balanced_acc", "brier", "pred_up_rate"]].to_string(index=False))
    cols = ["horizon", "method", "vs", "d_roc_auc", "d_roc_auc_lo", "d_roc_auc_hi", "p_roc_auc", "holm_roc_auc",
            "d_balanced_acc", "d_balanced_acc_lo", "d_balanced_acc_hi", "p_balanced_acc", "holm_balanced_acc",
            "d_brier", "d_brier_lo", "d_brier_hi", "p_brier"]
    print(C[cols].to_string(index=False))
    print(D.to_string(index=False))
    for h in HORIZONS:
        print(f"-- per-year AUC (exploratory), h={h}")
        print(Y[Y.horizon == h].pivot_table(index="role", columns="year", values="roc_auc").to_string())
    print(figure(L))


if __name__ == "__main__":
    {"select": select, "test": test, "report": report}[sys.argv[1]](*sys.argv[2:])

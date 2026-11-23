"""Does the VIX help on top of the best drift layer? (exploratory follow-up, nothing is selected)

Usage:  python timesfm_vixdrift_study.py lock      -> choices (inherited, nothing tuned) + hash-locked pre-registration
        python timesfm_vixdrift_study.py validate  -> F2 + VAL numbers, for information only (no choice uses them)
        python timesfm_vixdrift_study.py test      -> ONE-SHOT 2019-2025 run; refuses if the choices hash differs
        python timesfm_vixdrift_study.py report    -> prints the tables and draws slides/fig/vix_drift_methods.png

Baseline D0 = the drift layer of the drift study (h=1 stack-4, h=5 platt-6). D1-D4 change it only through the VIX.
TimesFM is never run here: the forecasts come from the caches of the earlier studies (official, and no-VIX).
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
import timesfm_vix_lib as vl
import timesfm_vixdrift_lib as xl
from tf_paths import RES_DIR, TF_DIR
from common import metrics, paired_bootstrap
from run_test import chance, holm

HORIZONS = (1, 5)
FOLDS = ("F2", "VAL")
SEL_END = "2018-12-31"
DRIFT_CH = RES_DIR / "timesfm_drift_choices.json"
VIX_CH = RES_DIR / "timesfm_vix_choices.json"
DRIFT_LEVELS = RES_DIR / "timesfm_drift_test_levels.csv"
DRIFT_PREDS = RES_DIR / "timesfm_drift_test_predictions.pkl"
CHOICES = RES_DIR / "timesfm_vixdrift_choices.json"
PREREG = TF_DIR / "timesfm_vixdrift_preregistration.md"
VLOG = RES_DIR / "timesfm_vixdrift_validation_log.csv"
STAMPS = RES_DIR / "timesfm_vixdrift_timestamps.csv"
TEST_FILES = [RES_DIR / f"timesfm_vixdrift_test_{x}" for x in
              ("levels.csv", "comparisons.csv", "by_year.csv", "predictions.pkl")]
OLD_PREDS = RES_DIR / "timesfm_test_predictions.pkl"
OLD_KEY = {1: "tfm.h1", 5: "tfm.h5"}
N_ROWS = {1: 1759, 5: 1755}
ROLES = ("d0", "d1", "d2", "d3", "d4")
NAME = {"d0": "D0 TimesFM + drift layer", "d1": "D1 + VIX-triggered refit", "d2": "D2 + VIX-regime selection",
        "d3": "D3 + VIX terms in the layer", "d4": "D4 drift layer, VIX removed from the input",
        "d2_blind": "regime-blind selection (D2 twin)", "fixed_lr": "team's fixed LR"}
CAVEAT = ("This follow-up was asked after the TimesFM, drift-layer and VIX test results were all known, and the drift "
          "layer itself was picked by a rule that was set after its own test. Nothing here is a clean out-of-sample "
          "confirmation. Treat the result as exploratory.")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stamp(event, **kw):
    row = pd.DataFrame([{"event": event, "utc": now(), "sha256": kw.get("sha256", "")}])
    row.to_csv(STAMPS, mode="a", header=not STAMPS.exists(), index=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_specs():
    d = json.loads(DRIFT_CH.read_text())["chosen"]
    v = json.loads(VIX_CH.read_text())["chosen"]
    out = {}
    for h in HORIZONS:
        out[h] = {"d0": d["h1.stack"]["spec"] if h == 1 else d["h5.platt"]["spec"],
                  "d0_cfg": d["h1.stack"]["cfg"] if h == 1 else d["h5.platt"]["cfg"],
                  "recal": v[f"h{h}.recal"]["spec"], "select": v[f"h{h}.select"]["spec"]}
    assert out[1]["d0"] == {"C": 0.03, "feats": "zms", "mode": "fixed"}
    assert out[5]["d0"] == {"N": 252, "R": 21, "cal": "offset"}
    assert out[1]["recal"]["cut"] == out[5]["recal"]["cut"] == 0.9
    return out


# ---------------------------------------------------------------------------
# lock
# ---------------------------------------------------------------------------
def lock(*args):
    assert not PREREG.exists() and not CHOICES.exists(), "already locked; the choices are final"
    assert not any(f.exists() for f in TEST_FILES) and not VLOG.exists(), "outputs exist: the lock must come first"
    stamp("lock_start")
    S = load_specs()
    choices = {
        "note": "Nothing is selected in this study: every setting is inherited from the drift and VIX studies or fixed "
                "here by design before any test metric. " + CAVEAT,
        "variant": dl.VARIANT, "ctx": dl.CTX,
        "inherited_from": {"timesfm_drift_choices.json": sha(DRIFT_CH), "timesfm_vix_choices.json": sha(VIX_CH)},
        "d0": {"h1": {"cfg": S[1]["d0_cfg"], "family": "stack", "spec": S[1]["d0"]},
               "h5": {"cfg": S[5]["d0_cfg"], "family": "platt", "spec": S[5]["d0"]}},
        "vix_recal_trigger": {"h1": S[1]["recal"], "h5": S[5]["recal"], "min_gap": vl.MIN_GAP,
                              "regime": "1[vix_pct_rank_252d >= cut], rank known at close t"},
        "vix_select": {"h1": S[1]["select"], "h5": S[5]["select"], "min_rows": vl.MIN_ROWS, "experts": list(vl.EXPERTS),
                       "d0_expert_history": f"walk-forward D0 from the first expert row to the day before TEST: expanding "
                                            f"refit every {xl.WF_EVERY} rows (h=1) or the D0 schedule (h=5); on TEST the D0 test frame"},
        "vix_terms": {"extra_features": ["vix_pct_rank_252d", "z x (vix_pct_rank_252d - 0.5)"],
                      "h1": "stack-4 features + the two terms, C 0.03, fixed",
                      "h5": f"logit p = z + a + b1 rank + b2 z (rank - 0.5); ML, L2 weight {xl.RIDGE} on b1 b2 only; "
                            "last 252 known rows, refit every 21 rows"},
        "no_vix_input": {"cache": "results/timesfm_vix_cache/novix/multi_ctx256.npz",
                         "rule": "z, median return and q90-q10 spread recomputed from the no-VIX quantiles, same D0 layer"},
        "test_rows": {"1": 1759, "5": 1755},
        "bootstrap": "common.paired_bootstrap, 2000 draws, block length max(20, 3h), seed 42; Holm (run_test.holm)",
        "families": {"primary": "d1-d0, d2-d0, d3-d0, d0-d4 on AUC and BA: 8 tests per horizon, Holm within horizon",
                     "isolating": "d2 - d2_blind on AUC and BA: 2 tests per horizon, Holm within horizon",
                     "vs_lr": "d0..d4 - fixed_lr on AUC and BA: 10 tests per horizon, Holm within horizon"},
    }
    CHOICES.write_text(json.dumps(choices, indent=2, sort_keys=True))
    digest = sha(CHOICES)
    stamp("choices_written", sha256=digest)
    PREREG.write_text(prereg_text(choices, digest), encoding="utf-8")
    stamp("prereg_locked", sha256=digest)
    print(PREREG.read_text(encoding="utf-8"))


def prereg_text(C, digest):
    S = load_specs()
    return f"""# Does the VIX help on top of the best drift layer? Pre-registration

Choices `results/timesfm_vixdrift_choices.json`  SHA-256: `{digest}`

Locked at (UTC): `{now()}`. `timesfm_vixdrift_study.py test` refuses to run if the hash differs, and the timestamps of every step are in `results/timesfm_vixdrift_timestamps.csv` (`lock_start` < `choices_written` < `prereg_locked` < `test_start`). Written before any test metric of this follow-up was computed.

**Caveat (read first).** {CAVEAT}

## Question

The VIX study put the VIX on the raw TimesFM score. The deck now reports "TimesFM + drift layer" everywhere, so the follow-up asks whether the VIX helps on top of that layer. TimesFM itself is frozen and is never run again here: forecasts come from the cached official run (variant `multi`, context 256) and from the cached run without the ^VIX covariate (`results/timesfm_vix_cache/novix/`, which covers 2013-01-02 to 2025-12-30, i.e. every row needed, so no new inference is made).

## Nothing is selected

No grid and no selection rule. Every setting is inherited or fixed here, before any test metric:

| item | value |
|---|---|
| D0 at h=1 | `{S[1]['d0_cfg']}` of the drift study: logistic regression (C=0.03, standardised) on [logit score z, median forecast return, q90-q10 spread], fit once at the start of the period on all known rows |
| D0 at h=5 | `{S[5]['d0_cfg']}`: offset-only recalibration of z (slope 1) on the last 252 known labels, refit every 21 rows |
| threshold of every frame | positive rate of the training labels of the current fit (as in the drift study) |
| VIX regime | `vix_pct_rank_252d` (rank of the VIX close over 252 days, known at close t) |
| trigger cut, min gap | 0.9, 21 rows (V1 study's choice at both horizons; its min gap) |
| selection split, lookback | h=1: tercile, all history; h=5: tercile, 504 rows (V2 study's choices); fewer than 30 same-regime rows: use all rows |

Labels are used only if their end date is on or before the decision date (`Dataset.n_known`, asserted in `Inp.train_rows`).

## The four variants (each changes D0 only through the VIX)

1. **D1 VIX-triggered refit.** h=5: the D0 offset recalibration (N=252) is refit when 1[rank >= 0.9] flips (at least 21 rows apart) instead of every 21 rows. h=1: the D0 stack is refit on all known rows at each flip (same gap) instead of once.
2. **D2 VIX-regime selection.** Experts: D0, the team's fixed LR (C=0.003, 23 core features, trained once per period), and always-up with the base rate known at t. Each day take the expert with the lowest mean Brier on past resolved predictions made in the same VIX regime (V2 study's split and lookback). The D0 expert needs out-of-sample history before 2019: it is built walk-forward from the first row with 250 known labels (expanding refit every 63 rows at h=1; the D0 schedule at h=5), and on the test rows it is the D0 test frame itself. Isolating twin: the same selection ignoring the regime.
3. **D3 VIX terms in the layer.** h=1: the D0 features plus the VIX rank and z x (rank - 0.5), same C, fit once. h=5: logit p = z + a + b1 rank + b2 z (rank - 0.5), maximum likelihood with an L2 weight of {xl.RIDGE} on b1 and b2 (a is free; the slope on z stays 1 before the interaction), last 252 known rows, refit every 21 rows.
4. **D4 VIX removed from the input.** D0 applied to the forecasts made without the ^VIX covariate: z, median return and spread are recomputed from the no-VIX quantiles, then exactly the D0 procedure. Primary comparison: D0 (official input) minus D4; positive means the VIX in the input helps.

## Test (2019-2025, run once)

Exactly the earlier studies' rows: `Dataset(h)` TEST rows, 1,759 for h=1 and 1,755 for h=5 (asserted; the official static frame and the fixed LR are checked equal to `results/timesfm_test_predictions.pkl`). D0 must reproduce the drift study's test AUC (`results/timesfm_drift_test_levels.csv`) to 1e-5, and its saved frame (`timesfm_drift_test_predictions.pkl`); otherwise the run stops.

**Metrics:** ROC-AUC, balanced accuracy (BA), Brier. **Inference:** paired moving-block bootstrap (`common.paired_bootstrap`, 2000 draws, block length max(20, 3h), seed 42) of first minus second, 95% percentile CI, one-sided p = P(delta* <= 0). Holm (`run_test.holm`) within each horizon and family.

| family | comparisons (each on AUC and BA) | Holm size per horizon |
|---|---|---|
| **primary** | D1 - D0, D2 - D0, D3 - D0, D0 - D4 | 8 |
| **isolating** | D2 - regime-blind selection | 2 |
| **also reported** | D0, D1, D2, D3, D4 each - the team's fixed LR | 10 |

Brier differences are reported but not tested. F2 + VAL (2015-2018) numbers of every method are saved in `results/timesfm_vixdrift_validation_log.csv` for information only; they are not used to choose anything.

## Reading the result

A VIX variant "helps" only if it beats D0 with a Holm p below 0.05 on both AUC and BA at the same horizon. A positive but non-significant difference is reported as "up, not significant". {CAVEAT}
"""


# ---------------------------------------------------------------------------
# validate (information only)
# ---------------------------------------------------------------------------
def frames_for(inp, inp_nv, S, X, period):
    f = {"d0": xl.run_d0(inp, S["d0"], period),
         "d1": xl.run_d1(inp, S["d0"], S, period),
         "d2": xl.run_d2(inp, X, S, period),
         "d2_blind": xl.run_d2(inp, X, S, period, regime=False),
         "d3": xl.run_d3(inp, S["d0"], period),
         "d4": xl.run_d0(inp_nv, S["d0"], period),
         "fixed_lr": inp.fixed_lr(period)}
    return f


def validate(*args):
    assert PREREG.exists() and not VLOG.exists() and not any(f.exists() for f in TEST_FILES)
    SP = load_specs()
    rows = []
    for h in HORIZONS:
        S = SP[h]
        inp, nv = xl.make_inp(h, upto=SEL_END), xl.make_inp(h, upto=SEL_END, novix=True)
        X = xl.expert_table(inp, S["d0"], ["F1", "F2", "VAL"])
        for fold in FOLDS:
            fr = frames_for(inp, nv, S, X, fold)
            base = metrics(fr["d0"])
            for role, f in fr.items():
                m = metrics(f)
                rows.append({"horizon": h, "role": role, "fold": fold, **m, "d_ba_vs_d0": m["balanced_acc"] - base["balanced_acc"],
                             "d_auc_vs_d0": m["roc_auc"] - base["roc_auc"], "n_refits": f.attrs.get("n_refits", np.nan)})
    pd.DataFrame(rows).to_csv(VLOG, index=False, float_format="%.5f")
    print(pd.DataFrame(rows).pivot_table(index=["horizon", "role"], columns="fold", values=["roc_auc", "balanced_acc"]).round(4))


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------
def _boot(a, b, h):
    return paired_bootstrap(a, b, h)


def test(*args):
    assert not any(f.exists() for f in TEST_FILES), "the test was already run once; it is not repeated"
    txt = PREREG.read_text(encoding="utf-8")
    want = re.search(r"SHA-256: `([0-9a-f]{64})`", txt).group(1)
    assert sha(CHOICES) == want, "choices changed after pre-registration"
    assert CHOICES.stat().st_mtime <= PREREG.stat().st_mtime, "choices written after the pre-registration"
    st = pd.read_csv(STAMPS)
    assert list(st.event[:3]) == ["lock_start", "choices_written", "prereg_locked"]
    ch = json.loads(CHOICES.read_text())
    SP = load_specs()
    assert ch["inherited_from"] == {"timesfm_drift_choices.json": sha(DRIFT_CH), "timesfm_vix_choices.json": sha(VIX_CH)}
    stamp("test_start", sha256=want)
    print(f"one-shot test, choices {want[:12]}...", flush=True)
    old, dold, dlev = pd.read_pickle(OLD_PREDS), pd.read_pickle(DRIFT_PREDS), pd.read_csv(DRIFT_LEVELS)

    P, lvl, comps, years = {}, [], [], []
    for h in HORIZONS:
        S = SP[h]
        inp, nv = xl.make_inp(h), xl.make_inp(h, novix=True)
        ds = inp.ds
        s, e = ds.period_positions("TEST")
        assert e - s == N_ROWS[h], (h, e - s)
        a_old, g_old = old[(OLD_KEY[h], "timesfm")], old[(OLD_KEY[h], "fixed_lr")]
        static = inp.static("TEST")
        assert static.index.equals(ds.index[s:e]) and a_old.index.equals(static.index)
        assert np.allclose(static["proba"], a_old["proba"]) and (static["pred"].to_numpy() == a_old["pred"].to_numpy()).all()
        assert g_old.index.equals(static.index) and (g_old["y"].to_numpy() == ds.y[s:e]).all()
        assert np.allclose(inp.fixed_lr("TEST")["proba"], g_old["proba"])
        d0 = xl.run_d0(inp, S["d0"], "TEST")
        # D0 reproduction: the drift study's saved frame and its AUC
        fam = xl.d0_family(h)
        ref = dold[(h, fam)]
        assert d0.index.equals(ref.index) and np.allclose(d0["proba"], ref["proba"], atol=1e-9)
        assert (d0["pred"].to_numpy() == ref["pred"].to_numpy()).all()
        auc_ref = float(dlev[(dlev.horizon == h) & (dlev.role == fam)].roc_auc.iloc[0])
        assert abs(metrics(d0)["roc_auc"] - auc_ref) < 1e-5, (h, metrics(d0)["roc_auc"], auc_ref)
        X = xl.expert_table(inp, S["d0"], ["F1", "F2", "VAL", "TEST"], d0_test=d0)
        fr = {"d0": d0, "d1": xl.run_d1(inp, S["d0"], S, "TEST"), "d2": xl.run_d2(inp, X, S, "TEST"),
              "d2_blind": xl.run_d2(inp, X, S, "TEST", regime=False), "d3": xl.run_d3(inp, S["d0"], "TEST"),
              "d4": xl.run_d0(nv, S["d0"], "TEST"), "fixed_lr": g_old, "static": static, "chance": chance(ds)}
        for k, f in fr.items():
            assert f.index.equals(static.index) and (f["y"].to_numpy() == static["y"].to_numpy()).all(), k
        P[h] = fr
        for role, f in fr.items():
            m = metrics(f)
            lvl.append({"horizon": h, "role": role, "n_refits": f.attrs.get("n_refits", np.nan),
                        "chosen": json.dumps(f.attrs.get("chosen", {})), "mean_slope": f.attrs.get("mean_slope", np.nan), **m})
            for yr, gg in f.groupby(f.index.year):
                mm = metrics(gg)
                years.append({"horizon": h, "role": role, "year": int(yr), **{k: mm[k] for k in ("balanced_acc", "roc_auc", "brier", "n")}})
        jobs = [("d1", "d1", "d0", "primary"), ("d2", "d2", "d0", "primary"), ("d3", "d3", "d0", "primary"),
                ("d4", "d0", "d4", "primary"), ("d2", "d2", "d2_blind", "isolating")]
        jobs += [(r, r, "fixed_lr", "vs_lr") for r in ROLES]
        res = Parallel(n_jobs=4)(delayed(_boot)(fr[a], fr[b], h) for _, a, b, _ in jobs)
        for (name, a, b, fam_), r in zip(jobs, res):
            comps.append({"horizon": h, "name": name, "a": a, "b": b, "family": fam_, **r})
        print(f"h={h} done", flush=True)

    C = pd.DataFrame(comps)
    for h in HORIZONS:
        for fam_ in ("primary", "isolating", "vs_lr"):
            sel = (C.horizon == h) & (C.family == fam_)
            adj = holm(np.concatenate([C.loc[sel, "p_balanced_acc"], C.loc[sel, "p_roc_auc"]]))
            n = int(sel.sum())
            C.loc[sel, "holm_balanced_acc"], C.loc[sel, "holm_roc_auc"] = adj[:n], adj[n:]
    L, Y = pd.DataFrame(lvl), pd.DataFrame(years)
    L.to_csv(RES_DIR / "timesfm_vixdrift_test_levels.csv", index=False, float_format="%.5f")
    C.to_csv(RES_DIR / "timesfm_vixdrift_test_comparisons.csv", index=False, float_format="%.5f")
    Y.to_csv(RES_DIR / "timesfm_vixdrift_test_by_year.csv", index=False, float_format="%.5f")
    pd.to_pickle({(h, k): f for h in P for k, f in P[h].items()}, RES_DIR / "timesfm_vixdrift_test_predictions.pkl")
    stamp("test_end")
    report()


# ---------------------------------------------------------------------------
# report: tables + figure
# ---------------------------------------------------------------------------
def figure(L):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["Arimo", "DejaVu Sans"], "font.size": 13})
    INK, MUTE, GRID, BLUE, VIOLET = "#1f1f1e", "#52514e", "#e6e5e1", "#2a78d6", "#6250d6"
    names = [("d0", "TimesFM +\ndrift layer"), ("d1", "+ VIX\ntrigger"), ("d2", "+ VIX\nselection"),
             ("d3", "+ VIX\nterms"), ("d4", "Drift layer,\nno VIX input"), ("fixed_lr", "Fixed\nLR")]
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
    ax.set_xticks(range(len(names))); ax.set_xticklabels([n for _, n in names], fontsize=11.5, color=INK)
    allv = L[L.role.isin([r for r, _ in names])].roc_auc
    lo = min(0.44, allv.min() - 0.012); hi = max(0.56, allv.max() + 0.015)
    ax.set_ylim(lo, hi)
    ax.set_yticks([t for t in (0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56) if lo <= t <= hi])
    ax.set_ylabel("Test AUC 2019-2025", color=MUTE)
    ax.legend(loc="upper left", frameon=False, fontsize=12, labelcolor=INK, borderaxespad=0, ncol=2)
    fig.tight_layout()
    out = TF_DIR / "slides" / "fig" / "vix_drift_methods.png"
    fig.savefig(out, dpi=220, facecolor="white")
    plt.close(fig)
    return out


def report(*args):
    L = pd.read_csv(RES_DIR / "timesfm_vixdrift_test_levels.csv")
    C = pd.read_csv(RES_DIR / "timesfm_vixdrift_test_comparisons.csv")
    pd.set_option("display.width", 250)
    print(L[["horizon", "role", "balanced_acc", "roc_auc", "brier", "n_refits", "chosen"]].round(4).to_string(index=False))
    print(C[["horizon", "name", "a", "b", "family", "d_balanced_acc", "d_balanced_acc_lo", "d_balanced_acc_hi",
             "holm_balanced_acc", "d_roc_auc", "d_roc_auc_lo", "d_roc_auc_hi", "p_roc_auc", "holm_roc_auc"]].round(4).to_string(index=False))
    print(figure(L))


if __name__ == "__main__":
    {"lock": lock, "validate": validate, "test": test, "report": report}[sys.argv[1]](*sys.argv[2:])

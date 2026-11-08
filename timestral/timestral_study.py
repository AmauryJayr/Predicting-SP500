"""Does a zero-shot time-series foundation model (Mistral Timestral-209M) predict SPY direction?

Mirrors timesfm3_experiment/timesfm_study.py step for step.

Usage:  python timestral_study.py smoke     -> results/timestral_timing.csv (shapes, s/forecast, RSS, grid cost)
        python timestral_study.py grid      -> results/timestral_cache/*.npz (resumable: re-run until done)
        python timestral_study.py checks    -> leakage self-check (truncate the prices after t)
        python timestral_study.py report    -> results/timestral_validation_log.csv + score sanity checks
        python timestral_study.py validate  -> smoke, grid, checks and report in one go
        python timestral_study.py select    -> results/timestral_shortlist.json + timestral_preregistration.md
        python timestral_study.py test      -> ONE-SHOT 2019-2025 run (--dry-val rehearses it on VAL)

Nothing is fitted on the price series: the forecaster is used zero-shot and frozen. The only
label-fitted part is a one-parameter offset in logit space (slope fixed at 1), fitted once on the
F1+F2+VAL rows whose label is known at the start of the evaluation period. Selection uses AUC only,
which no monotone calibration can change.
Run with PYTHONDONTWRITEBYTECODE=1 from inside this folder (the team tracks experiments/__pycache__).
"""
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))
import hashlib
import json
import re
import resource
import sys
import time

import numpy as np
import pandas as pd

import timestral_lib as tl
from ts_paths import RES_DIR, TS_DIR, TFM_RES
from common import PERIODS, SELECTION_FOLDS, Dataset, metrics, paired_bootstrap
from common import RES_DIR as PROJECT_RES_DIR   # the team's results/ (test_predictions.pkl)
from methods import run_spec
from run_test import chance, holm

HORIZONS = tl.HORIZONS
GRID_PERIOD = ("2013-01-01", "2018-12-31")                       # F1 + F2 + VAL, 1510 trading days
BATCH, THREADS, CHUNK = 8, 8, 128                                # fastest setting in timestral_timing.csv

# The full 3 x 4 grid costs 4.81 h of CPU time on F1+F2+VAL (results/timestral_timing.csv), over the
# 3 h this study is allowed: `multi` pushes 16 rows through the transformer and is 40x slower per forecast
# than the univariate variants at ctx 2048 (6.3 vs 0.13 s). Dropping the single most expensive cell, on
# cost alone and before any validation score existed, leaves 2.16 h.
DROPPED = {("multi", 2048): "6.33 s/forecast = 2.65 h of grid"}
CONFIGS = [(v, c) for v in tl.VARIANTS for c in tl.CTXS if (v, c) not in DROPPED]

# Dates before ~2014-02-25 have less than 2048 sessions of history (the panel starts 2006-01-03).
# Those forecasts use the longest window available (left-padded with NaN = missing) instead of being
# dropped, so that every config is scored on exactly the same rows. `ctx_used_min` in the validation
# log records how short it got.
LR_SPEC = {"feats": "core", "kind": "wf", "model": "LogReg", "norm": None, "params": {"C": 0.003},
           "refit": None}                                        # the project's headline trained-once LR
LR_KEY = {5: ("5", 1, "primary"), 1: ("1", 6, "primary")}        # its saved TEST frame in test_predictions.pkl

SHORT = RES_DIR / "timestral_shortlist.json"
PREREG = TS_DIR / "timestral_preregistration.md"
LOG = RES_DIR / "timestral_validation_log.csv"
TFM_PRED = TFM_RES / "timesfm_test_predictions.pkl"              # TimesFM-3 saved test frames, read only
BUDGET_HOURS = 3.0


class Tee:
    """Mirror stdout into a file (the test run keeps its own transcript)."""

    def __init__(self, path):
        self.f, self.o = open(path, "w", encoding="utf-8"), sys.stdout

    def write(self, s):
        self.o.write(s)
        self.f.write(s)

    def flush(self):
        self.o.flush()
        self.f.flush()


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ---------------------------------------------------------------------------
# 1. Timing smoke test
# ---------------------------------------------------------------------------
PROBE = [("price", 512), ("multi", 512)]                         # cheap and expensive representatives
SETTINGS = [(4, 16), (8, 16), (16, 16), (8, 8), (8, 12)]         # (batch, torch threads)
N_PROBE, N_TIME = 16, 16


def _time(m, v, c, dates) -> dict:
    t0 = time.time()
    q, uns = tl.forecast(m, v, c, dates, with_flags=True)
    dt = time.time() - t0
    assert q.shape == (len(dates), tl.H_MAX, tl.NQ) and np.isfinite(q).all()
    return {"variant": v, "ctx": c, "n": len(dates), "q_shape": "x".join(map(str, q.shape)),
            "ctx_used_min": int(min(len(x[0]) for x in tl.contexts(v, c, dates))),
            "n_unsorted": int(uns.sum()),
            "sec_total": round(dt, 2), "sec_per_forecast": round(dt / len(dates), 4), "peak_rss_mb": round(rss_mb())}


def smoke(*args):
    """Verify output shapes, batch invariance, memory; measure s/forecast under a few runtime settings."""
    dates = tl.grid_dates("2016-01-04", "2016-12-31")
    m = tl.load_model(BATCH, THREADS)
    print(f"RSS after load: {rss_mb():.0f} MB", flush=True)
    tl.forecast(m, "price", 256, dates[:2])                      # warm-up, not timed
    # batch invariance: the same forecast alone and inside a batch of unequal (NaN-padded) contexts
    early = tl.grid_dates("2013-01-02", "2013-01-04").append(dates[:1])
    inv = []
    for v, c in [("price", 2048), ("ret", 2048), ("multi", 512)]:
        tl.set_runtime(m, 8, THREADS)
        qb = tl.forecast(m, v, c, early)
        for i, t in enumerate(early):
            inv.append({"variant": v, "ctx": c, "date": str(t.date()),
                        "ctx_used": len(tl.contexts(v, c, [t])[0][0]),
                        "max_abs_diff_single_vs_batch": float(np.abs(tl.forecast(m, v, c, [t])[0] - qb[i]).max())})
    inv = pd.DataFrame(inv)
    inv.to_csv(RES_DIR / "timestral_batch_invariance.csv", index=False)
    print(inv.to_string(index=False), flush=True)
    rows = []
    for b, th in SETTINGS:
        r = tl.set_runtime(m, b, th)
        for v, c in PROBE:
            rows.append({"stage": "probe", **r, **_time(m, v, c, dates[:N_PROBE])})
            print(rows[-1], flush=True)
    probe = pd.DataFrame(rows)
    tot = probe.groupby(["batch", "torch_threads"])["sec_total"].sum()
    best = tot.idxmin()
    print(f"best setting (batch, threads) = {best}, probe total {tot.min():.1f}s", flush=True)
    r = tl.set_runtime(m, *best)
    for v in tl.VARIANTS:                                        # full grid timed at the best setting
        for c in tl.CTXS:
            rows.append({"stage": "grid", **r, **_time(m, v, c, dates[:N_TIME])})
            print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    n = len(tl.grid_dates(*GRID_PERIOD))
    g = t[t.stage == "grid"].copy()
    g["grid_hours"] = g["sec_per_forecast"] * n / 3600
    t = t.merge(g[["variant", "ctx", "stage", "grid_hours"]], on=["variant", "ctx", "stage"], how="left")
    t["kept"] = [(r.variant, r.ctx) in CONFIGS for r in t.itertuples()]
    t.to_csv(RES_DIR / "timestral_timing.csv", index=False)
    with pd.option_context("display.width", 200):
        print(g[["variant", "ctx", "sec_per_forecast", "grid_hours"]].to_string(index=False))
    print(f"{n} dates x full 3x4 grid: {g['grid_hours'].sum():.2f} CPU-hours (budget {BUDGET_HOURS})")
    print(f"kept grid ({len(CONFIGS)} configs): "
          f"{g[[(r.variant, r.ctx) in CONFIGS for r in g.itertuples()]]['grid_hours'].sum():.2f} CPU-hours")
    print(f"dropped: {sorted(DROPPED)}; peak RSS {rss_mb():.0f} MB")


# ---------------------------------------------------------------------------
# 2. Forecast grid (resumable; the cache is keyed by date)
# ---------------------------------------------------------------------------
def grid(*args):
    """Fill the forecast cache for every kept config over F1+F2+VAL. Safe to re-run."""
    dates = tl.grid_dates(*GRID_PERIOD)
    todo = [(v, c) for v, c in CONFIGS if len(dates.difference(tl.read_cache(v, c)[0]))]
    if not todo:
        print("cache complete")
        return
    m = tl.load_model(BATCH, THREADS)
    print(tl.set_runtime(m, BATCH, THREADS), f"{len(dates)} dates, {len(todo)} configs left", flush=True)
    for v, c in todo:
        tl.ensure(m, v, c, dates, chunk=CHUNK)
        print(f"peak RSS {rss_mb():.0f} MB", flush=True)


# ---------------------------------------------------------------------------
# 3. Leakage self-check
# ---------------------------------------------------------------------------
def checks(*args):
    """A forecast for date t must not move when every price series is truncated after t."""
    rng = np.random.default_rng(0)
    dates = tl.grid_dates(*GRID_PERIOD)
    pick = pd.DatetimeIndex(sorted(dates[rng.choice(len(dates), 3, replace=False)]))
    m = tl.load_model(BATCH, THREADS)
    rows = []
    for v, c in [("price", 512), ("ret", 512), ("multi", 512)]:
        d = tl.truncation_check(m, v, c, pick)
        print(f"truncation check {v} ctx{c}: max |delta quantile| = "
              f"{', '.join(f'{x:.3e}' for x in d)} on {[str(x.date()) for x in pick]}", flush=True)
        rows += [{"variant": v, "ctx": c, "date": str(t.date()), "max_abs_delta": x} for t, x in zip(pick, d)]
    pd.DataFrame(rows).to_csv(RES_DIR / "timestral_truncation_check.csv", index=False)


# ---------------------------------------------------------------------------
# 4. Validation log
# ---------------------------------------------------------------------------
def report(*args):
    """AUC / balanced accuracy of the raw score per horizon x config x fold, plus score sanity."""
    rows = []
    for h in HORIZONS:
        ds = Dataset(h)
        for v, c in CONFIGS:
            have, _, _, uns = tl.read_cache(v, c)
            for f in SELECTION_FOLDS:
                p = tl.raw_frame(ds, v, c, h, f)
                _, used = tl.aligned(v, c, p.index)
                extra = {}
                if f == "VAL":                                   # information only: AUC is unchanged by calibration
                    extra["cal_balanced_acc"] = metrics(tl.pred_frame(ds, v, c, h, f))["balanced_acc"]
                rows.append({"horizon": h, "variant": v, "ctx": c, "fold": f, **metrics(p),
                             "score_mean": p["proba"].mean(), "score_std": p["proba"].std(),
                             "share_at_clip": float(((p["proba"] <= tl.CLIP[0] + 1e-12) |
                                                     (p["proba"] >= tl.CLIP[1] - 1e-12)).mean()),
                             "n_unsorted": int(uns[have.isin(p.index)].sum()),
                             "ctx_used_min": int(used.min()), **extra})
    t = pd.DataFrame(rows)
    t.to_csv(LOG, index=False)
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
        print(t.pivot_table(index=["horizon", "variant", "ctx"], columns="fold",
                            values=["roc_auc", "balanced_acc"]).to_string())
        print("\nraw score mean / std by config (h=1 | h=5):")
        print(t.pivot_table(index=["variant", "ctx"], columns="horizon",
                            values=["score_mean", "score_std"]).to_string())
    print(f"\nnon-monotone quantile forecasts (before sorting) in the cached grid rows: {int(t['n_unsorted'].max())} max per config-horizon")
    print("\ncorrelation of raw scores over F1+F2+VAL:")
    g = tl.grid_dates(*GRID_PERIOD)
    for h in HORIZONS:
        S = pd.DataFrame({f"{v}{c}": tl.scores(v, c, h, g) for v, c in CONFIGS})
        print(f"h={h}\n{S.corr().round(3).to_string()}")


def validate(*args):
    smoke()
    grid()
    checks()
    report()


# ---------------------------------------------------------------------------
# 5. Selection + hash-locked pre-registration
# ---------------------------------------------------------------------------
def select(*args):
    """Best config per horizon by mean fold AUC, requiring VAL AUC > 0.5 (else flagged)."""
    assert not (RES_DIR / "timestral_test_predictions.pkl").exists(), "a test run already exists"
    log = pd.read_csv(LOG)
    t = (log.pivot_table(index=["horizon", "variant", "ctx"], columns="fold", values="roc_auc")
         .assign(mean_auc=lambda d: d[SELECTION_FOLDS].mean(axis=1)).reset_index())
    t.sort_values(["horizon", "mean_auc"], ascending=[True, False]).to_csv(
        RES_DIR / "timestral_validation_table.csv", index=False)
    items = []
    for h in HORIZONS:
        g = t[t.horizon == h].sort_values("mean_auc", ascending=False)
        ok = g[g["VAL"] > 0.5]
        r = (ok if len(ok) else g).iloc[0]
        items.append({"id": f"tsl.h{h}", "horizon": int(h), "variant": r["variant"], "ctx": int(r["ctx"]),
                      "mean_auc": round(float(r["mean_auc"]), 4),
                      "fold_auc": {f: round(float(r[f]), 4) for f in SELECTION_FOLDS},
                      "val_auc_above_half": bool(r["VAL"] > 0.5),
                      "mean_auc_above_half": bool(r["mean_auc"] > 0.5)})
    SHORT.write_text(json.dumps(items, indent=2))
    digest = hashlib.sha256(SHORT.read_bytes()).hexdigest()
    drop_txt = (", ".join(f"{v} ctx{c} ({why})" for (v, c), why in sorted(DROPPED.items())) or "none")
    lines = [
        "# Zero-shot Timestral-209M pre-registration", "",
        f"Shortlist `results/timestral_shortlist.json`  SHA-256: `{digest}`", "",
        "**Authorship.** Claude (Anthropic's model) wrote this document and ran the whole validation on "
        "2026-11-08 while the user was away; **no human reviewed it before the test**. It mirrors "
        "`timesfm3_experiment/timesfm_preregistration.md` (including that file's offset-only calibration "
        "amendment) with Timestral in place of TimesFM-3.", "",
        "Written after all validation runs (F1 2013-2014, F2 2015-2016, VAL 2017-2018) and **before any "
        "2019-2025 forecast was produced**. `timestral_study.py test` refuses to run if the hash differs.", "",
        "## Model and scores (frozen, zero-shot)", "",
        f"`TimestralForecaster.from_pretrained(\"{tl.REPO}\", device=\"cpu\", revision=\"{tl.REVISION}\")` "
        f"(package `timestral` 1.0.0 at commit `{tl.PKG_COMMIT}`), eval mode, `torch.inference_mode()`. "
        "`forecast(...)` with the vendor defaults as shipped: `symmetric_averaging=True`, "
        "`clip_nonnegative=True`, default caps (`quantile_real_cap_k=1e4`, `max_context=32768`). "
        f"`torch.manual_seed(42)`, `torch.set_num_threads({THREADS})`, batch size {BATCH} (speed only; "
        "single-forecast vs batched results are compared in `results/timestral_batch_invariance.csv`). "
        "The 209M size is fixed in advance and is not a selection dimension. Nothing is fitted or "
        "fine-tuned on the series.", "",
        "One forecast per decision date t: `prediction_length=5`, context ending at Close[t], no data after "
        "t. Step 1 serves h=1 and step 5 serves h=5. Variants: `price` (log SPY Close), `ret` (SPY daily log "
        "close-to-close returns), `multi` (`price` plus the log Close of the other 15 tickers as "
        "`past_covariates`). Context lengths 256/512/1024/2048 sessions. Contexts shorter than the window "
        "(history before ~2014-02) are left-padded with NaN (= missing) inside a batch.", "",
        "Raw score s(t,h) = P(up): for `price`/`multi`, P(logClose[t+h] > logClose[t]) from the step-h "
        "quantiles by linear interpolation of the CDF through the 21 points (q_i, level_i) with levels "
        "0.01, 0.05, 0.10, 0.15, ..., 0.90, 0.95, 0.99, clipped to [0.01, 0.99] outside q01..q99. Quantiles "
        "are sorted if not monotone (the count is logged as `n_unsorted`). For `ret`, h=1 is "
        "P(r[t+1] > 0) read the same way and h=5 uses a normal approximation of the 5-day sum, "
        "Phi(mu/sigma) with mu = sum of step medians and sigma^2 = sum((q90-q10)/(2*1.2816))^2 over the 5 "
        "steps. The label compares Close[t+h] with Open[t+1], not with Close[t], so the raw score is an "
        "approximation; selection used AUC only, which is calibration-free.", "",
        "## Selection rule (applied to validation only)", "",
        f"Grid: {len(CONFIGS)} of the 12 (variant, context) cells. Dropped before scoring, on CPU cost alone "
        f"(see `results/timestral_timing.csv`; rule: drop the most expensive cells while the full grid over "
        f"2013-01-01..2018-12-31 would exceed {BUDGET_HOURS:g} CPU-hours): {drop_txt}.", "",
        "Dates before ~2014-02-25 have fewer than 2048 sessions of history (the panel starts 2006-01-03); "
        "those forecasts use the longest window available rather than being dropped, so every config is "
        "scored on exactly the same rows. `ctx_used_min` in `results/timestral_validation_log.csv` records it.", "",
        "Per horizon, the config with the highest mean ROC-AUC of the raw score over F1/F2/VAL, requiring "
        "VAL AUC > 0.5. `val_auc_above_half` in the shortlist flags a selection made without that "
        "condition. Every config is logged in `results/timestral_validation_log.csv`.", "",
        "| id | h | variant | ctx | " + " | ".join(f"AUC {f}" for f in SELECTION_FOLDS)
        + " | mean AUC | VAL > 0.5 | mean > 0.5 |",
        "|---|---|---|---|" + "---|" * (len(SELECTION_FOLDS) + 3)]
    lines += [f"| {x['id']} | {x['horizon']} | {x['variant']} | {x['ctx']} | "
              + " | ".join(f"{x['fold_auc'][f]:.4f}" for f in SELECTION_FOLDS)
              + f" | {x['mean_auc']:.4f} | {x['val_auc_above_half']} | {x['mean_auc_above_half']} |"
              for x in items]
    lines += [
        "", "`mean > 0.5` is false where the selected config's mean validation AUC is itself below chance: "
        "the selection rule still returns the best eligible config, and the test is run on it as "
        "pre-registered, but no skill should be expected of it.",
        "", "## Calibration rule (fixed here, applied once at test time)", "",
        "proba = expit(logit(clip(s, 0.01, 0.99)) + b): an offset-only logistic calibration with the slope "
        "fixed at 1, so it cannot reverse or reorder Timestral's own scores (TEST AUC is the raw score's AUC). "
        "b is the maximum-likelihood offset, fitted once on "
        "all F1+F2+VAL rows whose label is known at the test start (purged with `ds.n_known(s)`: rows "
        "2013-01-02 .. 2018-12-24 for h=5 and .. 2018-12-31 for h=1) and never refitted. "
        "thr = `run_test.chance` rule, the positive rate of `ds.y[:ds.n_known(s)]`; "
        "pred = proba >= thr. Rows are exactly `Dataset(h)`'s TEST rows (`test` asserts this).", "",
        "## Reported comparisons (test 2019-01-02 .. last labelled day of 2025, run once)", "",
        "For each horizon, the selected config against:", "",
        "1. **chance** -- `run_test.chance(ds)`: constant proba = thr = the positive rate of the labels "
        "known at the test start, pred = 1 (BA = AUC = 0.5).",
        "2. **the project's headline trained-once logistic regression** -- "
        "`LogReg{'C': 0.003}` on the 23 `CORE_FEATURES`, fitted once on all labels known at 2019-01-02 and "
        "never adapted. Its saved TEST frames are reused from `experiments/results/test_predictions.pkl`, "
        "keys `('5', 1, 'primary')` and `('1', 6, 'primary')`; `test` re-checks that their index equals "
        "`Dataset(h)`'s TEST rows and otherwise recomputes the same spec with `methods.run_spec`.", "",
        "Each comparison: paired moving-block bootstrap (`common.paired_bootstrap`, 2000 draws, block "
        "length max(20, 3h), seed 42) of Timestral minus counterpart on balanced accuracy, ROC-AUC and "
        "Brier (sign flipped), with a 95% percentile CI and a one-sided p = P(delta* <= 0). Holm correction "
        "within each horizon over the 4 tests {2 comparisons} x {balanced_acc, roc_auc}.", "",
        "A win counts only if the Holm-adjusted p < 0.05 against **both** counterparts on balanced accuracy "
        "or on AUC, with a non-negative point estimate on the other metric. Both horizons are reported, "
        "win or not. No configuration changes after the test numbers are seen; anything later is "
        "exploratory. Also reported: test BA/AUC/Brier levels and per-year BA/AUC for 2019-2025.", "",
        "**Descriptive comparison with TimesFM-3 (not part of the win rule, not Holm-corrected).** The saved "
        "TimesFM-3 TEST frames `timesfm3_experiment/results/timesfm_test_predictions.pkl` (keys "
        "`('tfm.h1'|'tfm.h5', 'timesfm')`, read only) are compared with Timestral's frames by the same paired "
        "block bootstrap on AUC and BA (Timestral minus TimesFM-3, CI and one-sided p reported). The two "
        "studies share the test rows; `test` asserts that the indices are equal.", "",
        "## Secondary analysis: 2026 out-of-sample check (descriptive)", "",
        "SPY and the 15 covariates are re-downloaded from yfinance (2025-06-01 to today) and aligned to "
        "SPY's calendar exactly as the notebooks do (reindex, `ffill(limit=3)`). yfinance re-adjusts past "
        "prices for later dividends, so the fresh panel is re-based on the stored panel at the last shared "
        "session (an additive shift in log space; log returns are untouched) and the overlap mismatch in log "
        "returns is reported. Every 2026 decision date is forecast with the selected configs, labels are "
        "built as in `Dataset` (`y = 1[Close[t+h]/Open[t+1] - 1 > 0]`), proba uses the same pre-2019 offset "
        "calibrator, and thr is the positive rate of all labels known at 2026-01-01. Reported: n, AUC, BA "
        "and a paired block bootstrap against the chance rule. Descriptive only -- n is a few hundred rows "
        "of one regime, and no claim is corrected for it. If the download fails, the check is skipped and "
        "the report says so.", "",
        "## Deviations from the TimesFM-3 protocol (choices made because Timestral differs)", "",
        "* **Quantile grid.** Timestral returns 21 quantile levels (0.01..0.99) where TimesFM-3 returns 9 "
        "(0.1..0.9). The CDF is interpolated through all 21 and the score is clipped to [0.01, 0.99] "
        "(TimesFM: [0.05, 0.95]); the `ret` h=5 sigma still uses (q90-q10).",
        "* **Vendor defaults.** `symmetric_averaging=True` and `clip_nonnegative=True` are on (TimesFM was "
        "run with symmetric averaging, z-norm and make_positive off). For the log-Close targets the "
        "non-negative clamp does nothing: log SPY Close is far above 0 (code: `nonnegative_context` / "
        "`clamp_min(0)`), so it changes nothing for `price` and `multi`; `ret` has negative values and is "
        "not clamped. Timestral normalises internally (`scaler.py`) and has no "
        "z-norm switch.",
        "* **Batching.** Contexts of unequal length are left-padded with NaN to the longest in the batch "
        "(the library's own `pad_to_patch` pads the same way, with unobserved zeros); TimesFM took a ragged "
        "list. Batch invariance is measured in the smoke step, not assumed.",
        "* **Horizon.** The patch size is 32, so the 5 requested steps are read from the first patch of a "
        "single decoding block (no autoregressive feedback; code: `_block_ar_decode`).",
        "* **Covariates.** `multi` passes the same 15 tickers as `past_covariates`; the target is still "
        "SPY only (`TimestralForecaster` forecasts variates jointly, but only the target row is returned).",
        "* **Budget.** The 3 CPU-hour rule is applied to Timestral's own timings; see the grid line above.", "",
        "## Caveats", "",
        "* 2019-2025 has already been evaluated in earlier studies of this project, so it is not a pristine "
        "hold-out for the project as a whole (it is untouched by *this* study's selection).",
        "* **Possible pretraining leakage.** The model card of Timestral-209M (and the GitHub README) "
        "describe the architecture and benchmark scores (GIFT-Eval, TIME, fev-bench) but state **no "
        "pretraining data sources and no data cutoff**. SPY's history is public and ubiquitous, so the "
        "2019-2025 (and 2026) backtest cannot be guaranteed free of pretraining contamination. Any positive "
        "result has to be read as an upper bound; a null result is not affected.",
        "* The score is a model of Close-to-Close direction, while the label is Open[t+1]-to-Close[t+h]; the "
        "overnight gap is unmodelled and the offset absorbs only the level part of that mismatch.",
        "* Zero-shot means no horizon-specific or asset-specific fitting at all, so this is a test of the "
        "pretrained forecaster, not of what a fine-tuned Timestral could do."]
    PREREG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# 6. One-shot test
# ---------------------------------------------------------------------------
def lr_frame(ds, h: int, period: str) -> pd.DataFrame:
    """The project's headline trained-once LR on the same rows: reused if saved, else recomputed."""
    s, e = ds.period_positions(period)
    if period == "TEST":
        p = pd.read_pickle(PROJECT_RES_DIR / "test_predictions.pkl").get(LR_KEY[h])
        if p is not None and p.index.equals(ds.index[s:e]) and (p["y"].to_numpy() == ds.y[s:e]).all():
            p = p.copy()
            p.attrs["source"] = f"test_predictions.pkl{LR_KEY[h]}"
            return p
    p = run_spec(ds, LR_SPEC, period)
    p.attrs["source"] = "methods.run_spec(LR_SPEC)"
    return p


def check2026(m, items, h_cal_period: str):
    """Descriptive 2026 check: fresh yfinance data, same configs, same calibrator, chance rule."""
    fresh = tl.fresh_prices()
    pan, join, mism = tl.spliced_panel(fresh)
    print(f"\n2026 check: fresh panel {fresh['SPY'].index[0].date()} .. {fresh['SPY'].index[-1].date()}, "
          f"spliced at {join.date()}, max |log-return mismatch| over the overlap {mism:.2e}", flush=True)
    rows, frames = [], {}
    for x in items:
        h, v, c = x["horizon"], x["variant"], x["ctx"]
        ds = Dataset(h)
        lab = tl.fresh_labels(fresh, h)
        d = lab.index[lab.index >= pd.Timestamp("2026-01-01")]
        q = tl.forecast(m, v, c, d, pan)
        a = pd.Series(pan["spy"], index=pan["dates"]).reindex(d).to_numpy()
        sc = np.clip(np.array([tl.p_up(q[i], v, h, a[i]) for i in range(len(d))]), *tl.CLIP)
        b, _ = tl.calibrator(ds, v, c, h, h_cal_period)
        nk = int((ds.label_end <= pd.Timestamp("2026-01-01")).sum())
        r = float(ds.y[:nk].mean())                              # all labels known at 2026-01-01
        y = lab.loc[d, "y"].to_numpy().astype(int)
        proba = tl.apply_cal(b, sc)
        p = pd.DataFrame({"y": y, "proba": proba, "thr": r, "pred": (proba >= r).astype(int)}, index=d)
        ch = tl.chance_frame(y, d, r)
        frames[(h, v, c)] = p
        rows.append({"id": x["id"], "horizon": h, "variant": v, "ctx": c, "thr": r,
                     "first": str(d[0].date()), "last": str(d[-1].date()), "overlap_mismatch": mism,
                     **metrics(p), **paired_bootstrap(p, ch, h)})
    return pd.DataFrame(rows), frames


def win_rule(comp: pd.DataFrame) -> pd.DataFrame:
    """Pre-registered verdict per horizon: Holm p < 0.05 vs both counterparts on BA or AUC, other metric >= 0."""
    out = []
    for h, g in comp.groupby("horizon"):
        v = {}
        for met, oth in [("balanced_acc", "roc_auc"), ("roc_auc", "balanced_acc")]:
            v[met] = bool((g[f"holm_{met}"] < 0.05).all() and (g[f"d_{oth}"] >= 0).all()
                          and (g[f"d_{met}"] > 0).all() and len(g) == 2)
        out.append({"horizon": h, "win_on_ba": v["balanced_acc"], "win_on_auc": v["roc_auc"],
                    "win": v["balanced_acc"] or v["roc_auc"]})
    return pd.DataFrame(out)


def test(*args):
    """Run the pre-registered shortlist once. `--dry-val` rehearses the same code path on VAL."""
    dry = "--dry-val" in args
    period, tag = ("VAL", "_dryval") if dry else ("TEST", "")
    want = re.search(r"SHA-256: `([0-9a-f]{64})`", PREREG.read_text(encoding="utf-8")).group(1)
    assert hashlib.sha256(SHORT.read_bytes()).hexdigest() == want, "shortlist changed after pre-registration"
    items = json.loads(SHORT.read_text())
    sys.stdout = Tee(RES_DIR / f"timestral_test_stdout{tag}.txt")
    print(f"Timestral-209M one-shot run on {period} {PERIODS[period]}, shortlist {want[:12]}..., "
          f"threads={THREADS} batch={BATCH}, dry_val={dry}", flush=True)

    m, preds, lvl, comps, years, vs_tfm = None, {}, [], [], [], []
    tfm = pd.read_pickle(TFM_PRED) if period == "TEST" else {}
    for x in items:
        h, v, c = x["horizon"], x["variant"], x["ctx"]
        ds = Dataset(h)
        s, e = ds.period_positions(period)
        need = ds.index[tl.cal_rows(ds, period)].union(ds.index[s:e])
        if len(need.difference(tl.read_cache(v, c)[0])):         # only the dates still missing
            m = m or tl.load_model(BATCH, THREADS)
            tl.ensure(m, v, c, need, chunk=CHUNK)
        a = tl.pred_frame(ds, v, c, h, period)
        b = tl.chance_frame(ds.y[s:e], ds.index[s:e], tl.base_rate(ds, period))
        if period == "TEST":
            ref = chance(ds)
            assert ref.index.equals(b.index) and np.allclose(ref["proba"], b["proba"])
            assert a.index.equals(ds.index[s:e])
        g = lr_frame(ds, h, period)
        preds[(x["id"], "timestral")], preds[(x["id"], "chance")], preds[(x["id"], "fixed_lr")] = a, b, g
        tag_ = {"id": x["id"], "horizon": h, "variant": v, "ctx": c}
        for role, p in [("timestral", a), ("chance", b), ("fixed_lr", g)]:
            lvl.append({**tag_, "role": role, "source": p.attrs.get("source", ""),
                        "n_cal": p.attrs.get("n_cal", np.nan), "cal_offset": p.attrs.get("cal_offset", np.nan),
                        **metrics(p)})
            for yr, gg in p.groupby(p.index.year):
                mm = metrics(gg)
                years.append({**tag_, "role": role, "year": int(yr), **{k: mm[k] for k in
                              ("balanced_acc", "roc_auc", "brier", "n")}})
        for role in ("chance", "fixed_lr"):
            comps.append({**tag_, "vs": role, **paired_bootstrap(a, preds[(x["id"], role)], h)})
        if period == "TEST":                                     # descriptive: vs TimesFM-3's saved frames
            tf = tfm[(f"tfm.h{h}", "timesfm")]
            assert tf.index.equals(a.index) and (tf["y"].to_numpy() == a["y"].to_numpy()).all()
            preds[(x["id"], "timesfm3")] = tf
            lvl.append({**tag_, "role": "timesfm3", "source": "timesfm_test_predictions.pkl", **metrics(tf)})
            for yr, gg in tf.groupby(tf.index.year):
                mm = metrics(gg)
                years.append({**tag_, "role": "timesfm3", "year": int(yr), **{k: mm[k] for k in
                              ("balanced_acc", "roc_auc", "brier", "n")}})
            vs_tfm.append({**tag_, "vs": "timesfm3", **paired_bootstrap(a, tf, h)})

    comp = pd.DataFrame(comps)
    for _, g in comp.groupby("horizon"):                         # Holm over {2 comparisons} x {BA, AUC}
        adj = holm(np.concatenate([g["p_balanced_acc"], g["p_roc_auc"]]))
        comp.loc[g.index, "holm_balanced_acc"] = adj[: len(g)]
        comp.loc[g.index, "holm_roc_auc"] = adj[len(g):]
    L, Y = pd.DataFrame(lvl), pd.DataFrame(years)
    verdict = win_rule(comp)
    comp.to_csv(RES_DIR / f"timestral_test_comparisons{tag}.csv", index=False)
    L.to_csv(RES_DIR / f"timestral_test_levels{tag}.csv", index=False)
    Y.to_csv(RES_DIR / f"timestral_test_by_year{tag}.csv", index=False)
    verdict.to_csv(RES_DIR / f"timestral_test_verdict{tag}.csv", index=False)
    if vs_tfm:
        pd.DataFrame(vs_tfm).to_csv(RES_DIR / "timestral_test_vs_timesfm3.csv", index=False)
    pd.to_pickle(preds, RES_DIR / f"timestral_test_predictions{tag}.pkl")

    cols = ["id", "vs", "d_balanced_acc", "d_balanced_acc_lo", "d_balanced_acc_hi", "p_balanced_acc",
            "d_roc_auc", "d_roc_auc_lo", "d_roc_auc_hi", "p_roc_auc", "d_brier", "p_brier",
            "holm_balanced_acc", "holm_roc_auc"]
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
        print(L.to_string(index=False))
        print(comp[cols].to_string(index=False))
        print(verdict.to_string(index=False))
        if vs_tfm:
            print(pd.DataFrame(vs_tfm)[["id", "d_balanced_acc", "d_balanced_acc_lo", "d_balanced_acc_hi",
                                        "p_balanced_acc", "d_roc_auc", "d_roc_auc_lo", "d_roc_auc_hi",
                                        "p_roc_auc"]].to_string(index=False))
        print(Y.pivot_table(index=["id", "role"], columns="year", values="balanced_acc").to_string())
        print(Y.pivot_table(index=["id", "role"], columns="year", values="roc_auc").to_string())
        sys.stdout.flush()
        try:
            m = m or tl.load_model(BATCH, THREADS)
            t26, f26 = check2026(m, items, period)
        except Exception as ex:                                  # network or data failure: skip, say so
            print(f"2026 check SKIPPED: {type(ex).__name__}: {ex}")
            (RES_DIR / f"timestral_2026_skipped{tag}.txt").write_text(f"{type(ex).__name__}: {ex}\n")
        else:
            t26.to_csv(RES_DIR / f"timestral_2026{tag}.csv", index=False)
            pd.to_pickle(f26, RES_DIR / f"timestral_2026_predictions{tag}.pkl")
            print(t26[["id", "horizon", "variant", "ctx", "first", "last", "n", "balanced_acc", "roc_auc",
                       "brier", "pred_up_rate", "d_balanced_acc_lo", "d_balanced_acc_hi", "p_balanced_acc",
                       "d_roc_auc_lo", "d_roc_auc_hi", "p_roc_auc"]].to_string(index=False))
    sys.stdout.flush()


if __name__ == "__main__":
    {"smoke": smoke, "grid": grid, "checks": checks, "report": report, "validate": validate,
     "select": select, "test": test}[sys.argv[1]](*sys.argv[2:])

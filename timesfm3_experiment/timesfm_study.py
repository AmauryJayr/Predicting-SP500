"""Does a zero-shot time-series foundation model (Google TimesFM 3.0) predict SPY direction?

Usage:  python timesfm_study.py smoke     -> results/timesfm_timing.csv (shapes, s/forecast, grid cost)
        python timesfm_study.py grid      -> results/timesfm_cache/*.npz (resumable: re-run until done)
        python timesfm_study.py checks    -> leakage self-check (truncate the prices after t)
        python timesfm_study.py report    -> results/timesfm_validation_log.csv + score sanity checks
        python timesfm_study.py validate  -> smoke, grid, checks and report in one go
        python timesfm_study.py select    -> results/timesfm_shortlist.json + timesfm_preregistration.md
        python timesfm_study.py test      -> ONE-SHOT 2019-2025 run (--dry-val rehearses it on VAL)

Nothing is fitted on the price series: the forecaster is used zero-shot and frozen. The only
label-fitted part is a one-parameter offset in logit space (slope fixed at 1), fitted once on the
F1+F2+VAL rows whose label is known at the start of the evaluation period. Selection uses AUC only, which no monotone
calibration can change.
"""
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))
import hashlib
import json
import re
import sys
import time

import numpy as np
import pandas as pd

import timesfm_lib as tl
from common import PERIODS, RES_DIR, EXP_DIR, SELECTION_FOLDS, Dataset, metrics, paired_bootstrap
from methods import run_spec
from run_test import chance, holm

HORIZONS = tl.HORIZONS
GRID_PERIOD = ("2013-01-01", "2018-12-31")                       # F1 + F2 + VAL, 1510 trading days
BATCH, THREADS, CHUNK = 16, 8, 128                               # fastest setting in timesfm_timing.csv

# The full 3 x 4 grid costs 3.25 h of CPU time on F1+F2+VAL (results/timesfm_timing.csv), over the
# ~3 h this study is allowed: `multi` pushes 16 variates through the transformer and is an order of
# magnitude slower per forecast than the univariate variants (4.30 vs 0.14 s at ctx 2048). Its two
# longest contexts are therefore dropped, before any validation score was computed, leaving 0.68 h.
CONFIGS = [(v, c) for v in tl.VARIANTS for c in tl.CTXS if not (v == "multi" and c >= 1024)]
DROPPED = {("multi", 1024): "1.84 s/forecast = 0.77 h of grid", ("multi", 2048): "4.30 s/forecast = 1.80 h of grid"}

# Dates before ~2014-02-25 have less than 2048 sessions of history (the panel starts 2006-01-03).
# Those forecasts use the longest window available instead of being dropped, so that every config is
# scored on exactly the same rows; the model masks the left padding, so a short context is just a
# short context. `ctx_used_min` in the validation log records how short it got.
LR_SPEC = {"feats": "core", "kind": "wf", "model": "LogReg", "norm": None, "params": {"C": 0.003},
           "refit": None}                                        # the project's headline trained-once LR
LR_KEY = {5: ("5", 1, "primary"), 1: ("1", 6, "primary")}        # its saved TEST frame in test_predictions.pkl

SHORT = RES_DIR / "timesfm_shortlist.json"
PREREG = EXP_DIR / "timesfm_preregistration.md"
LOG = RES_DIR / "timesfm_validation_log.csv"


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


# ---------------------------------------------------------------------------
# 1. Timing smoke test
# ---------------------------------------------------------------------------
PROBE = [("price", 512), ("multi", 512)]                         # cheap and expensive representatives
SETTINGS = [(8, 16), (16, 16), (32, 16), (16, 8), (16, 4)]       # (per_core_batch_size, torch threads)
N_PROBE, N_TIME = 8, 16


def _time(m, v, c, dates) -> dict:
    t0 = time.time()
    q = tl.forecast(m, v, c, dates)
    dt = time.time() - t0
    assert q.shape == (len(dates), tl.H_MAX, tl.NQ) and np.isfinite(q).all()
    assert (np.diff(q, axis=-1) >= -1e-9).all(), "quantiles are not sorted"
    return {"variant": v, "ctx": c, "n": len(dates), "q_shape": "x".join(map(str, q.shape)),
            "ctx_used_min": int(min(len(x[0]) for x in tl.contexts(v, c, dates))),
            "sec_total": round(dt, 2), "sec_per_forecast": round(dt / len(dates), 4)}


def smoke(*args):
    """Verify output shapes, measure s/forecast under a few runtime settings, project the grid cost."""
    dates = tl.grid_dates("2016-01-04", "2016-12-31")
    m = tl.load_model(BATCH, THREADS)
    rows = []
    for b, th in SETTINGS:
        r = tl.set_runtime(m, b, th)
        for v, c in PROBE:
            rows.append({"stage": "probe", **r, **_time(m, v, c, dates[:N_PROBE])})
            print(rows[-1], flush=True)
    probe = pd.DataFrame(rows)
    tot = probe.groupby(["per_core_batch_size", "torch_threads"])["sec_total"].sum()
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
    t.to_csv(RES_DIR / "timesfm_timing.csv", index=False)
    with pd.option_context("display.width", 200):
        print(g[["variant", "ctx", "sec_per_forecast", "grid_hours"]].to_string(index=False))
    print(f"{n} dates x full 3x4 grid: {g['grid_hours'].sum():.2f} CPU-hours")
    print(f"kept grid ({len(CONFIGS)} configs): "
          f"{g[[(r.variant, r.ctx) in CONFIGS for r in g.itertuples()]]['grid_hours'].sum():.2f} CPU-hours")
    print(f"dropped: {sorted(DROPPED)}")


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


# ---------------------------------------------------------------------------
# 3. Leakage self-check
# ---------------------------------------------------------------------------
def checks(*args):
    """A forecast for date t must not move when every price series is truncated after t."""
    rng = np.random.default_rng(0)
    dates = tl.grid_dates(*GRID_PERIOD)
    pick = pd.DatetimeIndex(sorted(dates[rng.choice(len(dates), 3, replace=False)]))
    m = tl.load_model(BATCH, THREADS)
    for v, c in [("price", 512), ("ret", 512), ("multi", 512)]:
        d = tl.truncation_check(m, v, c, pick)
        print(f"truncation check {v} ctx{c}: max |delta quantile| = "
              f"{', '.join(f'{x:.3e}' for x in d)} on {[str(x.date()) for x in pick]}", flush=True)


# ---------------------------------------------------------------------------
# 4. Validation log
# ---------------------------------------------------------------------------
def report(*args):
    """AUC / balanced accuracy of the raw score per horizon x config x fold, plus score sanity."""
    rows = []
    for h in HORIZONS:
        ds = Dataset(h)
        for v, c in CONFIGS:
            for f in SELECTION_FOLDS:
                p = tl.raw_frame(ds, v, c, h, f)
                _, used = tl.aligned(v, c, p.index)
                extra = {}
                if f == "VAL":                                   # information only: AUC is unchanged by calibration
                    extra["cal_balanced_acc"] = metrics(tl.pred_frame(ds, v, c, h, f))["balanced_acc"]
                rows.append({"horizon": h, "variant": v, "ctx": c, "fold": f, **metrics(p),
                             "score_mean": p["proba"].mean(), "score_std": p["proba"].std(),
                             "ctx_used_min": int(used.min()), **extra})
    t = pd.DataFrame(rows)
    t.to_csv(LOG, index=False)
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
        print(t.pivot_table(index=["horizon", "variant", "ctx"], columns="fold",
                            values=["roc_auc", "balanced_acc"]).to_string())
        print("\nraw score mean / std by config (h=1 | h=5):")
        print(t.pivot_table(index=["variant", "ctx"], columns="horizon",
                            values=["score_mean", "score_std"]).to_string())
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
    log = pd.read_csv(LOG)
    t = (log.pivot_table(index=["horizon", "variant", "ctx"], columns="fold", values="roc_auc")
         .assign(mean_auc=lambda d: d[SELECTION_FOLDS].mean(axis=1)).reset_index())
    t.sort_values(["horizon", "mean_auc"], ascending=[True, False]).to_csv(
        RES_DIR / "timesfm_validation_table.csv", index=False)
    items = []
    for h in HORIZONS:
        g = t[t.horizon == h].sort_values("mean_auc", ascending=False)
        ok = g[g["VAL"] > 0.5]
        r = (ok if len(ok) else g).iloc[0]
        items.append({"id": f"tfm.h{h}", "horizon": int(h), "variant": r["variant"], "ctx": int(r["ctx"]),
                      "mean_auc": round(float(r["mean_auc"]), 4),
                      "fold_auc": {f: round(float(r[f]), 4) for f in SELECTION_FOLDS},
                      "val_auc_above_half": bool(r["VAL"] > 0.5),
                      "mean_auc_above_half": bool(r["mean_auc"] > 0.5)})
    SHORT.write_text(json.dumps(items, indent=2))
    digest = hashlib.sha256(SHORT.read_bytes()).hexdigest()
    lines = [
        "# Zero-shot TimesFM-3.0 pre-registration", "",
        f"Shortlist `results/timesfm_shortlist.json`  SHA-256: `{digest}`", "",
        "Written after all validation runs (F1 2013-2014, F2 2015-2016, VAL 2017-2018) and **before any "
        "2019-2025 forecast was produced**. `timesfm_study.py test` refuses to run if the hash differs.", "",
        "## Model and scores (frozen, zero-shot)", "",
        "`TimesFM3Forecaster.from_pretrained(\"google/timesfm-3.0-pytorch\", device=\"cpu\")`, eval mode, "
        "`make_positive=False`, no symmetric averaging, no z-norm, `torch.manual_seed(42)`, "
        f"`torch.set_num_threads({THREADS})`, `per_core_batch_size={BATCH}`. Nothing is fitted or fine-tuned "
        "on the series.", "",
        "One forecast per decision date t: horizon 5, context ending at Close[t], no data after t. Step 1 "
        "serves h=1 and step 5 serves h=5. Variants: `price` (log SPY Close), `ret` (SPY daily log "
        "close-to-close returns), `multi` (`price` plus the log Close of the other 15 tickers as "
        "`past_only_covariates`). Context lengths 256/512/1024/2048 sessions.", "",
        "Raw score s(t,h) = P(up): for `price`/`multi`, P(logClose[t+h] > logClose[t]) from the step-h "
        "quantiles by linear interpolation of the CDF through (q_i, 0.1(i+1)), clipped to [0.05, 0.95] "
        "outside q10..q90. For `ret`, h=1 is P(r[t+1] > 0) read the same way and h=5 uses a normal "
        "approximation of the 5-day sum, Phi(mu/sigma) with mu = sum of step medians and "
        "sigma^2 = sum((q90-q10)/(2*1.2816))^2. The label compares Close[t+h] with Open[t+1], not with "
        "Close[t], so the raw score is an approximation; selection used AUC only, which is "
        "calibration-free.", "",
        "## Selection rule (applied to validation only)", "",
        f"Grid: {len(CONFIGS)} of the 12 (variant, context) cells. Dropped before scoring, on CPU cost alone "
        f"(see `results/timesfm_timing.csv`): {', '.join(f'{v} ctx{c}' for v, c in sorted(DROPPED))} -- "
        "`multi` pushes 16 variates through the transformer and is about an order of magnitude slower per "
        "forecast than the univariate variants, which would have put the validation grid far beyond the "
        "3 CPU-hour budget of this study.", "",
        "Dates before ~2014-02-25 have fewer than 2048 sessions of history (the panel starts 2006-01-03); "
        "those forecasts use the longest window available (>= 1761 sessions) rather than being dropped, so "
        "every config is scored on exactly the same rows. `ctx_used_min` in "
        "`results/timesfm_validation_log.csv` records it.", "",
        "Per horizon, the config with the highest mean ROC-AUC of the raw score over F1/F2/VAL, requiring "
        "VAL AUC > 0.5. `val_auc_above_half` in the shortlist flags a selection made without that "
        "condition. Every config is logged in `results/timesfm_validation_log.csv`.", "",
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
        "fixed at 1, so it cannot reverse or reorder TimesFM's own scores (TEST AUC is the raw score's AUC). "
        "b is the maximum-likelihood offset, fitted once on "
        "all F1+F2+VAL rows whose label is known at the test start (purged with `ds.n_known(s)`: rows "
        "2013-01-02 .. 2018-12-24 for h=5 and .. 2018-12-31 for h=1) and never refitted. "
        "thr = `run_test.chance` rule, the positive rate of `ds.y[:ds.n_known(s)]`; "
        "pred = proba >= thr. Rows are exactly `Dataset(h)`'s TEST rows.", "",
        "**Amendment (made before any 2019-2025 forecast was produced).** The first version of this "
        "pre-registration used two-parameter Platt scaling (`LogisticRegression(C=1.0)` on the logit score). "
        "Fitting that calibrator on the pre-2019 rows gave a negative slope for h=1 (-0.048; pooled raw AUC "
        "0.4966) and a positive one for h=5 (+0.214), so at h=1 the test would have scored an *inverted* "
        "TimesFM -- a sign learned from 2013-2018 labels -- rather than the pretrained forecaster. The rule "
        "was therefore changed to the offset-only calibration above for both horizons. The shortlist and "
        "its hash are unchanged.", "",
        "## Reported comparisons (test 2019-01-02 .. last labelled day of 2025, run once)", "",
        "For each horizon, the selected config against:", "",
        "1. **chance** -- `run_test.chance(ds)`: constant proba = thr = the positive rate of the labels "
        "known at the test start, pred = 1 (BA = AUC = 0.5).",
        "2. **the project's headline trained-once logistic regression** -- "
        "`LogReg{'C': 0.003}` on the 23 `CORE_FEATURES`, fitted once on all labels known at 2019-01-02 and "
        "never adapted; this is the `primary` counterpart of shortlist entries 5.1-5.3 and 1.6 in "
        "`preregistration.md`. Its saved TEST frames are reused from `results/test_predictions.pkl`, keys "
        "`('5', 1, 'primary')` and `('1', 6, 'primary')`, whose indices were verified to equal "
        "`Dataset(h)`'s TEST rows exactly; `test` re-checks that and otherwise recomputes the same spec "
        "with `methods.run_spec`.", "",
        "Each comparison: paired moving-block bootstrap (`common.paired_bootstrap`, 2000 draws, block "
        "length max(20, 3h), seed 42) of TimesFM minus counterpart on balanced accuracy, ROC-AUC and Brier "
        "(sign flipped), with a 95% percentile CI and a one-sided p = P(delta* <= 0). Holm correction "
        "within each horizon over the 4 tests {2 comparisons} x {balanced_acc, roc_auc}, as in "
        "`fred_study.py`. Also reported: test BA/AUC/Brier levels and per-year BA/AUC for 2019-2025.", "",
        "A win counts only if the Holm-adjusted p < 0.05 against **both** counterparts on balanced accuracy "
        "or on AUC, with a non-negative point estimate on the other metric. Both horizons are reported, "
        "win or not. No configuration changes after the test numbers are seen; anything later is "
        "exploratory.", "",
        "## Secondary analysis: 2026 out-of-sample check (descriptive)", "",
        "SPY and the 15 covariates are re-downloaded from yfinance (2025-06-01 to today) and aligned to "
        "SPY's calendar exactly as the notebooks do (reindex, `ffill(limit=3)`). yfinance re-adjusts past "
        "prices for later dividends, so the fresh panel is re-based on the stored panel at the last shared "
        "session (an additive shift in log space; log returns are untouched) and the overlap mismatch in log "
        "returns is reported. Every 2026 decision date is forecast with the selected configs, labels are "
        "built as in `Dataset` (`y = 1[Close[t+h]/Open[t+1] - 1 > 0]`), proba uses the same pre-2019 offset "
        "calibrator, and thr is the positive rate of all labels known at 2026-01-01. Reported: n, AUC, BA "
        "and a paired block bootstrap against the chance rule. Descriptive only -- n is a few hundred rows "
        "of one regime, and no claim is corrected for it.", "",
        "## Caveats", "",
        "* 2019-2025 has already been evaluated in earlier studies of this project, so it is not a pristine "
        "hold-out for the project as a whole (it is untouched by *this* study's selection).",
        "* **Possible pretraining leakage.** TimesFM-3.0 publishes no overall data cutoff, and its "
        "pretraining mixture includes GiftEvalPretrain, which has an Economy/Finance domain. SPY's history "
        "is public and ubiquitous, so the 2019-2025 (and 2026) backtest cannot be guaranteed free of "
        "pretraining contamination. Any positive result has to be read as an upper bound.",
        "* The score is a model of Close-to-Close direction, while the label is Open[t+1]-to-Close[t+h]; the "
        "overnight gap is unmodelled and the offset absorbs only the level part of that mismatch.",
        "* Zero-shot means no horizon-specific or asset-specific fitting at all, so this is a test of the "
        "pretrained forecaster, not of what a fine-tuned TimesFM could do."]
    PREREG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# 6. One-shot test
# ---------------------------------------------------------------------------
def lr_frame(ds, h: int, period: str) -> pd.DataFrame:
    """The project's headline trained-once LR on the same rows: reused if saved, else recomputed."""
    s, e = ds.period_positions(period)
    if period == "TEST":
        p = pd.read_pickle(RES_DIR / "test_predictions.pkl").get(LR_KEY[h])
        if p is not None and p.index.equals(ds.index[s:e]) and (p["y"].to_numpy() == ds.y[s:e]).all():
            p = p.copy()
            p.attrs["source"] = f"test_predictions.pkl{LR_KEY[h]}"
            return p
    p = run_spec(ds, LR_SPEC, period)
    p.attrs["source"] = "methods.run_spec(LR_SPEC)"
    return p


def check2026(m, items, h_cal_period: str) -> pd.DataFrame:
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
        sc = np.array([tl.p_up(q[i], v, h, a[i]) for i in range(len(d))])
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


def test(*args):
    """Run the pre-registered shortlist once. `--dry-val` rehearses the same code path on VAL."""
    dry = "--dry-val" in args
    period, tag = ("VAL", "_dryval") if dry else ("TEST", "")
    want = re.search(r"SHA-256: `([0-9a-f]{64})`", PREREG.read_text(encoding="utf-8")).group(1)
    assert hashlib.sha256(SHORT.read_bytes()).hexdigest() == want, "shortlist changed after pre-registration"
    items = json.loads(SHORT.read_text())
    sys.stdout = Tee(RES_DIR / f"timesfm_test_stdout{tag}.txt")
    print(f"TimesFM-3.0 one-shot run on {period} {PERIODS[period]}, shortlist {want[:12]}..., "
          f"threads={THREADS} batch={BATCH}, dry_val={dry}", flush=True)

    m, preds, lvl, comps, years = None, {}, [], [], []
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
        g = lr_frame(ds, h, period)
        preds[(x["id"], "timesfm")], preds[(x["id"], "chance")], preds[(x["id"], "fixed_lr")] = a, b, g
        tag_ = {"id": x["id"], "horizon": h, "variant": v, "ctx": c}
        for role, p in [("timesfm", a), ("chance", b), ("fixed_lr", g)]:
            lvl.append({**tag_, "role": role, "source": p.attrs.get("source", ""),
                        "n_cal": p.attrs.get("n_cal", np.nan), **metrics(p)})
            for yr, gg in p.groupby(p.index.year):
                mm = metrics(gg)
                years.append({**tag_, "role": role, "year": int(yr), **{k: mm[k] for k in
                              ("balanced_acc", "roc_auc", "brier", "n")}})
        for role in ("chance", "fixed_lr"):
            comps.append({**tag_, "vs": role, **paired_bootstrap(a, preds[(x["id"], role)], h)})

    comp = pd.DataFrame(comps)
    for _, g in comp.groupby("horizon"):                         # Holm over {2 comparisons} x {BA, AUC}
        adj = holm(np.concatenate([g["p_balanced_acc"], g["p_roc_auc"]]))
        comp.loc[g.index, "holm_balanced_acc"] = adj[: len(g)]
        comp.loc[g.index, "holm_roc_auc"] = adj[len(g):]
    L, Y = pd.DataFrame(lvl), pd.DataFrame(years)
    comp.to_csv(RES_DIR / f"timesfm_test_comparisons{tag}.csv", index=False)
    L.to_csv(RES_DIR / f"timesfm_test_levels{tag}.csv", index=False)
    Y.to_csv(RES_DIR / f"timesfm_test_by_year{tag}.csv", index=False)
    pd.to_pickle(preds, RES_DIR / f"timesfm_test_predictions{tag}.pkl")

    cols = ["id", "vs", "d_balanced_acc", "d_balanced_acc_lo", "d_balanced_acc_hi", "p_balanced_acc",
            "d_roc_auc", "d_roc_auc_lo", "d_roc_auc_hi", "p_roc_auc", "d_brier", "p_brier",
            "holm_balanced_acc", "holm_roc_auc"]
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
        print(L.to_string(index=False))
        print(comp[cols].to_string(index=False))
        print(Y.pivot_table(index=["id", "role"], columns="year", values="balanced_acc").to_string())
        print(Y.pivot_table(index=["id", "role"], columns="year", values="roc_auc").to_string())
        m = m or tl.load_model(BATCH, THREADS)
        t26, f26 = check2026(m, items, period)
        t26.to_csv(RES_DIR / f"timesfm_2026{tag}.csv", index=False)
        pd.to_pickle(f26, RES_DIR / f"timesfm_2026_predictions{tag}.pkl")
        print(t26[["id", "horizon", "variant", "ctx", "first", "last", "n", "balanced_acc", "roc_auc",
                   "brier", "pred_up_rate", "d_balanced_acc_lo", "d_balanced_acc_hi", "p_balanced_acc",
                   "d_roc_auc_lo", "d_roc_auc_hi", "p_roc_auc"]].to_string(index=False))
    sys.stdout.flush()


if __name__ == "__main__":
    {"smoke": smoke, "grid": grid, "checks": checks, "report": report, "validate": validate,
     "select": select, "test": test}[sys.argv[1]](*sys.argv[2:])

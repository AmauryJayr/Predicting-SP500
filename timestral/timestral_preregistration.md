# Zero-shot Timestral-209M pre-registration

Shortlist `results/timestral_shortlist.json`  SHA-256: `1db2bb46d21c6c800e43c9f084a5ae8438852456f838ed783622690f6e8bb27a`

**Authorship.** Claude (Anthropic's model) wrote this document and ran the whole validation on 2026-11-08 while the user was away; **no human reviewed it before the test**. It mirrors `timesfm3_experiment/timesfm_preregistration.md` (including that file's offset-only calibration amendment) with Timestral in place of TimesFM-3.

Written after all validation runs (F1 2013-2014, F2 2015-2016, VAL 2017-2018) and **before any 2019-2025 forecast was produced**. `timestral_study.py test` refuses to run if the hash differs.

## Model and scores (frozen, zero-shot)

`TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cpu", revision="9874d252ab0aae16f7f7c999f98f98d6ad7d8365")` (package `timestral` 1.0.0 at commit `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`), eval mode, `torch.inference_mode()`. `forecast(...)` with the vendor defaults as shipped: `symmetric_averaging=True`, `clip_nonnegative=True`, default caps (`quantile_real_cap_k=1e4`, `max_context=32768`). `torch.manual_seed(42)`, `torch.set_num_threads(8)`, batch size 8 (speed only; single-forecast vs batched results are compared in `results/timestral_batch_invariance.csv`). The 209M size is fixed in advance and is not a selection dimension. Nothing is fitted or fine-tuned on the series.

One forecast per decision date t: `prediction_length=5`, context ending at Close[t], no data after t. Step 1 serves h=1 and step 5 serves h=5. Variants: `price` (log SPY Close), `ret` (SPY daily log close-to-close returns), `multi` (`price` plus the log Close of the other 15 tickers as `past_covariates`). Context lengths 256/512/1024/2048 sessions. Contexts shorter than the window (history before ~2014-02) are left-padded with NaN (= missing) inside a batch.

Raw score s(t,h) = P(up): for `price`/`multi`, P(logClose[t+h] > logClose[t]) from the step-h quantiles by linear interpolation of the CDF through the 21 points (q_i, level_i) with levels 0.01, 0.05, 0.10, 0.15, ..., 0.90, 0.95, 0.99, clipped to [0.01, 0.99] outside q01..q99. Quantiles are sorted if not monotone (the count is logged as `n_unsorted`). For `ret`, h=1 is P(r[t+1] > 0) read the same way and h=5 uses a normal approximation of the 5-day sum, Phi(mu/sigma) with mu = sum of step medians and sigma^2 = sum((q90-q10)/(2*1.2816))^2 over the 5 steps. The label compares Close[t+h] with Open[t+1], not with Close[t], so the raw score is an approximation; selection used AUC only, which is calibration-free.

## Selection rule (applied to validation only)

Grid: 11 of the 12 (variant, context) cells. Dropped before scoring, on CPU cost alone (see `results/timestral_timing.csv`; rule: drop the most expensive cells while the full grid over 2013-01-01..2018-12-31 would exceed 3 CPU-hours): multi ctx2048 (6.33 s/forecast = 2.65 h of grid).

Dates before ~2014-02-25 have fewer than 2048 sessions of history (the panel starts 2006-01-03); those forecasts use the longest window available rather than being dropped, so every config is scored on exactly the same rows. `ctx_used_min` in `results/timestral_validation_log.csv` records it.

Per horizon, the config with the highest mean ROC-AUC of the raw score over F1/F2/VAL, requiring VAL AUC > 0.5. `val_auc_above_half` in the shortlist flags a selection made without that condition. Every config is logged in `results/timestral_validation_log.csv`.

| id | h | variant | ctx | AUC F1 | AUC F2 | AUC VAL | mean AUC | VAL > 0.5 | mean > 0.5 |
|---|---|---|---|---|---|---|---|---|---|
| tsl.h1 | 1 | ret | 1024 | 0.4382 | 0.4478 | 0.5230 | 0.4697 | True | False |
| tsl.h5 | 5 | ret | 256 | 0.4437 | 0.4217 | 0.5400 | 0.4685 | True | False |

`mean > 0.5` is false where the selected config's mean validation AUC is itself below chance: the selection rule still returns the best eligible config, and the test is run on it as pre-registered, but no skill should be expected of it.

## Calibration rule (fixed here, applied once at test time)

proba = expit(logit(clip(s, 0.01, 0.99)) + b): an offset-only logistic calibration with the slope fixed at 1, so it cannot reverse or reorder Timestral's own scores (TEST AUC is the raw score's AUC). b is the maximum-likelihood offset, fitted once on all F1+F2+VAL rows whose label is known at the test start (purged with `ds.n_known(s)`: rows 2013-01-02 .. 2018-12-24 for h=5 and .. 2018-12-31 for h=1) and never refitted. thr = `run_test.chance` rule, the positive rate of `ds.y[:ds.n_known(s)]`; pred = proba >= thr. Rows are exactly `Dataset(h)`'s TEST rows (`test` asserts this).

## Reported comparisons (test 2019-01-02 .. last labelled day of 2025, run once)

For each horizon, the selected config against:

1. **chance** -- `run_test.chance(ds)`: constant proba = thr = the positive rate of the labels known at the test start, pred = 1 (BA = AUC = 0.5).
2. **the project's headline trained-once logistic regression** -- `LogReg{'C': 0.003}` on the 23 `CORE_FEATURES`, fitted once on all labels known at 2019-01-02 and never adapted. Its saved TEST frames are reused from `experiments/results/test_predictions.pkl`, keys `('5', 1, 'primary')` and `('1', 6, 'primary')`; `test` re-checks that their index equals `Dataset(h)`'s TEST rows and otherwise recomputes the same spec with `methods.run_spec`.

Each comparison: paired moving-block bootstrap (`common.paired_bootstrap`, 2000 draws, block length max(20, 3h), seed 42) of Timestral minus counterpart on balanced accuracy, ROC-AUC and Brier (sign flipped), with a 95% percentile CI and a one-sided p = P(delta* <= 0). Holm correction within each horizon over the 4 tests {2 comparisons} x {balanced_acc, roc_auc}.

A win counts only if the Holm-adjusted p < 0.05 against **both** counterparts on balanced accuracy or on AUC, with a non-negative point estimate on the other metric. Both horizons are reported, win or not. No configuration changes after the test numbers are seen; anything later is exploratory. Also reported: test BA/AUC/Brier levels and per-year BA/AUC for 2019-2025.

**Descriptive comparison with TimesFM-3 (not part of the win rule, not Holm-corrected).** The saved TimesFM-3 TEST frames `timesfm3_experiment/results/timesfm_test_predictions.pkl` (keys `('tfm.h1'|'tfm.h5', 'timesfm')`, read only) are compared with Timestral's frames by the same paired block bootstrap on AUC and BA (Timestral minus TimesFM-3, CI and one-sided p reported). The two studies share the test rows; `test` asserts that the indices are equal.

## Secondary analysis: 2026 out-of-sample check (descriptive)

SPY and the 15 covariates are re-downloaded from yfinance (2025-06-01 to today) and aligned to SPY's calendar exactly as the notebooks do (reindex, `ffill(limit=3)`). yfinance re-adjusts past prices for later dividends, so the fresh panel is re-based on the stored panel at the last shared session (an additive shift in log space; log returns are untouched) and the overlap mismatch in log returns is reported. Every 2026 decision date is forecast with the selected configs, labels are built as in `Dataset` (`y = 1[Close[t+h]/Open[t+1] - 1 > 0]`), proba uses the same pre-2019 offset calibrator, and thr is the positive rate of all labels known at 2026-01-01. Reported: n, AUC, BA and a paired block bootstrap against the chance rule. Descriptive only -- n is a few hundred rows of one regime, and no claim is corrected for it. If the download fails, the check is skipped and the report says so.

## Deviations from the TimesFM-3 protocol (choices made because Timestral differs)

* **Quantile grid.** Timestral returns 21 quantile levels (0.01..0.99) where TimesFM-3 returns 9 (0.1..0.9). The CDF is interpolated through all 21 and the score is clipped to [0.01, 0.99] (TimesFM: [0.05, 0.95]); the `ret` h=5 sigma still uses (q90-q10).
* **Vendor defaults.** `symmetric_averaging=True` and `clip_nonnegative=True` are on (TimesFM was run with symmetric averaging, z-norm and make_positive off). For the log-Close targets the non-negative clamp does nothing: log SPY Close is far above 0 (code: `nonnegative_context` / `clamp_min(0)`), so it changes nothing for `price` and `multi`; `ret` has negative values and is not clamped. Timestral normalises internally (`scaler.py`) and has no z-norm switch.
* **Batching.** Contexts of unequal length are left-padded with NaN to the longest in the batch (the library's own `pad_to_patch` pads the same way, with unobserved zeros); TimesFM took a ragged list. Batch invariance is measured in the smoke step, not assumed.
* **Horizon.** The patch size is 32, so the 5 requested steps are read from the first patch of a single decoding block (no autoregressive feedback; code: `_block_ar_decode`).
* **Covariates.** `multi` passes the same 15 tickers as `past_covariates`; the target is still SPY only (`TimestralForecaster` forecasts variates jointly, but only the target row is returned).
* **Budget.** The 3 CPU-hour rule is applied to Timestral's own timings; see the grid line above.

## Caveats

* 2019-2025 has already been evaluated in earlier studies of this project, so it is not a pristine hold-out for the project as a whole (it is untouched by *this* study's selection).
* **Possible pretraining leakage.** The model card of Timestral-209M (and the GitHub README) describe the architecture and benchmark scores (GIFT-Eval, TIME, fev-bench) but state **no pretraining data sources and no data cutoff**. SPY's history is public and ubiquitous, so the 2019-2025 (and 2026) backtest cannot be guaranteed free of pretraining contamination. Any positive result has to be read as an upper bound; a null result is not affected.
* The score is a model of Close-to-Close direction, while the label is Open[t+1]-to-Close[t+h]; the overnight gap is unmodelled and the offset absorbs only the level part of that mismatch.
* Zero-shot means no horizon-specific or asset-specific fitting at all, so this is a test of the pretrained forecaster, not of what a fine-tuned Timestral could do.

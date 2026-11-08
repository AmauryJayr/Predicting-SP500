# Zero-shot TimesFM-3.0 pre-registration

Shortlist `results/timesfm_shortlist.json`  SHA-256: `46f92f56973e796c4911397203098589e0f445114816f2197a40892955de240a`

Written after all validation runs (F1 2013-2014, F2 2015-2016, VAL 2017-2018) and **before any 2019-2025 forecast was produced**. `timesfm_study.py test` refuses to run if the hash differs.

## Model and scores (frozen, zero-shot)

`TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", device="cpu")`, eval mode, `make_positive=False`, no symmetric averaging, no z-norm, `torch.manual_seed(42)`, `torch.set_num_threads(8)`, `per_core_batch_size=16`. Nothing is fitted or fine-tuned on the series.

One forecast per decision date t: horizon 5, context ending at Close[t], no data after t. Step 1 serves h=1 and step 5 serves h=5. Variants: `price` (log SPY Close), `ret` (SPY daily log close-to-close returns), `multi` (`price` plus the log Close of the other 15 tickers as `past_only_covariates`). Context lengths 256/512/1024/2048 sessions.

Raw score s(t,h) = P(up): for `price`/`multi`, P(logClose[t+h] > logClose[t]) from the step-h quantiles by linear interpolation of the CDF through (q_i, 0.1(i+1)), clipped to [0.05, 0.95] outside q10..q90. For `ret`, h=1 is P(r[t+1] > 0) read the same way and h=5 uses a normal approximation of the 5-day sum, Phi(mu/sigma) with mu = sum of step medians and sigma^2 = sum((q90-q10)/(2*1.2816))^2. The label compares Close[t+h] with Open[t+1], not with Close[t], so the raw score is an approximation; selection used AUC only, which is calibration-free.

## Selection rule (applied to validation only)

Grid: 10 of the 12 (variant, context) cells. Dropped before scoring, on CPU cost alone (see `results/timesfm_timing.csv`): multi ctx1024, multi ctx2048 -- `multi` pushes 16 variates through the transformer and is about an order of magnitude slower per forecast than the univariate variants, which would have put the validation grid far beyond the 3 CPU-hour budget of this study.

Dates before ~2014-02-25 have fewer than 2048 sessions of history (the panel starts 2006-01-03); those forecasts use the longest window available (>= 1761 sessions) rather than being dropped, so every config is scored on exactly the same rows. `ctx_used_min` in `results/timesfm_validation_log.csv` records it.

Per horizon, the config with the highest mean ROC-AUC of the raw score over F1/F2/VAL, requiring VAL AUC > 0.5. `val_auc_above_half` in the shortlist flags a selection made without that condition. Every config is logged in `results/timesfm_validation_log.csv`.

| id | h | variant | ctx | AUC F1 | AUC F2 | AUC VAL | mean AUC | VAL > 0.5 | mean > 0.5 |
|---|---|---|---|---|---|---|---|---|---|
| tfm.h1 | 1 | multi | 256 | 0.4777 | 0.4961 | 0.5052 | 0.4930 | True | False |
| tfm.h5 | 5 | multi | 256 | 0.5255 | 0.5711 | 0.5334 | 0.5433 | True | True |

`mean > 0.5` is false where the selected config's mean validation AUC is itself below chance: the selection rule still returns the best eligible config, and the test is run on it as pre-registered, but no skill should be expected of it.

## Calibration rule (fixed here, applied once at test time)

proba = expit(logit(clip(s, 0.01, 0.99)) + b): an offset-only logistic calibration with the slope fixed at 1, so it cannot reverse or reorder TimesFM's own scores (TEST AUC is the raw score's AUC). b is the maximum-likelihood offset, fitted once on all F1+F2+VAL rows whose label is known at the test start (purged with `ds.n_known(s)`: rows 2013-01-02 .. 2018-12-24 for h=5 and .. 2018-12-31 for h=1) and never refitted. thr = `run_test.chance` rule, the positive rate of `ds.y[:ds.n_known(s)]`; pred = proba >= thr. Rows are exactly `Dataset(h)`'s TEST rows.

**Amendment (made before any 2019-2025 forecast was produced).** The first version of this pre-registration used two-parameter Platt scaling (`LogisticRegression(C=1.0)` on the logit score). Fitting that calibrator on the pre-2019 rows gave a negative slope for h=1 (-0.048; pooled raw AUC 0.4966) and a positive one for h=5 (+0.214), so at h=1 the test would have scored an *inverted* TimesFM -- a sign learned from 2013-2018 labels -- rather than the pretrained forecaster. The rule was therefore changed to the offset-only calibration above for both horizons. The shortlist and its hash are unchanged.

## Reported comparisons (test 2019-01-02 .. last labelled day of 2025, run once)

For each horizon, the selected config against:

1. **chance** -- `run_test.chance(ds)`: constant proba = thr = the positive rate of the labels known at the test start, pred = 1 (BA = AUC = 0.5).
2. **the project's headline trained-once logistic regression** -- `LogReg{'C': 0.003}` on the 23 `CORE_FEATURES`, fitted once on all labels known at 2019-01-02 and never adapted; this is the `primary` counterpart of shortlist entries 5.1-5.3 and 1.6 in `preregistration.md`. Its saved TEST frames are reused from `results/test_predictions.pkl`, keys `('5', 1, 'primary')` and `('1', 6, 'primary')`, whose indices were verified to equal `Dataset(h)`'s TEST rows exactly; `test` re-checks that and otherwise recomputes the same spec with `methods.run_spec`.

Each comparison: paired moving-block bootstrap (`common.paired_bootstrap`, 2000 draws, block length max(20, 3h), seed 42) of TimesFM minus counterpart on balanced accuracy, ROC-AUC and Brier (sign flipped), with a 95% percentile CI and a one-sided p = P(delta* <= 0). Holm correction within each horizon over the 4 tests {2 comparisons} x {balanced_acc, roc_auc}, as in `fred_study.py`. Also reported: test BA/AUC/Brier levels and per-year BA/AUC for 2019-2025.

A win counts only if the Holm-adjusted p < 0.05 against **both** counterparts on balanced accuracy or on AUC, with a non-negative point estimate on the other metric. Both horizons are reported, win or not. No configuration changes after the test numbers are seen; anything later is exploratory.

## Secondary analysis: 2026 out-of-sample check (descriptive)

SPY and the 15 covariates are re-downloaded from yfinance (2025-06-01 to today) and aligned to SPY's calendar exactly as the notebooks do (reindex, `ffill(limit=3)`). yfinance re-adjusts past prices for later dividends, so the fresh panel is re-based on the stored panel at the last shared session (an additive shift in log space; log returns are untouched) and the overlap mismatch in log returns is reported. Every 2026 decision date is forecast with the selected configs, labels are built as in `Dataset` (`y = 1[Close[t+h]/Open[t+1] - 1 > 0]`), proba uses the same pre-2019 offset calibrator, and thr is the positive rate of all labels known at 2026-01-01. Reported: n, AUC, BA and a paired block bootstrap against the chance rule. Descriptive only -- n is a few hundred rows of one regime, and no claim is corrected for it.

## Caveats

* 2019-2025 has already been evaluated in earlier studies of this project, so it is not a pristine hold-out for the project as a whole (it is untouched by *this* study's selection).
* **Possible pretraining leakage.** TimesFM-3.0 publishes no overall data cutoff, and its pretraining mixture includes GiftEvalPretrain, which has an Economy/Finance domain. SPY's history is public and ubiquitous, so the 2019-2025 (and 2026) backtest cannot be guaranteed free of pretraining contamination. Any positive result has to be read as an upper bound.
* The score is a model of Close-to-Close direction, while the label is Open[t+1]-to-Close[t+h]; the overnight gap is unmodelled and the offset absorbs only the level part of that mismatch.
* Zero-shot means no horizon-specific or asset-specific fitting at all, so this is a test of the pretrained forecaster, not of what a fine-tuned TimesFM could do.

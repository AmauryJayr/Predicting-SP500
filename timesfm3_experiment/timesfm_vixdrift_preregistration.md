# Does the VIX help on top of the best drift layer? Pre-registration

Choices `results/timesfm_vixdrift_choices.json`  SHA-256: `e921d7789c941edac400c0e5ebc46686fafb81d3960efdc35d0ba622767d7cd0`

Locked at (UTC): `2026-11-23T06:13:26+00:00`. `timesfm_vixdrift_study.py test` refuses to run if the hash differs, and the timestamps of every step are in `results/timesfm_vixdrift_timestamps.csv` (`lock_start` < `choices_written` < `prereg_locked` < `test_start`). Written before any test metric of this follow-up was computed.

**Caveat (read first).** This follow-up was asked after the TimesFM, drift-layer and VIX test results were all known, and the drift layer itself was picked by a rule that was set after its own test. Nothing here is a clean out-of-sample confirmation. Treat the result as exploratory.

## Question

The VIX study put the VIX on the raw TimesFM score. The deck now reports "TimesFM + drift layer" everywhere, so the follow-up asks whether the VIX helps on top of that layer. TimesFM itself is frozen and is never run again here: forecasts come from the cached official run (variant `multi`, context 256) and from the cached run without the ^VIX covariate (`results/timesfm_vix_cache/novix/`, which covers 2013-01-02 to 2025-12-30, i.e. every row needed, so no new inference is made).

## Nothing is selected

No grid and no selection rule. Every setting is inherited or fixed here, before any test metric:

| item | value |
|---|---|
| D0 at h=1 | `stack-4` of the drift study: logistic regression (C=0.03, standardised) on [logit score z, median forecast return, q90-q10 spread], fit once at the start of the period on all known rows |
| D0 at h=5 | `platt-6`: offset-only recalibration of z (slope 1) on the last 252 known labels, refit every 21 rows |
| threshold of every frame | positive rate of the training labels of the current fit (as in the drift study) |
| VIX regime | `vix_pct_rank_252d` (rank of the VIX close over 252 days, known at close t) |
| trigger cut, min gap | 0.9, 21 rows (V1 study's choice at both horizons; its min gap) |
| selection split, lookback | h=1: tercile, all history; h=5: tercile, 504 rows (V2 study's choices); fewer than 30 same-regime rows: use all rows |

Labels are used only if their end date is on or before the decision date (`Dataset.n_known`, asserted in `Inp.train_rows`).

## The four variants (each changes D0 only through the VIX)

1. **D1 VIX-triggered refit.** h=5: the D0 offset recalibration (N=252) is refit when 1[rank >= 0.9] flips (at least 21 rows apart) instead of every 21 rows. h=1: the D0 stack is refit on all known rows at each flip (same gap) instead of once.
2. **D2 VIX-regime selection.** Experts: D0, the team's fixed LR (C=0.003, 23 core features, trained once per period), and always-up with the base rate known at t. Each day take the expert with the lowest mean Brier on past resolved predictions made in the same VIX regime (V2 study's split and lookback). The D0 expert needs out-of-sample history before 2019: it is built walk-forward from the first row with 250 known labels (expanding refit every 63 rows at h=1; the D0 schedule at h=5), and on the test rows it is the D0 test frame itself. Isolating twin: the same selection ignoring the regime.
3. **D3 VIX terms in the layer.** h=1: the D0 features plus the VIX rank and z x (rank - 0.5), same C, fit once. h=5: logit p = z + a + b1 rank + b2 z (rank - 0.5), maximum likelihood with an L2 weight of 1.0 on b1 and b2 (a is free; the slope on z stays 1 before the interaction), last 252 known rows, refit every 21 rows.
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

A VIX variant "helps" only if it beats D0 with a Holm p below 0.05 on both AUC and BA at the same horizon. A positive but non-significant difference is reported as "up, not significant". This follow-up was asked after the TimesFM, drift-layer and VIX test results were all known, and the drift layer itself was picked by a rule that was set after its own test. Nothing here is a clean out-of-sample confirmation. Treat the result as exploratory.

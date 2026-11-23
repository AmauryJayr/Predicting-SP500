# Does the VIX help TimesFM-3? Pre-registration

Choices `results/timesfm_vix_choices.json`  SHA-256: `488751dc6cc4f96f1d6583dd45e305d4ef1da821dd6d9d9b04036584bf2bf65b`

Locked at (UTC): `2026-11-23T04:08:34+00:00`. `timesfm_vix_study.py test` refuses to run if the hash differs, and the timestamps of every step are in `results/timesfm_vix_timestamps.csv` (`ablation_inference_*` < `select_start` < `choices_written` < `prereg_locked` < `test_start`). Written after all selection runs and **before any 2019-2025 metric of a VIX method was computed**: `select` loads no TimesFM score after 2018-12-31 (`Inp(h, upto=...)`) and evaluates only the folds F2, VAL. The V4 ablation forecasts for 2019-2025 were generated before the lock (model inference only, scores and metrics were not looked at); V4 appears in the selection log only on F2 and VAL, descriptively.

## What is tested

The question: does the VIX help TimesFM-3, frozen and zero-shot, predict SPY direction? It mirrors the team's VIX study (`experiments/vix_study.py` on master, applied to the logistic regression) on the TimesFM score z = logit(P(up)) of the official setup (variant `multi`, context 256). The regime variable is `vix_pct_rank_252d`, the 1-year percentile rank of the VIX close, known at the close of day t. Every label-fitted step uses only labels with `label_end <= decision date` (`Dataset.n_known`, asserted); thr = positive rate of the training window, pred = proba >= thr.

1. **V1 VIX-triggered recalibration** (`recal`): the calibration layer on z (Platt, or offset only as the official frame) is refit on the last N known labels only when 1[rank >= cut] flips, at least 21 rows apart. Isolating comparison: the same layer on scheduled refits every 63 rows with the same window.
2. **V2 VIX-regime expert selection** (`select`): experts {TimesFM (offset-only calibration, expanding, refit every 63 rows), the team's fixed LR (C=0.003, 23 core features, trained once per period F1, F2, VAL, TEST), always-up}. Each day, take the expert with the lowest Brier on past resolved predictions made in the same VIX regime (median or tercile split of the rank; lookback 504 rows or all; at least 30 same-regime rows, else all rows). Isolating comparison: the same selection ignoring the regime. The expert history starts at the first row with 250 known scored labels (about Jan 2014).
3. **V3 VIX-conditioned stacking** (`stack`): logistic regression (standardised, C 0.03 or 0.3) on [z, VIX rank, z x (rank - 0.5)], fixed, expanding (refit 21) or rolling 252 / 504 (refit 21). Isolating comparison: the same stack on z alone (`timesfm_drift_lib.run_stack`, feats z).
4. **V4 input ablation** (no outer layer): the official `multi` forecasts (SPY + 15 covariates) minus the same run with ^VIX removed (14 covariates), context 256, default model flags, batch 16 / 8 threads, over 2013-2025 (new cache `results/timesfm_vix_cache/novix/`; the existing caches are only read). Reproduction check before use: the official `multi` re-run on 64 random 2019-2025 dates and `price` on 16 dates matches the existing caches exactly (max abs diff 0; `results/timesfm_vix_repro.csv`). Both runs get exactly the official calibration (offset fitted once on 2013-2018, threshold = base rate). Secondary: SPY + ^VIX only versus SPY alone (the `price` variant; its 2019-2025 forecasts are new, its 2013-2018 forecasts are the existing ones).
5. **V5 volatility-managed exposure** (secondary, the team's M3): weight w_t = min(cap, c / var_hat_t) decided at the close of t, held open t+1 to open t+2; c scales the managed return to the buy-and-hold volatility on all data from 2013-01-01 to the day before the period (the first TimesFM forecast is on that date, so every signal uses the same history). Variance signals: the squared TimesFM spread at h=1 ((q90 - q10) / (2 x 1.2816))^2, the VIX ((VIX/100)^2 / 252) and 20-day realised variance. 5 bps per unit of turnover (and 0 bps). The cap, 1.5 or 2.0, is chosen per signal on F2 and VAL only (below).

## Selection rule (selection folds only)

Folds F2, VAL (F2 2015-16, VAL 2017-18), as in the drift study: the cached TimesFM forecasts start on 2013-01-02, so the team's fold F1 has no earlier scores to calibrate on. The counterpart of every config in a fold is the *static official* TimesFM frame of that fold. Rule per horizon and family (V1-V3): eligible if delta BA > 0 and delta AUC > 0 in both folds; take the eligible config with the highest mean(delta BA) + mean(delta AUC); if none is eligible the best-ranked config is taken anyway and flagged (`eligible` below). Grid: V1 8 configs (Platt / offset x window 252, 504 x cut 0.8, 0.9), V2 4 (median / tercile x lookback 504 / all), V3 8 (C 0.03, 0.3 x fixed, expanding, rolling 252, rolling 504). All configs are in `results/timesfm_vix_validation_log.csv`.

## Chosen configs

| h | method | config | spec | eligible | mean dBA | mean dAUC | eligible / configs |
|---|---|---|---|---|---|---|---|
| 1 | V1 VIX-triggered recalibration | recal-5 | offset only on the last 252 known labels; refit when the VIX rank crosses 0.9 (min gap 21 rows) | False | +0.0100 | +0.0053 | 0 / 8 |
| 1 | V2 VIX-regime selection | select-3 | tercile split of the VIX rank, lookback all since the first expert row | False | +0.0072 | -0.0053 | 0 / 4 |
| 1 | V3 VIX-conditioned stacking | stack-4 | LR C=0.3 on [logit score, VIX rank, score x (rank - 0.5)]; fixed (fit once) | False | -0.0310 | -0.0245 | 0 / 8 |
| 5 | V1 VIX-triggered recalibration | recal-5 | offset only on the last 252 known labels; refit when the VIX rank crosses 0.9 (min gap 21 rows) | False | +0.0088 | +0.0191 | 0 / 8 |
| 5 | V2 VIX-regime selection | select-2 | tercile split of the VIX rank, lookback 504 | False | -0.0037 | -0.0333 | 0 / 4 |
| 5 | V3 VIX-conditioned stacking | stack-7 | LR C=0.3 on [logit score, VIX rank, score x (rank - 0.5)]; rolling 504, refit every 21 | False | +0.0018 | -0.0185 | 0 / 8 |

V5 caps (mean over F2 and VAL of Sharpe(managed) - Sharpe(buy-and-hold) at 5 bps): tfm_spread: cap 2.0 (1.5: -0.263, 2.0: -0.250); vix: cap 2.0 (1.5: -0.200, 2.0: -0.097); rv20: cap 1.5 (1.5: +0.024, 2.0: +0.021).

## Test (2019-2025, run once)

Exactly the TimesFM study's rows: `Dataset(h)` TEST rows, 1,759 for h=1 and 1,755 for h=5 (asserted; the static frame is checked equal to the saved `results/timesfm_test_predictions.pkl`, and so is the fixed LR). The test period has no effect on any choice; no config is changed after the numbers are seen, anything later is exploratory and labelled so.

**Metrics:** ROC-AUC, balanced accuracy (BA), Brier. **Inference:** paired moving-block bootstrap (`common.paired_bootstrap`, 2000 draws, block length max(20, 3h), seed 42) of method minus counterpart, 95% percentile CI, one-sided p = P(delta* <= 0). Holm within each horizon.

**Primary family** (V1-V3): each of the 3 methods versus (a) the **static official TimesFM** frame and (b) the team's **fixed LR**, on BA and AUC: 12 tests per horizon, Holm over those 12. **Isolating family** (V1-V3): each method versus its VIX-free twin (scheduled refits / regime-blind selection / stack without VIX terms), BA and AUC: 6 tests per horizon, Holm over those 6. **V4**: official `multi` minus no-VIX on BA and AUC, Holm over those 2 per horizon (positive = the VIX in the input helps); secondary: SPY + VIX only minus SPY alone, Holm over 2. **V5**: Sharpe(managed) - Sharpe(buy-and-hold) at 5 bps for each signal (Holm over the 3 signals) and the TimesFM-spread signal minus the VIX signal, and minus rv20, directly (paired moving-block bootstrap of the Sharpe difference, block 20, 2000 draws); reported at 5 and 0 bps, max drawdown included. Descriptive only (no Holm): each method versus chance.

## Win rule

* A VIX method (V1-V3) **helps TimesFM** at a horizon if its Holm-adjusted p < 0.05 against the static TimesFM frame on AUC or on BA, with a non-negative point estimate on the other.
* It is **useful** only if, in addition, (i) it is not significantly worse than the fixed LR (the 95% CI of method minus fixed LR does not lie entirely below 0 on AUC or on BA) and (ii) the isolating comparison shows the gain comes from the VIX: Holm-adjusted p < 0.05 against the VIX-free twin on AUC or on BA, with a positive point estimate on the metric concerned.
* **The VIX in the input helps** (V4) if Holm-adjusted p < 0.05 on AUC or on BA for official minus no-VIX.
* V5 is secondary and has no win rule; it is reported as the Sharpe table. Both horizons are reported, win or not. If no config is eligible on selection, the least-bad one is used and flagged (see the table).

## Caveats

* **The test period is not a pristine hold-out for this question.** TimesFM's 2019-2025 results were already known (AUC 0.482 at h=1, 0.459 at h=5), so were the drift study's results and the team's VIX study's results on the logistic regression, and the team's fixed LR is known to be weakest in high-VIX periods (h=5 AUC 0.489 in high VIX vs 0.536 in mid VIX). The question was asked knowing this. This pre-registration locks the *configs* before their own test metrics exist; it cannot undo that knowledge.
* Possible pretraining leakage of TimesFM-3.0 on SPY's public history (as in the TimesFM study); it is shared by every method and by the static frame.
* Only two selection folds (F2, VAL) and a small grid; selection on two folds is noisy, and for some families no config may be eligible (flagged above).
* The TimesFM expert of V2 is the walk-forward offset-calibrated score, not the once-fitted static frame, so that its out-of-sample history exists from 2014; V2 is compared with the official static frame and, descriptively, with that expert alone.
* V4 changes the model input: the no-VIX forecasts are a different forecast series, so the comparison shows what the covariate does inside the model, not a general statement about the VIX.
* The score approximates the project's label (Open[t+1] to Close[t+h]); the overnight gap is not modelled.
* The VIX methods cannot add information TimesFM does not contain, except through the VIX itself and, for V2, the other experts.

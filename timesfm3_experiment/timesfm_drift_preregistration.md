# Drift methods on frozen TimesFM-3: pre-registration

Choices `results/timesfm_drift_choices.json`  SHA-256: `5b052f2edc30071f03788754126c13e1c72841bc5ece2235fd797ec869272946`

Locked at (UTC): `2026-11-23T01:40:13+00:00`. `timesfm_drift_study.py test` refuses to run if the hash differs, and the timestamps of every step are in `results/timesfm_drift_timestamps.csv` (`select_start` < `choices_written` < `prereg_locked` < `test_start`). Written after all selection runs and **before any 2019-2025 metric of a drift method was computed**: `select` loads no TimesFM score after 2018-12-31 (`Inp(h, upto=...)`) and evaluates only the folds F2, VAL.

## What is tested

TimesFM-3.0 stays frozen and zero-shot (no retraining, no fine-tuning, no model call: the cached forecasts of the official setup, variant `multi`, context 256, are read). Four families of outer layers act on its daily P(up) score z = logit(score):

1. **recalibration** (`platt`): logistic recalibration of z (Platt: slope + offset, the slope may go negative = flip the direction; or offset only) on the last N known labels, expanding or rolling, refit every R rows;
2. **stacking** (`stack`): logistic regression (C=0.03, standardised) on TimesFM-derived features, fixed (fit once at the period start), expanding or rolling, as in the project's walk-forward experiment;
3. **Hedge** (`hedge`): exponentially weighted ensemble over three fixed experts {TimesFM calibrated (official static frame), the team's fixed LR, base rate}; weights from delayed, discounted log loss (`methods.run_hedge` logic); the thresholds are weighted the same way;
4. **detector** (`detect`): Page-Hinkley or ADWIN on TimesFM's own delayed log loss / error; when it fires (at least 21 rows since the last refit) the Platt calibration is refit on a recent window (`methods.run_detect` logic).

Every label-fitted step uses only labels with `label_end <= decision date` (`Dataset.n_known`, asserted), thr = positive rate of the training window, pred = proba >= thr. The logic mirrors `experiments/methods.py`; its functions need a feature matrix, so they are re-implemented on the score column in `timesfm_drift_lib.py` (detectors, loss and Dataset are imported unchanged).

## Selection rule (selection folds only)

Folds F2, VAL (F2 2015-16, VAL 2017-18). **Deviation from the team's three folds:** the cached TimesFM forecasts start on 2013-01-02, so F1 (2013-14) has no earlier scores to recalibrate on and no static counterpart; adding pre-2013 forecasts would need new model calls, which this study does not make. The counterpart of every config in a fold is the *static official* TimesFM frame of that fold (offset fitted once on the purged rows before the fold, threshold = their positive rate). Rule per horizon and family: eligible if delta BA > 0 and delta AUC > 0 in both folds; take the eligible config with the highest mean(delta BA) + mean(delta AUC); if none is eligible the best-ranked config is taken anyway and flagged (`eligible` below). Grid (34 configs per horizon): recalibration 12 (Platt / offset x window 252, 504, expanding x refit 21, 63), stacking 8 (2 feature sets x fixed, expanding, rolling 252, rolling 504), Hedge 6 (eta 20, 100, 500 x loss half-life 63, 252), detector 8 (PH lambda 0.5 / 2, ADWIN delta 0.002 / 0.05 x window 504 / expanding). All configs are logged in `results/timesfm_drift_validation_log.csv`.

## Chosen configs

| h | family | config | spec | eligible | mean dBA | mean dAUC | eligible / configs |
|---|---|---|---|---|---|---|---|
| 1 | recalibration | platt-7 | offset, window 252, refit every 63 rows | False | -0.0011 | +0.0055 | 0 / 12 |
| 1 | stacking | stack-4 | LR C=0.03 on logit score + median return + q90-q10 spread; fixed (fit once) | False | -0.0045 | +0.0199 | 0 / 8 |
| 1 | Hedge | hedge-1 | eta=20, loss half-life 252 | False | -0.0005 | -0.0047 | 0 / 6 |
| 1 | detector | detect-0 | Page-Hinkley (delta=0.005, lambda=0.5) on log loss; Platt refit window 504 | False | -0.0135 | -0.0029 | 0 / 8 |
| 5 | recalibration | platt-6 | offset, window 252, refit every 21 rows | False | +0.0194 | +0.0118 | 0 / 12 |
| 5 | stacking | stack-7 | LR C=0.03 on logit score + median return + q90-q10 spread; rolling 504, refit every 21 | False | +0.0095 | -0.0129 | 0 / 8 |
| 5 | Hedge | hedge-1 | eta=20, loss half-life 252 | False | -0.0014 | -0.0048 | 0 / 6 |
| 5 | detector | detect-5 | ADWIN (delta=0.002) on 0/1 error; Platt refit window expanding | False | -0.0057 | +0.0000 | 0 / 8 |

## Test (2019-2025, run once)

Exactly the TimesFM study's rows: `Dataset(h)` TEST rows, 1,759 for h=1 and 1,755 for h=5 (asserted, and the static frame is checked equal to the saved `results/timesfm_test_predictions.pkl`). The test period has no effect on any choice; no config is changed after the numbers are seen, anything later is exploratory.

**Primary comparisons**, for each of the 4 chosen methods and each horizon: (a) vs the **static official TimesFM** frame (offset fitted once on 2013-2018, never updated); (b) vs the project's **fixed LR** (`LogReg C=0.003`, 23 core features, fitted once; frames reused from `results/timesfm_test_predictions.pkl`). Descriptive only (not in the Holm family): each method vs chance.

**Metrics:** ROC-AUC, balanced accuracy (BA), Brier. **Inference:** paired moving-block bootstrap (`common.paired_bootstrap`, 2000 draws, block length max(20, 3h), seed 42) of method minus counterpart, 95% percentile CI, one-sided p = P(delta* <= 0). **Holm** within each horizon over the 16 tests {4 methods} x {2 counterparts} x {BA, AUC}.

A method **wins** only if its Holm-adjusted p < 0.05 against **both** counterparts on BA or on AUC, with a non-negative point estimate on the other of the two. Both horizons are reported, win or not. Exploratory (marked as such): per-year AUC 2019-2025 of the chosen methods.

## Caveats

* **TimesFM's 2019-2025 results were already seen** (the TimesFM study: AUC 0.482 at h=1, 0.459 at h=5), as were the fixed LR's. This pre-registration locks the *drift methods'* configs before their own test metrics exist, but the test period is not a pristine hold-out for the project: the choice of asking whether drift handling could rescue TimesFM was made knowing that TimesFM's static score did not work on it (cf. the similar caveat of the VIX study).
* Possible pretraining leakage of TimesFM-3.0 on SPY's public history, as in the TimesFM study; it does not affect the comparison of the drift methods with the static score, which share it.
* Only two selection folds (F2, VAL) and a small grid; selection on two folds is noisy, and for most families no config may be eligible (flagged above).
* The score approximates the project's label (Open[t+1] to Close[t+h]); the overnight gap is not modelled.
* The drift layers cannot add information TimesFM does not contain, except through the other experts (Hedge: the fixed LR, the base rate) and the two extra stacking features.

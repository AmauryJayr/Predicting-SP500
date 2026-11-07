# Chronos-2 pre-registration: SPY direction, h = 1 and h = 5

Written after all validation runs (folds F1 = 2013-2014, F2 = 2015-2016, VAL = 2017-2018) and **before any 2019-2025 Chronos prediction was produced**.
Frozen specification: `chronos_experiment/results/chronos_shortlist.json` (SHA-256 `336ce5edcd939de5cd9a46b0b1c5d63efcad0642b9dd8247a02709a2e3997b6e`). `chronos_experiment/chronos_study.py test` refuses to run if the hash differs, and refuses to run a second time.

## Model and setup

* `amazon/chronos-2`, zero-shot (no training or fine-tuning on our data), 21 quantile levels.
* Forecast series: g[t] = log(Close[t] / Open[t-h+1]), known at the close of t; g[t+h] equals the label's forward log return. P(up) = 1 - F(0) from the step-h forecast quantiles.
* Threshold: base rate of labels known at the period start (same rule as the trained-once models).
* Search space: 3 input variants (`uni` = SPY only; `joint` = SPY, QQQ, IWM, TLT, GLD jointly + log VIX; `core` = SPY + the 23 CORE_FEATURES as past covariates) x 4 context lengths (256, 512, 1024, 2048) = 12 per horizon. Every validation score is in `chronos_experiment/results/chronos_validation_log.csv`.

## Selection rule (fixed in code before validation was run)

Eligible if BA > 0.5 and AUC > 0.5 in at least 2 of 3 folds, and AUC > 0.5 on VAL. Rank by mean (BA - 0.5) + (AUC - 0.5). If none is eligible, take the best by the same score and mark it ineligible.

**No configuration was eligible at either horizon.** The fallback picks are:

| h | variant | context L | val mean dBA | val mean dAUC | eligible |
|---|---|---|---|---|---|
| 1 | joint | 512 | -0.0005 | -0.0087 | no |
| 5 | core | 256 | +0.0024 | -0.0094 | no |

A Platt recalibration of each pick was fitted on VAL predictions only: h = 1 slope 0.320, intercept 0.014; h = 5 slope **-0.092** (negative), intercept 0.528. The h = 5 calibrator reverses the ranking, so the `chronos_platt` row at h = 5 will have AUC = 1 - AUC of the raw forecast. It is reported only as a secondary result.

## Reported comparisons (test 2019-01-02 .. last labelled day of 2025, run once)

For each horizon:
1. Test BA, AUC and Brier for Chronos, Chronos + Platt, chance, and the notebook's trained-once LogReg (all features).
2. Paired moving-block bootstrap (2000 draws, block length max(20, 3h), seed 42), Δ = A - B (Brier sign flipped):
   * **Confirmatory:** Chronos vs chance; Chronos vs notebook LogReg fixed.
   * Secondary: Chronos + Platt vs chance.
3. Holm correction over the confirmatory p-values within each horizon (2 comparisons x BA and AUC).
4. Per-year BA and AUC, 2019-2025.

A confirmed win requires a Holm-adjusted p < 0.05 on BA or AUC, with the point estimate on the other metric not negative. All results are reported, wins or not. Running every variant x context on the test period (`chronos_experiment/chronos_study.py explore`) is done only after this test and is labelled exploratory.

## Caveat

Chronos-2's pretraining corpus may include financial series that overlap 2019-2025. We cannot rule this out, so any positive result would need that qualification.

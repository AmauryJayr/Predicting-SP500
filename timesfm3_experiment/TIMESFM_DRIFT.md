# Drift-aware methods on frozen TimesFM-3: does it help?

Zero-shot Google TimesFM-3.0 on SPY direction (h=1 and h=5), test 2019-2025, pre-registered and run once. TimesFM is **not** retrained or fine-tuned; the team's drift methods act on top of its daily score. Folder: `timesfm3_experiment/`. Pre-registration: `timesfm_drift_preregistration.md`.

## In short

* We took the team's drift ideas (recalibrate on recent data, refit a small model on a rolling window, mix experts with Hedge, refit when a drift detector fires) and put them on top of the unchanged TimesFM score. Settings were chosen on 2015-2018 only and locked before the 2019-2025 test was run, once.
* **No drift method rescues TimesFM.** All four methods stay within about +-0.05 AUC of 0.5 and none of them beats both the static TimesFM score and the team's fixed LR. After the Holm correction no AUC difference is significant (smallest adjusted p: 0.31 for AUC).
* The nearest thing to a signal is the stacking layer at h=1: AUC 0.524 (static TimesFM 0.482, fixed LR 0.508) and balanced accuracy 0.537. Its balanced-accuracy gain over static TimesFM survives the correction (adjusted p = 0.032), but against the fixed LR it does not (adjusted p = 1.0), the AUC gain over static TimesFM is not significant (CI includes 0), and at h=5 stacking is at 0.489. Not a win under the pre-registered rule.
* The detector never fired in 2019-2025, so that method ranks days exactly as static TimesFM does (same AUC to the last digit). The Hedge ensemble gave only 15-24% of its weight to TimesFM and ended near the base rate.
* The fixed LR stays the best method at h=5 (AUC 0.536 against 0.459 for static TimesFM). Reading: the weakness of TimesFM here is in its ranking of days, and recalibrating or re-weighting its score does not repair a ranking that is weak or inverted in most years. This is what the numbers are consistent with; we did not test why.

## 1. Results (test 2019-2025, run once)

Same rows as the TimesFM study: 1,759 days for h=1 and 1,755 for h=5 (asserted in the code; the static frame is checked equal to the saved one in `results/timesfm_test_predictions.pkl`). AUC = ranking quality (0.5 = coin flip), BA = balanced accuracy (0.5 = coin flip), Brier = squared error of the probabilities (lower is better; a constant base-rate forecast scores about 0.248 at h=1 and 0.238 at h=5).

### 1.1 Levels

**h=1** (n=1,759)

| Method | Config | AUC | BA | Brier | Share of days predicted up |
|---|---|---|---|---|---|
| Static TimesFM (official) |  | 0.482 | 0.483 | 0.2589 | 0.49 |
| Recalibration | platt-7 | 0.487 | 0.477 | 0.2606 | 0.47 |
| Stacking | stack-4 | 0.524 | 0.537 | 0.2476 | 0.66 |
| Hedge | hedge-1 | 0.481 | 0.470 | 0.2494 | 0.37 |
| Detector-triggered | detect-0 | 0.482 | 0.477 | 0.2486 | 0.41 |
| Fixed LR (team) |  | 0.508 | 0.514 | 0.2491 | 0.25 |
| Chance (base rate) |  | 0.500 | 0.500 | 0.2480 | 1.00 |

**h=5** (n=1,755)

| Method | Config | AUC | BA | Brier | Share of days predicted up |
|---|---|---|---|---|---|
| Static TimesFM (official) |  | 0.459 | 0.467 | 0.2612 | 0.54 |
| Recalibration | platt-6 | 0.481 | 0.487 | 0.2686 | 0.50 |
| Stacking | stack-7 | 0.489 | 0.507 | 0.2407 | 0.62 |
| Hedge | hedge-1 | 0.484 | 0.480 | 0.2399 | 0.50 |
| Detector-triggered | detect-5 | 0.459 | 0.463 | 0.2397 | 0.45 |
| Fixed LR (team) |  | 0.536 | 0.536 | 0.2390 | 0.48 |
| Chance (base rate) |  | 0.500 | 0.500 | 0.2380 | 1.00 |

In plain words: at h=1 static TimesFM, recalibration, Hedge and the detector all sit at 0.48-0.49 AUC, i.e. slightly worse than a coin flip. Stacking is at 0.524 and the fixed LR at 0.508. At h=5 static TimesFM (0.459) and the detector (identical, see 1.5) are the worst; the three adaptive layers lift it to 0.48-0.49, still below 0.5; only the fixed LR (0.536) is above 0.5. The "Share of days predicted up" column matters for BA: the stacking layer at h=1 says "up" on 66% of days, while up days are about 55% of the sample (its probabilities are above the 0.542 threshold on most days).

Tiny example for AUC: with 10 up days and 10 down days there are 100 up/down pairs; AUC 0.52 means the model gives the up day the higher score in about 52 of them (a coin flip gives 50, 0.48 gives 48).

### 1.2 Primary comparisons

**Method minus static official TimesFM** (positive = method better; Brier sign flipped; CI = 95% block bootstrap; p one-sided, raw / Holm over 16 tests per horizon)

| h | Method | dAUC [CI] | p / Holm | dBA [CI] | p / Holm | dBrier [CI] |
|---|---|---|---|---|---|---|
| 1 | Recalibration | +0.006 [-0.003, +0.015] | 0.103 / 1.000 | -0.006 [-0.020, +0.008] | 0.782 / 1.000 | -0.0018 [-0.0035, +0.0001] |
| 1 | Stacking | +0.043 [-0.003, +0.090] | 0.037 / 0.562 | +0.054 [+0.016, +0.091] | 0.002 / 0.032 | +0.0113 [+0.0066, +0.0164] |
| 1 | Hedge | -0.001 [-0.012, +0.008] | 0.621 / 1.000 | -0.013 [-0.031, +0.003] | 0.937 / 1.000 | +0.0095 [+0.0062, +0.0133] |
| 1 | Detector-triggered | +0.000 [+0.000, +0.000] | 1.000 / 1.000 | -0.005 [-0.017, +0.007] | 0.807 / 1.000 | +0.0102 [+0.0060, +0.0150] |
| 5 | Recalibration | +0.022 [+0.001, +0.046] | 0.019 / 0.312 | +0.020 [-0.007, +0.047] | 0.074 / 1.000 | -0.0074 [-0.0142, +0.0005] |
| 5 | Stacking | +0.030 [-0.050, +0.113] | 0.215 / 1.000 | +0.040 [-0.026, +0.108] | 0.113 / 1.000 | +0.0204 [+0.0068, +0.0349] |
| 5 | Hedge | +0.025 [-0.004, +0.056] | 0.042 / 0.630 | +0.013 [-0.016, +0.044] | 0.185 / 1.000 | +0.0213 [+0.0121, +0.0308] |
| 5 | Detector-triggered | +0.000 [+0.000, +0.000] | 1.000 / 1.000 | -0.003 [-0.019, +0.013] | 0.659 / 1.000 | +0.0215 [+0.0129, +0.0297] |

**Method minus the fixed LR** (positive = method better; Brier sign flipped; CI = 95% block bootstrap; p one-sided, raw / Holm over 16 tests per horizon)

| h | Method | dAUC [CI] | p / Holm | dBA [CI] | p / Holm | dBrier [CI] |
|---|---|---|---|---|---|---|
| 1 | Recalibration | -0.020 [-0.056, +0.017] | 0.837 / 1.000 | -0.037 [-0.067, -0.005] | 0.988 / 1.000 | -0.0115 [-0.0167, -0.0064] |
| 1 | Stacking | +0.017 [-0.023, +0.059] | 0.215 / 1.000 | +0.023 [-0.008, +0.057] | 0.076 / 1.000 | +0.0016 [-0.0004, +0.0036] |
| 1 | Hedge | -0.027 [-0.058, +0.002] | 0.965 / 1.000 | -0.044 [-0.070, -0.017] | 0.999 / 1.000 | -0.0003 [-0.0016, +0.0010] |
| 1 | Detector-triggered | -0.026 [-0.062, +0.011] | 0.907 / 1.000 | -0.036 [-0.065, -0.007] | 0.993 / 1.000 | +0.0005 [-0.0006, +0.0017] |
| 5 | Recalibration | -0.054 [-0.108, -0.000] | 0.976 / 1.000 | -0.050 [-0.093, -0.005] | 0.985 / 1.000 | -0.0296 [-0.0436, -0.0146] |
| 5 | Stacking | -0.046 [-0.113, +0.024] | 0.898 / 1.000 | -0.030 [-0.085, +0.031] | 0.847 / 1.000 | -0.0017 [-0.0094, +0.0062] |
| 5 | Hedge | -0.052 [-0.092, -0.012] | 0.995 / 1.000 | -0.056 [-0.094, -0.020] | 0.998 / 1.000 | -0.0008 [-0.0033, +0.0017] |
| 5 | Detector-triggered | -0.076 [-0.137, -0.020] | 0.995 / 1.000 | -0.073 [-0.120, -0.028] | 0.999 / 1.000 | -0.0007 [-0.0056, +0.0037] |

**Descriptive: method minus chance** (not in the Holm family; AUC minus 0.5, so the CI is the CI of the AUC itself minus 0.5)

| h | Method | AUC 95% CI | BA minus 0.5 [CI] | raw p (AUC) |
|---|---|---|---|---|
| 1 | Recalibration | 0.487 [0.459, 0.515] | -0.023 [-0.047, +0.003] | 0.806 |
| 1 | Stacking | 0.524 [0.498, 0.549] | +0.037 [+0.015, +0.059] | 0.031 |
| 1 | Hedge | 0.481 [0.450, 0.508] | -0.030 [-0.054, -0.008] | 0.922 |
| 1 | Detector-triggered | 0.482 [0.452, 0.509] | -0.023 [-0.046, -0.001] | 0.898 |
| 5 | Recalibration | 0.481 [0.433, 0.534] | -0.013 [-0.048, +0.026] | 0.736 |
| 5 | Stacking | 0.489 [0.443, 0.543] | +0.007 [-0.028, +0.046] | 0.603 |
| 5 | Hedge | 0.484 [0.437, 0.535] | -0.020 [-0.057, +0.022] | 0.726 |
| 5 | Detector-triggered | 0.459 [0.412, 0.510] | -0.037 [-0.072, +0.000] | 0.948 |

In plain words: each row is "method minus the comparator". A positive number means the method did better. The bracket is the range we would expect from resampling blocks of consecutive days; if it contains 0, the difference could be noise. The two p-values are the raw one-sided p and the Holm-adjusted p (the adjustment accounts for the 16 comparisons we made per horizon, so a result only counts if the second number is below 0.05).

Reading the tables: versus static TimesFM, stacking at h=1 has the only adjusted p below 0.05 (balanced accuracy, +0.054, adjusted p 0.032); its Brier improvement (+0.011) is also clear. At h=5, recalibration and Hedge have raw p of 0.019 and 0.042 on AUC (+0.022, +0.025) but these do not survive the correction (0.31 and 0.63). Versus the fixed LR no drift method is significantly better on any metric (all adjusted p = 1.0): at h=5 every method is 0.05-0.08 worse on AUC (the CIs of Hedge and the detector exclude 0).

A caution on Brier: the detector and Hedge improve Brier by about 0.010 (h=1) and 0.021 (h=5) over static TimesFM, but the detector has exactly the same AUC. That gain therefore comes only from shrinking the probabilities toward the base rate, not from better ranking; the resulting Brier is close to what a constant base-rate forecast scores (0.248 at h=1, 0.238 at h=5).

Worked example for the pre-registered win rule: stacking at h=1 passes "adjusted p < 0.05 vs static TimesFM on BA" (0.032) but fails the same test vs the fixed LR (adjusted p 1.0; its BA is +0.023 higher but the CI [-0.008, +0.057] contains 0). The rule needs both, so the verdict is "no win" for every method at both horizons.

### 1.3 Exploratory: AUC by year

*Exploratory, not pre-registered, no correction; seven years of about 250 days each are noisy.*

**h=1**, AUC by calendar year (exploratory)

| Method | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|
| Static TimesFM (official) | 0.467 | 0.466 | 0.480 | 0.516 | 0.505 | 0.421 | 0.512 |
| Recalibration | 0.456 | 0.474 | 0.478 | 0.505 | 0.506 | 0.427 | 0.499 |
| Stacking | 0.572 | 0.474 | 0.582 | 0.538 | 0.491 | 0.599 | 0.548 |
| Hedge | 0.463 | 0.463 | 0.483 | 0.509 | 0.491 | 0.432 | 0.506 |
| Detector-triggered | 0.467 | 0.466 | 0.480 | 0.516 | 0.505 | 0.421 | 0.512 |
| Fixed LR (team) | 0.502 | 0.515 | 0.504 | 0.493 | 0.463 | 0.498 | 0.515 |

**h=5**, AUC by calendar year (exploratory)

| Method | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|
| Static TimesFM (official) | 0.490 | 0.377 | 0.480 | 0.392 | 0.507 | 0.506 | 0.425 |
| Recalibration | 0.479 | 0.363 | 0.488 | 0.383 | 0.472 | 0.539 | 0.395 |
| Stacking | 0.359 | 0.614 | 0.478 | 0.514 | 0.443 | 0.357 | 0.575 |
| Hedge | 0.462 | 0.434 | 0.465 | 0.429 | 0.518 | 0.470 | 0.397 |
| Detector-triggered | 0.490 | 0.377 | 0.480 | 0.392 | 0.507 | 0.506 | 0.425 |
| Fixed LR (team) | 0.478 | 0.519 | 0.481 | 0.490 | 0.540 | 0.432 | 0.524 |

In plain words: the layers change the years, not the picture. At h=1 stacking is above 0.5 in five of seven years (2019, 2021, 2022, 2024, 2025) but static TimesFM is below 0.5 in four of seven; at h=5 the static score collapses in 2020 (0.38), 2022 (0.39) and 2025 (0.42), and the recalibration, Hedge and detector variants fall in the same years (the detector is the static score). Stacking at h=5 does the opposite of static in 2020 (0.61) but also loses badly in 2019 (0.36) and 2024 (0.36): it flipped the direction at the wrong time.

### 1.4 What each layer did

| h | Layer | Refits or fires in 2019-2025 | Share of fits with a negative slope on the score | Mean slope on the score | Mean Hedge weights (TimesFM / LR / base rate) |
|---|---|---|---|---|---|
| 1 | Recalibration (offset only) | 28 refits | 0% (slope fixed at 1) | 1.00 | |
| 1 | Stacking (fixed, 3 features) | 1 fit | 100% | -0.009 | |
| 1 | Hedge | | | | 0.24 / 0.37 / 0.39 |
| 1 | Detector (Page-Hinkley) | 0 fires | 0% | +0.023 | |
| 5 | Recalibration (offset only) | 84 refits | 0% (slope fixed at 1) | 1.00 | |
| 5 | Stacking (rolling 504, refit every 21 days) | 83 refits | 79% | -0.164 | |
| 5 | Hedge | | | | 0.15 / 0.42 / 0.44 |
| 5 | Detector (ADWIN) | 0 fires | 0% | +0.214 | |

In plain words: the detector settings picked on 2015-2018 never fired in 2019-2025, so that method equals "static TimesFM with a one-time Platt fit". The chosen recalibrations are offset-only (they can move the level, not flip the direction). The stacking layers are the only ones that learned a negative slope on the TimesFM score: the fixed h=1 stack has almost no weight on the score itself (slope -0.009) and its AUC (0.524) comes from the other two features; on test, the q90-q10 spread alone has AUC 0.520 and the median forecast return alone 0.491 at h=1 (exploratory single-feature check, computed after the test, not used for any choice; we did not test why the spread carries information). At h=5 the rolling stack flips the sign of the score in most windows, which did not improve AUC (0.489).

## 2. What was done

### 2.1 Setup

* TimesFM-3.0 frozen and zero-shot; the cached forecasts of the official setup (variant `multi`, context 256) are read, **no model call is made**. The score is the forecast P(up) read off the step-h quantiles (as in `TIMESFM_REPORT.md` section 4).
* Static reference: the official frame (offset-only calibration fitted once on all labels known at the period start; threshold = their up-rate). Fixed LR: the team's `LogReg C=0.003` on the 23 core features, trained once.
* Four layers (code `timesfm_drift_lib.py`, mirroring `experiments/methods.py`; we re-implemented the loops on the score column because the originals need a feature matrix, and import the detectors, the loss and `Dataset` unchanged):
  1. **Recalibration** (`run_platt` logic): logistic recalibration of logit(score) on the last N *known* labels, refit every R days; Platt (slope + offset, the slope may go negative) or offset-only.
  2. **Stacking** (`run_wf` logic): standardised logistic regression (C=0.03) on logit(score), optionally plus the median forecast return and the q90-q10 spread; fixed (fit once), expanding or rolling.
  3. **Hedge** (`run_hedge` logic): weights over {static TimesFM, fixed LR, base rate} from delayed, discounted log loss.
  4. **Detector** (`run_detect` logic): Page-Hinkley or ADWIN on TimesFM's own delayed log loss or error; when it fires (at least 21 days since the last fit) the Platt calibration is refitted on a recent window.
* Strict purging: at decision date t only labels with `label_end <= t` are used (asserted in every fit).

Tiny examples (round numbers, made up):

* *Recalibration:* over the last 252 days TimesFM's average P(up) was 0.50 but 60% of days were up. The offset moves every probability up so the average becomes 0.60. This changes the Brier score and the threshold, but not which days rank above which, so AUC barely moves. Only a Platt slope that turns negative would reverse the ranking.
* *Hedge:* suppose the three experts' recent average log losses are 0.70 (TimesFM), 0.69 (LR) and 0.69 (base rate). With eta=20 the weights are proportional to exp(-20 x 0.01) = 0.82, 1, 1, i.e. about 0.29 / 0.36 / 0.36 after normalising (a weak tilt away from TimesFM). Hedge never has more information than its experts; it only chooses among them.
* *Detector:* Page-Hinkley adds up how much the loss exceeds its running average (minus a small allowance 0.005). If the loss jumps from 0.69 to 0.79 for a week, the sum passes the alarm level lambda=0.5 after about five days and the calibration is refitted.

### 2.2 Choices (selection folds only)

Rule: per horizon and family, a config is *eligible* if its balanced accuracy and AUC beat the static TimesFM frame in **both** selection folds; the eligible config with the highest mean(dBA) + mean(dAUC) is taken; if none is eligible the best-ranked config is taken anyway and flagged. Folds: **F2 (2015-16) and VAL (2017-18) only**. F1 (2013-14) is not used (deviation from the team's three folds): the cached TimesFM forecasts start on 2013-01-02, so F1 has no earlier scores to recalibrate on and no static counterpart, and creating older forecasts would need new model calls. Grid: 34 configs per horizon (recalibration 12, stacking 8, Hedge 6, detector 8), logged in `results/timesfm_drift_validation_log.csv`.

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


Selection summary (comparator = static TimesFM in the same fold; each config is evaluated in the 2 folds):

| h | Family | Configs | Config-fold pairs with dAUC > 0 | with dBA > 0 | Best mean dAUC (2 folds) | Mean of all dAUC |
|---|---|---|---|---|---|---|
| 1 | Recalibration | 12 | 7 / 24 | 9 / 24 | +0.006 | -0.013 |
| 1 | Stacking | 8 | 3 / 16 | 5 / 16 | +0.020 | -0.023 |
| 1 | Hedge | 6 | 2 / 12 | 6 / 12 | -0.004 | -0.007 |
| 1 | Detector-triggered | 8 | 5 / 16 | 0 / 16 | -0.001 | -0.007 |
| 5 | Recalibration | 12 | 5 / 24 | 10 / 24 | +0.012 | -0.019 |
| 5 | Stacking | 8 | 3 / 16 | 7 / 16 | +0.000 | -0.032 |
| 5 | Hedge | 6 | 6 / 12 | 3 / 12 | -0.001 | -0.018 |
| 5 | Detector-triggered | 8 | 0 / 16 | 7 / 16 | +0.000 | -0.028 |

In plain words: **no config in any family was eligible at either horizon** (none beat static TimesFM on both metrics in both folds), so every chosen config is the "least bad" one and is flagged. Most config-fold pairs were *worse* than static TimesFM on AUC: the static score was a strong reference in the selection years (AUC 0.571 in F2 and 0.533 in VAL at h=5; 0.496 and 0.505 at h=1), so the 2013-2018 evidence did not favour drift handling in the first place. We ran the test as pre-registered nevertheless, which is why the table above shows almost no gain.

### 2.3 Order of events (evidence that the choices were locked first)

`results/timesfm_drift_timestamps.csv` (UTC): `select_start` 01:40:02, `choices_written` 01:40:13, `prereg_locked` 01:40:13, `test_start` 01:40:25, `test_end` 01:42:36 on the same day. `select` loads no TimesFM score after 2018-12-31, and `test` refuses to run if the SHA-256 of `results/timesfm_drift_choices.json` differs from the one in the pre-registration: `5b052f2edc30071f03788754126c13e1c72841bc5ece2235fd797ec869272946`. After the lock, only the column layout of the timestamps file was repaired (a missing empty `sha256` field on the first row) and the figure label position was moved; no choice or number changed.

## 3. Checked and not checked

Checked:
* Four drift families, each with a small grid, chosen on two selection folds and tested once on the exact TimesFM test rows; every comparison with block bootstrap and Holm.
* Purging (asserted), row match with the earlier study (asserted), the static frame reproduced to the saved one, no 2019+ score in the selection run.

Not checked:
* Whether other TimesFM variants (`price`, `ret`, longer contexts) or other grid points would behave differently; the grid is small and was fixed in advance.
* F1 as a selection fold; scores before 2013; fine-tuning TimesFM (out of scope by design).
* Other adaptive models (GBDT, online SGD, regime models) on top of the score; transaction costs or trading rules; data after 2025.
* Why the q90-q10 spread has some rank information at h=1, and whether it persists (single exploratory check only).
* Power: with about 1,750 overlapping or consecutive daily rows, AUC differences of 0.02 are not distinguishable from noise (the CIs above are 0.05-0.15 wide).

## 4. Caveats

* **TimesFM's 2019-2025 results were already seen** (AUC 0.482 at h=1, 0.459 at h=5, and the fixed LR's), as was the fact that the static score did not work. The pre-registration locks the drift methods' configs before their own test metrics existed, but the test period is not a pristine hold-out for the project, and asking "can drift handling help TimesFM?" came after that knowledge (the VIX study carries the same kind of caveat).
* Possible pretraining leakage of TimesFM-3.0 on SPY's public history (as in `TIMESFM_REPORT.md`); it affects the static score and the layers alike.
* Selection on two folds with an empty eligible set is weak evidence about which config is best; the test numbers for the chosen configs should not be read as "the best the family can do".
* The drift layers use the same labels the fixed LR uses; they add information only through the extra experts (Hedge) or features (stacking).

## 5. Proposed slide text

**Title:** Drift-aware layers on TimesFM: no rescue

* We put the team's drift methods on top of the frozen TimesFM score: recalibration, stacking, Hedge, drift detector. Settings chosen on 2015-2018, test 2019-2025 run once.
* No method beats both the static TimesFM score and the fixed LR; the best, a stacking layer at h=1, reaches AUC 0.524 but is not significant against the fixed LR.
* At h=5 only the fixed LR is above a coin flip (0.536); recalibrating or mixing TimesFM's score does not fix its ranking.

**Speaker notes (about 15 s):** "We asked whether the drift methods from our main experiment can fix TimesFM. We left the model frozen and added four layers on top of its score, chose the settings on 2015 to 2018, and ran 2019 to 2025 once. No layer beats both the plain TimesFM score and our fixed logistic regression. The best one, stacking at one day, gets AUC 0.52, which isn't significant. At five days only the logistic regression is above chance."

## 6. Reproduce

```
cd timesfm3_experiment
PYTHONDONTWRITEBYTECODE=1 python timesfm_drift_study.py select   # choices + pre-registration (already run; refuses to run twice)
PYTHONDONTWRITEBYTECODE=1 python timesfm_drift_study.py test     # one-shot (refuses if outputs exist or the hash differs)
PYTHONDONTWRITEBYTECODE=1 python timesfm_drift_study.py report   # tables + slides/fig/drift_methods.png
```

Files: `timesfm_drift_lib.py`, `timesfm_drift_study.py`, `timesfm_drift_preregistration.md`, `results/timesfm_drift_{choices.json, validation_log.csv, timestamps.csv, test_levels.csv, test_comparisons.csv, test_by_year.csv, test_diagnostics.csv, test_predictions.pkl}`, figure `slides/fig/drift_methods.png`. Logs: `results/timesfm_drift_*.log` (git-ignored).

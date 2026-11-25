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

## 7. Exploratory: why the drift layer raised AUC (computed after the test)

**Status: exploratory, not pre-registered.** Everything below was computed after the 2019-2025 test numbers were known, on the same rows, with no model call (cached forecasts only). Nothing here changes a choice of sections 1-6. Script `timesfm_drift_why.py`; tables `results/timesfm_drift_why_*.csv`. About 170 AUC figures and 11 paired bootstrap contrasts were computed; the CIs are the only inference offered, and no claim of significance should be read into a single one of them. The reproduced AUCs match sections 1.4 and 2 exactly (static 0.482 / 0.459, stack-4 0.524, platt-6 0.481; predictions equal to the saved pickle).

### In short

* **h=1 (stack-4, 0.482 -> 0.524).** The layer throws TimesFM's direction call away (standardised weight on the score -0.003, on the median forecast -0.022) and bets on the width of TimesFM's forecast band: a wider q90-q10 spread gives a higher P(up) (+0.046). The spread is mostly a volatility gauge (Spearman 0.80-0.87 with VIX and 0.81-0.86 with 20-day realised vol, both periods), and "more volatile than usual -> a bit more likely to close up" is the whole effect, worth a few AUC points. We cannot say the spread beats a plain VIX or rv20 layer: stack-4 minus LR on rv20 is +0.017 [-0.001, +0.038], minus LR on VIX +0.018 [-0.007, +0.042], and the spread alone (0.520) reaches nearly all of stack-4. The pattern is positive in 5 of 7 test years, and removing 2020 and 2022 raises it (spread AUC 0.553), so those two years are not what drives it.
* **h=5 (platt-6, 0.459 -> 0.481).** All of the gain is between refit blocks (the within-block ranking is identical pair by pair; 99% of the positive-negative pairs are between blocks) and 105% of it comes from pairs in different calendar years. The offset equals the trailing up-rate minus the trailing mean TimesFM P(up) (correlation 0.9997), and a version with no labels at all, "TimesFM logit minus its own trailing-252 mean", gets the same AUC (0.481). So the layer works by subtracting the slow-moving level of TimesFM's own score, not by learning a base rate (the trailing up-rate alone gets 0.491). Without 2022 the gain is gone (-0.006), and platt-6 is still below 0.5.

### 7.1 h=1: what stack-4 learned

Fit once on 2013-01-02 .. 2018-12-31 (n = 1,510 purged rows, up-rate 0.542). Label (confirmed in `common.Dataset`, and recomputed): up if Close[t+h] / Open[t+1] > 1.

| feature | standardised coef | per raw unit |
|---|---|---|
| logit score | -0.003 | -0.009 |
| median forecast return | -0.022 | -10.7 |
| spread q90-q10 | **+0.046** | +4.76 |
| intercept | +0.167 | |

*Plain words:* the layer is basically one number, the width of the forecast band; the direction call gets a near-zero weight. A wider band by one standard deviation (0.0096 in log price, about 1%) moves the odds of an up day only slightly (+0.046 on the log-odds scale, +1.1 percentage points).

Note: the quantiles are of log price, so the spread is already a relative width; there is no separate "spread scaled by price". The scaled variant below divides it by its own trailing-252 mean.

### 7.2 h=1: single-feature AUCs

| feature | AUC 2013-2018 (fit window) | AUC test 2019-2025 |
|---|---|---|
| TimesFM logit score | 0.497 | 0.482 |
| median forecast return | 0.500 | 0.491 |
| **spread q90-q10** | **0.520** | **0.520** |
| spread / its trailing-252 mean | 0.518 | 0.510 |
| VIX level | 0.513 | 0.506 |
| 20-day realised vol (rv20) | 0.524 | 0.507 |
| stack-4 (fixed layer) | - | 0.524 |

*Plain words:* with the layer's three inputs, only the width has any ranking power, and it is the same in the fit window and in the test (0.520 and 0.520). If you picked each day's rank by "higher spread = more likely up", you would order a random up day above a random down day 52 times in 100.

Per year (test), AUC of the feature:

| feature | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|
| spread | 0.556 | 0.452 | 0.585 | 0.563 | 0.477 | 0.561 | 0.551 |
| VIX | 0.561 | 0.445 | 0.546 | 0.561 | 0.466 | 0.525 | 0.525 |
| rv20 | 0.531 | 0.455 | 0.555 | 0.531 | 0.488 | 0.540 | 0.540 |
| stack-4 | 0.572 | 0.474 | 0.582 | 0.538 | 0.491 | 0.599 | 0.548 |
| static TimesFM | 0.467 | 0.466 | 0.480 | 0.516 | 0.505 | 0.421 | 0.512 |

*Plain words:* in 5 of the 7 years (not 2020, not 2023) the spread, VIX and rv20 all point the same way, and the three move together year by year (they fail in the same two years). Each year has about 250 rows, so a single year's AUC is uncertain by roughly +/-0.04.

### 7.3 h=1: is the spread a volatility proxy?

Spearman correlation of the spread with: VIX 0.867 (2013-18) and 0.801 (2019-25); rv20 0.806 and 0.855; TimesFM score 0.076 and -0.025; median return 0.288 and 0.172. By year the correlation with VIX is 0.62-0.90 and with rv20 0.62-0.92 (`why_h1_spread_correlations.csv`).

Fixed layers fitted once on the same fit window as stack-4 (C = 0.03, standardised), test AUC:

| model | test AUC |
|---|---|
| LR[spread] | 0.520 |
| LR[VIX] | 0.506 |
| LR[rv20] | 0.507 |
| LR[spread + VIX] | 0.518 |
| LR[spread + rv20] | 0.509 |
| stack-4 (score, median, spread) | 0.524 |
| spread residualised on VIX (linear / log) | 0.512 / 0.513 |
| spread residualised on rv20 (linear / log) | 0.521 / 0.518 |

*Plain words:* the spread and the volatility measures are nearly the same ranking of "calm day vs nervous day" (a 0.8-0.9 rank correlation). What is left of the spread after removing VIX still ranks days with AUC 0.512, and after removing rv20 0.521, i.e. the part the spread has beyond them is small and within noise.

Paired moving-block bootstrap (the study's `paired_bootstrap`, 2000 draws, block 20, seed 42), difference in test AUC:

| A | B | A - B | 95% CI | one-sided p |
|---|---|---|---|---|
| stack-4 | LR[rv20] | +0.017 | [-0.001, +0.038] | 0.029 |
| stack-4 | LR[VIX] | +0.018 | [-0.007, +0.042] | 0.082 |
| stack-4 | LR[spread] | +0.005 | [-0.012, +0.024] | 0.277 |
| stack-4 | LR[spread + VIX] | +0.006 | [-0.012, +0.026] | 0.239 |
| LR[spread] | LR[rv20] | +0.012 | [+0.001, +0.025] | 0.020 |
| LR[spread] | static | +0.038 | [-0.002, +0.073] | 0.031 |

*Plain words:* the stack-4 gain over a plain realised-vol or VIX layer is about 2 AUC points, and the CIs touch zero; against the spread alone it is 0.5 point, indistinguishable from nothing. So "the spread is a volatility proxy that carries a small amount of information" fits the data, and "the spread beats VIX" is not established (p-values here are unadjusted, and these are two to six looks at the same 1,759 rows).

### 7.4 h=1: up-rate by quintile and by year

Up-rate by within-period quintile of the feature (about 300 rows in each cell in 2013-18 and 350 in 2019-25, so the standard error of one cell is about 2.7 points):

| spread quintile (low -> high) | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 2013-2018 | 0.536 | 0.500 | 0.525 | 0.587 | 0.560 |
| 2019-2025 | 0.491 | 0.540 | 0.593 | 0.580 | 0.528 |

| VIX quintile (low -> high) | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 2013-2018 | 0.532 | 0.520 | 0.557 | 0.556 | 0.543 |
| 2019-2025 | 0.563 | 0.523 | 0.516 | 0.587 | 0.544 |

(rv20 in the CSV: 0.510, 0.536, 0.540, 0.570, 0.553 and 0.517, 0.568, 0.556, 0.551, 0.540.) With cut points fixed on the fit window, 2019-2025 is dominated by the top bins (749 of 1,759 days above the fit window's top spread quintile, 1,057 above its top VIX quintile): the test period is a higher-volatility world than the fit window.

*Plain words:* the relation is a weak upward slope, not a clean monotone one: in 2019-2025 the up-rate is 49% on the calmest fifth of days, 58-59% in the middle-high fifths, and falls back to 53% in the top fifth. The extremes are a few points apart on 350 days each, i.e. one to three standard errors; this is a small effect that an AUC of 0.52 accurately summarises.

Per year, up-rate on days with spread above vs below the fit window's median (`why_h1_year_uprate.csv`): 2019 0.59 vs 0.47, 2020 0.57 vs 0.64, 2021 0.58 vs 0.49, 2022 0.51 vs 0.00 (only 2 days below), 2023 0.59 vs 0.63, 2024 0.56 vs 0.46, 2025 0.54 vs 0.44.

Test AUC after dropping the stress years:

| sample | spread | VIX | rv20 | stack-4 | static |
|---|---|---|---|---|---|
| all 2019-2025 | 0.520 | 0.506 | 0.507 | 0.524 | 0.482 |
| without 2020 | 0.530 | 0.509 | 0.516 | 0.534 | 0.483 |
| without 2022 | 0.536 | 0.514 | 0.520 | 0.540 | 0.474 |
| without 2020 and 2022 | 0.553 | 0.520 | 0.534 | 0.557 | 0.474 |

*Plain words:* the layer works in the calmer years (2019, 2021, 2024, 2025) and fails in 2020 (the spread sat above its old median on 220 of 253 days, with a crash and a rebound) and 2023; 2022 contributes little because the spread sat above its old median on 249 of 251 days, so there was nothing to rank. The effect is not manufactured by 2020 or 2022; those years dilute it.

### 7.5 h=5: exact decomposition of platt-6's gain

Static and platt-6 scores are both monotone in the logit score within a refit block (platt-6 adds a constant per block). Blocks are the 84 refit blocks of 21 rows inside the test period (offsets re-derived with `fit_cal`; asserted equal to `run_platt`'s probabilities). AUC = concordant pairs / all (positive, negative) pairs (729,806 pairs).

| | all pairs | within-block pairs (1.0% of pairs) | between-block pairs (99.0%) |
|---|---|---|---|
| static | 0.4592 | 0.5219 | 0.4585 |
| platt-6 | 0.4810 | 0.5219 | 0.4806 |
| gain (AUC points) | **+0.0219** | **0.0000** | **+0.0219** |

By calendar structure of the pair: same block 1.0% of pairs, gain 0.0000; same year but different block 13.0%, AUC 0.448 -> 0.439, gain -0.0011; different years 86.0%, AUC 0.460 -> 0.487, gain +0.0230.

*Plain words:* inside one 21-day block the offset shifts every day equally, so the order is unchanged (every pair keeps its verdict, verified). Therefore the whole improvement is created by comparing days from different blocks, and almost entirely days from different years: e.g. a down-day in 2022 now sits lower in the ranking than an up-day in 2024 because the offset in 2022 was lower, not because TimesFM said anything about that day.

### 7.6 h=5: what does the offset track?

Over the 84 refit blocks (`why_h5_blocks.csv`, `why_h5_offset_correlations.csv`):

| pair | Pearson | Spearman |
|---|---|---|
| offset vs trailing-252 up-rate | +0.688 | +0.652 |
| offset vs trailing-252 mean TimesFM P(up) | -0.595 | -0.619 |
| offset vs (trailing up-rate minus mean TimesFM P(up)) | +0.9997 | +0.9995 |
| offset vs the next block's realised up-rate | +0.085 | +0.057 |
| trailing-252 mean TimesFM P(up) vs next block's up-rate | -0.122 | -0.102 |

Test AUC of candidate signals (all known at the decision date; 1,755 rows):

| signal | test AUC |
|---|---|
| static TimesFM | 0.459 |
| platt-6 | 0.481 |
| offset alone (constant within a block) | 0.518 |
| trailing-252 up-rate, refreshed every 21 rows (no TimesFM) | 0.491 |
| trailing-252 up-rate, refreshed daily (no TimesFM) | 0.498 |
| trailing-252 mean TimesFM P(up) | 0.470 |
| TimesFM logit minus its trailing-252 mean (21-row refresh; **no labels**) | 0.481 |
| same, refreshed daily | 0.482 |

Paired bootstrap, platt-6 minus B (difference in test AUC):

| B | platt-6 - B | 95% CI | one-sided p |
|---|---|---|---|
| static | +0.022 | [+0.001, +0.046] | 0.020 |
| TimesFM logit minus its trailing-252 mean | -0.0002 | [-0.016, +0.017] | 0.50 |
| trailing up-rate, 21-row refresh | -0.010 | [-0.052, +0.030] | 0.69 |
| trailing up-rate, daily | -0.017 | [-0.064, +0.028] | 0.77 |
| offset alone | -0.037 | [-0.077, -0.0004] | 0.98 |

*Plain words:* a base rate with no TimesFM at all (0.491-0.498) gets as high as platt-6 (0.481), so the gain is not evidence that TimesFM's daily calls got better. The offset is almost exactly "what the last year did minus what TimesFM expected in the last year". Subtracting TimesFM's own trailing average from its score (no labels used) gives the same AUC (0.481), meaning the useful part of the offset is the minus-TimesFM-average part: the slow level of TimesFM's P(up) is anti-informative, since its trailing mean has AUC 0.470 (sign chosen after seeing the test: reversed 0.529), and removing it helps. Within a block, TimesFM's own ordering has AUC 0.522. An example: after a long stretch of TimesFM saying 66% up, the offset is negative (-0.52 in January 2019, when the trailing mean P(up) was 0.66; -0.19 at the end of 2021), pushing all days down a bit.

### 7.7 h=5: per year

| year | rows | up-rate | AUC static | AUC platt-6 | change within year | mean offset | share of total gain from positives of that year |
|---|---|---|---|---|---|---|---|
| 2019 | 252 | 0.679 | 0.490 | 0.479 | -0.011 | -0.054 | 14% |
| 2020 | 253 | 0.632 | 0.377 | 0.363 | -0.014 | +0.153 | 24% |
| 2021 | 252 | 0.631 | 0.480 | 0.488 | +0.008 | +0.100 | 3% |
| 2022 | 251 | 0.458 | 0.392 | 0.383 | -0.010 | -0.508 | -62% |
| 2023 | 250 | 0.620 | 0.507 | 0.472 | -0.035 | +0.132 | 59% |
| 2024 | 252 | 0.643 | 0.506 | 0.539 | +0.033 | +0.285 | 80% |
| 2025 | 245 | 0.637 | 0.425 | 0.395 | -0.029 | -0.157 | -19% |

Robustness of the pooled gain to dropping a year: all years +0.022 (0.459 -> 0.481); without 2022 -0.006 (0.455 -> 0.449); without 2020 +0.025; without 2022 and 2024 -0.011.

*Plain words:* inside a year the layer makes the ranking worse in 5 of 7 years (it only helps in 2021 and 2024); the pooled gain exists because the layer ranks different years against each other, and 2022 (down year with up-rate 0.458 and the lowest offset, -0.51) drives it: remove 2022 and the +0.022 turns into -0.006. The "share" column is the attribution of the pair gains to the year of the up-day in the pair and sums to 100%; it shows the gain is not spread evenly (a negative share means that year's up-days lost ranking).

### 7.8 Reading of the two layers, and limits

* The two layers raise AUC for different reasons. At h=1 a feature that is a volatility gauge (the forecast band) carries a small, same-sign signal in both periods; at h=5 the recalibration only moves the whole score up or down between blocks, and its gain depends on one year.
* **Neither result shows that TimesFM's direction forecast is informative.** The h=1 layer ignores the direction, the h=5 gain is matched by a rule with no labels and no TimesFM beyond its own average, and platt-6 stays below 0.5 (0.481).
* All differences are of one to three AUC points on 1,755-1,759 overlapping rows; the paired CIs above are the only inference, they were not corrected for the many looks, and the analysis was shaped after the test result was known. Treat the mechanism as a hypothesis consistent with the numbers, not as a tested claim.
* The "sign chosen after seeing the test" entries (the reversed trailing mean, 0.529) are descriptive only.

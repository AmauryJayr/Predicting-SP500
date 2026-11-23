# Does the VIX help TimesFM-3 predict SPY direction?

Zero-shot Google TimesFM-3.0 on SPY direction (h=1 and h=5), test 2019-2025, pre-registered and run once. TimesFM is **not** retrained or fine-tuned. This mirrors the team's VIX study (which was applied to the logistic regression) on the frozen TimesFM score. Folder: `timesfm3_experiment/`. Pre-registration: `timesfm_vix_preregistration.md`.

## In short

* We tried five ways to use the VIX with the unchanged TimesFM: refit the calibration when the VIX regime flips (V1), choose between experts by their record in the same VIX regime (V2), a logistic layer that lets the VIX scale how much the score is trusted (V3), taking the VIX out of the model's inputs (V4), and a volatility-managed SPY position (V5). Settings were chosen on 2015-2018 and locked before the 2019-2025 test, which was run once.
* **The VIX does not help TimesFM.** None of V1-V3 beats the static TimesFM score after the Holm correction (smallest adjusted p: 0.61 for AUC, 0.81 for balanced accuracy), so under the pre-registered rule none "helps", and none is "useful". Their test AUCs are 0.463-0.497 at h=1 and 0.470-0.478 at h=5, against 0.482 and 0.459 for static TimesFM; the fixed LR is 0.508 and 0.536.
* **Removing the VIX from the model's inputs changes almost nothing.** Official `multi` minus the same run without ^VIX: AUC -0.001 at h=1 and -0.003 at h=5 (95% CIs about +-0.015; adjusted p 0.62 at best). The forecasts were re-run with the official settings and reproduce the existing cache exactly (max difference 0).
* The isolating comparisons say the same: scheduled refits, regime-blind selection and a stack without VIX terms are as good as or better than their VIX versions (the VIX-free stack at h=1 has AUC 0.518 against 0.497 with the VIX terms; not significant).
* As a risk signal (V5), TimesFM's predicted spread is worse than the VIX or 20-day realised variance: Sharpe 0.60 against 0.96 and 1.04 at 5 bps (buy-and-hold 0.95). None of the three beats buy-and-hold in Sharpe significantly; all cut the maximum drawdown from -32% to about -14% to -16%, at the cost of return.
* Exploratory, after the test: static TimesFM's AUC is about the same in low, mid and high VIX terciles (0.455-0.497 at both horizons). It does not have a regime in which it works.

## 1. Results (test 2019-2025, run once)

Same rows as the TimesFM study: 1,759 days for h=1 and 1,755 for h=5 (asserted in the code; the static frame and the fixed LR are checked equal to the saved ones in `results/timesfm_test_predictions.pkl`). AUC = ranking quality (0.5 = coin flip), BA = balanced accuracy (0.5 = coin flip), Brier = squared error of the probabilities (lower is better; a constant base-rate forecast scores about 0.248 at h=1 and 0.238 at h=5).

### 1.1 Levels

**h=1** (n=1,759)

| Method | Config | AUC | BA | Brier | Share predicted up |
|---|---|---|---|---|---|
| Static TimesFM (official, with VIX input) | | 0.482 | 0.483 | 0.2589 | 0.49 |
| V1 VIX-triggered recalibration | recal-5 | 0.486 | 0.483 | 0.2610 | 0.42 |
| V2 VIX-regime selection | select-3 | 0.463 | 0.494 | 0.2482 | 0.97 |
| V3 VIX-conditioned stacking | stack-4 | 0.497 | 0.505 | 0.2479 | 0.39 |
| V4 TimesFM without ^VIX input | | 0.483 | 0.490 | 0.2571 | 0.49 |
| Fixed LR (team) | | 0.508 | 0.514 | 0.2491 | 0.25 |
| Chance (base rate) | | 0.500 | 0.500 | 0.2480 | 1.00 |
| *VIX-free twins:* scheduled refits / regime-blind selection / stack on the score alone | | 0.487 / 0.461 / 0.518 | 0.477 / 0.500 / 0.518 | | |

**h=5** (n=1,755)

| Method | Config | AUC | BA | Brier | Share predicted up |
|---|---|---|---|---|---|
| Static TimesFM (official, with VIX input) | | 0.459 | 0.467 | 0.2612 | 0.54 |
| V1 VIX-triggered recalibration | recal-5 | 0.474 | 0.476 | 0.2801 | 0.41 |
| V2 VIX-regime selection | select-2 | 0.478 | 0.497 | 0.2426 | 0.66 |
| V3 VIX-conditioned stacking | stack-7 | 0.470 | 0.511 | 0.2453 | 0.57 |
| V4 TimesFM without ^VIX input | | 0.463 | 0.461 | 0.2583 | 0.54 |
| Fixed LR (team) | | 0.536 | 0.536 | 0.2390 | 0.48 |
| Chance (base rate) | | 0.500 | 0.500 | 0.2380 | 1.00 |
| *VIX-free twins:* scheduled refits / regime-blind selection / stack on the score alone | | 0.472 / 0.504 / 0.494 | 0.486 / 0.520 / 0.490 | | |

### 1.2 Primary comparisons (differences, 95% block-bootstrap CI, Holm-adjusted p)

Difference = method minus counterpart (positive = method better). Holm p is one-sided, adjusted within horizon over the 12 primary tests (3 methods x {static TimesFM, fixed LR} x {AUC, BA}). Only the AUC and BA of the **static** comparison decide the win rule; the fixed LR is the second condition.

**h=1**

| Method | vs | dAUC [CI] | Holm p | dBA [CI] | Holm p |
|---|---|---|---|---|---|
| V1 recalibration | static | +0.005 [-0.001, +0.011] | 0.61 | +0.000 [-0.013, +0.013] | 1.0 |
| V2 selection | static | -0.019 [-0.052, +0.014] | 1.0 | +0.012 [-0.013, +0.036] | 1.0 |
| V3 stacking | static | +0.016 [-0.030, +0.066] | 1.0 | +0.022 [-0.017, +0.063] | 1.0 |
| V1 recalibration | fixed LR | -0.021 [-0.058, +0.015] | 1.0 | -0.031 [-0.059, -0.001] | 1.0 |
| V2 selection | fixed LR | -0.045 [-0.084, -0.006] | 1.0 | -0.019 [-0.039, +0.004] | 1.0 |
| V3 stacking | fixed LR | -0.010 [-0.047, +0.030] | 1.0 | -0.009 [-0.039, +0.025] | 1.0 |

**h=5**

| Method | vs | dAUC [CI] | Holm p | dBA [CI] | Holm p |
|---|---|---|---|---|---|
| V1 recalibration | static | +0.014 [-0.004, +0.035] | 0.76 | +0.009 [-0.016, +0.034] | 1.0 |
| V2 selection | static | +0.019 [-0.032, +0.073] | 1.0 | +0.031 [-0.015, +0.077] | 0.93 |
| V3 stacking | static | +0.011 [-0.069, +0.090] | 1.0 | +0.044 [-0.016, +0.103] | 0.81 |
| V1 recalibration | fixed LR | -0.062 [-0.118, -0.007] | 1.0 | -0.061 [-0.106, -0.018] | 1.0 |
| V2 selection | fixed LR | -0.057 [-0.098, -0.017] | 1.0 | -0.039 [-0.072, -0.003] | 1.0 |
| V3 stacking | fixed LR | -0.065 [-0.124, -0.005] | 1.0 | -0.026 [-0.075, +0.027] | 1.0 |

In plain words: no VIX method is clearly better than the plain TimesFM score at either horizon (the intervals all include 0 and no adjusted p is below 0.6). Several are clearly worse than the fixed LR (intervals below 0), which is what the "useful" condition guards against. The win rule is therefore not met by any method.

### 1.3 Isolating comparisons: is the VIX the reason?

Each method minus its VIX-free twin (Holm over 6 tests per horizon).

| h | Method | twin | dAUC [CI] | Holm p | dBA [CI] | Holm p |
|---|---|---|---|---|---|---|
| 1 | V1 | scheduled refits every 63 rows | -0.001 [-0.009, +0.006] | 1.0 | +0.006 [-0.006, +0.018] | 1.0 |
| 1 | V2 | regime-blind selection | +0.002 [-0.005, +0.009] | 1.0 | -0.006 [-0.014, +0.003] | 1.0 |
| 1 | V3 | stack without VIX terms | -0.021 [-0.049, +0.010] | 1.0 | -0.013 [-0.034, +0.014] | 1.0 |
| 5 | V1 | scheduled refits every 63 rows | +0.002 [-0.022, +0.025] | 1.0 | -0.011 [-0.036, +0.012] | 1.0 |
| 5 | V2 | regime-blind selection | -0.026 [-0.056, +0.002] | 1.0 | -0.023 [-0.050, +0.002] | 1.0 |
| 5 | V3 | stack without VIX terms | -0.024 [-0.056, +0.011] | 1.0 | +0.020 [-0.021, +0.062] | 0.97 |

In plain words: replacing the VIX trigger, the VIX regime or the VIX terms by their VIX-free versions does not lose anything; if anything the VIX versions are a little worse on AUC. The VIX is not what makes any of these layers work, because none of them works.

### 1.4 What each layer did

| h | Layer | What happened in 2019-2025 |
|---|---|---|
| 1, 5 | V1 (offset only, window 252, cut 0.9) | 20 refits on VIX flips; the scheduled twin made 27 |
| 1 | V2 (tercile, all rows) | picked always-up on 1,685 of 1,759 days and the fixed LR on 74; never TimesFM. The regime-blind twin picked always-up every day |
| 5 | V2 (tercile, lookback 504) | always-up 831 days, fixed LR 800, TimesFM 124; the blind twin: fixed LR 892, always-up 863 |
| 1 | V3 (fixed, C=0.3) | slope on the score -0.04 (almost no weight on TimesFM), VIX rank -0.11, score x rank +0.20 (raw units) |
| 5 | V3 (rolling 504, refit every 21) | 83 refits; slope on the score negative in 80% of refits (mean -0.22) |

In plain words: the expert-selection layer mostly stops using TimesFM (at h=1 it never picks it; the always-up and fixed-LR experts had lower Brier on past rows), which is why its h=1 predictions are almost always "up" and its AUC is below 0.5. The stack learns a negative weight on the score, as in the drift study.

### 1.5 V4: the VIX as a model input

Same model, same context (256), same calibration and threshold rule, only the covariate set differs (new forecasts in `results/timesfm_vix_cache/`, the existing caches untouched). Reproduction check: the official `multi` re-run on 64 random 2019-2025 dates and `price` on 16 dates equal the existing caches exactly (max abs difference 0; `results/timesfm_vix_repro.csv`).

| Input | h=1 AUC | h=1 BA | h=5 AUC | h=5 BA |
|---|---|---|---|---|
| SPY + 15 covariates incl. ^VIX (official) | 0.482 | 0.483 | 0.459 | 0.467 |
| SPY + 14 covariates, no ^VIX | 0.483 | 0.490 | 0.463 | 0.461 |
| SPY + ^VIX only | 0.482 | 0.483 | 0.439 | 0.447 |
| SPY alone (`price`) | 0.501 | 0.498 | 0.453 | 0.451 |

Primary (official minus no-VIX; positive = the VIX in the input helps; Holm over AUC and BA): h=1 AUC -0.001 [-0.015, +0.012] (p 1.0), BA -0.008 [-0.024, +0.009] (p 1.0); h=5 AUC -0.003 [-0.020, +0.013] (p 0.62), BA +0.006 [-0.013, +0.026] (p 0.55). Secondary (SPY + ^VIX only minus SPY alone): h=1 AUC -0.018 [-0.038, -0.004], BA -0.015 [-0.036, +0.001]; h=5 AUC -0.014 [-0.042, +0.012], BA -0.005 [-0.033, +0.021] (the h=1 AUC interval is below 0, i.e. the VIX alone was slightly worse than no covariate at all; it is not part of the win rule).

In plain words: the 15-series input including the VIX is no better than the same input without it; the VIX is not what TimesFM's score is built on in a useful way. Differences of this size are within the noise.

### 1.6 V5: volatility-managed SPY (secondary)

Weight w = min(cap, c / estimated variance), decided at the close of t, held from the open of t+1 to the open of t+2; c scales the managed return to the buy-and-hold volatility on the earlier data. Caps chosen on 2015-2018 only (TimesFM spread 2.0, VIX 2.0, rv20 1.5). Test 2019-2025 (1,758 days with a next-day return).

| Strategy (5 bps per unit turnover) | Sharpe | Ann. return | Ann. vol | Max drawdown | Sharpe minus buy-and-hold [CI] |
|---|---|---|---|---|---|
| Buy-and-hold SPY | 0.95 | 17.5% | 18.9% | -32.1% | |
| TimesFM spread (q90-q10 at h=1, squared) | 0.60 | 6.5% | 11.8% | -15.1% | -0.35 [-0.90, +0.29] |
| VIX | 0.96 | 10.4% | 10.9% | -15.9% | +0.01 [-0.45, +0.50] |
| 20-day realised variance | 1.04 | 13.0% | 12.5% | -13.8% | +0.09 [-0.40, +0.65] |

At 0 bps: TimesFM spread 0.76, VIX 1.05, rv20 1.08, buy-and-hold 0.95 (differences to buy-and-hold -0.19, +0.10, +0.13, all CIs include 0; Holm p 0.99 for all three). Direct comparisons of the managed Sharpe ratios at 5 bps: TimesFM spread minus VIX -0.37 [-0.66, -0.00]; TimesFM spread minus rv20 -0.44 [-0.71, -0.15]; VIX minus rv20 -0.08 [-0.36, +0.19].

In plain words: all three signals shrink the position in turbulent times and roughly halve the worst drawdown, but none raises the Sharpe ratio of buy-and-hold in a way that is distinguishable from noise. TimesFM's own spread is the weakest risk signal of the three: it is worse than the VIX and than realised variance, and its lower Sharpe comes with the lowest return. This is a different use of the model from direction forecasting, so the result does not change the direction verdict.

### 1.7 Exploratory (after the test): TimesFM by VIX regime

Not used for any choice. VIX terciles from the VIX level before 2019 (cuts 14.5 and 20.4); AUC, test 2019-2025.

| h | Model | low VIX | mid VIX | high VIX | n (low / mid / high) |
|---|---|---|---|---|---|
| 1 | Static TimesFM | 0.482 | 0.468 | 0.497 | 340 / 764 / 655 |
| 1 | TimesFM without VIX input | 0.499 | 0.466 | 0.496 | |
| 1 | Fixed LR (our reuse of the saved frame, same terciles) | 0.517 | 0.497 | 0.516 | |
| 1 | LogReg fixed (team table on master, its own terciles) | 0.499 | 0.527 | 0.466 | 486 / 722 / 551 |
| 5 | Static TimesFM | 0.455 | 0.463 | 0.460 | 336 / 764 / 655 |
| 5 | TimesFM without VIX input | 0.433 | 0.471 | 0.469 | |
| 5 | Fixed LR (our reuse of the saved frame, same terciles) | 0.459 | 0.530 | 0.569 | |
| 5 | LogReg fixed (team table on master, its own terciles) | 0.520 | 0.536 | 0.489 | 482 / 722 / 551 |

In plain words: static TimesFM is below 0.5 in every regime at both horizons, with no regime where it works. The fixed LR's weakness in high VIX at h=5 (0.489 in the team's table) is not reproduced with our terciles (0.569): the result depends on where the VIX cuts are, and we could not recover the team's cuts (the counts differ), so the regime numbers of the two sources are not comparable. The by-regime values are descriptive and rest on 336-764 rows each; the V3 h=5 low-VIX AUC of 0.344 is the extreme case of what such small cells can show. Per-year AUC of every method is in `results/timesfm_vix_test_by_year.csv`.

## 2. What was done

### 2.1 Setup

* TimesFM-3.0 frozen and zero-shot. V1-V3 and V5 read the cached forecasts of the official setup (variant `multi`, context 256), **no model call**. Only V4 calls the model, and only to re-run it with a different covariate list (`timesfm_vix_ablation.py`, official settings: context 256, default flags, batch 16, 8 threads, CPU).
* Regime variable: `vix_pct_rank_252d` (1-year percentile rank of the VIX close), known at the close of day t. Every label-fitted step uses only labels with `label_end <= decision date` (asserted).
* References: static TimesFM (official frame: offset-only calibration fitted once on the purged rows before the period; threshold = their up-rate) and the team's fixed LR (`LogReg C=0.003`, 23 core features, trained once); the always-up/base-rate chance frame.
* **V1 VIX-triggered recalibration** (the team's M1 on the score): the calibration layer on z = logit(score) (Platt or offset-only) is refit on the last N known labels only when 1[rank >= cut] flips, at least 21 rows after the last refit. Isolating twin: same layer, refit every 63 rows.
* **V2 VIX-regime expert selection** (M2): experts {TimesFM with expanding offset calibration refit every 63 rows, the team's fixed LR (trained once per period), always-up}. Each day take the expert with the lowest Brier on past *resolved* predictions in the same regime (median or tercile split, lookback 504 or all; fewer than 30 rows -> all rows). Twin: same selection ignoring the regime.
* **V3 VIX-conditioned stacking**: standardised logistic regression on [z, rank, z x (rank - 0.5)], fixed / expanding / rolling. Twin: same stack on z only (`timesfm_drift_lib.run_stack`).
* **V4 input ablation**: re-run `multi` without ^VIX (14 covariates), over 2013-2025; secondary: SPY + ^VIX only against SPY alone. Calibration and threshold exactly as the official frame.
* **V5 volatility-managed exposure** (M3): see 1.6. The variance scale constant uses data from 2013-01-02 (first TimesFM forecast) to the day before the period for all three signals, so they share the same history (the team's code uses all earlier data).

Tiny examples (round numbers, made up):

* *Trigger:* the rank crosses 0.9 on a Monday, so TimesFM's calibration is refit on the last 252 days. If the last year had left its average P(up) 0.05 below the up-rate, the refit moves every probability up by about 0.05. The ranking of days is unchanged.
* *Selection:* in the last 300 high-VIX days the experts' Brier scores were 0.26 (TimesFM), 0.25 (LR), 0.25 (always-up), with the LR slightly lower. Today is a high-VIX day, so the LR's prediction is used, not TimesFM's.

### 2.2 Worked example: 2020-03-09 (h=1, real numbers from the test run)

The VIX closed at 54.5, rank 1.00. TimesFM's raw P(up) is 0.608 (logit 0.437). The next day turned out up (y = 1).

* **Static TimesFM:** after the official offset, P(up) = 0.594 against the threshold 0.536, so it predicts up (right).
* **V1:** the chosen config had refit on 2019-01-02, 2019-08-05 and 2020-01-31 (VIX flips on 2020-02-03 and 2020-02-24 were skipped by the 21-row minimum gap). The offset in use is +0.013 (it was -0.225 at the start of the test), giving P(up) = 0.611 against 0.540; predicts up (right).
* **V2:** tercile regime = high. Among the 501 resolved high-VIX days the Brier scores are TimesFM 0.2581, fixed LR 0.2472, always-up 0.2479, so the fixed LR is chosen: P(up) = 0.414 against 0.536 and it predicts down (wrong). Over all past rows, whatever the regime, the Brier scores are 0.2580, 0.2493 and 0.2490, so the regime-blind twin picks always-up instead: on this day the regime did change the choice.
* **V3:** fitted once on 2013-2018: P(up) = 0.2231 - 0.0418 z - 0.1094 rank + 0.1964 z (rank - 0.5), as a logit; on this day P(up) = 0.535 against 0.542, predicts down (wrong).

### 2.3 Choices (selection folds only)

Rule: per horizon and family, a config is *eligible* if its BA and AUC beat the static TimesFM frame in **both** F2 (2015-16) and VAL (2017-18); the eligible config with the highest mean(dBA) + mean(dAUC) is taken; if none is eligible the best-ranked config is taken and flagged. As in the drift study F1 is not used (no TimesFM score exists before 2013-01-02 to calibrate on). Grid: V1 8 configs, V2 4, V3 8, all logged in `results/timesfm_vix_validation_log.csv`.

| h | method | config | spec | eligible | mean dBA | mean dAUC | eligible / configs |
|---|---|---|---|---|---|---|---|
| 1 | V1 | recal-5 | offset only, last 252 labels, cut 0.9 | False | +0.0100 | +0.0053 | 0 / 8 |
| 1 | V2 | select-3 | tercile split, lookback all | False | +0.0072 | -0.0053 | 0 / 4 |
| 1 | V3 | stack-4 | LR C=0.3, fixed | False | -0.0310 | -0.0245 | 0 / 8 |
| 5 | V1 | recal-5 | offset only, last 252 labels, cut 0.9 | False | +0.0088 | +0.0191 | 0 / 8 |
| 5 | V2 | select-2 | tercile split, lookback 504 | False | -0.0037 | -0.0333 | 0 / 4 |
| 5 | V3 | stack-7 | LR C=0.3, rolling 504, refit every 21 | False | +0.0018 | -0.0185 | 0 / 8 |

V5 caps: TimesFM spread 2.0, VIX 2.0, rv20 1.5 (mean Sharpe gain over buy-and-hold in F2 and VAL at 5 bps: -0.25, -0.10, +0.02).

In plain words: **no config in any family was eligible at either horizon**, so every chosen config is the least bad one and is flagged. The selection years already pointed the same way: static TimesFM was a strong reference there (AUC 0.571 and 0.533 at h=5 in F2 and VAL), and most VIX configs were worse than it (config-fold pairs with dAUC > 0: V1 6/16 at h=1 and 4/16 at h=5; V2 2/8 and 1/8; V3 0/16 and 1/16). We ran the test as pre-registered nevertheless. The V4 runs on F2 and VAL (descriptive) also showed no consistent gain from the VIX input (dAUC of no-VIX minus official between -0.017 and +0.005).

### 2.4 Order of events (evidence that the choices were locked first)

`results/timesfm_vix_timestamps.csv` (UTC, 2026-11-23): `ablation_inference_start` 03:48:01, `ablation_inference_end` 04:08:25, `select_start` 04:08:30, `choices_written` 04:08:34, `prereg_locked` 04:08:34, `test_start` 04:08:40, `test_end` 04:10:00. `select` loads no TimesFM score after 2018-12-31, and `test` refuses to run if the SHA-256 of `results/timesfm_vix_choices.json` differs from the one in the pre-registration: `488751dc6cc4f96f1d6583dd45e305d4ef1da821dd6d9d9b04036584bf2bf65b` (re-checked after the test: identical). The V4 ablation forecasts for 2019-2025 existed before the lock, but no score or metric of them was computed before it (the selection run reads them only for F2 and VAL).

After the lock, the following were added without changing any number: the `explore` and `example` commands, a `fit_rows` entry in the V1 diagnostics (used by `example`), and the x-axis labels of the figure.

## 3. Checked and not checked

Checked:
* Three VIX uses on the score, each with a small grid chosen on two folds and tested once on the exact TimesFM test rows; the input ablation with an exact reproduction check of the model; every comparison with block bootstrap and Holm; purging (asserted); row match with the earlier study (asserted); no 2019+ score in the selection run.

Not checked:
* Other VIX features (VIX change, term structure), other regime definitions, other cuts than 0.8/0.9, or finer grids; the grid is small and was fixed in advance.
* F1 as a selection fold; scores before 2013; fine-tuning TimesFM on the VIX (out of scope by design).
* A VIX input in other TimesFM variants (`price` with a VIX covariate at other contexts), or a different covariate subset; V4 removes only ^VIX from the official 15.
* Realistic trading costs beyond 0 and 5 bps; leverage limits; taxes. V5's cap grid has two values.
* Power: with about 1,750 daily rows, AUC differences of 0.02 are not distinguishable from noise (the CIs above are 0.03-0.15 wide), so "no significant difference" does not prove "no effect".

## 4. Caveats

* **The test period is not a pristine hold-out for this question.** TimesFM's 2019-2025 results were already known (AUC 0.482 at h=1, 0.459 at h=5), so were the drift study's and the team's VIX study's results on the logistic regression, and the team's fixed LR was known to be weakest in high-VIX periods (h=5 AUC 0.489 in high VIX against 0.536 in mid VIX, in the team's table). The pre-registration locks the configs before their own test metrics existed; it cannot undo that knowledge.
* Possible pretraining leakage of TimesFM-3.0 on SPY's public history (as in `TIMESFM_REPORT.md`); it is shared by every method and by the static frame.
* Selection on two folds with an empty eligible set is weak evidence about which config is best; the test numbers for the chosen configs are not "the best each idea can do".
* V2's TimesFM expert is the walk-forward offset-calibrated score, not the once-fitted static frame (its out-of-sample history exists from 2014). Its test frame is within 0.001 AUC of the static one (0.4822 against 0.4819 at h=1, 0.4597 against 0.4592 at h=5), so this does not drive the result.
* V4 changes the model's input, so the no-VIX forecasts are a different forecast series; the result says what this covariate does inside TimesFM, not whether the VIX is informative in general. The univariate `price` forecasts for 2019-2025 were computed for this study (2013-2018 are the existing ones).
* The score approximates the project's label (Open[t+1] to Close[t+h]); the overnight gap is not modelled.
* The exploratory regime table depends on the VIX cuts; ours (terciles of the pre-2019 VIX level) differ from the team's.

## 5. Proposed slide text

**Title:** Does the VIX help TimesFM? No.

* We tried the VIX three ways on the frozen TimesFM score (refit on VIX regime changes, pick the expert by VIX regime, a VIX-scaled stacking layer); settings chosen on 2015-2018, test 2019-2025 run once.
* None beats the plain TimesFM score after correction, none beats the fixed LR, and VIX-free versions do as well; test AUC 0.46-0.50 at one day.
* Taking the VIX out of the model's inputs changes nothing (AUC 0.482 vs 0.483 at one day).
* As a risk signal the VIX does better than TimesFM's own spread, but no signal raises the Sharpe ratio over buy-and-hold.

**One-line answer:** No: the VIX does not help TimesFM predict direction, in any of the ways we tried.

**Speaker notes (about 15 s):** "Next we asked whether the VIX can help TimesFM. We tried it three ways on top of the frozen score and once as a model input, chose the settings on 2015 to 2018, and ran 2019 to 2025 once. None of the VIX versions beats the plain TimesFM score, and removing the VIX from the inputs changes nothing. The VIX helps more as a risk signal than TimesFM's own spread, but it still doesn't beat buy-and-hold."

Figure for the slide: `slides/fig/vix_methods.png` (test AUC of static TimesFM, the three VIX layers, TimesFM without the VIX input, and the fixed LR, at both horizons).

## 6. Reproduce

Interpreter as in `TIMESFM_REPORT.md` section 9 (`PY=.../envs/timesfm/bin/python`). Run times on this CPU-only machine:

```
cd timesfm3_experiment
export PYTHONDONTWRITEBYTECODE=1
$PY timesfm_vix_ablation.py repro            # exact-reproduction check of the official forecasts, ~30 s
$PY timesfm_vix_ablation.py run novix        # model calls, 3,269 dates, ~17 min (resumable)
$PY timesfm_vix_ablation.py run vixonly      # ~2 min
$PY timesfm_vix_ablation.py run uni          # ~1 min (2019-2025 only)
$PY timesfm_vix_study.py select              # choices + pre-registration, ~5 s (refuses to run twice)
$PY timesfm_vix_study.py test                # one-shot, ~1.5 min (refuses if outputs exist or the hash differs)
$PY timesfm_vix_study.py explore             # exploratory by-regime table (needs `git` access to origin/master), ~10 s
$PY timesfm_vix_study.py example             # worked example numbers, ~10 s
$PY timesfm_vix_study.py report              # tables + slides/fig/vix_methods.png, seconds
```

Files: `timesfm_vix_lib.py`, `timesfm_vix_study.py`, `timesfm_vix_ablation.py`, `timesfm_vix_preregistration.md`, `results/timesfm_vix_{choices.json, validation_log.csv, timestamps.csv, repro.csv, example.json, test_levels.csv, test_comparisons.csv, test_by_year.csv, test_diagnostics.csv, test_v4.csv, test_strategy.csv, test_predictions.pkl, exploratory_by_regime.csv}`, figure `slides/fig/vix_methods.png`. The new forecast caches are in `results/timesfm_vix_cache/` (git-ignored); logs `results/timesfm_vix_*.log` are git-ignored.

---

## Follow-up: the VIX on top of the drift layer

### Summary

The sections above put the VIX on the raw (static) TimesFM score. The slides now report "TimesFM + drift layer" everywhere (h=1: the stack-4 layer; h=5: the platt-6 recalibration, both chosen in `TIMESFM_DRIFT.md`), so this follow-up repeats the question with that layer as the baseline D0. **Nothing helps.** None of the four VIX variants is significantly better than D0 on AUC or balanced accuracy at either horizon (all Holm p = 1.00 in the primary family). Most differences are slightly negative, and the VIX-removed version of the layer scores as well as D0 (h=1 AUC 0.530 vs 0.524; h=5 0.479 vs 0.481).

**Caveat.** This follow-up was asked after the TimesFM, drift and VIX test results were all known, and the drift layer itself was picked by a rule that was set after its own test. The follow-up has its own hash-locked pre-registration (`timesfm_vixdrift_preregistration.md`), and nothing in it is tuned, but treat the result as exploratory.

### Variants (each changes D0 only through the VIX)

| | h=1 (D0 = stack-4: logit score, median return, q90-q10 spread, C=0.03, fit once) | h=5 (D0 = platt-6: offset-only recalibration, last 252 labels, refit every 21 rows) |
|---|---|---|
| D1 VIX-triggered refit | refit on all known rows when the VIX rank crosses 0.9 (min gap 21 rows), instead of once | same offset layer, refit at the flips instead of every 21 rows |
| D2 VIX-regime selection | experts D0, the team's fixed LR, always-up; lowest Brier on past resolved same-regime rows (tercile split, all history) | same experts; tercile split, lookback 504 rows |
| D3 VIX terms in the layer | stack-4 features + VIX rank + z x (rank - 0.5) | logit p = z + a + b1 rank + b2 z (rank - 0.5), last 252 rows, refit every 21 rows |
| D4 VIX removed from the input | D0 on the forecasts made without ^VIX (cached in `results/timesfm_vix_cache/novix/`, which covered 2013-2025, so no new inference) | same |

Every setting (cut 0.9, min gap 21, split, lookback) was inherited from the VIX study; nothing was selected. The D0 expert of D2 is built walk-forward before 2019 and is the D0 test frame itself on the test rows.

### Results (test 2019-2025, run once; 1,759 rows at h=1, 1,755 at h=5)

| method | h=1 AUC | h=1 BA | h=5 AUC | h=5 BA |
|---|---|---|---|---|
| D0 TimesFM + drift layer | 0.524 | 0.537 | 0.481 | 0.487 |
| D1 + VIX-triggered refit | 0.521 | 0.521 | 0.474 | 0.475 |
| D2 + VIX-regime selection | 0.463 | 0.494 | 0.486 | 0.507 |
| D3 + VIX terms in the layer | 0.514 | 0.511 | 0.476 | 0.491 |
| D4 drift layer, VIX removed from input | 0.530 | 0.531 | 0.479 | 0.485 |
| regime-blind selection (D2 twin) | 0.461 | 0.500 | 0.504 | 0.520 |
| team's fixed LR | 0.508 | 0.514 | 0.536 | 0.536 |

Primary family (first minus second; positive = the VIX variant is better; for D4 it is D0 minus D4, so positive = the VIX in the input helps). Difference with 95% block-bootstrap CI and Holm-adjusted p (8 tests per horizon, one-sided p):

| h | comparison | dAUC [95% CI] | Holm p | dBA [95% CI] | Holm p |
|---|---|---|---|---|---|
| 1 | D1 - D0 | -0.004 [-0.014, +0.007] | 1.00 | -0.015 [-0.032, +0.001] | 1.00 |
| 1 | D2 - D0 | -0.061 [-0.104, -0.020] | 1.00 | -0.042 [-0.067, -0.018] | 1.00 |
| 1 | D3 - D0 | -0.011 [-0.030, +0.009] | 1.00 | -0.026 [-0.047, -0.005] | 1.00 |
| 1 | D0 - D4 | -0.005 [-0.016, +0.005] | 1.00 | +0.005 [-0.010, +0.022] | 1.00 |
| 5 | D1 - D0 | -0.008 [-0.033, +0.017] | 1.00 | -0.011 [-0.035, +0.011] | 1.00 |
| 5 | D2 - D0 | +0.005 [-0.049, +0.058] | 1.00 | +0.021 [-0.025, +0.066] | 1.00 |
| 5 | D3 - D0 | -0.005 [-0.025, +0.012] | 1.00 | +0.004 [-0.018, +0.026] | 1.00 |
| 5 | D0 - D4 | +0.002 [-0.011, +0.015] | 1.00 | +0.002 [-0.021, +0.022] | 1.00 |

Isolating family (D2 minus the same selection ignoring the VIX regime; 2 tests per horizon): h=1 dAUC +0.002 [-0.005, +0.009] (Holm p 0.69), dBA -0.006 [-0.014, +0.003] (0.91); h=5 dAUC -0.018 [-0.045, +0.008] (1.00), dBA -0.013 [-0.036, +0.010] (1.00). The regime split adds nothing.

Against the team's fixed LR (all in `results/timesfm_vixdrift_test_comparisons.csv`, family `vs_lr`): no layer is significantly above it at either horizon (Holm p = 1.00 for AUC; at h=1 D0 is +0.017 AUC above it, CI [-0.023, +0.059]). At h=5 every TimesFM layer, with or without the VIX, is below it (D0 -0.055 AUC, CI [-0.108, -0.000]).

Diagnostics: D1 refit 20 times at both horizons over seven years. D2 at h=1 chose always-up on 1,685 of 1,759 days and the LR on 74 (D0 on none), so its AUC is mostly that of a near-constant score; at h=5 it chose the LR on 841 days, always-up on 840 and D0 on 74.

F2 + VAL validation numbers (2015-2018; information only, nothing was chosen with them) are in `results/timesfm_vixdrift_validation_log.csv`; they show no consistent pattern either: at h=1 D1 is below D0 on AUC in both folds, while at h=5 D1 is slightly above it in both (+0.013 in F2, +0.001 in VAL) and the test reverses that sign (-0.008).

### In plain words

We took the layer that now sits on top of TimesFM and asked four ways whether the VIX makes it better: refit it when the VIX jumps into or out of a high-fear state, choose the model by VIX state, feed the VIX into the layer, or take the VIX away from TimesFM's inputs and see whether the layer gets worse. The layer does not get better with the VIX, and it does not get worse without it. The small gains seen (for example D2 at h=5, AUC +0.005, BA +0.021) are far inside the noise, and the same selection without the VIX is as good or better.

### Caveats

- Exploratory: asked after the TimesFM, drift and VIX test results were known, and the drift layer was picked by a rule set after its own test. The follow-up's own choices were locked first (hash and timestamps below), but nothing here is a clean out-of-sample confirmation.
- Four variants, two horizons, one test period; the bootstrap intervals are wide (AUC half-widths of about 0.01 to 0.05), so small effects cannot be ruled out.
- The D0 expert history of D2 before 2019 is a walk-forward version of the layer (refit every 63 rows at h=1), not the layer's fit-once rule; on the test rows it is exactly D0.
- D3 at h=5 uses an L2 weight of 1 on the two VIX coefficients; this is a design choice made before the test, not tuned.

### Order of events

Timestamps in `results/timesfm_vixdrift_timestamps.csv` (UTC, 2026-11-23): `lock_start` 06:13:26, `choices_written` 06:13:26, `prereg_locked` 06:13:26, `test_start` 06:13:32, `test_end` 06:14:20. The choices file hash is `e921d7789c941edac400c0e5ebc46686fafb81d3960efdc35d0ba622767d7cd0` (SHA-256), the same value in the pre-registration, the timestamp log and the file on disk; `test` refuses to run if it differs. D0 reproduced the drift study's test frames (probabilities to 1e-9, AUC to 1e-5) before anything else was computed.

### Reproduce

```bash
cd timesfm3_experiment
PY=/home/<user>/miniconda3/envs/timesfm/bin/python
PYTHONDONTWRITEBYTECODE=1 $PY timesfm_vixdrift_study.py lock       # choices + hash-locked pre-registration (refuses to run twice)
PYTHONDONTWRITEBYTECODE=1 $PY timesfm_vixdrift_study.py validate   # F2 + VAL, information only
PYTHONDONTWRITEBYTECODE=1 $PY timesfm_vixdrift_study.py test       # one-shot, ~1 min (refuses if outputs exist or the hash differs)
PYTHONDONTWRITEBYTECODE=1 $PY timesfm_vixdrift_study.py report     # tables + slides/fig/vix_drift_methods.png
```

Files: `timesfm_vixdrift_lib.py`, `timesfm_vixdrift_study.py`, `timesfm_vixdrift_preregistration.md`, `results/timesfm_vixdrift_{choices.json, timestamps.csv, validation_log.csv, test_levels.csv, test_comparisons.csv, test_by_year.csv, test_predictions.pkl}`, `slides/fig/vix_drift_methods.png`. Inputs only read: the forecast caches of the official and no-VIX runs, `results/timesfm_drift_*` and `results/timesfm_vix_choices.json`.

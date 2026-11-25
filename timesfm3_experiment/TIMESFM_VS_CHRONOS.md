# TimesFM-3 vs Chronos-2: SPY direction forecasts

**Exploratory.** This comparison was made after the 2019-2025 test period had been opened by both studies. It is not pre-registered and nothing here is a confirmatory result. It makes **42 paired-bootstrap comparisons plus 6 bootstrap comparisons of drift-layer gains** (48 in all); all intervals are 95 % moving-block bootstrap intervals (`experiments/common.py`, `paired_bootstrap`, 2,000 draws), **uncorrected for multiplicity**. Cached predictions only: neither model was called.

Chronos-2 is the teammate study on branch `origin/chronos` (commit `6334147`, folder `chronos_experiment/`). That branch was not checked out or modified; its prediction pickles were read from a `git archive` extraction (see "How to rerun"). Code: `timesfm_vs_chronos.py`; tables: `results/timesfm_vs_chronos_*.csv`.

## In short

* **No real difference in direction skill.** On the same 1,759 (h=1) and 1,755 (h=5) test days with identical labels, TimesFM minus Chronos AUC is +0.005 [-0.033, +0.043] at h=1 and -0.020 [-0.087, +0.044] at h=5. Both models sit at or just below 0.5. Both CIs contain 0 and the two models' chosen configurations are not a clean comparison anyway (see the confounds below).
* **They make different, almost unrelated bets.** The Spearman correlation of the two daily scores is -0.003 (h=1) and -0.048 (h=5); they make the same up/down call on 47.6 % and 50.1 % of days. Chronos's P(up) is strongly tied to the width of its own forecast (Spearman with its q90-q10 spread -0.72 at h=1, -0.59 at h=5) and to trailing volatility (-0.61 / -0.36); at h=5 it follows the trend (Spearman with the trailing 5-day return +0.60). TimesFM's score follows its own median forecast (+0.95), is unrelated to its spread (-0.03 / -0.07) and to volatility (about 0), and at h=5 leans mean-reverting (-0.23).
* **TimesFM's h=5 deficit is not concentrated in 2020 / 2022 / 2025.** TimesFM is below Chronos in 2020, 2022 and 2023 and above in 2019, 2021, 2024 and 2025; no single year's difference excludes 0 at h=5. Only one of the 14 per-year differences excludes 0 (h=1, 2022: +0.096 [+0.006, +0.188]), about what chance gives over 14 uncorrected intervals.
* **The drift layer's gain is TimesFM-specific at the chosen settings.** Applied to Chronos in the same way, h=1 stack-4 moves Chronos by -0.009 (TimesFM +0.043) and h=5 platt-6 by -0.0003 (TimesFM +0.022). The difference in gains is +0.052 [+0.003, +0.100] at h=1 and +0.022 [+0.007, +0.041] at h=5. Neither gain is evidence that TimesFM's direction call improved (TIMESFM_DRIFT.md section 7): stack-4 puts almost no weight on the TimesFM score (standardised -0.003) and its lift comes from TimesFM's q90-q10 spread, a volatility gauge that Chronos's score already contains (Spearman -0.72); platt-6 keeps the slope at 1, so there is no sign flip, and its lift comes from shifting whole periods (mostly 2022). A pure sign flip appears only in the one-feature `z only` stacks (Table 6), which were not the chosen layers.
* **The causes cannot be separated from these caches.** The two setups differ in at least seven ways at once (Table 1). The only partly controlled cut available (selection folds 2013-2018, TimesFM `price` vs Chronos `uni` at context 256) shows TimesFM ahead at h=5 (+0.045, CI includes 0) and level at h=1, and TimesFM's `multi` advantage of 2013-2018 at h=5 (0.548 vs 0.500) did not carry into the test (0.459 vs 0.479). A controlled re-run is proposed at the end, not run.

## 1. Setup differences

| Item | TimesFM-3 (this folder) | Chronos-2 (`origin/chronos` 6334147) | Confounded? |
|---|---|---|---|
| Model | `google/timesfm-3.0-pytorch`, 330,710,976 parameters, 20 layers, d_model 1280 (`TIMESFM_REPORT.md` section 3) | `amazon/chronos-2`, 120M parameters (`chronos_experiment/README.md`) | Yes: architecture, size and pretraining differ together |
| Input series | log Close of SPY (price levels), the same series for h=1 and h=5; one 5-step forecast serves both horizons (`timesfm_lib.series`) | return series g[t] = log(Close[t] / Open[t-h+1]); a different series per horizon, built so that g[t+h] equals the label's forward return (`chronos_study.ChronosData`) | Yes with model, anchor |
| Covariates | official `multi`: 15 log-Close series as past-only covariates (QQQ, IWM, 9 sector ETFs, TLT, IEF, GLD, VIX) at both horizons | h=1 chosen `joint`: g series of SPY, QQQ, IWM, TLT, GLD forecast jointly + log VIX as past covariate. h=5 chosen `core`: SPY g + the 23 engineered `CORE_FEATURES` (returns, vol, RSI, VIX features, breadth) | Yes: not the same covariate set at either horizon |
| Context (sessions) | 256 (both horizons) | 512 (h=1), 256 (h=5) | Partly: Chronos's own test grid is in Table 2b |
| Quantiles | 9 (q10 ... q90) | 21 (0.01 ... 0.99) | Yes |
| P(up) | 1 - F(log Close[t]), the step-h CDF interpolated through the 9 quantiles, clipped to [0.05, 0.95] | 1 - F(0) of the step-h g forecast, interpolated through 21 quantiles, tails extrapolated to 0.005 / 0.995 | Yes |
| Anchor vs the label | the label is Close[t+h] > Open[t+1]; TimesFM compares with Close[t]. The label differs from the close-to-close sign on 20.3 % of rows at h=1 and 7.7 % at h=5 (all `Dataset` rows, n = 4,528 / 4,524) | exact: g[t+h] is the label's return (asserted in the code) | Yes, but small: TimesFM's raw test AUC against the close-to-close label is 0.492 (h=1) / 0.474 (h=5) vs 0.482 / 0.459 against the real label |
| Calibration, threshold | offset-only logit shift fitted on the purged 2013-2018 labels (slope fixed at 1, AUC unchanged); threshold = base rate | raw P(up), threshold = base rate (AUC is the same raw or calibrated with a positive slope; Chronos's Platt fit had slope 0.32 at h=1 and -0.09 at h=5, not used for the headline AUC) | Not for AUC |
| Choice of config | frozen shortlist `results/timesfm_shortlist.json`: `multi`, 256; mean selection-fold AUC 0.493 (h=1, below 0.5), 0.543 (h=5) | frozen `chronos_shortlist.json`; README states no configuration met its eligibility rule at either horizon (fallback choice) | Different candidate sets and rules |
| Internal scaling | linear detrend + iterative RevIN inside the model (`TIMESFM_REPORT.md` section 3) | not documented in the README; not claimed | unknown |

Plain words: nothing in the two studies was held fixed except the data, the labels, the test days and the bootstrap. If the two models differ, the row that did it cannot be named from these runs. Example: Chronos h=1 uses 512 days of five return series plus VIX, TimesFM uses 256 days of sixteen price series, so a gap could come from the model, the input form, the covariates, the length or any mix of these.

## 2. Head-to-head on the common test days

Labels and dates were checked on 46 prediction frames (both models' chosen test frames, Chronos `uni` L256 test frames, all 16 Chronos test configurations, and the fold frames used in Table 3): same dates as `Dataset(h)` and **0 label mismatches** everywhere.

**Table 2a. Test AUC, 2019-2025** (difference = TimesFM static minus Chronos; positive = TimesFM better)

| h | n | TimesFM `multi` 256 | Chronos chosen | difference [95 % CI] | Chronos `uni` L256 | difference [95 % CI] |
|---|---|---|---|---|---|---|
| 1 | 1,759 | 0.4819 | 0.4765 (`joint` 512) | +0.0053 [-0.0332, +0.0431] | 0.4774 | +0.0045 [-0.0335, +0.0406] |
| 5 | 1,755 | 0.4592 | 0.4786 (`core` 256) | -0.0195 [-0.0868, +0.0443] | 0.4808 | -0.0216 [-0.0823, +0.0334] |

Plain words: the two AUCs differ by about half a point at h=1 and two points at h=5, far inside the noise of 1,700 days. Example: an AUC of 0.482 vs 0.477 means that out of 100 up/down day pairs, the models rank 48.2 vs 47.7 of them correctly.

The `uni` row is not like-for-like: Chronos `uni` has no covariates but still reads g, not price levels, and TimesFM has no cached `price` forecasts after 2018. The like-for-like cut on the selection folds is Table 3.

**Table 2b. Chronos configurations on the test set vs TimesFM static** (post-test, descriptive; no CI)

| h | Chronos AUC range over its 11 (h=1) or 5 (h=5) test configurations | TimesFM minus each |
|---|---|---|
| 1 | 0.464 (`uni` 2048) to 0.488 (`joint` 256) | -0.006 to +0.018 |
| 5 | 0.479 (`core` 256) to 0.499 (`uni` 1024) | -0.040 to -0.019 |

Plain words: at h=1 TimesFM sits inside the spread of Chronos's own configurations; at h=5 it is below every Chronos configuration that has a test cache, by 0.02-0.04, yet the chosen-config interval above still includes 0. Per-configuration values: `results/timesfm_vs_chronos_chronos_grid_test.csv`.

**Table 2c. AUC by test year** (TimesFM / Chronos chosen; difference [95 % CI])

| year | h=1 TimesFM | h=1 Chronos | h=1 diff [CI] | h=5 TimesFM | h=5 Chronos | h=5 diff [CI] |
|---|---|---|---|---|---|---|
| 2019 | 0.467 | 0.449 | +0.018 [-0.086, +0.134] | 0.490 | 0.451 | +0.040 [-0.180, +0.248] |
| 2020 | 0.466 | 0.516 | -0.051 [-0.178, +0.081] | 0.377 | 0.453 | -0.076 [-0.288, +0.132] |
| 2021 | 0.480 | 0.391 | +0.088 [-0.012, +0.162] | 0.480 | 0.344 | +0.136 [-0.042, +0.277] |
| 2022 | 0.516 | 0.419 | **+0.096 [+0.006, +0.188]** | 0.392 | 0.438 | -0.046 [-0.227, +0.165] |
| 2023 | 0.505 | 0.464 | +0.040 [-0.071, +0.150] | 0.507 | 0.523 | -0.016 [-0.241, +0.189] |
| 2024 | 0.421 | 0.460 | -0.039 [-0.147, +0.038] | 0.506 | 0.429 | +0.078 [-0.019, +0.199] |
| 2025 | 0.512 | 0.482 | +0.031 [-0.098, +0.131] | 0.425 | 0.395 | +0.029 [-0.164, +0.240] |

Pooled h=5 difference leaving one year out: without 2019 -0.019, 2020 -0.013, 2021 -0.042, **2022 +0.014**, 2023 -0.013, 2024 -0.033, 2025 -0.023 (h=1: all between -0.005 and +0.014).

Plain words: the sign of the difference flips from year to year and each year has about 250 days, so the intervals are wide. The h=5 deficit does not come from 2020, 2022 or 2025 alone: removing 2022 is the only exclusion that turns it positive, and removing 2021 (when Chronos was at 0.344) makes it larger. Example: in 2021 Chronos's h=5 AUC of 0.344 means it ranked the wrong day higher in about 66 of 100 pairs.

## 3. Like-for-like cut on the selection folds (2013-2018; descriptive)

These are the folds both studies used to choose their configurations, so levels are optimistic and the cut is only a cross-check of representation effects. TimesFM `price` and `ret` caches end in 2018 (no test forecasts), so this is the only place where input form can be compared inside TimesFM.

**Table 3. Pooled F1+F2+VAL AUC (1,510 days), per fold in the CSV `folds_like_for_like`**

| h | TimesFM `price` 256 | TimesFM `ret` 256 | TimesFM `multi` 256 | Chronos `uni` L256 | Chronos chosen |
|---|---|---|---|---|---|
| 1 | 0.496 | 0.459 | 0.497 | 0.487 | 0.509 |
| 5 | 0.544 | 0.482 | 0.548 | 0.499 | 0.500 |

Differences with 95 % CI: `price` - Chronos `uni`: h=1 +0.009 [-0.031, +0.060], h=5 +0.045 [-0.009, +0.117]. `ret` - Chronos `uni`: h=1 -0.028 [-0.062, +0.021], h=5 -0.017 [-0.062, +0.037]. `multi` - Chronos chosen: h=1 -0.012 [-0.047, +0.033], h=5 +0.047 [-0.012, +0.123].

Plain words: on 2013-2018 TimesFM on price levels beat Chronos on returns at h=5 by about 4-5 AUC points (not significant) and lost nothing at h=1; TimesFM's own return-series version (`ret`; at h=5 it reads P(up) from a normal approximation of the summed step quantiles, so it is not a pure input-form swap) was worse than its price-level version in these folds. Then on the test the h=5 order reversed (0.459 vs 0.479). Example: a 0.548 selection-fold AUC for `multi` at h=5 became 0.459 on 2019-2025, a 0.09 drop, while Chronos moved from 0.500 to 0.479.

## 4. Do they make the same bets? (test days, raw scores)

**Table 4a. Agreement of the two daily scores**

| h | Spearman, all days | same up/down call (own base-rate threshold) | same side of own median | up-call rate TimesFM / Chronos | Spearman by year 2019 ... 2025 |
|---|---|---|---|---|---|
| 1 | -0.003 | 47.6 % | 48.2 % | 0.49 / 0.68 | -0.24, -0.18, -0.12, +0.07, -0.02, -0.28, -0.34 |
| 5 | -0.048 | 50.1 % | 50.3 % | 0.54 / 0.63 | -0.29, -0.02, -0.38, -0.36, -0.01, +0.03, -0.42 |

Plain words: the two score series are statistically unrelated overall and mostly negatively related within years (2025: -0.34 and -0.42). Example: on a given day, knowing that TimesFM says "up" tells you nothing about whether Chronos says "up"; averaging the two ranks gives AUC 0.471 (h=1) and 0.453 (h=5), below either model alone, which is what combining two anti-predictive scores would be expected to do.

**Table 4b. What each score follows** (Spearman of the daily score with the trailing quantity at the same close; all test days)

| h | model | trailing 1-day return | trailing 5-day return | trailing 20-day realized vol | q90-q10 spread of its own forecast | median forecast return |
|---|---|---|---|---|---|---|
| 1 | TimesFM | +0.07 | -0.16 | +0.02 | -0.03 | +0.95 |
| 1 | Chronos | -0.06 | +0.11 | -0.61 | -0.72 | +0.84 |
| 5 | TimesFM | -0.05 | -0.23 | +0.02 | -0.07 | +0.95 |
| 5 | Chronos | +0.28 | +0.60 | -0.36 | -0.59 | +0.94 |

(Negative with trailing returns = a bet on reversal; positive = a bet on trend. The last two columns use the log-return units of each model's step-h quantiles.)

Plain words: Chronos at h=5 is a trend follower that gets less bullish when volatility is high; TimesFM is closer to a mild mean-reversion bet that ignores volatility. Example: after a strong 5-day SPY rally, Chronos's h=5 P(up) is typically in its upper range (Spearman +0.60) and TimesFM's typically in its lower range (-0.23). Within years the signs are stable for Chronos at h=5 (+0.47 to +0.73 with the 5-day return every year) and for TimesFM at h=5 negative in all seven years (-0.07 to -0.59; per-year values in `trailing_corr.csv`). On the readout arithmetic: with a positive median, widening the forecast distribution pushes 1 - F(0) down toward 0.5, which is consistent with the -0.59 to -0.72 correlation between Chronos's score and its own spread; this is arithmetic of the readout, not a measured cause.

## 5. Does TimesFM's drift layer help Chronos too?

Method. The two layers chosen in `TIMESFM_DRIFT.md` (h=1 stack-4: fixed logistic regression C=0.03 on logit score + median forecast return + q90-q10 spread, trained once on all 2013-2018 rows; h=5 platt-6: offset-only recalibration, 252-day window, refit every 21 days) were run through the unchanged `timesfm_drift_lib` code on Chronos's chosen configs. Chronos's logit score, median and spread came from its cached 2013-2018 + test frames (the same first row, 2013-01-02, the same purging and thresholds). **Units:** TimesFM forecasts log prices, so its q50[h] - log Close[t] and q90 - q10 are already log returns (from Close[t]); Chronos's quantiles are log returns of g (from Open[t+1]). Both are log-return units, so the same feature construction was used with no rescaling; only the reference point differs. The TimesFM runs were checked equal to the saved `timesfm_drift_test_predictions.pkl` frames.

**Table 5. Test AUC before and after the layer** (headline layer per horizon in bold; change vs the model's own static frame [95 % CI])

| h | layer | TimesFM before | TimesFM after (change) | Chronos before | Chronos after (change) | gain difference TimesFM - Chronos [CI] |
|---|---|---|---|---|---|---|
| 1 | **stack-4 (z + median + spread)** | 0.482 | **0.525 (+0.043 [-0.003, +0.090])** | 0.477 | 0.467 (-0.009 [-0.033, +0.018]) | **+0.052 [+0.003, +0.100]** |
| 1 | stack, z only | 0.482 | 0.518 (+0.036 [-0.018, +0.097]) | 0.477 | 0.477 (0.000) | +0.036 [-0.018, +0.097] |
| 1 | platt-6 offset | 0.482 | 0.486 (+0.004 [-0.005, +0.013]) | 0.477 | 0.482 (+0.005 [-0.006, +0.016]) | -0.001 [-0.010, +0.008] |
| 5 | stack-4 | 0.459 | 0.484 (+0.025 [-0.040, +0.084]) | 0.479 | 0.517 (+0.039 [-0.049, +0.121]) | -0.014 [-0.097, +0.075] |
| 5 | stack, z only | 0.459 | 0.459 (0.000) | 0.479 | 0.521 (+0.043 [-0.045, +0.120]) | -0.043 [-0.120, +0.045] |
| 5 | **platt-6 offset** | 0.459 | **0.481 (+0.022 [+0.001, +0.046])** | 0.479 | 0.478 (-0.0003 [-0.013, +0.013]) | **+0.022 [+0.007, +0.041]** |

Plain words: at the two headline layers the gain is TimesFM's alone, not Chronos's: Chronos does not improve, so the gain is not a generic property of the layer. Example: h=1 stack-4 takes TimesFM from 0.482 to 0.525 and Chronos from 0.477 to 0.467.

**What the layer actually does.** The `z only` stacks are a one-parameter logistic fit on the 2013-2018 score. Their fitted slope is negative for TimesFM at h=1 (-0.045) and for Chronos at h=5 (-0.080), and in exactly those two cells the test AUC becomes 1 minus the static AUC (0.518 = 1 - 0.482; 0.521 = 1 - 0.479): the gain is the ranking being reversed because the 2013-2018 slope was negative. With a positive slope (Chronos h=1: +0.22; TimesFM h=5: +0.20) the AUC does not change at all. So the drift-layer "lift" in both models is a sign flip determined by the pre-2019 fit, which happened to help in 2019-2025 for TimesFM h=1 and Chronos h=5 and would have hurt if the test-period sign had been the opposite. The offset-only platt-6 layer cannot flip the ranking; its AUC change comes only from level shifts between refit blocks (it refits 84 times).

**Table 5b. Single-feature test AUC** (raw, no fitting)

| h | model | logit score | median forecast return | q90-q10 spread |
|---|---|---|---|---|
| 1 | TimesFM | 0.482 | 0.491 | 0.520 |
| 1 | Chronos | 0.477 | 0.469 | 0.506 |
| 5 | TimesFM | 0.459 | 0.462 | 0.512 |
| 5 | Chronos | 0.479 | 0.488 | 0.518 |

Plain words: the spread (a width, not a direction) is the only one of the three at or above 0.5 for both models. In TimesFM it is nearly unrelated to the score (Spearman -0.03 at h=1), so adding it gives the stack something new; in Chronos it is already built into the score (-0.72), so adding it adds little. This is consistent with, but does not prove, why stack-4 helps TimesFM and not Chronos at h=1.

## 6. Conclusion

* **Is there a real difference?** No. Direction AUC differences are +0.005 [-0.033, +0.043] (h=1) and -0.020 [-0.087, +0.044] (h=5), and the models differ in no consistent year. Both are at or below chance.
* **The models do differ in what they do.** Their scores are uncorrelated (Table 4a), and Chronos reads its own forecast width and recent trend while TimesFM reads its median (Table 4b). Two unrelated anti-predictive scores do not combine into a better one (AUC 0.471 / 0.453).
* **What could explain the structural difference?** Candidate causes that this data cannot separate: (1) the input representation (price levels with a built-in detrend and RevIN vs return series), (2) the covariate sets (15 log-price series vs g series + VIX, or engineered features), (3) context length (256 vs 512 at h=1), (4) the readout (9 vs 21 quantiles; reference Close[t] vs the exact label return), (5) the model itself. Scoring TimesFM against the close-to-close label instead of the real label changes its test AUC only by +0.010 (h=1) / +0.015 (h=5), so the anchor mismatch does not explain the sub-0.5 AUC. The selection-fold cut points to input form mattering inside TimesFM (`price` 0.496 / 0.544 vs `ret` 0.459 / 0.482 at h=1 / h=5), but those folds are exactly where the configurations were chosen.
* **Drift layer.** It does not lift Chronos at TimesFM's chosen specs; the lifts that exist are not better direction calls: TimesFM's h=1 stack-4 lift comes from its forecast spread (a volatility gauge) and its h=5 platt-6 lift from period-level shifts, mostly 2022 (TIMESFM_DRIFT.md section 7); only the one-feature `z only` stacks (TimesFM h=1, Chronos h=5) lift by reversing the ranking, so their value hinges on whether the 2013-2018 sign persisted.

**Proposed controlled swaps (not run).** Each isolates one factor and reuses the frozen test protocol:
1. **TimesFM on Chronos's series.** Run TimesFM `multi` or `uni` at context 256 on g[t] = log(Close[t]/Open[t-h+1]) for h=1 and h=5 (and optionally the same 5 joint series), score P(up) = 1 - F(0). This removes input form and the Close/Open anchor at once. About 3,269 forecasts per variant (2013-2025), CPU.
2. **Chronos on TimesFM's input.** Chronos-2 on log Close SPY alone at context 256 with threshold log Close[t], and with TimesFM's 15 log-Close series as past covariates. This removes the representation and covariate differences.
3. **Equalize context and readout.** Chronos `joint` at 256 (cached test AUC 0.488 at h=1) vs 512 is already in Table 2b; extend with matching 9-quantile reads (q10 ... q90 subset of Chronos's 21) to test whether the quantile count matters.
4. Score each with both label definitions (Close[t+h] > Open[t+1] and Close[t+h] > Close[t]); the second needs no new run for TimesFM (0.492 / 0.474 above).
Each re-run adds comparisons; keep a pre-registered primary pair (swap 1 vs swap 2 at each horizon) and report the rest as descriptive.

## How to rerun

```
D=<any empty dir>; mkdir -p $D
git archive origin/chronos chronos_experiment | tar -x -C $D     # only that folder; ~17 MB
cd timesfm3_experiment
OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python timesfm_vs_chronos.py $D/chronos_experiment/results
# or: CHRONOS_RESULTS=$D/chronos_experiment/results python timesfm_vs_chronos.py
```

Runtime about 5 minutes; no model call. Outputs: `results/timesfm_vs_chronos_{label_check,label_anchor,head2head,by_year,chronos_grid_test,folds_like_for_like,same_bets,trailing_corr,combine,drift_layers,drift_gain_diff,drift_features}.csv`. The Chronos pickles are not copied into this repository.

## Limits

* Everything is post-test and exploratory; 48 uncorrected bootstrap comparisons. Expect about two or three intervals to exclude 0 by chance.
* One TimesFM variant (`multi` 256) and Chronos's two chosen configs are compared; the 11 + 5 Chronos test configurations in Table 2b are a post-hoc look.
* The table 3 cut uses the selection folds both studies tuned on.
* The sign-flip explanation in section 5 is read from fitted slopes and the identity AUC = 1 - static AUC in two cells; it does not say why the sign was negative in 2013-2018.

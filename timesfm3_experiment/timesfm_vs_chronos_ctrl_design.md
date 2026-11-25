# Controlled swap: same input for TimesFM-3 and Chronos-2 (design, written before any new forecast)

**Exploratory.** The 2019-2025 test period was already seen by both studies (TIMESFM_VS_CHRONOS.md). Nothing here is
confirmatory; the design is fixed in advance only so that the analysis cannot be tuned to the outcome. Its SHA-256 and the
UTC time it was locked are in `results/timesfm_vs_chronos_ctrl_lock.json`, together with the UTC time of the first TEST
forecast of a new cell (written by the run script). Any later deviation is listed in section 7 of TIMESFM_VS_CHRONOS.md, not
by editing this file.

## Question
The official TimesFM-3 and Chronos-2 setups differ in at least 7 confounded ways and make nearly unrelated SPY direction bets
(score Spearman -0.003 h=1, -0.048 h=5; trend/mean-reversion and volatility links differ). Which factor, the MODEL or the INPUT
series, drives (a) AUC and (b) the behaviour of the bet?

## Phase 1: univariate 2x2, context 256 in every cell, h = 1 and h = 5
* Model: TimesFM-3 (exactly the `timesfm_lib.forecast` settings: `google/timesfm-3.0-pytorch`, 5-step forecast, sorted 9
  quantiles, `make_positive`/symmetric averaging/znorm off, seed 42) or Chronos-2 (`amazon/chronos-2`, exactly the teammate's
  `chronos_study.py` call: `Chronos2Pipeline.predict_quantiles(..., prediction_length, quantile_levels=21 levels)`, batch 64,
  no covariates, `cross_learning` default).
* Input: L = SPY log Close levels (TimesFM's `price` series); G = g_h[t] = log(Close[t]/Open[t-h+1]) (Chronos's `uni` series,
  h-specific, NaN head forward-filled then 0, as in `ChronosData`). g_h[t+h] is exactly the label's log return.
* Same read for every cell: step-h quantiles at levels 0.1..0.9 (9 levels; for Chronos the 9 of its 21), P(up) = `timesfm_lib.p_up`
  formula, 1 - interp(threshold, quantiles, levels, left=0.05, right=0.95); threshold = log Close[t] for L, 0 for G. Chronos
  quantiles pass `np.maximum.accumulate` first (the teammate's guard; TimesFM's are already sorted). Sensitivity: Chronos cells
  also read with all 21 quantiles (the teammate's `prob_up`, threshold as above).
* Periods: 2013-2018 (F1, F2, VAL) and TEST 2019-2025 on `Dataset(h)` rows; dates and labels are checked against `Dataset`.
* Cells: TF x L (existing cache 2013-2018, new TEST), TF x G (new), CH x G (cached `uni` L256, re-read only), CH x L (new;
  one 5-step forecast serves both horizons). New caches go to the git-ignored `results/timesfm_ctrl_cache/`; existing
  caches are read, never modified.
* Order: TEST cells first, then 2013-2018. Reproducibility check before the big runs: ~30 dates of the cached TimesFM `price`
  cell (2013-2018) and ~30 dates of the cached Chronos `uni` L256 TEST cell are re-forecast with this runner and compared.

## Phase 2 (only after phase 1 is complete)
CH x L+15cov: Chronos-2 on SPY log Close with the same 15 log-Close series as TimesFM `multi` (`timesfm_lib.COVS`) as past-only
covariates (Chronos-2's `past_covariates`), context 256, same 9-quantile read, threshold log Close[t]; compared with the cached
TimesFM `multi` L256 (both horizons, TEST primary, 2013-2018 descriptive).

## Pre-specified outputs
(a) TEST AUC per cell and horizon (raw score; the AUC of the raw score equals that of any monotone calibration) with each cell
vs chance. Four contrasts per horizon via `experiments/common.paired_bootstrap` (2,000 moving blocks, block length
max(20, 3h), seed 42): model effect within L (TF x L - CH x L), model effect within G (TF x G - CH x G), input effect within TF
(TF x L - TF x G), input effect within Chronos (CH x L - CH x G). 8 contrasts in all; Holm over these 8 on a two-sided p =
min(1, 2 min(P(draw<=0), P(draw>0))) floored at 1/2001. Cell-vs-chance intervals are reported uncorrected (descriptive).
(b) Behaviour on TEST days: Spearman of each cell's score with trailing 1-day return, trailing 5-day return, 20-day realized
volatility (std of daily log Close changes), and with its own step-h q90-q10 spread; Spearman between the cells' daily scores.
Key question: does the turn-vs-trend sign (score vs trailing 5-day return at h=5) and the volatility link follow the MODEL (same
sign across inputs within a model) or the INPUT (same sign across models within an input)? "Follows" is read from the signs
and magnitudes of those Spearman numbers, not from a test.
(c) Secondary: AUC against the close-to-close label (Close[t+h] > Close[t]); per-year AUC; the same tables for 2013-2018.
Phase 2 adds the contrast TF multi - CH x L+15cov (same paired bootstrap, same Holm family extended by these 2 contrasts only in
a second, separately labelled family) and the same behaviour table.

## Counting
Phase 1 makes 8 contrasts + 8 vs-chance intervals; phase 2 adds 2 contrasts + 4 vs-chance intervals; all post-test, exploratory.

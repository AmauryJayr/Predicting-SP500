# Root cause 2: what does our simple model see that TimesFM-3 does not? (plan)

**Status: exploratory. The test period (2019-2025) was already seen in earlier studies of this folder; nothing here is a
pre-registered confirmatory test and no choice of any earlier study changes.** The plan is written and hashed before any
test number of this study is computed (`results/timesfm_rootcause2_lock.json`). Cached forecasts and the team's code only;
TimesFM and Chronos are not called.

## Protocol (the team's)
- `experiments/common.py` Dataset (h = 1 and h = 5), purging with `n_known`, `make_estimator("LogReg", {"C": 0.003})` +
  `fit_model`, trained ONCE on all rows whose label is known at the first test date (before 2019), applied to TEST
  (2019-01-01 .. 2025-12-31). Threshold = positive rate of the training rows. Same C for every feature subset (not retuned).
- Step 0: reproduce the fixed-LR comparator of `timesfm_drift_lib` (`LR_SPEC`, run through `methods.run_spec`): test AUC must be
  0.508 (h=1) and 0.536 (h=5) to 3 decimals; the subset code path must give the same numbers for the full 23 features.
- Uncertainty: `common.paired_bootstrap` (2,000 moving blocks, block length max(20, 3h), seed 42) for differences; for "AUC vs
  chance (0.5)" the same moving-block resampling on one AUC. Two-sided p from the bootstrap draws (2 * min(P(draw <= null),
  P(draw >= null)) for chance; 2 * min(p_one_sided, 1 - p_one_sided) for paired differences), floored at 1/2001.

## Task 1: information ceiling (feature subsets of the 23 CORE_FEATURES, same LR spec)
Groups (every core feature is in exactly one): SPY short-term = spy_ret_1d, spy_ret_5d, spy_ret_20d, spy_rsi14; SPY long-term /
trend = spy_ret_60d, spy_ret_250d, spy_px_vs_sma50, spy_px_vs_sma200, spy_dist_252d_high; volatility = spy_vol_20d,
spy_vol_ratio_5_60; VIX = vix_level, vix_z60, vix_minus_rv20, vix_log_chg_5d; cross-asset = sector_breadth_above_sma50,
sector_dispersion_5d, corr_spy_tlt_60d, tlt_ret_20d, iwm_minus_spy_ret_20d, xly_minus_xlp_ret_20d; Open-based = spy_overnight_ret,
spy_intraday_ret.
Subsets: (a) all 23; (b) SPY Close-only features computable from a 256-day Close window = short + long + volatility groups (11;
definitions checked in code and listed in the report; spy_rsi14 is an exponentially weighted average computed in the dataset
from the full price history, so it is only approximately a 256-day-window feature, and is flagged); (c) all SPY features = (b) +
Open-based (13); (d) VIX group (4); (e) cross-asset group (6). Test AUC per subset at h=1 and h=5, each vs chance; (a) vs TimesFM
static (the official offset-only calibrated score, `Inp.static`) via paired bootstrap.
**Primary family for Holm correction (12 tests):** 5 subsets vs chance x 2 horizons + full LR vs TimesFM static x 2 horizons.

## Task 2: where the LR's 5-day edge comes from
Drop-one-group ablation of (a) (6 groups x 2 horizons; AUC and paired-bootstrap difference vs the full LR; descriptive,
12 extra comparisons, NOT Holm-corrected, reported with that label). Standardised coefficients of the full LR (both horizons).
Per-year test AUC, full LR vs TimesFM static (descriptive, no tests).

## Task 3: does TimesFM add anything to the core features?
Train once on rows 2013-01-01 .. before 2019 (rows with a TimesFM score, purged, `Inp.train_rows(test_start, None)`): LR on the 23
core features vs LR on 23 core + TimesFM features [logit score, median forecast return, q90 - q10 spread] from `timesfm_drift_lib.Inp`
(same C = 0.003). Paired bootstrap of the AUC difference (2 tests, Holm over these 2; also at h=1 and h=5). Context rows (no
test): TimesFM-3 features only (3 features, C = 0.003) and the core-only model trained on all pre-2019 labels (task 1a). One
sensitivity (C = 0.03, same two models) reported without a test.

## Task 4: same bet or opposite?
Spearman correlation of TimesFM's daily logit score with (i) the full LR's score and (ii) each of the 23 core features, on TEST
and on 2013-2018 (the LR there is the same fixed model, i.e. in sample for that period; labelled so). The LR's own lean on each
feature = Spearman of the LR score with the feature, and its standardised coefficient. "Opposite" is flagged when the two
Spearman values have opposite signs and both |rho| >= 0.10. Descriptive, no tests.

## Count
Counted tests: 12 (primary, Holm) + 12 (ablation, uncorrected, descriptive) + 2 (task 3, Holm over 2) = 26. Per-year and
correlation tables are descriptive. Interpretation limits: one split, one seed, overlapping 5-day labels (block bootstrap),
the 2019-2025 window already seen.

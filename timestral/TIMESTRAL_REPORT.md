# Timestral-209M for SPY direction

Mistral's Timestral is a pretrained forecasting model. Used "as is", without any training on our data, can it predict whether SPY goes up on the project's task? This study repeats the finished TimesFM-3 study (`timesfm3_experiment/`) with Timestral, using the same data, rows, comparisons and win rule, so the two can be compared directly.

**Short answer.** No. On the 2019–2025 test, Timestral is at a coin-flip level at both horizons (AUC 0.480 at h=1 and 0.496 at h=5; AUC 0.5 is a coin flip). It does not beat the project's fixed logistic regression or the "always say up" chance rule at either horizon, so the pre-registered win rule is not met. It is not clearly different from TimesFM-3 either: the one nominal difference (h=5 balanced accuracy, +0.045) was not corrected for multiple comparisons and was not part of the win rule.

The whole study was run by Claude (Anthropic's model) in one unattended session on 2026-11-08, with no human review before the test. The pre-registration was written and hash-locked after validation and before any 2019+ forecast existed (file times in section 6).

---

## In short (easy English)

**The result.** Over 2019–2025:

- Tomorrow (h=1): Timestral AUC **0.480**. The fixed LR gets 0.508, TimesFM-3 gets 0.482.
- Next 5 days (h=5): Timestral AUC **0.496**. The fixed LR gets 0.536, TimesFM-3 gets 0.459.
- No comparison passed the pre-registered test. All four Holm-adjusted p-values at h=1 are 1.0, and at h=5 the smallest is 0.896.

**What was chosen, and one surprise.** The selection rule picked a *returns* variant at both horizons (`ret`, context 1024 for h=1 and 256 for h=5). On 2013–2018, the three folds gave mean AUC 0.470 and 0.469, both *below* 0.5. This is a result of the rule, not a choice we made later: the rule requires an AUC above 0.5 on the last fold (VAL), and only the `ret` cells passed that. All price and multi-variate cells had a VAL AUC below 0.5. Section 3 has the full table.

**What it looks like inside each year (exploratory, after seeing the test).** At h=5 the AUC is below 0.5 in every single year (0.367 to 0.483), yet the pooled 2019–2025 AUC is 0.496. See section 1.6. Timestral's h=5 score also moves *with* SPY's last 20 days (correlation +0.39 to +0.64 in every year), the opposite of TimesFM-3, which bets on a turn (−0.21 to −0.69). Why the model does this was not checked.

---

## 1. Results

### 1.1 What was run

| item | value |
|---|---|
| Model | `mistralai/Timestral-209M`, revision `9874d252ab0aae16f7f7c999f98f98d6ad7d8365`, vendor defaults, CPU, nothing fitted on SPY |
| Grid | 3 variants (`price`, `ret`, `multi`) x 4 context lengths (256/512/1024/2048); `multi` at 2048 dropped on cost (section 4) |
| Selection data | 2013-01-01 to 2018-12-31 (folds F1, F2, VAL; 1,510 forecast dates) |
| Test | 2019-01-02 to the last labelled day of 2025, run once (1,759 rows at h=1; 1,755 at h=5) |
| Selected | h=1: `ret`, ctx 1024. h=5: `ret`, ctx 256 (`results/timestral_shortlist.json`) |

### 1.2 Headline: the 2019–2025 test

Source: `results/timestral_test_levels.csv`.

| h | model | AUC | BA | Brier | accuracy | share of days called "up" | days |
|---|---|---|---|---|---|---|---|
| 1 | **Timestral** | **0.480** | **0.476** | 0.2496 | 0.476 | 0.492 | 1,759 |
| 1 | fixed LR | 0.508 | 0.514 | 0.2491 | 0.491 | 0.253 | 1,759 |
| 1 | chance | 0.500 | 0.500 | 0.2480 | 0.546 | 1.000 | 1,759 |
| 1 | TimesFM-3 | 0.482 | 0.483 | 0.2589 | 0.482 | 0.487 | 1,759 |
| 5 | **Timestral** | **0.496** | **0.512** | 0.2443 | 0.549 | 0.663 | 1,755 |
| 5 | fixed LR | 0.536 | 0.536 | 0.2390 | 0.531 | 0.483 | 1,755 |
| 5 | chance | 0.500 | 0.500 | 0.2380 | 0.614 | 1.000 | 1,755 |
| 5 | TimesFM-3 | 0.459 | 0.467 | 0.2612 | 0.478 | 0.543 | 1,755 |

**In plain words:**

- **Timestral is at the coin-flip line.** Out of 100 random (up-day, down-day) pairs, it puts the up-day first in about 48 pairs at h=1 and about 50 at h=5. The fixed LR does it in about 51 and 54.
- **Don't read the accuracy column on its own.** SPY went up in 61.4% of the 5-day windows, so "always say up" already scores 0.614 accuracy without knowing anything. That is why we judge on AUC and BA, where "always say up" scores exactly 0.5.
- **Brier is slightly worse than chance at both horizons** (0.2496 vs 0.2480 at h=1; 0.2443 vs 0.2380 at h=5), so its probabilities are not better sized than a constant. The TimesFM-3 Brier is worse still (0.2589, 0.2612).

*Tiny example.* On 100 test days at h=5, "always say up" is right on about 61 of them but cannot tell a good day from a bad one (AUC 0.5). Timestral's 0.496 means that when it ranks one up-window and one down-window, it picks the up-window about as often as a coin would.

### 1.3 Comparisons with confidence intervals (pre-registered win rule)

Source: `results/timestral_test_comparisons.csv`. Paired moving-block bootstrap, 2000 draws, block length max(20, 3h), seed 42. Each cell is Timestral minus the counterpart, with a 95% interval. "Holm p" is adjusted over the 4 tests per horizon.

| h | vs | BA difference [95% CI] | Holm p (BA) | AUC difference [95% CI] | Holm p (AUC) |
|---|---|---|---|---|---|
| 1 | chance | −0.024 [−0.048, +0.001] | 1.000 | −0.020 [−0.047, +0.008] | 1.000 |
| 1 | fixed LR | −0.037 [−0.067, −0.004] | 1.000 | −0.028 [−0.057, +0.008] | 1.000 |
| 5 | chance | +0.012 [−0.023, +0.052] | 0.896 | −0.004 [−0.047, +0.043] | 1.000 |
| 5 | fixed LR | −0.024 [−0.065, +0.019] | 1.000 | −0.039 [−0.084, +0.007] | 1.000 |

Win rule (`results/timestral_test_verdict.csv`): a win needs Holm p < 0.05 against both counterparts on BA or on AUC, with a non-negative estimate on the other metric. **Verdict: no win at h=1, no win at h=5.**

**In plain words:**

- **Nothing here is clearly above zero.** The only interval that excludes zero is at h=1 against the fixed LR on BA, and it is negative: Timestral is worse there (−0.037).
- **At h=5 the intervals are wide.** Against chance, the BA difference could be anywhere from −0.023 to +0.052. A year of data holds far fewer independent outcomes than it seems, because 5-day windows overlap.

*Tiny example.* "+0.012 [−0.023, +0.052]" says: the best guess is that Timestral's BA is 0.012 above chance, but a gap of −0.023 (worse) is just as plausible as +0.052 (better).

### 1.4 Descriptive comparison with TimesFM-3 (not part of the win rule, not Holm-corrected)

Source: `results/timestral_test_vs_timesfm3.csv`. Same bootstrap, Timestral minus TimesFM-3, on the same test rows (index equality asserted).

| h | BA difference [95% CI] | one-sided p | AUC difference [95% CI] | one-sided p |
|---|---|---|---|---|
| 1 | −0.006 [−0.043, +0.031] | 0.614 | −0.002 [−0.042, +0.040] | 0.516 |
| 5 | +0.045 [+0.000, +0.094] | 0.025 | +0.037 [−0.025, +0.105] | 0.117 |

**In plain words:**

- **At h=1 the two models are the same.** Differences of 0.006 and 0.002 are far inside the noise.
- **At h=5 Timestral looks a little better on BA, but this is weak evidence.** The BA interval just touches zero (lower end +0.0004), the AUC interval includes zero, and these are extra, uncorrected tests. Most of the gap is TimesFM-3 being below 0.5 at h=5 (AUC 0.459), not Timestral being above it (0.496).

*Tiny example.* If you ran this comparison on 20 unrelated pairs of coin-flip models, you would expect about one to show p ≈ 0.05 by luck alone.

### 1.5 Test years

Source: `results/timestral_test_by_year.csv`. About 250 days per year, so each number moves by 0.1 or more by luck alone (see the intervals in 1.6).

| h | model | metric | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Timestral | AUC | 0.483 | 0.521 | 0.411 | 0.503 | 0.502 | 0.473 | 0.402 |
| 1 | TimesFM-3 | AUC | 0.467 | 0.466 | 0.480 | 0.516 | 0.505 | 0.421 | 0.512 |
| 5 | Timestral | AUC | 0.453 | 0.483 | 0.424 | 0.393 | 0.430 | 0.367 | 0.434 |
| 5 | TimesFM-3 | AUC | 0.490 | 0.377 | 0.480 | 0.392 | 0.507 | 0.506 | 0.425 |
| 1 | Timestral | BA | 0.501 | 0.475 | 0.480 | 0.472 | 0.486 | 0.471 | 0.419 |
| 5 | Timestral | BA | 0.457 | 0.523 | 0.494 | 0.451 | 0.479 | 0.462 | 0.486 |

**In plain words:**

- **No year stands out as good.** At h=5 every Timestral year is below 0.5 in AUC (0.367 to 0.483).
- **The worst Timestral h=1 year is 2025 (AUC 0.402).** The other years are within about 0.1 of 0.5. We did not look into why.

*Tiny example.* "2024, h=5: AUC 0.367" means: out of 100 (up-window, down-window) pairs in 2024, Timestral put the up-window first in about 37.

### 1.6 Exploratory: h=5 inside each year (after seeing the test)

Everything in this section is **exploratory**: it was made after seeing the test, from saved predictions only (no new model run), with `timestral_explore_years.py`. Source: `results/timestral_exploratory_years.csv`. Intervals are 95% block-bootstrap intervals (20-day blocks, 1000 draws), the same recipe as TimesFM's `timesfm_explore_split.py`. TimesFM-3's numbers are recomputed with the same code and checked against its own `timesfm_exploratory_years.csv`.

| | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | all |
|---|---|---|---|---|---|---|---|---|
| share of up-windows | 0.679 | 0.632 | 0.631 | 0.458 | 0.620 | 0.643 | 0.637 | 0.614 |
| Timestral mean P(up) | 0.610 | 0.650 | 0.675 | 0.506 | 0.544 | 0.636 | 0.650 | 0.610 |
| Timestral h=5 AUC | 0.453 | 0.483 | 0.424 | 0.393 | 0.430 | 0.367 | 0.434 | 0.496 |
| 95% interval | 0.35–0.55 | 0.33–0.60 | 0.31–0.50 | 0.30–0.50 | 0.31–0.56 | 0.28–0.47 | 0.29–0.55 | 0.45–0.54 |
| TimesFM-3 h=5 AUC | 0.490 | 0.377 | 0.480 | 0.392 | 0.507 | 0.506 | 0.425 | 0.459 |
| 95% interval | 0.34–0.64 | 0.25–0.51 | 0.38–0.58 | 0.32–0.49 | 0.34–0.68 | 0.43–0.64 | 0.32–0.56 | 0.41–0.51 |
| corr(score, SPY past 20-day change), Timestral | +0.55 | +0.56 | +0.59 | +0.39 | +0.42 | +0.59 | +0.64 | +0.47 |
| same, TimesFM-3 | −0.49 | −0.40 | −0.55 | −0.45 | −0.21 | −0.47 | −0.69 | −0.33 |
| AUC of the rule "bet on a turn" (`−past 20-day change`) | 0.538 | 0.541 | 0.564 | 0.507 | 0.483 | 0.531 | 0.497 | 0.497 |

The figure is `figures/timestral_years_h5.png`.

**In plain words:**

- **Pooled AUC (0.496) is higher than every single year (0.367–0.483).** The table shows one visible part of why: the year-average scores and the year up-rates line up. 2022 has the lowest up-rate (0.458) and also the lowest mean score (0.506). Pooling uses these between-year differences as well, and inside a year they don't count. We did not run a test that splits the two parts; "why" is not checked beyond this table.
- **Timestral follows the recent trend; TimesFM-3 bets on a turn.** In every year the Timestral h=5 score rises with SPY's past 20-day change (+0.39 to +0.64), where TimesFM-3's falls (−0.21 to −0.69). Why Timestral's `ret` score behaves like this is not checked (we read only the score recipe: sum of the 5 step medians over a width estimate).
- **Following the trend did not help in those years.** The opposite rule, "bet on a turn", has an AUC above 0.5 in 5 of 7 years (0.507 to 0.564 in those five), and Timestral is below 0.5 in all 7. This is a descriptive pattern in 7 short years, not a finding about why.

*Tiny example.* Suppose SPY rose 3% over the last 20 days. Timestral then tends to give a higher P(up) than on an average day, and TimesFM-3 a lower one (that is what the signs of the correlations mean).

---

## 2. What the study asks, and what is mirrored

The study mirrors `timesfm3_experiment/` (pre-registered protocol, amendment included: offset-only calibration). Same data (`data/aligned_prices.pkl`), same `Dataset(h)` rows, same chance rule, same fixed LR (saved test frames), same bootstrap and Holm correction. Deviations are in section 7.

## 3. Validation and selection (2013–2018)

Source: `results/timestral_validation_table.csv` (all cells), `results/timestral_validation_log.csv` (with sanity columns). AUC of the raw score, mean over F1 (2013–14), F2 (2015–16) and VAL (2017–18).

| h | variant | ctx | AUC F1 | AUC F2 | AUC VAL | mean AUC | VAL > 0.5 |
|---|---|---|---|---|---|---|---|
| 1 | price | 512 | 0.463 | 0.506 | 0.458 | 0.476 | no |
| 1 | multi | 1024 | 0.467 | 0.479 | 0.476 | 0.474 | no |
| 1 | **ret (selected)** | **1024** | 0.438 | 0.448 | 0.523 | 0.470 | yes |
| 1 | ret | 2048 | 0.422 | 0.459 | 0.523 | 0.468 | yes |
| 5 | multi | 512 | 0.479 | 0.580 | 0.461 | 0.506 | no |
| 5 | price | 512 | 0.483 | 0.552 | 0.475 | 0.503 | no |
| 5 | **ret (selected)** | **256** | 0.444 | 0.422 | 0.540 | 0.469 | yes |
| 5 | ret | 512 | 0.411 | 0.406 | 0.556 | 0.458 | yes |

(Top rows by mean AUC, plus the selected rows and the other eligible `ret` rows; the full list has 22 cells.)

**In plain words:**

- **No cell is clearly above 0.5.** The best mean AUC in the whole grid is 0.506 (h=5, `multi`, ctx 512), and every other cell is at or below 0.503.
- **The rule's "VAL > 0.5" condition decided the selection.** At h=5 the top mean AUC (`multi` 512, 0.506) fails VAL (0.461). Only `ret` cells pass, so the rule returns the best `ret` cell. Its mean AUC is below 0.5, and the shortlist flags that (`mean_auc_above_half: false`). The rule is mirrored from TimesFM-3's and fixed before the test; we ran it as written.
- **The `ret` cells look like they flip sign between folds.** At h=5, `ret` ctx 256 is 0.444 and 0.422 in F1 and F2, and 0.540 in VAL. Test AUC for it was 0.496. We take this as noise around 0.5, not as a lesson.

*Tiny example.* If a cell has AUC 0.44, 0.42 and 0.54 in three folds, its mean is 0.47, but each fold has only about 500 days with heavily overlapping windows, so a fold's AUC can easily move by ±0.05–0.10 by luck.

Score sanity (`timestral_validation_log.csv`, `results/timestral_report.log`): no score hits the 0.01/0.99 clip (`share_at_clip` is 0 in every row), the quantiles were never out of order (`n_unsorted` 0), mean scores 0.51 to 0.64, standard deviations 0.02 to 0.09. Mean scores above 0.5 reflect that SPY goes up more often than not.

## 4. Pipeline and budget

- **Forecasts.** One forecast per decision date t with `prediction_length=5` and the context ending at Close[t]. Step 1 serves h=1 and step 5 serves h=5. Only Close data up to t is used.
- **Raw score P(up).** The CDF is linearly interpolated through the 21 quantile points (levels 0.01, 0.05, 0.10 ... 0.90, 0.95, 0.99), clipped to [0.01, 0.99]. For `price`/`multi` this is P(logClose[t+h] > logClose[t]). For `ret` at h=1 it is P(r[t+1] > 0). For `ret` at h=5 a normal approximation is used: mu = sum of the 5 step medians, sigma from (q90 − q10)/(2·1.2816) per step.
- **Calibration.** Offset-only: proba = expit(logit(s) + b), slope fixed at 1, so the AUC is the raw score's AUC. b was fitted once on the purged pre-2019 rows (1,510 rows at h=1, 1,506 at h=5; offsets −0.080 and +0.020, `timestral_test_levels.csv`). The decision threshold is the chance rule's positive rate.
- **Budget rule.** On 1,510 dates the full 12-cell grid would take **4.81 CPU-hours** (`results/timestral_timing.csv`), over the 3-hour budget. Rule: drop the most expensive cell(s) on cost alone, before any validation score exists. Only `multi` at ctx 2048 was dropped (6.33 s per forecast = 2.65 h of grid). The remaining 11 cells cost about 2.16 h. The reason is written in `timestral_study.py` (`DROPPED`) and in the pre-registration.
- **Runtime.** The speed settings (batch 8, 8 torch threads) were the fastest in the smoke probe. They change speed only: the same forecast alone and inside a batch of unequal contexts differs by at most 4.8e-07 in log-price units (`timestral_batch_invariance.csv`). Peak memory was 2,118 MB in the smoke step and 2,050 MB in the grid.
- **Wall-clock.** Smoke about 10 minutes, grid about 2 h 18 min (`timestral_grid.log`), checks and report under a minute, test run about 5 minutes (`timestral_test_stdout.txt`).

## 5. Checks

- **Truncation / leakage self-check** (`results/timestral_truncation_check.csv`): a forecast for date t does not change when every price series is cut after t. Largest change over 3 random dates and the three variants at ctx 512: 4.8e-07 in the quantile values (float32 rounding), 0 for `multi`.
- **Quantile order**: none of the cached forecasts had crossing quantiles.
- **Test rows**: the Timestral frames have exactly `Dataset(h)`'s TEST index; the chance frame equals `run_test.chance`; the fixed LR frames come from `experiments/results/test_predictions.pkl` after an index and label check; the TimesFM-3 frames were checked the same way (asserted in `test`).
- **Dry run**: `test --dry-val` ran the same code path on VAL before the real test (`results/timestral_*_dryval*`).

### 5.1 2026 out-of-sample check (descriptive)

Fresh yfinance download (2025-06-02 to 2026-11-06), re-based on the stored panel at 2025-12-31 (largest log-return mismatch over the overlap 5.5e-07), same selected configs, same pre-2019 offsets. Source: `results/timestral_2026.csv`.

| h | config | rows | first–last | AUC | BA | share called "up" |
|---|---|---|---|---|---|---|
| 1 | ret ctx 1024 | 213 | 2026-01-02 – 2026-11-05 | 0.464 | 0.465 | 0.310 |
| 5 | ret ctx 256 | 209 | 2026-01-02 – 2026-10-30 | 0.495 | 0.508 | 0.732 |

Against chance, the BA difference is −0.036 [−0.093, +0.006] at h=1 and +0.008 [−0.096, +0.058] at h=5 (one-sided p 0.954 and 0.759); the AUC differences are −0.036 [−0.119, +0.040] and −0.005 [−0.164, +0.064].

**In plain words:**

- **2026 looks like the test years.** About 210 rows, AUC below 0.5 at both horizons, intervals that include zero. With this few rows of one market regime, nothing can be concluded from it.

*Tiny example.* 209 rows at h=5 is under one year, and neighbouring windows overlap, so it holds only about 40 independent 5-day outcomes.

## 6. Protocol timing

| file | modified |
|---|---|
| `timestral_preregistration.md` (SHA-256 of the shortlist inside) | 2026-11-08 05:09:27 |
| `results/timestral_test_stdout.txt` (created at the start of the test run) | 2026-11-08 05:10:44 |

The test forecasts were therefore produced after the pre-registration. `timestral_study.py test` also refuses to run if the shortlist hash differs. The test ran once; `results/timestral_test_*_dryval*` are the validation-fold rehearsal only.

## 7. Deviations from the TimesFM-3 protocol

Same list as in the pre-registration (written before the test):

- **Quantile grid.** 21 levels (0.01–0.99) instead of 9 (0.1–0.9); the score is clipped to [0.01, 0.99] instead of [0.05, 0.95].
- **Vendor defaults.** `symmetric_averaging=True` and `clip_nonnegative=True` are on (TimesFM was run with its equivalents off). For the log-Close targets the non-negative clamp does nothing (log SPY Close is far above 0, by code reading); `ret` has negative values and is not clamped. Timestral normalises internally and has no z-norm switch.
- **Batching.** Unequal contexts are left-padded with NaN (treated as missing) to the longest in the batch; TimesFM took a ragged list. Batch invariance was measured (section 4).
- **Horizon.** The patch size is 32, so the 5 steps are read from the first patch of one decoding block, with no autoregressive feedback (code reading of `_block_ar_decode`).
- **Covariates.** `multi` passes the 15 tickers as `past_covariates`; the target is SPY only.
- **Budget.** The 3 CPU-hour rule was applied to Timestral's own timings; only `multi` 2048 was dropped (TimesFM dropped `multi` 1024 and 2048).
- **Not a deviation but different here:** the selection rule's "VAL > 0.5" condition picked `ret` for both horizons, where TimesFM-3 picked `multi` ctx 256.

## 8. Limitations

- **Possible pretraining leakage.** The model card of Timestral-209M and the GitHub README describe the architecture and benchmark scores (GIFT-Eval, TIME, fev-bench). They state **no pretraining data sources and no data cutoff**. SPY's history is public, so the 2019–2025 and 2026 backtest cannot be guaranteed free of contamination. A positive result would have had to be read as an upper bound; a null result like this one is not made worse by it.
- **2019–2025 is not pristine for the project.** Earlier studies of this project evaluated it. It is untouched by *this* study's selection.
- **Only the selected configs were run on the test.** We do not know how other cells (for example `multi` 512) would have scored in 2019–2025; this was by design, to keep the test a single pre-registered run.
- **The score approximates the label.** The score models Close-to-Close direction, while the label is Open[t+1] to Close[t+h]; the offset absorbs only the level part of the overnight gap.
- **`ret` at h=5 uses a normal approximation** of a 5-day sum, not a direct forecast.
- **Overlapping windows.** The 5-day windows overlap, so the effective sample is much smaller than the row count; the block bootstrap accounts for that only approximately.
- **Zero-shot only.** This tests the pretrained model, not a fine-tuned Timestral. Other sizes of the family (5M, 28M, 123M) were not tried; the 209M was fixed in advance.
- **Exploratory parts** (1.6) were written after seeing the test and are not corrected for multiple looks.

## 9. Reproduce

Environment (CPU only; the `timesfm` conda env is cloned, not modified):

```bash
/home/tong-nong/miniconda3/bin/conda create -n timestral --clone timesfm -y
/home/tong-nong/miniconda3/envs/timestral/bin/pip install --no-deps \
  "git+https://github.com/mistralai/timestral.git@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"
# model: mistralai/Timestral-209M, revision 9874d252ab0aae16f7f7c999f98f98d6ad7d8365 (downloaded on first use, about 837 MB)
```

Run from inside `timestral/`, with `PYTHONDONTWRITEBYTECODE=1`:

```bash
python timestral_study.py smoke     # timing, batch invariance, memory
python timestral_study.py grid      # 11 cells x 1,510 dates, resumable, about 2.2 h
python timestral_study.py checks    # truncation check
python timestral_study.py report    # validation log + sanity
python timestral_study.py select    # shortlist + hash-locked pre-registration
python timestral_study.py test      # ONE-SHOT 2019-2025 + 2026 check (--dry-val rehearses on VAL)
python timestral_explore_years.py   # exploratory, from saved predictions only, seconds
```

| file | what it is |
|---|---|
| `ts_paths.py` | paths; imports shared code from `../experiments` |
| `timestral_lib.py` | panel, variants, contexts, cache, P(up), calibration, 2026 splice |
| `timestral_study.py` | all steps above |
| `timestral_explore_years.py` | exploratory per-year table and figure |
| `timestral_preregistration.md` | the hash-locked protocol (written 2026-11-08 05:09) |
| `results/timestral_shortlist.json` | selected configs (hash in the pre-registration) |
| `results/timestral_validation_log.csv`, `..._validation_table.csv` | all cells x folds |
| `results/timestral_timing.csv`, `..._batch_invariance.csv`, `..._truncation_check.csv` | smoke and checks |
| `results/timestral_test_levels.csv`, `..._comparisons.csv`, `..._verdict.csv`, `..._by_year.csv`, `..._vs_timesfm3.csv` | test results |
| `results/timestral_test_predictions.pkl`, `results/timestral_test_stdout.txt` | saved test frames and transcript |
| `results/timestral_2026.csv`, `..._2026_predictions.pkl` | 2026 check |
| `results/timestral_exploratory_years.csv`, `figures/timestral_years_h5.png` | exploratory |
| `results/timestral_cache/` | forecast caches (quantiles by date) |
| `results/*_dryval*` | VAL-fold rehearsal of the test code |

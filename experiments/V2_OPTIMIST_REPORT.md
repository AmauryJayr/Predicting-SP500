# V2 optimist report: where does accounting for drift actually pay?

**Bottom line.**

- **Next-month SPY-TLT correlation (h = 21): drift handling helps significantly.** The relationship moved from
  about −0.4 to about +0.1 in 2022-2025.
  - **Linear HAR-corr model refit monthly on an expanding window** (preregistered default slot `spy_tlt_corr.5`):
    test MSE falls from 0.1194 to 0.1018 (**−14.7%**), Δ = +0.0176 [+0.0042, +0.0300], Holm p = 0.02. R² rises
    from 0.07 to 0.21, and sign-classification balanced accuracy rises from 0.515 to 0.642 (+0.127
    [+0.044, +0.218], Holm p = 0.02).
  - This meets the preregistered "large, real" criterion: Holm p < 0.05 vs both the trained-once counterpart and
    the preregistered best baseline, with at least a 5% MSE reduction.
  - **Tree model:** recalibration, bias correction and refitting reduce MSE by **62-69%** vs the same tree trained
    once. That fixed tree is catastrophically broken (R² −1.6), however, so this is a repair, not a win over a
    good model.
- **Caveat that matters most:** no correlation model significantly beats simple **63-day persistence** on the
  test. The best case is +6.7% MSE vs persistence-63, with a CI that includes 0, and balanced accuracy is worse.
  All of the gain comes from 2022-2025, a single regime shift.
- **Volatility (SPY and TLT, h = 5): drift handling does not help.** None of the 12 preregistered configurations
  beats the HAR-X trained once (all Holm p ≥ 0.62; the best is +3.4% MSE on TLT, CI includes 0). The trained-once
  HAR-X is already "drift-robust by construction": its lagged-RV and VIX features carry the regime.

## 1. Setup

| file | role |
|---|---|
| `v2_lib.py` | targets, features, purged walk-forward fitting, all drift methods (refit, decay, RLS, corrected ADWIN/PH, Hedge ensemble, bias correction, recalibration), metrics, paired moving-block bootstrap, Holm. Notebook-importable. |
| `v2_validation.py` | 57-62 configurations per target × 3 folds, plus baselines → `results/v2_validation_log.csv` (555 rows) |
| `v2_select.py` | shortlist rule → `results/v2_shortlist.json`, `results/v2_validation_selection_table.csv` |
| `v2_checks.py` | pre-test truncation (leakage) test and bootstrap sanity checks → `results/v2_checks_truncation.csv` |
| `v2_preregistration.md` | frozen shortlist, comparisons and SHA-256 |
| `v2_test.py` | the one-shot test run (asserts the hash) → `results/v2_test_*` |
| `v2_exploratory.py` | **post-test, exploratory**: mechanism and block-length sensitivity → `results/v2_expl_*` |

**Targets.** Decisions are made at the close of t. A target computed over t+1..t+h is known at the close of t+h,
and a fit at t0 uses only rows with label_end ≤ t0.

| target | h | regression target | classification |
|---|---|---|---|
| `spy_vol` | 5 | log annualised RV of SPY close-to-close returns over t+1..t+5 | 1[RV > RV20(t)] |
| `spy_tlt_corr` | 21 | realised correlation of SPY and TLT daily returns over t+1..t+21. h = 21 because a correlation needs about a month of returns to be estimated at all. | 1[corr > 0] |
| `tlt_vol` | 5 | log RV of TLT (optional third target: bond volatility changed regime in 2022) | 1[RV > RV20(t)] |

The classification is derived from the regression forecast: score = ŷ − reference (log RV20, or 0), with a
threshold of 0. No separate classifier is fitted, so there are no extra degrees of freedom.

**Models.**

- **Linear:** standardised Ridge (α = 1).
  - HAR-X for volatility: daily Parkinson range vol, weekly RV5 and monthly RV22, all in logs, plus log VIX (and
    SPY RV22 for TLT).
  - HAR-corr for correlation: trailing 21-, 63- and 252-day correlation.
- **Tree:** HistGradientBoosting on a broader set (returns, semivariance, VIX change, cross-asset vols and
  returns). Its hyperparameters were fixed a priori.

**Methods per model class** (each compared with the same model trained once on everything known at 2019-01-02):

- expanding refit every 5 / 21 / 63 / 252 sessions;
- rolling windows of 252 / 504 / 1008 / 1512 sessions;
- exponential decay weights (hl 126-1008);
- RLS with a forgetting factor (λ 0.99-0.999, updated daily as labels mature; linear only);
- corrected ADWIN (δ 0.002 / 0.05) and Page-Hinkley (λ 1 / 3) on the delayed |residual| stream. The B1 fix:
  after a detection, refit on rows after the cut and keep refitting the growing window;
- a Hedge ensemble over 5 refitting experts (expanding, rolling 1008/504/252, decay 126), weighted by discounted
  delayed squared error;
- bias correction (trailing mean of the last N known residuals, N 21-252);
- level-and-scale recalibration (trailing OLS of y on ŷ, N 252/504).

**Baselines.**

- spy_vol: persistence (RV5, RV20), raw VIX, VIX-linear, HAR and HAR-X (all trained once).
- tlt_vol: persistence, HAR and HAR-X.
- spy_tlt_corr: persistence (c21, c63), the long-run mean (through 2018) and HAR-corr.

**Leakage check (pre-test, `v2_checks.py`).** I corrupted every label with label_end > 2017-09-29 and every
feature row after that date. All 134 method × target runs, including every online, delayed-feedback, detector and
ensemble mechanism on the linear model, produce **bit-identical** predictions up to the cutoff. Self-comparison
bootstrap Δ = 0.

## 2. Validation (2013-2018) and selection

The full table is in `results/v2_validation_selection_table.csv`.

- **spy_vol:** the HAR-X trained once is already strong (R² about 0.5). Linear adaptive variants gain 1-3% MSE.
  The tree is much worse than the linear model, and its adaptive variants "gain" 5-10% only by repairing it.
- **tlt_vol:** similar. Linear gains are 1-3% on average and 5-14% in VAL.
- **spy_tlt_corr:** 2013-2018 contains no sign change, and **no linear adaptive variant helps** (VAL −1.7% for
  expanding refit, worse for faster forgetting). Only tree repairs are eligible.

**Rule** (identical for all targets; details in `v2_preregistration.md`):

- Eligible means a positive MSE gain vs the counterpart in VAL **and** on average. VAL cannot be outvoted, which
  avoids B3.
- Ranking uses absolute skill vs the best baseline, not repair size.
- Up to 4 rule slots take the best configuration per family.
- 2 fixed a-priori default slots (linear expanding refit 21; linear rolling 1008, refit 21) are the same for
  every target.

**Disclosure:** I wrote the rule after seeing the validation table. The defaults were not chosen for their scores;
for the correlation target they had negative validation scores.

### Preregistered shortlist (SHA-256 `37380b7d…a5636c017`, asserted by `v2_test.py`)

| id | slot | configuration |
|---|---|---|
| spy_vol.1-6 | rule ×4, default ×2 | lin/harx: RLS λ = 0.998; decay hl = 252; Hedge (η 2, hl 63); rolling 504; expanding R21; rolling 1008 |
| spy_tlt_corr.1-6 | rule ×4, default ×2 | hgb: recal N504; bias N252; expanding R21; ADWIN δ = 0.002. lin/hc: expanding R21; rolling 1008 |
| tlt_vol.1-6 | rule ×4, default ×2 | lin/harx: bias N126; recal N252; Hedge (η 2, hl 63); RLS λ = 0.999; expanding R21; rolling 1008 |

The best baseline chosen on validation was the trained-once linear model for all three targets (harx_fixed,
hc_fixed, harx_fixed). For linear entries, "vs fixed" and "vs best baseline" are therefore the same test; it is
counted twice in the Holm family, which is conservative.

## 3. Test results (2019-2025, run once)

ΔMSE = MSE(other) − MSE(entry), so positive means the adaptive model is better. Paired moving-block bootstrap,
2000 draws, L = 20 (h = 5) or 63 (h = 21). Holm adjustment over the 12 confirmatory tests per target. Full
results are in `results/v2_test_bootstrap.csv` and `results/v2_test_levels.csv`.

### 3.1 SPY-TLT correlation (h = 21, n = 1739)

**Levels:**

| model | MSE | R² | BA (sign) | AUC |
|---|---|---|---|---|
| lin/hc trained once (= best baseline) | 0.1194 | 0.070 | 0.515 | 0.779 |
| hgb trained once | 0.3357 | −1.616 | 0.493 | 0.372 |
| persistence c63 | 0.1091 | 0.150 | **0.693** | 0.778 |
| persistence c21 | 0.1547 | −0.206 | 0.690 | 0.729 |
| long-run mean | 0.2371 | −0.847 | 0.500 | 0.500 |
| **.5 lin expanding R21** | **0.1018** | **0.207** | 0.642 | 0.775 |
| .6 lin rolling 1008 | 0.1082 | 0.157 | 0.664 | 0.728 |
| .2 hgb bias N252 | 0.1048 | 0.183 | 0.637 | 0.739 |
| .1 hgb recal N504 | 0.1142 | 0.110 | 0.637 | 0.714 |
| .3 hgb expanding R21 | 0.1257 | 0.021 | 0.578 | 0.710 |
| .4 hgb ADWIN (1 detection) | 0.1302 | −0.015 | 0.590 | 0.696 |

**Confirmatory comparisons:**

| id | ΔMSE vs fixed [95% CI] (Holm p) | ΔMSE vs best baseline (Holm p) | ΔBA vs fixed (Holm p) | ΔMSE vs persist-63 (descriptive) |
|---|---|---|---|---|
| .1 | +0.222 [+0.128, +0.320] (0.000) | +0.005 [−0.017, +0.026] (1.0) | +0.144 (0.02) | −0.005 [−0.038, +0.022] |
| .2 | +0.231 [+0.136, +0.331] (0.000) | +0.015 [−0.012, +0.037] (0.89) | +0.145 (0.03) | +0.004 [−0.022, +0.030] |
| .3 | +0.210 [+0.130, +0.297] (0.000) | −0.006 [−0.027, +0.013] (1.0) | +0.085 (0.02) | −0.017 [−0.049, +0.015] |
| .4 | +0.206 [+0.114, +0.298] (0.000) | −0.011 [−0.042, +0.016] (1.0) | +0.097 (0.02) | −0.021 [−0.057, +0.007] |
| **.5** | **+0.0176 [+0.0042, +0.0300] (0.020)** | **same test (0.020)** | **+0.127 [+0.044, +0.218] (0.02)** | +0.007 [−0.011, +0.023] |
| .6 | +0.011 [−0.013, +0.032] (1.0) | same (1.0) | +0.148 (0.02) | +0.001 [−0.027, +0.022] |

- **AUC does not improve for the linear entries** (ΔAUC vs fixed −0.004 for .5 and −0.051 for .6). The BA gain
  comes from removing the forecast's bias, not from better ranking: the fixed model's mean error is −0.155 overall
  and about −0.25 in 2022-2024. This is the regression-target analogue of audit issue B7.

### 3.2 SPY volatility (h = 5, n = 1755)

The HAR-X trained once has MSE 0.1775, R² 0.488, QLIKE 0.442, BA 0.668 and AUC 0.759. It beats every baseline:
VIX-linear 0.1861, HAR 0.1961, persistence-20 0.2696 and raw VIX 0.3548.

| id | ΔMSE vs fixed (= vs best baseline) [CI] | Holm p | ΔQLIKE | ΔAUC |
|---|---|---|---|---|
| .1 RLS 0.998 | −0.0018 [−0.0105, +0.0070] | 1.0 | +0.019 | −0.010 |
| .2 decay 252 | −0.0036 [−0.0145, +0.0072] | 1.0 | +0.019 | −0.014 |
| .3 Hedge | −0.0021 [−0.0128, +0.0083] | 1.0 | +0.016 | −0.006 |
| .4 rolling 504 | −0.0182 [−0.0373, −0.0018] | 1.0 | −0.014 | −0.028 |
| .5 expanding R21 | +0.0004 [−0.0010, +0.0019] | 1.0 | −0.002 | −0.001 |
| .6 rolling 1008 | −0.0088 [−0.0193, +0.0009] | 1.0 | +0.013 | −0.023 |

No configuration helps, and rolling 504 is significantly worse before adjustment. Small QLIKE gains (+0.016 to
+0.019) for .1-.3 are not significant.

### 3.3 TLT volatility (h = 5, n = 1755)

The HAR-X trained once has MSE 0.1539. Ex post, plain HAR without VIX was better (0.1497).

| id | ΔMSE vs fixed (= vs best baseline) [CI] | Holm p |
|---|---|---|
| .1 bias N126 | +0.0052 [−0.0054, +0.0147] (+3.4%) | 1.0 |
| .2 recal N252 | −0.0056 [−0.0217, +0.0087] | 1.0 |
| .3 Hedge | −0.0037 [−0.0124, +0.0048] | 1.0 |
| .4 RLS 0.999 | +0.0009 [−0.0054, +0.0068] | 1.0 |
| .5 expanding R21 | +0.0016 [−0.0003, +0.0035] | 0.62 |
| .6 rolling 1008 | −0.0071 [−0.0188, +0.0027] | 1.0 |

## 4. Per-period behaviour

Per-year results are in `results/v2_test_by_year.csv` and sub-periods in `results/v2_test_subperiods.csv`.

**Correlation, MSE by year:**

| model | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|
| lin fixed | 0.067 | 0.072 | 0.150 | 0.111 | 0.157 | 0.128 | 0.155 |
| .5 lin expanding R21 | 0.066 | 0.072 | 0.159 | 0.096 | 0.115 | 0.087 | 0.120 |
| persistence c63 | 0.081 | 0.089 | 0.175 | 0.060 | 0.121 | 0.101 | 0.139 |
| hgb fixed | 0.073 | 0.106 | 0.173 | 0.418 | 0.566 | 0.522 | 0.511 |

- **.5 vs fixed:**
  - 2019-2021: ΔMSE −0.0027 [−0.0061, −0.0001], a small, significant cost while the relationship was stable.
  - 2022-2025: **+0.0332 [+0.0132, +0.0528]**.
  - The fixed model's bias by year was −0.04, +0.01, −0.13, −0.26, −0.25, −0.23 and −0.19. For .5 it shrank to
    −0.13, −0.23, −0.13, −0.04 and −0.03 over 2021-2025.
- **Tree repairs vs hgb fixed:** about 0 in 2019-2021, and +0.36 to +0.40 in 2022-2025.
- **vs the best baseline in 2022-2025:** .2 (hgb bias) is +0.036 [−0.003, +0.068] and .1 is +0.027
  [−0.003, +0.053], both with unadjusted p < 0.05.
- **SPY volatility:** no sub-period gain except .5 in 2019-2021 (+0.0023, p = 0.025 unadjusted). The adaptive
  variants lose in 2020, when refits on post-COVID data overshot.
- **TLT volatility:** .1 (bias correction) gains in 2022-2025 (+0.010 [−0.001, +0.020], p = 0.037 unadjusted)
  and loses slightly in 2019-2021. This is the expected pattern for a bond-vol level shift, but it is too small
  to survive adjustment.

**Rolling plot data.** `results/v2_test_rolling252.csv` is plot-ready (date, model, rolling252_mse, target).
`FIXED[...]`, `BEST_ADAPTIVE[...]` and `BEST_BASELINE[...]` are the preregistered rank-1 trio. Every other entry
and baseline is included too. For the correlation target, plot `fixed:lin/hc/fixed` against `spy_tlt_corr.5` and
`base:persist_63` as well, because rank-1 there is the tree repair.

## 5. Exploratory (post-test, not preregistered)

- **Mechanism** (`results/v2_expl_corr_coefficients.csv`). When HAR-corr is refit on an expanding window, the c252
  coefficient rises from 0.10 (2019 fit) to 0.35 (2025 fit). The total persistence Σb rises from 0.55 to 0.81, and
  the implied long-run mean moves from −0.40 to −0.24.
  - The trained-once model keeps pulling every forecast toward −0.40.
  - The refit learns that the correlation regime is more persistent than 2008-2018 suggested.
  - The result is that the refit model approaches persistence. That is why it matches persistence-63 but does
    not beat it.
- **Block length** (`results/v2_expl_blocklen.csv`): the .5-vs-fixed ΔMSE CI excludes 0 at every block length
  from 21 to 252 sessions (L = 252: [+0.0013, +0.0354]). The .5-vs-persist-63 comparison is never significant.

## 6. Honest best case

**Supported claims.**

- When a target's long-run level genuinely shifts, and the model's features do not carry that level (the
  SPY-TLT correlation in 2022), a model trained once stays biased for years. In that situation, the most boring
  drift-aware policy, retraining monthly, cuts MSE by about 15%. The cut is significant after Holm correction, at
  every block length, and it raises sign-classification balanced accuracy from 0.515 to 0.642.
- A flexible model that cannot extrapolate (gradient boosting) fails catastrophically when trained once
  (R² −1.6). Every drift mechanism repairs it (−62% to −69% MSE).

**What I cannot claim.**

- (i) That any drift-aware model beats a simple 63-day persistence forecast. It does not, on MSE or on BA.
- (ii) That validation would have picked the winner. Validation ranked expanding refit as slightly *harmful* for
  the correlation target, and it entered only through a default slot.
- (iii) That the effect generalises. It rests on a single regime shift, about 4 years and about 48
  non-overlapping months.
- (iv) Any benefit for volatility forecasting. HAR-X trained once is as good as anything adaptive.

## 7. What the auditor should check

1. **Timeline.**
   - shortlist: 21:28:05
   - `v2_test.py` written: 21:28:38
   - truncation check: 21:30:09
   - `v2_preregistration.md`: 21:30:36
   - test predictions: 21:31:04
   - statistics: 21:33:26
   - exploratory: 21:34

   `v2_test.py` asserts the hash. The selection rule was written after I saw the validation table (disclosed).
2. **Default slots.** The headline winner (.5) is a default slot, not a validation pick. Check that the defaults
   are plausibly a priori. They are the same for all targets, and their validation scores for the correlation
   target were negative.
3. **Counterpart = best baseline for linear entries.** These are duplicated tests inside the Holm family of 12,
   which is conservative. The "counterpart vs BB" rows with Δ = 0 and p = 1 are degenerate self-comparisons.
4. **BA gains in the correlation target come from de-biasing, not ranking** (AUC unchanged), so a B7-type
   interpretation applies.
5. **QLIKE uses exp(2ŷ) without a log-normal bias correction**, which is the same for every model. A
   bias-corrected variance forecast could change the QLIKE rankings.
6. **The tree "repair" results are B3-type.** The tree trained once is a strawman: trees cannot predict outside
   the training target range, and 2008-2018 correlations were almost always negative. Do not headline them.
7. **Overlapping h = 21 labels** mean the effective sample is about 83 months, and the gain is concentrated in
   2022-2025. A single-event caveat applies.
8. **Detectors.** The corrected ADWIN fired only once (tree model, correlation) and never on the linear vol
   models. The Page-Hinkley variants were noisy in validation and were not shortlisted.
9. **The truncation test covers** all linear methods, every baseline, and the tree fixed / bias / detect methods.
   It does not cover the tree refit or ensemble methods, which share the same `fit()` code path.

Compute: validation about 2 minutes, test about 5 minutes, checks and exploratory about 2 minutes.

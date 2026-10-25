# V2 pessimist audit: volatility and SPY-TLT correlation targets

Scope: `V2_OPTIMIST_REPORT.md`, `v2_preregistration.md`, `v2_lib.py`, `v2_*.py`, `results/v2_*`.
Everything in this audit was computed **after** the one-shot v2 test, so all of it is exploratory.

My scripts (the originals are untouched):

| script | what it does | outputs |
|---|---|---|
| `v2audit_leakage.py` | re-derives labels from raw prices; extends the truncation test to tree / ensemble methods and into the test period | `results/v2audit_leakage_*` |
| `v2audit_robust.py` | robustness of the correlation win: block length, full linear grid, start dates, non-overlapping months, DM-HAC, leave-one-year-out, global Holm | `results/v2audit_robust_*` |
| `v2audit_beatp63.py` | exploratory: can any adaptive model beat 63-day persistence? Mechanism of the refit gain | `results/v2audit_beatp63_*` |
| `v2audit_vol.py` | volatility "sabotage" check (synthetic level shift), real-data bias, log-normal QLIKE, detector fires | `results/v2audit_vol_*` |

Total compute was about 12 minutes.

## 0. Summary verdicts

| claim | verdict |
|---|---|
| C1a. Linear HAR-corr, expanding refit every 21 sessions, beats the same model trained once (−14.7% MSE) | **Credible but narrow.** It survives every statistical robustness check. It depends on a single regime event, needs at least 2 years of the post-2022 regime in the test window, and is not significant under a study-wide Holm correction (p = 0.08). |
| C1b. Sign balanced accuracy 0.515 → 0.642 | **Credible but trivial.** This is pure de-biasing: AUC is unchanged, and persistence-63 reaches 0.693. |
| C1c. "Drift handling helps" for the correlation target | **Not supported** as a general statement. Only slow, expanding-window updating helps. Fast-forgetting methods (rolling ≤ 504, RLS λ ≤ 0.995, bias N ≤ 63, decay hl 126, Page-Hinkley) are worse than training once, and no adaptive model beats persistence-63. |
| C1d. The tree repairs (−62% to −69% MSE) | **Credible as repairs, irrelevant as wins.** No repaired tree beats the linear model trained once or persistence-63. |
| C2. SPY volatility: no drift method beats HAR-X trained once | **Credible.** The methods are not sabotaged: they recover a synthetic level shift. The real data had no persistent bias to correct. |
| C3. TLT volatility: bias correction +3.4% MSE, not significant | **Credible null under MSE.** Under a log-normal-corrected QLIKE (my post-hoc choice), bias correction becomes nominally significant (p = 0.023 unadjusted). That is suggestive at most. |

No leakage was found anywhere.

## 1. Correctness and leakage

**Labels.** I re-derived labels and features from the raw prices for 200 random rows per target
(`results/v2audit_leakage_labels.csv`).

- The forward correlation over t+1..t+21, the forward log-RV over t+1..t+5, the trailing c21 and lrv22 all match
  to 1e-14.
- `label_end` equals the calendar date of t+h in 600 of 600 rows.

Features use data through the close of t. VIX closes 15 minutes after SPY, which is immaterial.

**Truncation / corruption test, extended** (`results/v2audit_leakage_truncation.csv`). There are 46 runs. Every
label with label_end > cutoff and every feature row after the cutoff was replaced with N(0, 5) noise.

- **VAL, cutoff 2017-09-29, all 3 targets:**
  - tree expanding refit (R 21 / 63 / 252), tree rolling (252 / 1008), tree decay (hl 126);
  - tree recalibration (N 252 / 504), the tree **Hedge ensemble**, and tree Page-Hinkley detection.
- **Inside TEST, cutoff 2022-06-30, correlation target:**
  - all 6 shortlist entries and both counterparts, and persistence;
  - linear bias, recalibration, RLS, ensemble and ADWIN;
  - linear refit R5 / R63.

Predictions before the cutoff are **bit-identical in 46 of 46 runs**, and all of them change after it.

**Code review.**

- `fit()` asserts purging.
- Bias correction and recalibration use `cs[nk]` / `base[a:nk]`, i.e. only labels that are known.
- RLS feeds row s only once s < n_known(t).
- Ensemble losses use rows below n_known(t).
- The detectors consume only matured residuals.
- `.5` and the fixed model were reproduced from scratch with max |Δ| = 0.

## 2. Bugs and weaknesses

| # | issue | severity | direction | effect on conclusions |
|---|---|---|---|---|
| V1 | **The detectors are nearly blind to level shifts (bias).** The stream is \|residual\| / (4·sd), clipped to [0, 1]. On the correlation target, the trained-once linear model is biased by −0.23 for 4 years (2022-25), yet the stream mean only moves 0.235 → 0.287. ADWIN (δ = 0.002, n ≈ 1000) needs a jump of about 0.17. Result: **0 fires** for both linear ADWINs on the correlation target, and **0 fires** in my synthetic +0.3 log-vol shift. Page-Hinkley fires the same number of times (2) with or without the shift, so it reacts to vol spikes, not to the shift. Two causes compound: the absolute value removes the sign, and the 4·sd scaling uses about ¼ of the [0, 1] range that the Hoeffding bound assumes. | Medium | Pessimistic for detectors | The statement "detectors fired only once" is a property of this detector design, not evidence that drift detection is useless. Do not cite the detector results as a test of detection. |
| V2 | **The Holm family is per target.** Over the study's 36 confirmatory MSE tests (3 targets × 12), `.5`'s adjusted p is **0.08**. Over the 18 unique tests (duplicates removed) it is 0.035. With 20,000 bootstrap draws the raw p is 0.0034 (2000-draw value: 0.0025). | Low-Medium | Optimistic | "Holm p = 0.02" is specific to the per-target family the preregistration chose. That family is legitimate as preregistered, but it should be reported as such. |
| V3 | **The "large, real" criterion is a single hurdle for linear entries.** "vs counterpart AND vs best baseline" collapses into one test because BB = the counterpart. The baseline that was actually best on test, persistence-63, was *not* the validation BB, and `.5` does not beat it. | Medium (interpretive) | Optimistic | The criterion is formally met but substantively not: the win is over the model's own stale version, not over the best simple forecast. |
| V4 | **The timing of the gain is mis-described.** The report says the gain is "concentrated in 2022-2025, after the flip". In fact 2022 contributes only **12.5%** of the total gain. 2023 (35%), 2024 (34%) and 2025 (26%) carry it. A test ending in 2022 shows **no gain** (+0.0017 [−0.0036, +0.0061]). | Low | Optimistic | Expanding refit corrects a regime change with a lag of 1-2 years. It acts as a slow bias correction, not as drift detection. |
| V5 | **QLIKE uses exp(2ŷ) without a log-normal correction.** With a causal trailing-variance correction, rankings change. TLT bias-N126 becomes rank 1, at +0.022 [+0.0003, +0.041] (p = 0.023 unadjusted, about 0.14 after Holm over 6). For SPY, RLS / expanding move to the top and decay / Hedge drop, all within noise. | Low | Pessimistic for TLT bias correction | QLIKE conclusions are metric-definition sensitive. MSE conclusions are unchanged. |
| V6 | At the test start, bias correction and recalibration use **in-sample** residuals and fitted values of the training rows for their first N-window. These are shrunk for the tree, which delays early-test correction. | Low | Pessimistic (conservative) | Negligible; no leakage. |
| V7 | **Target selection was not blind (protocol).** v2 was designed after the v1 test had exposed 2019-2025 data. The 2022 stock-bond correlation flip and the 2022 bond-vol regime are common knowledge, and the report motivates TLT vol with them explicitly. The method comparison is preregistered; the choice of a target that is *known* to have drifted is not. | Medium | Optimistic | It weakens "confirmatory" to "confirmatory for the method ranking, given a target chosen because it drifted". |

**Default slots (item 2).** I find them plausibly a priori:

- They are identical for all three targets.
- "Retrain monthly on all data" and "rolling 4-year window" are the canonical practitioner choices.
- They had negative validation scores for the correlation target.

However, they were written into the rule after the researcher saw that the rule alone would shortlist only tree
repairs for the correlation target. Validation said "do not adapt" (VAL −1.7%), and the preregistration's own
expectation paragraph did not predict a linear win. So the headline result is one a validation-driven
practitioner would **not** have deployed.

Mitigating this: across the 36 linear configurations, validation and test relative gains have a rank correlation
of **0.79** (`v2audit_robust_lingrid.csv`). Validation ordered the methods correctly (slow adaptation ≫ fast
adaptation) but got the sign of the best one wrong. That is what one expects when the validation folds contain
no regime change.

**Duplicated tests (item 3):** duplicating a test with an identical p inside Holm is conservative, as claimed.
**Tree strawman (item 6):** confirmed. The trained-once tree has R² −1.6 (MSE 0.336), and its repairs only bring
it to roughly the linear level (0.105-0.130).

## 3. Robustness of the correlation win (`.5` vs trained once; Δ = MSE_fixed − MSE_.5)

### 3.1 Block length and DM-HAC (`v2audit_robust_blocklen.csv`, `v2audit_robust_dm.csv`)

| block L / NW lag | bootstrap Δ [95% CI], p | DM-HAC t (HLN), p | vs persist-63: Δ [CI] / DM p |
|---|---|---|---|
| 21 / 20 | +0.0176 [+0.0050, +0.0282], 0.002 | 3.00, 0.0014 | +0.0072 [−0.009, +0.024] / 0.19 |
| 63 / 63 | +0.0176 [+0.0042, +0.0300], 0.0025 | 2.61, 0.0046 | +0.0072 [−0.011, +0.023] / 0.21 |
| 126 / 126 | +0.0176 [+0.0042, +0.0321], 0.0025 | 2.54, 0.0056 | +0.0072 [−0.011, +0.022] / 0.19 |
| 252 / — | +0.0176 [+0.0013, +0.0354], 0.015 | — | +0.0072 [−0.009, +0.019] |

### 3.2 Non-overlapping monthly evaluation (`v2audit_robust_monthly.csv`)

I scored every 21st row, for all 21 phase offsets (n = 82-83 each).

- Δ ranges from +0.016 to +0.020, and t from 2.12 to 2.70.
- **21 of 21 offsets give one-sided p < 0.05** (median p ≈ 0.008).
- The effective sample size of the daily loss differential is about **92** (Bartlett, 126 lags), consistent
  with about 83 independent months.
- vs persist-63: t is 0.45-1.08, and no offset is significant.

### 3.3 Refit frequency and neighbours (`v2audit_robust_lingrid.csv`)

The frequency of 21 was not cherry-picked:

| expanding refit every | Δ vs fixed [CI] | Δ vs persist-63 [CI] |
|---|---|---|
| 5 | +0.0178 [+0.0044, +0.0304] | +0.0075 [−0.010, +0.023] |
| **21** | **+0.0176 [+0.0042, +0.0300]** | +0.0072 [−0.011, +0.023] |
| 63 | +0.0177 [+0.0046, +0.0300] | +0.0074 [−0.012, +0.024] |
| 252 | +0.0162 [+0.0038, +0.0281] | +0.0059 [−0.016, +0.025] |

Wider neighbours (36 linear configurations):

- 22 of 36 have a positive Δ vs fixed. Only 5 have a CI excluding 0: the 4 expanding refits and decay hl = 1008.
- The best point estimate is bias N = 252 (+0.0215 [−0.001, +0.043]).
- **Fast forgetting loses:** rolling 252 (−0.069), bias N 21 (−0.045), RLS 0.99 (−0.042), PH (−0.024 to −0.037),
  rolling 504 (−0.023), decay hl 126 (−0.017).
- **No configuration beats persist-63** with a CI excluding 0. The best is bias N 252 at +0.011 [−0.001, +0.022].

### 3.4 Alternative evaluation windows (`v2audit_robust_startdates.csv`)

In each window the fixed model is trained once at the window start.

| window | Δ vs fixed [CI] | rel. | Δ vs persist-63 [CI] |
|---|---|---|---|
| 2015-2025 | +0.0121 [+0.0018, +0.0220] | 10% | +0.0136 [+0.0003, +0.026] |
| 2017-2025 | +0.0179 [+0.0059, +0.0286] | 15% | +0.009 [−0.006, +0.023] |
| 2019-2025 (prereg) | +0.0176 [+0.0042, +0.0300] | 15% | +0.007 [−0.011, +0.023] |
| 2020-2025 | +0.0237 [+0.0067, +0.0403] | 18% | +0.006 [−0.014, +0.026] |
| 2021-2025 | +0.0311 [+0.0104, +0.0515] | 21% | +0.004 [−0.019, +0.029] |
| 2022-2025 | +0.0349 [+0.0140, +0.0556] | 25% | +0.0005 [−0.018, +0.016] |
| 2023-2025 (fixed model already trained on 2022) | +0.0056 [−0.0056, +0.0140] | 5% | +0.013 [−0.002, +0.027] |
| 2019-2021 | −0.0027 [−0.0061, −0.0001] | −3% | +0.016 [−0.020, +0.048] |
| 2019-2022 | +0.0017 [−0.0036, +0.0061] | 2% | +0.003 |
| 2019-2023 | +0.0098 [−0.0007, +0.0202] | 9% | +0.004 |
| 2015-2021 | +0.0026 [−0.0036, +0.0090] | 2% | +0.021 [+0.005, +0.038] |

Only the 2015-2025 window beats persist-63 with a CI excluding 0, and that is because persistence did badly in
2015-2018.

### 3.5 Leave-one-year-out (`v2audit_robust_loyo.csv`)

| dropped | Δ [CI] | DM p (lag 42) | that year's share of the total gain |
|---|---|---|---|
| 2019 | +0.0204 [+0.0045, +0.0348] | 0.003 | 0% |
| 2020 | +0.0205 [+0.0045, +0.0350] | 0.003 | 0% |
| 2021 | +0.0220 [+0.0066, +0.0362] | 0.001 | −7% |
| 2022 | +0.0179 [+0.0023, +0.0329] | 0.009 | 13% |
| 2023 | +0.0134 [−0.0006, +0.0263] | 0.023 | 35% |
| 2024 | +0.0136 [+0.0023, +0.0235] | 0.006 | 34% |
| 2025 | +0.0150 [+0.0007, +0.0267] | 0.012 | 26% |
| 2023 + 2024 | +0.0078 [−0.0012, +0.0140] | 0.049 | — |
| 2022 + 2023 | +0.0130 [−0.0034, +0.0270] | 0.052 | — |

**The result does not depend on 2022.** It depends on having about 2-3 years of the post-flip regime: dropping
any 2 of 2022-2024 pushes the CI across 0.

## 4. Is the win meaningful? (exploratory)

### 4.1 Mechanism (`v2audit_beatp63_coefpath.csv`, `v2audit_beatp63_decomp.csv`)

MSE = bias² + error variance, over the full test period:

| model | MSE | bias² | error variance | corr(ŷ, y) |
|---|---|---|---|---|
| fixed | 0.1194 | 0.0240 | 0.0954 | 0.514 |
| refit 21 | 0.1018 | 0.0075 | 0.0943 | 0.521 |
| persist-63 | 0.1091 | 0.0004 | 0.1086 | 0.503 |

**About 94% of the gain (0.0165 of 0.0176) is the reduction in squared bias.** Ranking and correlation are
unchanged. The refit model is a slow, automatic intercept correction.

How the coefficients move:

- From 2019 to 2025 the c252 coefficient rises from 0.10 to 0.39, and Σb rises from 0.55 to 0.82.
- The intercept moves from −0.18 to −0.04, so the implied long-run mean moves from −0.40 to −0.22.
- Almost all of this happens after mid-2022. It is the model learning that the correlation is more persistent
  than 2008-2018 implied and that its anchor (−0.40) is wrong.
- Hybrids: "fixed slopes + level update" recovers only 0.1194 → 0.1113, while "refit slopes + 2019 long-run mean"
  recovers 0.1194 → 0.1048. **The larger part comes from increased persistence (weight on c252), i.e. moving
  toward the persistence forecast.**
- Persistence has essentially zero bias and higher variance. The refit model trades between the two, which is why
  it ties persistence and does not beat it.

### 4.2 Adaptive models vs persistence-63 (`v2audit_beatp63_candidates.csv`, Δ = MSE_p63 − MSE_candidate)

| candidate | VAL 2017-18 Δ [CI] | TEST Δ [CI], boot p / DM p |
|---|---|---|
| refit HAR-corr (`.5`) | +0.015 [−0.006, +0.036] | +0.007 [−0.011, +0.023], 0.25 / 0.21 |
| **50/50 average of persist-63 and refit HAR-corr** | +0.010 [−0.001, +0.021] | **+0.0082 [−0.0002, +0.0164], 0.031 / 0.034** |
| inverse trailing-MSE weights (252) | +0.011 [−0.002, +0.022] | +0.0077 [−0.0005, +0.0153], 0.036 / 0.037 |
| trailing OLS stacking (504) | **+0.023 [+0.007, +0.051]** | −0.001 [−0.026, +0.016] |
| Hedge over 5 forecasts | +0.010 | +0.004 [−0.006, +0.011] |
| AR(1) on c63, rolling 504 | **+0.026 [+0.009, +0.053]** | −0.002 [−0.028, +0.016] |
| AR(1) on c63, expanding | +0.018 | −0.001 |

- The candidates that validation would have chosen (stacking, rolling AR) **fail on test**.
- The simple 50/50 combination is the only one positive in both periods, at −7.5% MSE vs persistence. It is
  borderline: its CI touches 0, and with 9 candidates tried, a Holm-adjusted p is about 0.27.
- **There is no credible adaptive model that beats persistence-63.**

## 5. Volatility nulls: were the methods sabotaged? (`v2audit_vol_synth.csv`, `v2audit_vol_realbias.csv`)

**Synthetic test.** I added δ = +0.3 to the log-RV target for every row from 2021-01-01, with features unchanged.
Post-shift MSE, relative to the fixed model:

| method | SPY rel. gain | TLT rel. gain | SPY bias after shift (fixed: −0.31) |
|---|---|---|---|
| bias N 126 | +38% | +48% | −0.02 |
| recal N 252 | +37% | +46% | −0.02 |
| Hedge | +35% | +43% | −0.07 |
| decay 252 | +33% | +40% | −0.10 |
| RLS 0.998 | +31% | +38% | −0.12 |
| rolling 504 | +30% | +36% | −0.09 |
| expanding R21 | +9% | +14% | −0.28 (slow, as expected) |
| ADWIN 0.002 | 0% (0 fires) | 0% (0 fires) | −0.31 (V1) |

The implementations work. With δ = 0 the same code reproduces the real (null) test results.

**Real data.** The trained-once HAR-X bias by year, SPY: −0.01, +0.07, +0.20, −0.14, −0.06, −0.09, +0.02. The
bias alternates in sign with no persistent level shift: the lagged-RV and VIX features carry the vol level. So
there is nothing for a bias corrector to fix. Contrast the correlation target, where the bias was −0.19 to −0.26
for 4 consecutive years.

The volatility nulls are genuine, not sabotage. The one caveat is V1: the detectors could not have shown anything.

## 6. Recommended wording for the class report

> We asked whether drift-aware learning helps when a predictable target's relationship to its features changes.
> For next-week realised volatility of SPY and TLT, it does not. A HAR model with VIX, trained once on 2008-2018,
> was as good as every adaptive variant on 2019-2025 (best +3% MSE, not significant). Its features already carry
> the volatility regime, and a synthetic level-shift experiment confirms that our adaptive methods would have
> helped if a persistent bias had existed.
>
> For next-month SPY-TLT correlation, which flipped from about −0.4 to about +0.1 after 2022, simply re-fitting the
> same linear model every month on all available data cut MSE by 15% relative to never re-fitting. The cut was
> robust to bootstrap block length, Newey-West Diebold-Mariano tests, non-overlapping monthly scoring (21 of 21
> phase offsets significant) and refit frequency (weekly to yearly). Almost all of the gain is the removal of a
> persistent forecast bias. It appears only once the test window contains 2-3 years of the new regime, and it is
> not significant after correcting for all 36 confirmatory tests in the study (p = 0.08).
>
> Faster-forgetting methods (short rolling windows, aggressive forgetting factors, short-window bias correction,
> change detectors) were worse than never re-fitting. None of the adaptive models beat the naive 63-day
> persistence forecast. Validation on 2013-2018, which contained no regime change, would have advised against
> re-fitting.
>
> **Answer:** we found no method that works much better because it accounts for drift. The one reliable benefit is
> modest and generic: periodically re-fitting on all data protects a model from a structural level shift. It does
> not let it outperform a simple persistence forecast.

## 7. Files

All under `C:\UT\381K\Final Project\experiments\`:

- **Scripts:**
  - `v2audit_leakage.py`
  - `v2audit_robust.py`
  - `v2audit_beatp63.py`
  - `v2audit_vol.py`
- **Results:** `results/v2audit_*` (CSV files plus `*_stdout.txt`, and `results/v2audit_robust_preds.pkl`)

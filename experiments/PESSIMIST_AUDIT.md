# Pessimist audit: drift-aware SPY experiments

**Bottom line.**
- **Leakage:** I found none. Purging, delayed feedback, trailing normalisation and the bootstrap all pass targeted tests.
- **Can the code adapt?** Yes. On synthetic data with an injected concept flip, the drift-aware implementations beat the fixed model by about +0.15 BA. The negative SPY result is therefore not caused by code that cannot adapt.
- **One design flaw that hurt the adaptive side:** detection-triggered retraining was crippled. Fixing it does not help on SPY; the fixed version does worse.
- **"Drift-aware methods don't help":** robust. It survives every block length, both test sub-periods, a stale fixed model (the "insurance" scenario) and alternative shortlist rules.
- **"Four h = 5 methods are significantly worse":** fragile. It depends on the 2-of-3-folds selection rule, and with a VAL-only rule it disappears.
- **The exploratory LR-core result (BA 0.536):** fragile, and most likely regime luck rather than signal. Its pooled AUC comes from between-year base-rate tracking, mainly 2022. The mean within-year AUC is 0.499. The same model was anti-predictive in 2013-16 (AUC 0.37 / 0.43).

All audit code is in `experiments/audit_*.py` and all outputs are in `experiments/results/audit_*`. No optimist file, notebook or builder was modified. The audit took about 15 minutes of compute.

---

## 1. Correctness and leakage audit

### 1.1 What was verified (no bug)

| check | method | result |
|---|---|---|
| Purging: a fit at t0 uses only `label_end <= t0` | Code read of `walk_forward`, `run_detect.fit_at`, `run_online`, `run_regime`. **Truncation test** (`audit_tests.py`, `results/audit_truncation.csv`): for all 36 preregistered specs (method, primary and secondary, both horizons) plus 10 extra mechanisms (prior shift, online SGD, Page-Hinkley, rank-norm regime, z1260+decay), I corrupted every label with `label_end` > 2020-03-31 and every feature/VIX row after that date. | All 46 runs give **bit-identical** predictions and thresholds up to the cutoff (max abs diff 0.0). Predictions after the cutoff do change (0.13-0.91), so the corruption took effect. |
| Delayed feedback (Platt, Hedge, ADWIN/PH, SGD, prior) uses only known outcomes | **Label-leak canary** (`audit_tests.py canary`, `results/audit_canary.csv`): synthetic labels are constant within aligned blocks of h rows and independent across blocks, and the features are noise. A mechanism that consumed even one label before its `label_end` would see the current block's label and score BA close to 1. | 86 runs: mean BA 0.496, max 0.527. **No leakage.** |
| Trailing normalisations are past-only | Covered by the truncation test, because `F_full` after the cutoff was corrupted before `causal_normalize` ran. | Pass. |
| Scalers fitted inside each training fold | `StandardScaler` sits inside the `Pipeline` and is refit on each training set. The SGD scaler is fit on `X[:n_known(start)]`. | Pass. |
| Threshold | Threshold = positive rate (decay-weighted where relevant) of the model's own training set. Platt uses the base rate of the calibration window. Hedge uses the weighted mix. In prior shift with `thr_mode="recent"`, BA is provably unchanged. | Correct. |
| Paired bootstrap, signs | Same block indices for A and B; Brier sign flipped; p = P(Δ* ≤ 0). My independent re-implementation (`audit_lib.boot`) reproduces `common.paired_bootstrap` exactly (CI bound −0.11406 in both; p 0.8165 in both). | Correct. Blocks are non-circular, which slightly under-weights the end points. Negligible. |
| Builder `walk_forward` (spot check) | `avail = LABEL_END <= t0`; rolling window = last W of those rows; threshold = training base rate. | Correct. |
| Counterpart matching | Same model, hyperparameters and raw features, trained once on all labels known at the test start. Hedge's counterpart uses the same experts, each trained once. | Matched, **except `detect`** (issue B2). |

### 1.2 Issues found

| # | issue | severity | direction | does fixing it change a conclusion? |
|---|---|---|---|---|
| B1 | **Detection-triggered retraining cannot recover from a change.** On a detection, `run_detect` retrains on the last W = 504 sessions (or the whole history). Right after a change, almost all of those rows are *pre-change*. It then fully resets the detector. A change detector reacts to a *change* in error, not to a persistently high error, so it never fires again and the model stays stale. Synthetic strong flip: ADWIN fires once and recovers +0.03 BA, while the corrected version recovers +0.16 (table in §2). | **Medium** | Handicaps the adaptive side | **No.** The corrected detector (`audit_fixed_methods.py`) retrains on rows after the estimated change point and keeps refitting while that window grows. On SPY it does *worse*. At h = 5 with ADWIN (spec 5.3), ΔBA is −0.044 [−0.082, −0.001] and ΔAUC −0.051 [−0.099, −0.002] vs the fixed model, against −0.022 / −0.026 for the original. A corrected Page-Hinkley version is also worse (ΔBA −0.067). At h = 1 (spec 1.5), ADWIN fires 0 times, so nothing changes. |
| B2 | **The `detect` primary comparison confounds two things.** The method starts from a 504-session model, while the counterpart uses the full history. Spec 1.5's +0.011 AUC is purely a window effect (0 detections). The optimist acknowledges this. | Medium (interpretation) | Either | No. The optimist already reports it correctly. |
| B3 | **The selection rule decides the sign of the h = 5 result.** "Δ > 0 in 2 of 3 folds" lets F1/F2 outvote VAL. In F1/F2 the fixed LR was anti-predictive (AUC 0.37 / 0.43), so the rule picks "repair" methods (Platt with a negative slope, hl = 126). See R4: under a VAL-only rule, the h = 5 picks have ΔBA between −0.021 and +0.009, and none is significant. | **Medium** | Makes the adaptive methods look *worse* | **Yes, for the "significantly worse" claim:** it is a property of this shortlist, not of drift adaptation in general. "No method helps" is unchanged. |
| B4 | **The `z1260` normalisation zero-fills 378 training rows** (2008-01 to 2009-07, the GFC) where about 39% of the columns are NaN (`nan_to_num(0)`). | Low | Handicaps the z1260 configurations | No. z1260 is not in the shortlist and only competed in validation. |
| B5 | **The Holm family is too small.** It is applied over 6 entries per metric, but the confirmatory win rule is "BA **or** AUC", so the family should be 12. | Low | Optimistic | No. Every adjusted p is already 1. |
| B6 | **The notebooks' Majority baseline with expanding/rolling refits has AUC 0.45 / 0.47**, not 0.5. Its constant probability changes at each refit, so the AUC "ranks" days by the base rate of the era. As a result, the h = 5 notebook's "LogReg expanding vs Majority-expanding ΔAUC +0.052, p = 0.049" is an artifact: LR-expanding's own AUC is 0.502. | Low-medium (notebooks only) | Optimistic | Yes, for that one row. Do not cite it as evidence of signal. Compare against AUC = 0.5. |
| B7 | **Pooled-AUC interpretation.** AUC pooled over 2019-2025 rewards a model whose average probability tracks the yearly base rate, which is not within-period ranking. This matters for the exploratory LR (§4). | Low (interpretive) | Optimistic | It changes how the exploratory finding should be read. |
| B8 | **Detector tuning.** Page-Hinkley with λ = 0.5 raises about 40 false alarms in 7 years on *stationary* synthetic data. ADWIN with δ = 0.002 never fires at SPY-like signal levels. This is a tuning observation, not a bug. | Info | – | No. |

## 2. Synthetic drift sanity check (crucial)

`audit_synth.py` builds a `SynthDataset` with the same calendar, periods, h = 5 label delay and interface as `Dataset`. It runs **the optimist's unmodified methods**: the 12 preregistered h = 5 specs plus generic refit, rolling, decay, Platt, prior, PH, ADWIN, SGD and z-norm mechanisms. Δ = method − trained-once counterpart on the 2019-2025 test period.

**Strong signal (fixed-model AUC ≈ 0.75).** Seed 0 for the first five columns; `results/audit_synth_summary_h5.csv`. The last column comes from 3 seeds in `results/audit_detect_fixed_synth.csv`.

| mechanism | flip in 2021 (ΔBA / ΔAUC) | stationary control (ΔBA / ΔAUC) | covariate shift | label shift (ΔBrier) | detector fires (flip) |
|---|---|---|---|---|---|
| 5.4 decay hl = 126 | **+0.164 / +0.216** | +0.001 / −0.008 | +0.031 / +0.025 | +0.021 | – |
| 5.2 rank-norm + decay | **+0.143 / +0.191** | −0.001 / −0.011 | +0.035 / +0.032 | +0.017 | – |
| 5.6 Hedge (diverse) | **+0.150 / +0.196** | +0.003 / −0.004 | +0.018 / +0.031 | +0.020 | – |
| online SGD | **+0.138 / +0.189** | +0.004 / −0.001 | +0.041 / +0.011 | +0.019 | – |
| rolling 504 | **+0.148 / +0.193** | +0.004 / −0.019 | +0.022 / −0.003 | −0.004 | – |
| PH (λ = 0.5) | **+0.147 / +0.192** | −0.005 / −0.020 | +0.015 / −0.005 | −0.005 | 40 |
| 5.1 Platt-recent | +0.041 / +0.059 | +0.002 / −0.003 | +0.025 / +0.021 | +0.025 | – |
| 5.3 ADWIN W = 504 (original) | +0.019 / +0.025 | −0.001 / −0.001 | −0.018 / −0.034 | −0.022 | 1 |
| ADWIN W = 504 (**corrected**, B1) | **+0.159 / +0.215** | −0.004 / −0.004 | | | 1 |
| prior shift N = 126 | −0.003 / −0.001 | −0.009 / −0.017 | | +0.001 | – |

**Verdict: the implementations can adapt.** When the main coefficient flips sign, every re-weighting or refitting mechanism recovers +0.14 to +0.16 BA, and on stationary data they cost almost nothing. Two exceptions:
- **ADWIN** cannot adapt because of the B1 design flaw. The corrected version works.
- **Platt** gains less, because a sign flip in one of two coefficients cannot be fixed by recalibrating the score.

**Weak, SPY-like signal (fixed-model AUC 0.549, 8 seeds).** `results/audit_synth_summary_h5_power.csv`. This is the more important result. With a *full sign flip of the main predictor* in 2021, the fixed model falls to AUC 0.484 after the flip. Even so, the best adaptive methods gain on average only:

| mechanism | ΔBA flip (mean ± sd across seeds) | ΔAUC flip | ΔAUC stationary |
|---|---|---|---|
| online SGD | +0.022 ± 0.013 | +0.030 ± 0.024 | −0.002 |
| decay hl = 252 | +0.022 ± 0.018 | +0.023 ± 0.028 | −0.012 |
| 5.4 decay hl = 126 | +0.021 ± 0.019 | +0.021 ± 0.029 | −0.017 |
| 5.2 rank-norm + decay | +0.019 ± 0.012 | +0.021 ± 0.029 | −0.013 |
| 5.6 Hedge | +0.014 ± 0.018 | +0.014 ± 0.021 | −0.002 |
| 5.1 Platt | +0.007 ± 0.011 | −0.003 ± 0.015 | −0.018 |
| ADWIN | 0 (never fires) | −0.004 | −0.015 |

The 95% CI half-width on the real test is about ±0.04-0.05 BA. So **even a real, large concept drift produces a benefit about half the size of what the 7-year test can detect**, and in stationary stretches the same methods cost 0.01-0.03 AUC. Drift detectors never fire at this signal-to-noise ratio. The SPY null therefore rules out *large* gains (above about +0.03 BA), not small ones.

## 3. Protocol audit

* **Hash:** `sha256(shortlist.json)` = `6550c75a…ed314`. This matches `preregistration.md`, and `run_test.py` asserts it.
* **"Test run once":** this can be checked only from file timestamps.
  - `shortlist.json`: 17:01:15
  - `preregistration.md`: 17:01:39
  - `run_test.py` last edited: 17:02:07
  - `test_predictions.pkl`: 17:02:40
  - statistics (bootstrap CSVs): 17:05:15

  The 28-second edit after preregistration is consistent with fixing a crash. The hash guarantees the specs were not changed, but nothing proves there was no earlier run. I see no sign of tampering.
* **Round-2 grid:** 40 entries were designed after reading the round-1 validation results. Three of the 12 shortlisted entries come from round 2 (5.2, 5.6 and 1.1). This inflates the validation optimism, as the optimist states, but has no effect on test validity.
* **Selection rule:** of 204 entries per horizon, 89 (h = 5) and 54 (h = 1) were eligible. Choosing the maximum per family creates a strong winner's curse: validation deltas of +0.03 to +0.08 shrank to between −0.07 and +0.00 on the test.
* **Multiple-testing burden:**
  - **Validation:** 262 unique specs; 1,236 method-vs-primary fold deltas plus 780 secondary deltas, each on 3 metrics, so about 6,000 validation deltas.
  - **Preregistered test:** 33 bootstrap comparisons × 3 metrics = 99 p-values, of which 24 are confirmatory.
  - **Exploratory after the test:** 7 fixed models vs chance × 3 metrics.
  - **Notebooks:** 15 model × scheme rows and 16 bootstrap comparisons × 3 metrics per horizon.
  - **Overall:** about 60 distinct models were scored on 2019-2025 at h = 5 alone before the LR-core number was highlighted. Among the preregistered tests, 0 of 33 BA p-values are below 0.05, and 2 of 66 AUC/Brier p-values are, about the 3 expected by chance.

## 4. Robustness of the negative conclusion (EXPLORATORY sensitivity checks; no new search)

**R1. Block length** (`results/audit_robust_blocklen_subperiod.csv`). At L = 20, 40, 60 and 120, the 12 point estimates are unchanged by construction, and the CIs barely move: for 5.2, ΔBA [−0.116, −0.027] at L = 20 and [−0.115, −0.028] at L = 120. All Holm-adjusted p = 1 at every L. The CIs are *not* too narrow, and no method crosses into significant improvement.

**R2. Sub-periods.**
- **2019-2021:** the best is 1.5 at ΔBA +0.018 [−0.020, +0.057] and ΔAUC +0.028 [−0.013, +0.068], Holm p = 0.64.
- **2022-2025:** the best is 1.1 / 1.2 / 1.4, all at ΔBA ≤ 0.
- **h = 5:** no entry has a positive ΔBA in either sub-period.

**R3. Stale fixed model, the "insurance" scenario** (`results/audit_robust_stale.csv`). Each method and its trained-once counterpart start on 2017-01-01 (the fixed model is trained through 2016) or 2013-01-01 (trained through 2012). Both are scored on 2019-2025, and the method keeps adapting the whole time.

* **There is no staleness penalty to insure against.** At h = 5, the LR-core model trained through 2016 has test BA 0.541 and AUC 0.539. The one trained through 2018 has 0.536 / 0.536.
* **h = 5:** all 12 stale comparisons are ≤ 0 (ΔBA −0.002 to −0.081). 5.2 and 5.4 remain significantly worse.
* **h = 1:** deltas range from −0.019 to +0.028. The largest is 1.5 (ADWIN window, start 2017) at +0.028 [+0.000, +0.054], an uncorrected borderline result out of 24 comparisons, and 0 detections. 1.2 (GBDT decay) is +0.019 [−0.013, +0.052].
* **Verdict:** adaptation does not pay as insurance over this period.

**R4. Alternative shortlist rules** (`results/audit_robust_altrules.csv`). Rule-selected entries were run on the test, exploratory.

| rule | h | entries | ΔBA range | significant |
|---|---|---|---|---|
| all 3 folds positive | 5 | 2 (GBDT VIX-regime, LR z504) | −0.036, −0.008 | none after Holm |
| all 3 folds positive | 1 | 3 | −0.012 to −0.002 | none |
| VAL-only | 5 | 5 | −0.021 to **+0.009** | none |
| VAL-only | 1 | 6 | −0.009 to +0.009 | none |

The null on "helps" survives every rule. The "significantly worse at h = 5" result does **not**: under VAL-only selection, the h = 5 deltas are centred on zero.

**R5. Why h = 5 and h = 1 differ.** At h = 5, all 6 entries are negative (mean ΔAUC −0.056). At h = 1 they average about −0.008. Three things drive the gap:
1. **Overlapping labels.** h = 5 labels overlap, so the effective sample is about 5 times smaller, and recent-window estimates (Platt N = 504 sessions ≈ 100 independent labels; decay hl = 126) are very noisy.
2. **Selection.** The h = 5 shortlist was dominated by aggressive "repair" configurations chosen because of the anti-predictive F1/F2 fixed models (B3).
3. **Lucky fixed counterpart.** The h = 5 fixed counterpart happened to score well (0.536; see §5).

None of these is evidence that drift adaptation is intrinsically harmful at weekly horizons.

## 5. Stress test of the exploratory finding: trained-once LR, core features, h = 5, BA 0.536

Outputs: `results/audit_corelr_variants.csv`, `results/audit_corelr_blocks_periods.csv` and `results/audit_corelr_folds.csv`.

| check | result |
|---|---|
| **C grid** (1e-4 … 1) | BA 0.508 / 0.524 / **0.541** / 0.536 / 0.533 / 0.524 / 0.520 / 0.518. Only 2 of 7 C values other than the chosen one reach p < 0.05 vs chance. The chosen C sits near the peak. |
| **Drop one feature** (23 variants) | BA between 0.524 and 0.546 (median 0.536). This is robust to single features. |
| **Random 18-of-23 subsets** (200) | Median BA 0.531 (range 0.505-0.557). 84% exceed 0.52, and 59% have p < 0.05 vs chance. |
| **All 122 features** | BA 0.526, AUC 0.542 |
| **Threshold** | Training base rate: 0.536. **Threshold 0.5: 0.496.** Training median: 0.527. |
| **Retraining the same model** | Expanding refits every 21 / 63 / 252 sessions give AUC 0.510 / 0.510 / 0.517. The ranking signal mostly disappears once the model is refreshed. |
| **Block length** vs chance | ΔBA +0.036 with CI lower bound between −0.002 and +0.002 for L = 20 to 120 (p 0.020-0.030). |
| **Sub-periods** | 2019-21: ΔBA +0.014, **ΔAUC −0.007**. 2022-25: ΔBA +0.041, ΔAUC +0.046 (CI includes 0). |
| **Per year** (ΔBA vs 0.5) | +0.013, +0.047, **−0.021**, +0.044, +0.027, **−0.059**, +0.041. Per-year AUC − 0.5 is −0.022, +0.019, −0.019, −0.010, +0.040, −0.068, +0.024, a **mean of −0.005**. |
| **Mechanism** | The probability correlates −0.76 with 20-day realised volatility and +0.60 with distance from the 200-day SMA. It is a "calm and trending → up" regime score. In 2022 (up-rate 0.46) it predicted "up" only 9% of the time. Pooled BA and AUC therefore mostly come from **between-year base-rate tracking (2022)**, not within-period discrimination (B7). |
| **Trivial rules** (BA / AUC on test) | SMA200 0.506 / 0.490; SMA50 0.505 / 0.500; 20-day momentum 0.510 / 0.503; 5-day reversal 0.508 / 0.501; 1-day reversal 0.508 / 0.512; VIX above its 60-day mean 0.504 / 0.488; constant 0.500 / 0.500. Against the LR: vs SMA200, ΔBA +0.030 [−0.010, +0.067] and ΔAUC +0.045 [+0.003, +0.091]; vs 5-day reversal, ΔBA +0.028 [−0.021, +0.078]. The LR beats the rules on pooled AUC but not significantly on BA. |
| **Same model on validation** | F1 (2013-14) BA 0.423, **AUC 0.371**; F2 (2015-16) 0.425 / **0.429**; VAL (2017-18) 0.518 / 0.505. It was strongly *anti*-predictive for 4 of the 6 out-of-sample years before the test. |
| **Same model at h = 1** | Test BA 0.514, AUC 0.508 |
| **Comparisons before it** | It is the best of 7 post-hoc "fixed vs chance" tests (Bonferroni p ≈ 0.18), and of about 11 trained-once-vs-baseline tests in the notebooks and the experiments together. The expected minimum p from 11 null tests is about 0.08, so p = 0.026 is not remarkable. Its specification (C = 0.003, core features) was itself picked indirectly through a drift-method search on validation. |

**Verdict: fragile, best treated as noise or regime luck.** It is robust to feature perturbations, so it is not one lucky feature. However:
- its within-year AUC is about 0.50;
- the edge is concentrated in one regime year (2022);
- it disappears at threshold 0.5 and with routine refits;
- it was strongly anti-predictive in 2013-2016;
- the CI touches 0 after selecting the best of about 7-11 candidates.

It is not credible evidence of predictive signal.

## 6. Final verdict

**(a) Does accounting for drift help on this problem?** **No detectable benefit.** This conclusion is robust.
- The code can adapt: the synthetic flip gives +0.15 BA.
- On SPY, 0 of 12 preregistered methods help.
- No alternative block length, sub-period, staleness scenario or selection rule produces a significant gain, and the CIs rule out gains larger than about +0.03 BA / +0.04 AUC.
- Two caveats:
  1. The claim that drift methods are *significantly harmful* at h = 5 is mostly an artifact of the selection rule (B3) and should not be headlined.
  2. The weak-signal simulation shows the test could not have detected realistic benefits (about +0.02) even if they existed. The honest wording is "no evidence that it helps; large benefits excluded", not "drift handling does not work".

  There *is* evidence of drift: the same fixed model went from AUC 0.37 in 2013-14 to 0.54 in 2019-25. However, the drift appears to be unpredictable sign-switching that adaptive methods cannot exploit at this signal-to-noise level and with a 5-day label delay.

**(b) Is there any credible predictive signal at all?** **No.**
- At h = 1, every model is within ±0.015 BA of chance.
- At h = 5, the best number (0.536) is fragile, driven by one regime, and preceded by many comparisons.
- Trivial rules score 0.50-0.51.

## 7. Recommended wording for the class report

> We compared fixed, periodically retrained and eight families of drift-aware learners (time-decay weighting, trailing feature normalisation, recalibration, expert weighting, detector-triggered retraining, online learning and volatility-regime models) for predicting SPY's 1-day and 5-day direction. Six configurations per horizon were preregistered after validation on 2013-2018. On the 2019-2025 test, none improved on the same model trained once (all Holm-adjusted p = 1). The 95% confidence intervals exclude gains larger than about 3 points of balanced accuracy. Several h = 5 configurations were significantly worse, but this depended on our shortlist rule: methods chosen on the most recent validation fold alone were neither better nor worse. A synthetic experiment shows that the same implementations recover about 15 points of balanced accuracy after an injected concept flip when the signal is strong. With SPY-like signal strength, however, the recovery is only about 2 points, below what a 7-year test can detect. Feature-target relationships in SPY clearly drift (the same model was anti-predictive in 2013-2016), but not in a way that recency-based adaptation could exploit. Absolute performance was at chance level: the best model, a trained-once logistic regression at h = 5 found in exploratory analysis, reached a balanced accuracy of 0.536. That edge came almost entirely from one year (2022), vanished under routine retraining or a 0.5 threshold, and does not survive correction for the number of models examined. We therefore find no credible evidence of predictive signal, and no evidence that drift-aware learning helps on this problem.

## Files

| file | purpose |
|---|---|
| `audit_tests.py` | truncation test (`trunc`) and label-leak canary (`canary`) |
| `audit_synth.py` | `SynthDataset` and the synthetic drift scenarios; `python audit_synth.py 5 8 power` runs the weak-signal power study |
| `audit_fixed_methods.py` | corrected detection-triggered retraining (B1), on synthetic data and SPY |
| `audit_robustness.py` | R1-R4: block length, sub-periods, stale fixed model, alternative shortlist rules |
| `audit_core_lr.py` | stress test of the exploratory LR-core finding |
| `audit_lib.py` | bootstrap with configurable block length, rank AUC, Holm |
| `results/audit_*.csv`, `results/audit_*_stdout.txt`, `results/audit_stale_predictions.pkl` | outputs |

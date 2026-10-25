# v2 preregistration: drift-aware learning for volatility and correlation targets

Written before any 2019-2025 prediction was computed. Everything below was decided with data through 2018-12-31
only (validation folds F1 = 2013-14, F2 = 2015-16, VAL = 2017-18; each fold's models use only labels known at
that date).

**Frozen shortlist:** `experiments/results/v2_shortlist.json`
**SHA-256:** `37380b7d6d210e7d4e6d5ac6460f25525a6aa3cc1e47f0893ad5c37a5636c017`
`v2_test.py` asserts this hash before running.

## Disclosure on how the rule was written

The validation grid (`v2_validation.py`, 171 configurations x 3 folds, plus 14 baseline runs x 3 folds, all
logged in `results/v2_validation_log.csv`) was run **before** the selection rule in `v2_select.py` was written.
I had seen the validation table when I wrote the rule. The rule is generic and applied identically to all three
targets. The two "default" slots are fixed in advance (they are not chosen by validation score) and are the same
for every target.

## Targets (all decisions at the close of t)

| target | h | regression target | classification target | label known |
|---|---|---|---|---|
| spy_vol | 5 | log annualised RV of SPY close-to-close returns over t+1..t+5 | 1[RV(t+1..t+5) > RV20(t)] | close of t+5 |
| spy_tlt_corr | 21 | realised corr of SPY and TLT daily log returns over t+1..t+21 | 1[corr > 0] | close of t+21 |
| tlt_vol (optional third target) | 5 | log annualised RV of TLT over t+1..t+5 | 1[RV > RV20(t)] | close of t+5 |

The classification output is derived from the regression forecast: score = forecast - reference (log RV20(t),
or 0 for the correlation), prediction = 1[score > 0]. No separate classifier is fitted.

## Selection rule (`v2_select.py`)

0. The best baseline (BB) is the baseline with the lowest mean validation MSE.
1. **Eligibility.** The relative MSE gain over the matched trained-once model must be positive in VAL **and**
   positive on average over the 3 folds. VAL can no longer be outvoted (this addresses audit issue B3).
2. **Ranking** is by mean skill vs BB, i.e. absolute quality, not the size of the repair over a weak fixed model.
3. **Up to 4 rule slots:** the best eligible configuration of each method family, in rank order.
4. **2 default slots** on the linear model: expanding refit every 21 sessions, and a rolling 1008-session window
   refit every 21 sessions.

## Shortlist

| id | slot | configuration | val mean rel. MSE gain vs fixed | VAL gain | val skill vs BB |
|---|---|---|---|---|---|
| spy_vol.1 | rule | lin/harx RLS, forgetting factor λ = 0.998 | +3.2% | +2.9% | +3.2% |
| spy_vol.2 | rule | lin/harx decay hl = 252, refit 21 | +2.7% | +2.7% | +2.7% |
| spy_vol.3 | rule | lin/harx Hedge ensemble (η = 2, hl = 63) | +2.7% | +1.9% | +2.7% |
| spy_vol.4 | rule | lin/harx rolling W = 504, refit 21 | +2.1% | +0.3% | +2.1% |
| spy_vol.5 | default | lin/harx expanding refit 21 | +1.2% | −0.3% | +1.2% |
| spy_vol.6 | default | lin/harx rolling W = 1008, refit 21 | +1.5% | +3.3% | +1.5% |
| spy_tlt_corr.1 | rule | hgb/full recalibration (a + b·ŷ on last 504 known) | +10.8% | +24.2% | −21.6% |
| spy_tlt_corr.2 | rule | hgb/full bias correction N = 252 | +9.0% | +19.7% | −24.3% |
| spy_tlt_corr.3 | rule | hgb/full expanding refit 21 | +5.5% | +13.0% | −30.0% |
| spy_tlt_corr.4 | rule | hgb/full ADWIN (corrected, δ = 0.002) | +2.0% | +5.9% | −34.1% |
| spy_tlt_corr.5 | default | lin/hc expanding refit 21 | −1.2% | −1.7% | −1.2% |
| spy_tlt_corr.6 | default | lin/hc rolling W = 1008, refit 21 | −2.5% | −2.3% | −2.5% |
| tlt_vol.1 | rule | lin/harx bias correction N = 126 | +2.7% | +9.4% | +2.7% |
| tlt_vol.2 | rule | lin/harx recalibration N = 252 | +2.3% | +14.2% | +2.3% |
| tlt_vol.3 | rule | lin/harx Hedge ensemble (η = 2, hl = 63) | +2.0% | +9.7% | +2.0% |
| tlt_vol.4 | rule | lin/harx RLS λ = 0.999 | +1.6% | +5.1% | +1.6% |
| tlt_vol.5 | default | lin/harx expanding refit 21 | +1.2% | +3.1% | +1.2% |
| tlt_vol.6 | default | lin/harx rolling W = 1008, refit 21 | +0.8% | +5.8% | +0.8% |

**Best baselines (BB):** harx_fixed (spy_vol), hc_fixed (spy_tlt_corr) and harx_fixed (tlt_vol). In every case
the strongest baseline is the trained-once linear HAR-type model, so for linear entries "vs fixed" and "vs BB" are
the same comparison.

**Expectation.** For spy_tlt_corr, validation (2013-2018, with no sign change) gives no evidence that adaptation
helps the linear model. The rule-selected entries are tree-model repairs that are clearly worse than BB. I expect
those to fail vs BB.

**Matched counterpart:** the same model, features and hyperparameters, fit once on all labels known at
2019-01-02 (i.e. trained through 2018).

**Baselines:**

- spy_vol: persistence RV5 and RV20, the raw VIX forecast log(VIX/100), VIX-linear (fixed), HAR (fixed) and
  HAR-X = HAR + log VIX (fixed).
- tlt_vol: persistence RV5 and RV20, HAR (fixed) and HAR-X (fixed).
- spy_tlt_corr: persistence c21 and c63, the long-run mean (training mean through 2018) and HAR-corr
  (c21, c63, c252; fixed).

## Comparisons and statistics

- **Primary metric:** MSE of the regression target. Δ = MSE(other) − MSE(entry), so positive means the entry is
  better.
- **Confirmatory comparisons:** 12 per target, i.e. each of the 6 entries vs its counterpart and vs BB.
  - One-sided bootstrap p = P(Δ* ≤ 0).
  - Holm adjustment over these 12, per target and per metric.
- **Secondary metrics:** R², QLIKE (vol targets), balanced accuracy and AUC of the derived classification. These
  get the same Holm treatment within their own target × metric family.
- **Bootstrap:** paired moving-block bootstrap, 2000 draws, block length max(20, 3h) = 20 for h = 5 and 63 for
  h = 21.
- **Descriptive only:**
  - every entry vs every other baseline;
  - the counterpart vs BB (the B3 check);
  - per-year metrics;
  - sub-periods 2019-2021 and 2022-2025;
  - rolling 252-session MSE of FIXED (rank-1 counterpart), BEST_ADAPTIVE (rank-1 entry, id `.1`) and BEST_BASELINE.
- **What counts as a "large, real improvement":** Holm p < 0.05 vs **both** the counterpart and BB, with an MSE
  reduction of at least 5%.

The test is run once by `v2_test.py`. Anything computed after that is labelled exploratory in the report.

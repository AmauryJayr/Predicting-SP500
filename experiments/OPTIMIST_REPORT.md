# Optimist report: can drift-aware learning beat the same model without drift handling?

**Bottom line.** No. I searched 206 drift-aware configurations across 8 method families on validation data only (2008-2018) and preregistered 6 per horizon. On 2019-2025, none beats its matched trained-once counterpart, on any metric, at either horizon. At h = 5, four of the six are *significantly worse* than the fixed model. The validation gains were real in-sample, but they came from a specific situation: fixed models trained on short, crisis-dominated histories (2008-2012/2014) that became anti-predictive. Recalibration and re-weighting could repair that. The test-period fixed model was trained on 11 years, was not broken, and adaptation only added variance.

## 1. Setup (all code in `experiments/`)

| file | role |
|---|---|
| `common.py` | data loading, the 122 builder features (verbatim), target, purged walk-forward with sample weights / causal feature transforms / warm-up, metrics, paired block bootstrap |
| `methods.py` | the drift-aware mechanisms (prior shift, Platt recalibration, Hedge ensemble, Page-Hinkley / ADWIN triggered retraining, online SGD, regime models) |
| `configs.py`, `configs2.py` | round-1 (166) and round-2 (40) search spaces; each entry = method + primary + secondary counterpart |
| `run_validation.py`, `select_shortlist.py` | validation folds, logging, shortlist rule |
| `preregistration.md`, `results/shortlist.json` | frozen shortlist (SHA-256 checked by `run_test.py`) |
| `run_test.py` | one-shot test run |
| `exploratory_diagnostics.py` | post-test diagnostics, labelled exploratory |

**Leakage controls.** A model fitted at decision row t0 uses only rows with `label_end <= date[t0]` (asserted in code). Every online mechanism (base-rate tracking, Platt windows, expert losses, drift-detector streams, SGD updates) consumes an outcome only once its `label_end` has passed, so losses arrive with the h-day delay. Trailing normalisations were checked by recomputing them on truncated histories (max |diff| = 0). Recalibrators and Hedge weights are trained on genuinely out-of-sample predictions, produced in a 252-session warm-up before each evaluation period.

**Selection folds.** F1 = 2013-14, F2 = 2015-16 and VAL = 2017-18. Each is a walk-forward period whose fixed model is trained on everything known at the fold start. Total compute was about 3 minutes.

## 2. Methods tried (each with a matched counterpart)

Primary counterpart = the same model, hyperparameters and features, trained **once** at the start of the period and never adapted. Secondary = a mechanism ablation.

| family | drift mechanism | grid | secondary counterpart |
|---|---|---|---|
| decay | exponential time-decay sample weights, refit every 21 (LR) / 63 (GBDT) sessions | half-life 126-1008; LR C ∈ {0.003, 0.03}; all/core features | same refits without decay |
| norm | covariate-drift removal: trailing z-score (252/504/1260) or trailing percentile rank (252/504) of every feature | LR and GBDT | – |
| norm+decay | both of the above | rank252/z1260 × hl 252/504 | fixed model on normalised features |
| prior | label-shift correction: logit shift to the trailing N-label base rate; threshold = recent or training rate | N ∈ {63, 126, 252} | – |
| platt | Platt recalibration on the last N known out-of-sample predictions (can flip sign) | N ∈ {126, 252, 504}, base refit 21/63 | static Platt fitted once |
| hedge | exponentially weighted experts (windows 252/504/1008/expanding, or a diverse LR/GBDT pool), weights from delayed discounted log-loss | η ∈ {20, 100, 500}, loss half-life 63/252 | equal-weight refitting experts |
| detect | Page-Hinkley (log-loss) / ADWIN (0-1 error) on the delayed loss stream; retrain on 504 sessions or expanding | 4 detectors × 2 windows | scheduled refit every 63 sessions |
| online | SGD logistic regression with partial_fit as labels arrive | α, η0 | same SGD frozen |
| regime | separate models by VIX regime (median VIX level, or VIX 1-year percentile rank) | LR / GBDT | – |

"core" is an a-priori 23-feature subset (trend, reversal, volatility, VIX, breadth, bonds), defined in `common.py` before any result was seen.

## 3. Validation results

Every configuration and its counterparts appear in `results/validation_log.csv` (3,252 rows: each entry's method and counterparts × 3 folds × 2 horizons; shared counterparts repeat). Per-fold deltas are in `results/validation_deltas.csv`, fold means in `results/validation_summary.csv`.

The validation deltas vs the trained-once model looked strong at h = 5 (up to +0.08 BA / +0.08 AUC, fold-averaged) and modest but consistent at h = 1 (about +0.03 / +0.03, positive in all 3 folds for GBDT decay). However:

* The h = 5 gains came almost entirely from F1/F2. There the fixed LR (trained on 2008-12 or 2008-14) had AUC of only 0.40-0.44, i.e. it was **anti-predictive**, and recent-window Platt recalibration (which learns a negative slope) or re-weighting repaired it. On VAL, the same methods lost (e.g. Platt N=504: ΔAUC −0.099).
* Absolute levels of the "winners" were still about 0.50-0.54 BA and 0.50-0.51 AUC.

Exploratory read of the validation log (`results/exploratory_val_delta_by_fold.csv`): averaged over all 206 entries, the drift-aware gain shrinks as the fixed model's history grows.

| h | fold (fixed-model history) | mean ΔBA | mean ΔAUC | share of entries with ΔAUC > 0 |
|---|---|---|---|---|
| 5 | F1 (5 y) | +0.018 | +0.019 | 48% |
| 5 | F2 (7 y) | +0.020 | +0.020 | 59% |
| 5 | VAL (9 y) | −0.004 | −0.016 | 34% |
| 1 | F1 (5 y) | +0.010 | +0.001 | 44% |
| 1 | F2 (7 y) | −0.003 | −0.003 | 43% |
| 1 | VAL (9 y) | −0.001 | −0.011 | 26% |

In hindsight this was the warning sign. My selection rule (positive in at least 2 of 3 folds) let F1/F2 outvote VAL, the fold most like the test.

## 4. Preregistered test results (2019-01-02 to 2025, run once)

Δ = method − primary (trained-once) counterpart; positive = method better (Brier sign flipped). 95% CIs come from a paired moving-block bootstrap (2000 draws, block length 20). p is one-sided (H1: method better), and Holm-adjusted p uses the 6 entries per horizon. **All Holm-adjusted p = 1.0.**

### h = 5 (n = 1755)

| id | method | BA (method / fixed) | AUC (method / fixed) | Δ BA [95% CI] | Δ AUC [95% CI] | Δ Brier [95% CI] |
|---|---|---|---|---|---|---|
| 5.1 | Platt-recent, LR core, N=504 | 0.511 / 0.536 | 0.476 / 0.536 | −0.025 [−0.077, +0.029] | **−0.060 [−0.114, −0.003]** | −0.002 [−0.006, +0.004] |
| 5.2 | rank252 norm + decay hl=252, LR core | 0.464 / 0.536 | 0.464 / 0.536 | **−0.072 [−0.116, −0.027]** | **−0.072 [−0.122, −0.020]** | **−0.008 [−0.016, −0.002]** |
| 5.3 | ADWIN-triggered retrain (W=504), LR core | 0.514 / 0.536 | 0.510 / 0.536 | −0.022 [−0.060, +0.015] | −0.026 [−0.075, +0.024] | **−0.009 [−0.015, −0.002]** |
| 5.4 | decay hl=126, LR core C=0.03 | 0.462 / 0.524 | 0.443 / 0.531 | **−0.062 [−0.114, −0.007]** | **−0.089 [−0.148, −0.023]** | **−0.040 [−0.055, −0.023]** |
| 5.5 | VIX-regime GBDT, all | 0.471 / 0.508 | 0.458 / 0.511 | −0.036 [−0.076, +0.001] | **−0.053 [−0.095, −0.011]** | **−0.011 [−0.018, −0.004]** |
| 5.6 | Hedge over diverse experts, η=100 | 0.501 / 0.521 | 0.494 / 0.529 | −0.020 [−0.053, +0.009] | **−0.035 [−0.069, −0.002]** | +0.000 [−0.003, +0.004] |

### h = 1 (n = 1759)

| id | method | BA (method / fixed) | AUC (method / fixed) | Δ BA [95% CI] | Δ AUC [95% CI] | Δ Brier [95% CI] |
|---|---|---|---|---|---|---|
| 1.1 | rank252 norm + decay hl=252, GBDT core | 0.497 / 0.508 | 0.498 / 0.511 | −0.012 [−0.041, +0.020] | −0.014 [−0.053, +0.027] | **−0.007 [−0.012, −0.002]** |
| 1.2 | decay hl=252, GBDT core | 0.506 / 0.508 | 0.510 / 0.511 | −0.002 [−0.032, +0.032] | −0.001 [−0.042, +0.038] | **−0.007 [−0.013, −0.001]** |
| 1.3 | rank252 norm, LR core | 0.495 / 0.514 | 0.495 / 0.508 | −0.019 [−0.041, +0.002] | −0.013 [−0.033, +0.006] | +0.000 [−0.001, +0.002] |
| 1.4 | VIX-rank regime LR, all | 0.495 / 0.499 | 0.482 / 0.492 | −0.005 [−0.030, +0.022] | −0.010 [−0.038, +0.019] | −0.002 [−0.004, +0.001] |
| 1.5 | ADWIN-triggered retrain (W=504), LR all | 0.502 / 0.499 | 0.502 / 0.492 | +0.003 [−0.026, +0.031] | +0.011 [−0.021, +0.043] | −0.003 [−0.006, +0.001] |
| 1.6 | Platt-recent, LR core, N=252 | 0.489 / 0.514 | 0.487 / 0.508 | −0.024 [−0.055, +0.008] | −0.021 [−0.055, +0.016] | +0.000 [−0.002, +0.002] |

**Confirmed wins under the preregistered criterion: 0 of 12.** Bold marks CIs that exclude zero, and every one of those is a loss.

**Secondary (mechanism-ablation) comparisons**, unadjusted, from `results/test_bootstrap.csv`. These are the only nominally positive results:
* 5.3: ADWIN-triggered vs scheduled 63-day refits, ΔAUC +0.056 [+0.002, +0.109], p = 0.022. But ADWIN fired only **once** in seven years, so this really says that "refitting rarely beats refitting quarterly", which agrees with the main finding.
* 5.6: adaptive Hedge vs equal-weight refitting experts, ΔBrier +0.0027 [+0.0001, +0.0056], p = 0.02; ΔBA +0.029 [−0.007, +0.065]. Weighting experts by recent loss helps relative to a naive refit ensemble, but not relative to the trained-once ensemble.
* 1.5: ADWIN fired **zero** times at h = 1. The "method" is therefore one model fitted at the start on the last 504 sessions, so its +0.011 AUC vs fixed is a window-length effect, not drift detection.

With about 30 secondary tests, one or two nominal p ≈ 0.02 results are expected by chance.

### Per-year test results (balanced accuracy; method / fixed)

| id | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|
| 5.1 Platt | 0.495 / 0.513 | 0.523 / 0.547 | 0.479 / 0.479 | 0.499 / 0.544 | **0.587** / 0.527 | **0.490** / 0.441 | 0.462 / 0.541 |
| 5.3 ADWIN | 0.479 / 0.513 | 0.529 / 0.547 | 0.433 / 0.479 | 0.472 / 0.544 | **0.541** / 0.527 | **0.521** / 0.441 | 0.528 / 0.541 |
| 5.6 Hedge | **0.525** / 0.502 | 0.438 / 0.486 | **0.499** / 0.486 | **0.511** / 0.490 | 0.506 / 0.554 | 0.443 / 0.460 | 0.468 / 0.516 |
| 1.2 GBDT decay | **0.520** / 0.466 | **0.504** / 0.492 | 0.490 / 0.509 | 0.502 / 0.507 | 0.510 / 0.553 | 0.490 / 0.516 | **0.525** / 0.509 |
| 1.5 ADWIN | **0.506** / 0.488 | **0.487** / 0.459 | **0.502** / 0.494 | **0.518** / 0.479 | 0.511 / 0.538 | 0.483 / 0.538 | **0.506** / 0.474 |

AUC by year for all 12 entries is in `results/test_by_year.csv`. Year-level wins and losses alternate with no persistent pattern. The exception is 2023-24 at h = 5, where the fixed model's AUC dropped (0.54 to 0.43) and the adaptive models were better.

## 5. Honest best case

* **The best "drift-aware" story I can defend:** at h = 5, a fixed model can go stale. In 2024 the fixed LR fell to BA 0.441, and Platt/ADWIN adaptation recovered to 0.49-0.52 that year. On validation, recent-window recalibration repaired fixed models that had turned anti-predictive after the 2008-12 regime (fold AUC 0.40 → 0.62). So drift handling acts as **insurance against a broken fixed model**. When the fixed model is not broken, the same insurance costs 2-9 AUC points, and over 2019-2025 the cost outweighed the payout.
* **Least bad over the full period:** 1.5 (ADWIN on a 504-session window, h = 1) at ΔBA +0.003 / ΔAUC +0.011, and 1.2 (GBDT time-decay, h = 1) at roughly 0. Neither is distinguishable from zero, and both are slightly worse on Brier.
* **Size of the drift-aware benefit on the preregistered test:** none detectable. The 95% CIs bound any gain at about +0.03 BA and +0.04 AUC (h = 1) or +0.03 / +0.02 (h = 5). Several methods are reliably harmful at h = 5.
* **Exploratory and not preregistered** (`results/exploratory_fixed_vs_chance.csv`): the trained-once LR on core features at h = 5 has test BA 0.536 and AUC 0.536 vs chance, ΔBA +0.036 [−0.000, +0.074], p = 0.026, ΔAUC +0.036 [−0.007, +0.082]. This is the single most promising number in the project, and it belongs to the **non-adaptive** model. It is one of several counterparts examined after the test was run, so treat it as a hypothesis, not a result.

## 6. Notes for the auditor

1. **Protocol.** `preregistration.md` embeds the SHA-256 of `shortlist.json`, and `run_test.py` asserts it. The test was run once (`results/test_run_stdout.txt`). Nothing was changed after the test. The only post-test code is `exploratory_diagnostics.py`, which is read-only.
2. **Round-2 grid.** `configs2.py` was designed after seeing round-1 *validation* results. That is legitimate selection, but it inflates the validation optimism. `validation_log.csv` has a `round` column.
3. **Selection rule weakness.** "2 of 3 folds positive" let the two early folds, whose fixed models had short histories, dominate. A rule weighting VAL more heavily, or requiring all 3 folds, would have shortlisted fewer h = 5 adaptive methods. I did not re-run the test with such a rule.
4. **Counterpart choice.** The primary counterpart retrains nothing, but it does see all data through 2018. Secondary counterparts isolate each mechanism. For `norm` and `regime` there is no secondary counterpart.
5. **Things worth checking in code:** `n_known` relies on `label_end` being monotone (asserted). Platt and Hedge use warm-up predictions from walk-forward refits, never in-sample fits. The prior-shift with `thr_mode="recent"` provably leaves BA unchanged, so only AUC/Brier can move. The ADWIN implementation is simplified (exhaustive cut every 5 points, increases only). GBDT hyperparameters were fixed a priori rather than tuned.
6. **Bootstrap.** Block length is max(20, 3h) = 20 for both horizons. That may be short for h = 5 with overlapping labels, so a longer block would widen the CIs. This only strengthens the null.

# Preregistration: drift-aware shortlist for the 2019-2025 test period

Written after all validation runs (folds F1 = 2013-2014, F2 = 2015-2016, VAL = 2017-2018) and **before any 2019-2025 prediction was produced**. 
Frozen specifications: `experiments/results/shortlist.json` (SHA-256 `6550c75a7322c46f42ac026b0770a3c4fcf5b426dc38392bd0376b57eb3ed314`). `run_test.py` refuses to run if the hash differs.

## Selection rule (applied to validation only)

All 206 entries (round 1: `configs.py`, 166 entries; round 2: `configs2.py`, 40 entries, designed after looking at round-1 validation results) were run on the three folds. Each fold's models use only labels known at each decision date (label_end <= decision date). Every configuration is logged in `results/validation_log.csv`.

For each horizon, an entry is eligible if, against its **primary counterpart**, Δ balanced accuracy > 0 in at least 2 of the 3 folds and Δ ROC-AUC > 0 in at least 2 of the 3 folds. Eligible entries are ranked by mean ΔBA + mean ΔAUC. The rule keeps the best entry per method family, then the six best families.

## Counterparts

* **Primary counterpart (main comparison):** the same model, hyperparameters and raw feature set, trained ONCE on all labels known at the first test day (2019-01-02), and never adapted, recalibrated or re-weighted. For `hedge`, it is the equal-weight average of the same experts, each trained once.
* **Secondary counterpart (mechanism ablation), where defined:** decay → the same refit schedule without decay; norm+decay → a fixed model on the same normalised features; platt → the same refitting base with a *static* Platt calibrator fitted once on the year before the test; detect → scheduled refits every 63 sessions on the same window; hedge → equal weights over the same refitting experts.
* **Chance baseline:** a constant prediction equal to the training positive rate (BA = AUC = 0.5).

## Shortlist (all hyperparameters frozen)

### h = 5

| # | family | configuration | val mean ΔBA | val mean ΔAUC | val mean ΔBrier (+ = better) |
|---|---|---|---|---|---|
| 5.1 | platt | `LogReg{'C': 0.003} core R=21 N=504` | +0.0813 | +0.0767 | +0.0043 |
| 5.2 | norm+decay | `LogReg{'C': 0.003} core rank252 hl=252` | +0.0609 | +0.0655 | -0.0021 |
| 5.3 | detect | `LogReg{'C': 0.003} core adwin(d=0.002,lam=None) W=504` | +0.0410 | +0.0757 | -0.0121 |
| 5.4 | decay | `LogReg{'C': 0.03} core hl=126` | +0.0439 | +0.0528 | -0.0179 |
| 5.5 | regime | `GBDT all vix_level` | +0.0281 | +0.0465 | -0.0027 |
| 5.6 | hedge | `diverse eta=100 lhl=63` | +0.0352 | +0.0137 | +0.0015 |

### h = 1

| # | family | configuration | val mean ΔBA | val mean ΔAUC | val mean ΔBrier (+ = better) |
|---|---|---|---|---|---|
| 1.1 | norm+decay | `GBDT core rank252 hl=252` | +0.0304 | +0.0287 | -0.0031 |
| 1.2 | decay | `GBDT core hl=252` | +0.0276 | +0.0283 | -0.0024 |
| 1.3 | norm | `LogReg{'C': 0.003} core rank252` | +0.0301 | +0.0197 | -0.0005 |
| 1.4 | regime | `LogReg{'C': 0.003} all vix_pct_rank_252d` | +0.0225 | +0.0233 | -0.0010 |
| 1.5 | detect | `LogReg{'C': 0.003} all adwin(d=0.05,lam=None) W=504` | +0.0115 | +0.0268 | -0.0076 |
| 1.6 | platt | `LogReg{'C': 0.003} core R=21 N=252` | +0.0289 | +0.0026 | +0.0003 |

Exact specs (method / primary / secondary) are in `shortlist.json`.

## Reported comparisons (test 2019-01-02 .. last labelled day of 2025, run once)

For each of the 12 shortlisted entries:
1. Test BA, AUC and Brier score of the method, its primary counterpart and its secondary counterpart.
2. Paired moving-block bootstrap (2000 draws, block length max(20, 3h): 20 for h = 1 and 20 for h = 5, seed 42) of method minus counterpart: Δ BA, Δ AUC and Δ Brier (sign flipped, so positive = better), with a 95% percentile CI and a one-sided p = P(Δ* <= 0), against (a) the primary counterpart (**the confirmatory test**), (b) the secondary counterpart, and (c) the chance baseline.
3. Holm correction of the primary one-sided p-values across the 6 entries within each horizon, separately for each metric.
4. Per-year BA and AUC (2019 ... 2025) for each method and its primary counterpart.

A method counts as a confirmed win only if its primary Δ BA or Δ AUC has a Holm-adjusted p < 0.05 and its point estimate on the other metric is not negative. All 12 results are reported, wins or not. No configuration is changed after the test results are seen. Any later analysis is labelled exploratory.

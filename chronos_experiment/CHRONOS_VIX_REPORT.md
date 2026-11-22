# Chronos-2 VIX study: implementation report

This study ports master's VIX study (`experiments/vix_study.py`, `experiments/vix_lib.py`) to Chronos-2. It asks the same question for the foundation model: **can the VIX, used as a label-free regime signal, help a model handle drift?** The code is written but **has not been run yet**: no result exists, and every result section below says which file will hold it.

* Script: `chronos_experiment/chronos_vix_study.py`
* Outputs, created when it runs: `chronos_experiment/results_vix/` and `chronos_experiment/chronos_vix_preregistration.md`

## 1. How to run

From the project root, in this order:

```
python chronos_experiment/chronos_vix_study.py plan       # no GPU needed: forecasts still to compute
python chronos_experiment/chronos_vix_study.py smoke      # loads Chronos-2, checks cache reuse, estimates minutes
python chronos_experiment/chronos_vix_study.py validate   # folds F1 2013-14, F2 2015-16, VAL 2017-18
python chronos_experiment/chronos_vix_study.py select     # freezes choices, writes chronos_vix_preregistration.md
#   -> commit results_vix/vix_choices.json and chronos_vix_preregistration.md here, before the test
python chronos_experiment/chronos_vix_study.py test       # one run on 2019-2025; refuses to run twice
```

* `--batch N`, placed before the command, sets the GPU batch size. It halves itself on an out-of-memory error, as in the other Chronos scripts.
* An interrupted `validate` or `test` resumes where it stopped, because forecasts are saved every 1,024.
* `plan` gives the exact forecast count and `smoke` converts it into minutes on your GPU. By my estimate, `validate` needs roughly 15-20k new forecasts and `test` roughly 7-10k. At the ~10 forecasts/s the earlier studies saw on an RTX 3050, that is under an hour for `validate`.

## 2. How master's three methods map onto Chronos-2

Master's methods assume a model that is trained. Chronos-2 is zero-shot: it is never fitted to our data and only reads a context window of recent history. Each method therefore needs a translation, not a copy.

| | master (`vix_lib.py`) | Chronos-2 version | why this is the right analogue |
|---|---|---|---|
| **base model** | logistic regression, C = 0.003, core features, trained once | the Chronos-2 configuration frozen in `results/chronos_shortlist.json` (h = 1: `joint`, context 512; h = 5: `core`, context 256), raw P(up) | It is already the project's pre-registered Chronos pick, so no new choice is made on the test period. |
| **M1** | **VIX-triggered retraining**: on a regime flip, refit on the last `window` labelled rows | **VIX-triggered context reset**: on a regime flip, the context restarts `min_ctx` sessions before the flip, then grows by one session a day back to the base length | For a zero-shot model the context *is* the training set, so "retrain on recent data" becomes "re-read recent data". The trigger rule (252-day VIX percentile rank crosses `cut`, at least 21 sessions apart) is master's exactly, and a unit test checks this. |
| M1 control | the same model refit every 63 sessions | the same context reset every 63 sessions | Isolates the VIX trigger from the act of resetting, as master did. |
| **M2** | **regime-conditional selection** among always-up, LR-core, LR-all and GBDT-core | the same selection among always-up and 4 Chronos-2 experts: `uni` 256, `uni` 2048, `joint` 512 and `core` 256 | The experts vary context length and inputs, so the method can learn, for example, that short contexts work in high-VIX regimes. master's `regime_select` is reused **unchanged**. |
| **M3** | **volatility-managed exposure** with 20-day realised variance or the VIX as risk signal | the same strategy, with **Chronos-2's next-week SPY volatility forecast** as risk signal | Chronos-2 is a forecaster, so its natural role here is a better variance forecast. master's `vol_managed`, `scale_constant`, `strat_stats` and `sharpe_bootstrap` are reused **unchanged**. |

Grids, all decided before any validation run:

* M1: `cut` ∈ {0.8, 0.9} × `min_ctx` ∈ {63, 126}. This replaces master's `window` ∈ {expanding, 504}.
* M2: `split` ∈ {median, tercile} × `lookback` ∈ {504, all since 2010}. This is master's grid.
* M3: `cap` ∈ {1.5, 2.0}. The variance forecaster is *not* re-chosen; see 3.3.

Selection rules are master's: M1 and M2 by mean fold AUC, M3 by mean fold Sharpe at 5 bps. `select` also records the Chronos baseline's mean fold AUC and master's rv20 strategy's mean fold Sharpe, for reference only.

Test comparisons follow master's, with Chronos counterparts. All use a paired moving-block bootstrap, with Holm correction within each horizon and method over BA, AUC and Brier.

* **M1:** vs the fixed Chronos-2 base (primary); vs the scheduled resets.
* **M2:** vs the fixed Chronos-2 base (primary); vs regime-blind selection; vs always-up.
* **M3:**
  * Sharpe(managed with Chronos variance) − Sharpe(buy-and-hold) at 5 bps (primary) and at 0 bps.
  * **New:** Chronos-managed − master's rv20-managed, cap 2. This asks directly whether a better variance forecast pays.

## 3. Changes and optimizations, and why

### 3.1 M1 restarts the context *at the change* (a methodological fix)

The project's pessimist audit (`experiments/PESSIMIST_AUDIT.md`, issue B1) found that master-style retraining "cannot recover from a change". On a trigger it refits on the last 504 rows, which right after a change are almost all *pre-change*. The corrected version, which trains only on rows after the change, recovered +0.16 balanced accuracy on synthetic drift, against +0.03 for the original.

* Master's VIX M1 still uses the flawed form (`window` ∈ {None, 504}).
* The Chronos version uses the corrected form: after a flip the model sees mostly post-flip history. `min_ctx` (63 or 126 sessions) keeps enough history for the model to work, and the context then grows back to its normal length.
* **Expected effect:** if the VIX regime flip marks a real change, M1 can now react to it. If the null result persists, it can no longer be blamed on this design flaw.

### 3.2 M2's experts vary context length

Master's experts differ in model class and features. Chronos-2 has no model class to vary, but context length is the lever the earlier Chronos study found to matter:

* longer was better for volatility;
* shorter was slightly better for 1-day direction.

Short (256) and long (2048) `uni` experts let the regime choose how much history to trust. The `joint` 512 and `core` 256 experts are the inputs each horizon's earlier study selected.

Two details:

* **Thresholds.** Chronos experts have no training set, so every expert's decision threshold is the base rate of labels known at that row. This is master's own rule for the always-up expert.
* **Calibration caveat.** Chronos-2's raw direction probabilities are worse calibrated than the base rate (Brier, `results/chronos_test_levels.csv`). Because master's selector ranks experts by Brier score, it is likely to pick always-up often. I kept master's selector unchanged so the two studies stay comparable. `vix_test_info.json` will show how often each expert was picked.

### 3.3 M3 uses an already-frozen Chronos variance forecaster

* **What:** the variance signal is the `spy_vol` pick frozen in `results_ext/ext_shortlist.json` (`cov`, context 2048). That model beat 20-day persistence by 32% in MSE on 2019-2025 (Holm p < 0.001).
* **Why not re-select it:** re-tuning it here would add another search over the same test window. The only new choice is the cap, chosen on validation as master did.
* **Conversion:** the forecast is next-week log annualised volatility y, so daily variance = exp(2y) / 252. Any constant bias, such as the Jensen gap from exponentiating a mean log-vol, cancels, because master's `scale_constant` rescales the signal to buy-and-hold volatility on pre-period data.
* **Timing:** decided at the close of t from context ≤ t, held from the open of t+1 to the open of t+2, exactly as master.
* **One difference from master:** Chronos variance forecasts start in 2008 (the study start), so the Chronos strategy's scaling constant is fitted on 2008 → period start. Master's rv20 uses 2006 →. Both use only pre-period data.

### 3.4 Compute: no forecast is computed twice (runtime optimization)

Chronos-2 inference is the only expensive part, so the script caches every forecast under a key of (calendar day, *effective* context length). The effective length is what actually reaches the model after the window is cut at the series' first valid row.

* **Reuses the existing caches:** the 88 direction caches of `chronos_study.py` and the `spy_vol` caches of `chronos_vol_corr_drift.py`, all tracked in git, seed the new caches. Almost all of the base model's fold and test forecasts, every expert's 2013-2025 forecasts except `joint` 512 at h = 5 on the test period, and the variance forecasts for 2013-2025 are therefore free.
* **M1 only recomputes rows whose context changed.** Rows not within reach of a reset have exactly the base context and reuse the base forecast.
* **Shared work across configurations:** the 4 M1 configurations, the folds, the scheduled control and the test all share one cache. A forecast needed twice is computed once.
* **One batch per phase:** `validate` and `test` first gather every forecast the phase needs (`direction_requests`) and fill each cache in one batched pass, instead of many small passes.
* **Resumable:** progress is saved every 1,024 forecasts, so an interrupted run loses at most one chunk.
* **Cost estimates up front:** `plan` counts missing forecasts without loading the model. `smoke` times real forecasts and estimates the minutes.
* **Equivalence check:** `smoke` recomputes 8 cached forecasts per stream and asserts they match the stored ones within 1e-3. That guards against a changed model version or changed inputs silently mixing with old caches.

### 3.5 Protocol hardening beyond master

* `validate` never requests a forecast dated in the test period. Expert histories are cut at the end of VAL and left NaN after it, and the variance frame stops at 2018-12-31.
* `select` refuses to run after `test` has run, because it would overwrite the pre-registration.
* `test` refuses to run twice (as the Chronos scripts do; master's did not). It also checks that four files are unchanged since `select`: the choices file, the two Chronos shortlists it builds on, and `vix_lib.py`.
* `test` checks that the fixed Chronos baseline reproduces `results/chronos_test_levels.csv` (AUC to 1e-6). This confirms the data and caches match the earlier study.
* Extra outputs master did not write: per-year metrics (`vix_test_by_year.csv`) and all test predictions and strategy returns (`vix_test_predictions.pkl`), for later analysis without recomputing.

### 3.6 Considered and not done

* **VIX as a Chronos covariate.** The existing `joint` variant already feeds log VIX, and `core` feeds four VIX features (level, z-score, VIX − RV20, 5-day change). The earlier study's validation covered both. A further covariate variant would repeat that search.
* **Recalibrating Chronos probabilities on VIX flips.** A monotone recalibration leaves AUC unchanged, and AUC is the selection metric, so validation could not choose it fairly. It would also turn the zero-shot model into a fitted one.
* **Holm correction for M3.** Master did not apply one. I kept that for comparability; with two M3 comparisons per cost level, read the p-values as unadjusted.

## 4. Files

| file | status |
|---|---|
| `chronos_experiment/chronos_vix_study.py` | new: the study (commands `plan`, `smoke`, `validate`, `select`, `test`) |
| `experiments/vix_lib.py` | copied **byte-for-byte** from `master` (git blob `0a490cf`), so this branch can import it. A later merge with master sees the same file on both sides and does not conflict. |
| `chronos_experiment/CHRONOS_VIX_REPORT.md` | this report |
| `chronos_experiment/results_vix/` | created by the script. `cache/` holds forecast caches; validation log, choices, test levels, comparisons, by-year, strategy, info and predictions. |
| `chronos_experiment/chronos_vix_preregistration.md` | written by `select` |

## 5. What was verified, and what was not

* **Verified on this machine:**
  * The script compiles.
  * A unit test with stubbed project modules (scikit-learn is blocked by policy on this machine) checks:
    * M1's trigger fires on exactly the rows master's `vix_trigger` rule fires on;
    * context lengths after a reset follow the stated rule;
    * the cache shares entries by effective context, reuses seeded entries, persists and resumes, and records instead of computing in `plan` mode;
    * a forecast dated before the series starts is rejected;
    * expert thresholds equal master's always-up base rate, and expert forecasts are NaN after the validation cut-off;
    * master's `regime_select` runs on the Chronos expert frames.
* **Not verified:** nothing that needs the model or scikit-learn has run. That covers `plan`, `smoke`, `validate`, `select` and `test`. Run `plan` and `smoke` first: they exercise every data path and cache seed before any long job.

## 6. Limitations to state in the write-up

* The 2019-2025 test window has been used by every study in the project, including master's VIX study. This is one more pre-specified look, not a pristine hold-out.
* The earlier Chronos direction forecasts were at chance or below, and no configuration passed its validation rule. M1 and M2 can only redistribute that weak signal, so a large direction gain is not expected. The informative questions are whether the VIX trigger beats scheduled resets (M1) and whether Chronos's variance forecast beats rv20 for exposure management (M3).
* M3's Sharpe differences have wide intervals over 7 years. Master's managed strategy was +0.15 Sharpe over buy-and-hold with a 95% CI of [−0.29, +0.67].
* Parts of this code were written with an AI coding assistant (Claude), which the proposal says must be disclosed.

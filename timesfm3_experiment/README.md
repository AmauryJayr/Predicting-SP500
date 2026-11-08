# TimesFM-3 study (zero-shot SPY direction)

Question: can Google's pretrained TimesFM 3.0 forecaster, used as is with no training on our
data, predict whether SPY goes up on the project's task (h = 1 and h = 5 trading days)?

**Headline.** On the 2019-2025 test period TimesFM-3 reaches AUC 0.482 at h=1 and 0.459 at h=5,
worse than the project's fixed logistic regression (0.508 / 0.536) and slightly below a coin flip.
The setup was pre-registered (`timesfm_preregistration.md`, hash-locked) and the test was run once.

**Start with `TIMESFM_REPORT.md`**: the "In short" section at the top has the result, then
the follow-up checks (all exploratory, made after seeing the test), a file table in section 9,
and each test year in detail as an appendix at the end.
`TIMESFM_OTHER_ASSETS.md` repeats the setup on QQQ, GLD, TLT, IEF.
`timesfm_eli5.md` and `timesfm_terms_explained.md` are plain-language explainers (Traditional Chinese).

## Environment
CPU only. Conda env `timesfm` (Python 3.11): torch (CPU build), `timesfm[torch]==3.0.2`, pandas,
scikit-learn, scipy, joblib, yfinance. Exact install commands: `TIMESFM_REPORT.md`, section 9.
Model weights: `google/timesfm-3.0-pytorch` (downloaded on first model call).

## Re-run
```bash
cd timesfm3_experiment
python timesfm_explore_years.py        # seconds, from saved predictions and forecasts
python timesfm_explore_split.py
python timesfm_explore_spy_years.py
python timesfm_explore_year_detail.py  # also redraws figures/timesfm_year_*.png
python timesfm_study.py report         # validation log + sanity checks
```
Scripts that call the model (`timesfm_study.py validate|select|test`, `timesfm_verify_*.py`,
`timesfm_other_assets.py`) need the weights; see the docstring at the top of each file.

## What is and is not in git
- Tracked: all outputs in `results/timesfm_*`, and the forecast caches `results/timesfm_cache/`
  and `results/timesfm_cache_detrend/`, which the exploratory scripts above need.
- Not tracked (git-ignored): `results/timesfm_{cause,trigger,rest}_cache/` (about 68 MB) and
  `results/*.log`. They are recomputed by the matching `timesfm_verify_*.py` script when missing.

## Paths
`tf_paths.py` sets `RES_DIR` to this folder's `results/` and adds `../experiments` to `sys.path`,
so shared project code (`experiments/common.py`, `experiments/methods.py`, `experiments/run_test.py`) is imported from there, as
`chronos_experiment/` does. The team's `experiments/results/test_predictions.pkl` is only read.
Data comes from `data/` at the repo root. Run scripts from inside this folder.

# Timestral study (zero-shot SPY direction)

Question: can Mistral's pretrained Timestral-209M forecaster, used as is with no training on our
data, predict whether SPY goes up on the project's task (h = 1 and h = 5 trading days)? The study
repeats the finished TimesFM-3 study (`../timesfm3_experiment/`) step for step, so the two can be compared.

**Headline.** On the 2019-2025 test Timestral reaches AUC 0.480 at h=1 and 0.496 at h=5: a coin-flip
level, below the project's fixed logistic regression (0.508 / 0.536) and not distinguishable from
chance (0.500). The pre-registered win rule is not met at either horizon. Against TimesFM-3 (0.482 / 0.459)
the difference is within noise (one uncorrected, descriptive BA difference at h=5, +0.045). The setup was
pre-registered (`timestral_preregistration.md`, hash-locked) and the test was run once. Claude ran the whole
study unattended on 2026-11-08; no human reviewed the pre-registration before the test.

**Start with `TIMESTRAL_REPORT.md`**: the short answer and "In short" at the top, then the headline test
table, comparisons with intervals, validation and selection, checks, limitations (including the
pretraining-leakage caveat: the model card states no data sources or cutoff) and a file table in section 9.
Section 1.6 is exploratory (made after seeing the test).

## Environment
CPU only. Conda env `timestral`, cloned from the existing `timesfm` env (Python 3.11, torch 2.14 CPU,
numpy, pandas, scikit-learn, scipy, joblib, yfinance, huggingface_hub, safetensors); the `timesfm` env is not modified.
```bash
/home/tong-nong/miniconda3/bin/conda create -n timestral --clone timesfm -y
/home/tong-nong/miniconda3/envs/timestral/bin/pip install --no-deps \
  "git+https://github.com/mistralai/timestral.git@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"
```
Model: `mistralai/Timestral-209M`, pinned `revision="9874d252ab0aae16f7f7c999f98f98d6ad7d8365"`
(about 837 MB, downloaded to the Hugging Face cache on first use; public, not gated).
Do not add a file named `timestral.py` to this folder: it would shadow the package.

## Re-run
Always with `PYTHONDONTWRITEBYTECODE=1` (the team tracks `experiments/__pycache__` by mistake), from inside this folder.
```bash
cd timestral
python timestral_explore_years.py     # seconds, from saved test predictions only; redraws figures/timestral_years_h5.png
python timestral_study.py report      # validation log + sanity checks (reads the forecast caches)
```
Steps that call the model need the weights: `timestral_study.py smoke | grid | checks | validate | select | test`
(see the docstring at the top of the file; the grid takes about 2.2 h on 16 CPU cores, the test about 5 minutes).
`test` refuses to run if `results/timestral_shortlist.json` does not match the hash in `timestral_preregistration.md`.
Do not re-run `select` or `test` unless you intend to redo the study: `select` overwrites the shortlist and the pre-registration.

## What is and is not in git
- Tracked: all outputs in `results/timestral_*`, including the forecast caches `results/timestral_cache/`
  (about 8 MB; `timestral_study.py report` and the one-shot `test` read them) and the saved prediction frames
  that `timestral_explore_years.py` needs. The whole folder stays under about 10 MB.
- Not tracked (git-ignored): `results/*.log` and Python bytecode.
- Nothing in this folder was added to git by the study run; staging and committing is left to you.

## Paths
`ts_paths.py` sets `RES_DIR` to this folder's `results/` and adds `../experiments` to `sys.path`, so shared
project code (`experiments/common.py`, `experiments/methods.py`, `experiments/run_test.py`) is imported from there,
as `timesfm3_experiment/` does. The team's `experiments/results/test_predictions.pkl` and TimesFM-3's
`timesfm3_experiment/results/timesfm_test_predictions.pkl` and `timesfm_exploratory_years.csv` are only read.
Data comes from `data/` at the repo root.

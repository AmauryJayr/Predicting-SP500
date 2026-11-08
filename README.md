# Drift-Aware Machine Learning for SPY Direction Prediction

This project tests whether machine-learning models that are periodically retrained on recent data predict SPY's direction better than models that are trained once and kept fixed.

| Notebook | Horizon | Label |
|---|---|---|
| `spy_drift_1day.ipynb` | 1 session | `Close[t+1] > Open[t+1]` |
| `spy_drift_5day.ipynb` | 5 sessions (1 week) | `Close[t+5] > Open[t+1]` |

Both notebooks run the same pipeline. Only `HORIZON` in the configuration cell differs.

## Setup

```bash
pip install -r requirements.txt
jupyter notebook
```

The first run downloads data from Yahoo Finance and caches it in `data/raw_prices.pkl`. Delete that file, or set `FORCE_DOWNLOAD = True`, to refresh the data. Each notebook takes about 3–6 minutes on a laptop.

## Pipeline

1. **Data:** daily OHLCV data for SPY, QQQ, IWM, 9 sector SPDRs, TLT, IEF, GLD and ^VIX from 2006 to 2025. The years 2006–2007 are only a warm-up period for the features.
2. **Quality audit:** missing values, calendar alignment to SPY trading days, and OHLC sanity checks. Forward-fill is limited to 3 days.
3. **Features:** 122 features (44 SPY-only and 78 cross-market). An automated truncation test checks for look-ahead leakage.
4. **Drift diagnostics:** label base rate by year, PSI, adversarial validation, and rolling feature–target correlations.
5. **Models:** a majority-class baseline, logistic regression (L1/L2) and HistGradientBoosting.
6. **Selection:** hyperparameters, retraining frequency and window length are chosen on 2017–2018 validation data only.
7. **Main experiment:** purged walk-forward evaluation over 2019–2025 comparing fixed, expanding-window and rolling-window training.
8. **Statistics:** a paired moving-block bootstrap gives 95% confidence intervals for every comparison.
9. **Extra analyses:**
   - cross-market feature ablation
   - VIX-regime and per-year breakdowns
   - coefficient drift and permutation importance
   - calibration
   - an exploratory trading simulation with transaction costs

Each notebook writes its CSVs and figures to `results/h1/` or `results/h5/`.

## Drift-aware methods study

`drift_methods_findings.ipynb` reports a wider search for drift-aware methods, run as an optimistic research pass followed by an adversarial audit. It loads saved results from `experiments/results/`, so it runs in seconds; its last cell lists the scripts that regenerate those results.

- **SPY direction:** 206 configurations across 8 method families were tried:
  - time-decay weights
  - trailing feature normalization
  - recalibration
  - drift-detector retraining
  - Hedge ensembles
  - online learning
  - regime models
  - scheduled refits

  None beats the same model trained once, on a pre-registered, single-run test. Synthetic tests confirm the methods do adapt when there is a learnable signal; SPY direction does not have one.
- **Other targets:** next-week SPY and TLT volatility, and next-month SPY–TLT correlation.
  - Monthly refitting cuts correlation-forecast MSE by about 15% versus never refitting, after the 2022 stock–bond regime flip. It does not beat 63-day persistence.
  - For volatility, adaptation adds nothing.

Full write-ups:
- `experiments/OPTIMIST_REPORT.md`
- `experiments/PESSIMIST_AUDIT.md`
- `experiments/V2_OPTIMIST_REPORT.md`
- `experiments/V2_PESSIMIST_AUDIT.md`
- **Feature engineering and selection (Part D of the findings notebook):**
  - 66 new features from 15 extra Yahoo series: VIX term structure, VVIX, SKEW, Treasury yields, credit, TIP, EEM/EFA, dollar, oil, commodities, RSP breadth, return-structure transforms and calendar effects.
  - 18 feature-selection strategies, each refit inside every training window.
  - None gives a significant improvement on a pre-registered, single-run test. The best was a drift-aware sign-stability selector, at +0.04 balanced accuracy for h = 5, not significant.
  - Scripts: `experiments/fe_*.py`. Pre-registration: `experiments/fe_preregistration.md`.
- **Non-Yahoo data (Part E):** 21 point-in-time features from FRED, each lagged to its publication date:
  - credit spread, yield curves, real yield and breakeven, dollar;
  - St. Louis financial stress index and jobless claims;
  - the news-based Economic Policy Uncertainty index.

  They raise AUC by about +0.01 in most comparisons, with no significant gain. Scripts: `experiments/fred_*.py`; data: `data/fred/`.

## Pretrained model: TimesFM-3 (zero-shot)

`timesfm3_experiment/` tests Google's pretrained TimesFM-3 forecaster on the same SPY task. It is used as is, with no training on our data. Its settings were chosen on 2013–2018 only, and the 2019–2025 test was pre-registered and run once.

- **Result:** test AUC 0.482 at h = 1 and 0.459 at h = 5. That is slightly below a coin flip and below the fixed logistic regression (0.508 / 0.536). At h = 5 the gap to the logistic regression is clearly real; at h = 1 it could be luck.
- **Follow-up checks:** year-by-year results, what inside the model causes its errors, and the same setup on QQQ, GLD, TLT and IEF. They were made after seeing the test, so they are exploratory.
- **Setup:** a separate conda env (Python 3.11, CPU only, `timesfm` 3.0.2). Shared code is imported from `experiments/`; outputs go to `timesfm3_experiment/results/` and `timesfm3_experiment/figures/`.

Write-up: `timesfm3_experiment/TIMESFM_REPORT.md`; start with its "In short" section. How to run: `timesfm3_experiment/README.md`.

## Pretrained model: Timestral-209M (zero-shot)

`timestral/` repeats the TimesFM-3 study with Mistral's pretrained Timestral-209M, using the same data, test rows, comparisons and win rule. It is used as is, with no training on our data. Its settings were chosen on 2013–2018 only, and the 2019–2025 test was pre-registered and run once.

- **Result:** test AUC 0.480 at h = 1 and 0.496 at h = 5. That is coin-flip level and below the fixed logistic regression (0.508 / 0.536). It beats neither the logistic regression nor the chance rule at either horizon.
- **Compared with TimesFM-3** (0.482 / 0.459): not clearly different. The one nominal gap, +0.045 balanced accuracy at h = 5, was not corrected for multiple comparisons and is not part of the win rule.
- **Follow-up check (exploratory, after the test):** Timestral's 5-day score moves with SPY's past 20-day change, while TimesFM-3's moves against it. Why is not checked.
- **Setup:** a conda env `timestral` (a clone of the TimesFM env plus the `timestral` package at a pinned commit), CPU only. Outputs go to `timestral/results/` and `timestral/figures/`.

Write-up: `timestral/TIMESTRAL_REPORT.md`; start with its "Short answer". How to run: `timestral/README.md`.

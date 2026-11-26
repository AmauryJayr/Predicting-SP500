# TimesFM-3 behaviour experiments: plan (locked before any new forecast)

Exploratory. The 2019-2025 test period was already seen in earlier studies, so nothing here is a confirmatory test. Black-box experiments with controlled inputs; the model is frozen (same `timesfm_lib.load_model`, same `predict_batch` flags as the official pipeline: `make_positive=False, use_symmetric_averaging=False, use_znorm=False, sort_quantiles=True`, `torch.manual_seed(42)` before each call, context 256, horizon 5, CPU). All scores are "P(up)" read from the 9 quantiles exactly as `timesfm_lib.p_up` (price-level input, anchor = last log price, clipped to [0.05, 0.95]). Interpretations in the write-up are marked as readings.

Common constants: `SEED_BEH = 20261125`. "SPY-like" daily volatility and drift = mean / std of SPY daily log-close returns 2008-01-01 to 2018-12-31 (computed in code). Call rule "up" = P(up) > 0.5.

## R. Reproduction (before anything else)
30 test dates (fixed random draw, seed SEED_BEH) with the official `multi` setting (SPY + all 15 side series, price level, ctx 256), computed by my own direct `predict_batch` calls (not `tl.forecast`), compared with `results/timesfm_cache/multi_ctx256.npz`. Report the max abs difference of the quantiles (expect 0). Same for 30 SPY-only dates against `results/timesfm_ctrl_cache/tf_L_TEST.npz`.

## E1. Which side series does TimesFM listen to?
- Dates: a fixed random subsample of 800 of the 1,759 test days (`default_rng(SEED_BEH)`, sorted), because multi costs about 0.27 s/forecast (8 threads) and 22 variants x 1,759 days would take about 3 h.
- Variants (all on the same 800 days, always SPY target + the listed side series, original timing):
  1. 15 "drop one": each single side series removed (the other 14 are passed).
  2. 5 "drop group": equity (QQQ, IWM); sectors (XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY); bonds (TLT, IEF); gold (GLD); VIX.
  3. "scrambled timing, each separately": for each date t and each side series j, draw t'_j uniformly from the positions in [255, min(last pre-2019 position, pos(t)-256)] (one independent draw per series, drawn from `default_rng([SEED_BEH, int(date)])`); the window of series j is its own 256 log-closes ending at t'_j, shifted additively so its last value equals the true log-close of series j at t. SPY is untouched.
  4. "shifted block": same as 3 but one common t' for all 15 series (keeps their relation to each other, breaks their timing against SPY).
- Reference outputs already cached: full (`multi_ctx256`) and SPY-only (`tf_L_TEST`).
- Per variant and h in {1, 5}: mean |P(up)_variant - P(up)_full|; Spearman of P(up) with the full run; % of days where the up call (P>0.5) differs from the full run; AUC vs label (descriptive, secondary; label as in `timesfm_verify_detrend.labels(h)`). Same three numbers for SPY-only vs full, as the "side series removed completely" reference. Comparisons: 22 variants x 2 horizons x 4 measures, descriptive, no correction.

## E2. What does it do with the side series? (all 1,759 test days, no new forecasts)
- effect_h = P(up)_full - P(up)_SPY-only (both cached), h = 1, 5.
- Spearman of effect_h with: for each of the 15 side assets, its trailing 5-day and 20-day log return minus SPY's same-window log return (30 numbers); the 6 cross-asset features (sector_breadth_above_sma50, sector_dispersion_5d, corr_spy_tlt_60d, tlt_ret_20d, iwm_minus_spy_ret_20d, xly_minus_xlp_ret_20d; from `Dataset(h).X("core")` on the test rows); and with the label y_h. Also the Spearman of each of those 36 numbers with the label itself, for sign comparison ("would pushing this way help?").
- Range: 95% moving-block bootstrap (block 20, 1000 resamples, as `timesfm_rootcause_lib`). Also AUC of full vs SPY-only with the paired bootstrap difference (`timesfm_rootcause_lib.evaluate`). Comparisons: 36 features x 2 horizons + label rows; descriptive, no correction.

## E3. Can it use a clean cross-asset signal? (synthetic)
- Per window: x_k and y_k are log prices, k = 0..255 (256 context points, starting at log 100). Daily returns z_x[k], z_e[k] iid N(0,1) (drawn once per window and shared across levels, common random numbers). r_x = sigma * z_x with sigma = SPY-like daily vol, no drift. r_y[k] = sigma * (rho * z_x[k-1] + sqrt(1-rho^2) * z_e[k]), so x leads y by one day and y keeps SPY-like vol. The target is next-day direction: 1[r_y[256] > 0] where r_y[256] uses the last context day's r_x[255] (known: x is a past-only covariate that ends at the same day as y).
- rho per level is found by bisection so the oracle (score = r_x[255]) has next-day direction AUC = 0.52, 0.55, 0.60, 0.70, 0.80 on a 2,000,000-draw reference (plus 0.90 and a null level rho = 0, AUC = 0.50, as controls).
- 3,000 windows per level (same x and noise across levels). Compared per level: oracle score r_x[255] (its realised AUC on the same windows); in-window OLS (regress r_y[k] on r_x[k-1], k = 1..255 with intercept, score = slope * r_x[255]); TimesFM with x as past-only covariate; TimesFM y only (no covariate, same windows). Outputs: AUC of each (with moving-free percentile bootstrap over windows, 1000 resamples, paired TimesFM-with minus TimesFM-without), and Spearman(P(up)_with - P(up)_without, r_x[255]) which is a lower-noise "does it listen" measure. We read off the weakest level where TimesFM-with beats TimesFM-without with the paired range excluding 0 and where the Spearman is clearly positive.

## E4. Reversal: built-in habit or learned from SPY?
- 4,000 simulated 256-point log-price paths each from: (A) iid normal returns, SPY-like drift and vol; (B) GARCH(1,1) with normal innovations, parameters fitted by Gaussian maximum likelihood (own scipy code) to SPY 2008-2018 daily log returns, 500-day burn-in; (C) iid bootstrap (with replacement) of the real 2008-2018 daily log returns. Future unrelated to the past by construction. Univariate TimesFM, price-level input. (Same seeds per family: `default_rng([SEED_BEH, 4, family])`.)
- Per family and h in {1, 5}: Spearman of P(up) with the trailing 1-, 5-, 20-day log return, and with (last log price - mean of last 20 log prices) and (last log price - mean of all 256). Range: +-1.96/sqrt(n) (approximately +-0.03). Same measures for real SPY on the 1,759 test days (cached SPY-only forecasts) and, for context, on every cached `price_ctx256` date.
- Also reported: mean P(up) and the share of paths on which the model's built-in linear-trend line is applied (see E6).

## E5. The rule in a picture
- Backgrounds: 1,000 iid-normal random-walk paths (SPY-like drift and vol) and, separately, 1,000 real SPY 256-day windows drawn at random from pre-2019 history (end position uniform in [255, last pre-2019 position]).
- For k = 5 and k = 20 (separately), the last k daily returns r are replaced by r - mean(r) + log(1+m)/k, i.e. the wiggle shape is kept but the total move over the last k days is exactly m, for m in {-5, -3, -1, 0, +1, +3, +5}%. The remainder of the path is untouched; the log-price path is rebuilt from the returns (the last k days now end at a different level but the earlier part is identical).
- Average P(up) per m, h = 1, 5, with standard error. Figure: `figures/timesfm_behaviour_reversal.png`.

## E6. Which part produces the reversal?
Using the decomposition of `timesfm_rootcause_decompose.py` (network quantiles N before the trend-line add-back, line value c and slope s per step recovered from Q - N, r_t = a - apply*c): on the E4 paths compute (i) network-only P(up) = P(N_h > r_t), (ii) line-slope-only P(up) = P(a + apply*s*h + (N_h - N_h,median) > a), and the same Spearman table with trailing returns. Capture is done by wrapping `model.forward/decode` of the loaded copy at run time (as `timesfm_rootcause_capture.py`); I assert that the captured Q (sorted) equals the pipeline output. Cheap (the same forecasts as E4), so it is run.

## Outputs
`timesfm_behaviour_{run,analyze}.py`, `timesfm_behaviour_sim.py`, `results/timesfm_behaviour_*.csv`, caches in `results/timesfm_behaviour_cache/` (git-ignored), `figures/timesfm_behaviour_reversal.png`, `TIMESFM_BEHAVIOUR.md`, `results/timesfm_behaviour_lock.json`.

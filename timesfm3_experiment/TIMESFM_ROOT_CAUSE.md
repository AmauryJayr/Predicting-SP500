# Why is TimesFM-3 below 0.5? Root-cause tests

Question from the team: *"Why does TimesFM-3 perform worse? What is the root cause? Is it the transformer?"*
Setting: frozen TimesFM-3, zero-shot, `multi` variant, context 256, SPY direction, test 2019-2025 (1,759 days at h=1, 1,755 at h=5). Official AUC 0.482 (h=1) and 0.459 (h=5). All numbers below were computed after seeing the test, so they are **exploratory** (not pre-registered, not corrected for multiple looks). Builds on `TIMESFM_REPORT.md` section 1.3; nothing there is repeated except where needed.

## In short

**Is it the transformer? Partly, but the data do not support "the transformer is the root cause".**

- **The below-0.5 AUC is not significantly below 0.5 for any part.** Official h=5: 0.459 (95% interval 0.412 to 0.509). Transformer part alone: 0.480 (0.434 to 0.526). Detrending-line part alone: 0.470 (0.431 to 0.514). Transformer-only minus official is +0.021 (-0.007 to +0.047): not distinguishable. Switching the detrending off altogether leaves the pooled h=5 AUC unchanged (0.460 vs 0.459).
- **Where the shortfall can be located, it is in the network's output, not the detrending.** In 2020 and 2022 the transformer-only part scores 0.382 and 0.405 at h=5 (official 0.377 and 0.392), and on the 59% of days where no detrending is applied the official score is already 0.392 and 0.388 in those years. In 2025 the detrending line's slope is the weaker part (0.380; transformer part 0.461).
- **But trivial non-transformer forecasters show the same pattern in those years.** h=5 AUC in 2020 / 2022 / 2025: random walk with drift 0.419 / 0.391 / 0.311, AR(5) on returns 0.401 / 0.394 / 0.495, TimesFM official 0.377 / 0.392 / 0.425. Pooled, all seven simple forecasters sit at 0.474 to 0.505 and none differs significantly from TimesFM (h=5). So 2020 and 2022 are hard for anything that extrapolates the past, and we found no evidence that TimesFM's network is worse than such an extrapolator.
- **What we did not show:** why the network behaves the way it does inside (attention, pretraining data), and whether a different model or setting would do better. The test is also too small to detect AUC differences below about 0.04 to 0.05.

Slide figure: `slides/fig/root_cause.png`.

## 1. What was done

| Step | What | Script |
|---|---|---|
| A1 | Re-ran the official forecast for all 1,759 test days with hooks on `model.forward` / `model.decode` of our own loaded copy (installed package untouched) and stored the network's quantiles `N`, the final quantiles `Q`, and RevIN's mean / std | `timesfm_rootcause_capture.py` |
| A2 | Exact decomposition `Q = N + line add-back`, scores for each part, AUC pooled / per year / by detrended or not | `timesfm_rootcause_decompose.py` |
| A3 | Reference: official vs detrending off vs always on (runs cached earlier) | `timesfm_rootcause_detrend_ref.py` |
| B | Seven non-transformer forecasters through the same forecast -> P(up) -> AUC pipeline | `timesfm_rootcause_baselines.py` |
| C | Plain past-return scores (no model), figure | `timesfm_rootcause_market.py`, `timesfm_rootcause_figure.py` |

All AUCs use the official read-out: the 9 quantiles of step h are interpolated into a CDF, P(up) = 1 - CDF(threshold), clipped to [0.05, 0.95]. Intervals: our own moving-block bootstrap (block 20 days, 1,000 resamples, same resamples for all forecasters, so differences are paired). They differ slightly from the team's earlier intervals (official h=5: -0.088 to +0.009 here vs +0.010 before).

### 1.1 Reproduction check (A1)

| Check | Result |
|---|---|
| Instrumented run vs cached official quantiles (all 1,759 x 5 x 9 values) | max abs difference **0.0** (identical) |
| Official AUC recomputed from the instrumented run | 0.482 (h=1), 0.459 (h=5), as in the report |
| Model's own add-back vs numpy line `c + m*h/L` | max abs difference 5e-6 (float32 rounding) |
| `N + add-back = Q` | max abs difference 5e-6 |
| Detrending applied | 40.5% of days (712 of 1,759); the float64 replica of the apply rule disagrees with the model on 1 borderline day (std ratio 0.50009 vs threshold 0.5), so the model's own decision is used |
| RevIN mean / std vs mean / std of the window the network was shown | max abs 1e-6 / max relative 9e-6 |

## 2. Experiment A: where does the official forecast come from?

**How the pieces fit.** On a day where detrending is applied, the model draws a straight line through the 256-day context, gives the network the series *minus the line*, and adds the line back to the network's forecast. In symbols (log-price units, `a` = today's log close, `c` = line value today, `m/L` = line slope per day):

`official median change = (network's median - residual today)  +  h * m/L`

The first term is the network's own call ("will the residual go up or down?"), the second is the line's slope. The identity holds on every test day (max error 5e-6). On days without detrending the second term is 0.

*Worked example (round numbers).* Today's log close `a` = 6.000, the line passes through 5.980 today, slope +0.001 per day. The network sees today's residual as +0.020 and forecasts the residual at day 5 as +0.015 (a pull back of 0.005). Official forecast at day 5 = line 5.980 + 0.005 slope + residual 0.015 = 6.000, i.e. change 0 = network -0.005 + slope +0.005. The three scores are: official P(up) from 6.000 vs 6.000, "transformer part" P(up) from the network's quantiles vs today's residual (0.015 vs 0.020, a down call), and "line slope part" P(up) using only the +0.005 shift (an up call).

Scores: `official`; `Transformer part only` = P(network quantiles > today's residual); `Detrending-line slope only` = P(a + slope + the network's spread around its own median > a), i.e. the line's slope with the network's uncertainty but no network drift on top. Point versions (sign and size of each term) give the same picture and are in `timesfm_rootcause_components.csv`.

### 2.1 Pooled and per year (AUC, h=5)

| Forecaster | Pooled | 95% interval | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|
| TimesFM official | 0.459 | 0.412 to 0.509 | 0.490 | 0.377 | 0.480 | 0.392 | 0.507 | 0.506 | 0.425 |
| Transformer part only | 0.480 | 0.434 to 0.526 | 0.488 | 0.382 | 0.533 | 0.405 | 0.553 | 0.541 | 0.461 |
| Detrending-line slope only | 0.470 | 0.431 to 0.514 | 0.521 | 0.458 | 0.396 | 0.459 | 0.370 | 0.352 | 0.380 |
| No model: -(z-score of today in the window) | 0.473 | 0.426 to 0.522 | 0.651 | 0.538 | 0.588 | 0.574 | 0.427 | 0.535 | 0.482 |
| Random walk + drift (256 d) | 0.496 | 0.452 to 0.545 | 0.489 | 0.419 | 0.509 | 0.391 | 0.479 | 0.369 | 0.311 |
| Line slope from today's close | 0.490 | 0.445 to 0.540 | 0.500 | 0.431 | 0.424 | 0.441 | 0.442 | 0.395 | 0.353 |
| Whole line extrapolated | 0.505 | 0.454 to 0.554 | 0.668 | 0.523 | 0.598 | 0.529 | 0.504 | 0.526 | 0.485 |
| AR(5) on returns | 0.485 | 0.445 to 0.532 | 0.512 | 0.401 | 0.503 | 0.394 | 0.450 | 0.461 | 0.495 |
| AR(1) on log-price | 0.474 | 0.427 to 0.518 | 0.661 | 0.474 | 0.417 | 0.572 | 0.521 | 0.439 | 0.440 |
| Line + AR(1) on the residual | 0.493 | 0.443 to 0.549 | 0.680 | 0.522 | 0.598 | 0.497 | 0.492 | 0.514 | 0.427 |
| ETS damped trend | 0.505 | 0.477 to 0.534 | 0.504 | 0.497 | 0.489 | 0.475 | 0.528 | 0.503 | 0.502 |

*In plain words.* Every row of the pooled column lies between 0.46 and 0.51 and every interval contains 0.5. Official TimesFM is the lowest pooled number, but it is not significantly below its own transformer-only part (+0.021, interval -0.007 to +0.047) or the line-slope part (+0.011, -0.031 to +0.052). The yearly pattern is more informative: 2020 and 2022 are low for the transformer part (0.382, 0.405) and also for a random walk with drift and AR(5); 2025 is low for the line slope (0.380) and for every drift-type forecaster (0.311 to 0.353), but not for the transformer part (0.461). (The row "No model" scores minus today's z-score inside the 256-day window; see 2.4.)

### 2.2 Same table at h=1

| Forecaster | Pooled | 95% interval | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|
| TimesFM official | 0.482 | 0.454 to 0.509 | 0.467 | 0.466 | 0.480 | 0.516 | 0.505 | 0.421 | 0.512 |
| Transformer part only | 0.487 | 0.461 to 0.513 | 0.460 | 0.461 | 0.495 | 0.525 | 0.520 | 0.436 | 0.505 |
| Detrending-line slope only | 0.488 | 0.467 to 0.511 | 0.499 | 0.513 | 0.433 | 0.489 | 0.466 | 0.436 | 0.512 |
| No model: -(z-score of today in the window) | 0.509 | 0.482 to 0.534 | 0.574 | 0.491 | 0.563 | 0.555 | 0.474 | 0.551 | 0.547 |
| Random walk + drift (256 d) | 0.494 | 0.471 to 0.522 | 0.469 | 0.509 | 0.459 | 0.501 | 0.534 | 0.434 | 0.467 |
| Line slope from today's close | 0.498 | 0.475 to 0.526 | 0.488 | 0.509 | 0.445 | 0.529 | 0.511 | 0.484 | 0.491 |
| Whole line extrapolated | 0.523 | 0.497 to 0.549 | 0.595 | 0.481 | 0.571 | 0.567 | 0.501 | 0.548 | 0.533 |
| AR(5) on returns | 0.484 | 0.460 to 0.509 | 0.469 | 0.508 | 0.497 | 0.449 | 0.512 | 0.489 | 0.453 |
| AR(1) on log-price | 0.495 | 0.467 to 0.518 | 0.585 | 0.490 | 0.451 | 0.558 | 0.533 | 0.458 | 0.505 |
| Line + AR(1) on the residual | 0.513 | 0.488 to 0.539 | 0.601 | 0.468 | 0.565 | 0.549 | 0.512 | 0.539 | 0.517 |
| ETS damped trend | 0.506 | 0.480 to 0.534 | 0.471 | 0.547 | 0.532 | 0.462 | 0.429 | 0.470 | 0.543 |

*In plain words.* At h=1 nothing separates: official 0.482, transformer part 0.487, line slope 0.488, and the simple forecasters 0.484 to 0.523. Two simple forecasters that bet on a return to the fitted line ("whole line extrapolated" 0.523, "line + AR(1) on the residual" 0.513) are significantly *above* TimesFM (+0.041, interval 0.009 to 0.070, and +0.031, 0.004 to 0.059), but they are not significantly above 0.5 themselves and they were not chosen in advance, so we do not read more into them.

### 2.3 Does the detrending matter? Detrended vs not-detrended days, and the on/off switch

| h | Subset | n | Official | Transformer part | Line slope | Official 2020 | Official 2022 | Official 2025 |
|---|---|---|---|---|---|---|---|---|
| 5 | detrending applied | 712 | 0.467 | 0.475 | 0.500 | 0.360 (n=46) | 0.463 (n=46) | 0.527 (n=60) |
| 5 | not applied | 1,043 | 0.486 | 0.486 (same) | 0.500 (tie) | 0.392 (n=207) | 0.388 (n=205) | 0.511 (n=185) |
| 1 | detrending applied | 712 | 0.457 | 0.469 | 0.471 | 0.429 | 0.577 | 0.270 |
| 1 | not applied | 1,047 | 0.502 | 0.502 (same) | 0.500 (tie) | 0.458 | 0.510 | 0.549 |

| Variant (h=5) | Pooled | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|
| Official (detrend when fit is good) | 0.459 | 0.490 | 0.377 | 0.480 | 0.392 | 0.507 | 0.506 | 0.425 |
| Detrending off | 0.460 | 0.404 | 0.365 | 0.449 | 0.435 | 0.582 | 0.428 | 0.467 |
| Detrending always on | 0.478 | 0.436 | 0.365 | 0.519 | 0.459 | 0.525 | 0.518 | 0.430 |

*In plain words.* On the 1,043 days with no detrending the forecast **is** the network's output, and 2020 / 2022 are already 0.392 / 0.388 there. So those two years cannot be blamed on the pre/post-processing. Detrending off versus on does not change the pooled AUC (0.460 vs 0.459; always-on 0.478, +0.019 with interval -0.008 to +0.047), although individual years move a lot (2019: 0.404 to 0.490). The small subsets (n = 46 or 60 days) have intervals far wider than the differences shown.

### 2.4 Sizes, and what RevIN does

| (median over days) | h | detrending applied | not applied |
|---|---|---|---|
| \|official median change\| (log) | 5 | 0.0066 | 0.0037 |
| \|network's own change\| | 5 | 0.0029 | 0.0037 |
| \|line slope\| (h x m/L) | 5 | 0.0048 | 0 |
| Share of days the line slope says up | 5 | 98.7% | n/a |
| Share of days the network says up / the official forecast says up | 5 | 61.9% / 96.3% | 60.6% / 60.6% |
| corr(network change, today's position in the window, in sigma units) | 5 | -0.60 | -0.39 |
| Slope of network change (sigma units) on today's z-score | 5 | -0.18 | -0.09 |

*In plain words.* On detrended days the line's slope is larger than the network's own call (0.0048 vs 0.0029), and it says "up" on 98.7% of days because the test period is a bull market; that is why the official P(up) is above 0.5 on 96% of such days. The network itself pulls partway towards the window mean (for each 1 sigma that today is above the mean, it lowers its 5-day forecast by 0.09 to 0.18 sigma), which is the "bet on a turn" already described in section 1.3 of the report. (These are least-squares slopes; they can be driven by a few days with large |z|, and we did not make them robust, so read them as direction, not size.)

**RevIN.** The mean and std RevIN uses are exactly the mean and std of the window the network is shown (max difference 1e-6). The network's output is mapped back with these same two numbers, so in the P(up) score they cancel: computing the network-only score in RevIN units or in log-price units gives the same result to 7e-16. RevIN's location and scale therefore cannot add a bias of their own to the P(up) ranking, and cannot be separated as an extra "component". The only remaining way to read a location effect is the network's response to today's z-score above (the last two rows of the table); a score that uses nothing but "minus today's z-score" scores 0.473 at h=5 (0.509 at h=1), the same as the network, with no model at all. (RevIN's iterative refinement for later horizon patches is not used here: the 5 steps are read from the last context patch, asserted for every batch.)

## 3. Experiment B: non-transformer forecasters, same pipeline

Each forecaster uses only the last 256 log closes up to day t, forecasts the mean path for 5 steps, adds a normal band from its own residual volatility, and goes through the identical P(up) read-out. (`statsmodels` is not installed, so AR is least squares and ETS a small scipy fit written for this study; the ETS variance is the simple `sqrt(h)` approximation.)

| Forecaster | What it does |
|---|---|
| Random walk + drift | continue today's price at the average daily move of the last 256 days |
| Line slope from today's close | continue at the 256-day line's slope (the detrending line's slope, anchored at today) |
| Whole line extrapolated | forecast = the line's value at day t+h (so a price above the line is forecast to come back down) |
| AR(5) on returns | least-squares AR on daily returns, iterated |
| AR(1) on log-price | mean-reverts to the window mean |
| Line + AR(1) on the residual | "detrend, then a simple model on what is left" |
| ETS damped trend | Holt additive damped trend on log price |

Results are in the tables of 2.1 and 2.2 (the last seven rows). Paired differences to official TimesFM at h=5 (baseline minus official): random walk +0.037 (-0.016 to +0.092), line slope +0.031, whole line +0.046, AR(5) +0.026, AR(1) +0.015, line + AR(1) +0.034, ETS +0.045; **all intervals include 0**. All pooled h=5 intervals include 0.5.

*In plain words.* No simple forecaster does clearly better than TimesFM, and none does clearly better than chance. Two of the simplest, random walk with drift and AR(5), have the same low scores in 2020 and 2022 (0.42 / 0.39 and 0.40 / 0.39) as the transformer. Plain past-return rules with no model at all behave alike (`timesfm_rootcause_market.csv`): at h=5 the 60-day return scores 0.331 in 2022, and the 255-day return 0.354 in 2025.

## 4. Putting it together (C)

| Question | Answer from the tests |
|---|---|
| Is the pooled shortfall caused by the transformer? | Not shown. The transformer part alone is 0.480 (h=5), 0.487 (h=1), with intervals containing 0.5 and containing the official value. |
| Is it caused by the pre/post-processing (detrending)? | Not shown either. The slope part alone is 0.470; detrending off gives the same pooled AUC (0.460). |
| Where is the shortfall concentrated? | 2020 and 2022 at h=5: in the network's output (0.382, 0.405; also 0.392, 0.388 on days with no detrending). In 2025 the line slope is the weaker part (0.380 vs 0.461). |
| Do simple forecasters show the same? | Yes in 2020 and 2022 (random walk with drift 0.419 / 0.391, AR(5) 0.401 / 0.394) and in 2025 for the drift-type ones (0.311 to 0.353). Not for ETS damped trend (0.48 to 0.53 in every year). Pooled, none is separable from TimesFM or from chance. |
| So is it the transformer? | **Partly**: in 2020/2022 the low score is in the network's output rather than the line. **But** a one-line extrapolator gets the same low score in the same years, so the evidence points to those years being hard for forecasters that carry the past move forward, not to a defect of the transformer. |

### 4.1 The other pretrained model, Timestral-209M (from this repo only)

From `timestral/TIMESTRAL_REPORT.md` and `timestral/README.md`: Timestral-209M (`mistralai/Timestral-209M`, zero-shot, same test rows) has AUC 0.480 at h=1 and 0.496 at h=5; at h=5 it is below 0.5 in every single year (0.367 to 0.483) although the pooled value is 0.496; its h=5 score moves *with* SPY's last 20 days (correlation +0.39 to +0.64 in every year), whereas TimesFM-3's moves against it (-0.21 to -0.69). The Timestral report says why it does this "was not checked". It was run on the `ret` (daily returns) input, chosen on the validation period, not on the price-level `multi` input used for TimesFM. Its architecture: the repo reports only a patch size of 32, block-autoregressive decoding (read from code, `_block_ar_decode`), internal normalisation and a 21-quantile grid; it points to the model card for the architecture and does not reproduce it, so we do not state it here.

*Reading (an inference, not a test).* Two pretrained models with opposite habits (one follows the recent trend, one bets on a turn) both land at 0.46 to 0.50, and both are low in the same kind of years. That fits "SPY direction is close to unpredictable from past prices, and extrapolators lose in reversal years" better than "one architecture is broken". We did not run an experiment that could separate these two explanations for Timestral.

## 5. Checked vs not found / not verified

**Checked**
- The instrumented run reproduces the official quantiles exactly (diff 0.0); the decomposition `official = network + line` holds on every day (error 5e-6).
- Detrending is applied on 40.5% of days; on those days the line slope is larger than the network's own median change (h=5: 0.0048 vs 0.0029) and mostly "up".
- Pooled AUC of the transformer part, the slope part and every simple forecaster: all between 0.46 and 0.51, all intervals contain 0.5, no paired difference to official TimesFM at h=5 excludes 0.
- On days without detrending, 2020 and 2022 are already low (0.392, 0.388) at h=5, so those two years are not a pre/post-processing effect.
- RevIN's mean / std equal the window mean / std, and cancel in the P(up) ranking.
- Random walk with drift and AR(5) score as low as TimesFM in 2020 and 2022.

**Not found / not verified**
- No component is shown to be significantly below 0.5 (pooled). The yearly lows (about 0.38 to 0.40 in 2020 and 2022, n of about 250 days with overlapping 5-day labels) were picked after seeing the test and are not corrected for the number of years and variants looked at.
- Why the network's output is low in 2020 and 2022 (what inside the model drives it) is not tested; we did not ablate layers, attention or training data, and the pretraining data are not known.
- We do not know whether another context length, the `price` / `ret` variants, or a different frozen model would behave differently here beyond what earlier reports show.
- The "line slope is bad in 2023-2025" finding (h=5: 0.370, 0.352, 0.380) is descriptive; we did not test why trend slope would rank the next 5 days badly in those years.
- ETS and AR here are own simple implementations, not `statsmodels`; the band widths are approximations (only ranking inside each forecaster matters for AUC, but the clip at 0.05 / 0.95 can tie extreme days).
- Power: intervals are about +/-0.045 wide at h=5, so a true difference of 0.02 to 0.03 between components cannot be detected or ruled out.

## 6. Reproduce

All from `timesfm3_experiment/` with `/home/tong-nong/miniconda3/envs/timesfm/bin/python` (timesfm 3.0.2, CPU, 8 threads; times measured while another job was using the CPU).

| Step | Command | Time |
|---|---|---|
| Capture (resumable; cache in git-ignored `results/timesfm_rootcause_cache/`, log `results/timesfm_rootcause_capture.log`) | `python timesfm_rootcause_capture.py` | about 12 min (0.3 to 0.7 s per forecast) |
| Decomposition | `python timesfm_rootcause_decompose.py` | under 1 min |
| Detrending on/off reference (needs the caches of `timesfm_verify_detrend.py run off / always`, present) | `python timesfm_rootcause_detrend_ref.py` | under 1 min |
| Baselines | `python timesfm_rootcause_baselines.py` | about 1 min (ETS about 50 s) |
| Plain past-return scores | `python timesfm_rootcause_market.py` | seconds |
| Figure and summary | `python timesfm_rootcause_figure.py` | seconds |

Outputs: `results/timesfm_rootcause_{repro,components,subsets,sizes,baselines,detrend_ref,market,summary}.csv`, `results/timesfm_rootcause_baseline_scores.csv.gz`, `slides/fig/root_cause.png`. Shared helpers: `timesfm_rootcause_lib.py`.

## 7. Proposed slide

**Title:** Is the transformer why TimesFM-3 is below 0.5?

- Not shown: the network alone scores 0.480 (h=5), the detrending line 0.470, official 0.459; all within noise of 0.5.
- 2020 and 2022 are low in the network's output, but a plain random walk with drift and AR(5) are just as low there.
- Two pretrained models with opposite habits both end near 0.5, consistent with SPY direction being close to unpredictable from past prices (not tested directly).

**Speaker notes (about 15 seconds):** "We split the official forecast into the transformer's part and the detrending line's part. Neither is significantly below one half, and removing the detrending changes nothing overall. The weak years, 2020 and 2022, sit in the network's output, but simple extrapolators score just as low there. So we cannot show a transformer defect; the simplest reading is that extrapolating past moves does not predict SPY direction."

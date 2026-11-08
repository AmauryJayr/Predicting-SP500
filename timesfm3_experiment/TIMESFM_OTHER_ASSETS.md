# TimesFM-3 on other assets (exploratory)

If SPY is too hard, does zero-shot TimesFM-3 do better on the Nasdaq-100 (QQQ), gold (GLD) or US Treasuries (TLT for long-term bonds, IEF for medium-term bonds)?

**Short answer.** No. QQQ behaves like SPY. The two bond funds are at coin-flip level. Gold at h=5 is slightly above 0.5, but the evidence is weak: it is 1 of 10 tests, its interval only just clears 0.5, and it would not survive a correction for running 10 tests.

This check was made **after** the SPY test and was not planned in advance. It is outside the project's SPY scope and is kept here as a side note to `TIMESFM_REPORT.md`.

---

## 1. Results

### 1.1 AUC, 2019–2025

| asset | h=1 AUC [95% CI] | h=5 AUC [95% CI] | correlation of P(up) with the past 20-day move (h=5) |
|---|---|---|---|
| SPY (S&P 500) | 0.500 [0.475, 0.524] | 0.453 [0.406, 0.499] | −0.36 |
| QQQ (Nasdaq-100) | 0.509 [0.481, 0.536] | 0.472 [0.422, 0.518] | −0.31 |
| GLD (gold) | 0.509 [0.484, 0.537] | **0.543 [0.501, 0.583]** | −0.22 |
| TLT (long-term Treasuries) | 0.498 [0.471, 0.524] | 0.496 [0.450, 0.540] | −0.07 |
| IEF (medium-term Treasuries) | 0.492 [0.468, 0.518] | 0.500 [0.455, 0.551] | −0.03 |

1,755 days per asset and horizon. The last column shows how strongly TimesFM bets on a turn (report section 1.3): a negative value means that the more the asset rose over the past 20 days, the lower TimesFM's P(up).

**In plain words:**

- **QQQ is SPY again.** At h=5 it is below 0.5, and the bet on a turn is about as strong as on SPY. The two indices move closely together, so this is no surprise.
- **Bonds are a coin flip.** Both bond funds sit at 0.50 at both horizons. The bet on a turn almost disappears for them, but the model doesn't get better at ranking days either.
- **Gold at h=5 is the only interval entirely above 0.5,** but its lower end is 0.501, right at the line.
- **h=1 is a coin flip for every asset** (0.49–0.51, every interval contains 0.5).

### 1.2 Gold, year by year (h=5)

| 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|
| 0.512 | 0.465 | 0.525 | 0.418 | 0.570 | 0.527 | 0.583 |

**In plain words:** gold's 0.543 is not steady. 2020 and 2022 are below 0.5, and 2022 (0.418) is close to SPY's worst years. The best years are 2023 and 2025. Each year has only about 250 days, so a single year moves a lot by chance.

### 1.3 Compared with one-line rules (h=5)

Each rule scores a day by the asset's own past log change: the bigger the rise, the higher the score.

| asset | TimesFM-3 | past 5 days | past 20 days | past 256 days |
|---|---|---|---|---|
| SPY | 0.453 | 0.499 | 0.503 | 0.501 |
| QQQ | 0.472 | 0.494 | 0.513 | 0.494 |
| GLD | 0.543 | 0.483 | 0.486 | 0.509 |
| TLT | 0.496 | 0.491 | 0.503 | 0.505 |
| IEF | 0.500 | 0.486 | 0.514 | 0.536 |

**In plain words:** the simple rules are all close to 0.5 on every asset, gold included (0.48–0.51). So gold is not an easy asset to call either. TimesFM's small edge on gold is more likely luck than a real skill.

## 2. Why the gold number is not evidence

1. **Ten tests, one pass.** 5 assets × 2 horizons is 10 tests. Even if all ten were pure coin flips, each 95% interval would have a 2.5% chance of landing entirely above 0.5 by luck, so at least one of the ten would do so about 22% of the time (1 − 0.975¹⁰). After a correction for 10 tests (for example Bonferroni), gold's result is no longer significant.
   - **Tiny example.** Ask 10 people to flip a coin 20 times each. One of them will usually look "lucky". That doesn't make them good at flipping coins.
2. **The settings were chosen on SPY.** Context length, input and detrending all come from the SPY study. Nothing was pre-registered for gold.
3. **No simple rule finds anything on gold** (section 1.3), so there is no known pattern in gold that TimesFM could be picking up.

**What a clean test would need.** Fix the setup in advance and score it once on data nobody has looked at, for example 2026. 2026 so far has only about 200 days, so the interval would be roughly ±0.13 wide (compare the 2026 SPY check in report section 1.7) and could not tell 0.54 from 0.50.

## 3. Setup

- **Model and input.** TimesFM-3, zero-shot, with each asset's own log close only (the `price` input), 256-day context, detrending as shipped. This is *not* the official SPY setup, which adds 15 other tickers (`multi`). That is why SPY scores 0.453 here and 0.459 in the report.
- **Score.** P(up) from the forecast quantiles, as in report section 4.2. No calibration; the report's offset-only calibration doesn't change AUC anyway.
- **Labels.** The same definitions as the project, per asset: h=1 is `Close[t+1] > Open[t+1]`, h=5 is `Close[t+5] > Open[t+1]`.
- **Days.** All trading days from 2019-01-02 to 2025-12-31 that have a `Close[t+5]`: 1,755 per asset and horizon (the report's h=1 test has 1,759).
- **Intervals.** Moving-block bootstrap of the AUC, 20-day blocks, 1,000 draws, percentile 95% interval.

## 4. Reproduce

```bash
cd experiments
/home/<user>/miniconda3/envs/timesfm/bin/python timesfm_other_assets.py   # ~3 min on CPU
```

| File | Content |
|---|---|
| `experiments/timesfm_other_assets.py` | forecasts, AUCs, intervals, yearly AUCs, one-line rules |
| `results/timesfm_other_assets.csv` | one row per asset and horizon; all numbers in this note |

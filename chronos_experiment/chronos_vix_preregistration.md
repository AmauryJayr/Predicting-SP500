# Chronos-2 VIX study pre-registration

Choices file `chronos_experiment/results_vix/vix_choices.json`  SHA-256: `ac23bb885cfdaae8eac3fb6e11f986c767713be6ed779f339c67e803e3692f10`

Choices made on folds 2013-14, 2015-16, 2017-18 only (`results_vix/vix_validation_log.csv`).
Selection rules (master's): M1 and M2 by mean fold AUC; M3 by mean fold Sharpe at 5 bps.

Frozen inputs (SHA-256):
- `chronos_experiment/results/chronos_shortlist.json`: `336ce5edcd939de5cd9a46b0b1c5d63efcad0642b9dd8247a02709a2e3997b6e`
- `chronos_experiment/results_ext/ext_shortlist.json`: `e6f21fb30662b8bb83dad06b8876d3aabfd4adc412d44547173a2977f318de83`
- `experiments/vix_lib.py`: `23bf30c964c0780cc9d6218d141ba3dbda3f0bae2602186c83ddc7715954eb9b`

- h=5: M1 = {'cut': 0.9, 'min_ctx': 126}, M2 = {'split': 'median', 'lookback': 504}
- h=1: M1 = {'cut': 0.8, 'min_ctx': 126}, M2 = {'split': 'tercile', 'lookback': None}
- M3 = Chronos-2 variance forecast, cap 1.5, 5 bps per unit turnover

Test comparisons (paired block bootstrap; Holm within horizon and method over BA/AUC/Brier):
- M1 vs the fixed Chronos-2 base (primary); M1 vs the same context reset every 63 sessions (isolates the VIX trigger).
- M2 vs the fixed Chronos-2 base (primary); M2 vs regime-blind selection with the same lookback (isolates the regime); M2 vs always-up.
- M3: Sharpe of the Chronos-managed strategy minus buy-and-hold at 5 bps (primary) and 0 bps; minus master's rv20-managed strategy (cap 2) at 5 and 0 bps (secondary: does a better variance forecast pay?).
Caveat: 2019-2025 has been evaluated in earlier studies of this project, including master's VIX study.

# VIX study pre-registration

Choices file `results/vix_choices.json`  SHA-256: `e61fae7a5141702c080bcca5d24a7ec93746c892e2ece66b4a66bcb6890de8f1`

Choices made on folds 2013-14, 2015-16, 2017-18 only.

- h=5: M1 = {'cut': 0.9, 'window': 504}, M2 = {'split': 'median', 'lookback': 504}
- h=1: M1 = {'cut': 0.9, 'window': 504}, M2 = {'split': 'tercile', 'lookback': 504}
- M3 = signal rv20, cap 2.0, 5 bps per unit turnover; secondary VIX variant = {'signal': 'vix', 'cap': 1.5}

Test comparisons (paired block bootstrap; Holm within horizon and method over BA/AUC/Brier):
- M1 vs the same model trained once (primary); M1 vs scheduled 63-day refits with the same window (isolates the trigger).
- M2 vs the trained-once base model (primary); M2 vs regime-blind selection with the same lookback (isolates the regime); M2 vs always-up.
- M3: Sharpe of the managed strategy minus buy-and-hold at 5 bps (primary) and 0 bps; same for the best VIX-based variant (secondary).
Caveat: 2019-2025 has been evaluated in earlier studies of this project.
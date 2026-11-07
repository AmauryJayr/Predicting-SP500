# ARMAX ladder pre-registration

Ladder file `results/armax_ladder.json`  SHA-256: `6af6862a61d23e0f6c5f73a16328db73c1adf57f27f6fec3db173642270be714`

Choices made on folds 2013-14, 2015-16, 2017-18 only: L5 exog set by mean AUC, L6 scheme by mean Brier.

- h=5: L5 exog = X0, L6 scheme = expanding  (validation: {'exog_mean_auc': {'X0': 0.516, 'X1': 0.5051, 'X2': 0.4064}, 'scheme_mean_brier': {'expanding': 0.24012, 'fixed': 0.24064, 'rolling1008': 0.24067, 'rolling504': 0.24281}})
- h=1: L5 exog = X0, L6 scheme = rolling1008  (validation: {'exog_mean_auc': {'X0': 0.5003, 'X1': 0.4738, 'X2': 0.4791}, 'scheme_mean_brier': {'expanding': 0.2486, 'fixed': 0.24881, 'rolling1008': 0.24844, 'rolling504': 0.24923}})

Test comparisons (paired block bootstrap, Holm within horizon and comparison family over BA/AUC/Brier):
each step vs the previous step; each step vs always-up; L6 vs the notebook's fixed logistic regression.
Caveat: 2019-2025 has been evaluated in earlier studies of this project.
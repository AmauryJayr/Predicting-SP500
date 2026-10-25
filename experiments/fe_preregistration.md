# Feature-study pre-registration

Written by `fe_select.py` from validation results only (folds 2013-14, 2015-16, 2017-18).
The 2019-2025 test is run once by `fe_test.py`, which refuses to run if the shortlist hash differs.

Shortlist file: `results/fe_shortlist.json`  SHA-256: `6fd62bd195ba25848524fc2d90a3f5505a76298e93ab8ec7d311d48316480012`

| id | horizon | selector | model | scheme | val mean AUC | VAL AUC | base122 mean AUC |
|---|---|---|---|---|---|---|---|
| h5.1 | 5 | mi_10 | LR_C0.01 | fixed | 0.5503 | 0.5911 | 0.4723 |
| h5.2 | 5 | family_fwd | LR_C0.01 | fixed | 0.5218 | 0.5585 | 0.4723 |
| h5.3 | 5 | stable_25 | LR_C0.1 | refit | 0.5205 | 0.5258 | 0.4897 |
| h5.4 | 5 | perm_25 | GBDT | refit | 0.5167 | 0.5128 | 0.4816 |
| h5.5 | 5 | pca_5 | GBDT | fixed | 0.5133 | 0.5167 | 0.4657 |
| h1.1 | 1 | cluster_0.7 | GBDT | fixed | 0.5367 | 0.531 | 0.5107 |
| h1.2 | 1 | family_fwd | GBDT | refit | 0.5265 | 0.5284 | 0.49 |
| h1.3 | 1 | all188 | GBDT | fixed | 0.526 | 0.531 | 0.5107 |
| h1.4 | 1 | l1stab_0.6 | GBDT | fixed | 0.5215 | 0.5324 | 0.5107 |
| h1.5 | 1 | coreplus33 | GBDT | refit | 0.5148 | 0.5156 | 0.49 |

Confirmatory comparisons per shortlisted config (Holm within horizon over shortlist x {BA, AUC}):
(a) vs base122 counterpart (same model and scheme); (b) vs chance.
Secondary: same selector/model with the other scheme (fixed vs refit-with-reselection).
Note: the 2019-2025 period has been evaluated in earlier studies of this project, so these results
are not a pristine hold-out; treat them as one more pre-specified look.
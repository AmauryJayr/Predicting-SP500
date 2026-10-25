# FRED-features pre-registration

Shortlist `results/fred_shortlist.json`  SHA-256: `510597fc41a89461d78a747c9a01f285d4912af2256ebb7dd6676e140bb05499`

Family A (no selection): base122 + 21 FRED features vs base122, every model x scheme.
Family B: top-3 FRED-containing feature sets by validation mean AUC (VAL AUC > 0.5), vs twin without FRED.
Holm within (family, horizon) over {BA, AUC}. Also reported: each config vs chance.

| id | h | feature set | twin | model | scheme |
|---|---|---|---|---|---|
| A.h5.0 | 5 | base_fred | base122 | LR_C0.01 | fixed |
| A.h5.1 | 5 | base_fred | base122 | LR_C0.01 | refit |
| A.h5.2 | 5 | base_fred | base122 | LR_C0.1 | fixed |
| A.h5.3 | 5 | base_fred | base122 | LR_C0.1 | refit |
| A.h5.4 | 5 | base_fred | base122 | GBDT | fixed |
| A.h5.5 | 5 | base_fred | base122 | GBDT | refit |
| A.h1.6 | 1 | base_fred | base122 | LR_C0.01 | fixed |
| A.h1.7 | 1 | base_fred | base122 | LR_C0.01 | refit |
| A.h1.8 | 1 | base_fred | base122 | LR_C0.1 | fixed |
| A.h1.9 | 1 | base_fred | base122 | LR_C0.1 | refit |
| A.h1.10 | 1 | base_fred | base122 | GBDT | fixed |
| A.h1.11 | 1 | base_fred | base122 | GBDT | refit |
| B.h5.12 | 5 | base_fred | base122 | LR_C0.1 | fixed |
| B.h5.13 | 5 | spy4_fred | spy4 | LR_C0.01 | fixed |
| B.h5.14 | 5 | base_fred | base122 | LR_C0.1 | refit |
| B.h1.15 | 1 | core_fred | core23 | GBDT | refit |
| B.h1.16 | 1 | base_fred | base122 | GBDT | fixed |
| B.h1.17 | 1 | core_fred | core23 | GBDT | fixed |

Caveat: 2019-2025 has been evaluated in earlier studies of this project (not a pristine hold-out).
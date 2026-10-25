"""EXPLORATORY (post-test) diagnostics. Nothing here changes the preregistered results.
(a) validation only: mean delta of all drift-aware configs vs primary counterpart, per fold
    (the fold's fixed model has 5 / 7 / 9 years of history);
(b) test: fixed primary counterparts vs the chance baseline (block bootstrap)."""
import pandas as pd
from common import RES_DIR, paired_bootstrap
D = pd.read_csv(RES_DIR / "validation_deltas.csv")
a = D.groupby(["horizon", "fold"])[["dBA_primary", "dAUC_primary", "dBrier_primary"]].agg(["mean", lambda s: (s > 0).mean()])
a.columns = [f"{c}_{'mean' if s == 'mean' else 'share_pos'}" for c, s in a.columns]
print("(a) validation deltas vs trained-once counterpart, all 206 entries\n", a.round(3).to_string())
P = pd.read_pickle(RES_DIR / "test_predictions.pkl")
rows, seen = [], set()
for (h, i, role), p in P.items():
    if role != "primary":
        continue
    k = (h, tuple(p["proba"].round(10)))
    if k in seen:
        continue
    seen.add(k)
    rows.append({"horizon": h, "primary_of": f"{h}.{i}", **paired_bootstrap(p, P[(h, 0, "chance")], int(h))})
b = pd.DataFrame(rows)
print("(b) fixed counterparts vs chance on test (EXPLORATORY)\n", b[["horizon", "primary_of", "d_balanced_acc", "d_balanced_acc_lo",
      "d_balanced_acc_hi", "p_balanced_acc", "d_roc_auc", "d_roc_auc_lo", "d_roc_auc_hi", "p_roc_auc", "d_brier"]].round(4).to_string(index=False))
a.to_csv(RES_DIR / "exploratory_val_delta_by_fold.csv"); b.to_csv(RES_DIR / "exploratory_fixed_vs_chance.csv", index=False)

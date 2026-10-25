"""Merge validation rounds and apply the (validation-only) shortlist rule.

Rule: for each horizon, an entry is eligible if, versus its PRIMARY (trained-once) counterpart,
dBA > 0 in >= 2 of the 3 folds AND dAUC > 0 in >= 2 of the 3 folds (F1 2013-14, F2 2015-16, VAL 2017-18).
Eligible entries are ranked by score = mean dBA + mean dAUC; keep the best entry per family,
then the 6 families with the highest score.
"""
import json
import pandas as pd
from common import RES_DIR
import configs, configs2
from methods import spec_key

logs = [pd.read_csv(RES_DIR / f"validation_log{t}.csv").assign(round=r) for t, r in [("", 1), ("_r2", 2)]]
pd.concat(logs).to_csv(RES_DIR / "validation_log.csv", index=False)
(RES_DIR / "validation_log_r2.csv").unlink()
D = pd.concat([pd.read_csv(RES_DIR / f"validation_deltas{t}.csv").assign(round=r) for t, r in [("", 1), ("_r2", 2)]])
D.to_csv(RES_DIR / "validation_deltas.csv", index=False); (RES_DIR / "validation_deltas_r2.csv").unlink()
S = pd.concat([pd.read_csv(RES_DIR / f"validation_summary{t}.csv").assign(round=r) for t, r in [("", 1), ("_r2", 2)]])
(RES_DIR / "validation_summary_r2.csv").unlink()
pos = D.assign(ba_pos=D.dBA_primary > 0, auc_pos=D.dAUC_primary > 0).groupby(["horizon", "family", "label"]) \
    [["ba_pos", "auc_pos"]].sum().reset_index()
S = S.merge(pos, on=["horizon", "family", "label"])
S["eligible"] = (S.ba_pos >= 2) & (S.auc_pos >= 2)
S.to_csv(RES_DIR / "validation_summary.csv", index=False)

entries = {(e["family"], e["label"]): e for e in configs.grid() + configs2.grid()}
short = {}
for h in (5, 1):
    a = S[(S.horizon == h) & S.eligible].sort_values("score", ascending=False)
    best = a.groupby("family", sort=False).head(1).head(6)
    print(f"\n=== h={h} shortlist ===")
    print(best[["family", "label", "dBA", "dAUC", "dBrier", "dBA_sec", "dAUC_sec", "BA", "AUC", "Brier", "ba_pos", "auc_pos", "score"]].round(4).to_string(index=False))
    short[h] = [dict(entries[(r.family, r.label)], val=dict(dBA=r.dBA, dAUC=r.dAUC, dBrier=r.dBrier, BA=r.BA, AUC=r.AUC,
                                                            Brier=r.Brier, score=r.score)) for r in best.itertuples()]
json.dump(short, open(RES_DIR / "shortlist.json", "w"), indent=1, default=str)

"""Shortlist rule for the feature study. Written BEFORE looking at fe_validation_log.csv.

Per horizon:
  1. score = mean ROC-AUC over folds F1, F2, VAL (absolute skill, not 'repair' of a weak counterpart).
  2. eligible = VAL AUC > 0.5  AND  mean AUC > mean AUC of the base122 counterpart (same model, same scheme).
  3. shortlist = the top-5 eligible configurations, at most one per selector family
     (family = selector name before the first '_', e.g. 'mi', 'l1stab').
  4. each shortlisted config is compared on TEST with
       (a) its base122 counterpart [confirmatory, Holm over the shortlist x {BA, AUC}],
       (b) chance (BA = 0.5, AUC = 0.5) [confirmatory, Holm likewise],
       (c) the same selector/model with the other scheme (fixed <-> refit) [secondary].
Writes results/fe_shortlist.json and fe_preregistration.md (with the SHA-256 of the shortlist file).
"""
import hashlib
import json

import pandas as pd

from common import EXP_DIR, RES_DIR

log = pd.read_csv(RES_DIR / "fe_validation_log.csv")
agg = (log.pivot_table(index=["horizon", "selector", "model", "scheme"], columns="fold", values="roc_auc")
       .assign(mean_auc=lambda d: d[["F1", "F2", "VAL"]].mean(axis=1)).reset_index())
base = agg[agg.selector == "base122"].set_index(["horizon", "model", "scheme"])["mean_auc"]
agg["base_mean_auc"] = [base[(h, m, s)] for h, m, s in zip(agg.horizon, agg.model, agg.scheme)]
agg["eligible"] = (agg["VAL"] > 0.5) & (agg["mean_auc"] > agg["base_mean_auc"]) & (agg.selector != "base122")
agg["family"] = agg.selector.str.split("_").str[0]
agg.sort_values(["horizon", "mean_auc"], ascending=[True, False]).to_csv(RES_DIR / "fe_validation_table.csv", index=False)

shortlist = []
for h, g in agg[agg.eligible].sort_values("mean_auc", ascending=False).groupby("horizon", sort=False):
    seen = set()
    for _, r in g.iterrows():
        if r.family in seen:
            continue
        seen.add(r.family)
        shortlist.append({"id": f"h{h}.{len(seen)}", "horizon": int(h), "selector": r.selector, "model": r.model,
                          "scheme": r.scheme, "val_mean_auc": round(float(r.mean_auc), 4),
                          "val_auc_VAL": round(float(r.VAL), 4), "base_mean_auc": round(float(r.base_mean_auc), 4)})
        if len(seen) == 5:
            break

path = RES_DIR / "fe_shortlist.json"
path.write_text(json.dumps(shortlist, indent=2))
digest = hashlib.sha256(path.read_bytes()).hexdigest()

lines = ["# Feature-study pre-registration", "",
         "Written by `fe_select.py` from validation results only (folds 2013-14, 2015-16, 2017-18).",
         "The 2019-2025 test is run once by `fe_test.py`, which refuses to run if the shortlist hash differs.", "",
         f"Shortlist file: `results/fe_shortlist.json`  SHA-256: `{digest}`", "",
         "| id | horizon | selector | model | scheme | val mean AUC | VAL AUC | base122 mean AUC |",
         "|---|---|---|---|---|---|---|---|"]
for s in shortlist:
    lines.append(f"| {s['id']} | {s['horizon']} | {s['selector']} | {s['model']} | {s['scheme']} | "
                 f"{s['val_mean_auc']} | {s['val_auc_VAL']} | {s['base_mean_auc']} |")
lines += ["", "Confirmatory comparisons per shortlisted config (Holm within horizon over shortlist x {BA, AUC}):",
          "(a) vs base122 counterpart (same model and scheme); (b) vs chance.",
          "Secondary: same selector/model with the other scheme (fixed vs refit-with-reselection).",
          "Note: the 2019-2025 period has been evaluated in earlier studies of this project, so these results",
          "are not a pristine hold-out; treat them as one more pre-specified look."]
(EXP_DIR / "fe_preregistration.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

"""v2 shortlist selection (uses only results/v2_validation_log.csv, i.e. data through 2018).

Rule (identical for every target):
  0. Best baseline (BB) = baseline with the lowest mean validation MSE over F1, F2, VAL.
  1. Eligibility vs the matched trained-once counterpart (same model + features):
       rel_f = 1 - MSE(config, f) / MSE(fixed, f);   require rel_VAL > 0 AND mean_f rel_f > 0.
     VAL (the fold closest to the test) cannot be outvoted by F1/F2 (fixes audit issue B3).
  2. Ranking by ABSOLUTE skill, not by repair: skill_f = 1 - MSE(config, f) / MSE(BB, f); rank on mean_f skill_f.
     A config that only repairs a weak fixed model (e.g. the tree model) ranks below one that is good in absolute
     terms.
  3. Up to 4 rule slots: the best eligible config of each method family, in rank order (one per family).
  4. 2 a-priori "practitioner default" slots, the same for every target and not chosen by validation score:
       lin expanding refit every 21 sessions;  lin rolling 1008-session window refit every 21 sessions.
     If a default is already selected in step 3 the slot is filled by the next family in the step-3 ranking.
  Total <= 6 configurations per target. "Rank-1" (top rule slot) is the designated 'best adaptive' for the rolling
  plot and per-year tables.
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd

import v2_lib as L

DEFAULTS = [{"method": "refit", "R": 21, "W": None}, {"method": "refit", "R": 21, "W": 1008}]


def select(log: pd.DataFrame, target: str):
    d = log[log.target == target]
    base = d[d.role == "baseline"].pivot_table(index="baseline", columns="fold", values="mse")
    bb = base.mean(1).idxmin()
    bb_mse = base.loc[bb]
    c = d[d.role == "config"].copy()
    c["skill"] = 1 - c["mse"] / c["fold"].map(bb_mse)
    S = c.pivot_table(index="key", columns="fold", values=["rel_vs_fixed", "skill"])
    S.columns = [f"{a}_{b}" for a, b in S.columns]
    S["rel_mean"] = S[[f"rel_vs_fixed_{f}" for f in L.FOLDS]].mean(1)
    S["skill_mean"] = S[[f"skill_{f}" for f in L.FOLDS]].mean(1)
    meta = c.drop_duplicates("key").set_index("key")[["label", "family", "model", "feats"]]
    S = S.join(meta)
    S = S[S.family != "fixed"]
    S["eligible"] = (S["rel_vs_fixed_VAL"] > 0) & (S["rel_mean"] > 0)
    S = S.sort_values("skill_mean", ascending=False)
    E = S[S.eligible]
    per_family = E.groupby("family", sort=False).head(1)
    picks = list(per_family.index[:4])
    slots = {k: "rule" for k in picks}
    # a-priori defaults on the linear model class
    lin_model, lin_feats = [mc for mc in L.MODEL_CLASSES[target] if mc[0] == "lin"][0]
    extra = list(per_family.index[4:])
    for dflt in DEFAULTS:
        k = L.key({"model": lin_model, "feats": lin_feats} | dflt)
        if k not in picks:
            picks.append(k); slots[k] = "default"
        elif extra:
            k2 = extra.pop(0); picks.append(k2); slots[k2] = "rule"
    S["selected"] = S.index.isin(picks)
    S["slot"] = [slots.get(k, "") for k in S.index]
    return bb, base.mean(1), S, picks


def main():
    log = pd.read_csv(L.RES_DIR / "v2_validation_log.csv")
    shortlist, tables = {}, []
    for tgt in ["spy_vol", "spy_tlt_corr", "tlt_vol"]:
        bb, bmeans, S, picks = select(log, tgt)
        entries = []
        for i, k in enumerate(picks):
            sp = json.loads(k)
            entries.append({"id": f"{tgt}.{i + 1}", "spec": sp, "label": L.label(sp), "slot": S.loc[k, "slot"],
                            "counterpart": L.counterpart(sp),
                            "val_rel_vs_fixed_mean": round(float(S.loc[k, "rel_mean"]), 4),
                            "val_rel_vs_fixed_VAL": round(float(S.loc[k, "rel_vs_fixed_VAL"]), 4),
                            "val_skill_vs_bb_mean": round(float(S.loc[k, "skill_mean"]), 4),
                            "eligible": bool(S.loc[k, "eligible"])})
        shortlist[tgt] = {"h": 21 if tgt == "spy_tlt_corr" else 5, "best_baseline": bb,
                          "baselines": L.baselines(tgt), "best_adaptive_id": f"{tgt}.1", "entries": entries}
        S2 = S.copy(); S2["target"] = tgt; S2["best_baseline"] = bb
        tables.append(S2.reset_index())
        print(f"== {tgt}: best baseline {bb}  (val mean MSE {bmeans.round(4).to_dict()})")
        for e in entries:
            print(f"   {e['id']} [{e['slot']}] {e['label']}  rel_vs_fixed={e['val_rel_vs_fixed_mean']} "
                  f"(VAL {e['val_rel_vs_fixed_VAL']})  skill_vs_BB={e['val_skill_vs_bb_mean']} eligible={e['eligible']}")
    pd.concat(tables).to_csv(L.RES_DIR / "v2_validation_selection_table.csv", index=False)
    p = L.RES_DIR / "v2_shortlist.json"
    p.write_text(json.dumps(shortlist, indent=1, sort_keys=True))
    print("sha256", hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()

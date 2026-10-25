"""Does non-Yahoo data (FRED macro / credit / stress / news-uncertainty) improve SPY direction prediction?

Usage:  python fred_study.py validate   -> results/fred_validation_log.csv
        python fred_study.py select     -> results/fred_shortlist.json + fred_preregistration.md (hash-locked)
        python fred_study.py test       -> one-shot 2019-2025 test

Family A (primary, no selection): base122+FRED vs base122, for every model x scheme.
Family B (selected): top-3 FRED-containing feature sets by validation mean AUC (VAL AUC > 0.5 required),
                     each vs its twin without FRED. Holm within family and horizon over {BA, AUC}.
"""
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))
os.environ["OMP_NUM_THREADS"] = "1"
import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import fe_lib
from common import CORE_FEATURES, DATA_DIR, EXP_DIR, RES_DIR
from fe_lib import ColumnSet, FEDataset, MODELS, bootstrap_vs_chance, metrics, paired_bootstrap, run, sel_cluster, sel_stable
from fe_test import holm
from fred_features import load_fred
from fred_sets import SPY4, TWIN, fred_dataset, job

ALL_SETS = sorted(set(TWIN) | set(TWIN.values()))
FOLDS = ["F1", "F2", "VAL"]
SHORT = RES_DIR / "fred_shortlist.json"
PREREG = EXP_DIR / "fred_preregistration.md"


def validate():
    rows = []
    for h in [5, 1]:
        ds = fred_dataset(h)
        tasks = [(s, m, sc, f) for s in ALL_SETS for m in MODELS for sc in ["fixed", "refit"] for f in FOLDS]
        out = Parallel(n_jobs=-1)(delayed(job)(ds, s, m, sc, *ds.period(f)) for s, m, sc, f in tasks)
        for (s, m, sc, f), (p, ch) in zip(tasks, out):
            rows.append({"horizon": h, "feature_set": s, "model": m, "scheme": sc, "fold": f,
                         "n_features": len(ch[0]), **metrics(p)})
        print(f"h={h} done", flush=True)
    pd.DataFrame(rows).to_csv(RES_DIR / "fred_validation_log.csv", index=False)


def select():
    log = pd.read_csv(RES_DIR / "fred_validation_log.csv")
    t = (log.pivot_table(index=["horizon", "feature_set", "model", "scheme"], columns="fold", values="roc_auc")
         .assign(mean_auc=lambda d: d[FOLDS].mean(axis=1)).reset_index())
    t.sort_values(["horizon", "mean_auc"], ascending=[True, False]).to_csv(RES_DIR / "fred_validation_table.csv", index=False)
    famA = [{"family": "A", "horizon": h, "feature_set": "base_fred", "twin": "base122", "model": m, "scheme": sc}
            for h in [5, 1] for m in MODELS for sc in ["fixed", "refit"]]
    famB = []
    for h in [5, 1]:
        g = t[(t.horizon == h) & t.feature_set.isin(TWIN) & (t.VAL > 0.5)].sort_values("mean_auc", ascending=False)
        for _, r in g.head(3).iterrows():
            famB.append({"family": "B", "horizon": h, "feature_set": r.feature_set, "twin": TWIN[r.feature_set],
                         "model": r.model, "scheme": r.scheme, "val_mean_auc": round(float(r.mean_auc), 4)})
    items = [{"id": f"{x['family']}.h{x['horizon']}.{i}", **x} for i, x in enumerate(famA + famB)]
    SHORT.write_text(json.dumps(items, indent=2))
    digest = hashlib.sha256(SHORT.read_bytes()).hexdigest()
    lines = ["# FRED-features pre-registration", "",
             f"Shortlist `results/fred_shortlist.json`  SHA-256: `{digest}`", "",
             "Family A (no selection): base122 + 21 FRED features vs base122, every model x scheme.",
             "Family B: top-3 FRED-containing feature sets by validation mean AUC (VAL AUC > 0.5), vs twin without FRED.",
             "Holm within (family, horizon) over {BA, AUC}. Also reported: each config vs chance.", "",
             "| id | h | feature set | twin | model | scheme |", "|---|---|---|---|---|---|"]
    lines += [f"| {x['id']} | {x['horizon']} | {x['feature_set']} | {x['twin']} | {x['model']} | {x['scheme']} |" for x in items]
    lines += ["", "Caveat: 2019-2025 has been evaluated in earlier studies of this project (not a pristine hold-out)."]
    PREREG.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def test():
    want = re.search(r"SHA-256: `([0-9a-f]{64})`", PREREG.read_text(encoding="utf-8")).group(1)
    assert hashlib.sha256(SHORT.read_bytes()).hexdigest() == want, "shortlist changed after pre-registration"
    items = json.loads(SHORT.read_text())
    levels, comps, preds = [], [], {}
    for h in [5, 1]:
        ds = fred_dataset(h)
        s0, e0 = ds.period("TEST")
        its = [x for x in items if x["horizon"] == h]
        need = sorted({(x[k], x["model"], x["scheme"]) for x in its for k in ["feature_set", "twin"]})
        out = Parallel(n_jobs=-1)(delayed(job)(ds, *k, s0, e0) for k in need)
        for k, (p, ch) in zip(need, out):
            preds[(h, *k)] = p
            levels.append({"horizon": h, "feature_set": k[0], "model": k[1], "scheme": k[2],
                           "n_features": len(ch[0]), **metrics(p)})
        for x in its:
            a = preds[(h, x["feature_set"], x["model"], x["scheme"])]
            b = preds[(h, x["twin"], x["model"], x["scheme"])]
            comps.append({"id": x["id"], "family": x["family"], "horizon": h, "comparison": "vs_twin", **paired_bootstrap(a, b, h)})
            comps.append({"id": x["id"], "family": x["family"], "horizon": h, "comparison": "vs_chance", **bootstrap_vs_chance(a, h)})
    comp = pd.DataFrame(comps)
    for _, g in comp.groupby(["family", "horizon", "comparison"]):
        adj = holm(np.concatenate([g["p_balanced_acc"], g["p_roc_auc"]]))
        comp.loc[g.index, "holm_balanced_acc"], comp.loc[g.index, "holm_roc_auc"] = adj[: len(g)], adj[len(g):]
    comp = comp.merge(pd.DataFrame(items)[["id", "feature_set", "twin", "model", "scheme"]], on="id")
    comp.to_csv(RES_DIR / "fred_test_comparisons.csv", index=False)
    pd.DataFrame(levels).to_csv(RES_DIR / "fred_test_levels.csv", index=False)
    pd.to_pickle(preds, RES_DIR / "fred_test_predictions.pkl")
    cols = ["id", "feature_set", "model", "scheme", "comparison", "d_balanced_acc", "d_balanced_acc_lo",
            "d_balanced_acc_hi", "d_roc_auc", "d_roc_auc_lo", "d_roc_auc_hi", "holm_balanced_acc", "holm_roc_auc"]
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
        print(pd.DataFrame(levels).to_string())
        print(comp[cols].to_string())


if __name__ == "__main__":
    {"validate": validate, "select": select, "test": test}[sys.argv[1]]()

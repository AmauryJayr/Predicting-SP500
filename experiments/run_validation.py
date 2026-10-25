"""Run every configuration (method + counterparts) on the selection folds F1, F2, VAL.

Only data through 2018 is touched (the largest fold, VAL, ends 2018-12-31 and its
models see only labels known by each decision date).
Outputs:
  results/validation_log.csv     one row per (horizon, fold, spec)  - every configuration evaluated
  results/validation_deltas.csv  one row per (horizon, entry, fold) - method minus counterparts
  results/validation_summary.csv one row per (horizon, entry)       - fold-averaged deltas, selection score
Usage: python run_validation.py [grid_module]   (default: configs)
"""
import importlib
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "16")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import RES_DIR, SELECTION_FOLDS, PERIODS, Dataset, metrics
from methods import run_spec, spec_key

_DS = {}


def _job(h, fold, key):
    import json
    if h not in _DS:
        _DS[h] = Dataset(h)
    ds = _DS[h]
    assert pd.Timestamp(PERIODS[fold][1]) <= pd.Timestamp("2018-12-31")
    spec = json.loads(key)
    p = run_spec(ds, spec, fold)
    return h, fold, key, metrics(p), p.attrs.get("n_refits", np.nan), p[["proba", "thr"]].to_numpy()


def main(grid_module="configs", tag=""):
    E = importlib.import_module(grid_module).grid()
    keys = sorted({spec_key(e[k]) for e in E for k in ("method", "primary", "secondary") if e[k] is not None})
    jobs = [(h, f, k) for h in (5, 1) for f in SELECTION_FOLDS for k in keys]
    print(f"{len(E)} entries, {len(keys)} unique specs, {len(jobs)} runs")
    t = time.time()
    out = Parallel(n_jobs=15, batch_size=4, verbose=5)(delayed(_job)(*j) for j in jobs)
    print(f"done in {time.time() - t:.0f}s")
    M = {(h, f, k): (m, nr) for h, f, k, m, nr, _ in out}
    pd.to_pickle({(h, f, k): a for h, f, k, _, _, a in out}, RES_DIR / f"val_preds{tag}.pkl")

    rows, drows = [], []
    for h in (5, 1):
        for f in SELECTION_FOLDS:
            for e in E:
                for role in ("method", "primary", "secondary"):
                    if e[role] is None:
                        continue
                    m, nr = M[(h, f, spec_key(e[role]))]
                    rows.append({"horizon": h, "fold": f, "family": e["family"], "label": e["label"], "role": role,
                                 "n_refits": nr, **m, "spec": spec_key(e[role])})
                mm = M[(h, f, spec_key(e["method"]))][0]
                d = {"horizon": h, "fold": f, "family": e["family"], "label": e["label"]}
                for role in ("primary", "secondary"):
                    if e[role] is None:
                        continue
                    cm = M[(h, f, spec_key(e[role]))][0]
                    d[f"dBA_{role}"] = mm["balanced_acc"] - cm["balanced_acc"]
                    d[f"dAUC_{role}"] = mm["roc_auc"] - cm["roc_auc"]
                    d[f"dBrier_{role}"] = cm["brier"] - mm["brier"]  # positive = method better
                d.update({"BA": mm["balanced_acc"], "AUC": mm["roc_auc"], "Brier": mm["brier"]})
                drows.append(d)
    log = pd.DataFrame(rows)
    log.to_csv(RES_DIR / f"validation_log{tag}.csv", index=False)
    D = pd.DataFrame(drows)
    D.to_csv(RES_DIR / f"validation_deltas{tag}.csv", index=False)
    agg = D.groupby(["horizon", "family", "label"], sort=False).agg(
        dBA=("dBA_primary", "mean"), dAUC=("dAUC_primary", "mean"), dBrier=("dBrier_primary", "mean"),
        dBA_min=("dBA_primary", "min"), dAUC_min=("dAUC_primary", "min"),
        dBA_sec=("dBA_secondary", "mean"), dAUC_sec=("dAUC_secondary", "mean"),
        dBrier_sec=("dBrier_secondary", "mean"),
        BA=("BA", "mean"), AUC=("AUC", "mean"), Brier=("Brier", "mean")).reset_index()
    agg["score"] = agg["dBA"] + agg["dAUC"]
    agg.to_csv(RES_DIR / f"validation_summary{tag}.csv", index=False)
    return agg


if __name__ == "__main__":
    mod = sys.argv[1] if len(sys.argv) > 1 else "configs"
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    agg = main(mod, tag)
    pd.set_option("display.width", 250)
    for h in (5, 1):
        a = agg[agg.horizon == h]
        best = a.sort_values("score", ascending=False).groupby("family").head(1)
        print(f"\n=== h={h}: best config per family (fold-mean deltas vs primary counterpart) ===")
        print(best.round(4).to_string(index=False))

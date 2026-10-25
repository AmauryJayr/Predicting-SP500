"""Validation sweep for the feature-engineering / feature-selection study (data through 2018 only).

Every (horizon, selector, model, scheme) is run walk-forward on folds F1 (2013-14), F2 (2015-16) and
VAL (2017-18); training at each fit uses all labels known at that date. Every run is logged.
"""
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))
os.environ["OMP_NUM_THREADS"] = "1"
import time
import pandas as pd
from joblib import Parallel, delayed

from common import RES_DIR
from fe_lib import FEDataset, MODELS, SELECTORS, metrics, run

FOLDS = ["F1", "F2", "VAL"]


def job(ds, sel, model, scheme, fold):
    s, e = ds.period(fold)
    pred, chosen = run(ds, sel, model, scheme, s, e)
    return {"horizon": ds.h, "selector": sel, "model": model, "scheme": scheme, "fold": fold,
            "n_features": round(sum(len(c) for c in chosen) / len(chosen), 1), **metrics(pred)}


if __name__ == "__main__":
    t = time.time()
    rows = []
    for h in [5, 1]:
        ds = FEDataset(h)
        tasks = [(sel, m, sc, f) for sel in SELECTORS for m in MODELS for sc in ["fixed", "refit"] for f in FOLDS]
        rows += Parallel(n_jobs=-1, verbose=0)(delayed(job)(ds, *a) for a in tasks)
        print(f"h={h}: {len(tasks)} runs done at {time.time() - t:.0f}s", flush=True)
    log = pd.DataFrame(rows)
    log.to_csv(RES_DIR / "fe_validation_log.csv", index=False)
    print(f"logged {len(log)} runs -> {RES_DIR / 'fe_validation_log.csv'} in {time.time() - t:.0f}s")

"""v2 validation: every grid configuration + counterparts + baselines on folds F1 (2013-14), F2 (2015-16),
VAL (2017-18). Uses data through 2018 only (each fold's models use labels known at that fold's dates).

Run: python v2_validation.py  -> results/v2_validation_log.csv, results/v2_val_preds.pkl
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "16")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import v2_lib as L

TARGETS = ["spy_vol", "spy_tlt_corr", "tlt_vol"]


def job(target, model_class, fold):
    T = L.build_targets()
    tg = T[target]
    s, e = tg.period(fold)
    assert tg.dates[e - 1] <= np.datetime64("2018-12-31")
    rows, preds = [], {}
    if model_class == "baselines":
        specs = [(name, sp) for name, sp in L.baselines(target).items()]
    else:
        specs = [(None, sp) for sp in L.grid(target) if (sp["model"], sp["feats"]) == model_class]
    cache = {}
    for name, sp in specs:
        t0 = time.time()
        p = L.run(tg, sp, fold, cache=cache)
        fires = getattr(L.m_detect, "last_fires", np.nan) if sp["method"] == "detect" else np.nan
        m = L.metrics(tg, fold, p)
        rows.append(dict(target=target, fold=fold, role="baseline" if name else "config",
                         baseline=name, label=L.label(sp), key=L.key(sp), family=L.family(sp) if not name else "baseline",
                         model=sp.get("model"), feats=sp.get("feats"), n_fires=fires, secs=time.time() - t0, **m))
        preds[(target, fold, L.key(sp))] = p
    return rows, preds


def main():
    t0 = time.time()
    jobs = []
    for tgt in TARGETS:
        for f in L.FOLDS:
            jobs.append((tgt, "baselines", f))
            for mc in L.MODEL_CLASSES[tgt]:
                jobs.append((tgt, mc, f))
    # heavy (hgb) jobs first
    jobs.sort(key=lambda j: 0 if j[1] != "baselines" and j[1][0] == "hgb" else 1)
    res = Parallel(n_jobs=15, verbose=5)(delayed(job)(*j) for j in jobs)
    rows = [r for rr, _ in res for r in rr]
    preds = {k: v for _, pp in res for k, v in pp.items()}
    log = pd.DataFrame(rows)
    # attach deltas vs the matched trained-once counterpart (same model + features)
    fx = log[(log.role == "config") & (log.label.str.endswith("/fixed"))][["target", "fold", "model", "feats", "mse", "qlike", "ba", "auc"]]
    fx = fx.rename(columns={c: f"fixed_{c}" for c in ["mse", "qlike", "ba", "auc"]})
    log = log.merge(fx, on=["target", "fold", "model", "feats"], how="left")
    log["rel_vs_fixed"] = 1 - log["mse"] / log["fixed_mse"]
    log.to_csv(L.RES_DIR / "v2_validation_log.csv", index=False)
    pd.to_pickle(preds, L.RES_DIR / "v2_val_preds.pkl")
    print(f"{len(log)} rows, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

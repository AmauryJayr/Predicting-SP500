"""Run the preregistered shortlist ONCE on the 2019-2025 test period."""
import hashlib
import json
import os
import re

os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "16")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import EXP_DIR, RES_DIR, Dataset, metrics, paired_bootstrap
from methods import run_spec

raw = open(RES_DIR / "shortlist.json", "rb").read()
sha = hashlib.sha256(raw).hexdigest()
prereg = open(EXP_DIR / "preregistration.md", encoding="utf-8").read()
assert sha in prereg, "shortlist.json does not match the preregistered hash"
SHORT = json.loads(raw)

_DS = {}


def _run(h, spec):
    if h not in _DS:
        _DS[h] = Dataset(h)
    return run_spec(_DS[h], spec, "TEST")


def chance(ds):
    s, e = ds.period_positions("TEST")
    r = ds.y[:ds.n_known(s)].mean()
    return pd.DataFrame({"y": ds.y[s:e], "proba": r, "thr": r, "pred": 1}, index=ds.index[s:e])


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(1, run)
    return adj


def main():
    jobs = [(h, i, role, e[role]) for h in ("5", "1") for i, e in enumerate(SHORT[h], 1)
            for role in ("method", "primary", "secondary") if e.get(role)]
    out = Parallel(n_jobs=15, verbose=0)(delayed(_run)(int(h), spec) for h, i, role, spec in jobs)
    P = {(h, i, role): p for (h, i, role, _), p in zip(jobs, out)}
    for h in ("5", "1"):
        P[(h, 0, "chance")] = chance(Dataset(int(h)))
    pd.to_pickle(P, RES_DIR / "test_predictions.pkl")

    lvl, boot, years = [], [], []
    for h in ("5", "1"):
        for i, e in enumerate(SHORT[h], 1):
            tag = dict(horizon=int(h), id=f"{h}.{i}", family=e["family"], label=e["label"])
            m = P[(h, i, "method")]
            for role in ("method", "primary", "secondary"):
                if (h, i, role) in P:
                    lvl.append({**tag, "role": role, **metrics(P[(h, i, role)]),
                                "n_refits": P[(h, i, role)].attrs.get("n_refits", np.nan)})
            for role in ("primary", "secondary", "chance"):
                key = (h, 0, "chance") if role == "chance" else (h, i, role)
                if key in P:
                    boot.append({**tag, "vs": role, **paired_bootstrap(m, P[key], int(h))})
            for role in ("method", "primary"):
                p = P[(h, i, role)]
                for y, g in p.groupby(p.index.year):
                    mm = metrics(g)
                    years.append({**tag, "role": role, "year": y, "balanced_acc": mm["balanced_acc"],
                                  "roc_auc": mm["roc_auc"], "brier": mm["brier"], "n": mm["n"]})
    L = pd.DataFrame(lvl); B = pd.DataFrame(boot); Y = pd.DataFrame(years)
    for h in (5, 1):
        sel = (B.horizon == h) & (B.vs == "primary")
        for mname in ("balanced_acc", "roc_auc", "brier"):
            B.loc[sel, f"p_{mname}_holm"] = holm(B.loc[sel, f"p_{mname}"])
    L.to_csv(RES_DIR / "test_levels.csv", index=False)
    B.to_csv(RES_DIR / "test_bootstrap.csv", index=False)
    Y.to_csv(RES_DIR / "test_by_year.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(L.drop(columns=["label"]).round(4).to_string(index=False))
    cols = ["id", "family", "vs", "d_balanced_acc", "d_balanced_acc_lo", "d_balanced_acc_hi", "p_balanced_acc",
            "d_roc_auc", "d_roc_auc_lo", "d_roc_auc_hi", "p_roc_auc", "d_brier", "d_brier_lo", "d_brier_hi", "p_brier",
            "p_balanced_acc_holm", "p_roc_auc_holm"]
    print(B[cols].round(4).to_string(index=False))
    print(Y.pivot_table(index=["id", "role"], columns="year", values="balanced_acc").round(3).to_string())
    print(Y.pivot_table(index=["id", "role"], columns="year", values="roc_auc").round(3).to_string())


if __name__ == "__main__":
    main()

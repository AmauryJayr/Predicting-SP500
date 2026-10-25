"""One-shot test (2019-2025) of the pre-registered feature-study shortlist."""
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))
os.environ["OMP_NUM_THREADS"] = "1"
import hashlib
import json
import re

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import EXP_DIR, RES_DIR
from fe_lib import FEDataset, bootstrap_vs_chance, metrics, paired_bootstrap, run


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


if __name__ == "__main__":
    path = RES_DIR / "fe_shortlist.json"
    reg = (EXP_DIR / "fe_preregistration.md").read_text(encoding="utf-8")
    want = re.search(r"SHA-256: `([0-9a-f]{64})`", reg).group(1)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == want, "shortlist changed after pre-registration"
    shortlist = json.loads(path.read_text())

    other = {"fixed": "refit", "refit": "fixed"}
    preds, chosen, levels, comps = {}, {}, [], []
    for h in sorted({s["horizon"] for s in shortlist}):
        ds = FEDataset(h)
        s0, e0 = ds.period("TEST")
        items = [s for s in shortlist if s["horizon"] == h]
        needed = set()
        for s in items:
            needed |= {(s["selector"], s["model"], s["scheme"]), ("base122", s["model"], s["scheme"]),
                       (s["selector"], s["model"], other[s["scheme"]])}
        needed = sorted(needed)
        out = Parallel(n_jobs=-1)(delayed(run)(ds, *k, s0, e0) for k in needed)
        for k, (p, c) in zip(needed, out):
            preds[(h, *k)] = p
            chosen[(h, *k)] = c
            levels.append({"horizon": h, "selector": k[0], "model": k[1], "scheme": k[2], **metrics(p)})
        for s in items:
            a = preds[(h, s["selector"], s["model"], s["scheme"])]
            comps.append({"id": s["id"], "horizon": h, "comparison": "vs_base122", "confirmatory": True,
                          **paired_bootstrap(a, preds[(h, "base122", s["model"], s["scheme"])], h)})
            comps.append({"id": s["id"], "horizon": h, "comparison": "vs_chance", "confirmatory": True,
                          **bootstrap_vs_chance(a, h)})
            comps.append({"id": s["id"], "horizon": h, "comparison": f"{s['scheme']}_vs_{other[s['scheme']]}",
                          "confirmatory": False,
                          **paired_bootstrap(a, preds[(h, s["selector"], s["model"], other[s["scheme"]])], h)})

    comp = pd.DataFrame(comps)
    for (h, c), g in comp[comp.confirmatory].groupby(["horizon", "comparison"]):
        ps = np.concatenate([g["p_balanced_acc"], g["p_roc_auc"]])
        adj = holm(ps)
        comp.loc[g.index, "holm_balanced_acc"] = adj[: len(g)]
        comp.loc[g.index, "holm_roc_auc"] = adj[len(g):]
    comp.to_csv(RES_DIR / "fe_test_comparisons.csv", index=False)
    pd.DataFrame(levels).to_csv(RES_DIR / "fe_test_levels.csv", index=False)
    pd.to_pickle({"preds": preds, "chosen": chosen}, RES_DIR / "fe_test_predictions.pkl")
    cols = ["id", "comparison", "d_balanced_acc", "d_balanced_acc_lo", "d_balanced_acc_hi", "d_roc_auc",
            "d_roc_auc_lo", "d_roc_auc_hi", "holm_balanced_acc", "holm_roc_auc"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(pd.DataFrame(levels).round(4).to_string())
        print(comp[cols].to_string())

"""v2 one-shot test run on 2019-2025 for the preregistered shortlist (results/v2_shortlist.json, hash checked).

Outputs (results/):
  v2_test_predictions.pkl   predictions of every entry, counterpart and baseline (test rows)
  v2_test_levels.csv        absolute metrics (MSE, R2, QLIKE, BA, AUC, bias) for every model
  v2_test_bootstrap.csv     paired moving-block bootstrap deltas (entry vs counterpart / best baseline / each
                            baseline), 95% CI, one-sided p, Holm-adjusted p within target x metric family
  v2_test_by_year.csv       per-calendar-year metrics of every model
  v2_test_subperiods.csv    bootstrap deltas for 2019-2021 and 2022-2025 (descriptive)
  v2_test_rolling252.csv    plot-ready rolling 252-row MSE of the fixed counterpart, best adaptive (rank-1) and
                            the best baseline (+ all other entries), per target
"""
import hashlib
import json
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

PREREG_HASH = "37380b7d6d210e7d4e6d5ac6460f25525a6aa3cc1e47f0893ad5c37a5636c017"
SUBPERIODS = {"2019-2021": ("2019-01-01", "2021-12-31"), "2022-2025": ("2022-01-01", "2025-12-31")}


def load_shortlist():
    p = L.RES_DIR / "v2_shortlist.json"
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    assert h == PREREG_HASH, f"shortlist hash mismatch {h}"
    return json.loads(p.read_text())


def _pred_job(target, name, spec):
    tg = L.build_targets()[target]
    t0 = time.time()
    p = L.run(tg, spec, "TEST")
    fires = getattr(L.m_detect, "last_fires", None) if spec.get("method") == "detect" else None
    return (target, name), p, fires, time.time() - t0


def main():
    SL = load_shortlist()
    T = L.build_targets()
    jobs = []
    for tgt, info in SL.items():
        for e in info["entries"]:
            jobs.append((tgt, e["id"], e["spec"]))
        cps = {L.key(e["counterpart"]): e["counterpart"] for e in info["entries"]}
        for k, sp in cps.items():
            jobs.append((tgt, "fixed:" + L.label(sp), sp))
        for b, sp in info["baselines"].items():
            jobs.append((tgt, "base:" + b, sp))
    jobs.sort(key=lambda j: 0 if j[2].get("model") == "hgb" else 1)
    res = Parallel(n_jobs=15, verbose=5)(delayed(_pred_job)(*j) for j in jobs)
    P = {k: p for k, p, _, _ in res}
    fires = {k: f for k, _, f, _ in res if f is not None}
    pd.to_pickle({"preds": P, "fires": fires}, L.RES_DIR / "v2_test_predictions.pkl")
    print("detector fires:", fires)

    levels, boots, years, subs, rolls = [], [], [], [], []
    for tgt, info in SL.items():
        tg = T[tgt]
        s, e = tg.period("TEST")
        rows = np.arange(s, e)
        dates = tg.index[rows]
        names = [k[1] for k in P if k[0] == tgt]
        for n in names:
            levels.append(dict(target=tgt, model=n, **L.metrics_rows(tg, rows, P[(tgt, n)])))
            for yr in sorted(set(dates.year)):
                m = dates.year == yr
                years.append(dict(target=tgt, model=n, year=yr, **L.metrics_rows(tg, rows[m], P[(tgt, n)][m])))
        bb = "base:" + info["best_baseline"]
        for ent in info["entries"]:
            pa = P[(tgt, ent["id"])]
            cp = "fixed:" + L.label(ent["counterpart"])
            comps = [("vs_fixed", cp), ("vs_best_baseline", bb)] + \
                    [("vs_" + b, "base:" + b) for b in info["baselines"] if "base:" + b != bb]
            for cname, other in comps:
                r = L.paired_boot(tg, rows, pa, P[(tgt, other)])
                boots.append(dict(target=tgt, id=ent["id"], label=ent["label"], slot=ent["slot"], comparison=cname,
                                  other=other, confirmatory=cname in ("vs_fixed", "vs_best_baseline"), **r))
                if cname in ("vs_fixed", "vs_best_baseline"):
                    for sp_name, (a, b) in SUBPERIODS.items():
                        m = (dates >= a) & (dates <= b)
                        rr = L.paired_boot(tg, rows[m], pa[m], P[(tgt, other)][m])
                        subs.append(dict(target=tgt, id=ent["id"], label=ent["label"], comparison=cname,
                                         subperiod=sp_name, **rr))
        # B3 check: is the trained-once counterpart itself weak relative to the best baseline?
        for cp in {"fixed:" + L.label(ent["counterpart"]) for ent in info["entries"]}:
            if cp != bb:
                r = L.paired_boot(tg, rows, P[(tgt, cp)], P[(tgt, bb)])
                boots.append(dict(target=tgt, id=cp, label=cp, slot="counterpart", comparison="vs_best_baseline",
                                  other=bb, confirmatory=False, **r))
        # rolling 252-row MSE (plot-ready)
        best = info["best_adaptive_id"]
        cp_best = "fixed:" + L.label([x for x in info["entries"] if x["id"] == best][0]["counterpart"])
        R = pd.DataFrame(index=dates)
        R["y"] = tg.y[rows]
        for n in names:
            R[n] = (tg.y[rows] - P[(tgt, n)]) ** 2
        R = R.drop(columns="y").rolling(252).mean()
        R = R.rename(columns={best: f"BEST_ADAPTIVE[{best}]", cp_best: f"FIXED[{cp_best}]",
                              bb: f"BEST_BASELINE[{bb}]"})
        R = R.stack().rename("rolling252_mse").reset_index()
        R.columns = ["date", "model", "rolling252_mse"]
        R["target"] = tgt
        rolls.append(R.dropna())

    B = pd.DataFrame(boots)
    mets = ["mse", "r2", "qlike", "ba", "auc"]
    for tgt in B.target.unique():
        conf = (B.target == tgt) & B.confirmatory
        for m in mets:
            col = f"p_{m}"
            if col in B and B.loc[conf, col].notna().any():
                idx = B.index[conf & B[col].notna()]
                B.loc[idx, f"holm_{m}"] = L.holm(B.loc[idx, col].to_numpy())
    pd.DataFrame(levels).to_csv(L.RES_DIR / "v2_test_levels.csv", index=False)
    B.to_csv(L.RES_DIR / "v2_test_bootstrap.csv", index=False)
    pd.DataFrame(years).to_csv(L.RES_DIR / "v2_test_by_year.csv", index=False)
    pd.DataFrame(subs).to_csv(L.RES_DIR / "v2_test_subperiods.csv", index=False)
    pd.concat(rolls).to_csv(L.RES_DIR / "v2_test_rolling252.csv", index=False)

    pd.set_option("display.width", 250)
    cols = ["id", "label", "comparison", "d_mse", "d_mse_lo", "d_mse_hi", "p_mse", "holm_mse", "d_ba", "d_auc"]
    print(B[B.confirmatory][cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

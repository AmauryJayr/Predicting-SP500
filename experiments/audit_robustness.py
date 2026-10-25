"""AUDIT (pessimist): EXPLORATORY sensitivity checks of the preregistered negative conclusion.
None of these is a new search; the 12 preregistered method specs are used unchanged.

R1 block length 20 / 40 / 60 / 120 for the 12 primary comparisons (from the saved test predictions)
R2 test sub-periods 2019-2021 vs 2022-2025
R3 stale fixed model ("insurance" scenario): method and its trained-once counterpart both start at
   2017-01-01 (fixed model trained on labels known then, i.e. through 2016) or 2013-01-01, and are
   scored on 2019-2025 only. The adaptive method keeps adapting from its start date.
R4 alternative shortlist rules (all 3 folds positive; VAL-only) -> which entries, and their test deltas
Outputs: results/audit_robust_*.csv
"""
import json
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
import common
from common import RES_DIR, Dataset, metrics, paired_bootstrap
from methods import run_spec, spec_key
from audit_lib import boot, holm

SHORT = json.load(open(RES_DIR / "shortlist.json"))
P = pd.read_pickle(RES_DIR / "test_predictions.pkl")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)


def r0_check():
    a, b = P[("5", 1, "method")], P[("5", 1, "primary")]
    x = paired_bootstrap(a, b, 5); y = boot(a, b, L=20)
    print("bootstrap re-implementation check (should match):",
          round(x["d_roc_auc_lo"], 5), round(y["dAUC_lo"], 5), round(x["p_balanced_acc"], 4), round(y["pBA"], 4))


def r1_r2():
    rows = []
    for h in ("5", "1"):
        for i, e in enumerate(SHORT[h], 1):
            m, f = P[(h, i, "method")], P[(h, i, "primary")]
            for L in (20, 40, 60, 120):
                rows.append(dict(h=int(h), id=f"{h}.{i}", family=e["family"], period="2019-2025", L=L, **boot(m, f, L=L)))
            for per, (a, b) in {"2019-2021": ("2019", "2021"), "2022-2025": ("2022", "2025")}.items():
                mm, ff = m.loc[a:b], f.loc[a:b]
                rows.append(dict(h=int(h), id=f"{h}.{i}", family=e["family"], period=per, L=20, **boot(mm, ff, L=20)))
    R = pd.DataFrame(rows)
    for (h, per, L), g in R.groupby(["h", "period", "L"]):
        for m in ("BA", "AUC"):
            R.loc[g.index, f"p{m}_holm"] = holm(g[f"p{m}"])
    R.to_csv(RES_DIR / "audit_robust_blocklen_subperiod.csv", index=False)
    print(R[["h", "id", "family", "period", "L", "dBA", "dBA_lo", "dBA_hi", "pBA", "dAUC", "dAUC_lo", "dAUC_hi",
             "pAUC", "pBA_holm", "pAUC_holm"]].round(3).to_string(index=False))
    return R


EXTRA_PERIODS = {"S17": ("2017-01-01", "2025-12-31"), "S13": ("2013-01-01", "2025-12-31")}


def _run(h, spec, period):
    common.PERIODS.update(EXTRA_PERIODS)  # worker processes do not inherit runtime edits
    return run_spec(Dataset(h), spec, period)


def r3_stale():
    common.PERIODS["S17"] = ("2017-01-01", "2025-12-31")
    common.PERIODS["S13"] = ("2013-01-01", "2025-12-31")
    jobs = []
    for h in ("5", "1"):
        for i, e in enumerate(SHORT[h], 1):
            for per in ("S17", "S13"):
                for role in ("method", "primary"):
                    jobs.append((h, i, per, role, e[role]))
    out = Parallel(n_jobs=15, verbose=0)(delayed(_run)(int(h), s, per) for h, i, per, role, s in jobs)
    Q = {(h, i, per, role): p.loc["2019-01-01":] for (h, i, per, role, _), p in zip(jobs, out)}
    pd.to_pickle(Q, RES_DIR / "audit_stale_predictions.pkl")
    rows = []
    for h in ("5", "1"):
        for i, e in enumerate(SHORT[h], 1):
            fresh = P[(h, i, "primary")]
            for per in ("S17", "S13"):
                m, f = Q[(h, i, per, "method")], Q[(h, i, per, "primary")]
                assert m.index.equals(fresh.index)
                mm, ff = metrics(m), metrics(f)
                rows.append(dict(h=int(h), id=f"{h}.{i}", family=e["family"], start=per,
                                 BA_method=mm["balanced_acc"], BA_stale_fixed=ff["balanced_acc"],
                                 AUC_method=mm["roc_auc"], AUC_stale_fixed=ff["roc_auc"],
                                 BA_fresh_fixed=metrics(fresh)["balanced_acc"],
                                 AUC_fresh_fixed=metrics(fresh)["roc_auc"], **boot(m, f, L=20)))
    S = pd.DataFrame(rows)
    S.to_csv(RES_DIR / "audit_robust_stale.csv", index=False)
    print(S[["h", "id", "family", "start", "BA_method", "BA_stale_fixed", "BA_fresh_fixed", "AUC_method",
             "AUC_stale_fixed", "AUC_fresh_fixed", "dBA", "dBA_lo", "dBA_hi", "dAUC", "dAUC_lo", "dAUC_hi"]]
          .round(3).to_string(index=False))
    return S


def r4_alt_rules():
    import configs, configs2
    D = pd.read_csv(RES_DIR / "validation_deltas.csv")
    S = pd.read_csv(RES_DIR / "validation_summary.csv")
    entries = {(e["family"], e["label"]): e for e in configs.grid() + configs2.grid()}
    piv = D.pivot_table(index=["horizon", "family", "label"], columns="fold",
                        values=["dBA_primary", "dAUC_primary"]).reset_index()
    piv.columns = ["_".join([c for c in col if c]) for col in piv.columns]
    rules = {
        "all3": lambda r: all(r[f"dBA_primary_{f}"] > 0 and r[f"dAUC_primary_{f}"] > 0 for f in ("F1", "F2", "VAL")),
        "VALonly": lambda r: r["dBA_primary_VAL"] > 0 and r["dAUC_primary_VAL"] > 0,
    }
    picks = []
    for rn, fn in rules.items():
        for h in (5, 1):
            a = piv[piv.horizon == h].copy()
            a = a[a.apply(fn, axis=1)]
            score_col = (a["dBA_primary_VAL"] + a["dAUC_primary_VAL"]) if rn == "VALonly" else \
                a[[c for c in a.columns if c.startswith("dBA_primary") or c.startswith("dAUC_primary")]].sum(1) / 3
            a = a.assign(score=score_col).sort_values("score", ascending=False)
            best = a.groupby("family", sort=False).head(1).head(6)
            for r in best.itertuples():
                picks.append((rn, h, r.family, r.label, float(r.score)))
    print("alternative-rule picks:", len(picks))
    jobs = []
    for rn, h, fam, lab, sc in picks:
        e = entries[(fam, lab)]
        for role in ("method", "primary"):
            jobs.append((rn, h, fam, lab, sc, role, e[role]))
    out = Parallel(n_jobs=15, verbose=0)(delayed(_run)(h, s, "TEST") for rn, h, fam, lab, sc, role, s in jobs)
    Q = {(rn, h, fam, lab, role): p for (rn, h, fam, lab, sc, role, _), p in zip(jobs, out)}
    rows = []
    for rn, h, fam, lab, sc in picks:
        m, f = Q[(rn, h, fam, lab, "method")], Q[(rn, h, fam, lab, "primary")]
        rows.append(dict(rule=rn, h=h, family=fam, label=lab, val_score=sc, BA_method=metrics(m)["balanced_acc"],
                         BA_fixed=metrics(f)["balanced_acc"], **boot(m, f, L=20)))
    A = pd.DataFrame(rows)
    for (rn, h), g in A.groupby(["rule", "h"]):
        for mname in ("BA", "AUC"):
            A.loc[g.index, f"p{mname}_holm"] = holm(g[f"p{mname}"])
    A.to_csv(RES_DIR / "audit_robust_altrules.csv", index=False)
    print(A[["rule", "h", "family", "label", "val_score", "BA_method", "BA_fixed", "dBA", "dBA_lo", "dBA_hi",
             "dAUC", "dAUC_lo", "dAUC_hi", "pBA_holm", "pAUC_holm"]].round(3).to_string(index=False))
    return A


if __name__ == "__main__":
    which = sys.argv[1:] or ["r0", "r12", "r3", "r4"]
    if "r0" in which: r0_check()
    if "r12" in which: r1_r2()
    if "r3" in which: r3_stale()
    if "r4" in which: r4_alt_rules()

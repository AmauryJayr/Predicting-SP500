"""ARMAX baseline and its improvement ladder, with the same protocol as the other studies.

  python armax_study.py validate   choices for L5 (exog set, by mean fold AUC) and L6 (re-estimation, by mean fold Brier)
  python armax_study.py select     lock the ladder -> results/armax_ladder.json + armax_preregistration.md (SHA-256)
  python armax_study.py test       one run on 2019-2025

Ladder (each step keeps everything above it):
  L0 always up (buy-and-hold)
  L1 textbook ARMAX(1,1): close-to-close returns, exog = VIX level + 20d momentum, Gaussian, fitted once
  L2 + empirical calibration of the forecast z-score (returns are negatively skewed)
  L3 + target alignment: separate intraday and overnight models summed exactly over the label's window
  L4 + GARCH(1,1)-t volatility forecasts (volatility clustering, fat tails)
  L5 + exogenous set chosen on validation from X0 / X1 / X2 (risk-premium set)
  L6 + drift-aware re-estimation chosen on validation: fixed / rolling 504 / rolling 1008 / expanding, refit every 63 days
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

from armax_lib import ArmaxData, always_up, run
from common import EXP_DIR, RES_DIR, ROOT, metrics, paired_bootstrap
from fe_test import holm

FOLDS = ["F1", "F2", "VAL"]
BASE = dict(target="split", garch=True, calibrate=True)
SCHEMES = {"fixed": ("fixed", None), "rolling504": ("refit", 504), "rolling1008": ("refit", 1008), "expanding": ("refit", None)}
LADDER_FILE = RES_DIR / "armax_ladder.json"
PREREG = EXP_DIR / "armax_preregistration.md"


def _run(D, cfg, period, scheme_key):
    s, e = D.ds.period(period)
    kind, window = SCHEMES[scheme_key]
    return run(D, cfg, s, e, scheme="fixed" if kind == "fixed" else "refit", window=window)


def validate():
    rows = []
    for h in [5, 1]:
        D = ArmaxData(h)
        tasks = [("exog", {**BASE, "exog": x}, f, "fixed", x) for x in ["X0", "X1", "X2"] for f in FOLDS]
        out = Parallel(n_jobs=-1)(delayed(_run)(D, cfg, f, sk) for _, cfg, f, sk, _ in tasks)
        for (kind, cfg, f, sk, label), p in zip(tasks, out):
            rows.append({"horizon": h, "choice": kind, "option": label, "fold": f, **metrics(p)})
        ex = (pd.DataFrame([r for r in rows if r["horizon"] == h and r["choice"] == "exog"])
              .groupby("option")["roc_auc"].mean().idxmax())
        tasks = [("scheme", {**BASE, "exog": ex}, f, sk, sk) for sk in SCHEMES for f in FOLDS]
        out = Parallel(n_jobs=-1)(delayed(_run)(D, cfg, f, sk) for _, cfg, f, sk, _ in tasks)
        for (kind, cfg, f, sk, label), p in zip(tasks, out):
            rows.append({"horizon": h, "choice": kind, "option": label, "exog_used": ex, "fold": f, **metrics(p)})
        print(f"h={h} done (exog chosen: {ex})", flush=True)
    pd.DataFrame(rows).to_csv(RES_DIR / "armax_validation_log.csv", index=False)


def select():
    log = pd.read_csv(RES_DIR / "armax_validation_log.csv")
    ladder = {}
    for h in [5, 1]:
        g = log[log.horizon == h]
        ex = g[g.choice == "exog"].groupby("option")["roc_auc"].mean()
        sc = g[g.choice == "scheme"].groupby("option")["brier"].mean()
        ladder[str(h)] = {
            "L1": {"target": "cc", "exog": "X1", "garch": False, "calibrate": False, "scheme": "fixed"},
            "L2": {"target": "cc", "exog": "X1", "garch": False, "calibrate": True, "scheme": "fixed"},
            "L3": {"target": "split", "exog": "X1", "garch": False, "calibrate": True, "scheme": "fixed"},
            "L4": {"target": "split", "exog": "X1", "garch": True, "calibrate": True, "scheme": "fixed"},
            "L5": {**BASE, "exog": ex.idxmax(), "scheme": "fixed"},
            "L6": {**BASE, "exog": ex.idxmax(), "scheme": sc.idxmin()},
            "_validation": {"exog_mean_auc": ex.round(4).to_dict(), "scheme_mean_brier": sc.round(5).to_dict()},
        }
    LADDER_FILE.write_text(json.dumps(ladder, indent=2))
    digest = hashlib.sha256(LADDER_FILE.read_bytes()).hexdigest()
    lines = ["# ARMAX ladder pre-registration", "",
             f"Ladder file `results/armax_ladder.json`  SHA-256: `{digest}`", "",
             "Choices made on folds 2013-14, 2015-16, 2017-18 only: L5 exog set by mean AUC, L6 scheme by mean Brier.", ""]
    for h, L in ladder.items():
        lines.append(f"- h={h}: L5 exog = {L['L5']['exog']}, L6 scheme = {L['L6']['scheme']}  "
                     f"(validation: {L['_validation']})")
    lines += ["", "Test comparisons (paired block bootstrap, Holm within horizon and comparison family over BA/AUC/Brier):",
              "each step vs the previous step; each step vs always-up; L6 vs the notebook's fixed logistic regression.",
              "Caveat: 2019-2025 has been evaluated in earlier studies of this project."]
    PREREG.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def test():
    want = re.search(r"SHA-256: `([0-9a-f]{64})`", PREREG.read_text(encoding="utf-8")).group(1)
    assert hashlib.sha256(LADDER_FILE.read_bytes()).hexdigest() == want, "ladder changed after pre-registration"
    ladder = json.loads(LADDER_FILE.read_text())
    levels, comps, preds = [], [], {}
    for h in [5, 1]:
        D = ArmaxData(h)
        s, e = D.ds.period("TEST")
        L = ladder[str(h)]
        steps = ["L1", "L2", "L3", "L4", "L5", "L6"]
        out = Parallel(n_jobs=-1)(delayed(_run)(D, {k: v for k, v in L[st].items() if k != "scheme"}, "TEST", L[st]["scheme"])
                                  for st in steps)
        P = {"L0": always_up(D, s, e), **dict(zip(steps, out))}
        nb = pd.read_csv(ROOT / "results" / f"h{h}" / "test_predictions.csv", parse_dates=["date"])
        lr = nb[(nb.features == "all") & (nb.model == "LogReg") & (nb.scheme == "fixed")].set_index("date")[["y", "proba", "thr", "pred"]]
        for k, p in P.items():
            preds[(h, k)] = p
            levels.append({"horizon": h, "step": k, **metrics(p)})
        order = ["L0"] + steps
        for prev, cur in zip(order[:-1], order[1:]):
            comps.append({"horizon": h, "family": "step_vs_previous", "comparison": f"{cur} vs {prev}", **paired_bootstrap(P[cur], P[prev], h)})
        for cur in steps:
            comps.append({"horizon": h, "family": "vs_always_up", "comparison": f"{cur} vs L0", **paired_bootstrap(P[cur], P["L0"], h)})
        common_idx = P["L6"].index.intersection(lr.index)
        assert len(common_idx) > 0.98 * len(P["L6"])
        a, b = P["L6"].loc[common_idx], lr.loc[common_idx]
        assert (a["y"].to_numpy() == b["y"].to_numpy()).all()
        comps.append({"horizon": h, "family": "vs_logreg", "comparison": "L6 vs LogReg fixed (notebook)", **paired_bootstrap(a, b, h)})
    comp = pd.DataFrame(comps)
    for _, g in comp.groupby(["horizon", "family"]):
        ps = np.concatenate([g["p_balanced_acc"], g["p_roc_auc"], g["p_brier"]])
        adj = holm(ps)
        n = len(g)
        comp.loc[g.index, "holm_balanced_acc"], comp.loc[g.index, "holm_roc_auc"], comp.loc[g.index, "holm_brier"] = adj[:n], adj[n:2 * n], adj[2 * n:]
    comp.to_csv(RES_DIR / "armax_test_comparisons.csv", index=False)
    pd.DataFrame(levels).to_csv(RES_DIR / "armax_test_levels.csv", index=False)
    pd.to_pickle(preds, RES_DIR / "armax_test_predictions.pkl")
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
        print(pd.DataFrame(levels).to_string())
        print(comp[["horizon", "comparison", "d_balanced_acc", "d_balanced_acc_lo", "d_balanced_acc_hi", "d_roc_auc",
                    "d_roc_auc_lo", "d_roc_auc_hi", "d_brier", "d_brier_lo", "d_brier_hi",
                    "holm_balanced_acc", "holm_roc_auc", "holm_brier"]].to_string())


if __name__ == "__main__":
    {"validate": validate, "select": select, "test": test}[sys.argv[1]]()

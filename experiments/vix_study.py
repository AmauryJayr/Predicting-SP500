"""VIX as a drift / regime indicator, with the same protocol as the other studies.

  python vix_study.py validate   grid on folds 2013-14, 2015-16, 2017-18 -> results/vix_validation_log.csv
  python vix_study.py select     lock the choices -> results/vix_choices.json + vix_preregistration.md (SHA-256)
  python vix_study.py test       one run on 2019-2025

M1 VIX-triggered retraining     grid: regime cut {0.8, 0.9} x window {expanding, 504}; chosen by mean fold AUC
M2 regime-conditional selection grid: split {median, tercile} x lookback {504, all since 2010}; mean fold AUC
M3 volatility-managed exposure  grid: variance signal {VIX, 20d realized} x cap {1.5, 2}, scaled to buy-and-hold volatility on prior data; mean fold Sharpe at 5 bps
"""
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))
import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import EXP_DIR, PERIODS, RES_DIR, Dataset, metrics, paired_bootstrap
from fe_test import holm
from vix_lib import (always_up, expert_predictions, fixed_base, regime_select, scheduled_base, sharpe,
                     sharpe_bootstrap, spy_daily, strat_stats, vix_trigger, vol_managed)

FOLDS = ["F1", "F2", "VAL"]
M1_GRID = [{"cut": c, "window": w} for c in (0.8, 0.9) for w in (None, 504)]
M2_GRID = [{"split": s, "lookback": L} for s in ("median", "tercile") for L in (504, None)]
M3_GRID = [{"signal": s, "cap": c} for s in ("vix", "rv20") for c in (1.5, 2.0)]
COST = 5
CHOICES = RES_DIR / "vix_choices.json"
PREREG = EXP_DIR / "vix_preregistration.md"


def _m1(ds, f, cfg):
    s, e = ds.period_positions(f)
    p = vix_trigger(ds, s, e, cfg["cut"], cfg["window"])
    return {**metrics(p), "n_fires": p.attrs["n_fires"]}


def _m2(ds, E, f, cfg):
    s, e = ds.period_positions(f)
    p = regime_select(ds, E, s, e, cfg["split"], cfg["lookback"])
    return {**metrics(p), "chosen": json.dumps(p.attrs["chosen"])}


def validate():
    rows = []
    for h in [5, 1]:
        ds = Dataset(h)
        E = expert_predictions(ds)
        out = Parallel(n_jobs=-1)(delayed(_m1)(ds, f, c) for c in M1_GRID for f in FOLDS)
        rows += [{"horizon": h, "method": "M1", "config": json.dumps(c), "fold": f, **r}
                 for (c, f), r in zip([(c, f) for c in M1_GRID for f in FOLDS], out)]
        for c in M2_GRID:
            for f in FOLDS:
                rows.append({"horizon": h, "method": "M2", "config": json.dumps(c), "fold": f, **_m2(ds, E, f, c)})
        for f in FOLDS:                                           # references, for reading the log only
            s, e = ds.period_positions(f)
            rows.append({"horizon": h, "method": "ref_fixed", "config": "", "fold": f, **metrics(fixed_base(ds, s, e))})
        print(f"h={h} done", flush=True)
    D = spy_daily()
    for c in M3_GRID:
        for f in FOLDS:
            a, b = PERIODS[f]
            r = vol_managed(D, a, b, c["signal"], c["cap"], COST)
            rows.append({"horizon": 0, "method": "M3", "config": json.dumps(c), "fold": f,
                         "sharpe": sharpe(r["managed"]), "sharpe_bh": sharpe(r["bh"]), "mean_w": r["w"].mean()})
    pd.DataFrame(rows).to_csv(RES_DIR / "vix_validation_log.csv", index=False)


def select():
    log = pd.read_csv(RES_DIR / "vix_validation_log.csv")
    ch = {}
    for h in [5, 1]:
        g = log[log.horizon == h]
        m1 = g[g.method == "M1"].groupby("config")["roc_auc"].mean()
        m2 = g[g.method == "M2"].groupby("config")["roc_auc"].mean()
        ch[str(h)] = {"M1": json.loads(m1.idxmax()), "M2": json.loads(m2.idxmax()),
                      "_validation": {"M1_mean_auc": m1.round(4).to_dict(), "M2_mean_auc": m2.round(4).to_dict()}}
    m3 = log[log.method == "M3"].groupby("config")["sharpe"].mean()
    m3v = m3[[k for k in m3.index if '"vix"' in k]]
    ch["M3"] = {**json.loads(m3.idxmax()), "vix_variant": json.loads(m3v.idxmax()),
                "_validation_mean_sharpe": m3.round(3).to_dict()}
    CHOICES.write_text(json.dumps(ch, indent=2))
    digest = hashlib.sha256(CHOICES.read_bytes()).hexdigest()
    lines = ["# VIX study pre-registration", "",
             f"Choices file `results/vix_choices.json`  SHA-256: `{digest}`", "",
             "Choices made on folds 2013-14, 2015-16, 2017-18 only.", ""]
    for h in ["5", "1"]:
        lines.append(f"- h={h}: M1 = {ch[h]['M1']}, M2 = {ch[h]['M2']}")
    lines.append(f"- M3 = signal {ch['M3']['signal']}, cap {ch['M3']['cap']}, {COST} bps per unit turnover; "
                 f"secondary VIX variant = {ch['M3']['vix_variant']}")
    lines += ["", "Test comparisons (paired block bootstrap; Holm within horizon and method over BA/AUC/Brier):",
              "- M1 vs the same model trained once (primary); M1 vs scheduled 63-day refits with the same window (isolates the trigger).",
              "- M2 vs the trained-once base model (primary); M2 vs regime-blind selection with the same lookback (isolates the regime); M2 vs always-up.",
              f"- M3: Sharpe of the managed strategy minus buy-and-hold at {COST} bps (primary) and 0 bps; same for the best VIX-based variant (secondary).",
              "Caveat: 2019-2025 has been evaluated in earlier studies of this project."]
    PREREG.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def test():
    want = re.search(r"SHA-256: `([0-9a-f]{64})`", PREREG.read_text(encoding="utf-8")).group(1)
    assert hashlib.sha256(CHOICES.read_bytes()).hexdigest() == want, "choices changed after pre-registration"
    ch = json.loads(CHOICES.read_text())
    levels, comps, info = [], [], {}
    for h in [5, 1]:
        ds = Dataset(h)
        s, e = ds.period_positions("TEST")
        c1, c2 = ch[str(h)]["M1"], ch[str(h)]["M2"]
        E = expert_predictions(ds)
        P = {"fixed": fixed_base(ds, s, e), "always_up": always_up(ds, s, e),
             "M1": vix_trigger(ds, s, e, c1["cut"], c1["window"]),
             "scheduled": scheduled_base(ds, s, e, c1["window"]),
             "M2": regime_select(ds, E, s, e, c2["split"], c2["lookback"]),
             "M2_blind": regime_select(ds, E, s, e, c2["split"], c2["lookback"], use_regime=False)}
        info[h] = {"M1_fires": P["M1"].attrs["n_fires"], "M2_chosen": P["M2"].attrs["chosen"],
                   "M2_blind_chosen": P["M2_blind"].attrs["chosen"]}
        for k, p in P.items():
            levels.append({"horizon": h, "model": k, **metrics(p)})
        for fam, a, b in [("M1", "M1", "fixed"), ("M1", "M1", "scheduled"),
                          ("M2", "M2", "fixed"), ("M2", "M2", "M2_blind"), ("M2", "M2", "always_up")]:
            comps.append({"horizon": h, "family": fam, "comparison": f"{a} vs {b}", **paired_bootstrap(P[a], P[b], h)})
    comp = pd.DataFrame(comps)
    for _, g in comp.groupby(["horizon", "family"]):
        adj = holm(np.concatenate([g["p_balanced_acc"], g["p_roc_auc"], g["p_brier"]]))
        n = len(g)
        comp.loc[g.index, "holm_balanced_acc"], comp.loc[g.index, "holm_roc_auc"], comp.loc[g.index, "holm_brier"] = adj[:n], adj[n:2 * n], adj[2 * n:]
    comp.to_csv(RES_DIR / "vix_test_comparisons.csv", index=False)
    pd.DataFrame(levels).to_csv(RES_DIR / "vix_test_levels.csv", index=False)

    D = spy_daily()
    a, b = PERIODS["TEST"]
    srows = []
    for var, m3 in [("primary", ch["M3"]), ("vix_variant", ch["M3"]["vix_variant"])]:
        for cost in (COST, 0):
            r = vol_managed(D, a, b, m3["signal"], m3["cap"], cost)
            tag = {"variant": var, "signal": m3["signal"], "cap": m3["cap"], "cost_bps": cost}
            for k in ("managed", "bh"):
                srows.append({**tag, "strategy": k, **strat_stats(r[k]),
                              "mean_weight": float(r["w"].mean()) if k == "managed" else 1.0})
            srows.append({**tag, "strategy": "managed - bh", **sharpe_bootstrap(r["managed"], r["bh"])})
    S = pd.DataFrame(srows)
    S.to_csv(RES_DIR / "vix_test_strategy.csv", index=False)
    (RES_DIR / "vix_test_info.json").write_text(json.dumps(info, indent=2, default=str))
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
        print(pd.DataFrame(levels).to_string())
        print(comp[["horizon", "comparison", "d_balanced_acc", "d_balanced_acc_lo", "d_balanced_acc_hi", "d_roc_auc",
                    "d_roc_auc_lo", "d_roc_auc_hi", "d_brier", "holm_balanced_acc", "holm_roc_auc", "holm_brier"]].to_string())
        print(S.to_string())
        print(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    {"validate": validate, "select": select, "test": test}[sys.argv[1]]()

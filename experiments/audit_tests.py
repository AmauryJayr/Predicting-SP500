"""AUDIT (pessimist): leakage / causality tests on the optimist's code.

Does not modify any optimist file; imports common.py / methods.py read-only.

T1  Truncation test on REAL data: for each of the 12 preregistered method specs (plus primaries),
    corrupt every label with label_end > cutoff and every feature row / regime variable dated
    after the cutoff, rerun, and require that every prediction dated <= cutoff is bit-identical.
T2  Label-leak canary on SYNTHETIC data: labels are i.i.d. per aligned block of h rows, features are
    pure noise. Any mechanism that consumes a label before label_end <= decision date would see the
    current block's label and score far above BA 0.5. Expected: every method ~0.5.
Outputs: results/audit_truncation.csv, results/audit_canary.csv
"""
import json
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import common
from common import RES_DIR, Dataset, metrics
from methods import run_spec
from audit_synth import SynthDataset

SHORT = json.load(open(RES_DIR / "shortlist.json"))


def specs():
    out = []
    for h in ("5", "1"):
        for i, e in enumerate(SHORT[h], 1):
            for role in ("method", "primary", "secondary"):
                if e.get(role):
                    out.append((int(h), f"{h}.{i}", role, e[role]))
    # extra mechanisms not in the shortlist (prior shift, online SGD, Page-Hinkley)
    lr = {"C": 0.003}
    base = {"kind": "wf", "model": "LogReg", "params": lr, "feats": "core", "norm": None, "refit": None}
    extra = [
        {"kind": "prior", "base": base, "N": 126, "thr_mode": "train"},
        {"kind": "online", "feats": "core", "norm": None, "alpha": 1e-3, "eta0": 1e-2},
        {"kind": "detect", "model": "LogReg", "params": lr, "feats": "core", "norm": None, "detector": "ph",
         "delta": 0.005, "lam": 0.5, "stream": "logloss", "window": 504},
        {"kind": "regime", "model": "LogReg", "params": lr, "feats": "core", "norm": "rank252",
         "regime_var": "vix_pct_rank_252d"},
        base | {"norm": "z1260", "refit": 21, "half_life": 252},
    ]
    for h in (5, 1):
        for j, s in enumerate(extra):
            out.append((h, f"x{j}", "method", s))
    return out


def corrupt(ds, cutoff, rng):
    cut = np.datetime64(pd.Timestamp(cutoff))
    bad = ds.label_end > cut
    ds.y = ds.y.copy()
    ds.y[bad] = rng.integers(0, 2, bad.sum())
    F = ds.F_full.copy()
    after = F.index > pd.Timestamp(cutoff)
    F.loc[after] = rng.normal(size=(after.sum(), F.shape[1])) * 5
    ds.F_full = F
    ds._X_cache = {}
    d = ds.data.copy()
    after_d = d.index > pd.Timestamp(cutoff)
    for c in ["vix_level", "vix_pct_rank_252d"]:
        d.loc[after_d, c] = rng.uniform(0, 80, after_d.sum()) if c == "vix_level" else rng.uniform(0, 1, after_d.sum())
    ds.data = d


def truncation_test():
    common.PERIODS["AUD"] = ("2019-01-01", "2020-12-31")
    cutoff = "2020-03-31"
    rows = []
    clean = {h: Dataset(h) for h in (5, 1)}
    dirty = {}
    for h in (5, 1):
        dirty[h] = Dataset(h)
        corrupt(dirty[h], cutoff, np.random.default_rng(h))
    for h, sid, role, spec in specs():
        a = run_spec(clean[h], spec, "AUD")
        b = run_spec(dirty[h], spec, "AUD")
        m = a.index <= pd.Timestamp(cutoff)
        da = np.abs(a.loc[m, "proba"].to_numpy() - b.loc[m, "proba"].to_numpy()).max()
        dt = np.abs(a.loc[m, "thr"].to_numpy() - b.loc[m, "thr"].to_numpy()).max()
        after = np.abs(a.loc[~m, "proba"].to_numpy() - b.loc[~m, "proba"].to_numpy()).max()
        rows.append(dict(h=h, id=sid, role=role, kind=spec["kind"], n_before=int(m.sum()),
                         max_abs_diff_proba_before=da, max_abs_diff_thr_before=dt,
                         max_abs_diff_after=after, passed=bool(da == 0 and dt == 0)))
        print(rows[-1], flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(RES_DIR / "audit_truncation.csv", index=False)
    return T


def canary():
    rows = []
    for h in (5, 1):
        for seed in (0, 1):
            ds = SynthDataset("canary", h=h, seed=seed)
            for _, sid, role, spec in specs():
                if _ != h:
                    continue
                p = run_spec(ds, spec, "TEST")
                mt = metrics(p)
                rows.append(dict(h=h, seed=seed, id=sid, role=role, kind=spec["kind"],
                                 BA=mt["balanced_acc"], AUC=mt["roc_auc"]))
    C = pd.DataFrame(rows)
    C.to_csv(RES_DIR / "audit_canary.csv", index=False)
    print(C.groupby(["h", "kind"])[["BA", "AUC"]].agg(["mean", "max"]).round(3))
    return C


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "trunc"):
        T = truncation_test()
        print("truncation: all passed =", T.passed.all())
    if which in ("all", "canary"):
        canary()

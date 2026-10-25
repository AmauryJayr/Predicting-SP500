"""AUDIT (pessimist): synthetic data with KNOWN drift, run through the optimist's unmodified methods.

Question: can the implemented drift-aware mechanisms adapt at all? If they cannot beat the
trained-once model when we inject drift, the negative SPY result says nothing about drift.

SynthDataset mimics the Dataset interface (dates, label_end with h-day delay, y, X(feats, norm),
n_known, period_positions, data[regime vars]) on the same business-day calendar, so the same
TRAIN/VAL/TEST periods and warm-ups apply.

Scenarios (x_k ~ AR(1), rho=0.3; y = 1[score + logistic noise > 0]):
  stationary   score = k*(x0 + 0.7 x1)                    (control: adaptation should NOT help)
  flip_test    b0 = +k before 2021-01-01, -k after         (concept drift inside the test period)
  flip_2y      b0 flips sign every 504 sessions            (recurring concept drift)
  covshift     observed x0 = x0 + m_t, m_t a slow random walk; y depends on true x0
  labelshift   intercept +1.0 from 2021-01-01               (prior / label drift)
  canary       y i.i.d. per ALIGNED block of h rows, features noise (leak detector, used by audit_tests)
Strength k: 'strong' (k=1, AUC ~0.75) and 'weak' (k=0.15, AUC ~0.54, roughly SPY-like).
Outputs: results/audit_synth.csv, results/audit_synth_summary.csv
"""
import json
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import RES_DIR, Dataset, causal_normalize, metrics, STUDY_START

P_FEAT = 6


class SynthDataset(Dataset):
    def __init__(self, scenario, h=5, seed=0, k=1.0):
        self.h = h
        rng = np.random.default_rng(seed)
        cal = pd.bdate_range("2006-01-02", "2025-12-31")
        n = len(cal)
        e = rng.normal(size=(n, P_FEAT))
        X = np.empty_like(e)
        X[0] = e[0]
        for t in range(1, n):
            X[t] = 0.3 * X[t - 1] + np.sqrt(1 - 0.09) * e[t]
        b0 = np.full(n, k)
        c = np.zeros(n)
        Xobs = X.copy()
        if scenario == "flip_test":
            b0[cal >= "2021-01-01"] = -k
        elif scenario == "flip_2y":
            b0 = k * np.where((np.arange(n) // 504) % 2 == 0, 1.0, -1.0)
        elif scenario == "covshift":
            Xobs[:, 0] = X[:, 0] + np.cumsum(rng.normal(0, 0.1, n))
        elif scenario == "labelshift":
            c[cal >= "2021-01-01"] = 1.0
        score = b0 * X[:, 0] + 0.7 * k * X[:, 1] + c
        if scenario == "canary":
            blk = np.arange(n) // h
            yb = rng.integers(0, 2, blk.max() + 1)
            yfull = yb[blk].astype(float)
            Xobs = rng.normal(size=(n, P_FEAT))
        else:
            yfull = (score + rng.logistic(size=n) > 0).astype(float)
        cols = [f"x{j}" for j in range(P_FEAT)]
        F = pd.DataFrame(Xobs, index=cal, columns=cols)
        self.F_full = F
        vix = pd.Series(np.exp(0.3 * np.cumsum(rng.normal(0, 0.05, n)) + 3), index=cal)
        data = F.copy()
        data["vix_level"] = vix
        data["vix_pct_rank_252d"] = vix.rolling(252).rank(pct=True)
        data["target"] = yfull
        data["label_end"] = pd.Series(cal, index=cal).shift(-h)
        data = data.loc[STUDY_START:].iloc[:-h]
        data["target"] = data["target"].astype(int)
        self.all_features = cols
        self.data = data
        self.index = data.index
        self.dates = data.index.to_numpy()
        self.label_end = data["label_end"].to_numpy()
        self.y = data["target"].to_numpy()
        self.n = len(data)
        self._X_cache = {}

    def X(self, feats="all", norm=None):
        key = norm
        if key not in self._X_cache:
            F = self.F_full
            if norm is not None:
                kind = "z" if norm.startswith("z") else "rank"
                F = causal_normalize(F, int(norm.lstrip("zrank")), kind)
            self._X_cache[key] = np.nan_to_num(F.reindex(self.index).to_numpy(float), nan=0.0)
        return self._X_cache[key]


def spec_list(h):
    """The 12 preregistered entries (method/primary/secondary) at this horizon + generic mechanisms."""
    short = json.load(open(RES_DIR / "shortlist.json"))
    out = []
    for i, e in enumerate(short[str(h)], 1):
        for role in ("method", "primary", "secondary"):
            if e.get(role):
                out.append((f"{h}.{i}", e["family"], role, e[role]))
    lr = {"C": 0.003}
    fx = {"kind": "wf", "model": "LogReg", "params": lr, "feats": "core", "norm": None, "refit": None}
    gen = [
        ("g.expand21", "refit", "method", fx | {"refit": 21}),
        ("g.roll504", "refit", "method", fx | {"refit": 21, "window": 504}),
        ("g.roll252", "refit", "method", fx | {"refit": 21, "window": 252}),
        ("g.decay252", "decay", "method", fx | {"refit": 21, "half_life": 252}),
        ("g.platt252", "platt", "method", {"kind": "platt", "base": fx | {"refit": 21}, "N": 252, "mode": "recent"}),
        ("g.prior126", "prior", "method", {"kind": "prior", "base": fx, "N": 126, "thr_mode": "train"}),
        ("g.ph504", "detect", "method", {"kind": "detect", "model": "LogReg", "params": lr, "feats": "core",
                                         "norm": None, "detector": "ph", "delta": 0.005, "lam": 0.5,
                                         "stream": "logloss", "window": 504}),
        ("g.adwin504", "detect", "method", {"kind": "detect", "model": "LogReg", "params": lr, "feats": "core",
                                            "norm": None, "detector": "adwin", "delta": 0.002, "lam": None,
                                            "stream": "error", "window": 504}),
        ("g.sgd", "online", "method", {"kind": "online", "feats": "core", "norm": None, "alpha": 1e-3, "eta0": 1e-2}),
        ("g.sgd", "online", "primary", {"kind": "online", "feats": "core", "norm": None, "alpha": 1e-3,
                                        "eta0": 1e-2, "adaptive": False}),
        ("g.z252", "norm", "method", fx | {"norm": "z252"}),
        ("g.fixed", "fixed", "primary", fx),
    ]
    return out + gen


def main(scen_list=None, strengths=(("strong", 1.0), ("weak", 0.15)), seeds=(0,), h=5, tag=""):
    from joblib import Parallel, delayed
    from methods import run_spec

    scen_list = scen_list or ["stationary", "flip_test", "flip_2y", "covshift", "labelshift"]

    def job(scen, sname, k, seed, sid, fam, role, spec):
        ds = SynthDataset(scen, h=h, seed=seed, k=k)
        p = run_spec(ds, spec, "TEST")
        m = metrics(p)
        q = p[p.index >= "2021-01-01"]
        m2 = metrics(q)
        return dict(scenario=scen, strength=sname, seed=seed, h=h, id=sid, family=fam, role=role,
                    BA=m["balanced_acc"], AUC=m["roc_auc"], Brier=m["brier"],
                    BA_post2021=m2["balanced_acc"], AUC_post2021=m2["roc_auc"], Brier_post2021=m2["brier"],
                    n_refits=p.attrs.get("n_refits", np.nan))

    jobs = [(sc, sn, k, sd, *s) for sc in scen_list for sn, k in strengths for sd in seeds for s in spec_list(h)]
    rows = Parallel(n_jobs=15, verbose=2)(delayed(job)(*j) for j in jobs)
    R = pd.DataFrame(rows)
    R.to_csv(RES_DIR / f"audit_synth_h{h}{tag}.csv", index=False)
    # deltas vs matched primary (for shortlisted ids) or vs g.fixed (generic)
    out = []
    for (sc, sn, sd), g in R.groupby(["scenario", "strength", "seed"]):
        fixed = g[(g.id == "g.fixed")].iloc[0]
        for sid, gg in g.groupby("id"):
            if "method" not in set(gg.role):
                continue
            m = gg[gg.role == "method"].iloc[0]
            prim = gg[gg.role == "primary"].iloc[0] if "primary" in set(gg.role) else fixed
            out.append(dict(scenario=sc, strength=sn, seed=sd, id=sid, family=m.family,
                            BA=m.BA, BA_fixed=prim.BA, dBA=m.BA - prim.BA, dAUC=m.AUC - prim.AUC,
                            dBrier=prim.Brier - m.Brier, dBA_post2021=m.BA_post2021 - prim.BA_post2021,
                            dAUC_post2021=m.AUC_post2021 - prim.AUC_post2021, n_refits=m.n_refits))
    S = pd.DataFrame(out)
    S.to_csv(RES_DIR / f"audit_synth_summary_h{h}{tag}.csv", index=False)
    return R, S


if __name__ == "__main__":
    h = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    seeds = tuple(range(int(sys.argv[2]))) if len(sys.argv) > 2 else (0,)
    if len(sys.argv) > 3 and sys.argv[3] == "power":  # weak signal, many seeds: power of a 7-year test
        R, S = main(scen_list=["stationary", "flip_test"], strengths=(("weak", 0.15),), seeds=seeds, h=h, tag="_power")
    else:
        R, S = main(seeds=seeds, h=h)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    piv = S.groupby(["strength", "scenario", "id"])[["dBA", "dAUC", "dBrier", "dAUC_post2021"]].mean().round(3)
    print(piv.to_string())

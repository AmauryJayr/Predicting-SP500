"""AUDIT (pessimist): corrected copy of the drift-detection family ('detect').

Problem in methods.run_detect (found with the synthetic sanity check, audit_synth.py):
  after a detection it retrains on the last W=504 sessions (or the full history), which right after a
  change are almost all PRE-change rows, then fully resets the detector. A change detector reacts to a
  change in the error, not to a persistently high error, so after the reset it never fires again and the
  model stays stale. In the synthetic 'flip_test' scenario ADWIN fires once and recovers only +0.02 BA,
  while a plain 504-session rolling refit recovers +0.15.
Fix (standard ADWIN usage): retrain on rows after the estimated change point (ADWIN cut / Page-Hinkley
  argmin), at least `min_rows`, and keep refitting on the growing post-change window every `R` sessions
  until it reaches `W` (then it behaves like the original W-window model). No-detection behaviour is unchanged.

Run: python audit_fixed_methods.py  -> results/audit_detect_fixed_synth.csv, results/audit_detect_fixed_spy.csv
EXPLORATORY: the SPY run is a post-test sensitivity check, not a preregistered result.
"""
import json
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
from common import RES_DIR, Dataset, fit_model, metrics, to_pred
from methods import _X, _est, logloss, run_spec
from audit_lib import boot


class ADWIN2:
    def __init__(self, delta, max_len=1000, min_side=30, step=5):
        self.delta, self.max_len, self.min_side, self.step = delta, max_len, min_side, step
        self.w, self.rows = [], []

    def update(self, x, row):
        self.w.append(float(x)); self.rows.append(row)
        if len(self.w) > self.max_len:
            self.w.pop(0); self.rows.pop(0)
        n = len(self.w)
        if n < 2 * self.min_side:
            return None
        cs = np.cumsum(self.w)
        for c in range(self.min_side, n - self.min_side + 1, self.step):
            n0, n1 = c, n - c
            mu0, mu1 = cs[c - 1] / n0, (cs[-1] - cs[c - 1]) / n1
            m = 1 / (1 / n0 + 1 / n1)
            if mu1 - mu0 > np.sqrt(np.log(4 * n / self.delta) / (2 * m)):
                cut_row = self.rows[c]
                self.w, self.rows = self.w[c:], self.rows[c:]  # keep the post-change window (no full reset)
                return cut_row
        return None


class PH2:
    def __init__(self, delta, lam):
        self.delta, self.lam = delta, lam
        self.n, self.mean, self.m, self.M, self.arg = 0, 0.0, 0.0, 0.0, None

    def update(self, x, row):
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.m += x - self.mean - self.delta
        if self.m <= self.M:
            self.M, self.arg = self.m, row
        if self.m - self.M > self.lam:
            cut = self.arg if self.arg is not None else row
            self.n, self.mean, self.m, self.M, self.arg = 0, 0.0, 0.0, 0.0, None
            return cut
        return None


def run_detect_fixed(ds, spec, start, end, min_rows=250, R=21):
    X, y, est = _X(ds, spec), ds.y, _est(spec)
    W, min_gap = spec.get("window") or 10 ** 9, spec.get("min_gap", 21)
    det = PH2(spec["delta"], spec["lam"]) if spec["detector"] == "ph" else ADWIN2(spec["delta"])
    stream = spec.get("stream", "logloss")

    def fit_from(t, lo):
        nk = ds.n_known(t)
        lo = max(0, min(lo, nk - min_rows), nk - W)
        tr = np.arange(lo, nk)
        assert ds.label_end[tr].max() <= ds.dates[t]
        return fit_model(est, X[tr], y[tr]), y[tr].mean(), nk - lo

    model, thr, _ = fit_from(start, 0)
    regime_lo, last_fit, n_fires, growing = 0, start, 0, False
    proba = np.empty(end - start); th = np.empty(end - start)
    fed = start
    for t in range(start, end):
        nk = ds.n_known(t)
        cut = None
        for s in range(max(fed, start), nk):
            p = proba[s - start]
            x = logloss(y[s], p) if stream == "logloss" else float((p >= th[s - start]) != y[s])
            c = det.update(x, s)
            cut = c if c is not None else cut
        fed = max(fed, nk)
        if cut is not None and t - last_fit >= min_gap:
            regime_lo, growing = cut, True
            model, thr, ntr = fit_from(t, regime_lo)
            last_fit = t; n_fires += 1
        elif growing and t - last_fit >= R:
            model, thr, ntr = fit_from(t, regime_lo)
            last_fit = t
            growing = ntr < W
        proba[t - start] = model.predict_proba(X[t:t + 1])[0, 1]
        th[t - start] = thr
    out = pd.DataFrame({"proba": proba, "thr": th}, index=np.arange(start, end))
    out.attrs["n_refits"] = n_fires
    return out


def run_fixed_spec(ds, spec, period):
    s, e = ds.period_positions(period)
    df = run_detect_fixed(ds, spec, s, e)
    p = to_pred(ds, df, s, e); p.attrs.update(df.attrs)
    return p


DETS = {
    "adwin d=0.002 W=504": dict(detector="adwin", delta=0.002, lam=None, stream="error", window=504),
    "adwin d=0.05 W=504": dict(detector="adwin", delta=0.05, lam=None, stream="error", window=504),
    "ph lam=2 W=504": dict(detector="ph", delta=0.005, lam=2.0, stream="logloss", window=504),
    "adwin d=0.002 expanding": dict(detector="adwin", delta=0.002, lam=None, stream="error", window=None),
}


def _synth_job(scen, k, sname, seed, dname, which):
    from audit_synth import SynthDataset
    ds = SynthDataset(scen, h=5, seed=seed, k=k)
    sp = {"kind": "detect", "model": "LogReg", "params": {"C": 0.003}, "feats": "core", "norm": None, **DETS[dname]}
    if which == "fixed_model":
        sp = {"kind": "wf", "model": "LogReg", "params": {"C": 0.003}, "feats": "core", "norm": None, "refit": None}
        p = run_spec(ds, sp, "TEST")
    elif which == "original":
        p = run_spec(ds, sp, "TEST")
    else:
        p = run_fixed_spec(ds, sp, "TEST")
    m = metrics(p)
    return dict(scenario=scen, strength=sname, seed=seed, detector=dname, impl=which, BA=m["balanced_acc"],
                AUC=m["roc_auc"], n_fires=p.attrs.get("n_refits", np.nan))


def _spy_job(h, dname, feats, which):
    ds = Dataset(h)
    sp = {"kind": "detect", "model": "LogReg", "params": {"C": 0.003}, "feats": feats, "norm": None, **DETS[dname]}
    p = run_spec(ds, sp, "TEST") if which == "original" else run_fixed_spec(ds, sp, "TEST")
    return (h, dname, feats, which), p


def main():
    jobs = []
    for scen in ("stationary", "flip_test"):
        for sname, k in (("strong", 1.0), ("weak", 0.15)):
            for seed in range(3):
                jobs.append((scen, k, sname, seed, "adwin d=0.002 W=504", "fixed_model"))
                for d in DETS:
                    for w in ("original", "corrected"):
                        jobs.append((scen, k, sname, seed, d, w))
    R = pd.DataFrame(Parallel(n_jobs=15)(delayed(_synth_job)(*j) for j in jobs))
    fx = R[R.impl == "fixed_model"][["scenario", "strength", "seed", "BA", "AUC"]].rename(columns={"BA": "BA_fix", "AUC": "AUC_fix"})
    R = R[R.impl != "fixed_model"].merge(fx, on=["scenario", "strength", "seed"])
    R["dBA"] = R.BA - R.BA_fix; R["dAUC"] = R.AUC - R.AUC_fix
    R.to_csv(RES_DIR / "audit_detect_fixed_synth.csv", index=False)
    print(R.groupby(["scenario", "strength", "detector", "impl"])[["dBA", "dAUC", "n_fires"]].mean().round(3).to_string())

    # SPY test (EXPLORATORY): same detector settings as preregistered 5.3 (core, d=0.002) and 1.5 (all, d=0.05)
    P = pd.read_pickle(RES_DIR / "test_predictions.pkl")
    sj = [(5, "adwin d=0.002 W=504", "core", w) for w in ("original", "corrected")] + \
         [(1, "adwin d=0.05 W=504", "all", w) for w in ("original", "corrected")] + \
         [(h, "ph lam=2 W=504", f, "corrected") for h, f in ((5, "core"), (1, "all"))]
    out = dict(Parallel(n_jobs=6)(delayed(_spy_job)(*j) for j in sj))
    rows = []
    for (h, d, f, w), p in out.items():
        prim = P[(str(h), 3 if h == 5 else 5, "primary")]
        rows.append(dict(h=h, detector=d, feats=f, impl=w, n_fires=p.attrs.get("n_refits"), **{
            k: v for k, v in metrics(p).items() if k in ("balanced_acc", "roc_auc")}, **boot(p, prim, L=20)))
    S = pd.DataFrame(rows)
    S.to_csv(RES_DIR / "audit_detect_fixed_spy.csv", index=False)
    print(S.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

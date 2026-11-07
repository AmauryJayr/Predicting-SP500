#!/usr/bin/env python
"""
Chronos-2 (zero-shot, multivariate) for SPY direction, in the same framework as the rest of experiments/.

Same labels, periods, purging rule, threshold rule, metrics and paired block bootstrap as common.py.

    python chronos_experiment/chronos_study.py smoke              # 1-2 min: checks install, output shapes, speed
    python chronos_experiment/chronos_study.py validate [--stride 1]
    python chronos_experiment/chronos_study.py select             # freezes one config per horizon (+ SHA-256)
    python chronos_experiment/chronos_study.py test               # one-shot 2019-2025 run, refuses to run twice
    python chronos_experiment/chronos_study.py explore            # OPTIONAL, exploratory: every config on TEST

How a forecast becomes a direction probability
----------------------------------------------
Label at decision day t (close):  up if Close[t+h] > Open[t+1].
Define the series  g[t] = log(Close[t] / Open[t-h+1]),  which is fully known at the close of t.
Then g[t+h] = log(Close[t+h] / Open[t+1]) is exactly the label's log forward return.
At each decision day we give Chronos-2 the context g[..t] (plus covariates up to t), forecast h steps,
read the quantiles of step h, and set P(up) = 1 - F(0), interpolating the forecast CDF at zero.
Nothing after the close of t is ever in the context, so the purging rule holds by construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent  # chronos_experiment/
sys.path.insert(0, str(HERE.parent / "experiments"))  # reuse the project's common.py
from common import (CORE_FEATURES, DATA_DIR, ROOT, SELECTION_FOLDS,  # noqa: E402
                    Dataset, metrics, paired_bootstrap)

OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)
VAL_LOG = OUT / "chronos_validation_log.csv"
SHORTLIST = OUT / "chronos_shortlist.json"
HASHFILE = OUT / "chronos_shortlist.sha256"

MODEL_ID = "amazon/chronos-2"
# Chronos-2's native 21 quantile levels: 0.01, 0.05, 0.10, 0.15, ..., 0.90, 0.95, 0.99
QLEVELS = [0.01, 0.05] + [round(0.10 + 0.05 * i, 2) for i in range(17)] + [0.95, 0.99]
JOINT = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
VARIANTS = {
    "uni": "SPY label series only (univariate)",
    "joint": "SPY, QQQ, IWM, TLT, GLD label series forecast jointly, log VIX as past covariate",
    "core": "SPY label series, the 23 CORE_FEATURES as past covariates",
}
CONTEXTS = [256, 512, 1024, 2048]  # context length = Chronos's analogue of a rolling training window
HORIZONS = [1, 5]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class ChronosData:
    def __init__(self, h: int):
        self.h = h
        self.ds = Dataset(h)
        px = pd.read_pickle(DATA_DIR / "aligned_prices.pkl")
        self.cal = px["SPY"].index

        G = {s: np.log(px[s]["Close"] / px[s]["Open"].shift(h - 1)) for s in JOINT}
        # g[t+h] must equal the label's forward return for every modelling row
        chk = np.exp(G["SPY"].shift(-h).reindex(self.ds.index)) - 1
        assert np.allclose(chk.to_numpy(), self.ds.data["fwd_ret"].to_numpy(), atol=1e-10), \
            "label series does not reproduce Dataset.fwd_ret"
        self.first = h - 1  # first non-NaN position of g
        self.G = {s: g.ffill().fillna(0.0).to_numpy(float) for s, g in G.items()}
        self.lvix = np.log(px["^VIX"]["Close"]).ffill().bfill().to_numpy(float)

        F = self.ds.F_full[CORE_FEATURES].reindex(self.cal)
        self.first_core = int(np.flatnonzero(F.notna().all(axis=1).to_numpy())[0])
        self.core = F.ffill().fillna(0.0).to_numpy(float)

        self.cal_pos = self.cal.get_indexer(self.ds.index)  # Dataset row -> calendar position
        assert (self.cal_pos >= 0).all()

    def make_input(self, t: int, L: int, variant: str) -> dict:
        """Context ending at calendar position t (the decision day's close). Nothing after t is used."""
        first = max(self.first, self.first_core) if variant == "core" else self.first
        sl = slice(max(t - L + 1, first), t + 1)
        if variant == "uni":
            return {"target": self.G["SPY"][sl]}
        if variant == "joint":
            return {"target": np.stack([self.G[s][sl] for s in JOINT]),  # (n_variates, T), SPY first
                    "past_covariates": {"log_vix": self.lvix[sl]}}
        if variant == "core":
            return {"target": self.G["SPY"][sl],
                    "past_covariates": {c: self.core[sl, j] for j, c in enumerate(CORE_FEATURES)}}
        raise ValueError(variant)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_pipeline():
    import torch
    from chronos import Chronos2Pipeline

    if torch.cuda.is_available():
        dev = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    print(f"loading {MODEL_ID} on {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))
    if dev == "cpu":
        print("  NOTE: running on CPU. For an NVIDIA GPU, install a CUDA build of PyTorch (see instructions).")
    return Chronos2Pipeline.from_pretrained(MODEL_ID, device_map=dev)


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=float)


BATCH = 64  # set with --batch; halves itself automatically on a GPU out-of-memory error


def _predict_chunk(pipe, chunk, h):
    """predict_quantiles on one chunk; on CUDA out-of-memory, halve the global batch size and retry."""
    global BATCH
    import torch
    try:
        q, _mean = pipe.predict_quantiles(chunk, prediction_length=h, quantile_levels=QLEVELS)
        return list(q)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(chunk) == 1:
            raise
        BATCH = max(1, len(chunk) // 2)
        print(f"  GPU out of memory -> batch size now {BATCH}")
        out = []
        for j in range(0, len(chunk), BATCH):
            out += _predict_chunk(pipe, chunk[j:j + BATCH], h)
        return out


def spy_quantiles(pipe, inputs: list[dict], h: int, batch: int | None = None) -> np.ndarray:
    """Quantiles of SPY's step-h forecast, shape (n_inputs, len(QLEVELS))."""
    out = []
    i = 0
    while i < len(inputs):
        b = batch or BATCH
        q = _predict_chunk(pipe, inputs[i:i + b], h)
        i += b
        for qi in q:
            a = _np(qi)
            if a.ndim == 3:  # (n_variates, h, n_quantiles): SPY is variate 0
                a = a[0]
            assert a.shape == (h, len(QLEVELS)), f"unexpected Chronos output shape {a.shape}"
            out.append(a[h - 1])
    return np.vstack(out)


def prob_up(Q: np.ndarray) -> np.ndarray:
    tau = np.array(QLEVELS)
    p = np.empty(len(Q))
    for i, qv in enumerate(Q):
        qv = np.maximum.accumulate(qv)  # guard against quantile crossing
        F0 = np.interp(0.0, qv, tau, left=tau[0] / 2, right=1 - (1 - tau[-1]) / 2)
        p[i] = 1.0 - F0
    return p


def run_period(pipe, cd: ChronosData, variant: str, L: int, period: str, stride: int = 1) -> pd.DataFrame:
    """Predictions for every `stride`-th Dataset row of `period`. Cached on disk, so runs can resume."""
    f = OUT / f"pred_h{cd.h}_{variant}_L{L}_{period}_s{stride}.pkl"
    if f.exists():
        return pd.read_pickle(f)
    ds = cd.ds
    s, e = ds.period_positions(period)
    rows = np.arange(s, e, stride)
    inputs = [cd.make_input(int(cd.cal_pos[r]), L, variant) for r in rows]
    t0 = time.time()
    Q = spy_quantiles(pipe, inputs, cd.h)
    # Same threshold rule as the trained-once models: base rate of labels known at the period start.
    thr = float(ds.y[: ds.n_known(s)].mean())
    df = pd.DataFrame({"y": ds.y[rows], "proba": prob_up(Q), "thr": thr}, index=ds.index[rows])
    df["pred"] = (df["proba"] >= df["thr"]).astype(int)
    for j, tq in enumerate(QLEVELS):
        df[f"q{tq:.2f}"] = Q[:, j]
    pd.to_pickle(df, f)
    print(f"  h={cd.h} {variant:5s} L={L:4d} {period:4s}: {len(rows)} forecasts in {time.time() - t0:.0f}s")
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    order, m, run = np.argsort(p), len(p), 0.0
    adj = np.empty(m)
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def notebook_logreg_fixed(h: int) -> pd.DataFrame | None:
    f = ROOT / "results" / f"h{h}" / "test_predictions.csv"
    if not f.exists():
        return None
    p = pd.read_csv(f, parse_dates=["date"])
    p = p[(p.features == "all") & (p.model == "LogReg") & (p.scheme == "fixed")]
    return p.set_index("date")[["y", "proba", "thr", "pred"]]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_smoke(args):
    pipe = load_pipeline()
    per = []
    for h in HORIZONS:
        cd = ChronosData(h)
        s, _ = cd.ds.period_positions("VAL")
        for v in VARIANTS:
            inp = [cd.make_input(int(cd.cal_pos[r]), 1024, v) for r in range(s, s + 8)]
            t0 = time.time()
            Q = spy_quantiles(pipe, inp, h, batch=8)
            dt = (time.time() - t0) / len(inp)
            per.append(dt)
            print(f"h={h} {v:5s}: quantiles {Q.shape}, P(up) {np.round(prob_up(Q), 3)}, {dt:.2f}s per forecast")
    n_val = len(HORIZONS) * len(VARIANTS) * len(CONTEXTS) * 1500 / args.stride
    print(f"\nRough runtime: validate ~{n_val * np.mean(per) / 3600:.1f} h (stride {args.stride}), "
          f"test ~{2 * 1760 * np.mean(per) / 60:.0f} min. Smoke test passed.")


def cmd_validate(args):
    pipe = load_pipeline()
    for h in args.horizons:
        cd = ChronosData(h)
        for v in args.variants:
            for L in args.contexts:
                for fold in SELECTION_FOLDS:
                    run_period(pipe, cd, v, L, fold, args.stride)
    # rebuild the log from every cached validation file
    rows = []
    for f in sorted(OUT.glob("pred_h*_*_L*_*_s*.pkl")):
        h, v, L, period, s = f.stem.split("_")[1:]
        if period == "TEST":
            continue
        d = pd.read_pickle(f)
        rows.append({"h": int(h[1:]), "variant": v, "L": int(L[1:]), "fold": period,
                     "stride": int(s[1:]), **metrics(d)})
    log = pd.DataFrame(rows)
    log.to_csv(VAL_LOG, index=False)
    print(log.pivot_table(index=["h", "variant", "L"], columns="fold",
                          values=["balanced_acc", "roc_auc"]).round(3).to_string())


def cmd_select(args):
    log = pd.read_csv(VAL_LOG)
    log["dBA"], log["dAUC"] = log.balanced_acc - 0.5, log.roc_auc - 0.5
    keys = ["h", "variant", "L"]
    tab = log.groupby(keys).agg(mean_dBA=("dBA", "mean"), mean_dAUC=("dAUC", "mean"),
                                n_ba_pos=("dBA", lambda x: int((x > 0).sum())),
                                n_auc_pos=("dAUC", lambda x: int((x > 0).sum())),
                                n_folds=("fold", "nunique")).reset_index()
    val = log[log.fold == "VAL"].set_index(keys)["dAUC"].rename("val_dAUC")
    tab = tab.join(val, on=keys)
    # Rule (fixed before looking): all 3 folds present; BA > 0.5 and AUC > 0.5 in >= 2 of 3 folds;
    # AUC > 0.5 on VAL itself (the fold most like the test, per the earlier audit). Rank by mean dBA + dAUC.
    tab["eligible"] = (tab.n_folds == 3) & (tab.n_ba_pos >= 2) & (tab.n_auc_pos >= 2) & (tab.val_dAUC > 0)
    tab["score"] = tab.mean_dBA + tab.mean_dAUC
    tab.sort_values(["h", "score"], ascending=[True, False]).to_csv(OUT / "chronos_selection_table.csv", index=False)

    selected = {}
    for h, g in tab.groupby("h"):
        pool = g[g.eligible] if g.eligible.any() else g
        best = pool.sort_values("score", ascending=False).iloc[0]
        # Platt recalibration fitted once on the VAL predictions of the chosen config (pre-test data only)
        from sklearn.linear_model import LogisticRegression
        vf = next(OUT.glob(f"pred_h{h}_{best.variant}_L{best.L}_VAL_s*.pkl"))
        d = pd.read_pickle(vf)
        z = np.log(np.clip(d.proba, 1e-4, 1 - 1e-4) / np.clip(1 - d.proba, 1e-4, 1))
        lr = LogisticRegression(C=1e6).fit(z.to_numpy()[:, None], d.y)
        selected[str(int(h))] = {
            "variant": best.variant, "L": int(best.L), "eligible": bool(best.eligible),
            "val_mean_dBA": round(float(best.mean_dBA), 4), "val_mean_dAUC": round(float(best.mean_dAUC), 4),
            "platt_a": float(lr.coef_[0, 0]), "platt_b": float(lr.intercept_[0]),
        }
    spec = {"model": MODEL_ID, "quantile_levels": QLEVELS, "selected": selected,
            "confirmatory": ["chronos vs chance", "chronos vs notebook LogReg fixed"],
            "rule": "win = Holm p < 0.05 on BA or AUC (Holm over 2 comparisons x 2 metrics per horizon), "
                    "other metric not negative"}
    SHORTLIST.write_text(json.dumps(spec, indent=2, sort_keys=True))
    HASHFILE.write_text(sha256(SHORTLIST))
    print(json.dumps(selected, indent=2))
    print(f"\nFrozen: {SHORTLIST}\nSHA-256: {sha256(SHORTLIST)}\n"
          "Write this into a chronos_preregistration.md BEFORE running `test`.")


def cmd_test(args):
    assert SHORTLIST.exists(), "run `select` first"
    assert sha256(SHORTLIST) == HASHFILE.read_text().strip(), "shortlist changed after it was frozen"
    out_c = OUT / "chronos_test_comparisons.csv"
    if out_c.exists() and not args.force:
        sys.exit("The test has already been run once. Use --force only if it crashed before finishing.")
    spec = json.loads(SHORTLIST.read_text())
    pipe = load_pipeline()
    levels, comps, years = [], [], []
    for hk, sel in spec["selected"].items():
        h = int(hk)
        cd = ChronosData(h)
        c = run_period(pipe, cd, sel["variant"], sel["L"], "TEST")
        cal = c.copy()
        z = np.log(np.clip(c.proba, 1e-4, 1 - 1e-4) / np.clip(1 - c.proba, 1e-4, 1))
        cal["proba"] = 1 / (1 + np.exp(-(sel["platt_a"] * z + sel["platt_b"])))
        cal["pred"] = (cal.proba >= cal.thr).astype(int)
        chance = c.copy()
        chance["proba"], chance["pred"] = c.thr, 1
        frames = {"chronos": c, "chronos_platt": cal, "chance": chance}
        lr = notebook_logreg_fixed(h)
        idx = c.index
        if lr is not None:
            idx = c.index.intersection(lr.index)
            print(f"h={h}: {len(idx)} of {len(c)} test days overlap with the notebook LogReg predictions")
            if (lr.loc[idx, "y"].to_numpy() == c.loc[idx, "y"].to_numpy()).all():
                frames["logreg_fixed_nb"] = lr
            else:
                print("  labels differ from the notebook's; skipping that comparison")
        for name, d in frames.items():
            levels.append({"h": h, "model": name, **metrics(d.loc[idx])})
            for yr, g in d.loc[idx].groupby(d.loc[idx].index.year):
                years.append({"h": h, "model": name, "year": yr, **metrics(g)})
        pairs = [("chronos", "chance", True), ("chronos", "logreg_fixed_nb", True),
                 ("chronos_platt", "chance", False)]
        hc = []
        for a, b, conf in pairs:
            if b in frames:
                r = paired_bootstrap(frames[a].loc[idx], frames[b].loc[idx], h)
                hc.append({"h": h, "comparison": f"{a} vs {b}", "confirmatory": conf, **r})
        hc = pd.DataFrame(hc)
        conf = hc.confirmatory
        pc = np.concatenate([hc.loc[conf, "p_balanced_acc"], hc.loc[conf, "p_roc_auc"]])
        adj = holm(pc)
        k = int(conf.sum())
        hc.loc[conf, "holm_balanced_acc"], hc.loc[conf, "holm_roc_auc"] = adj[:k], adj[k:]
        comps.append(hc)
    pd.DataFrame(levels).to_csv(OUT / "chronos_test_levels.csv", index=False)
    pd.DataFrame(years).to_csv(OUT / "chronos_test_by_year.csv", index=False)
    comps = pd.concat(comps)
    comps.to_csv(out_c, index=False)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(levels).round(3).to_string())
    print(comps[["h", "comparison", "d_balanced_acc", "d_balanced_acc_lo", "d_balanced_acc_hi",
                 "d_roc_auc", "d_roc_auc_lo", "d_roc_auc_hi", "holm_balanced_acc", "holm_roc_auc"]]
          .round(4).to_string())


def cmd_explore(args):
    """EXPLORATORY: every variant x context on TEST, vs chance. Run only after `test`."""
    assert (OUT / "chronos_test_comparisons.csv").exists(), "run the confirmatory `test` first"
    pipe = load_pipeline()
    rows = []
    for h in HORIZONS:
        cd = ChronosData(h)
        for v in VARIANTS:
            for L in CONTEXTS:
                d = run_period(pipe, cd, v, L, "TEST")
                ch = d.copy()
                ch["proba"], ch["pred"] = d.thr, 1
                rows.append({"h": h, "variant": v, "L": L, **metrics(d), **paired_bootstrap(d, ch, h)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "chronos_explore_test_all_configs.csv", index=False)
    print(t[["h", "variant", "L", "balanced_acc", "roc_auc", "brier", "p_balanced_acc", "p_roc_auc"]]
          .round(3).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=64, help="forecasts per GPU batch (put before the command)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--stride", type=int, default=1)
    v = sub.add_parser("validate")
    v.add_argument("--stride", type=int, default=1, help="score every k-th validation day (faster)")
    v.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    v.add_argument("--variants", nargs="+", default=list(VARIANTS))
    v.add_argument("--contexts", type=int, nargs="+", default=CONTEXTS)
    sub.add_parser("select")
    t = sub.add_parser("test")
    t.add_argument("--force", action="store_true")
    sub.add_parser("explore")
    args = ap.parse_args()
    global BATCH
    BATCH = args.batch
    {"smoke": cmd_smoke, "validate": cmd_validate, "select": cmd_select,
     "test": cmd_test, "explore": cmd_explore}[args.cmd](args)


if __name__ == "__main__":
    main()

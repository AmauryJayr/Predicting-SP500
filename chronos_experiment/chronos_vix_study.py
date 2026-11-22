#!/usr/bin/env python
"""
VIX as a drift / regime indicator for Chronos-2 (zero-shot): the Chronos counterpart of master's
experiments/vix_study.py, with the same protocol (validate on folds -> freeze with SHA-256 -> one test run).

    python chronos_experiment/chronos_vix_study.py plan       # no GPU: how many forecasts are still to compute
    python chronos_experiment/chronos_vix_study.py smoke      # loads Chronos-2, checks cache reuse, estimates runtime
    python chronos_experiment/chronos_vix_study.py validate   # folds 2013-14, 2015-16, 2017-18
    python chronos_experiment/chronos_vix_study.py select     # freezes choices + writes chronos_vix_preregistration.md
    python chronos_experiment/chronos_vix_study.py test       # one run on 2019-2025; refuses to run twice

Master's three uses of the VIX, translated to a model that is never trained
--------------------------------------------------------------------------
M1  VIX-triggered context reset            (master: VIX-triggered retraining)
    Chronos-2's only "training data" is its context window. When the VIX regime flips (1-year percentile rank
    crosses `cut` in either direction, at least 21 sessions after the previous reset: master's rule), the
    context restarts `min_ctx` sessions before the flip and grows back by one session a day to the base length.
    Control: the same reset every 63 sessions, which isolates the VIX trigger (master: scheduled refits).
M2  Regime-conditional expert selection    (master: the same, with logistic-regression / GBDT experts)
    Experts: always-up and four Chronos-2 configurations that differ in inputs and context length. At each
    decision date, use the expert with the lowest Brier score on past *resolved* forecasts made in the same VIX
    regime. master's `regime_select` is reused unchanged.
M3  Volatility-managed exposure            (master: weights from 20-day realised variance or the VIX)
    master's `vol_managed` strategy, unchanged, but the variance estimate is Chronos-2's forecast of next-week
    SPY realised volatility (the frozen pick of chronos_vol_corr_drift.py). Compared with buy-and-hold and with
    master's frozen choice (20-day realised variance, cap 2).

Base model (the "trained once" analogue): the configuration frozen in results/chronos_shortlist.json
(h=1: joint, context 512; h=5: core, context 256), raw P(up), threshold = base rate known at the period start.

No look-ahead
-------------
* Every Chronos context is the window [t-L+1, t] ending at the decision day's close (built by the existing
  make_input functions; ForecastCache only ever shortens the window's start).
* VIX regimes use the 252-day percentile rank of VIX closes <= t (Dataset feature `vix_pct_rank_252d`).
* M2 scores an expert only on forecasts whose label is resolved by t (master's `regime_select`).
* M3's variance forecast uses the context up to the close of t for a position held from the open of t+1 to the
  open of t+2; the scaling constant uses only days before the evaluation period (master's `scale_constant`).
* `validate` never requests a forecast dated in the test period.

Compute reuse
-------------
Forecasts are cached per (calendar position, effective context length) in results_vix/cache/, seeded from the
caches of chronos_study.py and chronos_vol_corr_drift.py. A forecast is computed at most once, whichever
method, fold or command asks for it, and interrupted runs resume. Written with an AI coding assistant (Claude).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chronos_study as CS  # noqa: E402  (also puts experiments/ on sys.path)
import vix_lib as V  # noqa: E402  master's VIX helpers, copied unchanged into experiments/
from common import PERIODS, metrics, paired_bootstrap  # noqa: E402

OUT = HERE / "results_vix"
CACHE = OUT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
VAL_LOG = OUT / "vix_validation_log.csv"
CHOICES = OUT / "vix_choices.json"
PREREG = HERE / "chronos_vix_preregistration.md"
TEST_DONE = OUT / "vix_test_comparisons.csv"

HORIZONS = [5, 1]
FOLDS = ["F1", "F2", "VAL"]
M1_GRID = [{"cut": c, "min_ctx": m} for c in (0.8, 0.9) for m in (63, 126)]
M2_GRID = [{"split": s, "lookback": L} for s in ("median", "tercile") for L in (504, None)]
M3_CAPS = [1.5, 2.0]
MIN_GAP = 21        # master: minimum spacing between triggered refits
SCHED_EVERY = 63    # master: scheduled-refit control
COST = 5            # master: bps per unit turnover
EXPERTS = {"always_up": None, "uni_256": ("uni", 256), "uni_2048": ("uni", 2048),
           "joint_512": ("joint", 512), "core_256": ("core", 256)}
MASTER_M3 = {"signal": "rv20", "cap": 2.0}  # master's frozen M3 choice (experiments/results/vix_choices.json)
VOL_START = "2008-01-01"  # first day with a Chronos variance forecast (the project's study start)
CHUNK = 1024              # forecasts per cache save

_PIPE = None
PLAN: dict | None = None  # set by `plan`/`smoke`: record cache misses instead of forecasting


def pipeline():
    global _PIPE
    if _PIPE is None:
        _PIPE = CS.load_pipeline()
    return _PIPE


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Forecast cache
# ---------------------------------------------------------------------------
class ForecastCache:
    """Chronos-2 output for (calendar position t, effective context length) of one input stream.

    The effective length is what actually reaches the model (a context is cut at the stream's first valid row),
    so two requests that produce the same input share one entry. Saved to results_vix/cache/ every CHUNK forecasts.
    """

    def __init__(self, name, first, make_input, compute, seeds=()):
        self.name, self.first = name, int(first)
        self.make_input, self.compute = make_input, compute
        self.file = CACHE / f"{name}.pkl"
        self.d = pd.read_pickle(self.file) if self.file.exists() else {}
        n0 = len(self.d)
        for t, L, vals in seeds:
            t = np.asarray(t, int)
            assert (t >= 0).all(), f"{name}: a seeded date is not on the calendar"
            for ti, vi in zip(t, vals):
                self.d.setdefault((int(ti), self.leff(int(ti), L)), float(vi))
        self.n_seeded = len(self.d) - n0

    def leff(self, t: int, L: int) -> int:
        """Context length that reaches the model: the window [t-L+1, t] cut at the first valid row."""
        return int(t + 1 - max(t - L + 1, self.first))

    def get(self, t, L) -> np.ndarray:
        t = np.asarray(t, int)
        L = np.broadcast_to(np.asarray(L, int), t.shape)
        assert (t >= self.first).all(), f"{self.name}: decision day before the first valid row"
        assert (L >= 1).all()
        keys = [(int(a), self.leff(int(a), int(b))) for a, b in zip(t, L)]
        miss = sorted({k for k in keys if k not in self.d})
        if miss:
            if PLAN is not None:
                PLAN.setdefault(self.name, set()).update(miss)
                return np.full(len(keys), np.nan)
            self._fill(miss)
        return np.array([self.d[k] for k in keys])

    def _fill(self, miss):
        print(f"  {self.name}: {len(miss)} new forecasts", flush=True)
        t0 = time.time()
        for i in range(0, len(miss), CHUNK):
            part = miss[i:i + CHUNK]
            # the context of key (t, L) is [t-L+1, t]: it ends at the decision day's close
            vals = self.compute([self.make_input(t, L) for t, L in part])
            self.d.update({k: float(v) for k, v in zip(part, vals)})
            pd.to_pickle(self.d, self.file)
            done = i + len(part)
            print(f"    {done}/{len(miss)}  {(time.time() - t0) / done:.2f}s per forecast", flush=True)


def direction_cache(cd: CS.ChronosData, variant: str) -> ForecastCache:
    """P(up) cache for one horizon and input variant, seeded from chronos_study.py's cached predictions."""
    first = max(cd.first, cd.first_core) if variant == "core" else cd.first  # as CS.ChronosData.make_input
    seeds = []
    for f in sorted(CS.OUT.glob(f"pred_h{cd.h}_{variant}_L*_*_s1.pkl")):
        L = int(f.stem.split("_")[3][1:])
        d = pd.read_pickle(f)
        seeds.append((cd.cal.get_indexer(d.index), L, d["proba"].to_numpy()))
    return ForecastCache(f"dir_h{cd.h}_{variant}", first,
                         lambda t, L: cd.make_input(t, L, variant),
                         lambda inp: CS.prob_up(CS.spy_quantiles(pipeline(), inp, cd.h)), seeds)


def vol_cache():
    """Next-week SPY log-vol forecast cache for the frozen spy_vol pick, seeded from chronos_vol_corr_drift.py."""
    import chronos_vol_corr_drift as CE
    assert sha256(CE.SHORTLIST) == CE.HASHFILE.read_text().strip(), "ext_shortlist.json changed after it was frozen"
    sel = json.loads(CE.SHORTLIST.read_text())["selected"]["spy_vol"]
    ed = CE.build_all()["spy_vol"]
    cal = pd.read_pickle(CS.DATA_DIR / "aligned_prices.pkl")["SPY"].index
    seeds = []
    for f in sorted(CE.OUT.glob(f"pred_spy_vol_{sel['variant']}_L{sel['L']}_*.pkl")):
        d = pd.read_pickle(f)  # 'pred' is the raw step-5 forecast mean (v2_lib._clip is the identity for vol)
        seeds.append((cal.get_indexer(d.index), sel["L"], d["pred"].to_numpy()))
    fc = ForecastCache(f"vol_spy_{sel['variant']}_L{sel['L']}", ed.first,
                       lambda t, L: ed.make_input(t, L, sel["variant"]),
                       lambda inp: CE.forecast(pipeline(), inp, ed.h)[0], seeds)
    return fc, sel, cal


# ---------------------------------------------------------------------------
# Direction: base model, M1, M2
# ---------------------------------------------------------------------------
def base_spec() -> dict:
    assert sha256(CS.SHORTLIST) == CS.HASHFILE.read_text().strip(), "chronos_shortlist.json changed after freezing"
    sel = json.loads(CS.SHORTLIST.read_text())["selected"]
    return {int(h): {"variant": s["variant"], "L": int(s["L"])} for h, s in sel.items()}


class Horizon:
    def __init__(self, h: int, base: dict):
        self.h, self.base = h, base
        self.cd = CS.ChronosData(h)
        self.ds = self.cd.ds
        self.caches: dict[str, ForecastCache] = {}

    def cache(self, variant: str) -> ForecastCache:
        if variant not in self.caches:
            self.caches[variant] = direction_cache(self.cd, variant)
        return self.caches[variant]

    def proba(self, rows, variant, L) -> np.ndarray:
        return self.cache(variant).get(self.cd.cal_pos[rows], L)

    def thr(self, s: int) -> float:
        """chronos_study.py's threshold rule: base rate of the labels known at the period start."""
        return float(self.ds.y[:self.ds.n_known(s)].mean())


def fixed_chronos(H: Horizon, s: int, e: int) -> pd.DataFrame:
    p = H.proba(np.arange(s, e), H.base["variant"], H.base["L"])
    return V._frame(H.ds, s, e, p, np.full(e - s, H.thr(s)))


def reset_rows(ds, s, e, cut=None, every=None) -> np.ndarray:
    """Rows where the context restarts: VIX regime flips (master's M1 trigger) or every `every` rows (control)."""
    hi = V.vix_rank(ds) >= cut if every is None else None
    out, last = [], s
    for t in range(s + 1, e):
        fire = (t - last >= every) if every is not None else (hi[t] != hi[t - 1] and t - last >= MIN_GAP)
        if fire:
            out.append(t)
            last = t
    return np.asarray(out, dtype=int)


def context_lengths(H: Horizon, rows, resets, min_ctx) -> np.ndarray:
    """Base context length, except after a reset: min_ctx sessions on the reset day, +1 per session, capped at L."""
    L = H.base["L"]
    tcal = H.cd.cal_pos[rows]
    out = np.full(len(rows), L)
    k = np.searchsorted(resets, rows, side="right") - 1  # latest reset at or before each row
    has = k >= 0
    out[has] = np.minimum(L, tcal[has] - H.cd.cal_pos[resets[k[has]]] + min_ctx)
    assert (out >= min(min_ctx, L)).all()
    return out


def vix_reset(H: Horizon, s, e, min_ctx, cut=None, every=None) -> pd.DataFrame:
    rows = np.arange(s, e)
    resets = reset_rows(H.ds, s, e, cut, every)
    Lr = context_lengths(H, rows, resets, min_ctx)
    p = H.proba(rows, H.base["variant"], Lr)
    return V._frame(H.ds, s, e, p, np.full(e - s, H.thr(s)),
                    {"n_fires": int(len(resets)), "mean_ctx": float(Lr.mean()),
                     "share_shortened": float((Lr < H.base["L"]).mean())})


def expert_frames(H: Horizon, until: int | None = None) -> dict:
    """Out-of-sample expert forecasts from V.EXPERT_START to the end of the data, in master's frame format.

    Rows at or after `until` are left NaN, so `validate` never forecasts test days. Every expert's threshold is
    the base rate of labels known at that row (master's always-up rule), since Chronos experts have no training set.
    """
    ds = H.ds
    s0 = ds.pos(V.EXPERT_START)
    rows = np.arange(s0, ds.n)
    nk = np.searchsorted(ds.label_end, ds.dates[rows], side="right")
    rate = np.cumsum(ds.y)[nk - 1] / nk
    live = rows < (ds.n if until is None else until)
    E = {}
    for k, spec in EXPERTS.items():
        if spec is None:
            p = rate
        else:
            p = np.full(len(rows), np.nan)
            p[live] = H.proba(rows[live], *spec)
        E[k] = V._frame(ds, s0, ds.n, p, rate)
    return E


def direction_requests(H: Horizon, periods, m1_cfgs, sched_min_ctx=(), experts_until=None) -> dict:
    """Every (calendar position, context length) a phase needs, per variant, so each cache fills in one batch."""
    req: dict[str, list] = {}

    def add(v, rows, L):
        req.setdefault(v, []).append((H.cd.cal_pos[rows], np.broadcast_to(np.asarray(L, int), rows.shape)))

    bv = H.base["variant"]
    for per in periods:
        s, e = H.ds.period_positions(per)
        rows = np.arange(s, e)
        add(bv, rows, H.base["L"])
        for c in m1_cfgs:
            add(bv, rows, context_lengths(H, rows, reset_rows(H.ds, s, e, cut=c["cut"]), c["min_ctx"]))
        for m in sched_min_ctx:
            add(bv, rows, context_lengths(H, rows, reset_rows(H.ds, s, e, every=SCHED_EVERY), m))
    if experts_until is not None:
        rows = np.arange(H.ds.pos(V.EXPERT_START), experts_until)
        for spec in EXPERTS.values():
            if spec is not None:
                add(spec[0], rows, spec[1])
    return {v: (np.concatenate([a for a, _ in x]), np.concatenate([b for _, b in x])) for v, x in req.items()}


def prefetch(H: Horizon, req: dict):
    for v, (t, L) in req.items():
        H.cache(v).get(t, L)


# ---------------------------------------------------------------------------
# M3: volatility-managed exposure with a Chronos-2 variance forecast
# ---------------------------------------------------------------------------
def chronos_frame(fc, sel, cal, D: pd.DataFrame, until=None) -> pd.DataFrame:
    """master's spy_daily frame from VOL_START, plus v_chronos = forecast daily variance (closes <= t).

    The forecast is y = log annualised RMS return over t+1..t+5, so the daily variance is exp(2y) / 252.
    A constant bias (e.g. the Jensen gap of exp(2 E[log vol])) cancels: master's scale_constant rescales any
    signal to buy-and-hold volatility on prior data.
    """
    Dc = D.loc[VOL_START:until].copy()
    t = cal.get_indexer(Dc.index)
    Dc["v_chronos"] = np.exp(2 * fc.get(t, sel["L"])) / 252.0
    return Dc.dropna()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_validate(args):
    base = base_spec()
    rows = []
    for h in HORIZONS:
        H = Horizon(h, base[h])
        val_end = H.ds.period_positions("VAL")[1]
        prefetch(H, direction_requests(H, FOLDS, M1_GRID, experts_until=val_end))
        E = expert_frames(H, until=val_end)
        for f in FOLDS:
            s, e = H.ds.period_positions(f)
            rows.append({"horizon": h, "method": "ref_fixed", "config": json.dumps(H.base), "fold": f,
                         **metrics(fixed_chronos(H, s, e))})
            for c in M1_GRID:
                p = vix_reset(H, s, e, c["min_ctx"], cut=c["cut"])
                rows.append({"horizon": h, "method": "M1", "config": json.dumps(c), "fold": f,
                             **metrics(p), **p.attrs})
            for c in M2_GRID:
                p = V.regime_select(H.ds, E, s, e, c["split"], c["lookback"])
                rows.append({"horizon": h, "method": "M2", "config": json.dumps(c), "fold": f,
                             **metrics(p), "chosen": json.dumps(p.attrs["chosen"])})
        print(f"h={h} done", flush=True)
    D = V.spy_daily()
    fc, sel, cal = vol_cache()
    Dc = chronos_frame(fc, sel, cal, D, until=PERIODS["VAL"][1])
    for f in FOLDS:
        a, b = PERIODS[f]
        for cap in M3_CAPS:
            r = V.vol_managed(Dc, a, b, "chronos", cap, COST)
            rows.append({"horizon": 0, "method": "M3", "config": json.dumps({"signal": "chronos", "cap": cap}),
                         "fold": f, "sharpe": V.sharpe(r["managed"]), "sharpe_bh": V.sharpe(r["bh"]),
                         "mean_w": r["w"].mean()})
        r = V.vol_managed(D, a, b, MASTER_M3["signal"], MASTER_M3["cap"], COST)  # reference, for reading only
        rows.append({"horizon": 0, "method": "ref_M3_master", "config": json.dumps(MASTER_M3), "fold": f,
                     "sharpe": V.sharpe(r["managed"]), "sharpe_bh": V.sharpe(r["bh"]), "mean_w": r["w"].mean()})
    log = pd.DataFrame(rows)
    log.to_csv(VAL_LOG, index=False)
    with pd.option_context("display.width", 250):
        d = log[log.horizon > 0]
        print(d.pivot_table(index=["horizon", "method", "config"], columns="fold", values="roc_auc").round(4))
        print(log[log.horizon == 0].pivot_table(index=["method", "config"], columns="fold", values="sharpe").round(3))


def cmd_select(args):
    if TEST_DONE.exists():
        sys.exit("The test has already been run; re-selecting would overwrite the pre-registration.")
    log = pd.read_csv(VAL_LOG)
    ch = {}
    for h in HORIZONS:
        g = log[log.horizon == h]
        m1 = g[g.method == "M1"].groupby("config")["roc_auc"].mean()
        m2 = g[g.method == "M2"].groupby("config")["roc_auc"].mean()
        ref = g[g.method == "ref_fixed"]["roc_auc"].mean()
        ch[str(h)] = {"M1": json.loads(m1.idxmax()), "M2": json.loads(m2.idxmax()),
                      "_validation": {"M1_mean_auc": m1.round(4).to_dict(), "M2_mean_auc": m2.round(4).to_dict(),
                                      "fixed_chronos_mean_auc": round(float(ref), 4)}}
    m3 = log[log.method == "M3"].groupby("config")["sharpe"].mean()
    ref3 = log[log.method == "ref_M3_master"]["sharpe"].mean()
    ch["M3"] = {**json.loads(m3.idxmax()), "_validation_mean_sharpe": m3.round(3).to_dict(),
                "_master_rv20_cap2_mean_sharpe": round(float(ref3), 3)}
    frozen = {"chronos_experiment/results/chronos_shortlist.json": sha256(CS.SHORTLIST),
              "chronos_experiment/results_ext/ext_shortlist.json": sha256(HERE / "results_ext" / "ext_shortlist.json"),
              "experiments/vix_lib.py": sha256(Path(V.__file__))}
    ch["_frozen_inputs"] = frozen
    CHOICES.write_text(json.dumps(ch, indent=2, sort_keys=True))
    digest = sha256(CHOICES)
    lines = ["# Chronos-2 VIX study pre-registration", "",
             f"Choices file `chronos_experiment/results_vix/vix_choices.json`  SHA-256: `{digest}`", "",
             "Choices made on folds 2013-14, 2015-16, 2017-18 only (`results_vix/vix_validation_log.csv`).",
             "Selection rules (master's): M1 and M2 by mean fold AUC; M3 by mean fold Sharpe at "
             f"{COST} bps.", "", "Frozen inputs (SHA-256):"]
    lines += [f"- `{k}`: `{v}`" for k, v in frozen.items()]
    lines.append("")
    for h in ["5", "1"]:
        lines.append(f"- h={h}: M1 = {ch[h]['M1']}, M2 = {ch[h]['M2']}")
    lines.append(f"- M3 = Chronos-2 variance forecast, cap {ch['M3']['cap']}, {COST} bps per unit turnover")
    lines += ["", "Test comparisons (paired block bootstrap; Holm within horizon and method over BA/AUC/Brier):",
              "- M1 vs the fixed Chronos-2 base (primary); M1 vs the same context reset every 63 sessions "
              "(isolates the VIX trigger).",
              "- M2 vs the fixed Chronos-2 base (primary); M2 vs regime-blind selection with the same lookback "
              "(isolates the regime); M2 vs always-up.",
              f"- M3: Sharpe of the Chronos-managed strategy minus buy-and-hold at {COST} bps (primary) and 0 bps; "
              f"minus master's rv20-managed strategy (cap 2) at {COST} and 0 bps (secondary: does a better variance "
              "forecast pay?).",
              "Caveat: 2019-2025 has been evaluated in earlier studies of this project, including master's VIX study."]
    PREREG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def cmd_test(args):
    want = re.search(r"SHA-256: `([0-9a-f]{64})`", PREREG.read_text(encoding="utf-8")).group(1)
    assert sha256(CHOICES) == want, "choices changed after pre-registration"
    if TEST_DONE.exists() and not args.force:
        sys.exit("The test has already been run once. Use --force only if it crashed before finishing.")
    ch = json.loads(CHOICES.read_text())
    for k, v in ch["_frozen_inputs"].items():
        assert sha256(CS.ROOT / k) == v, f"{k} changed after pre-registration"
    base = base_spec()
    levels, comps, years, info, saved = [], [], [], {}, {}
    for h in HORIZONS:
        H = Horizon(h, base[h])
        ds = H.ds
        s, e = ds.period_positions("TEST")
        c1, c2 = ch[str(h)]["M1"], ch[str(h)]["M2"]
        prefetch(H, direction_requests(H, ["TEST"], [c1], sched_min_ctx=[c1["min_ctx"]], experts_until=ds.n))
        E = expert_frames(H)
        P = {"fixed": fixed_chronos(H, s, e), "always_up": V.always_up(ds, s, e),
             "M1": vix_reset(H, s, e, c1["min_ctx"], cut=c1["cut"]),
             "scheduled": vix_reset(H, s, e, c1["min_ctx"], every=SCHED_EVERY),
             "M2": V.regime_select(ds, E, s, e, c2["split"], c2["lookback"]),
             "M2_blind": V.regime_select(ds, E, s, e, c2["split"], c2["lookback"], use_regime=False)}
        ref = OUT.parent / "results" / "chronos_test_levels.csv"  # regression check vs chronos_study.py's test
        if ref.exists():
            r0 = pd.read_csv(ref).query("h == @h and model == 'chronos'")
            m0 = metrics(P["fixed"])
            if len(r0) and int(r0.n.iloc[0]) == m0["n"]:
                ok = abs(float(r0.roc_auc.iloc[0]) - m0["roc_auc"]) < 1e-6
                print(f"h={h}: fixed Chronos reproduces chronos_study.py's test AUC: {ok}")
        info[h] = {"M1": P["M1"].attrs, "scheduled": P["scheduled"].attrs,
                   "M2_chosen": P["M2"].attrs["chosen"], "M2_blind_chosen": P["M2_blind"].attrs["chosen"]}
        saved[h] = {k: p[["y", "proba", "thr", "pred"]] for k, p in P.items()}
        for k, p in P.items():
            levels.append({"horizon": h, "model": k, **metrics(p)})
            for yr, g in p.groupby(p.index.year):
                years.append({"horizon": h, "model": k, "year": yr, **metrics(g)})
        for fam, a, b in [("M1", "M1", "fixed"), ("M1", "M1", "scheduled"),
                          ("M2", "M2", "fixed"), ("M2", "M2", "M2_blind"), ("M2", "M2", "always_up")]:
            comps.append({"horizon": h, "family": fam, "comparison": f"{a} vs {b}", **paired_bootstrap(P[a], P[b], h)})
    comp = pd.DataFrame(comps)
    for _, g in comp.groupby(["horizon", "family"]):
        adj = CS.holm(np.concatenate([g["p_balanced_acc"], g["p_roc_auc"], g["p_brier"]]))
        n = len(g)
        comp.loc[g.index, "holm_balanced_acc"] = adj[:n]
        comp.loc[g.index, "holm_roc_auc"] = adj[n:2 * n]
        comp.loc[g.index, "holm_brier"] = adj[2 * n:]

    D = V.spy_daily()
    fc, sel, cal = vol_cache()
    Dc = chronos_frame(fc, sel, cal, D)
    a, b = PERIODS["TEST"]
    cap = ch["M3"]["cap"]
    srows = []
    for cost in (COST, 0):
        rc = V.vol_managed(Dc, a, b, "chronos", cap, cost)
        rm = V.vol_managed(D, a, b, MASTER_M3["signal"], MASTER_M3["cap"], cost)
        idx = rc.index.intersection(rm.index)
        assert len(idx) == len(rc) == len(rm), "Chronos and master strategies cover different test days"
        tag = {"cost_bps": cost, "cap": cap}
        for name, r, w in [("managed_chronos", rc["managed"], rc["w"]), ("managed_rv20_master", rm["managed"], rm["w"]),
                           ("bh", rc["bh"], None)]:
            srows.append({**tag, "strategy": name, **V.strat_stats(r),
                          "mean_weight": float(w.mean()) if w is not None else 1.0})
        srows.append({**tag, "strategy": "chronos - bh", **V.sharpe_bootstrap(rc["managed"], rc["bh"])})
        srows.append({**tag, "strategy": "chronos - rv20_master", **V.sharpe_bootstrap(rc["managed"], rm["managed"])})
        saved[f"M3_{cost}bps"] = pd.DataFrame({"chronos": rc["managed"], "rv20_master": rm["managed"], "bh": rc["bh"],
                                               "w_chronos": rc["w"], "w_rv20": rm["w"]})
    S = pd.DataFrame(srows)

    pd.DataFrame(levels).to_csv(OUT / "vix_test_levels.csv", index=False)
    pd.DataFrame(years).to_csv(OUT / "vix_test_by_year.csv", index=False)
    S.to_csv(OUT / "vix_test_strategy.csv", index=False)
    (OUT / "vix_test_info.json").write_text(json.dumps(info, indent=2, default=str))
    pd.to_pickle(saved, OUT / "vix_test_predictions.pkl")
    comp.to_csv(TEST_DONE, index=False)  # written last: its presence marks a completed test
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
        print(pd.DataFrame(levels).to_string())
        print(comp[["horizon", "comparison", "d_balanced_acc", "d_balanced_acc_lo", "d_balanced_acc_hi", "d_roc_auc",
                    "d_roc_auc_lo", "d_roc_auc_hi", "d_brier", "holm_balanced_acc", "holm_roc_auc",
                    "holm_brier"]].to_string())
        print(S.to_string())
        print(json.dumps(info, indent=2, default=str))


def count_requests() -> dict:
    """Forecasts still missing from the caches, for `validate` and then `test` (test counted for every M1 config)."""
    global PLAN
    PLAN = {}
    base = base_spec()
    Hs = {h: Horizon(h, base[h]) for h in HORIZONS}
    fc, sel, cal = vol_cache()
    D = V.spy_daily()
    try:
        for h, H in Hs.items():
            prefetch(H, direction_requests(H, FOLDS, M1_GRID, experts_until=H.ds.period_positions("VAL")[1]))
        chronos_frame(fc, sel, cal, D, until=PERIODS["VAL"][1])
        val = {k: len(v) for k, v in PLAN.items()}
        for h, H in Hs.items():
            prefetch(H, direction_requests(H, ["TEST"], M1_GRID, sched_min_ctx=sorted({c["min_ctx"] for c in M1_GRID}),
                                           experts_until=H.ds.n))
        chronos_frame(fc, sel, cal, D)
        tot = {k: len(v) for k, v in PLAN.items()}
    finally:
        PLAN = None
    seeded = {c.name: c.n_seeded for H in Hs.values() for c in H.caches.values()}
    seeded[fc.name] = fc.n_seeded
    return {"validate": val, "test_upper_bound": {k: tot[k] - val.get(k, 0) for k in tot}, "seeded": seeded}


def cmd_plan(args):
    r = count_requests()
    print("entries reused from chronos_study.py / chronos_vol_corr_drift.py caches:", r["seeded"])
    for phase in ("validate", "test_upper_bound"):
        print(f"{phase}: {sum(r[phase].values())} forecasts to compute  {r[phase]}")


def cmd_smoke(args):
    base = base_spec()
    sec = {}
    for h in HORIZONS:
        H = Horizon(h, base[h])
        c = H.cache(H.base["variant"])
        keys = [k for k in list(c.d)[:8]]
        t0 = time.time()
        new = c.compute([c.make_input(t, L) for t, L in keys])
        sec[f"dir_h{h}"] = (time.time() - t0) / len(keys)
        diff = np.max(np.abs(np.asarray(new) - np.array([c.d[k] for k in keys])))
        print(f"h={h} {H.base}: recomputed 8 cached forecasts, max |diff| {diff:.2e}, "
              f"{sec[f'dir_h{h}']:.2f}s per forecast")
        assert diff < 1e-3, "cached chronos_study.py forecasts do not match a fresh run (model or input changed?)"
    fc, sel, cal = vol_cache()
    keys = list(fc.d)[:8]
    t0 = time.time()
    new = fc.compute([fc.make_input(t, L) for t, L in keys])
    sec["vol"] = (time.time() - t0) / len(keys)
    diff = np.max(np.abs(np.asarray(new) - np.array([fc.d[k] for k in keys])))
    print(f"spy_vol {sel}: recomputed 8 cached forecasts, max |diff| {diff:.2e}, {sec['vol']:.2f}s per forecast")
    assert diff < 1e-3, "cached chronos_vol_corr_drift.py forecasts do not match a fresh run"
    r = count_requests()
    for phase in ("validate", "test_upper_bound"):
        est = sum(n * (sec["vol"] if k.startswith("vol") else sec[k.rsplit("_", 1)[0]]) for k, n in r[phase].items())
        print(f"{phase}: {sum(r[phase].values())} new forecasts, about {est / 60:.0f} min  {r[phase]}")
    print("Smoke test passed.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=None, help="GPU batch size (halves itself on out-of-memory)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "smoke", "validate", "select"):
        sub.add_parser(name)
    sub.add_parser("test").add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.batch:
        CS.BATCH = args.batch
    {"plan": cmd_plan, "smoke": cmd_smoke, "validate": cmd_validate, "select": cmd_select,
     "test": cmd_test}[args.cmd](args)


if __name__ == "__main__":
    main()

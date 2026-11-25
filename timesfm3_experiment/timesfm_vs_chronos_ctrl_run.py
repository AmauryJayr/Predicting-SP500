"""Controlled-swap experiment, forecasting side (TimesFM env). Design: timesfm_vs_chronos_ctrl_design.md (locked).

Chronos-2 is not in this env: `prep` writes its input files, `timesfm_vs_chronos_ctrl_chronos.py` (chronos2 venv) forecasts them.

    python timesfm_vs_chronos_ctrl_run.py prep                      # Chronos input files + the reproducibility inputs
    python timesfm_vs_chronos_ctrl_run.py tf CELL PART [--limit N]  # CELL in L, G1, G5; PART in TEST, PRE
    python timesfm_vs_chronos_ctrl_run.py repro_tf                  # re-forecast 30 cached TimesFM `price` dates
    python timesfm_vs_chronos_ctrl_run.py repro_ch                  # compare Chronos re-forecasts with the cached ones / pred_len 1 vs 5
    python timesfm_vs_chronos_ctrl_run.py stamp                     # record the UTC time of the first TEST forecast (verifies the design sha)

Caches (git-ignored): results/timesfm_ctrl_cache/{tf_<cell>_<part>,ch_<cell>_<part>}.npz, dates + quantiles.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
import numpy as np
import pandas as pd

import timesfm_lib as tl
from tf_paths import RES_DIR, TF_DIR
from common import DATA_DIR, PERIODS, Dataset

CTX = 256
CACHE = RES_DIR / "timesfm_ctrl_cache"
CACHE.mkdir(parents=True, exist_ok=True)
LOCK = RES_DIR / "timesfm_vs_chronos_ctrl_lock.json"
DESIGN = TF_DIR / "timesfm_vs_chronos_ctrl_design.md"
CHR_RES = os.environ.get("CHRONOS_RESULTS", "/tmp/claude-1000/-home-tong-nong-Documents-Predicting-SP500/"
                         "7e6e832c-eb35-4db3-a775-fd9250864606/scratchpad/chronos_branch/chronos_experiment/results")
PRE = ["F1", "F2", "VAL"]


def stamp(part: str = "TEST"):
    lk = json.loads(LOCK.read_text())
    assert hashlib.sha256(DESIGN.read_bytes()).hexdigest() == lk["design_sha256"], "design file changed after the lock"
    if part == "TEST" and lk["first_new_test_forecast_utc"] is None:
        lk["first_new_test_forecast_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        LOCK.write_text(json.dumps(lk, indent=1))
    return lk


def cal_panel():
    p = tl.panel()
    px = pd.read_pickle(DATA_DIR / "aligned_prices.pkl")
    return p, px


def g_series(px, h: int) -> np.ndarray:
    """g_h[t] = log(Close[t] / Open[t-h+1]) exactly as chronos_study.ChronosData (ffill, then 0 for the NaN head)."""
    g = np.log(px["SPY"]["Close"] / px["SPY"]["Open"].shift(h - 1))
    return g.ffill().fillna(0.0).to_numpy(float)


def part_dates(cell: str, part: str) -> pd.DatetimeIndex:
    """Dataset(h) rows of the part (G cells: that horizon; L cells: the union of both horizons)."""
    hs = (1, 5) if cell in ("L", "Lcov") else (int(cell[1:]),)
    pers = ["TEST"] if part == "TEST" else PRE
    out = None
    for h in hs:
        ds = Dataset(h)
        for per in pers:
            s, e = ds.period_positions(per)
            ix = ds.index[s:e]
            out = ix if out is None else out.union(ix)
    return pd.DatetimeIndex(out).sort_values()


def level_panel(p, cell):
    """Fake `timesfm_lib` panel whose 'spy' field is the series the cell forecasts (so tl.forecast is used unchanged)."""
    return p


def windows(series: np.ndarray, cal: pd.DatetimeIndex, dates, ctx: int = CTX, cov: np.ndarray | None = None):
    pos = cal.get_indexer(pd.DatetimeIndex(dates))
    assert (pos >= ctx).all()
    x = np.stack([series[q + 1 - ctx:q + 1] for q in pos])
    c = None if cov is None else np.stack([cov[:, q + 1 - ctx:q + 1] for q in pos])
    return x, c


# ---------------------------------------------------------------------------------------------------------
def prep():
    p, px = cal_panel()
    cal = p["dates"]
    def save(name, dates, x, pred_len, cov=None):
        kw = {"covnames": np.array([f"c{j}" for j in range(cov.shape[1])])} if cov is not None else {}
        if cov is not None:
            kw["cov"] = cov
        np.savez(CACHE / f"in_{name}.npz", dates=dates.to_numpy(), x=x, pred_len=pred_len, **kw)
        print(f"in_{name}: {len(dates)} dates, x {x.shape}" + (f", cov {cov.shape}" if cov is not None else ""))
    for part in ("TEST", "PRE"):
        d = part_dates("L", part)
        x, _ = windows(p["spy"], cal, d)
        save(f"ch_L_{part}", d, x, 5)
        x, c = windows(p["spy"], cal, d, cov=p["cov"])
        save(f"ch_Lcov_{part}", d, x, 5, c)
    # reproducibility inputs: 30 cached Chronos `uni` L256 TEST dates (h-specific G series, prediction_length = h)
    rng = np.random.default_rng(0)
    for h in (1, 5):
        c = pd.read_pickle(f"{CHR_RES}/pred_h{h}_uni_L256_TEST_s1.pkl")
        d = c.index[np.sort(rng.choice(len(c), 30, replace=False))]
        x, _ = windows(g_series(px, h), cal, d)
        save(f"repro_ch_uni_h{h}", d, x, h)
    d = pd.read_pickle(f"{CHR_RES}/pred_h1_uni_L256_TEST_s1.pkl").index[::60][:30]
    x, _ = windows(p["spy"], cal, d)
    save("repro_ch_L_pl1", d, x, 1)                       # level input, prediction_length 1 vs the 5-step run


def run_tf(cell: str, part: str, limit: int = 0, threads: int = 8, chunk: int = 128):
    if part == "TEST":
        stamp("TEST")
    else:
        stamp("PRE")
    p, px = cal_panel()
    q_path = CACHE / f"tf_{cell}_{part}.npz"
    pp = dict(p)
    if cell != "L":
        pp = {"dates": p["dates"], "spy": g_series(px, int(cell[1:])), "cov": None}   # G_h as the target series
    dates = part_dates(cell, part)
    have, q, used = (pd.DatetimeIndex([]), np.zeros((0, 5, 9)), np.zeros(0, int))
    if q_path.exists():
        z = np.load(q_path)
        have, q, used = pd.DatetimeIndex(z["dates"]), z["q"], z["ctx_used"]
    todo = dates.difference(have)
    if limit:
        todo = todo[:limit]
    print(f"tf {cell} {part}: {len(dates)} dates, {len(have)} done, {len(todo)} to do", flush=True)
    m = tl.load_model(threads=threads)
    t_all = time.time()
    for i in range(0, len(todo), chunk):
        part_ix = todo[i:i + chunk]
        t0 = time.time()
        qc = tl.forecast(m, "price", CTX, part_ix, pp)         # timesfm_lib.forecast unchanged; `price` = target series of `pp`
        uc = np.array([len(c[0]) for c in tl.contexts("price", CTX, part_ix, pp)], dtype=int)
        assert (uc == CTX).all()
        have, q, used = have.append(part_ix), np.concatenate([q, qc]), np.concatenate([used, uc])
        o = np.argsort(have.to_numpy())
        np.savez_compressed(str(q_path) + ".tmp.npz", dates=have.to_numpy()[o], q=q[o], ctx_used=used[o])
        os.replace(str(q_path) + ".tmp.npz", q_path)
        print(f"  {i + len(part_ix)}/{len(todo)} new, {(time.time() - t0) / len(part_ix):.3f}s/forecast, "
              f"elapsed {time.time() - t_all:.0f}s", flush=True)


def repro_tf(threads: int = 8):
    p, px = cal_panel()
    have, q, _ = tl.read_cache("price", CTX)
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(len(have), 30, replace=False))
    m = tl.load_model(threads=threads)
    t0 = time.time()
    new = tl.forecast(m, "price", CTX, have[sel], p)
    d = np.abs(new - q[sel])
    res = {"n": 30, "dates": [str(x.date()) for x in have[sel][[0, -1]]], "max_abs_diff_quantiles": float(d.max()),
           "mean_abs_diff": float(d.mean()), "sec_per_forecast": (time.time() - t0) / 30}
    # the same call through the G-series path must also be a pure function of the context (batch-composition check)
    new2 = tl.forecast(m, "price", CTX, have[sel[:10]], p)
    res["max_abs_diff_batch10_vs_cache"] = float(np.abs(new2 - q[sel[:10]]).max())
    (RES_DIR / "timesfm_vs_chronos_ctrl_repro_tf.json").write_text(json.dumps(res, indent=1))
    print(res)


def repro_ch():
    res = {}
    for h in (1, 5):
        o = np.load(CACHE / f"out_repro_ch_uni_h{h}.npz")
        c = pd.read_pickle(f"{CHR_RES}/pred_h{h}_uni_L256_TEST_s1.pkl")
        cols = [f"q{t:.2f}" for t in [0.01, 0.05] + [round(0.10 + 0.05 * i, 2) for i in range(17)] + [0.95, 0.99]]
        ref = c.loc[pd.DatetimeIndex(o["dates"]), cols].to_numpy()
        new = o["q"][:, h - 1, :]
        sc = np.abs(ref).max()
        res[f"uni_h{h}"] = {"n": len(ref), "max_abs_diff": float(np.abs(new - ref).max()),
                           "mean_abs_diff": float(np.abs(new - ref).mean()),
                           "max_abs_diff_over_q_iqr_scale": float((np.abs(new - ref) / (ref[:, 15] - ref[:, 5]).reshape(-1, 1)).max())}
    a, b = np.load(CACHE / "out_repro_ch_L_pl1.npz"), np.load(CACHE / "ch_L_TEST.npz")
    ix = pd.DatetimeIndex(b["dates"]).get_indexer(pd.DatetimeIndex(a["dates"]))
    ok = ix >= 0
    if ok.any():
        res["L_step1_pred_len1_vs_pred_len5"] = {"n": int(ok.sum()), "max_abs_diff": float(np.abs(a["q"][ok, 0] - b["q"][ix[ok], 0]).max())}
    (RES_DIR / "timesfm_vs_chronos_ctrl_repro_ch.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("cell", nargs="?")
    ap.add_argument("part", nargs="?")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    {"prep": prep, "repro_ch": repro_ch, "stamp": lambda: print(stamp("TEST")),
     "repro_tf": lambda: repro_tf(a.threads),
     "tf": lambda: run_tf(a.cell, a.part, a.limit, a.threads)}[a.cmd]()

"""V4 of the VIX study: does the VIX as a TimesFM-3 *input* help?  Inference side (the only model calls of the study).

Re-runs the frozen, zero-shot model with exactly the official settings (context 256, default model flags, batch 16,
8 threads) but with a different covariate set, and writes the quantile forecasts to a NEW cache folder
`results/timesfm_vix_cache/<name>/` (git-ignored; the existing caches are only read):

  novix     SPY log close + the 14 covariates of `multi` without ^VIX        dates: all cached `multi` dates
  vixonly   SPY log close + ^VIX (1 covariate)                               dates: all cached `multi` dates
  uni       SPY log close alone (the `price` variant)                        dates: 2019-2025 only; 2013-2018 is
            copied from the existing `price` cache (read-only copy into the new folder)

Usage:  python timesfm_vix_ablation.py repro            exact-reproduction check of the official `multi` and `price`
        python timesfm_vix_ablation.py run <name>       resumable, written after every chunk of 128 dates
"""
import sys
import time

import numpy as np
import pandas as pd

import timesfm_lib as tl
from tf_paths import RES_DIR

CTX = 256
ROOT = RES_DIR / "timesfm_vix_cache"
BATCH, THREADS, CHUNK = 16, 8, 128                       # the setting of the official run (timesfm_study.py)
SPEC = {"novix": ("multi", slice(0, 14)), "vixonly": ("multi", slice(14, 15)), "uni": ("price", None)}
_ORIG = tl.CACHE


def folder(name):
    f = ROOT / name
    f.mkdir(parents=True, exist_ok=True)
    return f


def official_dates(first=None):
    d, _, _ = tl.read_cache("multi", CTX)
    return d if first is None else d[d >= pd.Timestamp(first)]


def panel_for(name):
    p = tl.panel()
    variant, sl = SPEC[name]
    if sl is None:
        return dict(p)
    assert p["cov"].shape[0] == 15
    return {**p, "cov": p["cov"][sl]}


def model():
    m = tl.load_model(BATCH, THREADS)
    tl.set_runtime(m, BATCH, THREADS)
    return m


def run(name):
    variant, _ = SPEC[name]
    f = folder(name)
    dates = official_dates("2019-01-01" if name == "uni" else None)
    if name == "uni":                                    # 2013-2018 from the existing cache (read-only), 2019-2025 new
        d0, q0, u0 = tl.read_cache("price", CTX)
        if not (f / f"price_ctx{CTX}.npz").exists():
            tl.CACHE = f
            tl.write_cache("price", CTX, d0, q0, u0)
    tl.CACHE = f
    m = model()
    p = panel_for(name)
    t0 = time.time()
    have, _, _ = tl.ensure(m, variant, CTX, dates, chunk=CHUNK, p=p)
    tl.CACHE = _ORIG
    print(f"{name}: {len(have)} dates cached, {time.time() - t0:.0f}s", flush=True)


def repro(n=64):
    """Official `multi` (all 15 covariates) on n dates and `price` on 16 dates vs the existing caches: must be exact."""
    rows = []
    m = model()
    rng = np.random.default_rng(0)
    for variant, k, first, last in (("multi", n, "2019-01-01", "2025-12-31"), ("price", 16, "2017-01-01", "2018-12-31")):
        d = official_dates()
        d = d[(d >= pd.Timestamp(first)) & (d <= pd.Timestamp(last))]
        pick = d[np.sort(rng.choice(len(d), k, replace=False))]
        old, _ = tl.aligned(variant, CTX, pick)
        new = tl.forecast(m, variant, CTX, pick)
        rows.append({"variant": variant, "n": k, "max_abs_diff": float(np.abs(new - old).max())})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RES_DIR / "timesfm_vix_repro.csv", index=False)


if __name__ == "__main__":
    repro() if sys.argv[1] == "repro" else run(sys.argv[2])

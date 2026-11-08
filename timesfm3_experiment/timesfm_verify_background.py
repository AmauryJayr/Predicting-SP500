"""VERIFICATION (post-test, exploratory): which part of the 256-day context triggers TimesFM-3's short-term
reversal on SPY?  Extra background conditions not covered by timesfm_verify_trigger.py.
Variant `price`, ctx 256, detrending off, 2019-2025 test dates. Nothing here touches a pre-registered file.

  python timesfm_verify_background.py        -> results/timesfm_background_extra.csv

Each condition keeps the real last 255 log returns and changes one part (L = 20 = the recent stretch,
bg = the 235 returns before it). The score is corr / slope of d5 = q50[t+5] - logC[t] on r20.
  A          real context (reference)
  B          bg -> Gaussian with the bg's mean and sd
  Bsum       bg -> Gaussian bridge: bg's sd, and exactly the bg's total move
  Grev       bg -> time-reversed
  Blow<w>    bg -> real w-day smoothed path + Gaussian noise (total sd and total move kept)
  O<n>       oldest n returns -> Gaussian bridge (rest real)
  N<n>       n returns just before the recent stretch -> Gaussian bridge (rest real)
  S32x<f>    oldest 32 returns: deviations from their mean scaled by f (tests RevIN first-patch scaling)
"""
import time

import numpy as np
import pandas as pd

import timesfm_lib as tl
from timesfm_verify_detrend import get_model, set_mode, _fc
from tf_paths import RES_DIR
from common import Dataset

CTX, L = 256, 20
CONDS = ["A", "B", "Bsum", "Grev", "Blow21", "Blow63", "O32", "O64", "O128", "N32", "S32x2", "S32x0.5"]


def smooth(a, w):
    k = np.ones(w) / w
    return np.convolve(a, k, "same") / np.convolve(np.ones_like(a), k, "same")


def build(cond, r, rng):
    """r: 255 real returns -> constructed 255 returns."""
    nb = len(r) - L
    bg, st = r[:nb], r[nb:]
    if cond == "A":
        return r

    def gb(a):
        g = rng.normal(0, a.std(), len(a))
        return g - g.mean() + a.mean()
    if cond[0] in "ON" and cond[1:].isdigit():
        n = int(cond[1:])
        out = r.copy()
        if cond[0] == "O":
            out[:n] = gb(r[:n])
        else:
            out[nb - n:nb] = gb(r[nb - n:nb])
        return out
    if cond.startswith("S32x"):
        f = float(cond[4:])
        out = r.copy()
        a = r[:32]
        out[:32] = a.mean() + f * (a - a.mean())
        return out
    if cond == "B":
        nbg = rng.normal(bg.mean(), bg.std(), nb)
    elif cond == "Bsum":
        nbg = gb(bg)
    elif cond == "Grev":
        nbg = bg[::-1]
    elif cond.startswith("Blow"):
        w = int(cond[4:])
        lo = smooth(bg, w)
        sd = np.sqrt(max(bg.var() - lo.var(), 0))
        g = rng.normal(0, sd, nb)
        nbg = lo + g - g.mean() + (bg.mean() - lo.mean())
    else:
        raise ValueError(cond)
    return np.concatenate([nbg, st])


def main():
    ds = Dataset(5)
    s, e = ds.period_positions("TEST")
    dates = ds.index[s:e]
    p = tl.panel()
    spy = p["spy"]
    pos = p["dates"].get_indexer(dates)
    r_all = np.diff(spy)
    m = get_model()
    set_mode(m, "off")
    rows = []
    for cond in CONDS:
        rng = np.random.default_rng([777, len(cond), ord(cond[-1])])
        t0 = time.time()
        xs = []
        for k in pos:
            lo = k - (CTX - 1)
            rr = build(cond, r_all[lo:k], rng)
            xs.append(np.concatenate([[spy[lo]], spy[lo] + np.cumsum(rr)]).astype(np.float32))
        q = _fc(m, xs)
        xs = np.stack(xs)
        d5 = q[:, 4, 4] - xs[:, -1]
        r20 = xs[:, -1] - xs[:, -21]
        rows.append({"cond": cond, "n": len(pos), "corr_d5_r20": np.corrcoef(d5, r20)[0, 1],
                     "slope_d5_r20": np.polyfit(r20, d5, 1)[0], "bgsum_sd": np.std(xs[:, -21] - xs[:, 0])})
        print(f"{cond:8s} corr={rows[-1]['corr_d5_r20']:+.3f} slope={rows[-1]['slope_d5_r20']:+.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(RES_DIR / "timesfm_background_extra.csv", index=False)


if __name__ == "__main__":
    main()

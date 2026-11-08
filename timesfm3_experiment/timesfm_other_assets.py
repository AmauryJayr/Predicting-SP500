"""EXPLORATORY (post-test): does zero-shot TimesFM-3 do better on assets other than SPY?
Same setup as the SPY `price` variant (target's own log close only, ctx 256, default detrending), same
2019-2025 dates and label definition (h=1: Close[t+1] > Open[t+1]; h=5: Close[t+5] > Open[t+1]).
Nothing here touches a pre-registered file.

  python timesfm_other_assets.py        -> results/timesfm_other_assets.csv

Columns: AUC with a 95% block-bootstrap CI (20-day blocks, 1000 draws), AUC per year, corr(score, r20),
and the AUC of three one-line rules (score = past 5 / 20 / 256-day log change) plus corr(r20, label).
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import timesfm_lib as tl
from timesfm_verify_detrend import get_model, set_mode, _fc
from tf_paths import RES_DIR
from common import DATA_DIR

CTX = 256
ASSETS = ["SPY", "QQQ", "GLD", "TLT", "IEF"]
rng = np.random.default_rng(0)


def block_ci(y, s, B=1000, b=20):
    n = len(y)
    k = int(np.ceil(n / b))
    out = []
    for _ in range(B):
        st = rng.integers(0, n - b, k)
        i = (st[:, None] + np.arange(b)).ravel()[:n]
        if 0 < y[i].sum() < n:
            out.append(roc_auc_score(y[i], s[i]))
    return np.percentile(out, [2.5, 97.5])


def main():
    px = pd.read_pickle(DATA_DIR / "aligned_prices.pkl")
    cal = px["SPY"].index
    dates = cal[(cal >= "2019-01-01") & (cal <= "2025-12-31")]
    dates = dates[cal.get_indexer(dates) + 5 < len(cal)]            # need Close[t+5]
    yr = dates.year
    m = get_model()
    set_mode(m, "default")
    rows = []
    for tk in ASSETS:
        t0 = time.time()
        d = px[tk]
        c = np.log(d["Close"].to_numpy(float))
        o = d["Open"].to_numpy(float)
        C = d["Close"].to_numpy(float)
        pos = cal.get_indexer(dates)
        assert np.isfinite(c[pos.min() - 256:]).all()
        q = _fc(m, [c[k - CTX + 1:k + 1].astype(np.float32) for k in pos])
        r5, r20, r256 = (c[pos] - c[pos - w] for w in (5, 20, 256))
        for h in (1, 5):
            y = (C[pos + h] > o[pos + 1]).astype(int)
            s = np.array([tl.p_up(q[i], "price", h, c[k]) for i, k in enumerate(pos)])
            lo, hi = block_ci(y, s)
            rows.append({"asset": tk, "h": h, "n": len(y), "up_rate": y.mean(), "auc": roc_auc_score(y, s),
                         "ci_lo": lo, "ci_hi": hi, "corr_score_r20": np.corrcoef(s, r20)[0, 1],
                         **{str(Y): roc_auc_score(y[yr == Y], s[yr == Y]) for Y in range(2019, 2026)},
                         "auc_rule_r5": roc_auc_score(y, r5), "auc_rule_r20": roc_auc_score(y, r20),
                         "auc_rule_r256": roc_auc_score(y, r256), "corr_r20_label": np.corrcoef(r20, y)[0, 1]})
        print(tk, f"{time.time() - t0:.0f}s", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(RES_DIR / "timesfm_other_assets.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

"""EXPLORATORY (written and run after the test, from saved predictions only; no model run).

Per test year, h=5: AUC of Timestral and of TimesFM-3 with 95% moving-block-bootstrap intervals
(20-day blocks, 1000 draws, seed 0 -- same recipe as timesfm_explore_split.py), and the correlation of
each model's h=5 score with SPY's trailing 20-session log change ("bet on a turn" signature of TimesFM:
-0.21 .. -0.68 in every year). Writes results/timestral_exploratory_years.csv.
Usage:  PYTHONDONTWRITEBYTECODE=1 python timestral_explore_years.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ts_paths import RES_DIR, TFM_RES
from common import DATA_DIR


def block_ci(y, s, rng, B=1000, b=20):
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
    tsl = pd.read_pickle(RES_DIR / "timestral_test_predictions.pkl")[("tsl.h5", "timestral")]
    tfm = pd.read_pickle(TFM_RES / "timesfm_test_predictions.pkl")[("tfm.h5", "timesfm")]
    assert tsl.index.equals(tfm.index)
    ls = np.log(pd.read_pickle(DATA_DIR / "aligned_prices.pkl")["SPY"]["Close"].astype(float))
    r20 = ls.diff(20).reindex(tsl.index)
    assert r20.notna().all()
    rows = []
    for yr in sorted(set(tsl.index.year)) + ["all"]:
        sel = np.ones(len(tsl), bool) if yr == "all" else (tsl.index.year == yr)
        row = {"year": yr, "n": int(sel.sum()), "up_rate": float(tsl["y"][sel].mean()),
               "auc_bounce20": float(roc_auc_score(tsl["y"][sel], -r20[sel]))}   # rule "bet on a turn", as in TimesFM's script
        for name, p in [("timestral", tsl), ("timesfm3", tfm)]:
            g, r = p[sel], r20[sel]
            rng = np.random.default_rng(0)
            lo, hi = block_ci(g["y"].to_numpy(), g["proba"].to_numpy(), rng)
            row.update({f"mean_proba_{name}": float(g["proba"].mean()), f"auc_{name}": roc_auc_score(g["y"], g["proba"]), f"auc_{name}_lo": lo,
                        f"auc_{name}_hi": hi, f"corr_r20_{name}": float(g["proba"].corr(r))})
        rows.append(row)
    t = pd.DataFrame(rows)
    # cross-check: TimesFM numbers recomputed here must equal the ones its own script saved
    ref = pd.read_csv(TFM_RES / "timesfm_exploratory_years.csv")
    ref = ref[ref["year"].astype(str) != "all"].assign(year=lambda d: d.year.astype(int)).set_index("year")
    y = t[t.year != "all"].assign(year=lambda d: d.year.astype(int)).set_index("year")
    assert np.allclose(y["auc_bounce20"], ref["auc_bounce20"]) and np.allclose(y["auc_timesfm3"], ref["auc_timesfm"]) and np.allclose(y["corr_r20_timesfm3"], ref["corr_r20"])
    t.to_csv(RES_DIR / "timestral_exploratory_years.csv", index=False)
    with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
        print(t.to_string(index=False))




def figure():
    """figures/timestral_years_h5.png: per-year h=5 AUC with 95% intervals, both models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ts_paths import FIG_DIR
    t = pd.read_csv(RES_DIR / "timestral_exploratory_years.csv")
    t = t[t.year != "all"]
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for k, (name, col, off) in enumerate([("Timestral (ret, ctx 256)", "timestral", -0.1),
                                          ("TimesFM-3 (multi, ctx 256)", "timesfm3", 0.1)]):
        a = t[f"auc_{col}"].to_numpy()
        ax.errorbar(x + off, a, yerr=[a - t[f"auc_{col}_lo"], t[f"auc_{col}_hi"] - a], fmt="o", capsize=3,
                    label=name, color=["#1f6f8b", "#c0562b"][k])
    ax.axhline(0.5, color="grey", lw=1, ls="--")
    ax.set_xticks(x, t.year.astype(int))
    ax.set_ylabel("h=5 AUC (95% block-bootstrap interval)")
    ax.set_title("Test years 2019-2025, h=5 (exploratory)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "timestral_years_h5.png", dpi=130)


if __name__ == "__main__":
    main()
    figure()

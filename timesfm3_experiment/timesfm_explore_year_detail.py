"""EXPLORATORY (post-test): a per-year look at TimesFM-3 on SPY, 2019-2025, for the "Each year in detail" report section.
Nothing here changes a pre-registered result. Uses saved forecasts only; no new TimesFM run.

Usage:  python timesfm_explore_year_detail.py -> results/timesfm_year_detail.csv   (one row per year x horizon, plus "all")
                                                 results/timesfm_year_phases.csv   (one row per year x phase)
                                                 figures/timesfm_year_<YYYY>.png   (one figure per year)

  official   the pre-registered tfm.h1 / tfm.h5 TEST scores (multi, ctx 256, calibrated), plus chance and fixed_lr.
             AUC 95% interval: moving-block bootstrap, 20-day blocks, 1000 draws, default_rng(0) per horizon,
             years in order (same as timesfm_explore_split.py, so the h=5 intervals are identical).
  SPY        timesfm_spy_years.csv (ret, max_fall, peak, trough, recovered, vol, big_days, up5) + the h=1 up-rate.
  phases     h=5 only. 2020 / 2022 / 2025: timesfm_explore_years.STRETCHES exactly (everything else = "other").
             Other years: fall = peak+1 day .. trough, recovery = trough+1 day .. recovered date, other = the rest.
             turn_points_up / rest_pushes_up come from the SPY-only forecast split of timesfm_explore_split.py.
  mechanism  h=5 only: official turn / rest AUC (fit of P(up) on r20); price-only AUC (detrending as shipped), its turn part and rest, and the price-only AUC with
             detrending off (results/timesfm_cache_detrend/off_price_ctx256.npz, same p_up path).
  h=1 checks AUC of the official h=1 score against three labels (open-to-close = the real one; close-to-close
             Close[t+1] > Close[t]; overnight gap Open[t+1] > Close[t]) and its correlation with SPY's past 1- and 5-day log move;
             also the AUC of those two past moves themselves as a rule (higher past move -> says up) against the real label.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

import timesfm_lib as tl
from common import DATA_DIR, RES_DIR
from timesfm_explore_years import STRETCHES, frame
from timesfm_explore_split import split, block_ci, price_frame, cols_for, SEGS
from timesfm_verify_detrend import labels

FIG_DIR = RES_DIR.parent / "figures"
YEARS = list(range(2019, 2026))


def off_price_pup(ls, ylab):
    z = np.load(RES_DIR / "timesfm_cache_detrend" / "off_price_ctx256.npz", allow_pickle=True)
    d = pd.DatetimeIndex(z["dates"])
    anc = ls.reindex(d).to_numpy()
    s = pd.Series([tl.p_up(z["q"][i], "price", 5, anc[i]) for i in range(len(d))], index=d)
    return s.loc[ylab.index]


def phases_for(yr, spy):
    """[(label, kind, start, end)] for the year; 'other' is added by the caller."""
    if yr in (2020, 2022, 2025):
        return [(lab[5:], kind, a, e) for lab, kind, a, e, _ in STRETCHES if lab.startswith(str(yr))]
    r = spy[spy.year == yr].iloc[0]
    pk, tr = pd.Timestamp(r["peak"]), pd.Timestamp(r["trough"])
    out = [("fall", "fall", pk + pd.Timedelta(days=1), tr)]
    if isinstance(r["recovered"], str) and r["recovered"]:
        out.append(("recovery", "recovery", tr + pd.Timedelta(days=1), pd.Timestamp(r["recovered"])))
    return out


def main():
    FIG_DIR.mkdir(exist_ok=True)
    px = pd.read_pickle(DATA_DIR / "aligned_prices.pkl")["SPY"]
    close, opn = px["Close"].astype(float), px["Open"].astype(float)
    ls = np.log(close)
    P = pd.read_pickle(RES_DIR / "timesfm_test_predictions.pkl")
    spy = pd.read_csv(RES_DIR / "timesfm_spy_years.csv")

    # ---- h=5 price-only split (turn / rest), as in timesfm_explore_split.py
    pr = price_frame(ls)
    pr["turn"], pr["rest"], _ = split(pr, "d5", cols_for(SEGS))
    pr["pup_off"] = off_price_pup(ls, pr["y"])

    off, _ = frame()    # official h=5 frame; turn part = fit of P(up) on r20, as in timesfm_explore_split.py
    off["turn"], off["rest"], _ = split(off, "proba", ["r20"])
    rows = []
    t5 = P[("tfm.h5", "timesfm")]
    assert (t5.index == pr.index).all()
    # ---- h=1 label checks: row t has y = Close[t+1] > Open[t+1]
    t1 = P[("tfm.h1", "timesfm")]
    nxt_o, nxt_c = opn.shift(-1).reindex(t1.index), close.shift(-1).reindex(t1.index)
    assert ((nxt_c > nxt_o).astype(int) == t1["y"]).all(), "h=1 label is not Close[t+1] > Open[t+1]"
    lab_cc = (nxt_c > close.reindex(t1.index)).astype(int)
    lab_gap = (nxt_o > close.reindex(t1.index)).astype(int)
    r1 = ls.diff(1).reindex(t1.index)
    r5 = ls.diff(5).reindex(t1.index)

    for h, rng_seed in ((1, 0), (5, 0)):
        rng = np.random.default_rng(rng_seed)
        T, C, F = (P[(f"tfm.h{h}", r)] for r in ("timesfm", "chance", "fixed_lr"))
        for yr in YEARS + ["all"]:
            m = (T.index.year == yr) if yr != "all" else np.ones(len(T), bool)
            g, c, f = T[m], C[m], F[m]
            row = {"year": yr, "horizon": h, "n": len(g), "up_rate": g["y"].mean(),
                   "auc": roc_auc_score(g["y"], g["proba"]),
                   "ba": balanced_accuracy_score(g["y"], g["pred"]), "called_up": g["pred"].mean(),
                   "mean_p": g["proba"].mean(), "brier": ((g["proba"] - g["y"]) ** 2).mean(),
                   "brier_chance": ((c["proba"] - c["y"]) ** 2).mean(), "auc_fixed_lr": roc_auc_score(f["y"], f["proba"])}
            if yr != "all":
                row["auc_ci_lo"], row["auc_ci_hi"] = block_ci(g["y"].to_numpy(), g["proba"].to_numpy(), rng)
                s = spy[spy.year == yr].iloc[0]
                row.update({k: s[k] for k in ("ret", "max_fall", "peak", "trough", "recovered", "vol", "big_days")})
            if h == 1:
                row["up_rate_cc"], row["up_rate_gap"] = lab_cc[m].mean(), lab_gap[m].mean()
                row["auc_cc"] = roc_auc_score(lab_cc[m], g["proba"])
                row["auc_gap"] = roc_auc_score(lab_gap[m], g["proba"])
                row["corr_r1"], row["corr_r5"] = g["proba"].corr(r1[m]), g["proba"].corr(r5[m])
                row["auc_rule_r1"], row["auc_rule_r5"] = roc_auc_score(g["y"], r1[m]), roc_auc_score(g["y"], r5[m])
            else:
                p = pr[m]
                row.update({"official_turn_auc": roc_auc_score(g["y"], off["turn"][m]),
                            "official_rest_auc": roc_auc_score(g["y"], off["rest"][m]),
                            "price_auc": roc_auc_score(p["y"], p["pup"]), "price_turn_auc": roc_auc_score(p["y"], p["turn"]),
                            "price_rest_auc": roc_auc_score(p["y"], p["rest"]),
                            "price_off_auc": roc_auc_score(p["y"], p["pup_off"])})
            rows.append(row)
    D = pd.DataFrame(rows)
    D.to_csv(RES_DIR / "timesfm_year_detail.csv", index=False)

    # ---- phases (h=5)
    prow = []
    for yr in YEARS:
        idx = t5.index[t5.index.year == yr]
        spans = phases_for(yr, spy)
        used = pd.Series(False, index=idx)
        for lab, kind, a, e in spans + [("other", "other", None, None)]:
            if lab == "other":
                msk = ~used
            else:
                msk = pd.Series((idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(e)), index=idx)
                used |= msk
            d = idx[msk.to_numpy()]
            if len(d) == 0:
                continue
            g, p = t5.loc[d], pr.loc[d]
            prow.append({"year": yr, "phase": lab, "kind": kind, "start": d[0].date(), "end": d[-1].date(), "n": len(d),
                         "up_rate": g["y"].mean(), "mean_p": g["proba"].mean(), "called_up": g["pred"].mean(),
                         "hit_rate": (g["pred"] == g["y"]).mean(), "turn_points_up": (p["turn"] > 0).mean(),
                         "rest_pushes_up": (p["rest"] > 0).mean()})
        if yr == 2022:   # pooled legs, as in timesfm_split_stretches.csv
            for kind, key in (("fall", "fall"), ("recovery", "rebound")):
                d = pd.DatetimeIndex(np.concatenate([x for x in
                    [t5.index[(t5.index >= a) & (t5.index <= e)].values for lab, k, a, e, _ in STRETCHES
                     if lab.startswith("2022") and k == kind for a, e in [(pd.Timestamp(a), pd.Timestamp(e))]]]))
                g, p = t5.loc[d], pr.loc[d]
                prow.append({"year": yr, "phase": f"{key}s (all)", "kind": kind, "start": "", "end": "", "n": len(d),
                             "up_rate": g["y"].mean(), "mean_p": g["proba"].mean(), "called_up": g["pred"].mean(),
                             "hit_rate": (g["pred"] == g["y"]).mean(), "turn_points_up": (p["turn"] > 0).mean(),
                             "rest_pushes_up": (p["rest"] > 0).mean()})
    PH = pd.DataFrame(prow)
    PH.to_csv(RES_DIR / "timesfm_year_phases.csv", index=False)

    # ---- cross-checks against existing CSVs
    sy = pd.read_csv(RES_DIR / "timesfm_split_years.csv")
    sy = sy[sy.year != "all"].assign(year=lambda x: x.year.astype(int)).set_index("year")
    d5 = D[(D.horizon == 5) & (D.year != "all")].assign(year=lambda x: x.year.astype(int)).set_index("year")
    print("max |diff| h5 vs split_years: auc %.2e ci_lo %.2e ci_hi %.2e price %.2e turn %.2e rest %.2e" % (
        (d5.auc - sy.official_auc).abs().max(), (d5.auc_ci_lo - sy.official_ci_lo).abs().max(),
        (d5.auc_ci_hi - sy.official_ci_hi).abs().max(), (d5.price_auc - sy.price_auc).abs().max(),
        (d5.price_turn_auc - sy.price_turn_auc).abs().max(), (d5.price_rest_auc - sy.price_rest_auc).abs().max()))
    by = pd.read_csv(RES_DIR / "timesfm_test_by_year.csv")
    for h in (1, 5):
        for role, col in (("timesfm", "auc"), ("fixed_lr", "auc_fixed_lr")):
            b = by[(by.horizon == h) & (by.role == role)].set_index("year")["roc_auc"]
            x = D[(D.horizon == h) & (D.year != "all")].assign(year=lambda z: z.year.astype(int)).set_index("year")[col]
            print(f"h{h} {role} AUC vs by_year csv: max diff {(x - b).abs().max():.2e}")
        for role, col in (("timesfm", "ba"), ("timesfm", "brier"), ("chance", "brier_chance")):
            b = by[(by.horizon == h) & (by.role == role)].set_index("year")["balanced_acc" if col == "ba" else "brier"]
            x = D[(D.horizon == h) & (D.year != "all")].assign(year=lambda z: z.year.astype(int)).set_index("year")[col]
            print(f"h{h} {role} {col} vs by_year csv: max diff {(x - b).abs().max():.2e}")
    ss = pd.read_csv(RES_DIR / "timesfm_split_stretches.csv").set_index("stretch")
    pj = PH.set_index(["year", "phase"])
    worst = 0
    for lab in ss.index:
        yr = int(lab[:4]); ph = lab[5:]
        r = pj.loc[(yr, ph)]
        worst = max(worst, abs(r.n - ss.loc[lab, "n"]), abs(r.up_rate - ss.loc[lab, "up_rate"]),
                    abs(r.turn_points_up - ss.loc[lab, "turn_points_up"]), abs(r.rest_pushes_up - ss.loc[lab, "rest_pushes_up"]))
    print(f"phases vs split_stretches ({len(ss)} stretches/pools): max abs diff {worst:.2e}")
    ex = pd.read_csv(RES_DIR / "timesfm_exploratory_stretches.csv").set_index("label")
    w2 = 0
    for lab in ex.index:
        r = pj.loc[(int(lab[:4]), lab[5:])]
        w2 = max(w2, abs(r.mean_p - ex.loc[lab, "mean_proba"]), abs(r.called_up - ex.loc[lab, "called_up"]), abs(r.n - ex.loc[lab, "n"]))
    print(f"phases vs exploratory_stretches: max abs diff {w2:.2e}")
    print(D[D.year == "all"][["horizon", "auc", "price_auc", "price_off_auc"]].to_string(index=False))
    print("official turn/rest vs split_years: %.2e %.2e" % ((d5.official_turn_auc - sy.official_turn_auc).abs().max(),
                                                         (d5.official_rest_auc - sy.official_rest_auc).abs().max()))

    # ---- figures
    INK, MUTE, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
    BLUE, ORANGE, VIOLET, YELLOW = "#2a78d6", "#eb6834", "#6250d6", "#eda100"
    prob = t5["proba"]
    roll_p = prob.rolling(20).mean()
    roll_y = t5["y"].rolling(20).mean()
    lo = min(prob.min(), roll_y.min(), 0.5) - 0.03
    hi = max(prob.max(), roll_y.max(), 0.5) + 0.03
    for yr in YEARS:
        d1 = D[(D.year == yr) & (D.horizon == 1)].iloc[0]
        d5r = D[(D.year == yr) & (D.horizon == 5)].iloc[0]
        cy = close[close.index.year == yr]
        spans = phases_for(yr, spy)
        multi = yr in (2020, 2022, 2025)
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.15]})
        fig.patch.set_facecolor("#fcfcfb")
        for ax in (a1, a2):
            ax.set_facecolor("#fcfcfb")
            ax.grid(axis="y", color=GRID, lw=0.8)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            ax.tick_params(colors=MUTE, labelsize=9, length=0)
        a1.plot(cy.index, cy.values, color=BLUE, lw=2)
        for lab, kind, a, e in spans:
            col = VIOLET if kind == "fall" else YELLOW
            for ax in (a1, a2):
                ax.axvspan(pd.Timestamp(a), pd.Timestamp(e) + pd.Timedelta(hours=23), color=col, alpha=0.16, lw=0)
            short = lab
            if yr == 2022:
                short = short.replace("fall ", "F").replace("rebound ", "R")
            a1.text(pd.Timestamp(a) + (pd.Timestamp(e) - pd.Timestamp(a)) / 2, 0.97, short, transform=a1.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8, color=INK)
        a1.set_ylabel("SPY close (USD)", color=MUTE, fontsize=9)
        a1.set_title(f"{yr}   AUC h=5 {d5r.auc:.3f} [{d5r.auc_ci_lo:.3f}, {d5r.auc_ci_hi:.3f}]   |   AUC h=1 {d1.auc:.3f} [{d1.auc_ci_lo:.3f}, {d1.auc_ci_hi:.3f}]",
                     loc="left", fontsize=11, color=INK)
        m = prob.index.year == yr
        a2.plot(prob.index[m], prob[m], color=BLUE, lw=0.8, alpha=0.45, label="TimesFM P(up), daily (h=5)")
        a2.plot(roll_p.index[m], roll_p[m], color=BLUE, lw=2.6, label="P(up), 20-day mean")
        a2.plot(roll_y.index[m], roll_y[m], color=ORANGE, lw=1.8, ls=(0, (5, 3)), label="actual share of 5-day windows up, 20-day mean")
        a2.axhline(0.5, color=MUTE, lw=1)
        a2.set_ylim(lo, hi)
        a2.set_ylabel("probability / share up", color=MUTE, fontsize=9)
        from matplotlib.patches import Patch
        h_, l_ = a2.get_legend_handles_labels()
        h_ += [Patch(color=VIOLET, alpha=0.3), Patch(color=YELLOW, alpha=0.3)]
        l_ += ["fall", "recovery"]
        a2.legend(h_, l_, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=8.5, labelcolor=MUTE)
        a2.xaxis.set_major_locator(mdates.MonthLocator())
        a2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        a2.set_xlim(pd.Timestamp(f"{yr}-01-01"), pd.Timestamp(f"{yr}-12-31"))
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"timesfm_year_{yr}.png", dpi=130)
        plt.close(fig)
    pd.set_option("display.width", 250)
    print(PH.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

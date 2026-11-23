"""EXPLORATORY (post-test): per-year detail of the best drift-applied TimesFM-3 on SPY, 2019-2025, for the talk slides.
Nothing here changes a pre-registered result. Uses saved frames only; no new TimesFM run.

Usage:  python timesfm_drift_year_detail.py -> results/timesfm_drift_year_detail.csv   (one row per year x horizon, plus "all")
                                              results/timesfm_drift_year_phases.csv   (one row per year x phase, h=5)

Best drift version (chosen on the 2015-2018 selection folds by mean AUC gain over static TimesFM, results/timesfm_drift_choices.json):
  h=1  stacking, config stack-4      h=5  recalibration, config platt-6
Frames: results/timesfm_drift_test_predictions.pkl, keys (horizon, role). AUCs are checked against timesfm_drift_test_levels.csv.
Same logic as timesfm_explore_year_detail.py: AUC 95% interval = moving-block bootstrap (20-day blocks, 1000 draws, default_rng(0)
per horizon, years in order); phases = timesfm_explore_years.STRETCHES for 2020/2022/2025, peak/trough/recovered dates otherwise.
Extra columns: static TimesFM and fixed-LR AUC per year, and for h=5 the "turn part" (least-squares fit of the score on SPY's past
20-day log move r20) and the rest, for the drift score and the static score.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from tf_paths import RES_DIR
from common import DATA_DIR
from timesfm_explore_years import STRETCHES
from timesfm_explore_split import split, block_ci

YEARS = list(range(2019, 2026))
BEST = {1: ("stack", "stack-4"), 5: ("platt", "platt-6")}


def phases_for(yr, spy):
    if yr in (2020, 2022, 2025):
        return [(lab[5:], kind, a, e) for lab, kind, a, e, _ in STRETCHES if lab.startswith(str(yr))]
    r = spy[spy.year == yr].iloc[0]
    pk, tr = pd.Timestamp(r["peak"]), pd.Timestamp(r["trough"])
    out = [("fall", "fall", pk + pd.Timedelta(days=1), tr)]
    if isinstance(r["recovered"], str) and r["recovered"]:
        out.append(("recovery", "recovery", tr + pd.Timedelta(days=1), pd.Timestamp(r["recovered"])))
    return out


def main():
    close = pd.read_pickle(DATA_DIR / "aligned_prices.pkl")["SPY"]["Close"].astype(float)
    ls = np.log(close)
    P = pd.read_pickle(RES_DIR / "timesfm_drift_test_predictions.pkl")
    spy = pd.read_csv(RES_DIR / "timesfm_spy_years.csv")
    lev = pd.read_csv(RES_DIR / "timesfm_drift_test_levels.csv")

    # ---- the frames must reproduce the saved test levels
    for h, (role, cfg) in BEST.items():
        for rl, cf in ((role, cfg), ("static", None), ("fixed_lr", None)):
            row = lev[(lev.horizon == h) & (lev.role == rl)]
            if cf is not None:
                row = row[row.cfg == cf]
            assert len(row) == 1, (h, rl, cf)
            f = P[(h, rl)]
            assert abs(roc_auc_score(f.y, f.proba) - row.roc_auc.iloc[0]) < 1e-5, (h, rl)

    rows = []
    t5 = P[(5, "platt")]
    r20 = ls.diff(20).reindex(t5.index)
    r1, r5 = ls.diff(1), ls.diff(5)
    for h in (1, 5):
        rng = np.random.default_rng(0)
        role, cfg = BEST[h]
        T, S, F = P[(h, role)], P[(h, "static")], P[(h, "fixed_lr")]
        assert (T.index == S.index).all() and (T.y == S.y).all()
        if h == 5:
            tt = pd.DataFrame({"d": T.proba, "s": S.proba, "r20": r20})
            tt["d_turn"], tt["d_rest"], _ = split(tt, "d", ["r20"])
            tt["s_turn"], tt["s_rest"], _ = split(tt, "s", ["r20"])
        for yr in YEARS + ["all"]:
            m = (T.index.year == yr) if yr != "all" else np.ones(len(T), bool)
            g, s, f = T[m], S[m], F[m]
            row = {"year": yr, "horizon": h, "cfg": cfg, "n": len(g), "up_rate": g["y"].mean(),
                   "auc": roc_auc_score(g["y"], g["proba"]), "ba": balanced_accuracy_score(g["y"], g["pred"]),
                   "called_up": g["pred"].mean(), "mean_p": g["proba"].mean(), "brier": ((g["proba"] - g["y"]) ** 2).mean(),
                   "auc_static": roc_auc_score(s["y"], s["proba"]), "called_up_static": s["pred"].mean(),
                   "auc_fixed_lr": roc_auc_score(f["y"], f["proba"])}
            if yr != "all":
                row["auc_ci_lo"], row["auc_ci_hi"] = block_ci(g["y"].to_numpy(), g["proba"].to_numpy(), rng)
                q = spy[spy.year == yr].iloc[0]
                row.update({k: q[k] for k in ("ret", "max_fall", "peak", "trough", "recovered", "vol", "big_days")})
            if h == 1:
                row["corr_r1"], row["corr_r5"] = g["proba"].corr(r1.reindex(g.index)), g["proba"].corr(r5.reindex(g.index))
                row["auc_rule_r1"] = roc_auc_score(g["y"], r1.reindex(g.index))
                row["auc_rule_r5"] = roc_auc_score(g["y"], r5.reindex(g.index))
            else:
                row.update({"drift_turn_auc": roc_auc_score(g["y"], tt["d_turn"][m]), "drift_rest_auc": roc_auc_score(g["y"], tt["d_rest"][m]),
                            "static_turn_auc": roc_auc_score(g["y"], tt["s_turn"][m]), "static_rest_auc": roc_auc_score(g["y"], tt["s_rest"][m]),
                            "corr_r20": g["proba"].corr(r20[m])})
            rows.append(row)
    D = pd.DataFrame(rows)
    D.to_csv(RES_DIR / "timesfm_drift_year_detail.csv", index=False)

    # ---- phases (h=5, drift-applied = platt-6)
    prow = []
    def add(yr, lab, kind, d, start, end):
        g = t5.loc[d]
        prow.append({"year": yr, "phase": lab, "kind": kind, "start": start, "end": end, "n": len(d),
                     "up_rate": g["y"].mean(), "mean_p": g["proba"].mean(), "called_up": g["pred"].mean(),
                     "hit_rate": (g["pred"] == g["y"]).mean(),
                     "called_up_static": P[(5, "static")].loc[d, "pred"].mean(), "mean_p_static": P[(5, "static")].loc[d, "proba"].mean()})
    for yr in YEARS:
        idx = t5.index[t5.index.year == yr]
        used = pd.Series(False, index=idx)
        for lab, kind, a, e in phases_for(yr, spy) + [("other", "other", None, None)]:
            if lab == "other":
                msk = ~used
            else:
                msk = pd.Series((idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(e)), index=idx)
                used |= msk
            d = idx[msk.to_numpy()]
            if len(d):
                add(yr, lab, kind, d, d[0].date(), d[-1].date())
        if yr == 2022:
            for kind, key in (("fall", "fall"), ("recovery", "rebound")):
                d = pd.DatetimeIndex(np.concatenate([t5.index[(t5.index >= pd.Timestamp(a)) & (t5.index <= pd.Timestamp(e))].values
                                                     for lab, k, a, e, _ in STRETCHES if lab.startswith("2022") and k == kind]))
                add(yr, f"{key}s (all)", kind, d, "", "")
    PH = pd.DataFrame(prow)
    PH.to_csv(RES_DIR / "timesfm_drift_year_phases.csv", index=False)

    # ---- cross-check against the saved per-year drift table
    by = pd.read_csv(RES_DIR / "timesfm_drift_test_by_year.csv").drop_duplicates(["horizon", "role", "year"])
    for h, (role, cfg) in BEST.items():
        b = by[(by.horizon == h) & (by.role == role)].set_index("year")["roc_auc"]
        x = D[(D.horizon == h) & (D.year != "all")].assign(year=lambda z: z.year.astype(int)).set_index("year")["auc"]
        print(f"h{h} {role} AUC vs drift_test_by_year.csv: max diff {(x - b).abs().max():.2e}")
        assert (x - b).abs().max() < 1e-5
    pd.set_option("display.width", 250)
    print(D[["year", "horizon", "n", "up_rate", "auc", "auc_ci_lo", "auc_ci_hi", "auc_static", "auc_fixed_lr", "called_up", "called_up_static"]].round(3).to_string(index=False))
    print(D[D.horizon == 5][["year", "drift_turn_auc", "drift_rest_auc", "static_turn_auc", "static_rest_auc", "corr_r20"]].round(3).to_string(index=False))
    print(D[D.horizon == 1][["year", "corr_r1", "corr_r5", "auc_rule_r1", "auc_rule_r5"]].round(3).to_string(index=False))
    print(PH.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

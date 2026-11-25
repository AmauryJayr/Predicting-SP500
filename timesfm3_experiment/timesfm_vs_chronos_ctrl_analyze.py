"""Controlled-swap analysis (reads caches only; no model call). Design: timesfm_vs_chronos_ctrl_design.md (locked).

Cells: TF x L, TF x G, CH x L, CH x G (phase 1); CH x L+15cov vs TimesFM `multi` (phase 2, if its caches exist).
Run (from timesfm3_experiment/, timesfm env):  PYTHONDONTWRITEBYTECODE=1 python timesfm_vs_chronos_ctrl_analyze.py
Outputs: results/timesfm_vs_chronos_ctrl_*.csv
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import timesfm_lib as tl
import timesfm_vs_chronos_ctrl_run as R
from tf_paths import RES_DIR
from common import Dataset, paired_bootstrap

OUT = "timesfm_vs_chronos_ctrl_"
CH_RES = R.CHR_RES
QL21 = np.array([0.01, 0.05] + [round(0.10 + 0.05 * i, 2) for i in range(17)] + [0.95, 0.99])
I9 = [int(np.flatnonzero(np.isclose(QL21, l))[0]) for l in tl.QLEVELS]       # the 0.1..0.9 columns of Chronos's 21
PRE, ALLP = R.PRE, R.PRE + ["TEST"]
NB = 0


def auc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else np.nan


def read9(q9: np.ndarray, thr: np.ndarray, chronos: bool) -> np.ndarray:
    """timesfm_lib.p_up formula on the 9 deciles: 1 - interp(thr, q, levels, left=0.05, right=0.95)."""
    out = np.empty(len(q9))
    for i in range(len(q9)):
        q = np.maximum.accumulate(q9[i]) if chronos else q9[i]
        out[i] = 1.0 - np.interp(thr[i], q, tl.QLEVELS, left=0.05, right=0.95)
    return out


def read21(q21: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """chronos_study.prob_up on the 21 quantiles (threshold generalised from 0 to thr)."""
    out = np.empty(len(q21))
    for i in range(len(q21)):
        q = np.maximum.accumulate(q21[i])
        out[i] = 1.0 - np.interp(thr[i], q, QL21, left=QL21[0] / 2, right=1 - (1 - QL21[-1]) / 2)
    return out


def frame(y, p, idx):
    p = np.asarray(p, float)
    return pd.DataFrame({"y": np.asarray(y), "proba": p, "pred": (p >= np.median(p)).astype(int)}, index=idx)


def boot(a, b, h):
    global NB
    NB += 1
    r = paired_bootstrap(a, b, h)
    p1 = r["p_roc_auc"]
    return {"d_auc": r["d_roc_auc"], "lo": r["d_roc_auc_lo"], "hi": r["d_roc_auc_hi"],
            "p_two_sided": min(1.0, max(2 * min(p1, 1 - p1), 1 / 2001))}


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def load_npz(name):
    z = np.load(R.CACHE / name)
    return pd.DatetimeIndex(z["dates"]), z["q"]


def merged(*pairs):
    d = pairs[0][0].append([x[0] for x in pairs[1:]]) if len(pairs) > 1 else pairs[0][0]
    q = np.concatenate([x[1] for x in pairs])
    return d, q


def take(d, q, idx):
    i = d.get_indexer(idx)
    assert (i >= 0).all(), f"{int((i < 0).sum())} dates missing"
    return q[i]


def main():
    p = tl.panel()
    lc = pd.Series(p["spy"], index=p["dates"])
    close = np.exp(lc)
    trail = pd.DataFrame({"ret1": lc.diff(1), "ret5": lc.diff(5), "vol20": lc.diff().rolling(20).std()})
    cache = {}
    def tf_L():
        if "tfL" not in cache:
            dp, qp, _ = tl.read_cache("price", R.CTX)
            cache["tfL"] = merged((dp, qp), load_npz("tf_L_TEST.npz"))
        return cache["tfL"]
    def ch_L(kind="L"):
        if kind not in cache:
            cache[kind] = merged(load_npz(f"ch_{kind}_PRE.npz"), load_npz(f"ch_{kind}_TEST.npz"))
        return cache[kind]

    have_p2 = all((R.CACHE / f).exists() for f in ("ch_Lcov_PRE.npz", "ch_Lcov_TEST.npz"))
    CELLS = ["TFxL", "TFxG", "CHxL", "CHxG"]
    S, S21, chk, rows_cell = {}, {}, [], []
    for h in (1, 5):
        ds = Dataset(h)
        idx_all, per_of, ys = [], [], []
        for per in ALLP:
            s, e = ds.period_positions(per)
            idx_all.append(ds.index[s:e]); per_of += [per] * (e - s); ys.append(ds.y[s:e])
        idx = idx_all[0].append(idx_all[1:])
        y = np.concatenate(ys)
        per_of = np.array(per_of)
        anchor = lc.reindex(idx).to_numpy()
        d = {}
        # Chronos `uni` (G) cached pickles: label/date checks and the 21-quantile reads
        pk = pd.concat([pd.read_pickle(f"{CH_RES}/pred_h{h}_uni_L256_{per}_s1.pkl") for per in ALLP])
        chk.append({"h": h, "frame": "chronos_uni_L256_cached", "n": len(pk), "same_dates_as_Dataset": bool(pk.index.equals(idx)),
                    "label_mismatches": int((pk["y"].to_numpy() != y).sum())})
        q21g = pk[[f"q{t:.2f}" for t in QL21]].to_numpy()
        # step-h quantiles (n, 9) per cell
        q9 = {"TFxL": take(*tf_L(), idx)[:, h - 1],
              "TFxG": take(*merged(load_npz(f"tf_G{h}_PRE.npz"), load_npz(f"tf_G{h}_TEST.npz")), idx)[:, h - 1],
              "CHxL": take(*ch_L(), idx)[:, h - 1][:, I9], "CHxG": q21g[:, I9]}
        q21 = {"CHxL": take(*ch_L(), idx)[:, h - 1], "CHxG": q21g}
        thr = {"TFxL": anchor, "TFxG": np.zeros(len(idx)), "CHxL": anchor, "CHxG": np.zeros(len(idx))}
        if have_p2:
            q9["CHxLcov"] = take(*ch_L("Lcov"), idx)[:, h - 1][:, I9]; q21["CHxLcov"] = take(*ch_L("Lcov"), idx)[:, h - 1]
            thr["CHxLcov"] = anchor
            CELLS_H = CELLS + ["CHxLcov"]
        else:
            CELLS_H = CELLS
        for c in CELLS_H:
            sc = read9(q9[c], thr[c], c.startswith("CH"))
            d[c] = pd.DataFrame({"score": sc, "spread": q9[c][:, 8] - q9[c][:, 0], "med": q9[c][:, 4] - thr[c]}, index=idx)
        # reproduces the cached Chronos proba from its 21 quantiles (sanity of read21)
        s21 = read21(q21["CHxG"], thr["CHxG"])
        chk.append({"h": h, "frame": "read21_vs_cached_proba_CHxG", "n": len(s21), "max_abs_diff": float(np.abs(s21 - pk["proba"].to_numpy()).max())})
        S21[h] = {c: pd.Series(read21(q21[c], thr[c]), index=idx) for c in q21}
        # official references (TEST/PRE both): TimesFM multi (9 quantile read of its cache) and Chronos chosen config
        qm, _ = tl.aligned("multi", 256, idx)
        d["TFmulti"] = pd.DataFrame({"score": tl.scores("multi", 256, h, idx).to_numpy(),
                                     "spread": qm[:, h - 1, 8] - qm[:, h - 1, 0], "med": qm[:, h - 1, 4] - anchor}, index=idx)
        v, L = {1: ("joint", 512), 5: ("core", 256)}[h]
        co = pd.concat([pd.read_pickle(f"{CH_RES}/pred_h{h}_{v}_L{L}_{per}_s1.pkl") for per in ALLP])
        chk.append({"h": h, "frame": f"chronos_chosen_{v}_L{L}", "n": len(co), "same_dates_as_Dataset": bool(co.index.equals(idx)),
                    "label_mismatches": int((co["y"].to_numpy() != y).sum())})
        d["CHofficial"] = pd.DataFrame({"score": co["proba"].to_numpy(), "spread": (co["q0.90"] - co["q0.10"]).to_numpy(),
                                        "med": co["q0.50"].to_numpy()}, index=idx)
        S[h] = {"d": d, "idx": idx, "y": y, "per": per_of, "ds": ds}

    # ------------------------------- (a) AUC per cell, vs chance, contrasts ------------------------------
    AUC, CON, VC, YR, ANC = [], [], [], [], []
    for h in (1, 5):
        d, idx, y, per, ds = S[h]["d"], S[h]["idx"], S[h]["y"], S[h]["per"], S[h]["ds"]
        cl, cn = close.reindex(idx).to_numpy(), close.shift(-h).reindex(idx).to_numpy()
        ycc = (cn > cl).astype(int)
        okcc = np.isfinite(cn)
        for c, dfc in d.items():
            sc = dfc["score"].to_numpy()
            for grp, m in (("TEST", per == "TEST"), ("2013-2018", per != "TEST"), ("F1", per == "F1"), ("F2", per == "F2"), ("VAL", per == "VAL")):
                row = {"h": h, "cell": c, "period": grp, "n": int(m.sum()), "auc": auc(y[m], sc[m]),
                       "auc_close_to_close_label": auc(ycc[m & okcc], sc[m & okcc])}
                if c in S21[h] and grp in ("TEST", "2013-2018"):
                    row["auc_21quantile_read"] = auc(y[m], S21[h][c].to_numpy()[m])
                AUC.append(row)
            if c in ("TFxL", "TFxG", "CHxL", "CHxG", "CHxLcov") and True:
                m = per == "TEST"
                fr = frame(y[m], sc[m], idx[m])
                ch_fr = tl.chance_frame(y[m], idx[m], float(ds.y[:ds.n_known(ds.period_positions("TEST")[0])].mean()))
                VC.append({"h": h, "cell": c, **{k: v for k, v in boot(fr, ch_fr, h).items() if k != "p_two_sided"}})
            for yr in range(2019, 2026):
                m = (per == "TEST") & (idx.year == yr)
                YR.append({"h": h, "cell": c, "year": yr, "n": int(m.sum()), "auc": auc(y[m], sc[m])})
        m = per == "TEST"
        fr = {c: frame(y[m], d[c]["score"].to_numpy()[m], idx[m]) for c in d}
        spec = [("model effect within L: TFxL - CHxL", "TFxL", "CHxL"), ("model effect within G: TFxG - CHxG", "TFxG", "CHxG"),
                ("input effect within TimesFM: TFxL - TFxG", "TFxL", "TFxG"), ("input effect within Chronos: CHxL - CHxG", "CHxL", "CHxG")]
        for name, a, b in spec:
            CON.append({"family": "phase1", "h": h, "contrast": name, "auc_a": auc(y[m], d[a]["score"].to_numpy()[m]),
                        "auc_b": auc(y[m], d[b]["score"].to_numpy()[m]), **boot(fr[a], fr[b], h)})
        if have_p2:
            CON.append({"family": "phase2", "h": h, "contrast": "TFmulti - CHxLcov", "auc_a": auc(y[m], d["TFmulti"]["score"].to_numpy()[m]),
                        "auc_b": auc(y[m], d["CHxLcov"]["score"].to_numpy()[m]), **boot(fr["TFmulti"], fr["CHxLcov"], h)})
            CON.append({"family": "phase2", "h": h, "contrast": "input/cov effect within Chronos: CHxLcov - CHxL",
                        "auc_a": auc(y[m], d["CHxLcov"]["score"].to_numpy()[m]), "auc_b": auc(y[m], d["CHxL"]["score"].to_numpy()[m]),
                        **boot(fr["CHxLcov"], fr["CHxL"], h)})
    con = pd.DataFrame(CON)
    con["p_holm"] = np.nan
    for fam in con.family.unique():
        k = con.family == fam
        if fam == "phase2":
            k = k & con.contrast.str.startswith("TFmulti")
        con.loc[k, "p_holm"] = holm(con.loc[k, "p_two_sided"].to_numpy())
    # ------------------------------- (b) behaviour -----------------------------------------------------------
    BEH, XC = [], []
    for h in (1, 5):
        d, idx, per = S[h]["d"], S[h]["idx"], S[h]["per"]
        tr = trail.reindex(idx)
        for grp, m in (("TEST", per == "TEST"), ("2013-2018", per != "TEST")):
            for c, dfc in d.items():
                sc = dfc["score"].to_numpy()[m]
                row = {"h": h, "period": grp, "cell": c}
                for k in ("ret1", "ret5", "vol20"):
                    row[f"rho_{k}"] = spearmanr(sc, tr[k].to_numpy()[m])[0]
                row["rho_own_spread"] = spearmanr(sc, dfc["spread"].to_numpy()[m])[0]
                row["rho_own_median_minus_thr"] = spearmanr(sc, dfc["med"].to_numpy()[m])[0]
                BEH.append(row)
            names = list(d)
            M = pd.DataFrame({c: d[c]["score"].to_numpy()[m] for c in names}).corr(method="spearman")
            for a in names:
                for b in names:
                    XC.append({"h": h, "period": grp, "cell_a": a, "cell_b": b, "spearman": M.loc[a, b]})
    pd.DataFrame(AUC).to_csv(RES_DIR / f"{OUT}auc.csv", index=False, float_format="%.5f")
    con.to_csv(RES_DIR / f"{OUT}contrasts.csv", index=False, float_format="%.5f")
    pd.DataFrame(VC).to_csv(RES_DIR / f"{OUT}vs_chance.csv", index=False, float_format="%.5f")
    pd.DataFrame(YR).to_csv(RES_DIR / f"{OUT}by_year.csv", index=False, float_format="%.5f")
    pd.DataFrame(BEH).to_csv(RES_DIR / f"{OUT}behaviour.csv", index=False, float_format="%.5f")
    pd.DataFrame(XC).to_csv(RES_DIR / f"{OUT}cell_corr.csv", index=False, float_format="%.5f")
    pd.DataFrame(chk).to_csv(RES_DIR / f"{OUT}checks.csv", index=False, float_format="%.6f")
    pd.set_option("display.width", 250)
    print(pd.DataFrame(chk).to_string())
    a = pd.DataFrame(AUC)
    print(a[a.period.isin(["TEST", "2013-2018"])].round(4).to_string())
    print(pd.DataFrame(VC).round(4).to_string())
    print(con.round(4).to_string())
    print(pd.DataFrame(BEH).round(3).to_string())
    print(f"paired-bootstrap comparisons made: {NB}")


if __name__ == "__main__":
    main()

"""Exploratory TimesFM-3 vs Chronos-2 comparison of SPY direction forecasts (cached predictions only).

No model is called. Inputs:
  * TimesFM-3: `results/timesfm_test_predictions.pkl` (static test frames), `results/timesfm_cache/*.npz`
    (quantiles, via `timesfm_lib`), `timesfm_drift_lib` (the drift layers).
  * Chronos-2 (teammate branch `origin/chronos`, commit 6334147): prediction pickles
    `chronos_experiment/results/pred_h{h}_{variant}_L{L}_{PERIOD}_s1.pkl` (columns y, proba, thr, pred, q0.01..q0.99).
    They are NOT copied into this repo. Point CHRONOS_RESULTS (env var) or argv[1] at the folder holding them; the default is
    the scratchpad extraction made with
        mkdir -p $D && git archive origin/chronos chronos_experiment | tar -x -C $D
    (then CHRONOS_RESULTS=$D/chronos_experiment/results). Nothing on the chronos branch is checked out or modified.

Everything here is exploratory: run after the TEST period was opened, not pre-registered. `NCOMP` counts the
paired-bootstrap comparisons made; confidence intervals are 95%, uncorrected for multiplicity.

Run (from timesfm3_experiment/):  PYTHONDONTWRITEBYTECODE=1 python timesfm_vs_chronos.py [chronos_results_dir]
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import timesfm_lib as tl
import timesfm_drift_lib as dl
from tf_paths import RES_DIR
from common import Dataset, block_indices, block_len, paired_bootstrap

DEFAULT = ("/tmp/claude-1000/-home-tong-nong-Documents-Predicting-SP500/7e6e832c-eb35-4db3-a775-fd9250864606/"
           "scratchpad/chronos_branch/chronos_experiment/results")
CH = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CHRONOS_RESULTS", DEFAULT))
OUT = "timesfm_vs_chronos_"
CHOSEN = {1: ("joint", 512), 5: ("core", 256)}          # Chronos's frozen choices (chronos_shortlist.json)
FOLDS = ["F1", "F2", "VAL"]
NCOMP = 0                                               # running count of paired-bootstrap comparisons (boot())


def save(df, name):
    df = pd.DataFrame(df)
    df.to_csv(RES_DIR / f"{OUT}{name}.csv", index=False, float_format="%.5f")
    return df


def auc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else np.nan


def boot(a, b, h):
    """paired moving-block bootstrap (experiments/common.py) of the AUC difference A - B; counted."""
    global NCOMP
    NCOMP += 1
    r = paired_bootstrap(a, b, h)
    return {"d_auc": r["d_roc_auc"], "lo": r["d_roc_auc_lo"], "hi": r["d_roc_auc_hi"], "p_one_sided": r["p_roc_auc"]}


def cf(h, variant, L, period):
    return pd.read_pickle(CH / f"pred_h{h}_{variant}_L{L}_{period}_s1.pkl")


def frame_of(y, p, idx):
    p = np.asarray(p, float)
    return pd.DataFrame({"y": np.asarray(y), "proba": p, "pred": (p >= np.median(p)).astype(int)}, index=idx)


def main():
    ds = {h: Dataset(h) for h in (1, 5)}
    pn = tl.panel()
    lc = pd.Series(pn["spy"], index=pn["dates"])
    close = np.exp(lc)
    old = pd.read_pickle(RES_DIR / "timesfm_test_predictions.pkl")
    trail = pd.DataFrame({"ret1": lc.diff(1), "ret5": lc.diff(5), "vol20": lc.diff().rolling(20).std()})

    # ---------------- label / date checks -----------------------------------------------------------------
    chk = []

    def check(name, h, fr, period="TEST"):
        s, e = ds[h].period_positions(period)
        same_idx = fr.index.equals(ds[h].index[s:e])
        mism = -1 if not same_idx else int((fr["y"].to_numpy() != ds[h].y[s:e]).sum())
        chk.append({"frame": name, "h": h, "period": period, "n": len(fr), "same_dates_as_Dataset": same_idx,
                    "label_mismatches": mism, "ok": same_idx and mism == 0})

    T = {}
    for h in (1, 5):
        v, L = CHOSEN[h]
        T[h] = {"tfm": old[(f"tfm.h{h}", "timesfm")], "chr": cf(h, v, L, "TEST"), "chr_uni": cf(h, "uni", 256, "TEST")}
        for k, fr in T[h].items():
            check(f"{k}_test", h, fr)

    # label anchor: TimesFM scores P(Close[t+h] > Close[t]); the label is Close[t+h] > Open[t+1]
    anc = []
    for h in (1, 5):
        d = ds[h]
        c0 = close.reindex(d.index).to_numpy()
        ch_ = close.shift(-h).reindex(d.index).to_numpy()
        ok = np.isfinite(ch_)
        y_cc = ch_ > c0
        anc.append({"h": h, "n": int(ok.sum()),
                    "share_label_differs_from_Close_t+h_gt_Close_t": float((y_cc[ok] != (d.y[ok] == 1)).mean()),
                    "share_up_label": float(d.y[ok].mean()), "share_up_close_to_close": float(y_cc[ok].mean())})
    for a_, h in zip(anc, (1, 5)):                       # TimesFM raw test AUC against both label definitions
        s_, e_ = ds[h].period_positions("TEST")
        ix = ds[h].index[s_:e_]
        ycc = (close.shift(-h).reindex(ix) > close.reindex(ix)).astype(int).to_numpy()
        sc = tl.scores("multi", 256, h, ix).to_numpy()
        a_.update({"test_auc_tfm_vs_label": auc(ds[h].y[s_:e_], sc), "test_auc_tfm_vs_close_to_close_label": auc(ycc, sc)})
    save(anc, "label_anchor")
    print(pd.DataFrame(anc).round(4).to_string(), flush=True)

    # ---------------- Task 2: head-to-head on the common test days -----------------------------------------
    H2H, YR, GRID = [], [], []
    for h in (1, 5):
        t, c, cu = T[h]["tfm"], T[h]["chr"], T[h]["chr_uni"]
        y = t["y"].to_numpy()
        a_t, a_c, a_u = auc(y, t["proba"]), auc(y, c["proba"]), auc(y, cu["proba"])
        for name, other, a_o in (("TimesFM static - Chronos chosen", c, a_c), ("TimesFM static - Chronos uni L256", cu, a_u)):
            H2H.append({"h": h, "comparison": name, "n": len(y), "auc_timesfm": a_t, "auc_chronos": a_o, **boot(t, other, h)})
        for v in ("uni", "joint", "core"):
            for L in (256, 512, 1024, 2048):
                f = CH / f"pred_h{h}_{v}_L{L}_TEST_s1.pkl"
                if f.exists():
                    fr = pd.read_pickle(f)
                    check(f"chr_{v}_L{L}_test", h, fr)
                    GRID.append({"h": h, "variant": v, "L": L, "auc_chronos": auc(y, fr["proba"]), "auc_timesfm": a_t,
                                 "d_tfm_minus_chr": a_t - auc(y, fr["proba"]), "chosen": (v, L) == CHOSEN[h]})
        for yr in range(2019, 2026):
            m = (t.index.year == yr)
            r = boot(t[m], c[m], h)
            YR.append({"h": h, "year": str(yr), "n": int(m.sum()), "auc_timesfm": auc(y[m], t["proba"][m]),
                       "auc_chronos": auc(y[m], c["proba"][m]), "d_tfm_minus_chr": r["d_auc"], "lo": r["lo"], "hi": r["hi"]})
        for yr in range(2019, 2026):                       # pooled difference without one year (descriptive)
            m = (t.index.year != yr)
            at, ac = auc(y[m], t["proba"][m]), auc(y[m], c["proba"][m])
            YR.append({"h": h, "year": f"all_but_{yr}", "n": int(m.sum()), "auc_timesfm": at, "auc_chronos": ac,
                       "d_tfm_minus_chr": at - ac, "lo": np.nan, "hi": np.nan})
    save(H2H, "head2head")
    save(YR, "by_year")
    save(GRID, "chronos_grid_test")
    print(pd.DataFrame(H2H).round(4).to_string(), flush=True)
    print(pd.DataFrame(YR).round(3).to_string(), flush=True)

    # ---------------- like-for-like on the selection folds (TimesFM price / ret caches end 2018) -------------
    LFL = []
    for h in (1, 5):
        v, L = CHOSEN[h]
        for per in FOLDS + ["pooled"]:
            ps = FOLDS if per == "pooled" else [per]
            ys, idx = [], []
            sc = {k: [] for k in ("tfm_price", "tfm_ret", "tfm_multi", "chr_uni", "chr_chosen")}
            for p in ps:
                s, e = ds[h].period_positions(p)
                ix = ds[h].index[s:e]
                u, cc = cf(h, "uni", 256, p), cf(h, v, L, p)
                check(f"chr_uni_L256_{p}", h, u, p)
                check(f"chr_chosen_{p}", h, cc, p)
                ys.append(ds[h].y[s:e])
                idx.append(ix)
                for var in ("price", "ret", "multi"):
                    sc[f"tfm_{var}"].append(tl.scores(var, 256, h, ix).to_numpy())
                sc["chr_uni"].append(u["proba"].to_numpy())
                sc["chr_chosen"].append(cc["proba"].to_numpy())
            y = np.concatenate(ys)
            ix = idx[0].append(idx[1:]) if len(idx) > 1 else idx[0]
            sc = {k: np.concatenate(v_) for k, v_ in sc.items()}
            row = {"h": h, "period": per, "n": len(y), **{f"auc_{k}": auc(y, v_) for k, v_ in sc.items()}}
            if per == "pooled":
                for a, b in (("tfm_price", "chr_uni"), ("tfm_ret", "chr_uni"), ("tfm_multi", "chr_chosen")):
                    r = boot(frame_of(y, sc[a], ix), frame_of(y, sc[b], ix), h)
                    row.update({f"d_{a}_minus_{b}": r["d_auc"], f"d_{a}_minus_{b}_lo": r["lo"], f"d_{a}_minus_{b}_hi": r["hi"]})
            LFL.append(row)
    save(LFL, "folds_like_for_like")
    print(pd.DataFrame(LFL).round(3).T.to_string(), flush=True)

    # ---------------- Task 3: same bets? -------------------------------------------------------------------
    SB, TR, CMB = [], [], []
    for h in (1, 5):
        t, c = T[h]["tfm"], T[h]["chr"]
        idx = t.index
        st, sc_ = tl.scores("multi", 256, h, idx).to_numpy(), c["proba"].to_numpy()        # raw scores
        assert spearmanr(st, t["proba"])[0] > 0.9999, "TimesFM calibration must be monotone"
        tr = trail.reindex(idx)
        y = t["y"].to_numpy()
        for yr in ["all"] + list(range(2019, 2026)):
            m = np.ones(len(idx), bool) if yr == "all" else (idx.year == yr)
            SB.append({"h": h, "year": str(yr), "n": int(m.sum()), "spearman_tfm_vs_chr": spearmanr(st[m], sc_[m])[0],
                       "same_call_share": float((t["pred"].to_numpy()[m] == c["pred"].to_numpy()[m]).mean()),
                       "same_side_of_own_median": float(((st[m] >= np.median(st[m])) == (sc_[m] >= np.median(sc_[m]))).mean()),
                       "up_rate_tfm": float(t["pred"].to_numpy()[m].mean()), "up_rate_chr": float(c["pred"].to_numpy()[m].mean())})
            for nm, s_ in (("timesfm", st), ("chronos", sc_)):
                for col in ("ret1", "ret5", "vol20"):
                    TR.append({"h": h, "year": str(yr), "model": nm, "vs": col, "spearman": spearmanr(s_[m], tr[col].to_numpy()[m])[0]})
        rk = pd.Series(st).rank().to_numpy() + pd.Series(sc_).rank().to_numpy()
        CMB.append({"h": h, "auc_tfm": auc(y, st), "auc_chr": auc(y, sc_), "auc_rank_average": auc(y, rk)})
    save(SB, "same_bets")
    save(TR, "trailing_corr")
    save(CMB, "combine")
    print(pd.DataFrame(SB).round(3).to_string(), flush=True)
    print(pd.DataFrame(TR)[lambda d: d.year == "all"].round(3).to_string(), flush=True)
    print(pd.DataFrame(CMB).round(3).to_string(), flush=True)

    # ---------------- Task 4: the drift layers on both models ---------------------------------------------
    # Units: TimesFM forecasts log Close, so q50[h] - logC[t] and q90 - q10 are already log returns (from Close[t]);
    # Chronos's quantiles are log returns of g (Open[t+1] -> Close[t+h]). Both are log-return units, so the two extra
    # stack features are built identically: med = step-h median, spr = step-h q90 - q10. No rescaling; only the
    # reference point differs (Close[t] vs Open[t+1]).
    LAYERS = {"stack4_zms_fixed": ("stack", {"feats": "zms", "C": 0.03, "mode": "fixed"}),
              "stack_z_only_fixed": ("stack", {"feats": "z", "C": 0.03, "mode": "fixed"}),
              "platt6_offset_252_21": ("platt", {"cal": "offset", "N": 252, "R": 21})}
    HEAD = {1: "stack4_zms_fixed", 5: "platt6_offset_252_21"}
    DR, FE, GAIN = [], [], []
    drift_old = pd.read_pickle(RES_DIR / "timesfm_drift_test_predictions.pkl")
    for h in (1, 5):
        d = ds[h]
        s, e = d.period_positions("TEST")
        y = d.y[s:e]
        inp_t = dl.Inp(h)
        v, L = CHOSEN[h]
        fr = pd.concat([cf(h, v, L, p) for p in FOLDS + ["TEST"]])
        p0 = d.pos(dl.FIRST)
        assert fr.index.equals(d.index[p0:p0 + len(fr)]) and (fr["y"].to_numpy() == d.y[p0:p0 + len(fr)]).all()
        assert p0 + len(fr) == d.n, "Chronos caches must cover every row from 2013-01-01 on"
        inp_c = object.__new__(dl.Inp)                     # same attributes as Inp.__init__ sets, filled from Chronos
        inp_c.h, inp_c.ds, inp_c.p0, inp_c._lr = h, d, p0, {}
        inp_c.z, inp_c.med, inp_c.spr = (np.full(d.n, np.nan) for _ in range(3))
        inp_c.z[p0:] = tl.logit(fr["proba"].to_numpy())
        inp_c.med[p0:] = fr["q0.50"].to_numpy()
        inp_c.spr[p0:] = fr["q0.90"].to_numpy() - fr["q0.10"].to_numpy()
        frames = {"timesfm": {"static": inp_t.static("TEST")}, "chronos": {"static": T[h]["chr"]}}
        assert np.allclose(frames["timesfm"]["static"]["proba"], T[h]["tfm"]["proba"])
        for ln, (fam, spec) in LAYERS.items():
            frames["timesfm"][ln] = dl.run(inp_t, fam, spec, "TEST")
            frames["chronos"][ln] = dl.run(inp_c, fam, spec, "TEST")
        key = {1: ("stack", "stack4_zms_fixed"), 5: ("platt", "platt6_offset_252_21")}[h]
        assert np.allclose(frames["timesfm"][key[1]]["proba"], drift_old[(h, key[0])]["proba"]), "TimesFM drift not reproduced"
        for mname in ("timesfm", "chronos"):
            for ln, fr_ in frames[mname].items():
                row = {"h": h, "model": mname, "layer": ln, "headline": ln == HEAD[h], "auc": auc(fr_["y"], fr_["proba"]),
                       "n_refits": fr_.attrs.get("n_refits", np.nan), "mean_slope": fr_.attrs.get("mean_slope", np.nan)}
                if ln != "static":
                    row.update({f"d_vs_static_{k}": val for k, val in boot(fr_, frames[mname]["static"], h).items()})
                DR.append(row)
        for ln in LAYERS:
            r = boot(frames["timesfm"][ln], frames["chronos"][ln], h)
            DR.append({"h": h, "model": "timesfm_minus_chronos(layered)", "layer": ln, "headline": ln == HEAD[h],
                       "auc": r["d_auc"], **{f"d_vs_static_{k}": val for k, val in r.items()}})
        # difference of the two models' gains, same bootstrap blocks for all four score vectors
        rng = np.random.default_rng(42)
        idxs = [block_indices(len(y), block_len(h), rng) for _ in range(2000)]
        for ln in LAYERS:
            sv = [frames["timesfm"][ln]["proba"].to_numpy(), frames["timesfm"]["static"]["proba"].to_numpy(),
                  frames["chronos"][ln]["proba"].to_numpy(), frames["chronos"]["static"]["proba"].to_numpy()]
            pt = [auc(y, sv[0]) - auc(y, sv[1]), auc(y, sv[2]) - auc(y, sv[3])]
            dr = np.array([[auc(y[i], sv[0][i]) - auc(y[i], sv[1][i]), auc(y[i], sv[2][i]) - auc(y[i], sv[3][i])] for i in idxs])
            g = dr[:, 0] - dr[:, 1]
            GAIN.append({"h": h, "layer": ln, "headline": ln == HEAD[h], "gain_timesfm": pt[0], "gain_chronos": pt[1],
                         "gain_diff_tfm_minus_chr": pt[0] - pt[1], "lo": np.percentile(g, 2.5), "hi": np.percentile(g, 97.5)})
        for mname, i_ in (("timesfm", inp_t), ("chronos", inp_c)):
            for nm, arr in (("z_logit_score", i_.z), ("median_forecast_return", i_.med), ("spread_q90_q10", i_.spr)):
                FE.append({"h": h, "model": mname, "feature": nm, "auc_test": auc(y, arr[s:e]),
                           "spearman_with_trailing_vol20": spearmanr(arr[s:e], trail["vol20"].reindex(d.index[s:e]))[0],
                           "spearman_with_z": spearmanr(arr[s:e], i_.z[s:e])[0],
                           "mean_test": float(np.mean(arr[s:e])), "sd_test": float(np.std(arr[s:e]))})
    save(DR, "drift_layers")
    save(GAIN, "drift_gain_diff")
    save(FE, "drift_features")
    print(pd.DataFrame(DR).round(4).to_string(), flush=True)
    print(pd.DataFrame(GAIN).round(4).to_string(), flush=True)
    print(pd.DataFrame(FE).round(4).to_string(), flush=True)

    ck = save(chk, "label_check")
    print(f"label check: {len(ck)} frames, all ok = {bool(ck.ok.all())}; "
          f"paired-bootstrap comparisons = {NCOMP}; gain-difference bootstraps = {len(GAIN)}", flush=True)


if __name__ == "__main__":
    main()

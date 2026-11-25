"""Check every number on the deck against the saved results (drift-applied TimesFM: h=1 stack-4, h=5 platt-6).
Run from slides/ with the project interpreter:  python check_numbers.py > check_numbers_output.txt
For each claim: the value is recomputed from the frames / CSVs, formatted as on the slide, and the formatted text must
appear on the named slide (slide text or notes), read straight from timesfm3_slides.pptx. Exit code 1 if any check fails."""
import re, sys, zipfile
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
R = "../results/"
Z = zipfile.ZipFile("timesfm3_slides.pptx")

def slide_text(n):
    out = []
    for part in (f"ppt/slides/slide{n}.xml", f"ppt/notesSlides/notesSlide{n}.xml"):
        x = Z.read(part).decode("utf8")
        out.append(" ".join(re.findall(r"<a:t>([^<]*)</a:t>", x)))
    t = " ".join(out)
    return re.sub(r"\s+", " ", t.replace("&amp;", "&").replace("&quot;", '"').replace("&lt;", "<").replace("&gt;", ">"))
TXT = {n: slide_text(n) for n in range(1, 18)}
n_ok = n_bad = 0
def chk(slide, shown, why):
    """`shown` is the text that must appear on the slide; `why` documents the computed value it comes from."""
    global n_ok, n_bad
    ok = shown in TXT[slide]
    n_ok += ok; n_bad += (not ok)
    print(f"{'ok  ' if ok else 'FAIL'} slide {slide:2d}  {shown!r:48}  <- {why}")

P = pd.read_pickle(R + "timesfm_drift_test_predictions.pkl")
S1, S5, L1, L5, F1, F5 = P[(1, "stack")], P[(5, "platt")], P[(1, "static")], P[(5, "static")], P[(1, "fixed_lr")], P[(5, "fixed_lr")]
lev = pd.read_csv(R + "timesfm_drift_test_levels.csv")
def level(h, role, cfg=None):
    r = lev[(lev.horizon == h) & (lev.role == role)]
    return (r[r.cfg == cfg] if cfg else r).roc_auc.iloc[0]
auc = lambda f: roc_auc_score(f.y, f.proba)
# frames reproduce the saved levels (1e-5)
for nm, f, h, role, cfg in (("stack-4", S1, 1, "stack", "stack-4"), ("platt-6", S5, 5, "platt", "platt-6"), ("static h1", L1, 1, "static", None),
                            ("static h5", L5, 5, "static", None), ("fixed_lr h1", F1, 1, "fixed_lr", None), ("fixed_lr h5", F5, 5, "fixed_lr", None)):
    assert abs(auc(f) - level(h, role, cfg)) < 1e-5, nm
print("frames reproduce timesfm_drift_test_levels.csv to 1e-5 (stack-4, platt-6, static, fixed LR; both horizons)\n")

det = pd.read_csv(R + "timesfm_drift_year_detail.csv"); det = det[det.year != "all"]; det["year"] = det.year.astype(int)
ph = pd.read_csv(R + "timesfm_drift_year_phases.csv")
d = lambda y, h: det[(det.year == y) & (det.horizon == h)].iloc[0]
# detail CSV vs frames (independent recomputation of AUC and calls)
for y in range(2019, 2026):
    for h, f, st in ((5, S5, L5), (1, S1, L1)):
        g = f[f.index.year == y]
        assert abs(roc_auc_score(g.y, g.proba) - d(y, h).auc) < 1e-9 and abs(g.pred.mean() - d(y, h).called_up) < 1e-9
        assert abs(st[st.index.year == y].pred.mean() - d(y, h).called_up_static) < 1e-9
print("drift_year_detail.csv matches the frames (AUC, called-up share, static called-up share)\n")

print("--- slide 4: result")
chk(4, "AUC 0.52 for tomorrow", f"stack-4 h=1 AUC {auc(S1):.3f}")
assert f"{auc(S1):.2f}" == "0.52" and f"{auc(S5):.2f}" == "0.48" and f"{auc(F1):.2f}" == "0.51" and f"{auc(F5):.2f}" == "0.54"
chk(4, "it scores 0.48", f"platt-6 h=5 AUC {auc(S5):.3f}")
chk(4, "regression's 0.51", f"fixed LR h=1 {auc(F1):.3f}")
chk(4, "regression's 0.54", f"fixed LR h=5 {auc(F5):.3f}")
a5 = {y: d(y, 5).auc for y in range(2019, 2026)}
weak = sorted(a5, key=a5.get)[:3]
assert sorted(weak) == [2020, 2022, 2025], weak
chk(4, "2020, 2022, 2025", f"three lowest h=5 years {sorted(weak)}")
chk(4, "1,755–1,759 days", f"n h=5 {len(S5)}, h=1 {len(S1)}")
cmp_ = pd.read_csv(R + "timesfm_drift_test_comparisons.csv")
def c(h, m, vs): return cmp_[(cmp_.horizon == h) & (cmp_.method == m) & (cmp_.vs == vs)].iloc[0]
f3 = lambda v: f"{v:+.3f}".replace("-", "−")
r = c(1, "stack", "static"); chk(4, f"h=1 {f3(r.d_roc_auc)} [{f3(r.d_roc_auc_lo)}, {f3(r.d_roc_auc_hi)}]", "stack-4 - static, h=1, bootstrap 95%")
r = c(5, "platt", "static"); chk(4, f"h=5 {f3(r.d_roc_auc)} [{f3(r.d_roc_auc_lo)}, {f3(r.d_roc_auc_hi)}]", "platt-6 - static, h=5, bootstrap 95%")
r = c(1, "stack", "fixed_lr"); chk(4, f"fixed LR {f3(r.d_roc_auc)}", "stack-4 - fixed LR, h=1")
# Holm: none significant after correction (adjusted p >= 0.05 for the AUC gain)
for h, m in ((1, "stack"), (5, "platt")):
    for vs in ("static", "fixed_lr"):
        assert c(h, m, vs).holm_roc_auc >= 0.05, (h, m, vs)
print(f"ok       Holm-adjusted AUC p: h1 stack vs static {c(1,'stack','static').holm_roc_auc:.3f}, vs LR {c(1,'stack','fixed_lr').holm_roc_auc:.3f}; "
      f"h5 platt vs static {c(5,'platt','static').holm_roc_auc:.3f}; none < 0.05")

print("\n--- year slides 5-11 (badges, 95% block-bootstrap CIs, red/amber)")
spy = pd.read_csv(R + "timesfm_spy_years.csv").set_index("year")
red = {(y, h) for y in range(2019, 2026) for h in (1, 5) if d(y, h).auc_ci_hi < 0.5}
amber = {(y, h) for y in range(2019, 2026) for h in (1, 5) if d(y, h).auc < 0.45 and (y, h) not in red}
assert red == {(2020, 5), (2022, 5)} and amber == {(2025, 5)}, (red, amber)
print(f"ok       red = {sorted(red)}, amber = {sorted(amber)}")
for i, y in enumerate(range(2019, 2026)):
    n = 5 + i
    chk(n, f"h=5 AUC {d(y,5).auc:.2f}", f"{y} h=5 AUC {d(y,5).auc:.3f}")
    chk(n, f"h=1 AUC {d(y,1).auc:.2f}", f"{y} h=1 AUC {d(y,1).auc:.3f}")
    chk(n, f"{y} · ", "title")
chk(5, "(0.35–0.64)", f"2019 h5 CI [{d(2019,5).auc_ci_lo:.3f}, {d(2019,5).auc_ci_hi:.3f}]")
chk(7, "(0.36–0.59)", f"2021 h5 CI [{d(2021,5).auc_ci_lo:.3f}, {d(2021,5).auc_ci_hi:.3f}]")
chk(9, "interval 0.30–0.68", f"2023 h5 CI [{d(2023,5).auc_ci_lo:.3f}, {d(2023,5).auc_ci_hi:.3f}]")
chk(10, "(0.56–0.68)", f"2024 h1 CI [{d(2024,1).auc_ci_lo:.3f}, {d(2024,1).auc_ci_hi:.3f}]")
assert d(2024, 1).auc_ci_lo > 0.5
chk(6, "whole interval below one half", f"2020 h5 CI hi {d(2020,5).auc_ci_hi:.3f} < 0.5")
chk(8, "whole interval below one half", f"2022 h5 CI hi {d(2022,5).auc_ci_hi:.3f} < 0.5")
chk(10, "Raw TimesFM scored 0.42", f"2024 h1 static AUC {d(2024,1).auc_static:.3f}")
chk(10, "The 5-day AUC is a coin flip (0.54)", f"2024 h5 AUC {d(2024,5).auc:.3f}, CI [{d(2024,5).auc_ci_lo:.3f}, {d(2024,5).auc_ci_hi:.3f}] includes 0.5")
assert d(2024, 5).auc_ci_lo < 0.5 < d(2024, 5).auc_ci_hi
chk(5, "tomorrow's 0.57", f"2019 h1 AUC {d(2019,1).auc:.3f}, CI lo {d(2019,1).auc_ci_lo:.3f} < 0.5")
assert d(2019, 1).auc_ci_lo < 0.5

print("\n--- up calls, up rates (h=5)")
pct = lambda v: f"{v*100:.0f}%"
chk(5, f"Said “up” on {pct(d(2019,5).called_up)} of days, while {pct(d(2019,5).up_rate)} of 5-day windows went up", "2019 called-up share, 5-day up rate")
chk(5, "TimesFM said up on 38 percent of days", f"2019 called up {d(2019,5).called_up:.3f}")
ph5 = ph.set_index(["year", "phase"])
fall, rec = ph5.loc[(2020, "fall")], ph5.loc[(2020, "recovery")]
chk(6, f"Said “up” every day of the crash (right on {pct(fall.hit_rate)})", f"2020 fall called up {fall.called_up:.2f}, hit {fall.hit_rate:.3f}")
assert fall.called_up == 1.0
chk(6, f"“up” on {pct(rec.called_up)} of days, right on {pct(rec.hit_rate)}; always “up” gets {pct(rec.up_rate)}", f"2020 recovery called {rec.called_up:.3f}, hit {rec.hit_rate:.3f}, up rate {rec.up_rate:.3f}")
chk(6, "up on only 41 percent", f"2020 recovery called up {rec.called_up:.3f}")
chk(7, f"Said “up” on {pct(d(2021,5).called_up)} of days (raw TimesFM: {pct(d(2021,5).called_up_static)}), while {pct(d(2021,5).up_rate)} of windows went up", "2021 called up, static called up, up rate")
chk(7, "68 percent of days instead of 90", f"2021 {d(2021,5).called_up:.3f} vs static {d(2021,5).called_up_static:.3f}")
rb, fl = ph5.loc[(2022, "rebounds (all)")], ph5.loc[(2022, "falls (all)")]
chk(8, f"the market rose in {pct(rb.up_rate)} of windows, but TimesFM said “up” on only {pct(rb.called_up)} of those days", f"2022 bounces up rate {rb.up_rate:.3f}, called up {rb.called_up:.3f} ({int(round(rb.called_up*rb.n))} of {int(rb.n)} days)")
chk(8, f"higher in the falls ({fl.mean_p:.2f}) than in the bounces ({rb.mean_p:.2f})", "2022 mean P(up) falls vs bounces")
assert fl.mean_p > rb.mean_p and int(rb.n) == 90
chk(8, "71 percent of the windows", f"{rb.up_rate:.3f}")
chk(8, "only 1 percent", f"{rb.called_up:.3f}")
chk(9, f"Said “up” on {pct(d(2023,5).called_up)} of days (raw TimesFM: {pct(d(2023,5).called_up_static)}), close to the {pct(d(2023,5).up_rate)} of windows that went up", "2023 called up, static, up rate")
chk(9, "from 29 to 61 percent", f"2023 static {d(2023,5).called_up_static:.3f} -> {d(2023,5).called_up:.3f}")
f25, r25 = ph5.loc[(2025, "fall")], ph5.loc[(2025, "recovery")]
chk(11, f"Said “up” on {pct(f25.called_up)} of fall days (right on {pct(f25.hit_rate)})", f"2025 fall called {f25.called_up:.4f}, hit {f25.hit_rate:.4f}")
chk(11, f"the market rose in {pct(r25.up_rate)} of windows, but TimesFM said “up” on only {pct(r25.called_up)} of days", f"2025 recovery up {r25.up_rate:.3f}, called {r25.called_up:.3f}")
chk(11, "76 percent of the windows", f"{r25.up_rate:.3f}"); chk(11, "only 7 percent", f"{r25.called_up:.3f}")
chk(11, "most fall days", f"2025 fall called up {f25.called_up:.2f} (majority)")
assert f25.called_up > 0.5
# the 2020 / 2022 / 2025 phases are the ones used in the raw analysis (timesfm_explore_years.STRETCHES)
assert int(fall.n) == 23 and int(rec.n) == 112 and int(f25.n) == 34 and int(r25.n) == 120

print("\n--- 'Why' claims re-checked on the drift frame (h=5): turn part = fit of the score on SPY's past 20-day move; rest = residual")
for y, tn, rs in ((2020, "0.54", "0.34"), (2022, "0.51", "0.37")):
    r_ = d(y, 5)
    assert f"{r_.drift_turn_auc:.2f}" == tn and f"{r_.drift_rest_auc:.2f}" == rs, (y, r_.drift_turn_auc, r_.drift_rest_auc)
chk(6, "that part alone scores 0.54. The rest of the forecast scores 0.34", f"2020 turn {d(2020,5).drift_turn_auc:.3f}, rest {d(2020,5).drift_rest_auc:.3f}")
chk(6, "The bounce habit scores 0.54; the rest scores 0.34", "2020 same")
chk(8, "that part scores 0.51); the rest of the forecast (0.37)", f"2022 turn {d(2022,5).drift_turn_auc:.3f}, rest {d(2022,5).drift_rest_auc:.3f}")
# offset-only recalibration: same ranking as raw inside each 21-day window; yearly AUC differences stay small except where windows differ
print("ok       2025 'does not fix it': recovery called-up 7% (static 14%), AUC", f"{d(2025,5).auc:.3f} vs static {d(2025,5).auc_static:.3f}")
assert d(2025, 5).auc < 0.45
chk(11, "The drift layer does not fix", "claim: platt-6 h=5 2025 AUC stays below 0.45")

print("\n--- SPY facts quoted on year slides (timesfm_spy_years.csv)")
for y, ret, dip in ((2019, "+31%", "−7%"), (2020, None, "−34%"), (2021, "+29%", "−5%"), (2022, "−18%", "−24%"), (2023, "+26%", "−10%"), (2024, "+25%", "−8%"), (2025, None, "−19%")):
    s_ = spy.loc[y]; f = lambda v: f"{v*100:+.0f}%".replace("-", "−")
    if ret: chk(5 + y - 2019, ret, f"{y} return {s_.ret:.4f}")
    chk(5 + y - 2019, dip, f"{y} max fall {s_.max_fall:.4f}" if y != 2022 else f"{y} max fall from high {s_.max_fall:.4f}")
    assert (ret is None or f(s_.ret) == ret) and f(s_.max_fall) == dip, (y, f(s_.ret), f(s_.max_fall))

print("\n--- slide 12 (raw TimesFM, unchanged numbers) vs timesfm_rootcause_summary.csv / components")
rc = pd.read_csv(R + "timesfm_rootcause_summary.csv"); rc5 = rc[rc.h == 5].set_index("model")
chk(12, f"h=5 AUC {rc5.loc['Transformer part only','auc']:.3f} and {rc5.loc['Detrending-line slope only','auc']:.3f} (official {rc5.loc['TimesFM official','auc']:.3f})", "transformer part, detrending-line slope, official")
chk(12, f"({rc5.loc['Detrending-line slope only','auc_2025']:.2f})", "2025 detrending-line slope AUC")
assert abs(rc5.loc["TimesFM official", "auc"] - level(5, "static")) < 1e-5


print("\n--- slide 13 (TimesFM vs Chronos-2, raw scores; timesfm_vs_chronos_*.csv)")
h2h = pd.read_csv(R + "timesfm_vs_chronos_head2head.csv"); h2h = h2h[h2h.comparison == "TimesFM static - Chronos chosen"].set_index("h")
sb = pd.read_csv(R + "timesfm_vs_chronos_same_bets.csv"); sb = sb[sb.year == "all"].set_index("h")
tcn = pd.read_csv(R + "timesfm_vs_chronos_trailing_corr.csv"); tcn = tcn[tcn.year == "all"]
dft = pd.read_csv(R + "timesfm_vs_chronos_drift_features.csv")
dl = pd.read_csv(R + "timesfm_vs_chronos_drift_layers.csv")
f2 = lambda v: f"{v:.2f}".replace("-", "−")
f3s = lambda v: f"{v:+.3f}".replace("-", "−")
assert abs(h2h.loc[1, "auc_timesfm"] - level(1, "static")) < 1e-5 and abs(h2h.loc[5, "auc_timesfm"] - level(5, "static")) < 1e-5
assert (h2h.n.loc[1], h2h.n.loc[5]) == (len(L1), len(L5)) == (1759, 1755)
chk(13, f"TimesFM {h2h.loc[1,'auc_timesfm']:.3f} / {h2h.loc[5,'auc_timesfm']:.3f}, Chronos-2 {h2h.loc[1,'auc_chronos']:.3f} / {h2h.loc[5,'auc_chronos']:.3f}", "head2head.csv, test AUC h=1 / h=5 (TimesFM = static, as in levels CSV; Chronos chosen config)")
chk(13, f"{f3s(h2h.loc[1,'d_auc'])} [{f3s(h2h.loc[1,'lo'])}, {f3s(h2h.loc[1,'hi'])}]", "TimesFM - Chronos AUC, h=1, bootstrap 95%")
chk(13, f"{f3s(h2h.loc[5,'d_auc'])} [{f3s(h2h.loc[5,'lo'])}, {f3s(h2h.loc[5,'hi'])}]", "TimesFM - Chronos AUC, h=5, bootstrap 95%")
assert all(h2h.loc[h, "lo"] < 0 < h2h.loc[h, "hi"] for h in (1, 5))
chk(13, f"Spearman {f3s(sb.loc[1,'spearman_tfm_vs_chr'])} / {f3s(sb.loc[5,'spearman_tfm_vs_chr'])}", "same_bets.csv, Spearman of the two daily scores")
chk(13, f"same up/down call on {sb.loc[1,'same_call_share']*100:.1f}% / {sb.loc[5,'same_call_share']*100:.1f}% of days", "same_bets.csv, same_call_share")
tcv = lambda h, m, vs: tcn[(tcn.h == h) & (tcn.model == m) & (tcn.vs == vs)].spearman.iloc[0]
spr = lambda h, m: dft[(dft.h == h) & (dft.model == m) & (dft.feature == "spread_q90_q10")].spearman_with_z.iloc[0]
chk(13, f"5-day: {f2(tcv(5,'timesfm','ret5'))} with the trailing 5-day return", "trailing_corr.csv TimesFM ret5 h=5")
chk(13, "ignores volatility (about 0)", f"TimesFM vol20 Spearman h=1 {tcv(1,'timesfm','vol20'):+.3f}, h=5 {tcv(5,'timesfm','vol20'):+.3f}")
assert abs(tcv(1, "timesfm", "vol20")) < 0.03 and abs(tcv(5, "timesfm", "vol20")) < 0.03
chk(13, f"at h=5 (+{tcv(5,'chronos','ret5'):.2f})", "trailing_corr.csv Chronos ret5 h=5")
chk(13, f"({f2(tcv(1,'chronos','vol20'))} at h=1, {f2(tcv(5,'chronos','vol20'))} at h=5)", "trailing_corr.csv Chronos vol20")
# figure values (bars drawn from the same CSVs; rounded as labelled on the figure)
print("ok       figure bars (Spearman, h=1 / h=5): TimesFM ret1 %+.2f/%+.2f ret5 %+.2f/%+.2f vol20 %+.2f/%+.2f band %+.2f/%+.2f; Chronos ret1 %+.2f/%+.2f ret5 %+.2f/%+.2f vol20 %+.2f/%+.2f band %+.2f/%+.2f" % tuple(
    x for m in ("timesfm", "chronos") for vs in ("ret1", "ret5", "vol20", "spread") for x in ((tcv(1, m, vs), tcv(5, m, vs)) if vs != "spread" else (spr(1, m), spr(5, m)))))
dlh = dl[(dl.model == "chronos") & (dl.headline == True)].set_index("h")  # noqa: E712
assert dlh.loc[1, "layer"] == "stack4_zms_fixed" and dlh.loc[5, "layer"] == "platt6_offset_252_21"
chk(13, f"h=1 {f3s(dlh.loc[1,'d_vs_static_d_auc'])}", "drift_layers.csv: stack-4 applied to Chronos, h=1 gain over its static")
chk(13, f"h=5 {dlh.loc[5,'d_vs_static_d_auc']:+.4f}".replace("-", "−"), "drift_layers.csv: platt-6 applied to Chronos, h=5 gain")
for tx in ("Raw scores, no drift layer", "Chronos-2 from the team's chronos branch", "Exploratory", "controlled swap", "no real difference in skill", "[~20 s]"):
    chk(13, tx, "text")

print("\n--- slide 15 (why the drift layer helped; TIMESFM_DRIFT.md section 7, timesfm_drift_why_*.csv)")
w = lambda n: pd.read_csv(R + f"timesfm_drift_why_{n}.csv")
rep_ = w("repro"); print("ok       repro.csv present (the 'why' script reproduces the headline AUCs):", len(rep_), "rows")
co = w("h1_coefs").set_index("feature"); sa_ = w("h1_single_auc").set_index("feature").auc_test_2019_2025
sc = w("h1_spread_correlations"); lrc = w("h1_lr_compare").set_index("model"); bt1 = w("h1_bootstrap")
rbst = w("h5_robust_years").set_index("sample"); dec = w("h5_decomposition").set_index("method"); sig = w("h5_signals").set_index("signal"); by5 = w("h5_by_year").set_index("year")
assert abs(sa_["static TimesFM (official)"] - level(1, "static")) < 1e-5 and abs(sa_["stack-4 (fixed layer)"] - auc(S1)) < 1e-5
assert abs(dec.loc["static", "auc_all"] - level(5, "static")) < 1e-5 and abs(dec.loc["platt-6", "auc_all"] - auc(S5)) < 1e-5
chk(15, f"(h=1, {level(1,'static'):.3f} → {auc(S1):.3f})", "static -> stack-4")
chk(15, f"(h=5, {level(5,'static'):.3f} → {auc(S5):.3f})", "static -> platt-6")
chk(15, f"direction call ({f3s(co.loc['score_logit','coef_standardised'])})", "h1_coefs.csv standardised weight on the logit score")
chk(15, f"q90−q10, {f3s(co.loc['spread_q90q10','coef_standardised'])})", "h1_coefs.csv standardised weight on q90-q10 spread")
assert co.loc["spread_q90q10", "coef_standardised"] > 0 > co.loc["score_logit", "coef_standardised"]
scp = sc[(sc.period == "2019-2025")].set_index("pair").spearman
chk(15, f"Spearman {scp['spread vs VIX']:.2f} with the VIX", "h1_spread_correlations.csv, test period")
chk(15, f"{scp['spread vs rv20']:.2f} with 20-day realized vol", "h1_spread_correlations.csv, test period")
r_ = bt1[(bt1.A == "stack-4") & (bt1.B == "LR[VIX]")].iloc[0]
chk(15, f"minus an LR on the VIX: {f3s(r_.d_auc)} [{f3s(r_.lo)}, {f3s(r_.hi)}]", "h1_bootstrap.csv, stack-4 - LR[VIX]"); assert r_.lo < 0 < r_.hi
chk(15, f"({dec.loc['static','auc_within_block']:.3f} for both)", "h5_decomposition.csv within-block AUC (static = platt-6)")
assert abs(dec.loc["static", "auc_within_block"] - dec.loc["platt-6", "auc_within_block"]) < 1e-9
chk(15, f"(mean {f2(by5.loc[2022,'mean_offset'])})", "h5_by_year.csv, 2022 mean offset")
assert by5.mean_offset.idxmin() == 2022
chk(15, f"gets the same {sig.loc['z minus trailing-252 mean z (21-row refresh; no labels used)','test_auc']:.3f}", "h5_signals.csv, score minus its trailing-252 mean, no labels")
assert f"{sig.loc['z minus trailing-252 mean z (21-row refresh; no labels used)','test_auc']:.3f}" == f"{auc(S5):.3f}"  # same 21-row refresh schedule as platt-6
print("ok       (daily-refreshed no-label version: %.3f; the slide quotes the 21-row-refresh version, same schedule as platt-6)" % sig.loc['z minus trailing-252 mean z (daily; no labels used)','test_auc'])
# figure: single-feature AUCs and h=5 gain with / without 2022
print("ok       figure bars h=1 test AUC: score %.3f, median %.3f, band %.3f, VIX %.3f, rv20 %.3f, stack-4 %.3f; h=5 gain all years %+.3f, without 2022 %+.3f" % (
    sa_["score_logit"], sa_["median_return"], sa_["spread_q90q10"], sa_["VIX"], sa_["rv20"], sa_["stack-4 (fixed layer)"], rbst.loc["all years", "d_auc"], rbst.loc["excl 2022", "d_auc"]))
assert f"{rbst.loc['all years','d_auc']:+.3f}" == "+0.022" and f"{rbst.loc['excl 2022','d_auc']:+.3f}" == "-0.006"
chk(15, f"without 2022 the gain is {f3s(rbst.loc['excl 2022','d_auc'])}", "h5_robust_years.csv: platt-6 - static AUC without 2022")
for tx in ("Exploratory", "it did not fix the direction call", "volatility gauge", "2022 bear year", "[~22 s]"):
    chk(15, tx, "text")

print("\n--- slide 14 (drift detail)")
chk(14, f"stacking at h=1 ({level(1,'static'):.3f} → {auc(S1):.3f})", "static -> stack-4, h=1")
chk(14, f"recalibration at h=5 ({level(5,'static'):.3f} → {auc(S5):.3f})", "static -> platt-6, h=5")
chk(14, f"the fixed LR ({auc(F5):.3f})", "fixed LR h=5")
chk(14, f"the stack is {f3(c(1,'stack','fixed_lr').d_roc_auc)} above the LR", "stack-4 - fixed LR, h=1")
gains = [c(1, "stack", "static").d_roc_auc, c(5, "platt", "static").d_roc_auc]
assert max(gains) < 0.05
chk(14, "at most 0.04", f"largest used gain {max(gains):.3f}")
import json
ch = json.load(open(R + "timesfm_drift_choices.json"))["chosen"]
gain = {k: v["mean_d_auc"] for k, v in ch.items()}
assert max((k for k in gain if k.startswith("h1.")), key=gain.get) == "h1.stack" and max((k for k in gain if k.startswith("h5.")), key=gain.get) == "h5.platt"
print("ok       largest 2015-2018 gain: h1.stack %.4f, h5.platt %.4f (max of their horizon's families)" % (gain["h1.stack"], gain["h5.platt"]))

print("\n--- slide 16 (VIX on top of the drift layer; baseline = TimesFM + drift layer)")
vx = pd.read_csv(R + "timesfm_vixdrift_test_levels.csv")
vc = pd.read_csv(R + "timesfm_vixdrift_test_comparisons.csv")
v = lambda h, role: vx[(vx.horizon == h) & (vx.role == role)].roc_auc.iloc[0]
# the baseline D0 is the drift layer of slides 5-15
assert abs(v(1, "d0") - auc(S1)) < 1e-5 and abs(v(5, "d0") - auc(S5)) < 1e-5
chk(16, f"Baseline AUC: {v(1,'d0'):.3f} (h=1) and {v(5,'d0'):.3f} (h=5)", "D0 = stack-4 (h=1), platt-6 (h=5)")
for h in (1, 5):
    lo, hi = min(v(h, r_) for r_ in ("d1", "d2", "d3")), max(v(h, r_) for r_ in ("d1", "d2", "d3"))
    chk(16, f"h={h} {lo:.3f}–{hi:.3f}", f"D1-D3 AUC range at h={h}")
prim = vc[vc.family == "primary"]
assert len(prim) == 8 and (prim[["holm_balanced_acc", "holm_roc_auc"]].min().min() > 0.05)
chk(16, "None is significantly better than the baseline", f"smallest Holm p over the 8 primary tests per horizon: {prim[['holm_balanced_acc','holm_roc_auc']].min().min():.3f} (> 0.05)")
chk(16, f"VIX removed from the input: {v(1,'d4'):.3f} (h=1) and {v(5,'d4'):.3f} (h=5)", "D4 = drift layer on the no-VIX forecasts")
d4 = prim[prim.name == "d4"]
assert (d4.holm_roc_auc > 0.05).all() and (d4.holm_balanced_acc > 0.05).all()
assert all(abs(v(h, r_) - vx[(vx.horizon == h) & (vx.role == r_)].roc_auc.iloc[0]) < 1e-12 for h in (1, 5) for r_ in ("d0", "d4"))
chk(16, "Baseline: TimesFM + drift layer", "subtitle")
chk(16, "Exploratory", "footnote")
chk(16, "[~15 s]", "speaker notes length tag")
ptf = pd.read_pickle(R + "timesfm_vixdrift_test_predictions.pkl")
assert abs(auc(ptf[(1, "d0")]) - v(1, "d0")) < 1e-5 and abs(auc(ptf[(5, "d4")]) - v(5, "d4")) < 1e-5
assert len(ptf[(1, "d0")]) == 1759 and len(ptf[(5, "d0")]) == 1755
print("ok       slide 16 numbers recomputed from timesfm_vixdrift_test_predictions.pkl (D0, D4) and match the levels CSV to 1e-5")

print("\n--- slide 17 (bottom line)")
chk(17, f"AUC {auc(S1):.2f} (h=1) and {auc(S5):.2f} (h=5)", "stack-4, platt-6")
chk(17, "at h=1 (0.51)", f"fixed LR h=1 {auc(F1):.3f}")
chk(17, "at h=5 (0.54)", f"fixed LR h=5 {auc(F5):.3f}")
chk(17, f"2020 ({d(2020,5).auc:.2f}), 2022 ({d(2022,5).auc:.2f}), 2025 ({d(2025,5).auc:.2f})", "h=5 weak years")

print("\n--- slide 3 worked example (made-up quantiles, unchanged)")
assert "331M" not in TXT[3] and "trillion" not in TXT[3] and "1 trillion" not in TXT[3], "slide 3 must not repeat slide 2"
print("ok       slide 3: the two repeated bullets (331M parameters, 1 trillion points) are gone; slide 2 keeps them")
assert "331M" in TXT[2] and "1T+" in TXT[2]
q40, q50 = 597, 601
lower = 0.40 + (600 - q40) / (q50 - q40) * 0.10
chk(3, "about 47.5% chance of ending lower", f"interpolated {lower:.3f}"); assert abs(lower - 0.475) < 1e-9
chk(3, "P(up) ≈ 52.5%", f"{1-lower:.3f}")

print(f"\nchecks passed: {n_ok}, failed: {n_bad}")
sys.exit(1 if n_bad else 0)

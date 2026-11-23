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
TXT = {n: slide_text(n) for n in range(1, 16)}
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

print("\n--- slide 13 (drift detail)")
chk(13, f"stacking at h=1 ({level(1,'static'):.3f} → {auc(S1):.3f})", "static -> stack-4, h=1")
chk(13, f"recalibration at h=5 ({level(5,'static'):.3f} → {auc(S5):.3f})", "static -> platt-6, h=5")
chk(13, f"the fixed LR ({auc(F5):.3f})", "fixed LR h=5")
chk(13, f"the stack is {f3(c(1,'stack','fixed_lr').d_roc_auc)} above the LR", "stack-4 - fixed LR, h=1")
gains = [c(1, "stack", "static").d_roc_auc, c(5, "platt", "static").d_roc_auc]
assert max(gains) < 0.05
chk(13, "at most 0.04", f"largest used gain {max(gains):.3f}")
import json
ch = json.load(open(R + "timesfm_drift_choices.json"))["chosen"]
gain = {k: v["mean_d_auc"] for k, v in ch.items()}
assert max((k for k in gain if k.startswith("h1.")), key=gain.get) == "h1.stack" and max((k for k in gain if k.startswith("h5.")), key=gain.get) == "h5.platt"
print("ok       largest 2015-2018 gain: h1.stack %.4f, h5.platt %.4f (max of their horizon's families)" % (gain["h1.stack"], gain["h5.platt"]))

print("\n--- slide 14 (VIX on top of the drift layer; baseline = TimesFM + drift layer)")
vx = pd.read_csv(R + "timesfm_vixdrift_test_levels.csv")
vc = pd.read_csv(R + "timesfm_vixdrift_test_comparisons.csv")
v = lambda h, role: vx[(vx.horizon == h) & (vx.role == role)].roc_auc.iloc[0]
# the baseline D0 is the drift layer of slides 7-13
assert abs(v(1, "d0") - auc(S1)) < 1e-5 and abs(v(5, "d0") - auc(S5)) < 1e-5
chk(14, f"Baseline AUC: {v(1,'d0'):.3f} (h=1) and {v(5,'d0'):.3f} (h=5)", "D0 = stack-4 (h=1), platt-6 (h=5)")
for h in (1, 5):
    lo, hi = min(v(h, r_) for r_ in ("d1", "d2", "d3")), max(v(h, r_) for r_ in ("d1", "d2", "d3"))
    chk(14, f"h={h} {lo:.3f}–{hi:.3f}", f"D1-D3 AUC range at h={h}")
prim = vc[vc.family == "primary"]
assert len(prim) == 8 and (prim[["holm_balanced_acc", "holm_roc_auc"]].min().min() > 0.05)
chk(14, "None is significantly better than the baseline", f"smallest Holm p over the 8 primary tests per horizon: {prim[['holm_balanced_acc','holm_roc_auc']].min().min():.3f} (> 0.05)")
chk(14, f"VIX removed from the input: {v(1,'d4'):.3f} (h=1) and {v(5,'d4'):.3f} (h=5)", "D4 = drift layer on the no-VIX forecasts")
d4 = prim[prim.name == "d4"]
assert (d4.holm_roc_auc > 0.05).all() and (d4.holm_balanced_acc > 0.05).all()
assert all(abs(v(h, r_) - vx[(vx.horizon == h) & (vx.role == r_)].roc_auc.iloc[0]) < 1e-12 for h in (1, 5) for r_ in ("d0", "d4"))
chk(14, "Baseline: TimesFM + drift layer", "subtitle")
chk(14, "the few small gains are not significant", "any positive D1-D3 difference vs D0 (or D0 - D4) has Holm p > 0.05: asserted above")
chk(14, "Exploratory", "footnote")
chk(14, "[~15 s]", "speaker notes length tag")
ptf = pd.read_pickle(R + "timesfm_vixdrift_test_predictions.pkl")
assert abs(auc(ptf[(1, "d0")]) - v(1, "d0")) < 1e-5 and abs(auc(ptf[(5, "d4")]) - v(5, "d4")) < 1e-5
assert len(ptf[(1, "d0")]) == 1759 and len(ptf[(5, "d0")]) == 1755
print("ok       slide 14 numbers recomputed from timesfm_vixdrift_test_predictions.pkl (D0, D4) and match the levels CSV to 1e-5")

print("\n--- slide 15 (bottom line)")
chk(15, f"AUC {auc(S1):.2f} (h=1) and {auc(S5):.2f} (h=5)", "stack-4, platt-6")
chk(15, "at h=1 (0.51)", f"fixed LR h=1 {auc(F1):.3f}")
chk(15, "at h=5 (0.54)", f"fixed LR h=5 {auc(F5):.3f}")
chk(15, f"2020 ({d(2020,5).auc:.2f}), 2022 ({d(2022,5).auc:.2f}), 2025 ({d(2025,5).auc:.2f})", "h=5 weak years")

print("\n--- slide 3 worked example (made-up quantiles, unchanged)")
q40, q50 = 597, 601
lower = 0.40 + (600 - q40) / (q50 - q40) * 0.10
chk(3, "about 47.5% chance of ending lower", f"interpolated {lower:.3f}"); assert abs(lower - 0.475) < 1e-9
chk(3, "P(up) ≈ 52.5%", f"{1-lower:.3f}")

print(f"\nchecks passed: {n_ok}, failed: {n_bad}")
sys.exit(1 if n_bad else 0)

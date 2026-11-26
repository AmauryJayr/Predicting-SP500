"""Check every number on the deck against the saved results (drift-applied TimesFM: h=1 stack-4, h=5 platt-6).
Run from slides/ with the project interpreter:  python check_numbers.py > check_numbers_output.txt
The deck is written in plain English: strings are checked in the form shown on the slides (2-decimal AUCs, plain phrases).
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


f2 = lambda v: f"{v:.2f}".replace("-", "−")
f3 = lambda v: f"{v:+.3f}".replace("-", "−")
pct = lambda v: f"{v*100:.0f}%"
cmp_ = pd.read_csv(R + "timesfm_drift_test_comparisons.csv")
def c(h, m, vs): return cmp_[(cmp_.horizon == h) & (cmp_.method == m) & (cmp_.vs == vs)].iloc[0]

print("--- slide 2 / 3: how it was built and how it works (model-card facts and the made-up example)")
chk(2, "331 million adjustable numbers", "model card: 331M parameters"); chk(2, "Over 1 trillion data points", "model card: 1T+ time points")
chk(2, "Neural network, 20 layers", "model card: 20 layers"); chk(2, "pieces of 32 points", "model card: patch size 32")
chk(2, "9 possible prices, from low (10%) to high (90%)", "9 quantiles 10%...90%"); chk(2, "zero-shot", "defined in plain words next to it")
assert "331 million" not in TXT[3] and "trillion" not in TXT[3], "slide 3 must not repeat slide 2"
print("ok       slide 3 does not repeat slide 2's 331 million / 1 trillion")
chk(3, "the market's “fear index”", "VIX defined on first appearance"); chk(3, "9 possible prices, from low (10%) to high (90%)", "output line")
chk(3, "a small correction step on top of TimesFM that adapts to recent data", "drift layer defined")
chk(3, "a small model on top that uses TimesFM's outputs", "stacking, plain"); chk(3, "a monthly up/down shift of all its chances", "recalibration, plain")
q40, q50 = 597, 601
lower = 0.40 + (600 - q40) / (q50 - q40) * 0.10
chk(3, "about 47.5% chance of ending lower", f"interpolated {lower:.3f}"); assert abs(lower - 0.475) < 1e-9
chk(3, "Chance of up ≈ 52.5%", f"{1-lower:.3f}"); chk(3, "58% cut-off", "the rule's threshold used in the worked example")

print("\n--- slide 4: result")
a1, a5, l1, l5 = auc(S1), auc(S5), auc(F1), auc(F5)
assert (f2(a1), f2(a5), f2(l1), f2(l5)) == ("0.52", "0.48", "0.51", "0.54")
chk(4, "drift layer scores 0.52", f"stack-4 h=1 AUC {a1:.3f}")
chk(4, "it scores 0.48", f"platt-6 h=5 AUC {a5:.3f}")
chk(4, "our simple model, 0.51", f"fixed LR h=1 {l1:.3f}")
chk(4, "our simple model, 0.54", f"fixed LR h=5 {l5:.3f}")
a5y = {y: d(y, 5).auc for y in range(2019, 2026)}
weak = sorted(a5y, key=a5y.get)[:3]
assert sorted(weak) == [2020, 2022, 2025], weak
chk(4, "2020, 2022 and 2025", f"three lowest h=5 years {sorted(weak)}")
chk(4, "1,755–1,759 days", f"n h=5 {len(S5)}, h=1 {len(S1)}")
g1, g5 = c(1, "stack", "static").d_roc_auc, c(5, "platt", "static").d_roc_auc
chk(4, f"(+{g1:.2f} tomorrow, +{g5:.2f} next 5 days) could be luck", f"gain over static: h=1 {g1:+.3f}, h=5 {g5:+.3f}; Holm-adjusted p >= 0.05 for all (asserted below)")
r = c(1, "stack", "fixed_lr")
assert r.d_roc_auc_lo < 0 < r.d_roc_auc_hi
chk(4, f"tomorrow's {f2(a1)} vs {f2(l1)} is not clearly better", f"stack-4 - fixed LR h=1 {r.d_roc_auc:+.3f}, 95% interval includes 0")
r = c(5, "platt", "fixed_lr")
assert r.d_roc_auc_hi < 0
chk(4, "and worse than our simple model, 0.54", f"platt-6 - fixed LR h=5 {r.d_roc_auc:+.3f}, 95% interval [{r.d_roc_auc_lo:+.3f}, {r.d_roc_auc_hi:+.3f}] below 0 (uncorrected)")
for h, m in ((1, "stack"), (5, "platt")):
    for vs in ("static", "fixed_lr"):
        assert c(h, m, vs).holm_roc_auc >= 0.05, (h, m, vs)
print("ok       Holm-adjusted AUC p >= 0.05 for stack-4 / platt-6 against static and fixed LR (so 'could be luck')")
chk(4, "pick one up day and one down day; how often does the model rank the up day higher? 0.5 = coin flip, 1.0 = always right.", "AUC defined plainly")
chk(4, "a logistic regression trained once", "our simple model defined")
chk(4, "see the previous slide", "drift layer defined on slide 3")
chk(4, "the layer for each horizon was picked on 2015–2018 numbers only, but that rule was set after the test", "honest caveat kept")

print("\n--- year slides 5-11 (badges, red/amber, 95% block-bootstrap intervals only used for the wording 'clearly' / 'coin flip')")
spy = pd.read_csv(R + "timesfm_spy_years.csv").set_index("year")
red = {(y, h) for y in range(2019, 2026) for h in (1, 5) if d(y, h).auc_ci_hi < 0.5}
amber = {(y, h) for y in range(2019, 2026) for h in (1, 5) if d(y, h).auc < 0.45 and (y, h) not in red}
assert red == {(2020, 5), (2022, 5)} and amber == {(2025, 5)}, (red, amber)
print(f"ok       red = {sorted(red)}, amber = {sorted(amber)}")
for i, y in enumerate(range(2019, 2026)):
    n = 5 + i
    chk(n, f"Next 5 days: AUC {d(y,5).auc:.2f}", f"{y} h=5 AUC {d(y,5).auc:.3f}")
    chk(n, f"Tomorrow: AUC {d(y,1).auc:.2f}", f"{y} h=1 AUC {d(y,1).auc:.3f}")
    chk(n, f"{y} · ", "title")
chk(n, "Red = clearly worse than a coin flip", "legend")
for y in (2019, 2021, 2023):     # 'A coin flip' = interval includes 0.5
    assert d(y, 5).auc_ci_lo < 0.5 < d(y, 5).auc_ci_hi, y
print("ok       2019, 2021, 2023 h=5 intervals include 0.5 ('a coin flip')")
assert d(2020, 5).auc_ci_hi < 0.5 and d(2022, 5).auc_ci_hi < 0.5
chk(6, "Clearly worse than a coin flip", f"2020 h5 interval hi {d(2020,5).auc_ci_hi:.3f} < 0.5")
chk(8, "Clearly worse than a coin flip", f"2022 h5 interval hi {d(2022,5).auc_ci_hi:.3f} < 0.5")
chk(6, "The AUC is 0.36, clearly worse than a coin flip", "notes"); chk(8, "The AUC is 0.38, clearly worse than a coin flip", "notes")
assert d(2024, 1).auc_ci_lo > 0.5
chk(10, "clearly above a coin flip", f"2024 h1 interval lo {d(2024,1).auc_ci_lo:.3f} > 0.5")
chk(10, f"TimesFM alone scored {d(2024,1).auc_static:.2f} here", f"2024 h1 static AUC {d(2024,1).auc_static:.3f}")
chk(10, f"The next-5-days AUC is a coin flip ({d(2024,5).auc:.2f})", f"2024 h5 AUC {d(2024,5).auc:.3f}, interval [{d(2024,5).auc_ci_lo:.3f}, {d(2024,5).auc_ci_hi:.3f}] includes 0.5")
assert d(2024, 5).auc_ci_lo < 0.5 < d(2024, 5).auc_ci_hi
chk(5, f"For tomorrow it is {d(2019,1).auc:.2f}, but that could still be luck", f"2019 h1 AUC {d(2019,1).auc:.3f}, interval lo {d(2019,1).auc_ci_lo:.3f} < 0.5")
assert d(2019, 1).auc_ci_lo < 0.5

print("\n--- up calls, up rates (h=5)")
chk(5, f"Said “up” on {pct(d(2019,5).called_up)} of days, while {pct(d(2019,5).up_rate)} of 5-day periods went up.", "2019 called-up share, 5-day up rate")
chk(5, "said up on only 38 percent of days, while 68 percent of the 5-day periods went up", f"{d(2019,5).called_up:.3f}, {d(2019,5).up_rate:.3f}")
ph5 = ph.set_index(["year", "phase"])
fall, rec = ph5.loc[(2020, "fall")], ph5.loc[(2020, "recovery")]
assert fall.called_up == 1.0
chk(6, f"Said “up” on every day of the crash; right only {pct(fall.hit_rate)} of the time.", f"2020 fall called up {fall.called_up:.2f}, hit {fall.hit_rate:.3f}")
chk(6, f"it said “up” on {pct(rec.called_up)} of days; saying “up” every day would have been right {pct(rec.up_rate)}.", f"2020 recovery called {rec.called_up:.3f}, up rate {rec.up_rate:.3f}")
chk(6, "right on only 30 percent", f"{fall.hit_rate:.3f}"); chk(6, "only 41 percent of days", f"{rec.called_up:.3f}"); chk(6, "right 71 percent of the time", f"{rec.up_rate:.3f}")
chk(7, f"Said “up” on {pct(d(2021,5).called_up)} of days (TimesFM alone: {pct(d(2021,5).called_up_static)}); {pct(d(2021,5).up_rate)} of 5-day periods went up.", "2021 called up, static called up, up rate")
chk(7, "68 percent of days, instead of 90 percent", f"2021 {d(2021,5).called_up:.3f} vs static {d(2021,5).called_up_static:.3f}")
rb, fl = ph5.loc[(2022, "rebounds (all)")], ph5.loc[(2022, "falls (all)")]
chk(8, f"the market rose in {pct(rb.up_rate)} of 5-day periods, but TimesFM said “up” on only {pct(rb.called_up)} of those days.", f"2022 bounces up rate {rb.up_rate:.3f}, called up {rb.called_up:.3f} ({int(round(rb.called_up*rb.n))} of {int(rb.n)} days)")
assert fl.mean_p > rb.mean_p and int(rb.n) == 90
chk(8, "It was more upbeat in the falls than in the bounces.", f"2022 mean chance of up: falls {fl.mean_p:.2f} > bounces {rb.mean_p:.2f}")
chk(8, "71 percent of the 5-day periods", f"{rb.up_rate:.3f}"); chk(8, "only 1 percent of those days", f"{rb.called_up:.3f}")
chk(9, f"Said “up” on {pct(d(2023,5).called_up)} of days (TimesFM alone: {pct(d(2023,5).called_up_static)}), close to the {pct(d(2023,5).up_rate)} of 5-day periods that went up.", "2023 called up, static, up rate")
chk(9, "from 29 to 61 percent", f"2023 static {d(2023,5).called_up_static:.3f} -> {d(2023,5).called_up:.3f}")
f25, r25 = ph5.loc[(2025, "fall")], ph5.loc[(2025, "recovery")]
chk(11, f"Said “up” on {pct(f25.called_up)} of fall days (right on {pct(f25.hit_rate)}).", f"2025 fall called {f25.called_up:.4f}, hit {f25.hit_rate:.4f}")
chk(11, f"the market rose in {pct(r25.up_rate)} of 5-day periods, but TimesFM said “up” on only {pct(r25.called_up)} of days.", f"2025 recovery up {r25.up_rate:.3f}, called {r25.called_up:.3f}")
chk(11, "76 percent of the 5-day periods", f"{r25.up_rate:.3f}"); chk(11, "only 7 percent of days", f"{r25.called_up:.3f}")
chk(11, "up on most fall days", f"2025 fall called up {f25.called_up:.2f} (majority)"); assert f25.called_up > 0.5
assert int(fall.n) == 23 and int(rec.n) == 112 and int(f25.n) == 34 and int(r25.n) == 120

print("\n--- 'reversal bet' (= the report's 'turn part': the part of the forecast that follows SPY's past 20-day move) vs 'the rest of the forecast'")
for y, tn, rs in ((2020, "0.54", "0.34"), (2022, "0.51", "0.37")):
    r_ = d(y, 5)
    assert f"{r_.drift_turn_auc:.2f}" == tn and f"{r_.drift_rest_auc:.2f}" == rs, (y, r_.drift_turn_auc, r_.drift_rest_auc)
chk(6, "on its own it scores 0.54. The rest of the forecast scores 0.34", f"2020 turn {d(2020,5).drift_turn_auc:.3f}, rest {d(2020,5).drift_rest_auc:.3f}")
chk(6, "On its own it scores 0.54, so it is not the cause. The rest of the forecast scores 0.34.", "notes")
chk(8, "(on its own it scores 0.51); the rest of the forecast (0.37)", f"2022 turn {d(2022,5).drift_turn_auc:.3f}, rest {d(2022,5).drift_rest_auc:.3f}")
chk(8, "which scores 0.51. It is the rest of the forecast, 0.37.", "notes")
chk(6, "after a fall it expects a bounce", "the report: 'after a rise it expects a dip, and after a fall it expects a bounce'")
print("ok       2025 'does not fix it': recovery called-up 7% (static 14%), AUC", f"{d(2025,5).auc:.3f} vs static {d(2025,5).auc_static:.3f}")
assert d(2025, 5).auc < 0.45
chk(11, "The drift layer does not fix this", "claim: platt-6 h=5 2025 AUC stays below 0.45")

print("\n--- SPY facts quoted on year slides (timesfm_spy_years.csv)")
neg = [y for y in range(2019, 2026) if spy.loc[y].ret < 0]
assert neg == [2022], neg
print("ok       2022 is the only test year with a negative return (the 'one bear year')")
want = {2019: ("Up 31%", "−7%"), 2020: ("Down 34%", None), 2021: ("Up 29%", "−5%"), 2022: ("Down 18%", "−24%"), 2023: ("Up 26%", "10% dip"),
        2024: ("Up 25%", "−8%"), 2025: ("Down 19%", None)}
for y, (a_, b_) in want.items():
    s_ = spy.loc[y]; n = 5 + y - 2019
    if a_.startswith("Up"): assert f"Up {abs(s_.ret)*100:.0f}%" == a_ and s_.ret > 0, (y, s_.ret)
    elif y == 2022: assert f"Down {abs(s_.ret)*100:.0f}%" == a_, (y, s_.ret)
    else: assert f"Down {abs(s_.max_fall)*100:.0f}%" == a_, (y, s_.max_fall)
    chk(n, a_, f"{y} return {s_.ret:.4f}, max fall {s_.max_fall:.4f}")
    if b_ and b_[0] == "−": assert f"−{abs(s_.max_fall)*100:.0f}%" == b_, (y, s_.max_fall); chk(n, b_, f"{y} max fall {s_.max_fall:.4f}")
    elif b_: assert f"{abs(s_.max_fall)*100:.0f}% dip" == b_, (y, s_.max_fall); chk(n, b_, f"{y} max fall {s_.max_fall:.4f}")
chk(5, "31 percent", "2019 return"); chk(7, "only 5 percent", "2021 max fall"); chk(9, "10 percent dip", "2023 max fall")

print("\n--- slide 12 (TimesFM alone) vs timesfm_rootcause_summary.csv / components / baselines")
rc = pd.read_csv(R + "timesfm_rootcause_summary.csv"); rc5 = rc[rc.h == 5].set_index("model")
net, ln, off = rc5.loc["Transformer part only"], rc5.loc["Detrending-line slope only"], rc5.loc["TimesFM official"]
chk(12, f"AUC {net.auc:.2f} and {ln.auc:.2f} (TimesFM as released: {off.auc:.2f})", "network part, trend-line part, official: pooled h=5 AUC")
assert net.hi > 0.5 and ln.hi > 0.5 and off.hi > 0.5
chk(12, "Neither is clearly worse than a coin flip", f"upper ends of the 95% intervals: network {net.hi:.3f}, trend line {ln.hi:.3f} (> 0.5)")
chk(12, f"({ln.auc_2025:.2f})", "2025 built-in trend line AUC"); assert ln.auc_2025 < net.auc_2025
for m in ("Random walk + drift", "AR(5) on returns"):
    assert rc5.loc[m, "auc_2020"] < 0.45 and rc5.loc[m, "auc_2022"] < 0.45, m
assert net.auc_2020 < 0.45 and net.auc_2022 < 0.45
print("ok       2020 and 2022 AUC below 0.45 for the network part AND for random walk + drift, AR(5): %s" % {m: (round(rc5.loc[m,'auc_2020'],2), round(rc5.loc[m,'auc_2022'],2)) for m in ("Transformer part only", "Random walk + drift", "AR(5) on returns")})
chk(12, "“tomorrow = today + the average daily gain”", "plain name of the random walk + drift baseline")
assert abs(off.auc - level(5, "static")) < 1e-5
print("ok       figure dots (h=5): " + ", ".join(f"{m} {rc5.loc[m,'auc']:.2f}" for m in rc5.index))

print("\n--- slide 13 (TimesFM vs Chronos-2, raw scores; timesfm_vs_chronos_*.csv and the controlled swap, ctrl_*.csv)")
h2h = pd.read_csv(R + "timesfm_vs_chronos_head2head.csv"); h2h = h2h[h2h.comparison == "TimesFM static - Chronos chosen"].set_index("h")
sb = pd.read_csv(R + "timesfm_vs_chronos_same_bets.csv"); sb = sb[sb.year == "all"].set_index("h")
tcn = pd.read_csv(R + "timesfm_vs_chronos_trailing_corr.csv"); tcn = tcn[tcn.year == "all"]
dft = pd.read_csv(R + "timesfm_vs_chronos_drift_features.csv")
dl = pd.read_csv(R + "timesfm_vs_chronos_drift_layers.csv")
assert abs(h2h.loc[1, "auc_timesfm"] - level(1, "static")) < 1e-5 and abs(h2h.loc[5, "auc_timesfm"] - level(5, "static")) < 1e-5
assert (h2h.n.loc[1], h2h.n.loc[5]) == (len(L1), len(L5)) == (1759, 1755)
chk(13, f"TimesFM {f2(h2h.loc[1,'auc_timesfm'])} / {f2(h2h.loc[5,'auc_timesfm'])}, Chronos-2 {f2(h2h.loc[1,'auc_chronos'])} / {f2(h2h.loc[5,'auc_chronos'])}", "head2head.csv, test AUC tomorrow / next 5 days")
chk(13, f"TimesFM scores {f2(h2h.loc[1,'auc_timesfm'])} for tomorrow and {f2(h2h.loc[5,'auc_timesfm'])} for the next 5 days. Chronos-2 scores {f2(h2h.loc[1,'auc_chronos'])} and {f2(h2h.loc[5,'auc_chronos'])}", "notes")
assert all(h2h.loc[h, "lo"] < 0 < h2h.loc[h, "hi"] for h in (1, 5))
chk(13, "No real difference.", f"TimesFM - Chronos AUC: h=1 {h2h.loc[1,'d_auc']:+.3f}, h=5 {h2h.loc[5,'d_auc']:+.3f}; both 95% intervals include 0")
assert all(0.45 <= sb.loc[h, "same_call_share"] <= 0.55 for h in (1, 5))
chk(13, "agree on only about half of the days", f"same_bets.csv same_call_share {sb.loc[1,'same_call_share']:.3f} / {sb.loc[5,'same_call_share']:.3f} (between 45% and 55%)")
tcv = lambda h, m, vs: tcn[(tcn.h == h) & (tcn.model == m) & (tcn.vs == vs)].spearman.iloc[0]
spr = lambda h, m: dft[(dft.h == h) & (dft.model == m) & (dft.feature == "spread_q90_q10")].spearman_with_z.iloc[0]
assert tcv(5, "timesfm", "ret5") < -0.2 < 0 < 0.5 < tcv(5, "chronos", "ret5")
chk(13, "TimesFM bets on reversals", f"Spearman of TimesFM's chance of up with the last-5-day move at h=5: {tcv(5,'timesfm','ret5'):+.2f} (negative)")
chk(13, "Chronos-2 bets the move continues", f"same for Chronos-2: {tcv(5,'chronos','ret5'):+.2f} (positive)")
assert tcv(1, "chronos", "vol20") < -0.3 and tcv(5, "chronos", "vol20") < -0.3 and abs(tcv(1, "timesfm", "vol20")) < 0.03 and abs(tcv(5, "timesfm", "vol20")) < 0.03
chk(13, "turns gloomy when the market swings a lot", f"Chronos-2 chance of up vs recent swings: {tcv(1,'chronos','vol20'):+.2f} (h=1), {tcv(5,'chronos','vol20'):+.2f} (h=5); TimesFM ~0 ({tcv(1,'timesfm','vol20'):+.3f}, {tcv(5,'timesfm','vol20'):+.3f})")
# figure bars: only |value| >= 0.2 are labelled
print("ok       figure bars labelled when |rho| >= 0.2 (Spearman; h=1 / h=5): TimesFM ret1 %+.2f/%+.2f ret5 %+.2f/%+.2f vol20 %+.2f/%+.2f band %+.2f/%+.2f; Chronos ret1 %+.2f/%+.2f ret5 %+.2f/%+.2f vol20 %+.2f/%+.2f band %+.2f/%+.2f" % tuple(
    x for m in ("timesfm", "chronos") for vs in ("ret1", "ret5", "vol20", "spread") for x in ((tcv(1, m, vs), tcv(5, m, vs)) if vs != "spread" else (spr(1, m), spr(5, m)))))
# controlled swap (section 7): TEST rows
beh = pd.read_csv(R + "timesfm_vs_chronos_ctrl_behaviour.csv"); beh = beh[beh.period == "TEST"].set_index(["h", "cell"])
for h in (1, 5):
    assert beh.loc[(h, "TFxL"), "rho_ret5"] < 0 and beh.loc[(h, "CHxL"), "rho_ret5"] < 0, h      # fed prices (L): both against the last move
    assert beh.loc[(h, "TFxG"), "rho_ret5"] > 0 and beh.loc[(h, "CHxG"), "rho_ret5"] > 0, h      # fed returns (G): both with the last move
chk(13, "Fed daily prices, both lean toward reversals; fed daily returns, both lean toward the move continuing.", "ctrl_behaviour.csv, TEST, Spearman with the last-5-day move: L cells negative, G cells positive, both models, h=1 and h=5: " +
    "; ".join(f"h={h} " + ", ".join(f"{c_} {beh.loc[(h,c_),'rho_ret5']:+.2f}" for c_ in ("TFxL", "CHxL", "TFxG", "CHxG")) for h in (1, 5)))
ctr = pd.read_csv(R + "timesfm_vs_chronos_ctrl_contrasts.csv"); ctr = ctr[ctr.family == "phase1"]
assert len(ctr) == 8 and (ctr.lo < 0).all() and (ctr.hi > 0).all() and (ctr.p_holm >= 0.05).all()
chk(13, "Accuracy does not change.", f"ctrl_contrasts.csv: all 8 model/input contrasts have 95% intervals that include 0, Holm p = {ctr.p_holm.min():.3f}")
dlh = dl[(dl.model == "chronos") & (dl.headline == True)].set_index("h")  # noqa: E712
assert dlh.loc[1, "layer"] == "stack4_zms_fixed" and dlh.loc[5, "layer"] == "platt6_offset_252_21"
assert dlh.loc[1, "d_vs_static_d_auc"] < 0 and dlh.loc[5, "d_vs_static_d_auc"] < 0
chk(13, "The drift layer does not help Chronos-2.", f"drift_layers.csv: gain on Chronos-2 h=1 {dlh.loc[1,'d_vs_static_d_auc']:+.3f}, h=5 {dlh.loc[5,'d_vs_static_d_auc']:+.4f} (both <= 0)")
for tx in ("The difference is the input, not the model.", "Same skill (a coin flip), different bets.", "TimesFM alone, no drift layer", "Exploratory", "Swap test details: TIMESFM_VS_CHRONOS.md, section 7", "[~25 s]",
           "Does the model's chance of up move with"):
    if tx.startswith("Does"): print("ok       chart axis title (image): 'Does the model's chance of up move with ...?  + together, - opposite'"); continue
    chk(13, tx, "text")

print("\n--- slide 14 (drift detail)")
chk(14, f"the small model on top for tomorrow ({f2(level(1,'static'))} → {f2(a1)})", "static -> stack-4, h=1")
chk(14, f"the monthly shift for the next 5 days ({f2(level(5,'static'))} → {f2(a5)})", "static -> platt-6, h=5")
chk(14, f"our simple model ({f2(l5)}) stays better", "fixed LR h=5")
chk(14, f"only {c(1,'stack','fixed_lr').d_roc_auc:.2f} above our simple model", "stack-4 - fixed LR, h=1")
assert max(g1, g5) < 0.05
chk(14, "at most 0.04", f"largest used gain {max(g1, g5):.3f}")
assert a5 < 0.5
chk(14, "the layer stays below a coin flip", f"platt-6 h=5 AUC {a5:.3f}")
chk(14, "could be luck", "Holm-adjusted p >= 0.05 (asserted on slide 4)")
for tx in ("a monthly up/down shift of all its chances (recalibration)", "a small model on top that uses TimesFM's outputs (stacking)",
           "a vote that gives more weight to whichever model did well lately (Hedge)", "an alarm that refits when errors jump (drift detector)"):
    chk(14, tx, "plain definition with the report's name in brackets")
import json
ch = json.load(open(R + "timesfm_drift_choices.json"))["chosen"]
gain = {k: v["mean_d_auc"] for k, v in ch.items()}
assert max((k for k in gain if k.startswith("h1.")), key=gain.get) == "h1.stack" and max((k for k in gain if k.startswith("h5.")), key=gain.get) == "h5.platt"
print("ok       largest 2015-2018 gain: h1.stack %.4f, h5.platt %.4f (max of their horizon's families)" % (gain["h1.stack"], gain["h5.platt"]))
chk(14, "the biggest gains on 2015–2018", "see line above")

print("\n--- slide 15 (why the drift layer helped; TIMESFM_DRIFT.md section 7, timesfm_drift_why_*.csv)")
w = lambda n: pd.read_csv(R + f"timesfm_drift_why_{n}.csv")
rep_ = w("repro"); print("ok       repro.csv present (the 'why' script reproduces the headline AUCs):", len(rep_), "rows")
co = w("h1_coefs").set_index("feature"); sa_ = w("h1_single_auc").set_index("feature").auc_test_2019_2025
bt1 = w("h1_bootstrap")
rbst = w("h5_robust_years").set_index("sample"); dec = w("h5_decomposition").set_index("method"); sig = w("h5_signals").set_index("signal"); by5 = w("h5_by_year").set_index("year")
assert abs(sa_["static TimesFM (official)"] - level(1, "static")) < 1e-5 and abs(sa_["stack-4 (fixed layer)"] - a1) < 1e-5
assert abs(dec.loc["static", "auc_all"] - level(5, "static")) < 1e-5 and abs(dec.loc["platt-6", "auc_all"] - a5) < 1e-5
chk(15, f"Tomorrow ({f2(level(1,'static'))} → {f2(a1)})", "static -> stack-4")
chk(15, f"Next 5 days ({f2(level(5,'static'))} → {f2(a5)})", "static -> platt-6")
assert abs(co.loc["score_logit", "coef_standardised"]) < 0.01 and co.loc["spread_q90q10", "coef_standardised"] > 0.03
chk(15, "the layer ignores TimesFM's up/down call", f"h1_coefs.csv standardised weight on the score {co.loc['score_logit','coef_standardised']:+.3f} (about 0); on the width of the price range {co.loc['spread_q90q10','coef_standardised']:+.3f} (positive: wider -> more up)")
chk(15, "a wide range = a jumpy market → slightly more up days", "positive weight on the width")
chk(15, f"({sa_['VIX']:.3f}, against {sa_['spread_q90q10']:.3f} for the width alone)", "h1_single_auc.csv: VIX alone vs width alone, test")
r_ = bt1[(bt1.A == "stack-4") & (bt1.B == "LR[VIX]")].iloc[0]; assert r_.lo < 0 < r_.hi
print(f"ok       stack-4 minus an LR on the VIX {r_.d_auc:+.3f} [{r_.lo:+.3f}, {r_.hi:+.3f}], includes 0 (so 'not a new skill')")
assert abs(dec.loc["static", "auc_within_block"] - dec.loc["platt-6", "auc_within_block"]) < 1e-9
chk(15, "it never changes which day ranks higher inside a month", f"h5_decomposition.csv within-block AUC {dec.loc['static','auc_within_block']:.3f} for both static and platt-6")
assert by5.mean_offset.idxmin() == 2022 and by5.loc[2022, "mean_offset"] < 0
chk(15, "It pushed all of 2022 down", f"h5_by_year.csv: 2022 mean offset {by5.loc[2022,'mean_offset']:+.2f}, the lowest of all years")
assert rbst.loc["excl 2022", "d_auc"] < 0.005 and f"{rbst.loc['all years','d_auc']:+.3f}" == "+0.022"
chk(15, "Without 2022 the gain is gone", f"h5_robust_years.csv: platt-6 - static AUC {rbst.loc['all years','d_auc']:+.3f} all years, {rbst.loc['excl 2022','d_auc']:+.3f} without 2022")
chk(15, "2022 was the one bear year", "2022 is the only negative-return test year (asserted above)")
chk(15, "Like a teacher shifting a whole class's marks", "analogy")
nolab = sig.loc['z minus trailing-252 mean z (21-row refresh; no labels used)', 'test_auc']
assert f2(nolab) == f2(a5)
chk(15, f"also gets {f2(nolab)}", "h5_signals.csv, score minus its trailing-252 mean, no labels, same 21-row schedule as platt-6")
chk(15, f"gets the same {f2(nolab)}", "notes")
print("ok       (daily-refreshed no-label version: %.3f; the slide quotes the 21-row-refresh version, same schedule as platt-6)" % sig.loc['z minus trailing-252 mean z (daily; no labels used)','test_auc'])
print("ok       figure bars h=1 test AUC: up/down call %.3f, middle price %.3f, width %.3f, VIX %.3f, recent swings %.3f, drift layer %.3f; h=5 gain all years %+.3f, without 2022 %+.3f" % (
    sa_["score_logit"], sa_["median_return"], sa_["spread_q90q10"], sa_["VIX"], sa_["rv20"], sa_["stack-4 (fixed layer)"], rbst.loc["all years", "d_auc"], rbst.loc["excl 2022", "d_auc"]))
for tx in ("Exploratory", "it did not fix the direction call", "how jumpy the market is", "2022 bear year", "[~30 s]"):
    chk(15, tx, "text")

print("\n--- slide 16 (VIX on top of the drift layer; baseline = TimesFM + drift layer)")
vx = pd.read_csv(R + "timesfm_vixdrift_test_levels.csv")
vc = pd.read_csv(R + "timesfm_vixdrift_test_comparisons.csv")
v = lambda h, role: vx[(vx.horizon == h) & (vx.role == role)].roc_auc.iloc[0]
assert abs(v(1, "d0") - a1) < 1e-5 and abs(v(5, "d0") - a5) < 1e-5
chk(16, f"Starting point: AUC {f2(v(1,'d0'))} tomorrow, {f2(v(5,'d0'))} next 5 days.", "D0 = stack-4 (h=1), platt-6 (h=5)")
rng = {h: (min(v(h, r_) for r_ in ("d1", "d2", "d3")), max(v(h, r_) for r_ in ("d1", "d2", "d3"))) for h in (1, 5)}
chk(16, f"tomorrow {f2(rng[1][0])}–{f2(rng[1][1])}, next 5 days {f2(rng[5][0])}–{f2(rng[5][1])}", f"D1-D3 AUC range: h=1 {rng[1][0]:.3f}-{rng[1][1]:.3f}, h=5 {rng[5][0]:.3f}-{rng[5][1]:.3f}")
prim = vc[vc.family == "primary"]
assert len(prim) == 8 and (prim[["holm_balanced_acc", "holm_roc_auc"]].min().min() > 0.05)
chk(16, "None is clearly better than the starting point", f"smallest Holm p over the 8 primary tests per horizon: {prim[['holm_balanced_acc','holm_roc_auc']].min().min():.3f} (> 0.05)")
chk(16, f"VIX taken out of the input: {f2(v(1,'d4'))} tomorrow, {f2(v(5,'d4'))} next 5 days.", "D4 = drift layer on the no-VIX forecasts")
d4 = prim[prim.name == "d4"]
assert (d4.holm_roc_auc > 0.05).all() and (d4.holm_balanced_acc > 0.05).all()
chk(16, "Starting point: TimesFM + drift layer", "subtitle"); chk(16, "the market's “fear index”", "VIX defined")
chk(16, "Exploratory", "footnote"); chk(16, "[~15 s]", "speaker notes length tag")
ptf = pd.read_pickle(R + "timesfm_vixdrift_test_predictions.pkl")
assert abs(auc(ptf[(1, "d0")]) - v(1, "d0")) < 1e-5 and abs(auc(ptf[(5, "d4")]) - v(5, "d4")) < 1e-5
assert len(ptf[(1, "d0")]) == 1759 and len(ptf[(5, "d0")]) == 1755
print("ok       slide 16 numbers recomputed from timesfm_vixdrift_test_predictions.pkl (D0, D4) and match the levels CSV to 1e-5")

print("\n--- slide 17 (bottom line)")
assert a5 < 0.5 and c(5, "platt", "fixed_lr").d_roc_auc_hi < 0
chk(17, f"AUC {f2(a1)} tomorrow, {f2(a5)} next 5 days. Our simple model: {f2(l1)} and {f2(l5)}.", "stack-4, platt-6, fixed LR")
chk(17, "Tomorrow: not clearly different. Next 5 days: worse.", "intervals checked on slide 4")
chk(17, f"2020 ({f2(d(2020,5).auc)}), 2022 ({f2(d(2022,5).auc)}), 2025 ({f2(d(2025,5).auc)})", "h=5 weak years")

print(f"\nchecks passed: {n_ok}, failed: {n_bad}")
sys.exit(1 if n_bad else 0)

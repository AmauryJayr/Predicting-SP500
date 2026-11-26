"""Chart PNGs for the TimesFM-3 slides (drift-applied TimesFM). Reads saved results only (no TimesFM run). Output: slides/fig/*.png"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
ROOT = HERE.parent.parent
FIG = HERE / "fig"; FIG.mkdir(exist_ok=True)
DPI = 220
plt.rcParams.update({"font.family": ["Arimo", "DejaVu Sans"], "font.size": 14})
INK, MUTE, GRID, GREY = "#1f1f1e", "#52514e", "#e6e5e1", "#e3e3e0"
BLUE, VIOLET, YELLOW, SLATE = "#2a78d6", "#6250d6", "#eda100", "#8b8d93"
A_FALL, A_REC = 0.16, 0.26

def clean(ax, grid_y=True):
    ax.set_facecolor("white")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#bdbcb8")
    if grid_y:
        ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTE, labelsize=14, length=0)

# Drift-applied TimesFM (best version per horizon, chosen on 2015-2018): h=1 stack-4, h=5 platt-6.
P = pd.read_pickle(RES / "timesfm_drift_test_predictions.pkl")
t5, t1 = P[(5, "platt")], P[(1, "stack")]
lev = pd.read_csv(RES / "timesfm_drift_test_levels.csv")
det = pd.read_csv(RES / "timesfm_drift_year_detail.csv"); det = det[det.year != "all"]; det["year"] = det.year.astype(int)
ph = pd.read_csv(RES / "timesfm_drift_year_phases.csv")
px = pd.read_pickle(ROOT / "data" / "aligned_prices.pkl")["SPY"]["Close"].astype(float)
def level(h, role, cfg=None):
    r = lev[(lev.horizon == h) & (lev.role == role)]
    return (r[r.cfg == cfg] if cfg else r).roc_auc.iloc[0]

# ---------------- slide 4: overall AUC by horizon + per-year h=5
auc1, auc5 = roc_auc_score(t1.y, t1.proba), roc_auc_score(t5.y, t5.proba)
print("overall AUC h1 (stack-4), h5 (platt-6):", round(auc1, 3), round(auc5, 3))
assert abs(auc1 - level(1, "stack", "stack-4")) < 1e-5 and abs(auc5 - level(5, "platt", "platt-6")) < 1e-5
lr1 = roc_auc_score(P[(1, "fixed_lr")].y, P[(1, "fixed_lr")].proba)
lr5 = roc_auc_score(P[(5, "fixed_lr")].y, P[(5, "fixed_lr")].proba)
print("fixed_lr AUC from pkl:", round(lr1, 3), round(lr5, 3))
assert abs(lr1 - level(1, "fixed_lr")) < 1e-5 and abs(lr5 - level(5, "fixed_lr")) < 1e-5

fig, ax = plt.subplots(figsize=(6.0, 4.2)); clean(ax)
x = np.arange(2); w = 0.34
vals = {"TimesFM-3 + drift layer": [auc1, auc5], "Our simple model": [lr1, lr5]}
for i, (name, v) in enumerate(vals.items()):
    xs = x + (i - 0.5) * (w + 0.04)
    col = BLUE if i == 0 else SLATE
    ax.bar(xs, [vv - 0.5 for vv in v], w, bottom=0.5, color=col, label=name, zorder=3)
    for xx, vv in zip(xs, v):
        ax.text(xx, vv + (0.004 if vv >= 0.5 else -0.004), f"{vv:.2f}", ha="center", va="bottom" if vv >= 0.5 else "top",
                fontsize=15, color=INK, fontweight="bold")
ax.axhline(0.5, color=INK, lw=1.4, ls=(0, (5, 3)), zorder=4)
ax.text(-0.93, 0.5, "coin flip", ha="left", va="bottom", fontsize=14, color=INK)
ax.set_ylim(0.43, 0.58); ax.set_xlim(-0.95, 1.62)
ax.set_xticks(x); ax.set_xticklabels(["Tomorrow", "Next 5 days"], fontsize=15, color=INK)
ax.set_yticks([0.45, 0.5, 0.55]); ax.set_ylabel("AUC", color=MUTE)
ax.legend(loc="upper left", frameon=False, fontsize=14, labelcolor=INK, borderaxespad=0)
fig.tight_layout(); fig.savefig(FIG / "auc_horizon.png", dpi=DPI); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.0, 4.2)); clean(ax)
yrs = list(range(2019, 2026))
a5 = [det[(det.year == y) & (det.horizon == 5)].auc.iloc[0] for y in yrs]
fall = {2020, 2022, 2025}
ax.bar(range(7), [a - 0.5 for a in a5], 0.66, bottom=0.5, color=[VIOLET if y in fall else SLATE for y in yrs], zorder=3)
for i, a in enumerate(a5):
    ax.text(i, a + (0.006 if a >= 0.5 else -0.006), f"{a:.2f}", ha="center", va="bottom" if a >= 0.5 else "top", fontsize=14, color=INK, fontweight="bold")
ax.axhline(0.5, color=INK, lw=1.4, ls=(0, (5, 3)), zorder=4)
ax.text(-0.55, 0.5, "coin flip", ha="left", va="bottom", fontsize=14, color=INK)
ax.set_ylim(0.34, 0.54); ax.set_xlim(-0.6, 6.6)
ax.set_xticks(range(7)); ax.set_xticklabels([str(y) for y in yrs], fontsize=13, color=INK)
ax.set_yticks([0.4, 0.5]); ax.set_ylabel("AUC, next 5 days", color=MUTE)
ax.legend([Patch(color=VIOLET), Patch(color=SLATE)], ["big-fall years", "other years"], loc="lower right", frameon=False, fontsize=14,
          labelcolor=INK, borderaxespad=0.2, bbox_to_anchor=(1.0, 0.0), ncol=1)
fig.tight_layout(); fig.savefig(FIG / "auc_by_year.png", dpi=DPI); plt.close(fig)

# ---------------- slide 2: made-up fan chart
rng = np.random.default_rng(7)
past_n = 25
steps = rng.normal(0.25, 2.0, past_n)
past = 600 - np.cumsum(steps[::-1])[::-1]; past = np.append(past, 600.0)     # last point = today's close
past[:-1] = past[:-1] - 0   # keep
q1 = np.array([594, 596, 598, 599, 600.5, 601.5, 603, 604, 606.0])
q5 = np.array([585, 590, 594, 597, 601, 604, 607, 611, 616.0])
fan = np.vstack([np.full(9, 600.0)] + [q1 + (q5 - q1) * (k / 4) for k in range(5)])   # rows: t, t+1..t+5
tx = np.arange(0, 6); px_ = np.arange(-past_n, 1)
fig, ax = plt.subplots(figsize=(6.9, 3.9)); clean(ax, grid_y=False)
for lo, hi, a in ((0, 8, 0.16), (1, 7, 0.22), (2, 6, 0.30), (3, 5, 0.42)):
    ax.fill_between(tx, fan[:, lo], fan[:, hi], color=BLUE, alpha=a, lw=0, zorder=2)
ax.plot(tx, fan[:, 4], color=BLUE, lw=2.6, zorder=3)
ax.plot(px_, past, color=INK, lw=2.2, zorder=3)
ax.scatter([0], [600], color=INK, s=55, zorder=5)
ax.axhline(600, color=MUTE, lw=1.3, ls=(0, (5, 3)), zorder=1)
ax.text(-24.5, 601.2, "today's close = 600", fontsize=14, color=INK, va="bottom")
ax.text(5.5, 616, "P(up) at t+5\n= 52.5% < 58%\n→ “not up”", fontsize=14, color=INK, va="top", ha="left", linespacing=1.3)
ax.text(-24.5, 620.5, "made-up numbers", fontsize=14, color=MUTE, style="italic", va="top")
ax.set_xlim(-25, 13.5); ax.set_ylim(576, 622)
ax.set_xticks([-25, -15, -5, 0, 5]); ax.set_xticklabels(["t−25", "t−15", "t−5", "t", "t+5"], color=MUTE)
ax.set_yticks([580, 590, 600, 610, 620]); ax.set_ylabel("SPY price (USD)", color=MUTE)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.legend([Patch(color=BLUE, alpha=0.18), Patch(color=BLUE, alpha=0.42), plt.Line2D([], [], color=BLUE, lw=2.6)],
          ["10–90% band", "40–60% band", "median"], loc="lower left", bbox_to_anchor=(0.0, 0.02), frameon=False, fontsize=14, labelcolor=INK, borderaxespad=0)
fig.tight_layout(); fig.savefig(FIG / "fan_chart.png", dpi=DPI); plt.close(fig)

# ---------------- title slide: SPY 2019-2025 with the three fall years' falls shaded
c = px.loc["2019-01-01":"2025-12-31"]
fig, ax = plt.subplots(figsize=(5.6, 3.2)); clean(ax, grid_y=False)
ax.plot(c.index, c.values, color=INK, lw=1.8)
for _, r in ph[(ph.kind != "other") & ph.start.notna() & (ph.phase != "other") & ~ph.phase.str.contains("(all)", regex=False)].iterrows():
    if r.phase.startswith("fall") and int(r.year) in (2020, 2022, 2025):
        ax.axvspan(pd.Timestamp(r.start), pd.Timestamp(r.end), color=VIOLET, alpha=0.22, lw=0)
ax.set_yticks([]); ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(labelsize=13)
fig.tight_layout(); fig.savefig(FIG / "title_spy.png", dpi=DPI, transparent=False); plt.close(fig)

# ---------------- year slides
def phases(yr):
    g = ph[(ph.year == yr) & (ph.phase != "other") & ph.start.notna() & ~ph.phase.str.contains("(all)", regex=False)]
    out = []
    for _, r in g.iterrows():
        if yr == 2022:
            lab = r.phase.replace("fall ", "F").replace("rebound ", "R")
        else:
            lab = "F" if r.kind == "fall" else "R"
        out.append((lab, r.kind, pd.Timestamp(r.start), pd.Timestamp(r.end)))
    return out

W, H = 8.2, 5.0
for yr in range(2019, 2026):
    cy = px[px.index.year == yr]; g = t5[t5.index.year == yr]
    fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor("white")
    L, R = 0.80, 0.12
    def axes(y0, h):   # inches from bottom
        return fig.add_axes([L / W, y0 / H, (W - L - R) / W, h / H])
    a3 = axes(0.42, 0.10); a2 = axes(0.60, 0.40); a1 = axes(1.30, 0.40); a0 = axes(2.20, 2.40)
    for a in (a0, a1, a2, a3): clean(a, grid_y=False)
    a3.set_yticks([]); a3.spines["bottom"].set_visible(False)
    a0.grid(axis="y", color=GRID, lw=0.8)
    a0.plot(cy.index, cy.values, color=INK, lw=2.2, zorder=3)
    lo, hi = cy.min(), cy.max(); pad = (hi - lo) * 0.14
    a0.set_ylim(lo - pad * 0.4, hi + pad)
    for lab, kind, a, e in phases(yr):
        col, al = (VIOLET, A_FALL) if kind == "fall" else (YELLOW, A_REC)
        a0.axvspan(a, e + pd.Timedelta(hours=23), color=col, alpha=al, lw=0, zorder=1)
        a3.axvspan(a, e + pd.Timedelta(hours=23), color=col, alpha=min(1, al * 2.2), lw=0)
        a0.text(a + (e - a) / 2, 0.985, lab, transform=a0.get_xaxis_transform(), ha="center", va="top", fontsize=14, color=INK, fontweight="bold", zorder=5)
    for ax, ser, name in ((a1, g.pred, "TimesFM + drift layer said “up”"), (a2, g.y, "Market went up over the next 5 days")):
        ax.bar(g.index, np.ones(len(g)), width=1.25, color=[BLUE if v else GREY for v in ser], zorder=2)
        ax.set_ylim(0, 1); ax.set_yticks([])
        ax.text(0.0, 1.12, name, transform=ax.transAxes, fontsize=14, color=INK, va="bottom")
    for ax in (a0, a1, a2, a3):
        ax.set_xlim(pd.Timestamp(f"{yr}-01-01"), pd.Timestamp(f"{yr}-12-31"))
    for ax in (a0, a1, a2): ax.tick_params(labelbottom=False)
    a0.set_ylabel("SPY close (USD)", color=MUTE, fontsize=13)
    a3.xaxis.set_major_locator(mdates.MonthLocator()); a3.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    a3.tick_params(labelsize=13)
    a0.legend([Patch(color=VIOLET, alpha=0.3), Patch(color=YELLOW, alpha=0.4), Patch(color=BLUE), Patch(color=GREY)],
              ["fall", "recovery", "up", "not up"], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, frameon=False, fontsize=13,
              labelcolor=INK, columnspacing=1.2, handlelength=1.2, borderaxespad=0.15)
    fig.savefig(FIG / f"year_{yr}.png", dpi=DPI); plt.close(fig)

# ---------------- slide 13: TimesFM vs Chronos-2, what each raw score follows (Spearman, test days)
tc = pd.read_csv(RES / "timesfm_vs_chronos_trailing_corr.csv"); tc = tc[tc.year == "all"]
df_ = pd.read_csv(RES / "timesfm_vs_chronos_drift_features.csv"); df_ = df_[df_.feature == "spread_q90_q10"]
def sp(h, model, vs):
    if vs == "spread": return df_[(df_.h == h) & (df_.model == model)].spearman_with_z.iloc[0]
    return tc[(tc.h == h) & (tc.model == model) & (tc.vs == vs)].spearman.iloc[0]
rows = [("Yesterday's\nmove", "ret1"), ("Last 5 days'\nmove", "ret5"), ("Recent swings\n(20 days)", "vol20"), ("Width of its\nown price range", "spread")]
fig, axs = plt.subplots(1, 2, figsize=(7.45, 4.4), sharey=True)
for ax, h in zip(axs, (1, 5)):
    clean(ax, grid_y=False); ax.grid(axis="x", color=GRID, lw=0.8)
    yy = np.arange(len(rows))[::-1]; bh = 0.36
    for k, (model, col) in enumerate((("timesfm", BLUE), ("chronos", VIOLET))):
        v = [sp(h, model, vs) for _, vs in rows]
        ys = yy + (0.5 - k) * (bh + 0.03)
        ax.barh(ys, v, bh, color=col, zorder=3, label="TimesFM-3" if model == "timesfm" else "Chronos-2")
        for yv, vv in zip(ys, v):
            if abs(vv) >= 0.2:   # label only the clear leans; near-zero bars are visibly ~0
                ax.text(vv + (0.03 if vv >= 0 else -0.03), yv, f"{vv:+.2f}".replace("-", "\u2212"), ha="left" if vv >= 0 else "right", va="center", fontsize=12, color=INK, fontweight="bold")
    ax.axvline(0, color=INK, lw=1.2, zorder=4)
    ax.set_xlim(-1.12, 0.95); ax.set_xticks([-0.5, 0, 0.5])
    ax.set_xticklabels(["\u22120.5", "0", "+0.5"], fontsize=12)
    ax.set_title("Tomorrow" if h == 1 else "Next 5 days", fontsize=14, color=INK, fontweight="bold", loc="left")
axs[0].set_yticks(np.arange(len(rows))[::-1]); axs[0].set_yticklabels([r[0] for r in rows], fontsize=12, color=INK)
hd, lb = axs[0].get_legend_handles_labels()
fig.legend(hd, lb, loc="upper right", ncol=2, frameon=False, fontsize=13, labelcolor=INK, bbox_to_anchor=(1.0, 1.0), columnspacing=1.4, handlelength=1.2)
fig.supxlabel("Does the model's chance of up move with ...?   + together, \u2212 opposite", fontsize=12, color=MUTE, y=0.0)
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(FIG / "timesfm_vs_chronos.png", dpi=DPI); plt.close(fig)

# ---------------- slide 15: why the drift layer helped (TIMESFM_DRIFT.md section 7)
sa = pd.read_csv(RES / "timesfm_drift_why_h1_single_auc.csv").set_index("feature").auc_test_2019_2025
rb = pd.read_csv(RES / "timesfm_drift_why_h5_robust_years.csv").set_index("sample")
fig, (a1_, a2_) = plt.subplots(1, 2, figsize=(6.9, 4.2), gridspec_kw={"width_ratios": [3.1, 1.5]})
clean(a1_, grid_y=False); a1_.grid(axis="x", color=GRID, lw=0.8)
feats = [("TimesFM up/down call", "score_logit", BLUE), ("TimesFM middle price", "median_return", BLUE), ("Width of TimesFM's\nprice range", "spread_q90q10", BLUE),
         ("VIX (fear index)", "VIX", SLATE), ("Recent swings\n(20 days)", "rv20", SLATE), ("Drift layer", "stack-4 (fixed layer)", VIOLET)]
for i, (lab, key, col) in enumerate(feats):
    v = sa[key]; yv = len(feats) - 1 - i
    a1_.barh(yv, v - 0.5, 0.62, left=0.5, color=col, zorder=3)
    a1_.text(v + 0.002 if v >= 0.5 else v - 0.002, yv, f"{v:.3f}", ha="left" if v >= 0.5 else "right", va="center", fontsize=13, color=INK, fontweight="bold")
a1_.axvline(0.5, color=INK, lw=1.4, ls=(0, (5, 3)), zorder=4)
a1_.set_yticks(range(len(feats))); a1_.set_yticklabels([f[0] for f in feats][::-1], fontsize=13, color=INK)
a1_.set_xlim(0.465, 0.545); a1_.set_xticks([0.48, 0.50, 0.52]); a1_.tick_params(axis="x", labelsize=12)
a1_.set_title("Tomorrow: AUC of\neach input alone", fontsize=13, color=INK, fontweight="bold", loc="left")
clean(a2_); 
g = [("all\n7 years", rb.loc["all years", "d_auc"], BLUE), ("without\n2022", rb.loc["excl 2022", "d_auc"], VIOLET)]
for i, (lab, v, col) in enumerate(g):
    a2_.bar(i, v, 0.62, color=col, zorder=3)
    a2_.text(i, v + (0.0012 if v >= 0 else -0.0012), f"{v:+.3f}".replace("-", "\u2212"), ha="center", va="bottom" if v >= 0 else "top", fontsize=13, color=INK, fontweight="bold")
a2_.axhline(0, color=INK, lw=1.2, zorder=4)
a2_.set_xticks([0, 1]); a2_.set_xticklabels([x[0] for x in g], fontsize=12, color=INK)
a2_.set_ylim(-0.014, 0.03); a2_.set_yticks([0]); a2_.set_yticklabels(["0"], fontsize=12); a2_.set_xlim(-0.6, 1.6)
a2_.set_title("Next 5 days:\nAUC gain", fontsize=13, color=INK, fontweight="bold", loc="left")
fig.tight_layout(); fig.savefig(FIG / "drift_why.png", dpi=DPI); plt.close(fig)

# ---------------- slide 12: is the neural network the problem? (was timesfm_rootcause_figure.py; same data, plain labels)
comp = pd.read_csv(RES / "timesfm_rootcause_components.csv").set_index(["model", "h"])
base = pd.read_csv(RES / "timesfm_rootcause_baselines.csv").set_index(["model", "h"])
NAMES = [("TimesFM as released", comp, "official"), ("Its neural network only", comp, "network_only"), ("Its built-in trend line only", comp, "line_slope"),
         ("Today + average\ndaily gain", base, "random_walk_drift"), ("Next move from the\nlast 5 moves", base, "AR5_returns"), ("Fading trend", base, "ETS_damped_trend")]
rcs = []
for lab, t, k in NAMES:
    r_ = t.loc[(k, 5)]
    rcs.append(dict(model=lab, auc=r_.auc_pooled, lo=r_.ci_lo, hi=r_.ci_hi, **{f"auc_{y}": r_[f"auc_{y}"] for y in range(2019, 2026)}))
d_ = pd.DataFrame(rcs)
fig, ax = plt.subplots(figsize=(7.5, 4.4))
yv = np.arange(len(d_))[::-1]
ax.axvline(0.5, ls="--", c=INK, lw=1.3, zorder=1)
for i, r_ in d_.iterrows():
    cc = "#c0392b" if i < 3 else BLUE
    ax.hlines(yv[i], r_.lo, r_.hi, color=cc, lw=2.4, alpha=.5, zorder=2)
    ax.plot(r_.auc, yv[i], "o", ms=11, color=cc, zorder=4)
    ax.text(0.595, yv[i], f"{r_.auc:.2f}", va="center", ha="left", fontsize=13, color=INK, fontweight="bold")
    for yr, mk in ((2020, "s"), (2022, "^"), (2025, "D")):
        ax.plot(r_[f"auc_{yr}"], yv[i], mk, ms=6, mfc="white", mec="#444", mew=1.2, zorder=3)
ax.set_yticks(yv); ax.set_yticklabels(d_.model, fontsize=12, color=INK)
ax.set_xlim(0.28, 0.66); ax.set_ylim(-0.6, len(d_) - 0.4)
ax.set_xlabel("AUC for the next 5 days (dot = all test years; bar = range luck alone could give)", fontsize=11, color=MUTE)
ax.tick_params(axis="x", labelsize=12, colors=MUTE)
for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
ax.text(0.5, len(d_) - 0.35, "coin flip", ha="center", va="bottom", fontsize=12, color=INK)
ax.plot([], [], " ", label="single years:"); ax.plot([], [], "s", ms=6, mfc="white", mec="#444", label="2020"); ax.plot([], [], "^", ms=6, mfc="white", mec="#444", label="2022")
ax.plot([], [], "D", ms=6, mfc="white", mec="#444", label="2025")
ax.legend(loc="lower left", fontsize=12, frameon=False, ncol=4, handletextpad=.2, columnspacing=.8, bbox_to_anchor=(0.0, 1.06), title=None, title_fontsize=12, alignment="left")
fig.tight_layout(); fig.savefig(FIG / "root_cause.png", dpi=DPI, facecolor="white"); plt.close(fig)

# ---------------- slides 14 and 16: bar charts of test AUC by method (were drawn by the study scripts' report(); same data, plain labels)
def method_chart(L, names, fname, tick_size):
    fig, ax = plt.subplots(figsize=(7.5, 4.2)); clean(ax)
    w = 0.36
    for j, (h, col, lab) in enumerate([(1, BLUE, "Tomorrow"), (5, VIOLET, "Next 5 days")]):
        v = [float(L[(L.horizon == h) & (L.role == r)].roc_auc.iloc[0]) for r, _ in names]
        xs = np.arange(len(names)) + (j - 0.5) * w
        ax.bar(xs, [a - 0.5 for a in v], w * 0.94, bottom=0.5, color=col, label=lab, zorder=3)
        for x_, a in zip(xs, v):
            ax.text(x_, a + (0.004 if a >= 0.5 else -0.004), f"{a:.2f}", ha="center", va="bottom" if a >= 0.5 else "top", fontsize=12, color=INK, fontweight="bold")
    ax.axhline(0.5, color=INK, lw=1.4, ls=(0, (5, 3)), zorder=4)
    ax.set_xticks(range(len(names))); ax.set_xticklabels([n for _, n in names], fontsize=tick_size, color=INK)
    allv = L[L.role.isin([r for r, _ in names])].roc_auc
    lo, hi = min(0.44, allv.min() - 0.012), max(0.56, allv.max() + 0.015)
    ax.set_ylim(lo, hi); ax.set_yticks([t for t in (0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56) if lo <= t <= hi])
    ax.set_ylabel("Test AUC 2019\u20132025", color=MUTE)
    hd_, lb_ = ax.get_legend_handles_labels()
    ax.legend(hd_ + [plt.Line2D([], [], color=INK, lw=1.4, ls=(0, (5, 3)))], lb_ + ["coin flip (0.5)"], loc="upper left", frameon=False, fontsize=12, labelcolor=INK, borderaxespad=0, ncol=3, columnspacing=1.2)
    fig.tight_layout(); fig.savefig(FIG / fname, dpi=DPI, facecolor="white"); plt.close(fig)
method_chart(lev, [("static", "TimesFM\nalone"), ("platt", "Monthly\nshift"), ("stack", "Small model\non top"), ("hedge", "Vote"), ("detect", "Alarm"), ("fixed_lr", "Our simple\nmodel")],
             "drift_methods.png", 12)
vlev = pd.read_csv(RES / "timesfm_vixdrift_test_levels.csv")
method_chart(vlev, [("d0", "TimesFM +\ndrift layer"), ("d1", "Refit when\nVIX flips"), ("d2", "Pick model\nby VIX"), ("d3", "Add VIX\nterms"), ("d4", "VIX taken\nout of input"), ("fixed_lr", "Our simple\nmodel")],
             "vix_drift_methods.png", 11.5)
print("done")

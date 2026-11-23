"""ROOT CAUSE: slide figure slides/fig/root_cause.png (h=5 AUC: official TimesFM, its parts, simple forecasters)
and the combined table results/timesfm_rootcause_summary.csv.  Reads the csv files of the two experiments."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

from tf_paths import RES_DIR, TF_DIR

for f in fm.findSystemFonts():
    if "Arimo" in f:
        fm.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Arimo", "DejaVu Sans"]

comp = pd.read_csv(RES_DIR / "timesfm_rootcause_components.csv")
base = pd.read_csv(RES_DIR / "timesfm_rootcause_baselines.csv")
c = comp.set_index(["model", "h"])
b = base.set_index(["model", "h"])
NAMES = [("TimesFM official", c, "official"),
         ("Transformer part only", c, "network_only"),
         ("Detrending-line slope only", c, "line_slope"),
         ("Random walk + drift", b, "random_walk_drift"),
         ("AR(5) on returns", b, "AR5_returns"),
         ("ETS damped trend", b, "ETS_damped_trend")]
rows = []
for h in (1, 5):
    for lab, t, k in NAMES:
        r = t.loc[(k, h)]
        rows.append(dict(h=h, model=lab, auc=r.auc_pooled, lo=r.ci_lo, hi=r.ci_hi,
                         **{f"auc_{y}": r[f"auc_{y}"] for y in range(2019, 2026)}))
S = pd.DataFrame(rows)
S.to_csv(RES_DIR / "timesfm_rootcause_summary.csv", index=False)

h = 5
d = S[S.h == h].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=220)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
y = np.arange(len(d))[::-1]
col = {0: "#c0392b", 1: "#c0392b", 2: "#c0392b"}
ax.axvline(0.5, ls="--", c="#555", lw=1.3, zorder=1)
for i, r in d.iterrows():
    cc = "#c0392b" if i < 3 else "#2c6fbb"
    ax.hlines(y[i], r.lo, r.hi, color=cc, lw=2.2, alpha=.55, zorder=2)
    ax.plot(r.auc, y[i], "o", ms=11, color=cc, zorder=4)
    ax.text(0.595, y[i], f"{r.auc:.3f}", va="center", ha="left", fontsize=11.5, color="#222", fontweight="bold")
    for yr, mk in ((2020, "s"), (2022, "^"), (2025, "D")):
        ax.plot(r[f"auc_{yr}"], y[i], mk, ms=5.5, mfc="white", mec="#444", mew=1.2, zorder=3)
ax.set_yticks(y); ax.set_yticklabels(d.model, fontsize=11.5)
ax.set_xlim(0.28, 0.66); ax.set_ylim(-0.6, len(d) - 0.4)
ax.set_xlabel("AUC at h=5, test 2019-2025 (dot = pooled, bar = 95% interval)", fontsize=10.5)
ax.tick_params(axis="x", labelsize=10.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.text(0.5, len(d) - 0.35, "chance 0.5", ha="center", va="bottom", fontsize=10, color="#555")
ax.plot([], [], "s", ms=6, mfc="white", mec="#444", label="2020"); ax.plot([], [], "^", ms=6, mfc="white", mec="#444", label="2022")
ax.plot([], [], "D", ms=6, mfc="white", mec="#444", label="2025")
ax.legend(loc="lower left", fontsize=10, frameon=False, ncol=3, handletextpad=.2, columnspacing=.8, bbox_to_anchor=(0.0, 1.0), title="single years:", title_fontsize=10, alignment="left")
fig.tight_layout()
out = TF_DIR / "slides" / "fig" / "root_cause.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=220, facecolor=fig.get_facecolor())
print("wrote", out)

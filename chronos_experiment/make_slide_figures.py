#!/usr/bin/env python
"""
Slide-ready figures for the Chronos-2 presentation, drawn only from saved results (no model, no GPU).

    python chronos_experiment/make_slide_figures.py

Writes to chronos_experiment/figures/:
  fig_direction_auc_by_year.png   slide 6   Chronos-2 test AUC by year, 1-day and 5-day   (results_ext/direction_drift_by_year.csv)
  fig_corr_rolling_mse.png        slide 8   rolling 252-day MSE on the SPY-TLT correlation (results_ext/ext_test_rolling252_mse.csv)
  fig_m3_equity.png               slide 10  growth of $1: Chronos-managed vs 20-day-vol managed vs buy-and-hold, 5 bps
                                            (results_vix/vix_test_predictions.pkl)
Colours and background match the deck (navy / blue / orange on #F7F6F2).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

BG, INK, BODY, GRID = "#F7F6F2", "#14213D", "#46506A", "#DEDAD0"
NAVY, BLUE, ORANGE, MUTED = "#14213D", "#2559B8", "#C8601A", "#8A93A6"
plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14, "legend.fontsize": 13,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "font.family": "sans-serif"})


def style(ax):
    ax.set_facecolor(BG)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=BODY)


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"saved {path}")


def direction_by_year():
    Y = pd.read_csv(HERE / "results_ext" / "direction_drift_by_year.csv")
    Y = Y[Y.model == "chronos"]
    fig, ax = plt.subplots(figsize=(8, 6.2), facecolor=BG)
    style(ax)
    ax.axvspan(2021.5, 2022.5, color=GRID, alpha=0.7, lw=0)
    ax.axhline(0.5, color=INK, lw=1.2, ls=(0, (5, 4)))
    ax.text(2025.45, 0.503, "chance", color=INK, ha="right", va="bottom")
    for h, c, mk, ls in [(1, BLUE, "o", "-"), (5, ORANGE, "s", "--")]:
        g = Y[Y.h == h].sort_values("year")
        ax.plot(g.year, g.roc_auc, color=c, marker=mk, ms=9, lw=2.5, ls=ls, mec=BG, mew=2, label=f"{h}-day")
    ax.set_xticks(sorted(Y.year.unique()))
    ax.set_ylabel("AUC within the year")
    ax.text(2022, ax.get_ylim()[1], "2022", ha="center", va="top", color=BODY)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Chronos-2 direction AUC by test year", loc="left", color=INK)
    save(fig, "fig_direction_auc_by_year.png")


def corr_rolling():
    R = pd.read_csv(HERE / "results_ext" / "ext_test_rolling252_mse.csv", parse_dates=["Date"], index_col="Date")
    d = R[R.target == "spy_tlt_corr"]
    fig, ax = plt.subplots(figsize=(9, 6.2), facecolor=BG)
    style(ax)
    lines = [("har_fixed", ORANGE, "--", "HAR, trained once"), ("chronos", BLUE, "-", "Chronos-2, no retraining"),
             ("har_refit21", NAVY, "-.", "HAR, refit monthly"), ("persist_63", MUTED, ":", "63-day persistence")]
    for col, c, ls, lab in lines:
        if col in d:
            ax.plot(d.index, d[col], color=c, ls=ls, lw=2.5, label=lab)
    ax.axvline(pd.Timestamp("2022-01-01"), color=INK, lw=1.2, ls=(0, (5, 4)))
    ax.text(pd.Timestamp("2022-02-15"), ax.get_ylim()[1], "2022 flip", color=INK, va="top")
    ax.set_ylabel("Rolling 252-day MSE (lower is better)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    ax.set_title("SPY–TLT correlation forecast error over the test", loc="left", color=INK)
    save(fig, "fig_corr_rolling_mse.png")


def m3_equity():
    f = HERE / "results_vix" / "vix_test_predictions.pkl"
    if not f.exists():
        print(f"skipped fig_m3_equity.png: {f} not found (run chronos_vix_study.py test first)")
        return
    S = pd.read_pickle(f)["M3_5bps"]
    fig, ax = plt.subplots(figsize=(9, 6.2), facecolor=BG)
    style(ax)
    for col, c, ls, lab in [("chronos", BLUE, "-", "Chronos-managed"), ("rv20_master", NAVY, "-.", "20-day-vol managed"),
                            ("bh", ORANGE, "--", "Buy-and-hold")]:
        ax.plot(S.index, (1 + S[col]).cumprod(), color=c, ls=ls, lw=2.5, label=lab)
    ax.set_yscale("log")
    ax.set_yticks([1, 1.5, 2, 3, 4])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Volatility-managed SPY exposure, 2019–2025, 5 bps costs", loc="left", color=INK)
    save(fig, "fig_m3_equity.png")


if __name__ == "__main__":
    direction_by_year()
    corr_rolling()
    m3_equity()

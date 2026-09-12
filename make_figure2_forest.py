# -*- coding: utf-8 -*-
"""Generate Figure 2: a two-panel forest plot with publication styling.

Panel A: fully adjusted Firth penalized logistic regression estimates for
all-cause death within 28 days, 3 months, and 6 months (complete cases).
Panel B: sensitivity analyses for the 28-day endpoint (unadjusted, multiple
imputation, stabilized IPTW, and exclusion of hospital stays < 3 days).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from figure_style import GREY, INK, set_sci_style


OUT = Path(__file__).resolve().parents[1] / "Final_Submission_Package" / "Figures"
set_sci_style()

panel_a = [
    ("Death within 28 days", 39.06, 12.79, 138.13, 1841, 25),
    ("Death within 3 months", 30.64, 10.85, 95.36, 1841, 28),
    ("Death within 6 months", 15.01, 6.53, 34.54, 1841, 42),
]

panel_b = [
    ("Unadjusted", 51.89, 23.41, 125.17, 1997, 28),
    ("Multiple imputation", 46.44, 14.88, 144.95, 1997, 28),
    ("Stabilized IPTW", 15.13, 5.39, 42.48, 1867, 25),
    ("Excluding stay <3 days", 17.87, 3.83, 95.50, 1792, 12),
]

PANELS = [
    (
        "A. Adjusted all-cause mortality outcomes",
        "Adjusted OR (95% CI)",
        "Adjusted odds ratio (95% CI; log scale)",
        panel_a,
    ),
    (
        "B. Sensitivity analyses for 28-day death",
        "OR (95% CI)",
        "Odds ratio (95% CI; log scale)",
        panel_b,
    ),
]

TOP = 2.5
STEP = 1.2
TICKS = [0.5, 1, 2, 5, 10, 20, 50, 100, 200]

fig = plt.figure(figsize=(9.8, 6.9), dpi=300)
grid = GridSpec(
    2,
    1,
    figure=fig,
    height_ratios=[len(rows) for _, _, _, rows in PANELS],
    left=0.24,
    right=0.60,
    top=0.94,
    bottom=0.10,
    hspace=0.62,
)

for index, (title, column_header, xlabel, rows) in enumerate(PANELS):
    ax = fig.add_subplot(grid[index])
    ax.set_xscale("log")
    ax.set_xlim(0.5, 320)
    ax.set_ylim(TOP - (len(rows) - 1) * STEP - 0.95, TOP + 1.05)
    ax.set_title(title, loc="left", fontsize=9.5, pad=10)
    ax.axvline(1.0, color=GREY, linewidth=0.7, linestyle=(0, (4, 2.5)), zorder=1)

    ax.text(
        1.02,
        TOP + 0.8,
        column_header,
        ha="left",
        va="center",
        fontsize=8.5,
        color=INK,
        transform=ax.get_yaxis_transform(),
        clip_on=False,
    )

    for i, (label, or_v, lo, hi, n_row, events_row) in enumerate(rows):
        y = TOP - i * STEP
        lower_err = or_v - lo
        upper_err = hi - or_v
        ax.errorbar(
            [or_v],
            [y],
            xerr=[[lower_err], [upper_err]],
            fmt="none",
            ecolor=INK,
            elinewidth=1.0,
            capsize=2.2,
            capthick=1.0,
            zorder=2,
        )
        ax.scatter(
            [or_v],
            [y],
            s=26,
            marker="s",
            color=INK,
            linewidths=0.5,
            zorder=3,
        )
        ax.text(
            -0.03,
            y,
            label,
            ha="right",
            va="center",
            fontsize=9,
            color=INK,
            transform=ax.get_yaxis_transform(),
            clip_on=False,
        )
        ax.text(
            1.02,
            y,
            f"{or_v:.2f} ({lo:.2f}-{hi:.2f})",
            ha="left",
            va="center",
            fontsize=8,
            color=INK,
            transform=ax.get_yaxis_transform(),
            clip_on=False,
        )
        ax.text(
            1.02,
            y - 0.42,
            f"N = {n_row}; events = {events_row}",
            ha="left",
            va="center",
            fontsize=6.8,
            color=INK,
            transform=ax.get_yaxis_transform(),
            clip_on=False,
        )

    ax.set_yticks([])
    ax.set_xticks(TICKS)
    ax.set_xticklabels([f"{t:g}" for t in TICKS])
    ax.tick_params(axis="x", length=3, labelsize=8)
    ax.set_xlabel(xlabel, fontsize=8.5)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

fig.savefig(OUT / "07_Figure2_ForestPlot.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "07_Figure2_ForestPlot.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved Figure 2 PNG and PDF")

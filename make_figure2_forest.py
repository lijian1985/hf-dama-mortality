# -*- coding: utf-8 -*-
"""Generate Figure 2: a two-panel forest plot with publication styling."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    ("Stabilized IPTW", 15.13, 5.39, 42.48, 1867, 25),
    ("Excluding stay <3 days", 17.87, 3.83, 95.50, 1792, 12),
]


fig, axes = plt.subplots(2, 1, figsize=(9.8, 5.7), dpi=300)
fig.subplots_adjust(left=0.24, right=0.60, top=0.92, bottom=0.11, hspace=1.05)

for ax, title, rows in zip(
    axes,
    ("A. Adjusted all-cause mortality outcomes", "B. Sensitivity analyses for 28-day death"),
    (panel_a, panel_b),
):
    ax.set_xscale("log")
    ax.set_xlim(0.5, 320)
    ax.set_ylim(-0.75, 3.55)
    ax.set_title(title, loc="left", fontsize=9.5, pad=10)
    ax.axvline(1.0, color=GREY, linewidth=0.7, linestyle=(0, (4, 2.5)), zorder=1)

    ax.text(
        1.02,
        3.12,
        "Adjusted OR (95% CI)",
        ha="left",
        va="center",
        fontsize=8.5,
        color=INK,
        transform=ax.get_yaxis_transform(),
        clip_on=False,
    )

    for i, (label, or_v, lo, hi, n_row, events_row) in enumerate(rows):
        y = 2.4 - i * 1.2
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
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50, 100, 200])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20", "50", "100", "200"])
    ax.tick_params(axis="x", length=3)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

axes[0].tick_params(axis="x", labelbottom=False)
axes[1].set_xlabel("Adjusted odds ratio (95% CI; log scale)")

fig.savefig(OUT / "07_Figure2_ForestPlot.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "07_Figure2_ForestPlot.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved Figure 2 PNG and PDF")

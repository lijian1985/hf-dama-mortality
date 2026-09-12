# -*- coding: utf-8 -*-
"""Generate Figure 3: mortality/readmission rates with 95% Wilson CIs."""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figure_style import INK, set_sci_style


OUT = Path(__file__).resolve().parents[1] / "Final_Submission_Package" / "Figures"
set_sci_style()

categories = ["28 days", "3 months", "6 months"]
death_routine = [0.4, 0.6, 1.4]
death_dama = [18.7, 18.7, 18.7]
readm_routine = [7.0, 25.1, 39.4]
readm_dama = [7.5, 21.5, 26.2]

death_routine_count = [8, 11, 26]
death_dama_count = [20, 20, 20]
readm_routine_count = [132, 475, 745]
readm_dama_count = [8, 23, 28]

routine_color = "#eceff1"
dama_color = "#9c2b2b"


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.96
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (centre - margin) * 100, (centre + margin) * 100


fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), dpi=300)
fig.subplots_adjust(left=0.085, right=0.985, top=0.85, bottom=0.15, wspace=0.45)
x = np.arange(len(categories))
width = 0.34


def add_panel(ax, routine, dama, count_r, count_d, title, yticks):
    upper_bounds = []
    for rc, dc in zip(count_r, count_d):
        upper_bounds.append(wilson(rc, 1890)[1])
        upper_bounds.append(wilson(dc, 107)[1])
    ymax = max(yticks[-1] * 1.16, max(upper_bounds) * 1.10)
    routine_bar = ax.bar(
        x - width / 2,
        routine,
        width,
        facecolor=routine_color,
        edgecolor=INK,
        linewidth=0.7,
        label="Routine discharge (n=1890)",
        zorder=2,
    )
    dama_bar = ax.bar(
        x + width / 2,
        dama,
        width,
        facecolor=dama_color,
        edgecolor=INK,
        linewidth=0.7,
        label="DAMA (n=107)",
        zorder=2,
    )
    for xi, (rv, dv, rc, dc) in enumerate(zip(routine, dama, count_r, count_d)):
        lo, hi = wilson(rc, 1890)
        ax.errorbar(
            xi - width / 2,
            rv,
            yerr=[[rv - lo], [hi - rv]],
            fmt="none",
            ecolor=INK,
            elinewidth=0.7,
            capsize=2,
            zorder=3,
        )
        lo, hi = wilson(dc, 107)
        ax.errorbar(
            xi + width / 2,
            dv,
            yerr=[[dv - lo], [hi - dv]],
            fmt="none",
            ecolor=INK,
            elinewidth=0.7,
            capsize=2,
            zorder=3,
        )
        ax.text(
            xi - width / 2,
            rv + yticks[-1] * 0.025,
            f"{rv:.1f} ({rc})",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=INK,
        )
        ax.text(
            xi + width / 2,
            dv + yticks[-1] * 0.025,
            f"{dv:.1f} ({dc})",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=INK,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, ymax)
    ax.set_yticks(yticks)
    ax.set_title(title, loc="left", fontsize=9, pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    return routine_bar, dama_bar


r1, d1 = add_panel(
    axes[0],
    death_routine,
    death_dama,
    death_routine_count,
    death_dama_count,
    "A. All-cause death from admission",
    [0, 10, 20, 30],
)
add_panel(
    axes[1],
    readm_routine,
    readm_dama,
    readm_routine_count,
    readm_dama_count,
    "B. All-cause readmission from admission",
    [0, 10, 20, 30, 40],
)

axes[0].set_ylabel("Percentage of patients (%)")
fig.legend(
    [r1, d1],
    ["Routine discharge (n=1890)", "DAMA (n=107)"],
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=2,
    frameon=False,
    columnspacing=1.6,
    handlelength=1.2,
    handletextpad=0.5,
)

fig.savefig(OUT / "08_Figure3_OutcomeRates.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "08_Figure3_OutcomeRates.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved Figure 3 PNG and PDF")

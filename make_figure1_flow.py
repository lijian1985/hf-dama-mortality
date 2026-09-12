# -*- coding: utf-8 -*-
"""Generate a publication-style participant flow diagram (rectangular boxes)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from figure_style import INK, set_sci_style


OUT = Path(__file__).resolve().parents[1] / "Final_Submission_Package" / "Figures"
set_sci_style()


def box(ax, cx, cy, w, h, text, fontsize=9.5):
    ax.add_patch(
        Rectangle(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            facecolor="white",
            edgecolor=INK,
            linewidth=0.9,
            zorder=2,
        )
    )
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        linespacing=1.35,
        zorder=3,
    )


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=0.9,
            color=INK,
            shrinkA=0,
            shrinkB=0,
            zorder=1,
        )
    )


def line(ax, x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color=INK, linewidth=0.9, zorder=1)


fig, ax = plt.subplots(figsize=(8.6, 5.9), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(25, 100)
ax.axis("off")

# Source cohort and the exclusion branch.
box(ax, 43, 88, 46, 10, "Hospitalized patients with heart failure\n(December 2016-June 2019; n = 2008)", 9.5)
box(ax, 84, 88, 30, 11, "Excluded:\nin-hospital death (n = 11)\nmissing hospital outcome (n = 0)", 8.5)
arrow(ax, 66, 88, 69, 88)

# Included analytical cohort.
box(ax, 43, 69, 46, 9.5, "Patients discharged alive and included\nin the analysis (n = 1997)", 10)
arrow(ax, 43, 83.25, 43, 73.75)

# Two exposure groups and the branch connector.
box(ax, 21, 38, 40, 11, "Routine discharge\n(n = 1890; 94.6%)", 9.5)
box(ax, 65, 38, 40, 11, "Discharge against medical advice\n(n = 107; 5.4%)", 9.5)
line(ax, 43, 64.25, 43, 57)
line(ax, 21, 57, 65, 57)
arrow(ax, 21, 57, 21, 43.5)
arrow(ax, 65, 57, 65, 43.5)

fig.savefig(OUT / "06_Figure1_Flowchart.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "06_Figure1_Flowchart.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved Figure 1 PNG and PDF")

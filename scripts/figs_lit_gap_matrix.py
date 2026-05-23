"""Three-literature gap matrix figure for the literature review chapter.

Produces ``thesis/figures/lit_gap_matrix.pdf`` showing five literature
questions (rows) crossed with the three bodies of work this thesis sits
between, plus a fourth column for this thesis's coverage.

The cell-fill semantics use the shared thesis palette so the figure is
visually consistent with every other figure in the document:

  * full coverage (class_accent green)
  * partial / proxy coverage (accent sienna)
  * absent (rule grey, soft text)
  * this thesis (data deep-blue)

Every claim encoded in the cells is supported by a citation in the
caption and the surrounding prose; nothing is asserted in this figure
that is not also asserted with a citation in the chapter text.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _plot_style import PALETTE, apply_style  # noqa: E402

apply_style()

OUT_PDF = PROJECT_ROOT / "thesis" / "figures" / "lit_gap_matrix.pdf"

ROWS = [
    "Formal group-fairness metrics\n(DP, DI, EOdds, EO)",
    "Multi-class fairness extensions",
    "Procedural fairness operationalised\nas ML metrics",
    "Real HR-decision datasets\n(performance, promotion, hiring)",
    "Behavioural validation of\nperceived fairness",
]

COLS = [
    "HR-ML applied\nliterature",
    "Fairness-ML\ntheory",
    "Organisational\nbehaviour",
    "This thesis",
]

# Cell states: 'full', 'partial', 'absent', 'thesis'.
# Each entry is the verdict at the intersection of (column-literature, row-question).
# Verdicts are drawn directly from sources cited in the chapter:
#  * HR-ML applied        (juirivas2024fairness, pagano2023fairness Section 3.4):
#                         no formal group-fairness metric reported; binary or
#                         occasionally multi-class targets; no procedural metric;
#                         primary dataset is the audited HR study; no behavioural
#                         validation.
#  * Fairness-ML theory   (pagano2023fairness, juirivas2024fairness, lequy2022survey):
#                         metrics fully developed; multi-class flagged as gap;
#                         procedural fairness absent; HR datasets absent (Ricci
#                         is the lone partial exception); some behavioural overlap.
#  * Organisational beh.  (greenberg1987taxonomy, colquitt2001dimensionality,
#                         newman2020when, qin2023fairness):
#                         distributive metrics implicit only; no algorithmic
#                         multi-class concern; procedural fairness fully developed
#                         as constructs (not as ML metrics); HR-decision context;
#                         behavioural validation is the disciplinary core.
#  * This thesis: full coverage by construction (Contributions C1-C5).
CELLS = [
    ["absent",   "full",    "absent",  "thesis"],   # Group-fairness metrics
    ["absent",   "partial", "absent",  "thesis"],   # Multi-class extensions
    ["absent",   "absent",  "partial", "thesis"],   # Procedural fairness operationalised
    ["partial",  "absent",  "full",    "thesis"],   # Real HR-decision datasets
    ["absent",   "partial", "full",    "thesis"],   # Behavioural validation
]

# Compact one-line evidence labels per cell (textual annotation inside cells).
# Markers are kept brief; the surrounding prose carries the citations.
CELL_LABELS = [
    ["none",        "DP, EO,\nEOdds, DI",   "none",      "C1"],
    ["binary only", "binary\ndominant",     "none",      "C2"],
    ["none",        "none",                 "constructs\nonly",  "C3"],
    ["proxy",       "Ricci only",           "core",      "C4 + C5"],
    ["none",        "vignette",             "core",      "future"],
]

STATE_FILL = {
    "full":    PALETTE['class_accent'],
    "partial": PALETTE['accent'],
    "absent":  PALETTE['rule'],
    "thesis":  PALETTE['data'],
}

STATE_TEXT = {
    "full":    "white",
    "partial": "white",
    "absent":  PALETTE['ink_soft'],
    "thesis":  "white",
}

def _draw_cell(ax, x, y, w, h, state, label):
    fill = STATE_FILL[state]
    txt_colour = STATE_TEXT[state]
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.7,
        facecolor=fill,
        edgecolor=PALETTE['ink'],
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2,
        label,
        ha="center", va="center",
        fontsize=7.5,
        color=txt_colour,
        zorder=3,
    )

def main() -> None:
    n_rows = len(ROWS)
    n_cols = len(COLS)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    left_margin = 0.32
    top_margin = 0.86
    bottom_margin = 0.16

    cell_w = (1.0 - left_margin - 0.02) / n_cols
    cell_h = (top_margin - bottom_margin) / n_rows

    # Column headers
    for j, col in enumerate(COLS):
        x = left_margin + j * cell_w
        emphasis = "bold" if col == "This thesis" else "normal"
        colour = PALETTE['data'] if col == "This thesis" else PALETTE['ink']
        ax.text(
            x + cell_w / 2, top_margin + 0.04,
            col,
            ha="center", va="center",
            fontsize=8.5,
            fontweight=emphasis,
            color=colour,
        )

    # Row labels (right-aligned, just left of the grid)
    for i, row in enumerate(ROWS):
        y = top_margin - (i + 0.5) * cell_h
        ax.text(
            left_margin - 0.015, y,
            row,
            ha="right", va="center",
            fontsize=8,
            color=PALETTE['ink'],
        )

    # Cells
    for i in range(n_rows):
        for j in range(n_cols):
            x = left_margin + j * cell_w
            y = top_margin - (i + 1) * cell_h
            _draw_cell(
                ax,
                x + 0.005, y + 0.005,
                cell_w - 0.01, cell_h - 0.01,
                CELLS[i][j],
                CELL_LABELS[i][j],
            )

    # Legend at the bottom
    legend_handles = [
        Patch(facecolor=STATE_FILL['full'], edgecolor=PALETTE['ink'],
              label="addressed"),
        Patch(facecolor=STATE_FILL['partial'], edgecolor=PALETTE['ink'],
              label="partial / proxy"),
        Patch(facecolor=STATE_FILL['absent'], edgecolor=PALETTE['ink'],
              label="absent"),
        Patch(facecolor=STATE_FILL['thesis'], edgecolor=PALETTE['ink'],
              label="this thesis"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.55, 0.005),
        ncol=4,
        fontsize=8,
        frameon=False,
    )

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"[lit-figs] Wrote {OUT_PDF}")

if __name__ == "__main__":
    main()

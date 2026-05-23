"""Cross-architecture OULAD gender-attribution-share bar chart.

Reads results/phase6/oulad_cross_arch.csv and produces
thesis/figures/phase6_oulad_cross_arch.pdf:

  Horizontal bar chart of LR / RF / GB / XGB gender-attribution shares
  on OULAD, with a vertical reference line at the IBM HR Attrition
  tree-model level (~ 2 %) to show the dataset-property gap.

The visual claim: all four architectures cluster in a 5.9-8.7 % band on
OULAD, well separated from the < 2 % observed on IBM HR for tree-based
models. The narrowness of the OULAD band relative to the IBM HR gap
indicates that the elevated gender attribution on OULAD is a dataset
property, not a classifier-architecture property.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _plot_style import MODEL_COLOURS, PALETTE, apply_style  # noqa: E402

apply_style()

CSV_IN = PROJECT_ROOT / "results" / "phase6" / "oulad_cross_arch.csv"
OUT_PDF = PROJECT_ROOT / "thesis" / "figures" / "phase6_oulad_cross_arch.pdf"

# The LR baseline share comes from the existing  Pareto-optimal-cell
# analysis (Section 4.6) and is included for direct comparison; the row is
# not in oulad_cross_arch.csv (which covers only RF/GB/XGB).
LR_GENDER_SHARE = 0.087345508990378

# IBM HR Attrition tree-model gender share, from  baseline analysis.
IBM_HR_TREE_REFERENCE = 0.018

def main() -> None:
    df = pd.read_csv(CSV_IN)
    rows = [("LR", LR_GENDER_SHARE)]
    for _, row in df.iterrows():
        rows.append((row["model"], float(row["gender_share"])))

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.set_facecolor(PALETTE["bg"])
    fig.patch.set_facecolor(PALETTE["bg"])

    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colours = [MODEL_COLOURS.get(label, PALETTE["data"]) for label in labels]
    y_pos = list(range(len(labels)))
    ax.barh(y_pos, values, color=colours, edgecolor="white", linewidth=0.8,
            alpha=0.95, zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.105)
    ax.set_xlabel("Gender attribution share (mean |SHAP| / total |SHAP|)",
                  fontsize=9.5, color=PALETTE["ink"])
    ax.tick_params(axis="x", labelsize=8.5, colors=PALETTE["ink_soft"])

    # Reference: IBM HR tree-model gender share. Place the label above the
    # top bar (empty area between y_pos = -0.5 and y_pos = -0.05) so it does
    # not overlap any data bar.
    ax.axvline(IBM_HR_TREE_REFERENCE, color=PALETTE["highlight"],
               linestyle="--", linewidth=1.2, alpha=0.85, zorder=2)
    ax.text(IBM_HR_TREE_REFERENCE + 0.0015, -0.55,
            "IBM HR Attrition tree-model reference (~2%)",
            ha="left", va="bottom", fontsize=8,
            color=PALETTE["highlight"], style="italic")

    # Value labels at end of each bar.
    for yi, val in zip(y_pos, values):
        ax.text(val + 0.0015, yi, f"{val * 100:.1f}%",
                va="center", fontsize=9, color=PALETTE["ink"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["rule"])
    ax.spines["bottom"].set_color(PALETTE["rule"])
    ax.grid(axis="x", color=PALETTE["rule"], alpha=0.4, lw=0.5, zorder=0)

    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"[figs] Wrote {OUT_PDF}")

if __name__ == "__main__":
    main()

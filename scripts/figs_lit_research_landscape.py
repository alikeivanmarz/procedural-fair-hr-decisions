"""Research-landscape scatter for the literature review chapter.

Produces ``thesis/figures/lit_research_landscape.pdf`` showing where
existing HR-machine-learning, fairness-machine-learning, and this thesis
sit on a (sample-size, fairness-metric-count) plane.

Headline visual claim: HR-ML applied work clusters at zero formal fairness
metrics; classical fairness-ML benchmark datasets compute a small handful
of metrics on demographic-proxy data; this thesis occupies the
upper-right quadrant -- multi-metric audit (including procedural-fairness
metrics that no prior work computes) on HR-relevant datasets.

Inspired by the Deepa Singh proposal's `fig6_dataset_landscape.py`.

Data points are evidence-based:
  * Nayem & Uddin 2024 -- sample size 1,109 stated in the published abstract.
  * The four classical fairness-ML benchmark datasets and their typical
    metric coverage are reported in Le Quy et al. (2022) Table 15
    (Demographic Parity, Equalised Odds, ABROCA = three metrics).
  * This thesis's per-dataset metric coverage is the count of unique
    fairness-metric families computed in this thesis's results chapter:
    DP, EOdds, EO, DI, KNN-Consistency, Counterfactual Fairness, Process
    Consistency, Voice/Representation, Transparency = nine families on
    D1 and D6, with reduced coverage on D2 (no procedural sensitivity
    rerun) and D7 (controllable benchmark, three families).
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _plot_style import PALETTE, apply_style  # noqa: E402

apply_style()

OUT_PDF = PROJECT_ROOT / "thesis" / "figures" / "lit_research_landscape.pdf"

# ---------------------------------------------------------------------------
# Evidence-based data records.
# Tuple: (short_label, full_label, n_rows, n_metrics, tradition, dx, dy, ha)
#   tradition in {"hr_ml", "fairness_ml", "thesis"}
#   (dx, dy) in data units for label offset; ha in {"left", "right", "center"}
# ---------------------------------------------------------------------------
records = [
    # HR-ML applied literature (zero formal fairness metrics).
    # Nayem 2024 has its sample size stated in the abstract; the other four
    # HR-ML applied papers cited in this thesis are represented by the
    # cluster annotation (rather than as individual points whose sample
    # sizes are not stated in the available abstracts).
    ("Nayem 2024", "Nayem &\nUddin 2024", 1109, 0, "hr_ml", -0.18, 0.65, "right"),

    # Classical fairness-ML benchmark datasets and their typical metric
    # coverage (Demographic Parity, Equalised Odds, ABROCA = 3 metrics)
    # per Le Quy et al. (2022) Table 15.
    ("Adult", "Adult\n(48,842)", 48842, 3, "fairness_ml", 0.15, 0.65, "left"),
    ("Ricci", "Ricci\n(118)", 118, 3, "fairness_ml", 0.10, 0.65, "left"),
    ("German credit", "German credit\n(1,000)", 1000, 3, "fairness_ml", 0.10, -0.75, "left"),
    ("COMPAS", "COMPAS\n(~7k)", 7000, 4, "fairness_ml", 0.20, -0.75, "left"),

    # This thesis -- the two primary HR datasets carrying every
    # contribution, plus the high-scale ACS-Income at reduced procedural
    # coverage. Labels go BELOW the stars to avoid clashing with the
    # target-region annotation in the top-right corner.
    ("D1 IBM HR", "D1 IBM HR\n(this thesis)", 1470, 9, "thesis", -0.10, -0.85, "center"),
    ("D6 OULAD", "D6 OULAD\n(this thesis)", 32593, 9, "thesis", 0.00, -0.85, "center"),
    ("D2 ACS-Income", "D2 ACS-Income\n(this thesis)", 124226, 8, "thesis", -0.15, -0.85, "right"),
]

STYLE = {
    "hr_ml": dict(
        color=PALETTE["highlight"],
        marker="o",
        size=180,
        zorder=3,
        edgecolor="white",
        linewidth=1.4,
        label="HR-ML applied literature",
    ),
    "fairness_ml": dict(
        color=PALETTE["data"],
        marker="D",
        size=140,
        zorder=3,
        edgecolor="white",
        linewidth=1.2,
        label="Fairness-ML benchmark datasets",
    ),
    "thesis": dict(
        color=PALETTE["accent"],
        marker="*",
        size=320,
        zorder=5,
        edgecolor="white",
        linewidth=1.6,
        label="This thesis",
    ),
}

def main() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    ax.set_facecolor(PALETTE["bg"])
    fig.patch.set_facecolor(PALETTE["bg"])

    # Target-region rectangle: high-N + multi-metric on HR data.
    # Drawn first so points sit on top. Annotation is anchored to the
    # bottom-left corner of the region to avoid clashing with the
    # individual star labels that point upward from the data points.
    target = FancyBboxPatch(
        (np.log10(800), 6.5),
        np.log10(200000) - np.log10(800),
        3.5,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=PALETTE["accent"],
        edgecolor=PALETTE["accent"],
        alpha=0.07,
        linewidth=0.0,
        zorder=1,
    )
    ax.add_patch(target)
    ax.text(
        np.log10(900), 6.85,
        "Target region: HR-relevant scale, multi-metric audit",
        ha="left", va="bottom",
        fontsize=8.0, color=PALETTE["accent"], style="italic",
        zorder=4,
    )

    # HR-ML cluster annotation: a soft band at y=0 extending from ~1k to ~10k
    # to indicate the four other HR-ML papers whose sample sizes are not
    # stated in their abstracts but cluster at zero formal fairness metrics.
    ax.axhspan(
        -0.45, 0.45,
        xmin=(np.log10(700) - 1.9) / (5.4 - 1.9),
        xmax=(np.log10(15000) - 1.9) / (5.4 - 1.9),
        color=PALETTE["highlight"], alpha=0.05, zorder=0,
    )
    ax.text(
        np.log10(3500), -0.85,
        "HR-ML applied literature: zero formal fairness metrics\n(Bhattacharya 2023; Sahinbas 2022; Kediya 2023; Patel 2022)",
        ha="center", va="center",
        fontsize=7.8, color=PALETTE["highlight"], style="italic",
    )

    # Plot each record.
    for short, full, n_rows, n_metrics, tradition, dx, dy, ha in records:
        x = np.log10(n_rows)
        y = float(n_metrics)
        s = STYLE[tradition]
        ax.scatter(
            x, y,
            c=s["color"], marker=s["marker"], s=s["size"],
            edgecolors=s["edgecolor"], linewidths=s["linewidth"],
            zorder=s["zorder"],
            alpha=0.95,
        )
        weight = "bold" if tradition == "thesis" else "normal"
        col = PALETTE["accent"] if tradition == "thesis" else PALETTE["ink"]
        ax.annotate(
            full,
            xy=(x, y),
            xytext=(x + dx, y + dy),
            fontsize=8.0 if tradition != "thesis" else 8.5,
            color=col,
            fontweight=weight,
            ha=ha, va="center",
            multialignment="center",
            arrowprops=dict(
                arrowstyle="-",
                color=PALETTE["rule"],
                lw=0.6, shrinkA=4, shrinkB=2,
            ) if (abs(dx) + abs(dy)) > 0.1 else None,
            zorder=6,
        )

    # X-axis: sample size on log10 scale.
    tick_vals = [100, 300, 1000, 3000, 10000, 30000, 100000]
    ax.set_xticks([np.log10(v) for v in tick_vals])
    ax.set_xticklabels([f"{v:,}" for v in tick_vals], fontsize=8.5)
    ax.set_xlim(1.9, 5.4)

    # Y-axis: integer metric counts.
    ax.set_yticks(list(range(0, 11, 1)))
    ax.set_ylim(-1.4, 10.5)

    ax.set_xlabel("Sample size (rows, log scale)", fontsize=9.5, color=PALETTE["ink"])
    ax.set_ylabel("Number of formal fairness metrics computed", fontsize=9.5, color=PALETTE["ink"])

    ax.grid(axis="y", color=PALETTE["rule"], lw=0.6, alpha=0.5, zorder=0)
    ax.tick_params(colors=PALETTE["ink_soft"], labelsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["rule"])
    ax.spines["bottom"].set_color(PALETTE["rule"])

    # Legend.
    handles = [
        mlines.Line2D(
            [], [], marker="o", color="w",
            markerfacecolor=STYLE["hr_ml"]["color"],
            markeredgecolor="white", markersize=10,
            label="HR-ML applied literature",
        ),
        mlines.Line2D(
            [], [], marker="D", color="w",
            markerfacecolor=STYLE["fairness_ml"]["color"],
            markeredgecolor="white", markersize=9,
            label="Fairness-ML benchmark datasets",
        ),
        mlines.Line2D(
            [], [], marker="*", color="w",
            markerfacecolor=STYLE["thesis"]["color"],
            markeredgecolor="white", markersize=14,
            label="This thesis",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        frameon=False,
        fontsize=8.5,
    )

    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"[lit-figs] Wrote {OUT_PDF}")

if __name__ == "__main__":
    main()

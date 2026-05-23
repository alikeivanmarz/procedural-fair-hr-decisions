"""Generate phase5_eqodds_per_dataset_compact.pdf for thesis_final.

Three-panel grouped bar chart, one panel per fairness metric (macro-DP,
Equal Opportunity, Equalised Odds). Each panel shows three dataset bars
(IBM HR Attrition, OULAD, Ricci) reporting the mean shift in the gap
between the lambda=0 baseline cells and the mitigated lambda>0 cells for
the Equalised Odds Post-processor. Positive bars = gap widening = fairness
worsening.

Reads results/phase5/audit_n30_2026-05-21.csv. Writes the PDF into
thesis_final/figures/. The original thesis_compact/figures/ directory
is not touched, consistent with .
"""
from __future__ import annotations

import sys
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _plot_style import PALETTE, apply_style  # noqa: E402

apply_style()

OUT_DIR = ROOT / "thesis_final" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_CSV = ROOT / "results" / "phase5" / "audit_n30_2026-05-21.csv"

DATASETS = [
    ("ibm_hr_attrition", "D1 IBM HR\nAttrition"),
    ("oulad",            "D2 OULAD"),
    ("ricci",            "D4 Ricci"),
]
METRICS = [
    ("macro_dp",            "macro-DP gap"),
    ("equal_opportunity",   "Equal Opportunity gap"),
    ("macro_eodds",         "Equalised Odds gap"),
]

def _load_eqodds_shifts():
    df = pd.read_csv(AUDIT_CSV)
    df = df[df["method"] == "eqodds_postproc"].copy()
    df = df[df["dataset"].isin([d for d, _ in DATASETS])]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["lambda_"] = pd.to_numeric(df["lambda_"], errors="coerce")
    rows = []
    for dataset, _ in DATASETS:
        for metric, _ in METRICS:
            sub = df[(df["dataset"] == dataset) & (df["metric"] == metric)]
            v0 = sub.loc[sub["lambda_"] == 0, "value"].mean()
            vp = sub.loc[sub["lambda_"] > 0, "value"].mean()
            rows.append({
                "dataset": dataset,
                "metric": metric,
                "lambda0_mean": v0,
                "lambda_pos_mean": vp,
                "shift_pp": (vp - v0) * 100.0,
            })
    return pd.DataFrame(rows)

def render():
    shifts = _load_eqodds_shifts()
    n_metrics = len(METRICS)
    n_datasets = len(DATASETS)

    fig, axes = plt.subplots(1, n_metrics, figsize=(10.6, 3.7), sharey=True)
    x = np.arange(n_datasets)
    bar_w = 0.55
    bar_colour = PALETTE.get("data", "#2b6cb0")

    for ax, (metric, metric_label) in zip(axes, METRICS):
        rows = shifts[shifts["metric"] == metric]
        shifts_pp = [
            rows.loc[rows["dataset"] == d, "shift_pp"].iloc[0]
            for d, _ in DATASETS
        ]
        ax.axhline(0.0, color=PALETTE.get("ink_soft", "#888"),
                   lw=0.6, alpha=0.6)
        ax.bar(x, shifts_pp, bar_w, color=bar_colour,
               edgecolor=PALETTE.get("ink", "#222"), linewidth=0.6)
        for i, v in enumerate(shifts_pp):
            ax.annotate(
                f"{v:+.1f}",
                xy=(x[i], v),
                xytext=(0, 4 if v >= 0 else -10),
                textcoords="offset points",
                ha="center", fontsize=9,
                color=PALETTE.get("ink", "#222"),
            )
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in DATASETS], fontsize=9)
        ax.set_title(metric_label, fontsize=10.5,
                     color=PALETTE.get("ink", "#222"))
        ax.set_ylim(-5.0, 55.0)

    axes[0].set_ylabel("Gap shift, baseline -> mitigated (pp)",
                       fontsize=9.8)
    fig.suptitle(
        "Equalised Odds post-processor: per-dataset gap shifts "
        "(positive = fairness worsening)",
        fontsize=11.2, y=1.04,
    )
    fig.tight_layout()
    out = OUT_DIR / "phase5_eqodds_per_dataset_compact.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print(shifts.to_string(index=False))

if __name__ == "__main__":
    render()

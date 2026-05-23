"""Phase-5 Tier-6 figure script.

Produces two PDFs:
  thesis/figures/phase5_pareto_scatter.pdf  — accuracy vs macro_dp Pareto scatter
  thesis/figures/phase5_effectiveness_heatmap.pdf — accuracy-drop vs DP-improvement

Default inputs are the post-Block-C-bis consolidated outputs from
(pareto_n30_2026-05-21.csv + audit_n30_2026-05-21.csv) so the figures
reflect the four-dataset audit coverage (IBM HR Attrition, Ricci, OULAD,
Dutch Census). ACS-Income's RF-only baseline is disclosed separately in
the chapter prose and is not aggregated into the heatmap because its
per-method coverage is asymmetric (only ~5 of 12 methods ran).

Usage
-----
  python scripts/figs_phase5_pareto.py
    [--in-pareto results/phase5/pareto_n30_2026-05-21.csv]
    [--in-audit  results/phase5/audit_n30_2026-05-21.csv]
    [--out-dir   thesis/figures/]
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared thesis palette + style
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _plot_style import (MITIGATION_COLOURS, PALETTE, apply_style,
                           diverging_cmap)
apply_style()
plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 300,
})

# ---------------------------------------------------------------------------
# Colour mapping: 12 mitigation methods + 3 identity baselines.
# Imported from the shared style module so all thesis figures use the same
# semantic colouring (pre-processing = data-blue family, in-processing =
# accent-sienna family, post-processing = class-accent green family,
# identity = rule grey). See scripts/_plot_style.py for the full mapping.
# ---------------------------------------------------------------------------
METHOD_COLORS: dict[str, str] = MITIGATION_COLOURS

METHOD_KIND_PATCH = {
    "preprocessing":  mpatches.Patch(color=PALETTE['data'],         label="Pre-processing"),
    "inprocessing":   mpatches.Patch(color=PALETTE['accent'],       label="In-processing"),
    "postprocessing": mpatches.Patch(color=PALETTE['class_accent'], label="Post-processing"),
    "baseline":       mpatches.Patch(color=PALETTE['rule'],         label="Baseline"),
}

DATASET_LABELS: dict[str, str] = {
    "ibm_hr_attrition": "IBM HR Attrition",
    "ricci":            "Ricci",
    "oulad":            "OULAD",
    "dutch_census":     "Dutch Census",
}

PLOT_DATASETS = ["ibm_hr_attrition", "ricci", "oulad", "dutch_census"]

# ---------------------------------------------------------------------------
# Figure 1 — Pareto scatter (accuracy vs macro_dp)
# ---------------------------------------------------------------------------

def fig_pareto_scatter(pareto: pd.DataFrame, out_dir: pathlib.Path) -> None:
    # 2x2 grid for the four near-full-coverage datasets.
    fig, axes_2d = plt.subplots(2, 2, figsize=(6.8, 5.0), sharey=False)
    axes = axes_2d.flatten()
    fairness_metric = "macro_dp"

    for ax, ds in zip(axes, PLOT_DATASETS):
        sub = pareto[
            (pareto.dataset == ds) & (pareto.fairness_metric == fairness_metric)
        ].copy()
        if sub.empty:
            ax.set_title(DATASET_LABELS[ds])
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="grey")
            continue

        frontier = sub[sub.on_pareto_frontier]
        non_front = sub[~sub.on_pareto_frontier]

        ax.scatter(
            non_front.seed_mean_accuracy, non_front.seed_mean_fairness,
            c=non_front["method"].map(METHOD_COLORS).fillna("#888888"),
            s=12, alpha=0.4, linewidths=0, zorder=2,
        )
        ax.scatter(
            frontier.seed_mean_accuracy, frontier.seed_mean_fairness,
            c=frontier["method"].map(METHOD_COLORS).fillna("#888888"),
            s=24, alpha=0.9, edgecolors="black", linewidths=0.5, zorder=3,
        )
        ax.set_xlabel("Accuracy", fontsize=8)
        ax.set_ylabel("Macro-DP (lower = fairer)", fontsize=8)
        ax.set_title(DATASET_LABELS[ds], fontsize=9)
        ax.tick_params(labelsize=7)

    # Legend
    handles = list(METHOD_KIND_PATCH.values())
    handles += [
        plt.scatter([], [], s=24, c="white", edgecolors="black",
                    linewidths=0.5, label="Pareto-optimal"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.05), fontsize=7,
               framealpha=0.8)

    fig.suptitle(
        "Phase-5 mitigation matrix: accuracy vs Macro-DP Pareto scatter",
        fontsize=9, y=1.02,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = out_dir / "phase5_pareto_scatter.pdf"
    fig.savefig(out, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"[figs] wrote {out}")

# ---------------------------------------------------------------------------
# Figure 2 — Effectiveness heatmap
# ---------------------------------------------------------------------------

def fig_effectiveness_heatmap(audit: pd.DataFrame, out_dir: pathlib.Path) -> None:
    METHODS = [
        "reweighing", "smote_nc", "di_remover", "optim_preproc", "lfr",
        "adv_debias", "exp_gradient", "gerryfair", "prejudice_remover",
        "eqodds_postproc", "calib_eqodds", "reject_option",
    ]
    ds_filter = audit.dataset.isin(PLOT_DATASETS)
    df = audit[ds_filter & audit.method.isin(METHODS)].copy()

    # Seed-mean per (dataset, base, method, lambda, metric)
    cell = (
        df.groupby(["dataset", "base_model", "method", "lambda_", "metric"])["value"]
        .mean()
        .reset_index()
    )

    # Baseline = lambda=0 per method
    base = (
        cell[cell.lambda_ == 0]
        .rename(columns={"value": "base_val"})
        [["dataset", "base_model", "method", "metric", "base_val"]]
    )
    m = cell[cell.lambda_ > 0].merge(base, on=["dataset", "base_model", "method", "metric"])

    # Accuracy drop (positive = worse accuracy)
    acc = m[m.metric == "accuracy"].copy()
    acc["acc_drop"] = acc["base_val"] - acc["value"]
    acc_agg = acc.groupby("method")["acc_drop"].mean()

    # DP improvement (positive = lower DP = fairer)
    dp = m[m.metric == "macro_dp"].copy()
    dp["dp_imp"] = dp["base_val"] - dp["value"]
    dp_agg = dp.groupby("method")["dp_imp"].mean()

    # Build matrix: rows=methods, cols=[acc_drop, dp_gap_reduction].
    # Values scaled to percentage points to match the units of Table 4.7
    # in the thesis (where 'Acc drop %' and 'DP red %' are expressed in pp).
    tbl = pd.DataFrame({
        "Accuracy drop %\n(positive = costlier)": acc_agg * 100,
        "DP gap reduction %\n(positive = fairer)": dp_agg * 100,
    })
    tbl = tbl.reindex(METHODS).dropna(how="all")

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    im = ax.imshow(tbl.values, aspect="auto", cmap=diverging_cmap(),
                   vmin=-15, vmax=15)

    ax.set_xticks(range(len(tbl.columns)))
    ax.set_xticklabels(tbl.columns, fontsize=8)
    ax.set_yticks(range(len(tbl)))
    ax.set_yticklabels(tbl.index, fontsize=7)

    for i in range(len(tbl)):
        for j in range(len(tbl.columns)):
            v = tbl.iloc[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=6, color=PALETTE['ink'])

    plt.colorbar(im, ax=ax, label="Mean change in percentage points (λ>0 vs λ=0)", fraction=0.03)
    ax_title = "Effectiveness: accuracy vs DP across " + ", ".join(
        DATASET_LABELS[d] for d in PLOT_DATASETS
    )
    ax.set_title(ax_title, fontsize=8)
    plt.tight_layout()
    out = out_dir / "phase5_effectiveness_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"[figs] wrote {out}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-pareto", type=pathlib.Path,
                   default=pathlib.Path("results/phase5/pareto_n30_2026-05-21.csv"))
    p.add_argument("--in-audit", type=pathlib.Path,
                   default=pathlib.Path("results/phase5/audit_n30_2026-05-21.csv"))
    p.add_argument("--out-dir", type=pathlib.Path,
                   default=pathlib.Path("thesis/figures"))
    return p.parse_args(argv)

def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pareto = pd.read_csv(args.in_pareto)
    audit = pd.read_csv(args.in_audit)
    fig_pareto_scatter(pareto, args.out_dir)
    fig_effectiveness_heatmap(audit, args.out_dir)
    print("[figs] done.")

if __name__ == "__main__":
    main()

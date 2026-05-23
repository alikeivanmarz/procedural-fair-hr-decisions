"""Phase-4 followup figures.

Emits the three thesis figures that visualise the C3 separability claim
post followup:

1. ``thesis/figures/phase4_divergence_target.pdf`` — D1 IBM HR
   leaky-vs-honest target divergence under procedural fairness. 2x2 grid,
   one panel per procedural-metric family (process_consistency at σ=0.3,
   voice_enrichment, model_flippability_validity, actionable_validity).
   Bars per model with bootstrap-CI error bars; Constant + Shuffled
   baselines as dashed reference lines.

2. ``thesis/figures/phase4_divergence_notion.pdf`` — statistical-vs-
   procedural rankings per dataset (5-panel: D1-Attrition, D1-PerfRating,
   ACS-Income, Dutch-Census, OULAD-3). Dual-axis with vacuity-flag
   annotations on model labels (vacuity per  / ).

3. ``thesis/figures/phase4_consistency_curve.pdf`` — Process consistency
   noise sweep. 1×5 grid (one panel per dataset), each panel a per-model
   curve over noise σ ∈ {0.1, 0.3, 1.0, 3.0} with bootstrap-CI bands.

Inputs:
    * ``results/phase4/procedural.csv``    — produced by `make phase4`
    * ``results/phase4/per_group_tpr.csv`` — produced by `make phase4-significance`
    * ``results/phase4/significance.csv``  — produced by `make phase4-significance`

Determinism : all model fits in upstream scripts are seeded;
this figure script only consumes CSVs and renders deterministically with
matplotlib's Agg backend. No model fits happen here (the previous
incarnation did fit models inline; the followup moved that work into
`run_phase4_significance.py` so this script is data-only).

CLI
---
.. code-block:: bash

    python scripts/figs_phase4_divergence.py \\
        [--out-dir thesis/figures/]

References
----------
* `` — Phase-4 followup spec (Tier 5 = ).
* the project documentation — determinism.
* the project documentation — procedural CSV schema.
"""

from __future__ import annotations

# Determinism prelude — MUST come before numpy / sklearn imports.
import os

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import argparse  # noqa: E402
import math  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
from typing import Any  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless render

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants — must match run_phase4_significance.py.
# ---------------------------------------------------------------------------

NON_VACUOUS_TPR_THRESHOLD = 0.05

TRAINED_MODELS = (
    "RandomForestClassifier",
    "LogisticRegression",
    "MLPClassifier",
    "XGBClassifier",
    "GradientBoostingClassifier",
    "KNeighborsClassifier",
)
ALL_MODELS = TRAINED_MODELS + ("ConstantPredictor", "ShuffledPredictor")
MODEL_LABELS = {
    "RandomForestClassifier": "RF",
    "LogisticRegression": "LR",
    "MLPClassifier": "MLP",
    "XGBClassifier": "XGB",
    "GradientBoostingClassifier": "GB",
    "KNeighborsClassifier": "KNN",
    "ConstantPredictor": "Const",
    "ShuffledPredictor": "Shuf",
}

# Shared thesis palette: 6 ML model families + 2 reference predictors
# come from scripts/_plot_style.py::MODEL_COLOURS so every figure in the
# thesis uses the same model -> colour mapping.
import sys
import pathlib as _pathlib
sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
from _plot_style import MODEL_COLOURS, PALETTE, apply_style
apply_style()

DATASETS_FIG2 = (
    ("ibm_hr_attrition", "D1 IBM HR — Attrition (honest)"),
    ("ibm_hr_perfrating", "D1 IBM HR — PerformanceRating (leaky)"),
    ("acs_income", "D2 ACS-Income (CA-2018)"),
    ("ricci", "D4 Ricci (firefighter promotion)"),
    ("dutch_census", "D5 Dutch Census"),
    ("oulad", "D6 OULAD (3-class)"),
)

# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _agg_row(
    df: pd.DataFrame,
    *,
    dataset: str,
    model: str,
    metric: str,
    noise_std: float = -1.0,
) -> tuple[float, float, float] | None:
    """Return ``(mean, ci_lo, ci_hi)`` from the seed=-1 aggregate row, or None."""
    sub = df[
        (df["dataset"] == dataset)
        & (df["model"] == model)
        & (df["metric"] == metric)
        & (np.isclose(df["noise_std"], noise_std))
        & (df["seed"] == -1)
    ]
    if not len(sub):
        return None
    row = sub.iloc[0]
    if pd.isna(row["mean"]):
        return None
    return float(row["mean"]), float(row["ci_lo"]), float(row["ci_hi"])

def _vacuity_flag_caption(tpr_df: pd.DataFrame, dataset: str, model: str) -> str:
    """Build the ``|EO|=X.XXX (TPR_g1=Y.YY, TPR_g2=Z.ZZ, non-vacuous=...)`` caption
    chunk per (dataset, model). Returns "" if no row matches.
    """
    sub = tpr_df[(tpr_df["dataset"] == dataset) & (tpr_df["model"] == model)]
    if not len(sub):
        return ""
    if (sub["target_form"] == "binary").all():
        r = sub.iloc[0]
        return (
            f"|EO|={r['abs_eo']:.3f} ({r['tpr_per_group']}, "
            f"non-vacuous={bool(r['non_vacuous_tpr'])})"
        )
    # Multiclass: one TPR string per class, plus non-vacuous flag aggregate.
    pieces = []
    all_non_vac = True
    for _, r in sub.iterrows():
        pieces.append(
            f"c{int(r['class_idx'])}: |EO|={r['abs_eo']:.3f} ({r['tpr_per_group']})"
        )
        if not bool(r["non_vacuous_tpr"]):
            all_non_vac = False
    return "; ".join(pieces) + f"; non-vacuous(all classes)={all_non_vac}"

# ---------------------------------------------------------------------------
# Figure 1 — divergence_target (D1 leaky-vs-honest, 2×2 procedural metrics).
# ---------------------------------------------------------------------------

def render_divergence_target(
    proc_df: pd.DataFrame,
    tpr_df: pd.DataFrame,
    out_path: pathlib.Path,
    log,
) -> None:
    panels = [
        ("process_consistency", 0.3, "Process consistency (σ=0.3)"),
        ("voice_enrichment", -1.0, "Voice enrichment"),
        ("model_flippability_validity", -1.0, "Model flippability (validity)"),
        ("actionable_validity", -1.0, "Explanation actionability (validity)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=False)
    axes = axes.flatten()

    bar_models = list(TRAINED_MODELS)
    x = np.arange(len(bar_models))
    width = 0.38

    captions: list[str] = []

    for ax, (metric, noise, title) in zip(axes, panels):
        leaky_means, leaky_los, leaky_his = [], [], []
        honest_means, honest_los, honest_his = [], [], []
        for m in bar_models:
            l = _agg_row(
                proc_df, dataset="ibm_hr_perfrating", model=m, metric=metric, noise_std=noise
            )
            h = _agg_row(
                proc_df, dataset="ibm_hr_attrition", model=m, metric=metric, noise_std=noise
            )
            for v, mu, lo, hi in (
                (l, leaky_means, leaky_los, leaky_his),
                (h, honest_means, honest_los, honest_his),
            ):
                if v is None:
                    mu.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                else:
                    mu.append(v[0]); lo.append(v[1]); hi.append(v[2])

        leaky_means = np.asarray(leaky_means)
        honest_means = np.asarray(honest_means)
        leaky_err = np.vstack(
            [leaky_means - np.asarray(leaky_los), np.asarray(leaky_his) - leaky_means]
        )
        honest_err = np.vstack(
            [honest_means - np.asarray(honest_los), np.asarray(honest_his) - honest_means]
        )

        ax.bar(
            x - width / 2,
            leaky_means,
            width,
            yerr=leaky_err,
            label="Leaky (PerfRating)",
            color=PALETTE['highlight'],  # rust: alarm semantic for leaky target
            edgecolor=PALETTE['ink'],
            capsize=3,
        )
        ax.bar(
            x + width / 2,
            honest_means,
            width,
            yerr=honest_err,
            label="Honest (Attrition)",
            color=PALETTE['data'],  # deep blue: trustworthy / canonical
            edgecolor=PALETTE['ink'],
            capsize=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in bar_models], rotation=0)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Metric value (higher = fairer)")
        ax.grid(axis="y", linestyle=":", alpha=0.4)

        # Reference lines: Constant + Shuffled honest baseline.
        for ref_model, colour, ls, label in (
            ("ConstantPredictor", "0.6", "--", "Constant"),
            ("ShuffledPredictor", "0.3", ":", "Shuffled"),
        ):
            ref = _agg_row(
                proc_df,
                dataset="ibm_hr_attrition",
                model=ref_model,
                metric=metric,
                noise_std=noise,
            )
            if ref is not None:
                ax.axhline(
                    ref[0], color=colour, linestyle=ls, linewidth=1, label=f"{label} ref"
                )

        # Auto-scaled y-axis. Voice-enrichment can exceed 1 — leave it un-clipped.
        if metric == "voice_enrichment":
            ymax = max(1.4, float(np.nanmax([leaky_means.max(), honest_means.max()]) * 1.1))
            ax.set_ylim(0, ymax)
        else:
            ax.set_ylim(0, 1.05)

        # Caption: rank reversal at top-1.
        if not (np.isnan(leaky_means).any() or np.isnan(honest_means).any()):
            l_best = bar_models[int(np.nanargmax(leaky_means))]
            h_best = bar_models[int(np.nanargmax(honest_means))]
            note = (
                f"{title}: leaky-best={MODEL_LABELS[l_best]}, "
                f"honest-best={MODEL_LABELS[h_best]}"
                f" {'(rank reversal)' if l_best != h_best else '(no top-1 reversal)'}"
            )
            captions.append(note)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        "D1 IBM HR procedural fairness: "
        "leaky vs honest targets (5-seed bootstrap 95% CIs; Holm-corrected significance)",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300, transparent=True)
    plt.close(fig)
    log(f"Wrote {out_path}")
    for c in captions:
        log(f"  caption: {c}")
    # Statistical-fairness caption appendix per  / .
    log("  per-(dataset, model) statistical-fairness vacuity (ibm_hr_attrition):")
    for m in bar_models:
        log(f"    {MODEL_LABELS[m]}: {_vacuity_flag_caption(tpr_df, 'ibm_hr_attrition', m)}")
    log("  per-(dataset, model) statistical-fairness vacuity (ibm_hr_perfrating):")
    for m in bar_models:
        log(f"    {MODEL_LABELS[m]}: {_vacuity_flag_caption(tpr_df, 'ibm_hr_perfrating', m)}")

# ---------------------------------------------------------------------------
# Figure 2 — divergence_notion (5-dataset, statistical-vs-procedural ranks).
# ---------------------------------------------------------------------------

def _statistical_eo_per_model(
    tpr_df: pd.DataFrame, dataset: str
) -> dict[str, float]:
    """Return {model: |EO|} from per_group_tpr.csv. For multiclass datasets,
    |EO| is averaged over non-vacuous classes (Phase-3 macro_filtered convention)."""
    sub = tpr_df[tpr_df["dataset"] == dataset]
    out: dict[str, float] = {}
    for m in TRAINED_MODELS:
        m_sub = sub[sub["model"] == m]
        if not len(m_sub):
            continue
        if (m_sub["target_form"] == "multiclass").any():
            non_vac = m_sub[m_sub["non_vacuous_tpr"]]
            if len(non_vac):
                eo = float(non_vac["abs_eo"].mean())
            else:
                eo = float(m_sub["abs_eo"].mean())
        else:
            eo = float(m_sub["abs_eo"].iloc[0])
        if not np.isnan(eo):
            out[m] = eo
    return out

def _procedural_aggregate_per_model(
    proc_df: pd.DataFrame, dataset: str
) -> dict[str, tuple[float, float, float]]:
    """Per-model procedural aggregate (mean, ci_lo, ci_hi) reproducing the
    headline-claims metric: mean of voice_representation, voice_enrichment
    (clipped at 1), model_flippability_validity, actionable_validity, and
    process_consistency at σ=0.3 — using the seed=-1 aggregate-CSV rows.
    """
    out: dict[str, tuple[float, float, float]] = {}
    metrics = [
        ("voice_representation", -1.0, False),
        ("voice_enrichment", -1.0, True),
        ("model_flippability_validity", -1.0, False),
        ("actionable_validity", -1.0, False),
        ("process_consistency", 0.3, False),
    ]
    for m in TRAINED_MODELS:
        means, los, his = [], [], []
        for metric, noise, clip1 in metrics:
            row = _agg_row(proc_df, dataset=dataset, model=m, metric=metric, noise_std=noise)
            if row is None:
                continue
            mu, lo, hi = row
            if clip1:
                mu = min(1.0, mu); lo = min(1.0, lo); hi = min(1.0, hi)
            means.append(mu); los.append(lo); his.append(hi)
        if not means:
            continue
        out[m] = (
            float(np.mean(means)),
            float(np.mean(los)),
            float(np.mean(his)),
        )
    return out

def render_divergence_notion(
    proc_df: pd.DataFrame,
    tpr_df: pd.DataFrame,
    out_path: pathlib.Path,
    log,
    sig_df: pd.DataFrame | None = None,
    weighting_scheme: str = "equal_weights",
) -> None:
    # 2 x 3 grid (6 datasets). Note  / : Ricci is 3-group sensitive
    # (Race ∈ {W, B, H}); the |EO| column is max-min spread, which is
    # consistent with how multiclass datasets handle multiple TPRs already.
    n_ds = len(DATASETS_FIG2)
    nrows, ncols = 2, 3
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(15, 7), sharey=True)
    axes = axes_grid.flatten()

    for ax, (ds_key, ds_label) in zip(axes, DATASETS_FIG2):
        stat = _statistical_eo_per_model(tpr_df, ds_key)
        proc = _procedural_aggregate_per_model(proc_df, ds_key)
        models = [m for m in TRAINED_MODELS if m in stat and m in proc]
        if not models:
            ax.set_title(ds_label, fontsize=9)
            ax.text(
                0.5, 0.5, "no comparable rows",
                ha="center", va="center", transform=ax.transAxes
            )
            continue

        # Statistical rank: ascending (lower |EO| = fairer = rank 1).
        stat_sorted = sorted(models, key=lambda m: stat[m])
        stat_rank = {m: r for r, m in enumerate(stat_sorted, start=1)}
        # Procedural rank: descending (higher = fairer).
        proc_sorted = sorted(models, key=lambda m: -proc[m][0])
        proc_rank = {m: r for r, m in enumerate(proc_sorted, start=1)}

        x = np.arange(len(models))

        stat_arr = np.asarray([stat_rank[m] for m in models], dtype=float)
        proc_arr = np.asarray([proc_rank[m] for m in models], dtype=float)
        # CI in proc_rank: derive a small ribbon by ranking proc[m][1] /
        # proc[m][2] separately. (Approximate: rank stability across CIs.)
        proc_lo_arr = np.asarray([
            sorted(models, key=lambda mm: -proc[mm][1]).index(m) + 1 for m in models
        ], dtype=float)
        proc_hi_arr = np.asarray([
            sorted(models, key=lambda mm: -proc[mm][2]).index(m) + 1 for m in models
        ], dtype=float)
        proc_err = np.vstack(
            [
                np.maximum(0, proc_arr - np.minimum(proc_lo_arr, proc_hi_arr)),
                np.maximum(0, np.maximum(proc_lo_arr, proc_hi_arr) - proc_arr),
            ]
        )

        # Use the paired-bootstrap rho from significance.csv if available,
        # so the per-panel headers match Table 4.5 in the thesis. Fallback
        # to the small-n raw Spearman (n=6 ranks per dataset) only if
        # significance data is missing.
        rho, rho_lo, rho_hi = float("nan"), float("nan"), float("nan")
        if sig_df is not None and len(sig_df):
            rd = sig_df[
                (sig_df["dataset"] == ds_key)
                & (sig_df["comparison_type"] == "rank_disagreement")
                & (sig_df["weighting_scheme"] == weighting_scheme)
            ]
            if len(rd) >= 1:
                row = rd.iloc[0]
                rho = float(row["mean"])
                rho_lo = float(row.get("ci_lo", float("nan")))
                rho_hi = float(row.get("ci_hi", float("nan")))
        if math.isnan(rho):
            try:
                rho = float(spearmanr(stat_arr, proc_arr).statistic)
            except Exception:
                rho = float("nan")

        ax.plot(x, stat_arr, marker="o", color=PALETTE['accent'],
                linewidth=2, markersize=6, label="Statistical-fairness rank")
        ax.errorbar(
            x, proc_arr, yerr=proc_err,
            marker="s", color=PALETTE['class_accent'],
            linewidth=2, markersize=6,
            capsize=3, label="Procedural-fairness rank (CI)",
        )
        ax.set_xticks(x)
        # Tag vacuous-EO models in the x-axis labels.
        labels = []
        any_non_vac_panel = False
        for m in models:
            sub = tpr_df[(tpr_df["dataset"] == ds_key) & (tpr_df["model"] == m)]
            non_vac = bool((sub["non_vacuous_tpr"] == False).any()) if len(sub) else False
            tag = "*" if non_vac else ""
            labels.append(MODEL_LABELS[m] + tag)
            if not non_vac and len(sub):
                any_non_vac_panel = True
        ax.set_xticklabels(labels, fontsize=8, rotation=0)
        #  /  §Tier-D — annotate panels where every model has
        # vacuous statistical fairness. The reader sees that a "rank
        # reversal" in such a panel is *because* statistical fairness is
        # silent, not despite a clean reading.
        if not any_non_vac_panel and len(models):
            ax.text(
                0.5, 0.92, "[VACUOUS STATISTICAL FAIRNESS]",
                transform=ax.transAxes, fontsize=8.5, ha="center", va="top",
                color=PALETTE['highlight'], fontweight="bold",
                bbox=dict(facecolor=PALETTE['card'],
                          edgecolor=PALETTE['highlight'],
                          boxstyle="round,pad=0.2", linewidth=0.8),
            )
        ax.set_yticks(list(range(1, len(models) + 1)))
        ax.set_ylim(len(models) + 0.5, 0.5)  # rank 1 at top
        ax.set_ylabel("Rank (1 = best)")
        if not math.isnan(rho_lo):
            title = (
                f"{ds_label}\n"
                f"Spearman ρ = {rho:+.3f} "
                f"[{rho_lo:+.2f}, {rho_hi:+.2f}]"
            )
        else:
            title = f"{ds_label}\nSpearman ρ = {rho:+.3f}"
        ax.set_title(title, fontsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        # Reversal note.
        reversed_models = [
            MODEL_LABELS[m] for m in models if stat_rank[m] != proc_rank[m]
        ]
        if reversed_models:
            ax.text(
                0.02, 0.02, "rev: " + ",".join(reversed_models),
                transform=ax.transAxes, fontsize=7, va="bottom",
            )

    # Hide unused panels (none expected; 2x3 = 6 datasets exactly).
    for ax_idx in range(len(DATASETS_FIG2), nrows * ncols):
        axes[ax_idx].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02),
        frameon=False, fontsize=9,
    )
    fig.suptitle(
        "Statistical-vs-procedural rank disagreement: "
        "6-dataset evidence base (Holm-corrected) "
        "(* = at least one group/class with vacuous TPR)",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300, transparent=True)
    plt.close(fig)
    log(f"Wrote {out_path}")
    log("  per-(dataset, model) statistical-fairness vacuity:")
    for ds_key, _ in DATASETS_FIG2:
        log(f"    [{ds_key}]")
        for m in TRAINED_MODELS:
            cap = _vacuity_flag_caption(tpr_df, ds_key, m)
            if cap:
                log(f"      {MODEL_LABELS[m]}: {cap}")

# ---------------------------------------------------------------------------
# Figure 3 — phase4_consistency_curve.
# ---------------------------------------------------------------------------

def render_consistency_curve(
    proc_df: pd.DataFrame,
    out_path: pathlib.Path,
    log,
) -> None:
    nrows, ncols = 2, 3
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(15, 7), sharey=True)
    axes = axes_grid.flatten()
    noise_levels = sorted(
        v for v in proc_df.loc[proc_df["metric"] == "process_consistency", "noise_std"].unique()
        if v > 0
    )

    for ax, (ds_key, ds_label) in zip(axes, DATASETS_FIG2):
        for m in ALL_MODELS:
            means, los, his = [], [], []
            for sigma in noise_levels:
                row = _agg_row(
                    proc_df,
                    dataset=ds_key,
                    model=m,
                    metric="process_consistency",
                    noise_std=sigma,
                )
                if row is None:
                    means.append(np.nan); los.append(np.nan); his.append(np.nan)
                else:
                    means.append(row[0]); los.append(row[1]); his.append(row[2])
            means = np.asarray(means)
            if np.all(np.isnan(means)):
                continue
            los = np.asarray(los); his = np.asarray(his)
            ax.plot(
                noise_levels,
                means,
                marker="o",
                color=MODEL_COLOURS[m],
                linewidth=1.5,
                markersize=5,
                label=MODEL_LABELS[m],
            )
            ax.fill_between(noise_levels, los, his, color=MODEL_COLOURS[m], alpha=0.15)
        ax.set_xscale("log")
        ax.set_xticks(noise_levels)
        ax.set_xticklabels([str(s) for s in noise_levels], fontsize=8)
        ax.set_xlabel("Perturbation σ (log)")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(ds_label, fontsize=9)
        ax.grid(True, which="both", linestyle=":", alpha=0.3)

    # Set y-label on left-column axes.
    for r in range(nrows):
        axes_grid[r, 0].set_ylabel("Process consistency")
    # Hide unused panels (2x3 = 6 datasets exactly; expect none).
    for ax_idx in range(len(DATASETS_FIG2), nrows * ncols):
        axes[ax_idx].axis("off")
    # Pull legend from a panel that actually has lines (some panels can be
    # bare if the dataset is missing).
    legend_handles, legend_labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            legend_handles, legend_labels = h, l
            break
    fig.legend(
        legend_handles, legend_labels,
        loc="lower center", ncol=len(ALL_MODELS), bbox_to_anchor=(0.5, -0.02),
        frameon=False, fontsize=8,
    )
    fig.suptitle(
        "Process consistency vs perturbation magnitude across 6 datasets",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300, transparent=True)
    plt.close(fig)
    log(f"Wrote {out_path}")

# ---------------------------------------------------------------------------
# Top-level driver.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Figure 4 — phase4_rank_disagreement.
# ---------------------------------------------------------------------------

def render_rank_disagreement(
    sig_df: pd.DataFrame,
    out_path: pathlib.Path,
    log,
    *,
    weighting_scheme: str = "equal_weights",
) -> None:
    """Per-dataset Spearman ρ between procedural and statistical rankings.

    Reads the ``rank_disagreement`` rows produced by  in
    ``significance.csv`` (one per (dataset, weighting_scheme)) and emits
    a 1×6 grid of bars with bootstrap-CI error bars.
    """
    if "comparison_type" not in sig_df.columns:
        log("  significance.csv lacks comparison_type column; skipping fig 4")
        return
    rd = sig_df[sig_df["comparison_type"] == "rank_disagreement"].copy()
    if "weighting_scheme" in rd.columns:
        rd = rd[rd["weighting_scheme"] == weighting_scheme]
    if rd.empty:
        log(
            "  no rank_disagreement rows in significance.csv "
            f"(weighting_scheme={weighting_scheme!r}); skipping fig 4"
        )
        return

    fig, ax = plt.subplots(figsize=(11, 4))
    datasets = [ds_key for ds_key, _ in DATASETS_FIG2]
    labels = [ds_label for _, ds_label in DATASETS_FIG2]
    means = []
    los = []
    his = []
    for ds_key in datasets:
        sub = rd[rd["dataset"] == ds_key]
        if sub.empty:
            means.append(np.nan); los.append(np.nan); his.append(np.nan)
            continue
        r = sub.iloc[0]
        means.append(float(r["mean"]))
        los.append(float(r["ci_lo"]))
        his.append(float(r["ci_hi"]))
    means = np.asarray(means, dtype=float)
    los = np.asarray(los, dtype=float)
    his = np.asarray(his, dtype=float)
    err = np.vstack([
        np.where(np.isnan(means), 0.0, means - los),
        np.where(np.isnan(means), 0.0, his - means),
    ])

    x = np.arange(len(datasets))
    # Strength-of-agreement palette: data (deep blue) for strong agreement
    # rho>=0.7; accent (sienna) for weak-positive 0<=rho<0.7; highlight (rust)
    # for negative rho (disagreement).
    colours = [
        PALETTE['data'] if (m >= 0.7) else
        (PALETTE['accent'] if (m >= 0) else PALETTE['highlight'])
        for m in means
    ]
    ax.bar(x, means, yerr=err, color=colours, edgecolor=PALETTE['ink'], capsize=4)
    # Reference lines.
    ax.axhline(1.0, color=PALETTE['ink'], linestyle="--", linewidth=1,
               label="ρ = 1 (perfect agreement)")
    ax.axhline(0.0, color=PALETTE['ink_soft'], linestyle=":", linewidth=1,
               label="ρ = 0 (no relationship)")
    ax.axhline(0.7, color=PALETTE['data'], linestyle=":", linewidth=0.8, alpha=0.6,
               label="ρ = 0.7 (strong agreement)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(-1.20, 1.30)
    ax.set_ylabel("Spearman ρ (procedural-rank vs statistical-rank)")
    ax.set_title(
        "Per-dataset Spearman rank disagreement: procedural vs statistical "
        f"fairness\n(weighting={weighting_scheme}; bootstrap 95% CI)",
        pad=24,
    )
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    # Annotate the value with offsets large enough to clear both the bar end
    # and the bootstrap-CI error caps; positive bars get the label above,
    # negative bars get it below. err is shape (2, N) (lower, upper); we
    # take the appropriate side for each bar.
    for idx, (xi, m) in enumerate(zip(x, means)):
        if np.isnan(m):
            continue
        if m >= 0:
            cap = float(err[1, idx])
            offset = cap + 0.10
        else:
            cap = float(err[0, idx])
            offset = -(cap + 0.10)
        ax.text(xi, m + offset,
                f"{m:+.2f}", ha="center", fontsize=8,
                color=PALETTE['ink'])
    # Legend anchored just above the plotting area so neither the dashed
    # rho = 1 reference line nor the dotted rho = 0 / rho = 0.7 lines
    # cross through the legend text.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=3, fontsize=8, frameon=False)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300, transparent=True)
    plt.close(fig)
    log(f"Wrote {out_path}")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="figs_phase4_divergence",
        description=(
            "Procedural-fairness diagnostic figures. Emits 4 "
            "PDFs under thesis/figures/: divergence_target (D1 leaky-vs-"
            "honest), divergence_notion (6-dataset statistical-vs-"
            "procedural ranks), consistency_curve (process_consistency "
            "vs noise_std), rank_disagreement (per-dataset Spearman ρ "
            "with bootstrap CIs)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(_PROJECT_ROOT / "thesis" / "figures"),
    )
    parser.add_argument(
        "--procedural-csv",
        type=str,
        default=str(_PROJECT_ROOT / "results" / "phase4" / "procedural.csv"),
    )
    parser.add_argument(
        "--tpr-csv",
        type=str,
        default=str(_PROJECT_ROOT / "results" / "phase4" / "per_group_tpr.csv"),
    )
    parser.add_argument(
        "--significance-csv",
        type=str,
        default=str(_PROJECT_ROOT / "results" / "phase4" / "significance.csv"),
    )
    parser.add_argument(
        "--weighting-scheme",
        type=str,
        default="equal_weights",
        help=(
            "Weighting scheme to use for rank-disagreement and headline "
            "filtering when significance.csv has multiple schemes "
            "(default: equal_weights)."
        ),
    )
    args = parser.parse_args(argv)

    proc_csv = pathlib.Path(args.procedural_csv)
    tpr_csv = pathlib.Path(args.tpr_csv)
    sig_csv = pathlib.Path(args.significance_csv)
    if not proc_csv.exists():
        print(f"ERROR: procedural CSV not found at {proc_csv}", file=sys.stderr)
        return 2
    if not tpr_csv.exists():
        print(
            f"ERROR: per_group_tpr CSV not found at {tpr_csv}. "
            "Run `make phase4-significance` first.",
            file=sys.stderr,
        )
        return 2

    proc_df = pd.read_csv(proc_csv)
    tpr_df = pd.read_csv(tpr_csv)
    sig_df = pd.read_csv(sig_csv) if sig_csv.exists() else pd.DataFrame()

    out_dir = pathlib.Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        print(msg, flush=True)

    render_divergence_target(
        proc_df, tpr_df, out_dir / "phase4_divergence_target.pdf", _log
    )
    render_divergence_notion(
        proc_df, tpr_df, out_dir / "phase4_divergence_notion.pdf", _log,
        sig_df=sig_df if not sig_df.empty else None,
        weighting_scheme=args.weighting_scheme,
    )
    render_consistency_curve(
        proc_df, out_dir / "phase4_consistency_curve.pdf", _log
    )
    if not sig_df.empty:
        render_rank_disagreement(
            sig_df, out_dir / "phase4_rank_disagreement.pdf", _log,
            weighting_scheme=args.weighting_scheme,
        )
    else:
        _log("  significance.csv missing or empty; skipping fig 4")
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

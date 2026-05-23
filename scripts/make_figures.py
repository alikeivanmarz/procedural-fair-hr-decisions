"""Regenerate all _compact figures for the compact thesis variant.

Reads existing committed CSV results from results/{phase4,phase5,phase6}
and writes _compact-suffixed PDFs into thesis_compact/figures/.
Per , the originals in thesis/figures/ are never touched.

Twelve mandatory figures (+1 optional):

  1. method_pipeline_compact.pdf       (3 RQs, single contribution x 3 parts, 4 datasets)
  2. method_mitigation_compact.pdf     (D1 + D2 + D4 Ricci, 4 representative methods)
  3. method_inference_stack_compact.pdf (no audit.csv/procedural.csv, no Phase labels)
  4. method_procedural_compact.pdf     (Newman framing pointer corrected)
  5. method_multiclass_proof_compact.pdf (Reduction theorem in Sec 3.5)
  6. lit_gap_matrix_compact.pdf        (single-contribution x three-parts mapping)
  7. lit_research_landscape_compact.pdf (Ricci labelled "this thesis"; optional)
  8. phase4_consistency_curve_compact.pdf
  9. phase4_rank_disagreement_compact.pdf
 10. phase4_divergence_notion_compact.pdf
 11. phase5_pareto_scatter_compact.pdf
 12. phase5_effectiveness_heatmap_compact.pdf
 13. phase6_shap_importance_compact.pdf  (no "" legend label)

Run from repository root:
    python scripts/figs_compact_all.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import textwrap
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from _plot_style import (  # noqa: E402
    PALETTE, MODEL_COLOURS, MITIGATION_COLOURS, FEATURE_COLOURS,
    apply_style, categorical_palette, diverging_cmap,
)

apply_style()

RESULTS = ROOT / "results"
OUT_DIR = ROOT / "thesis_compact" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- dataset filters for the compact thesis ---
COMPACT_PHASE5_DATASETS = ["ibm_hr_attrition", "oulad", "ricci"]
COMPACT_PHASE4_DATASETS = ["ibm_hr_attrition", "ibm_hr_perfrating",
                           "oulad", "ricci", "acs_income"]
HEADLINE_SEPARABILITY_DATASETS = ["ibm_hr_attrition", "ibm_hr_perfrating",
                                  "oulad", "acs_income"]

DATASET_DISPLAY = {
    "ibm_hr_attrition":  "D1 IBM HR Attrition",
    "ibm_hr_perfrating": "D1 IBM HR PerformanceRating",
    "oulad":             "D2 OULAD",
    "acs_income":        "D3 ACS-Income",
    "ricci":             "D4 Ricci",
}

TRAINED_MODELS = (
    "RandomForestClassifier",
    "LogisticRegression",
    "MLPClassifier",
    "XGBClassifier",
    "GradientBoostingClassifier",
    "KNeighborsClassifier",
)

MODEL_LABELS = {
    "RandomForestClassifier": "RF",
    "LogisticRegression": "LR",
    "MLPClassifier": "MLP",
    "XGBClassifier": "XGB",
    "GradientBoostingClassifier": "GB",
    "KNeighborsClassifier": "KNN",
}

def _wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))

def _save(fig, name: str):
    out = OUT_DIR / f"{name}.pdf"
    if out.is_symlink():
        # The compact thesis should not overwrite shared original-thesis figures.
        out.unlink()
    fig.savefig(out, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")

# ===========================================================
# 1. method_pipeline_compact.pdf
# ===========================================================
def fig_method_pipeline_compact():
    fig, ax = plt.subplots(figsize=(11, 7.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Title strip
    ax.text(50, 94,
            "Can procedural fairness be operationalised as ML metrics, "
            "shown to be empirically distinct from\nstatistical fairness on real "
            "HR-relevant data, and used alongside it to evaluate mitigation choices?",
            ha="center", va="center", fontsize=10.8, style="italic",
            color=PALETTE["ink"])
    ax.text(50, 87, "Composite research question",
            ha="center", va="center", fontsize=8.8, color=PALETTE["ink_soft"])

    # RQ row
    rq_y = 76
    rq_labels = ["RQ1", "RQ2", "RQ3"]
    rq_descs = ["Audit", "Procedural fairness\n& separability", "Mitigation across\nboth dimensions"]
    for i, (lbl, desc) in enumerate(zip(rq_labels, rq_descs)):
        x = 20 + i * 30
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 9, rq_y - 5), 18, 10,
            boxstyle="round,pad=0.3", linewidth=1.2,
            facecolor=PALETTE["card"], edgecolor=PALETTE["ink"]))
        ax.text(x, rq_y + 2.5, lbl, ha="center", va="center",
                fontsize=11.8, fontweight="bold", color=PALETTE["ink"])
        ax.text(x, rq_y - 2.0, desc, ha="center", va="center",
                fontsize=8.3, color=PALETTE["ink_soft"])

    # Single contribution band
    ax.add_patch(mpatches.FancyBboxPatch(
        (8, 56), 84, 8,
        boxstyle="round,pad=0.4", linewidth=1.5,
        facecolor=PALETTE["card"], edgecolor=PALETTE["accent"]))
    ax.text(50, 60, "One contribution in three interlocking parts",
            ha="center", va="center", fontsize=11.8, fontweight="bold",
            color=PALETTE["ink"])

    part_x = [20, 50, 80]
    part_lbls = ["Part 1: Audit", "Part 2: Procedural\nfairness operationalisation\n& separability",
                 "Part 3: Mitigation evaluated\non both dimensions"]
    for x, lbl in zip(part_x, part_lbls):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 11, 41), 22, 11,
            boxstyle="round,pad=0.3", linewidth=1.0,
            facecolor=PALETTE["bg"], edgecolor=PALETTE["ink_soft"]))
        ax.text(x, 46.5, lbl, ha="center", va="center", fontsize=8.7,
                color=PALETTE["ink"])
        # Connecting line to RQ above
        rq_idx = part_x.index(x)
        ax.plot([x, 20 + rq_idx * 30], [52.5, 71], lw=0.8,
                color=PALETTE["ink_soft"], linestyle="--", alpha=0.5)

    # Dataset coverage matrix
    ax.text(50, 33, "Dataset coverage", ha="center", va="center",
            fontsize=10.8, fontweight="bold", color=PALETTE["ink"])

    datasets = ["D1 Attrition\n(honest)", "D1 PerfRating\n(leaky motivator)",
                "D2 OULAD\n(multi-class)", "D3 ACS-Income\n(scale + proxy)",
                "D4 Ricci\n(institutional HR)"]
    parts = ["Audit", "Procedural", "Mitigation", "SHAP"]
    cell_w = 14
    cell_h = 4.0
    x0 = 10
    y0 = 8

    # Header row
    for i, ds in enumerate(datasets):
        ax.text(x0 + 7 + i * cell_w, y0 + 4 * cell_h + 3.5, ds,
                ha="center", va="center", fontsize=7.8, color=PALETTE["ink"])
    # Row labels
    for j, p in enumerate(parts):
        ax.text(x0 - 1, y0 + (3 - j) * cell_h + cell_h / 2, p,
                ha="right", va="center", fontsize=8.7, color=PALETTE["ink"])

    # Coverage matrix:
    # rows: Audit / Procedural / Mitigation / SHAP
    # cols: D1-A, D1-PR, D2 OULAD, D3 ACS, D4 Ricci
    coverage = [
        ["F", "F", "F", "F", "F"],     # Audit
        ["F", "F", "F", "F", "F"],     # Procedural
        ["F", "n/a", "F", "P", "F"],   # Mitigation (D1-PR not used; D3 partial; Ricci yes)
        ["F", "n/a", "F", "F", "n/a"], # SHAP (D1-A, D2, D3; not D1-PR, not Ricci)
    ]
    for j, row in enumerate(coverage):
        for i, val in enumerate(row):
            if val == "F":
                fc = PALETTE["class_accent"]
                txt = "used"
            elif val == "P":
                fc = "#D7E6EE"
                txt = "partial"
            else:
                fc = PALETTE["bg"]
                txt = "n/a"
            ax.add_patch(mpatches.Rectangle(
                (x0 + i * cell_w, y0 + (3 - j) * cell_h),
                cell_w, cell_h,
                facecolor=fc, edgecolor=PALETTE["ink_soft"], linewidth=0.6,
                alpha=0.85))
            ax.text(x0 + 7 + i * cell_w, y0 + (3 - j) * cell_h + cell_h / 2,
                    txt, ha="center", va="center", fontsize=8.1,
                    color=PALETTE["ink"])

    _save(fig, "method_pipeline_compact")

# ===========================================================
# 2. method_mitigation_compact.pdf
# ===========================================================
def fig_method_mitigation_compact():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 95, "Mitigation comparison",
            ha="center", va="center", fontsize=12.8, fontweight="bold",
            color=PALETTE["ink"])
    ax.text(50, 89, "Three datasets, four representative methods, N = 30 seeds",
            ha="center", va="center", fontsize=9.8, color=PALETTE["ink_soft"])

    # Datasets column
    ax.text(15, 80, "Datasets", ha="center", va="center",
            fontsize=10.8, fontweight="bold", color=PALETTE["ink"])
    ds_labels = ["D1 IBM HR\nAttrition\n(94.46%)", "D2 OULAD\n(86.03%)",
                 "D4 Ricci\n(91.67%)"]
    for i, lbl in enumerate(ds_labels):
        ax.add_patch(mpatches.FancyBboxPatch(
            (8, 60 - i * 18), 14, 13,
            boxstyle="round,pad=0.3",
            facecolor=PALETTE["card"], edgecolor=PALETTE["ink"], linewidth=1.0))
        ax.text(15, 66.5 - i * 18, lbl, ha="center", va="center",
                fontsize=8.7, color=PALETTE["ink"])

    # Base classifiers column
    ax.text(40, 80, "Base classifiers", ha="center", va="center",
            fontsize=10.8, fontweight="bold", color=PALETTE["ink"])
    bcs = ["LR", "RF", "GB", "XGB", "MLP", "KNN"]
    for i, bc in enumerate(bcs):
        ax.add_patch(mpatches.Rectangle(
            (33 + (i % 3) * 5, 67 - (i // 3) * 7), 4, 5,
            facecolor=MODEL_COLOURS.get(bc, PALETTE["data"]),
            edgecolor=PALETTE["ink"], linewidth=0.6, alpha=0.8))
        ax.text(35 + (i % 3) * 5, 69.5 - (i // 3) * 7, bc,
                ha="center", va="center", fontsize=7.8,
                color="white" if bc in {"RF", "LR"} else PALETTE["ink"])

    # Mitigation methods column
    ax.text(67, 80, "Mitigation methods", ha="center", va="center",
            fontsize=10.8, fontweight="bold", color=PALETTE["ink"])
    methods = [
        ("Reweighing", "pre", "reweighing"),
        ("Adv-Debias", "in", "adv_debias"),
        ("EOdds-Post", "post", "eqodds_postproc"),
        ("LFR", "pre", "lfr"),
    ]
    for i, (name, stage, key) in enumerate(methods):
        ax.add_patch(mpatches.FancyBboxPatch(
            (58, 65 - i * 8), 18, 6,
            boxstyle="round,pad=0.2",
            facecolor=MITIGATION_COLOURS.get(key, PALETTE["data"]),
            edgecolor=PALETTE["ink"], linewidth=0.6, alpha=0.85))
        ax.text(67, 68 - i * 8, f"{name}\n({stage})",
                ha="center", va="center", fontsize=7.8,
                color="white" if stage == "pre" else PALETTE["ink"])

    # Lambda axis
    ax.text(90, 80, "Reg. λ", ha="center", va="center",
            fontsize=10.8, fontweight="bold", color=PALETTE["ink"])
    lambdas = [0, 0.05, 0.1, 0.3, 1, 3, 10]
    for i, l in enumerate(lambdas):
        ax.text(90, 70 - i * 5, f"λ = {l}", ha="center", va="center",
                fontsize=8.2, color=PALETTE["ink_soft"])

    ax.text(50, 6,
            "Nominal grid: base classifiers × methods × λ × seeds; "
            "metrics: accuracy, macro-DP, EO, EOdds, process consistency.",
            ha="center", va="center", fontsize=8.8, style="italic",
            color=PALETTE["ink_soft"])

    _save(fig, "method_mitigation_compact")

# ===========================================================
# 3. method_inference_stack_compact.pdf
# ===========================================================
def fig_method_inference_stack_compact():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 96, "Statistical-inference rigour stack",
            ha="center", va="center", fontsize=12.8, fontweight="bold",
            color=PALETTE["ink"])
    ax.text(50, 91,
            "Each layer filters candidate empirical claims before they enter the headline results.",
            ha="center", va="center", fontsize=8.8, style="italic",
            color=PALETTE["ink_soft"])

    layers = [
        ("Layer 0", "Per-seed metric values",
         "Per-cell measurements for each dataset, model, method, λ, and metric."),
        ("Layer 1", "Bootstrap confidence intervals",
         "10,000 resamples; 95% CI for each mean estimate."),
        ("Layer 2", "Variance-audited effect size",
         "Paired differences with a small variance floor."),
        ("Layer 3", "Weighting robustness",
         "Equal, voice-heavy, transparency-heavy, consistency-heavy, and rank aggregation."),
        ("Layer 4", "Family-wise error control",
         "Holm-Bonferroni at α = 0.05."),
    ]
    y0 = 80
    for i, (name, title, desc) in enumerate(layers):
        y = y0 - i * 12
        ax.add_patch(mpatches.FancyBboxPatch(
            (8, y - 5.4), 67, 10.2,
            boxstyle="round,pad=0.3",
            facecolor=PALETTE["card"], edgecolor=PALETTE["ink"], linewidth=0.8))
        ax.text(11, y + 2.5, name, ha="left", va="center",
                fontsize=8.8, fontweight="bold", color=PALETTE["ink"])
        ax.text(24, y + 2.5, title, ha="left", va="center",
                fontsize=9.1, color=PALETTE["ink"])
        ax.text(11, y - 1.4, _wrap(desc, 70), ha="left", va="center",
                fontsize=7.9, style="italic", color=PALETTE["ink_soft"])
        # downward arrow except for last
        if i < len(layers) - 1:
            ax.annotate("", xy=(42, y - 8.2), xytext=(42, y - 5.7),
                        arrowprops=dict(arrowstyle="->", lw=1.0,
                                        color=PALETTE["ink_soft"]))

    # Parallel branch: Spearman rank-disagreement
    ax.add_patch(mpatches.FancyBboxPatch(
        (79, 31), 19, 49,
        boxstyle="round,pad=0.3",
        facecolor=PALETTE["card"], edgecolor=PALETTE["accent"], linewidth=1.0))
    ax.text(88.5, 76, "Spearman rank\nseparability test",
            ha="center", va="center", fontsize=8.8, fontweight="bold",
            color=PALETTE["ink"])
    ax.text(88.5, 65, "Per dataset:\nH0: ρ ≥ 0.7\nReject if the\nupper CI < 0.7",
            ha="center", va="center", fontsize=7.8, color=PALETTE["ink"])
    ax.text(88.5, 52, "Checked with\npercentile CI,\nBCa CI, Holm,\nand BH-FDR",
            ha="center", va="center", fontsize=7.8, style="italic",
            color=PALETTE["ink_soft"])

    # Headline
    ax.add_patch(mpatches.FancyBboxPatch(
        (8, 12), 90, 10,
        boxstyle="round,pad=0.3",
        facecolor=PALETTE["card"], edgecolor=PALETTE["accent"], linewidth=1.2))
    ax.text(53, 18.5, "Headline empirical claims",
            ha="center", va="center", fontsize=10.8, fontweight="bold",
            color=PALETTE["ink"])
    ax.text(53, 15.2,
            "Weighting-robust pairwise procedural gaps + 4/4 separability rejection on the primary diagnostic set",
            ha="center", va="center", fontsize=8.5, color=PALETTE["ink_soft"])

    _save(fig, "method_inference_stack_compact")

# ===========================================================
# 4. method_procedural_compact.pdf
# ===========================================================
def fig_method_procedural_compact():
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    axes = axes.flatten()
    titles = [
        "Process Consistency",
        "Voice / Representation",
        "Model-Flippability",
        "Explanation-Actionability",
    ]
    subtitles = [
        "How stable is the prediction\nunder semantic noise?",
        "How much does the model use\nmodifiable features?",
        "Can the prediction be reversed by\na sparse counterfactual?",
        "Does the counterfactual flip\na modifiable feature?",
    ]
    body = [
        ("Trained model f\n+ test sample x\n(sample_n = 500)",
         "Gaussian noise on numerics\nσ ∈ {0.1, 0.3, 1.0, 3.0}",
         "JS divergence between original\nand perturbed predictions\nMean over rows; bootstrap CI"),
        ("Trained model f\n+ test sample x",
         "TreeExplainer SHAP attribution\n(or KernelExplainer)",
         "Voice = modifiable attribution\nshare over total attribution\nEnrichment adjusts for feature count"),
        ("Trained model f\n+ test sample x (sample_n = 30)",
         "Greedy k_max = 1 counterfactual\nover numeric features within ±1σ",
         "Sparsity: 1 − #changed/|F|\nValidity: ∃ x′ flipping ŷ"),
        ("Greedy k_max = 1 counterfactual\nreuses model-flippability output",
         "Is the flipped feature in F_mod?",
         "Fraction over rows; bootstrap CI"),
    ]
    for ax, title, subtitle, b in zip(axes, titles, subtitles, body):
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # Title block
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.02, 0.82), 0.96, 0.15,
            boxstyle="round,pad=0.02",
            facecolor=PALETTE["card"], edgecolor=PALETTE["ink"], linewidth=0.8))
        ax.text(0.5, 0.93, title, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=PALETTE["ink"])
        ax.text(0.5, 0.855, subtitle, ha="center", va="center",
                fontsize=8.3, style="italic", color=PALETTE["ink_soft"])
        # Three body blocks
        for i, txt in enumerate(b):
            y = 0.74 - i * 0.22
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.04, y - 0.155), 0.92, 0.16,
                boxstyle="round,pad=0.02",
                facecolor=PALETTE["bg"], edgecolor=PALETTE["ink_soft"], linewidth=0.5))
            ax.text(0.5, y - 0.07, txt, ha="center", va="center",
                    fontsize=8.7, color=PALETTE["ink"])

    fig.suptitle("Procedural-fairness compute pipelines",
                 fontsize=12, fontweight="bold", color=PALETTE["ink"], y=0.98)
    fig.text(0.5, 0.02,
             "Foundation: F = F_mod ∪ F_imm, the modifiable / immutable feature partition "
             "(ascribed-vs-achieved framing of Newman 2020).",
             ha="center", va="center", fontsize=8.8, style="italic",
             color=PALETTE["ink_soft"])
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    _save(fig, "method_procedural_compact")

# ===========================================================
# 5. method_multiclass_proof_compact.pdf
# ===========================================================
def fig_method_multiclass_proof_compact():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.1))
    for ax, title in zip(axes, [
            "Binary EOdds (one point)",
            "Macro-EOdds on three classes (three points)"]):
        ax.set_xlim(-0.02, 0.45)
        ax.set_ylim(-0.02, 0.45)
        ax.set_xlabel(r"$|\Delta\mathrm{FPR}|$ (per class $c$ in right panel)",
                      fontsize=9.8)
        ax.set_ylabel(r"$|\Delta\mathrm{TPR}|$ (per class $c$ in right panel)",
                      fontsize=9.8)
        ax.set_title(title, fontsize=10.8, color=PALETTE["ink"])
        # Symmetric-error diagonal
        ax.plot([0, 0.4], [0, 0.4], lw=0.8, ls="--",
                color=PALETTE["ink_soft"],
                label="symmetric error" if title.startswith("Binary") else None)
        ax.grid(True, alpha=0.3)

    # Left: one point representing binary EOdds
    axes[0].scatter([0.20], [0.20], s=160, color=PALETTE["data"], zorder=5,
                    edgecolor=PALETTE["ink"], linewidth=0.8)
    axes[0].annotate("Binary EOdds\n= max(|ΔTPR|, |ΔFPR|)\n= 0.20",
                     xy=(0.20, 0.20), xytext=(0.27, 0.10),
                     fontsize=8.9, color=PALETTE["ink"],
                     arrowprops=dict(arrowstyle="->", color=PALETTE["ink_soft"]))
    axes[0].text(0.05, 0.40, "Hardt et al. 2016",
                 fontsize=8.8, style="italic", color=PALETTE["ink_soft"])

    # Right: three points per class
    class_pts = [(0.10, 0.10, "c = 1"), (0.20, 0.20, "c = 2"),
                 (0.30, 0.30, "c = 3")]
    for x, y, name in class_pts:
        axes[1].scatter([x], [y], s=140, color=PALETTE["data"], zorder=5,
                        edgecolor=PALETTE["ink"], linewidth=0.8)
        axes[1].annotate(name, xy=(x, y), xytext=(x + 0.012, y - 0.022),
                         fontsize=8.4, color=PALETTE["ink_soft"])
    axes[1].annotate("Macro-EOdds\n= mean of per-class\nmaxima = 0.20",
                     xy=(0.30, 0.30), xytext=(0.16, 0.37),
                     fontsize=8.9, color=PALETTE["ink"],
                     arrowprops=dict(arrowstyle="->", color=PALETTE["ink_soft"]))

    fig.suptitle("Binary-restriction reduction of Macro-EOdds (Methodology §3.5)",
                 fontsize=11.8, fontweight="bold", color=PALETTE["ink"], y=1.01)
    fig.text(0.5, 0.025,
             "Reduction theorem (§3.5): if every class has |ΔTPR_c| = |ΔFPR_c|, "
             "Macro-EOdds equals canonical binary EOdds.\n"
             "For asymmetric errors, Macro-EOdds reduces to the average rather than the maximum.",
             ha="center", va="center", fontsize=9.8, style="italic",
             color=PALETTE["ink_soft"], wrap=True)
    fig.tight_layout(rect=[0, 0.10, 1, 0.94])
    _save(fig, "method_multiclass_proof_compact")

# ===========================================================
# 6. lit_gap_matrix_compact.pdf
# ===========================================================
def fig_lit_gap_matrix_compact():
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    columns = ["HR-ML\napplied", "Fairness-ML\ntheory",
               "Organisational\njustice", "Behavioural\nHR", "This thesis"]
    rows = [
        ("Formal group-fairness metrics", ["absent", "DP, EO,\nEOdds, DI", "absent", "absent", "Part 1"]),
        ("Multi-class fairness extensions", ["binary only", "binary dominant", "absent", "n/a", "Part 1"]),
        ("Procedural fairness as ML metrics", ["absent", "absent", "constructs", "perceptions", "Part 2"]),
        ("Empirical separability test", ["absent", "absent", "constructs", "perceptions", "Part 2"]),
        ("Real HR-decision datasets", ["proxy", "proxy", "n/a", "perceived", "Ricci + IBM"]),
        ("Mitigation evaluated on both dimensions", ["absent", "single criterion", "n/a", "n/a", "Part 3"]),
    ]
    col_x = [29, 43, 58, 73, 88]
    # Header
    ax.text(8, 92, "Property", ha="left", va="center",
            fontsize=9.8, fontweight="bold", color=PALETTE["ink"])
    for c, x in zip(columns, col_x):
        ax.text(x, 92, c, ha="center", va="center",
                fontsize=8.8, fontweight="bold", color=PALETTE["ink"])

    cell_h = 11
    for j, (prop, cells) in enumerate(rows):
        y = 85 - j * cell_h
        ax.text(4, y, _wrap(prop, 22), ha="left", va="center",
                fontsize=8.8, color=PALETTE["ink"])
        for i, val in enumerate(cells):
            x = col_x[i]
            if val.startswith("Part") or val.startswith("Ricci"):
                fc = PALETTE["class_accent"]
                txtcol = "white"
            elif val == "absent":
                fc = PALETTE["highlight"]
                txtcol = "white"
            else:
                fc = PALETTE["card"]
                txtcol = PALETTE["ink"]
            ax.add_patch(mpatches.FancyBboxPatch(
                (x - 6.7, y - 4.2), 13.4, 8.4,
                boxstyle="round,pad=0.15",
                facecolor=fc, edgecolor=PALETTE["ink_soft"], linewidth=0.6,
                alpha=0.85))
            ax.text(x, y, val, ha="center", va="center",
                    fontsize=8.0, color=txtcol, wrap=True)

    fig.suptitle("Literature gap matrix: one contribution closing the procedural-operationalisation gap",
                 fontsize=10.8, fontweight="bold", color=PALETTE["ink"], y=0.99)
    _save(fig, "lit_gap_matrix_compact")

# ===========================================================
# 7. lit_research_landscape_compact.pdf  (optional, included)
# ===========================================================
def fig_lit_research_landscape_compact():
    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    # Scatter: HR-relevant scale vs formal fairness-metric count
    points = [
        # (rows, metric_count, label, color, marker, annotation offset)
        (1470, 9, "D1 IBM HR\n(this thesis)", PALETTE["data"], "o"),
        (31482, 9, "D2 OULAD\n(this thesis)", PALETTE["data"], "o"),
        (195665, 8, "D3 ACS-Income\n(this thesis)", PALETTE["data"], "o"),
        (118, 7, "D4 Ricci\n(this thesis)", PALETTE["data"], "o"),
        # Reference datasets in fairness-ML benchmark ecology
        (48842, 3, "Adult", PALETTE["ink_soft"], "s"),
        (7000, 3, "COMPAS", PALETTE["ink_soft"], "s"),
        (1000, 3, "German credit", PALETTE["ink_soft"], "s"),
        # Applied HR-ML examples with zero formal fairness metrics
        (1109, 0, "", PALETTE["highlight"], "v"),
        (1500, 0, "", PALETTE["highlight"], "v"),
        (1200, 0, "", PALETTE["highlight"], "v"),
        (1100, 0, "", PALETTE["highlight"], "v"),
    ]
    offsets = {
        "D1 IBM HR\n(this thesis)": (8, 8),
        "D2 OULAD\n(this thesis)": (-44, 8),
        "D3 ACS-Income\n(this thesis)": (-58, 12),
        "D4 Ricci\n(this thesis)": (8, 6),
        "Adult": (8, -2),
        "COMPAS": (8, 6),
        "German credit": (8, -10),
    }
    for n, m, lbl, col, mk in points:
        ax.scatter([n], [m], s=140, c=col, marker=mk,
                   edgecolors=PALETTE["ink"], linewidths=0.6, zorder=3)
        if lbl:
            ax.annotate(lbl, xy=(n, m), xytext=offsets.get(lbl, (8, 6)),
                        textcoords="offset points",
                        fontsize=8.1, color=PALETTE["ink"],
                        bbox=dict(boxstyle="round,pad=0.15",
                                  fc="white", ec="none", alpha=0.82))

    # Target region
    ax.add_patch(mpatches.Rectangle(
        (100, 5), 2.4e5 - 100, 6,
        facecolor=PALETTE["class_accent"], edgecolor="none", alpha=0.10,
        zorder=1))
    ax.text(1.5e4, 10.25, "Target region: HR-relevant scale, multi-metric audit",
            fontsize=8.8, color=PALETTE["class_accent"], style="italic",
            ha="center")
    ax.text(2300, 0.45,
            "Applied HR-ML examples:\nzero formal fairness metrics",
            fontsize=8.8, color=PALETTE["highlight"], style="italic",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

    ax.set_xscale("log")
    ax.set_xlabel("Dataset size (rows, log scale)", fontsize=10.5,
                  color=PALETTE["ink"])
    ax.set_ylabel("Number of formal fairness metrics computed",
                  fontsize=10.5, color=PALETTE["ink"])
    ax.set_xlim(80, 3e5)
    ax.set_ylim(-1.0, 11)
    ax.grid(True, which="both", alpha=0.3)

    # Legend
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor=PALETTE["data"], markeredgecolor=PALETTE["ink"],
                   markersize=8, label="This thesis"),
        plt.Line2D([0], [0], marker="s", linestyle="",
                   markerfacecolor=PALETTE["ink_soft"], markeredgecolor=PALETTE["ink"],
                   markersize=8, label="Fairness-ML benchmark"),
        plt.Line2D([0], [0], marker="v", linestyle="",
                   markerfacecolor=PALETTE["highlight"], markeredgecolor=PALETTE["ink"],
                   markersize=8, label="Applied HR-ML literature"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8.7,
              frameon=True, edgecolor=PALETTE["ink_soft"])

    fig.suptitle("Research landscape: HR-relevant scale vs formal-fairness coverage",
                 fontsize=11.6, fontweight="bold", color=PALETTE["ink"], y=0.99)
    fig.tight_layout()
    _save(fig, "lit_research_landscape_compact")

# ===========================================================
# 8. phase4_consistency_curve_compact.pdf
# ===========================================================
def _load_phase4_procedural():
    path = RESULTS / "phase4" / "procedural_n30.csv"
    df = pd.read_csv(path)
    return df

def fig_phase4_consistency_curve_compact():
    df = _load_phase4_procedural()
    # filter to process_consistency metric, compact datasets
    pc = df[(df["metric"] == "process_consistency") &
            (df["dataset"].isin(COMPACT_PHASE4_DATASETS))].copy()
    # use aggregate rows (seed == -1) for mean & CI
    agg = pc[pc["seed"] == -1].copy()
    if agg.empty:
        # fallback: aggregate from per-seed rows
        agg = pc.groupby(["dataset", "target", "model", "noise_std"], as_index=False).agg(
            mean=("value", "mean"),
            ci_lo=("value", lambda v: np.percentile(v, 2.5)),
            ci_hi=("value", lambda v: np.percentile(v, 97.5)),
        )

    agg = agg[agg["model"].isin(TRAINED_MODELS)].copy()

    datasets = COMPACT_PHASE4_DATASETS
    fig, axes_grid = plt.subplots(2, 3, figsize=(10.8, 6.2), sharey=True)
    axes = axes_grid.flatten()

    for ax, ds in zip(axes, datasets):
        sub = agg[agg["dataset"] == ds]
        if sub.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color=PALETTE["ink_soft"])
            ax.set_title(DATASET_DISPLAY.get(ds, ds), fontsize=10,
                         color=PALETTE["ink"])
            continue
        models = [m for m in TRAINED_MODELS if m in set(sub["model"])]
        for m in models:
            mrows = sub[sub["model"] == m].sort_values("noise_std")
            colour = MODEL_COLOURS.get(m, PALETTE["data"])
            ax.plot(mrows["noise_std"], mrows["mean"], "-o",
                    color=colour, ms=4, lw=1.2, label=MODEL_LABELS.get(m, m))
            ax.fill_between(mrows["noise_std"], mrows["ci_lo"], mrows["ci_hi"],
                            color=colour, alpha=0.15)
        ax.set_xscale("log")
        ax.set_xlabel(r"Perturbation $\sigma$ (log)", fontsize=9.5)
        ax.set_ylim(0, 1.05)
        ax.set_title(DATASET_DISPLAY.get(ds, ds), fontsize=10,
                     color=PALETTE["ink"])
        ax.tick_params(labelsize=8.3)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Process Consistency", fontsize=9.8)
    axes[3].set_ylabel("Process Consistency", fontsize=9.8)

    # Single shared legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].axis("off")
    if handles:
        axes[-1].legend(handles, labels, loc="center", ncol=2,
                        frameon=False, title="Model", fontsize=8.5,
                        title_fontsize=9.2)

    fig.suptitle("Process Consistency across perturbation magnitude",
                 fontsize=11.8, fontweight="bold", color=PALETTE["ink"], y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "phase4_consistency_curve_compact")

# ===========================================================
# 9. phase4_rank_disagreement_compact.pdf
# ===========================================================
def _load_separability():
    path = RESULTS / "phase4" / "headline_separability_triangulated.csv"
    df = pd.read_csv(path)
    return df

def fig_phase4_rank_disagreement_compact():
    df = _load_separability()
    # filter to equal_weights scheme and the four headline diagnostic
    # target/dataset combinations
    sub = df[df["weighting_scheme"] == "equal_weights"].copy()
    # dataset values in headline_separability use names like ricci, ACS-Income,
    # IBM-HR-Attrition, IBM-HR-PerformanceRating, OULAD, Dutch
    name_map = {
        "acs_income": "D3 ACS-Income",
        "ibm_hr_attrition": "D1 Attrition",
        "ibm_hr_perfrating": "D1 PerformanceRating",
        "oulad": "D2 OULAD",
        "dutch_census": "(excluded)",
        "ricci": "(excluded; supporting)",
    }
    headline_keys = ["ibm_hr_attrition", "ibm_hr_perfrating",
                     "oulad", "acs_income"]
    sub = sub[sub["dataset"].isin(headline_keys)].copy()
    sub["display"] = sub["dataset"].map(name_map)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    order = headline_keys
    sub = sub.set_index("dataset").loc[order].reset_index()
    xs = np.arange(len(sub))
    rho = sub["rho_mean"].values
    lo = sub["rho_ci_lo"].values
    hi = sub["rho_ci_hi"].values
    err_lo = rho - lo
    err_hi = hi - rho
    colours = [PALETTE["highlight"] if r < 0 else PALETTE["data"] for r in rho]

    ax.bar(xs, rho, yerr=[err_lo, err_hi], width=0.55,
           color=colours, edgecolor=PALETTE["ink"], linewidth=0.7,
           ecolor=PALETTE["ink"], capsize=4, alpha=0.85)

    for x, r, lo_, hi_ in zip(xs, rho, lo, hi):
        sign = "+" if r >= 0 else ""
        ax.text(x, r + (0.05 if r >= 0 else -0.08), f"{sign}{r:.3f}",
                ha="center", va="center", fontsize=8.8, color=PALETTE["ink"])

    ax.axhline(0.7, lw=0.9, ls="--", color=PALETTE["class_accent"])
    ax.text(len(sub) - 0.5, 0.72,
            r"strong-agreement floor $\rho = 0.7$",
            ha="right", va="bottom", fontsize=8.8,
            color=PALETTE["class_accent"])
    ax.axhline(0, lw=0.6, color=PALETTE["ink_soft"])

    ax.set_xticks(xs)
    ax.set_xticklabels(sub["display"], fontsize=9.2)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel(r"Spearman $\rho$ (procedural vs statistical rank)",
                  fontsize=9.7)
    ax.set_title(
        "Per-target Spearman rank disagreement on the primary diagnostic set "
        "(equal_weights scheme; bootstrap 95% CI)",
        fontsize=10.7, color=PALETTE["ink"])
    ax.tick_params(axis="y", labelsize=8.6)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, "phase4_rank_disagreement_compact")

# ===========================================================
# 10. phase4_divergence_notion_compact.pdf
# ===========================================================
def _statistical_rank_per_dataset(tpr_df: pd.DataFrame, dataset: str) -> dict[str, int]:
    """Statistical fairness rank from per-group TPR gaps; lower |EO| is better."""
    sub = tpr_df[tpr_df["dataset"] == dataset]
    rows: list[tuple[str, float]] = []
    for model in TRAINED_MODELS:
        m_sub = sub[sub["model"] == model]
        if m_sub.empty:
            continue
        if (m_sub["target_form"] == "multiclass").any():
            non_vacuous = m_sub[m_sub["non_vacuous_tpr"]]
            eo = float((non_vacuous if not non_vacuous.empty else m_sub)["abs_eo"].mean())
        else:
            eo = float(m_sub["abs_eo"].iloc[0])
        if np.isfinite(eo):
            rows.append((model, eo))
    rows.sort(key=lambda item: item[1])
    return {model: rank for rank, (model, _) in enumerate(rows, start=1)}

def _seed_component(
    df: pd.DataFrame, dataset: str, model: str, seed: int,
    metric: str, noise_std: float,
) -> float | None:
    row = df[
        (df["dataset"] == dataset)
        & (df["model"] == model)
        & (df["seed"] == seed)
        & (df["metric"] == metric)
        & (np.isclose(df["noise_std"], noise_std))
    ]
    if row.empty:
        return None
    value = float(row["value"].iloc[0])
    return value if np.isfinite(value) else None

def _procedural_rank_summary_by_seed(
    proc_df: pd.DataFrame, dataset: str,
) -> dict[str, tuple[float, float, float]]:
    """Mean procedural rank and percentile band across the N=30 seed ranks."""
    metrics = [
        ("voice_representation", -1.0, False),
        ("voice_enrichment", -1.0, True),
        ("model_flippability_validity", -1.0, False),
        ("actionable_validity", -1.0, False),
        ("process_consistency", 0.3, False),
    ]
    seeds = sorted(int(s) for s in proc_df.loc[
        (proc_df["dataset"] == dataset) & (proc_df["seed"] >= 0), "seed"
    ].unique())
    ranks: dict[str, list[int]] = {model: [] for model in TRAINED_MODELS}
    for seed in seeds:
        scores: list[tuple[str, float]] = []
        for model in TRAINED_MODELS:
            vals: list[float] = []
            for metric, noise, clip_one in metrics:
                value = _seed_component(proc_df, dataset, model, seed, metric, noise)
                if value is None:
                    continue
                vals.append(min(1.0, value) if clip_one else value)
            if vals:
                scores.append((model, float(np.mean(vals))))
        scores.sort(key=lambda item: -item[1])
        for rank, (model, _) in enumerate(scores, start=1):
            ranks[model].append(rank)

    summary: dict[str, tuple[float, float, float]] = {}
    for model, model_ranks in ranks.items():
        if not model_ranks:
            continue
        arr = np.asarray(model_ranks, dtype=float)
        summary[model] = (
            float(np.mean(arr)),
            float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 97.5)),
        )
    return summary

def fig_phase4_divergence_notion_compact():
    df = _load_separability()
    proc_df = _load_phase4_procedural()
    tpr_df = pd.read_csv(RESULTS / "phase4" / "per_group_tpr_n30.csv")
    name_map = {
        "acs_income": "D3 ACS-Income",
        "ibm_hr_attrition": "D1 Attrition (honest)",
        "ibm_hr_perfrating": "D1 PerformanceRating (leaky / vacuous)",
        "oulad": "D2 OULAD",
    }
    keys = ["ibm_hr_attrition", "ibm_hr_perfrating",
            "oulad", "acs_income"]
    sub_eq = df[(df["weighting_scheme"] == "equal_weights") &
                df["dataset"].isin(keys)].copy()
    sub_eq = sub_eq.set_index("dataset").loc[keys]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.flatten()

    for ax, key in zip(axes, keys):
        rho = sub_eq.loc[key, "rho_mean"]
        title = name_map[key]
        ax.set_title(f"{title}\nSpearman ρ = {rho:+.3f}",
                     fontsize=9.8, color=PALETTE["ink"])
        stat_rank = _statistical_rank_per_dataset(tpr_df, key)
        proc_rank = _procedural_rank_summary_by_seed(proc_df, key)
        models = [m for m in TRAINED_MODELS if m in stat_rank and m in proc_rank]
        n = len(models)
        for model in models:
            x = stat_rank[model]
            y, lo, hi = proc_rank[model]
            colour = MODEL_COLOURS.get(model, PALETTE["data"])
            ax.errorbar(
                x, y,
                yerr=[[max(0.0, y - lo)], [max(0.0, hi - y)]],
                fmt="o", ms=7, lw=0.8, capsize=2.5,
                color=colour, ecolor=colour,
                markeredgecolor=PALETTE["ink"], markeredgewidth=0.5,
                zorder=4,
            )
            ax.annotate(
                MODEL_LABELS.get(model, model),
                xy=(x, y), xytext=(4, 3), textcoords="offset points",
                fontsize=8.0, color=PALETTE["ink"],
            )
        ax.plot([1, n], [1, n], lw=0.7, ls="--",
                color=PALETTE["ink_soft"], alpha=0.6)
        ax.set_xlim(0.5, n + 0.5)
        ax.set_ylim(0.5, n + 0.5)
        ax.invert_yaxis()
        ax.set_xticks(range(1, n + 1))
        ax.set_yticks(range(1, n + 1))
        ax.set_xlabel("Statistical-fairness rank (1 = best)", fontsize=8.8)
        ax.set_ylabel("Procedural rank, seed mean (1 = best)", fontsize=8.8)
        ax.tick_params(labelsize=8.2)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Statistical-vs-procedural model rankings on the primary diagnostic set",
                 fontsize=11.8, fontweight="bold", color=PALETTE["ink"])
    fig.tight_layout()
    _save(fig, "phase4_divergence_notion_compact")

# ===========================================================
# 11. phase5_pareto_scatter_compact.pdf
# ===========================================================
def _load_phase5_audit():
    path = RESULTS / "phase5" / "audit_n30_2026-05-21.csv"
    df = pd.read_csv(path)
    return df

def _load_phase5_pareto():
    path = RESULTS / "phase5" / "pareto_n30_2026-05-21.csv"
    df = pd.read_csv(path)
    return df

def fig_phase5_pareto_scatter_compact():
    df = _load_phase5_pareto()
    df = df[
        (df["dataset"].isin(COMPACT_PHASE5_DATASETS))
        & (df["accuracy_metric"] == "accuracy")
        & (df["fairness_metric"] == "macro_dp")
    ].copy()

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.2))
    panels = [("ibm_hr_attrition", "D1 IBM HR Attrition"),
              ("oulad", "D2 OULAD"),
              ("ricci", "D4 Ricci")]

    for ax, (ds, title) in zip(axes, panels):
        sub = df[df["dataset"] == ds].copy()
        sub = sub.dropna(subset=["seed_mean_accuracy", "seed_mean_fairness"])
        if sub.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color=PALETTE["ink_soft"])
            ax.set_title(title, fontsize=10, color=PALETTE["ink"])
            continue
        sub["abs_macro_dp"] = sub["seed_mean_fairness"].abs()
        frontier = sub[sub["on_pareto_frontier"]].copy()
        ax.scatter(
            sub["seed_mean_accuracy"], sub["abs_macro_dp"],
            s=16, color=PALETTE["ink_soft"], alpha=0.25,
            edgecolors="none", label="candidate cells",
        )
        if not frontier.empty:
            frontier = frontier.sort_values("seed_mean_accuracy")
            ax.plot(
                frontier["seed_mean_accuracy"], frontier["abs_macro_dp"],
                "-", lw=1.0, color=PALETTE["data"], alpha=0.8,
            )
            ax.scatter(
                frontier["seed_mean_accuracy"], frontier["abs_macro_dp"],
                s=46, color=PALETTE["data"], alpha=0.92,
                edgecolors=PALETTE["ink"], linewidths=0.45,
                label="Pareto frontier",
            )
        ax.set_xlabel("Accuracy", fontsize=9.8)
        ax.set_ylabel("|Macro-DP| (lower = fairer)", fontsize=9.8)
        ax.set_title(
            f"{title}\n{len(sub)} cells; {len(frontier)} frontier",
            fontsize=10.8, color=PALETTE["ink"],
        )
        ax.tick_params(labelsize=8.4)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2,
                   frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=8.7)

    fig.suptitle("Accuracy vs Macro-DP Pareto scatter across the mitigation panel",
                 fontsize=11.8, fontweight="bold", color=PALETTE["ink"], y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    _save(fig, "phase5_pareto_scatter_compact")

# ===========================================================
# 12. phase5_effectiveness_heatmap_compact.pdf
# ===========================================================
def fig_phase5_effectiveness_heatmap_compact():
    df = _load_phase5_audit()
    df = df[df["dataset"].isin(COMPACT_PHASE5_DATASETS)].copy()
    # Match scripts/compute_effectiveness_table_compact.py: average over
    # seeds per lambda, subtract the lambda=0 baseline, then average deltas.
    metrics_to_track = ["accuracy", "macro_dp", "equal_opportunity", "macro_eodds"]
    sub = df[df["metric"].isin(metrics_to_track)].copy()
    grouped = sub.groupby(
        ["dataset", "base_model", "method", "lambda_", "metric"],
        as_index=False,
    )["value"].mean()
    base = grouped[grouped["lambda_"] == 0.0].rename(columns={"value": "value_l0"})
    base = base.drop(columns=["lambda_"])
    mit = grouped[grouped["lambda_"] > 0.0].rename(columns={"value": "value_lmit"})
    merged = mit.merge(base, on=["dataset", "base_model", "method", "metric"],
                       how="left")
    merged["delta"] = merged["value_lmit"] - merged["value_l0"]
    agg = merged.groupby(["method", "metric"], as_index=False)["delta"].mean()
    pivot = agg.pivot(index="method", columns="metric", values="delta")
    # ensure all 4 metric columns exist
    for col in metrics_to_track:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot = pivot[metrics_to_track]
    # for accuracy: negative delta = accuracy improved; flip sign so "good" is +
    pivot_disp = pivot.copy()
    pivot_disp["accuracy"] = -pivot_disp["accuracy"]
    # for fairness metrics: lower gap = better, so delta < 0 = better; flip sign
    for c in ["macro_dp", "equal_opportunity", "macro_eodds"]:
        pivot_disp[c] = -pivot_disp[c]
    # multiply by 100 → percentage points
    pivot_disp = pivot_disp * 100
    pivot_disp = pivot_disp.rename(columns={
        "accuracy": "Acc drop pp",
        "macro_dp": "DP red pp",
        "equal_opportunity": "EO red pp",
        "macro_eodds": "EOdds red pp",
    })

    # Restrict to the 4 representative compact methods, plus a couple of
    # references if present
    compact_methods = ["reweighing", "lfr", "adv_debias", "eqodds_postproc"]
    pivot_disp = pivot_disp.reindex([m for m in compact_methods
                                     if m in pivot_disp.index])

    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    cmap = diverging_cmap()
    vmax = float(np.nanmax(np.abs(pivot_disp.values))) if not pivot_disp.empty else 1.0
    if vmax == 0:
        vmax = 1.0
    im = ax.imshow(pivot_disp.values, cmap=cmap, aspect="auto",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot_disp.columns)))
    ax.set_xticklabels(pivot_disp.columns, fontsize=9.6, color=PALETTE["ink"])
    method_labels = {
        "reweighing": "Reweighing (pre)",
        "lfr": "LFR (pre)",
        "adv_debias": "Adversarial Debiasing (in)",
        "eqodds_postproc": "Equalised-Odds Post (post)",
    }
    ax.set_yticks(range(len(pivot_disp.index)))
    ax.set_yticklabels([method_labels.get(m, m) for m in pivot_disp.index],
                       fontsize=9.6, color=PALETTE["ink"])
    for i in range(len(pivot_disp.index)):
        for j in range(len(pivot_disp.columns)):
            v = pivot_disp.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=9,
                        color="white" if abs(v) > vmax * 0.4 else PALETTE["ink"])
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04,
                 label="signed percentage-point change")
    im.colorbar.ax.tick_params(labelsize=8.5)
    im.colorbar.set_label("signed percentage-point change", fontsize=9.2)
    fig.suptitle("Effectiveness heatmap, averaged across D1 Attrition, D2 OULAD, D4 Ricci",
                 fontsize=10.8, fontweight="bold", color=PALETTE["ink"], y=1.02)
    fig.tight_layout()
    _save(fig, "phase5_effectiveness_heatmap_compact")
    return pivot_disp

# ===========================================================
# 13. phase6_shap_importance_compact.pdf
# ===========================================================
def fig_phase6_shap_importance_compact():
    path = RESULTS / "phase6" / "shap_results.csv"
    if not path.exists():
        # fallback to n200 variant
        path = RESULTS / "phase6" / "shap_results_n200.csv"
    if not path.exists():
        print("WARN: no SHAP results CSV found")
        return
    df = pd.read_csv(path)
    df = df[df["dataset"].isin(["ibm_hr_attrition", "acs_income", "oulad"])].copy()
    # filter to Pareto-optimal mitigated (use mitigated rows) and the overall
    # 'all' group if present
    if "group" in df.columns:
        df = df[df["group"].fillna("all").astype(str).isin(
            ["all", "All", "overall", ""])]
    # take top-5 features per dataset
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.4))
    panels = [("ibm_hr_attrition", "IBM HR Attrition (GB)"),
              ("acs_income", "ACS-Income (RF)"),
              ("oulad", "OULAD (LR)")]
    for ax, (ds, title) in zip(axes, panels):
        sub = df[df["dataset"] == ds].copy()
        if sub.empty:
            ax.set_title(title, fontsize=10, color=PALETTE["ink"])
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color=PALETTE["ink_soft"])
            continue
        # rank features by normalised_share (already normalised) descending
        if "normalised_share" not in sub.columns:
            metric_col = "mean_abs_shap"
        else:
            metric_col = "normalised_share"
        sub = sub.groupby("feature", as_index=False)[metric_col].mean()
        sub = sub.sort_values(metric_col, ascending=False).head(5)
        sub = sub.iloc[::-1]  # flip for horizontal bars (top is largest)
        # colour by feature type if available
        feat_types = []
        type_lookup = (
            df[df["dataset"] == ds]
            .groupby("feature")[["is_sensitive", "is_proxy"]]
            .max()
            .to_dict()
        )
        for f in sub["feature"]:
            is_sens = bool(type_lookup.get("is_sensitive", {}).get(f, False))
            is_proxy = bool(type_lookup.get("is_proxy", {}).get(f, False))
            if is_sens:
                feat_types.append("sensitive")
            elif is_proxy:
                feat_types.append("proxy")
            else:
                feat_types.append("other")
        colours = [FEATURE_COLOURS[t] for t in feat_types]
        labels = [_wrap(str(f), 18) for f in sub["feature"]]
        ax.barh(labels, sub[metric_col], color=colours,
                edgecolor=PALETTE["ink"], linewidth=0.5, alpha=0.85)
        ax.set_xlabel("Normalised mean |SHAP|", fontsize=9.8)
        ax.set_title(title, fontsize=10.8, color=PALETTE["ink"])
        ax.tick_params(axis="both", labelsize=8.5)
        ax.grid(True, axis="x", alpha=0.3)
    # Legend at the right
    legend_handles = [
        mpatches.Patch(color=FEATURE_COLOURS["sensitive"], label="Sensitive attribute"),
        mpatches.Patch(color=FEATURE_COLOURS["proxy"], label="Proxy feature"),
        mpatches.Patch(color=FEATURE_COLOURS["other"], label="Other feature"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=8.8, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("SHAP feature attribution (Pareto-optimal mitigated models)",
                 fontsize=11.8, fontweight="bold", color=PALETTE["ink"], y=1.02)
    fig.tight_layout()
    _save(fig, "phase6_shap_importance_compact")

# ===========================================================
# 14. method_taxonomy.pdf (compact-local copy, not the original symlink)
# ===========================================================
def fig_method_taxonomy_compact():
    fig, ax = plt.subplots(figsize=(11.0, 7.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_axis_off()

    px0, px1 = 12, 92
    py0, py1 = 12, 82
    pxmid = (px0 + px1) / 2
    pymid = (py0 + py1) / 2

    quadrants = [
        (px0, pymid, pxmid, py1, PALETTE["data"], 0.05, "group + distributive"),
        (pxmid, pymid, px1, py1, PALETTE["highlight"], 0.05, "group + procedural"),
        (px0, py0, pxmid, pymid, PALETTE["data"], 0.035, "individual + distributive"),
        (pxmid, py0, px1, pymid, PALETTE["highlight"], 0.07, "individual + procedural"),
    ]
    for x0, y0, x1, y1, colour, alpha, label in quadrants:
        ax.add_patch(mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            facecolor=colour, alpha=alpha, edgecolor="none"))
        ha = "left" if x0 < pxmid else "right"
        va = "top" if y0 >= pymid else "bottom"
        tx = x0 + 1.5 if ha == "left" else x1 - 1.5
        ty = y1 - 1.5 if va == "top" else y0 + 1.5
        ax.text(tx, ty, label, ha=ha, va=va, fontsize=8.8,
                color=PALETTE["ink_soft"], style="italic", weight="bold")

    ax.plot([px0, px1], [pymid, pymid], color=PALETTE["ink"], linewidth=0.9)
    ax.plot([pxmid, pxmid], [py0, py1], color=PALETTE["ink"], linewidth=0.9)
    ax.annotate("", xy=(px1 + 1.2, pymid), xytext=(px0 - 1.2, pymid),
                arrowprops=dict(arrowstyle="<|-|>", color=PALETTE["ink"], lw=0.9))
    ax.annotate("", xy=(pxmid, py1 + 1.2), xytext=(pxmid, py0 - 1.2),
                arrowprops=dict(arrowstyle="<|-|>", color=PALETTE["ink"], lw=0.9))

    ax.text(px0 - 2.5, pymid, "distributive", ha="right", va="center",
            fontsize=9.7, weight="bold")
    ax.text(px1 + 2.5, pymid, "procedural", ha="left", va="center",
            fontsize=9.7, weight="bold")
    ax.text(pxmid, py1 + 2.5, "group level", ha="center", va="bottom",
            fontsize=9.7, weight="bold")
    ax.text(pxmid, py0 - 2.5, "individual level", ha="center", va="top",
            fontsize=9.7, weight="bold")

    metrics = [
        (20, 76, "Demographic Parity", "inherited"),
        (20, 70, "Disparate Impact", "inherited"),
        (30, 73, "Equalised Odds", "inherited"),
        (30, 67, "Equal Opportunity", "inherited"),
        (20, 62, "ABROCA", "inherited"),
        (45, 75, "Macro-DP", "new"),
        (45, 69, "Macro-EOdds", "new"),
        (45, 63, "Macro-EO", "new"),
        (34, 58, "Multinomial CF", "new"),
        (75, 73, "Voice / Representation", "new"),
        (75, 67, "Voice-Enrichment", "new"),
        (24, 32, "KNN Consistency", "inherited"),
        (40, 26, "Counterfactual Fairness (Level 1)", "inherited"),
        (70, 38, "Process Consistency", "new"),
        (75, 28, "Model-Flippability", "new"),
        (75, 18, "Explanation-Actionability", "new"),
    ]
    for x, y, label, kind in metrics:
        edge = PALETTE["highlight"] if kind == "new" else PALETTE["data"]
        weight = "bold" if kind == "new" else "normal"
        linewidth = 1.2 if kind == "new" else 0.9
        box_w = max(11, len(label) * 0.46 + 4)
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - box_w / 2, y - 1.7), box_w, 3.4,
            boxstyle="round,pad=0.18,rounding_size=0.5",
            linewidth=linewidth, edgecolor=edge, facecolor=PALETTE["card"],
            zorder=3))
        ax.text(x, y, label, ha="center", va="center",
                fontsize=8.2, color=PALETTE["ink"], weight=weight, zorder=4)

    ax.text(50, 92, "Fairness criteria used in this thesis",
            ha="center", va="top", fontsize=12.7, weight="bold")
    ax.text(
        50, 88,
        "Placed by what they measure and the level at which they operate",
        ha="center", va="top", fontsize=8.8, color=PALETTE["ink_soft"],
        style="italic",
    )

    for k, (kind, edge, label) in enumerate([
        ("new", PALETTE["highlight"], "introduced or first operationalised in this thesis"),
        ("inherited", PALETTE["data"], "inherited from prior fairness-ML literature"),
    ]):
        lx = 18 + k * 42
        ax.add_patch(mpatches.FancyBboxPatch(
            (lx, 3.0), 5, 2.0,
            boxstyle="round,pad=0.1,rounding_size=0.3",
            linewidth=1.2 if kind == "new" else 0.9,
            edgecolor=edge, facecolor=PALETTE["card"]))
        ax.text(lx + 6, 4.0, label, ha="left", va="center",
                fontsize=8.2, color=PALETTE["ink"])

    _save(fig, "method_taxonomy")

# ===========================================================
# 15. phase6_oulad_cross_arch.pdf (compact-local copy)
# ===========================================================
def fig_phase6_oulad_cross_arch_compact():
    path = RESULTS / "phase6" / "oulad_cross_arch.csv"
    if not path.exists():
        print("WARN: no OULAD cross-architecture CSV found")
        return
    df = pd.read_csv(path)
    rows = [("LR", 0.087345508990378)]
    for _, row in df.iterrows():
        rows.append((str(row["model"]), float(row["gender_share"])))

    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    labels = [model for model, _ in rows]
    values = [value for _, value in rows]
    colours = [MODEL_COLOURS.get(label, PALETTE["data"]) for label in labels]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colours, edgecolor=PALETTE["ink"],
            linewidth=0.45, alpha=0.92, zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.105)
    ax.set_xlabel("Gender attribution share (mean |SHAP| / total |SHAP|)",
                  fontsize=9.8)
    ax.axvline(0.018, color=PALETTE["highlight"], linestyle="--",
               linewidth=1.0, alpha=0.85, zorder=2)
    ax.text(0.020, -0.58, "IBM HR Attrition tree-model reference (~2%)",
            ha="left", va="bottom", fontsize=8.8, color=PALETTE["highlight"],
            style="italic")
    for yi, value in zip(y_pos, values):
        ax.text(value + 0.0015, yi, f"{value * 100:.1f}%",
                va="center", fontsize=9.8, color=PALETTE["ink"])
    ax.tick_params(axis="x", labelsize=8.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["rule"])
    ax.spines["bottom"].set_color(PALETTE["rule"])
    ax.grid(axis="x", color=PALETTE["rule"], alpha=0.65, lw=0.5, zorder=0)
    fig.tight_layout()
    _save(fig, "phase6_oulad_cross_arch")

# ===========================================================
# Main driver
# ===========================================================
def main():
    print(f"Writing compact figures to {OUT_DIR.relative_to(ROOT)}/")

    fig_method_pipeline_compact()
    fig_method_mitigation_compact()
    fig_method_inference_stack_compact()
    fig_method_procedural_compact()
    fig_method_multiclass_proof_compact()
    fig_method_taxonomy_compact()
    fig_lit_gap_matrix_compact()
    fig_lit_research_landscape_compact()
    fig_phase4_consistency_curve_compact()
    fig_phase4_rank_disagreement_compact()
    fig_phase4_divergence_notion_compact()
    fig_phase5_pareto_scatter_compact()
    fig_phase5_effectiveness_heatmap_compact()
    fig_phase6_shap_importance_compact()
    fig_phase6_oulad_cross_arch_compact()

    print("Done.")

if __name__ == "__main__":
    main()

"""Phase-6 SHAP feature-importance figure.

Reads results/phase6/shap_results[_n{N}].csv and produces
thesis/figures/phase6_shap_importance[_n{N}].pdf:

  3 panels (one per dataset), each with a horizontal bar chart of
  normalised mean |SHAP| for top-10 features, coloured by:
    * red    — sensitive attribute ( attribute (S))
    * orange — known proxy features (from Phase-2 Bayesian audit)
    * blue   — other features

  Two bars per feature: unmitigated baseline vs Pareto-optimal mitigated.

CLI
---
  python scripts/figs_phase6_shap.py [--background-n N]

When --background-n N is given (N != 50), reads shap_results_n{N}.csv
and writes phase6_shap_importance_n{N}.pdf.  The default (N=50) reads
the bare filename and writes the bare output filename so existing
behaviour is preserved .

References
----------

* results/phase6/shap_results.csv — produced by scripts/run_phase6_shap.py.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Shared thesis palette (semantic feature-type roles).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _plot_style import FEATURE_COLOURS, PALETTE, apply_style
apply_style()

_DEFAULT_RESULTS_CSV = PROJECT_ROOT / "results" / "phase6" / "shap_results.csv"
_DEFAULT_OUT_PDF = PROJECT_ROOT / "thesis" / "figures" / "phase6_shap_importance.pdf"

DATASETS_ORDER = ["ibm_hr_attrition", "acs_income", "oulad"]
DATASET_LABELS = {
    "ibm_hr_attrition": "IBM HR (Attrition)",
    "acs_income": "ACS Income",
    "oulad": "OULAD",
}

TOP_N = 10
COLOUR_SENSITIVE = FEATURE_COLOURS['sensitive']  # palette: highlight (rust)
COLOUR_PROXY     = FEATURE_COLOURS['proxy']      # palette: accent (sienna)
COLOUR_OTHER     = FEATURE_COLOURS['other']      # palette: data (deep blue)

METHOD_LABELS = {
    "identity_preprocessing": "Unmitigated",
}
DEFAULT_MITIGATED_LABEL = "Pareto-optimal"

def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, DEFAULT_MITIGATED_LABEL)

def _feature_colour(row: pd.Series) -> str:
    if row["is_sensitive"]:
        return COLOUR_SENSITIVE
    if row["is_proxy"]:
        return COLOUR_PROXY
    return COLOUR_OTHER

def _plot_dataset(ax: plt.Axes, df: pd.DataFrame, dataset: str, title: str) -> None:
    """Render one horizontal-bar panel for *dataset*.

    Shows top-10 features by mean normalised_share across all methods,
    with two bars per feature (unmitigated vs mitigated).
    """
    sub = df[(df["dataset"] == dataset) & (df["group"] == "all")].copy()
    if sub.empty:
        ax.set_title(f"{title}\n(no data)")
        ax.axis("off")
        return

    # Identify the two method variants present.
    methods = sub["method"].unique().tolist()
    identity = [m for m in methods if m == "identity_preprocessing"]
    mitigated = [m for m in methods if m != "identity_preprocessing"]

    baseline_method = identity[0] if identity else methods[0]
    mitigated_method = mitigated[0] if mitigated else None

    # Pick top-10 features by mean normalised_share across both methods.
    pivot = (
        sub.groupby("feature")["normalised_share"]
        .mean()
        .sort_values(ascending=False)
        .head(TOP_N)
    )
    top_features = pivot.index.tolist()

    # Build lookup: feature -> normalised_share per method.
    def _shares(method: str) -> dict[str, float]:
        m_sub = sub[sub["method"] == method]
        return m_sub.set_index("feature")["normalised_share"].to_dict()

    base_shares = _shares(baseline_method)
    mitig_shares = _shares(mitigated_method) if mitigated_method else {}

    # Colour per feature (based on any row of that feature).
    feat_meta = sub.drop_duplicates("feature").set_index("feature")

    y = np.arange(len(top_features))
    bar_height = 0.35

    colours = [
        _feature_colour(feat_meta.loc[f]) if f in feat_meta.index
        else COLOUR_OTHER
        for f in top_features
    ]

    base_vals = [base_shares.get(f, 0.0) for f in top_features]
    mitig_vals = [mitig_shares.get(f, 0.0) for f in top_features]

    # Two bars per feature; unmitigated offset up, mitigated offset down.
    bars1 = ax.barh(
        y + bar_height / 2,
        base_vals,
        height=bar_height,
        color=colours,
        alpha=0.9,
        label=_method_label(baseline_method),
    )
    if mitig_vals and any(v > 0 for v in mitig_vals):
        ax.barh(
            y - bar_height / 2,
            mitig_vals,
            height=bar_height,
            color=colours,
            alpha=0.4,
            label=_method_label(mitigated_method or ""),
        )

    ax.set_yticks(y)
    ax.set_yticklabels(top_features, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Normalised mean |SHAP|", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.tick_params(axis="x", labelsize=8)

    # Colour y-tick labels to match feature classification.
    for tick, feat in zip(ax.get_yticklabels(), top_features):
        if feat in feat_meta.index:
            tick.set_color(_feature_colour(feat_meta.loc[feat]))

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SHAP feature-importance figure"
    )
    p.add_argument(
        "--background-n",
        type=int,
        default=50,
        dest="background_n",
        help=(
            "KernelExplainer background N used when generating shap_results.csv "
            "(default: 50).  When N != 50, reads shap_results_n{N}.csv and writes "
            "phase6_shap_importance_n{N}.pdf so the N=50 baseline is preserved "
            ""
        ),
    )
    return p.parse_args()

def main(background_n: int = 50) -> None:
    """Generate the Phase-6 SHAP importance figure.

    Parameters
    ----------
    background_n:
        The KernelExplainer background N that was used to produce the
        results CSV.  Controls both the input and output filenames:
        * 50  → shap_results.csv           → phase6_shap_importance.pdf
        * 200 → shap_results_n200.csv      → phase6_shap_importance_n200.pdf
    """
    suffix = "" if background_n == 50 else f"_n{background_n}"
    results_csv = (
        PROJECT_ROOT / "results" / "phase6" / f"shap_results{suffix}.csv"
    )
    out_pdf = (
        PROJECT_ROOT / "thesis" / "figures" / f"phase6_shap_importance{suffix}.pdf"
    )

    if not results_csv.exists():
        raise FileNotFoundError(
            f"results CSV not found: {results_csv}\n"
            f"Run scripts/run_phase6_shap.py --background-n {background_n} first."
        )

    df = pd.read_csv(results_csv)

    # Ensure bool columns are bool.
    for col in ("is_sensitive", "is_proxy"):
        if col in df.columns:
            df[col] = df[col].astype(bool)

    datasets_present = [d for d in DATASETS_ORDER if d in df["dataset"].unique()]
    n_panels = len(datasets_present)
    if n_panels == 0:
        raise ValueError(f"No datasets found in {results_csv}")

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(5 * n_panels, 8),
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets_present):
        _plot_dataset(ax, df, ds, DATASET_LABELS.get(ds, ds))

    # Legend explaining colours (added to the last panel).
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOUR_SENSITIVE, label="Sensitive attribute"),
        Patch(facecolor=COLOUR_PROXY, label="Proxy feature"),
        Patch(facecolor=COLOUR_OTHER, label="Other feature"),
    ]
    axes[-1].legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=7,
        title="Feature type",
        title_fontsize=8,
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"[phase6-figs] Wrote {out_pdf}")

if __name__ == "__main__":
    args = _parse_args()
    main(background_n=args.background_n)

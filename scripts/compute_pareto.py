"""Phase-5 Pareto-frontier computation.

Reads the consolidated Phase-5 audit CSV (``results/phase5/audit.csv``,
schema per ) and computes, per ``(dataset, fairness_metric)``,
the Pareto-optimal frontier over points
``(seed-mean accuracy, seed-mean fairness)`` indexed by
``(base_model, method, lambda_)``.

Pareto definition
-----------------

For an accuracy axis where higher is better and a fairness axis where
"more fair" means SMALLER absolute gap (e.g., ``macro_dp``,
``macro_eodds``, ``equal_opportunity``), point ``p`` is Pareto-dominated
by ``q`` iff:

    q.accuracy >= p.accuracy  AND  q.fairness <= p.fairness
    AND  (q.accuracy >  p.accuracy  OR  q.fairness <  p.fairness)

For fairness metrics where higher is better
(``counterfactual_fairness``: 1 = perfectly fair), the inequality on
the fairness axis flips. The sign convention is encoded in
``FAIRNESS_HIGHER_IS_BETTER`` below.

Inputs / outputs
----------------

CLI::

    python scripts/compute_pareto.py [--in-csv RESULTS] [--out-dir DIR]
                                     [--accuracy-metric NAME]
                                     [--fairness-metrics CSV]

Outputs:
    * ``results/phase5/pareto.csv`` with columns
      ``[dataset, accuracy_metric, fairness_metric, base_model, method,
        lambda_, seed_mean_accuracy, seed_mean_fairness,
        on_pareto_frontier]``.
    * ``results/phase5/pareto_summary.md`` — top 1-2 frontier-optimal
      cells per ``(dataset, fairness_metric)``.

Determinism
-----------

The output CSV is sorted lexicographically by
``(dataset, fairness_metric, base_model, method, lambda_)`` so two
clean runs produce a byte-identical file (extends  to the
Pareto step).

Limitations / scope
-------------------

Per  we do not silently drop "failed" cells: cells whose
``notes`` column flags a graceful fallback to the base estimator
(e.g., ``reweighing_failed``, ``optim_preproc_failed``,
``lfr_failed``) are EXCLUDED from the frontier because their
metric values are really base-estimator points masquerading as
the named method.
The exclusion is logged in the summary report so a reader can audit it.

 augmented filter
------------------------

The existing token-based filter does not catch *silent* backend
fallbacks: cases where a wrapper completes without raising but
internally ignores ``base_estimator`` and substitutes its own
fixed classifier (the canonical offender is the AIF360 Prejudice
Remover wrapper at ``src/mitigation/inprocessing.py:302--306``).
Symptom: the resulting ``(accuracy, balanced_accuracy)`` pair is
byte-identical across every base classifier present for the
``(dataset, method, lambda_, seed)`` group.

Per  we therefore add a structural-invariance pass: for
every ``(dataset, method, lambda_, seed)`` group with at least two
base classifiers, if the inter-base-model range of both
``accuracy`` and ``balanced_accuracy`` is below
``ADR033_TOLERANCE`` (1e-9), every row of that group is dropped
before aggregation. The check is method-agnostic — it triggers
symmetrically on any wrapper that exhibits the same structural
signature, not only Prejudice Remover. A documentation-only
whitelist ``KNOWN_BASE_BLIND_METHODS`` records the methods we
already know about so the augmented-filter audit can attribute
the drops; the whitelist does not affect filter behaviour.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------
# Sign convention for fairness metrics.
#   * False = SMALLER is more fair (gap-style metrics: DP, EOdds, EO).
#   * True  = LARGER is more fair (similarity-style metrics: CF).
# Add new fairness metrics here as the Tier-5 panel evolves.
# ---------------------------------------------------------------------

FAIRNESS_HIGHER_IS_BETTER: dict[str, bool] = {
    "macro_dp": False,
    "macro_eodds": False,
    "equal_opportunity": False,
    "counterfactual_fairness": True,
}

# Fairness metrics included by default on the CLI (matches ).
DEFAULT_FAIRNESS_METRICS: list[str] = list(FAIRNESS_HIGHER_IS_BETTER.keys())

# Performance / accuracy metrics emitted by the Tier-5 panel.
ACCURACY_METRICS: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
)

# Substrings in the ``notes`` column that mark a graceful fallback to
# the base estimator.
# Cells flagged by any of these tokens are EXCLUDED from the frontier.
# Only exclude cells where the MITIGATION METHOD itself failed and the runner
# fell back to the base estimator (making the cell a disguised baseline point).

# IMPORTANT: individual metric failures (counterfactual_fairness_failed:,
# process_consistency_failed:, eo_modal_class=1, etc.) do NOT invalidate the
# cell's accuracy / fairness values and must NOT appear in this list.

# These tokens match method-level prefixes only — none of them match
# "counterfactual_fairness_failed:", "equal_opportunity_failed:", etc.
FAILED_CELL_NOTE_TOKENS: tuple[str, ...] = (
    "reweighing_failed:",            # Reweighing → base estimator
    "lfr_failed:",                   # LFR → base estimator
    "adv_debias_failed:",            # AdvDebias → base estimator
    "eqodds_postproc_failed:",       # EQOdds postproc → base estimator
    "_predict_failed:",              # predict-path subprocess failure → base estimator
    "N/A —",                         # method not applicable (e.g. reweighing on KNN)
)

# ---------------------------------------------------------------------
#  augmented filter — structural-invariance detection.

# A group of audit rows sharing the same (dataset, method, lambda_,
# seed) and varying only over base_model is flagged as silent backend
# fallback iff the inter-base-model range of BOTH ``accuracy`` and
# ``balanced_accuracy`` is below ``ADR033_TOLERANCE``. Single-base-model
# groups cannot be assessed and are never flagged.

# The KNOWN_BASE_BLIND_METHODS list is documentation-only: it records
# methods whose wrappers we have already audited and confirmed to ignore
# ``base_estimator``. It does not affect filter behaviour; it is logged
# alongside the augmented-filter count for traceability.
# ---------------------------------------------------------------------

ADR033_TOLERANCE: float = 1e-9

KNOWN_BASE_BLIND_METHODS: tuple[str, ...] = (
    # AIF360 wrapper at src/mitigation/inprocessing.py:302--306 documents
    # that base_estimator is ignored;  (A+ extension Wave 2)
    # confirmed the symptom empirically on IBM HR Attrition.
    "prejudice_remover",
)

# ---------------------------------------------------------------------
# Pareto-frontier core.
# ---------------------------------------------------------------------

def compute_pareto_mask(
    accuracy: np.ndarray,
    fairness: np.ndarray,
    fairness_higher_is_better: bool,
) -> np.ndarray:
    """Return a boolean array; ``True`` iff the point is on the frontier.

    Convention: accuracy is ALWAYS treated as "higher is better".
    Fairness is "higher is better" iff ``fairness_higher_is_better``.

    A point ``p`` is dominated by ``q`` iff ``q`` is at least as good on
    BOTH axes and STRICTLY better on at least one. NaN points are never
    on the frontier (and never dominate anything).
    """
    accuracy = np.asarray(accuracy, dtype=float)
    fairness = np.asarray(fairness, dtype=float)
    n = len(accuracy)
    on_frontier = np.zeros(n, dtype=bool)

    # Encode "higher is better" for both axes by flipping fairness sign.
    fair_signed = fairness if fairness_higher_is_better else -fairness

    finite = np.isfinite(accuracy) & np.isfinite(fair_signed)
    for i in range(n):
        if not finite[i]:
            continue
        dominated = False
        for j in range(n):
            if i == j or not finite[j]:
                continue
            ge_acc = accuracy[j] >= accuracy[i]
            ge_fair = fair_signed[j] >= fair_signed[i]
            gt_acc = accuracy[j] > accuracy[i]
            gt_fair = fair_signed[j] > fair_signed[i]
            if ge_acc and ge_fair and (gt_acc or gt_fair):
                dominated = True
                break
        on_frontier[i] = not dominated
    return on_frontier

# ---------------------------------------------------------------------
# Filtering helpers.
# ---------------------------------------------------------------------

def is_failed_cell(notes: str | float) -> bool:
    """Return True if the cell's notes flag a graceful-fallback failure."""
    if not isinstance(notes, str) or not notes:
        return False
    return any(tok in notes for tok in FAILED_CELL_NOTE_TOKENS)

def detect_silent_backend_fallback(
    df: pd.DataFrame,
    *,
    tol: float = ADR033_TOLERANCE,
) -> pd.Series:
    """Flag audit rows belonging to a silent-backend-fallback cell.

    A group of rows sharing ``(dataset, method, lambda_, seed)`` is
    flagged iff (a) it spans at least two distinct ``base_model``
    values and (b) the inter-base-model range of BOTH ``accuracy`` and
    ``balanced_accuracy`` is below ``tol`` (default ``ADR033_TOLERANCE``
    = 1e-9). Groups with a single base_model cannot be assessed and
    are never flagged.

    Parameters
    ----------
    df:
        Long-format audit DataFrame conforming to  schema. Must
        contain rows for both ``accuracy`` and ``balanced_accuracy`` —
        groups lacking either metric are not flagged.
    tol:
        Float tolerance for "invariant".

    Returns
    -------
    pandas.Series
        Boolean Series indexed identically to ``df``; ``True`` for rows
        in a flagged group.
    """
    if df.empty:
        return pd.Series([], dtype=bool, index=df.index)

    perf = df[df["metric"].isin(("accuracy", "balanced_accuracy"))]
    if perf.empty:
        return pd.Series(False, index=df.index)

    # Per-cell (one row per ds, method, lam, seed, base_model, metric)
    # value — defensive against duplicate rows (uses mean as aggfunc).
    panel = perf.pivot_table(
        index=["dataset", "method", "lambda_", "seed", "base_model"],
        columns="metric",
        values="value",
        aggfunc="mean",
    )
    if not {"accuracy", "balanced_accuracy"}.issubset(panel.columns):
        return pd.Series(False, index=df.index)

    panel = panel.reset_index()
    # Per-group inter-base-model range.
    grp_keys = ["dataset", "method", "lambda_", "seed"]
    grp = panel.groupby(grp_keys, dropna=False)
    n_base = grp["base_model"].nunique()
    acc_range = grp["accuracy"].apply(lambda s: s.max() - s.min())
    bacc_range = grp["balanced_accuracy"].apply(lambda s: s.max() - s.min())
    invariant = (n_base >= 2) & (acc_range < tol) & (bacc_range < tol)
    invariant.name = "_silent_fallback"

    # Broadcast back to df. Build a key-DataFrame index-aligned to df.
    flagged_keys = invariant[invariant].index.to_frame(index=False)
    if flagged_keys.empty:
        return pd.Series(False, index=df.index)

    merged = df.reset_index().merge(
        flagged_keys.assign(_silent_fallback=True),
        on=grp_keys,
        how="left",
    )
    # Convert via numpy to avoid pandas object-dtype downcasting warnings.
    sf_bool = (
        merged["_silent_fallback"].to_numpy() == True  # noqa: E712
    )
    return pd.Series(sf_bool, index=df.index, name="_silent_fallback")

def summarise_silent_fallback(
    df: pd.DataFrame,
    flagged: pd.Series,
) -> pd.DataFrame:
    """Summarise the augmented-filter drops per (dataset, method).

    Returns a DataFrame with columns
    ``[dataset, method, n_rows_dropped, n_groups_dropped,
       in_known_base_blind_whitelist]``.

    The ``in_known_base_blind_whitelist`` column is True iff the method
    appears in ``KNOWN_BASE_BLIND_METHODS`` — documentation only; the
    filter applies symmetrically regardless of whitelist membership.
    """
    if not flagged.any():
        return pd.DataFrame(
            columns=[
                "dataset",
                "method",
                "n_rows_dropped",
                "n_groups_dropped",
                "in_known_base_blind_whitelist",
            ]
        )
    sub = df.loc[flagged]
    rows = (
        sub.groupby(["dataset", "method"], as_index=False)
        .agg(
            n_rows_dropped=("value", "size"),
            n_groups_dropped=(
                "seed",
                lambda s: sub.loc[s.index]
                .groupby(["lambda_", "seed"])
                .ngroups,
            ),
        )
    )
    rows["in_known_base_blind_whitelist"] = rows["method"].isin(
        KNOWN_BASE_BLIND_METHODS
    )
    return rows.sort_values(["dataset", "method"]).reset_index(drop=True)

def aggregate_seed_means(
    df: pd.DataFrame,
    *,
    apply_adr033_filter: bool = True,
) -> pd.DataFrame:
    """Aggregate per-seed metric rows into per-cell seed-mean values.

    Input: a long-format DataFrame matching  schema.
    Output: a long-format DataFrame with ``value_mean`` (mean across
    seeds), one row per ``(dataset, base_model, method, lambda_, metric)``.

    Cells flagged as failed via ``notes`` are dropped BEFORE aggregation
    (a reweighing-failed cell at seed=3 is really a base-estimator
    point per  / mitigation-fidelity contract).

    When ``apply_adr033_filter`` is True (default), the
    augmented structural-invariance filter is applied as a second pass:
    any ``(dataset, method, lambda_, seed)`` group whose
    ``(accuracy, balanced_accuracy)`` pair is invariant across at least
    two base classifiers is dropped as a silent backend fallback.
    """
    if df.empty:
        return df.assign(value_mean=pd.Series(dtype=float)).iloc[0:0]

    df = df.copy()
    df["_failed"] = df["notes"].apply(is_failed_cell)
    clean = df[~df["_failed"]].drop(columns="_failed")
    if apply_adr033_filter and not clean.empty:
        flagged = detect_silent_backend_fallback(clean)
        clean = clean.loc[~flagged]
    if clean.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "base_model",
                "method",
                "lambda_",
                "metric",
                "value_mean",
                "n_seeds",
            ]
        )

    agg = (
        clean.groupby(
            ["dataset", "base_model", "method", "lambda_", "metric"],
            as_index=False,
        )["value"]
        .agg(value_mean="mean", n_seeds="count")
    )
    return agg

def pivot_to_panel(agg: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long-format seed-mean DataFrame to wide form.

    One row per ``(dataset, base_model, method, lambda_)``; one column
    per metric.
    """
    if agg.empty:
        return agg.assign().iloc[0:0]
    panel = agg.pivot_table(
        index=["dataset", "base_model", "method", "lambda_"],
        columns="metric",
        values="value_mean",
        aggfunc="mean",
    ).reset_index()
    panel.columns.name = None
    return panel

# ---------------------------------------------------------------------
# Top-level frontier computation.
# ---------------------------------------------------------------------

def compute_frontier_table(
    df: pd.DataFrame,
    accuracy_metric: str,
    fairness_metrics: Iterable[str],
    *,
    apply_adr033_filter: bool = True,
) -> pd.DataFrame:
    """Compute the Pareto-frontier table per (dataset, fairness_metric).

    Returns a DataFrame with columns
    ``[dataset, accuracy_metric, fairness_metric, base_model, method,
       lambda_, seed_mean_accuracy, seed_mean_fairness,
       on_pareto_frontier]``,
    sorted lexicographically for deterministic output.

    The ``apply_adr033_filter`` flag is forwarded to
    ``aggregate_seed_means``; see  in
    the project documentation.
    """
    agg = aggregate_seed_means(df, apply_adr033_filter=apply_adr033_filter)
    panel = pivot_to_panel(agg)
    if panel.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "accuracy_metric",
                "fairness_metric",
                "base_model",
                "method",
                "lambda_",
                "seed_mean_accuracy",
                "seed_mean_fairness",
                "on_pareto_frontier",
            ]
        )

    rows: list[dict] = []
    for fmetric in fairness_metrics:
        if fmetric not in FAIRNESS_HIGHER_IS_BETTER:
            raise ValueError(
                f"Unknown fairness metric {fmetric!r}; expected one of "
                f"{sorted(FAIRNESS_HIGHER_IS_BETTER)}."
            )
        if fmetric not in panel.columns:
            # Audit CSV does not contain this fairness metric (e.g.,
            # smoke run with a reduced panel). Skip silently — but only
            # for metrics that are absent FROM THE INPUT, not for
            # metrics the caller asked for and that should be there.
            continue
        if accuracy_metric not in panel.columns:
            raise ValueError(
                f"accuracy metric {accuracy_metric!r} is not in the audit "
                f"CSV (got columns: {sorted(panel.columns)})."
            )

        higher_better = FAIRNESS_HIGHER_IS_BETTER[fmetric]
        for ds, ds_panel in panel.groupby("dataset"):
            ds_panel = ds_panel.copy().reset_index(drop=True)
            mask = compute_pareto_mask(
                ds_panel[accuracy_metric].to_numpy(),
                ds_panel[fmetric].to_numpy(),
                fairness_higher_is_better=higher_better,
            )
            for i, r in ds_panel.iterrows():
                rows.append(
                    {
                        "dataset": ds,
                        "accuracy_metric": accuracy_metric,
                        "fairness_metric": fmetric,
                        "base_model": r["base_model"],
                        "method": r["method"],
                        "lambda_": float(r["lambda_"]),
                        "seed_mean_accuracy": float(r[accuracy_metric])
                        if pd.notna(r[accuracy_metric]) else float("nan"),
                        "seed_mean_fairness": float(r[fmetric])
                        if pd.notna(r[fmetric]) else float("nan"),
                        "on_pareto_frontier": bool(mask[i]),
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(
        ["dataset", "fairness_metric", "base_model", "method", "lambda_"],
        kind="mergesort",
    ).reset_index(drop=True)
    return out

# ---------------------------------------------------------------------
# Markdown summary.
# ---------------------------------------------------------------------

def render_summary_md(
    frontier_df: pd.DataFrame,
    n_failed_cells_excluded: int,
    *,
    top_k: int = 2,
    n_adr033_dropped: int = 0,
    adr033_summary: pd.DataFrame | None = None,
) -> str:
    """Render a short Markdown report of the top-K frontier-optimal cells
    per ``(dataset, fairness_metric)``.

    When the  augmented filter was applied at run time, the
    ``n_adr033_dropped`` count and the per-method breakdown
    ``adr033_summary`` are reported as a separate section.
    """
    lines: list[str] = []
    lines.append("# Pareto Frontier Summary\n")
    lines.append(
        f"Frontier table size: {len(frontier_df)} rows. "
        f"Cells flagged as graceful-fallback failures (excluded from "
        f"frontier): {n_failed_cells_excluded}.\n"
    )
    if n_adr033_dropped:
        lines.append(
            f"\nAugmented filter (silent-backend-fallback): "
            f"{n_adr033_dropped} additional audit rows dropped before "
            f"aggregation.\n"
        )
        if adr033_summary is not None and not adr033_summary.empty:
            lines.append(
                "\n| dataset | method | rows dropped | groups dropped "
                "| known-blind |"
            )
            lines.append("|---|---|---|---|---|")
            for _, r in adr033_summary.iterrows():
                lines.append(
                    f"| {r['dataset']} | {r['method']} "
                    f"| {int(r['n_rows_dropped'])} "
                    f"| {int(r['n_groups_dropped'])} "
                    f"| {'yes' if r['in_known_base_blind_whitelist'] else 'no (anomaly)'} |"
                )
            lines.append("")
    if frontier_df.empty:
        lines.append("\n_No Pareto-frontier rows; the audit CSV has no usable cells._\n")
        return "\n".join(lines)

    on_front = frontier_df[frontier_df["on_pareto_frontier"]]
    grouped = on_front.groupby(["dataset", "fairness_metric"], sort=True)

    for (ds, fmetric), block in grouped:
        higher = FAIRNESS_HIGHER_IS_BETTER.get(fmetric, False)
        order_dir = "highest" if higher else "lowest"
        lines.append(f"## {ds} × {fmetric}\n")
        # Sort frontier points by accuracy desc; tie-break on fairness
        # in the appropriate direction.
        block_sorted = block.sort_values(
            ["seed_mean_accuracy", "seed_mean_fairness"],
            ascending=[False, not higher],
            kind="mergesort",
        )
        top = block_sorted.head(top_k)
        lines.append(
            f"_Top-{top_k} frontier-optimal cells (accuracy desc; "
            f"{order_dir}-fairness tie-break):_\n"
        )
        lines.append(
            "| base_model | method | lambda | acc | fairness |"
        )
        lines.append("|---|---|---|---|---|")
        for _, r in top.iterrows():
            lines.append(
                f"| {r['base_model']} | {r['method']} | "
                f"{r['lambda_']:.4g} | {r['seed_mean_accuracy']:.4f} | "
                f"{r['seed_mean_fairness']:.4f} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"

# ---------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------

def _csv_strs(s: str) -> list[str]:
    return [t.strip() for t in s.split(",") if t.strip()]

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--in-csv",
        type=pathlib.Path,
        default=PROJECT_ROOT / "results" / "phase5" / "audit.csv",
        help="Path to the consolidated Phase-5 audit CSV.",
    )
    p.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=PROJECT_ROOT / "results" / "phase5",
        help="Output directory for pareto.csv + pareto_summary.md.",
    )
    p.add_argument(
        "--accuracy-metric",
        choices=ACCURACY_METRICS,
        default="accuracy",
    )
    p.add_argument(
        "--fairness-metrics",
        type=_csv_strs,
        default=DEFAULT_FAIRNESS_METRICS,
        help=(
            "Comma-separated list of fairness metric names; defaults to "
            "the full Tier-5 statistical panel."
        ),
    )
    p.add_argument(
        "--apply-adr033-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "drop audit cells whose (accuracy, balanced_accuracy) "
            "is invariant across base classifiers (silent backend fallback). "
            "Default: enabled. Use --no-apply-adr033-filter to reproduce "
            "the original frontier (e.g., for regression auditing)."
        ),
    )
    return p.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.in_csv.exists():
        print(
            f"[pareto] input CSV not found at {args.in_csv}; "
            f"run `make phase5-smoke` or `make phase5` first.",
            file=sys.stderr,
        )
        return 1

    df = pd.read_csv(args.in_csv)
    n_failed = int(df["notes"].apply(is_failed_cell).sum())
    print(
        f"[pareto] loaded {len(df)} rows from {args.in_csv}; "
        f"{n_failed} flagged as failed cells will be excluded."
    )

    n_adr033 = 0
    adr033_summary: pd.DataFrame | None = None
    if args.apply_adr033_filter:
        # Apply the failed-cell filter first (matches aggregate_seed_means'
        # ordering) so the  audit reflects only the augmented drops.
        pre = df[~df["notes"].apply(is_failed_cell)]
        flagged = detect_silent_backend_fallback(pre)
        n_adr033 = int(flagged.sum())
        adr033_summary = summarise_silent_fallback(pre, flagged)
        print(
            f"[pareto] augmented filter: dropping {n_adr033} rows "
            f"across {len(adr033_summary)} (dataset, method) pairs."
        )
        if not adr033_summary.empty:
            for _, r in adr033_summary.iterrows():
                tag = (
                    "known-blind"
                    if r["in_known_base_blind_whitelist"]
                    else "NEW-anomaly"
                )
                print(
                    f"[pareto]   {r['dataset']:>18s} / {r['method']:>20s}: "
                    f"{r['n_rows_dropped']:>5d} rows "
                    f"({r['n_groups_dropped']} groups) [{tag}]"
                )

    frontier = compute_frontier_table(
        df,
        accuracy_metric=args.accuracy_metric,
        fairness_metrics=args.fairness_metrics,
        apply_adr033_filter=args.apply_adr033_filter,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "pareto.csv"
    md_path = args.out_dir / "pareto_summary.md"
    frontier.to_csv(csv_path, index=False)
    md_path.write_text(
        render_summary_md(
            frontier,
            n_failed,
            n_adr033_dropped=n_adr033,
            adr033_summary=adr033_summary,
        )
    )

    on_front = frontier[frontier["on_pareto_frontier"]]
    print(
        f"[pareto] wrote {csv_path} ({len(frontier)} rows; "
        f"{len(on_front)} frontier-optimal); "
        f"summary at {md_path}."
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())

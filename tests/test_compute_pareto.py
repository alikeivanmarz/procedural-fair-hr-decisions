"""Tests for ``scripts/compute_pareto.py``.

Covers:

  * ``test_pareto_smoke`` — synthetic 100-cell DataFrame; verifies
    the ``on_pareto_frontier`` flag is correct under a known-good
    configuration on the (accuracy, macro_dp) axes.
  * ``test_pareto_handles_failed_cells`` — cells whose ``notes`` flag
    a graceful fallback (``*_failed:``) are excluded from the
    frontier, per  / mitigation-fidelity contract.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from compute_pareto import (  # noqa: E402
    ADR033_TOLERANCE,
    KNOWN_BASE_BLIND_METHODS,
    aggregate_seed_means,
    compute_frontier_table,
    compute_pareto_mask,
    detect_silent_backend_fallback,
    is_failed_cell,
    summarise_silent_fallback,
)

# ---------------------------------------------------------------------
# Pareto-mask unit tests.
# ---------------------------------------------------------------------

def test_pareto_mask_smaller_is_better() -> None:
    """Three-point sanity: A=(0.9, 0.1), B=(0.95, 0.05), C=(0.8, 0.2).

    With smaller-is-better fairness, B dominates both A and C (better
    on both axes), so only B is on the frontier.
    """
    acc = np.array([0.9, 0.95, 0.8])
    fair = np.array([0.1, 0.05, 0.2])
    mask = compute_pareto_mask(acc, fair, fairness_higher_is_better=False)
    assert mask.tolist() == [False, True, False]

def test_pareto_mask_higher_is_better() -> None:
    """When fairness HIGHER is better (e.g., counterfactual_fairness),
    the dominance direction flips on that axis.
    """
    # A=(0.9, 0.7), B=(0.95, 0.6), C=(0.8, 0.95)
    # B has better acc but worse fair than A — neither dominates the other.
    # B has better acc but worse fair than C. C has better fair but
    # worse acc than B. → A, B, C all on the frontier.
    acc = np.array([0.9, 0.95, 0.8])
    fair = np.array([0.7, 0.6, 0.95])
    mask = compute_pareto_mask(acc, fair, fairness_higher_is_better=True)
    assert mask.tolist() == [True, True, True]

def test_pareto_mask_nan_excluded() -> None:
    """NaN points are never on the frontier and never dominate."""
    acc = np.array([0.9, np.nan, 0.85])
    fair = np.array([0.1, 0.05, 0.2])
    mask = compute_pareto_mask(acc, fair, fairness_higher_is_better=False)
    # Only (0.9, 0.1) is on the frontier; (0.85, 0.2) is dominated.
    assert mask.tolist() == [True, False, False]

def test_pareto_mask_ties_both_on_frontier() -> None:
    """Two identical points are both on the frontier (neither strictly
    dominates the other).
    """
    acc = np.array([0.9, 0.9, 0.8])
    fair = np.array([0.1, 0.1, 0.2])
    mask = compute_pareto_mask(acc, fair, fairness_higher_is_better=False)
    assert mask.tolist() == [True, True, False]

# ---------------------------------------------------------------------
# is_failed_cell.
# ---------------------------------------------------------------------

def test_is_failed_cell_recognises_failure_tokens() -> None:
    assert is_failed_cell("reweighing_failed: ValueError")
    assert is_failed_cell("reweighing_failed: TypeError")
    assert is_failed_cell("lfr_failed: RuntimeError")
    assert is_failed_cell("_predict_failed: IndexError")
    assert is_failed_cell("N/A — Reweighing requires sample_weight")
    assert not is_failed_cell("")
    assert not is_failed_cell(float("nan"))
    assert not is_failed_cell("eo_modal_class=2")
    # Metric-level failures must NOT exclude the cell from the frontier.
    assert not is_failed_cell("counterfactual_fairness_failed: ValueError")
    assert not is_failed_cell("process_consistency_failed: RuntimeError")

# ---------------------------------------------------------------------
# Synthetic Pareto end-to-end tests.
# ---------------------------------------------------------------------

def _make_synthetic_audit_df(
    *,
    n_seeds: int = 5,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic 100-cell audit CSV (long-format,  schema).

    Constructs a known-good configuration where the (method, lambda)
    point ``("methodA", 0.0)`` strictly dominates every other point on
    both accuracy and macro_dp, so the frontier collapses to that one
    cell per (dataset, base_model).
    """
    rng = np.random.default_rng(rng_seed)
    rows: list[dict] = []
    methods = ["methodA", "methodB", "methodC", "methodD"]
    lambdas = [0.0, 0.5, 1.0, 3.0, 10.0]
    base_models = ["RF"]
    datasets = ["ricci"]
    accuracy_metric = "accuracy"
    fairness_metric = "macro_dp"

    for ds in datasets:
        for bm in base_models:
            for m in methods:
                for lam in lambdas:
                    # methodA / lam=0 is the "winner" — high acc / low DP.
                    if m == "methodA" and lam == 0.0:
                        true_acc, true_dp = 0.95, 0.02
                    elif m == "methodA":
                        true_acc, true_dp = 0.93, 0.05
                    elif m == "methodB":
                        true_acc, true_dp = 0.90 - 0.01 * lam, 0.10
                    elif m == "methodC":
                        true_acc, true_dp = 0.85, 0.20 - 0.005 * lam
                    else:
                        true_acc, true_dp = 0.80, 0.30
                    for seed in range(n_seeds):
                        # Tiny per-seed noise so the seed-mean still
                        # matches the canonical ranking.
                        eps_a = rng.normal(0.0, 0.001)
                        eps_d = rng.normal(0.0, 0.001)
                        rows.append({
                            "dataset": ds,
                            "target": "Class",
                            "base_model": bm,
                            "method": m,
                            "method_kind": "in",
                            "lambda_": float(lam),
                            "seed": int(seed),
                            "metric": accuracy_metric,
                            "value": float(true_acc + eps_a),
                            "metric_kind": "performance",
                            "sample_n": 100,
                            "random_state": int(seed),
                            "notes": "",
                        })
                        rows.append({
                            "dataset": ds,
                            "target": "Class",
                            "base_model": bm,
                            "method": m,
                            "method_kind": "in",
                            "lambda_": float(lam),
                            "seed": int(seed),
                            "metric": fairness_metric,
                            "value": float(true_dp + eps_d),
                            "metric_kind": "statistical",
                            "sample_n": 100,
                            "random_state": int(seed),
                            "notes": "",
                        })
    return pd.DataFrame(rows)

def test_pareto_smoke() -> None:
    """: synthetic audit table; verify the frontier flag is correct.

    By construction the (methodA, lambda=0.0) cell strictly dominates
    everything else on both accuracy and macro_dp; the frontier should
    contain exactly that single cell.
    """
    df = _make_synthetic_audit_df()
    out = compute_frontier_table(
        df,
        accuracy_metric="accuracy",
        fairness_metrics=["macro_dp"],
    )
    assert not out.empty
    on_front = out[out["on_pareto_frontier"]]
    assert len(on_front) == 1, (
        f"expected exactly 1 frontier-optimal cell, got {len(on_front)}:\n"
        f"{on_front}"
    )
    winner = on_front.iloc[0]
    assert winner["method"] == "methodA"
    assert winner["lambda_"] == 0.0
    # Sanity: dataset/fairness_metric columns populated; sort-stable.
    assert (out["dataset"] == "ricci").all()
    assert (out["fairness_metric"] == "macro_dp").all()

def test_pareto_handles_failed_cells() -> None:
    """: cells flagged ``*_failed:`` are excluded BEFORE the frontier
    is computed (they are really base-estimator points masquerading as
    the named method per ).
    """
    # Build a synthetic table where (methodB, lambda=10) would be
    # frontier-optimal, BUT every seed of that cell carries a
    # ``methodB_failed:`` note. Without filtering, methodB/lam=10 would
    # appear on the frontier; with filtering it should be dropped.
    rows: list[dict] = []
    base_meta = {
        "dataset": "ricci",
        "target": "Class",
        "base_model": "RF",
        "method_kind": "in",
        "metric_kind": "performance",
        "sample_n": 100,
        "random_state": 0,
    }

    def _add(method: str, lam: float, acc: float, dp: float, notes: str) -> None:
        for seed in range(3):
            rows.append({
                **base_meta,
                "method": method,
                "lambda_": float(lam),
                "seed": int(seed),
                "metric": "accuracy",
                "value": float(acc),
                "metric_kind": "performance",
                "notes": notes,
            })
            rows.append({
                **base_meta,
                "method": method,
                "lambda_": float(lam),
                "seed": int(seed),
                "metric": "macro_dp",
                "value": float(dp),
                "metric_kind": "statistical",
                "notes": notes,
            })

    # methodA cell at (0.92, 0.05) — clean, real frontier candidate.
    _add("methodA", 0.0, 0.92, 0.05, "")
    # methodB cell at (0.99, 0.0) — would dominate everything if real,
    # BUT every row is flagged failed (uses a real method-level token).
    # Should be excluded from the frontier.
    _add("methodB", 10.0, 0.99, 0.0, "lfr_failed: TypeError")
    # methodC cell at (0.85, 0.10) — dominated; clean.
    _add("methodC", 0.5, 0.85, 0.10, "")

    df = pd.DataFrame(rows)
    # Pre-flight: aggregate_seed_means must drop the failed cell.
    agg = aggregate_seed_means(df)
    assert "methodB" not in set(agg["method"]), (
        "methodB cell should have been dropped by aggregate_seed_means; "
        f"got methods: {sorted(agg['method'].unique())}"
    )

    out = compute_frontier_table(
        df,
        accuracy_metric="accuracy",
        fairness_metrics=["macro_dp"],
    )
    on_front = out[out["on_pareto_frontier"]]
    # methodB excluded → methodA's (0.92, 0.05) is the sole frontier
    # point (methodC at (0.85, 0.10) is dominated).
    assert "methodB" not in set(out["method"]), (
        f"methodB leaked into pareto.csv: \n{out}"
    )
    assert len(on_front) == 1
    assert on_front.iloc[0]["method"] == "methodA"

# ---------------------------------------------------------------------
#  augmented filter — structural-invariance detection.
# ---------------------------------------------------------------------

def _adr033_row(
    *,
    dataset: str,
    method: str,
    lam: float,
    seed: int,
    base_model: str,
    metric: str,
    value: float,
    notes: str = "",
) -> dict:
    return {
        "dataset": dataset,
        "target": "Class",
        "base_model": base_model,
        "method": method,
        "method_kind": "in",
        "lambda_": float(lam),
        "seed": int(seed),
        "metric": metric,
        "value": float(value),
        "metric_kind": "performance" if metric in ("accuracy", "balanced_accuracy") else "statistical",
        "sample_n": 100,
        "random_state": int(seed),
        "notes": notes,
    }

def _adr033_emit(rows: list[dict], *, dataset, method, lam, seed, base_model,
                 accuracy, balanced_accuracy, notes="") -> None:
    rows.append(_adr033_row(
        dataset=dataset, method=method, lam=lam, seed=seed,
        base_model=base_model, metric="accuracy", value=accuracy, notes=notes,
    ))
    rows.append(_adr033_row(
        dataset=dataset, method=method, lam=lam, seed=seed,
        base_model=base_model, metric="balanced_accuracy",
        value=balanced_accuracy, notes=notes,
    ))

def test_detect_silent_backend_fallback_flags_invariant_group() -> None:
    """A (dataset, method, lambda_, seed) group whose accuracy AND
    balanced_accuracy are byte-identical across two or more base_models
    is flagged.
    """
    rows: list[dict] = []
    # PR-style: identical accuracy across all five base_models.
    for bm in ("GB", "LR", "MLP", "RF", "XGB"):
        _adr033_emit(rows, dataset="ricci", method="prejudice_remover",
                     lam=0.1, seed=0, base_model=bm,
                     accuracy=0.85, balanced_accuracy=0.80)
    # Non-PR: distinct accuracy per base_model.
    for bm, acc in (("GB", 0.86), ("LR", 0.84), ("RF", 0.87)):
        _adr033_emit(rows, dataset="ricci", method="reweighing",
                     lam=0.1, seed=0, base_model=bm,
                     accuracy=acc, balanced_accuracy=acc - 0.02)
    df = pd.DataFrame(rows)
    flagged = detect_silent_backend_fallback(df)
    assert flagged.dtype == bool
    # All PR rows flagged, all reweighing rows clean.
    assert flagged[df["method"] == "prejudice_remover"].all()
    assert (~flagged[df["method"] == "reweighing"]).all()

def test_detect_silent_backend_fallback_skips_single_base_model() -> None:
    """A group with only one base_model present cannot be assessed
    structurally and must never be flagged.
    """
    rows: list[dict] = []
    _adr033_emit(rows, dataset="ricci", method="some_method",
                 lam=0.1, seed=0, base_model="LR",
                 accuracy=0.85, balanced_accuracy=0.80)
    df = pd.DataFrame(rows)
    flagged = detect_silent_backend_fallback(df)
    assert not flagged.any()

def test_detect_silent_backend_fallback_tolerance_strict() -> None:
    """Accuracy differing by more than the tolerance must NOT be flagged."""
    rows: list[dict] = []
    # Tiny but supra-tolerance variation (~1e-3).
    for bm, acc in (("GB", 0.850), ("LR", 0.851), ("RF", 0.852)):
        _adr033_emit(rows, dataset="ricci", method="reweighing",
                     lam=0.1, seed=0, base_model=bm,
                     accuracy=acc, balanced_accuracy=acc)
    df = pd.DataFrame(rows)
    flagged = detect_silent_backend_fallback(df, tol=ADR033_TOLERANCE)
    assert not flagged.any()

def test_detect_silent_backend_fallback_requires_both_metrics_invariant() -> None:
    """If accuracy is invariant but balanced_accuracy varies, the group
    is NOT flagged — both metrics must be invariant.
    """
    rows: list[dict] = []
    for bm, bacc in (("GB", 0.80), ("LR", 0.81), ("RF", 0.82)):
        _adr033_emit(rows, dataset="ricci", method="some_method",
                     lam=0.1, seed=0, base_model=bm,
                     accuracy=0.85, balanced_accuracy=bacc)
    df = pd.DataFrame(rows)
    flagged = detect_silent_backend_fallback(df)
    assert not flagged.any()

def test_aggregate_seed_means_applies_adr033_filter_by_default() -> None:
    """``aggregate_seed_means`` drops invariant groups by default; the
    opt-out flag restores pre behaviour.
    """
    rows: list[dict] = []
    # PR-style invariant group at lam=0.1 across 3 seeds × 3 base_models.
    for seed in range(3):
        for bm in ("GB", "LR", "RF"):
            _adr033_emit(rows, dataset="ricci", method="prejudice_remover",
                         lam=0.1, seed=seed, base_model=bm,
                         accuracy=0.85, balanced_accuracy=0.80)
    # Clean reweighing group at lam=0.1 across 3 seeds × 3 base_models.
    for seed in range(3):
        for bm, acc in (("GB", 0.86), ("LR", 0.84), ("RF", 0.87)):
            _adr033_emit(rows, dataset="ricci", method="reweighing",
                         lam=0.1, seed=seed, base_model=bm,
                         accuracy=acc, balanced_accuracy=acc - 0.01)
    df = pd.DataFrame(rows)

    # Default: PR rows dropped.
    agg_filtered = aggregate_seed_means(df)
    assert "prejudice_remover" not in set(agg_filtered["method"])
    assert "reweighing" in set(agg_filtered["method"])

    # Opt-out: PR rows survive.
    agg_unfiltered = aggregate_seed_means(df, apply_adr033_filter=False)
    assert "prejudice_remover" in set(agg_unfiltered["method"])
    assert "reweighing" in set(agg_unfiltered["method"])

def test_summarise_silent_fallback_marks_whitelist_correctly() -> None:
    """Methods in ``KNOWN_BASE_BLIND_METHODS`` are tagged as such; any
    other method that triggers the filter is flagged as an anomaly.
    """
    rows: list[dict] = []
    # PR (whitelisted) invariant group.
    for bm in ("GB", "LR", "RF"):
        _adr033_emit(rows, dataset="ricci", method="prejudice_remover",
                     lam=0.1, seed=0, base_model=bm,
                     accuracy=0.85, balanced_accuracy=0.80)
    # A different method (NOT whitelisted) also triggers.
    for bm in ("GB", "LR", "RF"):
        _adr033_emit(rows, dataset="ricci", method="adv_debias",
                     lam=0.1, seed=0, base_model=bm,
                     accuracy=0.90, balanced_accuracy=0.85)
    df = pd.DataFrame(rows)
    flagged = detect_silent_backend_fallback(df)
    summary = summarise_silent_fallback(df, flagged)
    assert set(summary["method"]) == {"prejudice_remover", "adv_debias"}
    pr_row = summary[summary["method"] == "prejudice_remover"].iloc[0]
    ad_row = summary[summary["method"] == "adv_debias"].iloc[0]
    assert bool(pr_row["in_known_base_blind_whitelist"]) is True
    assert bool(ad_row["in_known_base_blind_whitelist"]) is False
    # Sanity on whitelist content.
    assert "prejudice_remover" in KNOWN_BASE_BLIND_METHODS

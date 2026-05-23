"""Smoke tests for the Phase-4 procedural-fairness audit runner.

Two tests:

1. ``test_run_phase4_smoke`` — runs ``scripts/run_phase4_procedural.py``
   on the IBM HR Attrition dataset only with a tiny ``--sample-n`` so
   the test completes in under a minute. Checks the schema and row count.

2. ``test_run_phase4_byte_identical`` — runs the smoke once, copies the
   CSV, runs again with the cache cleared, and asserts byte-identical
   output.

The full audit (all three datasets × three models) is invoked via
``make phase4`` and is too slow for a unit test; the smoke ensures the
runner itself works end-to-end.

References
----------
* `` — Gaussian-noise perturbation contract.
* `` — modifiable / immutable feature partition.
* the project documentation — determinism (seed=0).
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

def _run_phase4(out_dir: pathlib.Path) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_phase4_procedural.py"),
        "--datasets",
        "ibm_hr_attrition",
        "--out-dir",
        str(out_dir),
        "--sample-n",
        "10",
        "--sample-n-transparency",
        "10",
    ]
    return subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT)
    )

def test_run_phase4_smoke(tmp_path: pathlib.Path) -> None:
    """End-to-end Phase-4 audit on IBM HR Attrition with a tiny sample.

    Validates the post schema, row count, and that the
    backward-compatible metric names + the new split metric names
    are all populated.
    """
    out_dir = tmp_path / "phase4"
    proc = _run_phase4(out_dir)
    assert proc.returncode == 0, (
        f"audit script failed (rc={proc.returncode}):\n"
        f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
    )

    csv_path = out_dir / "procedural.csv"
    assert csv_path.exists(), f"procedural.csv not created at {csv_path}"
    df = pd.read_csv(csv_path)

    # Post schema.
    expected_cols = {
        "dataset",
        "target",
        "model",
        "metric",
        "value",
        "seed",
        "mean",
        "std",
        "ci_lo",
        "ci_hi",
        "noise_std",
        "target_form",
        "n_classes",
        "sample_n",
        "random_state",
        "notes",
    }
    assert set(df.columns) == expected_cols, (
        f"procedural.csv columns mismatch: got {set(df.columns)}, "
        f"expected {expected_cols}"
    )

    # Backward-compat: original four metric names from  must still
    # appear (transparency_metrics alias preserved per ).
    metrics_in_df = set(df["metric"].unique())
    legacy_metrics = {
        "process_consistency",
        "voice_representation",
        "transparency_sparsity",
        "transparency_validity",
    }
    assert legacy_metrics.issubset(metrics_in_df), (
        f"Legacy metric names missing from CSV: "
        f"{legacy_metrics - metrics_in_df}"
    )

    # New split metrics from .
    new_metrics = {
        "voice_enrichment",
        "model_flippability_sparsity",
        "model_flippability_validity",
        "actionable_validity",
        "actionable_sparsity",
    }
    assert new_metrics.issubset(metrics_in_df), (
        f"Required metric names missing from CSV: "
        f"{new_metrics - metrics_in_df}"
    )

    # All three default models present (smoke does not exercise XGB/GB/KNN).
    assert set(df["model"].unique()) == {
        "RandomForestClassifier",
        "LogisticRegression",
        "MLPClassifier",
    }

    # Per-seed rows have seed >= 0; aggregated rows have seed == -1.
    assert set(df["seed"].unique()) == {0, -1}, (
        f"Expected seed in {{0, -1}}; got {set(df['seed'].unique())}"
    )

    # Values bounded in [0, 1] for non-NaN entries.
    finite_mask = df["value"].notna()
    assert (df.loc[finite_mask, "value"] >= 0.0).all()
    assert (df.loc[finite_mask, "value"] <= 1.0).all() | (
        # voice_enrichment can exceed 1 by construction (it's voice / share)
        df.loc[finite_mask, "metric"] == "voice_enrichment"
    ).all() or True  # always-true relaxation; explicit check below
    over_one = df.loc[finite_mask & (df["value"] > 1.0), "metric"].unique()
    assert set(over_one).issubset({"voice_enrichment"}), (
        f"Values > 1 found in unexpected metrics: {set(over_one)}"
    )

    # Non-leaky honest target.
    assert (df["target"] == "Attrition").all()
    assert (df["notes"] == "").all() or df["notes"].isna().all()

def test_run_phase4_seed_aggregate_matches_mean(tmp_path: pathlib.Path) -> None:
    """: with 2 seeds, the seed=-1 aggregate `mean` matches np.mean of
    the per-seed `value`s exactly for every (model, metric, noise_std).
    """
    import sys
    out_dir = tmp_path / "phase4_seedagg"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_phase4_procedural.py"),
        "--datasets",
        "ibm_hr_attrition",
        "--models",
        "RF",
        "--seeds",
        "0,1",
        "--noise-stds",
        "0.1",
        "--out-dir",
        str(out_dir),
        "--sample-n",
        "10",
        "--sample-n-transparency",
        "10",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT)
    )
    assert proc.returncode == 0, (
        f"seed-agg run failed:\n{proc.stdout}\n{proc.stderr}"
    )
    df = pd.read_csv(out_dir / "procedural.csv")
    per_seed = df[df["seed"] >= 0]
    aggregated = df[df["seed"] == -1]
    grouped = per_seed.groupby(["model", "metric", "noise_std"])["value"].mean()
    for _, row in aggregated.iterrows():
        key = (row["model"], row["metric"], row["noise_std"])
        expected = grouped.get(key)
        assert expected is not None, f"missing per-seed group for {key}"
        # mean column on aggregated row must match np.mean of per-seed values
        assert abs(row["mean"] - expected) < 1e-12, (
            f"aggregate mean mismatch for {key}: "
            f"agg.mean={row['mean']} vs np.mean={expected}"
        )

def test_run_phase4_byte_identical(tmp_path: pathlib.Path) -> None:
    """Re-running the audit produces a byte-identical CSV.

    Clears the per-cell cache between runs so the second run actually
    re-computes (rather than reading the cached parquet which is itself
    byte-stable).
    """
    out_dir = tmp_path / "phase4"
    proc1 = _run_phase4(out_dir)
    assert proc1.returncode == 0, (
        f"first audit run failed:\n{proc1.stdout}\n{proc1.stderr}"
    )
    csv1 = (out_dir / "procedural.csv").read_bytes()

    # Wipe the cache to force a fresh re-compute.
    cache_dir = out_dir / "cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    # Also remove the consolidated CSV/parquet so the script re-writes them.
    (out_dir / "procedural.csv").unlink()
    (out_dir / "procedural.parquet").unlink(missing_ok=True)

    proc2 = _run_phase4(out_dir)
    assert proc2.returncode == 0, (
        f"second audit run failed:\n{proc2.stdout}\n{proc2.stderr}"
    )
    csv2 = (out_dir / "procedural.csv").read_bytes()

    assert csv1 == csv2, (
        "CSV bytes differ between two runs (determinism violation). "
        f"First run had {len(csv1)} bytes; second had {len(csv2)}."
    )

# ---------------------------------------------------------------------------
#  — XGBoost SHAP smoke test ( followup²).
# ---------------------------------------------------------------------------

def test_xgb_label_encoder_subclass_shap_finite() -> None:
    """: ``_XGBWithLabelEncoder`` subclass + ``_xgboost_shap_compat_patch``
    produce finite SHAP values on labels {3, 4} (the D1-PerformanceRating
    case that previously emitted NaN voice).

    Two assertions:
      * ``isinstance(model, XGBClassifier)`` is True after the subclass
        change (was False under the old composition wrapper, which made
        ``shap.TreeExplainer`` fall through to a NaN-emitting branch).
      * Inside ``_xgboost_shap_compat_patch``, ``shap.TreeExplainer``
        returns finite values for an xgboost-3.x booster.
    """
    pytest.importorskip("xgboost")
    pytest.importorskip("shap")

    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from xgboost import XGBClassifier  # noqa: E402

    import shap  # noqa: E402

    from scripts.run_procedural import (  # noqa: E402
        _build_model,
        _xgboost_shap_compat_patch,
    )

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    # Labels {3, 4} — this is the case that crashed plain XGBClassifier
    # (XGBoost requires labels in [0, n_classes-1]).
    y = rng.choice([3, 4], size=100)

    estimator, name, kind = _build_model("XGB", seed=0)
    assert name == "XGBClassifier"
    assert kind == "tree"
    #  Path 1 (preferred): subclass-based wrapper.
    assert isinstance(estimator, XGBClassifier), (
        "_XGBWithLabelEncoder must subclass XGBClassifier so "
        "shap.TreeExplainer's isinstance dispatch picks the right path."
    )
    estimator.fit(X, y)
    # Original labels are restored on classes_.
    assert set(estimator.classes_.tolist()) == {3, 4}
    # Predictions are in the original label space.
    preds = estimator.predict(X[:10])
    assert set(np.unique(preds).tolist()).issubset({3, 4})

    # SHAP under the compat patch returns finite values.
    with _xgboost_shap_compat_patch():
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X[:5], check_additivity=False)
    arr = np.asarray(shap_values)
    assert arr.size > 0, "SHAP returned an empty array"
    assert np.all(np.isfinite(arr)), (
        f"SHAP values are not all finite (got {np.isnan(arr).sum()} NaNs / "
        f"{(~np.isfinite(arr)).sum()} non-finite). The XGBoost-SHAP compat "
        "patch should have prevented this."
    )

# ---------------------------------------------------------------------------
#  — Multi-class XGBoost voice_representation finite-output regression.
# ---------------------------------------------------------------------------

def test_voice_representation_multiclass_xgb_finite() -> None:
    """: ``voice_representation`` on a 3-class XGBClassifier returns
    finite voice + voice_enrichment values.

    Pre the OULAD-XGB cell of ``procedural.csv`` had
    ``mean = NaN, ci_lo = NaN, ci_hi = NaN`` because xgboost-3.x
    multi-class boosters serialise ``base_score`` as the comma-separated
    bracketed vector ``"[v0,v1,...,vK]"`` (one entry per class), which
    the  compat patch (which stripped only the brackets) could not
    parse. The  patch reduces the per-class vector to its mean
    before SHAP's ``float()`` parses it.
    """
    pytest.importorskip("xgboost")
    pytest.importorskip("shap")

    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from procedural_fair_hr.procedural_fairness import voice_representation  # noqa: E402

    from scripts.run_procedural import _build_model  # noqa: E402

    rng = np.random.default_rng(0)
    n, n_feat = 200, 6
    X_arr = rng.normal(size=(n, n_feat))
    y = rng.choice([0, 1, 2], size=n)  # 3-class — multi-class branch.

    feature_names = [f"feat_{i}" for i in range(n_feat)]
    X = pd.DataFrame(X_arr, columns=feature_names)

    estimator, name, kind = _build_model("XGB", seed=0)
    estimator.fit(X.values, y)
    assert len(estimator.classes_) == 3

    partition = {
        "modifiable": feature_names[:3],
        "immutable": feature_names[3:],
    }

    voice, enrichment, per_feature = voice_representation(
        estimator,
        X.iloc[:50],
        feature_partition=partition,
        shap_explainer="tree",
        random_state=0,
    )

    assert np.isfinite(voice), (
        f"voice_representation on 3-class XGBClassifier returned NaN "
        f"voice (={voice!r}). The multi-class base_score patch "
        "should have prevented this."
    )
    assert np.isfinite(enrichment), (
        f"voice_representation on 3-class XGBClassifier returned NaN "
        f"voice_enrichment (={enrichment!r})."
    )
    assert 0.0 <= voice <= 1.0
    assert enrichment >= 0.0

def test_voice_representation_multiclass_gb_finite() -> None:
    """: ``voice_representation`` on a 3-class
    ``sklearn.GradientBoostingClassifier`` returns finite values.

    Pre (the same audit pass as the XGBoost issue) the OULAD-GB
    cell of ``procedural.csv`` had ``voice = NaN`` because SHAP's
    ``TreeExplainer`` raises ``InvalidModelError("GradientBoostingClassifier
    is only supported for binary classification right now")`` on multi-
    class sklearn boosting.  says fix, don't punt: the regression
    patch in ``voice_representation`` falls back to ``KernelExplainer``
    for this specific failure mode (model-agnostic; slower but works).
    The fall-through is engaged only when SHAP raises a known-unsupported
    error, leaving the fast tree path intact for binary GB and for
    XGB / RF / etc. on any class count.
    """
    pytest.importorskip("shap")
    from sklearn.ensemble import GradientBoostingClassifier

    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from procedural_fair_hr.procedural_fairness import voice_representation  # noqa: E402

    rng = np.random.default_rng(0)
    n, n_feat = 200, 6
    X_arr = rng.normal(size=(n, n_feat))
    y = rng.choice([0, 1, 2], size=n)  # 3-class — multi-class GB branch.

    feature_names = [f"feat_{i}" for i in range(n_feat)]
    X = pd.DataFrame(X_arr, columns=feature_names)

    estimator = GradientBoostingClassifier(n_estimators=20, random_state=0)
    estimator.fit(X.values, y)
    assert len(estimator.classes_) == 3

    partition = {
        "modifiable": feature_names[:3],
        "immutable": feature_names[3:],
    }

    voice, enrichment, per_feature = voice_representation(
        estimator,
        X.iloc[:30],
        feature_partition=partition,
        shap_explainer="tree",
        random_state=0,
    )
    assert np.isfinite(voice), (
        f"voice_representation on 3-class GradientBoostingClassifier "
        f"returned NaN voice (={voice!r}). The KernelExplainer "
        "fall-back should have prevented this."
    )
    assert np.isfinite(enrichment)
    assert 0.0 <= voice <= 1.0
    assert enrichment >= 0.0


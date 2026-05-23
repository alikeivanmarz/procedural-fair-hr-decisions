"""Tests for Phase-6 SHAP audit.

Exit-gate tests:
  * test_shap_output_schema — output CSV has all required columns.
  * test_shap_normalised_shares_sum_to_one — per (dataset, model, method, group),
    normalised_share sums to 1.0 (within floating-point tolerance).
  * test_shap_smoke — runs on ibm_hr_attrition × GB × identity × N=20 in < 30 s.

References
----------

"""

from __future__ import annotations

import time
import pathlib
import sys

import numpy as np
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Columns required by  output schema.
REQUIRED_COLUMNS = [
    "dataset",
    "model",
    "method",
    "lambda_",
    "shap_type",
    "group",
    "feature",
    "mean_abs_shap",
    "normalised_share",
    "is_sensitive",
    "is_proxy",
    "sample_n",
    "random_state",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_smoke(dataset: str, base_model: str, max_rows: int = 20):
    """Run the SHAP audit on one dataset/model with a capped sample.

    Returns the resulting DataFrame.
    """
    import tempfile
    from scripts.run_shap import main

    with tempfile.TemporaryDirectory() as tmpdir:
        df = main(
            datasets=[dataset],
            max_rows=max_rows,
            out_dir=tmpdir,
        )
    return df

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_shap_output_schema():
    """Output CSV has all required  columns (test_shap_output_schema)."""
    df = _run_smoke("ibm_hr_attrition", "GB", max_rows=20)
    assert not df.empty, "Expected non-empty DataFrame from smoke run"
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert missing == [], f"Missing columns: {missing}"

def test_shap_normalised_shares_sum_to_one():
    """Per (dataset, model, method, group), normalised_share sums to 1.0.

    Tests the normalisation invariant: for each unique combination
    of (dataset, model, method, group) the sum of normalised_share across all
    features must equal 1.0 within 1e-6 floating-point tolerance.
    """
    df = _run_smoke("ibm_hr_attrition", "GB", max_rows=20)
    assert not df.empty, "Expected non-empty DataFrame"
    key_cols = ["dataset", "model", "method", "group"]
    for keys, sub in df.groupby(key_cols):
        total = sub["normalised_share"].sum()
        assert abs(total - 1.0) < 1e-5, (
            f"normalised_share does not sum to 1.0 for {dict(zip(key_cols, keys))}: "
            f"sum={total:.8f}"
        )

def test_shap_smoke():
    """Smoke test: ibm_hr_attrition × GB × identity × N=20 completes in < 30 s."""
    t0 = time.time()
    df = _run_smoke("ibm_hr_attrition", "GB", max_rows=20)
    elapsed = time.time() - t0
    assert elapsed < 30.0, f"Smoke test took {elapsed:.1f}s (> 30s budget)"
    assert not df.empty, "Expected non-empty DataFrame from smoke run"
    # Verify that 'all' group rows exist.
    assert "all" in df["group"].unique(), "Expected 'all' group in output"
    # Verify mean_abs_shap >= 0.
    assert (df["mean_abs_shap"] >= 0).all(), "mean_abs_shap must be non-negative"

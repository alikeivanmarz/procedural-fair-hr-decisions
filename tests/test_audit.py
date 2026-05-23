"""Smoke test for the Phase-2 audit runner.

Runs ``scripts/run_phase2_audit.py`` on Ricci only (118 rows total; 24-row
test split) and checks that the no-shortcut deliverables are produced and
well-formed:

1. ``results/phase2/audit.csv`` and ``audit.parquet`` exist; the schema has
   the ten columns documented in the runner's module docstring; at least
   eight metric rows for Ricci are present.
2. ``results/phase2/figs/abroca_ricci_Race.png`` exists and weighs more than
   1 KB (sanity check that matplotlib rendered the figure).
3. ``results/phase2/proxy_edges.csv`` exists.
4. ``results/phase2/audit_ricci.parquet`` (per-dataset checkpoint) exists.
5. ``results/phase2/progress.log`` contains the dataset's start + done lines.
6. The ``counterfactual_fairness`` row's ``cf_n_samples`` equals 24 — the
   FULL Ricci test split — proving no subsampling was applied (Ricci is
   below the 5,000-row CF_FULL_THRESHOLD in the runner).
7. Every ``metric_value`` is a real float (no NaN);
   ``disparate_impact_ratio`` may be infinite if a group has zero positives.
8. ``class_idx == 1`` for every Ricci row (binary dataset → canonical
   positive class).
9. ``target_form == "binary"`` for every Ricci row.

References
----------
* `` — ``make audit`` reproducibility.
* `` — determinism (seed=0 throughout).
* `` — Absolute Between-ROC Area``.
"""

from __future__ import annotations

import math
import pathlib
import subprocess
import sys

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

def test_run_phase2_audit_ricci_smoke(tmp_path: pathlib.Path) -> None:
    """End-to-end no-shortcut audit on Ricci."""
    out_dir = tmp_path / "phase2"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_phase2_audit.py"),
        "--datasets",
        "ricci",
        "--out-dir",
        str(out_dir),
        "--no-commit",
        "--no-push",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT)
    )
    assert proc.returncode == 0, (
        f"audit script failed (rc={proc.returncode}):\n"
        f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
    )

    audit_csv = out_dir / "audit.csv"
    audit_parquet = out_dir / "audit.parquet"
    proxy_csv = out_dir / "proxy_edges.csv"
    fig_path = out_dir / "figs" / "abroca_ricci_Race.png"
    partial_parquet = out_dir / "audit_ricci.parquet"
    progress_log = out_dir / "progress.log"

    # 1. audit.csv + audit.parquet exist with the new 11-column schema.
    assert audit_csv.exists(), f"audit.csv not created: {audit_csv}"
    assert audit_parquet.exists(), f"audit.parquet not created: {audit_parquet}"
    df = pd.read_csv(audit_csv)
    expected_cols = {
        "dataset",
        "protected_attribute",
        "class_idx",
        "metric_name",
        "metric_value",
        "model",
        "target_form",
        "n_train",
        "n_test",
        "cf_n_samples",
        "non_vacuous",  # Pattern A guard (constant predictions)
        "non_vacuous_tpr",  # Pattern B guard (uncorrelated predictions; IBM HR case)
        "pc_n_rows",  #  disclosure column
    }
    assert set(df.columns) == expected_cols, (
        f"audit.csv columns mismatch: got {set(df.columns)}, "
        f"expected {expected_cols}"
    )
    ricci_rows = df[df["dataset"] == "ricci"]
    assert len(ricci_rows) >= 8, (
        f"Expected >= 8 metric rows for ricci, got {len(ricci_rows)}"
    )

    # Ricci is balanced (~50 % positive class); the LR baseline is NOT
    # degenerate (Pattern A or B), so every Ricci row must have BOTH
    # non_vacuous == True AND non_vacuous_tpr == True.
    assert ricci_rows["non_vacuous"].all(), (
        f"All Ricci rows should have non_vacuous=True (baseline is not "
        f"Pattern-A degenerate on this balanced 118-row dataset); got "
        f"{ricci_rows['non_vacuous'].value_counts().to_dict()}"
    )
    assert ricci_rows["non_vacuous_tpr"].all(), (
        f"All Ricci rows should have non_vacuous_tpr=True (baseline is not "
        f"Pattern-B degenerate on this balanced 118-row dataset); got "
        f"{ricci_rows['non_vacuous_tpr'].value_counts().to_dict()}"
    )

    # 7. Every metric_value is a float; allow inf for disparate_impact_ratio.
    for _, row in ricci_rows.iterrows():
        val = row["metric_value"]
        assert isinstance(val, float), (
            f"metric_value not float for {row['metric_name']}: type={type(val)}"
        )
        if row["metric_name"] == "disparate_impact_ratio":
            assert math.isfinite(val) or math.isinf(val)
        else:
            assert math.isfinite(val), (
                f"non-finite metric_value for {row['metric_name']}: {val}"
            )

    # 8 + 9. class_idx==1 and target_form=="binary" for every Ricci row.
    assert (ricci_rows["class_idx"] == 1).all(), (
        f"Ricci rows must have class_idx==1; got {ricci_rows['class_idx'].unique().tolist()}"
    )
    assert (ricci_rows["target_form"] == "binary").all(), (
        f"Ricci rows must have target_form=='binary'; "
        f"got {ricci_rows['target_form'].unique().tolist()}"
    )

    # 6. CF was run on the FULL Ricci test split (24 rows, ≤ CF_FULL_THRESHOLD).
    cf_rows = ricci_rows[ricci_rows["metric_name"] == "counterfactual_fairness"]
    assert len(cf_rows) == 1, (
        f"Expected exactly one CF row for binary Ricci, got {len(cf_rows)}"
    )
    cf_n = int(cf_rows["cf_n_samples"].iloc[0])
    assert cf_n == 24, (
        f"Ricci test split is 24 rows; CF must use ALL of them (no shortcut). "
        f"Got cf_n_samples={cf_n}"
    )
    n_test = int(cf_rows["n_test"].iloc[0])
    assert n_test == 24, f"Ricci n_test should be 24; got {n_test}"

    # 2. ABROCA figure exists and is > 1 KB.
    assert fig_path.exists(), f"ABROCA figure not created: {fig_path}"
    assert fig_path.stat().st_size > 1000, (
        f"ABROCA figure too small ({fig_path.stat().st_size} bytes)"
    )

    # 3. proxy_edges.csv exists with the documented schema.
    assert proxy_csv.exists(), f"proxy_edges.csv not created: {proxy_csv}"
    proxy_df = pd.read_csv(proxy_csv)
    assert set(proxy_df.columns) == {
        "dataset",
        "protected_attribute",
        "target",
        "edge_type",
        "source",
        "sink",
        "pc_n_rows",  #  disclosure column
    }, f"proxy_edges.csv columns mismatch: got {set(proxy_df.columns)}"

    # 4. Per-dataset checkpoint parquet exists.
    assert partial_parquet.exists(), (
        f"per-dataset partial parquet not created: {partial_parquet}"
    )
    partial = pd.read_parquet(partial_parquet)
    assert set(partial.columns) == expected_cols, (
        f"audit_ricci.parquet schema drift: {set(partial.columns)}"
    )

    # 5. progress.log contains the dataset's milestones.
    assert progress_log.exists(), f"progress.log not created: {progress_log}"
    log_text = progress_log.read_text()
    assert "ricci  LOAD" in log_text, "progress.log missing 'ricci LOAD' milestone"
    assert "ricci  DONE" in log_text, "progress.log missing 'ricci DONE' milestone"
    assert "PC START" in log_text, "progress.log missing 'PC START' milestone"
    assert "PC DONE" in log_text or "PC FAILED" in log_text, (
        "progress.log missing PC outcome milestone"
    )
    assert "CF START" in log_text, "progress.log missing 'CF START' milestone"
    assert "CF DONE" in log_text, "progress.log missing 'CF DONE' milestone"

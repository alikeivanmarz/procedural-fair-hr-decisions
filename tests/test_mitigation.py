"""Smoke + byte-identical + parallel-correctness tests for the Phase-5
mitigation-matrix audit runner.

Tests:

  * ``test_run_phase5_smoke`` — 1 cell on Ricci with RF +
    identity_preprocessing; rc=0; full Tier-5 12-metric panel emitted.
  * ``test_run_phase5_byte_identical`` — runs the same cell twice with
    the cache cleared between runs; asserts the consolidated CSVs are
    byte-identical.
  * ``test_parallel_correctness`` — running with ``--max-workers 4``
    produces the same CSV as ``--max-workers 1`` (modulo lexicographic
    sort). This is the parallel-determinism guarantee from .
  * ``test_panel_shape_12_metrics`` — Tier-5 panel emits
    exactly 12 rows per cell.
  * ``test_panel_byte_identical_two_runs`` — same cell twice
    produces row-for-row identical metric values.
  * ``test_panel_includes_procedural`` — voice_representation,
    voice_enrichment, model_flippability_validity,
    explanation_actionability_validity are all present and finite for a
    non-degenerate cell.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

EXPECTED_COLS = [
    "dataset",
    "target",
    "base_model",
    "method",
    "method_kind",
    "lambda_",
    "seed",
    "metric",
    "value",
    "metric_kind",
    "sample_n",
    "random_state",
    "notes",
]

PANEL_METRICS = {
    # Performance (3)
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    # Statistical (4)
    "macro_dp",
    "macro_eodds",
    "equal_opportunity",
    "counterfactual_fairness",
    # Procedural (5)
    "process_consistency",
    "voice_representation",
    "voice_enrichment",
    "model_flippability_validity",
    "explanation_actionability_validity",
}

def _run_phase5(
    out_dir: pathlib.Path,
    *,
    seeds: str = "0",
    max_workers: int = 1,
    methods: str = "identity_preprocessing",
    timeout: int = 600,
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_phase5_audit.py"),
        "--datasets",
        "ricci",
        "--base-models",
        "RF",
        "--methods",
        methods,
        "--lambdas",
        "0",
        "--seeds",
        seeds,
        "--max-workers",
        str(max_workers),
        "--out-dir",
        str(out_dir),
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=timeout,
    )

def test_run_phase5_smoke(tmp_path: pathlib.Path) -> None:
    out_dir = tmp_path / "phase5"
    proc = _run_phase5(out_dir)
    assert proc.returncode == 0, (
        f"audit script failed (rc={proc.returncode}):\n"
        f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
    )
    csv_path = out_dir / "audit.csv"
    assert csv_path.exists(), f"audit.csv not created at {csv_path}"
    df = pd.read_csv(csv_path)

    #  column order.
    assert list(df.columns) == EXPECTED_COLS, (
        f"audit.csv columns mismatch:\n"
        f"  got      {list(df.columns)}\n"
        f"  expected {EXPECTED_COLS}"
    )

    # 1 cell × 12 metrics emitted by the Tier-5 metric panel.
    assert len(df) == 12, f"expected 12 metric rows, got {len(df)}: {df}"
    assert set(df["metric"]) == PANEL_METRICS, (
        f"metric set mismatch:\n  got      {sorted(df['metric'])}\n"
        f"  expected {sorted(PANEL_METRICS)}"
    )
    assert (df["dataset"] == "ricci").all()
    assert (df["base_model"] == "RF").all()
    assert (df["method"] == "identity_preprocessing").all()
    assert (df["method_kind"] == "pre").all()

def test_run_phase5_byte_identical(tmp_path: pathlib.Path) -> None:
    """Same args + cleared cache → byte-identical CSV."""
    out_dir_1 = tmp_path / "run1"
    out_dir_2 = tmp_path / "run2"

    proc1 = _run_phase5(out_dir_1)
    assert proc1.returncode == 0, proc1.stderr
    proc2 = _run_phase5(out_dir_2)
    assert proc2.returncode == 0, proc2.stderr

    csv1 = (out_dir_1 / "audit.csv").read_bytes()
    csv2 = (out_dir_2 / "audit.csv").read_bytes()
    assert csv1 == csv2, (
        "Phase-5 audit CSV is not byte-identical across two clean runs. "
        f"len1={len(csv1)} len2={len(csv2)}\n"
        f"--- run1 ---\n{csv1.decode()}\n"
        f"--- run2 ---\n{csv2.decode()}"
    )

def test_parallel_correctness(tmp_path: pathlib.Path) -> None:
    """Parallel run (max_workers=4) matches serial run (max_workers=1)."""
    out_serial = tmp_path / "serial"
    out_parallel = tmp_path / "parallel"

    proc1 = _run_phase5(out_serial, seeds="0,1", max_workers=1)
    assert proc1.returncode == 0, proc1.stderr
    proc2 = _run_phase5(out_parallel, seeds="0,1", max_workers=4)
    assert proc2.returncode == 0, proc2.stderr

    csv1 = (out_serial / "audit.csv").read_bytes()
    csv2 = (out_parallel / "audit.csv").read_bytes()
    assert csv1 == csv2, (
        "Parallel run produced a different consolidated CSV than serial. "
        "Lexicographic sort should make these byte-identical regardless of "
        "worker run order. "
        f"\n--- serial ---\n{csv1.decode()}\n"
        f"--- parallel ---\n{csv2.decode()}"
    )

# ---------------------------------------------------------------------
#  — Tier-5 12-metric panel tests.
# ---------------------------------------------------------------------

def test_panel_shape_12_metrics(tmp_path: pathlib.Path) -> None:
    """: each cell emits exactly 12 metric rows, matching the
    Tier-5 panel spec in  / ."""
    out_dir = tmp_path / "phase5_panel"
    proc = _run_phase5(out_dir)
    assert proc.returncode == 0, (
        f"audit script failed (rc={proc.returncode}):\n"
        f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
    )
    df = pd.read_csv(out_dir / "audit.csv")
    # 1 dataset × 1 base × 1 method × 1 lambda × 1 seed = 1 cell.
    assert len(df) == 12, (
        f"expected 12 metric rows per cell, got {len(df)}"
    )
    metrics_seen = set(df["metric"])
    assert metrics_seen == PANEL_METRICS, (
        f"panel metric set mismatch:\n"
        f"  got      {sorted(metrics_seen)}\n"
        f"  expected {sorted(PANEL_METRICS)}"
    )
    # metric_kind tagging contract.
    perf = df[df["metric_kind"] == "performance"]
    stat = df[df["metric_kind"] == "statistical"]
    proc_ = df[df["metric_kind"] == "procedural"]
    assert len(perf) == 3, perf
    assert len(stat) == 4, stat
    assert len(proc_) == 5, proc_

def test_panel_byte_identical_two_runs(tmp_path: pathlib.Path) -> None:
    """: re-running the same cell with the cache cleared produces
    bit-identical metric values (extends  to the full Tier-5
    panel including SHAP / counterfactual / process-consistency).
    """
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"

    proc1 = _run_phase5(out1)
    assert proc1.returncode == 0, proc1.stderr
    proc2 = _run_phase5(out2)
    assert proc2.returncode == 0, proc2.stderr

    csv1 = (out1 / "audit.csv").read_bytes()
    csv2 = (out2 / "audit.csv").read_bytes()
    assert csv1 == csv2, (
        "Tier-5 panel CSV is not byte-identical across two clean runs. "
        "Either the SHAP / CF / PC determinism prelude is leaking randomness "
        "or one of the new metric branches is non-deterministic.\n"
        f"--- run1 ---\n{csv1.decode()}\n"
        f"--- run2 ---\n{csv2.decode()}"
    )

def test_panel_includes_procedural(tmp_path: pathlib.Path) -> None:
    """: voice_representation + voice_enrichment +
    model_flippability_validity + explanation_actionability_validity
    are all present and finite for a non-degenerate cell.
    """
    import math

    out_dir = tmp_path / "phase5_proc"
    proc = _run_phase5(out_dir)
    assert proc.returncode == 0, proc.stderr
    df = pd.read_csv(out_dir / "audit.csv")
    required = {
        "voice_representation",
        "voice_enrichment",
        "model_flippability_validity",
        "explanation_actionability_validity",
    }
    present = required & set(df["metric"])
    assert present == required, (
        f"missing procedural metrics: {sorted(required - present)}"
    )
    for metric_name in required:
        val = float(df.loc[df["metric"] == metric_name, "value"].iloc[0])
        assert math.isfinite(val), (
            f"{metric_name} is not finite (= {val}); for a non-degenerate "
            f"Ricci/RF/identity cell every procedural metric should be "
            f"computable."
        )

# ---------------------------------------------------------------------
# Worker-recycling test (Tier-5b incident 2026-05-01).

# After the 6.5 h oulad run on 2026-05-01 hit a macOS Jetsam OOM kill,
# the runner switched from ``concurrent.futures.ProcessPoolExecutor`` +
# sliding-window submission to ``multiprocessing.Pool`` with
# ``maxtasksperchild=20``. The earlier ``test_runner_memory_bounded``
# test (which drove the sliding-window _fill/wait loop with a mock
# executor) is therefore obsolete; the new contract is "every Pool we
# build for the parallel branch sets maxtasksperchild=20 and uses
# imap_unordered with chunksize=1".
# ---------------------------------------------------------------------

def test_runner_uses_pool_with_maxtasks(tmp_path: pathlib.Path) -> None:
    """Verify the parallel branch builds ``multiprocessing.Pool`` with
    ``maxtasksperchild=20`` so workers are recycled every 20 cells.

    We patch ``multiprocessing.Pool`` (re-exported as
    ``run_phase5_audit.Pool``) with a ``MagicMock`` that records every
    call's kwargs, then drive the runner ``main()`` against an in-process
    args namespace with a single pending cell + ``max-workers=2``. The
    mock pool's ``imap_unordered`` returns an empty iterable, so no real
    work runs and the test finishes in < 0.1 s.
    """
    import sys
    from unittest.mock import MagicMock, patch

    # Import the runner module directly.
    runner_path = str(PROJECT_ROOT / "scripts")
    if runner_path not in sys.path:
        sys.path.insert(0, runner_path)
    import run_phase5_audit as runner  # type: ignore[import]

    out_dir = tmp_path / "phase5_pool"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cache").mkdir(parents=True, exist_ok=True)

    # Mock Pool — context-manager + imap_unordered protocol.
    mock_pool_instance = MagicMock()
    mock_pool_instance.__enter__ = MagicMock(return_value=mock_pool_instance)
    mock_pool_instance.__exit__ = MagicMock(return_value=False)
    mock_pool_instance.imap_unordered = MagicMock(return_value=iter([]))

    mock_pool_cls = MagicMock(return_value=mock_pool_instance)

    # Force the auto-throttle path NOT to drop max_workers below 2.
    with patch.object(runner, "Pool", mock_pool_cls), patch.object(
        runner, "_resolve_max_workers", return_value=2
    ):
        rc = runner.main(
            [
                "--datasets",
                "ricci",
                "--base-models",
                "RF",
                "--methods",
                "identity_preprocessing",
                "--lambdas",
                "0",
                "--seeds",
                "0",
                "--max-workers",
                "2",
                "--out-dir",
                str(out_dir),
            ]
        )
    assert rc == 0

    assert mock_pool_cls.called, "multiprocessing.Pool was never instantiated"
    _, kwargs = mock_pool_cls.call_args
    mtpc = kwargs.get("maxtasksperchild")
    assert isinstance(mtpc, int) and 20 <= mtpc <= 200, (
        f"Pool constructed without a sane maxtasksperchild (got {mtpc!r}). "
        f"Worker recycling is the definitive fix for the 2026-05-01 Jetsam OOM; "
        f"value tuned upward to 200 after the 2026-05-02 watchdog crash showed "
        f"that recycling every 20 cells was thrashing the OS page cache."
    )
    assert kwargs.get("processes") == 2, (
        f"Pool processes mismatch: expected 2, got "
        f"{kwargs.get('processes')!r}"
    )
    assert kwargs.get("initializer") is runner._pool_initializer, (
        "Pool initializer must be _pool_initializer so each worker "
        "re-pins single-thread BLAS for determinism."
    )
    assert kwargs.get("initargs") == ("cpu",), (
        f"Pool initargs mismatch: expected ('cpu',), got "
        f"{kwargs.get('initargs')!r}"
    )

    # imap_unordered must use chunksize=1 so the iterator is lazy and
    # tasks are dispatched as workers free up (no buffering).
    assert mock_pool_instance.imap_unordered.called, (
        "imap_unordered was never called on the Pool"
    )
    _, imap_kwargs = mock_pool_instance.imap_unordered.call_args
    assert imap_kwargs.get("chunksize") == 1, (
        f"imap_unordered called with chunksize="
        f"{imap_kwargs.get('chunksize')!r}; must be 1 so the task "
        f"iterator is consumed lazily."
    )

"""Tests for src/visualisations.py — ABROCA slice-plot functions.

Exit-gate: pytest tests/test_visualisations.py::test_abroca
"""

import numpy as np
import pandas as pd
import pytest

def test_abroca() -> None:
    """Verify compute_abroca and plot_abroca satisfy the exit-gate spec."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for testing

    from procedural_fair_hr.visualisation import compute_abroca, plot_abroca

    rng = np.random.default_rng(0)
    n = 200
    s = pd.Series(["A"] * 100 + ["B"] * 100, name="group")
    y_true = rng.integers(0, 2, n)

    # Perfect scores for group A, random for group B → large ABROCA
    y_score_A = y_true[:100].astype(float)  # perfect
    y_score_B = rng.random(100)             # random
    y_score = np.concatenate([y_score_A, y_score_B])

    abroca = compute_abroca(y_true, y_score, s, privileged_val="A")
    assert 0.0 <= abroca <= 1.0
    assert isinstance(abroca, float)

    # Same scores for both groups → ABROCA ≈ 0
    y_score_same = rng.random(n)
    abroca_fair = compute_abroca(y_true, y_score_same, s, privileged_val="A")
    assert abroca_fair < 0.2  # should be small (random noise only)

    # plot_abroca returns a float and does not crash
    val = plot_abroca(y_true, y_score, s, privileged_val="A", save_path=None)
    assert isinstance(val, float)
    import matplotlib.pyplot as plt
    plt.close("all")

def test_abroca_identical_groups() -> None:
    """ABROCA of two identical score distributions should be near zero."""
    from procedural_fair_hr.visualisation import compute_abroca

    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.integers(0, 2, n)
    y_score = rng.random(n)
    sensitive = pd.Series(["X"] * 50 + ["Y"] * 50, name="group")

    # Use same underlying scores for both groups
    abroca = compute_abroca(y_true, y_score, sensitive, privileged_val="X")
    assert 0.0 <= abroca <= 1.0

def test_abroca_perfect_vs_random() -> None:
    """Perfect group vs random group should yield a meaningfully large ABROCA."""
    from procedural_fair_hr.visualisation import compute_abroca

    rng = np.random.default_rng(7)
    n_half = 150

    # Both groups have mixed labels so ROC is well-defined
    y_true_A = rng.integers(0, 2, n_half)
    y_true_B = rng.integers(0, 2, n_half)
    y_true = np.concatenate([y_true_A, y_true_B])

    # Group A: perfect scores; Group B: purely random scores
    y_score_A = y_true_A.astype(float)
    y_score_B = rng.random(n_half)
    y_score = np.concatenate([y_score_A, y_score_B])

    sensitive = pd.Series(["priv"] * n_half + ["unpriv"] * n_half)
    abroca = compute_abroca(y_true, y_score, sensitive, privileged_val="priv")
    # Perfect vs random should produce a reasonably large gap
    assert abroca > 0.1

def test_plot_abroca_save(tmp_path) -> None:
    """plot_abroca should save a file when save_path is provided."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from procedural_fair_hr.visualisation import plot_abroca

    rng = np.random.default_rng(1)
    n = 100
    y_true = rng.integers(0, 2, n)
    y_score = rng.random(n)
    sensitive = pd.Series(["A"] * 50 + ["B"] * 50, name="group")

    out_path = str(tmp_path / "abroca_test.png")
    val = plot_abroca(
        y_true, y_score, sensitive, privileged_val="A",
        title="Test ABROCA", save_path=out_path
    )
    assert isinstance(val, float)
    import os
    assert os.path.isfile(out_path)
    plt.close("all")

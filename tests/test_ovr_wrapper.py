"""Tests for ``src/mitigation/ovr_wrapper.py``.

Exit-gate:
    pytest tests/test_ovr_wrapper.py
    pytest tests/test_invariants.py::test_ovr_binary_restriction

Tests covered:
  * ``test_binary_restriction_equivalence``: when |𝒞| = 2, OvR(f)
    produces predictions identical to the unwrapped binary ``f`` on the
    same (X, y, A).
  * ``test_predict_returns_correct_shape`` — multi-class (3-class) predict
    returns a length-n integer ndarray with values drawn from {0, 1, 2}.
  * ``test_predict_proba_returns_n_by_K`` — multi-class predict_proba
    returns shape (n, |𝒞|).
  * ``test_calibration_platt_changes_scores_but_not_argmax_in_simple_cases``
    — sanity: Platt calibration alters the raw score matrix but for an
    easy 3-class problem leaves the argmax (= predict) unchanged.

All seeds are fixed to 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from procedural_fair_hr.mitigation.ovr_wrapper import OneVsRestFairnessAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _logreg_factory():
    """Zero-arg factory returning a fresh ``LogisticRegression(seed=0)``.

    Used as the stand-in "binary mitigation method" for these tests; the
    OvR wrapper's correctness does not depend on the wrapped method's
    fairness behaviour, only on the algebraic OvR construction.
    """
    return LogisticRegression(random_state=0, max_iter=1000)

def _make_binary_dataset(n: int = 200, seed: int = 0):
    """Return (X, y, A) for a simple binary task with a sensitive attribute."""
    X, y = make_classification(
        n_samples=n,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        random_state=seed,
    )
    rng = np.random.default_rng(seed)
    A = pd.DataFrame({"sex": rng.integers(0, 2, size=n)})
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]), pd.Series(y), A

def _make_three_class_dataset(n: int = 300, seed: int = 0):
    """Return (X, y, A) for a 3-class task with a sensitive attribute."""
    X, y = make_classification(
        n_samples=n,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=seed,
    )
    rng = np.random.default_rng(seed)
    A = pd.DataFrame({"sex": rng.integers(0, 2, size=n)})
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]), pd.Series(y), A

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_binary_restriction_equivalence():
    """: with |𝒞| = 2, OvR(method) ≡ method on the same (X, y, A).

    See  §Decision (item 4). Plain ``LogisticRegression`` is a
    valid stand-in: the OvR-equivalence property is purely algebraic, the
    method's fairness behaviour is irrelevant for this test.
    """
    X, y, A = _make_binary_dataset(seed=0)

    # Unwrapped binary baseline.
    baseline = _logreg_factory().fit(X, y)
    baseline_pred = baseline.predict(X)

    # OvR-wrapped binary.
    adapter = OneVsRestFairnessAdapter(_logreg_factory).fit(X, y, A)
    ovr_pred = adapter.predict(X)

    assert ovr_pred.shape == baseline_pred.shape
    assert np.array_equal(
        ovr_pred, baseline_pred
    ), "OvR(method).predict must equal method.predict when |𝒞| = 2."

def test_predict_returns_correct_shape():
    """Multi-class (3-class): predict returns (n,) ints in {0, 1, 2}."""
    X, y, A = _make_three_class_dataset(seed=0)

    adapter = OneVsRestFairnessAdapter(_logreg_factory).fit(X, y, A)
    pred = adapter.predict(X)

    assert isinstance(pred, np.ndarray)
    assert pred.shape == (len(X),)
    assert set(np.unique(pred)).issubset({0, 1, 2})

def test_predict_proba_returns_n_by_K():
    """Multi-class (3-class): predict_proba returns (n, 3)."""
    X, y, A = _make_three_class_dataset(seed=0)

    adapter = OneVsRestFairnessAdapter(_logreg_factory).fit(X, y, A)
    proba = adapter.predict_proba(X)

    assert isinstance(proba, np.ndarray)
    assert proba.shape == (len(X), 3)
    # Per  the columns are not required to be a simplex, but each
    # individual column comes from a binary predict_proba and should
    # therefore be in [0, 1].
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

def test_calibration_platt_changes_scores_but_not_argmax_in_simple_cases():
    """Platt calibration alters the raw score matrix.

    On an easy, well-separated 3-class problem the argmax (= predict)
    should remain identical to the raw-score variant — the calibrator is
    a monotone transformation of each column independently and on a
    well-separated problem the column-wise rank ordering of (n × |𝒞|)
    score matrix is preserved at most rows. We therefore assert:

        (a) at least one row's score vector differs between raw and
            Platt — i.e., calibration is doing something;
        (b) the predicted classes agree on a strong majority of rows
            (>= 90 %).
    """
    X, y, A = _make_three_class_dataset(n=300, seed=0)

    raw = OneVsRestFairnessAdapter(_logreg_factory, calibration="raw").fit(X, y, A)
    platt = OneVsRestFairnessAdapter(_logreg_factory, calibration="platt").fit(X, y, A)

    raw_scores = raw.predict_proba(X)
    platt_scores = platt.predict_proba(X)

    # (a) Calibration is doing something — scores actually differ.
    assert not np.allclose(raw_scores, platt_scores), (
        "Platt calibration produced identical scores to raw — calibration "
        "appears not to be applied."
    )

    # (b) Argmax agrees on a strong majority of rows on this easy problem.
    raw_pred = raw.predict(X)
    platt_pred = platt.predict(X)
    agreement = float(np.mean(raw_pred == platt_pred))
    assert agreement >= 0.9, (
        f"raw vs platt argmax agreement only {agreement:.2%}; expected >= 90% "
        "on an easy 3-class problem."
    )

def test_invalid_calibration_raises():
    """Constructor rejects calibration values outside the documented set."""
    with pytest.raises(ValueError):
        OneVsRestFairnessAdapter(_logreg_factory, calibration="bogus")

def test_non_callable_base_method_raises():
    """Constructor rejects a non-callable base_method."""
    with pytest.raises(TypeError):
        OneVsRestFairnessAdapter(base_method=42)  # type: ignore[arg-type]

"""Unit tests for the Phase-5 mitigation adapter base class.

Covers:

  * Registry signatures — every entry in ``MITIGATION_REGISTRY`` is a
    subclass of :class:`MitigationBase` and exposes the required class
    attributes / methods.
  * Hyperparameter grid — :attr:`hyperparameter_grid` is the canonical
    sweep ``[0.0, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0]`` .
  * Multi-class compat — wrappers with ``multi_class_native = False``
    raise ``NotImplementedError`` when fed ``|𝒞| > 2`` targets unless
    OvR-wrapped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from procedural_fair_hr.mitigation import (
    CANONICAL_HYPERPARAMETER_GRID,
    MITIGATION_REGISTRY,
    MitigationBase,
    register,
)

# ---------------------------------------------------------------------
# Stubs used by the unit tests
# ---------------------------------------------------------------------

class _BinaryStub(MitigationBase):
    """Minimal binary-only mitigation stub used to exercise the
    multi-class guard."""

    method_name = "_binary_stub"
    method_kind = "in"
    library_origin = "thesis"
    multi_class_native = False

    def fit(self, X, y, A):
        self._check_multiclass(y)
        self._fitted = True
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return None

# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_registry_signatures() -> None:
    """Every registered class is a MitigationBase subclass and exposes
    the canonical interface (class attributes + fit/predict/predict_proba)."""
    assert len(MITIGATION_REGISTRY) >= 3, (
        f"At minimum we expect identity_preprocessing + adversarial_debiasing "
        f"+ identity_postprocessing in the registry; got "
        f"{sorted(MITIGATION_REGISTRY)}"
    )
    for name, cls in MITIGATION_REGISTRY.items():
        assert issubclass(cls, MitigationBase), (
            f"{name!r} is not a MitigationBase subclass"
        )
        for attr in ("method_name", "method_kind", "library_origin",
                        "multi_class_native"):
            assert hasattr(cls, attr), f"{name!r} missing attr {attr!r}"
        assert cls.method_kind in ("pre", "in", "post"), (
            f"{name!r}.method_kind must be one of {{pre,in,post}}; "
            f"got {cls.method_kind!r}"
        )
        for method_name in ("fit", "predict", "predict_proba"):
            assert callable(getattr(cls, method_name, None)), (
                f"{name!r} missing callable {method_name}"
            )

def test_hyperparameter_grid_canonical() -> None:
    """Default grid matches  spec."""
    inst = next(iter(MITIGATION_REGISTRY.values()))()
    grid = inst.hyperparameter_grid
    assert grid == CANONICAL_HYPERPARAMETER_GRID
    assert grid == [0.0, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0]

def test_multiclass_guard_raises_for_binary_only() -> None:
    """A wrapper with ``multi_class_native=False`` rejects |𝒞|>2 ``y``."""
    rng = np.random.default_rng(0)
    n = 30
    X = pd.DataFrame(rng.normal(size=(n, 4)))
    y_multi = np.array([0, 1, 2] * (n // 3))
    A = pd.DataFrame({"sensitive": rng.integers(0, 2, n)})

    stub = _BinaryStub()
    with pytest.raises(NotImplementedError, match="binary-only"):
        stub.fit(X, y_multi, A)

    # Same stub on binary y must succeed.
    y_bin = (y_multi > 0).astype(int)
    stub.fit(X, y_bin, A)

def test_register_decorator_replaces_existing_entry() -> None:
    """Registering a name a second time replaces the prior entry — this
    is the contract used by tests that swap a real method for a stub."""

    @register("_register_test_stub")
    class A(MitigationBase):
        method_name = "_register_test_stub"
        method_kind = "in"
        library_origin = "thesis"
        multi_class_native = True

        def fit(self, X, y, A_):
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            return None

    @register("_register_test_stub")
    class B(MitigationBase):
        method_name = "_register_test_stub"
        method_kind = "in"
        library_origin = "thesis"
        multi_class_native = True

        def fit(self, X, y, A_):
            return self

        def predict(self, X):
            return np.ones(len(X), dtype=int)

        def predict_proba(self, X):
            return None

    assert MITIGATION_REGISTRY["_register_test_stub"] is B

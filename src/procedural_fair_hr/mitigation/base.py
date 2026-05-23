"""Mitigation adapter base class + global registry.

This module defines the canonical interface every Phase-5 mitigation
wrapper exposes (`MitigationBase`) and a process-global registry
(`MITIGATION_REGISTRY`) populated via the `@register("name")` decorator
in :mod:`procedural_fair_hr.mitigation.preprocessing`, :mod:`procedural_fair_hr.mitigation.inprocessing`,
:mod:`procedural_fair_hr.mitigation.postprocessing`.

Contract (mirrors the project documentation):

  Every concrete subclass:

    * Accepts the constructor signature
      ``(base_estimator, lambda_, sensitive_col, random_state)`` (in any
      order via keyword) at minimum; subclasses may add their own
      hyperparameters.
    * Exposes ``fit(X, y, A) -> self``, ``predict(X) -> ndarray``, and
      ``predict_proba(X) -> ndarray | None``.
    * Declares the class attributes ``method_name``, ``method_kind``
      (``"pre" | "in" | "post"``), ``library_origin`` (``"aif360" |
      "fairlearn" | "imblearn" | "pytorch" | "thesis"``), and
      ``multi_class_native`` (``True`` iff the underlying method handles
      ``|𝒞| > 2`` without an OvR wrapper).
    * Registers itself by name at import time using the
      :func:`register` decorator.

The canonical hyperparameter grid for parametric methods is
``[0.0, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0]``. Methods with
non-numeric or method-specific hyperparameters override
:attr:`hyperparameter_grid` to surface their own canonical sweep.

Multi-class compatibility:

  Methods declaring ``multi_class_native = False`` raise
  ``NotImplementedError`` from ``fit`` when ``len(np.unique(y)) > 2``
  unless wrapped via :class:`procedural_fair_hr.mitigation.ovr_wrapper.OneVsRestFairnessAdapter`
  per  / .

References (this contract),  (binary-restriction equivalence).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------

MITIGATION_REGISTRY: dict[str, type["MitigationBase"]] = {}

def register(name: str) -> Callable[[type["MitigationBase"]], type["MitigationBase"]]:
    """Class decorator: register a :class:`MitigationBase` subclass by ``name``.

    Re-registering the same name overrides the prior entry (last
    decorator wins). This is intentional for tests that swap a real
    method for a stub — production code never re-registers.
    """

    def decorator(cls: type["MitigationBase"]) -> type["MitigationBase"]:
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"register() requires a non-empty string name; got {name!r}"
            )
        MITIGATION_REGISTRY[name] = cls
        return cls

    return decorator

# ---------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------

# Canonical hyperparameter sweep .
CANONICAL_HYPERPARAMETER_GRID: list[float] = [0.0, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0]

class MitigationBase(ABC):
    """Abstract base for every Phase-5 mitigation wrapper.

    Subclasses MUST set the class attributes ``method_name``,
    ``method_kind``, ``library_origin``, and ``multi_class_native`` AND
    implement :meth:`fit`, :meth:`predict`, :meth:`predict_proba`. See
    the module docstring +  for the full contract.
    """

    # Class-level metadata (overridden by every concrete subclass).
    method_name: ClassVar[str] = "mitigation_base"
    method_kind: ClassVar[str] = "in"  # "pre" | "in" | "post"
    library_origin: ClassVar[str] = "thesis"
    multi_class_native: ClassVar[bool] = False

    def __init__(
        self,
        base_estimator: Optional[Any] = None,
        lambda_: float = 1.0,
        sensitive_col: str = "sensitive",
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        self.base_estimator = base_estimator
        self.lambda_ = float(lambda_)
        self.sensitive_col = sensitive_col
        self.random_state = int(random_state)
        # Surface any extra kwargs on ``self`` so subclasses can pick them
        # up by name without each overriding ``__init__`` boilerplate.
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        A: pd.DataFrame,
    ) -> "MitigationBase":
        """Fit the mitigation method.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.
        y : numpy.ndarray
            Target vector. Subclasses with ``multi_class_native = False``
            must raise ``NotImplementedError`` when ``len(np.unique(y)) > 2``.
        A : pandas.DataFrame
            One-column-per-attribute sensitive frame. The
            wrapper picks ``A[self.sensitive_col]`` when an underlying
            method needs a single column.

        Returns
        -------
        self
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return hard label predictions."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Return per-class probabilities, or ``None`` if not supported."""

    # ------------------------------------------------------------------
    # Defaults shared by every subclass
    # ------------------------------------------------------------------

    @property
    def hyperparameter_grid(self) -> list[float]:
        """Canonical λ sweep .

        Subclasses that take a non-λ hyperparameter (e.g., AdvDebias's
        adversary-loss weight) reuse this grid by re-interpreting it.
        Methods that have NO tunable hyperparameter (e.g., Reweighing)
        return ``[0.0]``.
        """
        return list(CANONICAL_HYPERPARAMETER_GRID)

    def _check_multiclass(self, y: np.ndarray) -> None:
        """Guard for binary-only methods.

        Concrete subclasses with ``multi_class_native = False`` should
        call this at the top of ``fit`` so the user gets a clear error
        rather than a silent contract violation. Wrapping via
        :class:`OneVsRestFairnessAdapter` lifts the binary method to
        ``|𝒞| > 2`` per  / .
        """
        n_classes = int(np.unique(np.asarray(y)).size)
        if n_classes > 2 and not self.multi_class_native:
            raise NotImplementedError(
                f"{type(self).__name__} (method_name={self.method_name!r}) "
                f"is binary-only (multi_class_native={self.multi_class_native}); "
                f"got y with {n_classes} unique classes. Wrap with "
                f"procedural_fair_hr.mitigation.ovr_wrapper.OneVsRestFairnessAdapter "
                f"for multi-class targets."
            )

    # ------------------------------------------------------------------
    # Convenience helpers (sklearn compatibility)
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover — debugging aid
        return (
            f"{type(self).__name__}(method={self.method_name!r}, "
            f"kind={self.method_kind!r}, lambda_={self.lambda_:.4g}, "
            f"sensitive_col={self.sensitive_col!r}, "
            f"random_state={self.random_state})"
        )

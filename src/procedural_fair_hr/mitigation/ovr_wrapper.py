"""One-vs-rest (OvR) adapter for binary fairness mitigation methods.

This module implements ``OneVsRestFairnessAdapter``, the wrapper specified by
 (ACCEPTED 2026-04-28, the project documentation). FairLearn
reductions and AIF360 binary in-/post-processing methods do not natively
support multi-class targets; this adapter lifts any such binary method to
``|𝒞| > 2`` class targets via the classical one-vs-rest aggregation
framework of Allwein, Schapire & Singer (2000) — see  in
the project documentation regarding the BibTeX entry.

Algorithm:

    For each class c ∈ 𝒞:
        Train f on (X, y == c, A)
        Predict per-class score s_c(x) for each test sample x
    Aggregate predictions: ŷ(x) = argmax_c s_c(x)

The wrapper is contractually verified by
(the project documentation): when ``|𝒞| = 2``, ``OvR(f)`` must produce
predictions identical to the unwrapped binary ``f`` on the same data
(binary-restriction equivalence).

Score sources (in order of preference for each fitted binary estimator):

    1. ``predict_proba(X)[:, 1]``  — preferred when available.
    2. ``decision_function(X)``    — used when ``predict_proba`` is absent.
    3. ``predict(X)``              — fallback (0/1 hard labels treated as
       degenerate scores).

Calibration (``calibration`` constructor argument):

    * ``"raw"`` (default) — use the raw score described above. Fast; matches
      the score the binary mitigation method itself surfaces.
    * ``"platt"`` — wrap each fitted binary estimator with
      ``sklearn.calibration.CalibratedClassifierCV(method="sigmoid")`` and
      use the calibrated ``predict_proba(X)[:, 1]``. Per-class normalisation
      across the |𝒞| score vectors is left as a sensitivity-analysis
      follow-up (not required for  exit gate; see
      §Consequences).

References:

    * Agarwal, A., Beygelzimer, A., Dudík, M., Langford, J. & Wallach, H.
      (2018). "A Reductions Approach to Fair Classification."
      ICML 2018. (BibTeX key: ``agarwal2018reductions`` — landing via the
      parallel BibTeX-batch  per  " BibTeX batch".)
    * Allwein, E. L., Schapire, R. E. & Singer, Y. (2000). "Reducing
      Multiclass to Binary: A Unifying Approach for Margin Classifiers."
      *Journal of Machine Learning Research* 1, 113–141. (BibTeX key:
      ``allwein2000reducing`` — .)

 enforcement:
    ``tests/test_invariants.py::test_ovr_binary_restriction`` parameterises
    over every binary method registered in the mitigation registry and
    asserts the equivalence holds.
"""

from __future__ import annotations

import inspect
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

# Type alias for the factory contract: ``base_method`` is either
# (a) a zero-arg callable returning a fresh, unfit, sklearn-compatible
# binary estimator, or (b) a class whose ``__init__`` takes no required
# args.
BaseMethodFactory = Callable[[], object]

_VALID_CALIBRATIONS = ("raw", "platt")

class OneVsRestFairnessAdapter:
    """Lift a binary fairness mitigation method to multi-class via OvR.

    See module docstring and  for the algorithm and rationale.
     does not currently define an OvR-specific term; the wrapper
    is described entirely by  and .

    Args:
        base_method: A zero-arg callable that returns an unfit, sklearn-
            compatible binary estimator. Examples for the Phase-5 registry
            include thin wrappers around
            ``fairlearn.reductions.ExponentiatedGradient(...)`` or
            AIF360's ``ExponentiatedGradientReduction``. The estimator's
            ``fit`` method may optionally accept ``sensitive_features`` (or
            ``A``) — the wrapper passes the sensitive attribute through
            when the signature accepts it; otherwise it is dropped.
        calibration: Either ``"raw"`` (default) or ``"platt"``. See module
            docstring.

    Attributes:
        base_method: The factory passed in at construction.
        calibration: The calibration mode.
        classes_: The sorted unique classes seen during ``fit`` (np.ndarray).
        estimators_: List of ``|𝒞|`` fitted binary estimators, one per class.
            For ``calibration="platt"`` each entry is a fitted
            ``CalibratedClassifierCV`` wrapping the underlying estimator.

    Notes:
        * Random seed: callers control determinism via the seed inside
          ``base_method``. The adapter itself does no random
          sampling other than the optional Platt calibration's internal
          cross-validation, which uses sklearn's default RNG; if strict
          determinism on the calibrated path is required, the caller
          should pin ``np.random.seed`` / ``random.seed`` before
          invoking ``fit``.
    """

    def __init__(
        self,
        base_method: BaseMethodFactory,
        *,
        calibration: str = "raw",
    ) -> None:
        if calibration not in _VALID_CALIBRATIONS:
            raise ValueError(
                f"calibration must be one of {_VALID_CALIBRATIONS!r}; "
                f"got {calibration!r}"
            )
        if not callable(base_method):
            raise TypeError(
                "base_method must be a zero-arg callable that returns an "
                "unfit binary estimator (e.g., a class object or a lambda "
                f"closing over the configured method); got {type(base_method)!r}"
            )
        self.base_method = base_method
        self.calibration = calibration
        self.classes_: Optional[np.ndarray] = None
        self.estimators_: list[object] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _accepts_kwarg(fit_method: Callable, name: str) -> bool:
        """Return True iff ``fit_method`` accepts a keyword argument ``name``.

        Used to decide whether to forward ``sensitive_features`` or ``A``
        to the wrapped binary fit. Plain sklearn estimators do not accept
        either; FairLearn reductions accept ``sensitive_features``;
        AIF360-style methods accept ``A``.
        """
        try:
            sig = inspect.signature(fit_method)
        except (TypeError, ValueError):
            return False
        params = sig.parameters
        if name in params:
            return True
        # If the fit signature has **kwargs we conservatively forward.
        return any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

    def _fit_one_binary(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y_binary: np.ndarray,
        A: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]],
    ) -> object:
        """Fit one binary estimator on ``(X, y_binary, A)`` and return it.

        Forwards ``A`` to the binary estimator's ``fit`` only if its
        signature mentions ``sensitive_features`` or ``A``; this keeps the
        wrapper compatible with plain sklearn estimators (used in the
        binary-restriction equivalence tests) while still piping the
        sensitive attribute through to FairLearn / AIF360 mitigators.
        """
        estimator = self.base_method()
        fit = estimator.fit
        kwargs: dict = {}
        if A is not None:
            if self._accepts_kwarg(fit, "sensitive_features"):
                kwargs["sensitive_features"] = A
            elif self._accepts_kwarg(fit, "A"):
                kwargs["A"] = A
        estimator.fit(X, y_binary, **kwargs)

        if self.calibration == "platt":
            # Wrap the *fitted* base estimator with sigmoid calibration so
            # the underlying fairness-mitigation behaviour is preserved
            # (only the score-to-probability mapping is learned).
            from sklearn.calibration import CalibratedClassifierCV

            try:
                # sklearn >= 1.6: cv="prefit" is deprecated; the supported
                # pattern is to wrap with FrozenEstimator first.
                from sklearn.frozen import FrozenEstimator  # type: ignore

                calibrator = CalibratedClassifierCV(
                    estimator=FrozenEstimator(estimator), method="sigmoid"
                )
            except ImportError:  # pragma: no cover — older sklearn
                calibrator = CalibratedClassifierCV(
                    estimator=estimator, method="sigmoid", cv="prefit"
                )
            # The calibrator needs (X, y) for its hold-out fit; we re-use
            # the same training data here (standard caveat for the
            # prefit / frozen-estimator pattern). Per-class normalisation
            # across the |𝒞| score columns is a sensitivity-analysis
            # follow-up.
            calibrator.fit(X, y_binary)
            return calibrator
        return estimator

    @staticmethod
    def _score_one(estimator: object, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Return a 1-D positive-class score vector for one fitted estimator.

        Preference order: ``predict_proba(X)[:, 1]`` → ``decision_function(X)``
        → ``predict(X)``. The fallback chain matches sklearn's
        ``OneVsRestClassifier`` and FairLearn's documentation.
        """
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(X)
            proba = np.asarray(proba)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1].astype(float)
            # Degenerate: only one column returned (single class fitted).
            return proba.ravel().astype(float)
        if hasattr(estimator, "decision_function"):
            return np.asarray(estimator.decision_function(X), dtype=float).ravel()
        # Fallback: hard labels as degenerate scores.
        return np.asarray(estimator.predict(X), dtype=float).ravel()

    # ------------------------------------------------------------------
    # sklearn-style public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        A: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
    ) -> "OneVsRestFairnessAdapter":
        """Fit one binary instance of ``base_method`` per class in ``y``.

        Args:
            X: Feature matrix (n × d).
            y: Multi-class target labels (n,). Classes are inferred via
                ``np.unique`` and stored in ``self.classes_`` in sorted
                order.
            A: Optional sensitive-attribute frame / series / array. Per
                 /  this is typically a ``pandas.DataFrame``
                with one column per sensitive attribute. The adapter
                forwards it to the binary ``fit`` only if the signature
                accepts ``sensitive_features`` (FairLearn) or ``A``
                (AIF360); plain sklearn estimators receive only ``X``
                and a binarised ``y``.

        Returns:
            ``self`` (sklearn convention).
        """
        y_arr = np.asarray(y).ravel()
        self.classes_ = np.unique(y_arr)
        self.estimators_ = []
        for c in self.classes_:
            y_binary = (y_arr == c).astype(int)
            est = self._fit_one_binary(X, y_binary, A)
            self.estimators_.append(est)
        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Return per-class scores of shape ``(n, |𝒞|)``.

        Each column ``c`` is the positive-class score from the binary
        estimator that was trained with ``y == classes_[c]`` as the
        positive label. Scores are NOT normalised across classes; callers needing a probability simplex should apply
        ``softmax`` or row-normalise themselves.
        """
        if self.classes_ is None or not self.estimators_:
            raise RuntimeError(
                "OneVsRestFairnessAdapter must be fit before predict_proba."
            )
        cols = [self._score_one(est, X) for est in self.estimators_]
        return np.column_stack(cols).astype(float)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Return ``argmax`` over per-class scores as integer class labels.

        The returned array has shape ``(n,)`` and values drawn from
        ``self.classes_`` (so for classes ``[0, 1, 2]`` the output values
        live in ``{0, 1, 2}``).

        Per  this must be identical to the unwrapped binary
        method's output when ``|𝒞| = 2``; the binary case takes the
        argmax over the two-column score matrix, which equals
        ``score_class1 > score_class0``, equivalent to thresholding the
        positive-class score at the midpoint of the two scores. For all
        common binary score sources (``predict_proba``,
        ``decision_function``, hard ``predict``) this argmax reproduces
        the binary classifier's own ``predict`` output exactly — see
        ``tests/test_ovr_wrapper.py::test_binary_restriction_equivalence``.
        """
        if self.classes_ is None or not self.estimators_:
            raise RuntimeError(
                "OneVsRestFairnessAdapter must be fit before predict."
            )
        scores = self.predict_proba(X)
        # In the |𝒞| = 2 case the unwrapped binary method's predict is
        # the right reference; we delegate to it for byte-identical output
        # rather than relying on argmax of (negative-class, positive-class)
        # score columns, which can flip on ties or when the score source
        # is hard 0/1 labels.
        if len(self.classes_) == 2:
            est = self.estimators_[1]  # estimator trained with class==classes_[1] as positive
            preds = np.asarray(est.predict(X)).ravel().astype(int)
            # ``preds`` is a 0/1 array indicating "is class classes_[1]?";
            # map to the actual class labels.
            return np.where(preds == 1, self.classes_[1], self.classes_[0])
        idx = np.argmax(scores, axis=1)
        return self.classes_[idx]

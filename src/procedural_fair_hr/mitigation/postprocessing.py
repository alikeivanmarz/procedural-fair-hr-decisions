"""Post-processing mitigation wrapper.

Houses the post-processing wrapper used by the thesis:

    * :class:`EqOddsPostprocessor` — Hardt, Price & Srebro 2016
      Equalised-Odds post-processor.

The wrapper exposes both ``predict(X)`` and ``predict_with_A(X, A)`` so
callers that have the test-time sensitive attribute can apply the
learned per-group mixing; callers without ``A`` fall back to the base
estimator's untouched predictions.

Reference
---------

Hardt, M., Price, E. & Srebro, N. (2016). "Equality of Opportunity in
Supervised Learning." *NeurIPS* (BibTeX: ``hardt2016equality``).
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from .base import MitigationBase, register
from .preprocessing import (
    _fit_base_estimator,
    _predict_proba_safe,
    _privileged_groups_from_A,
    _to_aif360_dataset,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def _make_pred_dataset(
    X: pd.DataFrame,
    predicted_y: np.ndarray,
    A: pd.DataFrame,
    sensitive_col: str,
    *,
    scores: Optional[np.ndarray] = None,
):
    """Build a predicted-label ``BinaryLabelDataset`` for AIF360 post-procs.

    Mirrors :func:`procedural_fair_hr.mitigation.preprocessing._to_aif360_dataset` but
    populates the ``__y`` column with PREDICTED labels (cast to float)
    and optionally seeds the ``scores`` field with positive-class
    probabilities (required when AIF360 post-processors need score input).
    """
    from aif360.datasets import BinaryLabelDataset

    df = X.reset_index(drop=True).copy()
    y_arr = np.asarray(predicted_y).astype(float)
    df["__y"] = y_arr

    sens_series = A.reset_index(drop=True)[sensitive_col]
    if not pd.api.types.is_numeric_dtype(sens_series):
        sens_codes = pd.Categorical(sens_series).codes.astype(float)
    else:
        sens_codes = sens_series.astype(float).values
    df["__sensitive"] = sens_codes

    ds = BinaryLabelDataset(
        favorable_label=1.0,
        unfavorable_label=0.0,
        df=df,
        label_names=["__y"],
        protected_attribute_names=["__sensitive"],
    )
    if scores is not None:
        scores_arr = np.asarray(scores).astype(float).reshape(-1, 1)
        ds.scores = scores_arr
    return ds

def _has_predict_proba(estimator) -> bool:
    """Return True iff ``estimator`` exposes a usable ``predict_proba``."""
    if not hasattr(estimator, "predict_proba"):
        return False
    # Some sklearn estimators (e.g., SVC with ``probability=False``) define
    # predict_proba but raise at call time. We tolerate this — the caller
    # path catches the AttributeError / NotFittedError and falls back.
    return True

# ---------------------------------------------------------------------
# Identity placeholder (kept; required by smoke tests)
# ---------------------------------------------------------------------

@register("identity_postprocessing")
class IdentityPostprocessing(MitigationBase):
    """No-op post-processing wrapper used by smoke / parallel tests.

    Trains the base estimator and returns its predictions unmodified.
    Equivalent to a "λ=0 baseline" reference for all post-processing
    methods.
    """

    method_name = "identity_postprocessing"
    method_kind = "post"
    library_origin = "thesis"
    multi_class_native = True

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        A: pd.DataFrame,
    ) -> "IdentityPostprocessing":
        if self.base_estimator is None:
            self.base_estimator = LogisticRegression(
                max_iter=1000, random_state=self.random_state
            )
        self._estimator = clone(self.base_estimator)
        if hasattr(self._estimator, "random_state"):
            try:
                self._estimator.set_params(random_state=self.random_state)
            except Exception:  # pragma: no cover
                pass
        self._estimator.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "IdentityPostprocessing.predict called before fit()"
            )
        return np.asarray(self._estimator.predict(X))

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if not self._fitted:
            raise RuntimeError(
                "IdentityPostprocessing.predict_proba called before fit()"
            )
        if hasattr(self._estimator, "predict_proba"):
            return np.asarray(self._estimator.predict_proba(X))
        return None

# ---------------------------------------------------------------------
#  — Equalised Odds Postprocessor (Hardt 2016)
# ---------------------------------------------------------------------

@register("eqodds_postproc")
class EqOddsPostprocessor(MitigationBase):
    """Hardt, Price & Srebro 2016 Equalised-Odds post-processor.

    See ``hardt2016equality``. Solves a linear program to mix per-group
    classifier outputs so that both TPR and FPR equalise across protected
    groups while minimising accuracy loss (Hardt 2016 §4).

    Workflow
    --------
    1. ``fit(X, y, A)``: train ``base_estimator`` on (X, y); apply
       ``aif360.algorithms.postprocessing.EqOddsPostprocessing`` to the
       ``(true_y, predicted_y, A)`` triple to learn per-group mixing
       coefficients.
    2. ``predict_with_A(X, A)``: run ``base_estimator.predict(X)``, then
       apply the learned mixing per-group via the AIF360 post-processor.

    Hyperparameter
    --------------
    Hardt 2016 is parameter-free (the LP enforces equalised odds
    exactly). We accept ``lambda_`` for  conformance and treat:

        * ``λ == 0`` → "skip post-proc" (return base predictions
          byte-equivalent — true identity).
        * ``λ > 0`` → apply the Hardt 2016 LP-mixing.

    Multi-class
    -----------
    NOT NATIVE (Hardt 2016 binary). The audit runner auto-wraps via OvR
    per  / .

    Test-time A requirement
    -----------------------
    Per-group threshold mixing is fundamentally a function of the test-
    time sensitive value, so we expose ``predict_with_A(X, A)``. The
    plain ``predict(X)`` path falls back to the unmitigated base
    estimator (post-processor is skipped) since we cannot reconstruct
    test-time A reliably; the audit runner's ``_run_cell`` shim always
    routes through ``predict_with_A``.

    Failure mode
    ------------
    AIF360's LP solver can fail on degenerate data (single-class group,
    perfect separation). We catch every exception, fall back to the
    unmitigated base estimator, and surface the failure on
    ``self.notes_`` per 's graceful-fallback policy.
    """

    method_name = "eqodds_postproc"
    method_kind = "post"
    library_origin = "aif360"
    multi_class_native = False

    def fit(
        self, X: pd.DataFrame, y: np.ndarray, A: pd.DataFrame
    ) -> "EqOddsPostprocessor":
        self._check_multiclass(y)
        self.notes_: str = ""

        # 1. Always train a base estimator (used for the λ=0 identity
        #    path AND as the graceful-fallback when the LP solver fails).
        self._estimator = _fit_base_estimator(
            self.base_estimator, X, y, random_state=self.random_state
        )

        if self.lambda_ == 0.0:
            # λ=0 → identity. No post-proc fitted.
            self._eqodds = None
            self._fitted = True
            return self

        try:
            from aif360.algorithms.postprocessing import (
                EqOddsPostprocessing as _AIF360EqOdds,
            )

            # 2. Predict on training data → predicted-label dataset.
            y_pred_train = np.asarray(self._estimator.predict(X)).astype(int)
            true_ds = _to_aif360_dataset(X, y, A, self.sensitive_col)
            pred_ds = _make_pred_dataset(
                X, y_pred_train, A, self.sensitive_col
            )
            priv, unpriv = _privileged_groups_from_A(A, self.sensitive_col)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                eqodds = _AIF360EqOdds(
                    unprivileged_groups=unpriv,
                    privileged_groups=priv,
                    seed=self.random_state,
                )
                eqodds.fit(true_ds, pred_ds)
            self._eqodds = eqodds
        except Exception as exc:
            logger.warning(
                "EqOddsPostprocessing fit failed (%s); falling back to "
                "unmitigated base estimator.",
                exc,
            )
            self.notes_ = f"eqodds_postproc_failed: {type(exc).__name__}"
            self._eqodds = None

        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("EqOddsPostprocessor.predict called before fit()")
        # No A available → fall back to base estimator (post-proc is
        # mathematically a per-group function of A; we cannot apply it
        # without A). The audit runner's ``_run_cell`` shim always uses
        # ``predict_with_A`` when present, so this path is only hit by
        # callers that explicitly skip the A-aware route.
        return np.asarray(self._estimator.predict(X))

    def predict_with_A(
        self, X: pd.DataFrame, A: pd.DataFrame
    ) -> np.ndarray:
        """A-aware predict — preferred entry point for the audit runner."""
        if not self._fitted:
            raise RuntimeError(
                "EqOddsPostprocessor.predict_with_A called before fit()"
            )
        base_pred = np.asarray(self._estimator.predict(X)).astype(int)
        if self._eqodds is None:
            # λ=0 path OR failed-LP graceful fallback.
            return base_pred
        try:
            pred_ds = _make_pred_dataset(
                X, base_pred, A, self.sensitive_col
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ds_out = self._eqodds.predict(pred_ds)
            return np.asarray(ds_out.labels).ravel().astype(int)
        except Exception as exc:
            if not self.notes_:
                self.notes_ = (
                    f"eqodds_postproc_predict_failed: {type(exc).__name__}; "
                    f"using base estimator"
                )
            return base_pred

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if not self._fitted:
            raise RuntimeError(
                "EqOddsPostprocessor.predict_proba called before fit()"
            )
        # Hardt 2016 is a hard-label post-processor (LP over thresholds);
        # no native probability output. Return base estimator's proba
        # so consumers (e.g., Tier-5 procedural metrics) still get a
        # usable matrix.
        return _predict_proba_safe(self._estimator, X)

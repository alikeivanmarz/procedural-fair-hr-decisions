"""Pre-processing mitigation wrappers.

Houses the two pre-processing wrappers used by the thesis:

    * :class:`Reweighing` — Kamiran & Calders 2012 sample reweighing.
    * :class:`LFR` — Zemel et al. 2013 Learning Fair Representations.

Both methods wrap AIF360 implementations. ``Reweighing`` is parameter-free
(``lambda_`` ignored); ``LFR`` consumes the canonical ``lambda_`` grid
defined in :data:`procedural_fair_hr.mitigation.base.CANONICAL_HYPERPARAMETER_GRID`.

References
----------

* Kamiran, F. & Calders, T. (2012). "Data preprocessing techniques for
  classification without discrimination." *KDD* (BibTeX:
  ``kamiran2012data``).
* Zemel, R., Wu, Y., Swersky, K., Pitassi, T. & Dwork, C. (2013).
  "Learning Fair Representations." *ICML* (BibTeX: ``zemel2013learning``).
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def _to_aif360_dataset(
    X: pd.DataFrame,
    y: np.ndarray,
    A: pd.DataFrame,
    sensitive_col: str,
):
    """Convert ``(X, y, A)`` into an ``aif360.datasets.BinaryLabelDataset``.

    AIF360's ``BinaryLabelDataset`` is the load-bearing object for every
    AIF360 algorithm in this module (Reweighing / DI Remover /
    LFR). It requires:

        * Float labels (``favorable_label=1.0``, ``unfavorable_label=0.0``).
        * A single ``label_names`` column.
        * One or more ``protected_attribute_names`` columns.

    Single-sensitive-attribute path. Multi-sensitive callers (e.g., Law
    School with ``male`` + ``race``) need a multi-attr extension; for the
    Phase-5 thesis sweep all loaders ship a one-column sensitive frame.

    Returns
    -------
    aif360.datasets.BinaryLabelDataset
        With label ``__y`` and protected attribute ``__sensitive``.

    Notes
    -----
    The AIF360 protected-attribute column must be numeric. Non-numeric
    sensitive values (e.g., Ricci's ``Race ∈ {B, H, W}``) are mapped via
    ``pd.Categorical(...).codes`` to integer codes; the privileged group
    is the largest-code value for stability across runs (the choice is
    arbitrary for binary mitigation methods that are sensitive only to
    the partition, not the label naming — Feldman 2015 §3, Kamiran 2012
    §3).
    """
    from aif360.datasets import BinaryLabelDataset

    df = X.reset_index(drop=True).copy()
    y_arr = np.asarray(y).astype(float)
    df["__y"] = y_arr

    sens_series = A.reset_index(drop=True)[sensitive_col]
    if not pd.api.types.is_numeric_dtype(sens_series):
        sens_codes = pd.Categorical(sens_series).codes.astype(float)
    else:
        sens_codes = sens_series.astype(float).values
    df["__sensitive"] = sens_codes

    return BinaryLabelDataset(
        favorable_label=1.0,
        unfavorable_label=0.0,
        df=df,
        label_names=["__y"],
        protected_attribute_names=["__sensitive"],
    )

def _privileged_groups_from_A(
    A: pd.DataFrame, sensitive_col: str
) -> tuple[list[dict], list[dict]]:
    """Compute (privileged, unprivileged) group dicts for AIF360 algorithms.

    Returns the larger code as privileged (deterministic tiebreak: the
    larger integer wins). Both groups carry the AIF360 column name
    ``__sensitive`` (matching :func:`_to_aif360_dataset`'s convention).
    """
    sens = A.reset_index(drop=True)[sensitive_col]
    if not pd.api.types.is_numeric_dtype(sens):
        codes = pd.Categorical(sens).codes
    else:
        codes = sens.astype(int).values
    unique = sorted(np.unique(codes).tolist())
    # Two-group fallback: privileged = larger code; unprivileged = smaller.
    # For multi-valued sensitive attributes (e.g., Ricci 3 race codes) we
    # collapse to a binary partition ``{>median}`` vs ``{≤median}`` so
    # AIF360's binary-only API (DI Remover / Reweighing / LFR) accepts
    # the input. This is documented in JOURNAL.
    if len(unique) <= 2:
        priv = unique[-1]
        unpriv = unique[0]
    else:
        median = float(np.median(unique))
        priv = next(c for c in reversed(unique) if c > median)
        unpriv = next(c for c in unique if c <= median)
    return (
        [{"__sensitive": int(priv)}],
        [{"__sensitive": int(unpriv)}],
    )

def _accepts_sample_weight(estimator) -> bool:
    """Return True iff ``estimator.fit`` accepts ``sample_weight``.

    Used by Reweighing to detect base estimators that cannot consume
    weighted samples. Sklearn estimators that DO accept it: RF, GB,
    XGB, LR, DecisionTree. Estimators that DON'T: KNN, MLP (sklearn's
    ``MLPClassifier.fit`` does not take ``sample_weight``).
    """
    import inspect

    try:
        sig = inspect.signature(estimator.fit)
    except (TypeError, ValueError):
        return False
    return "sample_weight" in sig.parameters

def _fit_base_estimator(
    base_estimator,
    X,
    y,
    *,
    sample_weight: Optional[np.ndarray] = None,
    random_state: int = 0,
):
    """Clone the base estimator, thread ``random_state``, and fit.

    Returns the fitted clone. If ``sample_weight`` is non-None and the
    estimator does not support ``sample_weight``, raises
    ``NotImplementedError`` with a clear message naming the estimator.
    """
    if base_estimator is None:
        base_estimator = LogisticRegression(max_iter=1000, random_state=random_state)
    est = clone(base_estimator)
    if hasattr(est, "random_state"):
        try:
            est.set_params(random_state=random_state)
        except Exception:
            pass

    if sample_weight is not None:
        if not _accepts_sample_weight(est):
            raise NotImplementedError(
                f"{type(est).__name__}.fit does not accept sample_weight; "
                f"Reweighing requires a sample-weight-aware base estimator. "
                f"Use RF / GB / XGB / LR / DecisionTree (the audit runner "
                f"flags KNN / MLP cells as N/A for Reweighing)."
            )
        est.fit(X, y, sample_weight=sample_weight)
    else:
        est.fit(X, y)
    return est

def _predict_proba_safe(estimator, X) -> Optional[np.ndarray]:
    """Return ``predict_proba(X)`` if available else None."""
    if hasattr(estimator, "predict_proba"):
        try:
            return np.asarray(estimator.predict_proba(X))
        except Exception:
            return None
    return None

# ---------------------------------------------------------------------
# Identity placeholder (kept; required by smoke tests)
# ---------------------------------------------------------------------

@register("identity_preprocessing")
class IdentityPreprocessing(MitigationBase):
    """No-op pre-processing wrapper used by smoke / parallel tests.

    Trains the base estimator on (X, y) untouched. Equivalent to a
    "λ=0 baseline" reference for all pre-processing methods.
    """

    method_name = "identity_preprocessing"
    method_kind = "pre"
    library_origin = "thesis"
    multi_class_native = True  # delegates to base estimator

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        A: pd.DataFrame,
    ) -> "IdentityPreprocessing":
        self._estimator = _fit_base_estimator(
            self.base_estimator, X, y, random_state=self.random_state
        )
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("IdentityPreprocessing.predict called before fit()")
        return np.asarray(self._estimator.predict(X))

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if not self._fitted:
            raise RuntimeError(
                "IdentityPreprocessing.predict_proba called before fit()"
            )
        return _predict_proba_safe(self._estimator, X)

# ---------------------------------------------------------------------
#  — Reweighing (Kamiran & Calders 2012)
# ---------------------------------------------------------------------

@register("reweighing")
class Reweighing(MitigationBase):
    """Kamiran & Calders 2012 sample reweighing (``kamiran2012data``).

    Computes per-sample weights ``w(s, y) = P(s) · P(y) / P(s, y)`` via
    AIF360 (`aif360.algorithms.preprocessing.Reweighing`); these weights
    balance the (sensitive, label) joint distribution. The weights are
    passed to the base estimator's ``fit(X, y, sample_weight=...)``.

    Hyperparameter
    --------------
    Reweighing has no native λ knob (it always fully balances). We
    interpret the canonical λ ∈ {0, 0.05, 0.1, 0.3, 1, 3, 10} as a
    simple gating switch: ``λ == 0`` skips reweighing entirely (identity
    behaviour, byte-identical to base estimator); ``λ > 0`` applies the
    full Kamiran reweighing. This keeps the canonical 7-point grid usable
    for cross-method comparison plots in §6.

    Multi-class
    -----------
    NOT NATIVE — Reweighing assumes binary y. Wrap with the project's
    OvR adapter for |𝒞|>2. The base ``fit``
    raises ``NotImplementedError`` on multi-class y so the audit runner
    can route through the OvR wrapper instead.

    Sample-weight passthrough
    -------------------------
    Sklearn estimators that accept ``sample_weight``: RF, GB, XGB, LR,
    DecisionTree. Estimators that DON'T: KNN, MLPClassifier. For
    non-supporting bases the wrapper raises ``NotImplementedError`` with
    a clear message at fit time; the audit runner records these cells as
    ``N/A — base estimator does not accept sample_weight`` in ``notes``.
    """

    method_name = "reweighing"
    method_kind = "pre"
    library_origin = "aif360"
    multi_class_native = False

    def fit(
        self, X: pd.DataFrame, y: np.ndarray, A: pd.DataFrame
    ) -> "Reweighing":
        self._check_multiclass(y)

        if self.lambda_ == 0.0:
            # λ=0 → identity (no reweighing). Byte-identical to base.
            sample_weight = None
        else:
            from aif360.algorithms.preprocessing import (
                Reweighing as _AIF360Reweighing,
            )

            ds = _to_aif360_dataset(X, y, A, self.sensitive_col)
            priv, unpriv = _privileged_groups_from_A(A, self.sensitive_col)
            rw = _AIF360Reweighing(
                unprivileged_groups=unpriv, privileged_groups=priv
            )
            ds_transformed = rw.fit_transform(ds)
            sample_weight = np.asarray(ds_transformed.instance_weights).astype(float)

        self._estimator = _fit_base_estimator(
            self.base_estimator,
            X,
            y,
            sample_weight=sample_weight,
            random_state=self.random_state,
        )
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Reweighing.predict called before fit()")
        return np.asarray(self._estimator.predict(X))

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if not self._fitted:
            raise RuntimeError("Reweighing.predict_proba called before fit()")
        return _predict_proba_safe(self._estimator, X)

# ---------------------------------------------------------------------
#  — Optimised Pre-processing (Calmon 2017)
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
#  — Learning Fair Representations (Zemel 2013)
# ---------------------------------------------------------------------

@register("lfr")
class LFR(MitigationBase):
    """Zemel 2013 Learning Fair Representations (``zemel2013learning``).

    Learns a fair representation ``Z`` such that
    ``P(Z | s = 0) ≈ P(Z | s = 1)`` while preserving accuracy in
    ``Z → y``. Uses k prototype vectors + adversarial training.
    Implementation delegates to ``aif360.algorithms.preprocessing.LFR``.

    Hyperparameters
    ---------------
    LFR has 4 hyperparameters per Zemel 2013 Eq. 1:

        * ``Az`` — adversarial-fairness weight (mapped from our λ).
        * ``Ay`` — utility weight (fixed = 1.0).
        * ``Ax`` — reconstruction weight (fixed = 0.1).
        * ``k`` — # prototypes (fixed = 5).

    We map λ ∈ {0, 0.05, 0.1, 0.3, 1, 3, 10} → ``Az`` directly
    (no clipping; LFR's loss accepts arbitrary positive weights). λ=0 →
    adversarial weight zero, essentially identity (within reconstruction
    noise from the autoencoder).

    Multi-class
    -----------
    NOT NATIVE (Zemel 2013 binary-only); the audit runner auto-wraps
    with OvR per  / .

    Convergence + failure mode
    --------------------------
    LFR's adversarial optimisation can fail to converge or return NaN
    representations on small / degenerate datasets. If ``LFR.fit``
    raises or returns NaN, we fall back to the unmitigated base
    estimator with a notes annotation (graceful degradation per
    ).
    """

    method_name = "lfr"
    method_kind = "pre"
    library_origin = "aif360"
    multi_class_native = False

    def fit(
        self, X: pd.DataFrame, y: np.ndarray, A: pd.DataFrame
    ) -> "LFR":
        self._check_multiclass(y)
        self.notes_: str = ""

        Az = float(self.lambda_)
        # Cache A for predict-time transform.
        self._A_train = A.reset_index(drop=True).copy()

        if Az <= 0.0:
            X_used, y_used = X, y
            self._lfr = None
        else:
            try:
                from aif360.algorithms.preprocessing import LFR as _AIF360LFR

                ds = _to_aif360_dataset(X, y, A, self.sensitive_col)
                priv, unpriv = _privileged_groups_from_A(A, self.sensitive_col)
                lfr = _AIF360LFR(
                    unprivileged_groups=unpriv,
                    privileged_groups=priv,
                    k=5,
                    Ax=0.1,
                    Ay=1.0,
                    Az=Az,
                    verbose=0,
                    seed=self.random_state,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    lfr.fit(ds, maxiter=2000, maxfun=2000)
                    ds_rep = lfr.transform(ds, threshold=0.5)

                feats = np.asarray(ds_rep.features)
                if not np.all(np.isfinite(feats)):
                    raise RuntimeError("LFR produced non-finite features")

                df_rep = pd.DataFrame(feats, columns=ds_rep.feature_names)
                X_rep = pd.DataFrame(index=X.index, columns=X.columns)
                for col in X.columns:
                    if col in df_rep.columns:
                        X_rep[col] = df_rep[col].values
                    else:
                        X_rep[col] = X[col].values
                X_used = X_rep.astype(float, errors="ignore")
                # LFR's transform also rewrites y; honour it.
                y_used = np.asarray(ds_rep.labels).ravel().astype(int)
                self._lfr = lfr
            except Exception as exc:
                logger.warning(
                    "LFR adversarial training failed (%s); falling back to "
                    "unmitigated base estimator.",
                    exc,
                )
                self.notes_ = f"lfr_failed: {type(exc).__name__}"
                X_used, y_used = X, y
                self._lfr = None

        self._estimator = _fit_base_estimator(
            self.base_estimator,
            X_used,
            y_used,
            random_state=self.random_state,
        )
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LFR.predict called before fit()")
        return np.asarray(self._estimator.predict(X))

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if not self._fitted:
            raise RuntimeError("LFR.predict_proba called before fit()")
        return _predict_proba_safe(self._estimator, X)

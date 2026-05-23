"""In-processing mitigation wrapper.

Houses the in-processing wrapper used by the thesis:

    * :class:`AdversarialDebiasing` — Zhang, Lemoine & Mitchell 2018,
      reimplemented in PyTorch.

The PyTorch implementation lives in
:mod:`procedural_fair_hr.mitigation._adv_debias_pytorch`; this module
adapts it to the :class:`MitigationBase` constructor / fit / predict /
predict_proba contract.

Reference
---------

Zhang, B. H., Lemoine, B. & Mitchell, M. (2018). "Mitigating Unwanted
Biases with Adversarial Learning." *AAAI/ACM Conference on AI, Ethics,
and Society* (BibTeX: ``zhang2018mitigating``).
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from ._adv_debias_pytorch import AdversarialDebiasingPyTorch
from .base import MitigationBase, register
from .preprocessing import (
    _fit_base_estimator,
    _predict_proba_safe,
    _privileged_groups_from_A,
    _to_aif360_dataset,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
#  — Adversarial Debiasing (PyTorch reimpl, )
# ---------------------------------------------------------------------

@register("adversarial_debiasing")
class AdversarialDebiasing(MitigationBase):
    """Zhang, Lemoine & Mitchell 2018 — PyTorch reimpl .

    See ``zhang2018mitigating``. Trains a predictor + adversary alternately
    via the gradient-projection trick (ZLM 2018 Eq. 4) on a weighted-sum
    loss ``predictor_loss − λ · adversary_loss``. The ``lambda_`` argument
    controls the fairness-vs-utility trade-off directly (matches the
    canonical λ grid in  without remapping); ``λ=0`` recovers a
    vanilla 2-layer MLP classifier.

    The underlying PyTorch implementation lives in
    :mod:`procedural_fair_hr.mitigation._adv_debias_pytorch`. This wrapper is just
    adapter glue from the canonical ``(X, y, A)`` interface to the
    PyTorch class; see that module for the training-loop details and
    determinism guarantees.

    Multi-class
    -----------
    NATIVE — the predictor head is ``MLP(input → 64 → n_classes)`` so
    ``predict_proba`` returns a proper softmax over n_classes.

    Sensitive-attribute binarisation (DISCLOSURE per )
    ---------------------------------------------------------
    Zhang 2018's adversary head is BCE-with-logits — i.e., the canonical
    ZLM 2018 formulation predicts a SINGLE binary sensitive value, not a
    multi-valued one. On datasets whose sensitive attribute is
    multi-valued or non-numeric (e.g., D4 Ricci ``Race ∈ {W, B, H}``;
    D2 ACS ``RAC1P ∈ {1..9}``), this wrapper collapses sensitive to
    binary via ``pd.Categorical(...).codes > median`` (consistent with
    the Tier-2 helper ``_privileged_groups_from_A``).

    **Implication:** AdvDebias's fairness claims on multi-valued
    sensitive datasets are about a **median-split binarisation** of the
    protected attribute, NOT about the original multi-group structure.
    For Ricci this means the comparison is essentially "alphabetically-
    earlier race code vs the rest" rather than a per-race fairness claim.

    This is faithful to ZLM 2018 (which is binary-only); a multi-class
    adversary head (cross-entropy over n_sensitive_classes) would be a
    research extension to the canonical paper and is genuinely out of
    thesis scope per  (PyTorch reimplementation = canonical ZLM
    2018, not an extension). Disclosed per  and surfaced in the
     phase report's methodological-caveats list.

    ``base_estimator`` argument
    ---------------------------
    Adversarial Debiasing REPLACES the base estimator (it is itself a
    classifier with its own MLP predictor). The ``base_estimator``
    constructor argument is therefore IGNORED; we accept it for
    conformance but emit a ``UserWarning`` at fit time when a non-None
    base is passed (so the audit runner's logs surface the dropped base).
    """

    method_name = "adversarial_debiasing"
    method_kind = "in"
    library_origin = "pytorch"  # custom PyTorch reimpl per
    multi_class_native = True

    def __init__(
        self,
        base_estimator=None,
        lambda_: float = 1.0,
        sensitive_col: str = "sensitive",
        random_state: int = 0,
        *,
        n_epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            base_estimator=base_estimator,
            lambda_=lambda_,
            sensitive_col=sensitive_col,
            random_state=random_state,
            **kwargs,
        )
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.device = device  # resolved by underlying class

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        A: pd.DataFrame,
    ) -> "AdversarialDebiasing":
        # Tier-3 contract: warn (don't raise) on non-None base_estimator.
        # AdvDebias replaces the base classifier with its own MLP per
        # ;  conformance is preserved (we accept the kwarg)
        # but the user gets a clear log line.
        if self.base_estimator is not None:
            warnings.warn(
                f"AdversarialDebiasing replaces the base estimator with its "
                f"own PyTorch MLP; the supplied "
                f"base_estimator={type(self.base_estimator).__name__!r} is "
                f"ignored. Pass base_estimator=None to silence this warning.",
                UserWarning,
                stacklevel=2,
            )

        y = np.asarray(y).astype(int)
        n_classes = int(np.unique(y).size)
        n_features = int(X.shape[1])
        sens = (
            A[self.sensitive_col]
            if self.sensitive_col in A.columns
            else A.iloc[:, 0]
        )
        # Sensitive may be string (e.g., Ricci's Race ∈ {B, H, W}). The
        # PyTorch adversary's BCE-with-logits takes binary 0/1 input; for
        # multi-valued non-numeric sensitive we collapse to a binary
        # partition via pd.Categorical(...).codes > median (mirrors the
        # Tier-2 ``_privileged_groups_from_A`` convention).
        if not pd.api.types.is_numeric_dtype(sens):
            codes = pd.Categorical(sens).codes
            unique = sorted(np.unique(codes).tolist())
            if len(unique) <= 2:
                sens_arr = codes.astype(float)
            else:
                median = float(np.median(unique))
                sens_arr = (codes > median).astype(float)
        else:
            sens_vals = np.asarray(sens).astype(float)
            unique_v = np.unique(sens_vals)
            if len(unique_v) > 2:
                median = float(np.median(unique_v))
                sens_arr = (sens_vals > median).astype(float)
            else:
                sens_arr = sens_vals

        self._impl = AdversarialDebiasingPyTorch(
            n_classes=n_classes,
            n_features=n_features,
            lambda_=self.lambda_,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            device=self.device,
            random_state=self.random_state,
        )
        # Cast X to a numeric float32 ndarray for PyTorch. Callers that
        # pass non-numeric columns (e.g., raw object dtype) will get a
        # clear pandas error here; the audit runner's _preprocess_xy
        # ordinal-encodes everything before this fit() is reached.
        X_arr = np.asarray(X.values, dtype=np.float32)
        self._impl.fit(X_arr, y, sens_arr)
        self._fitted = True
        return self

    @staticmethod
    def _coerce_X(X) -> np.ndarray:
        """Accept DataFrame, ndarray, or list of arrays.

        SHAP's KernelExplainer and the procedural-fairness metrics call
        ``predict_proba`` with already-encoded numpy arrays; the audit
        runner's _preprocess_xy yields DataFrames. Both paths must work
        per  (fix, don't punt to a graceful-fallback NaN row).
        """
        if hasattr(X, "values"):
            return np.asarray(X.values, dtype=np.float32)
        return np.asarray(X, dtype=np.float32)

    def predict(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("AdversarialDebiasing.predict called before fit()")
        return self._impl.predict(self._coerce_X(X))

    def predict_proba(self, X) -> Optional[np.ndarray]:
        if not self._fitted:
            raise RuntimeError(
                "AdversarialDebiasing.predict_proba called before fit()"
            )
        return self._impl.predict_proba(self._coerce_X(X))

# Convenience alias under the short name used by some Tier-3 dispatch
# documentation. Both names
# resolve to the same class so callers can use either.
MITIGATION_REGISTRY_ALIAS_NAME = "adv_debias"
register(MITIGATION_REGISTRY_ALIAS_NAME)(AdversarialDebiasing)

"""Binary group-fairness metrics for the employee-performance fairness audit.

Implements 8 binary group-fairness metrics (Eqs. 1–9) from:
    Reference: pagano2023fairness
    Pagano, T. P., Loureiro, R. B., Lisboa, F. V. N., Peixoto, R. M.,
    Oliveira, G. S., Cruz, G. O. S., Araujo, M. M., Santos, L. L., Cruz,
    M. A. S., Serrano, A. M. L., Soares, E., & Winkler, I. (2023).
    Bias and Unfairness in Machine Learning Models: A Systematic Review on
    Datasets, Tools, Fairness Metrics, and Identification and Mitigation
    Methods. *Big Data and Cognitive Computing*, 7(1), 15.
    DOI pending — .

Signature contract:
    Every public function accepts (y_true, y_pred, sensitive) in that order.
    Binary metrics return float. Multi-class metrics () return
    tuple(float, dict[int, float]).

Term references ():
    - Sensitive attribute (S):  attribute (S)
    - Group g, protected group s:  g, protected group s, non-protected group s̄
    - Demographic Parity (DP):  Parity (DP) — binary
    - Equalised Odds (EOdds):  Odds (EOdds) — binary
    - Equal Opportunity (EOO):  Opportunity (EOO) — binary
    - Disparate Impact (DI):  Impact (DI)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Union

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_array(x: Union[np.ndarray, list, pd.Series]) -> np.ndarray:
    """Convert array-like to a 1-D numpy integer array."""
    return np.asarray(x, dtype=int).ravel()

def _privileged_mask(sensitive: pd.Series) -> np.ndarray:
    """Return a boolean mask where True = privileged (majority) group.

    The privileged group is defined as the mode of ``sensitive``
    (see  g, protected group s, non-protected group s̄).

    Args:
        sensitive: A ``pandas.Series`` of group labels
            (see  attribute (S)).

    Returns:
        Boolean numpy array, True for the majority (privileged) group.
    """
    priv_label = sensitive.mode().iloc[0]
    return (sensitive == priv_label).to_numpy(dtype=bool)

def _positive_rate(y_pred: np.ndarray, mask: np.ndarray) -> float:
    """P(ŷ=1 | group defined by mask)."""
    group_preds = y_pred[mask]
    if len(group_preds) == 0:
        return float("nan")
    return float(group_preds.mean())

def _tpr(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    """True-positive rate = P(ŷ=1 | y=1, group)."""
    pos_mask = mask & (y_true == 1)
    if pos_mask.sum() == 0:
        return float("nan")
    return float(y_pred[pos_mask].mean())

def _fpr(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    """False-positive rate = P(ŷ=1 | y=0, group)."""
    neg_mask = mask & (y_true == 0)
    if neg_mask.sum() == 0:
        return float("nan")
    return float(y_pred[neg_mask].mean())

def _accuracy(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    """Accuracy for the group defined by mask."""
    group_true = y_true[mask]
    group_pred = y_pred[mask]
    if len(group_true) == 0:
        return float("nan")
    return float((group_true == group_pred).mean())

# ---------------------------------------------------------------------------
# Public API — 8 binary group-fairness metrics
# ---------------------------------------------------------------------------

def demographic_parity_difference(
    y_true,
    y_pred,
    sensitive: pd.Series,
) -> float:
    """Demographic Parity Difference (DPD / SPD).

    DPD = P(ŷ=1|S=privileged) − P(ŷ=1|S=unprivileged).
    Positive values favour the privileged (majority) group.

    Reference: pagano2023fairness Eq. 1.
    See also:  Parity (DP) — binary.

    Args:
        y_true: Ground-truth binary labels (array-like, ignored in this metric
            but required by the  signature).
        y_pred: Predicted binary labels (array-like).
        sensitive: Group membership labels
            (see  attribute (S)).
            The privileged group is the mode of this Series.

    Returns:
        float: P(ŷ=1|privileged) − P(ŷ=1|unprivileged).
    """
    y_pred_arr = _to_array(y_pred)
    priv = _privileged_mask(sensitive)
    unpriv = ~priv
    return float(_positive_rate(y_pred_arr, priv) - _positive_rate(y_pred_arr, unpriv))

def disparate_impact_ratio(
    y_true,
    y_pred,
    sensitive: pd.Series,
) -> float:
    """Disparate Impact Ratio (DI).

    DI = P(ŷ=1|S=unprivileged) / P(ŷ=1|S=privileged).
    The legal four-fifths rule requires DI ≥ 0.8.
    Returns 0.0 when the privileged-group positive rate is zero.

    Reference: pagano2023fairness Eq. 2.
    See also:  Impact (DI).

    Args:
        y_true: Ground-truth binary labels (array-like, unused).
        y_pred: Predicted binary labels (array-like).
        sensitive: Group membership labels
            (see  attribute (S)).

    Returns:
        float: DI ratio ∈ [0, ∞).
    """
    y_pred_arr = _to_array(y_pred)
    priv = _privileged_mask(sensitive)
    unpriv = ~priv
    rate_priv = _positive_rate(y_pred_arr, priv)
    rate_unpriv = _positive_rate(y_pred_arr, unpriv)
    if rate_priv == 0.0:
        return 0.0
    return float(rate_unpriv / rate_priv)

def equal_opportunity_difference(
    y_true,
    y_pred,
    sensitive: pd.Series,
) -> float:
    """Equal Opportunity Difference (EOO).

    EOO = TPR(privileged) − TPR(unprivileged)
        = P(ŷ=1|y=1, S=priv) − P(ŷ=1|y=1, S=unpriv).
    Positive values indicate higher recall for the privileged group.

    Reference: pagano2023fairness Eq. 3.
    See also:  Opportunity (EOO) — binary.

    Args:
        y_true: Ground-truth binary labels (array-like).
        y_pred: Predicted binary labels (array-like).
        sensitive: Group membership labels
            (see  attribute (S)).

    Returns:
        float: TPR difference (privileged − unprivileged).
    """
    y_true_arr = _to_array(y_true)
    y_pred_arr = _to_array(y_pred)
    priv = _privileged_mask(sensitive)
    unpriv = ~priv
    return float(_tpr(y_true_arr, y_pred_arr, priv) - _tpr(y_true_arr, y_pred_arr, unpriv))

def equalised_odds_difference(
    y_true,
    y_pred,
    sensitive: pd.Series,
) -> float:
    """Equalised Odds Difference (EOdds).

    EOdds = max(|TPR gap|, |FPR gap|).
    Non-negative; zero indicates perfect equalised odds.

    Reference: pagano2023fairness Eq. 4.
    See also:  Odds (EOdds) — binary.

    Args:
        y_true: Ground-truth binary labels (array-like).
        y_pred: Predicted binary labels (array-like).
        sensitive: Group membership labels
            (see  attribute (S)).

    Returns:
        float: max(|TPR_priv − TPR_unpriv|, |FPR_priv − FPR_unpriv|).
    """
    y_true_arr = _to_array(y_true)
    y_pred_arr = _to_array(y_pred)
    priv = _privileged_mask(sensitive)
    unpriv = ~priv
    tpr_gap = abs(_tpr(y_true_arr, y_pred_arr, priv) - _tpr(y_true_arr, y_pred_arr, unpriv))
    fpr_gap = abs(_fpr(y_true_arr, y_pred_arr, priv) - _fpr(y_true_arr, y_pred_arr, unpriv))
    # handle NaN (e.g. no negatives in dataset)
    values = [v for v in (tpr_gap, fpr_gap) if not np.isnan(v)]
    if not values:
        return float("nan")
    return float(max(values))

def statistical_parity_difference(
    y_true,
    y_pred,
    sensitive: pd.Series,
) -> float:
    """Statistical Parity Difference (SPD) — alias for demographic_parity_difference.

    Different literature uses different names for the same quantity.
    This function delegates to ``demographic_parity_difference``.

    Reference: pagano2023fairness Eq. 5.
    See also:  Parity (DP) — binary.

    Args:
        y_true: Ground-truth binary labels (array-like, unused).
        y_pred: Predicted binary labels (array-like).
        sensitive: Group membership labels
            (see  attribute (S)).

    Returns:
        float: P(ŷ=1|privileged) − P(ŷ=1|unprivileged).
    """
    return demographic_parity_difference(y_true, y_pred, sensitive)

def average_absolute_odds_difference(
    y_true,
    y_pred,
    sensitive: pd.Series,
) -> float:
    """Average Absolute Odds Difference (AAOD).

    AAOD = 0.5 * (|FPR gap| + |TPR gap|).
    Non-negative; zero indicates perfect equalised odds.

    Reference: pagano2023fairness Eq. 6.
    See also:  Odds (EOdds) — binary.

    Args:
        y_true: Ground-truth binary labels (array-like).
        y_pred: Predicted binary labels (array-like).
        sensitive: Group membership labels
            (see  attribute (S)).

    Returns:
        float: 0.5 * (|FPR_priv − FPR_unpriv| + |TPR_priv − TPR_unpriv|).
    """
    y_true_arr = _to_array(y_true)
    y_pred_arr = _to_array(y_pred)
    priv = _privileged_mask(sensitive)
    unpriv = ~priv
    tpr_gap = abs(_tpr(y_true_arr, y_pred_arr, priv) - _tpr(y_true_arr, y_pred_arr, unpriv))
    fpr_gap = abs(_fpr(y_true_arr, y_pred_arr, priv) - _fpr(y_true_arr, y_pred_arr, unpriv))
    # if either is NaN (e.g. all-positive dataset), use only the defined component
    components = [v for v in (tpr_gap, fpr_gap) if not np.isnan(v)]
    if not components:
        return float("nan")
    return float(0.5 * sum(components)) if len(components) == 2 else float(components[0])

def average_equalised_odds_difference(
    y_true,
    y_pred,
    sensitive: pd.Series,
) -> float:
    """Average Equalised Odds Difference (AEORD).

    AEORD = 0.5 * (EOO + FPR_diff).
    May be negative when direction matters.

    Reference: pagano2023fairness Eq. 7.
    See also:  Odds (EOdds) — binary.

    Args:
        y_true: Ground-truth binary labels (array-like).
        y_pred: Predicted binary labels (array-like).
        sensitive: Group membership labels
            (see  attribute (S)).

    Returns:
        float: 0.5 * ((TPR_priv − TPR_unpriv) + (FPR_priv − FPR_unpriv)).
    """
    y_true_arr = _to_array(y_true)
    y_pred_arr = _to_array(y_pred)
    priv = _privileged_mask(sensitive)
    unpriv = ~priv
    tpr_diff = _tpr(y_true_arr, y_pred_arr, priv) - _tpr(y_true_arr, y_pred_arr, unpriv)
    fpr_diff = _fpr(y_true_arr, y_pred_arr, priv) - _fpr(y_true_arr, y_pred_arr, unpriv)
    components = [v for v in (tpr_diff, fpr_diff) if not np.isnan(v)]
    if not components:
        return float("nan")
    return float(0.5 * sum(components)) if len(components) == 2 else float(components[0])

def accuracy_balance(
    y_true,
    y_pred,
    sensitive: pd.Series,
) -> float:
    """Accuracy Balance / Accuracy Difference (ABAD).

    ABAD = |Acc(S=privileged) − Acc(S=unprivileged)|.
    Non-negative; zero means both groups have equal accuracy.

    Reference: pagano2023fairness Eq. 8.

    Args:
        y_true: Ground-truth binary labels (array-like).
        y_pred: Predicted binary labels (array-like).
        sensitive: Group membership labels
            (see  attribute (S)).

    Returns:
        float: |Acc_priv − Acc_unpriv|.
    """
    y_true_arr = _to_array(y_true)
    y_pred_arr = _to_array(y_pred)
    priv = _privileged_mask(sensitive)
    unpriv = ~priv
    acc_priv = _accuracy(y_true_arr, y_pred_arr, priv)
    acc_unpriv = _accuracy(y_true_arr, y_pred_arr, unpriv)
    return float(abs(acc_priv - acc_unpriv))

# ---------------------------------------------------------------------------
# Individual fairness metrics
# ---------------------------------------------------------------------------

def knn_consistency(
    y_true,
    y_pred: np.ndarray,
    sensitive: pd.Series,
    X: pd.DataFrame,
    k: int = 5,
) -> float:
    """KNN Consistency (KNNC): fraction of instances whose prediction matches
    the majority prediction among their k nearest neighbours in feature space.

    1.0 = perfect individual fairness (all neighbours predict the same).
    Reference: pagano2023fairness.

    Args:
        y_true: Unused; kept for  compatibility.
        y_pred: Predicted labels (array-like).
        sensitive: Group labels (unused directly; included for API consistency).
        X: Feature matrix used to compute pairwise distances.
        k: Number of nearest neighbours.
    """
    from sklearn.preprocessing import StandardScaler

    y_pred_arr = np.asarray(y_pred).ravel()
    num_cols = X.select_dtypes(include=[np.number]).columns
    X_num = X[num_cols].fillna(0.0).values.astype(float)
    if X_num.shape[1] > 0:
        X_scaled = StandardScaler().fit_transform(X_num)
    else:
        X_scaled = np.zeros((len(X), 1))

    n = len(X_scaled)
    consistent = 0
    for i in range(n):
        dists = np.linalg.norm(X_scaled - X_scaled[i], axis=1)
        dists[i] = np.inf
        nn_idx = np.argpartition(dists, min(k, n - 1))[:k]
        majority = int(np.bincount(y_pred_arr[nn_idx]).argmax())
        if y_pred_arr[i] == majority:
            consistent += 1
    return float(consistent / n)

def lipschitz_fairness(
    y_pred: np.ndarray,
    sensitive: pd.Series,
    X: pd.DataFrame,
    epsilon: float = 0.01,
    n_sample_pairs: int = 1000,
) -> float:
    """Lipschitz Fairness Through Awareness (Dwork et al. 2012): fraction of
    sampled pairs (i, j) satisfying |ŷ_i - ŷ_j| ≤ d(x_i, x_j).

    Uses L=1 Lipschitz constant and Euclidean distance on z-scored numeric X.
    For n > 100, samples ``n_sample_pairs`` pairs randomly for scalability.
    Reference: dwork2012fairness.

    Args:
        y_pred: Predicted labels (array-like of int or float).
        sensitive: Unused; included for API consistency.
        X: Feature matrix.
        epsilon: Not used in this implementation; retained for API.
        n_sample_pairs: Max random pairs to evaluate.
    """
    from sklearn.preprocessing import StandardScaler

    y_arr = np.asarray(y_pred, dtype=float).ravel()
    num_cols = X.select_dtypes(include=[np.number]).columns
    X_num = X[num_cols].fillna(0.0).values.astype(float)
    if X_num.shape[1] > 0:
        X_scaled = StandardScaler().fit_transform(X_num)
    else:
        X_scaled = np.zeros((len(X), 1))

    n = len(X_scaled)
    rng = np.random.default_rng(0)
    if n * (n - 1) // 2 > n_sample_pairs:
        idx_i = rng.integers(0, n, n_sample_pairs)
        idx_j = rng.integers(0, n, n_sample_pairs)
        mask = idx_i != idx_j
        idx_i, idx_j = idx_i[mask], idx_j[mask]
    else:
        idx_i, idx_j = np.triu_indices(n, k=1)

    dists = np.linalg.norm(X_scaled[idx_i] - X_scaled[idx_j], axis=1)
    pred_diffs = np.abs(y_arr[idx_i] - y_arr[idx_j])
    satisfied = (pred_diffs <= dists).mean()
    return float(satisfied)

# ---------------------------------------------------------------------------
# Counterfactual fairness metric
# ---------------------------------------------------------------------------

def _ordinal_encode_df(X: pd.DataFrame) -> np.ndarray:
    """Encode all object columns with ordinal integers; leave numerics as-is."""
    from sklearn.preprocessing import OrdinalEncoder
    X_out = X.copy()
    obj_cols = X_out.select_dtypes(include="object").columns
    if len(obj_cols) > 0:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_out[obj_cols] = enc.fit_transform(X_out[obj_cols])
    return X_out.values.astype(float)

def counterfactual_fairness(
    y_true,
    y_pred: np.ndarray,
    sensitive: pd.Series,
    X: pd.DataFrame,
    model,
    sensitive_col: str,
    n_cf: int = 50,
) -> float:
    """Level-1 Counterfactual Fairness (Kusner et al. 2017 §3.1): fraction of
    instances whose prediction is unchanged when the sensitive attribute is
    flipped to a different value, all else equal.

    1.0 = perfectly counterfactually fair (flipping sensitive col has no effect).
    Reference: kusner2017counterfactual.

    Args:
        y_true: Unused; kept for  compatibility.
        y_pred: Original predictions (unused; model is called directly).
        sensitive: Sensitive attribute Series (unused directly).
        X: Feature DataFrame containing ``sensitive_col`` as one column.
        model: Fitted sklearn-compatible classifier with a ``predict`` method.
        sensitive_col: Name of the sensitive column in ``X``.
        n_cf: Max instances to evaluate (random sample for speed).
    """
    rng = np.random.default_rng(0)
    n = len(X)
    idx = rng.choice(n, size=min(n_cf, n), replace=False)
    X_sample = X.iloc[idx].copy().reset_index(drop=True)

    unique_vals = X_sample[sensitive_col].unique().tolist()
    if len(unique_vals) < 2:
        return 1.0  # only one group value — trivially fair

    X_enc = _ordinal_encode_df(X_sample)
    orig_preds = model.predict(X_enc)

    # For each instance, flip to a different sensitive value
    X_cf = X_sample.copy()
    for i in range(len(X_cf)):
        current = X_cf.at[i, sensitive_col]
        others = [v for v in unique_vals if v != current]
        X_cf.at[i, sensitive_col] = rng.choice(others)

    X_cf_enc = _ordinal_encode_df(X_cf)
    cf_preds = model.predict(X_cf_enc)

    return float((orig_preds == cf_preds).mean())

# ---------------------------------------------------------------------------
# Multi-class fairness extensions (..)
# ---------------------------------------------------------------------------

# Formal extensions of the binary group-fairness metrics to multi-class
# targets. All four functions return a tuple ``(macro_value, per_class)``
# per 's documented multi-class signature.

# Binary-restriction equivalence (verified by unit tests + ):

# * ``macro_dp`` and ``multinomial_counterfactual_fairness`` reduce to
#   their binary counterparts (``demographic_parity_difference`` and
#   ``counterfactual_fairness``) as exact algebraic identities for
#   ``|𝒞| = 2``.

# * ``macro_eo`` and ``macro_eodds`` reduce to ``equal_opportunity_
#   difference`` and ``equalised_odds_difference`` ONLY when the binary
#   confusion matrix is symmetric across classes — i.e., when
#   ``|TPR_diff| == |FPR_diff|`` (Macro-EOdds) or ``|TPR_diff| ==
#   |TNR_diff|`` (Macro-EO). When errors are asymmetric the macro and
#   binary forms diverge: the binary form takes a max / classical sum
#   over the two errors while the macro form averages per-class TPRs,
#   which mixes both error directions through the negation term. This
#   conditionality is empirically demonstrated by
#   ``test_macro_*_asymmetric_non_equivalence`` in
#   ``tests/test_fairness_metrics.py`` and is the standard equivalence
#   check used in the macro-averaged-fairness literature
#   (Pagano et al. 2023 §5; Putzel & Lee 2022). See GLOSSARY entries
#   for ``Macro-EO`` and ``Macro-EOdds``.

# Filtered macro (Phase-3 followup, post-review concern Issue 1):
# every multi-class group-metric (`macro_dp`, `macro_eo`, `macro_eodds`)
# accepts an optional ``filter_classes: set[int] | None`` parameter.
# When provided, the macro mean is taken only over classes NOT in the
# filter set. This is used by the audit script to emit a parallel
# "filtered-macro" row that excludes per-class rows whose
# `non_vacuous_tpr` flag is False (e.g., OULAD's degenerate class-2
# Distinction), so a reader sees both the full macro AND the macro
# with degenerate classes excluded. When every class is filtered, the
# macro returns NaN (documented per-function).

def macro_dp(
    y_true,
    y_pred,
    sensitive: pd.Series,
    *,
    filter_classes: set[int] | None = None,
) -> tuple[float, dict[int, float]]:
    """Macro Demographic Parity (Macro-DP) — multi-class extension.

    For each class ``c ∈ 𝒞``::

        DP_c = max_g P(ŷ=c | S=g) − min_g P(ŷ=c | S=g)

    and ``macro_dp`` is the unweighted mean of ``DP_c`` over classes.

    Generalises Dwork et al. 2012 / Pagano et al. 2023 Eq. 1 from the
    binary positive-class form to a per-class rate-difference form.

    Binary-restriction equivalence: when ``|𝒞| = 2``, ``macro_dp`` equals
    ``|demographic_parity_difference|`` as an exact algebraic identity
    (since ``P(ŷ=0|S=g) = 1 − P(ŷ=1|S=g)``).

    Reference: pagano2023fairness §Eq. 1 (binary form);
    dwork2012fairness (DP-FtA origin); see
    Parity — multi-class (Macro-DP).

    Args:
        y_true: Ground-truth labels (array-like; unused — DP is a
            prediction-rate parity metric, kept for  signature).
        y_pred: Predicted class labels (array-like of int).
        sensitive: Group membership labels
            (see  attribute (S)).
        filter_classes: Optional set of class indices to EXCLUDE from
            the macro mean. Per-class entries are still returned in the
            ``per_class`` dict for transparency, but they do not
            contribute to the macro value. This is the "filtered macro"
            convention used by the Phase-2 audit to drop degenerate
            classes (e.g., OULAD's class-2 Distinction with
            ``non_vacuous_tpr == False``) from the macro average. If
            ``filter_classes`` excludes every observed class, the macro
            value is ``NaN`` (the per-class dict is still complete).

    Returns:
        Tuple ``(macro_value, per_class)`` where ``macro_value`` is a
        ``float`` and ``per_class`` is a ``dict[int, float]`` mapping
        each observed class index ``c`` to its ``DP_c`` value.
    """
    y_pred_arr = np.asarray(y_pred).ravel()
    sens_arr = np.asarray(sensitive)

    classes = np.unique(np.concatenate([y_pred_arr, np.asarray(y_true).ravel()]))
    classes = np.sort(classes)
    groups = np.unique(sens_arr)

    per_class: dict[int, float] = {}
    for c in classes:
        rates = []
        for g in groups:
            mask = sens_arr == g
            if mask.sum() == 0:
                continue
            rate = float((y_pred_arr[mask] == c).mean())
            rates.append(rate)
        if not rates:
            per_class[int(c)] = float("nan")
        else:
            per_class[int(c)] = float(max(rates) - min(rates))

    excluded = set(filter_classes) if filter_classes else set()
    finite = [
        v for c, v in per_class.items()
        if not np.isnan(v) and int(c) not in excluded
    ]
    macro = float(np.mean(finite)) if finite else float("nan")
    return macro, per_class

def macro_eo(
    y_true,
    y_pred,
    sensitive: pd.Series,
    *,
    filter_classes: set[int] | None = None,
) -> tuple[float, dict[int, float]]:
    """Macro Equal Opportunity (Macro-EO) — multi-class extension.

    For each class ``c ∈ 𝒞``::

        EO_c = max_g P(ŷ=c | y=c, S=g) − min_g P(ŷ=c | y=c, S=g)

    and ``macro_eo`` is the unweighted mean of ``EO_c`` over classes
    that have at least one ground-truth instance in every group.

    EO is the class-conditional true-positive rate (recall) difference;
    for binary, EO on the positive class is the classical Hardt 2016
    equal-opportunity metric.

    **Binary-restriction equivalence is CONDITIONAL.** The reduction
    ``macro_eo == |equal_opportunity_difference|`` holds only when the
    binary errors are symmetric — specifically when
    ``|TPR_diff| == |TNR_diff|`` — because the binary EOO uses the
    positive class only while the macro form averages the per-class
    TPR-difference over both classes (``c=0`` contributes a TNR-difference
    via the ``y=0`` conditioning). When errors are asymmetric, the macro
    value diverges from the binary form by up to half the asymmetry.
    The unconditional reduction is to ``0.5 * (|TPR_diff| + |TNR_diff|)``,
    not to ``|equal_opportunity_difference|``. This is documented
    empirically by ``test_macro_eo_asymmetric_non_equivalence``.

    Reference: hardt2016equality (binary form generalised); see
     Opportunity — multi-class (Macro-EO).

    Args:
        y_true: Ground-truth class labels (array-like of int).
        y_pred: Predicted class labels (array-like of int).
        sensitive: Group membership labels
            (see  attribute (S)).
        filter_classes: Optional set of class indices to EXCLUDE from
            the macro mean (filtered-macro convention; see module
            docstring). Per-class entries are still computed and
            returned. If every class is filtered, the macro value is
            ``NaN``.

    Returns:
        Tuple ``(macro_value, per_class)``.
    """
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()
    sens_arr = np.asarray(sensitive)

    classes = np.sort(np.unique(np.concatenate([y_true_arr, y_pred_arr])))
    groups = np.unique(sens_arr)

    per_class: dict[int, float] = {}
    for c in classes:
        rates = []
        for g in groups:
            mask = (sens_arr == g) & (y_true_arr == c)
            if mask.sum() == 0:
                continue
            rate = float((y_pred_arr[mask] == c).mean())
            rates.append(rate)
        if len(rates) < 2:
            # Class has positive instances in fewer than 2 groups —
            # the per-class EO is not defined in the usual sense.
            per_class[int(c)] = float("nan")
        else:
            per_class[int(c)] = float(max(rates) - min(rates))

    excluded = set(filter_classes) if filter_classes else set()
    finite = [
        v for c, v in per_class.items()
        if not np.isnan(v) and int(c) not in excluded
    ]
    macro = float(np.mean(finite)) if finite else float("nan")
    return macro, per_class

def macro_eodds(
    y_true,
    y_pred,
    sensitive: pd.Series,
    *,
    filter_classes: set[int] | None = None,
) -> tuple[float, dict[int, float]]:
    """Macro Equalised Odds (Macro-EOdds) — multi-class extension.

    Definition (Hardt 2016 generalised): the average of
    ``macro_eo(y, ŷ)`` and ``macro_eo(1−y, 1−ŷ)`` ("EO of the negation",
    capturing the false-positive component for binary). For multi-class
    targets the negation is applied class-wise: each class's EO is
    averaged with the EO of the "not-c" target. ``per_class`` here maps
    ``c`` to ``(EO_c + EO_c_neg) / 2``.

    **Binary-restriction equivalence is CONDITIONAL.** For ``|𝒞| = 2``,
    ``macro_eodds == equalised_odds_difference`` holds only when the
    binary errors are symmetric (``|TPR_diff| == |FPR_diff|``). When
    errors are asymmetric, the binary EOdds takes a max while the macro
    form averages, so the two diverge: the unconditional reduction is to
    ``0.5 * (|TPR_diff| + |FPR_diff|)``
    (= :func:`average_absolute_odds_difference`), NOT to
    ``equalised_odds_difference``. This is documented empirically by
    ``test_macro_eodds_asymmetric_non_equivalence``.

    Reference: hardt2016equality (binary form generalised); see
     Odds — multi-class (Macro-EOdds).

    Args:
        y_true: Ground-truth class labels (array-like of int).
        y_pred: Predicted class labels (array-like of int).
        sensitive: Group membership labels
            (see  attribute (S)).
        filter_classes: Optional set of class indices to EXCLUDE from
            the macro mean (filtered-macro convention; see module
            docstring). Per-class entries are still computed and
            returned. If every class is filtered, the macro value is
            ``NaN``.

    Returns:
        Tuple ``(macro_value, per_class)``.
    """
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()
    sens_arr = np.asarray(sensitive)

    classes = np.sort(np.unique(np.concatenate([y_true_arr, y_pred_arr])))
    groups = np.unique(sens_arr)

    per_class: dict[int, float] = {}
    for c in classes:
        # EO_c on the standard target (recall for class c).
        rates_pos = []
        for g in groups:
            mask = (sens_arr == g) & (y_true_arr == c)
            if mask.sum() == 0:
                continue
            rates_pos.append(float((y_pred_arr[mask] == c).mean()))

        # EO_c_neg: recall for "not c" — i.e., the rate at which the
        # model predicts not-c when the ground truth is not-c.
        rates_neg = []
        for g in groups:
            mask = (sens_arr == g) & (y_true_arr != c)
            if mask.sum() == 0:
                continue
            rates_neg.append(float((y_pred_arr[mask] != c).mean()))

        components = []
        if len(rates_pos) >= 2:
            components.append(max(rates_pos) - min(rates_pos))
        if len(rates_neg) >= 2:
            components.append(max(rates_neg) - min(rates_neg))
        if not components:
            per_class[int(c)] = float("nan")
        else:
            per_class[int(c)] = float(np.mean(components))

    excluded = set(filter_classes) if filter_classes else set()
    finite = [
        v for c, v in per_class.items()
        if not np.isnan(v) and int(c) not in excluded
    ]
    macro = float(np.mean(finite)) if finite else float("nan")
    return macro, per_class

def multinomial_counterfactual_fairness(
    y_true,
    y_pred,
    sensitive: pd.Series,
    X: pd.DataFrame,
    model,
    sensitive_col: str,
    n_cf: int = 50,
) -> float:
    """Multi-class Level-1 Counterfactual Fairness via TV-distance.

    Generalises ``counterfactual_fairness`` (Kusner 2017 §3.1, Level-1
    sensitive-flip) to multi-class targets. For each sampled instance
    ``x`` with sensitive value ``a``::

        score_orig = model.predict_proba(x | A=a)
        for each a' ≠ a:
            score_cf  = model.predict_proba(x | A=a')
            TV(orig, cf) = 0.5 * Σ_c | score_orig[c] − score_cf[c] |
        max_TV = max over a' of TV
    macro_value = 1 − mean(max_TV) ∈ [0, 1].

    1.0 = perfectly counterfactually fair (flipping the sensitive value
    leaves the prediction distribution unchanged for every instance).

    When ``model`` does not expose ``predict_proba`` the function falls
    back to hard predictions (one-hot vectors); TV-distance over one-hot
    is exactly ``I(orig != cf)``, which reproduces the binary
    sensitive-flip count of ``counterfactual_fairness``.

    Consistent with  ( used Level-1 sensitive-flip;
    will add a DiCE-based variant); see  §1.

    Reference: kusner2017counterfactual §3.1 (Level-1 form generalised);
    see  Fairness — multi-class.

    Args:
        y_true: Unused; kept for  compatibility.
        y_pred: Unused; model is called directly.
        sensitive: Sensitive attribute Series (unused directly; sensitive
            values come from ``X[sensitive_col]``).
        X: Feature DataFrame containing ``sensitive_col`` as one column.
        model: Fitted sklearn-compatible classifier; ``predict_proba``
            is used when available, otherwise ``predict``.
        sensitive_col: Name of the sensitive column in ``X``.
        n_cf: Max instances to evaluate (random sample with seed=0).

    Returns:
        ``float`` ∈ [0, 1] — the macro counterfactual-fairness score.
    """
    rng = np.random.default_rng(0)
    n = len(X)
    idx = rng.choice(n, size=min(n_cf, n), replace=False)
    X_sample = X.iloc[idx].copy().reset_index(drop=True)

    unique_vals = X_sample[sensitive_col].unique().tolist()
    if len(unique_vals) < 2:
        return 1.0

    use_proba = hasattr(model, "predict_proba")

    X_enc = _ordinal_encode_df(X_sample)
    if use_proba:
        try:
            orig_scores = np.asarray(model.predict_proba(X_enc), dtype=float)
        except (NotImplementedError, AttributeError):
            use_proba = False
    if not use_proba:
        orig_hard = np.asarray(model.predict(X_enc)).ravel()

    # Build counterfactual matrices: one per alternative sensitive value.
    # For each instance, flip to a different value (matches the existing
    # binary `counterfactual_fairness` API: a single random alternative).
    X_cf = X_sample.copy()
    for i in range(len(X_cf)):
        current = X_cf.at[i, sensitive_col]
        others = [v for v in unique_vals if v != current]
        X_cf.at[i, sensitive_col] = rng.choice(others)

    X_cf_enc = _ordinal_encode_df(X_cf)
    if use_proba:
        cf_scores = np.asarray(model.predict_proba(X_cf_enc), dtype=float)
        # TV-distance per row.
        tv = 0.5 * np.abs(orig_scores - cf_scores).sum(axis=1)
    else:
        cf_hard = np.asarray(model.predict(X_cf_enc)).ravel()
        # TV over one-hot is exactly I(orig != cf).
        tv = (orig_hard != cf_hard).astype(float)

    return float(1.0 - tv.mean())

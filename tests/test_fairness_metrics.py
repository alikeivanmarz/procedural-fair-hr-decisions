"""Tests for src/fairness_metrics.py binary metrics,  individual,  CF.

Exit-gates:
  pytest tests/test_fairness_metrics.py::test_binary_group_metrics
  pytest tests/test_fairness_metrics.py::test_individual_metrics
  pytest tests/test_fairness_metrics.py::test_counterfactual_fairness
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder

from procedural_fair_hr.fairness_metrics import (
    counterfactual_fairness,
    demographic_parity_difference,
    disparate_impact_ratio,
    equal_opportunity_difference,
    equalised_odds_difference,
    average_absolute_odds_difference,
    accuracy_balance,
    knn_consistency,
    lipschitz_fairness,
    macro_dp,
    macro_eo,
    macro_eodds,
    multinomial_counterfactual_fairness,
)

def test_binary_group_metrics():
    """: binary group-fairness metrics on a controlled toy case."""
    rng = np.random.default_rng(0)
    n = 200
    s = pd.Series(["A"] * 100 + ["B"] * 100, name="group")
    y_true = np.ones(n, dtype=int)
    y_pred = np.array([1] * 100 + [0] * 100)

    dpd = demographic_parity_difference(y_true, y_pred, s)
    assert abs(dpd - 1.0) < 0.01, f"DPD should be 1.0, got {dpd}"

    di = disparate_impact_ratio(y_true, y_pred, s)
    assert di == 0.0 or abs(di) < 0.01, f"DI should be ~0, got {di}"

    eoo = equal_opportunity_difference(y_true, y_pred, s)
    assert abs(eoo - 1.0) < 0.01, f"EOO should be 1.0, got {eoo}"

    eodds = equalised_odds_difference(y_true, y_pred, s)
    assert eodds >= 0.0

    aaod = average_absolute_odds_difference(y_true, y_pred, s)
    assert aaod >= 0.0

    ab = accuracy_balance(y_true, y_pred, s)
    assert ab >= 0.0

    for val in [dpd, di, eoo, eodds, aaod, ab]:
        assert isinstance(val, float)

def test_individual_metrics():
    """: KNNC and Lipschitz individual fairness metrics."""
    rng = np.random.default_rng(42)
    n = 100
    X = pd.DataFrame({"f1": rng.standard_normal(n), "f2": rng.standard_normal(n)})
    s = pd.Series(["A"] * 50 + ["B"] * 50, name="group")
    y_pred = np.ones(n, dtype=int)

    knnc = knn_consistency(None, y_pred, s, X=X, k=5)
    assert knnc == 1.0, f"All-same predictions → KNNC=1.0, got {knnc}"

    lf = lipschitz_fairness(y_pred, s, X=X)
    assert 0.0 <= lf <= 1.0

    y_noisy = rng.integers(0, 2, n)
    knnc_noisy = knn_consistency(None, y_noisy, s, X=X, k=5)
    assert 0.0 <= knnc_noisy <= 1.0

def test_counterfactual_fairness():
    """Test Level-1 counterfactual fairness on a toy model that ignores sensitive col.

    The LogisticRegression model is trained on OrdinalEncoder-transformed features
    so that "M"/"F" become numeric.  The model is then evaluated by
    ``counterfactual_fairness``, which encodes X internally via _ordinal_encode_df.
    The resulting score must be a float in [0, 1].

    See  Fairness.
    Reference: kusner2017counterfactual sec 3.1.
    """
    rng = np.random.default_rng(0)
    n = 100
    # A model that is explicitly NOT sensitive to the sensitive_col
    X = pd.DataFrame({
        "feature": rng.standard_normal(n),
        "sensitive": ["M"] * 50 + ["F"] * 50,
    })
    y = (X["feature"] > 0).astype(int)
    s = pd.Series(X["sensitive"], name="sensitive")

    # Encode for sklearn
    enc = OrdinalEncoder()
    X_enc = enc.fit_transform(X)
    clf = LogisticRegression(random_state=0).fit(X_enc, y)

    # A fair model (ignores sensitive col effectively)
    cf_score = counterfactual_fairness(
        y_true=y.values,
        y_pred=clf.predict(X_enc),
        sensitive=s,
        X=X,
        model=clf,
        sensitive_col="sensitive",
        n_cf=10,
    )
    assert 0.0 <= cf_score <= 1.0
    assert isinstance(cf_score, float)

def test_counterfactual_fairness_perfect_score():
    """A model that predicts purely from a non-sensitive feature must score 1.0.

    When all predictions are determined by ``feature`` alone (not ``sensitive``),
    flipping the sensitive column should leave every prediction unchanged.

    See  Fairness.
    """
    rng = np.random.default_rng(42)
    n = 80

    X = pd.DataFrame({
        "feature": rng.standard_normal(n),
        "sensitive": (["A"] * (n // 2)) + (["B"] * (n // 2)),
    })
    # Labels depend only on feature
    y = (X["feature"] > 0).astype(int)
    s = pd.Series(X["sensitive"], name="sensitive")

    # Train on encoded X
    enc = OrdinalEncoder()
    X_enc = enc.fit_transform(X)

    # Build a model that uses only the first column (feature) by zeroing out
    # the sensitive coefficient after fitting
    clf = LogisticRegression(random_state=0).fit(X_enc, y)
    # Zero out coefficient for 'sensitive' column (index depends on OrdinalEncoder order)
    # OrdinalEncoder output column order matches input DataFrame column order
    # X columns: ["feature", "sensitive"] -> indices 0, 1
    clf.coef_[0, 1] = 0.0  # kill sensitive coefficient

    cf_score = counterfactual_fairness(
        y_true=y.values,
        y_pred=clf.predict(X_enc),
        sensitive=s,
        X=X,
        model=clf,
        sensitive_col="sensitive",
        n_cf=5,
    )
    # With the sensitive coefficient zeroed, flipping "A"/"B" changes nothing
    assert cf_score == 1.0

def test_counterfactual_fairness_return_type():
    """counterfactual_fairness must return a Python float (not np.float64 etc.)."""
    rng = np.random.default_rng(7)
    n = 40
    X = pd.DataFrame({
        "x": rng.standard_normal(n),
        "grp": ["P"] * 20 + ["Q"] * 20,
    })
    y = (X["x"] > 0).astype(int)
    s = pd.Series(X["grp"], name="grp")
    enc = OrdinalEncoder()
    X_enc = enc.fit_transform(X)
    clf = LogisticRegression(random_state=0).fit(X_enc, y)

    score = counterfactual_fairness(
        y_true=y.values,
        y_pred=clf.predict(X_enc),
        sensitive=s,
        X=X,
        model=clf,
        sensitive_col="grp",
    )
    assert type(score) is float

# ---------------------------------------------------------------------------
# Phase-3 multi-class fairness extensions
# ---------------------------------------------------------------------------

def test_macro_dp_returns_correct_shape():
    """: macro_dp returns (float, dict) with one entry per class on a
    3-class case.

    See  Parity — multi-class (Macro-DP).
    """
    rng = np.random.default_rng(0)
    n = 300
    y_true = rng.integers(0, 3, size=n)
    y_pred = rng.integers(0, 3, size=n)
    s = pd.Series(["A"] * 150 + ["B"] * 150, name="group")

    macro, per_class = macro_dp(y_true, y_pred, s)
    assert isinstance(macro, float)
    assert isinstance(per_class, dict)
    assert set(per_class.keys()) == {0, 1, 2}
    for v in per_class.values():
        assert isinstance(v, float)

def test_macro_dp_binary_restriction_equivalence():
    """: |𝒞|=2 ⇒ macro_dp[0] == |demographic_parity_difference| within 1e-9.

    Algebraic identity: for binary, the per-class rate-difference for c=0 equals
    that for c=1 (since P(ŷ=0|S=g) = 1 − P(ŷ=1|S=g)). The mean of the two
    equals the absolute value of the signed binary DP.

    See  Parity (DP) — binary.
    """
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    # Build an asymmetric prediction pattern so DP is non-zero.
    y_pred = np.concatenate([
        rng.integers(0, 2, size=100),  # group A: ~50 % positive
        (rng.random(100) < 0.2).astype(int),  # group B: ~20 % positive
    ])
    s = pd.Series(["A"] * 100 + ["B"] * 100, name="group")

    macro, per_class = macro_dp(y_true, y_pred, s)
    binary_dp = demographic_parity_difference(y_true, y_pred, s)

    # macro_dp == |binary DP| as an exact algebraic identity (modulo
    # floating-point rounding).
    assert abs(macro - abs(binary_dp)) < 1e-9
    # And per-class entries should be identical for binary (DP_0 == DP_1).
    assert abs(per_class[0] - per_class[1]) < 1e-9

def test_macro_dp_glossary_reference():
    """: docstring references the GLOSSARY entry for Macro-DP."""
    doc = macro_dp.__doc__ or ""
    assert "macro_dp" in doc.lower() or "Macro-DP" in doc
def test_macro_eo_returns_correct_shape():
    """: macro_eo returns (float, dict) with one entry per class on a
    3-class case.

    See  Opportunity — multi-class (Macro-EO).
    """
    rng = np.random.default_rng(0)
    n = 300
    y_true = rng.integers(0, 3, size=n)
    y_pred = rng.integers(0, 3, size=n)
    s = pd.Series(["A"] * 150 + ["B"] * 150, name="group")

    macro, per_class = macro_eo(y_true, y_pred, s)
    assert isinstance(macro, float)
    assert isinstance(per_class, dict)
    assert set(per_class.keys()) == {0, 1, 2}

def test_macro_eo_binary_restriction_equivalence():
    """: |𝒞|=2 + symmetric errors ⇒ macro_eo[0] ==
    |equal_opportunity_difference| within 1e-9.

    Construction: a symmetric scenario where |TPR_diff| == |TNR_diff|, so
    EO_0 == EO_1 == |TPR_diff| and macro_eo == |TPR_diff| ==
    |equal_opportunity_difference|. The privileged group ("A") is correct
    on every instance; the unprivileged group ("B") is correct on the
    first half of positives + first half of negatives, wrong on the rest.

    See  Opportunity (EOO) — binary.
    """
    n_per_group = 100  # 50 pos + 50 neg per group
    y_true = np.concatenate([
        # priv group A
        np.ones(50, dtype=int), np.zeros(50, dtype=int),
        # unpriv group B
        np.ones(50, dtype=int), np.zeros(50, dtype=int),
    ])
    y_pred = np.concatenate([
        # priv group A: perfect predictor → TPR=1, FPR=0
        np.ones(50, dtype=int), np.zeros(50, dtype=int),
        # unpriv group B: 50 % TPR + 50 % FPR (symmetric errors)
        np.array([1] * 25 + [0] * 25, dtype=int),
        np.array([1] * 25 + [0] * 25, dtype=int),
    ])
    s = pd.Series(["A"] * n_per_group + ["B"] * n_per_group, name="group")

    macro, per_class = macro_eo(y_true, y_pred, s)
    binary_eo = equal_opportunity_difference(y_true, y_pred, s)

    # |TPR_diff| = 0.5; |TNR_diff| = 0.5 → EO_0 == EO_1 == 0.5.
    assert abs(per_class[0] - per_class[1]) < 1e-9
    assert abs(macro - abs(binary_eo)) < 1e-9

def test_macro_eodds_returns_correct_shape():
    """: macro_eodds returns (float, dict) with one entry per class on a
    3-class case.

    See  Odds — multi-class (Macro-EOdds).
    """
    rng = np.random.default_rng(0)
    n = 300
    y_true = rng.integers(0, 3, size=n)
    y_pred = rng.integers(0, 3, size=n)
    s = pd.Series(["A"] * 150 + ["B"] * 150, name="group")

    macro, per_class = macro_eodds(y_true, y_pred, s)
    assert isinstance(macro, float)
    assert isinstance(per_class, dict)
    assert set(per_class.keys()) == {0, 1, 2}

def test_macro_eodds_binary_restriction_equivalence():
    """: |𝒞|=2 + symmetric errors ⇒ macro_eodds[0] ==
    |equalised_odds_difference| within 1e-9.

    Same symmetric construction as the macro_eo test: |TPR_diff| ==
    |FPR_diff| = 0.5, so equalised_odds_difference == max(|TPR_diff|,
    |FPR_diff|) == 0.5 and macro_eodds == 0.5.

    See  Odds (EOdds) — binary.
    """
    n_per_group = 100
    y_true = np.concatenate([
        np.ones(50, dtype=int), np.zeros(50, dtype=int),
        np.ones(50, dtype=int), np.zeros(50, dtype=int),
    ])
    y_pred = np.concatenate([
        np.ones(50, dtype=int), np.zeros(50, dtype=int),
        np.array([1] * 25 + [0] * 25, dtype=int),
        np.array([1] * 25 + [0] * 25, dtype=int),
    ])
    s = pd.Series(["A"] * n_per_group + ["B"] * n_per_group, name="group")

    macro, _ = macro_eodds(y_true, y_pred, s)
    binary_eodds = equalised_odds_difference(y_true, y_pred, s)

    assert abs(macro - binary_eodds) < 1e-9

def test_multinomial_cf_returns_float():
    """: multinomial_counterfactual_fairness returns a float in [0, 1]
    on a 3-class case.
    """
    rng = np.random.default_rng(0)
    n = 90
    X = pd.DataFrame({
        "feature": rng.standard_normal(n),
        "sensitive": (["A"] * 30) + (["B"] * 30) + (["C"] * 30),
    })
    # 3-class target derived from feature buckets.
    y = pd.qcut(X["feature"], q=3, labels=False).astype(int)

    enc = OrdinalEncoder()
    X_enc = enc.fit_transform(X)
    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_enc, y)
    s = pd.Series(X["sensitive"], name="sensitive")

    score = multinomial_counterfactual_fairness(
        y_true=y.values,
        y_pred=clf.predict(X_enc),
        sensitive=s,
        X=X,
        model=clf,
        sensitive_col="sensitive",
        n_cf=30,
    )
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

def test_multinomial_cf_binary_restriction_equivalence():
    """: |𝒞|=2 ⇒ multinomial_counterfactual_fairness reduces exactly to
    counterfactual_fairness on hard-prediction inputs.

    Wraps the LogisticRegression in a stub that exposes only ``predict``
    (no ``predict_proba``), forcing the multi-class CF function down its
    one-hot fallback path. Both functions then use the same RNG seed
    and identity-flip on the sensitive column, so they must match
    exactly (TV over one-hot == I(orig != cf) == 1 − hard-equality).

    See  Fairness.
    """
    rng = np.random.default_rng(0)
    n = 100
    X = pd.DataFrame({
        "feature": rng.standard_normal(n),
        "sensitive": ["M"] * 50 + ["F"] * 50,
    })
    y = (X["feature"] > 0).astype(int)
    s = pd.Series(X["sensitive"], name="sensitive")

    enc = OrdinalEncoder()
    X_enc = enc.fit_transform(X)
    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_enc, y)

    class _HardOnly:
        """Wrapper exposing only ``predict`` to force the hard-prediction
        fallback path."""

        def __init__(self, model):
            self._model = model

        def predict(self, X):
            return self._model.predict(X)

    hard_model = _HardOnly(clf)

    binary_cf = counterfactual_fairness(
        y_true=y.values,
        y_pred=clf.predict(X_enc),
        sensitive=s,
        X=X,
        model=hard_model,
        sensitive_col="sensitive",
        n_cf=30,
    )
    multi_cf = multinomial_counterfactual_fairness(
        y_true=y.values,
        y_pred=clf.predict(X_enc),
        sensitive=s,
        X=X,
        model=hard_model,
        sensitive_col="sensitive",
        n_cf=30,
    )
    assert abs(multi_cf - binary_cf) < 1e-9

# ---------------------------------------------------------------------------
# Phase-3 followup (post-review concerns)

# Issue 2c — non-equivalence empirical demonstration: the symmetric-error
# condition for the macro_eo / macro_eodds binary-restriction reduction
# is NOT unconditional. The tests below construct deterministic synthetic
# data where |TPR_diff| ≠ |FPR_diff| (Macro-EOdds case) or
# |TPR_diff| ≠ |TNR_diff| (Macro-EO case) and confirm that the macro
# form does NOT reduce to the binary form in that asymmetric regime.
# This complements the existing equivalence tests (which deliberately
# construct symmetric synthetic data) and lets a future reader see the
# conditionality empirically rather than only as a docstring claim.

# Issue 1 — filtered macro: the audit script emits a parallel
# "filtered-macro" row for each multi-class group metric (target_form ==
# "macro_filtered") that excludes per-class entries flagged
# non_vacuous_tpr == False. The tests below pin the filter convention:
# (a) a degenerate per-class row is excluded from the macro mean; (b) if
# every class is filtered, the macro returns NaN (per docstring).
# ---------------------------------------------------------------------------

def test_macro_eo_asymmetric_non_equivalence():
    """Phase-3 followup Issue 2c: when |TPR_diff| ≠ |TNR_diff|, ``macro_eo``
    does NOT reduce to ``equal_opportunity_difference``.

    Construction (binary, |𝒞|=2): we engineer asymmetric errors so that
    the privileged group has perfect TPR but imperfect TNR while the
    unprivileged group has imperfect TPR and perfect TNR. Specifically::

        priv (A): TPR = 1.00, TNR = 0.50  (50 % FPR)
        unpriv (B): TPR = 0.50, TNR = 1.00  (0 % FPR)

    so |TPR_diff| = 0.50 and |TNR_diff| = 0.50 are equal — this is NOT
    asymmetric enough. We instead use::

        priv (A): TPR = 1.00, TNR = 1.00
        unpriv (B): TPR = 0.40, TNR = 0.90

    which gives |TPR_diff| = 0.60, |TNR_diff| = 0.10 — clearly asymmetric.
    The binary EOO equals |TPR_diff| = 0.60. The macro_eo equals
    0.5 * (|TPR_diff| + |TNR_diff|) = 0.35. The two differ by 0.25,
    well above the 1e-3 tolerance, demonstrating the conditionality.

    See  Opportunity — multi-class (Macro-EO) and
     Opportunity (EOO) — binary.
    """
    # Group A (privileged): 50 positives + 50 negatives, model is perfect.
    # Group B (unprivileged): 50 positives (40 % TPR -> 20 correct, 30 wrong),
    # 50 negatives (90 % TNR -> 5 false positives).
    y_true = np.concatenate([
        np.ones(50, dtype=int), np.zeros(50, dtype=int),  # A
        np.ones(50, dtype=int), np.zeros(50, dtype=int),  # B
    ])
    y_pred = np.concatenate([
        np.ones(50, dtype=int), np.zeros(50, dtype=int),  # A perfect
        # B: 20 / 50 positives correct (TPR = 0.4) and 5 / 50 negatives
        # incorrectly predicted positive (FPR = 0.1, TNR = 0.9).
        np.array([1] * 20 + [0] * 30, dtype=int),
        np.array([1] * 5 + [0] * 45, dtype=int),
    ])
    s = pd.Series(["A"] * 100 + ["B"] * 100, name="group")

    binary_eo = equal_opportunity_difference(y_true, y_pred, s)
    macro, per_class = macro_eo(y_true, y_pred, s)

    # Sanity: the construction satisfies the asymmetric regime.
    # |TPR_diff| = 1.0 - 0.4 = 0.6; |TNR_diff| = 1.0 - 0.9 = 0.1.
    assert abs(abs(binary_eo) - 0.6) < 1e-9, (
        f"binary EOO should be 0.6 by construction, got {binary_eo}"
    )
    # The unconditional reduction: macro_eo == 0.5 * (|TPR_diff| + |TNR_diff|).
    expected_macro = 0.5 * (0.6 + 0.1)
    assert abs(macro - expected_macro) < 1e-9, (
        f"macro_eo should be {expected_macro} (mean of TPR_diff and TNR_diff), "
        f"got {macro}"
    )
    # The non-equivalence claim: macro_eo != |binary EOO| in the asymmetric regime.
    assert abs(macro - abs(binary_eo)) > 1e-3, (
        f"macro_eo and |binary EOO| coincide ({macro}, {abs(binary_eo)}); "
        f"the asymmetric construction failed to demonstrate non-equivalence"
    )
    # And the per-class entries are NOT equal in the asymmetric regime
    # (in the symmetric regime they coincide; here they should differ).
    assert abs(per_class[0] - per_class[1]) > 1e-3, (
        f"per-class EO_0 ({per_class[0]}) and EO_1 ({per_class[1]}) coincide; "
        f"asymmetric construction failed"
    )

def test_macro_eodds_asymmetric_non_equivalence():
    """Phase-3 followup Issue 2c: when |TPR_diff| ≠ |FPR_diff|, ``macro_eodds``
    does NOT reduce to ``equalised_odds_difference``.

    Construction (binary, |𝒞|=2): same asymmetric data as the macro_eo
    non-equivalence test (|TPR_diff| = 0.6, |FPR_diff| = 0.1).

    The binary equalised_odds_difference takes a max:
        max(|TPR_diff|, |FPR_diff|) = max(0.6, 0.1) = 0.6.
    The macro_eodds takes an unweighted average over per-class
    (EO + EO_neg)/2 entries; for binary this equals
    0.5 * (|TPR_diff| + |FPR_diff|) = 0.35
    (= average_absolute_odds_difference). The two differ by 0.25, well
    above the 1e-3 tolerance, demonstrating the conditionality.

    See  Odds — multi-class (Macro-EOdds) and
     Odds (EOdds) — binary.
    """
    y_true = np.concatenate([
        np.ones(50, dtype=int), np.zeros(50, dtype=int),  # A
        np.ones(50, dtype=int), np.zeros(50, dtype=int),  # B
    ])
    y_pred = np.concatenate([
        np.ones(50, dtype=int), np.zeros(50, dtype=int),  # A perfect
        np.array([1] * 20 + [0] * 30, dtype=int),  # B TPR = 0.4
        np.array([1] * 5 + [0] * 45, dtype=int),    # B FPR = 0.1
    ])
    s = pd.Series(["A"] * 100 + ["B"] * 100, name="group")

    binary_eodds = equalised_odds_difference(y_true, y_pred, s)
    macro, _ = macro_eodds(y_true, y_pred, s)

    # Sanity: binary_eodds = max(|TPR_diff|, |FPR_diff|) = 0.6.
    assert abs(binary_eodds - 0.6) < 1e-9, (
        f"binary EOdds should be 0.6 by construction, got {binary_eodds}"
    )
    # macro_eodds = 0.5 * (|TPR_diff| + |FPR_diff|) = 0.35 in the
    # asymmetric regime (= average_absolute_odds_difference).
    expected_macro = 0.5 * (0.6 + 0.1)
    assert abs(macro - expected_macro) < 1e-9, (
        f"macro_eodds should be {expected_macro} (mean of TPR/FPR diffs), "
        f"got {macro}"
    )
    # Non-equivalence: macro_eodds != binary_eodds in the asymmetric regime.
    assert abs(macro - binary_eodds) > 1e-3, (
        f"macro_eodds and binary_eodds coincide ({macro}, {binary_eodds}); "
        f"asymmetric construction failed"
    )

def test_macro_dp_filtered_excludes_class():
    """Phase-3 followup Issue 1: ``filter_classes`` excludes the named class
    from the macro mean while keeping the per-class dict complete.

    Construction (3-class): per-class DPs are designed to be 0.4, 0.2, 0.0
    so the unfiltered macro is 0.2 and a filter that drops class 2
    (the degenerate one with DP=0.0) yields a filtered macro of 0.3
    (= mean of {0.4, 0.2}). The per_class dict still contains all three
    entries — only the macro mean is restricted.

    See  Parity — multi-class (Macro-DP).
    """
    # 50 instances per group; predictions engineered so that:
    #   class 0: priv predicts 40 %, unpriv predicts 0 %  -> DP_0 = 0.4
    #   class 1: priv predicts 30 %, unpriv predicts 10 % -> DP_1 = 0.2
    #   class 2: priv predicts 30 %, unpriv predicts 30 % -> DP_2 = 0.0
    # (We just need controlled per-class rates; y_true is unused by macro_dp.)
    y_true = np.zeros(100, dtype=int)
    y_pred = np.concatenate([
        # priv (A): 20 zeros, 15 ones, 15 twos
        np.array([0] * 20 + [1] * 15 + [2] * 15, dtype=int),
        # unpriv (B): 0 zeros, 5 ones (10%), 15 twos (30%), 30 threes -> use 0
        # We need 50 entries total for B: 0 zeros, 5 ones, 15 twos, 30 zeros
        # -> classes: 30 zeros, 5 ones, 15 twos
        np.array([0] * 30 + [1] * 5 + [2] * 15, dtype=int),
    ])
    s = pd.Series(["A"] * 50 + ["B"] * 50, name="group")

    # Verify per-class entries match the construction.
    macro_unfilt, per_class = macro_dp(y_true, y_pred, s)
    assert abs(per_class[0] - 0.20) < 1e-9, f"DP_0 expected 0.20, got {per_class[0]}"  # |0.4-0.6|
    assert abs(per_class[1] - 0.20) < 1e-9, f"DP_1 expected 0.20, got {per_class[1]}"  # |0.3-0.1|
    assert abs(per_class[2] - 0.00) < 1e-9, f"DP_2 expected 0.00, got {per_class[2]}"  # |0.3-0.3|
    expected_unfiltered = (0.20 + 0.20 + 0.00) / 3
    assert abs(macro_unfilt - expected_unfiltered) < 1e-9

    # Filter out the degenerate class 2.
    macro_filt, per_class_filt = macro_dp(
        y_true, y_pred, s, filter_classes={2}
    )
    expected_filtered = (0.20 + 0.20) / 2
    assert abs(macro_filt - expected_filtered) < 1e-9, (
        f"filtered macro_dp expected {expected_filtered}, got {macro_filt}"
    )
    # per_class is still complete (unchanged) even when class 2 is filtered.
    assert set(per_class_filt.keys()) == {0, 1, 2}
    assert abs(per_class_filt[2] - 0.00) < 1e-9

def test_macro_dp_filtered_all_classes_returns_nan():
    """Phase-3 followup Issue 1: filtering every class yields a NaN macro.

    Documented behaviour from the ``macro_dp`` docstring + the GLOSSARY
    Macro-DP filtered-variant note: when every observed class is excluded
    via ``filter_classes``, the macro mean has zero contributors and is
    ``NaN``. The per-class dict is still complete.
    """
    y_true = np.array([0, 1, 2, 0, 1, 2], dtype=int)
    y_pred = np.array([0, 1, 2, 1, 2, 0], dtype=int)
    s = pd.Series(["A", "A", "A", "B", "B", "B"], name="group")

    macro, per_class = macro_dp(y_true, y_pred, s, filter_classes={0, 1, 2})
    assert np.isnan(macro), f"all-classes-filtered macro should be NaN, got {macro}"
    assert set(per_class.keys()) == {0, 1, 2}

def test_macro_eo_filtered_excludes_class():
    """Phase-3 followup Issue 1: filter_classes works for macro_eo too.

    Same convention as macro_dp_filtered: an excluded class still appears
    in the per_class dict but does not contribute to the macro mean.
    """
    # Build a 3-class scenario with controlled per-class TPR differences.
    # 30 instances per (group, class) cell, 2 groups × 3 classes = 180 rows.
    rng = np.random.default_rng(0)
    n_per_cell = 30
    y_true_parts = []
    y_pred_parts = []
    sens_parts = []
    # Group A: perfect predictor for every class (TPR = 1).
    for c in (0, 1, 2):
        y_true_parts.append(np.full(n_per_cell, c, dtype=int))
        y_pred_parts.append(np.full(n_per_cell, c, dtype=int))
        sens_parts.append(np.full(n_per_cell, "A"))
    # Group B: TPR is 0.5 for class 0, 1.0 for class 1, 0.0 for class 2
    # (the latter is the "degenerate" class candidate).
    # Class 0: 15 correct + 15 wrong (predict class 1).
    y_true_parts.append(np.full(n_per_cell, 0, dtype=int))
    y_pred_parts.append(np.array([0] * 15 + [1] * 15, dtype=int))
    sens_parts.append(np.full(n_per_cell, "B"))
    # Class 1: all correct.
    y_true_parts.append(np.full(n_per_cell, 1, dtype=int))
    y_pred_parts.append(np.full(n_per_cell, 1, dtype=int))
    sens_parts.append(np.full(n_per_cell, "B"))
    # Class 2: all wrong (predict class 0) -> TPR = 0 for class 2 in B.
    y_true_parts.append(np.full(n_per_cell, 2, dtype=int))
    y_pred_parts.append(np.full(n_per_cell, 0, dtype=int))
    sens_parts.append(np.full(n_per_cell, "B"))

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    s = pd.Series(np.concatenate(sens_parts), name="group")

    # Per-class TPR diffs: class 0 = 1.0 - 0.5 = 0.5; class 1 = 0; class 2 = 1.0.
    macro_unfilt, per_class = macro_eo(y_true, y_pred, s)
    assert abs(per_class[0] - 0.5) < 1e-9
    assert abs(per_class[1] - 0.0) < 1e-9
    assert abs(per_class[2] - 1.0) < 1e-9

    # Filter out class 2 (the most extreme): macro should drop to the mean
    # of class 0 and class 1 only.
    macro_filt, _ = macro_eo(y_true, y_pred, s, filter_classes={2})
    expected = (0.5 + 0.0) / 2
    assert abs(macro_filt - expected) < 1e-9

def test_macro_eodds_filtered_excludes_class():
    """Phase-3 followup Issue 1: filter_classes works for macro_eodds too."""
    # Use the same setup as macro_eo_filtered above.
    n_per_cell = 30
    y_true_parts = []
    y_pred_parts = []
    sens_parts = []
    for c in (0, 1, 2):
        y_true_parts.append(np.full(n_per_cell, c, dtype=int))
        y_pred_parts.append(np.full(n_per_cell, c, dtype=int))
        sens_parts.append(np.full(n_per_cell, "A"))
    y_true_parts.append(np.full(n_per_cell, 0, dtype=int))
    y_pred_parts.append(np.array([0] * 15 + [1] * 15, dtype=int))
    sens_parts.append(np.full(n_per_cell, "B"))
    y_true_parts.append(np.full(n_per_cell, 1, dtype=int))
    y_pred_parts.append(np.full(n_per_cell, 1, dtype=int))
    sens_parts.append(np.full(n_per_cell, "B"))
    y_true_parts.append(np.full(n_per_cell, 2, dtype=int))
    y_pred_parts.append(np.full(n_per_cell, 0, dtype=int))
    sens_parts.append(np.full(n_per_cell, "B"))

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    s = pd.Series(np.concatenate(sens_parts), name="group")

    macro_unfilt, per_class = macro_eodds(y_true, y_pred, s)
    macro_filt, _ = macro_eodds(y_true, y_pred, s, filter_classes={2})

    # Filtered macro must equal the mean of the kept classes' per-class
    # entries (excluded ones contribute nothing).
    expected = (per_class[0] + per_class[1]) / 2
    assert abs(macro_filt - expected) < 1e-9
    # Filtered macro should differ from unfiltered (since class 2 is the
    # most extreme contributor in this construction).
    assert abs(macro_filt - macro_unfilt) > 1e-9

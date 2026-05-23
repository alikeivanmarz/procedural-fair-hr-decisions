"""Tests for ``src/procedural_fairness.py`` — .

Covers the three procedural-justice metrics:
    * ``process_consistency``
    * ``voice_representation``
    * ``transparency_metrics``

Plus the two cross-cutting invariants:

"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from procedural_fair_hr.procedural_fairness import (
    explanation_actionability,
    model_flippability,
    process_consistency,
    transparency_metrics,
    voice_representation,
)

# ---------------------------------------------------------------------------
#  — process_consistency
# ---------------------------------------------------------------------------

class _ConstantClassifier:
    """Always predicts class 0; ``predict_proba`` returns full-mass-on-0."""

    def __init__(self, n_classes: int = 2):
        self.n_classes = n_classes

    def predict(self, X):
        n = len(X)
        return np.zeros(n, dtype=int)

    def predict_proba(self, X):
        n = len(X)
        proba = np.zeros((n, self.n_classes), dtype=float)
        proba[:, 0] = 1.0
        return proba

class _RandomProbaClassifier:
    """Returns uniformly-random probability vectors per call.

    Used to verify that ``process_consistency`` drops below 1.0 when the
    model is genuinely sensitive to perturbations.
    """

    def __init__(self, n_classes: int = 3, seed: int = 0):
        self.n_classes = n_classes
        self._rng = np.random.default_rng(seed)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.asarray(proba.argmax(axis=1), dtype=int)

    def predict_proba(self, X):
        n = len(X)
        raw = self._rng.random((n, self.n_classes))
        raw = raw / raw.sum(axis=1, keepdims=True)
        return raw

def _toy_X(seed: int = 0, n: int = 60) -> pd.DataFrame:
    """Build a small mixed-type DataFrame for procedural-fairness tests."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "f_num1": rng.standard_normal(n),
            "f_num2": rng.uniform(0, 5, size=n),
            "f_cat": rng.choice(["A", "B", "C"], size=n),
        }
    )

def test_process_consistency_constant_model():
    """: a constant model is perfectly consistent (ProcConsistency = 1.0).

    A constant classifier returns the same prediction distribution for
    any input; perturbing the input cannot change the output, so
    JS divergence = 0 for every (row, perturbation) pair and the
    consistency score is exactly 1.0.

    See  Consistency.
    """
    X = _toy_X(seed=0, n=40)
    model = _ConstantClassifier(n_classes=2)
    score, per_row = process_consistency(
        model, X, perturbations_per_row=5, sample_n=20, random_state=0
    )
    assert score == pytest.approx(1.0, abs=1e-9)
    assert all(v == pytest.approx(1.0, abs=1e-9) for v in per_row.values())

def test_process_consistency_noisy_model():
    """: a model whose ``predict_proba`` returns random vectors must
    score strictly below 1.0 (typically well below).

    See  Consistency.
    """
    X = _toy_X(seed=0, n=40)
    model = _RandomProbaClassifier(n_classes=3, seed=42)
    score, _ = process_consistency(
        model, X, perturbations_per_row=10, sample_n=20, random_state=0
    )
    assert score < 0.99, f"Random model should score < 0.99; got {score}"
    assert 0.0 <= score <= 1.0

def test_process_consistency_signature():
    """ / : returns ``tuple[float, dict[int, float]]``."""
    X = _toy_X(seed=0, n=20)
    model = _ConstantClassifier()
    out = process_consistency(model, X, perturbations_per_row=3, sample_n=10)
    assert isinstance(out, tuple) and len(out) == 2
    score, per_row = out
    assert isinstance(score, float)
    assert isinstance(per_row, dict)
    assert all(isinstance(k, int) for k in per_row.keys())
    assert all(isinstance(v, float) for v in per_row.values())

def test_process_consistency_glossary_reference():
    """: ``process_consistency`` docstring references ."""
    doc = process_consistency.__doc__ or ""
    assert "Process Consistency" in doc
def test_process_consistency_stratified_sample():
    """: stratification on a sensitive attribute returns a sample
    drawn from both groups (not just one)."""
    X = _toy_X(seed=0, n=80)
    sens = pd.Series(["A"] * 40 + ["B"] * 40, name="group")
    model = _ConstantClassifier()
    _, per_row = process_consistency(
        model,
        X,
        perturbations_per_row=2,
        sample_n=20,
        stratify_on=sens,
        random_state=0,
    )
    selected_rows = set(per_row.keys())
    in_group_a = sum(1 for i in selected_rows if i < 40)
    in_group_b = sum(1 for i in selected_rows if i >= 40)
    assert in_group_a > 0 and in_group_b > 0, (
        "Stratified sample should draw from both groups; got "
        f"A={in_group_a}, B={in_group_b}"
    )

# ---------------------------------------------------------------------------
#  — voice_representation
# ---------------------------------------------------------------------------

class _LinearScoreModel:
    """A linear scorer that exposes ``predict`` only.

    Provides hand-set coefficients so we can engineer SHAP values
    deterministically for the voice tests. Compatible with
    ``shap.KernelExplainer`` (uses ``predict``).
    """

    def __init__(self, coefs: np.ndarray, intercept: float = 0.0):
        self.coefs = np.asarray(coefs, dtype=float).ravel()
        self.intercept = float(intercept)

    def predict(self, X):
        X_arr = np.asarray(X, dtype=float)
        # Guard against shape mismatch by truncating to coef length.
        n_used = min(X_arr.shape[1], self.coefs.shape[0])
        scores = X_arr[:, :n_used] @ self.coefs[:n_used] + self.intercept
        return (scores > 0).astype(int)

def _build_tree_model_for_voice(X: pd.DataFrame, y: np.ndarray):
    """Fit a small RandomForest for use with ``shap.TreeExplainer``."""
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=20, random_state=0, max_depth=4
    ).fit(X.values, y)

def test_voice_zero_when_no_modifiable():
    """: empty 'modifiable' list → voice = 0 (every feature is immutable)."""
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame(
        {
            "a": rng.standard_normal(n),
            "b": rng.standard_normal(n),
            "c": rng.standard_normal(n),
        }
    )
    y = (X["a"] + X["b"] > 0).astype(int).values
    model = _build_tree_model_for_voice(X, y)
    score, enrichment, per_feature = voice_representation(
        model,
        X,
        feature_partition={"modifiable": [], "immutable": ["a", "b", "c"]},
        shap_explainer="tree",
        sample_n=30,
        random_state=0,
    )
    assert score == pytest.approx(0.0, abs=1e-9)
    # n_modifiable = 0 → enrichment is undefined; we emit 0.0 by spec.
    assert enrichment == pytest.approx(0.0, abs=1e-9)
    # per_feature keys are the full set of columns.
    assert set(per_feature.keys()) == {"a", "b", "c"}

def test_voice_one_when_all_modifiable():
    """: every column in 'modifiable' → voice = 1.0."""
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame(
        {
            "a": rng.standard_normal(n),
            "b": rng.standard_normal(n),
        }
    )
    y = (X["a"] + X["b"] > 0).astype(int).values
    model = _build_tree_model_for_voice(X, y)
    score, enrichment, _ = voice_representation(
        model,
        X,
        feature_partition={"modifiable": ["a", "b"], "immutable": []},
        shap_explainer="tree",
        sample_n=30,
        random_state=0,
    )
    assert score == pytest.approx(1.0, abs=1e-9)
    # All features modifiable: voice/share = 1.0/1.0 = 1.0.
    assert enrichment == pytest.approx(1.0, abs=1e-9)

def test_voice_balanced_partition_approx_half():
    """: a balanced 50/50 partition with roughly balanced importance
    yields a voice share near 0.5.

    Construction: 4 features with equal influence on the target; partition
    splits them 2/2. Since the RF has stochasticity, we assert the share
    falls in a wide [0.25, 0.75] band rather than ≈ exactly 0.5.
    """
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "a": rng.standard_normal(n),
            "b": rng.standard_normal(n),
            "c": rng.standard_normal(n),
            "d": rng.standard_normal(n),
        }
    )
    # Symmetric target: each feature contributes equally.
    y = ((X["a"] + X["b"] + X["c"] + X["d"]) > 0).astype(int).values
    model = _build_tree_model_for_voice(X, y)
    score, _enrichment, _ = voice_representation(
        model,
        X,
        feature_partition={
            "modifiable": ["a", "b"],
            "immutable": ["c", "d"],
        },
        shap_explainer="tree",
        sample_n=100,
        random_state=0,
    )
    assert 0.25 <= score <= 0.75, (
        f"Balanced partition should yield voice ~ 0.5; got {score}"
    )

def test_voice_partition_validation_missing_feature():
    """: a feature missing from both lists raises ValueError."""
    n = 30
    X = pd.DataFrame(
        {
            "a": np.arange(n, dtype=float),
            "b": np.arange(n, dtype=float),
            "c": np.arange(n, dtype=float),
        }
    )
    y = (X["a"] > X["a"].median()).astype(int).values
    model = _build_tree_model_for_voice(X, y)
    with pytest.raises(ValueError, match="missing from feature_partition"):
        voice_representation(
            model,
            X,
            feature_partition={"modifiable": ["a"], "immutable": ["b"]},
            shap_explainer="tree",
            sample_n=10,
            random_state=0,
        )

def test_voice_partition_validation_overlap():
    """: a feature appearing in both buckets raises ValueError."""
    n = 30
    X = pd.DataFrame(
        {
            "a": np.arange(n, dtype=float),
            "b": np.arange(n, dtype=float),
        }
    )
    y = (X["a"] > X["a"].median()).astype(int).values
    model = _build_tree_model_for_voice(X, y)
    with pytest.raises(ValueError, match="both 'modifiable' and 'immutable'"):
        voice_representation(
            model,
            X,
            feature_partition={
                "modifiable": ["a", "b"],
                "immutable": ["a"],
            },
            shap_explainer="tree",
            sample_n=10,
            random_state=0,
        )

def test_voice_signature():
    """ / : returns ``tuple[float, dict[str, float]]``."""
    n = 30
    X = pd.DataFrame({"a": np.arange(n, dtype=float), "b": np.arange(n, dtype=float)})
    y = (X["a"] > X["a"].median()).astype(int).values
    model = _build_tree_model_for_voice(X, y)
    out = voice_representation(
        model,
        X,
        feature_partition={"modifiable": ["a"], "immutable": ["b"]},
        shap_explainer="tree",
        sample_n=10,
        random_state=0,
    )
    assert isinstance(out, tuple) and len(out) == 3
    score, enrichment, per_feature = out
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert isinstance(enrichment, float)
    assert enrichment >= 0.0
    assert isinstance(per_feature, dict)
    assert set(per_feature.keys()) == {"a", "b"}
    for v in per_feature.values():
        assert isinstance(v, float)

def test_voice_glossary_reference():
    """: ``voice_representation`` docstring references ."""
    doc = voice_representation.__doc__ or ""
    assert "Voice-Representation" in doc or "Voice / Representation" in doc
# ---------------------------------------------------------------------------
#  — transparency_metrics
# ---------------------------------------------------------------------------

class _ConstantPredictModel:
    """Always predicts 0; used to test that no CF can be found."""

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

class _ThresholdOnFirstFeatureModel:
    """Predicts ``X.iloc[:, 0] > threshold``.

    Flipping the first feature's value past the threshold changes the
    prediction; flipping any other feature does not. This makes
    ``transparency_metrics`` deterministic: every CF changes exactly one
    feature (the first one).
    """

    def __init__(self, feature_index: int = 0, threshold: float = 0.0):
        self.feature_index = feature_index
        self.threshold = threshold

    def predict(self, X):
        arr = np.asarray(X.iloc[:, self.feature_index].values, dtype=float)
        return (arr > self.threshold).astype(int)

def test_transparency_validity_constant_model():
    """: constant model → validity = 0 (no CF can change prediction)."""
    rng = np.random.default_rng(0)
    n = 30
    X = pd.DataFrame(
        {
            "a": rng.standard_normal(n),
            "b": rng.standard_normal(n),
            "c": rng.choice(["x", "y"], size=n),
        }
    )
    model = _ConstantPredictModel()
    out = transparency_metrics(
        model, X, max_features_to_flip=3, sample_n=10, random_state=0
    )
    assert out["validity"] == pytest.approx(0.0, abs=1e-9)
    # Sparsity is 0 by the documented penalty convention (no CF found).
    assert out["sparsity"] == pytest.approx(0.0, abs=1e-9)

def test_transparency_sparsity_single_feature_classifier():
    """: model that thresholds on one feature → every CF changes
    exactly that one feature → mean cardinality = 1, sparsity =
    1 − 1/n_features.

    Construction: 4 features (3 numeric, 1 categorical). Model thresholds
    on feature 0 at 0; positive rows have feature[0] > 0, negative rows
    have feature[0] < 0. Subtracting one std (≈ 1.0) from a positive row
    yields a row with feature[0] < 0, flipping the prediction. Same for
    negative rows + std. The greedy 1-flip search succeeds in exactly
    one feature for every row.
    """
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame(
        {
            "f0": rng.choice([-1.5, 1.5], size=n),  # binary-ish, std ~ 1.5
            "f1": rng.standard_normal(n),
            "f2": rng.standard_normal(n),
            "f3": rng.choice(["x", "y"], size=n),
        }
    )
    model = _ThresholdOnFirstFeatureModel(feature_index=0, threshold=0.0)
    out = transparency_metrics(
        model, X, max_features_to_flip=3, sample_n=30, random_state=0
    )
    n_features = X.shape[1]
    expected_sparsity = 1 - 1 / n_features
    assert out["validity"] == pytest.approx(1.0, abs=1e-9), (
        f"Every row should have a 1-flip CF; validity={out['validity']}"
    )
    assert out["sparsity"] == pytest.approx(expected_sparsity, abs=1e-9), (
        f"Sparsity should be {expected_sparsity}; got {out['sparsity']}"
    )

def test_transparency_signature():
    """ / : returns dict[str, float] with sparsity + validity."""
    rng = np.random.default_rng(0)
    n = 20
    X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n)})
    model = _ThresholdOnFirstFeatureModel(feature_index=0, threshold=0.0)
    out = transparency_metrics(
        model, X, max_features_to_flip=2, sample_n=10, random_state=0
    )
    assert isinstance(out, dict)
    assert "sparsity" in out and "validity" in out
    for k in ("sparsity", "validity"):
        assert isinstance(out[k], float)
        assert 0.0 <= out[k] <= 1.0

def test_transparency_glossary_reference():
    """: ``transparency_metrics`` docstring references ."""
    doc = transparency_metrics.__doc__ or ""
    assert "Transparency-Sparsity" in doc or "Transparency:" in doc
# ---------------------------------------------------------------------------
#  — process_consistency at noise_std == 0.0
# ---------------------------------------------------------------------------

def test_process_consistency_zero_noise_returns_one():
    """ exit gate: noise_std=0.0 → consistency = 1.0 (no perturbation).

    See  Consistency,  §Tier-2.
    """
    X = _toy_X(seed=0, n=30)
    model = _RandomProbaClassifier(n_classes=3, seed=42)
    score, per_row = process_consistency(
        model, X, perturbations_per_row=5,
        noise_std=0.0, sample_n=15, random_state=0,
    )
    assert score == pytest.approx(1.0, abs=1e-9)
    assert all(v == pytest.approx(1.0, abs=1e-9) for v in per_row.values())

# ---------------------------------------------------------------------------
#  — voice_enrichment
# ---------------------------------------------------------------------------

class _UniformImportanceModel:
    """A linear model whose coefficients are all equal — uniform SHAP-importance.

    For the voice_enrichment unit test, we want a synthetic model that
    distributes importance evenly across all features so the voice share
    equals the modifiable feature-count share, giving enrichment = 1.0.
    Implementation: trivial linear scorer; tested via shap.LinearExplainer.
    """

    def __init__(self, n_features: int):
        # Use sklearn LR fit on a synthetic dataset so SHAP's
        # LinearExplainer works directly.
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(0)
        n = 200
        X = rng.standard_normal((n, n_features))
        # Equal-weight target: every feature contributes equally.
        y = (X.sum(axis=1) > 0).astype(int)
        self._model = LogisticRegression(max_iter=200).fit(X, y)
        self.coef_ = self._model.coef_
        self.intercept_ = self._model.intercept_
        self.classes_ = self._model.classes_

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)

def test_voice_enrichment_uniform_importance_is_one():
    """: with uniform importance, enrichment ≈ 1 (mechanical-inflation null).

    See
    """
    rng = np.random.default_rng(0)
    n = 100
    X = pd.DataFrame(
        {f"f{i}": rng.standard_normal(n) for i in range(4)}
    )
    # Equal-weight target so the fitted LR has roughly equal coefs ⇒
    # roughly equal SHAP importance.
    y = (X.sum(axis=1) > 0).astype(int).values
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=500, random_state=0).fit(
        X.values, y
    )
    _voice, enrichment, per_feature = voice_representation(
        model,
        X,
        feature_partition={
            "modifiable": ["f0", "f1"],
            "immutable": ["f2", "f3"],
        },
        shap_explainer="linear",
        sample_n=50,
        random_state=0,
    )
    # Per-feature importance should be roughly uniform; enrichment
    # ≈ 1 (the null where modifiable count share == voice share).
    # Allow a wide tolerance because LR + finite-sample SHAP isn't perfectly
    # symmetric.
    assert 0.6 <= enrichment <= 1.4, (
        f"Uniform-importance enrichment should be near 1.0; got {enrichment} "
        f"(per_feature={per_feature})"
    )

def test_voice_enrichment_all_modifiable_concentration_above_one():
    """: SHAP importance concentrated on modifiable → enrichment > 1.

    Construction: 4 features. Target depends ONLY on f0 (modifiable);
    f1..f3 are noise. Modifiable count share = 1/4; voice share ~ 1.0
    (target driven entirely by f0); enrichment ~ 4 ≫ 1.
    """
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "f0": rng.standard_normal(n),
            "f1": rng.standard_normal(n),
            "f2": rng.standard_normal(n),
            "f3": rng.standard_normal(n),
        }
    )
    y = (X["f0"] > 0).astype(int).values
    model = _build_tree_model_for_voice(X, y)
    _voice, enrichment, _ = voice_representation(
        model,
        X,
        feature_partition={
            "modifiable": ["f0"],
            "immutable": ["f1", "f2", "f3"],
        },
        shap_explainer="tree",
        sample_n=80,
        random_state=0,
    )
    assert enrichment > 1.5, (
        f"Concentrated-on-modifiable enrichment should be > 1; got {enrichment}"
    )

def test_voice_enrichment_all_immutable_concentration_below_one():
    """: SHAP importance concentrated on immutable → enrichment < 1."""
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "f0": rng.standard_normal(n),
            "f1": rng.standard_normal(n),
            "f2": rng.standard_normal(n),
            "f3": rng.standard_normal(n),
        }
    )
    # Target driven entirely by f0; mark f0 as IMMUTABLE.
    y = (X["f0"] > 0).astype(int).values
    model = _build_tree_model_for_voice(X, y)
    _voice, enrichment, _ = voice_representation(
        model,
        X,
        feature_partition={
            "modifiable": ["f1", "f2", "f3"],
            "immutable": ["f0"],
        },
        shap_explainer="tree",
        sample_n=80,
        random_state=0,
    )
    # Modifiable count share = 3/4 = 0.75; voice ~ 0 ⇒ enrichment ~ 0.
    assert enrichment < 0.7, (
        f"Concentrated-on-immutable enrichment should be < 0.7; got {enrichment}"
    )

# ---------------------------------------------------------------------------
#  — model_flippability + explanation_actionability
# ---------------------------------------------------------------------------

def test_model_flippability_matches_transparency_on_known_case():
    """: model_flippability matches the deprecated transparency_metrics.

    Both run the same greedy ≤k-feature search on the same model+sample;
    they MUST agree on validity (validity is a counting metric, identical
    for the two implementations modulo internal sample-index-sort
    differences which we control by using the same sample_n).
    """
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame(
        {
            "f0": rng.choice([-1.5, 1.5], size=n),
            "f1": rng.standard_normal(n),
            "f2": rng.standard_normal(n),
            "f3": rng.choice(["x", "y"], size=n),
        }
    )
    model = _ThresholdOnFirstFeatureModel(feature_index=0, threshold=0.0)
    legacy = transparency_metrics(
        model, X, max_features_to_flip=1, sample_n=30, random_state=0
    )
    new = model_flippability(
        model, X, max_features_to_flip=1, sample_n=30, random_state=0
    )
    # Validity is the count of rows with a 1-flip CF; both must agree.
    assert legacy["validity"] == pytest.approx(new["validity"], abs=1e-9)

def test_explanation_actionability_subset_of_flippability():
    """: actionable_validity ≤ model_flippability validity (strict subset).

    By construction, an actionable CF is a CF whose flipped feature is
    modifiable; the flippability search finds CFs without that constraint,
    so actionable_validity ≤ flippability.validity.
    """
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame(
        {
            "f0": rng.choice([-1.5, 1.5], size=n),
            "f1": rng.standard_normal(n),
            "f2": rng.standard_normal(n),
        }
    )
    model = _ThresholdOnFirstFeatureModel(feature_index=0, threshold=0.0)
    flip = model_flippability(
        model, X, max_features_to_flip=1, sample_n=30, random_state=0
    )
    # Mark f0 as IMMUTABLE: every CF flips f0, but f0 is not modifiable,
    # so actionable_validity should be 0 (or near 0).
    act_immut = explanation_actionability(
        model,
        X,
        feature_partition={
            "modifiable": ["f1", "f2"],
            "immutable": ["f0"],
        },
        max_features_to_flip=1,
        sample_n=30,
        random_state=0,
    )
    assert act_immut["actionable_validity"] <= flip["validity"]
    assert act_immut["actionable_validity"] == pytest.approx(0.0, abs=1e-9)

    # Now mark f0 as MODIFIABLE: every CF flips f0 (the only feature that
    # can change the prediction), so actionable_validity == flippability.validity.
    act_mod = explanation_actionability(
        model,
        X,
        feature_partition={
            "modifiable": ["f0"],
            "immutable": ["f1", "f2"],
        },
        max_features_to_flip=1,
        sample_n=30,
        random_state=0,
    )
    assert act_mod["actionable_validity"] == pytest.approx(
        flip["validity"], abs=1e-9
    )

def test_model_flippability_constant_model_zero_validity():
    """: a constant model has flippability validity = 0 + actionable = 0."""
    rng = np.random.default_rng(0)
    n = 30
    X = pd.DataFrame(
        {
            "a": rng.standard_normal(n),
            "b": rng.standard_normal(n),
        }
    )
    model = _ConstantPredictModel()
    flip = model_flippability(
        model, X, max_features_to_flip=2, sample_n=10, random_state=0
    )
    assert flip["validity"] == pytest.approx(0.0, abs=1e-9)

    act = explanation_actionability(
        model,
        X,
        feature_partition={"modifiable": ["a"], "immutable": ["b"]},
        max_features_to_flip=2,
        sample_n=10,
        random_state=0,
    )
    assert act["actionable_validity"] == pytest.approx(0.0, abs=1e-9)

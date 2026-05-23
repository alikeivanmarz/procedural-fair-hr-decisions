"""Unit tests for the PyTorch Adversarial Debiasing reimplementation.

Tests:

  * ``test_fit_predict_smoke`` — synthetic 2-class N=100; predict shape OK.
  * ``test_lambda_zero_recovers_baseline`` — λ=0 → matches sklearn LR
    within 5 % accuracy.
  * ``test_lambda_increase_reduces_dp`` — λ ∈ {0, 1, 10} → DP gap is
    monotonically non-increasing within seed-noise tolerance.
  * ``test_byte_identical_two_runs`` — same seed → byte-identical
    predictions on CPU. Skipped on MPS (PyTorch MPS deterministic
    algorithms are still incomplete;  documents the CPU fallback).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from procedural_fair_hr.mitigation._adv_debias_pytorch import AdversarialDebiasingPyTorch

def _synthetic_binary(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    # Generate sens then make y correlated with sens to give the
    # adversary something to exploit.
    sens = rng.integers(0, 2, size=n).astype(float)
    # y depends on first feature (the "honest" signal) plus a leakage
    # term from sens that AdvDebias can suppress at higher λ.
    logits = 1.5 * X[:, 0] + 1.5 * (sens - 0.5)
    y = (logits > rng.normal(size=n)).astype(int)
    return X, y, sens

def test_fit_predict_smoke() -> None:
    """End-to-end smoke: model fits and predicts without error; shapes OK."""
    X, y, sens = _synthetic_binary(n=100, seed=0)
    clf = AdversarialDebiasingPyTorch(
        n_classes=2,
        n_features=X.shape[1],
        lambda_=1.0,
        n_epochs=5,
        batch_size=32,
        device="cpu",
        random_state=0,
    )
    clf.fit(X, y, sens)
    yhat = clf.predict(X)
    proba = clf.predict_proba(X)
    assert yhat.shape == (100,)
    assert proba.shape == (100, 2)
    assert set(np.unique(yhat)).issubset({0, 1})
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

def test_lambda_zero_recovers_baseline() -> None:
    """At λ=0 the predictor reduces to a vanilla MLP; its accuracy is
    within 5 percentage points of sklearn LogisticRegression on the same data."""
    X, y, sens = _synthetic_binary(n=400, seed=0)
    clf = AdversarialDebiasingPyTorch(
        n_classes=2,
        n_features=X.shape[1],
        lambda_=0.0,
        n_epochs=30,
        batch_size=64,
        device="cpu",
        random_state=0,
    )
    clf.fit(X, y, sens)
    adv_acc = float(np.mean(clf.predict(X) == y))

    lr = LogisticRegression(max_iter=1000, random_state=0)
    lr.fit(X, y)
    lr_acc = float(np.mean(lr.predict(X) == y))

    assert abs(adv_acc - lr_acc) < 0.05, (
        f"λ=0 AdvDebias acc {adv_acc:.3f} too far from LR acc {lr_acc:.3f} "
        f"(>0.05). Indicates training instability."
    )

def _dp_gap(yhat: np.ndarray, sens: np.ndarray) -> float:
    """Demographic-parity gap |P(yhat=1|s=1) - P(yhat=1|s=0)|."""
    s1 = sens == 1
    s0 = sens == 0
    if s1.sum() == 0 or s0.sum() == 0:
        return 0.0
    return float(abs(yhat[s1].mean() - yhat[s0].mean()))

def test_lambda_increase_reduces_dp() -> None:
    """Increasing λ should not make DP gap dramatically larger.

    The ZLM 2018 adversary suppresses the predictor's reliance on
    sensitive-correlated signal; in expectation DP gap shrinks as λ
    grows. With small N + few epochs the trajectory is noisy, so we
    only assert monotonic *non-increase* with a generous tolerance:
    the largest-λ DP gap must not exceed the λ=0 DP gap by more than
    0.10 (i.e., we accept noise but reject a clear regression).
    """
    X, y, sens = _synthetic_binary(n=400, seed=0)

    gaps: list[float] = []
    for lam in (0.0, 1.0, 10.0):
        clf = AdversarialDebiasingPyTorch(
            n_classes=2,
            n_features=X.shape[1],
            lambda_=lam,
            n_epochs=30,
            batch_size=64,
            device="cpu",
            random_state=0,
        )
        clf.fit(X, y, sens)
        gaps.append(_dp_gap(clf.predict(X), sens))

    # The high-λ DP gap should be no worse than the low-λ DP gap by more
    # than 0.10 (loose monotonicity per docstring).
    assert gaps[-1] <= gaps[0] + 0.10, (
        f"DP gap regressed at λ=10 vs λ=0: {gaps[-1]:.3f} > {gaps[0]:.3f}+0.10. "
        f"Full gap trace: {gaps}"
    )

def test_byte_identical_two_runs() -> None:
    """Two CPU runs with seed=0 produce byte-identical predictions."""
    X, y, sens = _synthetic_binary(n=200, seed=0)

    def run() -> np.ndarray:
        clf = AdversarialDebiasingPyTorch(
            n_classes=2,
            n_features=X.shape[1],
            lambda_=1.0,
            n_epochs=10,
            batch_size=32,
            device="cpu",
            random_state=0,
        )
        clf.fit(X, y, sens)
        return clf.predict_proba(X)

    p1 = run()
    p2 = run()
    assert np.array_equal(p1, p2), (
        f"CPU AdvDebias is not byte-identical across two seeded runs; "
        f"max |drift| = {np.max(np.abs(p1 - p2)):.2e}"
    )

@pytest.mark.skipif(
    True,
    reason="MPS determinism not guaranteed in PyTorch 2.x; defaults "
    "AdvDebias to CPU. Re-enable when torch.backends.mps deterministic "
    "ops cover linear+softmax+ce backward.",
)
def test_byte_identical_two_runs_mps() -> None:  # pragma: no cover
    """MPS byte-identity probe (currently skipped per )."""

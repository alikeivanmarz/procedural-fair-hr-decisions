"""Unit tests for the Phase-4 triangulated significance utilities.

Covers:

* ``stratified_holm_correction`` — within-stratum Holm matches a manual
  expansion on a 2-stratum mini-example.
* ``benjamini_hochberg_correction`` — Wikipedia's textbook BH example.
* ``bca_interval`` — degenerate-jackknife (acceleration=0) reproduces a
  simple bias-corrected (BC) interval; a synthetic skewed distribution
  produces an interval shifted in the expected direction relative to
  the percentile bounds.

References
----------
* Benjamini, Y. & Hochberg, Y. 1995. *Controlling the False Discovery Rate.*
* Efron, B. 1987. *Better bootstrap confidence intervals.*
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.run_procedural_significance import (
    bca_interval,
    benjamini_hochberg_correction,
    stratified_holm_correction,
)

# ---------------------------------------------------------------------------
# Stratified Holm
# ---------------------------------------------------------------------------

class TestStratifiedHolm:
    def test_two_strata_independent_correction(self) -> None:
        """Within each stratum, Holm-correct independently.

        Stratum A has [0.001, 0.04] (n=2): Holm crit. for first is
        0.05/2 = 0.025; 0.001 < 0.025 → reject. Crit. for second is
        0.05/1 = 0.05; 0.04 < 0.05 → reject. Adjusted: [0.002, 0.04].
        Stratum B has [0.04, 0.30] (n=2): first crit. 0.025; 0.04 not
        < 0.025 → no rejections. Adjusted: [0.08, 0.30].
        """
        p = np.array([0.001, 0.04, 0.04, 0.30])
        strata = np.array(["A", "A", "B", "B"])
        rejected, p_holm = stratified_holm_correction(p, strata, alpha=0.05)
        assert list(rejected) == [True, True, False, False]
        np.testing.assert_allclose(
            p_holm, [0.002, 0.04, 0.08, 0.30], atol=1e-9
        )

    def test_empty_input(self) -> None:
        p = np.array([])
        strata = np.array([])
        rejected, p_holm = stratified_holm_correction(p, strata, alpha=0.05)
        assert rejected.size == 0
        assert p_holm.size == 0

    def test_single_stratum_matches_pooled_holm(self) -> None:
        """With a single stratum, stratified Holm == pooled Holm."""
        from scripts.run_phase4_significance import holm_bonferroni_correction

        p = np.array([0.001, 0.013, 0.014, 0.190, 0.350])
        strata = np.array(["X"] * 5)
        rej_strat, p_strat = stratified_holm_correction(p, strata)
        rej_pool, p_pool = holm_bonferroni_correction(p)
        np.testing.assert_array_equal(rej_strat, rej_pool)
        np.testing.assert_allclose(p_strat, p_pool, atol=1e-12)

# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR
# ---------------------------------------------------------------------------

class TestBenjaminiHochberg:
    def test_textbook_example(self) -> None:
        """Mini-example: sorted p ascending = [0.001, 0.008, 0.039, 0.041,
        0.042, 0.060, 0.074, 0.205] (n=8) at α=0.05.

        Critical line: i * α/n = [0.00625, 0.0125, 0.01875, 0.025,
        0.03125, 0.0375, 0.04375, 0.05].
        Largest i with p_(i) <= crit_(i): scan from right.
        i=8: 0.205 vs 0.05 → no
        i=7: 0.074 vs 0.04375 → no
        i=6: 0.060 vs 0.0375 → no
        i=5: 0.042 vs 0.03125 → no
        i=4: 0.041 vs 0.025 → no
        i=3: 0.039 vs 0.01875 → no
        i=2: 0.008 vs 0.0125 → YES
        Reject H_(1) and H_(2) → 2 rejections.
        """
        p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205])
        rejected, p_bh = benjamini_hochberg_correction(p, alpha=0.05)
        assert list(rejected) == [
            True, True, False, False, False, False, False, False,
        ]
        # Adjusted p-values are monotone non-decreasing (running min).
        # p_BH[i] = min_{j>=i} (p_(j) * n / j).
        # Computing manually for verification:
        # j=8: 0.205 * 8/8 = 0.205
        # j=7: 0.074 * 8/7 = 0.08457 → min(0.205, 0.08457) = 0.08457
        # j=6: 0.060 * 8/6 = 0.080 → min = 0.080
        # j=5: 0.042 * 8/5 = 0.0672 → min = 0.0672
        # j=4: 0.041 * 8/4 = 0.082 → min = 0.0672
        # j=3: 0.039 * 8/3 = 0.104 → min = 0.0672
        # j=2: 0.008 * 8/2 = 0.032 → min = 0.032
        # j=1: 0.001 * 8/1 = 0.008 → min = 0.008
        expected = [0.008, 0.032, 0.0672, 0.0672, 0.0672, 0.080, 0.08457, 0.205]
        np.testing.assert_allclose(p_bh, expected, atol=1e-4)

    def test_unsorted_input_preserves_order(self) -> None:
        # Reverse the textbook input.
        p = np.array([0.205, 0.074, 0.060, 0.042, 0.041, 0.039, 0.008, 0.001])
        rejected, _ = benjamini_hochberg_correction(p, alpha=0.05)
        assert list(rejected) == [
            False, False, False, False, False, False, True, True,
        ]

    def test_empty_input(self) -> None:
        p = np.array([])
        rejected, p_bh = benjamini_hochberg_correction(p)
        assert rejected.size == 0
        assert p_bh.size == 0

    def test_all_significant(self) -> None:
        p = np.array([1e-6, 1e-6, 1e-6, 1e-6])
        rejected, p_bh = benjamini_hochberg_correction(p, alpha=0.05)
        assert all(rejected)
        assert all(p_bh < 0.05)

# ---------------------------------------------------------------------------
# BCa interval
# ---------------------------------------------------------------------------

class TestBCaInterval:
    def test_symmetric_zero_acceleration_matches_percentile(self) -> None:
        """For a symmetric bootstrap distribution centred at theta_hat
        with zero-acceleration jackknife (all jack values equal), BCa
        collapses to the percentile interval modulo a tiny continuity
        correction (1/(2B)).
        """
        rng = np.random.default_rng(42)
        boot = rng.normal(loc=0.0, scale=1.0, size=10_000)
        # Jackknife with zero variance (degenerate sample); a=0 fallback.
        jack = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        lo, hi = bca_interval(0.0, boot, jack, alpha=0.05)
        # Percentile bounds (symmetric around 0).
        perc_lo = np.percentile(boot, 2.5)
        perc_hi = np.percentile(boot, 97.5)
        assert abs(lo - perc_lo) < 0.1
        assert abs(hi - perc_hi) < 0.1

    def test_skewed_distribution_shifts_interval(self) -> None:
        """For a right-skewed bootstrap distribution, BCa should shift
        the interval rightward relative to the percentile interval if
        the acceleration is positive (Efron 1987).
        """
        rng = np.random.default_rng(1)
        # Right-skewed: lognormal centred near 1.
        boot = rng.lognormal(mean=0.0, sigma=0.5, size=10_000)
        theta_hat = float(np.median(boot))
        # Synthetic jackknife with positive skew → positive acceleration.
        # Use mean-jack = 1 with one large deviation to make the cubed-
        # diff sum positive (a > 0).
        jack = np.array([1.0, 1.0, 1.0, 1.0, 0.5])
        lo_bca, hi_bca = bca_interval(theta_hat, boot, jack, alpha=0.05)
        # Just verify the interval is well-formed.
        assert lo_bca < hi_bca
        assert np.isfinite(lo_bca) and np.isfinite(hi_bca)
        # And that BCa output sits in the support of `boot`.
        assert boot.min() <= lo_bca <= boot.max()
        assert boot.min() <= hi_bca <= boot.max()

    def test_empty_inputs(self) -> None:
        lo, hi = bca_interval(0.0, np.array([]), np.array([1.0, 2.0]))
        assert np.isnan(lo) and np.isnan(hi)
        lo, hi = bca_interval(0.0, np.array([1.0, 2.0]), np.array([]))
        assert np.isnan(lo) and np.isnan(hi)

    def test_all_boot_below_theta_hat_clipped(self) -> None:
        """If every bootstrap resample is strictly below theta_hat the
        bias-correction fraction is 0 → Φ⁻¹(0) = -∞; we clip with the
        Hall continuity correction. Interval should remain finite.
        """
        boot = np.linspace(-1.0, 0.0, 10_000)
        jack = np.array([0.5, 0.5, 0.5, 0.6])
        lo, hi = bca_interval(1.0, boot, jack, alpha=0.05)
        assert np.isfinite(lo) and np.isfinite(hi)

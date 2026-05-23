"""Phase-4 followup — triangulated multiple-comparison significance.

Extends the standard ``run_phase4_significance.py`` output
(``results/phase4/significance_n30.csv``) along two axes that the existing
N=30 file cannot answer from stored summaries alone:

1. **Bias-corrected accelerated (BCa) bootstrap intervals** (Efron 1987,
   *JASA* 82(397)): require per-resample bootstrap statistics + a
   jackknife over the original sample, neither of which is preserved by
   the standard percentile-only output. We therefore re-run the paired
   bootstrap with the same RNG seed (`seed=0` for procedural_gap,
   `seed=1` for separability_gap, `seed=2` for rank_disagreement) and
   compute BCa on the captured samples.
2. **Three multiple-comparison procedures** applied side-by-side:

   * ``p_holm_pooled``: Holm step-down across the full 930-test family
     (the existing convention from ``significance_n30.csv``; equivalent
     to ``p_holm``).
   * ``p_holm_stratified``: Holm step-down within each weighting scheme
     stratum (5 strata × 186 tests = 930 total; Westfall & Young 1993
     §2.3.5 — stratified Holm preserves family-wise α inside each
     scheme but does not pool across schemes).
   * ``p_bh_fdr``: Benjamini-Hochberg FDR (Benjamini & Hochberg 1995,
     *J. R. Stat. Soc. B* 57(1)) applied to the pooled 930-test family
     at α=0.05.

Output schema (strict superset of ``significance_n30.csv`` per ):
the 15 existing columns are preserved in order, then the 8 new columns
are appended:
``bca_lo, bca_hi, p_holm_pooled, p_holm_stratified, p_bh_fdr,
rejected_pooled, rejected_stratified, rejected_fdr``.
``p_holm_pooled == p_holm`` and ``rejected_pooled == rejected_holm`` by
construction (the pooled Holm correction is the same procedure as the
original); the rename is provided for clarity in the triangulation.

A second output, ``results/phase4/headline_separability_triangulated.csv``,
reports REJECT / NOT-REJECT of the headline hypothesis
:math:`H_0 : \\rho \\ge 0.7` (procedural-vs-statistical Spearman rank
correlation) per (dataset × weighting_scheme × correction-procedure),
under each of the three multiple-comparison procedures AND the BCa
percentile (BCa upper bound < 0.7 → reject).

References
----------
* Efron, B. 1987. *Better bootstrap confidence intervals.* JASA 82(397):171-185.
* Benjamini, Y. & Hochberg, Y. 1995. *Controlling the False Discovery Rate.*
  J. R. Stat. Soc. B 57(1):289-300.
* Holm, S. 1979. *A simple sequentially rejective multiple test procedure.*
  Scandinavian J. Statistics 6:65-70.
* Westfall, P.H. & Young, S.S. 1993. *Resampling-based multiple testing.* Wiley.

CLI
---
.. code-block:: bash

    python scripts/run_phase4_significance_triangulated.py \\
        [--procedural-csv results/phase4/procedural_n30.csv] \\
        [--baseline-csv results/phase4/significance_n30.csv] \\
        [--out-dir results/phase4]
"""

from __future__ import annotations

import os

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import argparse  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm, spearmanr  # noqa: E402

# Resolve project root so we can re-use the production script's helpers.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._procedural_significance_helpers import (  # noqa: E402
    COHENS_D_VARIANCE_FLOOR,
    MODEL_ORDER,
    TRAINED_MODELS,
    WEIGHTING_SCHEMES,
    _per_seed_procedural_aggregate,
    _statistical_rank_per_dataset,
    _superiority_probability,
    holm_bonferroni_correction,
)

# ---------------------------------------------------------------------------
#  — multiple-comparison procedures: stratified Holm + BH-FDR.
# ---------------------------------------------------------------------------

def stratified_holm_correction(
    p_values: np.ndarray,
    strata: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Holm step-down correction applied independently within each stratum.

    Parameters
    ----------
    p_values : array of float, shape (n,)
        Raw two-sided p-values. NaNs pass through to ``rejected[i]=False``
        and ``p_holm[i]=NaN``.
    strata : array of object, shape (n,)
        Stratum label per test. Tests sharing a label form a sub-family.
    alpha : float
        Family-wise α controlled *within each stratum* (Westfall & Young
        1993 §2.3.5).

    Returns
    -------
    (rejected_mask, p_stratified_holm) — aligned with the input order.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    rejected = np.zeros(n, dtype=bool)
    p_corrected = np.full(n, np.nan, dtype=float)
    if n == 0:
        return rejected, p_corrected
    strata = np.asarray(strata)
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        sub_p = p[idx]
        sub_rej, sub_p_holm = holm_bonferroni_correction(sub_p, alpha=alpha)
        rejected[idx] = sub_rej
        p_corrected[idx] = sub_p_holm
    return rejected, p_corrected

def benjamini_hochberg_correction(
    p_values: np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR step-up correction (BH 1995).

    Procedure:
      1. Sort the n p-values ascending (NaNs sort last).
      2. For sorted position i (1-indexed), critical value is
         ``alpha * i / n``.
      3. Find the largest i with ``p_(i) <= alpha * i / n``; reject all
         hypotheses with rank ≤ that i.
      4. Adjusted p-value is ``p_BH[i] = min_{j>=i} ( p_(j) * n / j )``
         clipped to [0, 1] (Benjamini-Yekutieli 2001 convention).

    Returns
    -------
    (rejected_mask, p_bh) — aligned with the input order.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    rejected = np.zeros(n, dtype=bool)
    p_bh = np.full(n, np.nan, dtype=float)
    if n == 0:
        return rejected, p_bh
    finite_mask = ~np.isnan(p)
    finite_idx = np.where(finite_mask)[0]
    if finite_idx.size == 0:
        return rejected, p_bh
    order = finite_idx[np.argsort(p[finite_idx], kind="stable")]
    m = len(order)
    # Adjusted p-values: scan from largest p to smallest, maintaining the
    # running min of ``p_(j) * m / j`` (j is the sorted rank, 1-indexed).
    running_min = 1.0
    for sorted_pos in range(m - 1, -1, -1):
        j = sorted_pos + 1  # 1-indexed rank
        idx = order[sorted_pos]
        adj = p[idx] * m / j
        if adj < running_min:
            running_min = adj
        p_bh[idx] = min(1.0, running_min)
    # Reject H_(i) iff p_BH_(i) <= alpha.
    for sorted_pos in range(m):
        idx = order[sorted_pos]
        if p_bh[idx] <= alpha:
            rejected[idx] = True
    return rejected, p_bh

# ---------------------------------------------------------------------------
#  — BCa interval (Efron 1987).
# ---------------------------------------------------------------------------

def bca_interval(
    theta_hat: float,
    boot_samples: np.ndarray,
    jackknife_samples: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bias-corrected accelerated (BCa) bootstrap interval.

    Parameters
    ----------
    theta_hat : float
        Point estimate of the statistic on the original sample.
    boot_samples : array of float, shape (B,)
        Bootstrap distribution of the statistic. Must have ``B >= 1``.
    jackknife_samples : array of float, shape (n,)
        Leave-one-out estimates of the statistic on the original sample
        of size n (delete-1 jackknife, NOT bootstrap resampling).
    alpha : float
        Two-sided significance level; the interval has confidence
        ``1 - alpha``.

    Returns
    -------
    (bca_lo, bca_hi) : lower and upper BCa interval bounds.

    Notes
    -----
    Formulae (Efron 1987 eqs. 2.7, 2.8):

    .. math::
       z_0 = \\Phi^{-1}\\!\\left( \\frac{\\#\\{b : \\hat\\theta^*_b
       < \\hat\\theta\\}}{B} \\right)

    .. math::
       a = \\frac{ \\sum_i (\\bar{\\hat\\theta_{(\\cdot)}} -
       \\hat\\theta_{(i)})^3 }{ 6 \\left( \\sum_i
       (\\bar{\\hat\\theta_{(\\cdot)}} - \\hat\\theta_{(i)})^2
       \\right)^{3/2} }

    .. math::
       \\alpha_1 = \\Phi\\!\\left( z_0 + \\frac{z_0 + z_{\\alpha/2}}
       {1 - a (z_0 + z_{\\alpha/2})} \\right), \\quad
       \\alpha_2 = \\Phi\\!\\left( z_0 + \\frac{z_0 + z_{1 - \\alpha/2}}
       {1 - a (z_0 + z_{1 - \\alpha/2})} \\right)

    Edge cases:
      * If all bootstrap samples are < or > theta_hat, the fraction is
        0 or 1 and Φ⁻¹ is ±∞. We clip the fraction to ``(1/(2B), 1 -
        1/(2B))`` to keep z_0 finite (Hall 1992 continuity correction).
      * If the jackknife variance is 0 (statistic constant under
        leave-one-out, e.g., a degenerate sample), we fall back to a
        bias-corrected (BC) interval (set a = 0).
      * If a denominator ``1 - a (z_0 + z_α)`` is non-positive, we clip
        α_1 / α_2 to [0, 1] so the percentile lookup remains valid; this
        is the standard "BCa is unstable" warning regime (Efron &
        Tibshirani 1993 §14.3) and we flag it via the return value
        (caller can compare bca_lo, bca_hi to NaN if both endpoints
        collapse to the same percentile).
    """
    boot = np.asarray(boot_samples, dtype=float)
    jack = np.asarray(jackknife_samples, dtype=float)
    B = boot.size
    if B == 0 or jack.size == 0 or not np.isfinite(theta_hat):
        return float("nan"), float("nan")

    # z_0: bias-correction constant.
    frac_below = float((boot < theta_hat).sum()) / B
    # Continuity-correct to avoid Φ⁻¹(0) / Φ⁻¹(1).
    eps = 1.0 / (2.0 * B)
    frac_below = min(max(frac_below, eps), 1.0 - eps)
    z0 = float(norm.ppf(frac_below))

    # a: acceleration constant from delta jackknife.
    jack_mean = float(np.mean(jack))
    diffs = jack_mean - jack
    num = float(np.sum(diffs ** 3))
    den = 6.0 * (float(np.sum(diffs ** 2)) ** 1.5)
    if den == 0.0:
        a = 0.0  # fall back to BC.
    else:
        a = num / den

    z_lo = float(norm.ppf(alpha / 2.0))
    z_hi = float(norm.ppf(1.0 - alpha / 2.0))

    def _alpha_bca(z_score: float) -> float:
        denom = 1.0 - a * (z0 + z_score)
        if denom <= 0:
            # Degenerate regime; clip to (0, 1).
            return float("nan")
        return float(norm.cdf(z0 + (z0 + z_score) / denom))

    a1 = _alpha_bca(z_lo)
    a2 = _alpha_bca(z_hi)
    if np.isnan(a1):
        a1 = 0.0
    if np.isnan(a2):
        a2 = 1.0
    a1 = min(max(a1, 0.0), 1.0)
    a2 = min(max(a2, 0.0), 1.0)
    lo = float(np.percentile(boot, 100.0 * a1))
    hi = float(np.percentile(boot, 100.0 * a2))
    if lo > hi:
        # If endpoints crossed (extreme regime), swap to keep
        # lo <= hi convention.
        lo, hi = hi, lo
    return lo, hi

# ---------------------------------------------------------------------------
# Paired bootstrap with stored samples + jackknife (for BCa).
# ---------------------------------------------------------------------------

def _paired_bootstrap_with_samples(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Reimplements ``_paired_bootstrap_gap`` but stores per-resample
    bootstrap statistics + jackknife so we can compute BCa.

    Statistic of interest: ``mean(a - b)`` (the paired mean gap).

    Output keys:
      * ``mean``, ``ci_lo``, ``ci_hi``, ``p_value``, ``cohens_d``,
        ``effect_size_kind``, ``n_bootstrap``, ``notes`` — match the
        production helper byte-for-byte under the same RNG seed (this is
        critical so ``p_holm_pooled`` reproduces the existing CSV).
      * ``boot_samples`` — length-B numpy array of bootstrap means.
      * ``jackknife`` — length-n numpy array of leave-one-out means.
      * ``theta_hat`` — observed ``mean(a - b)`` on the full sample.
      * ``bca_lo``, ``bca_hi`` — BCa-95 % interval bounds.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)
    if n == 0:
        return {
            "mean": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "p_value": float("nan"),
            "cohens_d": float("nan"),
            "effect_size_kind": "undefined",
            "n_bootstrap": n_boot,
            "notes": "no overlapping non-NaN seeds",
            "boot_samples": np.array([], dtype=float),
            "jackknife": np.array([], dtype=float),
            "theta_hat": float("nan"),
            "bca_lo": float("nan"),
            "bca_hi": float("nan"),
        }
    diffs = a - b
    obs = float(np.mean(diffs))
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)
    # Match production helper's resampling loop byte-for-byte
    # (rng.integers in the same order with the same shape).
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = float(np.mean(diffs[idx]))
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    # Two-sided p-value (same convention as production helper).
    if obs >= 0:
        p_one = float((boot_means <= 0).mean())
    else:
        p_one = float((boot_means >= 0).mean())
    p_value = min(1.0, 2.0 * p_one)

    # Variance-floor + effect-size policy (mirror production).
    sd_diffs = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    a_sd = float(np.std(a, ddof=1)) if n > 1 else 0.0
    b_sd = float(np.std(b, ddof=1)) if n > 1 else 0.0
    DETERMINISTIC_SD = 0.1 * COHENS_D_VARIANCE_FLOOR
    notes_parts: list[str] = []
    if a_sd < DETERMINISTIC_SD:
        notes_parts.append(
            f"model_a effectively deterministic across seeds (sd={a_sd:.2g})"
        )
    if b_sd < DETERMINISTIC_SD:
        notes_parts.append(
            f"model_b effectively deterministic across seeds (sd={b_sd:.2g})"
        )
    if sd_diffs == 0.0 and obs == 0.0:
        cohens_d = float("nan")
        kind = "undefined"
        notes_parts.append("identical per-seed values; effect undefined")
    elif sd_diffs == 0.0:
        cohens_d = _superiority_probability(diffs)
        kind = "superiority_prob"
        notes_parts.append(
            "pooled sd(diffs)=0; reporting probability of superiority "
            "instead of Cohen's d"
        )
    elif sd_diffs < COHENS_D_VARIANCE_FLOOR:
        cohens_d = obs / COHENS_D_VARIANCE_FLOOR
        kind = "cohens_d_floored"
        notes_parts.append(
            f"sd(diffs)={sd_diffs:.4g} < floor={COHENS_D_VARIANCE_FLOOR}; "
            "Cohen's d denominator floored"
        )
    else:
        cohens_d = obs / sd_diffs
        kind = "cohens_d"

    # Jackknife — delete-1 over the n paired diffs.
    if n >= 2:
        jack = np.empty(n, dtype=float)
        # Mean-of-leave-one-out = (n * mean - x_i) / (n - 1)
        total = float(np.sum(diffs))
        for i in range(n):
            jack[i] = (total - diffs[i]) / (n - 1)
    else:
        jack = np.array([], dtype=float)

    bca_lo, bca_hi = bca_interval(obs, boot_means, jack, alpha=0.05)

    return {
        "mean": obs,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "effect_size_kind": kind,
        "n_bootstrap": n_boot,
        "notes": "; ".join(notes_parts),
        "boot_samples": boot_means,
        "jackknife": jack,
        "theta_hat": obs,
        "bca_lo": bca_lo,
        "bca_hi": bca_hi,
    }

# ---------------------------------------------------------------------------
# Spearman-ρ bootstrap with stored samples + headline H0: ρ >= 0.7.
# ---------------------------------------------------------------------------

def _spearman_rho_bootstrap_with_samples(
    proc_rank_per_seed: dict[int, dict[str, int]],
    stat_rank: dict[str, int],
    *,
    n_boot: int = 10_000,
    seed: int = 2,
    h0_threshold: float = 0.7,
) -> dict[str, Any]:
    """Bootstrap Spearman ρ between procedural-rank and statistical-rank
    vectors, storing per-resample bootstrap means and a delete-1
    jackknife over the per-seed ρ values (so BCa is computable).

    Returns:
      * ``mean``, ``ci_lo``, ``ci_hi`` — percentile bootstrap (matches
        existing CSV byte-for-byte under same RNG seed).
      * ``p_value`` — original ``H₀: ρ=1`` test (existing convention,
        retained for back-compat with the baseline CSV).
      * ``p_value_h07`` — NEW one-sided test for ``H₀: ρ >= 0.7`` vs
        ``H₁: ρ < 0.7``. Computed as the fraction of bootstrap means
        that are >= 0.7 (i.e., compatible with the null).
      * ``bca_lo``, ``bca_hi`` — BCa-95 % bounds.
      * ``boot_samples``, ``jackknife``, ``theta_hat`` — internal.
    """
    seeds = sorted(proc_rank_per_seed.keys())
    if not seeds:
        return {
            "mean": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "p_value": float("nan"),
            "p_value_h07": float("nan"),
            "n_bootstrap": n_boot,
            "boot_samples": np.array([], dtype=float),
            "jackknife": np.array([], dtype=float),
            "theta_hat": float("nan"),
            "bca_lo": float("nan"),
            "bca_hi": float("nan"),
        }
    per_seed_rho: list[float] = []
    for s in seeds:
        pr = proc_rank_per_seed[s]
        shared = sorted(set(pr.keys()) & set(stat_rank.keys()))
        if len(shared) < 2:
            continue
        pa = np.asarray([pr[m] for m in shared], dtype=float)
        sa = np.asarray([stat_rank[m] for m in shared], dtype=float)
        try:
            rho = float(spearmanr(pa, sa).statistic)
        except Exception:
            continue
        if np.isnan(rho):
            continue
        per_seed_rho.append(rho)
    per_seed_rho_arr = np.asarray(per_seed_rho, dtype=float)
    if per_seed_rho_arr.size == 0:
        return {
            "mean": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "p_value": float("nan"),
            "p_value_h07": float("nan"),
            "n_bootstrap": n_boot,
            "boot_samples": np.array([], dtype=float),
            "jackknife": np.array([], dtype=float),
            "theta_hat": float("nan"),
            "bca_lo": float("nan"),
            "bca_hi": float("nan"),
        }
    rng = np.random.default_rng(seed)
    n = per_seed_rho_arr.size
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(per_seed_rho_arr[idx]))
    mean_rho = float(np.mean(per_seed_rho_arr))
    ci_lo = float(np.percentile(boot, 2.5))
    ci_hi = float(np.percentile(boot, 97.5))
    # Original convention: H₀: ρ=1; one-sided p = P(boot >= 1).
    p_value_rho1 = float((boot >= 1.0).mean())
    # NEW headline: H₀: ρ >= h0_threshold vs H₁: ρ < h0_threshold.
    # Reject H₀ if upper-tail support at h0_threshold is small, i.e.
    # if very few bootstrap resamples exceed the threshold.
    # Bootstrap p = P(boot >= h0_threshold).
    p_value_h07 = float((boot >= h0_threshold).mean())

    # Jackknife — delete-1 on per_seed_rho.
    if n >= 2:
        total = float(np.sum(per_seed_rho_arr))
        jack = np.empty(n, dtype=float)
        for i in range(n):
            jack[i] = (total - per_seed_rho_arr[i]) / (n - 1)
    else:
        jack = np.array([], dtype=float)

    bca_lo, bca_hi = bca_interval(mean_rho, boot, jack, alpha=0.05)

    return {
        "mean": mean_rho,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": p_value_rho1,
        "p_value_h07": p_value_h07,
        "n_bootstrap": n_boot,
        "boot_samples": boot,
        "jackknife": jack,
        "theta_hat": mean_rho,
        "bca_lo": bca_lo,
        "bca_hi": bca_hi,
    }

# ---------------------------------------------------------------------------
# Top-level driver — re-walks the (dataset × scheme × pair) combinations
# under the same RNG seeds as the production script so byte-for-byte
# parity with significance_n30.csv is achievable on the shared columns.
# ---------------------------------------------------------------------------

def compute_triangulated_table(
    proc_df: pd.DataFrame, tpr_df: pd.DataFrame, log
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the full triangulated significance table + headline
    separability summary.

    Returns
    -------
    (sig_df_triangulated, headline_df)
    """
    rows: list[dict[str, Any]] = []
    # We also accumulate one rho_stats dict per (dataset, scheme) so we
    # can write the headline file at the end without re-running the
    # bootstrap.
    rho_lookup: dict[tuple[str, str], dict[str, Any]] = {}

    datasets = sorted(proc_df["dataset"].unique())
    schemes = list(WEIGHTING_SCHEMES.keys())
    for ds in datasets:
        for scheme in schemes:
            log(f"triangulated / dataset={ds} / scheme={scheme}")
            per_seed: dict[str, np.ndarray] = {}
            for m in TRAINED_MODELS:
                agg = _per_seed_procedural_aggregate(
                    proc_df, ds, m, weighting_scheme=scheme
                )
                per_seed[m] = agg

            models = [
                m for m in TRAINED_MODELS if not np.all(np.isnan(per_seed[m]))
            ]

            # Pairwise procedural-gap (seed=0 — matches production).
            for i, ma in enumerate(models):
                for mb in models[i + 1:]:
                    stats = _paired_bootstrap_with_samples(
                        per_seed[ma], per_seed[mb], n_boot=10_000, seed=0,
                    )
                    rows.append({
                        "dataset": ds,
                        "weighting_scheme": scheme,
                        "model_a": ma,
                        "model_b": mb,
                        "comparison_type": "procedural_gap",
                        "mean": stats["mean"],
                        "ci_lo": stats["ci_lo"],
                        "ci_hi": stats["ci_hi"],
                        "cohens_d": stats["cohens_d"],
                        "effect_size_kind": stats["effect_size_kind"],
                        "p_value": stats["p_value"],
                        "n_bootstrap": stats["n_bootstrap"],
                        "notes": stats["notes"],
                        "bca_lo": stats["bca_lo"],
                        "bca_hi": stats["bca_hi"],
                    })

            # Separability-gap bootstrap (seed=1 — matches production).
            stat_rank = _statistical_rank_per_dataset(tpr_df, ds)
            if not stat_rank:
                log(f"  no statistical rank for {ds}; skipping separability rows")
                continue
            seeds = sorted(int(s) for s in proc_df["seed"].unique() if s >= 0)
            proc_rank_by_seed: dict[int, dict[str, int]] = {}
            for s in seeds:
                seed_means: list[tuple[str, float]] = []
                for m in TRAINED_MODELS:
                    if m not in per_seed or np.isnan(per_seed[m]).all():
                        continue
                    idx = seeds.index(s) if s in seeds else None
                    if idx is None or idx >= len(per_seed[m]):
                        continue
                    v = per_seed[m][idx]
                    if not np.isnan(v):
                        seed_means.append((m, float(v)))
                seed_means.sort(key=lambda kv: -kv[1])
                proc_rank_by_seed[s] = {
                    m: r for r, (m, _) in enumerate(seed_means, start=1)
                }

            for i, ma in enumerate(models):
                if ma not in stat_rank:
                    continue
                for mb in models[i + 1:]:
                    if mb not in stat_rank:
                        continue
                    stat_diff = stat_rank[ma] - stat_rank[mb]
                    a_seed: list[float] = []
                    b_seed: list[float] = []
                    for s in seeds:
                        pr = proc_rank_by_seed.get(s, {})
                        if ma in pr and mb in pr:
                            proc_diff = pr[ma] - pr[mb]
                            a_seed.append(float(proc_diff - stat_diff))
                            b_seed.append(0.0)
                    stats = _paired_bootstrap_with_samples(
                        np.asarray(a_seed), np.asarray(b_seed),
                        n_boot=10_000, seed=1,
                    )
                    rows.append({
                        "dataset": ds,
                        "weighting_scheme": scheme,
                        "model_a": ma,
                        "model_b": mb,
                        "comparison_type": "separability_gap",
                        "mean": stats["mean"],
                        "ci_lo": stats["ci_lo"],
                        "ci_hi": stats["ci_hi"],
                        "cohens_d": stats["cohens_d"],
                        "effect_size_kind": stats["effect_size_kind"],
                        "p_value": stats["p_value"],
                        "n_bootstrap": stats["n_bootstrap"],
                        "notes": stats["notes"],
                        "bca_lo": stats["bca_lo"],
                        "bca_hi": stats["bca_hi"],
                    })

            # Spearman-ρ rank-disagreement (seed=2 — matches production).
            rho_stats = _spearman_rho_bootstrap_with_samples(
                proc_rank_by_seed, stat_rank,
                n_boot=10_000, seed=2, h0_threshold=0.7,
            )
            rho_lookup[(ds, scheme)] = rho_stats
            rows.append({
                "dataset": ds,
                "weighting_scheme": scheme,
                "model_a": None,
                "model_b": None,
                "comparison_type": "rank_disagreement",
                "mean": rho_stats["mean"],
                "ci_lo": rho_stats["ci_lo"],
                "ci_hi": rho_stats["ci_hi"],
                "cohens_d": float("nan"),
                "effect_size_kind": "spearman_rho",
                "p_value": rho_stats["p_value"],
                "n_bootstrap": rho_stats["n_bootstrap"],
                "notes": (
                    "Spearman ρ between procedural-rank and "
                    "statistical-rank vectors over trained models; "
                    "H₀: ρ=1"
                ),
                "bca_lo": rho_stats["bca_lo"],
                "bca_hi": rho_stats["bca_hi"],
            })

    sig_df = pd.DataFrame(rows)

    # Stable sort to mirror the production CSV's row order.
    sig_df = sig_df.sort_values(
        ["dataset", "weighting_scheme", "comparison_type", "model_a", "model_b"],
        na_position="last",
    ).reset_index(drop=True)

    # ---------- Multiple-comparison corrections ----------------------------
    p_values = sig_df["p_value"].to_numpy()

    # Pooled Holm (the existing convention from significance_n30.csv).
    rej_pool, p_holm_pool = holm_bonferroni_correction(p_values, alpha=0.05)

    # Stratified Holm (one stratum per weighting_scheme).
    strata = sig_df["weighting_scheme"].astype(str).to_numpy()
    rej_strat, p_holm_strat = stratified_holm_correction(
        p_values, strata, alpha=0.05
    )

    # BH-FDR (pooled).
    rej_fdr, p_fdr = benjamini_hochberg_correction(p_values, alpha=0.05)

    # Existing 15-col schema preservation: keep `p_holm` + `rejected_holm`.
    sig_df["p_holm"] = p_holm_pool
    sig_df["rejected_holm"] = rej_pool

    sig_df["bca_lo"] = sig_df["bca_lo"]  # already present
    sig_df["bca_hi"] = sig_df["bca_hi"]  # already present
    sig_df["p_holm_pooled"] = p_holm_pool
    sig_df["p_holm_stratified"] = p_holm_strat
    sig_df["p_bh_fdr"] = p_fdr
    sig_df["rejected_pooled"] = rej_pool
    sig_df["rejected_stratified"] = rej_strat
    sig_df["rejected_fdr"] = rej_fdr

    # Strict-superset column order.
    final_cols = [
        "dataset", "weighting_scheme", "model_a", "model_b", "comparison_type",
        "mean", "ci_lo", "ci_hi", "cohens_d", "effect_size_kind",
        "p_value", "n_bootstrap", "notes",
        "p_holm", "rejected_holm",
        # new columns appended in -compatible order.
        "bca_lo", "bca_hi",
        "p_holm_pooled", "p_holm_stratified", "p_bh_fdr",
        "rejected_pooled", "rejected_stratified", "rejected_fdr",
    ]
    sig_df = sig_df[final_cols]

    # ---------- Headline separability summary ------------------------------
    # For each (dataset, scheme), test H₀: ρ >= 0.7 under four procedures:
    #   * BCa upper bound < 0.7 -> reject (interval-based test).
    #   * pooled Holm-adjusted p_value_h07 < 0.05 -> reject.
    #   * stratified Holm-adjusted p_value_h07 < 0.05 -> reject.
    #   * BH-FDR-adjusted p_value_h07 < 0.05 -> reject.

    # The h07 p-values are corrected over the family of 30 rank_disagreement
    # rows (6 datasets x 5 schemes). We apply the same three procedures
    # (Holm pooled, Holm stratified by scheme, BH-FDR pooled) for parity.
    headline_rows = []
    h07_p_raw = []
    h07_keys = []
    h07_strata = []
    for (ds, scheme), rho in rho_lookup.items():
        h07_p_raw.append(float(rho["p_value_h07"]))
        h07_keys.append((ds, scheme))
        h07_strata.append(scheme)
    h07_p_raw_arr = np.asarray(h07_p_raw, dtype=float)
    rej_h07_pool, p_h07_pool = holm_bonferroni_correction(h07_p_raw_arr, alpha=0.05)
    rej_h07_strat, p_h07_strat = stratified_holm_correction(
        h07_p_raw_arr, np.asarray(h07_strata), alpha=0.05
    )
    rej_h07_fdr, p_h07_fdr = benjamini_hochberg_correction(h07_p_raw_arr, alpha=0.05)

    for i, (ds, scheme) in enumerate(h07_keys):
        rho = rho_lookup[(ds, scheme)]
        bca_hi = float(rho["bca_hi"])
        bca_lo = float(rho["bca_lo"])
        # BCa-interval test of H₀: ρ >= 0.7. Reject iff BCa upper bound
        # < 0.7 (the entire 95% interval lies below the threshold).
        reject_bca = bool(bca_hi < 0.7)
        headline_rows.append({
            "dataset": ds,
            "weighting_scheme": scheme,
            "rho_mean": float(rho["mean"]),
            "rho_ci_lo": float(rho["ci_lo"]),
            "rho_ci_hi": float(rho["ci_hi"]),
            "rho_bca_lo": bca_lo,
            "rho_bca_hi": bca_hi,
            "p_h07_raw": float(rho["p_value_h07"]),
            "p_h07_holm_pooled": float(p_h07_pool[i]),
            "p_h07_holm_stratified": float(p_h07_strat[i]),
            "p_h07_bh_fdr": float(p_h07_fdr[i]),
            "reject_h07_bca": reject_bca,
            "reject_h07_holm_pooled": bool(rej_h07_pool[i]),
            "reject_h07_holm_stratified": bool(rej_h07_strat[i]),
            "reject_h07_bh_fdr": bool(rej_h07_fdr[i]),
        })

    headline_df = pd.DataFrame(headline_rows)
    headline_df = headline_df.sort_values(
        ["dataset", "weighting_scheme"]
    ).reset_index(drop=True)
    return sig_df, headline_df

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_phase4_significance_triangulated",
        description=(
            "Triangulated multiple-comparison + BCa significance "
            "extension to results/phase4/significance_n30.csv. Emits "
            "results/phase4/significance_triangulated.csv and "
            "results/phase4/headline_separability_triangulated.csv."
        ),
    )
    parser.add_argument(
        "--procedural-csv",
        type=str,
        default=str(_PROJECT_ROOT / "results" / "phase4" / "procedural_n30.csv"),
    )
    parser.add_argument(
        "--tpr-csv",
        type=str,
        default=str(_PROJECT_ROOT / "results" / "phase4" / "per_group_tpr_n30.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(_PROJECT_ROOT / "results" / "phase4"),
    )
    args = parser.parse_args(argv)

    proc_csv = pathlib.Path(args.procedural_csv)
    tpr_csv = pathlib.Path(args.tpr_csv)
    if not proc_csv.exists():
        print(f"ERROR: procedural CSV not found at {proc_csv}.", file=sys.stderr)
        return 2
    if not tpr_csv.exists():
        # Fall back to non-N30 TPR (per_group_tpr.csv) if the N=30
        # variant isn't on disk — TPR doesn't change with seed count.
        fallback = _PROJECT_ROOT / "results" / "phase4" / "per_group_tpr.csv"
        if fallback.exists():
            tpr_csv = fallback
        else:
            print(f"ERROR: TPR CSV not found at {args.tpr_csv}.", file=sys.stderr)
            return 2

    out_dir = pathlib.Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        print(msg, flush=True)

    proc_df = pd.read_csv(proc_csv)
    tpr_df = pd.read_csv(tpr_csv)

    sig_df, headline_df = compute_triangulated_table(proc_df, tpr_df, _log)

    sig_path = out_dir / "significance_triangulated.csv"
    sig_df.to_csv(sig_path, index=False)
    _log(
        f"Wrote {sig_path} ({len(sig_df)} rows; "
        f"{int(sig_df['rejected_pooled'].sum())} pooled-Holm survivors, "
        f"{int(sig_df['rejected_stratified'].sum())} stratified-Holm survivors, "
        f"{int(sig_df['rejected_fdr'].sum())} BH-FDR survivors)"
    )

    head_path = out_dir / "headline_separability_triangulated.csv"
    headline_df.to_csv(head_path, index=False)
    _log(f"Wrote {head_path} ({len(headline_df)} rows)")
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

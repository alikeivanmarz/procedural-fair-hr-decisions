"""Procedural-fairness metrics for  (contribution C3).

This module operationalises three procedural-justice constructs from
organisational-behaviour theory as ML-evaluable metrics:

    1. ``process_consistency``  — Greenberg 1987 + Colquitt 2001 +
       Hancox-Li 2020. Stability of model predictions under
       semantically-irrelevant Gaussian / categorical-resample
       perturbations of the input.

    2. ``voice_representation`` — Newman, Fast & Harmon 2020. Share of
       SHAP-importance attributable to *modifiable* (data-subject-
       actionable) features versus *immutable* (demographic /
       historical) features. Per  the function ALSO returns
       ``voice_enrichment = voice / (n_modifiable / n_total)``, which
       disentangles the voice signal from the mechanical inflation
       caused by the modifiable / immutable count ratio.

    3. ``model_flippability`` (formerly the architectural half of
       ``transparency_metrics``) — Jui & Rivas 2024 §6.5; Wachter,
       Mittelstadt & Russell 2018. Sparsity + validity of greedy
       minimum-feature-flip counterfactuals (DiCE-based diversity is
       deferred to  per  §4). This metric is an
       ARCHITECTURAL property of the model.

    4. ``explanation_actionability`` (added in ) — the
       PROCEDURAL half of the original transparency metric. Fraction
       of rows whose sparsest CF involves a *modifiable* feature . A CF that flips a modifiable feature gives the data
       subject something to act on; one that flips an immutable
       feature is procedurally meaningless.

    Backward-compat: ``transparency_metrics`` is kept as an alias that
    returns the union of ``model_flippability`` and
    ``explanation_actionability`` outputs (under the historical key
    names). It is deprecated; new code should call the split functions.

Per , the Phase-4 perturbation method is **simple Gaussian noise
on numerics + uniform-resample on categoricals**, NOT DiCE. The
Phase-6 XAI module will revisit consistency under DiCE-generated valid
counterfactuals (`dice_consistency`); the two will be reported side by
side in the thesis.

Per , the modifiable / immutable feature partition required by
``voice_representation`` is per-dataset and explicitly justified in
the ADR. The audit script (``scripts/run_phase4_procedural.py``)
imports its partitions from there.

Signature contract:
    * ``process_consistency`` → ``tuple[float, dict[int, float]]``
    * ``voice_representation`` → ``tuple[float, float, dict[str, float]]``
      (overall voice, voice_enrichment, per-feature share dict — the
      enrichment scalar was added in  / ; older code that
      unpacks two values must be updated).
    * ``model_flippability`` → ``dict[str, float]`` with at-minimum
      keys ``{sparsity, validity}``.
    * ``explanation_actionability`` → ``dict[str, float]`` with
      at-minimum keys ``{actionable_validity, actionable_sparsity}``.
    * ``transparency_metrics`` (deprecated alias) →
      ``dict[str, float]`` exposing the union of the keys above.

Term references ():
    * Process Consistency:         Consistency
    * Voice-Representation:
    * Voice-Enrichment:
    * Model-Flippability:
    * Explanation-Actionability:
    * Transparency-Sparsity:
    * Transparency-Validity:

References:
    * greenberg1987taxonomy   — Greenberg 1987.
    * colquitt2001dimensionality — Colquitt 2001.
    * newman2020when          — Newman, Fast & Harmon 2020.
    * juirivas2024fairness    — Jui & Rivas 2024 (TBD per ).
    * hancoxli2020robustness  — Hancox-Li 2020 (TBD per ).
    * wachter2018counterfactual — Wachter, Mittelstadt & Russell 2018.
"""

from __future__ import annotations

import contextlib
import itertools
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
#  /  — SHAP-XGBoost compatibility patch (multi-class aware).
# ---------------------------------------------------------------------------

# xgboost >= 3.0 serialises ``base_score`` in the booster's ubjson dump as
# a bracketed string. For BINARY classifiers it looks like ``"[5E-1]"``
# (single value). For MULTI-CLASS classifiers it looks like
# ``"[1.18567824E-1,6.5677845E-1,-7.7534616E-1]"`` (one entry per class).
# shap <= 0.49.x's ``XGBTreeModelLoader`` calls
# ``float(learner_model_param["base_score"])`` directly, which fails on
# both forms.

#  (-3) handled the binary form by stripping brackets.
#  (-4) extends the patch to multi-class: when the
# bracketed string contains commas, decode the per-class vector and
# fold it into a single scalar by taking the MEAN. SHAP's downstream
# `base_score` is used as the additive baseline; for multi-class the
# per-class baseline averages out under the mean(|·|)-over-classes
# aggregation that ``voice_representation`` already applies, so the
# scalar mean is information-preserving for the metric we compute.

# The patch is scoped (context manager); the original function is
# restored on exit so unrelated SHAP calls (e.g., LR's LinearExplainer)
# are unaffected. Auto-applied inside ``voice_representation`` whenever
# a tree explainer is requested so callers do not need to wrap calls
# themselves.
@contextlib.contextmanager
def _xgboost_shap_compat_patch():
    """Make shap.TreeExplainer parse xgboost-3.x boosters correctly.

    Coerces the ``learner_model_param["base_score"]`` field from the
    bracketed-vector string form (binary: ``"[<scalar>]"``; multi-class:
    ``"[<v0>,<v1>,...,<vK>]"``) into a single scalar that ``float()``
    can parse. Multi-class vectors are reduced to their MEAN.

    No-op on older xgboost / newer shap that already handle the new
    format, or when SHAP is not importable.
    """
    try:
        from shap.explainers import _tree as _shap_tree
    except ImportError:
        yield
        return

    if not hasattr(_shap_tree, "decode_ubjson_buffer"):
        yield
        return

    _orig_decode = _shap_tree.decode_ubjson_buffer

    def _patched_decode(fd):
        result = _orig_decode(fd)
        try:
            bs = result["learner"]["learner_model_param"]["base_score"]
            if isinstance(bs, str) and bs.startswith("[") and bs.endswith("]"):
                inner = bs.strip("[]")
                if "," in inner:
                    # Multi-class: per-class base scores; fold to mean.
                    parts = [float(p.strip()) for p in inner.split(",") if p.strip()]
                    if parts:
                        scalar = sum(parts) / len(parts)
                        result["learner"]["learner_model_param"][
                            "base_score"
                        ] = repr(scalar)
                else:
                    # Binary: single bracketed scalar.
                    result["learner"]["learner_model_param"][
                        "base_score"
                    ] = inner
        except (KeyError, TypeError, ValueError):
            pass
        return result

    _shap_tree.decode_ubjson_buffer = _patched_decode
    try:
        yield
    finally:
        _shap_tree.decode_ubjson_buffer = _orig_decode

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stratified_sample_indices(
    X: pd.DataFrame,
    sample_n: int | None,
    stratify_on: pd.Series | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return row indices for an optionally-stratified sample of ``X``.

    If ``sample_n`` is ``None`` or ≥ ``len(X)``, returns ``arange(len(X))``.
    If ``stratify_on`` is provided, samples proportionally per group.
    """
    n = len(X)
    if sample_n is None or sample_n >= n:
        return np.arange(n)
    if stratify_on is None:
        return rng.choice(n, size=sample_n, replace=False)

    strata = pd.Series(stratify_on).reset_index(drop=True)
    indices: list[int] = []
    groups = strata.unique()
    # Floor allocation per group then top up the remainder uniformly.
    base = sample_n // max(len(groups), 1)
    remainder = sample_n - base * len(groups)
    for g in groups:
        g_idx = np.flatnonzero(strata.to_numpy() == g)
        take = min(base, len(g_idx))
        if take > 0:
            indices.extend(rng.choice(g_idx, size=take, replace=False).tolist())
    if remainder > 0:
        # Pick the remainder from any not-yet-selected row.
        already = set(indices)
        pool = [i for i in range(n) if i not in already]
        if pool:
            remainder = min(remainder, len(pool))
            indices.extend(rng.choice(pool, size=remainder, replace=False).tolist())
    return np.asarray(sorted(set(indices)), dtype=int)

def _model_proba(model, X: pd.DataFrame, n_classes_hint: int | None = None) -> np.ndarray:
    """Return a ``(n_rows, n_classes)`` probability matrix.

    Uses ``model.predict_proba`` when available; otherwise one-hot encodes
    ``model.predict`` outputs over the union of seen labels (with a hint).
    """
    if hasattr(model, "predict_proba"):
        try:
            proba = np.asarray(model.predict_proba(X), dtype=float)
            if proba.ndim == 1:
                proba = proba[:, None]
            # Ensure rows sum to 1 (tolerate floating-point drift).
            row_sums = proba.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            return proba / row_sums
        except (NotImplementedError, AttributeError, ValueError):
            pass

    preds = np.asarray(model.predict(X)).ravel().astype(int)
    classes_seen = np.unique(preds)
    if n_classes_hint is not None and n_classes_hint > 0:
        n_classes = max(int(n_classes_hint), int(classes_seen.max()) + 1)
    else:
        n_classes = int(classes_seen.max()) + 1
    out = np.zeros((len(preds), n_classes), dtype=float)
    out[np.arange(len(preds)), preds] = 1.0
    return out

def _js_divergence_base2(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence (base 2) between two distributions.

    Bounded in [0, 1] base 2. Adds ``eps`` to handle zero-probability
    classes (avoids ``log(0)`` NaNs); clips the final value into [0, 1]
    to defend against floating-point drift.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        n = max(p.shape[0], q.shape[0])
        p_pad = np.zeros(n, dtype=float)
        p_pad[: p.shape[0]] = p
        q_pad = np.zeros(n, dtype=float)
        q_pad[: q.shape[0]] = q
        p, q = p_pad, q_pad
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    # KL divergence in base 2.
    kl_pm = np.sum(p * (np.log2(p) - np.log2(m)))
    kl_qm = np.sum(q * (np.log2(q) - np.log2(m)))
    js = 0.5 * (kl_pm + kl_qm)
    # Clip to [0, 1] (already bounded base-2; defend against drift).
    return float(min(max(js, 0.0), 1.0))

def _perturb_row(
    row: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    column_stds: dict[str, float],
    column_levels: dict[str, np.ndarray],
    noise_std: float,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Return ``n`` perturbed copies of ``row`` as a DataFrame.

    Numeric cols receive Gaussian noise σ = noise_std × column_std;
    categorical cols are uniform-resampled from their observed levels.
    """
    rep = pd.DataFrame([row.to_dict()] * n)
    for col in numeric_cols:
        sigma = column_stds.get(col, 0.0) * noise_std
        if sigma > 0:
            noise = rng.normal(loc=0.0, scale=sigma, size=n)
            rep[col] = row[col] + noise
    for col in categorical_cols:
        levels = column_levels.get(col)
        if levels is not None and len(levels) > 0:
            rep[col] = rng.choice(levels, size=n)
    # Preserve dtypes — rep is constructed from a dict, so dtypes match
    # the original row's element-types where possible.
    return rep

# ---------------------------------------------------------------------------
# Public API — Procedural-fairness metrics
# ---------------------------------------------------------------------------

def process_consistency(
    model,
    X: pd.DataFrame,
    perturbations_per_row: int = 10,
    noise_std: float = 0.1,
    random_state: int = 0,
    sample_n: int | None = 500,
    stratify_on: pd.Series | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, dict[int, float]]:
    """Process Consistency (procedural fairness, Greenberg 1987 / Colquitt 2001 / Hancox-Li 2020).

    Operationalises the OB-theoretic *consistency* construct as the mean
    Jensen-Shannon divergence between the model's prediction distribution
    on an instance ``x`` and on N perturbed copies ``x + ε``:

        ProcConsistency = 1 − E_x[ E_ε[ JS_2(P(ŷ|x), P(ŷ|x+ε)) ] ]

    where ε is Gaussian on numerics (σ = ``noise_std × column_std``) and
    a uniform resample on categoricals (within their observed levels).
    JS divergence is computed in base 2 (bounded in [0, 1]) so the
    consistency score is in [0, 1]; **higher is more consistent**
    (1.0 = perfectly consistent, e.g. constant model or noise-invariant
    model).

    Per  the perturbation method is simple noise / categorical
    resample, NOT DiCE.  will add a DiCE-based ``dice_consistency``
    that uses valid counterfactuals as the perturbation source.

    See  Consistency.

    Reference: greenberg1987taxonomy; colquitt2001dimensionality;
    hancoxli2020robustness (TBD per ).

    Args:
        model: A fitted estimator with either ``predict_proba`` (preferred)
            or ``predict``. When only ``predict`` is available the function
            falls back to one-hot encoding the hard predictions, which is
            equivalent to a "fraction of perturbations that flip the
            predicted class" measure.
        X: Feature DataFrame (numeric + categorical / object columns).
        perturbations_per_row: Number of perturbed copies per sampled row
            (default 10).
        noise_std: Multiplier for the per-column standard deviation that
            sets the Gaussian noise scale on numerics (default 0.1).
        random_state: Random seed (default 0;  determinism).
        sample_n: Optional cap on the number of rows evaluated. If
            ``None`` or larger than ``len(X)``, every row is used.
            Default 500.
        stratify_on: Optional ``pandas.Series`` whose values are used to
            stratify the sample (typically the sensitive attribute, so
            both groups are proportionally represented).

    Returns:
        Tuple ``(overall_consistency, per_row_consistency)``:
            * ``overall_consistency`` — float in [0, 1].
            * ``per_row_consistency`` — ``dict[int, float]`` mapping the
              positional index of each evaluated row in ``X`` to its
              consistency score (1 − mean JS over its N perturbations).

    Notes:
        * Per  limitation: this metric measures stability under
          random noise; it does NOT measure stability under valid
          counterfactual flips.  surfaces the latter as a
          companion metric.

          float]]``.
    """
    if rng is None:
        rng = np.random.default_rng(random_state)
    n_rows = len(X)
    if n_rows == 0:
        return float("nan"), {}

    #  short-circuit: noise_std == 0 means no perturbation; consistency
    # = 1.0 by definition. Avoids spurious JS divergence from categorical
    # uniform-resample (which still varies even at noise_std=0).
    sample_idx_for_zero = _stratified_sample_indices(X, sample_n, stratify_on, rng)
    if noise_std == 0.0:
        zero_per_row = {int(i): 1.0 for i in sample_idx_for_zero}
        return 1.0, zero_per_row

    sample_idx = sample_idx_for_zero
    X_sample = X.iloc[sample_idx].reset_index(drop=True)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    column_stds = {
        c: float(X[c].astype(float).std(ddof=0)) for c in numeric_cols
    }
    column_levels = {c: X[c].dropna().unique() for c in categorical_cols}

    # Predictions on the sample (used as the "anchor").
    orig_proba = _model_proba(model, X_sample)
    n_classes = orig_proba.shape[1]

    per_row: dict[int, float] = {}
    js_scores: list[float] = []
    for i in range(len(X_sample)):
        row = X_sample.iloc[i]
        perturbed = _perturb_row(
            row,
            numeric_cols,
            categorical_cols,
            column_stds,
            column_levels,
            noise_std,
            perturbations_per_row,
            rng,
        )
        # Predict on the perturbed batch.
        perturbed_proba = _model_proba(model, perturbed, n_classes_hint=n_classes)
        anchor = orig_proba[i]
        # Mean JS over the N perturbations.
        js_per_p = [
            _js_divergence_base2(anchor, perturbed_proba[k])
            for k in range(perturbed_proba.shape[0])
        ]
        mean_js = float(np.mean(js_per_p)) if js_per_p else 0.0
        consistency = float(min(max(1.0 - mean_js, 0.0), 1.0))
        # Map back to the original positional index in ``X``.
        per_row[int(sample_idx[i])] = consistency
        js_scores.append(mean_js)

    overall_js = float(np.mean(js_scores)) if js_scores else 0.0
    overall_consistency = float(min(max(1.0 - overall_js, 0.0), 1.0))
    return overall_consistency, per_row

def voice_representation(
    model,
    X: pd.DataFrame,
    feature_partition: dict[str, list[str]],
    shap_explainer: str = "tree",
    sample_n: int | None = 500,
    random_state: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, dict[str, float]]:
    """Voice / Representation (procedural fairness, Newman, Fast & Harmon 2020).

    Operationalises the OB-theoretic *voice* construct as the share of
    total |SHAP| feature importance attributable to *modifiable* (action-
    actionable) features versus *immutable* (demographic / historical)
    features:

        Voice = Σ_{f ∈ M} mean(|SHAP_f|) / Σ_f mean(|SHAP_f|)

    where ``M`` is the set of modifiable features per the per-dataset
    partition specified in . **Higher is more voice** (the data
    subject has more actionable channels to influence the prediction);
    range [0, 1].

    See

    Reference: newman2020when; colquitt2001dimensionality (voice
    dimension of organisational-justice theory).

    Args:
        model: A fitted estimator. The choice of ``shap_explainer``
            determines which SHAP backend is used:
                * ``"tree"`` — ``shap.TreeExplainer`` (RandomForest,
                  GradientBoosting, XGBoost, LightGBM, ...).
                * ``"linear"`` — ``shap.LinearExplainer``.
                * ``"kernel"`` — ``shap.KernelExplainer`` with a
                  100-row background sample (slow, model-agnostic).
        X: Feature DataFrame. Every column in ``X`` must appear in
            exactly one of ``feature_partition["modifiable"]`` or
            ``feature_partition["immutable"]`` — otherwise a
            ``ValueError`` is raised listing the unaccounted features.
        feature_partition: ``{"modifiable": [...], "immutable": [...]}``.
            See  for the per-dataset partitions.
        shap_explainer: Which SHAP backend to use. Default ``"tree"``.
        sample_n: Optional cap on the number of rows used to compute
            SHAP values (default 500). SHAP scales with sample size, so
            this matters for performance; the |SHAP| share is averaged
            over the sample.
        random_state: Random seed for sampling (default 0; ).

    Returns:
        Tuple ``(overall_voice_share, voice_enrichment, per_feature_share)``
        where:
            * ``overall_voice_share`` — float in [0, 1]: total
              |SHAP|-share attributable to modifiable features.
            * ``voice_enrichment`` — float ≥ 0: ratio of voice share to
              the modifiable feature-count share, i.e.
              ``voice / (n_modifiable / n_total)``. Values:
                - ``> 1``: modifiable features carry MORE importance
                  than their numerical share would predict (real voice
                  signal).
                - ``≈ 1``: voice is mechanically inflated by the
                  feature-count ratio (no enrichment).
                - ``< 1``: modifiable features carry LESS importance
                  than their share (voice deficit).
              Returns ``0.0`` when ``n_modifiable == 0`` (no voice
              expressible) or ``n_total == 0``. The cap above 1 is the
              theoretical maximum ``n_total / n_modifiable`` (when
              voice = 1.0); we leave it un-clipped so plots can show
              the full enrichment factor.
            * ``per_feature_share`` — ``dict[str, float]`` mapping each
              feature in ``X.columns`` to its individual share of the
              total |SHAP|. Sums to 1.0 (modulo floating-point drift)
              across all features.

    Raises:
        ValueError: If any column in ``X`` is missing from both
            ``modifiable`` and ``immutable`` lists, or appears in
            both. Lists the offending columns.

    Notes:
        * Per , the modifiable / immutable partition is a
          per-dataset judgement call documented verbatim in .

          ``tuple[float, float, dict[str, float]]``.
        * If ``sum(|SHAP|) == 0`` (degenerate constant model), returns
          ``(0.0, 0.0, {f: 0.0 for f in X.columns})`` to avoid division
          by zero.
        * Determinism: SHAP background sample is fixed via
          ``shap.sample(..., random_state=random_state)`` and
          ``TreeExplainer`` uses ``feature_perturbation="interventional"``.
    """
    modifiable = list(feature_partition.get("modifiable", []))
    immutable = list(feature_partition.get("immutable", []))
    cols = list(X.columns)

    overlap = set(modifiable) & set(immutable)
    if overlap:
        raise ValueError(
            f"Features appear in both 'modifiable' and 'immutable' "
            f"buckets: {sorted(overlap)}"
        )
    missing = [c for c in cols if c not in modifiable and c not in immutable]
    if missing:
        raise ValueError(
            f"Features in X are missing from feature_partition: {missing}. "
            "Every column in X must be in either 'modifiable' or "
            "'immutable'."
        )

    if rng is None:
        rng = np.random.default_rng(random_state)
    n_rows = len(X)
    if sample_n is not None and sample_n < n_rows:
        idx = rng.choice(n_rows, size=sample_n, replace=False)
        X_sample = X.iloc[idx].reset_index(drop=True)
    else:
        X_sample = X.reset_index(drop=True)

    import shap  # local import keeps the module importable when shap is absent

    # Deterministic SHAP background : fixed sample with explicit
    # random_state, plus feature_perturbation="interventional" for
    # TreeExplainer (the "tree_path_dependent" default has been observed
    # to drift across sklearn / xgboost / shap version pairs).
    if shap_explainer == "tree":
        # : auto-apply the xgboost-3.x base_score patch (multi-class
        # aware). Idempotent / scoped — leaves SHAP untouched on exit and
        # is a no-op for non-xgboost trees.
        shap_values = None
        with _xgboost_shap_compat_patch():
            try:
                try:
                    explainer = shap.TreeExplainer(
                        model, feature_perturbation="interventional",
                        data=shap.sample(X_sample, min(100, len(X_sample)),
                                         random_state=random_state),
                    )
                except Exception:  # pragma: no cover - some models reject background
                    explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(
                    X_sample, check_additivity=False
                )
            except Exception as exc:
                #  (-4) — SHAP's TreeExplainer raises
                # ``InvalidModelError`` on multi-class
                # ``sklearn.GradientBoostingClassifier`` ("only supported
                # for binary classification right now"). This is the same
                # class of NaN-emitting failure as the xgboost-3.x
                # base_score bug.  says fix, don't punt: fall back
                # to KernelExplainer which is model-agnostic. Slower
                # (~10x for the 500-row sample) but well within wall-
                # clock budget for the audit. The fallback is applied
                # inside the patch context so other tree models retain
                # the fast path.
                _name = type(model).__name__
                _is_known_unsupported = (
                    "InvalidModelError" in type(exc).__name__
                    or "only supported for binary" in str(exc)
                )
                if not _is_known_unsupported:
                    raise
                background_n = min(100, len(X_sample))
                background = shap.sample(
                    X_sample, background_n, random_state=random_state
                )
                predictor = (
                    model.predict_proba
                    if hasattr(model, "predict_proba")
                    else model.predict
                )
                np.random.seed(random_state)
                explainer = shap.KernelExplainer(predictor, background)
                shap_values = explainer.shap_values(
                    X_sample, nsamples=100, silent=True
                )
    elif shap_explainer == "linear":
        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)
    elif shap_explainer == "kernel":
        background_n = min(100, len(X_sample))
        background = shap.sample(
            X_sample, background_n, random_state=random_state
        )
        # KernelExplainer wants a callable. Use predict_proba when present
        # (richer signal); else predict.
        predictor = (
            model.predict_proba
            if hasattr(model, "predict_proba")
            else model.predict
        )
        # : KernelExplainer.shap_values() internally calls
        # ``np.random.choice`` and ``np.random.permutation`` on the
        # GLOBAL NumPy random state. Seed it explicitly here so the
        # SHAP estimates are byte-stable across processes (the local
        # ``rng`` Generator does not control the global state).
        np.random.seed(random_state)
        explainer = shap.KernelExplainer(predictor, background)
        shap_values = explainer.shap_values(
            X_sample, nsamples=100, silent=True
        )
    else:
        raise ValueError(
            f"Unknown shap_explainer={shap_explainer!r}; expected "
            "'tree' | 'linear' | 'kernel'."
        )

    # SHAP returns either a 2-D array (binary / regression) or a list /
    # 3-D array of per-class SHAP values. Aggregate by mean(|·|) over
    # both axes (samples + classes) so we get a single (n_features,)
    # importance vector.
    if isinstance(shap_values, list):
        # list of (n_rows, n_features) arrays, one per class
        stacked = np.stack(
            [np.abs(np.asarray(sv)) for sv in shap_values], axis=0
        )
        feature_importance = stacked.mean(axis=(0, 1))
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            # (n_rows, n_features, n_classes) — common newer SHAP layout
            feature_importance = np.abs(arr).mean(axis=(0, 2))
        else:
            feature_importance = np.abs(arr).mean(axis=0)

    feature_importance = np.asarray(feature_importance, dtype=float).ravel()
    if feature_importance.shape[0] != len(cols):
        # Fallback: if SHAP returned an unexpected shape, pad / truncate.
        # In practice this would indicate an explainer mismatch; raise.
        raise ValueError(
            f"SHAP returned a feature-importance vector of length "
            f"{feature_importance.shape[0]}; expected {len(cols)} "
            "(matching X.columns)."
        )

    total = float(feature_importance.sum())
    n_total = len(cols)
    n_modifiable = len(modifiable)
    if total == 0.0:
        per_feature = {c: 0.0 for c in cols}
        return 0.0, 0.0, per_feature

    per_feature = {
        cols[i]: float(feature_importance[i] / total) for i in range(len(cols))
    }
    voice_share = float(sum(per_feature[c] for c in modifiable))
    voice_share = float(min(max(voice_share, 0.0), 1.0))

    # Voice enrichment: voice / (n_modifiable / n_total).
    if n_modifiable == 0 or n_total == 0:
        voice_enrichment = 0.0
    else:
        modifiable_share = n_modifiable / n_total
        voice_enrichment = float(voice_share / modifiable_share)
    return voice_share, voice_enrichment, per_feature

def _greedy_cf_search(
    model,
    X: pd.DataFrame,
    max_features_to_flip: int,
    sample_n: int | None,
    random_state: int,
    rng: np.random.Generator | None = None,
) -> tuple[list[tuple[str, ...] | None], list[int], np.ndarray, int]:
    """Internal: run greedy minimum-feature-flip CF search per row.

    Returns
    -------
    flip_combos : list[tuple[str, ...] | None]
        Per-row tuple of feature names changed in the sparsest CF, or
        ``None`` if no CF was found within ``max_features_to_flip``.
    n_changes : list[int]
        Per-row count of features changed (= ``len(flip_combo)`` if
        found, else ``n_features``: failure penalty for sparsity).
    sample_idx : np.ndarray
        Positional indices in ``X`` corresponding to evaluated rows.
    n_features : int
        Total feature count of ``X`` (denominator for sparsity).
    """
    if rng is None:
        rng = np.random.default_rng(random_state)
    n_rows = len(X)
    if sample_n is not None and sample_n < n_rows:
        idx = np.sort(rng.choice(n_rows, size=sample_n, replace=False))
    else:
        idx = np.arange(n_rows)
    X_sample = X.iloc[idx].reset_index(drop=True)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    column_stds = {
        c: float(X[c].astype(float).std(ddof=0)) for c in numeric_cols
    }
    column_mins = {c: float(X[c].astype(float).min()) for c in numeric_cols}
    column_maxs = {c: float(X[c].astype(float).max()) for c in numeric_cols}
    column_levels = {c: X[c].dropna().unique() for c in categorical_cols}
    n_features = len(X.columns)

    orig_preds = np.asarray(model.predict(X_sample)).ravel()

    def _candidate_flips_for(row: pd.Series, col: str) -> list[Any]:
        if col in numeric_cols:
            sigma = column_stds.get(col, 0.0)
            base = float(row[col])
            cands: list[Any] = []
            if sigma > 0:
                cands.append(base + sigma)
                cands.append(base - sigma)
            col_min = column_mins.get(col)
            col_max = column_maxs.get(col)
            if col_min is not None and col_min != base:
                cands.append(col_min)
            if col_max is not None and col_max != base:
                cands.append(col_max)
            seen = set()
            unique_cands: list[Any] = []
            for c in cands:
                key = round(c, 9)
                if key not in seen:
                    seen.add(key)
                    unique_cands.append(c)
            return unique_cands
        levels = column_levels.get(col, np.array([]))
        return [v for v in levels if v != row[col]]

    flip_combos: list[tuple[str, ...] | None] = []
    n_changes: list[int] = []

    for i in range(len(X_sample)):
        row = X_sample.iloc[i]
        orig_pred = orig_preds[i]
        best_flip: tuple[str, ...] | None = None

        for k in range(1, max_features_to_flip + 1):
            for combo in itertools.combinations(X.columns, k):
                col_candidates = [_candidate_flips_for(row, c) for c in combo]
                if any(len(cands) == 0 for cands in col_candidates):
                    continue
                bounded = [cands[:4] for cands in col_candidates]
                for choice in itertools.product(*bounded):
                    test_row = row.copy()
                    for c, val in zip(combo, choice):
                        test_row[c] = val
                    test_df = pd.DataFrame([test_row])
                    pred = np.asarray(model.predict(test_df)).ravel()[0]
                    if pred != orig_pred:
                        best_flip = combo
                        break
                if best_flip is not None:
                    break
            if best_flip is not None:
                break

        flip_combos.append(best_flip)
        n_changes.append(len(best_flip) if best_flip is not None else n_features)

    return flip_combos, n_changes, idx, n_features

def model_flippability(
    model,
    X: pd.DataFrame,
    sensitive: pd.Series | None = None,
    max_features_to_flip: int = 1,
    sample_n: int | None = 500,
    random_state: int = 0,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Architectural property — fraction of rows reachable by any ≤k-feature flip.

    Computes ``sparsity`` + ``validity`` averaged over rows using the
    same greedy ≤k-feature CF search as the (deprecated)
    ``transparency_metrics``. This metric is an ARCHITECTURAL property of
    the trained model: it characterises how "flippable" the model is
    under small input perturbations, regardless of WHICH features get
    flipped.

    See

    Reference: this metric is the Phase-4-original
    ``transparency_metrics`` renamed to make the
    conflation with explanation-actionability explicit. The procedurally
    meaningful complement is :func:`explanation_actionability`.

    Args:
        model: A fitted estimator with ``predict``.
        X: Feature DataFrame.
        sensitive: Optional ``pandas.Series`` (currently unused; reserved
            for future per-group breakdowns).
        max_features_to_flip: Max cardinality of the greedy search
            (default 1 per  — the simplest flip is the
            least-architecturally-noisy reading).
        sample_n: Optional cap on rows evaluated (default 500).
        random_state: Random seed (default 0; ).
        rng: Optional pre-built ``numpy.random.Generator``; when
            provided overrides ``random_state``.

    Returns:
        ``dict[str, float]`` with keys:
            * ``sparsity`` ∈ [0, 1] — 1 − mean(n_changed / n_features);
              rows where no CF found contribute 1.0 to the
              fraction-changed (sparsity = 0 for those rows).
            * ``validity`` ∈ [0, 1] — fraction of rows for which a CF
              was found within ``max_features_to_flip`` features.
    """
    if rng is None:
        rng = np.random.default_rng(random_state)
    n_rows = len(X)
    if n_rows == 0:
        return {"sparsity": float("nan"), "validity": float("nan")}

    flip_combos, n_changes, _, n_features = _greedy_cf_search(
        model, X, max_features_to_flip, sample_n, random_state, rng=rng
    )
    found_flags = [c is not None for c in flip_combos]
    if not n_changes:
        return {"sparsity": float("nan"), "validity": float("nan")}

    fraction_changed = np.asarray(n_changes, dtype=float) / max(n_features, 1)
    sparsity = float(1.0 - fraction_changed.mean())
    sparsity = float(min(max(sparsity, 0.0), 1.0))
    validity = float(np.mean(found_flags))
    return {"sparsity": sparsity, "validity": validity}

def explanation_actionability(
    model,
    X: pd.DataFrame,
    feature_partition: dict[str, list[str]],
    max_features_to_flip: int = 1,
    sample_n: int | None = 500,
    random_state: int = 0,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Procedural property — fraction of CFs that involve a *modifiable* feature.

    For each row, run the same greedy ≤k-feature CF search as
    :func:`model_flippability`; among rows where a CF was found, count
    the share whose CF involves AT LEAST ONE modifiable feature. A CF that flips a modifiable feature gives the data
    subject something to act on; one that flips an immutable feature
    is procedurally meaningless ("you'd need to be a different age").

    See

    Reference:  / . This is the procedurally meaningful
    complement to :func:`model_flippability`; the union of the two
    constitutes the original ``transparency_metrics`` framing.

    Args:
        model: A fitted estimator with ``predict``.
        X: Feature DataFrame.
        feature_partition: ``{"modifiable": [...], "immutable": [...]}``.
            Same contract as :func:`voice_representation`. Every column
            in ``X`` must appear in exactly one bucket.
        max_features_to_flip: Max cardinality of the greedy search
            (default 1 per ).
        sample_n: Optional cap on rows evaluated (default 500).
        random_state: Random seed (default 0; ).
        rng: Optional pre-built ``numpy.random.Generator``.

    Returns:
        ``dict[str, float]`` with keys:
            * ``actionable_validity`` ∈ [0, 1] — fraction of rows for
              which an actionable CF was found (a strict subset of
              :func:`model_flippability`'s validity rows).
            * ``actionable_sparsity`` ∈ [0, 1] — among rows with an
              actionable CF, ``1 − mean(n_changed / n_features)``.
              Rows without an actionable CF contribute 0 to the
              numerator (penalty), mirroring the
              :func:`model_flippability` convention.

    Raises:
        ValueError: same partition-validation rules as
            :func:`voice_representation`.
    """
    modifiable = list(feature_partition.get("modifiable", []))
    immutable = list(feature_partition.get("immutable", []))
    cols = list(X.columns)

    overlap = set(modifiable) & set(immutable)
    if overlap:
        raise ValueError(
            f"Features appear in both 'modifiable' and 'immutable' "
            f"buckets: {sorted(overlap)}"
        )
    missing = [c for c in cols if c not in modifiable and c not in immutable]
    if missing:
        raise ValueError(
            f"Features in X are missing from feature_partition: {missing}. "
            "Every column in X must be in either 'modifiable' or "
            "'immutable'."
        )

    if rng is None:
        rng = np.random.default_rng(random_state)
    n_rows = len(X)
    if n_rows == 0:
        return {
            "actionable_validity": float("nan"),
            "actionable_sparsity": float("nan"),
        }

    flip_combos, n_changes, _, n_features = _greedy_cf_search(
        model, X, max_features_to_flip, sample_n, random_state, rng=rng
    )
    if not flip_combos:
        return {
            "actionable_validity": float("nan"),
            "actionable_sparsity": float("nan"),
        }

    modifiable_set = set(modifiable)
    actionable_flags: list[bool] = []
    actionable_n_changes: list[int] = []
    for combo, n_changed in zip(flip_combos, n_changes):
        if combo is None:
            actionable_flags.append(False)
            actionable_n_changes.append(n_features)
            continue
        # A CF is actionable iff AT LEAST ONE of the flipped features
        # is modifiable.
        if any(c in modifiable_set for c in combo):
            actionable_flags.append(True)
            actionable_n_changes.append(n_changed)
        else:
            actionable_flags.append(False)
            actionable_n_changes.append(n_features)

    actionable_validity = float(np.mean(actionable_flags))
    fraction_changed = (
        np.asarray(actionable_n_changes, dtype=float) / max(n_features, 1)
    )
    actionable_sparsity = float(1.0 - fraction_changed.mean())
    actionable_sparsity = float(min(max(actionable_sparsity, 0.0), 1.0))
    return {
        "actionable_validity": actionable_validity,
        "actionable_sparsity": actionable_sparsity,
    }

def transparency_metrics(
    model,
    X: pd.DataFrame,
    sensitive: pd.Series | None = None,
    max_features_to_flip: int = 3,
    sample_n: int | None = 500,
    random_state: int = 0,
) -> dict[str, float]:
    """Transparency: Sparsity + Validity (Jui & Rivas 2024 §6.5; Wachter 2018).

    DEPRECATED: this function conflates two
    distinct procedural-fairness constructs — the architectural
    "how flippable is this model under any small change" (kept here)
    and the procedural "does the explanation give the data subject
    something to *act* on" (which requires the modifiable / immutable
    partition; :func:`explanation_actionability`). New code SHOULD
    call :func:`model_flippability` (architectural) and
    :func:`explanation_actionability` (procedural) directly. This
    function is preserved for backward compatibility (smoke tests in
    ``tests/test_phase4_audit.py`` rely on the historical key names).

    For each sampled row, find a *minimum-feature-flip* counterfactual
    that changes the model's prediction, using a greedy search up to
    ``max_features_to_flip`` features. For numerics, candidate flips
    are ``±1 column std``; for categoricals, each alternative observed
    level. The function returns:

        sparsity = 1 − E_x[ #features_changed_in_CF / n_features ]
        validity = Pr_x[ a CF was found within max_features_to_flip ]

    **Higher sparsity = more transparent** (CFs change few features —
    explanations are easy to communicate). **Higher validity = the
    model is reachable** (a CF actually exists for most rows).

    Per  +  §4, DiCE-based diversity is deferred to
    ; this Phase-4 implementation uses a deterministic greedy
    flip search with no DiCE dependency.

    See  and

    Reference: juirivas2024fairness §6.5 (TBD per );
    wachter2018counterfactual.

    Args:
        model: A fitted estimator with ``predict``.
        X: Feature DataFrame.
        sensitive: Optional ``pandas.Series`` (currently unused; reserved
            for stratification and for future per-group sparsity reports).
        max_features_to_flip: Max cardinality of the greedy search
            (default 3). For each row the search tries 1-feature flips
            first, then 2-feature combinations, then 3, ..., stopping at
            the smallest cardinality that flips the prediction.
        sample_n: Optional cap on rows evaluated (default 500).
        random_state: Random seed (default 0; ).

    Returns:
        ``dict[str, float]`` with at-minimum keys:
            * ``sparsity``  ∈ [0, 1] — fraction of features unchanged in
              the CF (1 − mean flip-cardinality / n_features). For rows
              where no CF was found within ``max_features_to_flip``,
              the per-row sparsity contribution is 0 (penalty), and
              the failure is captured separately by ``validity``.
            * ``validity``  ∈ [0, 1] — fraction of rows for which a CF
              was found within ``max_features_to_flip`` features.

    Notes:

        * The "no CF found ⇒ sparsity contribution = 0" convention
          ensures transparency is penalised for hard-to-flip rows even
          though validity captures the same failure separately. This
          mirrors the standard CF-explanation literature (Wachter 2018,
          Jui & Rivas 2024 §6.5).
    """
    rng = np.random.default_rng(random_state)
    n_rows = len(X)
    if n_rows == 0:
        return {"sparsity": float("nan"), "validity": float("nan")}

    if sample_n is not None and sample_n < n_rows:
        idx = rng.choice(n_rows, size=sample_n, replace=False)
    else:
        idx = np.arange(n_rows)
    X_sample = X.iloc[idx].reset_index(drop=True)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    column_stds = {
        c: float(X[c].astype(float).std(ddof=0)) for c in numeric_cols
    }
    column_mins = {c: float(X[c].astype(float).min()) for c in numeric_cols}
    column_maxs = {c: float(X[c].astype(float).max()) for c in numeric_cols}
    column_levels = {c: X[c].dropna().unique() for c in categorical_cols}
    n_features = len(X.columns)

    # Original predictions on the sample.
    orig_preds = np.asarray(model.predict(X_sample)).ravel()

    def _candidate_flips_for(row: pd.Series, col: str) -> list[Any]:
        """Return candidate replacement values for ``col`` in ``row``.

        For numerics, candidates include ±1 std plus the column's
        observed min and max (to span the empirical range so a CF can
        be found even for features that need a large flip).
        For categoricals, every alternative observed level.
        """
        if col in numeric_cols:
            sigma = column_stds.get(col, 0.0)
            base = float(row[col])
            cands: list[Any] = []
            if sigma > 0:
                cands.append(base + sigma)
                cands.append(base - sigma)
            # Also try the column extrema (min / max) as candidates.
            col_min = column_mins.get(col)
            col_max = column_maxs.get(col)
            if col_min is not None and col_min != base:
                cands.append(col_min)
            if col_max is not None and col_max != base:
                cands.append(col_max)
            # Deduplicate while preserving order.
            seen = set()
            unique_cands = []
            for c in cands:
                key = round(c, 9)
                if key not in seen:
                    seen.add(key)
                    unique_cands.append(c)
            return unique_cands
        levels = column_levels.get(col, np.array([]))
        return [v for v in levels if v != row[col]]

    n_changes_list: list[int] = []
    found_flags: list[bool] = []

    for i in range(len(X_sample)):
        row = X_sample.iloc[i]
        orig_pred = orig_preds[i]
        found = False
        n_changed = 0

        # Search over k = 1, 2, ..., max_features_to_flip.
        for k in range(1, max_features_to_flip + 1):
            best_flip: tuple[str, ...] | None = None
            for combo in itertools.combinations(X.columns, k):
                # For each column, generate candidate replacement values.
                col_candidates = [_candidate_flips_for(row, c) for c in combo]
                if any(len(cands) == 0 for cands in col_candidates):
                    continue
                # Try every Cartesian combination of one candidate per
                # chosen column. To bound combinatorial cost, cap each
                # column's candidate list to its first 4 entries.
                bounded = [cands[:4] for cands in col_candidates]
                for choice in itertools.product(*bounded):
                    test_row = row.copy()
                    for c, val in zip(combo, choice):
                        test_row[c] = val
                    test_df = pd.DataFrame([test_row])
                    pred = np.asarray(model.predict(test_df)).ravel()[0]
                    if pred != orig_pred:
                        best_flip = combo
                        break
                if best_flip is not None:
                    break
            if best_flip is not None:
                found = True
                n_changed = len(best_flip)
                break

        found_flags.append(found)
        # Per-row sparsity contribution: when no CF found, n_changed = 0
        # is replaced with n_features so the per-row sparsity is 0 (penalty).
        n_changes_list.append(n_changed if found else n_features)

    fraction_changed = np.asarray(n_changes_list, dtype=float) / max(n_features, 1)
    sparsity = float(1.0 - fraction_changed.mean()) if len(n_changes_list) else float("nan")
    sparsity = float(min(max(sparsity, 0.0), 1.0))
    validity = float(np.mean(found_flags)) if found_flags else float("nan")
    return {"sparsity": sparsity, "validity": validity}

"""Phase-4 followup — paired bootstrap + Cohen's d significance.

Consumes ``results/phase4/procedural.csv`` (per-seed values from the audit
runner) and emits:

* ``results/phase4/significance.csv`` — paired-bootstrap (n=10,000) CIs +
  Cohen's d effect sizes for pairwise model comparisons on the
  procedural-aggregate, plus the procedural-vs-statistical separability
  bootstrap.
* ``results/phase4/per_group_tpr.csv`` — per-(dataset, model) TPR per
  protected group + non-vacuity flag, computed inline by re-fitting each
  model with seed=0 on the audit's preprocessing pipeline. The
  Phase-2 ``audit.csv`` only carries ``LR + RF`` rows — Phase-4 spans 8
  models, so the per-group TPR has to be re-derived for the missing
  cells.
* ``results/phase4/headline_claims.md`` — top-line significant gaps per
  dataset, ready for thesis Results §4.4 talking points.

Determinism : all model fits in this script are seeded with
``random_state=0``; bootstrap RNGs are explicit ``numpy.random.default_rng``
instances; environment thread counts pinned to 1 before any numpy import.

CLI
---
.. code-block:: bash

    python scripts/run_phase4_significance.py \\
        [--procedural-csv results/phase4/procedural.csv] \\
        [--out-dir results/phase4]

References
----------
* `` — Phase-4 followup spec (Tier 4 =  + ).
* the project documentation — determinism.
* the project documentation — procedural CSV schema.
"""

from __future__ import annotations

# Determinism prelude — MUST come before numpy / sklearn imports.
import os

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import argparse  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OrdinalEncoder, StandardScaler  # noqa: E402

# Ensure ``procedural_fair_hr.*`` and ``scripts.*`` resolve from the project root.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from procedural_fair_hr.data_loaders import (  # noqa: E402
    load_acs,
    load_ibm_hr,
    load_oulad,
    load_ricci,
)

# ---------------------------------------------------------------------------
# Constants — match the Phase-4 audit registry.
# ---------------------------------------------------------------------------

NON_VACUOUS_TPR_THRESHOLD = 0.05  # consistent with Phase-2 convention.

# Models in the Phase-4 audit's procedural CSV. ConstantPredictor and
# ShuffledPredictor are excluded from re-fits (they don't have a meaningful
# per-group TPR — Constant predicts the same class for all rows; Shuffled
# permutes predictions arbitrarily).
TRAINED_MODELS = (
    "RandomForestClassifier",
    "LogisticRegression",
    "MLPClassifier",
    "XGBClassifier",
    "GradientBoostingClassifier",
    "KNeighborsClassifier",
)

# Order shown in CSVs / figs (best-procedural-first roughly).
MODEL_ORDER = (
    "RandomForestClassifier",
    "LogisticRegression",
    "MLPClassifier",
    "XGBClassifier",
    "GradientBoostingClassifier",
    "KNeighborsClassifier",
    "ConstantPredictor",
    "ShuffledPredictor",
)
MODEL_LABELS = {
    "RandomForestClassifier": "RF",
    "LogisticRegression": "LR",
    "MLPClassifier": "MLP",
    "XGBClassifier": "XGB",
    "GradientBoostingClassifier": "GB",
    "KNeighborsClassifier": "KNN",
    "ConstantPredictor": "Const",
    "ShuffledPredictor": "Shuf",
}

# (dataset_key, sensitive_col, target_form) — must match Phase-4 audit.
DATASET_REGISTRY = {
    "ibm_hr_attrition": ("Gender", "binary"),
    "ibm_hr_perfrating": ("Gender", "binary"),
    "oulad": ("gender", "multiclass"),
    "acs_income": ("RAC1P", "binary"),
    "dutch_census": ("sex", "binary"),
    "ricci": ("Race", "binary"),
}

# ---------------------------------------------------------------------------
# Loader wrappers (mirror the Phase-4 audit's wrappers).
# ---------------------------------------------------------------------------

def _load_dataset(key: str) -> dict:
    """Return a :class:`DatasetBundle` for the given Phase-4 dataset key."""
    if key == "ibm_hr_attrition":
        return load_ibm_hr(target="Attrition")
    if key == "ibm_hr_perfrating":
        bundle = load_ibm_hr(target="PerformanceRating")
        if "PercentSalaryHike" in bundle["X_train"].columns:
            bundle["X_train"] = bundle["X_train"].drop(columns=["PercentSalaryHike"])
            bundle["X_test"] = bundle["X_test"].drop(columns=["PercentSalaryHike"])
            bundle["feature_names"] = list(bundle["X_train"].columns)
        return bundle
    if key == "oulad":
        return load_oulad()
    if key == "acs_income":
        bundle = load_acs(state="CA", year=2018, task="income")
        target_n_test = min(4000, len(bundle["X_test"]))
        target_n_train = min(16000, len(bundle["X_train"]))
        rng = np.random.default_rng(0)
        if len(bundle["X_train"]) > target_n_train:
            idx = np.sort(
                rng.choice(len(bundle["X_train"]), size=target_n_train, replace=False)
            )
            bundle["X_train"] = bundle["X_train"].iloc[idx].reset_index(drop=True)
            bundle["y_train"] = bundle["y_train"].iloc[idx].reset_index(drop=True)
            bundle["A_train"] = bundle["A_train"].iloc[idx].reset_index(drop=True)
        if len(bundle["X_test"]) > target_n_test:
            idx = np.sort(
                rng.choice(len(bundle["X_test"]), size=target_n_test, replace=False)
            )
            bundle["X_test"] = bundle["X_test"].iloc[idx].reset_index(drop=True)
            bundle["y_test"] = bundle["y_test"].iloc[idx].reset_index(drop=True)
            bundle["A_test"] = bundle["A_test"].iloc[idx].reset_index(drop=True)
        return bundle
    if key == "ricci":
        return load_ricci()
    raise ValueError(f"Unknown dataset key: {key!r}")

def _build_estimator(model_name: str, seed: int = 0) -> Any:
    """Match Phase-4 audit's _build_model for trained models only."""
    if model_name == "RandomForestClassifier":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
    if model_name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if model_name == "MLPClassifier":
        return MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=seed)
    if model_name == "GradientBoostingClassifier":
        return GradientBoostingClassifier(n_estimators=100, random_state=seed)
    if model_name == "KNeighborsClassifier":
        return KNeighborsClassifier(n_neighbors=5, n_jobs=1)
    if model_name == "XGBClassifier":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "xgboost is required for XGBClassifier; install with "
                "`pip install xgboost`."
            ) from exc
        from sklearn.preprocessing import LabelEncoder

        class _XGBWithLabelEncoder:
            def __init__(self, **kwargs):
                self._xgb = XGBClassifier(**kwargs)
                self._le = LabelEncoder()
                self.classes_: np.ndarray | None = None

            def fit(self, X, y):
                y_enc = self._le.fit_transform(y)
                self._xgb.fit(X, y_enc)
                self.classes_ = self._le.classes_.copy()
                return self

            def predict(self, X):
                preds_enc = self._xgb.predict(X)
                return self._le.inverse_transform(preds_enc)

            def predict_proba(self, X):
                return self._xgb.predict_proba(X)

            def __getattr__(self, name):
                return getattr(self._xgb, name)

        return _XGBWithLabelEncoder(
            n_estimators=100,
            random_state=seed,
            eval_metric="logloss",
            n_jobs=1,
        )
    raise ValueError(f"Unknown model name: {model_name!r}")

def _build_pipeline(X_train: pd.DataFrame, model: Any) -> Pipeline:
    """Audit-compatible preprocessing pipeline."""
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                ),
            ),
        ]
    )
    num_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    pre = ColumnTransformer(
        [("cat", cat_pipe, cat_cols), ("num", num_pipe, num_cols)],
        remainder="drop",
    )
    return Pipeline([("preprocessor", pre), ("clf", model)])

# ---------------------------------------------------------------------------
#  — per-group TPR + non-vacuity surfacing.
# ---------------------------------------------------------------------------

def _per_group_tpr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sens: pd.Series,
    *,
    pos_label: int,
) -> dict[str, dict[str, float]]:
    """Return ``{group_value: {tpr, n_pos, n_pred_pos}}`` for each protected
    group present in ``sens``.

    TPR is class-conditional recall on the binary indicator
    ``y_true == pos_label`` / ``y_pred == pos_label``. A group's TPR is
    NaN when ``n_pos == 0`` (no positive ground-truth instances to recall).
    """
    out: dict[str, dict[str, float]] = {}
    sens = pd.Series(sens).reset_index(drop=True)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    for grp in sorted(sens.unique(), key=lambda v: str(v)):
        mask = (sens == grp).values
        if not mask.any():
            continue
        y_t = y_true_arr[mask]
        y_p = y_pred_arr[mask]
        pos_mask = y_t == pos_label
        n_pos = int(pos_mask.sum())
        n_pred_pos = int((y_p == pos_label).sum())
        if n_pos == 0:
            tpr = float("nan")
        else:
            tpr = float((y_p[pos_mask] == pos_label).mean())
        out[str(grp)] = {
            "tpr": tpr,
            "n_pos": float(n_pos),
            "n_pred_pos": float(n_pred_pos),
            "n_total": float(int(mask.sum())),
        }
    return out

def _binarise_for_eo(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Phase-2 convention: rarer class is positive when labels aren't
    already {0, 1}. Returns (y_true_bin, y_pred_bin, pos_label_used).
    """
    uniques = np.unique(y_true)
    if set(int(v) for v in uniques) == {0, 1}:
        return y_true.astype(int), y_pred.astype(int), 1
    counts = {int(v): int((y_true == v).sum()) for v in uniques}
    pos = min(counts, key=counts.get)
    y_t_bin = (y_true == pos).astype(int)
    y_p_bin = (y_pred == pos).astype(int)
    return y_t_bin, y_p_bin, pos

def compute_per_group_tpr_table(log) -> pd.DataFrame:
    """Re-fit every (dataset, trained-model) pair with seed=0 and emit one
    row per (dataset, model, group) with the group's TPR + a
    ``non_vacuous_tpr`` flag.

    The table also includes the |EO| (= |TPR_g - TPR_g'|) so the figure
    captions can pull statistical fairness directly from this CSV.
    """
    rows: list[dict[str, Any]] = []
    for dataset_key, (sens_col, target_form) in DATASET_REGISTRY.items():
        log(f"per-group TPR / dataset={dataset_key}")
        bundle = _load_dataset(dataset_key)
        sens_test = bundle["A_test"][sens_col].reset_index(drop=True)
        y_test = np.asarray(bundle["y_test"])
        for model_name in TRAINED_MODELS:
            try:
                est = _build_estimator(model_name, seed=0)
                pipe = _build_pipeline(bundle["X_train"], est)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    warnings.simplefilter("ignore", UserWarning)
                    warnings.simplefilter("ignore", FutureWarning)
                    pipe.fit(bundle["X_train"], bundle["y_train"])
                    y_pred = np.asarray(pipe.predict(bundle["X_test"]))
            except Exception as exc:  # pragma: no cover
                log(f"  WARNING: {model_name} fit failed: {exc!r}")
                continue

            if target_form == "binary":
                y_t_bin, y_p_bin, pos_used = _binarise_for_eo(y_test, y_pred)
                groups = _per_group_tpr(y_t_bin, y_p_bin, sens_test, pos_label=1)
                tprs = [g["tpr"] for g in groups.values() if not np.isnan(g["tpr"])]
                if len(tprs) >= 2:
                    eo = float(max(tprs) - min(tprs))
                else:
                    eo = float("nan")
                non_vacuous = bool(
                    all(
                        not np.isnan(g["tpr"])
                        and g["tpr"] >= NON_VACUOUS_TPR_THRESHOLD
                        for g in groups.values()
                    )
                ) if groups else False
                # Encode group columns.
                grp_keys = sorted(groups.keys())
                tpr_strs = [
                    f"{k}={groups[k]['tpr']:.3f}" if not np.isnan(groups[k]["tpr"])
                    else f"{k}=NaN"
                    for k in grp_keys
                ]
                rows.append(
                    {
                        "dataset": dataset_key,
                        "model": model_name,
                        "target_form": "binary",
                        "class_idx": -1,
                        "pos_label_used": int(pos_used),
                        "groups": ";".join(grp_keys),
                        "tpr_per_group": ";".join(tpr_strs),
                        "abs_eo": eo,
                        "non_vacuous_tpr": non_vacuous,
                        "n_groups_with_positives": int(
                            sum(1 for g in groups.values() if g["n_pos"] > 0)
                        ),
                    }
                )
            else:
                # Multiclass — surface per-class TPR via OvR (matches Phase-2
                # ``ovr_class_*`` convention).
                classes = sorted(int(c) for c in np.unique(y_test))
                for c in classes:
                    y_t_bin = (y_test == c).astype(int)
                    y_p_bin = (y_pred == c).astype(int)
                    groups = _per_group_tpr(y_t_bin, y_p_bin, sens_test, pos_label=1)
                    tprs = [
                        g["tpr"] for g in groups.values() if not np.isnan(g["tpr"])
                    ]
                    eo = float(max(tprs) - min(tprs)) if len(tprs) >= 2 else float("nan")
                    non_vacuous = bool(
                        all(
                            not np.isnan(g["tpr"])
                            and g["tpr"] >= NON_VACUOUS_TPR_THRESHOLD
                            for g in groups.values()
                        )
                    ) if groups else False
                    grp_keys = sorted(groups.keys())
                    tpr_strs = [
                        f"{k}={groups[k]['tpr']:.3f}"
                        if not np.isnan(groups[k]["tpr"])
                        else f"{k}=NaN"
                        for k in grp_keys
                    ]
                    rows.append(
                        {
                            "dataset": dataset_key,
                            "model": model_name,
                            "target_form": "multiclass",
                            "class_idx": int(c),
                            "pos_label_used": int(c),
                            "groups": ";".join(grp_keys),
                            "tpr_per_group": ";".join(tpr_strs),
                            "abs_eo": eo,
                            "non_vacuous_tpr": non_vacuous,
                            "n_groups_with_positives": int(
                                sum(1 for g in groups.values() if g["n_pos"] > 0)
                            ),
                        }
                    )
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
#  — paired bootstrap + Cohen's d on procedural-aggregate.
# ---------------------------------------------------------------------------

# Procedural-aggregate definition (documented for thesis methodology):
# unweighted mean of four metrics that all live in [0, 1] with higher=better:
#   * voice_representation
#   * voice_enrichment / max(1, voice_enrichment)  -- clipped at 1 for mixing
#                                                     into the mean (it can
#                                                     exceed 1; clipping is
#                                                     ONLY for aggregate)
#   * model_flippability_validity
#   * actionable_validity
#   * process_consistency at sigma = 0.3 (a 5th)
PROC_AGG_METRICS_NOISE_NA = (
    "voice_representation",
    "model_flippability_validity",
    "actionable_validity",
)
PROC_AGG_PC_NOISE = 0.3

#  — Procedural-aggregate weighting schemes.

# The 5 component metrics that enter the procedural aggregate (all in [0, 1],
# higher = fairer) are:
#     m1 = voice_representation
#     m2 = voice_enrichment (clipped at 1 to prevent feature-count
#                             inflation; see PROC_AGG_VE_CLIP)
#     m3 = model_flippability_validity
#     m4 = actionable_validity
#     m5 = process_consistency at σ = 0.3

# The 5 weighting schemes below answer "how do we combine these into a
# scalar?" — each implements a different theoretical prior on which
# procedural dimension to favour. A finding is **weighting-robust** iff it
# survives Holm-Bonferroni under ALL 5 schemes.

# Authorial decisions:
#   - "voice_heavy"  prioritises Newman-Fast-Harmon 2020's voice
#     dimension (m1+m2 share 0.60).
#   - "transparency_heavy" prioritises post-hoc explainability /
#     actionability (m3+m4 share 0.60).
#   - "consistency_heavy" prioritises process consistency (m5 carries
#     0.55; remaining 0.45 split equally over the other 4).
#   - "rank_aggregate" is a non-weighted alternative: per-(dataset, seed)
#     rank each model 1..N within each metric (1 = best, NaN-tolerant),
#     then average the per-metric ranks for the model. Lower-is-better
#     under this scheme — to keep the orientation "higher = fairer" we
#     map the per-model mean rank ``r`` to ``1 - (r - 1) / (N - 1)`` so
#     the best model on every metric scores 1.0 and the worst scores 0.0.
PROC_AGG_METRIC_KEYS = (
    "voice_representation",          # m1
    "voice_enrichment",              # m2 (clipped at 1)
    "model_flippability_validity",   # m3
    "actionable_validity",           # m4
    "process_consistency",           # m5 at σ=0.3
)
PROC_AGG_VE_CLIP = 1.0  # voice_enrichment clip into the aggregate

# Equal weights — 0.20 each. Verified to recover the prior unweighted-mean
# behaviour byte-identically on the existing N=5 procedural.csv (
# regression test).
_EQUAL = 1.0 / len(PROC_AGG_METRIC_KEYS)
WEIGHTING_SCHEMES: dict[str, dict[str, float]] = {
    "equal_weights": {k: _EQUAL for k in PROC_AGG_METRIC_KEYS},
    # voice_heavy: 0.30 + 0.30 = 0.60 to voice; remaining 0.40 / 3 ≈ 0.1333.
    "voice_heavy": {
        "voice_representation": 0.30,
        "voice_enrichment": 0.30,
        "model_flippability_validity": 0.40 / 3.0,
        "actionable_validity": 0.40 / 3.0,
        "process_consistency": 0.40 / 3.0,
    },
    # transparency_heavy: 0.30 + 0.30 = 0.60 to model_flippability +
    # actionable; remaining 0.40 / 3 split equally.
    "transparency_heavy": {
        "voice_representation": 0.40 / 3.0,
        "voice_enrichment": 0.40 / 3.0,
        "model_flippability_validity": 0.30,
        "actionable_validity": 0.30,
        "process_consistency": 0.40 / 3.0,
    },
    # consistency_heavy: 0.55 to process_consistency; remaining 0.45 / 4.
    "consistency_heavy": {
        "voice_representation": 0.45 / 4.0,
        "voice_enrichment": 0.45 / 4.0,
        "model_flippability_validity": 0.45 / 4.0,
        "actionable_validity": 0.45 / 4.0,
        "process_consistency": 0.55,
    },
    # rank_aggregate is special-cased — see _per_seed_procedural_aggregate.
    # Stored here (with NaN sentinel) so callers iterating over the dict
    # can pick up the scheme name.
    "rank_aggregate": {k: float("nan") for k in PROC_AGG_METRIC_KEYS},
}

def _per_seed_metric_components(
    df: pd.DataFrame, dataset: str, model: str
) -> dict[int, dict[str, float]]:
    """Return ``{seed: {metric_key: value}}`` for the 5 components that
    enter the procedural aggregate. NaN-tolerant: missing values absent
    from the dict for that seed.

    The return shape lets callers apply weighted means or per-metric
    ranks without re-querying the CSV.
    """
    sub = df[(df["dataset"] == dataset) & (df["model"] == model) & (df["seed"] >= 0)]
    seeds = sorted(sub["seed"].unique())
    out: dict[int, dict[str, float]] = {}
    for s in seeds:
        s_sub = sub[sub["seed"] == s]
        comps: dict[str, float] = {}
        for m in ("voice_representation", "model_flippability_validity",
                  "actionable_validity"):
            row = s_sub[(s_sub["metric"] == m) & (s_sub["noise_std"] == -1.0)]
            if len(row) and not pd.isna(row["value"].iloc[0]):
                comps[m] = float(row["value"].iloc[0])
        ve_row = s_sub[
            (s_sub["metric"] == "voice_enrichment")
            & (s_sub["noise_std"] == -1.0)
        ]
        if len(ve_row) and not pd.isna(ve_row["value"].iloc[0]):
            comps["voice_enrichment"] = min(
                PROC_AGG_VE_CLIP, float(ve_row["value"].iloc[0])
            )
        pc_row = s_sub[
            (s_sub["metric"] == "process_consistency")
            & (np.isclose(s_sub["noise_std"], PROC_AGG_PC_NOISE))
        ]
        if len(pc_row) and not pd.isna(pc_row["value"].iloc[0]):
            comps["process_consistency"] = float(pc_row["value"].iloc[0])
        out[int(s)] = comps
    return out

def _weighted_aggregate(
    components: dict[str, float], weights: dict[str, float]
) -> float:
    """Weighted mean of present components (re-normalising weights over
    the present subset to handle NaNs, byte-identical to the original
    unweighted-mean-over-present-keys when weights are equal).
    """
    total_w = 0.0
    total_v = 0.0
    for k, v in components.items():
        if k not in weights:
            continue
        w = float(weights[k])
        if not np.isfinite(w) or w == 0.0:
            continue
        total_w += w
        total_v += w * float(v)
    if total_w == 0.0:
        return float("nan")
    return total_v / total_w

def _rank_aggregate_per_seed(
    per_seed_components: dict[str, dict[int, dict[str, float]]],
    seed: int,
    model: str,
) -> float:
    """Rank-aggregate scheme: per-metric, rank the models present at this
    seed (1 = highest); return the mean rank for ``model`` mapped to
    ``[0, 1]`` with higher = fairer.

    ``per_seed_components`` is ``{model: {seed: {metric: value}}}``.
    """
    metric_keys = list(PROC_AGG_METRIC_KEYS)
    per_metric_ranks: list[float] = []
    for mk in metric_keys:
        # Models present for this metric at this seed.
        vals: list[tuple[str, float]] = []
        for m, seed_comps in per_seed_components.items():
            comps = seed_comps.get(seed, {})
            if mk in comps:
                vals.append((m, float(comps[mk])))
        if not vals:
            continue
        # Higher = fairer → rank 1 to the largest.
        vals.sort(key=lambda kv: -kv[1])
        rank_map = {m: r for r, (m, _) in enumerate(vals, start=1)}
        if model not in rank_map:
            continue
        N = len(vals)
        if N <= 1:
            scaled = 1.0
        else:
            # Map rank r ∈ [1, N] -> score in [0, 1] with rank 1 -> 1.0
            scaled = 1.0 - (rank_map[model] - 1) / (N - 1)
        per_metric_ranks.append(scaled)
    if not per_metric_ranks:
        return float("nan")
    return float(np.mean(per_metric_ranks))

def _per_seed_procedural_aggregate(
    df: pd.DataFrame, dataset: str, model: str,
    *, weighting_scheme: str = "equal_weights",
) -> np.ndarray:
    """Return a length-N_seeds array of per-seed procedural-aggregate scalars
    for the given (dataset, model) pair under ``weighting_scheme``.

    : supports 5 weighting schemes (see ``WEIGHTING_SCHEMES``).
    For ``rank_aggregate``, all models at the dataset are needed at once
    to compute per-metric ranks; this function therefore lazily collects
    components for all trained models.

    Byte-identical regression: ``equal_weights`` recovers the prior
    behaviour exactly when present-keys are equal across models, since
    ``_weighted_aggregate`` re-normalises over present components. The
     unit test verifies this.
    """
    if weighting_scheme not in WEIGHTING_SCHEMES:
        raise ValueError(
            f"Unknown weighting_scheme={weighting_scheme!r}; expected one of "
            f"{sorted(WEIGHTING_SCHEMES.keys())}"
        )
    if weighting_scheme == "rank_aggregate":
        # Need components for *all* models at this dataset for ranking.
        all_models = list(MODEL_ORDER)
        all_comps = {
            m: _per_seed_metric_components(df, dataset, m) for m in all_models
        }
        seeds = sorted({s for d in all_comps.values() for s in d.keys()})
        own = all_comps.get(model, {})
        out = []
        for s in seeds:
            if s not in own:
                out.append(float("nan"))
            else:
                out.append(_rank_aggregate_per_seed(all_comps, s, model))
        return np.asarray(out, dtype=float)

    weights = WEIGHTING_SCHEMES[weighting_scheme]
    comps_by_seed = _per_seed_metric_components(df, dataset, model)
    seeds = sorted(comps_by_seed.keys())
    out = []
    for s in seeds:
        agg = _weighted_aggregate(comps_by_seed[s], weights)
        out.append(agg)
    return np.asarray(out, dtype=float)

# ---------------------------------------------------------------------------
#  — variance-audited effect size + Holm-Bonferroni correction.
# ---------------------------------------------------------------------------

# : minimum-allowable pooled standard deviation for Cohen's d. With 5
# seeds and a deterministic model (e.g., LR which converges to a unique
# optimum on convex objectives), the empirical pooled sd of the per-seed
# differences can be ~0, driving Cohen's d to ±∞. We floor sd at this
# value to keep the effect-size scale interpretable. The choice of 0.005
# is documented per  / : it is roughly half the smallest
# meaningfully-resolvable procedural-aggregate increment (per-metric
# rounding to ~0.001 × 5 components ≈ 0.005). Effect sizes computed under
# the floor are flagged in the ``effect_size_kind`` column.
COHENS_D_VARIANCE_FLOOR = 0.005

def _superiority_probability(diffs: np.ndarray) -> float:
    """Probability of superiority on the per-seed paired diffs.

    Defined as ``P(diff > 0) + 0.5 * P(diff == 0)``. A non-parametric
    effect size that is robust to zero pooled variance — useful when one
    or both arms are deterministic across seeds.
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    if diffs.size == 0:
        return float("nan")
    pos = float((diffs > 0).sum())
    eq = float((diffs == 0).sum())
    return float((pos + 0.5 * eq) / diffs.size)

def _paired_bootstrap_gap(
    a: np.ndarray, b: np.ndarray, *, n_boot: int = 10_000, seed: int = 0
) -> dict[str, float | str]:
    """Paired-bootstrap on the gap (a - b) over the per-seed paired array.

     — extends the original return shape with two new fields:

    * ``effect_size_kind`` ∈ {``"cohens_d"``, ``"cohens_d_floored"``,
      ``"superiority_prob"``, ``"undefined"``}
    * ``notes`` — free-form string flagging zero-variance edge cases (used
      by both -2's tabulation and the headline-claims renderer).

    Effect-size policy (variance audit, ):
      * If pooled sd of diffs > ``COHENS_D_VARIANCE_FLOOR`` → vanilla
        Cohen's d, ``effect_size_kind = "cohens_d"``.
      * If 0 < pooled sd ≤ floor → flooring kicks in:
        ``d = obs / max(sd, COHENS_D_VARIANCE_FLOOR)``,
        ``effect_size_kind = "cohens_d_floored"`` (note in the column).
      * If pooled sd == 0 → both arms are deterministic. Cohen's d is
        undefined; we report the probability-of-superiority on the
        per-seed diffs instead (a non-parametric effect size in [0, 1]),
        ``effect_size_kind = "superiority_prob"``. Note flagged.
      * If both arms have identical per-seed values (diffs all zero) →
        no effect, ``effect_size_kind = "undefined"``.
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
        }
    diffs = a - b
    obs = float(np.mean(diffs))
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = float(np.mean(diffs[idx]))
    ci_lo, ci_hi = float(np.percentile(boot_means, 2.5)), float(
        np.percentile(boot_means, 97.5)
    )
    # Two-sided p-value: fraction of bootstrap means whose sign opposes
    # the observed sign. If obs > 0, p = 2 * Pr[boot_mean <= 0]; if
    # obs < 0, p = 2 * Pr[boot_mean >= 0]; cap at 1.0.
    if obs >= 0:
        p_one = float((boot_means <= 0).mean())
    else:
        p_one = float((boot_means >= 0).mean())
    p_value = min(1.0, 2.0 * p_one)

    # : variance-audited Cohen's d.
    sd_diffs = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    notes_parts: list[str] = []
    a_sd = float(np.std(a, ddof=1)) if n > 1 else 0.0
    b_sd = float(np.std(b, ddof=1)) if n > 1 else 0.0
    # : arm is "effectively deterministic" if its per-seed std is at
    # least an order of magnitude below the variance floor. This catches
    # cases where one arm is a closed-form / convex optimiser (e.g.,
    # LogisticRegression on a separable dataset) and converges to the
    # same solution every seed; raw std may be ~1e-6 instead of exact 0.
    # Threshold 5e-4 = 0.1 * COHENS_D_VARIANCE_FLOOR.
    DETERMINISTIC_SD = 0.1 * COHENS_D_VARIANCE_FLOOR  # 0.0005
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
        # Both arms deterministic → use probability-of-superiority.
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
    return {
        "mean": obs,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "effect_size_kind": kind,
        "n_bootstrap": n_boot,
        "notes": "; ".join(notes_parts),
    }

def holm_bonferroni_correction(
    p_values: np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Holm step-down multiple-comparison correction.

    Procedure (Holm 1979):
      1. Sort the n p-values ascending.
      2. For the i-th sorted p (1-indexed), the corrected α is
         ``alpha / (n - i + 1)``.
      3. Reject H0_i iff p_i < α_i AND every preceding H0_(<i) was also
         rejected (step-down).
      4. The Holm-corrected p-value is ``min(1, max_{j<=i} (n-j+1) * p_j)``
         (running max ensures monotonicity).

    Inputs/outputs:
      * ``p_values`` — array-like of length n. NaNs are passed through to
        ``rejected[i] = False`` and ``p_holm[i] = NaN``.
      * Returns ``(rejected_mask, p_holm)`` aligned with the *input* order.

    Verified against Wikipedia's textbook example
    ``[0.001, 0.013, 0.014, 0.190, 0.350]`` at α=0.05: only the first p
    is rejected; the corrected p-values are
    ``[0.005, 0.052, 0.052, 0.380, 0.380]``.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    rejected = np.zeros(n, dtype=bool)
    p_holm = np.full(n, np.nan, dtype=float)
    if n == 0:
        return rejected, p_holm
    # Stable sort (lower-index ties keep input order); NaNs sort last.
    finite_mask = ~np.isnan(p)
    finite_idx = np.where(finite_mask)[0]
    if finite_idx.size == 0:
        return rejected, p_holm
    order = finite_idx[np.argsort(p[finite_idx], kind="stable")]
    m = len(order)
    # Running max of (n_remaining * p) gives the corrected p.
    running_max = 0.0
    stop_rejecting = False
    for i, idx in enumerate(order):
        n_remaining = m - i  # m, m-1, ..., 1
        adj = n_remaining * p[idx]
        if adj > running_max:
            running_max = adj
        p_holm[idx] = min(1.0, running_max)
        if not stop_rejecting and p[idx] < alpha / n_remaining:
            rejected[idx] = True
        else:
            stop_rejecting = True
    return rejected, p_holm

def _statistical_rank_per_dataset(
    tpr_df: pd.DataFrame, dataset: str
) -> dict[str, int]:
    """Return ``{model: rank}`` (rank=1 = best) using statistical |EO| from
    ``per_group_tpr.csv`` (lower = fairer).

    For multiclass datasets, average |EO| across non-vacuous classes
    (Phase-3 macro_filtered convention reused informally here).
    """
    sub = tpr_df[(tpr_df["dataset"] == dataset)]
    rows: list[tuple[str, float]] = []
    for m in TRAINED_MODELS:
        m_sub = sub[sub["model"] == m]
        if not len(m_sub):
            continue
        if (m_sub["target_form"] == "multiclass").any():
            non_vac = m_sub[m_sub["non_vacuous_tpr"]]
            if len(non_vac):
                eo = float(non_vac["abs_eo"].mean())
            else:
                eo = float(m_sub["abs_eo"].mean())
        else:
            eo = float(m_sub["abs_eo"].iloc[0])
        if not np.isnan(eo):
            rows.append((m, eo))
    rows.sort(key=lambda kv: kv[1])  # ascending — lower |EO| is fairer
    return {m: r for r, (m, _) in enumerate(rows, start=1)}

def _procedural_rank_per_dataset(
    proc_df: pd.DataFrame, dataset: str,
    *, weighting_scheme: str = "equal_weights",
) -> dict[str, int]:
    """Return ``{model: rank}`` (rank=1 = best) using mean per-seed
    procedural-aggregate (higher = fairer)."""
    means: list[tuple[str, float]] = []
    for m in TRAINED_MODELS:
        agg = _per_seed_procedural_aggregate(
            proc_df, dataset, m, weighting_scheme=weighting_scheme
        )
        if len(agg) == 0 or np.all(np.isnan(agg)):
            continue
        means.append((m, float(np.nanmean(agg))))
    means.sort(key=lambda kv: -kv[1])  # descending — higher = fairer
    return {m: r for r, (m, _) in enumerate(means, start=1)}

# ---------------------------------------------------------------------------
#  — Spearman-rank separability test (procedural vs statistical
# rankings, per dataset, with bootstrap CI on ρ).
# ---------------------------------------------------------------------------

def _spearman_rho_bootstrap_ci(
    proc_rank_per_seed: dict[int, dict[str, int]],
    stat_rank: dict[str, int],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap CI on Spearman ρ between procedural-rank and statistical-
    rank vectors over the trained-models axis.

    The per-seed procedural ranks are bootstrapped over seeds (sampling
    with replacement), and for each resample we compute Spearman ρ
    against the (constant) statistical-rank vector. Returns the mean ρ,
    95 % percentile CI, and a p-value for H₀: ρ = 1 (the perfect-
    agreement null) computed as ``2 * P(boot_rho >= 1)`` clipped to
    [0, 1] (equivalently, the fraction of resamples with ρ < 1, doubled).
    """
    from scipy.stats import spearmanr

    seeds = sorted(proc_rank_per_seed.keys())
    if not seeds:
        return {
            "mean": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "p_value": float("nan"),
            "n_bootstrap": n_boot,
        }
    # Per-seed ρ between procedural rank vector and statistical rank vector.
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
    per_seed_rho = np.asarray(per_seed_rho, dtype=float)
    if per_seed_rho.size == 0:
        return {
            "mean": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "p_value": float("nan"),
            "n_bootstrap": n_boot,
        }
    # Bootstrap over seeds.
    rng = np.random.default_rng(seed)
    n = per_seed_rho.size
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(per_seed_rho[idx]))
    mean_rho = float(np.mean(per_seed_rho))
    ci_lo, ci_hi = (
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    )
    # One-sided p-value for H₀: ρ = 1. The alternative is ρ < 1. Under
    # H₀, all bootstrap resamples have ρ = 1; under H₁, the observed
    # bootstrap distribution sits strictly below 1. We therefore compute
    # the percentile bootstrap p-value as ``P(boot_rho >= 1)`` — the
    # fraction of resamples where ρ would be at-least-as-consistent with
    # the null. When the upper-CI is below 1, p ≈ 0 (reject H₀); when
    # ρ is centred around 1, p ≈ 1 (fail to reject).
    p_value = float((boot >= 1.0).mean())
    return {
        "mean": mean_rho,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": p_value,
        "n_bootstrap": n_boot,
    }

def compute_significance_table(
    proc_df: pd.DataFrame, tpr_df: pd.DataFrame, log
) -> pd.DataFrame:
    """ +  +  — pairwise procedural-gap bootstrap + separability
    bootstrap + Spearman-rank separability test, **swept over 5 weighting
    schemes**.

    For each (dataset, weighting_scheme):
      * pairwise paired-bootstrap of procedural-aggregate gaps
        (``comparison_type = "procedural_gap"``);
      * pairwise rank-disagreement bootstrap reusing the per-seed
        procedural ranks (``comparison_type = "separability_gap"``);
      * one Spearman-ρ row (``comparison_type = "rank_disagreement"``)
        per dataset reporting bootstrap-CI on ρ between procedural and
        statistical rankings, with H₀: ρ=1 p-value.

    Schema column ``weighting_scheme`` takes values from the keys of
    ``WEIGHTING_SCHEMES`` (5 schemes). The ``rank_disagreement`` rows
    use ``model_a = NULL`` / ``model_b = NULL``.
    """
    rows: list[dict[str, Any]] = []
    datasets = sorted(proc_df["dataset"].unique())
    schemes = list(WEIGHTING_SCHEMES.keys())
    for ds in datasets:
        for scheme in schemes:
            log(f"significance / dataset={ds} / scheme={scheme}")
            # Per-seed aggregates per model under this scheme.
            per_seed: dict[str, np.ndarray] = {}
            for m in TRAINED_MODELS:
                agg = _per_seed_procedural_aggregate(
                    proc_df, ds, m, weighting_scheme=scheme
                )
                per_seed[m] = agg

            # Pairwise procedural-gap bootstrap.
            models = [
                m for m in TRAINED_MODELS if not np.all(np.isnan(per_seed[m]))
            ]
            for i, ma in enumerate(models):
                for mb in models[i + 1 :]:
                    stats = _paired_bootstrap_gap(
                        per_seed[ma], per_seed[mb], n_boot=10_000, seed=0
                    )
                    rows.append(
                        {
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
                        }
                    )

            # Separability-gap bootstrap.
            stat_rank = _statistical_rank_per_dataset(tpr_df, ds)
            if not stat_rank:
                log(
                    f"  no statistical rank for {ds}; skipping separability rows"
                )
                continue
            # Per-seed procedural ranks (under the current scheme).
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
                for mb in models[i + 1 :]:
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
                    stats = _paired_bootstrap_gap(
                        np.asarray(a_seed), np.asarray(b_seed),
                        n_boot=10_000, seed=1,
                    )
                    rows.append(
                        {
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
                        }
                    )

            #  — Spearman-ρ rank-disagreement bootstrap.
            rho_stats = _spearman_rho_bootstrap_ci(
                proc_rank_by_seed, stat_rank, n_boot=10_000, seed=2,
            )
            rows.append(
                {
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
                        f"Spearman ρ between procedural-rank and "
                        f"statistical-rank vectors over trained models; "
                        f"H₀: ρ=1"
                    ),
                }
            )

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Headline claims markdown.
# ---------------------------------------------------------------------------

def _is_artefactual_determinism(notes: str, kind: str) -> bool:
    """ — heuristic: is the claim's apparent strength inflated by an
    artefactual deterministic arm?

    Triggers when:
      * ``effect_size_kind`` is ``cohens_d_floored`` or ``undefined`` or
        ``superiority_prob`` (probability of superiority is fine but the
        underlying determinism context warrants framing); OR
      * the notes column flags a deterministic arm.
    """
    if kind in {"cohens_d_floored", "undefined", "superiority_prob"}:
        return True
    if "deterministic" in notes:
        return True
    return False

def _framing_prefix(row: pd.Series, winner_label: str, loser_label: str) -> str:
    """ — generate a contextualizing sentence to prepend to a claim
    where artefactual-determinism inflates apparent claim strength.

    Returns "" when no framing is needed; otherwise returns a sentence
    that does not duplicate any phrase already in ``notes``.
    """
    notes = str(row.get("notes", "") or "")
    kind = str(row.get("effect_size_kind", "cohens_d"))
    if not _is_artefactual_determinism(notes, kind):
        return ""
    ds = str(row.get("dataset", ""))
    # Tag the deterministic arm — try to figure out which arm.
    a_det = "model_a effectively deterministic" in notes
    b_det = "model_b effectively deterministic" in notes
    pooled_zero = "pooled sd(diffs)=0" in notes
    floored = kind == "cohens_d_floored"
    undefined = kind == "undefined"
    # Which arm corresponds to the "loser" label after re-orientation?
    # The mean is sign-flipped during _format_claim — so winner_label is
    # whichever model has higher procedural aggregate. The deterministic
    # arm in the original (a, b) ordering is ``a`` if a_det (and
    # symmetrically for b_det). We just describe it as "the deterministic
    # arm" without trying to map back perfectly.
    parts = []
    if undefined:
        parts.append(
            f"On {ds}, both arms collapse to identical per-seed values; "
            "the apparent gap is exactly zero and the effect size is undefined."
        )
    elif pooled_zero:
        parts.append(
            f"On {ds}, both models' per-seed procedural aggregates are "
            "deterministic (pooled diff sd = 0); P(superiority) reports the "
            "fraction of seeds where the winner exceeded the loser, not a "
            "Cohen's d effect size."
        )
    elif floored or a_det or b_det:
        parts.append(
            f"On {ds}, the {loser_label if (a_det) else winner_label if (b_det) else 'losing'}"
            f" arm has near-zero per-seed variance (effectively deterministic "
            "across seeds); the reported Cohen's d is large in magnitude "
            "because the diff sd is artefactually small, NOT because the "
            "procedural gap is large in absolute terms — read the mean-gap "
            "and CI as the primary effect-size, not the d."
        )
    if not parts:
        return ""
    return " ".join(parts)

def _format_claim(row: pd.Series) -> str:
    """One-line procedural-gap claim formatted for the headline_claims.md.

    The CSV stores ``gap = proc_agg(a) - proc_agg(b)``. We re-orient the
    text so the *winner* (higher procedural aggregate) appears first, and
    the printed CI / Cohen's d are sign-flipped accordingly so they refer
    to ``proc_agg(winner) - proc_agg(loser)`` (always positive mean).

     / : surfaces ``effect_size_kind`` (Cohen's d vs probability
    of superiority vs floored Cohen's d) and the Holm-corrected p-value
    ``p_holm``. : the renderer is only called for Holm-survivors,
    so all displayed claims are family-wise-α=0.05 significant.

    : when ``effect_size_kind ∈ {cohens_d_floored,
    undefined, superiority_prob}`` OR ``notes`` mentions a deterministic
    arm, prepend a contextualising sentence so the prose does not
    overclaim from artefactual determinism.
    """
    ma = MODEL_LABELS.get(row["model_a"], row["model_a"])
    mb = MODEL_LABELS.get(row["model_b"], row["model_b"])
    kind = str(row.get("effect_size_kind", "cohens_d"))
    if row["mean"] >= 0:
        winner, loser = ma, mb
        mean = float(row["mean"])
        ci_lo = float(row["ci_lo"])
        ci_hi = float(row["ci_hi"])
        d = float(row["cohens_d"])
    else:
        winner, loser = mb, ma
        mean = -float(row["mean"])
        ci_lo = -float(row["ci_hi"])
        ci_hi = -float(row["ci_lo"])
        if kind == "superiority_prob":
            d = 1.0 - float(row["cohens_d"])
        else:
            d = -float(row["cohens_d"])
    if kind == "superiority_prob":
        eff_str = f"P(superiority) = {d:.2f}"
    elif kind == "cohens_d_floored":
        eff_str = f"Cohen's d = {d:+.2f} (sd-floored)"
    elif kind == "undefined":
        eff_str = "effect size undefined (zero diff variance)"
    else:
        eff_str = f"Cohen's d = {d:+.2f}"
    p_holm = row.get("p_holm", float("nan"))
    p_str = (
        f"p_Holm = {float(p_holm):.3g}"
        if not pd.isna(p_holm)
        else f"p_raw = {row['p_value']:.3g}"
    )
    note = str(row.get("notes", "") or "")
    note_suffix = f" — {note}" if note else ""
    framing = _framing_prefix(row, winner, loser)
    framing_prefix = f"{framing} " if framing else ""
    return (
        f"- {framing_prefix}**{winner} procedurally outperforms {loser}** "
        f"with mean gap **{mean:.3f}**, 95 % CI [{ci_lo:+.3f}, {ci_hi:+.3f}], "
        f"{eff_str}, {p_str}.{note_suffix}"
    )

def _weighting_robust_pairs(
    sig_df: pd.DataFrame,
) -> set[tuple[str, str, str]]:
    """ /  — return the set of ``(dataset, model_a, model_b)``
    procedural-gap pairs that survive Holm-Bonferroni under ALL 5
    weighting schemes.

    These are the "weighting-robust" headline claims; the
    ``headline_claims.md`` reports only these.
    """
    schemes = list(WEIGHTING_SCHEMES.keys())
    sub = sig_df[sig_df["comparison_type"] == "procedural_gap"]
    # Build per-pair set of schemes where survived.
    survived: dict[tuple[str, str, str], set[str]] = {}
    for _, r in sub.iterrows():
        if not bool(r.get("rejected_holm", False)):
            continue
        key = (str(r["dataset"]), str(r["model_a"]), str(r["model_b"]))
        survived.setdefault(key, set()).add(str(r["weighting_scheme"]))
    robust = {k for k, sset in survived.items() if sset >= set(schemes)}
    return robust

def render_headline_claims(
    sig_df: pd.DataFrame, out_path: pathlib.Path, log
) -> None:
    """Holm-corrected, weighting-robust procedural / separability /
    rank-disagreement headlines per dataset.

    Reports ONLY claims that:
      * survive Holm-Bonferroni at family-wise α=0.05 (across the full
        scheme-spanned family in ``significance.csv``); AND
      * (for procedural-gap rows) survive under ALL 5 weighting schemes
        (= weighting-robust).

    Surfaces:
      * weighting-robust procedural-gap survivors (one section per
        dataset; effect-size kind respected per );
      * separability-gap survivors (rank-gap reorientation);
      * rank-disagreement Spearman ρ rows from  with bootstrap CI
        and p-value for H₀: ρ=1.

     framing: artefactual-determinism cases are prefixed with a
    contextualising sentence so the prose does not overclaim.
    """
    n_total = int(len(sig_df))
    n_proc = int((sig_df["comparison_type"] == "procedural_gap").sum())
    n_sep = int((sig_df["comparison_type"] == "separability_gap").sum())
    n_rd = int((sig_df["comparison_type"] == "rank_disagreement").sum())
    n_rejected = int(sig_df["rejected_holm"].sum())

    robust_pairs = _weighting_robust_pairs(sig_df)

    lines: list[str] = []
    lines.append(
        "# Phase 4 followup³ — Holm-corrected, weighting-robust headline "
        "procedural-fairness claims"
    )
    lines.append("")
    lines.append(
        "Auto-generated from `results/phase4/significance.csv` (paired "
        "bootstrap n=10,000) by `scripts/run_phase4_significance.py`. "
        "**Procedural aggregate** = weighted mean of voice_representation, "
        "min(1, voice_enrichment), model_flippability_validity, "
        "actionable_validity, and process_consistency at σ=0.3 "
        "(5 metrics in [0, 1] with higher = fairer). "
        "Five weighting schemes are evaluated (ADR-022 / T-093): "
        "`equal_weights`, `voice_heavy`, `transparency_heavy`, "
        "`consistency_heavy`, `rank_aggregate`. The voice_enrichment "
        "component is clipped at 1.0 before mixing into the aggregate "
        "(ADR-020) to prevent its theoretical maximum (n_total/n_modifiable) "
        "from dominating; we leave the un-clipped value in `procedural.csv` "
        "for downstream analysis."
    )
    lines.append("")
    lines.append(
        f"After **Holm-Bonferroni step-down correction at family-wise α=0.05 "
        f"across {n_total} hypothesis tests** ({n_proc} pairwise "
        f"procedural-gap tests + {n_sep} separability-gap tests + "
        f"{n_rd} rank-disagreement tests, swept over 5 weighting schemes), "
        f"{n_rejected} survivors are reported. **The procedural-gap "
        f"section reports only the {len(robust_pairs)} pairs that survive "
        "Holm under ALL 5 schemes (= weighting-robust).** Pairs that survive "
        "under 1-4 schemes but not all 5 are scheme-fragile and are not "
        "promoted to the headline."
    )
    lines.append("")
    lines.append(
        "Effect sizes are Cohen's d for paired diffs unless otherwise noted "
        "(T-084): per-pair pooled-sd is audited; if the pooled sd is below "
        "**0.005** the d denominator is floored (`effect_size_kind = "
        "cohens_d_floored`), and if both arms are deterministic across seeds "
        "(pooled sd = 0) we report probability-of-superiority instead "
        "(`effect_size_kind = superiority_prob`). T-097 (ADR-022): "
        "claims with artefactual-determinism (one or both arms collapse to "
        "majority-class predictions across all seeds) are prepended with a "
        "contextualising sentence so the prose does not overclaim from "
        "vanishing variance — read the mean-gap and CI as the primary "
        "effect-size in such rows, NOT the inflated Cohen's d."
    )
    lines.append("")
    lines.append(
        "Per ADR-022, the headline procedural-gap rows below use the "
        "`equal_weights` scheme as the displayed effect size (consistent "
        "with the prior follow-up²); the weighting-robustness filter "
        "above ensures the *qualitative* claim survives under all 5 "
        "schemes. Per-scheme rows for the same pair are available in "
        "`significance.csv`."
    )
    lines.append("")

    datasets = sorted(sig_df["dataset"].unique())
    for ds in datasets:
        lines.append(f"## {ds}")
        lines.append("")
        # Procedural-gap rows: filter to weighting-robust + use
        # equal_weights for prose presentation.
        proc = sig_df[
            (sig_df["dataset"] == ds)
            & (sig_df["comparison_type"] == "procedural_gap")
            & (sig_df["weighting_scheme"] == "equal_weights")
        ].copy()
        sep = sig_df[
            (sig_df["dataset"] == ds)
            & (sig_df["comparison_type"] == "separability_gap")
            & (sig_df["weighting_scheme"] == "equal_weights")
        ].copy()
        rank_dis = sig_df[
            (sig_df["dataset"] == ds)
            & (sig_df["comparison_type"] == "rank_disagreement")
        ].copy()

        # Filter to weighting-robust pairs (survives under ALL 5 schemes).
        proc["__robust"] = proc.apply(
            lambda r: (
                str(r["dataset"]),
                str(r["model_a"]),
                str(r["model_b"]),
            ) in robust_pairs,
            axis=1,
        )
        proc_sig = proc[proc["__robust"]].copy()
        if not proc_sig.empty:
            def _abs_effect(row):
                v = row.get("cohens_d", float("nan"))
                if pd.isna(v):
                    return -1.0
                if str(row.get("effect_size_kind", "cohens_d")) == "superiority_prob":
                    return abs(float(v) - 0.5)
                return abs(float(v))

            proc_sig["__abs_effect"] = proc_sig.apply(_abs_effect, axis=1)
            proc_sig = proc_sig.sort_values("__abs_effect", ascending=False)
            for _, r in proc_sig.iterrows():
                lines.append(_format_claim(r))
        else:
            n_scheme_fragile = int(
                proc[proc["rejected_holm"] == True].shape[0]  # noqa: E712
            )
            if n_scheme_fragile:
                lines.append(
                    f"- _No procedural-fairness gap on this dataset is "
                    f"weighting-robust (survives Holm under ALL 5 schemes). "
                    f"{n_scheme_fragile} pair(s) survive under "
                    "`equal_weights` only; see `significance.csv` for the "
                    "full per-scheme breakdown._"
                )
            else:
                lines.append(
                    "- _No Holm-corrected significant pairwise "
                    "procedural-fairness gaps on this dataset under any "
                    "weighting scheme._"
                )

        #  — Spearman-ρ rank-disagreement (per dataset, equal_weights
        # row promoted to headline; per-scheme rows in CSV).
        rd_eq = rank_dis[rank_dis["weighting_scheme"] == "equal_weights"]
        if not rd_eq.empty:
            r = rd_eq.iloc[0]
            rho = float(r["mean"])
            ci_lo = float(r["ci_lo"])
            ci_hi = float(r["ci_hi"])
            p_val = float(r["p_value"])
            p_holm = r.get("p_holm", float("nan"))
            p_str = (
                f"p_Holm = {float(p_holm):.3g}"
                if not pd.isna(p_holm)
                else f"p_raw = {p_val:.3g}"
            )
            rejected_h0 = (
                not pd.isna(p_val) and p_val < 0.05
            )
            verdict = (
                "rejecting H₀: ρ=1 — procedural and statistical rankings "
                "are statistically distinct"
                if rejected_h0
                else "consistent with H₀: ρ=1 — procedural and statistical "
                "rankings are not statistically distinguishable on this "
                "dataset under the rank-correlation test"
            )
            lines.append(
                f"- **Procedural-vs-statistical Spearman rank correlation:** "
                f"ρ = {rho:+.2f}, 95 % CI [{ci_lo:+.2f}, {ci_hi:+.2f}], "
                f"{p_str}, {verdict} (T-094, equal_weights scheme)."
            )

        # Separability-gap (paired rank-gap shift; only equal_weights).
        if not sep.empty:
            sep_sig = sep[sep["rejected_holm"] == True].copy()  # noqa: E712
            if not sep_sig.empty:
                sep_sig["abs_mean"] = sep_sig["mean"].abs()
                sep_sig = sep_sig.sort_values("abs_mean", ascending=False)
                top = sep_sig.iloc[0]
                p_holm_str = (
                    f"p_Holm = {float(top['p_holm']):.3g}"
                    if not pd.isna(top.get("p_holm", float("nan")))
                    else f"p_raw = {top['p_value']:.3g}"
                )
                lines.append(
                    f"- **Pairwise rank-gap shift survives Holm correction** "
                    f"for {MODEL_LABELS.get(top['model_a'], top['model_a'])} "
                    f"vs {MODEL_LABELS.get(top['model_b'], top['model_b'])}: "
                    f"shift = {top['mean']:+.2f}, 95 % CI "
                    f"[{top['ci_lo']:+.2f}, {top['ci_hi']:+.2f}], "
                    f"{p_holm_str}."
                )
            else:
                lines.append(
                    "- _No pairwise rank-gap shift survives Holm "
                    "correction under `equal_weights`._"
                )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"Wrote {out_path}")

# ---------------------------------------------------------------------------
# Top-level driver.
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_phase4_significance",
        description=(
            "Phase-4 followup: per-group TPR table (T-079) + paired-bootstrap "
            "significance + Cohen's d (T-081). Emits "
            "results/phase4/per_group_tpr.csv, results/phase4/significance.csv, "
            "and results/phase4/headline_claims.md."
        ),
    )
    parser.add_argument(
        "--procedural-csv",
        type=str,
        default=str(_PROJECT_ROOT / "results" / "phase4" / "procedural.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(_PROJECT_ROOT / "results" / "phase4"),
    )
    args = parser.parse_args(argv)

    proc_csv = pathlib.Path(args.procedural_csv)
    if not proc_csv.exists():
        print(
            f"ERROR: procedural CSV not found at {proc_csv}. "
            "Run `make phase4` first.",
            file=sys.stderr,
        )
        return 2
    out_dir = pathlib.Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        print(msg, flush=True)

    proc_df = pd.read_csv(proc_csv)

    tpr_path = out_dir / "per_group_tpr.csv"
    tpr_df = compute_per_group_tpr_table(_log)
    tpr_df = tpr_df.sort_values(
        ["dataset", "model", "target_form", "class_idx"]
    ).reset_index(drop=True)
    tpr_df.to_csv(tpr_path, index=False)
    _log(f"Wrote {tpr_path} ({len(tpr_df)} rows)")

    #  +  (Holm-Bonferroni family-wise correction) + /094
    sig_path = out_dir / "significance.csv"
    sig_df = compute_significance_table(proc_df, tpr_df, _log)
    sig_df = sig_df.sort_values(
        ["dataset", "weighting_scheme", "comparison_type", "model_a", "model_b"],
        na_position="last",
    ).reset_index(drop=True)

    # : Holm-Bonferroni correction across the full family of tests
    # (procedural_gap + separability_gap, all datasets). Preserves CSV
    # row count; adds two new columns (`p_holm`, `rejected_holm`).
    rejected, p_holm = holm_bonferroni_correction(
        sig_df["p_value"].to_numpy(), alpha=0.05
    )
    sig_df["p_holm"] = p_holm
    sig_df["rejected_holm"] = rejected

    sig_df.to_csv(sig_path, index=False)
    _log(
        f"Wrote {sig_path} ({len(sig_df)} rows; "
        f"{int(rejected.sum())} survive Holm at family-wise α=0.05)"
    )

    # Headline claims
    render_headline_claims(sig_df, out_dir / "headline_claims.md", _log)
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

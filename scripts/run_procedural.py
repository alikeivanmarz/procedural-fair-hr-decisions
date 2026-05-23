"""Phase-4 procedural-fairness audit runner.

Computes the Phase-4 procedural-fairness metrics from
``src/procedural_fairness.py`` on five (dataset, target) combinations
crossed with eight model families (six trained + two null baselines),
across multiple seeds and noise levels, and emits a single CSV.

Per  §2 +  (Phase-4 followup scope):

    Datasets / targets / models
    ===========================

    | Dataset            | Target              | Notes (ADR)              |
    |--------------------|---------------------|--------------------------|
    | D1 IBM HR          | Attrition           | Honest         |
    | D1 IBM HR          | PerformanceRating   | Leaky          |
    | D6 OULAD-3         | course_outcome      | Multi-class robustness   |
    | D2 ACS-Income      | high_income         |  /           |
    | D5 Dutch Census    | high_level_occ      |  /           |

    Models (one per row):
      * RandomForestClassifier(n_estimators=100, random_state=seed)
      * LogisticRegression(max_iter=1000, random_state=seed)
      * MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=seed)
      * XGBClassifier(n_estimators=100, random_state=seed) [ / ]
      * GradientBoostingClassifier(n_estimators=100, random_state=seed)
      * KNeighborsClassifier(n_neighbors=5)
      * ConstantPredictor (majority class)              [ / ]
      * ShuffledPredictor (RF predictions permuted)     [ / ]

The runner uses the modifiable / immutable feature partitions documented
verbatim in  + the  ADDENDUM for ACS / Dutch.

Per , when target is ``PerformanceRating`` we drop
``PercentSalaryHike`` from features (the leaky covariate). The audit row's
``notes`` column flags this with ``"leaky"``.

Per  +  the consistency metric uses Gaussian noise +
categorical resample (no DiCE).  sweeps ``noise_std`` over
{0.1, 0.3, 1.0, 3.0} so the metric is reported as a curve rather than
a saturated point.

Determinism:
    * MKL/OMP/OpenBLAS thread counts pinned to 1 + ``PYTHONHASHSEED=0``
      BEFORE numpy / sklearn imports.
    * Every metric call receives an explicit ``rng`` parameter.
    * SHAP background uses ``shap.sample(..., random_state=seed)`` and
      TreeExplainer uses ``feature_perturbation="interventional"``.

Schema (``results/phase4/procedural.csv``):

    | Column         | Notes                                                |
    |----------------|------------------------------------------------------|
    | dataset        | str — dataset key                                    |
    | target         | str — target attribute                               |
    | model          | str — model class name (or ``ConstantPredictor`` /  |
    |                | ``ShuffledPredictor``)                               |
    | metric         | str — one of ``process_consistency``,                |
    |                | ``voice_representation``, ``voice_enrichment``,      |
    |                | ``transparency_sparsity``, ``transparency_validity``,|
    |                | ``model_flippability_sparsity``,                     |
    |                | ``model_flippability_validity``,                     |
    |                | ``actionable_validity``, ``actionable_sparsity``     |
    | value          | float                                                |
    | seed           | int — 0..N-1 for per-seed rows; -1 for aggregated    |
    |                | (sentinel; see "Aggregation convention" below).      |
    | mean           | float — only populated on the seed=-1 sentinel row;  |
    |                | NaN otherwise.                                       |
    | std            | float — same.                                        |
    | ci_lo          | float — bootstrap 95 % CI lower; same.               |
    | ci_hi          | float — bootstrap 95 % CI upper; same.               |
    | noise_std      | float — perturbation σ (only meaningful for          |
    |                | process_consistency; ``-1.0`` sentinel for other     |
    |                | metrics so the column is never missing).             |
    | target_form    | ``binary`` or ``multiclass``                         |
    | n_classes      | int                                                  |
    | sample_n       | int — actual N used (capped at len(X_test))          |
    | random_state   | int — seed (== seed column for per-seed rows;        |
    |                | -1 on aggregated rows).                              |
    | notes          | str — ``leaky`` for PerformanceRating; empty else    |

Aggregation convention:
    For each (dataset, target, model, metric, noise_std) tuple the
    runner emits ``len(--seeds)`` rows (one per seed; ``seed = 0..N-1``)
    PLUS one ``seed = -1`` aggregated row carrying:
        * ``value`` == bootstrap mean,
        * ``mean`` == ``np.mean(per_seed_values)``,
        * ``std``  == ``np.std(per_seed_values, ddof=0)``,
        * ``ci_lo, ci_hi`` == 95 % percentile bootstrap CI on the mean
          (10,000 resamples).
    Per-seed rows have ``mean / std / ci_lo / ci_hi == NaN``. The
    sentinel was chosen over a separate ``aggregate=True/False``
    column because it keeps the schema flat with no NaN-vs-bool
    type collisions; downstream consumers filter ``df[df.seed >= 0]``
    for per-seed work or ``df[df.seed == -1]`` for plotting CIs.

CLI
---
.. code-block:: bash

    python scripts/run_phase4_procedural.py \\
        [--out-dir results/phase4/] \\
        [--sample-n 500] \\
        [--datasets ibm_hr_attrition,...] \\
        [--models RF,LR,MLP,XGB,GB,KNN,Constant,Shuffled] \\
        [--seeds 0,1,2,3,4] \\
        [--noise-stds 0.1,0.3,1.0,3.0]

References
----------
* `` — IBM HR target switch.
* `` §2 — Phase-4 scope.
* `` — Gaussian-noise perturbation.
* `` (+  ADDENDUM) — modifiable/immutable
  partitions for IBM HR / OULAD / ACS-Income / Dutch Census.
* `` — Phase-4 methodology hardening (this script
  is its .. +  implementation).
* the project documentation — procedural-fairness signature.
"""

from __future__ import annotations

#  — Determinism prelude. MUST come BEFORE numpy / sklearn imports.
import os

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from typing import Any, Callable  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
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

# Ensure ``src.*`` resolves from the project root.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from procedural_fair_hr.data_loaders import (  # noqa: E402
    load_acs,
    load_ibm_hr,
    load_oulad,
)
from procedural_fair_hr.procedural_fairness import (  # noqa: E402
    _xgboost_shap_compat_patch,
    explanation_actionability,
    model_flippability,
    process_consistency,
    transparency_metrics,
    voice_representation,
)

# ---------------------------------------------------------------------------
#  /  — SHAP-XGBoost compatibility patch.
# ---------------------------------------------------------------------------

# The compat patch was promoted to ``src/procedural_fairness.py`` in
# so that ``voice_representation`` can apply it unconditionally on tree
# explainers (the previous Phase-4 audit had to wrap the call manually,
# which the sensitivity script forgot to do — see results/phase4/
# sensitivity.log).  also extended the patch to handle the
# multi-class ``base_score`` form ``"[v0,v1,...,vK]"`` by reducing the
# per-class vector to its mean. The symbol is re-exported here for
# backward-compat with ``tests/test_phase4_audit.py::
# test_xgb_label_encoder_subclass_shap_finite``.
__all_compat__ = (_xgboost_shap_compat_patch,)

# ---------------------------------------------------------------------------
#  (+  ADDENDUM) — modifiable / immutable feature partition.
# ---------------------------------------------------------------------------

# Single source of truth:  + the  ADDENDUM. A
# change to either partition requires a superseding ADR. The lists are
# written verbatim from those ADRs (DO NOT redefine inline anywhere else;
#  single-source-of-truth).
ADR019_PARTITIONS: dict[str, dict[str, list[str]]] = {
    "ibm_hr_attrition": {
        "modifiable": [
            "JobInvolvement",
            "JobSatisfaction",
            "EnvironmentSatisfaction",
            "RelationshipSatisfaction",
            "WorkLifeBalance",
            "OverTime",
            "PerformanceRating",
            "MonthlyIncome",
            "MonthlyRate",
            "DailyRate",
            "HourlyRate",
            "StockOptionLevel",
            "JobLevel",
            "TrainingTimesLastYear",
            "PercentSalaryHike",
        ],
        "immutable": [
            "Age",
            "MaritalStatus",
            "Education",
            "EducationField",
            "Department",
            "JobRole",
            "BusinessTravel",
            "DistanceFromHome",
            "NumCompaniesWorked",
            "TotalWorkingYears",
            "YearsAtCompany",
            "YearsInCurrentRole",
            "YearsSinceLastPromotion",
            "YearsWithCurrManager",
            "Over18",
            "EmployeeNumber",
            "EmployeeCount",
            "StandardHours",
        ],
    },
    "ibm_hr_perfrating": {
        "modifiable": [
            "JobInvolvement",
            "JobSatisfaction",
            "EnvironmentSatisfaction",
            "RelationshipSatisfaction",
            "WorkLifeBalance",
            "OverTime",
            "MonthlyIncome",
            "MonthlyRate",
            "DailyRate",
            "HourlyRate",
            "StockOptionLevel",
            "JobLevel",
            "TrainingTimesLastYear",
        ],
        "immutable": [
            "Age",
            "MaritalStatus",
            "Education",
            "EducationField",
            "Department",
            "JobRole",
            "BusinessTravel",
            "DistanceFromHome",
            "NumCompaniesWorked",
            "TotalWorkingYears",
            "YearsAtCompany",
            "YearsInCurrentRole",
            "YearsSinceLastPromotion",
            "YearsWithCurrManager",
            "Over18",
            "EmployeeNumber",
            "EmployeeCount",
            "StandardHours",
            "Attrition",
        ],
    },
    "oulad": {
        "modifiable": [
            "studied_credits",
            "num_of_prev_attempts",
            "code_module",
            "code_presentation",
        ],
        "immutable": [
            "gender",
            "age_band",
            "disability",
            "region",
            "imd_band",
            "highest_education",
        ],
    },
    #  ADDENDUM — ACS-Income (D2). Sensitive: SEX (or RAC1P).
    # See  ADDENDUM for per-feature rationale.
    # ACSIncome has 10 columns: AGEP, COW, SCHL, MAR, OCCP, POBP, RELP,
    # WKHP, SEX, RAC1P. Other ACS feature codes (ESR, MIG, ANC, DREM,
    # FER, NATIVITY, DEAR, DEYE, CIT, DECADE) live in other ACS tasks
    # (e.g. ACSEmployment) and are listed in the addendum as a future-
    # proofing reference; they are not present here so are absent
    # from this dict.
    "acs_income": {
        "modifiable": [
            "SCHL",  # educational attainment — modifiable through study
            "MAR",   # marital status — life choice, modifiable
            "COW",   # class of worker — job-class change is feasible
            "WKHP",  # usual hours per week — modifiable on the job
            "OCCP",  # occupation code — career change
            "POBP",  # place of birth (BORDERLINE — see addendum)
        ],
        "immutable": [
            "SEX",   # sensitive
            "RAC1P",  # sensitive
            "AGEP",  # age
            "RELP",  # relationship — household-structural, slow-moving
        ],
    },
    #  ADDENDUM — Dutch Census (D5). Sensitive: sex.
    # See  ADDENDUM for per-feature rationale.
    #  ADDENDUM 2 — D4 Ricci firefighter promotion.
    "ricci": {
        "modifiable": [
            "Oral",      # oral-exam score — improvable through preparation
            "Written",   # written-exam score — improvable through study
            "Combine",   # 0.6*Written + 0.4*Oral, derived; modifiable
        ],
        "immutable": [
            "Race",      # sensitive demographic
            "Position",  # current rank (Captain/Lieutenant) — pre-existing
                         # structural for the promotion-decision horizon
        ],
    },
}

# ---------------------------------------------------------------------------
# Null-baseline predictors.
# ---------------------------------------------------------------------------

class ConstantPredictor:
    """Always predicts the majority training class (procedural CEILING).

    Per  /  — reference CEILING for procedural fairness:
      * process_consistency = 1.0 (constant under any perturbation),
      * voice_representation undefined (SHAP on a constant function is
        zero everywhere; we emit ``NaN`` for voice and ``NaN`` for
        voice_enrichment to flag the degeneracy),
      * model_flippability validity = 0 (no CF can change a constant
        prediction),
      * explanation_actionability validity = 0 (subset of above).
    """

    def __init__(self) -> None:
        self.majority_class_: int = 0
        self.classes_: np.ndarray = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConstantPredictor":
        y_arr = np.asarray(y).ravel()
        classes, counts = np.unique(y_arr, return_counts=True)
        self.classes_ = classes.astype(int)
        self.majority_class_ = int(classes[counts.argmax()])
        return self

    def predict(self, X) -> np.ndarray:
        n = len(X)
        return np.full(n, self.majority_class_, dtype=int)

    def predict_proba(self, X) -> np.ndarray:
        n = len(X)
        n_cls = max(2, int(self.classes_.max()) + 1)
        proba = np.zeros((n, n_cls), dtype=float)
        proba[:, self.majority_class_] = 1.0
        return proba

class ShuffledPredictor:
    """RandomForest with predictions permuted at predict time (procedural FLOOR).

    Per  /  — reference FLOOR. The underlying RF is trained
    normally; at predict time we shuffle the predictions deterministically
    using a seed-derived permutation. This decouples the prediction from
    the input row, so:
      * process_consistency drops to "near random" levels,
      * voice ≈ 0 (SHAP on uncorrelated output),
      * model_flippability validity is high (any flip is likely to land
        in a different shuffled bucket).

    The shuffle is deterministic given the seed, so re-running with the
    same seed reproduces byte-identical predictions.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self._rf = RandomForestClassifier(
            n_estimators=100, random_state=self.seed, n_jobs=1
        )
        self.classes_: np.ndarray = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ShuffledPredictor":
        self._rf.fit(X, y)
        self.classes_ = np.asarray(self._rf.classes_)
        return self

    def _shuffle_preds(self, preds: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(preds))
        return np.asarray(preds)[perm]

    def predict(self, X) -> np.ndarray:
        return self._shuffle_preds(self._rf.predict(X))

    def predict_proba(self, X) -> np.ndarray:
        proba = self._rf.predict_proba(X)
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(proba))
        return proba[perm]

# ---------------------------------------------------------------------------
# Dataset and model registries.
# ---------------------------------------------------------------------------

def _load_ibm_hr_attrition() -> dict:
    """Load IBM HR with target=Attrition (the honest target per )."""
    return load_ibm_hr(target="Attrition")

def _load_ibm_hr_perfrating() -> dict:
    """Load IBM HR with target=PerformanceRating (leaky per )."""
    bundle = load_ibm_hr(target="PerformanceRating")
    if "PercentSalaryHike" in bundle["X_train"].columns:
        bundle["X_train"] = bundle["X_train"].drop(columns=["PercentSalaryHike"])
        bundle["X_test"] = bundle["X_test"].drop(columns=["PercentSalaryHike"])
        bundle["feature_names"] = list(bundle["X_train"].columns)
    return bundle

def _load_acs_income() -> dict:
    """Load Folktables ACS-Income (D2).  / .

    Per the implementer instructions, we use a defensible larger sample
    (20,000 rows total before split) when the underlying frame is
    larger, to keep the audit tractable; the full frame can have ~150k+
    rows for CA-2018. The 20,000 cap is taken AFTER the loader's 80/20
    split via random subsampling on (X, y, A) jointly with seed=0 to
    preserve  / .
    """
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

def _load_ricci() -> dict:
    """D4 Ricci firefighter-promotion loader thunk."""
    from procedural_fair_hr.data_loaders import load_ricci

    return load_ricci()


# Dataset registry: (key, loader, sensitive-column, target-name, target-form,
# notes, shap_kernel_for_kernel_models). Order is the canonical row order
# in the output CSV.
DATASETS: list[dict[str, Any]] = [
    {
        "key": "ibm_hr_attrition",
        "loader": _load_ibm_hr_attrition,
        "sensitive_col": "Gender",
        "target_name": "Attrition",
        "target_form": "binary",
        "notes": "",
    },
    {
        "key": "ibm_hr_perfrating",
        "loader": _load_ibm_hr_perfrating,
        "sensitive_col": "Gender",
        "target_name": "PerformanceRating",
        "target_form": "binary",
        "notes": "leaky",
    },
    {
        "key": "oulad",
        "loader": load_oulad,
        "sensitive_col": "gender",
        "target_name": "final_result",
        "target_form": "multiclass",
        "notes": "",
    },
    {
        "key": "acs_income",
        "loader": _load_acs_income,
        "sensitive_col": "RAC1P",
        "target_name": "high_income",
        "target_form": "binary",
        "notes": "",
    },
    #  /  — D4 Ricci firefighter promotion (genuine HR per
    # Le Quy 2022 §3.3.2 + Supreme Court 2009 *Ricci v. DeStefano*).
    {
        "key": "ricci",
        "loader": _load_ricci,
        "sensitive_col": "Race",
        "target_name": "Class",
        "target_form": "binary",
        "notes": "",
    },
]

# Model registry. Each entry: (short_key, builder(seed), display_name,
# shap_explainer). ``shap_explainer`` may be ``None`` when SHAP isn't
# applicable (constant / shuffled baselines emit NaN voice).
def _build_model(key: str, seed: int) -> tuple[Any, str, str | None]:
    """Build a single model instance for the given short-key + seed.

    Returns (estimator, display_name, shap_explainer_kind).
    """
    if key == "RF":
        return (
            RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1),
            "RandomForestClassifier",
            "tree",
        )
    if key == "LR":
        return (
            LogisticRegression(max_iter=1000, random_state=seed),
            "LogisticRegression",
            "linear",
        )
    if key == "MLP":
        return (
            MLPClassifier(
                hidden_layer_sizes=(64,), max_iter=500, random_state=seed
            ),
            "MLPClassifier",
            "kernel",
        )
    if key == "XGB":
        # Local import — xgboost is optional; fail loudly with a hint if
        # the env doesn't have it.
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "xgboost is required for --models XGB; install with "
                "`pip install xgboost` in the thesis env."
            ) from exc

        # XGBoost requires class labels in [0, n_classes-1]; IBM HR
        # PerformanceRating ships labels {3, 4} which crashes the fit.
        # : the wrapper is now a SUBCLASS of XGBClassifier (rather
        # than composition + ``__getattr__`` delegation) so that
        # ``isinstance(model, XGBClassifier)`` returns True. shap's
        # TreeExplainer dispatches via isinstance checks BEFORE delegating,
        # so the previous composition-based wrapper fell through to a
        # NaN-emitting branch. The subclass form keeps inherited
        # ``predict_proba`` / ``feature_importances_`` / ``get_booster``
        # working natively on the encoded-label space (which matches what
        # SHAP introspects).
        from sklearn.preprocessing import LabelEncoder

        class _XGBWithLabelEncoder(XGBClassifier):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._le = LabelEncoder()
                # ``XGBClassifier.classes_`` is a property; we override it
                # below so user-facing code sees the ORIGINAL label set
                # (e.g., {3, 4}) rather than the encoded {0, 1}.
                self._original_classes: np.ndarray | None = None

            def fit(self, X, y, **fit_kwargs):
                y_enc = self._le.fit_transform(y)
                super().fit(X, y_enc, **fit_kwargs)
                self._original_classes = self._le.classes_.copy()
                return self

            @property
            def classes_(self):  # type: ignore[override]
                if self._original_classes is not None:
                    return self._original_classes
                return super().classes_

            def predict(self, X):
                preds_enc = super().predict(X)
                return self._le.inverse_transform(preds_enc)

            # predict_proba inherits from XGBClassifier — column order is
            # the encoded class order, which is identical (numerically
            # sorted) to ``self._original_classes``.

        return (
            _XGBWithLabelEncoder(
                n_estimators=100,
                random_state=seed,
                eval_metric="logloss",
                n_jobs=1,
            ),
            "XGBClassifier",
            "tree",
        )
    if key == "GB":
        return (
            GradientBoostingClassifier(n_estimators=100, random_state=seed),
            "GradientBoostingClassifier",
            "tree",
        )
    if key == "KNN":
        return (
            KNeighborsClassifier(n_neighbors=5, n_jobs=1),
            "KNeighborsClassifier",
            "kernel",
        )
    if key == "Constant":
        return (ConstantPredictor(), "ConstantPredictor", None)
    if key == "Shuffled":
        return (ShuffledPredictor(seed=seed), "ShuffledPredictor", None)
    raise ValueError(f"Unknown model key: {key!r}")

DEFAULT_MODEL_KEYS = ("RF", "LR", "MLP")
ALL_MODEL_KEYS = (
    "RF",
    "LR",
    "MLP",
    "XGB",
    "GB",
    "KNN",
    "Constant",
    "Shuffled",
)

def _build_models(
    keys: tuple[str, ...] = ("RF", "LR", "MLP"),
    seed: int = 0,
) -> list[tuple[str, Any, str]]:
    """Backward-compat: return the (name, estimator, shap_kind) list.

    Used by ``scripts/run_phase4_sensitivity.py`` and any external caller
    that pre-dates the  model-key API. Default keys are the
    legacy three-model baseline (RF / LR / MLP);  /  extends
    callers to optionally pass the post 6-model roster
    ``("RF", "LR", "MLP", "XGB", "GB", "KNN")`` for a granular
    Spearman-ρ on the partition-sensitivity check.
    """
    out: list[tuple[str, Any, str]] = []
    for k in keys:
        est, name, shap_kind = _build_model(k, seed=seed)
        # ``shap_kind`` is non-None for all trainable model keys.
        assert shap_kind is not None, (
            f"_build_models received non-trainable key {k!r}; "
            "Constant / Shuffled baselines are not supported here."
        )
        out.append((name, est, shap_kind))
    return out

# Output column order — fixed for byte-identical CSV.
CSV_COLUMNS = [
    "dataset",
    "target",
    "model",
    "metric",
    "value",
    "seed",
    "mean",
    "std",
    "ci_lo",
    "ci_hi",
    "noise_std",
    "target_form",
    "n_classes",
    "sample_n",
    "random_state",
    "notes",
]

# Sentinel for the noise_std column on metric rows where noise_std is
# inapplicable (everything except process_consistency).
NOISE_NA = -1.0
SEED_AGGREGATE = -1

# ---------------------------------------------------------------------------
# Pipeline / preprocessing helpers.
# ---------------------------------------------------------------------------

def _preprocess_xy(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode categoricals to numeric so the procedural metrics see purely
    numeric features. See the original  docstring for rationale.
    """
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

    if cat_cols:
        cat_pipe.fit(X_train[cat_cols])
    if num_cols:
        num_pipe.fit(X_train[num_cols])

    def _apply(X: pd.DataFrame) -> pd.DataFrame:
        out_parts: list[pd.DataFrame] = []
        if cat_cols:
            cat_arr = cat_pipe.transform(X[cat_cols])
            out_parts.append(
                pd.DataFrame(cat_arr, columns=cat_cols, index=X.index)
            )
        if num_cols:
            num_arr = num_pipe.transform(X[num_cols])
            out_parts.append(
                pd.DataFrame(num_arr, columns=num_cols, index=X.index)
            )
        out = pd.concat(out_parts, axis=1)
        return out[X.columns]

    return _apply(X_train), _apply(X_test)

# ---------------------------------------------------------------------------
# Bootstrap CI helper.
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    values: list[float], n_resamples: int = 10_000, seed: int = 0
) -> tuple[float, float, float, float]:
    """Return (mean, std, ci_lo, ci_hi) using percentile bootstrap on the mean.

    A NaN in ``values`` propagates to NaN outputs so the schema stays
    populated even when a metric was not computed (e.g., voice on
    ConstantPredictor).
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.any(np.isnan(arr)):
        return (float("nan"),) * 4
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    if arr.size == 1:
        return mean, std, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    boot = arr[idx].mean(axis=1)
    ci_lo = float(np.percentile(boot, 2.5))
    ci_hi = float(np.percentile(boot, 97.5))
    return mean, std, ci_lo, ci_hi

# ---------------------------------------------------------------------------
# Per-(dataset, model, seed) audit cell — emits per-seed metric rows.
# ---------------------------------------------------------------------------

def _row_template(
    *,
    dataset_key: str,
    target_name: str,
    model_name: str,
    metric: str,
    value: float,
    seed: int,
    noise_std: float,
    target_form: str,
    n_classes: int,
    sample_n: int,
    notes: str,
) -> dict:
    """Build a per-seed schema-conformant row (mean/std/ci NaN)."""
    return {
        "dataset": dataset_key,
        "target": target_name,
        "model": model_name,
        "metric": metric,
        "value": float(value),
        "seed": int(seed),
        "mean": float("nan"),
        "std": float("nan"),
        "ci_lo": float("nan"),
        "ci_hi": float("nan"),
        "noise_std": float(noise_std),
        "target_form": target_form,
        "n_classes": int(n_classes),
        "sample_n": int(sample_n),
        "random_state": int(seed),
        "notes": notes,
    }

def _audit_dataset_cell(
    *,
    dataset_key: str,
    loader: Callable[[], dict],
    sensitive_col: str,
    target_name: str,
    target_form: str,
    notes: str,
    model_keys: list[str],
    seeds: list[int],
    noise_stds: list[float],
    sample_n: int,
    sample_n_transparency: int,
    max_features_to_flip: int,
    log: Callable[[str], None],
) -> list[dict]:
    """Compute all metric rows (per-seed + aggregated) for one dataset.

    For each model_key × seed produce per-seed rows; then aggregate
    across seeds for each (model, metric, noise_std) tuple into a
    seed=-1 sentinel row.
    """
    bundle = loader()
    X_train: pd.DataFrame = bundle["X_train"]
    X_test: pd.DataFrame = bundle["X_test"]
    y_train: pd.Series = bundle["y_train"]
    A_test: pd.DataFrame = bundle["A_test"]
    n_classes = int(bundle["n_classes"])

    log(
        f"  [{dataset_key}] preprocess  X_train={X_train.shape} "
        f"X_test={X_test.shape} n_classes={n_classes}"
    )
    X_train_enc, X_test_enc = _preprocess_xy(X_train, X_test)

    sample_for_call = min(sample_n, len(X_test_enc))
    sens_for_call = A_test[sensitive_col].reset_index(drop=True)
    transparency_n = min(sample_n_transparency, len(X_test_enc))

    partition = ADR019_PARTITIONS[dataset_key]
    cols_set = set(X_test_enc.columns)
    part_set = set(partition["modifiable"]) | set(partition["immutable"])
    missing = sorted(cols_set - part_set)
    if missing:
        raise RuntimeError(
            f"modifiable/immutable partition for dataset_key={dataset_key!r} is missing "
            f"these features in X_test: {missing}. Add them to the partition "
            "(with a superseding ADR) or remove them from the loader output."
        )
    trimmed_partition = {
        "modifiable": [c for c in partition["modifiable"] if c in cols_set],
        "immutable": [c for c in partition["immutable"] if c in cols_set],
    }

    rows: list[dict] = []

    # Per-seed values keyed by (model_name, metric, noise_std) → list of
    # per-seed values for aggregation.
    per_seed_values: dict[tuple[str, str, float], list[float]] = {}

    for model_key in model_keys:
        for seed in seeds:
            estimator, model_name, shap_explainer = _build_model(model_key, seed)

            log(
                f"  [{dataset_key} / {model_name} / seed={seed}] fit"
            )
            t_fit = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", UserWarning)
                warnings.simplefilter("ignore", FutureWarning)
                estimator.fit(X_train_enc.values, y_train.values)
            log(
                f"  [{dataset_key} / {model_name} / seed={seed}] fit "
                f"done in {time.time() - t_fit:.1f}s"
            )

            # ----- 1. process_consistency over noise_std grid ---------------
            for nstd in noise_stds:
                t0 = time.time()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    warnings.simplefilter("ignore", UserWarning)
                    warnings.simplefilter("ignore", FutureWarning)
                    pc_overall, _ = process_consistency(
                        estimator,
                        X_test_enc,
                        perturbations_per_row=10,
                        noise_std=nstd,
                        sample_n=sample_for_call,
                        stratify_on=sens_for_call,
                        random_state=seed,
                    )
                log(
                    f"    pc[noise={nstd}]={pc_overall:.4f} "
                    f"in {time.time() - t0:.1f}s"
                )
                rows.append(
                    _row_template(
                        dataset_key=dataset_key,
                        target_name=target_name,
                        model_name=model_name,
                        metric="process_consistency",
                        value=pc_overall,
                        seed=seed,
                        noise_std=nstd,
                        target_form=target_form,
                        n_classes=n_classes,
                        sample_n=sample_for_call,
                        notes=notes,
                    )
                )
                per_seed_values.setdefault(
                    (model_name, "process_consistency", nstd), []
                ).append(float(pc_overall))

            # ----- 2. voice_representation + voice_enrichment ---------------
            if shap_explainer is None:
                # Baselines: emit NaN voice per  spec.
                for metric in ("voice_representation", "voice_enrichment"):
                    rows.append(
                        _row_template(
                            dataset_key=dataset_key,
                            target_name=target_name,
                            model_name=model_name,
                            metric=metric,
                            value=float("nan"),
                            seed=seed,
                            noise_std=NOISE_NA,
                            target_form=target_form,
                            n_classes=n_classes,
                            sample_n=sample_for_call,
                            notes=notes,
                        )
                    )
                    per_seed_values.setdefault(
                        (model_name, metric, NOISE_NA), []
                    ).append(float("nan"))
            else:
                t0 = time.time()
                try:
                    with warnings.catch_warnings(), _xgboost_shap_compat_patch():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        warnings.simplefilter("ignore", UserWarning)
                        warnings.simplefilter("ignore", FutureWarning)
                        # : ``_xgboost_shap_compat_patch`` is the
                        # context manager that fixes shap-0.49.x's
                        # base_score parser for xgboost >= 3.0 boosters.
                        # No-op for non-XGB models (the patched decoder
                        # is only invoked inside ``shap.TreeExplainer``
                        # for xgboost models).
                        voice_overall, voice_enrich, _ = voice_representation(
                            estimator,
                            X_test_enc,
                            feature_partition=trimmed_partition,
                            shap_explainer=shap_explainer,
                            sample_n=sample_for_call,
                            random_state=seed,
                        )
                except Exception as exc:  # pragma: no cover - defensive
                    log(f"    voice failed: {exc!r}; emitting NaN")
                    voice_overall = float("nan")
                    voice_enrich = float("nan")
                log(
                    f"    voice={voice_overall:.4f} "
                    f"enrich={voice_enrich:.4f} in {time.time() - t0:.1f}s"
                )
                rows.append(
                    _row_template(
                        dataset_key=dataset_key,
                        target_name=target_name,
                        model_name=model_name,
                        metric="voice_representation",
                        value=voice_overall,
                        seed=seed,
                        noise_std=NOISE_NA,
                        target_form=target_form,
                        n_classes=n_classes,
                        sample_n=sample_for_call,
                        notes=notes,
                    )
                )
                rows.append(
                    _row_template(
                        dataset_key=dataset_key,
                        target_name=target_name,
                        model_name=model_name,
                        metric="voice_enrichment",
                        value=voice_enrich,
                        seed=seed,
                        noise_std=NOISE_NA,
                        target_form=target_form,
                        n_classes=n_classes,
                        sample_n=sample_for_call,
                        notes=notes,
                    )
                )
                per_seed_values.setdefault(
                    (model_name, "voice_representation", NOISE_NA), []
                ).append(float(voice_overall))
                per_seed_values.setdefault(
                    (model_name, "voice_enrichment", NOISE_NA), []
                ).append(float(voice_enrich))

            # ----- 3. transparency_metrics (backward-compat alias) -----------
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", UserWarning)
                warnings.simplefilter("ignore", FutureWarning)
                trans = transparency_metrics(
                    estimator,
                    X_test_enc,
                    sensitive=sens_for_call,
                    max_features_to_flip=max_features_to_flip,
                    sample_n=transparency_n,
                    random_state=seed,
                )
            log(
                f"    transparency sparsity={trans['sparsity']:.4f} "
                f"validity={trans['validity']:.4f} in {time.time() - t0:.1f}s"
            )
            for old_metric, key in (
                ("transparency_sparsity", "sparsity"),
                ("transparency_validity", "validity"),
            ):
                rows.append(
                    _row_template(
                        dataset_key=dataset_key,
                        target_name=target_name,
                        model_name=model_name,
                        metric=old_metric,
                        value=trans[key],
                        seed=seed,
                        noise_std=NOISE_NA,
                        target_form=target_form,
                        n_classes=n_classes,
                        sample_n=transparency_n,
                        notes=notes,
                    )
                )
                per_seed_values.setdefault(
                    (model_name, old_metric, NOISE_NA), []
                ).append(float(trans[key]))

            # ----- 4. model_flippability (architectural split) --------------
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", UserWarning)
                warnings.simplefilter("ignore", FutureWarning)
                flip = model_flippability(
                    estimator,
                    X_test_enc,
                    sensitive=sens_for_call,
                    max_features_to_flip=max_features_to_flip,
                    sample_n=transparency_n,
                    random_state=seed,
                )
            log(
                f"    flippability sparsity={flip['sparsity']:.4f} "
                f"validity={flip['validity']:.4f} in {time.time() - t0:.1f}s"
            )
            for new_metric, key in (
                ("model_flippability_sparsity", "sparsity"),
                ("model_flippability_validity", "validity"),
            ):
                rows.append(
                    _row_template(
                        dataset_key=dataset_key,
                        target_name=target_name,
                        model_name=model_name,
                        metric=new_metric,
                        value=flip[key],
                        seed=seed,
                        noise_std=NOISE_NA,
                        target_form=target_form,
                        n_classes=n_classes,
                        sample_n=transparency_n,
                        notes=notes,
                    )
                )
                per_seed_values.setdefault(
                    (model_name, new_metric, NOISE_NA), []
                ).append(float(flip[key]))

            # ----- 5. explanation_actionability (procedural split) -----------
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", UserWarning)
                warnings.simplefilter("ignore", FutureWarning)
                act = explanation_actionability(
                    estimator,
                    X_test_enc,
                    feature_partition=trimmed_partition,
                    max_features_to_flip=max_features_to_flip,
                    sample_n=transparency_n,
                    random_state=seed,
                )
            log(
                f"    actionability validity={act['actionable_validity']:.4f} "
                f"sparsity={act['actionable_sparsity']:.4f} "
                f"in {time.time() - t0:.1f}s"
            )
            for new_metric, key in (
                ("actionable_validity", "actionable_validity"),
                ("actionable_sparsity", "actionable_sparsity"),
            ):
                rows.append(
                    _row_template(
                        dataset_key=dataset_key,
                        target_name=target_name,
                        model_name=model_name,
                        metric=new_metric,
                        value=act[key],
                        seed=seed,
                        noise_std=NOISE_NA,
                        target_form=target_form,
                        n_classes=n_classes,
                        sample_n=transparency_n,
                        notes=notes,
                    )
                )
                per_seed_values.setdefault(
                    (model_name, new_metric, NOISE_NA), []
                ).append(float(act[key]))

    # ----- Aggregate per (model_name, metric, noise_std) across seeds -------
    for (model_name, metric, nstd), values in per_seed_values.items():
        mean, std, ci_lo, ci_hi = _bootstrap_ci(values, n_resamples=10_000, seed=0)
        # Choose sample_n / target_form / etc. from the first matching per-seed
        # row (they must be identical across seeds for a given key).
        # This avoids duplicating the bookkeeping above.
        ref = next(
            r for r in rows
            if r["model"] == model_name
            and r["metric"] == metric
            and r["noise_std"] == nstd
            and r["seed"] != SEED_AGGREGATE
        )
        rows.append(
            {
                "dataset": dataset_key,
                "target": target_name,
                "model": model_name,
                "metric": metric,
                "value": float(mean),
                "seed": SEED_AGGREGATE,
                "mean": float(mean),
                "std": float(std),
                "ci_lo": float(ci_lo),
                "ci_hi": float(ci_hi),
                "noise_std": float(nstd),
                "target_form": ref["target_form"],
                "n_classes": ref["n_classes"],
                "sample_n": ref["sample_n"],
                "random_state": SEED_AGGREGATE,
                "notes": notes,
            }
        )

    return rows

# ---------------------------------------------------------------------------
# Top-level driver.
# ---------------------------------------------------------------------------

def _sort_rows_for_csv(rows: list[dict]) -> list[dict]:
    """Sort rows deterministically so re-runs produce byte-identical CSV."""
    return sorted(
        rows,
        key=lambda r: (
            r["dataset"],
            r["target"],
            r["model"],
            r["metric"],
            r["noise_std"],
            r["seed"],
        ),
    )

def _cache_key(
    dataset_key: str,
    model_keys: list[str],
    seeds: list[int],
    noise_stds: list[float],
    sample_n: int,
    sample_n_transparency: int,
    max_features_to_flip: int,
) -> str:
    """Stable cache filename suffix encoding the call shape."""
    seeds_s = "-".join(str(s) for s in seeds)
    noise_s = "-".join(f"{n:g}" for n in noise_stds)
    models_s = "-".join(model_keys)
    return (
        f"{dataset_key}__M{models_s}__S{seeds_s}__N{noise_s}__"
        f"sn{sample_n}_st{sample_n_transparency}_mf{max_features_to_flip}"
    )

def run(
    *,
    out_dir: pathlib.Path,
    sample_n: int,
    sample_n_transparency: int,
    max_features_to_flip: int,
    seeds: list[int],
    noise_stds: list[float],
    model_keys: list[str],
    dataset_keys: list[str] | None,
    log: Callable[[str], None],
) -> int:
    """Drive the full Phase-4 procedural audit. Returns 0 on success."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    selected = (
        [d for d in DATASETS if d["key"] in dataset_keys]
        if dataset_keys
        else DATASETS
    )

    all_rows: list[dict] = []
    started_at = time.perf_counter()

    for ds in selected:
        cache_name = _cache_key(
            ds["key"],
            model_keys,
            seeds,
            noise_stds,
            sample_n,
            sample_n_transparency,
            max_features_to_flip,
        )
        cache_path = cache_dir / f"{cache_name}.parquet"
        if cache_path.exists():
            log(f"[{ds['key']}] CACHED  {cache_path.name}")
            cached = pd.read_parquet(cache_path)
            all_rows.extend(cached.to_dict(orient="records"))
            continue

        t_cell = time.perf_counter()
        log(
            f"[{ds['key']}] START  models={model_keys} seeds={seeds} "
            f"noise={noise_stds} sample_n={sample_n} "
            f"sample_n_transparency={sample_n_transparency} "
            f"max_features_to_flip={max_features_to_flip}"
        )
        rows = _audit_dataset_cell(
            dataset_key=ds["key"],
            loader=ds["loader"],
            sensitive_col=ds["sensitive_col"],
            target_name=ds["target_name"],
            target_form=ds["target_form"],
            notes=ds["notes"],
            model_keys=model_keys,
            seeds=seeds,
            noise_stds=noise_stds,
            sample_n=sample_n,
            sample_n_transparency=sample_n_transparency,
            max_features_to_flip=max_features_to_flip,
            log=log,
        )
        elapsed = time.perf_counter() - t_cell
        log(f"[{ds['key']}] DONE  {len(rows)} rows in {elapsed:.1f}s")

        pd.DataFrame(rows, columns=CSV_COLUMNS).to_parquet(
            cache_path, index=False
        )
        all_rows.extend(rows)

    all_rows_sorted = _sort_rows_for_csv(all_rows)
    df = pd.DataFrame(all_rows_sorted, columns=CSV_COLUMNS)

    csv_path = out_dir / "procedural.csv"
    parquet_path = out_dir / "procedural.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    total_elapsed = time.perf_counter() - started_at
    log(
        f"AUDIT END  rows={len(df)} csv={csv_path} "
        f"total_elapsed={total_elapsed:.1f}s"
    )

    print(f"\nWrote {len(df)} rows -> {csv_path}", flush=True)
    print(f"Wrote {len(df)} rows -> {parquet_path}", flush=True)
    return 0

def _parse_csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def _parse_csv_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def _parse_csv_strs(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_phase4_procedural",
        description=(
            "Procedural-fairness audit. "
            "Output: results/phase4/procedural.csv."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(_PROJECT_ROOT / "results" / "phase4"),
        help="Output directory.",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=500,
        help="Stratified sample size for process_consistency / voice.",
    )
    parser.add_argument(
        "--sample-n-transparency",
        type=int,
        default=None,
        help="Stratified sample size for transparency / flippability / "
        "actionability (default: same as --sample-n).",
    )
    parser.add_argument(
        "--max-features-to-flip",
        type=int,
        default=1,
        help="max_features_to_flip for the greedy CF search (default 1).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0",
        help="Comma-separated seeds (default '0' for backward-compat smoke).",
    )
    parser.add_argument(
        "--noise-stds",
        type=str,
        default="0.1",
        help="Comma-separated process_consistency noise σ values "
        "(default '0.1' for backward-compat smoke).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODEL_KEYS),
        help=(
            "Comma-separated model short-keys; choices: "
            + ",".join(ALL_MODEL_KEYS)
            + ". Default: RF,LR,MLP."
        ),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help=(
            "Comma-separated dataset keys to run "
            "(default: all of "
            + ",".join(d["key"] for d in DATASETS)
            + ")."
        ),
    )
    return parser.parse_args(argv)

def _stdout_logger(msg: str) -> None:
    print(msg, flush=True)

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = pathlib.Path(args.out_dir).resolve()

    dataset_keys = (
        _parse_csv_strs(args.datasets) if args.datasets else None
    )
    if dataset_keys:
        unknown = [
            d for d in dataset_keys if d not in {ds["key"] for ds in DATASETS}
        ]
        if unknown:
            print(f"Unknown dataset key(s): {unknown!r}", file=sys.stderr)
            return 2

    model_keys = _parse_csv_strs(args.models)
    unknown_models = [m for m in model_keys if m not in ALL_MODEL_KEYS]
    if unknown_models:
        print(
            f"Unknown model key(s): {unknown_models!r}; "
            f"valid: {ALL_MODEL_KEYS!r}",
            file=sys.stderr,
        )
        return 2

    seeds = _parse_csv_ints(args.seeds)
    noise_stds = _parse_csv_floats(args.noise_stds)
    if not seeds or not noise_stds or not model_keys:
        print("--seeds, --noise-stds, --models must be non-empty", file=sys.stderr)
        return 2

    sample_n_transparency = (
        args.sample_n_transparency
        if args.sample_n_transparency is not None
        else args.sample_n
    )

    return run(
        out_dir=out_dir,
        sample_n=args.sample_n,
        sample_n_transparency=sample_n_transparency,
        max_features_to_flip=args.max_features_to_flip,
        seeds=seeds,
        noise_stds=noise_stds,
        model_keys=model_keys,
        dataset_keys=dataset_keys,
        log=_stdout_logger,
    )

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

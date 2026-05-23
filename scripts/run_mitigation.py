"""Phase-5 mitigation-matrix audit runner.

Runs each (dataset, base_model, mitigation_method, lambda, seed) cell in
parallel via ``multiprocessing.Pool(maxtasksperchild=20)`` (Tier-5b
incident 2026-05-01) and consolidates per-cell parquet outputs into a
single CSV (``results/phase5/audit.csv``). Workers are recycled every
20 cells so PyTorch grad buffers, SHAP caches, and AIF360 datasets
allocated inside C extensions are released definitively (cf. macOS
Jetsam OOM kill at 21:03 NZ on 2026-05-01).

Architecture (mirrors ``scripts/run_phase4_procedural.py`` but with
process-pool parallelism from day 1, per 's hardware-utilization
spec):

  * Each worker pins single-thread BLAS in its initialiser
    (``MKL_NUM_THREADS = OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = 1``)
    so per-cell determinism is preserved.
  * Outer parallelism comes from ``max_workers=9`` (10 cores - 1 OS).
  * Each cell writes one parquet under ``results/phase5/cache/<cell_key>.parquet``;
    the consolidator stitches them into the headline CSV after every
    submitted cell finishes (or fails).
  * Lexicographic sort on ``(dataset, base_model, method, lambda, seed,
    metric)`` makes the consolidated CSV byte-identical across runs
    regardless of worker run order.

Schema (````):

    [dataset, target, base_model, method, method_kind, lambda_, seed,
     metric, value, metric_kind, sample_n, random_state, notes]

CLI summary::

    python scripts/run_phase5_audit.py \\
        --datasets ricci \\
        --base-models RF \\
        --methods identity_preprocessing \\
        --lambdas 0 \\
        --seeds 0 \\
        --max-workers 1 \\
        --out-dir results/phase5

References
----------

* ``scripts/benchmark_parallelism.py`` — _worker_init pattern.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Any, Iterable

# --- Determinism prelude (env vars before numpy import) ---
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------

# : this is the canonical column order for the Phase-5 audit CSV.
CSV_COLUMNS: list[str] = [
    "dataset",
    "target",
    "base_model",
    "method",
    "method_kind",
    "lambda_",
    "seed",
    "metric",
    "value",
    "metric_kind",
    "sample_n",
    "random_state",
    "notes",
]

# ---------------------------------------------------------------------
# Dataset registry (mirrors run_phase4_procedural.py§DATASETS)
# ---------------------------------------------------------------------

def _load_ibm_hr_attrition() -> dict:
    from procedural_fair_hr.data_loaders import load_ibm_hr

    return load_ibm_hr(target="Attrition")

def _load_ibm_hr_perfrating() -> dict:
    from procedural_fair_hr.data_loaders import load_ibm_hr

    bundle = load_ibm_hr(target="PerformanceRating")
    return bundle

def _load_acs_income() -> dict:
    from procedural_fair_hr.data_loaders import load_acs

    return load_acs(state="CA", year=2018, task="income")

def _load_oulad() -> dict:
    from procedural_fair_hr.data_loaders import load_oulad

    return load_oulad()

def _load_ricci() -> dict:
    from procedural_fair_hr.data_loaders import load_ricci

    return load_ricci()

DATASETS: dict[str, dict[str, Any]] = {
    "ibm_hr_attrition": {
        "loader": _load_ibm_hr_attrition,
        "sensitive_col": "Gender",
        "target": "Attrition",
        "notes": "",
    },
    "ibm_hr_perfrating": {
        "loader": _load_ibm_hr_perfrating,
        "sensitive_col": "Gender",
        "target": "PerformanceRating",
        "notes": "leaky",
    },
    "acs_income": {
        "loader": _load_acs_income,
        "sensitive_col": "RAC1P",
        "target": "high_income",
        "notes": "",
    },
    "oulad": {
        "loader": _load_oulad,
        "sensitive_col": "gender",
        "target": "final_result",
        "notes": "",
    },
    "ricci": {
        "loader": _load_ricci,
        "sensitive_col": "Race",
        "target": "Class",
        "notes": "",
    },
}

# ---------------------------------------------------------------------
# Base-model factory (subset of run_phase4_procedural.py for the Tier-1
# smoke; full roster lands in Tier 5 audit at ).
# ---------------------------------------------------------------------

def build_base_model(key: str, seed: int):
    """Construct one fresh sklearn-compatible base estimator."""
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier

    if key == "RF":
        return RandomForestClassifier(
            n_estimators=50, random_state=seed, n_jobs=1
        )
    if key == "LR":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if key == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(64,), max_iter=200, random_state=seed
        )
    if key == "GB":
        return GradientBoostingClassifier(n_estimators=50, random_state=seed)
    if key == "KNN":
        return KNeighborsClassifier(n_neighbors=5, n_jobs=1)
    if key == "XGB":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "xgboost is required for --base-models XGB"
            ) from exc
        return XGBClassifier(
            n_estimators=50,
            random_state=seed,
            eval_metric="logloss",
            n_jobs=1,
        )
    raise ValueError(f"Unknown base-model key: {key!r}")

# ---------------------------------------------------------------------
# Cell spec + key
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class CellSpec:
    dataset: str
    base_model: str
    method: str
    lambda_: float
    seed: int
    sensitive_col: str
    target: str
    notes: str = ""

    def key(self) -> str:
        # Stable cell key for parquet caching + log line.
        return (
            f"{self.dataset}__{self.base_model}__{self.method}"
            f"__lam{self.lambda_:g}__s{self.seed}"
        )

def cell_cache_path(out_dir: pathlib.Path, spec: CellSpec) -> pathlib.Path:
    return out_dir / "cache" / f"{spec.key()}.parquet"

def cell_cache_exists(out_dir: pathlib.Path, spec: CellSpec) -> bool:
    return cell_cache_path(out_dir, spec).exists()

# ---------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------

def _worker_init(device: str = "cpu") -> None:
    """Pin single-thread BLAS + seeded hash for cell-level determinism."""
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["PHASE5_DEVICE"] = device

# ---------------------------------------------------------------------
# Pool-based worker recycling (Tier-5b incident 2026-05-01).

# We run cells through ``multiprocessing.Pool`` with
# ``maxtasksperchild=20`` rather than ``ProcessPoolExecutor`` so that
# every worker is killed and respawned every 20 cells. The previous
# sliding-window ``ProcessPoolExecutor`` pattern bounded parent-process
# memory but worker processes themselves accumulated PyTorch grad
# buffers, SHAP background caches, and AIF360 dataset objects across
# hundreds of cells. ``gc.collect()`` cannot reclaim memory allocated
# inside C extensions; only worker exit does. A ~6.5 h oulad run on
# 2026-05-01 21:03 NZ triggered macOS Jetsam OOM kills (WindowServer
# hang + reboot at 21:14) — see DiagnosticReports/JetsamEvent-2026-05-
# 01-210416.ips.

# ``Pool.imap_unordered`` with ``chunksize=1`` is lazy (does not
# materialise all tasks up front) and yields results as they complete,
# so combined with ``maxtasksperchild=20`` the parent + worker memory
# footprint stays bounded indefinitely.
# ---------------------------------------------------------------------

# Module-level so the ``Pool`` initializer can stash the device choice
# from the parent into each freshly-spawned worker without re-passing
# it on every task.
_WORKER_DEVICE: str | None = None

def _pool_initializer(device: str) -> None:
    """``multiprocessing.Pool`` initializer.

    Stores the device choice in a module-level global so the per-task
    ``_pool_run_cell`` does not need to re-receive it (Pool tasks take a
    single argument). Then delegates to the existing ``_worker_init``
    BLAS/hash-pinning routine for  determinism.

    Also silences sklearn ``UserWarning`` chatter ("X does not have valid
    feature names") which fired millions of times in the 2026-05-04
    run and inflated stdout to 113 MB, contributing to the IDE-side OOM
    incident at 12:36 NZST that hard-rebooted the host machine.
    """
    import warnings as _warnings
    _warnings.filterwarnings("ignore", category=UserWarning)
    _warnings.filterwarnings("ignore", category=FutureWarning)
    _warnings.filterwarnings("ignore", category=DeprecationWarning)
    global _WORKER_DEVICE
    _WORKER_DEVICE = device
    _worker_init(device)

#: Per-cell wall-clock budget. A cell exceeding this raises ``CellTimeoutError``
#: inside the worker and is reported as FAIL; the next cell continues. Set as
#: a module-level constant so tests can monkey-patch it. The 2026-05-04
#: incident hung indefinitely on a single ``reject_option`` cell because the
#: previous Pool wrapper had no timeout; this constant prevents recurrence.
CELL_TIMEOUT_S: int = 600  # 10 minutes

class CellTimeoutError(TimeoutError):
    """Raised inside a Pool worker when a cell exceeds CELL_TIMEOUT_S."""

def _pool_run_cell(args_tuple: tuple) -> str:
    """``multiprocessing.Pool`` task wrapper around ``_run_cell``.

    Pool tasks take exactly one argument; we pack ``(spec, out_dir)``
    into a tuple in the parent and unpack here. Keeps ``_run_cell``'s
    own signature unchanged so the synchronous code path and tests
    remain compatible.

    Per-cell exceptions are CAUGHT here (not re-raised) and returned as
    a ``[phase5] FAIL ...`` string so a single bad cell does not kill
    the whole pool — ``Pool.imap_unordered`` propagates worker
    exceptions to the main thread, which would otherwise abort the
    entire 50K-cell run on the first degenerate split.

    A ``signal.SIGALRM``-based per-cell wall-clock timeout fires after
    ``CELL_TIMEOUT_S`` seconds. This catches infinite loops that
    Python-level ``try/except`` cannot interrupt (the 2026-05-04
    ``reject_option`` hang would have been caught here).
    """
    import signal

    spec, out_dir = args_tuple

    def _alarm_handler(_signum, _frame):
        raise CellTimeoutError(
            f"cell exceeded CELL_TIMEOUT_S={CELL_TIMEOUT_S}s wall-clock budget"
        )

    # SIGALRM is process-wide; only the main thread of each worker sees it.
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(CELL_TIMEOUT_S)
    try:
        return _run_cell(spec, out_dir)
    except CellTimeoutError as exc:
        return f"FAIL {spec.key()}: TIMEOUT: {exc}"
    except Exception as exc:  # noqa: BLE001 — defensive boundary
        return f"FAIL {spec.key()}: {type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)  # disarm so a late SIGALRM cannot poison the next cell

def _preprocess_xy(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Numericise categoricals (mirror of 's _preprocess_xy)."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler

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

    out_train = pd.DataFrame(index=X_train.index)
    out_test = pd.DataFrame(index=X_test.index)

    if num_cols:
        num_pipe.fit(X_train[num_cols])
        out_train[num_cols] = num_pipe.transform(X_train[num_cols])
        out_test[num_cols] = num_pipe.transform(X_test[num_cols])
    if cat_cols:
        cat_pipe.fit(X_train[cat_cols])
        out_train[cat_cols] = cat_pipe.transform(X_train[cat_cols])
        out_test[cat_cols] = cat_pipe.transform(X_test[cat_cols])

    # Preserve original column order.
    return out_train[X_train.columns], out_test[X_test.columns]

# ---------------------------------------------------------------------
# Tier-5 metric panel.
# ---------------------------------------------------------------------

# Per cell the panel emits 12 metric rows:

#   Performance (3):  accuracy, balanced_accuracy, f1_macro
#   Statistical (4):  macro_dp, macro_eodds, equal_opportunity,
#                     counterfactual_fairness  (Kusner Level-1, )
#   Procedural  (5):  process_consistency (σ=0.3), voice_representation,
#                     voice_enrichment, model_flippability_validity,
#                     explanation_actionability_validity

# Counterfactual_fairness uses the binary form on |𝒞|=2 and the
# multinomial TV-distance form on |𝒞|>2.
# Per-cell sample sizes are tightened relative to  (sample_n=200
# for PC/voice/CF, sample_n=20 for flippability/actionability) so the
# 52,080-cell matrix fits in the wall-clock budget .

# Canonical procedural σ for the Phase-5 audit.
PHASE5_PC_NOISE_STD: float = 0.3
# Tightened sample sizes for the audit.  used 500 / 30;
# does 12 metrics × 52,080 cells, so we tighten further to fit in the
# parallel wall-clock budget. Verified empirically on the smoke test
# (Tier-5 ricci/RF/identity finishes in <30s per cell).
PHASE5_SAMPLE_N_FAST: int = 200       # process_consistency / voice / CF
PHASE5_SAMPLE_N_FLIP: int = 20         # flippability / actionability

def _trimmed_partition(
    dataset_key: str, x_columns: list[str]
) -> dict[str, list[str]] | None:
    """Look up the  partition and trim to columns present in X.

    Returns ``None`` if the dataset has no registered partition (caller
    should record an N/A note for the procedural metrics).
    """
    # Late import: scripts/ is added to sys.path inside the worker so
    # ADR019_PARTITIONS resolves to the canonical Phase-4 source.
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from run_phase4_procedural import ADR019_PARTITIONS  # type: ignore

    partition = ADR019_PARTITIONS.get(dataset_key)
    if partition is None:
        return None
    cols_set = set(x_columns)
    trimmed = {
        "modifiable": [c for c in partition["modifiable"] if c in cols_set],
        "immutable": [c for c in partition["immutable"] if c in cols_set],
    }
    # Per : every column in X must appear in exactly one bucket.
    part_set = set(trimmed["modifiable"]) | set(trimmed["immutable"])
    missing = [c for c in x_columns if c not in part_set]
    if missing:
        # Spread orphan columns into "immutable" by default so the
        # voice_representation contract is satisfied without inventing a
        # judgement call for unknown features. : surface this in
        # the cell notes; do NOT silently triage as "good enough".
        trimmed = dict(trimmed)
        trimmed["immutable"] = list(trimmed["immutable"]) + missing
    return trimmed

def _compute_metrics(
    *,
    method,
    method_cls,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    A_test: pd.DataFrame,
    sens_test: pd.Series,
    sensitive_col: str,
    dataset_key: str,
    seed: int,
) -> tuple[list[dict], str]:
    """Compute the Tier-5 12-metric panel for one cell.

    Returns ``(rows, extra_notes)`` where ``rows`` is the list of metric
    dicts (with ``metric`` / ``value`` / ``metric_kind`` keys) and
    ``extra_notes`` accumulates per-metric graceful-fallback strings.
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
    )

    from procedural_fair_hr.fairness_metrics import (
        counterfactual_fairness,
        equal_opportunity_difference,
        macro_dp,
        macro_eodds,
        multinomial_counterfactual_fairness,
    )
    from procedural_fair_hr.procedural_fairness import (
        explanation_actionability,
        model_flippability,
        process_consistency,
        voice_representation,
    )

    rows: list[dict] = []
    notes_parts: list[str] = []
    n_classes = int(np.unique(np.concatenate([y_test, y_pred])).size)

    # --- Performance axis (3) -------------------------------------------------
    try:
        rows.append({
            "metric": "accuracy",
            "value": float(accuracy_score(y_test, y_pred)),
            "metric_kind": "performance",
        })
    except Exception as exc:
        rows.append({
            "metric": "accuracy",
            "value": float("nan"),
            "metric_kind": "performance",
        })
        notes_parts.append(f"accuracy_failed: {type(exc).__name__}")

    try:
        rows.append({
            "metric": "balanced_accuracy",
            "value": float(balanced_accuracy_score(y_test, y_pred)),
            "metric_kind": "performance",
        })
    except Exception as exc:
        rows.append({
            "metric": "balanced_accuracy",
            "value": float("nan"),
            "metric_kind": "performance",
        })
        notes_parts.append(f"balanced_accuracy_failed: {type(exc).__name__}")

    try:
        rows.append({
            "metric": "f1_macro",
            "value": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "metric_kind": "performance",
        })
    except Exception as exc:
        rows.append({
            "metric": "f1_macro",
            "value": float("nan"),
            "metric_kind": "performance",
        })
        notes_parts.append(f"f1_macro_failed: {type(exc).__name__}")

    # --- Statistical axis (4) -------------------------------------------------
    try:
        macro_dp_val, _ = macro_dp(y_test, y_pred, sens_test)
        rows.append({
            "metric": "macro_dp",
            "value": float(macro_dp_val),
            "metric_kind": "statistical",
        })
    except Exception as exc:
        rows.append({
            "metric": "macro_dp",
            "value": float("nan"),
            "metric_kind": "statistical",
        })
        notes_parts.append(f"macro_dp_failed: {type(exc).__name__}")

    try:
        macro_eodds_val, _ = macro_eodds(y_test, y_pred, sens_test)
        rows.append({
            "metric": "macro_eodds",
            "value": float(macro_eodds_val),
            "metric_kind": "statistical",
        })
    except Exception as exc:
        rows.append({
            "metric": "macro_eodds",
            "value": float("nan"),
            "metric_kind": "statistical",
        })
        notes_parts.append(f"macro_eodds_failed: {type(exc).__name__}")

    # equal_opportunity_difference is binary-only (Hardt 2016); on
    # multi-class targets we report the |𝒞|=2 binary-restriction value
    # of macro_eo on the positive-class restriction. Per  the
    # binary version takes y_true / y_pred as 0/1; we coerce to (==1)
    # when the label set is binary, otherwise emit NaN with note.
    try:
        if n_classes == 2:
            rows.append({
                "metric": "equal_opportunity",
                "value": float(
                    equal_opportunity_difference(y_test, y_pred, sens_test)
                ),
                "metric_kind": "statistical",
            })
        else:
            # Multi-class: report Hardt binary EO on the modal-class
            # restriction so the metric remains a float. : this
            # is documented in the notes column rather than triaged as
            # NaN-by-default.
            modal = int(pd.Series(y_test).mode().iloc[0])
            y_true_bin = (np.asarray(y_test) == modal).astype(int)
            y_pred_bin = (np.asarray(y_pred) == modal).astype(int)
            rows.append({
                "metric": "equal_opportunity",
                "value": float(
                    equal_opportunity_difference(y_true_bin, y_pred_bin, sens_test)
                ),
                "metric_kind": "statistical",
            })
            notes_parts.append(f"eo_modal_class={modal}")
    except Exception as exc:
        rows.append({
            "metric": "equal_opportunity",
            "value": float("nan"),
            "metric_kind": "statistical",
        })
        notes_parts.append(f"equal_opportunity_failed: {type(exc).__name__}")

    # Counterfactual fairness — Kusner Level-1. Binary on
    # |𝒞|=2; multinomial TV-distance on |𝒞|>2.
    try:
        # The CF metric needs the sensitive column to be a column in X.
        # The audit runner has already preprocessed X to ordinal-encoded
        # columns; the sensitive column is in the encoded space.
        if sensitive_col in X_test.columns:
            X_for_cf = X_test
            sens_col_for_cf = sensitive_col
        else:
            # Append the (encoded) sensitive column back to X for the CF
            # flip, using A_test as the source of truth.
            X_for_cf = X_test.copy()
            X_for_cf[sensitive_col] = A_test[sensitive_col].reset_index(drop=True).values
            sens_col_for_cf = sensitive_col

        if n_classes == 2:
            cf_val = counterfactual_fairness(
                y_true=None,
                y_pred=None,
                sensitive=sens_test,
                X=X_for_cf,
                model=method,
                sensitive_col=sens_col_for_cf,
                n_cf=min(PHASE5_SAMPLE_N_FAST, len(X_for_cf)),
            )
        else:
            cf_val = multinomial_counterfactual_fairness(
                y_true=None,
                y_pred=None,
                sensitive=sens_test,
                X=X_for_cf,
                model=method,
                sensitive_col=sens_col_for_cf,
                n_cf=min(PHASE5_SAMPLE_N_FAST, len(X_for_cf)),
            )
        rows.append({
            "metric": "counterfactual_fairness",
            "value": float(cf_val),
            "metric_kind": "statistical",
        })
    except Exception as exc:
        rows.append({
            "metric": "counterfactual_fairness",
            "value": float("nan"),
            "metric_kind": "statistical",
        })
        notes_parts.append(f"counterfactual_fairness_failed: {type(exc).__name__}")

    # --- Procedural axis (5) --------------------------------------------------
    # Process consistency at σ=0.3 (the canonical Phase-4 noise level).
    try:
        pc_overall, _ = process_consistency(
            method,
            X_test,
            perturbations_per_row=10,
            noise_std=PHASE5_PC_NOISE_STD,
            sample_n=min(PHASE5_SAMPLE_N_FAST, len(X_test)),
            stratify_on=sens_test,
            random_state=seed,
        )
        rows.append({
            "metric": "process_consistency",
            "value": float(pc_overall),
            "metric_kind": "procedural",
        })
    except Exception as exc:
        rows.append({
            "metric": "process_consistency",
            "value": float("nan"),
            "metric_kind": "procedural",
        })
        notes_parts.append(f"process_consistency_failed: {type(exc).__name__}")

    # Voice / Voice-Enrichment via SHAP. Use kernel explainer because the
    # mitigation wrappers are heterogeneous (sklearn / AIF360 / Fairlearn /
    # PyTorch); kernel works on any predict_proba/predict callable.
    partition = _trimmed_partition(dataset_key, list(X_test.columns))
    if partition is None:
        rows.append({
            "metric": "voice_representation",
            "value": float("nan"),
            "metric_kind": "procedural",
        })
        rows.append({
            "metric": "voice_enrichment",
            "value": float("nan"),
            "metric_kind": "procedural",
        })
        notes_parts.append("voice_partition_unknown")
    else:
        try:
            voice_overall, voice_enrich, _ = voice_representation(
                method,
                X_test,
                feature_partition=partition,
                shap_explainer="kernel",
                sample_n=min(PHASE5_SAMPLE_N_FAST, len(X_test)),
                random_state=seed,
            )
            rows.append({
                "metric": "voice_representation",
                "value": float(voice_overall),
                "metric_kind": "procedural",
            })
            rows.append({
                "metric": "voice_enrichment",
                "value": float(voice_enrich),
                "metric_kind": "procedural",
            })
        except Exception as exc:
            rows.append({
                "metric": "voice_representation",
                "value": float("nan"),
                "metric_kind": "procedural",
            })
            rows.append({
                "metric": "voice_enrichment",
                "value": float("nan"),
                "metric_kind": "procedural",
            })
            notes_parts.append(f"voice_failed: {type(exc).__name__}")

    # Model flippability validity (architectural).
    try:
        flip = model_flippability(
            method,
            X_test,
            sensitive=sens_test,
            max_features_to_flip=1,
            sample_n=min(PHASE5_SAMPLE_N_FLIP, len(X_test)),
            random_state=seed,
        )
        rows.append({
            "metric": "model_flippability_validity",
            "value": float(flip["validity"]),
            "metric_kind": "procedural",
        })
    except Exception as exc:
        rows.append({
            "metric": "model_flippability_validity",
            "value": float("nan"),
            "metric_kind": "procedural",
        })
        notes_parts.append(f"flippability_failed: {type(exc).__name__}")

    # Explanation actionability validity (procedural).
    if partition is None:
        rows.append({
            "metric": "explanation_actionability_validity",
            "value": float("nan"),
            "metric_kind": "procedural",
        })
        notes_parts.append("actionability_partition_unknown")
    else:
        try:
            act = explanation_actionability(
                method,
                X_test,
                feature_partition=partition,
                max_features_to_flip=1,
                sample_n=min(PHASE5_SAMPLE_N_FLIP, len(X_test)),
                random_state=seed,
            )
            rows.append({
                "metric": "explanation_actionability_validity",
                "value": float(act["actionable_validity"]),
                "metric_kind": "procedural",
            })
        except Exception as exc:
            rows.append({
                "metric": "explanation_actionability_validity",
                "value": float("nan"),
                "metric_kind": "procedural",
            })
            notes_parts.append(f"actionability_failed: {type(exc).__name__}")

    return rows, "; ".join(notes_parts)

def _run_cell(spec: CellSpec, out_dir_str: str) -> str:
    """Execute one Phase-5 cell and emit its parquet to the cache.

    This function runs inside a ``multiprocessing.Pool`` worker (with
    ``maxtasksperchild=20`` so the worker is recycled every 20 cells).
    It is designed to be self-contained: every import lives inside the
    function so the parent process doesn't carry the workers' deps.
    """
    out_dir = pathlib.Path(out_dir_str)
    cache_path = cell_cache_path(out_dir, spec)
    if cache_path.exists():
        return f"CACHED {spec.key()}"

    # Re-pin BLAS in the worker (idempotent).
    _worker_init(os.environ.get("PHASE5_DEVICE", "cpu"))

    # Seed numpy + Python random.
    import random as _random

    np.random.seed(spec.seed)
    _random.seed(spec.seed)

    from procedural_fair_hr.mitigation import MITIGATION_REGISTRY

    if spec.method not in MITIGATION_REGISTRY:
        raise KeyError(
            f"Method {spec.method!r} not in MITIGATION_REGISTRY "
            f"(known: {sorted(MITIGATION_REGISTRY)})"
        )

    ds_meta = DATASETS[spec.dataset]
    bundle = ds_meta["loader"]()

    X_train_raw = bundle["X_train"]
    X_test_raw = bundle["X_test"]
    y_train = np.asarray(bundle["y_train"]).astype(int)
    y_test = np.asarray(bundle["y_test"]).astype(int)
    A_train = bundle["A_train"]
    A_test = bundle["A_test"]

    X_train, X_test = _preprocess_xy(X_train_raw, X_test_raw)

    base = build_base_model(spec.base_model, seed=spec.seed)
    method_cls = MITIGATION_REGISTRY[spec.method]
    n_classes = int(np.unique(y_train).size)
    cell_notes = ds_meta.get("notes", "")

    # OvR auto-wrap for binary-only methods on multi-class targets per
    #  / . We instantiate the inner method via a factory
    # closure so the OvR adapter can clone it once per class.
    if n_classes > 2 and not getattr(method_cls, "multi_class_native", False):
        from procedural_fair_hr.mitigation.ovr_wrapper import OneVsRestFairnessAdapter

        def _factory():
            return method_cls(
                base_estimator=base,
                lambda_=spec.lambda_,
                sensitive_col=spec.sensitive_col,
                random_state=spec.seed,
            )

        method = OneVsRestFairnessAdapter(_factory, calibration="raw")
        try:
            method.fit(X_train, y_train, A_train)
        except NotImplementedError as exc:
            # Base estimator does not support sample_weight (e.g.,
            # Reweighing × KNN/MLP). Record N/A cell and short-circuit.
            cell_notes = (
                (cell_notes + "; " if cell_notes else "")
                + f"N/A — {exc}"
            )
            df = pd.DataFrame(
                [
                    {
                        "dataset": spec.dataset,
                        "target": ds_meta["target"],
                        "base_model": spec.base_model,
                        "method": spec.method,
                        "method_kind": getattr(method_cls, "method_kind", "in"),
                        "lambda_": float(spec.lambda_),
                        "seed": int(spec.seed),
                        "metric": "accuracy",
                        "value": float("nan"),
                        "metric_kind": "performance",
                        "sample_n": int(len(y_test)),
                        "random_state": int(spec.seed),
                        "notes": cell_notes,
                    }
                ],
                columns=CSV_COLUMNS,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)
            return f"NA {spec.key()}"
        # Some Tier-3 methods need the
        # sensitive attribute at predict time. Use predict_with_A when
        # available; otherwise fall back to predict(X).
        if hasattr(method, "predict_with_A"):
            y_pred = np.asarray(method.predict_with_A(X_test, A_test))
        else:
            y_pred = np.asarray(method.predict(X_test))
    else:
        method = method_cls(
            base_estimator=base,
            lambda_=spec.lambda_,
            sensitive_col=spec.sensitive_col,
            random_state=spec.seed,
        )
        try:
            method.fit(X_train, y_train, A_train)
        except NotImplementedError as exc:
            cell_notes = (
                (cell_notes + "; " if cell_notes else "")
                + f"N/A — {exc}"
            )
            df = pd.DataFrame(
                [
                    {
                        "dataset": spec.dataset,
                        "target": ds_meta["target"],
                        "base_model": spec.base_model,
                        "method": spec.method,
                        "method_kind": getattr(method_cls, "method_kind", "in"),
                        "lambda_": float(spec.lambda_),
                        "seed": int(spec.seed),
                        "metric": "accuracy",
                        "value": float("nan"),
                        "metric_kind": "performance",
                        "sample_n": int(len(y_test)),
                        "random_state": int(spec.seed),
                        "notes": cell_notes,
                    }
                ],
                columns=CSV_COLUMNS,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)
            return f"NA {spec.key()}"
        # Tier-3 predict-time A passthrough (see comment above).
        if hasattr(method, "predict_with_A"):
            y_pred = np.asarray(method.predict_with_A(X_test, A_test))
        else:
            y_pred = np.asarray(method.predict(X_test))
        # Surface optional method-side notes (graceful-fallback flags
        # from LFR convergence failures).
        method_notes = getattr(method, "notes_", "")
        if method_notes:
            cell_notes = (
                (cell_notes + "; " if cell_notes else "") + method_notes
            )

    # Sensitive series for fairness metrics.
    if spec.sensitive_col in A_test.columns:
        sens_test = A_test[spec.sensitive_col].reset_index(drop=True)
    else:
        sens_test = A_test.iloc[:, 0].reset_index(drop=True)

    metric_rows, metric_notes = _compute_metrics(
        method=method,
        method_cls=method_cls,
        X_test=X_test.reset_index(drop=True),
        y_test=y_test,
        y_pred=y_pred,
        A_test=A_test.reset_index(drop=True),
        sens_test=sens_test,
        sensitive_col=spec.sensitive_col,
        dataset_key=spec.dataset,
        seed=spec.seed,
    )
    if metric_notes:
        cell_notes = (cell_notes + "; " if cell_notes else "") + metric_notes

    rows: list[dict] = []
    for m in metric_rows:
        rows.append(
            {
                "dataset": spec.dataset,
                "target": ds_meta["target"],
                "base_model": spec.base_model,
                "method": spec.method,
                "method_kind": getattr(method_cls, "method_kind", "in"),
                "lambda_": float(spec.lambda_),
                "seed": int(spec.seed),
                "metric": m["metric"],
                "value": float(m["value"]) if m["value"] == m["value"] else float("nan"),
                "metric_kind": m["metric_kind"],
                "sample_n": int(len(y_test)),
                "random_state": int(spec.seed),
                "notes": cell_notes,
            }
        )

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    # Belt-and-braces leak mitigation: release Python-level references
    # (PyTorch grad buffers, SHAP caches, AIF360 dataset objects)
    # accumulated during this cell before returning to the parent. The
    # definitive bound on worker RSS comes from ``maxtasksperchild=20``
    # in the parent's Pool — see the parallel branch in ``main()``.
    gc.collect()
    return f"OK {spec.key()}"

# ---------------------------------------------------------------------
# Consolidator
# ---------------------------------------------------------------------

def consolidate_csv(out_dir: pathlib.Path) -> pathlib.Path:
    """Stitch every per-cell parquet into ``audit.csv`` with deterministic
    lexicographic sort. Returns the CSV path."""
    cache_dir = out_dir / "cache"
    out_csv = out_dir / "audit.csv"
    if not cache_dir.exists():
        # Empty run.
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(out_csv, index=False)
        return out_csv

    parquet_files = sorted(cache_dir.glob("*.parquet"))
    if not parquet_files:
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(out_csv, index=False)
        return out_csv

    frames = [pd.read_parquet(p) for p in parquet_files]
    df = pd.concat(frames, ignore_index=True)
    # Ensure column order.
    df = df.reindex(columns=CSV_COLUMNS)
    sort_keys = [
        "dataset",
        "base_model",
        "method",
        "lambda_",
        "seed",
        "metric",
    ]
    df = df.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    return out_csv

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _csv_strs(s: str) -> list[str]:
    return [t.strip() for t in s.split(",") if t.strip()]

def _csv_floats(s: str) -> list[float]:
    return [float(t.strip()) for t in s.split(",") if t.strip()]

def _csv_ints(s: str) -> list[int]:
    return [int(t.strip()) for t in s.split(",") if t.strip()]

DEFAULT_DATASETS = ["ibm_hr_attrition"]
DEFAULT_BASE_MODELS = ["RF"]
DEFAULT_METHODS = ["identity_preprocessing"]
DEFAULT_LAMBDAS = [0.0]
DEFAULT_SEEDS = [0]

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", type=_csv_strs, default=DEFAULT_DATASETS)
    p.add_argument("--base-models", type=_csv_strs, default=DEFAULT_BASE_MODELS)
    p.add_argument("--methods", type=_csv_strs, default=DEFAULT_METHODS)
    p.add_argument("--lambdas", type=_csv_floats, default=DEFAULT_LAMBDAS)
    p.add_argument("--seeds", type=_csv_ints, default=DEFAULT_SEEDS)
    p.add_argument("--max-workers", type=int, default=1)
    p.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/phase5"),
    )
    p.add_argument(
        "--memory-headroom-gb",
        type=float,
        default=4.0,
        help="If available RAM at startup is below this, reduce max_workers.",
    )
    p.add_argument(
        "--device",
        choices=("cpu", "mps", "auto"),
        default="cpu",
        help="PyTorch device for AdvDebias workers.",
    )
    return p.parse_args(argv)

def _enumerate_cells(args: argparse.Namespace) -> list[CellSpec]:
    cells: list[CellSpec] = []
    for ds in args.datasets:
        if ds not in DATASETS:
            raise KeyError(f"Unknown dataset {ds!r} (known: {list(DATASETS)})")
        ds_meta = DATASETS[ds]
        for bm in args.base_models:
            for method in args.methods:
                for lam in args.lambdas:
                    for seed in args.seeds:
                        cells.append(
                            CellSpec(
                                dataset=ds,
                                base_model=bm,
                                method=method,
                                lambda_=float(lam),
                                seed=int(seed),
                                sensitive_col=ds_meta["sensitive_col"],
                                target=ds_meta["target"],
                                notes=ds_meta.get("notes", ""),
                            )
                        )
    return cells

def _resolve_max_workers(requested: int, headroom_gb: float) -> int:
    """Auto-throttle if available memory is low."""
    try:
        import psutil

        avail_gb = psutil.virtual_memory().available / 1e9
    except Exception:
        return max(1, requested)
    if avail_gb < headroom_gb:
        return max(1, min(requested, 8))
    return max(1, requested)

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cache").mkdir(parents=True, exist_ok=True)

    cells = _enumerate_cells(args)
    pending = [c for c in cells if not cell_cache_exists(out_dir, c)]
    print(
        f"[phase5] {len(cells)} cells total; {len(pending)} pending; "
        f"max_workers={args.max_workers}; device={args.device}"
    )

    max_workers = _resolve_max_workers(
        args.max_workers, args.memory_headroom_gb
    )

    if not pending:
        consolidate_csv(out_dir)
        return 0

    t0 = time.perf_counter()

    if max_workers <= 1:
        # Synchronous path — useful for debugging + the byte-identical
        # parallel-vs-serial test that asserts the parallel run matches
        # the serial baseline.
        _worker_init(args.device)
        for spec in pending:
            try:
                msg = _run_cell(spec, str(out_dir))
                print(f"[phase5] {msg}")
            except Exception as e:
                print(f"[phase5] FAIL {spec.key()}: {e}", file=sys.stderr)
                raise
    else:
        # Tier-5b incident 2026-05-01: switched from ProcessPoolExecutor
        # to multiprocessing.Pool(maxtasksperchild=20) to recycle workers
        # every 20 cells. The previous sliding-window ProcessPoolExecutor
        # pattern bounded parent-process memory but workers themselves
        # accumulated PyTorch / SHAP / AIF360 C-extension allocations
        # across hundreds of cells, eventually triggering macOS Jetsam
        # OOM kills. ``imap_unordered(chunksize=1)`` is lazy and yields
        # as cells complete; combined with ``maxtasksperchild=20`` total
        # parent + worker RSS stays bounded for arbitrarily long runs.
        with Pool(
            processes=max_workers,
            initializer=_pool_initializer,
            initargs=(args.device,),
            maxtasksperchild=50,
        ) as pool:
            args_iter = ((spec, str(out_dir)) for spec in pending)
            for msg in pool.imap_unordered(
                _pool_run_cell, args_iter, chunksize=1
            ):
                print(f"[phase5] {msg}")

    consolidate_csv(out_dir)
    dt = time.perf_counter() - t0
    print(f"[phase5] done in {dt:.1f}s; CSV: {out_dir / 'audit.csv'}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

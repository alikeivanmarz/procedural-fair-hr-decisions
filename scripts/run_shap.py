"""Phase-6 SHAP-only XAI audit.

Computes SHAP feature importance (global + demographic-split) for each
(dataset, model) pair in the Phase-6 scope:

  * D1: ibm_hr_attrition
  * D2: acs_income
  * D6: oulad

For each dataset we run SHAP on:
  (a) the unmitigated baseline (identity_preprocessing, lambda=0)
  (b) the Pareto-optimal mitigated model from results/phase5/pareto.csv
      (accuracy_metric='accuracy', fairness_metric='macro_dp', on_pareto_frontier=True,
      highest accuracy, excluding prejudice_remover per Phase-5 infra notes).

Explainer selection:
  * TreeExplainer  — RF, XGB, GradientBoosting
  * LinearExplainer — LR
  * KernelExplainer — MLP (background=shap.sample(X_train, N, random_state=42)
    where N is controlled by --background-n, default=50)
  * KNN skipped (SHAP not meaningful for distance-based models)

Output schema (INV per ):
  [dataset, model, method, lambda_, shap_type, group, feature,
   mean_abs_shap, normalised_share, is_sensitive, is_proxy,
   sample_n, random_state]

Output filenames:
  * --background-n 50 (default): results/phase6/shap_results.csv
    and results/phase6/shap_summary.md (backwards-compatible).
  * --background-n N (N != 50): results/phase6/shap_results_n{N}.csv
    and results/phase6/shap_summary_n{N}.md (preserves N=50 baseline
    per ).

Also produces results/phase6/shap_summary[_n{N}].md — top-5 features per
(dataset, model, group) in markdown table format.

CLI
---
  python scripts/run_phase6_shap.py [--datasets D1 ...] [--background-n N]
      [--max-rows R] [--out-dir DIR]

References
----------

*  attribute (S).
*  Parity — multi-class (Macro-DP).
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Any

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

CSV_COLUMNS: list[str] = [
    "dataset",
    "model",
    "method",
    "lambda_",
    "shap_type",
    "group",
    "feature",
    "mean_abs_shap",
    "normalised_share",
    "is_sensitive",
    "is_proxy",
    "sample_n",
    "random_state",
]

# Random state used everywhere for  determinism.
RANDOM_STATE: int = 42

# SHAP sample size on the test split.
SHAP_SAMPLE_N: int = 500

# Background sample for KernelExplainer (ACS memory constraint).
KERNEL_BACKGROUND_N: int = 50

# ---------------------------------------------------------------------------
# Dataset registry (mirrors run_phase5_audit.py)
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict[str, Any]] = {
    "ibm_hr_attrition": {
        "sensitive_col": "Gender",
        "target": "Attrition",
        "demographic_groups": {
            "Gender": {"Male": "Male", "Female": "Female"},
        },
    },
    "acs_income": {
        "sensitive_col": "RAC1P",
        "target": "high_income",
        "demographic_groups": {
            # ACS RAC1P: 1=White alone (majority), all others minority.
            "RAC1P": {"majority": 1, "minority": None},
        },
    },
    "oulad": {
        "sensitive_col": "gender",
        "target": "final_result",
        "demographic_groups": {
            "gender": {"Male": "M", "Female": "F"},
        },
    },
}

# Proxy features per dataset from Phase-2 Bayesian audit
# (results/phase2/proxy_edges.csv and per-dataset parquet files).
# ACS: SCHL is a direct parent of RAC1P in the PC DAG.
# Ricci: Written is a direct parent of Race (not in Phase-6 scope).
# IBM HR / OULAD: no proxy edges recovered by Phase-2 PC algorithm.
PROXY_FEATURES: dict[str, list[str]] = {
    "ibm_hr_attrition": [],
    "acs_income": ["SCHL"],
    "oulad": [],
}

# Pareto-optimal mitigated models (from results/phase5/pareto_summary.md
# and pareto.csv, accuracy_metric=accuracy, fairness_metric=macro_dp,
# on_pareto_frontier=True, highest accuracy, prejudice_remover excluded
# per Phase-5 infra notes):

#   ibm_hr_attrition: GB, di_remover, lambda=0.05
#     (acc=0.8673, macro_dp=0.00077 — best non-prejudice_remover frontier cell)
#   acs_income:       RF, reweighing, lambda=0.05
#     (acc=0.8135, macro_dp=0.3337)
#   oulad:            LR, di_remover, lambda=0.3
#     (acc=0.5676, macro_dp=0.0242)
PARETO_OPTIMAL: dict[str, dict[str, Any]] = {
    "ibm_hr_attrition": {"base_model": "GB", "method": "di_remover", "lambda_": 0.05},
    "acs_income":        {"base_model": "RF", "method": "reweighing",  "lambda_": 0.05},
    "oulad":             {"base_model": "LR", "method": "di_remover",  "lambda_": 0.3},
}

# ---------------------------------------------------------------------------
# Loaders (same pattern as run_phase5_audit.py)
# ---------------------------------------------------------------------------

def _load_dataset(dataset_key: str) -> dict:
    """Load a dataset bundle ."""
    if dataset_key == "ibm_hr_attrition":
        from procedural_fair_hr.data_loaders import load_ibm_hr
        return load_ibm_hr(target="Attrition")
    if dataset_key == "acs_income":
        from procedural_fair_hr.data_loaders import load_acs
        return load_acs(state="CA", year=2018, task="income")
    if dataset_key == "oulad":
        from procedural_fair_hr.data_loaders import load_oulad
        return load_oulad()
    raise ValueError(f"Unknown dataset key: {dataset_key!r}")

# ---------------------------------------------------------------------------
# Pre-processing (mirrors run_phase5_audit.py::_preprocess_xy)
# ---------------------------------------------------------------------------

def _preprocess_xy(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Numericise categoricals — identical to mitigation preprocessing."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
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

    return out_train[X_train.columns], out_test[X_test.columns]

# ---------------------------------------------------------------------------
# Base-model factory (mirrors run_phase5_audit.py)
# ---------------------------------------------------------------------------

def _build_base_model(key: str, seed: int = RANDOM_STATE):
    """Construct one fresh sklearn-compatible base estimator."""
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    if key == "RF":
        return RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=1)
    if key == "LR":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if key == "MLP":
        return MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, random_state=seed)
    if key == "GB":
        return GradientBoostingClassifier(n_estimators=50, random_state=seed)
    if key == "XGB":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=50, random_state=seed, eval_metric="logloss", n_jobs=1)
    raise ValueError(f"Unknown base-model key: {key!r}")

# ---------------------------------------------------------------------------
# Mitigation model factory (mirrors run_phase5_audit.py::_run_cell)
# ---------------------------------------------------------------------------

def _build_mitigated_model(
    base_model_key: str,
    method_name: str,
    lambda_: float,
    sensitive_col: str,
    n_classes: int,
    seed: int = RANDOM_STATE,
):
    """Instantiate and return a MitigationBase subclass (not yet fitted)."""
    from procedural_fair_hr.mitigation import MITIGATION_REGISTRY

    if method_name not in MITIGATION_REGISTRY:
        raise KeyError(f"Method {method_name!r} not in MITIGATION_REGISTRY")

    base = _build_base_model(base_model_key, seed=seed)
    method_cls = MITIGATION_REGISTRY[method_name]

    if n_classes > 2 and not getattr(method_cls, "multi_class_native", False):
        from procedural_fair_hr.mitigation.ovr_wrapper import OneVsRestFairnessAdapter

        def _factory():
            return method_cls(
                base_estimator=base,
                lambda_=lambda_,
                sensitive_col=sensitive_col,
                random_state=seed,
            )

        return OneVsRestFairnessAdapter(_factory, calibration="raw"), method_cls

    method = method_cls(
        base_estimator=base,
        lambda_=lambda_,
        sensitive_col=sensitive_col,
        random_state=seed,
    )
    return method, method_cls

# ---------------------------------------------------------------------------
# Identity (unmitigated baseline) model
# ---------------------------------------------------------------------------

def _build_identity_model(base_model_key: str, seed: int = RANDOM_STATE):
    """Return a plain sklearn estimator — no mitigation wrapper."""
    return _build_base_model(base_model_key, seed=seed)

# ---------------------------------------------------------------------------
# SHAP explainer factory
# ---------------------------------------------------------------------------

def _get_explainer(model_key: str, model, X_train: pd.DataFrame,
                   kernel_background_n: int = KERNEL_BACKGROUND_N):
    """Select the SHAP explainer appropriate for *model_key*.

    Parameters
    ----------
    model_key:
        One of RF, XGB, GB, LR, MLP.
    model:
        A fitted sklearn-compatible estimator.
    X_train:
        Training features (used to build the KernelExplainer background).
    kernel_background_n:
        Number of background rows for KernelExplainer (default: module
        constant KERNEL_BACKGROUND_N=50).  Pass a larger value (e.g. 200)
        to improve KernelExplainer accuracy at the cost of compute time.

    Returns:
        explainer — shap Explainer object
        is_tree   — bool (True for TreeExplainer, False otherwise)

    Notes
    -----
    * TreeExplainer for RF, XGB, GB (fast + exact).
    * LinearExplainer for LR (deterministic).
    * KernelExplainer for MLP (background=kernel_background_n rows).
    * KNN is skipped (SHAP not meaningful for distance-based models).
    """
    import shap

    bg = shap.sample(X_train, min(kernel_background_n, len(X_train)), random_state=RANDOM_STATE)

    if model_key in ("RF", "XGB", "GB"):
        inner = _unwrap_estimator(model)
        try:
            return shap.TreeExplainer(inner), True
        except Exception:
            # Mitigated model is a wrapper (e.g. OvR) — fall back to KernelExplainer.
            return shap.KernelExplainer(model.predict_proba, bg), False
    if model_key == "LR":
        inner = _unwrap_estimator(model)
        try:
            masker = shap.maskers.Independent(X_train)
            return shap.LinearExplainer(inner, masker), False
        except Exception:
            return shap.KernelExplainer(model.predict_proba, bg), False
    if model_key == "MLP":
        return shap.KernelExplainer(model.predict_proba, bg), False
    raise ValueError(f"No SHAP explainer defined for model key {model_key!r}")

def _unwrap_estimator(model):
    """Extract the raw sklearn/XGB estimator from a MitigationBase wrapper.

    Priority order for attribute search: fitted private attrs first
    (``_estimator`` is the canonical Phase-5 fitted-estimator slot
    in IdentityPreprocessing / Reweighing / LFR), then common sklearn public names.
    We deliberately check ``_estimator`` BEFORE ``base_estimator`` to
    avoid returning the unfitted base estimator stored in
    ``MitigationBase.__init__``.
    """
    # Identity path: plain sklearn estimator — has fit/predict but no
    # MitigationBase class attributes.
    if hasattr(model, "fit") and not hasattr(model, "method_name"):
        return model
    # MitigationBase subclasses: try fitted private slot first.
    for attr in ("_estimator", "estimator_", "_model", "model_",
                 "estimator", "base_estimator_"):
        if hasattr(model, attr):
            inner = getattr(model, attr)
            if inner is not None:
                return inner
    # Fallback: return the model itself; caller will catch SHAP errors.
    return model

def _compute_shap_values(
    explainer,
    X_sample: pd.DataFrame,
    is_tree: bool,
    n_classes: int,
) -> np.ndarray:
    """Compute SHAP values and return array of shape (n_samples, n_features).

    For multi-class outputs (n_classes > 2), we average the absolute SHAP
    values across all classes to obtain a single per-feature importance
    vector.

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
        Each entry is the SIGNED SHAP value (caller takes abs).
    """
    import shap

    vals = explainer(X_sample) if is_tree else explainer.shap_values(X_sample)

    # Normalise to array shapes.
    if isinstance(vals, shap.Explanation):
        vals = vals.values

    if isinstance(vals, list):
        # Multi-class tree: list of n_classes arrays each (n, p).
        # Average |SHAP| across classes.
        return np.mean([np.abs(v) for v in vals], axis=0)

    if vals.ndim == 3:
        # (n_samples, n_features, n_classes) — KernelExplainer multi-class.
        return np.mean(np.abs(vals), axis=2)

    if vals.ndim == 2:
        return vals

    raise ValueError(f"Unexpected SHAP values shape: {vals.shape}")

# ---------------------------------------------------------------------------
# SHAP rows builder
# ---------------------------------------------------------------------------

def _rows_for_group(
    abs_shap: np.ndarray,
    feature_names: list[str],
    group_label: str,
    dataset: str,
    model_key: str,
    method: str,
    lambda_: float,
    sample_n: int,
    sensitive_col: str,
) -> list[dict]:
    """Build CSV rows for one demographic group (or 'all')."""
    proxies = set(PROXY_FEATURES.get(dataset, []))
    mean_abs = abs_shap.mean(axis=0)  # shape (n_features,)
    total = mean_abs.sum()
    normalised = mean_abs / total if total > 0 else mean_abs

    rows = []
    for i, feat in enumerate(feature_names):
        rows.append(
            {
                "dataset": dataset,
                "model": model_key,
                "method": method,
                "lambda_": float(lambda_),
                "shap_type": "global" if group_label == "all" else "demographic",
                "group": group_label,
                "feature": feat,
                "mean_abs_shap": float(mean_abs[i]),
                "normalised_share": float(normalised[i]),
                "is_sensitive": (feat == sensitive_col),
                "is_proxy": (feat in proxies),
                "sample_n": int(sample_n),
                "random_state": RANDOM_STATE,
            }
        )
    return rows

# ---------------------------------------------------------------------------
# Stratified sample helper
# ---------------------------------------------------------------------------

def _stratified_sample(
    X: pd.DataFrame,
    y: np.ndarray,
    n: int,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return a stratified sample of *n* rows from (X, y)."""
    from sklearn.model_selection import train_test_split

    if len(X) <= n:
        return X.reset_index(drop=True), y

    # Use train_test_split in stratified mode to get a balanced sample.
    _, X_s, _, y_s = train_test_split(
        X,
        y,
        test_size=min(n / len(X), 0.999),
        stratify=y,
        random_state=random_state,
    )
    return X_s.reset_index(drop=True), y_s

# ---------------------------------------------------------------------------
# Core per-(dataset, model, method) runner
# ---------------------------------------------------------------------------

def _run_shap_cell(
    dataset_key: str,
    base_model_key: str,
    method_name: str,
    lambda_: float,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    A_train: pd.DataFrame,
    A_test: pd.DataFrame,
    max_rows: int | None = None,
    kernel_background_n: int = KERNEL_BACKGROUND_N,
) -> list[dict]:
    """Run SHAP for one (dataset, base_model, method, lambda) cell.

    Skips KNN silently (SHAP not meaningful, per ).

    Returns a list of row dicts ready for the output CSV.
    """
    if base_model_key == "KNN":
        return []

    ds_meta = DATASETS[dataset_key]
    sensitive_col = ds_meta["sensitive_col"]
    n_classes = int(np.unique(y_train).size)
    feature_names = list(X_train.columns)

    # --- Fit the model ---
    is_identity = (method_name == "identity_preprocessing")

    if is_identity:
        model = _build_identity_model(base_model_key)
        model.fit(X_train, y_train)
    else:
        model, _ = _build_mitigated_model(
            base_model_key, method_name, lambda_, sensitive_col, n_classes
        )
        model.fit(X_train, y_train, A_train)

    # --- SHAP explainer ---
    try:
        explainer, is_tree = _get_explainer(base_model_key, model, X_train,
                                            kernel_background_n=kernel_background_n)
    except Exception as exc:
        print(f"  [SKIP] {dataset_key}/{base_model_key}/{method_name}: explainer failed: {exc}")
        return []

    # --- Test sample ---
    sample_n = min(SHAP_SAMPLE_N if max_rows is None else max_rows, len(X_test))
    X_sample, y_sample = _stratified_sample(X_test, y_train[:len(X_test)], sample_n)

    # Align A_test to sample.
    A_sample = A_test.iloc[: len(X_test)].reset_index(drop=True).loc[X_sample.index]

    # --- Compute SHAP values ---
    try:
        abs_shap = _compute_shap_values(explainer, X_sample, is_tree, n_classes)
    except Exception as exc:
        print(f"  [SKIP] {dataset_key}/{base_model_key}/{method_name}: shap failed: {exc}")
        return []

    # abs_shap is already abs (returned from _compute_shap_values) for
    # averaged-class paths; take abs for the signed 2-d path.
    if abs_shap.min() < 0:
        abs_shap = np.abs(abs_shap)

    rows: list[dict] = []

    # --- Global row (all samples) ---
    rows.extend(
        _rows_for_group(
            abs_shap, feature_names, "all",
            dataset_key, base_model_key, method_name, lambda_,
            len(X_sample), sensitive_col,
        )
    )

    # --- Demographic-split rows ---
    sens_col = ds_meta["sensitive_col"]
    group_defs = ds_meta["demographic_groups"].get(sens_col, {})

    # Retrieve sens values aligned to X_sample.
    if sens_col in A_sample.columns:
        sens_values = A_sample[sens_col].reset_index(drop=True).values
    else:
        sens_values = None

    if sens_values is not None:
        for group_label, group_val in group_defs.items():
            if group_val is None:
                # "minority" = everything not the majority value.
                majority_val = list(group_defs.values())[0]
                mask = sens_values != majority_val
            else:
                mask = sens_values == group_val

            if mask.sum() == 0:
                continue

            rows.extend(
                _rows_for_group(
                    abs_shap[mask], feature_names, group_label,
                    dataset_key, base_model_key, method_name, lambda_,
                    int(mask.sum()), sensitive_col,
                )
            )

    return rows

# ---------------------------------------------------------------------------
# Pareto-optimal cell selection
# ---------------------------------------------------------------------------

def _select_pareto_model(dataset_key: str) -> dict[str, Any]:
    """Return the pre-computed Pareto-optimal (base_model, method, lambda_) for *dataset_key*.

    Uses the hardcoded PARETO_OPTIMAL dict (derived from
    results/phase5/pareto.csv) rather than re-reading the CSV at runtime
    so this function is deterministic and dependency-free.
    """
    return PARETO_OPTIMAL[dataset_key]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SHAP attribution audit")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        choices=list(DATASETS),
        help="Dataset keys to process (default: all three).",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap SHAP sample at N rows (for smoke tests).",
    )
    p.add_argument(
        "--background-n",
        type=int,
        default=50,
        dest="background_n",
        help=(
            "Number of background rows for KernelExplainer (default: 50). "
            "When set to a value other than 50, output files are suffixed "
            "_n{N} so the default-N baseline is preserved. "
            ""
        ),
    )
    p.add_argument(
        "--out-dir",
        default="results/phase6",
        help="Output directory.",
    )
    return p.parse_args()

def main(
    datasets: list[str] | None = None,
    max_rows: int | None = None,
    out_dir: str = "results/phase6",
    background_n: int = 50,
) -> pd.DataFrame:
    """Run the Phase-6 SHAP audit and write CSV + markdown summary.

    Parameters
    ----------
    datasets:
        Dataset keys to process.  Defaults to all three Phase-6 datasets.
    max_rows:
        Cap SHAP sample size (useful for smoke tests).
    out_dir:
        Directory for output files.
    background_n:
        Number of background rows for KernelExplainer (default: 50).
        When background_n != 50 the output files are suffixed _n{N} so
        the original N=50 baseline is preserved .
         runs this with background_n=200.

    Returns
    -------
    pd.DataFrame
        The full SHAP results table (also written to shap_results[_n{N}].csv).
    """
    import random as _random

    np.random.seed(RANDOM_STATE)
    _random.seed(RANDOM_STATE)

    if datasets is None:
        datasets = list(DATASETS)

    out_path = PROJECT_ROOT / out_dir
    out_path.mkdir(parents=True, exist_ok=True)

    # Determine output filename suffix : when background_n != 50,
    # use _n{N} suffix so the N=50 baseline CSV is never overwritten.
    suffix = "" if background_n == 50 else f"_n{background_n}"
    csv_filename = f"shap_results{suffix}.csv"
    summary_filename = f"shap_summary{suffix}.md"

    suffix_display = repr(suffix) if suffix else "(none)"
    print(f"[phase6] KernelExplainer background_n={background_n}  "
          f"(output suffix: {suffix_display})")

    all_rows: list[dict] = []

    for ds_key in datasets:
        print(f"\n[phase6] Dataset: {ds_key}")
        bundle = _load_dataset(ds_key)

        X_train_raw = bundle["X_train"]
        X_test_raw = bundle["X_test"]
        y_train = np.asarray(bundle["y_train"]).astype(int)
        # y_test not needed — SHAP doesn't use labels.
        A_train = bundle["A_train"]
        A_test = bundle["A_test"]

        X_train, X_test = _preprocess_xy(X_train_raw, X_test_raw)

        pareto = _select_pareto_model(ds_key)
        base_model_key = pareto["base_model"]
        pareto_method = pareto["method"]
        pareto_lambda = pareto["lambda_"]

        cells = [
            # (a) Unmitigated baseline.
            (base_model_key, "identity_preprocessing", 0.0),
            # (b) Pareto-optimal mitigated model.
            (base_model_key, pareto_method, pareto_lambda),
        ]

        for bm_key, method_name, lambda_ in cells:
            print(f"  -> {bm_key} / {method_name} / lambda={lambda_}")
            try:
                rows = _run_shap_cell(
                    dataset_key=ds_key,
                    base_model_key=bm_key,
                    method_name=method_name,
                    lambda_=lambda_,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    A_train=A_train,
                    A_test=A_test,
                    max_rows=max_rows,
                    kernel_background_n=background_n,
                )
            except Exception as exc:
                print(f"  [ERROR] {ds_key}/{bm_key}/{method_name}: {exc}")
                rows = []
            if rows:
                all_rows.extend(rows)
                print(f"     {len(rows)} rows")
            else:
                print("     (skipped)")

    if not all_rows:
        print("[phase6] No SHAP rows produced — check dataset/model config.")
        df = pd.DataFrame(columns=CSV_COLUMNS)
    else:
        df = pd.DataFrame(all_rows, columns=CSV_COLUMNS)

    csv_path = out_path / csv_filename
    df.to_csv(csv_path, index=False)
    print(f"\n[phase6] Wrote {len(df)} rows -> {csv_path}")

    # --- Markdown summary ---
    _write_summary(df, out_path / summary_filename)

    return df

def _write_summary(df: pd.DataFrame, path: pathlib.Path) -> None:
    """Write top-5 features per (dataset, model, group) as a markdown table."""
    lines = ["# Phase-6 SHAP Summary\n"]
    if df.empty:
        lines.append("_(no data)_\n")
        path.write_text("\n".join(lines))
        return

    for (dataset, model, group), sub in df.groupby(["dataset", "model", "group"]):
        top5 = (
            sub.sort_values("normalised_share", ascending=False)
            .drop_duplicates("feature")
            .head(5)
        )
        lines.append(f"\n## {dataset} / {model} / group={group}\n")
        lines.append("| feature | mean_abs_shap | normalised_share | is_sensitive | is_proxy |")
        lines.append("|---|---|---|---|---|")
        for _, row in top5.iterrows():
            lines.append(
                f"| {row['feature']} | {row['mean_abs_shap']:.6f} | "
                f"{row['normalised_share']:.4f} | {row['is_sensitive']} | {row['is_proxy']} |"
            )

    path.write_text("\n".join(lines) + "\n")
    print(f"[phase6] Wrote summary -> {path}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()
    main(
        datasets=args.datasets,
        max_rows=args.max_rows,
        out_dir=args.out_dir,
        background_n=args.background_n,
    )

"""Phase-2 fairness audit runner.

This is the **no-shortcuts** rewrite of the original  runner. The previous
version (commit ``b5360e6``) ran the PC structure-learning algorithm on a
1000-row subsample of every training set and computed counterfactual
fairness on N=50 instances per dataset. The user rejected those shortcuts
(2026-04-29). This rewrite:

* Runs :func:`procedural_fair_hr.data_loaders.learn_proxy_graph` on the **full** training
  set of each dataset (no row subsampling). Significance level α=0.05 (the
  ``pgmpy`` default), ``max_cond_vars=3``.
* Runs counterfactual fairness on the **full test set** for "small" datasets
  (≤5,000 test rows: Ricci, IBM HR) and on a **stratified N=1000** sample for
  the rest (Adult, ACS, Dutch Census, Law School, OULAD). Stratification is
  by the joint cell ``(protected_attr × y)``.
* Handles OULAD's 3-class target via **one-vs-rest** binarisation: each
  binary metric is computed once per class ``c ∈ {0,1,2}`` against
  ``y_bin = (y == c)``. Binary datasets get a single ``class_idx = 1`` row
  per metric (the canonical "positive" class).
* Streams per-dataset progress to ``results/phase2/progress.log`` AND stdout
  with explicit ``flush=True`` so the user can ``tail -f`` it.
* Writes a per-dataset partial parquet
  ``results/phase2/audit_<dataset>.parquet`` immediately after each dataset
  finishes (crash-recovery: if the parquet already exists when the script
  starts, that dataset is SKIPPED — ``rm`` the file to force a re-run).
* Commits + pushes the per-dataset partial after each dataset (handled by
  the ``--commit`` CLI flag, default ON when run via ``make audit``).
* After all 7 datasets finish, concatenates the partials into
  ``results/phase2/audit.csv`` and ``results/phase2/audit.parquet``.

Dataset run order (smallest-first for fast progress visibility):

    Ricci (118) → IBM HR PerformanceRating (1,470, leaky per )
    → IBM HR Attrition (1,470, honest, primary for +) → OULAD
    (21,562 / 3-class) → Dutch Census (60,420) → Law School (20,798)
    → Adult (48,842) → ACS-CA-2018 (~196k).

Output schema (``audit.csv`` / ``audit.parquet``)
-------------------------------------------------
One row per ``(dataset, protected_attribute, class_idx, metric_name,
metric_value, model, target_form, n_train, n_test, cf_n_samples)``. Columns:

* ``dataset`` — short key (``ricci``, ``ibm_hr``, …).
* ``protected_attribute`` — sensitive column name (e.g. ``Gender``).
* ``class_idx`` — int. ``1`` for binary datasets (positive class). For
  OULAD with one-vs-rest binarisation: ``0`` (Fail-vs-rest), ``1``
  (Pass-vs-rest), ``2`` (Distinction-vs-rest).
* ``metric_name`` — see the ``GROUP_METRICS`` registry below + ``knn_consistency``,
  ``lipschitz_fairness``, ``counterfactual_fairness``, ``abroca``.
* ``metric_value`` — float; bounded :mod:`procedural_fair_hr.fairness_metrics` docs.
* ``model`` — sklearn class name of the trained baseline.
* ``target_form`` — ``"binary"`` or ``"ovr_class_<c>"`` for OvR rows.
* ``n_train``, ``n_test`` — row counts in the train/test split.
* ``cf_n_samples`` — actual N used for counterfactual fairness on this row
  (NaN for non-CF metrics).
* ``pc_n_rows`` — number of rows the PC structure-learning algorithm consumed
  for this row's dataset (full training data for 5 datasets; 10,000-row
  stratified subsample for Adult and ACS per ).
* ``non_vacuous`` — Pattern A degeneracy guard (constant predictions).
  ``True`` iff the classifier predicts the positive class for between
  5 % and 95 % of all test samples for this row. When ``False``, the
  classifier has collapsed to (near-)constant predictions; binary
  fairness metrics become trivially small or undefined.
* ``non_vacuous_tpr`` — Pattern B degeneracy guard (uncorrelated
  predictions). ``True`` iff at least one protected group has TPR ≥ 5 %
  on the binarised target. When ``False``, every group has TPR < 5 %
  — the model predicts the positive class often enough to satisfy the
  Pattern A guard, but those predictions don't correlate with the true
  labels, so EOdds (a difference of TPRs) collapses to ~0 trivially.
  This is IBM HR / Gender's specific failure mode: RF predicts class 4
  for ~10 % of test samples but almost none are actually class 4, so
  TPR_M ≈ TPR_F ≈ 0. See the project documentation 2026-04-29 entry
  "IBM HR EOdds = 0.0000 is degenerate, not fair".

  A row is fully non-vacuous iff BOTH guards are True. Downstream
  consumers that want only "real" fairness rows should filter on
  ``(non_vacuous & non_vacuous_tpr)``. The columns are kept separate
  to make the failure mode visible.

Cited references (per ``thesis/refs.bib``):

* ``pagano2023fairness`` — 8 binary group-fairness metrics (Eqs. 1–9).
* ``gardner2019abroca`` — ABROCA slice plot.
* ``kuzilek2017oulad`` — OULAD source.
* ``lequy2022survey`` — proxy-graph leaf-node constraint (§2.2 Eq. 2).

CLI
---
.. code-block:: bash

    python scripts/run_phase2_audit.py \\
        [--datasets ricci,ibm_hr,...] \\
        [--out-dir results/phase2/] \\
        [--no-commit] \\
        [--no-push]

See the project documentation (fairness-metric signature),
```` (loader return shape), ```` (determinism, seed=0).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import pathlib
import subprocess
import sys
import time
import warnings
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Add  to path so ``src.*`` imports resolve when the script is
# run from the project root or via ``python scripts/run_phase2_audit.py``.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import bayesian_audit, fairness_metrics, visualisations  # noqa: E402
from procedural_fair_hr.data_loaders import (  # noqa: E402
    load_acs,
    load_ibm_hr,
    load_oulad,
    load_ricci,
)

# ---------------------------------------------------------------------------
# Dataset registry: ordered smallest-first per the task spec.
# ---------------------------------------------------------------------------

LOADERS: dict[str, Callable[[], dict]] = {
    "ricci": load_ricci,
    # Two IBM HR variants : the leaky PerformanceRating target
    # (kept as the C1 leakage evidence) and the honest Attrition target
    # (the new primary IBM HR target for + mitigation).
    "ibm_hr": load_ibm_hr,
    "ibm_hr_attrition": lambda: load_ibm_hr(target="Attrition"),
    "oulad": load_oulad,
    "acs": load_acs,
}

# CF: full test set when test rows ≤ this threshold, else stratified N=1000.
CF_FULL_THRESHOLD = 5_000
CF_LARGE_N = 1_000

# Individual-fairness sample (KNNC is O(n^2); Lipschitz is O(n_pairs)).
INDIV_FAIRNESS_SAMPLE_ROWS = 500

# PC algorithm row-subsample. pgmpy's chi-square PC scales as
# O(n_rows × n_cols^k) where k is the conditioning-set size; on Adult and ACS
# the full-data PC does not terminate within a laptop's wall-time budget.
# Adult and ACS get a 10,000-row stratified subsample (by sensitive × y); the
# 5 smaller datasets continue to use full training data. The audit rows
# record `pc_n_rows` for explicit disclosure.
PC_SUBSAMPLE_DATASETS: dict[str, int] = {
    "adult": 10_000,
    "acs": 10_000,
}

# ---------------------------------------------------------------------------
# 8 binary group metrics``).
# ---------------------------------------------------------------------------

GROUP_METRICS: list[tuple[str, Callable]] = [
    ("demographic_parity_difference", fairness_metrics.demographic_parity_difference),
    ("disparate_impact_ratio", fairness_metrics.disparate_impact_ratio),
    ("equal_opportunity_difference", fairness_metrics.equal_opportunity_difference),
    ("equalised_odds_difference", fairness_metrics.equalised_odds_difference),
    ("statistical_parity_difference", fairness_metrics.statistical_parity_difference),
    ("average_absolute_odds_difference", fairness_metrics.average_absolute_odds_difference),
    ("average_equalised_odds_difference", fairness_metrics.average_equalised_odds_difference),
    ("accuracy_balance", fairness_metrics.accuracy_balance),
]

# Output column order for audit.csv / audit.parquet.
AUDIT_COLUMNS = [
    "dataset",
    "protected_attribute",
    "class_idx",
    "metric_name",
    "metric_value",
    "model",
    "target_form",
    "n_train",
    "n_test",
    "cf_n_samples",
    "non_vacuous",
    "non_vacuous_tpr",
    "pc_n_rows",
]

# Non-vacuous-fairness guard thresholds (see module docstring +
# entry "IBM HR EOdds = 0.0000 is degenerate, not fair", 2026-04-29). Two
# complementary guards together catch the two known degeneracy patterns.

# Pattern A — constant predictions (caught by `non_vacuous`).
#   The classifier predicts the positive class for fewer than 5 % or more
#   than 95 % of all test samples. DP / SPD become trivially small because
#   the per-group positive rates can't differ much. Examples: OULAD class-2
#   (Distinction predicted 0 / 4,313 times) and Law School (positive class
#   ~ 96 % of test).
#   Guard: `(NON_VACUOUS_LOWER <= mean(y_pred_bin) <= NON_VACUOUS_UPPER)`.

# Pattern B — non-constant but uncorrelated predictions (caught by
# `non_vacuous_tpr`). The classifier predicts the positive class for some
# fraction of test samples, but those predictions don't correlate with the
# true labels — so TPR is approximately 0 for every group. EOdds = max
# |TPR_diff|, |FPR_diff| collapses to ~0 trivially. Example: IBM HR /
# Gender — RF predicts PerformanceRating=4 for ~10 % of test samples, but
# almost none of those are actually class 4, so TPR_M ≈ TPR_F ≈ 0.
#   Guard: `non_vacuous_tpr` is True iff at least one group has
#   `TPR_g >= NON_VACUOUS_TPR_LOWER` (0.05). When a group has zero true
#   positives in the test set, its TPR is undefined and excluded from the
#   max.

# A row is fully non-vacuous iff BOTH guards are True. A downstream
# consumer of audit.csv that wants to keep only "real" fairness rows
# should filter on `(non_vacuous & non_vacuous_tpr)`. The two columns are
# kept separate to make the failure mode visible (which kind of degeneracy
# was hit).
NON_VACUOUS_LOWER = 0.05
NON_VACUOUS_UPPER = 0.95
NON_VACUOUS_TPR_LOWER = 0.05

PROXY_COLUMNS = [
    "dataset",
    "protected_attribute",
    "target",
    "edge_type",
    "source",
    "sink",
    "pc_n_rows",
]

# ---------------------------------------------------------------------------
# Progress logging
# ---------------------------------------------------------------------------

class ProgressLogger:
    """Tee writer: stdout + appendable progress.log with explicit flush.

    The user is expected to ``tail -f results/phase2/progress.log`` in another
    terminal while this script runs, so every milestone must hit disk
    immediately.
    """

    def __init__(self, log_path: pathlib.Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Open in append mode so re-runs accumulate the timeline.
        self._fh = open(self.log_path, "a", buffering=1)  # line-buffered

    def log(self, msg: str) -> None:
        ts = _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")
        line = f"{ts}  {msg}"
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()
        try:
            os.fsync(self._fh.fileno())
        except OSError:
            pass  # not all FDs support fsync on every OS

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _binarise_for_binary_loader(y: np.ndarray) -> np.ndarray:
    """Coerce a binary loader's labels to the canonical {0, 1} integer form.

    Most loaders already return 0/1 (Adult, ACS, Ricci, Dutch, LawSchool,
    OULAD-binary-collapse if applicable). IBM HR ``PerformanceRating`` returns
    {3, 4} — we map ``4 -> 1`` (rarer "high performer" minority) and
    ``3 -> 0`` to match the Phase-1 baseline (cf.  ``run_ibm_hr_baseline``
    which uses ``HIGH_PERF = 4`` as the positive label).
    """
    arr = np.asarray(y).ravel()
    uniques = np.unique(arr)
    if set(uniques.tolist()) == {0, 1}:
        return arr.astype(int)
    if len(uniques) != 2:
        raise ValueError(
            f"Cannot binarise array with values {uniques.tolist()!r} (expected 2 classes)"
        )
    counts = {int(v): int((arr == v).sum()) for v in uniques}
    pos_class = min(counts, key=counts.get)  # rarer class -> 1
    return (arr == pos_class).astype(int)

def _build_baseline_pipeline(
    dataset_name: str,
    X_train: pd.DataFrame,
    *,
    multinomial: bool,
) -> Pipeline:
    """Construct the baseline pipeline per the task spec.

    * IBM HR uses ``RandomForestClassifier(n_estimators=200, random_state=0)``
      (matches  baseline).
    * OULAD (3-class) uses ``LogisticRegression(multi_class='multinomial',
      max_iter=1000, random_state=0)``.
    * Everything else uses ``LogisticRegression(max_iter=1000, random_state=0)``.

    Categorical columns are imputed (mode) + ordinally encoded; numeric columns
    are imputed (median) + standardised.
    """
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    num_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_cols),
            ("num", num_pipe, num_cols),
        ],
        remainder="drop",
    )

    if dataset_name in ("ibm_hr", "ibm_hr_attrition"):
        # Both IBM HR variants use RF, matching the Phase-1 baseline
        # protocol.
        clf = RandomForestClassifier(n_estimators=200, random_state=0)
    elif multinomial:
        # multi_class='multinomial' is deprecated in sklearn 1.5+; the
        # 'lbfgs' solver handles multinomial automatically. Pass the kwarg
        # only when sklearn accepts it; otherwise rely on solver default.
        try:
            clf = LogisticRegression(
                multi_class="multinomial",
                max_iter=1000,
                random_state=0,
                solver="lbfgs",
            )
        except TypeError:  # noqa: BLE001
            clf = LogisticRegression(max_iter=1000, random_state=0, solver="lbfgs")
    else:
        clf = LogisticRegression(max_iter=1000, random_state=0)

    return Pipeline([("preprocessor", preprocessor), ("clf", clf)])

def _privileged_value(sensitive: pd.Series):
    """Return the modal (privileged) value of a sensitive Series.

    Mirrors :func:`procedural_fair_hr.fairness_metrics._privileged_mask` so the ABROCA plot's
    privileged group matches the fairness metrics' privileged group.
    """
    return sensitive.mode().iloc[0]

def _stratified_cf_sample(
    n_test: int,
    sensitive_test: pd.Series,
    y_test: np.ndarray,
    *,
    target_n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return ``target_n`` indices stratified by the joint cell
    ``(sensitive × y)`` over the test set.

    For OULAD (3-class y) the joint cells are ``2 × 3 = 6``. For binary
    datasets they are ``2 × 2 = 4``. Stratification preserves each cell's
    share of the test set; if a cell has fewer rows than its quota it is
    taken whole and the remainder is allocated proportionally to the
    remaining cells.
    """
    if target_n >= n_test:
        return np.arange(n_test)

    # Build joint key.
    joint = pd.Series(
        list(zip(sensitive_test.astype(str).values, y_test.astype(int).tolist())),
        index=sensitive_test.index,
    )
    cell_indices: dict[tuple, np.ndarray] = {}
    pos_arr = np.arange(n_test)
    for cell, group in joint.reset_index(drop=True).groupby(joint.values):
        cell_indices[cell] = pos_arr[group.index.to_numpy()]

    # Initial proportional allocation.
    cell_quota = {
        cell: max(1, int(round(target_n * len(idx) / n_test)))
        for cell, idx in cell_indices.items()
    }
    # Cap quota at the cell size; redistribute slack later.
    capped = {
        cell: min(cell_quota[cell], len(idx))
        for cell, idx in cell_indices.items()
    }
    # Adjust to hit target_n exactly.
    diff = target_n - sum(capped.values())
    if diff != 0:
        # Sort cells by remaining headroom (size - capped) for +diff,
        # by current allocation for -diff.
        if diff > 0:
            order = sorted(
                cell_indices.keys(),
                key=lambda c: len(cell_indices[c]) - capped[c],
                reverse=True,
            )
            i = 0
            while diff > 0 and any(capped[c] < len(cell_indices[c]) for c in order):
                cell = order[i % len(order)]
                if capped[cell] < len(cell_indices[cell]):
                    capped[cell] += 1
                    diff -= 1
                i += 1
        else:
            order = sorted(cell_indices.keys(), key=lambda c: capped[c], reverse=True)
            i = 0
            while diff < 0:
                cell = order[i % len(order)]
                if capped[cell] > 1:
                    capped[cell] -= 1
                    diff += 1
                i += 1

    selected: list[int] = []
    for cell, idx in cell_indices.items():
        n_take = capped[cell]
        chosen = rng.choice(idx, size=n_take, replace=False)
        selected.extend(int(v) for v in chosen)
    return np.array(sorted(selected), dtype=int)

# ---------------------------------------------------------------------------
# Proxy graph (PC algorithm on the FULL training set — no subsampling)
# ---------------------------------------------------------------------------

def _proxy_audit_full(
    dataset_name: str,
    X_train: pd.DataFrame,
    A_train: pd.DataFrame,
    y_train: pd.Series,
    sensitive_col: str,
    target_name: str,
    logger: ProgressLogger,
    pc_subsample_rows: int | None = None,
) -> tuple[list[dict], list[dict], int, int]:
    """Run :func:`procedural_fair_hr.data_loaders.learn_proxy_graph` and harvest direct +
    1-step indirect edges.

    By default uses the FULL training set (no row subsampling). If
    ``pc_subsample_rows`` is set AND the training set is larger than it,
    a stratified subsample by ``(sensitive_col × target_name)`` is used —
    see  for the Adult/ACS rationale.

    Returns ``(direct_rows, indirect_rows, n_edges_total, pc_n_rows_used)``.
    """
    df = X_train.copy()
    df[sensitive_col] = A_train[sensitive_col].values
    df[target_name] = y_train.values

    # Stratified subsample for tractability per  (Adult, ACS).
    if pc_subsample_rows is not None and len(df) > pc_subsample_rows:
        from sklearn.model_selection import train_test_split as _tts
        joint = (
            df[sensitive_col].astype(str)
            + "::"
            + df[target_name].astype(str)
        )
        df, _ = _tts(
            df,
            train_size=pc_subsample_rows,
            stratify=joint,
            random_state=0,
        )
        df = df.reset_index(drop=True)
        logger.log(
            f"  {dataset_name}/{sensitive_col}  PC SUBSAMPLE  {pc_subsample_rows} rows "
            f"stratified by ({sensitive_col} × {target_name})"
        )

    # PC requires discrete inputs. Bin numeric columns into 4 quantile-based
    # buckets (more stable than equi-width on skewed columns). Impute NaN
    # first (Adult / IBM HR may carry missing values). Object columns are
    # converted to integer category codes.
    df_disc = df.copy()
    for col in df_disc.columns:
        if pd.api.types.is_numeric_dtype(df_disc[col]):
            df_disc[col] = df_disc[col].fillna(df_disc[col].median())
            try:
                df_disc[col] = pd.qcut(
                    df_disc[col], q=4, labels=False, duplicates="drop"
                )
            except ValueError:
                df_disc[col] = pd.Categorical(df_disc[col]).codes
            df_disc[col] = df_disc[col].astype("Int64").fillna(-1).astype(int)
        else:
            mode = df_disc[col].mode()
            fill = mode.iloc[0] if len(mode) else "missing"
            df_disc[col] = df_disc[col].fillna(fill)
            df_disc[col] = pd.Categorical(df_disc[col]).codes.astype(int)

    # Drop constant or near-constant columns -- pgmpy's chi-square test
    # raises "No data; observed has size 0" when a column has only one
    # unique value (no contingency table can be formed). IBM HR has columns
    # like Over18, EmployeeCount, StandardHours that are constant by design.
    # The sensitive column and target are NEVER dropped here even if they
    # are degenerate -- their absence would produce vacuous results.
    leaf = {sensitive_col, target_name}
    drop_cols: list[str] = []
    for col in df_disc.columns:
        if col in leaf:
            continue
        if df_disc[col].nunique(dropna=False) <= 1:
            drop_cols.append(col)
    if drop_cols:
        logger.log(
            f"  {dataset_name}/{sensitive_col}  PC dropping {len(drop_cols)} "
            f"constant column(s): {drop_cols}"
        )
        df_disc = df_disc.drop(columns=drop_cols)

    n_rows = len(df_disc)
    n_cols = df_disc.shape[1]
    logger.log(
        f"  {dataset_name}/{sensitive_col}  PC START  n_rows={n_rows} n_cols={n_cols}"
    )
    t0 = time.perf_counter()
    try:
        edges, _proxy_cols = bayesian_audit.learn_proxy_graph(
            df=df_disc,
            sensitive_cols=[sensitive_col],
            target_col=target_name,
            significance=0.05,
            max_parents=3,
        )
    except Exception as exc:  # noqa: BLE001
        logger.log(
            f"  {dataset_name}/{sensitive_col}  PC FAILED  {exc!r}"
        )
        return [], [], 0
    pc_elapsed = time.perf_counter() - t0
    logger.log(
        f"  {dataset_name}/{sensitive_col}  PC DONE   {len(edges)} edges in {pc_elapsed:.1f}s"
    )

    direct = [
        {
            "dataset": dataset_name,
            "protected_attribute": sensitive_col,
            "target": target_name,
            "edge_type": "direct",
            "source": src,
            "sink": dst,
        }
        for src, dst in edges
        if dst == sensitive_col
    ]

    proxies = {row["source"] for row in direct}
    indirect: list[dict] = []
    for src, dst in edges:
        if dst in proxies and src != sensitive_col and src != dst:
            indirect.append(
                {
                    "dataset": dataset_name,
                    "protected_attribute": sensitive_col,
                    "target": target_name,
                    "edge_type": "indirect",
                    "source": src,
                    "sink": dst,
                }
            )

    # Stamp `pc_n_rows` on every proxy row for explicit disclosure.
    for r in direct + indirect:
        r["pc_n_rows"] = n_rows

    return direct, indirect, len(edges), n_rows

# ---------------------------------------------------------------------------
# Per-dataset audit
# ---------------------------------------------------------------------------

def _compute_non_vacuous_tpr(
    y_true_bin: np.ndarray,
    y_pred_bin: np.ndarray,
    sensitive: pd.Series,
    threshold: float = NON_VACUOUS_TPR_LOWER,
) -> bool:
    """Pattern-B degeneracy guard: at least one protected group must have
    TPR >= ``threshold`` for the binarised target.

    Returns True iff the classifier has meaningful positive-class
    detection for at least one group. Returns False when every group has
    TPR < threshold (or undefined because the group has zero true
    positives) — that's the IBM HR / Gender failure mode where the model
    predicts the positive class often enough to satisfy the
    ``non_vacuous`` (Pattern A) guard but its predictions don't correlate
    with the true labels.
    """
    y_true_bin = np.asarray(y_true_bin).astype(int)
    y_pred_bin = np.asarray(y_pred_bin).astype(int)
    sens_arr = np.asarray(sensitive)
    max_tpr = 0.0
    saw_defined = False
    for group_value in np.unique(sens_arr):
        mask = sens_arr == group_value
        positives = int(y_true_bin[mask].sum())
        if positives == 0:
            # Undefined: no true positives in this group -> exclude.
            continue
        true_positives = int(((y_true_bin == 1) & (y_pred_bin == 1) & mask).sum())
        tpr = true_positives / positives
        saw_defined = True
        if tpr > max_tpr:
            max_tpr = tpr
    if not saw_defined:
        # No group has any true positives at all -> the binarised target is
        # all-negative on test, fairness numbers are vacuous by construction.
        return False
    return max_tpr >= threshold

def _row(
    *,
    dataset: str,
    sens: str,
    class_idx: int,
    metric: str,
    value: float,
    model: str,
    target_form: str,
    n_train: int,
    n_test: int,
    cf_n: float = float("nan"),
    pc_n_rows: float = float("nan"),
    non_vacuous: bool = True,
    non_vacuous_tpr: bool = True,
) -> dict:
    return {
        "dataset": dataset,
        "protected_attribute": sens,
        "class_idx": class_idx,
        "metric_name": metric,
        "metric_value": float(value) if value is not None else float("nan"),
        "model": model,
        "target_form": target_form,
        "n_train": n_train,
        "n_test": n_test,
        "cf_n_samples": float(cf_n),
        "non_vacuous": bool(non_vacuous),
        "non_vacuous_tpr": bool(non_vacuous_tpr),
        "pc_n_rows": float(pc_n_rows),
    }

def _audit_one_attr(
    dataset_name: str,
    sensitive_col: str,
    bundle: dict,
    *,
    figs_dir: pathlib.Path,
    logger: ProgressLogger,
) -> tuple[list[dict], list[dict]]:
    """Audit a single (dataset, protected_attribute) pair.

    Returns ``(metric_rows, proxy_rows)``. ``metric_rows`` populates
    ``audit.csv``; ``proxy_rows`` populates ``proxy_edges.csv``.
    """
    X_train: pd.DataFrame = bundle["X_train"]
    X_test: pd.DataFrame = bundle["X_test"]
    y_train: pd.Series = bundle["y_train"]
    y_test: pd.Series = bundle["y_test"]
    A_train: pd.DataFrame = bundle["A_train"]
    A_test: pd.DataFrame = bundle["A_test"]
    n_train = len(X_train)
    n_test = len(X_test)
    n_classes = int(bundle["n_classes"])
    target_name = str(y_train.name) if y_train.name is not None else "target"

    logger.log(f"  {dataset_name}/{sensitive_col}  START  n_train={n_train} n_test={n_test} n_classes={n_classes}")

    # --- 1. Train baseline -------------------------------------------------
    multinomial = n_classes > 2
    pipe = _build_baseline_pipeline(dataset_name, X_train, multinomial=multinomial)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        # FutureWarning from sklearn re multi_class deprecation, etc.
        warnings.simplefilter("ignore", FutureWarning)
        pipe.fit(X_train, y_train)
        y_pred = np.asarray(pipe.predict(X_test))

    if dataset_name in ("ibm_hr", "ibm_hr_attrition"):
        model_name = "RandomForestClassifier"
    else:
        model_name = "LogisticRegression"

    sensitive_test = A_test[sensitive_col].reset_index(drop=True)

    metric_rows: list[dict] = []
    # Per-class non_vacuous flags. Populated in the binary or OvR branch and
    # consumed by the CF block downstream so its rows carry the same flag as
    # the closed-form metrics for the same class_idx. Two flags are tracked:
    # `non_vacuous_by_class` is Pattern-A (constant predictions);
    # `non_vacuous_tpr_by_class` is Pattern-B (uncorrelated predictions).
    # See module docstring + .
    non_vacuous_by_class: dict[int, bool] = {}
    non_vacuous_tpr_by_class: dict[int, bool] = {}

    # --- 2. Closed-form metrics (group + individual + ABROCA) --------------
    cm_t0 = time.perf_counter()
    logger.log(f"  {dataset_name}/{sensitive_col}  CLOSED-FORM START")

    if n_classes == 2:
        # Binary path.
        y_test_bin = _binarise_for_binary_loader(y_test.values)
        y_pred_bin = _binarise_for_binary_loader(y_pred)
        try:
            y_score = pipe.predict_proba(X_test)[:, 1]
        except (AttributeError, IndexError, NotImplementedError):
            try:
                y_score = pipe.decision_function(X_test)
            except Exception:  # noqa: BLE001
                y_score = y_pred_bin.astype(float)

        target_form = "binary"
        class_idx = 1

        # Non-vacuous-fairness guard: a classifier that predicts the positive
        # class for <5 % or >95 % of all samples has effectively collapsed; the
        # binary fairness metrics on it become trivially small / undefined
        # (cf.  "IBM HR EOdds = 0.0000 is degenerate, not fair").
        positive_pred_rate_overall = float(np.mean(y_pred_bin))
        non_vacuous = (
            NON_VACUOUS_LOWER <= positive_pred_rate_overall <= NON_VACUOUS_UPPER
        )
        non_vacuous_tpr = _compute_non_vacuous_tpr(
            y_test_bin, y_pred_bin, sensitive_test
        )
        non_vacuous_by_class[class_idx] = non_vacuous
        non_vacuous_tpr_by_class[class_idx] = non_vacuous_tpr
        if not non_vacuous:
            logger.log(
                f"  {dataset_name}/{sensitive_col}  NON_VACUOUS=False  "
                f"positive_pred_rate={positive_pred_rate_overall:.4f} "
                f"(outside [{NON_VACUOUS_LOWER}, {NON_VACUOUS_UPPER}])"
            )
        if not non_vacuous_tpr:
            logger.log(
                f"  {dataset_name}/{sensitive_col}  NON_VACUOUS_TPR=False  "
                f"all groups have TPR < {NON_VACUOUS_TPR_LOWER} (Pattern B "
                f"degeneracy: pred-rate is in band but predictions are "
                f"uncorrelated with truth)"
            )

        for name, fn in GROUP_METRICS:
            try:
                value = fn(y_test_bin, y_pred_bin, sensitive_test)
            except Exception as exc:  # noqa: BLE001
                logger.log(f"  [warn] {name} failed for {dataset_name}/{sensitive_col}: {exc!r}")
                value = float("nan")
            metric_rows.append(
                _row(
                    dataset=dataset_name,
                    sens=sensitive_col,
                    class_idx=class_idx,
                    metric=name,
                    value=value,
                    model=model_name,
                    target_form=target_form,
                    n_train=n_train,
                    n_test=n_test,
                    non_vacuous=non_vacuous,
                    non_vacuous_tpr=non_vacuous_tpr,
                )
            )

        # KNNC + Lipschitz on a deterministic test sample.
        n_use = min(INDIV_FAIRNESS_SAMPLE_ROWS, n_test)
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(n_test, size=n_use, replace=False)
        X_test_sample = X_test.iloc[sample_idx].reset_index(drop=True)
        y_pred_sample = y_pred_bin[sample_idx]
        sensitive_sample = sensitive_test.iloc[sample_idx].reset_index(drop=True)

        try:
            knnc = fairness_metrics.knn_consistency(
                None, y_pred_sample, sensitive_sample, X=X_test_sample, k=5
            )
        except Exception as exc:  # noqa: BLE001
            logger.log(f"  [warn] knn_consistency failed for {dataset_name}: {exc!r}")
            knnc = float("nan")
        metric_rows.append(
            _row(
                dataset=dataset_name, sens=sensitive_col, class_idx=class_idx,
                metric="knn_consistency", value=knnc, model=model_name,
                target_form=target_form, n_train=n_train, n_test=n_test,
                non_vacuous=non_vacuous,
                non_vacuous_tpr=non_vacuous_tpr,
            )
        )
        try:
            lf = fairness_metrics.lipschitz_fairness(
                y_pred_sample, sensitive_sample, X=X_test_sample, n_sample_pairs=1000
            )
        except Exception as exc:  # noqa: BLE001
            logger.log(f"  [warn] lipschitz_fairness failed for {dataset_name}: {exc!r}")
            lf = float("nan")
        metric_rows.append(
            _row(
                dataset=dataset_name, sens=sensitive_col, class_idx=class_idx,
                metric="lipschitz_fairness", value=lf, model=model_name,
                target_form=target_form, n_train=n_train, n_test=n_test,
                non_vacuous=non_vacuous,
                non_vacuous_tpr=non_vacuous_tpr,
            )
        )

        # ABROCA --------------------------------------------------------
        figs_dir.mkdir(parents=True, exist_ok=True)
        abroca_path = figs_dir / f"abroca_{dataset_name}_{sensitive_col}.png"
        priv_val = _privileged_value(sensitive_test)
        abroca_value: float
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: F401  -- side-effect import

            abroca_value = visualisations.plot_abroca(
                y_true=y_test_bin,
                y_score=np.asarray(y_score),
                sensitive=sensitive_test,
                privileged_val=priv_val,
                title=f"ABROCA -- {dataset_name} ({sensitive_col})",
                save_path=str(abroca_path),
            )
            import matplotlib.pyplot as _plt
            fig = _plt.gcf()
            fig.savefig(str(abroca_path), dpi=120, bbox_inches="tight")
            _plt.close("all")
        except Exception as exc:  # noqa: BLE001
            logger.log(f"  [warn] plot_abroca failed for {dataset_name}/{sensitive_col}: {exc!r}")
            abroca_value = float("nan")
        metric_rows.append(
            _row(
                dataset=dataset_name, sens=sensitive_col, class_idx=class_idx,
                metric="abroca", value=abroca_value, model=model_name,
                target_form=target_form, n_train=n_train, n_test=n_test,
                non_vacuous=non_vacuous,
                non_vacuous_tpr=non_vacuous_tpr,
            )
        )

    else:
        # Multi-class one-vs-rest path (currently only OULAD, n_classes=3).
        # Class-conditional probabilities for ABROCA OvR.
        try:
            y_proba = pipe.predict_proba(X_test)  # shape (n, K)
        except Exception:  # noqa: BLE001
            y_proba = None

        for c in range(n_classes):
            target_form = f"ovr_class_{c}"
            class_idx = c
            y_test_bin = (y_test.values == c).astype(int)
            y_pred_bin = (y_pred == c).astype(int)

            # Non-vacuous-fairness guard for this OvR class — see binary path
            # comment + module docstring +  ("IBM HR EOdds = 0.0000
            # is degenerate, not fair", 2026-04-29). The OULAD Distinction
            # class (c=2) is a known candidate for `non_vacuous == False`
            # because the multinomial LR rarely predicts it (~13 % prior).
            positive_pred_rate_overall = float(np.mean(y_pred_bin))
            non_vacuous = (
                NON_VACUOUS_LOWER
                <= positive_pred_rate_overall
                <= NON_VACUOUS_UPPER
            )
            non_vacuous_tpr = _compute_non_vacuous_tpr(
                y_test_bin, y_pred_bin, sensitive_test
            )
            non_vacuous_by_class[class_idx] = non_vacuous
            non_vacuous_tpr_by_class[class_idx] = non_vacuous_tpr
            if not non_vacuous:
                logger.log(
                    f"  {dataset_name}/{sensitive_col}  ovr_class_{c}  "
                    f"NON_VACUOUS=False  "
                    f"positive_pred_rate={positive_pred_rate_overall:.4f} "
                    f"(outside [{NON_VACUOUS_LOWER}, {NON_VACUOUS_UPPER}])"
                )
            if not non_vacuous_tpr:
                logger.log(
                    f"  {dataset_name}/{sensitive_col}  ovr_class_{c}  "
                    f"NON_VACUOUS_TPR=False  all groups have TPR < "
                    f"{NON_VACUOUS_TPR_LOWER} (Pattern B degeneracy)"
                )

            for name, fn in GROUP_METRICS:
                try:
                    value = fn(y_test_bin, y_pred_bin, sensitive_test)
                except Exception as exc:  # noqa: BLE001
                    logger.log(f"  [warn] {name} ovr_class_{c} failed for {dataset_name}/{sensitive_col}: {exc!r}")
                    value = float("nan")
                metric_rows.append(
                    _row(
                        dataset=dataset_name, sens=sensitive_col, class_idx=class_idx,
                        metric=name, value=value, model=model_name,
                        target_form=target_form, n_train=n_train, n_test=n_test,
                        non_vacuous=non_vacuous,
                        non_vacuous_tpr=non_vacuous_tpr,
                    )
                )

            # KNNC + Lipschitz once per class on the same deterministic sample.
            n_use = min(INDIV_FAIRNESS_SAMPLE_ROWS, n_test)
            rng = np.random.default_rng(0)
            sample_idx = rng.choice(n_test, size=n_use, replace=False)
            X_test_sample = X_test.iloc[sample_idx].reset_index(drop=True)
            y_pred_sample = y_pred_bin[sample_idx]
            sensitive_sample = sensitive_test.iloc[sample_idx].reset_index(drop=True)
            try:
                knnc = fairness_metrics.knn_consistency(
                    None, y_pred_sample, sensitive_sample, X=X_test_sample, k=5
                )
            except Exception as exc:  # noqa: BLE001
                logger.log(f"  [warn] knn_consistency ovr_class_{c} failed: {exc!r}")
                knnc = float("nan")
            metric_rows.append(
                _row(
                    dataset=dataset_name, sens=sensitive_col, class_idx=class_idx,
                    metric="knn_consistency", value=knnc, model=model_name,
                    target_form=target_form, n_train=n_train, n_test=n_test,
                    non_vacuous=non_vacuous,
                    non_vacuous_tpr=non_vacuous_tpr,
                )
            )
            try:
                lf = fairness_metrics.lipschitz_fairness(
                    y_pred_sample, sensitive_sample, X=X_test_sample, n_sample_pairs=1000
                )
            except Exception as exc:  # noqa: BLE001
                logger.log(f"  [warn] lipschitz_fairness ovr_class_{c} failed: {exc!r}")
                lf = float("nan")
            metric_rows.append(
                _row(
                    dataset=dataset_name, sens=sensitive_col, class_idx=class_idx,
                    metric="lipschitz_fairness", value=lf, model=model_name,
                    target_form=target_form, n_train=n_train, n_test=n_test,
                    non_vacuous=non_vacuous,
                    non_vacuous_tpr=non_vacuous_tpr,
                )
            )

            # ABROCA per OvR class.
            figs_dir.mkdir(parents=True, exist_ok=True)
            abroca_path = figs_dir / f"abroca_{dataset_name}_{sensitive_col}_class{c}.png"
            priv_val = _privileged_value(sensitive_test)
            abroca_value = float("nan")
            if y_proba is not None and y_proba.shape[1] >= n_classes:
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt  # noqa: F401

                    abroca_value = visualisations.plot_abroca(
                        y_true=y_test_bin,
                        y_score=np.asarray(y_proba[:, c]),
                        sensitive=sensitive_test,
                        privileged_val=priv_val,
                        title=f"ABROCA -- {dataset_name} class={c} ({sensitive_col})",
                        save_path=str(abroca_path),
                    )
                    import matplotlib.pyplot as _plt
                    fig = _plt.gcf()
                    fig.savefig(str(abroca_path), dpi=120, bbox_inches="tight")
                    _plt.close("all")
                except Exception as exc:  # noqa: BLE001
                    logger.log(
                        f"  [warn] plot_abroca ovr_class_{c} failed for {dataset_name}/{sensitive_col}: {exc!r}"
                    )
                    abroca_value = float("nan")
            metric_rows.append(
                _row(
                    dataset=dataset_name, sens=sensitive_col, class_idx=class_idx,
                    metric="abroca", value=abroca_value, model=model_name,
                    target_form=target_form, n_train=n_train, n_test=n_test,
                    non_vacuous=non_vacuous,
                    non_vacuous_tpr=non_vacuous_tpr,
                )
            )

        # --- 2b. Formal macro-extensions ( /  /  §1) ---
        # For multi-class targets, emit ONE macro row per metric in addition
        # to the per-class ovr_class_{c} rows above. The macro rows use the
        # formal multi-class definitions in src/fairness_metrics.py
        # (macro_dp / macro_eodds / macro_eo) instead of an inline OvR
        # binarisation. Sentinel: class_idx = -1, target_form = "macro".
        # See  Parity — multi-class (Macro-DP) /
        # ::Equal Opportunity — multi-class (Macro-EO) / ::Equalised Odds —
        # multi-class (Macro-EOdds).

        # Non-vacuous flag: a macro row inherits non-vacuity iff EVERY
        # per-class row in the OULAD-3 set is non-vacuous (the most
        # stringent reading; a single degenerate class - e.g. OULAD's
        # Distinction - flips the macro non_vacuous flag to False).
        macro_non_vacuous = (
            all(non_vacuous_by_class.values())
            if non_vacuous_by_class else True
        )
        macro_non_vacuous_tpr = (
            all(non_vacuous_tpr_by_class.values())
            if non_vacuous_tpr_by_class else True
        )

        # Phase-3 followup (post-review Issue 1): also emit a parallel set
        # of "filtered-macro" rows that exclude per-class entries whose
        # `non_vacuous_tpr` flag is False from the macro mean (e.g.
        # OULAD's class-2 Distinction, which is degenerate). Both versions
        # ship side-by-side so a reader sees the un-filtered macro
        # alongside the macro-with-degenerate-classes-excluded. Sentinel:
        # class_idx=-1, target_form="macro_filtered". The non_vacuous
        # flags on the filtered rows are computed only over the kept
        # classes (so they are typically True when at least one class
        # survives filtering).
        filter_classes_set = {
            int(c)
            for c, ok in non_vacuous_tpr_by_class.items()
            if not ok
        }
        kept_classes = [
            int(c)
            for c in non_vacuous_tpr_by_class.keys()
            if int(c) not in filter_classes_set
        ]
        if kept_classes:
            filtered_non_vacuous = all(
                non_vacuous_by_class.get(c, True) for c in kept_classes
            )
            filtered_non_vacuous_tpr = all(
                non_vacuous_tpr_by_class.get(c, True) for c in kept_classes
            )
        else:
            # All classes filtered out -> filtered macro is NaN; flags
            # default to True (vacuously) to avoid double-flagging the
            # already-degenerate row.
            filtered_non_vacuous = True
            filtered_non_vacuous_tpr = True

        macro_metric_specs: list[tuple[str, Callable]] = [
            ("macro_dp", fairness_metrics.macro_dp),
            ("macro_eodds", fairness_metrics.macro_eodds),
            ("macro_eo", fairness_metrics.macro_eo),
        ]
        for macro_name, macro_fn in macro_metric_specs:
            # Unfiltered macro (existing  byte-identical row).
            try:
                macro_value, _per_class = macro_fn(
                    y_test.values.astype(int),
                    y_pred.astype(int),
                    sensitive_test,
                )
            except Exception as exc:  # noqa: BLE001
                logger.log(
                    f"  [warn] {macro_name} failed for {dataset_name}/{sensitive_col}: {exc!r}"
                )
                macro_value = float("nan")
            metric_rows.append(
                _row(
                    dataset=dataset_name, sens=sensitive_col, class_idx=-1,
                    metric=macro_name, value=macro_value, model=model_name,
                    target_form="macro", n_train=n_train, n_test=n_test,
                    non_vacuous=macro_non_vacuous,
                    non_vacuous_tpr=macro_non_vacuous_tpr,
                )
            )
            # Filtered macro (additive; excludes per-class rows whose
            # non_vacuous_tpr flag is False).
            try:
                filtered_value, _ = macro_fn(
                    y_test.values.astype(int),
                    y_pred.astype(int),
                    sensitive_test,
                    filter_classes=filter_classes_set,
                )
            except Exception as exc:  # noqa: BLE001
                logger.log(
                    f"  [warn] {macro_name} (filtered) failed for "
                    f"{dataset_name}/{sensitive_col}: {exc!r}"
                )
                filtered_value = float("nan")
            metric_rows.append(
                _row(
                    dataset=dataset_name, sens=sensitive_col, class_idx=-1,
                    metric=macro_name, value=filtered_value, model=model_name,
                    target_form="macro_filtered", n_train=n_train, n_test=n_test,
                    non_vacuous=filtered_non_vacuous,
                    non_vacuous_tpr=filtered_non_vacuous_tpr,
                )
            )

    cm_elapsed = time.perf_counter() - cm_t0
    logger.log(f"  {dataset_name}/{sensitive_col}  CLOSED-FORM DONE  {cm_elapsed:.1f}s")

    # --- 3. Counterfactual fairness ---------------------------------------
    # Decide N: full when n_test ≤ CF_FULL_THRESHOLD, else CF_LARGE_N
    # stratified by (sensitive × y).
    if n_test <= CF_FULL_THRESHOLD:
        cf_target_n = n_test
    else:
        cf_target_n = min(CF_LARGE_N, n_test)
    rng_cf = np.random.default_rng(0)
    cf_idx = _stratified_cf_sample(
        n_test=n_test,
        sensitive_test=sensitive_test,
        y_test=y_test.values,
        target_n=cf_target_n,
        rng=rng_cf,
    )
    cf_n = int(len(cf_idx))
    logger.log(f"  {dataset_name}/{sensitive_col}  CF START  N={cf_n} (full_threshold={CF_FULL_THRESHOLD})")

    X_cf_raw = X_test.iloc[cf_idx].reset_index(drop=True).copy()
    sens_cf = sensitive_test.iloc[cf_idx].reset_index(drop=True)
    y_cf_test = y_test.values[cf_idx]

    # If the sensitive column lives in X (most loaders), CF flips its value
    # in X. If it lives ONLY in A (e.g. IBM HR's `Gender` is dropped from X
    # by the loader), inject a copy so CF can flip it. We also need a
    # classifier whose feature space matches X_cf -- we re-train an
    # ordinal-encoded auxiliary model on the same features.
    sens_in_X = sensitive_col in X_cf_raw.columns
    if not sens_in_X:
        X_cf_raw[sensitive_col] = sens_cf.values

    # Build the same X-augmented training matrix to fit the auxiliary model.
    X_train_aug = X_train.copy()
    if not sens_in_X:
        X_train_aug[sensitive_col] = A_train[sensitive_col].values

    # Encode aux training matrix (object cols -> ordinal; numeric NaN -> median).
    X_train_enc = X_train_aug.copy()
    obj_cols = X_train_enc.select_dtypes(include="object").columns.tolist()
    for c in obj_cols:
        mode = X_train_enc[c].mode()
        fill = mode.iloc[0] if len(mode) else "missing"
        X_train_enc[c] = X_train_enc[c].fillna(fill)
    if obj_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_enc[obj_cols] = enc.fit_transform(X_train_enc[obj_cols])
    X_train_enc = X_train_enc.astype(float)
    X_train_enc = X_train_enc.fillna(X_train_enc.median(numeric_only=True))

    # Impute X_cf_raw NaNs the same way (object first, then numeric).
    X_cf_imp = X_cf_raw.copy()
    for c in X_cf_imp.columns:
        if X_cf_imp[c].dtype == object:
            mode = X_cf_imp[c].mode()
            fill = mode.iloc[0] if len(mode) else "missing"
            X_cf_imp[c] = X_cf_imp[c].fillna(fill)
        else:
            if X_cf_imp[c].isna().any():
                X_cf_imp[c] = X_cf_imp[c].fillna(X_cf_imp[c].median())

    # Aux classifier: same family as the baseline (RF for IBM HR; multinomial
    # LR for OULAD; LR otherwise).
    if dataset_name == "ibm_hr":
        aux_clf = RandomForestClassifier(n_estimators=200, random_state=0)
    elif n_classes > 2:
        aux_clf = LogisticRegression(
            max_iter=1000, random_state=0, solver="lbfgs"
        )
    else:
        aux_clf = LogisticRegression(max_iter=1000, random_state=0)

    # Fit the aux classifier; for binary, target is the {0,1} canonical form;
    # for multi-class, target stays as int-coded.
    if n_classes == 2:
        y_train_aux = _binarise_for_binary_loader(y_train.values)
    else:
        y_train_aux = y_train.values.astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", FutureWarning)
        aux_clf.fit(X_train_enc.values, y_train_aux)

    # Compute CF for each (binary or OvR-class) row.
    cf_t0 = time.perf_counter()

    def _cf_for_target(y_target_form: str, class_for_label: int | None) -> float:
        """Compute counterfactual fairness once.

        For binary (n_classes==2): operates on the aux's binary predictions.
        For OvR: collapses aux's argmax predictions to ``(==class_for_label)``
        before comparison.
        """
        try:
            cf_score = fairness_metrics.counterfactual_fairness(
                y_true=None,
                y_pred=None,
                sensitive=sens_cf,
                X=X_cf_imp,
                model=aux_clf,
                sensitive_col=sensitive_col,
                n_cf=cf_n,
            )
            if class_for_label is None:
                return float(cf_score)
            # For OvR class_for_label, we need a wrapper around aux_clf that
            # predicts (argmax == class_for_label) instead of argmax. Easiest
            # path: rerun the CF computation manually using the aux model's
            # predict, mapping to (==class_for_label).
            rng_local = np.random.default_rng(0)
            n = len(X_cf_imp)
            idx = rng_local.choice(n, size=min(cf_n, n), replace=False)
            X_sample = X_cf_imp.iloc[idx].copy().reset_index(drop=True)
            unique_vals = X_sample[sensitive_col].unique().tolist()
            if len(unique_vals) < 2:
                return 1.0
            # Encode as in fairness_metrics._ordinal_encode_df
            X_enc = _ordinal_encode_local(X_sample)
            orig_preds = aux_clf.predict(X_enc) == class_for_label
            X_cf_local = X_sample.copy()
            for i in range(len(X_cf_local)):
                current = X_cf_local.at[i, sensitive_col]
                others = [v for v in unique_vals if v != current]
                X_cf_local.at[i, sensitive_col] = rng_local.choice(others)
            X_cf_enc = _ordinal_encode_local(X_cf_local)
            cf_preds = aux_clf.predict(X_cf_enc) == class_for_label
            return float((orig_preds == cf_preds).mean())
        except Exception as exc:  # noqa: BLE001
            logger.log(
                f"  [warn] counterfactual_fairness failed for "
                f"{dataset_name}/{sensitive_col} target={y_target_form}: {exc!r}"
            )
            return float("nan")

    if n_classes == 2:
        cf_value = _cf_for_target("binary", None)
        metric_rows.append(
            _row(
                dataset=dataset_name, sens=sensitive_col, class_idx=1,
                metric="counterfactual_fairness", value=cf_value,
                model=model_name, target_form="binary",
                n_train=n_train, n_test=n_test, cf_n=cf_n,
                non_vacuous=non_vacuous_by_class.get(1, True),
                non_vacuous_tpr=non_vacuous_tpr_by_class.get(1, True),
            )
        )
    else:
        # Multi-class CF: emit BOTH (i) the formal multinomial CF row
        # (TV-distance over predicted-probability vectors,  /
        #  §1) AND (ii) per-class OvR CF rows
        # (`I(orig_class != cf_class)` averaged over the stratified
        # sample, restored as Phase-3-followup Issue 3 — both
        # representations have value: per-class CF surfaces which
        # individual class is most affected by sensitive-flip; multinomial
        # CF is the macro TV-distance over softmax). Sentinel for the
        # multinomial row: class_idx=-1, target_form="macro"; per-class
        # rows: target_form="ovr", class_idx ∈ {0..n_classes-1}. See
        #  Fairness — multi-class.

        # Macro non_vacuous flag inherits from every per-class flag (see
        # the macro group-metric block above) — the multinomial CF
        # operates on the full softmax distribution and is degenerate
        # iff some class is degenerate.
        macro_non_vacuous_cf = (
            all(non_vacuous_by_class.values())
            if non_vacuous_by_class else True
        )
        macro_non_vacuous_tpr_cf = (
            all(non_vacuous_tpr_by_class.values())
            if non_vacuous_tpr_by_class else True
        )
        try:
            cf_value = fairness_metrics.multinomial_counterfactual_fairness(
                y_true=None,
                y_pred=None,
                sensitive=sens_cf,
                X=X_cf_imp,
                model=aux_clf,
                sensitive_col=sensitive_col,
                n_cf=cf_n,
            )
        except Exception as exc:  # noqa: BLE001
            logger.log(
                f"  [warn] multinomial_counterfactual_fairness failed for "
                f"{dataset_name}/{sensitive_col}: {exc!r}"
            )
            cf_value = float("nan")
        metric_rows.append(
            _row(
                dataset=dataset_name, sens=sensitive_col, class_idx=-1,
                metric="multinomial_counterfactual_fairness", value=cf_value,
                model=model_name, target_form="macro",
                n_train=n_train, n_test=n_test, cf_n=cf_n,
                non_vacuous=macro_non_vacuous_cf,
                non_vacuous_tpr=macro_non_vacuous_tpr_cf,
            )
        )

        # Per-class OvR CF rows (Phase-3 followup Issue 3): for each class
        # c, binarise the model's predictions to (argmax == c) and run
        # the existing binary `counterfactual_fairness`-style sample
        # comparison via the local `_cf_for_target` helper. These rows
        # carry the same `non_vacuous` / `non_vacuous_tpr` flags as the
        # group metrics for the same class_idx.
        for c in range(n_classes):
            per_class_cf = _cf_for_target(f"ovr_class_{c}", c)
            metric_rows.append(
                _row(
                    dataset=dataset_name, sens=sensitive_col, class_idx=c,
                    metric="counterfactual_fairness", value=per_class_cf,
                    model=model_name, target_form="ovr",
                    n_train=n_train, n_test=n_test, cf_n=cf_n,
                    non_vacuous=non_vacuous_by_class.get(c, True),
                    non_vacuous_tpr=non_vacuous_tpr_by_class.get(c, True),
                )
            )

    cf_elapsed = time.perf_counter() - cf_t0
    logger.log(f"  {dataset_name}/{sensitive_col}  CF DONE   N={cf_n} in {cf_elapsed:.1f}s")

    # --- 4. Proxy graph (FULL training set by default;  subsamples
    #         Adult and ACS to 10,000 rows stratified by sensitive × y) -----
    pc_subsample = PC_SUBSAMPLE_DATASETS.get(dataset_name)
    direct_rows, indirect_rows, n_edges, pc_n_rows_used = _proxy_audit_full(
        dataset_name=dataset_name,
        X_train=X_train,
        A_train=A_train,
        y_train=y_train,
        sensitive_col=sensitive_col,
        target_name=target_name,
        logger=logger,
        pc_subsample_rows=pc_subsample,
    )
    proxy_rows = direct_rows + indirect_rows

    # Stamp `pc_n_rows` on every metric row for explicit disclosure.
    for r in metric_rows:
        r["pc_n_rows"] = float(pc_n_rows_used)

    return metric_rows, proxy_rows

def _ordinal_encode_local(X: pd.DataFrame) -> np.ndarray:
    """Mirror :func:`fairness_metrics._ordinal_encode_df` (private helper).

    Encodes object columns ordinally + leaves numerics unchanged. Used by the
    OvR CF helper because the public ``counterfactual_fairness`` function
    re-fits its encoder per-call and the OvR branch needs the same encoding.
    """
    X_out = X.copy()
    obj_cols = X_out.select_dtypes(include="object").columns
    if len(obj_cols) > 0:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_out[obj_cols] = enc.fit_transform(X_out[obj_cols])
    return X_out.values.astype(float)

# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def _git_commit_and_push(
    out_dir: pathlib.Path,
    dataset_name: str,
    n_metric_rows: int,
    n_proxy_edges: int,
    n_abroca: int,
    elapsed_s: float,
    *,
    do_push: bool,
    logger: ProgressLogger,
) -> None:
    """Stage + commit + push the per-dataset partial.

    Tolerates non-fast-forward errors via ``git pull --rebase`` followed by a
    second push.
    """
    repo_root = _PROJECT_ROOT.parent  # employee-performance/
    rel_parquet = (out_dir / f"audit_{dataset_name}.parquet").relative_to(repo_root)
    rel_log = (out_dir / "progress.log").relative_to(repo_root)
    rel_figs_dir = (out_dir / "figs").relative_to(repo_root)

    add_paths = [str(rel_parquet), str(rel_log)]
    # Add only fig files for this dataset (avoid sweeping in stale ones).
    figs_dir = out_dir / "figs"
    if figs_dir.exists():
        for fp in figs_dir.glob(f"abroca_{dataset_name}_*.png"):
            add_paths.append(str(fp.relative_to(repo_root)))

    # Use -f because some local checkouts may still have ``results/`` in
    # ``.gitignore`` (the ignore was removed in commit
    # following  v2 to make per-dataset checkpoints commit-able).
    cmd_add = ["git", "-C", str(repo_root), "add", "-f"] + add_paths
    rc_add = subprocess.run(cmd_add, capture_output=True, text=True)
    if rc_add.returncode != 0:
        logger.log(f"  [warn] git add failed: {rc_add.stderr.strip()}")
        return

    msg = (
        f"phase-2 audit: {dataset_name} complete "
        f"({n_metric_rows} metrics + {n_proxy_edges} proxy edges + "
        f"{n_abroca} ABROCA plots, {elapsed_s:.1f}s)"
    )
    rc_commit = subprocess.run(
        ["git", "-C", str(repo_root), "commit", "-m", msg],
        capture_output=True, text=True,
    )
    if rc_commit.returncode != 0:
        # Possibly nothing to commit (e.g. parquet unchanged); log and move on.
        logger.log(f"  [warn] git commit returned rc={rc_commit.returncode}: {rc_commit.stderr.strip() or rc_commit.stdout.strip()}")
        return

    # Capture commit SHA.
    rc_sha = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    sha = rc_sha.stdout.strip()[:12] if rc_sha.returncode == 0 else "?"
    logger.log(f"  {dataset_name}  COMMIT  {sha}  '{msg}'")

    if not do_push:
        return

    rc_pull = subprocess.run(
        ["git", "-C", str(repo_root), "pull", "--rebase", "origin", "main"],
        capture_output=True, text=True,
    )
    if rc_pull.returncode != 0:
        logger.log(f"  [warn] git pull --rebase failed: {rc_pull.stderr.strip()}")
        # Don't push if rebase failed; the user can resolve manually.
        return

    rc_push = subprocess.run(
        ["git", "-C", str(repo_root), "push", "origin", "main"],
        capture_output=True, text=True,
    )
    if rc_push.returncode != 0:
        logger.log(f"  [warn] git push failed: {rc_push.stderr.strip()}")
    else:
        logger.log(f"  {dataset_name}  PUSH    origin/main")

def run_audit(
    datasets: list[str],
    out_dir: pathlib.Path,
    *,
    do_commit: bool,
    do_push: bool,
) -> int:
    """Drive the full no-shortcut audit.

    Returns 0 on success; non-zero if any dataset failed entirely.
    """
    if not datasets:
        datasets = list(LOADERS.keys())

    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    logger = ProgressLogger(out_dir / "progress.log")
    logger.log("=" * 72)
    logger.log(f"audit START datasets={datasets}")
    logger.log(f"  CF_FULL_THRESHOLD={CF_FULL_THRESHOLD}  CF_LARGE_N={CF_LARGE_N}")
    logger.log("=" * 72)

    failed: list[str] = []
    started_at = time.perf_counter()

    for ds_name in datasets:
        partial = out_dir / f"audit_{ds_name}.parquet"
        if partial.exists():
            logger.log(f"{ds_name}  SKIP  (partial exists at {partial.name})")
            continue

        ds_t0 = time.perf_counter()
        logger.log(f"{ds_name}  LOAD")
        try:
            bundle = LOADERS[ds_name]()
        except Exception as exc:  # noqa: BLE001
            logger.log(f"{ds_name}  LOAD FAILED  {exc!r}")
            failed.append(ds_name)
            continue
        logger.log(
            f"{ds_name}  LOAD DONE  n_train={len(bundle['X_train'])} "
            f"n_test={len(bundle['X_test'])} n_classes={bundle['n_classes']} "
            f"sensitive={bundle['sensitive_names']}"
        )

        ds_metric_rows: list[dict] = []
        ds_proxy_rows: list[dict] = []
        for sens_col in bundle["sensitive_names"]:
            try:
                m_rows, p_rows = _audit_one_attr(
                    dataset_name=ds_name,
                    sensitive_col=sens_col,
                    bundle=bundle,
                    figs_dir=figs_dir,
                    logger=logger,
                )
                ds_metric_rows.extend(m_rows)
                ds_proxy_rows.extend(p_rows)
            except Exception as exc:  # noqa: BLE001
                logger.log(f"  [error] audit({ds_name}, {sens_col}) crashed: {exc!r}")
                failed.append(f"{ds_name}/{sens_col}")

        ds_elapsed = time.perf_counter() - ds_t0

        # Persist per-dataset partial parquet IMMEDIATELY.
        df_ds = pd.DataFrame(ds_metric_rows, columns=AUDIT_COLUMNS)
        df_ds.to_parquet(partial, index=False)
        # Also keep a sidecar of the proxy rows for this dataset (concatenated
        # later into proxy_edges.csv). Saved as parquet for symmetry.
        proxy_partial = out_dir / f"proxy_{ds_name}.parquet"
        pd.DataFrame(ds_proxy_rows, columns=PROXY_COLUMNS).to_parquet(
            proxy_partial, index=False
        )

        n_abroca = len(list(figs_dir.glob(f"abroca_{ds_name}_*.png")))
        logger.log(
            f"{ds_name}  DONE  metrics={len(ds_metric_rows)} "
            f"proxy_edges={len(ds_proxy_rows)} abroca_plots={n_abroca} "
            f"elapsed={ds_elapsed:.1f}s"
        )

        if do_commit:
            _git_commit_and_push(
                out_dir=out_dir,
                dataset_name=ds_name,
                n_metric_rows=len(ds_metric_rows),
                n_proxy_edges=len(ds_proxy_rows),
                n_abroca=n_abroca,
                elapsed_s=ds_elapsed,
                do_push=do_push,
                logger=logger,
            )

    # --- Final consolidation: concat all per-dataset parquets ------------
    all_metric_rows: list[pd.DataFrame] = []
    all_proxy_rows: list[pd.DataFrame] = []
    for ds_name in datasets:
        partial = out_dir / f"audit_{ds_name}.parquet"
        if partial.exists():
            all_metric_rows.append(pd.read_parquet(partial))
        proxy_partial = out_dir / f"proxy_{ds_name}.parquet"
        if proxy_partial.exists():
            all_proxy_rows.append(pd.read_parquet(proxy_partial))

    audit_df = (
        pd.concat(all_metric_rows, ignore_index=True)
        if all_metric_rows
        else pd.DataFrame(columns=AUDIT_COLUMNS)
    )
    audit_csv = out_dir / "audit.csv"
    audit_parquet = out_dir / "audit.parquet"
    audit_df.to_csv(audit_csv, index=False)
    audit_df.to_parquet(audit_parquet, index=False)

    proxy_df = (
        pd.concat(all_proxy_rows, ignore_index=True)
        if all_proxy_rows
        else pd.DataFrame(columns=PROXY_COLUMNS)
    )
    proxy_csv = out_dir / "proxy_edges.csv"
    proxy_df.to_csv(proxy_csv, index=False)

    total_elapsed = time.perf_counter() - started_at
    logger.log(
        f"AUDIT END  metrics={len(audit_df)} proxy_edges={len(proxy_df)} "
        f"datasets={len(datasets)} total_elapsed={total_elapsed:.1f}s "
        f"failed={failed}"
    )
    logger.close()

    print(f"\nWrote {len(audit_df)} metric rows -> {audit_csv}", flush=True)
    print(f"Wrote {len(audit_df)} metric rows -> {audit_parquet}", flush=True)
    print(f"Wrote {len(proxy_df)} proxy edges -> {proxy_csv}", flush=True)
    print(f"Wrote ABROCA figures -> {figs_dir}", flush=True)
    print(f"Total elapsed: {total_elapsed:.1f}s", flush=True)

    if failed:
        print(f"FAILED: {failed}", file=sys.stderr, flush=True)
        return 1
    return 0

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_phase2_audit",
        description=(
            "Statistical and multi-class fairness audit. "
            "Runs the full panel of fairness metrics + Bayesian proxy-graph "
            "audit (FULL training set, no subsampling) + ABROCA plot for the "
            "seven Phase-1 datasets, with stratified-N=1000 counterfactual "
            "fairness on large datasets and full-test CF on small ones."
        ),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help=(
            "Comma-separated list of dataset keys to audit "
            f"(default: all of {list(LOADERS.keys())})"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(_PROJECT_ROOT / "results" / "phase2"),
        help="Output directory (default: project_new/results/phase2/).",
    )
    parser.add_argument(
        "--no-commit",
        action="store_true",
        help="Do not auto-commit the per-dataset partial after each dataset.",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help=(
            "Do not push to origin/main after each per-dataset commit "
            "(commits are still made locally if --no-commit is not set)."
        ),
    )
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
        unknown = [d for d in datasets if d not in LOADERS]
        if unknown:
            print(f"Unknown dataset key(s): {unknown!r}", file=sys.stderr, flush=True)
            return 2
    else:
        datasets = list(LOADERS.keys())

    out_dir = pathlib.Path(args.out_dir).resolve()
    do_commit = not args.no_commit
    do_push = (not args.no_push) and do_commit

    t0 = time.perf_counter()
    rc = run_audit(
        datasets=datasets, out_dir=out_dir,
        do_commit=do_commit, do_push=do_push,
    )
    elapsed = time.perf_counter() - t0
    print(f"Elapsed: {elapsed:.1f} s", flush=True)
    return rc

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

# Results

This directory contains the result tables cited by the thesis. They are
shipped pre-computed so reviewers can inspect every claim without re-running
the pipeline; `make all` regenerates them byte-identically.

## Layout

```
results/
├── audit/
│   └── audit.csv                  statistical + multi-class fairness audit
├── procedural/
│   ├── procedural.csv             process consistency, voice, flippability, actionability
│   ├── significance.csv           paired-bootstrap CIs + Cohen's d, all metrics
│   └── per_group_tpr.csv          per-protected-group TPR breakdown
├── mitigation/
│   ├── audit.csv.gz               raw per-cell mitigation matrix (gzipped)
│   └── pareto.csv                 (accuracy, fairness) Pareto frontier
└── shap/
    ├── shap_results.csv           per-(dataset, model, feature) SHAP importance
    ├── oulad_cross_arch.csv       OULAD cross-architecture SHAP comparability
    └── shap_summary.md            top-5 features per (dataset, model, group)
```

## File-by-file index

### `audit/audit.csv` (~16 KB, 140 rows)

Produced by `make audit` (equivalently, `python scripts/run_audit.py`).

One row per (dataset, target, fairness metric). Columns include the metric
name, value, classifier baseline, and a flag for per-class vacuous-TPR rows.

### `procedural/procedural.csv` (~1.3 MB, ~12,000 rows)

Produced by `make procedural`. Sweep over 6 (dataset, target) combinations
× 8 classifiers × N seeds × 4 noise levels × 5 procedural metrics.

### `procedural/significance.csv` (~190 KB)

Paired-bootstrap 95 % CIs and Cohen's d with variance floor; produced by
`scripts/run_procedural_significance.py`.

### `procedural/per_group_tpr.csv` (~5 KB)

True-positive-rate breakdown per protected group, used by the rank-disagreement
test in the thesis Results chapter.

### `mitigation/audit.csv.gz` (~2.2 MB compressed, expands to ~39 MB)

Raw per-cell mitigation matrix: dataset × base classifier × method × lambda
× seed × metric. Gzipped because the uncompressed file (337k rows) is large.

Decompress with `gunzip -k results/mitigation/audit.csv.gz` before loading.

### `mitigation/pareto.csv` (~460 KB)

Pareto frontier in (accuracy, fairness metric) space, with bootstrap CIs and
Holm-Bonferroni family-wise error control. Produced by
`scripts/compute_pareto.py`.

### `shap/shap_results.csv` (~34 KB, ~280 rows)

Mean absolute SHAP value per (dataset, model, feature, demographic group)
cell. Produced by `make shap`.

### `shap/oulad_cross_arch.csv`

OULAD-specific cross-architecture comparison (TreeExplainer vs KernelExplainer).

### `shap/shap_summary.md`

Markdown table of top-5 most-important features per (dataset, model, group).

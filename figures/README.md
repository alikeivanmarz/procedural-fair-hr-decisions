# Figures

PDF figures as cited in the thesis. They ship pre-rendered so reviewers can
inspect them without re-running the pipeline; `make figures` regenerates them
byte-identically from the result CSVs in `../results/`.

## Index

### Methodology chapter

| File                                       | Thesis section                                 |
|--------------------------------------------|-------------------------------------------------|
| `method_pipeline_compact.pdf`              | Overall experimental pipeline                  |
| `method_taxonomy.pdf`                      | Taxonomy of fairness metrics                   |
| `method_procedural_compact.pdf`            | Procedural-fairness operationalisation         |
| `method_mitigation_compact.pdf`            | Mitigation-method placement (pre / in / post)  |
| `method_multiclass_proof_compact.pdf`      | Binary-restriction equivalence proof sketch    |
| `method_inference_stack_compact.pdf`       | Statistical-inference stack                    |

### Results chapter

| File                                            | Thesis section                                                |
|-------------------------------------------------|----------------------------------------------------------------|
| `phase4_consistency_curve_compact.pdf`          | Process Consistency vs noise magnitude (RQ2)                  |
| `phase4_divergence_notion_compact.pdf`          | Statistical-vs-procedural divergence across notions           |
| `phase4_rank_disagreement_compact.pdf`          | Spearman rank-disagreement test (RQ2 headline)                |
| `phase5_effectiveness_heatmap_compact.pdf`      | Mitigation-method effectiveness heatmap                       |
| `phase5_eqodds_per_dataset_compact.pdf`         | Equalised-odds postproc per dataset (RQ3 dataset-conditioning)|
| `phase5_pareto_scatter_compact.pdf`             | Pareto frontier in (accuracy, fairness) space                 |
| `phase6_shap_importance_compact.pdf`            | SHAP feature-importance attribution                           |
| `phase6_oulad_cross_arch.pdf`                   | OULAD cross-architecture SHAP comparability                   |

### Literature-review positioning

| File                                       | Thesis section                              |
|--------------------------------------------|----------------------------------------------|
| `lit_gap_matrix_compact.pdf`               | Literature gap-matrix                       |
| `lit_research_landscape_compact.pdf`       | Research-landscape positioning              |

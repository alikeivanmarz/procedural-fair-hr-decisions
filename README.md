# Procedural Fairness in HR Machine Learning

> Reproducibility package for the Master's thesis **"Procedural Fairness in HR
> Machine Learning: Auditing Statistical Bias and Evaluating Mitigation"**
> (Lipu, 2026, Unitec Institute of Technology).

This repository contains the code, datasets specification, result tables, and
figures needed to reproduce every empirical claim in the thesis. It
operationalises three organisational-justice constructs — process consistency,
voice and representation, and transparency — as machine-learning metrics, and
benchmarks them alongside the standard statistical-fairness suite against four
HR-relevant datasets and four bias-mitigation methods.

## Contributions

1. **Statistical-fairness audit** of the standard random-forest HR-ML pipeline
   across four datasets, including the detection of target leakage on the IBM
   HR PerformanceRating target.
2. **Procedural-fairness operationalisation:** three organisational-justice
   constructs turned into quantitative ML metrics (process consistency, voice
   and representation, transparency), with a Spearman-rank disagreement test
   showing they rank models differently from statistical-fairness metrics.
3. **Bias-mitigation head-to-head comparison** of four representative methods
   (Reweighing, Learning Fair Representations, Adversarial Debiasing,
   Equalised-Odds Post-processor) with paired-bootstrap confidence intervals
   and Holm-Bonferroni family-wise error control.

## Repository layout

```
procedural-fair-hr-decisions/
├── README.md                     this file
├── LICENSE                       MIT
├── CITATION.cff                  citation metadata
├── pyproject.toml                package metadata, dependency pins
├── environment.yml               conda environment specification
├── Makefile                      reproducibility entry points
├── data/
│   ├── README.md                 dataset provenance, fetch instructions
│   ├── CHECKSUMS.md              SHA-256 checksums of the raw files
│   ├── raw/                      populated by `make data` (not redistributed)
│   └── processed/                populated by `make data`
├── src/procedural_fair_hr/
│   ├── data_loaders.py           canonical loaders for the four datasets
│   ├── fairness_metrics.py       statistical, individual, counterfactual metrics
│   ├── multiclass_metrics.py     macro-averaged extensions, ABROCA
│   ├── procedural_fairness.py    process consistency, voice, transparency
│   ├── inference.py              bootstrap, Cohen's d, Holm, Benjamini-Hochberg
│   ├── visualisation.py          ABROCA plotting helpers
│   └── mitigation/               Reweighing, LFR, AdvDebias, EqOdds postproc
├── scripts/
│   ├── download_data.sh          fetch all four datasets
│   ├── verify_checksums.py       verify SHA-256 hashes
│   ├── run_audit.py              statistical and multi-class fairness audit
│   ├── run_procedural.py         procedural-fairness measurement suite
│   ├── run_procedural_significance.py
│   ├── run_mitigation.py         mitigation head-to-head
│   ├── compute_pareto.py         Pareto frontier extraction
│   ├── run_shap.py               SHAP attribution analysis
│   └── make_figures.py           regenerate all figures from result CSVs
├── tests/                        pytest suite
├── results/
│   ├── README.md                 file index, producing commands
│   ├── audit/audit.csv
│   ├── procedural/{procedural,significance,per_group_tpr}.csv
│   ├── mitigation/{audit.csv.gz,pareto.csv}
│   └── shap/shap_results.csv
└── figures/                      PDFs as cited in the thesis
```

## Installation

The package targets Python 3.10. The canonical environment is specified by
[`environment.yml`](environment.yml).

```bash
git clone https://github.com/alikeivanmarz/procedural-fair-hr-decisions.git
cd procedural-fair-hr-decisions
conda env create -f environment.yml
conda activate procedural-fair-hr
pip install -e .
```

## Reproducing the experiments

The Makefile defines one target per published experiment. The full pipeline
is wall-clock dominated by the mitigation grid and takes approximately one day
on a 9-core machine; the other targets each finish within an hour.

```bash
make data         # download, verify, and pre-process the four datasets
make audit        # statistical and multi-class fairness audit
make procedural   # procedural-fairness metrics and significance tests
make mitigation   # mitigation head-to-head and Pareto frontier
make shap         # SHAP attribution analysis
make figures      # regenerate every figure from the result CSVs
make all          # run the full pipeline end-to-end
```

`make test` runs the pytest suite. `make lint` and `make format` run the
ruff/black/mypy checks.

### Determinism

All experiments seed every stochastic step. The mitigation pipeline pins
single-thread BLAS (`MKL_NUM_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`)
inside its process pool, so reruns are byte-identical at the cell level.

### Hardware reference

The published results were produced on an Apple M2 Pro (10-core, 32 GB) with a
9-worker process pool. Reproduction on a comparable workstation should match
the wall-clock figures above.

## Datasets

The four datasets cover binary and multi-class HR-relevant settings at a range
of sample sizes. The raw files are fetched at runtime; this repository does
not redistribute the data. See [`data/README.md`](data/README.md) for fetch
commands, sources, and licence notes; [`data/CHECKSUMS.md`](data/CHECKSUMS.md)
records the SHA-256 hashes verified by `make data`.

| ID  | Dataset            | Source                                       | Target(s)                            | Sensitive attribute(s)        |
|-----|--------------------|----------------------------------------------|--------------------------------------|--------------------------------|
| D1  | IBM HR Analytics   | Kaggle `pavansubhasht/ibm-hr-analytics-attrition-dataset` | Attrition (binary); PerformanceRating | Gender, Age, MaritalStatus     |
| D2  | OULAD              | Kaggle `anlgrbz/student-demographics-online-education-dataoulad` | Final result (3-class and 4-class)   | gender                         |
| D3  | ACS-Income         | Folktables (ACS 2018 1-yr PUMS, California) | Income > \$50k                       | SEX, RAC1P                     |
| D4  | Ricci v. DeStefano | OpenML #42665                                | Promoted (binary)                    | Race                           |

The IBM HR data and OULAD have non-redistribution clauses; users supply their
own Kaggle credentials in `~/.kaggle/kaggle.json` before running `make data`.

## Methods

### Base classifiers

Logistic Regression, Random Forest (`n_estimators=50`), Gradient Boosting,
XGBoost, K-Nearest Neighbours, and a Multilayer Perceptron — plus two
reference predictors (constant-class and shuffled-label) used as ceiling and
noise baselines in the procedural-fairness analysis.

### Fairness metrics

| Family               | Metrics                                                                                       |
|----------------------|-----------------------------------------------------------------------------------------------|
| Statistical (group)  | Demographic Parity, Equalised Odds, Equal Opportunity, Disparate Impact, SPD, AAOD, AEORD     |
| Individual           | K-Nearest Neighbours Consistency                                                              |
| Counterfactual       | Level-1 Binary CF, Multinomial CF (total variation over class probabilities)                  |
| Multi-class          | Macro-DP, Macro-EOdds, Macro-EO, filtered variants, ABROCA                                    |
| Procedural           | Process Consistency, Voice / Representation, Voice-Enrichment, Model-Flippability, Explanation-Actionability |

### Mitigation methods

| Method                          | Stage           | Library / source                   |
|---------------------------------|-----------------|------------------------------------|
| Reweighing                      | Pre-processing  | AIF360 (Kamiran & Calders, 2012)   |
| Learning Fair Representations   | Pre-processing  | AIF360 (Zemel et al., 2013)        |
| Adversarial Debiasing           | In-processing   | PyTorch reimplementation (Zhang, Lemoine & Mitchell, 2018) |
| Equalised-Odds Post-processor   | Post-processing | AIF360 (Hardt, Price & Srebro, 2016) |

### Statistical inference

Percentile and bias-corrected accelerated (BCa) bootstrap intervals (N = 30
seeds per cell), paired-difference Cohen's d with variance floor, Holm-Bonferroni
family-wise error control, and Benjamini-Hochberg false-discovery-rate control.
Headline claims survive Holm correction under five weighting schemes.

## Citation

If you use this code or its results, please cite the thesis:

```bibtex
@mastersthesis{lipu2026procedural,
  title  = {Procedural Fairness in {HR} Machine Learning:
            Auditing Statistical Bias and Evaluating Mitigation},
  author = {Lipu, Nahid Hasan},
  school = {Unitec Institute of Technology},
  year   = {2026},
  month  = may,
  type   = {Master's thesis},
  address = {Auckland, New Zealand}
}
```

Machine-readable citation metadata is in
[`CITATION.cff`](CITATION.cff).

## License

This project is released under the [MIT License](LICENSE). The licence applies
to the code in this repository; each dataset retains its own licence as noted
in [`data/README.md`](data/README.md).

## Acknowledgements

Developed at Unitec Institute of Technology, New Zealand.

.PHONY: help env data test audit procedural mitigation shap figures all clean lint format

ENV_NAME ?= procedural-fair-hr
CONDA    ?= conda
PYTHON   ?= python3

help:
	@echo "Targets:"
	@echo "  make env         create the conda environment"
	@echo "  make data        download and verify all four datasets"
	@echo "  make test        run the pytest suite"
	@echo "  make audit       run the statistical and multi-class fairness audit"
	@echo "  make procedural  run the procedural-fairness measurement suite"
	@echo "  make mitigation  run the bias-mitigation head-to-head comparison"
	@echo "  make shap        run the SHAP attribution analysis"
	@echo "  make figures     regenerate every figure from the result CSVs"
	@echo "  make all         data -> audit -> procedural -> mitigation -> shap -> figures"
	@echo "  make clean       remove processed data, results, and figures"
	@echo "  make lint        run ruff, black --check, and mypy"
	@echo "  make format      run black and ruff --fix"

env:
	$(CONDA) env create -n $(ENV_NAME) -f environment.yml

data: data/processed/.sentinel

data/processed/.sentinel: scripts/download_data.sh
	bash scripts/download_data.sh
	$(PYTHON) scripts/verify_checksums.py
	$(PYTHON) -m procedural_fair_hr.data_loaders --build-all
	touch data/processed/.sentinel

test:
	$(PYTHON) -m pytest tests/ -q

audit:
	$(PYTHON) scripts/run_audit.py

procedural:
	$(PYTHON) scripts/run_procedural.py
	$(PYTHON) scripts/run_procedural_significance.py

mitigation:
	$(PYTHON) scripts/run_mitigation.py
	$(PYTHON) scripts/compute_pareto.py

shap:
	$(PYTHON) scripts/run_shap.py

figures:
	$(PYTHON) scripts/make_figures.py

all: data test audit procedural mitigation shap figures

clean:
	rm -rf data/processed/* results/*/cache* results/*/output*

lint:
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m black --check src tests scripts
	$(PYTHON) -m mypy src

format:
	$(PYTHON) -m black src tests scripts
	$(PYTHON) -m ruff check --fix src tests scripts

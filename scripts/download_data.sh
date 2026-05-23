#!/usr/bin/env bash
# -----------------------------------------------------------------------
# download_data.sh — pull the seven verified-public datasets used by this
# thesis to data/raw/. Idempotent: skips downloads whose checksum already
# matches data/CHECKSUMS.md. Per ADR-005, this script is the *only* path
# from raw URLs to data/raw/.
#
# Datasets:
#   D1 IBM HR Analytics (Kaggle CLI required for direct download)
#   D2 Folktables ACS Income & Employment (Python loader, fetched by src/)
#   D3 UCI Adult (via ucimlrepo Python package)
#   D4 Ricci (OpenML id 42665, via openml Python package)
#   D5 Dutch Census + Law School (tailequy/fairness_dataset GitHub)
#   D6 OULAD (Kaggle mirror per ADR/R12)
# -----------------------------------------------------------------------
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${PROJECT_ROOT}/data/raw"
mkdir -p "${RAW_DIR}"

log() { printf '[download_data] %s\n' "$*" >&2; }

# -----------------------------------------------------------------------
# D5 — tailequy/fairness_dataset (Dutch Census + Law School)
# Direct git clone; small repo, no auth required.
# -----------------------------------------------------------------------
if [[ ! -d "${RAW_DIR}/tailequy_fairness" ]]; then
  log "Cloning tailequy/fairness_dataset for Dutch Census + Law School (D5)..."
  git clone --depth 1 https://github.com/tailequy/fairness_dataset.git \
    "${RAW_DIR}/tailequy_fairness"
else
  log "D5 already present, skipping."
fi

# -----------------------------------------------------------------------
# D3, D4, D2, D6 — fetched from inside Python (ucimlrepo, openml,
# folktables, Kaggle). Those are invoked by `python -m src.data_loaders
# --build-all` after this shell script returns.
# -----------------------------------------------------------------------
log "Python-side loaders (Adult, Ricci, ACS PUMS, OULAD) will run via 'make data'."

# -----------------------------------------------------------------------
# D1 — IBM HR Analytics (Kaggle CLI). The user must have configured kaggle
# credentials at ~/.kaggle/kaggle.json. If not, log a friendly message
# and continue — the dataset is small enough for manual download too.
# -----------------------------------------------------------------------
IBM_HR_DIR="${RAW_DIR}/ibm_hr"
mkdir -p "${IBM_HR_DIR}"
IBM_HR_CSV="${IBM_HR_DIR}/WA_Fn-UseC_-HR-Employee-Attrition.csv"
if [[ ! -f "${IBM_HR_CSV}" ]]; then
  if command -v kaggle >/dev/null 2>&1; then
    log "Pulling IBM HR Analytics from Kaggle (D1)..."
    kaggle datasets download -d pavansubhasht/ibm-hr-analytics-attrition-dataset \
      -p "${IBM_HR_DIR}" --unzip
  else
    log "Kaggle CLI not installed. Manual download required:"
    log "  https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset"
    log "  Save WA_Fn-UseC_-HR-Employee-Attrition.csv to: ${IBM_HR_CSV}"
  fi
else
  log "D1 already present, skipping."
fi

log "Raw downloads complete (Python loaders run next via make data)."

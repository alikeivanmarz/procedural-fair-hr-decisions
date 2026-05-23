# Dataset checksums

Every raw dataset file is listed below with its SHA-256, source URL, retrieval
date, and licence note. `scripts/verify_checksums.py` re-computes the SHA-256
for each entry after `make data` and aborts on any mismatch.

| ID  | File                                                            | SHA-256                                                              | Source URL                                                                                          | Licence note                                                                                       |
|-----|-----------------------------------------------------------------|----------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| D1  | `data/raw/ibm_hr/WA_Fn-UseC_-HR-Employee-Attrition.csv`          | (computed on first `make data`)                                       | https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset                     | Kaggle "Other (specified in description)"; copyright IBM. Research use only; not redistributed.    |
| D2  | `data/raw/oulad/studentInfo.csv`                                 | `7e6f3e474a5eee00639d2a414a6c7e928745823c2d2c2563ca1780145f99b0d6`     | Kaggle: `anlgrbz/student-demographics-online-education-dataoulad`                                    | CC-BY 4.0 (Kuzilek, Hlosta & Zdrahal, 2017, *Sci. Data* 4:170171). Kaggle mirror.                  |
| D3  | `data/raw/folktables/acs_2018_ca.csv`                            | (computed on first `make data`)                                       | https://github.com/socialfoundations/folktables (downloads via US Census Bureau)                     | US Census Bureau public-domain Microdata.                                                          |
| D4  | `data/raw/ricci/ricci.csv`                                       | (computed on first `make data`)                                       | https://www.openml.org/d/42665                                                                       | OpenML public dataset.                                                                              |

## Validation procedure

`scripts/verify_checksums.py` re-computes the SHA-256 of every listed file
after `make data` and exits non-zero on any mismatch. This guard prevents
downstream stages from silently running on a different dataset version.

# Datasets

The thesis uses four publicly available datasets covering binary and multi-class
HR-relevant settings. None of the raw files are redistributed in this
repository; they are fetched at runtime by `make data`. The fetch script
downloads each file, verifies its SHA-256 against
[`CHECKSUMS.md`](CHECKSUMS.md), and writes derived parquet artefacts to
`data/processed/`.

## Summary

| ID  | Dataset            | Rows (raw)       | Target(s)                            | Sensitive attribute(s)         |
|-----|--------------------|------------------|--------------------------------------|---------------------------------|
| D1  | IBM HR Analytics   | 1,470            | Attrition (binary); PerformanceRating | Gender, Age, MaritalStatus      |
| D2  | OULAD              | 32,593           | Final result (3- and 4-class)        | gender                          |
| D3  | ACS-Income (CA, 2018) | ~195,000     | Income > \$50k                       | SEX, RAC1P                      |
| D4  | Ricci v. DeStefano | 118              | Promoted (binary)                    | Race ∈ {W, B, H}                |

## Fetch instructions

All four datasets are downloaded by:

```bash
make data
```

which calls [`scripts/download_data.sh`](../scripts/download_data.sh). Two of
the four datasets require user-supplied credentials.

### D1 — IBM HR Analytics

Source: Kaggle dataset
[`pavansubhasht/ibm-hr-analytics-attrition-dataset`](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset).

Requires a configured Kaggle API token at `~/.kaggle/kaggle.json` (see the
[Kaggle API documentation](https://github.com/Kaggle/kaggle-api)).

> **Licence:** IBM copyright; Kaggle terms permit use for research and
> education. The data are **not redistributed** by this repository.

### D2 — OULAD

Source: Kaggle mirror
[`anlgrbz/student-demographics-online-education-dataoulad`](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad)
of the canonical Open University Learning Analytics Dataset
(Kuzilek, Hlosta & Zdrahal, 2017, *Scientific Data* 4:170171).

Requires a configured Kaggle API token. The Kaggle mirror is used because the
canonical KMI archive at `https://analyse.kmi.open.ac.uk/open_dataset` returns
HTTP 404; the Kaggle copy contains the same data.

> **Licence:** CC-BY 4.0.

### D3 — ACS-Income

Source: US Census Bureau American Community Survey, 2018 one-year Public Use
Microdata Sample (PUMS) for California, accessed via the
[`folktables`](https://github.com/socialfoundations/folktables) Python API
(Ding et al., 2021).

No credentials required.

> **Licence:** US Census Bureau public-domain microdata.

### D4 — Ricci v. DeStefano

Source: OpenML dataset
[#42665](https://www.openml.org/d/42665) (2003 New Haven Fire Department
promotion exam, made famous by *Ricci v. DeStefano*, 557 U.S. 557).

No credentials required.

> **Licence:** OpenML public dataset.

## Pre-processing

`make data` calls `python -m procedural_fair_hr.data_loaders --build-all` to
produce model-ready parquet files in `data/processed/`. The pre-processing for
each dataset is described in the thesis methodology chapter and is fully
deterministic given the fetched raw files.

## Integrity

After `make data`, `scripts/verify_checksums.py` re-computes the SHA-256 of
each raw file and aborts if any hash does not match
[`CHECKSUMS.md`](CHECKSUMS.md). A mismatch breaks `make all` cleanly rather
than silently introducing a different dataset version.

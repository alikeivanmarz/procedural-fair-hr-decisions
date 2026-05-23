"""Canonical loaders for the four datasets used in the thesis.

Each loader returns a :class:`DatasetBundle` with the structure::

 {
 "X_train": pandas.DataFrame,
 "X_test": pandas.DataFrame,
 "y_train": pandas.Series,
 "y_test": pandas.Series,
 "A_train": pandas.DataFrame, # sensitive attributes
 "A_test": pandas.DataFrame,
 "feature_names": list[str],
 "sensitive_names": list[str],
 "class_names": list[str],
 "n_classes": int,
 }

Datasets:

* IBM HR Analytics (Kaggle) -- binary Attrition target and
 multi-class PerformanceRating target.
* Open University Learning Analytics Dataset (Kaggle mirror) -- 3-class
 and 4-class final-result targets.
* ACS-Income (Folktables, ACS 2018 California PUMS) -- binary high-income
 target.
* Ricci v. DeStefano (OpenML 42665) -- binary promotion target.

Every loader uses ``random_state=0`` for an 80/20 stratified split so reruns
are deterministic.
"""

from __future__ import annotations

import pathlib
from typing import TypedDict

import pandas as pd

# Repo root resolved relative to this source file:
# src/procedural_fair_hr/data_loaders.py -> parents[2] is the repo root.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

class DatasetBundle(TypedDict):
    """Canonical loader return shape.

 ``A_train`` and ``A_test`` are always :class:`pandas.DataFrame` so that
 single- and multi-sensitive-attribute datasets share one type contract.
 """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    A_train: pd.DataFrame
    A_test: pd.DataFrame
    feature_names: list[str]
    sensitive_names: list[str]
    class_names: list[str]
    n_classes: int

# ---------------------------------------------------------------------------
# D1 -- IBM HR Analytics
# ---------------------------------------------------------------------------

def load_ibm_hr(target: str = "PerformanceRating") -> DatasetBundle:
    """Load IBM HR Analytics from the local Kaggle download.

 The raw CSV must be at
 ``data/raw/ibm_hr/WA_Fn-UseC_-HR-Employee-Attrition.csv`` relative to the
 repository root; ``scripts/download_data.sh`` fetches it via the Kaggle CLI.

 Parameters
 ----------
 target:
 ``"PerformanceRating"`` (multi-class, values ``{3, 4}``) or
 ``"Attrition"`` (binary).

 Returns
 -------
 DatasetBundle
 """
    from sklearn.model_selection import train_test_split

    csv_path = (
        REPO_ROOT / "data" / "raw" / "ibm_hr"
        / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    )
    if not csv_path.exists():
        raise FileNotFoundError(
            f"IBM HR raw CSV not found at {csv_path}. "
            "Run: kaggle datasets download "
            "-d pavansubhasht/ibm-hr-analytics-attrition-dataset "
            f"-p {csv_path.parent} --unzip"
        )

    df = pd.read_csv(csv_path)

    if target == "PerformanceRating":
        y = df["PerformanceRating"].astype(int)
        unique_classes = sorted(y.unique().tolist())
        n_classes = len(unique_classes)
        class_names = [str(c) for c in unique_classes]
    elif target == "Attrition":
        y = df["Attrition"].map({"Yes": 1, "No": 0}).astype(int)
        if y.isna().any():
            unexpected = sorted(df["Attrition"][y.isna()].unique().tolist())
            raise ValueError(f"Unexpected Attrition labels: {unexpected!r}")
        n_classes = 2
        class_names = ["No", "Yes"]
    else:
        raise ValueError(
            f"Unknown target {target!r}. Choose 'PerformanceRating' or 'Attrition'."
        )

    gender = df["Gender"].astype(str)
    drop_cols = {target, "Gender"}
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X,
        y,
        gender,
        test_size=0.2,
        random_state=0,
        stratify=y,
    )

    y_train = pd.Series(y_train.values, index=X_train.index, name=target, dtype=int)
    y_test = pd.Series(y_test.values, index=X_test.index, name=target, dtype=int)
    A_train = pd.DataFrame({"Gender": A_train.values}, index=X_train.index)
    A_test = pd.DataFrame({"Gender": A_test.values}, index=X_test.index)

    return DatasetBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        A_train=A_train,
        A_test=A_test,
        feature_names=list(X_train.columns),
        sensitive_names=["Gender"],
        class_names=class_names,
        n_classes=n_classes,
    )

def load_ibm_hr_perfrating_deleaked() -> DatasetBundle:
    """Load IBM HR PerformanceRating with the leakage feature removed.

 Sibling of :func:`load_ibm_hr`. Identical to
 ``load_ibm_hr(target="PerformanceRating")`` except the ``PercentSalaryHike``
 feature is dropped from ``X_train`` and ``X_test``.

 Rationale: on the fictional IBM HR data, ``PercentSalaryHike`` is a
 deterministic function of ``PerformanceRating`` (linear-regression R^2 = 1.0),
 so any classifier trained on the raw target trivially recovers it.
 """
    bundle = load_ibm_hr(target="PerformanceRating")
    leak_col = "PercentSalaryHike"
    if leak_col not in bundle["X_train"].columns:
        raise AssertionError(
            f"Expected leakage column {leak_col!r} in IBM HR feature matrix; "
            f"got columns: {list(bundle['X_train'].columns)!r}"
        )
    bundle["X_train"] = bundle["X_train"].drop(columns=[leak_col])
    bundle["X_test"] = bundle["X_test"].drop(columns=[leak_col])
    bundle["feature_names"] = list(bundle["X_train"].columns)
    return bundle

# ---------------------------------------------------------------------------
# D3 -- ACS-Income (Folktables, ACS 2018 California PUMS)
# ---------------------------------------------------------------------------

def load_acs(state: str = "CA", year: int = 2018, task: str = "income") -> DatasetBundle:
    """Load Folktables ACS PUMS data via the ``folktables`` package.

 Downloads ACS 1-Year person-survey microdata for the given state and year,
 caches the raw frame as parquet under
 ``data/raw/folktables/{state}_{year}_{task}.parquet``, and returns the
 requested task as a :class:`DatasetBundle`.

 Parameters
 ----------
 state:
 Two-letter US state code (default ``"CA"``).
 year:
 ACS survey year (default ``2018``).
 task:
 Either ``"income"`` (target ``PINCP > 50000``) or ``"employment"``
 (target ``ESR == 1``).

 Returns
 -------
 DatasetBundle
 With sensitive attribute ``RAC1P`` (race recode).

 Reference
 ---------
 Ding, Hardt, Miller & Schmidt (2021) "Retiring Adult: New Datasets for
 Fair Machine Learning", NeurIPS.
 """
    import numpy as np
    from folktables import ACSDataSource, ACSEmployment, ACSIncome
    from sklearn.model_selection import train_test_split

    if task not in ("income", "employment"):
        raise ValueError(f"task must be 'income' or 'employment', got {task!r}")

    raw_dir = REPO_ROOT / "data" / "raw" / "folktables"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{state}_{year}_{task}.parquet"

    if raw_path.exists():
        acs_data = pd.read_parquet(raw_path)
    else:
        data_source = ACSDataSource(
            survey_year=str(year), horizon="1-Year", survey="person"
        )
        acs_data = data_source.get_data(states=[state], download=True)
        acs_data.to_parquet(raw_path)

    if task == "income":
        acs_task = ACSIncome
        class_names: list[str] = ["low_income", "high_income"]
    else:
        acs_task = ACSEmployment
        class_names = ["not_employed", "employed"]

    # ``df_to_pandas`` already applies the task's binary target_transform and
    # returns booleans, so we cast directly to int.
    X_all, y_raw, group_raw = acs_task.df_to_pandas(acs_data)
    y_raw = np.ravel(y_raw)
    group_raw = np.ravel(group_raw)

    y_int = y_raw.astype(int)
    y_all = pd.Series(y_int, name="target", dtype=int)
    A_all = pd.Series(group_raw, name="RAC1P")

    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X_all, y_all, A_all,
        test_size=0.2,
        random_state=0,
        stratify=y_all,
    )

    y_train = pd.Series(y_train.values, index=X_train.index, name="target", dtype=int)
    y_test = pd.Series(y_test.values, index=X_test.index, name="target", dtype=int)
    A_train = pd.DataFrame({"RAC1P": A_train.values}, index=X_train.index)
    A_test = pd.DataFrame({"RAC1P": A_test.values}, index=X_test.index)

    return DatasetBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        A_train=A_train,
        A_test=A_test,
        feature_names=list(X_train.columns),
        sensitive_names=["RAC1P"],
        class_names=class_names,
        n_classes=2,
    )

# ---------------------------------------------------------------------------
# D4 -- Ricci v. DeStefano (OpenML 42665)
# ---------------------------------------------------------------------------

def load_ricci() -> DatasetBundle:
    """Load the Ricci v. DeStefano firefighter promotion dataset via OpenML.

 Fetches OpenML id 42665, caches the raw CSV to
 ``data/raw/ricci/ricci.csv``, and returns a binary promotion target with
 ``Race`` (``W``, ``B``, ``H``) as the sensitive attribute. 118 rows total
 (94 train + 24 test under an 80/20 stratified split).

 Reference
 ---------
 Le Quy et al. (2022) "A survey on datasets for fairness-aware machine
 learning", WIREs Data Mining and Knowledge Discovery.
 """
    import openml
    from sklearn.model_selection import train_test_split

    dataset = openml.datasets.get_dataset(42665)
    X_raw, y_raw, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    raw_dir = REPO_ROOT / "data" / "raw" / "ricci"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = raw_dir / "ricci.csv"
    if not raw_csv.exists():
        combined = X_raw.copy()
        combined[y_raw.name] = y_raw.values
        combined.to_csv(raw_csv, index=False)

    label_map = {"Promotion": 1, "No promotion": 0}
    y = y_raw.map(label_map)
    if y.isna().any():
        unexpected = sorted(y_raw[y.isna()].unique().tolist())
        raise ValueError(
            f"Unexpected Promotion labels in Ricci dataset: {unexpected!r}"
        )
    y = y.astype(int)
    y.name = "Promotion"

    race = X_raw["Race"].astype(str).copy()

    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X_raw,
        y,
        race,
        test_size=0.2,
        random_state=0,
        stratify=y,
    )

    y_train = pd.Series(y_train, index=X_train.index, name="Promotion", dtype=int)
    y_test = pd.Series(y_test, index=X_test.index, name="Promotion", dtype=int)
    A_train = pd.DataFrame({"Race": A_train}, index=X_train.index)
    A_test = pd.DataFrame({"Race": A_test}, index=X_test.index)

    return DatasetBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        A_train=A_train,
        A_test=A_test,
        feature_names=list(X_raw.columns),
        sensitive_names=["Race"],
        class_names=["Not Promoted", "Promoted"],
        n_classes=2,
    )

# ---------------------------------------------------------------------------
# D2 -- Open University Learning Analytics Dataset (OULAD)
# ---------------------------------------------------------------------------

def load_oulad() -> DatasetBundle:
    """Load OULAD with the 3-class ``final_result`` target.

 Downloads the Kaggle mirror
 ``anlgrbz/student-demographics-online-education-dataoulad`` (the canonical
 KMI archive currently returns 404), drops rows with ``final_result ==
 "Withdrawn"`` and rows with missing values, and returns a 3-class target
 ``{Fail=0, Pass=1, Distinction=2}`` with ``gender`` (``M=1``, ``F=0``) as
 the sensitive attribute.

 Reference
 ---------
 Kuzilek, Hlosta & Zdrahal (2017) "Open University Learning Analytics
 dataset", Scientific Data 4, 170171.
 """
    import subprocess

    from sklearn.model_selection import train_test_split

    raw_dir = REPO_ROOT / "data" / "raw" / "oulad"
    csv_path = raw_dir / "studentInfo.csv"
    if not csv_path.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                "anlgrbz/student-demographics-online-education-dataoulad",
                "--unzip",
                "-p",
                str(raw_dir),
            ],
            check=True,
        )
    if not csv_path.exists():
        raise FileNotFoundError(
            f"OULAD studentInfo.csv not found after Kaggle download at: {csv_path}."
        )
    df = pd.read_csv(csv_path)

    expected_cols = [
        "code_module",
        "code_presentation",
        "id_student",
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "num_of_prev_attempts",
        "studied_credits",
        "disability",
        "final_result",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"OULAD CSV missing expected columns: {missing!r}.")

    if df.isna().any().any():
        df = df.dropna().reset_index(drop=True)
    if "Withdrawn" in set(df["final_result"].astype(str).unique()):
        df = df[df["final_result"] != "Withdrawn"].reset_index(drop=True)

    label_map = {"Fail": 0, "Pass": 1, "Distinction": 2}
    y = df["final_result"].map(label_map)
    if y.isna().any():
        unexpected = sorted(df["final_result"][y.isna()].unique().tolist())
        raise ValueError(
            f"Unexpected final_result labels in OULAD CSV: {unexpected!r}"
        )
    y = y.astype(int)
    y.name = "final_result"

    gender_map = {"M": 1, "F": 0}
    gender = df["gender"].map(gender_map)
    if gender.isna().any():
        unexpected = sorted(df["gender"][gender.isna()].unique().tolist())
        raise ValueError(
            f"Unexpected gender labels in OULAD CSV: {unexpected!r}"
        )
    gender = gender.astype(int)

    X = df.drop(columns=["final_result", "id_student"]).copy()
    X["gender"] = gender.values

    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X,
        y,
        gender,
        test_size=0.20,
        random_state=0,
        stratify=y,
    )

    y_train = pd.Series(y_train.values, index=X_train.index, name="final_result", dtype=int)
    y_test = pd.Series(y_test.values, index=X_test.index, name="final_result", dtype=int)
    A_train = pd.DataFrame({"gender": A_train.values}, index=X_train.index)
    A_test = pd.DataFrame({"gender": A_test.values}, index=X_test.index)

    return DatasetBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        A_train=A_train,
        A_test=A_test,
        feature_names=list(X_train.columns),
        sensitive_names=["gender"],
        class_names=["Fail", "Pass", "Distinction"],
        n_classes=3,
    )

def load_oulad_4class() -> DatasetBundle:
    """Load OULAD with the 4-class ``final_result`` target (Withdrawn retained).

 Identical to :func:`load_oulad` but the Withdrawn class is kept, giving
 class encoding ``{Withdrawn=0, Fail=1, Pass=2, Distinction=3}``.
 """
    import subprocess

    from sklearn.model_selection import train_test_split

    raw_dir = REPO_ROOT / "data" / "raw" / "oulad"
    csv_path = raw_dir / "studentInfo.csv"
    if not csv_path.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                "anlgrbz/student-demographics-online-education-dataoulad",
                "--unzip",
                "-p",
                str(raw_dir),
            ],
            check=True,
        )
    if not csv_path.exists():
        raise FileNotFoundError(
            f"OULAD studentInfo.csv not found after Kaggle download at: {csv_path}."
        )
    df = pd.read_csv(csv_path)

    expected_cols = [
        "code_module",
        "code_presentation",
        "id_student",
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "num_of_prev_attempts",
        "studied_credits",
        "disability",
        "final_result",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"OULAD CSV missing expected columns: {missing!r}.")

    if df.isna().any().any():
        df = df.dropna().reset_index(drop=True)

    label_map = {"Withdrawn": 0, "Fail": 1, "Pass": 2, "Distinction": 3}
    y = df["final_result"].map(label_map)
    if y.isna().any():
        unexpected = sorted(df["final_result"][y.isna()].unique().tolist())
        raise ValueError(
            f"Unexpected final_result labels in OULAD CSV: {unexpected!r}"
        )
    y = y.astype(int)
    y.name = "final_result"

    gender_map = {"M": 1, "F": 0}
    gender = df["gender"].map(gender_map)
    if gender.isna().any():
        unexpected = sorted(df["gender"][gender.isna()].unique().tolist())
        raise ValueError(
            f"Unexpected gender labels in OULAD CSV: {unexpected!r}"
        )
    gender = gender.astype(int)

    X = df.drop(columns=["final_result", "id_student"]).copy()
    X["gender"] = gender.values

    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X,
        y,
        gender,
        test_size=0.20,
        random_state=0,
        stratify=y,
    )

    y_train = pd.Series(y_train.values, index=X_train.index, name="final_result", dtype=int)
    y_test = pd.Series(y_test.values, index=X_test.index, name="final_result", dtype=int)
    A_train = pd.DataFrame({"gender": A_train.values}, index=X_train.index)
    A_test = pd.DataFrame({"gender": A_test.values}, index=X_test.index)

    return DatasetBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        A_train=A_train,
        A_test=A_test,
        feature_names=list(X_train.columns),
        sensitive_names=["gender"],
        class_names=["Withdrawn", "Fail", "Pass", "Distinction"],
        n_classes=4,
    )

# ---------------------------------------------------------------------------
# CLI: invoked by ``make data``
# ---------------------------------------------------------------------------

_LOADERS: list[tuple[str, object]] = [
    ("ibm_hr_attrition", lambda: load_ibm_hr(target="Attrition")),
    ("ibm_hr_perfrating", load_ibm_hr_perfrating_deleaked),
    ("acs_income", load_acs),
    ("ricci", load_ricci),
    ("oulad", load_oulad),
    ("oulad_4class", load_oulad_4class),
]

def _sha256_file(path: pathlib.Path) -> str:
    """Return the hex SHA-256 digest of ``path``."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _parse_checksums_md(checksums_path: pathlib.Path) -> list[dict[str, str]]:
    """Parse ``data/CHECKSUMS.md`` and return a list of row dicts."""
    rows: list[dict[str, str]] = []
    if not checksums_path.exists():
        return rows

    lines = checksums_path.read_text(encoding="utf-8").splitlines()
    header: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if not header:
            header = cells
            continue
        if all(set(c) <= {"-", " ", ":"} for c in cells):
            continue
        if len(cells) < len(header):
            cells += [""] * (len(header) - len(cells))
        rows.append(dict(zip(header, cells)))
    return rows

def _build_all(dry_run: bool, project_root: pathlib.Path) -> int:
    """Run every loader and write the concatenated train+test frame to parquet."""
    processed_dir = project_root / "data" / "processed"

    loaders_info = [
        (name, fn, processed_dir / f"{name}.parquet")
        for name, fn in _LOADERS
    ]

    if dry_run:
        print("--build-all --dry-run: the following actions would be taken:\n")
        for name, _fn, out_path in loaders_info:
            print(
                f"  [{name}]  call loader -> concatenate X_train+X_test"
                f" -> write {out_path}"
            )
        print(
            f"\n  Total: {len(loaders_info)} dataset(s) would be written"
            f" to {processed_dir}"
        )
        print("\n[dry-run] No downloads or writes performed. Exit 0.")
        return 0

    processed_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, str]] = []
    errors: list[tuple[str, str]] = []

    for name, fn, out_path in loaders_info:
        try:
            bundle = fn()  # type: ignore[operator]
            df = pd.concat(
                [bundle["X_train"], bundle["X_test"]], ignore_index=True
            )
            df.to_parquet(out_path, index=False)
            results.append((name, f"OK  {len(df):>7,} rows -> {out_path}"))
        except Exception as exc:  # noqa: BLE001
            errors.append((name, str(exc)))
            results.append((name, f"ERR {exc!r}"))

    print("\n=== build-all summary ===")
    for name, msg in results:
        status = "FAIL" if any(n == name for n, _ in errors) else "PASS"
        print(f"  [{status}] {name}: {msg}")
    print(
        f"\n  {len(results) - len(errors)}/{len(results)} datasets built"
        f" successfully."
    )

    return 1 if errors else 0

def _verify_checksums(project_root: pathlib.Path) -> int:
    """Recompute SHA-256 for each file listed in ``data/CHECKSUMS.md``."""
    checksums_path = project_root / "data" / "CHECKSUMS.md"
    rows = _parse_checksums_md(checksums_path)

    if not rows:
        print("No checksum rows found in data/CHECKSUMS.md - nothing to verify.")
        return 0

    any_fail = False
    print("\n=== verify-checksums ===")
    for row in rows:
        rel_file = row.get("File", "").strip("`")
        stored_sha = row.get("sha256", "").strip()
        skip_markers = {"(pending)", "", "(per file, pending)", "(computed on first `make data`)"}
        if not rel_file or stored_sha in skip_markers:
            print(f"  [SKIP] {rel_file or '(empty)'}: no stored checksum yet")
            continue
        abs_path = project_root / rel_file
        if not abs_path.exists():
            print(f"  [SKIP] {rel_file}: file not present locally")
            continue
        actual_sha = _sha256_file(abs_path)
        if actual_sha == stored_sha:
            print(f"  [PASS] {rel_file}")
        else:
            print(f"  [FAIL] {rel_file}")
            print(f"         stored : {stored_sha}")
            print(f"         actual : {actual_sha}")
            any_fail = True

    if any_fail:
        print("\nChecksum verification FAILED - at least one mismatch detected.")
        return 1
    print("\nChecksum verification PASSED.")
    return 0

def _record_checksums(project_root: pathlib.Path) -> int:
    """Compute SHA-256 for present files and write back to ``data/CHECKSUMS.md``."""
    checksums_path = project_root / "data" / "CHECKSUMS.md"
    rows = _parse_checksums_md(checksums_path)

    if not rows:
        print("No checksum rows found in data/CHECKSUMS.md - nothing to record.")
        return 0

    updated_count = 0
    print("\n=== record-checksums ===")
    for row in rows:
        rel_file = row.get("File", "").strip("`")
        if not rel_file:
            continue
        abs_path = project_root / rel_file
        if not abs_path.exists():
            print(f"  [SKIP] {rel_file}: not present")
            continue
        digest = _sha256_file(abs_path)
        old = row.get("sha256", "")
        row["sha256"] = digest
        updated_count += 1
        if old == digest:
            print(f"  [SAME] {rel_file}: {digest}")
        else:
            print(f"  [UPDT] {rel_file}: {digest}  (was: {old})")

    if updated_count == 0:
        print("No files present to hash - CHECKSUMS.md unchanged.")
        return 0

    original_text = checksums_path.read_text(encoding="utf-8")
    lines = original_text.splitlines(keepends=True)

    header: list[str] = []
    header_lineno = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|") and not header:
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if any("sha256" in c.lower() for c in cells):
                header = cells
                header_lineno = i
                break

    if header_lineno < 0 or not header:
        print("WARNING: Could not locate CHECKSUMS.md table header - file not updated.")
        return 0

    sha_col_idx = next(
        (i for i, h in enumerate(header) if "sha256" in h.lower()), None
    )
    if sha_col_idx is None:
        print("WARNING: sha256 column not found in table header - file not updated.")
        return 0

    file_col_idx = next(
        (i for i, h in enumerate(header) if h.lower() == "file"), None
    )
    if file_col_idx is None:
        print("WARNING: File column not found in table header - file not updated.")
        return 0

    digest_map: dict[str, str] = {
        row.get("File", "").strip("`"): row["sha256"]
        for row in rows
        if "sha256" in row
    }

    new_lines = list(lines)
    data_start = header_lineno + 2
    row_cursor = 0
    for i in range(data_start, len(new_lines)):
        line = new_lines[i]
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if row_cursor >= len(rows):
            break
        if file_col_idx < len(cells):
            file_val = cells[file_col_idx].strip("`")
        else:
            file_val = ""
        if file_val in digest_map:
            if sha_col_idx < len(cells):
                cells[sha_col_idx] = digest_map[file_val]
            new_lines[i] = "| " + " | ".join(cells) + " |\n"
        row_cursor += 1

    checksums_path.write_text("".join(new_lines), encoding="utf-8")
    print(f"\nUpdated {updated_count} checksum(s) in {checksums_path}.")
    return 0

def _main() -> int:
    """CLI for dataset build / checksum utilities.

 Usage::

 python -m procedural_fair_hr.data_loaders --build-all [--dry-run]
 python -m procedural_fair_hr.data_loaders --verify-checksums
 python -m procedural_fair_hr.data_loaders --record-checksums
 """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m procedural_fair_hr.data_loaders",
        description="Build and verify processed datasets.",
    )
    parser.add_argument(
        "--build-all",
        action="store_true",
        help="Run all loaders and write parquets to data/processed/.",
    )
    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="Recompute sha256 for files in data/CHECKSUMS.md and compare.",
    )
    parser.add_argument(
        "--record-checksums",
        action="store_true",
        help="Compute sha256 for files in data/raw/ and update data/CHECKSUMS.md.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="(modifier for --build-all) Print actions without executing them.",
    )
    args = parser.parse_args()

    if not any([args.build_all, args.verify_checksums, args.record_checksums]):
        parser.print_help()
        return 0

    rc = 0
    if args.build_all:
        rc = max(rc, _build_all(dry_run=args.dry_run, project_root=REPO_ROOT))
    if args.verify_checksums:
        rc = max(rc, _verify_checksums(project_root=REPO_ROOT))
    if args.record_checksums:
        rc = max(rc, _record_checksums(project_root=REPO_ROOT))
    return rc

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())

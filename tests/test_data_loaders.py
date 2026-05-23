"""Unit tests for :mod:`procedural_fair_hr.data_loaders`.

Each test asserts the canonical loader contract from
the project documentation (return-shape contract).

CLI tests (``test_main_*``) verify the ``_main()`` function implemented in
  (``--build-all``, ``--verify-checksums``,
``--record-checksums``, ``--dry-run``).
"""

from __future__ import annotations

import pathlib
import textwrap

import pandas as pd
import pytest

from procedural_fair_hr.data_loaders import (
    _build_all,
    _main,
    _parse_checksums_md,
    _record_checksums,
    _sha256_file,
    _verify_checksums,
    load_acs,
    load_ibm_hr,
    load_ibm_hr_perfrating_deleaked,
    load_oulad,
    load_oulad_4class,
    load_ricci,
)

INV002_KEYS = {
    "X_train",
    "X_test",
    "y_train",
    "y_test",
    "A_train",
    "A_test",
    "feature_names",
    "sensitive_names",
    "class_names",
    "n_classes",
}

def test_ricci() -> None:
    """``load_ricci`` honours ; 118-row dataset splits 94 train + 24 test."""
    bundle = load_ricci()

    # : every key present.
    assert set(bundle.keys()) == INV002_KEYS

    # Types .
    assert isinstance(bundle["X_train"], pd.DataFrame)
    assert isinstance(bundle["X_test"], pd.DataFrame)
    assert isinstance(bundle["y_train"], pd.Series)
    assert isinstance(bundle["y_test"], pd.Series)
    assert isinstance(bundle["A_train"], pd.DataFrame)
    assert isinstance(bundle["A_test"], pd.DataFrame)

    # Ricci has 118 rows; 80/20 stratified split = 94 train + 24 test.
    assert len(bundle["X_train"]) == 94
    assert len(bundle["X_test"]) == 24
    assert len(bundle["y_train"]) == len(bundle["X_train"])
    assert len(bundle["y_test"]) == len(bundle["X_test"])
    assert len(bundle["A_train"]) == len(bundle["X_train"])
    assert len(bundle["A_test"]) == len(bundle["X_test"])

    # Binary target is correct. Both classes present.
    assert bundle["n_classes"] == 2
    assert bundle["class_names"] == ["Not Promoted", "Promoted"]
    assert set(bundle["y_train"].unique()) == {0, 1}, (
        f"y_train must contain both classes; got {set(bundle['y_train'].unique())}"
    )
    assert set(bundle["y_test"].unique()) == {0, 1}, (
        f"y_test must contain both classes; got {set(bundle['y_test'].unique())}"
    )

    # Sensitive attribute is Race.
    assert bundle["sensitive_names"] == ["Race"]
    assert bundle["A_train"].columns.tolist() == ["Race"]
    assert bundle["A_test"].columns.tolist() == ["Race"]

    # No index leakage between train and test.
    assert set(bundle["X_train"].index).isdisjoint(set(bundle["X_test"].index))

def test_ibm_hr() -> None:
    """``load_ibm_hr`` honours  and produces a clean 80/20 stratified split."""
    bundle = load_ibm_hr()

    # : every key present.
    assert set(bundle.keys()) == INV002_KEYS

    # Types .
    assert isinstance(bundle["X_train"], pd.DataFrame)
    assert isinstance(bundle["X_test"], pd.DataFrame)
    assert isinstance(bundle["y_train"], pd.Series)
    assert isinstance(bundle["y_test"], pd.Series)
    assert isinstance(bundle["A_train"], pd.DataFrame)
    assert isinstance(bundle["A_test"], pd.DataFrame)

    # IBM HR ships 1,470 rows; 80/20 split => 1176 train + 294 test.
    assert len(bundle["X_train"]) + len(bundle["X_test"]) == 1470

    # n_classes and class_names consistent.
    assert bundle["n_classes"] >= 2
    assert len(bundle["class_names"]) == bundle["n_classes"]

    # Primary sensitive attribute is Gender.
    assert bundle["sensitive_names"] == ["Gender"]
    assert bundle["A_train"].columns.tolist() == ["Gender"]

    # Stratification check: same label set in train and test.
    assert set(bundle["y_train"].unique()) == set(bundle["y_test"].unique())

    # No index leakage between train and test.
    assert set(bundle["X_train"].index).isdisjoint(set(bundle["X_test"].index))

    # Attrition variant: binary with n_classes == 2.
    bundle_attr = load_ibm_hr(target="Attrition")
    assert bundle_attr["n_classes"] == 2
    assert bundle_attr["class_names"] == ["No", "Yes"]

def test_load_ibm_hr_perfrating_deleaked() -> None:
    """De-leaked variant returns 4-class PerformanceRating without PercentSalaryHike.

    Companion to ``test_ibm_hr``: verifies the sibling loader honours
    and produces a bundle whose feature matrix is exactly the baseline IBM HR
    PerformanceRating feature matrix minus the leakage column
    ``PercentSalaryHike`` (Phase-2 audit;  Block C).
    """
    bundle = load_ibm_hr_perfrating_deleaked()

    # : every key present, correct types.
    assert set(bundle.keys()) == INV002_KEYS
    assert isinstance(bundle["X_train"], pd.DataFrame)
    assert isinstance(bundle["X_test"], pd.DataFrame)
    assert isinstance(bundle["y_train"], pd.Series)
    assert isinstance(bundle["y_test"], pd.Series)
    assert isinstance(bundle["A_train"], pd.DataFrame)
    assert isinstance(bundle["A_test"], pd.DataFrame)

    # Leakage column is removed from both partitions.
    assert "PercentSalaryHike" not in bundle["X_train"].columns
    assert "PercentSalaryHike" not in bundle["X_test"].columns
    assert "PercentSalaryHike" not in bundle["feature_names"]

    # Multi-class structure preserved: PerformanceRating has 2 classes in the
    # IBM HR fictional data ({3, 4}) -- documented as a known dataset
    # limitation; the loader inherits ``n_classes`` from
    # :func:`load_ibm_hr`.
    baseline = load_ibm_hr(target="PerformanceRating")
    assert bundle["n_classes"] == baseline["n_classes"]
    assert bundle["class_names"] == baseline["class_names"]

    # Sensitive attribute unchanged (Gender).
    assert bundle["sensitive_names"] == ["Gender"]
    assert bundle["A_train"].columns.tolist() == ["Gender"]
    assert bundle["A_test"].columns.tolist() == ["Gender"]

    # Row counts unchanged vs. baseline (same split, seed=0).
    assert len(bundle["X_train"]) == len(baseline["X_train"])
    assert len(bundle["X_test"]) == len(baseline["X_test"])

    # Feature set is exactly baseline minus PercentSalaryHike.
    expected_cols = set(baseline["X_train"].columns) - {"PercentSalaryHike"}
    assert set(bundle["X_train"].columns) == expected_cols
    assert set(bundle["X_test"].columns) == expected_cols

    # No index leakage between train and test.
    assert set(bundle["X_train"].index).isdisjoint(set(bundle["X_test"].index))

def test_acs() -> None:
    """``load_acs`` honours ; default CA/2018/income split is in range."""
    bundle = load_acs()  # default: state="CA", year=2018, task="income"

    # : every key present.
    assert set(bundle.keys()) == INV002_KEYS

    # Types .
    assert isinstance(bundle["X_train"], pd.DataFrame)
    assert isinstance(bundle["X_test"], pd.DataFrame)
    assert isinstance(bundle["y_train"], pd.Series)
    assert isinstance(bundle["y_test"], pd.Series)
    assert isinstance(bundle["A_train"], pd.DataFrame)
    assert isinstance(bundle["A_test"], pd.DataFrame)

    # ACS CA 2018 income task: expect ~100k–300k rows total.
    total = len(bundle["X_train"]) + len(bundle["X_test"])
    assert 100_000 <= total <= 300_000, f"Unexpected total rows: {total}"
    assert len(bundle["y_train"]) == len(bundle["X_train"])
    assert len(bundle["y_test"]) == len(bundle["X_test"])
    assert len(bundle["A_train"]) == len(bundle["X_train"])
    assert len(bundle["A_test"]) == len(bundle["X_test"])

    # Binary target and class metadata. Both classes must be present
    # (: an earlier double-application of `>= 50_000` against an
    # already-boolean target produced an all-zero y. Plain `issubset({0,1})`
    # was satisfied by `{0}` and missed the bug — assert exact equality so a
    # degenerate target now fails loudly).
    assert bundle["n_classes"] == 2
    assert bundle["class_names"] == ["low_income", "high_income"]
    assert set(bundle["y_train"].unique()) == {0, 1}, (
        f"y_train must contain both classes; got {set(bundle['y_train'].unique())}"
    )
    assert set(bundle["y_test"].unique()) == {0, 1}, (
        f"y_test must contain both classes; got {set(bundle['y_test'].unique())}"
    )

    # Primary sensitive attribute is RAC1P (race/ethnicity).
    assert bundle["sensitive_names"] == ["RAC1P"]
    assert bundle["A_train"].columns.tolist() == ["RAC1P"]
    assert bundle["A_test"].columns.tolist() == ["RAC1P"]

    # No index leakage between train and test.
    assert set(bundle["X_train"].index).isdisjoint(set(bundle["X_test"].index))

def test_oulad() -> None:
    """``load_oulad`` honours ; cleaned 21,562-row dataset splits 80/20.

    Single-sensitive loader: ``A_train`` / ``A_test`` are :class:`pandas.DataFrame`
    with one column ``"gender"``. Total cleaned row count
    is fixed at 21,562 per Le Quy 2022 §3.4.2 (raw OULAD has 32,593 rows; rows
    with ``final_result == "Withdrawn"`` and rows with missing values are
    dropped upstream by the tailequy/fairness_dataset cleaning).

    The target is **3-class** as of 2026-04-29: ``{0: Fail, 1: Pass,
    2: Distinction}``. The previous binary collapse (Distinction → Pass) was
    discarded per the user's decision to support contribution C2 (multi-class
    fairness).
    """
    bundle = load_oulad()

    # : every key present.
    assert set(bundle.keys()) == INV002_KEYS

    # Types .
    assert isinstance(bundle["X_train"], pd.DataFrame)
    assert isinstance(bundle["X_test"], pd.DataFrame)
    assert isinstance(bundle["y_train"], pd.Series)
    assert isinstance(bundle["y_test"], pd.Series)
    assert isinstance(bundle["A_train"], pd.DataFrame)
    assert isinstance(bundle["A_test"], pd.DataFrame)

    # Single-sensitive: A_*.columns == ["gender"].
    assert bundle["A_train"].columns.tolist() == ["gender"]
    assert bundle["A_test"].columns.tolist() == ["gender"]

    # Total cleaned row count is fixed at 21,562 per Le Quy 2022 §3.4.2.
    total = len(bundle["X_train"]) + len(bundle["X_test"])
    assert total == 21_562, f"Expected 21,562 rows total (cleaned), got {total}"
    assert len(bundle["y_train"]) == len(bundle["X_train"])
    assert len(bundle["y_test"]) == len(bundle["X_test"])
    assert len(bundle["A_train"]) == len(bundle["X_train"])
    assert len(bundle["A_test"]) == len(bundle["X_test"])

    # 80/20 stratified split (allow ±0.1%).
    test_frac = len(bundle["X_test"]) / total
    assert abs(test_frac - 0.20) < 0.001, (
        f"Expected ~20% test split, got {test_frac:.4f} (total={total})"
    )

    # 3-class target and class metadata. All three classes must be present
    # in BOTH train and test ( lesson: a too-lax issubset assertion
    # masked an ACS bug; we now require strict equality).
    assert bundle["n_classes"] == 3
    assert bundle["class_names"] == ["Fail", "Pass", "Distinction"]
    assert set(bundle["y_train"].unique()) == {0, 1, 2}, (
        f"y_train must contain all 3 classes; got {set(bundle['y_train'].unique())}"
    )
    assert set(bundle["y_test"].unique()) == {0, 1, 2}, (
        f"y_test must contain all 3 classes; got {set(bundle['y_test'].unique())}"
    )

    # Single-sensitive: sensitive_names is exactly ["gender"].
    assert bundle["sensitive_names"] == ["gender"]

    # Train and test indices are disjoint (no leakage).
    assert set(bundle["X_train"].index).isdisjoint(set(bundle["X_test"].index))

def test_load_oulad_4class() -> None:
    """``load_oulad_4class`` returns a 4-class OULAD bundle with Withdrawn retained.

    Sibling of :func:`test_oulad`: this variant of the OULAD loader skips the
    ``Withdrawn`` filter applied by :func:`load_oulad`, so the bundle has
    ``n_classes == 4`` and contains more rows than the 3-class variant.
    Class ordering is fixed at ``["Withdrawn", "Fail", "Pass", "Distinction"]``
    (indices 0..3) per the loader's docstring; ordering propagates to audit
    CSVs and ``filter_classes`` configuration so it is part of the public
    contract.
    """
    bundle = load_oulad_4class()

    # : every key present.
    assert set(bundle.keys()) == INV002_KEYS

    # Types .
    assert isinstance(bundle["X_train"], pd.DataFrame)
    assert isinstance(bundle["X_test"], pd.DataFrame)
    assert isinstance(bundle["y_train"], pd.Series)
    assert isinstance(bundle["y_test"], pd.Series)
    assert isinstance(bundle["A_train"], pd.DataFrame)
    assert isinstance(bundle["A_test"], pd.DataFrame)

    # 4-class metadata + ordering.
    assert bundle["n_classes"] == 4
    assert bundle["class_names"] == ["Withdrawn", "Fail", "Pass", "Distinction"]

    # All four classes must appear in train and test (mirrors the strict
    # equality used in test_oulad to avoid the  issubset trap).
    assert set(bundle["y_train"].unique()) == {0, 1, 2, 3}, (
        f"y_train must contain all 4 classes; got {set(bundle['y_train'].unique())}"
    )
    assert set(bundle["y_test"].unique()) == {0, 1, 2, 3}, (
        f"y_test must contain all 4 classes; got {set(bundle['y_test'].unique())}"
    )

    # Withdrawn rows (class 0) are retained --- the distinguishing
    # behaviour of this variant vs load_oulad.
    n_withdrawn_train = int((bundle["y_train"] == 0).sum())
    n_withdrawn_test = int((bundle["y_test"] == 0).sum())
    assert n_withdrawn_train > 0, "Withdrawn rows must be present in train"
    assert n_withdrawn_test > 0, "Withdrawn rows must be present in test"

    # Single-sensitive: gender column matches load_oulad.
    assert bundle["A_train"].columns.tolist() == ["gender"]
    assert bundle["A_test"].columns.tolist() == ["gender"]
    assert bundle["sensitive_names"] == ["gender"]

    # 80/20 stratified split sanity (allow +/- 0.1%).
    total_4c = len(bundle["X_train"]) + len(bundle["X_test"])
    test_frac = len(bundle["X_test"]) / total_4c
    assert abs(test_frac - 0.20) < 0.001, (
        f"Expected ~20% test split, got {test_frac:.4f} (total={total_4c})"
    )

    # 4-class variant must have strictly more rows than the 3-class loader
    # (Withdrawn rows are the only delta).
    baseline_3c = load_oulad()
    total_3c = len(baseline_3c["X_train"]) + len(baseline_3c["X_test"])
    assert total_4c > total_3c, (
        f"4-class variant must have more rows than 3-class "
        f"(4c={total_4c}, 3c={total_3c})"
    )

    # The number of new rows equals the number of Withdrawn rows.
    n_withdrawn = n_withdrawn_train + n_withdrawn_test
    assert total_4c - total_3c == n_withdrawn, (
        f"Row delta ({total_4c - total_3c}) must equal Withdrawn count "
        f"({n_withdrawn}) --- only Withdrawn rows distinguish the variants."
    )

    # Train and test indices are disjoint (no leakage).
    assert set(bundle["X_train"].index).isdisjoint(set(bundle["X_test"].index))

# ---------------------------------------------------------------------------
#  CLI tests: _main(), _build_all(), _verify_checksums(),
# _record_checksums(), _sha256_file(), _parse_checksums_md()
# ---------------------------------------------------------------------------

def test_sha256_file(tmp_path: pathlib.Path) -> None:
    """``_sha256_file`` returns a 64-hex-char SHA-256 digest for any file."""
    import hashlib

    content = b"hello thesis\n"
    p = tmp_path / "test.bin"
    p.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert _sha256_file(p) == expected
    assert len(_sha256_file(p)) == 64

def test_parse_checksums_md_empty(tmp_path: pathlib.Path) -> None:
    """``_parse_checksums_md`` returns an empty list for a missing file."""
    missing = tmp_path / "nope.md"
    assert _parse_checksums_md(missing) == []

def test_parse_checksums_md_with_rows(tmp_path: pathlib.Path) -> None:
    """``_parse_checksums_md`` parses a minimal markdown table correctly."""
    content = textwrap.dedent("""\
        # Dataset CHECKSUMS

        | Dataset ID | File | sha256 | Source URL | Retrieved | Licence note |
        |---|---|---|---|---|---|
        | D1 | `data/raw/foo/bar.csv` | abc123 | https://example.com | 2026-01-01 | MIT |
        | D2 | `data/raw/baz/qux.csv` | (pending) | https://example.com | (pending) | CC-BY |
    """)
    md = tmp_path / "CHECKSUMS.md"
    md.write_text(content)
    rows = _parse_checksums_md(md)
    assert len(rows) == 2
    assert rows[0]["File"] == "`data/raw/foo/bar.csv`"
    assert rows[0]["sha256"] == "abc123"
    assert rows[1]["sha256"] == "(pending)"

def test_build_all_dry_run_exit_zero(tmp_path: pathlib.Path) -> None:
    """``_build_all(dry_run=True, ...)`` prints a summary and returns 0.

    Verifies the exit-gate for : ``python -m procedural_fair_hr.data_loaders
    --build-all --dry-run`` must exit 0 without touching the filesystem.
    """
    # Point to a fake project_root; dry-run must not write anything.
    rc = _build_all(dry_run=True, project_root=tmp_path)
    assert rc == 0
    # Nothing should have been written (no parquet files created).
    assert list(tmp_path.rglob("*.parquet")) == []

def test_verify_checksums_no_rows(tmp_path: pathlib.Path) -> None:
    """``_verify_checksums`` returns 0 when CHECKSUMS.md has no table rows."""
    checksums_md = tmp_path / "data" / "CHECKSUMS.md"
    checksums_md.parent.mkdir(parents=True)
    checksums_md.write_text("# Dataset CHECKSUMS\n\nNo table here.\n")
    rc = _verify_checksums(project_root=tmp_path)
    assert rc == 0

def test_verify_checksums_all_pending(tmp_path: pathlib.Path) -> None:
    """``_verify_checksums`` skips rows with ``(pending)`` and returns 0."""
    content = textwrap.dedent("""\
        # Dataset CHECKSUMS

        | Dataset ID | File | sha256 | Source URL | Retrieved | Licence note |
        |---|---|---|---|---|---|
        | D1 | `data/raw/foo/bar.csv` | (pending) | https://example.com | (pending) | MIT |
    """)
    checksums_md = tmp_path / "data" / "CHECKSUMS.md"
    checksums_md.parent.mkdir(parents=True)
    checksums_md.write_text(content)
    rc = _verify_checksums(project_root=tmp_path)
    assert rc == 0

def test_verify_checksums_pass(tmp_path: pathlib.Path) -> None:
    """``_verify_checksums`` returns 0 when all checksums match."""
    import hashlib

    raw_file = tmp_path / "data" / "raw" / "foo" / "bar.csv"
    raw_file.parent.mkdir(parents=True)
    raw_file.write_bytes(b"col1,col2\n1,2\n")
    digest = hashlib.sha256(b"col1,col2\n1,2\n").hexdigest()

    content = textwrap.dedent(f"""\
        # Dataset CHECKSUMS

        | Dataset ID | File | sha256 | Source URL | Retrieved | Licence note |
        |---|---|---|---|---|---|
        | D1 | `data/raw/foo/bar.csv` | {digest} | https://example.com | 2026-01-01 | MIT |
    """)
    checksums_md = tmp_path / "data" / "CHECKSUMS.md"
    checksums_md.parent.mkdir(parents=True, exist_ok=True)
    checksums_md.write_text(content)
    rc = _verify_checksums(project_root=tmp_path)
    assert rc == 0

def test_verify_checksums_fail(tmp_path: pathlib.Path) -> None:
    """``_verify_checksums`` returns 1 when a checksum mismatches."""
    raw_file = tmp_path / "data" / "raw" / "foo" / "bar.csv"
    raw_file.parent.mkdir(parents=True)
    raw_file.write_bytes(b"col1,col2\n1,2\n")

    content = textwrap.dedent("""\
        # Dataset CHECKSUMS

        | Dataset ID | File | sha256 | Source URL | Retrieved | Licence note |
        |---|---|---|---|---|---|
        | D1 | `data/raw/foo/bar.csv` | deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef | https://example.com | 2026-01-01 | MIT |
    """)
    checksums_md = tmp_path / "data" / "CHECKSUMS.md"
    checksums_md.parent.mkdir(parents=True, exist_ok=True)
    checksums_md.write_text(content)
    rc = _verify_checksums(project_root=tmp_path)
    assert rc == 1

def test_record_checksums_updates_digest(tmp_path: pathlib.Path) -> None:
    """``_record_checksums`` computes and writes SHA-256 into CHECKSUMS.md."""
    import hashlib

    raw_file = tmp_path / "data" / "raw" / "foo" / "bar.csv"
    raw_file.parent.mkdir(parents=True)
    raw_file.write_bytes(b"a,b\n1,2\n")
    expected_digest = hashlib.sha256(b"a,b\n1,2\n").hexdigest()

    content = textwrap.dedent("""\
        # Dataset CHECKSUMS

        | Dataset ID | File | sha256 | Source URL | Retrieved | Licence note |
        |---|---|---|---|---|---|
        | D1 | `data/raw/foo/bar.csv` | (pending) | https://example.com | (pending) | MIT |
    """)
    checksums_md = tmp_path / "data" / "CHECKSUMS.md"
    checksums_md.parent.mkdir(parents=True, exist_ok=True)
    checksums_md.write_text(content)

    rc = _record_checksums(project_root=tmp_path)
    assert rc == 0
    updated = checksums_md.read_text()
    assert expected_digest in updated

def test_main_dry_run_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_main()`` called with ``--build-all --dry-run`` returns 0."""
    import sys

    monkeypatch.setattr(sys, "argv", ["procedural_fair_hr.data_loaders", "--build-all", "--dry-run"])
    rc = _main()
    assert rc == 0

def test_main_no_args_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_main()`` with no action flags prints help and returns 0."""
    import sys

    monkeypatch.setattr(sys, "argv", ["procedural_fair_hr.data_loaders"])
    rc = _main()
    assert rc == 0

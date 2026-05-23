#!/usr/bin/env python3
"""Verify SHA-256 checksums of raw dataset files.

Thin wrapper that delegates to
``python -m procedural_fair_hr.data_loaders --verify-checksums``.
Returns non-zero on any mismatch so ``make data`` aborts cleanly.
"""
from __future__ import annotations

import pathlib
import sys

# Ensure the project source tree is on sys.path when this script is run
# directly (i.e., not via ``python -m``). When the package is installed in
# editable mode (``pip install -e .``) this is unnecessary, but the explicit
# path keeps the script usable in a fresh clone before installation.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from procedural_fair_hr.data_loaders import _verify_checksums  # noqa: E402


def main() -> int:
    return _verify_checksums(project_root=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())

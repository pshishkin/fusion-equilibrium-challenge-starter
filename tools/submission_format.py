"""Submission array dtypes and the cache identity of the exported format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.lib.npyio import NpzFile

PSI_DTYPE = np.dtype("float32")
SCALAR_DTYPE = np.dtype("float32")
EXPORT_FORMAT_VERSION = "psirz-float32-scalars-float32-v1"


def require_submission_dtypes(packed: NpzFile, path: Path) -> None:
    """Reject old float16 caches instead of quietly promoting their rounded maps."""
    for key in packed.files:
        if key.endswith("_psirz"):
            expected = PSI_DTYPE
        elif key.endswith("_q95") or key.endswith("_betaN"):
            expected = SCALAR_DTYPE
        else:
            raise ValueError(f"{path}: unexpected prediction key {key!r}")
        got = packed[key].dtype
        if got != expected:
            raise ValueError(f"{path}: {key} dtype {got}, expected {expected} "
                             f"({EXPORT_FORMAT_VERSION})")


def check_file(path: Path) -> None:
    with np.load(path, allow_pickle=False) as packed:
        require_submission_dtypes(packed, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        check_file(path)
        print(f"{path}: {EXPORT_FORMAT_VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

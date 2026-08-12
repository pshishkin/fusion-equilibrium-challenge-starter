#!/usr/bin/env python3
"""
Same as eda.py, but for the MAST public test split. Run with:

    uv run python my_experiments/eda_test_mast.py

A different machine: a spherical tokamak with its own coil set and its own grid, so the
column list differs from DIII-D.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from my_experiments.eda import dump

if __name__ == "__main__":
    dump("mast_public_test")

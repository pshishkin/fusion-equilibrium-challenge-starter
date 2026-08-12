#!/usr/bin/env python3
"""
Same as eda.py, but for the DIII-D public test split. Run with:

    uv run python my_experiments/eda_test_diii_d.py

No targets here — efit_psirz / efit_q95 / efit_beta_n and the other efit_* columns are
withheld. The difference against the training split's column list is exactly what you predict.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from my_experiments.eda import dump

if __name__ == "__main__":
    dump("diii_d_public_test")

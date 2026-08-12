#!/usr/bin/env python3
"""
Same as eda.py, but for the DIII-D public test split. Run with:

    uv run python my_experiments/eda_test_diii_d.py

No targets here — efit_psirz / efit_q95 / efit_beta_n and the other efit_* columns are
withheld. The difference against the training split's column list is exactly what you predict.
"""
from eda import dump

if __name__ == "__main__":
    dump("diii_d_public_test")

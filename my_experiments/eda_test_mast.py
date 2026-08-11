#!/usr/bin/env python3
"""
То же, что eda.py, но для публичного теста MAST. Запуск:

    uv run python my_experiments/eda_test_mast.py

Другая машина: сферический токамак, свой набор катушек и своя сетка,
поэтому и список колонок отличается от DIII-D.
"""
from eda import dump

if __name__ == "__main__":
    dump("mast_public_test")

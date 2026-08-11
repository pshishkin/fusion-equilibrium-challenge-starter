#!/usr/bin/env python3
"""
То же, что eda.py, но для публичного теста DIII-D. Запуск:

    uv run python my_experiments/eda_test_diii_d.py

Здесь целей нет — колонки efit_psirz / efit_q95 / efit_beta_n и прочие efit_* withheld.
Разница со списком колонок обучающего сплита и есть то, что вы предсказываете.
"""
from eda import dump

if __name__ == "__main__":
    dump("diii_d_public_test")

#!/usr/bin/env python3
"""
EDA-песочница: обучающий сплит DIII-D. Запуск:

    uv run python my_experiments/eda.py

Один parquet — это один шот, одна строка, в каждой ячейке лежит целый массив.
Соседние файлы делают то же самое для тестовых сплитов:
    eda_test_diii_d.py, eda_test_mast.py
"""
from pathlib import Path

import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "downloaded_huggingface" / "hf_dataset" / "data"
SHOT = 0          # индекс файла в отсортированном списке

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 70)
pd.set_option("display.width", 200)


def shape(v):
    """Спускаемся по вложенности, пока это не строка и не скаляр."""
    s = []
    while hasattr(v, "__len__") and not isinstance(v, str) and len(v):
        s.append(len(v))
        v = v[0]
    return tuple(s)


def dump(config: str, shot: int = SHOT) -> None:
    """Печатает один шот конфига транспонированным, со столбцом форм."""
    data_dir = DATA_ROOT / config
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(
            f"Нет parquet-файлов в {data_dir}. Скачать:\n"
            f'  hf download Sophelio/fusion-equilibrium-challenge --repo-type dataset '
            f'--local-dir {DATA_ROOT.parent} --include "data/{config}/*"'
        )
    file = files[shot]
    df = pd.read_parquet(file)
    t = df.T
    t["shape"] = [shape(v) for v in df.iloc[0]]

    print(f"{config}: {len(files)} шотов")
    print(f"{file.name}   {df.shape[0]} строк x {df.shape[1]} колонок\n")
    print(t)


if __name__ == "__main__":
    dump("diii_d_train")

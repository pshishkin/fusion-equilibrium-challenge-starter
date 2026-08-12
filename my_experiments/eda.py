#!/usr/bin/env python3
"""
EDA scratchpad: the DIII-D training split. Run with:

    uv run python my_experiments/eda.py

One parquet is one shot, one row, and every cell holds a whole array.
Sibling files do the same for the test splits:
    eda_test_diii_d.py, eda_test_mast.py
"""
from pathlib import Path

import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "downloaded_huggingface" / "hf_dataset" / "data"
SHOT = 0          # index of the file in the sorted list

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 70)
pd.set_option("display.width", 200)


def shape(v):
    """Descend through the nesting while it is neither a string nor a scalar."""
    s = []
    while hasattr(v, "__len__") and not isinstance(v, str) and len(v):
        s.append(len(v))
        v = v[0]
    return tuple(s)


def dump(config: str, shot: int = SHOT) -> None:
    """Print one shot of the config transposed, with a column of shapes."""
    data_dir = DATA_ROOT / config
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(
            f"No parquet files in {data_dir}. Download them with:\n"
            f'  hf download Sophelio/fusion-equilibrium-challenge --repo-type dataset '
            f'--local-dir {DATA_ROOT.parent} --include "data/{config}/*"'
        )
    file = files[shot]
    df = pd.read_parquet(file)
    t = df.T
    t["shape"] = [shape(v) for v in df.iloc[0]]

    print(f"{config}: {len(files)} shots")
    print(f"{file.name}   {df.shape[0]} rows x {df.shape[1]} columns\n")
    print(t)


if __name__ == "__main__":
    dump("diii_d_train")

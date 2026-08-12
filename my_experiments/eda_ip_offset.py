#!/usr/bin/env python3
"""
Is the plasma-current time offset a constant? Run with:

    uv run python my_experiments/eda_ip_offset.py

Short answer: it is not an offset at all. `magnetics_plasma_current_times` is one and the same
array in every shot — a template — while `magnetics_time` differs per shot. So the Ip axis was not
shifted, it was substituted, and the apparent shift is just the distance between that template and
whatever the shot's real timing was.

The script prints the evidence in four steps and ends with the correction that follows from it.
"""
from __future__ import annotations

import hashlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from my_experiments.baseline_model import IP_ON, sorted_shots  # noqa: E402

N_SHOTS = 150


def main() -> int:
    files = sorted_shots()
    sel = [files[i] for i in np.linspace(0, len(files) - 1, N_SHOTS).astype(int)]

    ip_axis_hashes, mag_axis_hashes = Counter(), Counter()
    rows = []
    for path in tqdm(sel, desc="reading shots", unit="shot"):
        r = pd.read_parquet(path).iloc[0]
        te = np.asarray(r["efit_times"], np.float64)
        tip = np.asarray(r["magnetics_plasma_current_times"], np.float64)
        ip = np.asarray(r["magnetics_plasma_current"], np.float64)
        tm = np.asarray(r["magnetics_time"], np.float64)

        ip_axis_hashes[hashlib.md5(tip.tobytes()).hexdigest()[:8]] += 1
        mag_axis_hashes[hashlib.md5(tm.tobytes()).hexdigest()[:8]] += 1

        peak = np.abs(ip).max()

        def coverage(axis: np.ndarray) -> float:
            """Fraction of EFIT frames that land where current is actually flowing."""
            return float((np.abs(np.interp(te, axis, ip)) > IP_ON * peak).mean())

        rows.append(dict(
            step=float(np.median(np.diff(tm))),
            mag_start=float(tm[0]),
            ip_start=float(tip[0]),
            cov_raw=coverage(tip),
            cov_reorigin=coverage(tm[0] + (tip - tip[0])),
        ))

    df = pd.DataFrame(rows)
    fast = df["step"] < 0.2          # the 0.05 ms acquisition
    slow = ~fast

    print(f"\n{'=' * 78}\n1. Is the Ip axis per shot, or one shared array?\n{'=' * 78}")
    print(f"  distinct magnetics_plasma_current_times arrays: {len(ip_axis_hashes)}")
    for h, n in ip_axis_hashes.most_common(3):
        print(f"     {h}: {n} shots")
    print(f"  distinct magnetics_time arrays:                 {len(mag_axis_hashes)} "
          f"(of {len(sel)} shots)")
    print(f"  -> the Ip axis is a template; the magnetics axis is genuinely per shot")

    print(f"\n{'=' * 78}\n2. Two acquisition configurations\n{'=' * 78}")
    print(f"  {'group':26s} {'shots':>6} {'magnetics_time[0], ms':>22} {'ip_times[0], ms':>17}")
    for lab, m in [("step 0.50 ms", slow), ("step 0.05 ms", fast)]:
        print(f"  {lab:26s} {m.sum():6d} {df.loc[m, 'mag_start'].median():22.0f} "
              f"{df.loc[m, 'ip_start'].median():17.4f}")

    print(f"\n{'=' * 78}\n3. Do the EFIT frames see any current?\n{'=' * 78}")
    print(f"  {'group':26s} {'as shipped':>12} {'re-origined at magnetics_time[0]':>34}")
    for lab, m in [("step 0.50 ms", slow), ("step 0.05 ms", fast)]:
        print(f"  {lab:26s} {df.loc[m, 'cov_raw'].median():12.2f} "
              f"{df.loc[m, 'cov_reorigin'].median():34.2f}")
    print(f"\n  worst case among the 0.05 ms shots, re-origined: "
          f"{df.loc[fast, 'cov_reorigin'].min():.2f}")
    print(f"  worst case among the 0.50 ms shots, re-origined: "
          f"{df.loc[slow, 'cov_reorigin'].min():.2f}  <- leave those alone")

    print(f"\n{'=' * 78}\n4. So the 'offset' is not a constant\n{'=' * 78}")
    implied = df.loc[fast, "ip_start"] - df.loc[fast, "mag_start"]
    print(f"  implied shift = ip_times[0] - magnetics_time[0], over the 0.05 ms group:")
    for q in (0, 25, 50, 75, 100):
        print(f"     percentile {q:3d}: {np.percentile(implied, q):8.0f} ms")
    print(f"  it varies because the template is fixed while each shot's acquisition is not.")
    print(f"  Quoting '~3 s' is a median, not a constant to subtract.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Read-only audit of a proposed two-signal, four-descriptor raw-current bank.

Run: .venv/bin/python my_experiments/diagnose_fast_magnetics.py
Only input columns are read. The public-test rows contain no requested labels.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import DEFAULT_LOCAL_DATA_DIR  # noqa: E402
from my_experiments.baseline_model import (  # noqa: E402
    RAW_DERIV_HALF_MS,
    _interp_derivatives,
    align_ip_times,
    shot_key,
)
from my_experiments.data_check import MANIFEST, check_dataset  # noqa: E402
from toolkit.parallel import pimap, release  # noqa: E402
from toolkit.records import file_hash  # noqa: E402

SIGNALS = ("ECOILA", "plasma_current")
HALFWIDTHS_MS = (0.2, 5.0, 20.0)
SECOND_HALFWIDTH_MS = 5.0
COLUMNS = (
    "efit_times", "magnetics_time", "magnetics_plasma_current_times",
    "magnetics_ECOILA", "magnetics_plasma_current",
)
COUNTS = {"diii_d_train": 100, "diii_d_public_test": 20}
OUT = ROOT / "experiments/fast-signal-audit.json"


def _old_raw_derivative(efit: np.ndarray, t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The existing raw one-millisecond formula for one selected signal."""
    step = float(np.median(np.diff(t)))
    k = max(1, round(RAW_DERIV_HALF_MS / step))
    idx = np.arange(len(t))
    lo = np.clip(idx - k, 0, len(t) - 1)
    hi = np.clip(idx + k, 0, len(t) - 1)
    dt = t[hi] - t[lo]
    d = np.where(dt > 0, (v[hi] - v[lo]) / np.maximum(dt, 1e-30), 0.0)
    return np.interp(efit, t, d).astype(np.float32).astype(np.float64)


def _describe_signal(efit: np.ndarray, t: np.ndarray, v: np.ndarray) -> dict[str, Any]:
    """Measure candidate columns only where both raw-window endpoints exist."""
    if len(t) != len(v):
        raise ValueError(f"signal time/value lengths differ: {len(t)} versus {len(v)}")
    if len(t) < 3 or not np.isfinite(t).all() or not np.isfinite(v).all():
        raise ValueError(f"signal has {len(t)} samples or nonfinite time/value samples")
    dt = np.diff(t)
    duplicates = int(np.count_nonzero(dt == 0))
    decreasing = int(np.count_nonzero(dt < 0))
    if duplicates or decreasing:
        return {"duplicates": duplicates, "decreasing": decreasing, "n_raw": len(t)}
    # Match the production interpolation's scipy call and float32 source values exactly.
    level = interp1d(t, v.astype(np.float32), kind="linear", fill_value="extrapolate",
                     bounds_error=False)(efit).astype(np.float32)
    old_raw = _old_raw_derivative(efit, t, v)
    old_interp = _interp_derivatives(level[:, None], efit)[:, 0].astype(np.float64)
    base = np.column_stack([level.astype(np.float64), old_raw, old_interp])
    bank: dict[str, Any] = {}
    for w in HALFWIDTHS_MS:
        valid = (efit - w >= t[0]) & (efit + w <= t[-1])
        vals = (np.interp(efit[valid] + w, t, v)
                - np.interp(efit[valid] - w, t, v)) / (2.0 * w)
        key = f"d1_{w:g}ms"
        bank[key] = {"valid": valid, "values": vals}
    w = SECOND_HALFWIDTH_MS
    valid = (efit - w >= t[0]) & (efit + w <= t[-1])
    vals = ((np.interp(efit[valid] + w, t, v) - 2 * np.interp(efit[valid], t, v)
             + np.interp(efit[valid] - w, t, v)) / w**2)
    bank[f"d2_{w:g}ms"] = {"valid": valid, "values": vals}
    return {
        "duplicates": 0, "decreasing": 0, "n_raw": len(t),
        "median_step_ms": float(np.median(dt)),
        "step_min_ms": float(dt.min()), "step_max_ms": float(dt.max()),
        "efit_before_raw": int(np.count_nonzero(efit < t[0])),
        "efit_after_raw": int(np.count_nonzero(efit > t[-1])),
        "base": base, "bank": bank,
    }


def _read_one(task: tuple[str, Path]) -> dict[str, Any]:
    """Read the five requested input columns for one shot."""
    config, path = task
    row = pd.read_parquet(path, columns=list(COLUMNS)).iloc[0]
    for col in COLUMNS:
        if col not in row:
            raise ValueError(f"{path}: missing {col}")
    efit = np.asarray(row["efit_times"], dtype=np.float64)
    shared = np.asarray(row["magnetics_time"], dtype=np.float64)
    ip_shipped = np.asarray(row["magnetics_plasma_current_times"], dtype=np.float64)
    ip_values = np.asarray(row["magnetics_plasma_current"], dtype=np.float32)
    if len(efit) < 2 or not np.isfinite(efit).all() or np.any(np.diff(efit) <= 0):
        raise ValueError(f"{path}: efit_times length={len(efit)} or nonfinite/nonmonotonic")
    if len(shared) != len(row["magnetics_ECOILA"]):
        raise ValueError(f"{path}: magnetics_time length={len(shared)} versus "
                         f"ECOILA length={len(row['magnetics_ECOILA'])}")
    if len(ip_shipped) != len(ip_values):
        raise ValueError(f"{path}: plasma-current time length={len(ip_shipped)} versus "
                         f"value length={len(ip_values)}")
    ip_axis = align_ip_times(efit, ip_shipped, ip_values, shared)
    signals = {
        "ECOILA": _describe_signal(efit, shared,
                                    np.asarray(row["magnetics_ECOILA"], dtype=np.float32)),
        "plasma_current": _describe_signal(efit, ip_axis, ip_values),
    }
    return {
        "config": config, "file": path.name, "n_efit": len(efit),
        "efit_times": efit,
        "regular_interior": np.r_[False, np.isclose(np.diff(efit)[:-1], 20.0, atol=0.05)
                                  & np.isclose(np.diff(efit)[1:], 20.0, atol=0.05), False],
        "ip_reorigined": bool(ip_axis is not ip_shipped), "signals": signals,
    }


def _column_stats(x: np.ndarray, base: np.ndarray) -> dict[str, Any]:
    """Correlation and linear novelty relative to old level/raw/interp columns."""
    if len(x) < 4 or np.std(x) == 0:
        raise ValueError(f"bank column has {len(x)} values or zero variance")
    corr = [float(np.corrcoef(x, base[:, i])[0, 1]) for i in range(3)]
    if not np.isfinite(corr).all():
        raise ValueError("nonfinite feature correlation")
    scaled = (base - base.mean(0)) / base.std(0)
    y = x - x.mean()
    fitted = scaled @ np.linalg.lstsq(scaled, y, rcond=None)[0]
    novelty = float(np.mean((y - fitted) ** 2) / np.mean(y**2))
    return {
        "n": len(x), "mean": float(np.mean(x)), "std": float(np.std(x)),
        "quantiles": [float(q) for q in np.quantile(x, [0, .01, .5, .99, 1])],
        "corr_level_raw1_interp": corr, "linear_residual_variance_fraction": novelty,
    }


def _aggregate(rows: list[dict[str, Any]], config: str) -> dict[str, Any]:
    """Aggregate all valid frames without replacing unavailable raw windows."""
    selected = [r for r in rows if r["config"] == config]
    out: dict[str, Any] = {
        "files": [r["file"] for r in selected],
        "n_efit_frames": sum(r["n_efit"] for r in selected),
        "ip_reorigined_shots": sum(r["ip_reorigined"] for r in selected),
        "signals": {},
    }
    for signal in SIGNALS:
        parts = [r["signals"][signal] for r in selected]
        duplicate = sum(p["duplicates"] for p in parts)
        decreasing = sum(p["decreasing"] for p in parts)
        if duplicate or decreasing:
            raise ValueError(f"{config}/{signal}: {duplicate} duplicate and {decreasing} "
                             "decreasing raw steps; no rows may be silently skipped")
        s: dict[str, Any] = {
            "raw_samples": sum(p["n_raw"] for p in parts),
            "median_raw_step_ms_by_shot": [p["median_step_ms"] for p in parts],
            "raw_step_range_ms": [min(p["step_min_ms"] for p in parts),
                                  max(p["step_max_ms"] for p in parts)],
            "duplicate_steps": duplicate, "decreasing_steps": decreasing,
            "efit_before_raw": sum(p["efit_before_raw"] for p in parts),
            "efit_after_raw": sum(p["efit_after_raw"] for p in parts),
            "bank": {},
        }
        for key in parts[0]["bank"]:
            valid = [p["bank"][key]["valid"] for p in parts]
            vals = np.concatenate([p["bank"][key]["values"] for p in parts])
            bases = np.concatenate([p["base"][m] for p, m in zip(parts, valid, strict=True)])
            regular = np.concatenate([r["regular_interior"][m]
                                      for r, m in zip(selected, valid, strict=True)])
            lagged: list[np.ndarray] = []
            leading: list[np.ndarray] = []
            for row, part, mask in zip(selected, parts, valid, strict=True):
                frame_pair = (mask[:-1] & mask[1:]
                              & np.isclose(np.diff(row["efit_times"]), 20.0, atol=0.05))
                full = np.full(len(mask), np.nan)
                full[mask] = part["bank"][key]["values"]
                lagged.append(full[:-1][frame_pair])
                leading.append(full[1:][frame_pair])
            lag_x = np.concatenate(lagged)
            lag_y = np.concatenate(leading)
            s["bank"][key] = {
                "outside_window_frames": sum(len(m) - int(np.count_nonzero(m)) for m in valid),
                "adjacent_20ms_pairs": len(lag_x),
                "adjacent_20ms_correlation": float(np.corrcoef(lag_x, lag_y)[0, 1]),
                "regular_20ms_interior": _column_stats(vals[regular], bases[regular]),
                "edge_or_gap": _column_stats(vals[~regular], bases[~regular]),
                **_column_stats(vals, bases),
            }
        out["signals"][signal] = s
    return out


def main() -> None:
    """Select fixed hash-ordered shots, validate the corpus, audit, and save compact JSON."""
    manifest = json.loads(MANIFEST.read_text())
    tasks: list[tuple[str, Path]] = []
    identities: dict[str, Any] = {}
    for config, count in COUNTS.items():
        identities[config] = check_dataset(DEFAULT_LOCAL_DATA_DIR, config)
        folder = DEFAULT_LOCAL_DATA_DIR / "data" / config
        paths = sorted(folder.glob("*.parquet"), key=shot_key)[:count]
        if len(paths) != count:
            raise ValueError(f"{folder}: selected {len(paths)} paths, expected {count}")
        tasks.extend((config, p) for p in paths)
    hashes = {f"{config}/{p.name}": manifest["configs"][config][p.name][1]
              for config, p in tasks}
    actual_hashes = {f"{config}/{p.name}": file_hash(p) for config, p in tasks}
    if hashes != actual_hashes:
        bad = [name for name in hashes if hashes[name] != actual_hashes[name]]
        raise ValueError(f"selected parquet SHA-256 mismatch for {bad}")
    rows = list(tqdm(pimap(_read_one, tasks, jobs=2), total=len(tasks), desc="fast-signal audit"))
    release()
    result = {
        "hypothesis": "two signals times three centered first differences at 0.2/5/20 ms "
                      "plus a centered second difference at 5 ms",
        "scope": "input-only; descriptive audit; no model fit or score",
        "selection": "first 100 train and first 20 public-test files ordered by salt-0 shot_key",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__), ROOT / "my_experiments/baseline_model.py"]},
        "dataset_identity": identities,
        "selected_verified_sha256": actual_hashes,
        "results": {config: _aggregate(rows, config) for config in COUNTS},
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

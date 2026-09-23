"""Audit source-li preservation on the fixed OOF100 maps without EFIT labels."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# The byte-pinned vendored scorer uses flat imports within fusion_scoring.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fusion_scoring"))

from fusion_scoring import common, derive, lcfs
from toolkit.parallel import pimap, release
from toolkit.records import file_hash, write_json

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "experiments/oof-li-discrepancy-audit.json"


def summarize_absolute(values: np.ndarray) -> dict[str, float]:
    """Describe a finite vector of absolute source-to-final li differences."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or len(arr) == 0 or not np.isfinite(arr).all() or np.any(arr < 0):
        raise ValueError(f"li absolute differences shape/size/finiteness/min "
                         f"{arr.shape}/{arr.size}/{int((~np.isfinite(arr)).sum())}/"
                         f"{float(arr.min()) if arr.size else 'empty'}, expected finite "
                         "nonnegative one-dimensional values")
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
        "fraction_at_most_1e_minus_4": float(np.mean(arr <= 1e-4)),
        "fraction_at_most_1e_minus_3": float(np.mean(arr <= 1e-3)),
        "fraction_at_most_1e_minus_2": float(np.mean(arr <= 1e-2)),
    }


def _one(task: tuple[str, int, int, str, str, str, str]) -> dict[str, Any]:
    name, index, frames, data_s, source_s, final_s, mask_s = task
    path = Path(data_s) / "data/diii_d_train" / name
    row = pd.read_parquet(path, columns=["efit_grid_R", "efit_grid_Z"]).iloc[0]
    R = np.asarray(row["efit_grid_R"], dtype=np.float64)
    Z = np.asarray(row["efit_grid_Z"], dtype=np.float64)
    with np.load(mask_s) as saved:
        mask = saved["mask_coarse"].astype(bool)
        if not np.array_equal(R, saved["grid_R"]) or not np.array_equal(Z, saved["grid_Z"]):
            raise ValueError(f"{path}: physical grid differs from pinned scorer mask")
    key = f"shot_{index:04d}_psirz"
    with np.load(source_s, allow_pickle=False) as saved:
        source = np.asarray(saved[key])
    with np.load(final_s, allow_pickle=False) as saved:
        final = np.asarray(saved[key])
    if (source.shape != final.shape or source.shape != (frames, 65, 65)
            or source.dtype != final.dtype or source.dtype != np.float32
            or not np.isfinite(source).all() or not np.isfinite(final).all()):
        raise ValueError(f"{path}: source/final map shape/dtype/finiteness "
                         f"{source.shape}/{final.shape}/{source.dtype}/{final.dtype}, "
                         f"expected ({frames},65,65)/float32 both finite")
    sign = int(common.AXIS_SIGN["DIII-D"])
    source_li: list[float] = []
    final_li: list[float] = []
    for frame, (initial, changed) in enumerate(zip(source, final, strict=True)):
        values = []
        for label, psi in (("source", initial), ("full_normal", changed)):
            contour, _ = lcfs.extract_lcfs_with_sign(
                psi.astype(np.float64), R, Z, "DIII-D", mask, mask.astype(float),
                axis_sign=sign)
            li = (float("nan") if contour is None else float(
                derive.internal_inductance(psi.astype(np.float64), R, Z, "DIII-D", contour)))
            if not np.isfinite(li):
                raise ValueError(f"{path} frame {frame}: {label} scorer-native li "
                                 f"{li}, expected finite")
            values.append(li)
        source_li.append(values[0])
        final_li.append(values[1])
    return {"shot": name, "frames": frames,
            "source_li": np.asarray(source_li), "final_li": np.asarray(final_li)}


def run(output: Path, jobs: int) -> None:
    """Use only saved maps and input grids; never read any true target column."""
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError(f"{output}: read-only audit already recorded")
    spec = json.loads(SPEC.read_text())
    if file_hash(Path(__file__)) != spec["helper_sha256"]:
        raise ValueError(f"{SPEC}: read-only audit helper SHA differs from frozen spec")
    for label, record in spec["pinned_inputs"].items():
        path = ROOT / record["path"]
        if file_hash(path) != record["sha256"]:
            raise ValueError(f"{label} {path}: SHA differs from frozen audit spec")
    for label, record in spec["pinned_scorer_sources"].items():
        path = ROOT / record["path"]
        if file_hash(path) != record["sha256"]:
            raise ValueError(f"{label} {path}: scorer source SHA differs")
    old = json.loads((ROOT / spec["pinned_inputs"]["old_files_manifest"]["path"])
                     .read_text())
    campaign = json.loads((ROOT / spec["pinned_inputs"]["old_campaign"]["path"])
                          .read_text())
    names = old["pilot_train_oof_files"]
    if (len(names) != 100 or old["pilot_train_oof_ordered_sha256"]
            != spec["pilot_ordered_names_sha256"]
            or sum(campaign["pilot_input_frame_counts"]) != spec["pilot_full_frames"]
            or set(names) & (set(old["gate_files"]) | set(old["reserved_tail_files"]))):
        raise ValueError(f"{SPEC}: pilot names/frames overlap gate or tail or differ from spec")
    data = Path(campaign["data_dir"])
    source = ROOT / spec["pinned_inputs"]["source_prediction"]["path"]
    final = ROOT / spec["pinned_inputs"]["full_normal_prediction"]["path"]
    mask = ROOT / spec["pinned_scorer_sources"]["mask"]["path"]
    tasks = [(name, i, campaign["pilot_input_frame_counts"][i], str(data),
              str(source), str(final), str(mask)) for i, name in enumerate(names)]
    source_parts: list[np.ndarray] = []
    final_parts: list[np.ndarray] = []
    per_shot: list[dict[str, Any]] = []
    top: list[dict[str, Any]] = []
    try:
        for i, item in enumerate(pimap(_one, tasks, jobs)):
            if item["shot"] != names[i] or item["frames"] != tasks[i][2]:
                raise ValueError(f"{names[i]}: audit shot/frame order differs")
            initial = item["source_li"]
            changed = item["final_li"]
            difference = changed - initial
            absolute = np.abs(difference)
            source_parts.append(initial)
            final_parts.append(changed)
            per_shot.append({"shot": names[i], "frames": len(initial),
                             "mean_abs_li_change": float(np.mean(absolute)),
                             "max_abs_li_change": float(np.max(absolute)),
                             "sum_squared_li_change": float(np.sum(difference**2))})
            for frame in np.argsort(absolute)[-3:]:
                top.append({"shot": names[i], "frame": int(frame),
                            "source_li": float(initial[frame]),
                            "final_li": float(changed[frame]),
                            "signed_change": float(difference[frame]),
                            "absolute_change": float(absolute[frame])})
            if (i + 1) % 20 == 0:
                print(f"li discrepancy audited {i + 1}/100 TRAIN shots", flush=True)
    finally:
        release()
    source_li = np.concatenate(source_parts)
    final_li = np.concatenate(final_parts)
    difference = final_li - source_li
    absolute = np.abs(difference)
    if len(difference) != spec["pilot_full_frames"]:
        raise ValueError(f"li audit frames {len(difference)}, "
                         f"expected {spec['pilot_full_frames']}")
    squared = np.sort(difference**2)[::-1]
    total_sq = float(np.sum(squared))
    if total_sq <= 0:
        raise ValueError(f"source/final li squared difference {total_sq}, expected positive")
    old_native = json.loads((ROOT / spec["pinned_inputs"]["old_native_summary"]["path"])
                            .read_text())
    score = old_native["native"]
    write_json(output, {
        "classification": "same100_train_oof_source_vs_full_normal_scorer_li_discrepancy",
        "spec_sha256": file_hash(SPEC),
        "source_prediction_sha256": file_hash(source),
        "full_normal_prediction_sha256": file_hash(final),
        "pilot_ordered_names_sha256": spec["pilot_ordered_names_sha256"],
        "shots": len(names), "frames": len(difference),
        "source_li_summary": {"mean": float(np.mean(source_li)),
                              "std": float(np.std(source_li)),
                              "p05": float(np.quantile(source_li, 0.05)),
                              "p95": float(np.quantile(source_li, 0.95))},
        "final_li_summary": {"mean": float(np.mean(final_li)),
                             "std": float(np.std(final_li))},
        "signed_li_change_mean": float(np.mean(difference)),
        "absolute_li_change": summarize_absolute(absolute),
        "li_change_rms": float(np.sqrt(np.mean(difference**2))),
        "source_final_li_pearson": float(np.corrcoef(source_li, final_li)[0, 1]),
        "share_squared_difference_top_1pct_frames": float(
            np.sum(squared[:max(1, round(0.01 * len(squared)))]) / total_sq),
        "share_squared_difference_top_10pct_frames": float(
            np.sum(squared[:max(1, round(0.10 * len(squared)))]) / total_sq),
        "top_20_frame_discrepancies": sorted(top, key=lambda row: row["absolute_change"],
                                             reverse=True)[:20],
        "per_shot": per_shot,
        "native_truth_label_R2_li_context_not_recomputed": {
            "source": score["source"]["r2_cons_each"]["li"],
            "full_normal": score["full_normal_truth_assisted"]["r2_cons_each"]["li"],
        },
        "target_columns_requested": [],
        "gate200_and_tail70_shots_accessed": 0,
        "wall_seconds": time.perf_counter() - started,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 24:
        raise ValueError(f"jobs={args.jobs}, expected 1..24")
    run(args.output, args.jobs)


if __name__ == "__main__":
    main()

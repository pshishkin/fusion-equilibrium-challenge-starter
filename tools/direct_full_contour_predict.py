"""Input-only direct contour inference and fixed source-map correction.

Validation is selection-only. The original salt-0 tail remains sealed until the
frozen validation gain, fallback and contour round-trip gates all pass.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from my_experiments import baseline_model
from toolkit.parallel import pimap, release
from toolkit.records import file_hash, write_json
from tools.contour_learnability import correct_map_from_displacement
from tools.direct_full_contour_cache import ROOT, YAML, allowed_input_features, verify_campaign
from tools.direct_full_contour_train import DirectContourMLP
from tools.geometry_map_smoke import NumericalInfeasibility, _cyclic_match
from tools.geometry_source_li_compensation import _compensate_frame
from tools.historical_compare import _save
from tools.run_direct_full_contour_oracle import _target_modules, _valid_polygon

INPUT_COLUMNS = list(dict.fromkeys((
    *baseline_model.GEOMETRY_COLUMNS,
    "magnetics_time", "magnetics_plasma_current_times",
    *(f"magnetics_{signal}" for signal in baseline_model.D3D_MAGNETICS_SIGNALS),
    "thomson_chord_name", "thomson_chord_R", "thomson_chord_Z",
    "thomson_core_Te", "thomson_core_ne", "thomson_core_times",
    "thomson_edge_Te", "thomson_edge_ne", "thomson_edge_times",
)))


@functools.lru_cache(maxsize=1)
def _load_model(path_s: str, expected_sha: str) -> tuple[DirectContourMLP, np.ndarray,
                                                         np.ndarray, np.ndarray, float]:
    path = Path(path_s)
    if file_hash(path) != expected_sha:
        raise ValueError(f"{path}: direct contour checkpoint SHA changed")
    checkpoint = joblib.load(path)
    net = DirectContourMLP().cpu()
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    mean_x = checkpoint["feature_mean"].numpy()
    std_x = checkpoint["feature_std"].numpy()
    mean_y = checkpoint["target_mean"].numpy()
    scale_y = float(checkpoint["target_scale"].item())
    if (mean_x.shape != (209,) or std_x.shape != (209,) or mean_y.shape != (1024,)
            or not np.isfinite(mean_x).all() or not np.isfinite(std_x).all()
            or not np.isfinite(mean_y).all() or np.any(std_x <= 0)
            or not np.isfinite(scale_y) or scale_y <= 0):
        raise ValueError(f"{path}: checkpoint scaler shape/finiteness invalid")
    return net, mean_x, std_x, mean_y, scale_y


def predict_polygons(features: np.ndarray, checkpoint_path: Path,
                     checkpoint_sha256: str) -> np.ndarray:
    """Predict finite physical polygons from target-stripped inputs on CPU only."""
    if features.ndim != 2 or features.shape[1] != 209 or not np.isfinite(features).all():
        raise ValueError(f"input feature shape/nonfinite {features.shape}/"
                         f"{int((~np.isfinite(features)).sum())}, expected (T,209)/0")
    net, mean_x, std_x, mean_y, scale_y = _load_model(
        str(checkpoint_path), checkpoint_sha256
    )
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), 1024):
            X = ((features[start:start + 1024].astype(np.float32)
                  - mean_x) / std_x).astype(np.float32)
            Y = net(torch.from_numpy(X)).numpy()
            output.append((Y * scale_y + mean_y).astype(np.float32))
    result = np.concatenate(output).reshape(len(features), 512, 2)
    if result.shape != (len(features), 512, 2) or not np.isfinite(result).all():
        raise ValueError("direct contour output is wrong shape or nonfinite")
    return result


def _one(task: tuple[str, int, str, str, str, str, str, dict[str, Any], dict[str, Any]]
         ) -> dict[str, Any]:
    name, index, data_s, source_s, checkpoint_s, checkpoint_sha, origin_s, spec, plan = task
    origin = Path(origin_s)
    path = Path(data_s) / "data/diii_d_train" / name
    pinned = json.loads((ROOT / "my_experiments/diii_d_dataset.json").read_text())
    expected_size, expected_sha = pinned["configs"]["diii_d_train"][name]
    if path.stat().st_size != expected_size or file_hash(path) != expected_sha:
        raise ValueError(f"{path}: parquet size/SHA differs from pinned DIII-D manifest")
    row = pd.read_parquet(path, columns=INPUT_COLUMNS).iloc[0]
    if any(key.startswith("efit_") and key not in (
            "efit_times", "efit_grid_R", "efit_grid_Z") for key in row.index):
        raise ValueError(f"{path}: direct contour inference row contains an EFIT target")
    features = allowed_input_features(row)
    polygons = predict_polygons(features, Path(checkpoint_s), checkpoint_sha)
    _, derive, lcfs, contour, mask, mask_r, mask_z, sign = _target_modules(origin, spec)
    R = np.asarray(row["efit_grid_R"], dtype=np.float64)
    Z = np.asarray(row["efit_grid_Z"], dtype=np.float64)
    if not np.array_equal(R, mask_r) or not np.array_equal(Z, mask_z):
        raise ValueError(f"{path}: physical grid differs from frozen scorer")
    with np.load(source_s, allow_pickle=False) as saved:
        original = saved[f"shot_{index:04d}_psirz"]
        q95 = saved[f"shot_{index:04d}_q95"]
        beta = saved[f"shot_{index:04d}_betaN"]
    frames = len(np.asarray(row["efit_times"]))
    if (original.shape != (frames, 65, 65) or original.dtype != np.float32
            or q95.shape != (frames,) or beta.shape != (frames,)
            or q95.dtype != np.float32 or beta.dtype != np.float32):
        raise ValueError(f"{name}: source/scalar shapes or float32 dtypes differ from {frames}")
    a_grid = json.loads((ROOT / plan["geometry"]["interior_source_li_family"])
                        .read_text())["family"]["candidate_a_grid"]
    if a_grid != [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]:
        raise ValueError("frozen source-li candidate grid differs")
    mapped: list[np.ndarray] = []
    profiled: list[np.ndarray] = []
    fallbacks: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    roundtrip: list[float] = []
    for frame, (source_psi, candidate) in enumerate(zip(original, polygons, strict=True)):
        reason: str | None = None
        if not _valid_polygon(candidate, R, Z):
            reason = "invalid_predicted_polygon"
        c_source, _ = lcfs.extract_lcfs_with_sign(
            source_psi, R, Z, "DIII-D", mask, mask.astype(float), axis_sign=sign
        )
        if c_source is None:
            raise ValueError(f"{name} frame {frame}: source-map scorer LCFS missing")
        if reason is None:
            matched, _ = _cyclic_match(c_source, candidate)
            movement = plan["geometry"]["correction_fraction"] * (matched - c_source)
            blended = c_source + movement
            if not _valid_polygon(blended, R, Z):
                reason = "invalid_half_blended_polygon"
        if reason is None:
            axis_r, axis_z, *_ = derive.magnetic_axis(
                source_psi, R, Z, "DIII-D", mask, sign
            )
            try:
                corrected, _ = correct_map_from_displacement(
                    source_psi, c_source, movement, (axis_r, axis_z), R, Z
                )
            except NumericalInfeasibility as exc:
                raise NumericalInfeasibility(f"{name} frame {frame}: {exc}") from exc
            c_new, _ = lcfs.extract_lcfs_with_sign(
                corrected, R, Z, "DIII-D", mask, mask.astype(float), axis_sign=sign
            )
            if c_new is None:
                reason = "corrected_map_LCFS_missing"
        if reason is not None:
            corrected = source_psi.copy()
            chosen = source_psi.copy()
            fallbacks[reason] += 1
        else:
            roundtrip.append(contour.symmetric_hausdorff(c_new, blended))
            try:
                chosen, detail = _compensate_frame(
                    source_psi, corrected, R, Z, mask, sign, a_grid,
                    derive, lcfs, contour,
                )
            except NumericalInfeasibility as exc:
                raise NumericalInfeasibility(f"{name} frame {frame}: source-li {exc}") from exc
            rejected.update(item["reason"] for item in detail["rejected"])
        mapped.append(corrected)
        profiled.append(chosen)
    return {
        "name": name, "frames": frames,
        "source": original, "map": np.stack(mapped), "profile": np.stack(profiled),
        "q95": q95, "betaN": beta,
        "fallback_by_reason": dict(fallbacks),
        "invalid_source_li_a_by_reason": dict(rejected),
        "roundtrip_m": roundtrip,
    }


def build_predictions(run: Path, scope: str, jobs: int) -> None:
    """Seal input-only candidate maps; no target is loaded in this stage."""
    started = time.perf_counter()
    manifest, spec = verify_campaign(run)
    if scope not in ("validation200", "evaluation70"):
        raise ValueError(f"unsupported direct contour inference scope {scope}")
    fit_record = json.loads((run / "fit/fit.json").read_text())
    checkpoint = run / "fit/direct_contour.joblib"
    if file_hash(checkpoint) != fit_record["checkpoint_sha256"]:
        raise ValueError(f"{checkpoint}: fitted checkpoint SHA changed")
    if scope == "evaluation70":
        gate_path = run / "validation200/stage2_result.json"
        gate = json.loads(gate_path.read_text())
        if not gate["gate_passed"]:
            raise ValueError(f"{gate_path}: native validation gate failed; tail70 prohibited")
        names = manifest["reserved_evaluation_files"]
        source = Path(manifest["reserved_evaluation_source_prediction"])
        expected = manifest["reserved_evaluation_source_prediction_sha256"]
    else:
        names = manifest["validation200_files"]
        source = Path(manifest["compact_validation200_prediction"])
        expected = manifest["compact_validation200_prediction_sha256"]
    if file_hash(source) != expected:
        raise ValueError(f"{source}: source-map prediction SHA changed")
    output = run / scope
    if output.exists():
        raise FileExistsError(f"{output}: scope output already exists")
    output.mkdir(parents=True)
    plan = yaml.safe_load((ROOT / YAML).read_text())
    tasks = [(name, i, manifest["data_dir"], str(source), str(checkpoint),
              fit_record["checkpoint_sha256"], manifest["origin_root"], spec, plan)
             for i, name in enumerate(names)]
    arrays: dict[str, dict[str, np.ndarray]] = {
        label: {} for label in ("source", "learned_map", "learned_map_source_li")
    }
    fallbacks: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    roundtrip: list[float] = []
    total_frames = 0
    try:
        for i, result in enumerate(pimap(_one, tasks, jobs)):
            if result["name"] != names[i]:
                raise ValueError(f"{names[i]}: candidate shot order changed")
            total_frames += result["frames"]
            for field, source_value, mapped, profiled in (
                ("psirz", result["source"], result["map"], result["profile"]),
                ("q95", result["q95"], result["q95"], result["q95"]),
                ("betaN", result["betaN"], result["betaN"], result["betaN"]),
            ):
                key = f"shot_{i:04d}_{field}"
                arrays["source"][key] = source_value
                arrays["learned_map"][key] = mapped
                arrays["learned_map_source_li"][key] = profiled
            fallbacks.update(result["fallback_by_reason"])
            rejected.update(result["invalid_source_li_a_by_reason"])
            roundtrip.extend(result["roundtrip_m"])
            if (i + 1) % 10 == 0:
                print(f"direct contour {scope}: {i + 1}/{len(names)} shots", flush=True)
    finally:
        release()
    if scope == "evaluation70" and total_frames != 14830:
        raise ValueError(f"reserved tail frames {total_frames}, expected 14830")
    for label, values in arrays.items():
        path = output / f"{label}.npz"
        _save(path, values)
        write_json(path.with_suffix(".json"), {
            "classification": "sealed_input_only_direct_contour_" + label,
            "prediction_sha256": file_hash(path),
            "files": names,
            "shots": len(names), "frames": total_frames,
            "model_checkpoint_sha256": fit_record["checkpoint_sha256"],
            "source_sha256": expected,
            "source_scalar_bits_unchanged": True,
            "input_target_columns_requested": [],
        })
    for i, name in enumerate(names):
        for field in ("q95", "betaN"):
            key = f"shot_{i:04d}_{field}"
            if not (arrays["source"][key].tobytes()
                    == arrays["learned_map"][key].tobytes()
                    == arrays["learned_map_source_li"][key].tobytes()):
                raise ValueError(f"{name}: {field} submitted scalar bits changed")
    spacing = 0.02656245231628418
    if not roundtrip:
        raise ValueError(f"{scope}: no non-fallback corrected contour for round-trip gate")
    write_json(output / "build.json", {
        "classification": "input_only_direct_contour_predictions_sealed",
        "scope": scope, "files": names,
        "shots": len(names), "frames": total_frames,
        "fallback_by_reason": dict(sorted(fallbacks.items())),
        "fallback_frames": sum(fallbacks.values()),
        "invalid_source_li_a_by_reason": dict(sorted(rejected.items())),
        "nonfallback_frames": len(roundtrip),
        "roundtrip_median_grid_cells": float(np.median(roundtrip) / spacing),
        "roundtrip_p95_grid_cells": float(np.quantile(roundtrip, 0.95) / spacing),
        "prediction_sha256": {label: file_hash(output / f"{label}.npz")
                              for label in arrays},
        "build_wall_seconds": time.perf_counter() - started,
        "no_truth_read_before_native_scorer": True,
    })


def score_scope(run: Path, scope: str, jobs: int) -> None:
    """Only after sealing arrays, use frozen native scorer on fixed shot order."""
    started = time.perf_counter()
    manifest, _ = verify_campaign(run)
    if scope == "evaluation70":
        gate_path = run / "validation200/stage2_result.json"
        if not json.loads(gate_path.read_text())["gate_passed"]:
            raise ValueError(f"{gate_path}: validation gate failed; tail70 scoring prohibited")
    output = run / scope
    build = json.loads((output / "build.json").read_text())
    if (output / "stage2_result.json").exists() or (output / "evaluation70_result.json").exists():
        raise FileExistsError(f"{output}: native score record already exists")
    names = (manifest["validation200_files"] if scope == "validation200"
             else manifest["reserved_evaluation_files"])
    if build["files"] != names:
        raise ValueError(f"{output}: prediction filenames differ from frozen scope")
    files_json = output / "files.json"
    write_json(files_json, {"files": names, "config": "diii_d_train"})
    scorer = (Path(manifest["origin_root"])
              / "runs/geometry-loss-screen-4a5516b9b1684353/source")
    scores: dict[str, dict[str, Any]] = {}
    walls: dict[str, float] = {}
    for label, expected in build["prediction_sha256"].items():
        prediction = output / f"{label}.npz"
        if file_hash(prediction) != expected:
            raise ValueError(f"{prediction}: sealed prediction SHA changed")
        began = time.perf_counter()
        metrics = output / f"{label}.metrics.json"
        command = [sys.executable, "-m", "tools.historical_compare", "score",
                   "--prediction", str(prediction), "--files-json", str(files_json),
                   "--data-dir", manifest["data_dir"], "--metrics-json", str(metrics),
                   "--jobs", str(jobs), "--scorer-root", str(scorer)]
        subprocess.run(command, check=True, cwd=ROOT,
                       env={**os.environ, "PYTHONPATH": str(ROOT)})
        native = json.loads(metrics.read_text())
        if native["files"] != names:
            raise ValueError(f"{metrics}: native scorer file order differs")
        scores[label] = native["results"]["model"]
        walls[label] = time.perf_counter() - began
    gain = round(scores["learned_map_source_li"]["S"] - scores["source"]["S"], 6)
    plan = yaml.safe_load((ROOT / YAML).read_text())
    no_fail = all(value["lcfs_fail_frac"] == 0.0 and value["cons_fail_frac"] == 0.0
                  for value in scores.values())
    if scope == "validation200":
        passed = bool(
            gain >= plan["gate"]["compact2_learned_delta_S_minimum"]
            and build["fallback_frames"] / build["frames"]
            <= plan["geometry"]["maximum_fallback_frame_fraction_on_validation"]
            and build["roundtrip_median_grid_cells"]
            <= plan["geometry"]["corrected_to_intended_blend_median_max_grid_cells"]
            and build["roundtrip_p95_grid_cells"]
            <= plan["geometry"]["corrected_to_intended_blend_p95_max_grid_cells"]
            and no_fail
        )
        result_name = "stage2_result.json"
    else:
        passed = None
        result_name = "evaluation70_result.json"
    write_json(output / result_name, {
        "classification": "input_only_direct_contour_native_validation"
                          if scope == "validation200" else "single_reserved_tail_evaluation",
        "scope": scope,
        "manifest_sha256": file_hash(run / "manifest.json"),
        "build_sha256": file_hash(output / "build.json"),
        "shots": len(names), "frames": build["frames"],
        "ordered_filenames": names,
        "native_metrics": scores,
        "native_metrics_sha256": {label: file_hash(output / f"{label}.metrics.json")
                                  for label in scores},
        "profile_minus_source_S": gain,
        "fallback_frames": build["fallback_frames"],
        "fallback_by_reason": build["fallback_by_reason"],
        "roundtrip_median_grid_cells": build["roundtrip_median_grid_cells"],
        "roundtrip_p95_grid_cells": build["roundtrip_p95_grid_cells"],
        "zero_native_failures": no_fail,
        "gate_passed": passed,
        "scorer_wall_seconds": walls,
        "score_stage_wall_seconds": time.perf_counter() - started,
        "caveat": "Validation200 is selection-only and seen by compact-source early stopping; "
                  "the reserved tail70, if reached, is one repeatedly studied local fold, "
                  "not a public-score guarantee.",
    })
    print(f"Direct contour {scope}: native delta {gain:+.6f}, gate {passed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build", "score"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--scope", choices=("validation200", "evaluation70"), required=True)
    parser.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 24:
        raise ValueError(f"jobs={args.jobs}, expected 1..24")
    run = args.run.resolve()
    if args.stage == "build":
        build_predictions(run, args.scope, args.jobs)
    else:
        score_scope(run, args.scope, args.jobs)


if __name__ == "__main__":
    main()

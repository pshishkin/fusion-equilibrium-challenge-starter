"""Run one frozen 35/35 input-only contour learnability screen.

Prepare only snapshots metadata and code. Train reads EFIT maps solely for the first
35 shots; map and profile stages read target-stripped evaluation inputs. Only the
last score stage passes the sealed prediction files to the frozen native scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from my_experiments.baseline_model import D3D_MAGNETICS_SIGNALS, GEOMETRY_COLUMNS
from my_experiments.data_check import check_dataset
from my_experiments.run import source_files
from toolkit.parallel import pimap, release
from toolkit.records import file_hash, write_json
from tools.contour_learnability import (
    N_COEFFICIENTS,
    ContourRidge,
    allowed_input_features,
    canonical_contour,
    correct_map_from_displacement,
    decode_normal_coefficients,
    encode_normal_target,
    outward_normals,
)
from tools.geometry_map_smoke import NumericalInfeasibility, _frozen_modules
from tools.geometry_source_li_compensation import _compensate_frame
from tools.historical_compare import _save, score

ROOT = Path(__file__).resolve().parent.parent
REVIEW_SPEC = ROOT / "experiments/contour-displacement-learnability-review.json"
REVIEW_SPEC_SHA256 = "6e0626808ca36852f5edf68e38ed8a03d3bae4080ff743c4e657e44ccde5bed6"
FULL70_SPEC = ROOT / "experiments/geometry-additive-source-li-full70-v3.json"
THOMSON_COLUMNS = (
    "thomson_chord_name", "thomson_chord_R", "thomson_chord_Z",
    "thomson_core_Te", "thomson_core_ne", "thomson_core_times",
    "thomson_edge_Te", "thomson_edge_ne", "thomson_edge_times",
)
INPUT_COLUMNS = list(dict.fromkeys((
    *GEOMETRY_COLUMNS,
    "magnetics_time", "magnetics_plasma_current_times",
    *(f"magnetics_{signal}" for signal in D3D_MAGNETICS_SIGNALS),
    *THOMSON_COLUMNS,
)))


def _hash_names(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def _source_paths(origin: Path, spec: dict[str, Any]) -> tuple[Path, Path, Path]:
    source = origin / spec["input"]["base_prediction"]
    files = origin / spec["input"]["ordered_files_record"]
    data = (origin / spec["input"]["data_dir"]).resolve()
    return source, files, data


def _verify_review(origin: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    spec_path = origin / "experiments/contour-displacement-learnability-review.json"
    if file_hash(spec_path) != REVIEW_SPEC_SHA256:
        raise ValueError(f"{spec_path}: SHA differs from the root-reviewed protocol")
    spec = json.loads(spec_path.read_text())
    source, files_path, _ = _source_paths(origin, spec)
    for path, expected in (
        (source, spec["input"]["base_prediction_sha256"]),
        (files_path, spec["input"]["ordered_files_record_sha256"]),
        (origin / spec["input"]["dataset_manifest"],
         spec["input"]["dataset_manifest_sha256"]),
    ):
        actual = file_hash(path)
        if actual != expected:
            raise ValueError(f"{path}: SHA {actual}, expected {expected}")
    for group in ("feature_producer_sha256", "producer_sha256"):
        entries = (spec["allowed_input_features"] if group == "feature_producer_sha256"
                   else spec["map_and_profile"])[group]
        for name, expected in entries.items():
            path = origin / name
            actual = file_hash(path)
            if actual != expected:
                raise ValueError(f"{path}: SHA {actual}, expected {expected}")
    ordered = json.loads(files_path.read_text())["files"]
    full70 = json.loads((origin / "experiments/geometry-additive-source-li-full70-v3.json")
                        .read_text())
    lengths = full70["frames_per_shot"]
    if (ordered != full70["files"] or len(ordered) != 70 or sum(lengths) != 14830
            or len(set(ordered)) != 70):
        raise ValueError(f"{files_path}: ordered 70 files/14,830 frames differ")
    for label, lo, hi in (("train_first_35", 0, 35), ("evaluate_last_35", 35, 70)):
        record = spec["screen_split"][label]
        if (record["shots"] != hi - lo or record["frames"] != sum(lengths[lo:hi])
                or record["ordered_filenames_sha256"] != _hash_names(ordered[lo:hi])):
            raise ValueError(f"{spec_path}: {label} shot/frame/hash mismatch")
    if set(ordered[:35]) & set(ordered[35:]):
        raise ValueError(f"{files_path}: train/evaluation filenames overlap")
    return spec, {"files": ordered, "frames_per_shot": lengths}


def prepare(output: Path) -> None:
    """Copy the whole runner source tree and seal filenames before reading labels."""
    origin = ROOT.resolve()
    spec, windows = _verify_review(origin)
    _, _, data = _source_paths(origin, spec)
    corpus = check_dataset(data, "diii_d_train")
    if output.exists():
        raise FileExistsError(f"{output}: frozen screen output already exists")
    output.mkdir(parents=True)
    frozen = output / "source"
    hashes: dict[str, str] = {}
    for path in source_files():
        relative = path.relative_to(origin)
        dest = frozen / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        hashes[str(relative)] = file_hash(dest)
        if hashes[str(relative)] != file_hash(path):
            raise ValueError(f"{path}: snapshot copy SHA differs")
    files = windows["files"]
    lengths = windows["frames_per_shot"]
    file_record = {
        "config": "diii_d_train",
        "dataset_revision": corpus["revision"],
        "files": files[35:],
        "frames_per_shot": lengths[35:],
    }
    write_json(output / "evaluation_files.json", file_record)
    manifest = {
        "protocol": "experiments/contour-displacement-learnability-review.json",
        "protocol_sha256": REVIEW_SPEC_SHA256,
        "origin_root": str(origin),
        "snapshot_source_sha256": hashes,
        "frozen_runner_sha256": hashes["tools/run_contour_learnability.py"],
        "full70_spec_sha256": file_hash(
            origin / "experiments/geometry-additive-source-li-full70-v3.json"
        ),
        "files": files,
        "frames_per_shot": lengths,
        "train_filenames_sha256": _hash_names(files[:35]),
        "evaluation_filenames_sha256": _hash_names(files[35:]),
        "dataset_preflight": corpus,
        "evaluation_files_sha256": file_hash(output / "evaluation_files.json"),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "pid_namespace": os.readlink("/proc/self/ns/pid"),
        "prepared_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "status": "frozen_before_any_contour_target_extraction_or_fit",
    }
    write_json(output / "manifest.json", manifest)
    print(f"Frozen {output}: train 35/{sum(lengths[:35])}, "
          f"evaluate 35/{sum(lengths[35:])}, protocol {REVIEW_SPEC_SHA256[:12]}")


def _run_context(output: Path) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    manifest = json.loads((output / "manifest.json").read_text())
    frozen = output / "source"
    if ROOT.resolve() != frozen.resolve():
        raise ValueError(f"runner imported from {ROOT}, expected frozen {frozen}")
    for relative, expected in manifest["snapshot_source_sha256"].items():
        path = frozen / relative
        if file_hash(path) != expected:
            raise ValueError(f"{path}: frozen source SHA changed")
    origin = Path(manifest["origin_root"])
    spec, windows = _verify_review(origin)
    if (windows["files"] != manifest["files"]
            or windows["frames_per_shot"] != manifest["frames_per_shot"]
            or file_hash(origin / FULL70_SPEC.relative_to(ROOT))
            != manifest["full70_spec_sha256"]
            or file_hash(output / "evaluation_files.json")
            != manifest["evaluation_files_sha256"]):
        raise ValueError(f"{output}: sealed windows/extension changed")
    source, _, data = _source_paths(origin, spec)
    return manifest, spec, source, data


def _scorer_spec(origin: Path, spec: dict[str, Any]) -> dict[str, Any]:
    evaluator = spec["evaluation"]
    keys = {
        "local_score_py_sha256", "derive_py_sha256", "lcfs_py_sha256",
        "contour_py_sha256", "o_point_py_sha256", "common_py_sha256",
    }
    old = json.loads((origin / "experiments/geometry-map-additive-contour-v3.json")
                     .read_text())["scoring"]
    return {"scorer_provenance": {
        "root": str((origin / evaluator["frozen_scorer_root"]).resolve()),
        **{key: old[key] for key in keys},
    }}


def _mask_and_modules(origin: Path, spec: dict[str, Any]) -> tuple[Any, Any, Any, Any,
                                                                     np.ndarray, np.ndarray,
                                                                     np.ndarray, int]:
    local_score, derive, lcfs, contour, common = _frozen_modules(_scorer_spec(origin, spec))
    mask_path = (origin / spec["evaluation"]["frozen_scorer_root"]
                 / "fusion_scoring/masks/d3d_envelope.npz")
    with np.load(mask_path) as masks:
        mask = masks["mask_coarse"].astype(bool)
        grid_r = masks["grid_R"].copy()
        grid_z = masks["grid_Z"].copy()
    return local_score, derive, lcfs, contour, mask, grid_r, grid_z, int(common.AXIS_SIGN["DIII-D"])


def _verify_shot(path: Path, origin: Path) -> None:
    dataset = json.loads((origin / "my_experiments/diii_d_dataset.json").read_text())
    expected_size, expected_hash = dataset["configs"]["diii_d_train"][path.name]
    if path.stat().st_size != expected_size:
        raise ValueError(f"{path}: size {path.stat().st_size}, expected {expected_size}")
    actual = file_hash(path)
    if actual != expected_hash:
        raise ValueError(f"{path}: SHA {actual}, expected {expected_hash}")


def _shot_grid(row: Any, grid_r: np.ndarray, grid_z: np.ndarray, path: Path
               ) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(row["efit_grid_R"], dtype=np.float64)
    Z = np.asarray(row["efit_grid_Z"], dtype=np.float64)
    if not np.array_equal(R, grid_r) or not np.array_equal(Z, grid_z):
        raise ValueError(f"{path}: grid {R.shape}/{Z.shape} differs from frozen scorer grid "
                         f"{grid_r.shape}/{grid_z.shape}")
    return R, Z


def _train_one(task: tuple[str, int, int, str, str, str, dict[str, Any]]) -> dict[str, Any]:
    name, index, n, data_s, source_s, origin_s, spec = task
    origin = Path(origin_s)
    path = Path(data_s) / "data/diii_d_train" / name
    _verify_shot(path, origin)
    row = pd.read_parquet(path, columns=[*INPUT_COLUMNS, "efit_psirz"]).iloc[0]
    local_score, _, lcfs, contour, mask, grid_r, grid_z, sign = _mask_and_modules(origin, spec)
    R, Z = _shot_grid(row, grid_r, grid_z, path)
    with np.load(source_s, allow_pickle=False) as saved:
        maps = saved[f"shot_{index:04d}_psirz"]
    if maps.shape != (n, 65, 65) or maps.dtype != np.float32:
        raise ValueError(f"{path}: source map {maps.shape}/{maps.dtype}, "
                         f"expected ({n},65,65)/float32")
    pred_contours: list[np.ndarray] = []
    for frame, psi in enumerate(maps):
        c_pred, _ = lcfs.extract_lcfs_with_sign(
            psi, R, Z, "DIII-D", mask, mask.astype(float), axis_sign=sign
        )
        if c_pred is None:
            raise NumericalInfeasibility(f"{name} frame {frame}: source LCFS missing")
        pred_contours.append(c_pred)
    X = allowed_input_features(row, np.stack(pred_contours))
    truth = local_score._as_psi_stack(row["efit_psirz"])
    if truth.shape != maps.shape or not np.isfinite(truth).all():
        raise ValueError(f"{path}: true psi shape/finiteness {truth.shape}/"
                         f"{int((~np.isfinite(truth)).sum())}, expected {maps.shape}/0")
    Y: list[np.ndarray] = []
    energy_numerator = 0.0
    energy_denominator = 0.0
    rms: list[float] = []
    source_hausdorff: list[float] = []
    lowpass_hausdorff: list[float] = []
    for frame, (psi_true, c_pred) in enumerate(zip(truth, pred_contours, strict=True)):
        c_true = lcfs.extract_lcfs(
            psi_true, R, Z, "DIII-D", mask, mask.astype(float)
        )
        if c_true is None:
            raise NumericalInfeasibility(f"{name} frame {frame}: target LCFS missing")
        coefficients, d = encode_normal_target(c_pred, c_true)
        u = decode_normal_coefficients(coefficients, c_pred)
        canonical, start = canonical_contour(c_pred)
        normal = outward_normals(canonical)
        reconstructed = np.sum(np.roll(u, -start, axis=0) * normal, axis=1)
        energy_numerator += float(np.sum(reconstructed**2))
        energy_denominator += float(np.sum(d**2))
        rms.append(float(np.sqrt(np.mean((d - reconstructed) ** 2))))
        source_hausdorff.append(contour.symmetric_hausdorff(c_pred, c_true))
        lowpass_hausdorff.append(contour.symmetric_hausdorff(c_pred + u, c_true))
        Y.append(coefficients)
    return {
        "name": name, "frames": n, "X": X, "Y": np.stack(Y),
        "energy_numerator": energy_numerator,
        "energy_denominator": energy_denominator,
        "rms_m": rms, "source_hausdorff_m": source_hausdorff,
        "lowpass_hausdorff_m": lowpass_hausdorff,
        "minimum_grid_spacing_m": float(min(np.diff(R).min(), np.diff(Z).min())),
    }


def train(output: Path, jobs: int) -> None:
    """Extract only train35 targets, apply the frozen representation gate, then fit once."""
    started = time.perf_counter()
    manifest, spec, source, data = _run_context(output)
    if (output / "train_gate.json").exists():
        raise FileExistsError(f"{output / 'train_gate.json'}: train stage already recorded")
    names = manifest["files"][:35]
    lengths = manifest["frames_per_shot"][:35]
    origin = Path(manifest["origin_root"])
    tasks = [(name, i, lengths[i], str(data), str(source), str(origin), spec)
             for i, name in enumerate(names)]
    rows: list[dict[str, Any]] = []
    try:
        for i, item in enumerate(pimap(_train_one, tasks, jobs)):
            if item["name"] != names[i] or item["frames"] != lengths[i]:
                raise ValueError(f"{names[i]}: returned {item['name']}/{item['frames']}, "
                                 f"expected {names[i]}/{lengths[i]}")
            rows.append(item)
            if (i + 1) % 5 == 0:
                print(f"train contour targets: {i + 1}/35 shots", flush=True)
    finally:
        release()
    energy_num = sum(item["energy_numerator"] for item in rows)
    energy_den = sum(item["energy_denominator"] for item in rows)
    if energy_den <= 0 or not np.isfinite(energy_den):
        raise ValueError(f"train35 normal displacement energy {energy_den}, expected positive")
    energy = energy_num / energy_den
    rms = [value for item in rows for value in item["rms_m"]]
    spacing = min(item["minimum_grid_spacing_m"] for item in rows)
    passed = bool(energy >= 0.8 and float(np.median(rms)) <= spacing)
    record = {
        "classification": "representation_gate_pass" if passed else "representation_gate_fail",
        "protocol_sha256": REVIEW_SPEC_SHA256,
        "run_manifest_sha256": file_hash(output / "manifest.json"),
        "training_filenames_sha256": manifest["train_filenames_sha256"],
        "train_shots": len(rows), "train_frames": sum(lengths),
        "retained_normal_energy": energy,
        "median_frame_rms_residual_m": float(np.median(rms)),
        "minimum_grid_spacing_m": spacing,
        "mean_source_to_true_hausdorff_m": float(np.mean(
            [value for item in rows for value in item["source_hausdorff_m"]]
        )),
        "mean_lowpass_to_true_hausdorff_m": float(np.mean(
            [value for item in rows for value in item["lowpass_hausdorff_m"]]
        )),
        "read_target_only_on_training_shots": True,
        "gate_passed": passed,
        "target_extraction_wall_seconds": time.perf_counter() - started,
    }
    if passed:
        X = np.concatenate([item["X"] for item in rows])
        Y = np.concatenate([item["Y"] for item in rows])
        if X.shape[0] != sum(lengths) or Y.shape != (sum(lengths), N_COEFFICIENTS):
            raise ValueError(f"train matrix shapes {X.shape}/{Y.shape}, expected "
                             f"{sum(lengths)} rows and {N_COEFFICIENTS} targets")
        fit_started = time.perf_counter()
        model = ContourRidge().fit(X, Y, lengths)
        model_path = output / "contour_ridge.joblib"
        joblib.dump(model, model_path)
        record["ridge_fit_wall_seconds"] = time.perf_counter() - fit_started
        record["model_sha256"] = file_hash(model_path)
        record["weighted_training_shots"] = 35
    write_json(output / "train_gate.json", record)
    print(f"Train representation gate {record['classification']}: energy {energy:.6f}, "
          f"median RMS {np.median(rms):.6f} m vs cell {spacing:.6f} m")


def _evaluation_one(task: tuple[str, int, int, str, str, str, str, str,
                                dict[str, Any]]) -> dict[str, Any]:
    name, index, n, data_s, source_s, model_s, origin_s, expected_model_sha, spec = task
    origin = Path(origin_s)
    path = Path(data_s) / "data/diii_d_train" / name
    _verify_shot(path, origin)
    # A column projection, not a full row followed by deletion: no evaluation label is loaded.
    row = pd.read_parquet(path, columns=INPUT_COLUMNS).iloc[0]
    if any(key.startswith("efit_") and key not in (
            "efit_times", "efit_grid_R", "efit_grid_Z") for key in row.index):
        raise ValueError(f"{path}: evaluation row contains an EFIT target column")
    _, derive, lcfs, _, mask, grid_r, grid_z, sign = _mask_and_modules(origin, spec)
    R, Z = _shot_grid(row, grid_r, grid_z, path)
    with np.load(source_s, allow_pickle=False) as saved:
        maps = saved[f"shot_{index:04d}_psirz"]
        q95 = saved[f"shot_{index:04d}_q95"]
        beta = saved[f"shot_{index:04d}_betaN"]
    if (maps.shape != (n, 65, 65) or maps.dtype != np.float32
            or q95.shape != (n,) or beta.shape != (n,)
            or q95.dtype != np.float32 or beta.dtype != np.float32):
        raise ValueError(f"{path}: source map/scalar shape or dtype differs from "
                         f"{n} frames of float32")
    pred_contours: list[np.ndarray] = []
    axes: list[tuple[float, float]] = []
    for frame, psi in enumerate(maps):
        c_pred, _ = lcfs.extract_lcfs_with_sign(
            psi, R, Z, "DIII-D", mask, mask.astype(float), axis_sign=sign
        )
        if c_pred is None:
            raise NumericalInfeasibility(f"{name} frame {frame}: source LCFS missing")
        ra, za, *_ = derive.magnetic_axis(psi, R, Z, "DIII-D", mask, sign)
        pred_contours.append(c_pred)
        axes.append((ra, za))
    X = allowed_input_features(row, np.stack(pred_contours))
    if file_hash(Path(model_s)) != expected_model_sha:
        raise ValueError(f"{model_s}: fitted Ridge SHA changed")
    model: ContourRidge = joblib.load(model_s)
    coefficients = model.predict(X)
    output_maps: list[np.ndarray] = []
    max_moves: list[float] = []
    for frame, (psi, contour, axis, coef) in enumerate(
        zip(maps, pred_contours, axes, coefficients, strict=True)
    ):
        try:
            displacement = decode_normal_coefficients(coef, contour)
            corrected, detail = correct_map_from_displacement(
                psi, contour, displacement, axis, R, Z
            )
            zero, _ = correct_map_from_displacement(
                psi, contour, np.zeros_like(displacement), axis, R, Z
            )
        except (ValueError, NumericalInfeasibility) as exc:
            raise NumericalInfeasibility(f"{name} frame {frame}: {exc}") from exc
        if zero.tobytes() != psi.tobytes():
            raise AssertionError(f"{name} frame {frame}: zero coefficient control lost source bits")
        output_maps.append(corrected)
        max_moves.append(detail.get("max_matched_contour_move_m", 0.0))
    return {
        "name": name, "frames": n, "original": maps, "corrected": np.stack(output_maps),
        "q95": q95, "betaN": beta,
        "max_predicted_contour_move_m": max(max_moves),
    }


def _write_prediction(output: Path, label: str, names: list[str],
                      arrays: dict[str, np.ndarray], manifest: dict[str, Any],
                      seconds: float) -> None:
    path = output / f"{label}.npz"
    _save(path, arrays)
    write_json(path.with_suffix(".json"), {
        "classification": "sealed_input_only_prediction",
        "prediction_sha256": file_hash(path),
        "protocol_sha256": REVIEW_SPEC_SHA256,
        "run_manifest_sha256": file_hash(output / "manifest.json"),
        "files": names,
        "ordered_filenames_sha256": _hash_names(names),
        "shots": len(names),
        "frames": sum(manifest["frames_per_shot"][35:]),
        "prediction_wall_seconds": seconds,
        "submitted_scalar_bits_unchanged": True,
        "evaluation_targets_read": False,
    })


def map_evaluation(output: Path, jobs: int) -> None:
    """Write the original and learned-map evaluation NPZs without labels."""
    started = time.perf_counter()
    manifest, spec, source, data = _run_context(output)
    gate = json.loads((output / "train_gate.json").read_text())
    if not gate["gate_passed"]:
        raise ValueError(f"{output}: representation gate failed; evaluation is prohibited")
    if (output / "original.npz").exists() or (output / "predicted_map.npz").exists():
        raise FileExistsError(f"{output}: evaluation maps already exist")
    model_path = output / "contour_ridge.joblib"
    if file_hash(model_path) != gate["model_sha256"]:
        raise ValueError(f"{model_path}: fitted Ridge SHA changed")
    origin = Path(manifest["origin_root"])
    names = manifest["files"][35:]
    lengths = manifest["frames_per_shot"][35:]
    tasks = [(name, 35 + i, lengths[i], str(data), str(source), str(model_path),
              str(origin), gate["model_sha256"], spec) for i, name in enumerate(names)]
    original: dict[str, np.ndarray] = {}
    corrected: dict[str, np.ndarray] = {}
    movement: list[dict[str, Any]] = []
    try:
        for i, item in enumerate(pimap(_evaluation_one, tasks, jobs)):
            if item["name"] != names[i] or item["frames"] != lengths[i]:
                raise ValueError(f"{names[i]}: evaluation result order/count differs")
            for field, orig, cand in (
                ("psirz", item["original"], item["corrected"]),
                ("q95", item["q95"], item["q95"]),
                ("betaN", item["betaN"], item["betaN"]),
            ):
                key = f"shot_{i:04d}_{field}"
                original[key] = orig
                corrected[key] = cand
            movement.append({"shot": names[i], "frames": lengths[i],
                             "max_predicted_contour_move_m":
                             item["max_predicted_contour_move_m"]})
            if (i + 1) % 5 == 0:
                print(f"sealed input-only maps: {i + 1}/35 shots", flush=True)
    finally:
        release()
    if sum(lengths) != 7563:
        raise ValueError(f"{output}: eval frames {sum(lengths)}, expected 7563")
    seconds = time.perf_counter() - started
    _write_prediction(output, "original", names, original, manifest, seconds)
    _write_prediction(output, "predicted_map", names, corrected, manifest, seconds)
    write_json(output / "map_stage.json", {
        "wall_seconds": seconds, "model_sha256": gate["model_sha256"],
        "original_prediction_sha256": file_hash(output / "original.npz"),
        "map_prediction_sha256": file_hash(output / "predicted_map.npz"),
        "per_shot_max_predicted_move": movement,
        "evaluation_target_columns_requested": [],
    })


def _profile_one(task: tuple[str, int, int, str, str, str, str, dict[str, Any]]
                 ) -> dict[str, Any]:
    name, local_index, n, data_s, source_s, map_s, origin_s, spec = task
    origin = Path(origin_s)
    path = Path(data_s) / "data/diii_d_train" / name
    # Only the permitted grid is read; neither true psi nor stored li is available here.
    row = pd.read_parquet(path, columns=["efit_grid_R", "efit_grid_Z"]).iloc[0]
    _, derive, lcfs, contour, mask, grid_r, grid_z, sign = _mask_and_modules(origin, spec)
    R, Z = _shot_grid(row, grid_r, grid_z, path)
    with (np.load(source_s, allow_pickle=False) as source,
          np.load(map_s, allow_pickle=False) as predicted):
        original = source[f"shot_{local_index + 35:04d}_psirz"]
        corrected = predicted[f"shot_{local_index:04d}_psirz"]
        q95 = predicted[f"shot_{local_index:04d}_q95"]
        beta = predicted[f"shot_{local_index:04d}_betaN"]
    if original.shape != (n, 65, 65) or corrected.shape != original.shape:
        raise ValueError(f"{name}: profile source/map shapes {original.shape}/{corrected.shape}, "
                         f"expected ({n},65,65)")
    chosen: list[np.ndarray] = []
    selected_a: list[float] = []
    rejected: Counter[str] = Counter()
    for frame, (source_frame, map_frame) in enumerate(zip(original, corrected, strict=True)):
        try:
            result, detail = _compensate_frame(
                source_frame, map_frame, R, Z, mask, sign,
                [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
                derive, lcfs, contour,
            )
        except NumericalInfeasibility as exc:
            raise NumericalInfeasibility(f"{name} frame {frame}: {exc}") from exc
        chosen.append(result)
        selected_a.append(detail["selected_a"])
        rejected.update(item["reason"] for item in detail["rejected"])
    return {
        "name": name, "frames": n, "psirz": np.stack(chosen),
        "q95": q95, "betaN": beta,
        "nonzero_a_frames": sum(a != 0 for a in selected_a),
        "invalid_nonzero_a_by_reason": dict(rejected),
    }


def compensate_evaluation(output: Path, jobs: int) -> None:
    """Apply the fixed source-li stage to already sealed, input-only maps."""
    started = time.perf_counter()
    manifest, spec, source, data = _run_context(output)
    stage = json.loads((output / "map_stage.json").read_text())
    map_path = output / "predicted_map.npz"
    if file_hash(map_path) != stage["map_prediction_sha256"]:
        raise ValueError(f"{map_path}: sealed learned-map SHA changed")
    if (output / "predicted_map_source_li.npz").exists():
        raise FileExistsError(f"{output}: compensated prediction already exists")
    origin = Path(manifest["origin_root"])
    names = manifest["files"][35:]
    lengths = manifest["frames_per_shot"][35:]
    tasks = [(name, i, lengths[i], str(data), str(source), str(map_path), str(origin), spec)
             for i, name in enumerate(names)]
    arrays: dict[str, np.ndarray] = {}
    nonzero = 0
    rejected: Counter[str] = Counter()
    try:
        for i, item in enumerate(pimap(_profile_one, tasks, jobs)):
            if item["name"] != names[i] or item["frames"] != lengths[i]:
                raise ValueError(f"{names[i]}: profile result order/count differs")
            for field in ("psirz", "q95", "betaN"):
                arrays[f"shot_{i:04d}_{field}"] = item[field]
            nonzero += item["nonzero_a_frames"]
            rejected.update(item["invalid_nonzero_a_by_reason"])
            if (i + 1) % 5 == 0:
                print(f"source-li evaluation: {i + 1}/35 shots", flush=True)
    finally:
        release()
    _write_prediction(output, "predicted_map_source_li", names, arrays, manifest,
                      time.perf_counter() - started)
    write_json(output / "profile_stage.json", {
        "wall_seconds": time.perf_counter() - started,
        "source_map_sha256": stage["map_prediction_sha256"],
        "compensated_prediction_sha256": file_hash(output / "predicted_map_source_li.npz"),
        "nonzero_a_frames": nonzero,
        "invalid_nonzero_a_by_reason": dict(sorted(rejected.items())),
        "evaluation_target_columns_requested": [],
    })


def score_evaluation(output: Path, jobs: int) -> None:
    """Only now let the unchanged native scorer read evaluation labels."""
    started = time.perf_counter()
    manifest, spec, _, data = _run_context(output)
    if (output / "final_score.json").exists():
        raise FileExistsError(f"{output}: score already recorded")
    map_stage = json.loads((output / "map_stage.json").read_text())
    profile_stage = json.loads((output / "profile_stage.json").read_text())
    expected = {
        "original": map_stage["original_prediction_sha256"],
        "predicted_map": map_stage["map_prediction_sha256"],
        "predicted_map_source_li": profile_stage["compensated_prediction_sha256"],
    }
    for label, digest in expected.items():
        path = output / f"{label}.npz"
        record = json.loads(path.with_suffix(".json").read_text())
        if (file_hash(path) != digest or record["prediction_sha256"] != digest
                or record["files"] != manifest["files"][35:]):
            raise ValueError(f"{path}: sealed prediction/files SHA changed")
    scores: dict[str, dict[str, Any]] = {}
    score_times: dict[str, float] = {}
    origin = Path(manifest["origin_root"])
    scorer_root = origin / spec["evaluation"]["frozen_scorer_root"]
    files_json = output / "evaluation_files.json"
    for label in expected:
        started_one = time.perf_counter()
        metrics = output / f"{label}.metrics.json"
        score(output / f"{label}.npz", files_json, data, metrics, jobs, scorer_root)
        scored = json.loads(metrics.read_text())
        if scored["files"] != manifest["files"][35:]:
            raise ValueError(f"{metrics}: native scorer filename order differs")
        scores[label] = scored["results"]["model"]
        score_times[label] = time.perf_counter() - started_one
    delta_map = round(scores["predicted_map"]["S"] - scores["original"]["S"], 6)
    delta_profile = round(scores["predicted_map_source_li"]["S"]
                          - scores["original"]["S"], 6)
    write_json(output / "final_score.json", {
        "classification": "reused_tail_learnability_screen",
        "protocol_sha256": REVIEW_SPEC_SHA256,
        "manifest_sha256": file_hash(output / "manifest.json"),
        "shots": 35, "frames": sum(manifest["frames_per_shot"][35:]),
        "evaluation_filenames_sha256": manifest["evaluation_filenames_sha256"],
        "native_metrics": scores,
        "native_metrics_sha256": {
            label: file_hash(output / f"{label}.metrics.json") for label in expected
        },
        "predicted_map_minus_original_S": delta_map,
        "predicted_map_source_li_minus_original_S": delta_profile,
        "fixed_screen_gate_passed": bool(delta_profile >= 0.001),
        "scorer_wall_seconds": score_times,
        "score_stage_wall_seconds": time.perf_counter() - started,
        "caveat": "Evaluation35 is part of the repeatedly used salt-0 tail. Even a passed fixed "
                  "screen would require a new user-approved common holdout and paired refits; "
                  "it is not independent confirmation or a public-score estimate.",
    })
    print(f"Native screen: map delta {delta_map:+.6f}, map+source-li "
          f"delta {delta_profile:+.6f}; fixed gate {delta_profile >= 0.001}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "map", "compensate", "score"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    output = args.run.resolve()
    if args.jobs < 1 or args.jobs > 4:
        raise ValueError(f"--jobs {args.jobs}, expected 1..4 for the frozen screen")
    if args.stage == "prepare":
        prepare(output)
    elif args.stage == "train":
        train(output, args.jobs)
    elif args.stage == "map":
        map_evaluation(output, args.jobs)
    elif args.stage == "compensate":
        compensate_evaluation(output, args.jobs)
    else:
        score_evaluation(output, args.jobs)


if __name__ == "__main__":
    main()

"""Freeze and build target-stripped 209-input / true-512-contour shot caches.

Only compact TRAIN and validation filenames are accepted. The original salt-0
tail is excluded by exact artifact/scorer filename comparison before any target
extraction. Each ready cache entry is atomic and checks its own source hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from my_experiments import baseline_model
from my_experiments.data_check import check_dataset
from my_experiments.run import source_files
from toolkit.parallel import pimap, release
from toolkit.records import file_hash, write_json
from tools.diagnose_direct_contour_representation import _canonical_true, _name_hash
from tools.geometry_map_smoke import _frozen_modules

ROOT = Path(__file__).resolve().parent.parent
SPEC = Path("experiments/direct-full-contour-mlp-review.json")
YAML = Path("baselines/direct-full-contour-mlp.review.params.yaml")
ARTIFACT = Path("runs/compact-production-9301af36275620f7/base_s0/baseline.joblib")
METRICS = Path("runs/compact-production-9301af36275620f7/base_s0/metrics.json")
FULL_TRAIN_FRAMES = 1248457
FULL_VALIDATION_FRAMES = 296053


def allowed_input_features(row: Any) -> np.ndarray:
    """Build only 208 existing permitted inputs plus one within-shot phase."""
    slim = baseline_model.slim_row(row)
    forbidden = [name for name in slim if name.startswith("efit_")
                 and name not in ("efit_times", "efit_grid_R", "efit_grid_Z")]
    if forbidden:
        raise ValueError(f"target-stripped input row contains EFIT targets {forbidden}")
    times = np.asarray(slim["efit_times"], dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all() or len(times) < 1:
        raise ValueError(f"efit_times shape/finiteness {times.shape}, expected finite (T,)")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("efit_times must increase strictly within a shot")
    base = np.asarray(baseline_model.features_for_row(slim), dtype=np.float32)
    if base.shape != (len(times), 208) or not np.isfinite(base).all():
        raise ValueError(f"permitted features shape/nonfinite {base.shape}/"
                         f"{int((~np.isfinite(base)).sum())}, expected ({len(times)},208)/0")
    phase = np.zeros(len(times), dtype=np.float32) if len(times) == 1 else (
        (times - times[0]) / (times[-1] - times[0])
    ).astype(np.float32)
    result = np.column_stack([base, phase])
    if result.shape != (len(times), 209) or not np.isfinite(result).all():
        raise ValueError(f"allowed input shape/nonfinite {result.shape}/"
                         f"{int((~np.isfinite(result)).sum())}, expected ({len(times)},209)/0")
    return result


def _verify_spec(origin: Path) -> tuple[dict[str, Any], dict[str, Any], list[str], list[str]]:
    spec = json.loads((origin / SPEC).read_text())
    params = origin / YAML
    if file_hash(params) != spec["effective_yaml_sha256"]:
        raise ValueError(f"{params}: YAML SHA differs from review protocol")
    plan = yaml.safe_load(params.read_text())
    if (plan["data"]["train_frame_share"] != 0.2
            or plan["data"]["validation_frame_share"] != 0.2
            or plan["input"]["total_width"] != 209
            or plan["target"]["output_width"] != 1024):
        raise ValueError(f"{params}: fixed cache input/target shares differ")
    artifact = origin / ARTIFACT
    if file_hash(artifact) != spec["source_and_split"]["compact_artifact_sha256"]:
        raise ValueError(f"{artifact}: compact artifact SHA differs")
    saved = joblib.load(artifact)
    train = list(saved["train_files"])
    validation = list(saved["val_files"])
    tail = json.loads((origin / METRICS).read_text())["files"]
    split = spec["source_and_split"]
    if (len(train) != 5633 or len(validation) != 1338 or len(tail) != 70
            or _name_hash(train) != split["train_ordered_names_sha256"]
            or _name_hash(validation) != split["validation_ordered_names_sha256"]
            or _name_hash(tail) != split["reserved_evaluation_ordered_names_sha256"]
            or set(train) & set(validation) or set(train) & set(tail)
            or set(validation) & set(tail)):
        raise ValueError(f"{artifact}: 5633/1338/70 windows changed or overlap")
    return spec, plan, train, validation


def prepare(output: Path) -> None:
    """Freeze full source/config and exact train/validation identities first."""
    started = time.perf_counter()
    origin = ROOT.resolve()
    spec, _, train, validation = _verify_spec(origin)
    stage0_path = origin / "results/direct_full_contour_stage0_20260923.json"
    stage0 = json.loads(stage0_path.read_text())
    if (stage0["classification"] != "truth_assisted_investment_gate_passed_selection_only"
            or not stage0["preregistered_oracle_gain_gate"]["passed"]
            or stage0["validation_filenames"] != validation[:200]):
        raise ValueError(f"{stage0_path}: fixed Stage-0 gate or val200 files differ")
    compact_val_source = (
        origin / "runs/direct-full-contour-stage0-20260923/compact_validation200.npz"
    )
    if file_hash(compact_val_source) != stage0["prediction_sha256"]["compact_control"]:
        raise ValueError(f"{compact_val_source}: Stage-0 compact validation source changed")
    tail_source = origin / spec["source_and_split"]["reserved_evaluation_source_prediction"]
    if (file_hash(tail_source)
            != spec["source_and_split"]["reserved_evaluation_source_prediction_sha256"]):
        raise ValueError(f"{tail_source}: reserved evaluation source SHA changed")
    data = (origin / "../downloaded_huggingface/hf_dataset").resolve()
    corpus = check_dataset(data, "diii_d_train")
    if corpus["manifest_sha256"] != spec["source_and_split"]["dataset_manifest_sha256"]:
        raise ValueError("DIII-D corpus manifest SHA changed")
    if output.exists():
        raise FileExistsError(f"{output}: direct contour campaign exists")
    output.mkdir(parents=True)
    snapshot = output / "source"
    hashes: dict[str, str] = {}
    for path in source_files():
        relative = path.relative_to(origin)
        dest = snapshot / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        hashes[str(relative)] = file_hash(dest)
        if hashes[str(relative)] != file_hash(path):
            raise ValueError(f"{path}: source snapshot SHA differs")
    for relative in (SPEC, YAML,
                     Path("experiments/geometry-additive-source-li-compensation-v3.json")):
        dest = snapshot / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin / relative, dest)
        hashes[str(relative)] = file_hash(dest)
    fingerprint = hashlib.sha256(json.dumps({
        "spec": hashes[str(SPEC)], "yaml": hashes[str(YAML)],
        "producer": hashes["tools/direct_full_contour_cache.py"],
        "baseline": hashes["my_experiments/baseline_model.py"],
        "experiments": hashes["experiments.py"],
        "lock": hashes["uv.lock"],
        "scorer": hashes["fusion_scoring/lcfs.py"],
        "mask": hashes["fusion_scoring/masks/d3d_envelope.npz"],
    }, sort_keys=True).encode()).hexdigest()[:20]
    manifest = {
        "status": "frozen_before_contour_target_extraction_or_fit",
        "origin_root": str(origin),
        "data_dir": str(data),
        "spec_sha256": hashes[str(SPEC)],
        "yaml_sha256": hashes[str(YAML)],
        "source_hashes": hashes,
        "train_files": train,
        "train_ordered_names_sha256": _name_hash(train),
        "validation_files": validation,
        "validation_ordered_names_sha256": _name_hash(validation),
        "target_cache_key": fingerprint,
        "target_cache_root": str(origin / ".direct_contour_targets" / fingerprint),
        "stage0_result": str(stage0_path),
        "stage0_result_sha256": file_hash(stage0_path),
        "validation200_files": validation[:200],
        "validation200_ordered_names_sha256": _name_hash(validation[:200]),
        "compact_validation200_prediction": str(compact_val_source),
        "compact_validation200_prediction_sha256": file_hash(compact_val_source),
        "reserved_evaluation_source_prediction": str(tail_source),
        "reserved_evaluation_source_prediction_sha256": file_hash(tail_source),
        "reserved_evaluation_files": json.loads((origin / METRICS).read_text())["files"],
        "dataset_preflight": corpus,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "pid_namespace": os.readlink("/proc/self/ns/pid"),
        "prepare_wall_seconds": time.perf_counter() - started,
    }
    write_json(output / "manifest.json", manifest)
    print(f"Frozen {output}: train {len(train)}, validation {len(validation)}, "
          f"cache key {fingerprint}")


def verify_campaign(output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reject a moving source tree or shot split before every cache/fit stage."""
    manifest = json.loads((output / "manifest.json").read_text())
    if ROOT.resolve() != (output / "source").resolve():
        raise ValueError(f"runner imported from {ROOT}, expected frozen {output / 'source'}")
    for relative, digest in manifest["source_hashes"].items():
        path = ROOT / relative
        if file_hash(path) != digest:
            raise ValueError(f"{path}: frozen source SHA changed")
    origin = Path(manifest["origin_root"])
    if (file_hash(origin / SPEC) != manifest["spec_sha256"]
            or file_hash(origin / YAML) != manifest["yaml_sha256"]):
        raise ValueError("direct contour spec or YAML changed")
    if file_hash(Path(manifest["stage0_result"])) != manifest["stage0_result_sha256"]:
        raise ValueError(f"{manifest['stage0_result']}: Stage-0 result changed")
    if (len(manifest["train_files"]) != 5633
            or len(manifest["validation_files"]) != 1338
            or len(manifest["reserved_evaluation_files"]) != 70
            or _name_hash(manifest["train_files"])
            != manifest["train_ordered_names_sha256"]
            or _name_hash(manifest["validation_files"])
            != manifest["validation_ordered_names_sha256"]
            or _name_hash(manifest["reserved_evaluation_files"])
            != json.loads((ROOT / SPEC).read_text())["source_and_split"][
                "reserved_evaluation_ordered_names_sha256"
            ]
            or set(manifest["train_files"]) & set(manifest["reserved_evaluation_files"])
            or set(manifest["validation_files"]) & set(manifest["reserved_evaluation_files"])):
        raise ValueError(f"{output}: sealed train/validation filenames changed")
    spec = json.loads((ROOT / SPEC).read_text())
    return manifest, spec


def _numeric_scorer(origin: Path, spec: dict[str, Any]
                    ) -> tuple[Any, Any, np.ndarray, np.ndarray, np.ndarray]:
    old = json.loads((origin / "experiments/geometry-map-additive-contour-v3.json")
                     .read_text())["scoring"]
    keys = ("derive_py_sha256", "lcfs_py_sha256", "contour_py_sha256",
            "o_point_py_sha256", "common_py_sha256")
    for key in keys:
        relative = "fusion_scoring/" + key.removesuffix("_py_sha256") + ".py"
        if file_hash(ROOT / relative) != old[key]:
            raise ValueError(f"{ROOT / relative}: numerical scorer differs from pinned native")
    local_score, _, lcfs, _, _ = _frozen_modules({"scorer_provenance": {
        "root": str(ROOT),
        "local_score_py_sha256": file_hash(ROOT / "local_score.py"),
        **{key: old[key] for key in keys},
    }})
    with np.load(ROOT / "fusion_scoring/masks/d3d_envelope.npz") as saved:
        mask = saved["mask_coarse"].astype(bool)
        R, Z = saved["grid_R"], saved["grid_Z"]
    return local_score, lcfs, mask, R, Z


def _save_atomic(path: Path, features: np.ndarray, contours: np.ndarray,
                 indices: np.ndarray, frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, features=features, contours=contours,
                                indices=indices, full_frames=frames)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _cache_one(task: tuple[str, str, str, str, str, dict[str, Any]]
               ) -> dict[str, Any]:
    name, window, data_s, cache_s, origin_s, spec = task
    started = time.perf_counter()
    origin = Path(origin_s)
    path = Path(data_s) / "data/diii_d_train" / name
    pinned = json.loads((ROOT / "my_experiments/diii_d_dataset.json").read_text())
    expected_size, source_hash = pinned["configs"]["diii_d_train"][name]
    if path.stat().st_size != expected_size or file_hash(path) != source_hash:
        raise ValueError(f"{path}: parquet size/SHA differs from pinned manifest")
    cache = Path(cache_s) / window / f"{Path(name).stem}.npz"
    ready = cache.with_suffix(".ready.json")
    cache_key = Path(cache_s).name
    yaml_sha = spec["effective_yaml_sha256"]
    lock_sha = file_hash(ROOT / "uv.lock")
    mask_sha = file_hash(ROOT / "fusion_scoring/masks/d3d_envelope.npz")
    lcfs_sha = file_hash(ROOT / "fusion_scoring/lcfs.py")
    features_sha = file_hash(ROOT / "my_experiments/baseline_model.py")
    if cache.exists() and ready.exists():
        info = json.loads(ready.read_text())
        if (info["source_sha256"] == source_hash
                and info["cache_sha256"] == file_hash(cache)
                and info["producer_sha256"] == file_hash(Path(__file__))
                and info["cache_key"] == cache_key
                and info["yaml_sha256"] == yaml_sha
                and info["dependency_lock_sha256"] == lock_sha
                and info["scorer_mask_sha256"] == mask_sha
                and info["scorer_lcfs_sha256"] == lcfs_sha
                and info["input_feature_producer_sha256"] == features_sha):
            with np.load(cache, allow_pickle=False) as saved:
                features, targets, indices = saved["features"], saved["contours"], saved["indices"]
                full_frames = int(saved["full_frames"])
            if (features.shape == (len(indices), 209)
                    and targets.shape == (len(indices), 512, 2)
                    and features.dtype == np.float32 and targets.dtype == np.float32
                    and np.array_equal(indices, baseline_model.thin_frames(
                        full_frames, 0.2))):
                return {**info, "cache_hit": True, "worker_seconds": time.perf_counter() - started}
        raise ValueError(f"{cache}: existing ready cache has mismatched identity or arrays")
    if cache.exists() or ready.exists():
        raise ValueError(f"{cache}: incomplete cache pair exists; refusing to overwrite")
    row = pd.read_parquet(path).iloc[0]
    times = np.asarray(row["efit_times"], dtype=np.float64)
    indices = baseline_model.thin_frames(len(times), 0.2)
    full_inputs = allowed_input_features(row)
    if full_inputs.shape != (len(times), 209):
        raise ValueError(f"{name}: full input shape {full_inputs.shape}, expected "
                         f"({len(times)},209)")
    local_score, lcfs, mask, mask_r, mask_z = _numeric_scorer(origin, spec)
    R = np.asarray(row["efit_grid_R"], dtype=np.float64)
    Z = np.asarray(row["efit_grid_Z"], dtype=np.float64)
    if not np.array_equal(R, mask_r) or not np.array_equal(Z, mask_z):
        raise ValueError(f"{name}: physical grid differs from frozen scorer")
    psi = local_score._as_psi_stack(row["efit_psirz"])
    if psi.shape != (len(times), 65, 65) or not np.isfinite(psi).all():
        raise ValueError(f"{name}: true psi shape/nonfinite {psi.shape}/"
                         f"{int((~np.isfinite(psi)).sum())}, expected "
                         f"({len(times)},65,65)/0")
    contours: list[np.ndarray] = []
    for frame in indices:
        c = lcfs.extract_lcfs(psi[int(frame)], R, Z, "DIII-D", mask, mask.astype(float))
        if c is None:
            raise ValueError(f"{name} frame {frame}: scorer-native true LCFS missing")
        try:
            contours.append(_canonical_true(c).astype(np.float32))
        except ValueError as exc:
            raise ValueError(f"{name} frame {frame}: {exc}") from exc
    features = full_inputs[indices].astype(np.float32)
    target = np.stack(contours)
    if (features.shape != (len(indices), 209) or target.shape != (len(indices), 512, 2)
            or not np.isfinite(features).all() or not np.isfinite(target).all()):
        raise ValueError(f"{name}: cache feature/target shape/finiteness differs from "
                         f"({len(indices)},209)/({len(indices)},512,2)")
    _save_atomic(cache, features, target, indices.astype(np.int32), len(times))
    info = {
        "shot": name, "window": window, "source_sha256": source_hash,
        "cache_path": str(cache), "cache_sha256": file_hash(cache),
        "producer_sha256": file_hash(Path(__file__)),
        "cache_key": cache_key, "yaml_sha256": yaml_sha,
        "dependency_lock_sha256": lock_sha,
        "scorer_mask_sha256": mask_sha,
        "scorer_lcfs_sha256": lcfs_sha,
        "input_feature_producer_sha256": features_sha,
        "full_frames": len(times), "kept_frames": len(indices),
        "kept_indices_sha256": hashlib.sha256(indices.astype(np.int32).tobytes()).hexdigest(),
        "features_shape": list(features.shape), "targets_shape": list(target.shape),
        "features_dtype": str(features.dtype), "targets_dtype": str(target.dtype),
        "cache_hit": False, "worker_seconds": time.perf_counter() - started,
    }
    write_json(ready, info)
    return info


def build(output: Path, jobs: int) -> None:
    """Build the fixed train/validation target cache with one shared shot pool."""
    started = time.perf_counter()
    manifest, spec = verify_campaign(output)
    if (output / "target_cache_ready.json").exists():
        raise FileExistsError(f"{output}: target-cache completion already recorded")
    origin = Path(manifest["origin_root"])
    tasks = [(name, window, manifest["data_dir"], manifest["target_cache_root"], str(origin), spec)
             for window, names in (("train", manifest["train_files"]),
                                   ("validation", manifest["validation_files"])) for name in names]
    entries: list[dict[str, Any]] = []
    try:
        for i, result in enumerate(pimap(_cache_one, tasks, jobs)):
            wanted = tasks[i]
            if result["shot"] != wanted[0] or result["window"] != wanted[1]:
                raise ValueError(f"{wanted[0]}: cache return order/window changed")
            entries.append(result)
            if (i + 1) % 250 == 0:
                print(f"direct contour cache: {i + 1}/{len(tasks)} shots", flush=True)
    finally:
        release()
    train = entries[:5633]
    validation = entries[5633:]
    train_full = sum(item["full_frames"] for item in train)
    validation_full = sum(item["full_frames"] for item in validation)
    if train_full != FULL_TRAIN_FRAMES or validation_full != FULL_VALIDATION_FRAMES:
        raise ValueError(f"full-frame totals {train_full}/{validation_full}, expected "
                         f"{FULL_TRAIN_FRAMES}/{FULL_VALIDATION_FRAMES}")
    record = {
        "classification": "full_train_validation_contour_cache_ready",
        "manifest_sha256": file_hash(output / "manifest.json"),
        "spec_sha256": manifest["spec_sha256"],
        "yaml_sha256": manifest["yaml_sha256"],
        "cache_key": manifest["target_cache_key"],
        "train_ordered_names_sha256": manifest["train_ordered_names_sha256"],
        "validation_ordered_names_sha256": manifest["validation_ordered_names_sha256"],
        "train_shots": 5633, "validation_shots": 1338,
        "train_full_frames": train_full,
        "validation_full_frames": validation_full,
        "train_kept_frames": sum(item["kept_frames"] for item in train),
        "validation_kept_frames": sum(item["kept_frames"] for item in validation),
        "ready_entries": entries,
        "cache_build_wall_seconds": time.perf_counter() - started,
        "no_reserved_tail70_training_row_or_target_read": True,
    }
    write_json(output / "target_cache_ready.json", record)
    print(f"Contour cache ready: train {record['train_kept_frames']} frames, "
          f"validation {record['validation_kept_frames']} frames, "
          f"{record['cache_build_wall_seconds']:.1f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "build"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 24:
        raise ValueError(f"jobs={args.jobs}, expected 1..24")
    output = args.run.resolve()
    if args.stage == "prepare":
        prepare(output)
    else:
        build(output, args.jobs)


if __name__ == "__main__":
    main()

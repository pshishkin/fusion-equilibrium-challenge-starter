"""Frozen Stage-0, validation-only half-contour oracle for a proposed direct MLP.

No new model is fitted. The compact two-member validation source has already seen
this window for early stopping, so every native score here is selection-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from my_experiments.data_check import check_dataset
from my_experiments.run import source_files
from toolkit.parallel import pimap, release
from toolkit.records import file_hash, write_json
from tools.contour_learnability import correct_map_from_displacement
from tools.geometry_map_smoke import (
    NumericalInfeasibility,
    _cyclic_match,
    _frozen_modules,
    _simple_closed,
)
from tools.geometry_source_li_compensation import _compensate_frame
from tools.historical_compare import _save

ROOT = Path(__file__).resolve().parent.parent
SPEC = Path("experiments/direct-full-contour-mlp-review.json")
YAML = Path("baselines/direct-full-contour-mlp.review.params.yaml")
COMPACT_SOURCE = Path("runs/compact-production-9301af36275620f7/source")
COMPACT_ARTIFACT = Path("runs/compact-production-9301af36275620f7/base_s0/baseline.joblib")
SCORE_FILES = Path("runs/compact-production-9301af36275620f7/base_s0/metrics.json")
SOURCE_FIELDS = ("psirz", "q95", "betaN")


def _names_sha(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def _length_one(task: tuple[str, str]) -> tuple[str, int]:
    name, data_s = task
    path = Path(data_s) / "data/diii_d_train" / name
    row = pd.read_parquet(path, columns=["efit_times"]).iloc[0]
    times = np.asarray(row["efit_times"], dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all() or len(times) < 1:
        raise ValueError(f"{path}: EFIT time shape/finiteness {times.shape}, expected nonempty")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError(f"{path}: EFIT times not strictly increasing")
    return name, len(times)


def _verify_spec(origin: Path) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    spec = json.loads((origin / SPEC).read_text())
    plan_path = origin / YAML
    if file_hash(plan_path) != spec["effective_yaml_sha256"]:
        raise ValueError(f"{plan_path}: effective YAML SHA differs from frozen review")
    plan = yaml.safe_load(plan_path.read_text())
    if (plan["geometry"]["correction_fraction"] != 0.5
            or plan["gate"]["compact2_oracle_delta_S_minimum"] != 0.0013
            or plan["gate"]["native_validation_shots"] != 200):
        raise ValueError(f"{plan_path}: fixed Stage-0 YAML values changed")
    artifact = origin / COMPACT_ARTIFACT
    if file_hash(artifact) != spec["source_and_split"]["compact_artifact_sha256"]:
        raise ValueError(f"{artifact}: compact artifact SHA changed")
    loaded = joblib.load(artifact)
    train = list(loaded["train_files"])
    validation = list(loaded["val_files"])
    scored = json.loads((origin / SCORE_FILES).read_text())["files"]
    split = spec["source_and_split"]
    if (len(train) != 5633 or _names_sha(train) != split["train_ordered_names_sha256"]
            or len(validation) != 1338
            or _names_sha(validation) != split["validation_ordered_names_sha256"]
            or len(scored) != 70
            or _names_sha(scored) != split["reserved_evaluation_ordered_names_sha256"]
            or _names_sha(validation[:200])
            != split["internal_native_validation_ordered_names_sha256"]
            or set(train) & set(validation) or set(train) & set(scored)
            or set(validation) & set(scored)):
        raise ValueError(f"{artifact}: compact 5633/1338/70 split or val200 identity changed")
    return spec, plan, validation[:200]


def prepare(output: Path, jobs: int) -> None:
    """Freeze the exact validation200 and both source trees before reading targets."""
    started = time.perf_counter()
    origin = ROOT.resolve()
    spec, plan, names = _verify_spec(origin)
    data = (origin / "../downloaded_huggingface/hf_dataset").resolve()
    corpus = check_dataset(data, "diii_d_train")
    if corpus["manifest_sha256"] != spec["source_and_split"]["dataset_manifest_sha256"]:
        raise ValueError("pinned DIII-D manifest SHA changed")
    lengths: list[int] = []
    try:
        for i, (name, n) in enumerate(pimap(
            _length_one, [(name, str(data)) for name in names], jobs
        )):
            if name != names[i]:
                raise ValueError(f"{names[i]}: metadata returned wrong filename {name}")
            lengths.append(n)
    finally:
        release()
    if output.exists():
        raise FileExistsError(f"{output}: oracle campaign already prepared")
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
            raise ValueError(f"{path}: snapshot copy SHA differs")
    for relative in (SPEC, YAML, Path(plan["geometry"]["interior_source_li_family"])):
        dest = snapshot / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin / relative, dest)
        hashes[str(relative)] = file_hash(dest)
    compact_source = origin / COMPACT_SOURCE
    compact_hashes = {
        relative: file_hash(compact_source / relative)
        for relative in ("my_experiments/baseline_model.py", "my_experiments/models.py",
                         "my_experiments/seq_model.py", "experiments.py")
    }
    manifest = {
        "status": "frozen_before_stage0_truth_contours_or_scorer",
        "origin_root": str(origin),
        "data_dir": str(data),
        "spec_sha256": file_hash(origin / SPEC),
        "yaml_sha256": file_hash(origin / YAML),
        "source_hashes": hashes,
        "compact_source": str(compact_source),
        "compact_predictor_source_sha256": compact_hashes,
        "compact_artifact": str(origin / COMPACT_ARTIFACT),
        "compact_artifact_sha256": spec["source_and_split"]["compact_artifact_sha256"],
        "validation200_files": names,
        "validation200_ordered_names_sha256": _names_sha(names),
        "validation200_frames": lengths,
        "validation200_total_frames": sum(lengths),
        "dataset_preflight": corpus,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "pid_namespace": os.readlink("/proc/self/ns/pid"),
        "prepare_wall_seconds": time.perf_counter() - started,
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "validation200.files.json", {
        "files": names, "config": "diii_d_train",
        "dataset_revision": corpus["revision"]
    })
    print(f"Frozen Stage-0 oracle: {len(names)} compact validation shots, "
          f"{sum(lengths)} full frames, {manifest['prepare_wall_seconds']:.2f} s")


def _verify_run(output: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = json.loads((output / "manifest.json").read_text())
    if ROOT.resolve() != (output / "source").resolve():
        raise ValueError(f"runner imported from {ROOT}, expected frozen {output / 'source'}")
    for relative, expected in manifest["source_hashes"].items():
        path = ROOT / relative
        if file_hash(path) != expected:
            raise ValueError(f"{path}: frozen source SHA differs")
    origin = Path(manifest["origin_root"])
    if (file_hash(origin / SPEC) != manifest["spec_sha256"]
            or file_hash(origin / YAML) != manifest["yaml_sha256"]):
        raise ValueError("review spec or YAML changed since campaign freeze")
    spec = json.loads((ROOT / SPEC).read_text())
    plan = yaml.safe_load((ROOT / YAML).read_text())
    if (_names_sha(manifest["validation200_files"])
            != manifest["validation200_ordered_names_sha256"]
            or len(manifest["validation200_files"]) != 200
            or len(manifest["validation200_frames"]) != 200
            or sum(manifest["validation200_frames"])
            != manifest["validation200_total_frames"]):
        raise ValueError(f"{output}: frozen validation200 filenames/lengths changed")
    for relative, digest in manifest["compact_predictor_source_sha256"].items():
        path = Path(manifest["compact_source"]) / relative
        if file_hash(path) != digest:
            raise ValueError(f"{path}: compact source SHA changed")
    if file_hash(Path(manifest["compact_artifact"])) != manifest["compact_artifact_sha256"]:
        raise ValueError("compact artifact SHA changed")
    frozen_scorer = (origin / "runs/geometry-loss-screen-4a5516b9b1684353/source")
    original_scorer = json.loads((origin / "experiments/geometry-map-additive-contour-v3.json")
                                 .read_text())["scoring"]
    for key, relative in (
        ("local_score_py_sha256", "local_score.py"),
        ("derive_py_sha256", "fusion_scoring/derive.py"),
        ("lcfs_py_sha256", "fusion_scoring/lcfs.py"),
        ("contour_py_sha256", "fusion_scoring/contour.py"),
        ("o_point_py_sha256", "fusion_scoring/o_point.py"),
        ("common_py_sha256", "fusion_scoring/common.py"),
    ):
        path = frozen_scorer / relative
        if file_hash(path) != original_scorer[key]:
            raise ValueError(f"{path}: native scorer source SHA differs")
    mask = frozen_scorer / "fusion_scoring/masks/d3d_envelope.npz"
    if file_hash(ROOT / "fusion_scoring/masks/d3d_envelope.npz") != file_hash(mask):
        raise ValueError(f"{mask}: native and runner scorer mask SHA differ")
    return manifest, spec, plan


def predict_source(output: Path, jobs: int) -> None:
    """Run the frozen compact predictor in its own interpreter, with no label columns."""
    manifest, _, _ = _verify_run(output)
    script = output / "source/tools/direct_contour_compact_val_predict.py"
    if file_hash(script) != manifest["source_hashes"][
        "tools/direct_contour_compact_val_predict.py"
    ]:
        raise ValueError(f"{script}: frozen isolated predictor script changed")
    env = {**os.environ, "PYTHONPATH": manifest["compact_source"]}
    command = [sys.executable, str(script), "--run", str(output), "--jobs", str(jobs)]
    subprocess.run(command, check=True, cwd=manifest["compact_source"], env=env)


def _target_modules(origin: Path, spec: dict[str, Any]
                    ) -> tuple[Any, Any, Any, Any, np.ndarray, np.ndarray, np.ndarray, int]:
    old = json.loads((origin / "experiments/geometry-map-additive-contour-v3.json")
                     .read_text())["scoring"]
    keys = ("derive_py_sha256", "lcfs_py_sha256", "contour_py_sha256",
            "o_point_py_sha256", "common_py_sha256")
    for key in keys:
        relative = "fusion_scoring/" + key.removesuffix("_py_sha256") + ".py"
        path = ROOT / relative
        if file_hash(path) != old[key]:
            raise ValueError(f"{path}: vendored numerical scorer differs from pinned native")
    local_score, derive, lcfs, contour, common = _frozen_modules({"scorer_provenance": {
        "root": str(ROOT),
        "local_score_py_sha256": file_hash(ROOT / "local_score.py"),
        **{key: old[key] for key in keys},
    }})
    with np.load(ROOT / "fusion_scoring/masks/d3d_envelope.npz") as saved:
        mask = saved["mask_coarse"].astype(bool)
        grid_r, grid_z = saved["grid_R"], saved["grid_Z"]
    return local_score, derive, lcfs, contour, mask, grid_r, grid_z, int(common.AXIS_SIGN["DIII-D"])


def _valid_polygon(c: np.ndarray, R: np.ndarray, Z: np.ndarray) -> bool:
    return bool(c.shape == (512, 2) and np.isfinite(c).all() and _simple_closed(c)
                and (c[:, 0] >= R[0]).all() and (c[:, 0] <= R[-1]).all()
                and (c[:, 1] >= Z[0]).all() and (c[:, 1] <= Z[-1]).all())


def _oracle_one(task: tuple[str, int, int, str, str, str, dict[str, Any], dict[str, Any]]
                ) -> dict[str, Any]:
    name, index, n, data_s, source_s, origin_s, spec, plan = task
    origin = Path(origin_s)
    path = Path(data_s) / "data/diii_d_train" / name
    pinned = json.loads((ROOT / "my_experiments/diii_d_dataset.json").read_text())
    expected_size, expected_hash = pinned["configs"]["diii_d_train"][name]
    if path.stat().st_size != expected_size or file_hash(path) != expected_hash:
        raise ValueError(f"{path}: validation parquet size/SHA differs from pinned manifest")
    local_score, derive, lcfs, contour, mask, grid_r, grid_z, sign = _target_modules(origin, spec)
    row = pd.read_parquet(path, columns=["efit_times", "efit_grid_R", "efit_grid_Z",
                                         "efit_psirz"]).iloc[0]
    R = np.asarray(row["efit_grid_R"], dtype=np.float64)
    Z = np.asarray(row["efit_grid_Z"], dtype=np.float64)
    if not np.array_equal(R, grid_r) or not np.array_equal(Z, grid_z):
        raise ValueError(f"{path}: physical grid differs from frozen scorer")
    truth = local_score._as_psi_stack(row["efit_psirz"])
    with np.load(source_s, allow_pickle=False) as saved:
        source_maps = saved[f"shot_{index:04d}_psirz"]
        q95 = saved[f"shot_{index:04d}_q95"]
        beta = saved[f"shot_{index:04d}_betaN"]
    if (truth.shape != (n, 65, 65) or source_maps.shape != truth.shape
            or source_maps.dtype != np.float32 or q95.shape != (n,)
            or beta.shape != (n,) or q95.dtype != np.float32
            or beta.dtype != np.float32 or not np.isfinite(truth).all()):
        raise ValueError(f"{path}: oracle true/source map or scalar shape/dtype changed, "
                         f"expected ({n},65,65) and ({n},) float32")
    mapped: list[np.ndarray] = []
    profiled: list[np.ndarray] = []
    fallback: Counter[str] = Counter()
    roundtrip: list[float] = []
    rejected_profile: Counter[str] = Counter()
    family_path = ROOT / plan["geometry"]["interior_source_li_family"]
    family = json.loads(family_path.read_text())["family"]
    a_grid = family["candidate_a_grid"]
    if a_grid != [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]:
        raise ValueError(f"{family_path}: fixed source-li candidate grid changed")
    for frame, (psi, true_psi) in enumerate(zip(source_maps, truth, strict=True)):
        reason: str | None = None
        c_pred, _ = lcfs.extract_lcfs_with_sign(
            psi, R, Z, "DIII-D", mask, mask.astype(float), axis_sign=sign
        )
        c_true = lcfs.extract_lcfs(true_psi, R, Z, "DIII-D", mask, mask.astype(float))
        if c_pred is None or c_true is None:
            raise ValueError(f"{name} frame {frame}: source or scorer-native true LCFS missing")
        matched, _ = _cyclic_match(c_pred, c_true)
        displacement = plan["geometry"]["correction_fraction"] * (matched - c_pred)
        blended = c_pred + displacement
        if not _valid_polygon(blended, R, Z):
            reason = "invalid_half_blended_polygon"
        if reason is None:
            ra, za, *_ = derive.magnetic_axis(psi, R, Z, "DIII-D", mask, sign)
            try:
                corrected, _ = correct_map_from_displacement(
                    psi, c_pred, displacement, (ra, za), R, Z
                )
            except NumericalInfeasibility as exc:
                raise NumericalInfeasibility(f"{name} frame {frame}: {exc}") from exc
            c_new, _ = lcfs.extract_lcfs_with_sign(
                corrected, R, Z, "DIII-D", mask, mask.astype(float), axis_sign=sign
            )
            if c_new is None:
                reason = "corrected_map_LCFS_missing"
        if reason is not None:
            corrected = psi.copy()
            chosen = psi.copy()
            fallback[reason] += 1
        else:
            roundtrip.append(contour.symmetric_hausdorff(c_new, blended))
            try:
                chosen, detail = _compensate_frame(
                    psi, corrected, R, Z, mask, sign,
                    a_grid,
                    derive, lcfs, contour,
                )
            except NumericalInfeasibility as exc:
                raise NumericalInfeasibility(f"{name} frame {frame}: source-li {exc}") from exc
            rejected_profile.update(item["reason"] for item in detail["rejected"])
        mapped.append(corrected)
        profiled.append(chosen)
    return {
        "name": name, "frames": n, "source": source_maps,
        "map": np.stack(mapped), "profile": np.stack(profiled),
        "q95": q95, "betaN": beta,
        "fallback_by_reason": dict(fallback),
        "source_li_invalid_a_by_reason": dict(rejected_profile),
        "roundtrip_m": roundtrip,
    }


def _seal_prediction(path: Path, names: list[str], arrays: dict[str, np.ndarray],
                     manifest: dict[str, Any], label: str) -> None:
    _save(path, arrays)
    write_json(path.with_suffix(".json"), {
        "classification": "selection_only_half_contour_oracle_" + label,
        "prediction_sha256": file_hash(path),
        "manifest_sha256": file_hash(path.parent / "manifest.json"),
        "files": names,
        "shots": len(names),
        "frames": manifest["validation200_total_frames"],
        "all_arrays_float32": True,
        "q95_betaN_copied_from_compact_source": True,
    })


def build_oracle(output: Path, jobs: int) -> None:
    """Use val200 truth only for the preregistered half-contour mechanism arm."""
    started = time.perf_counter()
    manifest, spec, plan = _verify_run(output)
    source_path = output / "compact_validation200.npz"
    source_record = json.loads(source_path.with_suffix(".json").read_text())
    if (file_hash(source_path) != source_record["prediction_sha256"]
            or source_record["files"] != manifest["validation200_files"]):
        raise ValueError(f"{source_path}: compact source prediction/files changed")
    if (output / "oracle_half_map.npz").exists():
        raise FileExistsError(f"{output}: oracle was already sealed")
    names = manifest["validation200_files"]
    lengths = manifest["validation200_frames"]
    tasks = [(name, i, lengths[i], manifest["data_dir"], str(source_path),
              manifest["origin_root"], spec, plan) for i, name in enumerate(names)]
    control: dict[str, np.ndarray] = {}
    map_arrays: dict[str, np.ndarray] = {}
    profile_arrays: dict[str, np.ndarray] = {}
    fallback: Counter[str] = Counter()
    invalid_a: Counter[str] = Counter()
    roundtrip: list[float] = []
    try:
        for i, result in enumerate(pimap(_oracle_one, tasks, jobs)):
            if result["name"] != names[i] or result["frames"] != lengths[i]:
                raise ValueError(f"{names[i]}: oracle result order/frame count changed")
            for field, source, corrected, profile in (
                ("psirz", result["source"], result["map"], result["profile"]),
                ("q95", result["q95"], result["q95"], result["q95"]),
                ("betaN", result["betaN"], result["betaN"], result["betaN"]),
            ):
                key = f"shot_{i:04d}_{field}"
                control[key] = source
                map_arrays[key] = corrected
                profile_arrays[key] = profile
            fallback.update(result["fallback_by_reason"])
            invalid_a.update(result["source_li_invalid_a_by_reason"])
            roundtrip.extend(result["roundtrip_m"])
            if (i + 1) % 20 == 0:
                print(f"half-contour oracle: {i + 1}/200 validation shots", flush=True)
    finally:
        release()
    for label, arrays in (("compact_control", control), ("oracle_half_map", map_arrays),
                          ("oracle_half_map_source_li", profile_arrays)):
        _seal_prediction(output / f"{label}.npz", names, arrays, manifest, label)
    for i in range(200):
        for field in ("q95", "betaN"):
            key = f"shot_{i:04d}_{field}"
            if not (control[key].tobytes() == map_arrays[key].tobytes()
                    == profile_arrays[key].tobytes()):
                raise ValueError(f"{names[i]}: submitted {field} bits changed")
    spacing = 0.02656245231628418
    if not roundtrip:
        raise ValueError("validation200 oracle has no non-fallback contour round-trips")
    write_json(output / "oracle_build.json", {
        "classification": "truth_assisted_validation200_mechanism_only",
        "manifest_sha256": file_hash(output / "manifest.json"),
        "files": names,
        "frames": sum(lengths),
        "fallback_by_reason": dict(sorted(fallback.items())),
        "fallback_frames": sum(fallback.values()),
        "source_li_invalid_a_by_reason": dict(sorted(invalid_a.items())),
        "roundtrip_nonfallback_frames": len(roundtrip),
        "roundtrip_median_grid_cells": float(np.median(roundtrip) / spacing),
        "roundtrip_p95_grid_cells": float(np.quantile(roundtrip, 0.95) / spacing),
        "source_prediction_sha256": source_record["prediction_sha256"],
        "control_sha256": file_hash(output / "compact_control.npz"),
        "oracle_map_sha256": file_hash(output / "oracle_half_map.npz"),
        "oracle_profile_sha256": file_hash(output / "oracle_half_map_source_li.npz"),
        "all_submitted_scalars_bit_exact": True,
        "wall_seconds": time.perf_counter() - started,
    })


def score_oracle(output: Path, jobs: int) -> None:
    """Score three sealed validation200 arms through the isolated native scorer."""
    started = time.perf_counter()
    manifest, _, plan = _verify_run(output)
    built = json.loads((output / "oracle_build.json").read_text())
    if (output / "stage0_result.json").exists():
        raise FileExistsError(f"{output}: Stage-0 native score already recorded")
    expected = {"compact_control": built["control_sha256"],
                "oracle_half_map": built["oracle_map_sha256"],
                "oracle_half_map_source_li": built["oracle_profile_sha256"]}
    for label, sha in expected.items():
        path = output / f"{label}.npz"
        meta = json.loads(path.with_suffix(".json").read_text())
        if (file_hash(path) != sha or meta["prediction_sha256"] != sha
                or meta["files"] != manifest["validation200_files"]):
            raise ValueError(f"{path}: sealed oracle prediction changed")
    scorer_root = (Path(manifest["origin_root"])
                   / "runs/geometry-loss-screen-4a5516b9b1684353/source")
    native: dict[str, dict[str, Any]] = {}
    seconds: dict[str, float] = {}
    for label in expected:
        begin = time.perf_counter()
        metrics = output / f"{label}.metrics.json"
        command = [sys.executable, "-m", "tools.historical_compare", "score",
                   "--prediction", str(output / f"{label}.npz"),
                   "--files-json", str(output / "validation200.files.json"),
                   "--data-dir", manifest["data_dir"], "--metrics-json", str(metrics),
                   "--jobs", str(jobs), "--scorer-root", str(scorer_root)]
        subprocess.run(command, check=True, cwd=ROOT,
                       env={**os.environ, "PYTHONPATH": str(ROOT)})
        result = json.loads(metrics.read_text())
        if result["files"] != manifest["validation200_files"]:
            raise ValueError(f"{metrics}: native shot order differs")
        native[label] = result["results"]["model"]
        seconds[label] = time.perf_counter() - begin
    delta = round(native["oracle_half_map_source_li"]["S"]
                  - native["compact_control"]["S"], 6)
    no_fail = all(native[label]["lcfs_fail_frac"] == 0.0
                  and native[label]["cons_fail_frac"] == 0.0 for label in native)
    passing = bool(delta >= plan["gate"]["compact2_oracle_delta_S_minimum"] and no_fail)
    write_json(output / "stage0_result.json", {
        "classification": "fixed_half_contour_oracle_gate_pass" if passing
                          else "fixed_half_contour_oracle_gate_fail",
        "spec_sha256": manifest["spec_sha256"],
        "yaml_sha256": manifest["yaml_sha256"],
        "manifest_sha256": file_hash(output / "manifest.json"),
        "validation200_ordered_names_sha256":
            manifest["validation200_ordered_names_sha256"],
        "shots": 200, "frames": manifest["validation200_total_frames"],
        "native_metrics": native,
        "native_metrics_sha256": {label: file_hash(output / f"{label}.metrics.json")
                                  for label in expected},
        "oracle_profile_minus_control_S": delta,
        "fixed_minimum_delta_S": plan["gate"]["compact2_oracle_delta_S_minimum"],
        "zero_native_failures": no_fail,
        "gate_passed": passing,
        "fallback_by_reason": built["fallback_by_reason"],
        "fallback_frames": built["fallback_frames"],
        "roundtrip_median_grid_cells": built["roundtrip_median_grid_cells"],
        "roundtrip_p95_grid_cells": built["roundtrip_p95_grid_cells"],
        "source_li_invalid_a_by_reason": built["source_li_invalid_a_by_reason"],
        "prediction_sha256": expected,
        "native_scorer_root": str(scorer_root),
        "native_scorer_source_sha256": file_hash(scorer_root / "local_score.py"),
        "scorer_wall_seconds": seconds,
        "score_stage_wall_seconds": time.perf_counter() - started,
        "interpretation": "Truth-assisted fixed-half-contour mechanism on a compact artifact's "
                          "already early-stopped validation window. It is a prefit investment "
                          "gate, neither a mathematical upper bound nor independent/public gain.",
        "no_new_model_fit_or_reserved_eval70_access": True,
    })
    print(f"Stage-0 oracle native delta {delta:+.6f}; pass={passing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "predict", "oracle", "score"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 24:
        raise ValueError(f"jobs={args.jobs}, expected 1..24")
    output = args.run.resolve()
    if args.stage == "prepare":
        prepare(output, args.jobs)
    elif args.stage == "predict":
        predict_source(output, min(args.jobs, 8))
    elif args.stage == "oracle":
        build_oracle(output, args.jobs)
    else:
        score_oracle(output, args.jobs)


if __name__ == "__main__":
    main()

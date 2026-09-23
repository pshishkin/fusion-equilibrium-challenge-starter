"""Fixed input-only li discrepancy guard on the sealed OOF100 oracle maps."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from my_experiments.run import source_files
from toolkit.parallel import pimap, release
from toolkit.records import file_hash, write_json
from tools.oof_li_discrepancy import _one as scorer_li_for_saved_maps

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "experiments/oof-li-guard-v2.json"


def guarded_maps(source: np.ndarray, candidate: np.ndarray, source_li: np.ndarray,
                 candidate_li: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Choose whole original frames for strictly excessive input-only li changes."""
    original = np.asarray(source)
    changed = np.asarray(candidate)
    before = np.asarray(source_li, dtype=np.float64)
    after = np.asarray(candidate_li, dtype=np.float64)
    if (original.ndim != 3 or original.shape[1:] != (65, 65)
            or changed.shape != original.shape or original.dtype != np.float32
            or changed.dtype != np.float32 or before.shape != (len(original),)
            or after.shape != before.shape or not np.isfinite(original).all()
            or not np.isfinite(changed).all() or not np.isfinite(before).all()
            or not np.isfinite(after).all() or not np.isfinite(threshold)
            or threshold <= 0):
        raise ValueError(f"guard source/candidate/maps/li/threshold "
                         f"{original.shape}/{changed.shape}/{original.dtype}/"
                         f"{changed.dtype}/{before.shape}/{after.shape}/{threshold}, "
                         "expected matched finite float32 maps, finite per-frame li and "
                         "positive threshold")
    replace = np.abs(after - before) > threshold
    result = changed.copy()
    result[replace] = original[replace]
    for index in range(len(result)):
        expected = original[index] if replace[index] else changed[index]
        if result[index].tobytes() != expected.tobytes():
            raise ValueError(f"guarded frame {index}: selected map bytes differ from source")
    return result, replace


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"{path}: guarded prediction already exists")
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _verify_inputs(spec: dict[str, Any], origin: Path) -> None:
    for label, record in spec["pinned_inputs"].items():
        path = origin / record["path"]
        if file_hash(path) != record["sha256"]:
            raise ValueError(f"{label} {path}: frozen input SHA differs")
    scorer = origin / spec["pinned_scorer"]["root"]
    checks = {
        "local_score.py": "local_score_py_sha256",
        "fusion_scoring/lcfs.py": "lcfs_py_sha256",
        "fusion_scoring/derive.py": "derive_py_sha256",
        "fusion_scoring/masks/d3d_envelope.npz": "mask_sha256",
    }
    for relative, key in checks.items():
        path = scorer / relative
        if file_hash(path) != spec["pinned_scorer"][key]:
            raise ValueError(f"{path}: pinned scorer {key} differs")


def prepare(run: Path) -> None:
    """Freeze the fixed threshold, source code and same100 input identities."""
    started = time.perf_counter()
    run = run.resolve()
    if run.exists():
        raise FileExistsError(f"{run}: li guard campaign already exists")
    spec = json.loads(SPEC.read_text())
    parent = ROOT / spec["amends_spec"]
    if file_hash(parent) != spec["amends_spec_sha256"]:
        raise ValueError(f"{parent}: reviewed v1 li guard spec SHA differs")
    if file_hash(Path(__file__)) != spec["guard_helper_sha256"]:
        raise ValueError(f"{SPEC}: guard helper source SHA differs")
    _verify_inputs(spec, ROOT)
    files = json.loads((ROOT / spec["pinned_inputs"]["old_files_manifest"]["path"])
                       .read_text())
    campaign = json.loads((ROOT / spec["pinned_inputs"]["old_campaign"]["path"])
                          .read_text())
    names = files["pilot_train_oof_files"]
    if (len(names) != 100 or files["pilot_train_oof_ordered_sha256"]
            != spec["frozen_protocol"]["ordered_filenames_sha256"]
            or sum(campaign["pilot_input_frame_counts"]) != 21667
            or set(names) & (set(files["gate_files"]) | set(files["reserved_tail_files"]))):
        raise ValueError(f"{SPEC}: pilot filenames/frames overlap gate or tail or differ")
    run.mkdir(parents=True)
    source = run / "source"
    source.mkdir()
    for path in source_files():
        target = source / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for path in (SPEC, parent):
        target = source / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    hashes = {str(path.relative_to(ROOT)): file_hash(source / path.relative_to(ROOT))
              for path in source_files()}
    hashes[str(SPEC.relative_to(ROOT))] = file_hash(source / SPEC.relative_to(ROOT))
    hashes[str(parent.relative_to(ROOT))] = file_hash(source / parent.relative_to(ROOT))
    write_json(run / "manifest.json", {
        "classification": "fixed_li_guard_frozen_before_map_build_or_native_score",
        "origin_root": str(ROOT),
        "source_sha256": hashes,
        "spec_sha256": file_hash(SPEC),
        "pilot_files": names,
        "pilot_frame_counts": campaign["pilot_input_frame_counts"],
        "pilot_full_frames": 21667,
        "gate_shots_excluded": len(files["gate_files"]),
        "tail_shots_excluded": len(files["reserved_tail_files"]),
        "prepare_wall_seconds": time.perf_counter() - started,
    })
    subprocess.run([
        sys.executable, "-c",
        "import pathlib,tools.oof_li_guard as g; "
        "assert pathlib.Path(g.__file__).resolve()=="
        "pathlib.Path('tools/oof_li_guard.py').resolve()",
    ], cwd=source, env={**os.environ, "PYTHONPATH": str(source)}, check=True)
    print(f"Frozen li guard {run}: {len(names)} TRAIN shots/21667 frames, threshold 0.05")


def _context(run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if ROOT.resolve() != (run / "source").resolve():
        raise ValueError(f"li guard imported from {ROOT}, expected {run / 'source'}")
    manifest = json.loads((run / "manifest.json").read_text())
    spec_path = run / "source/experiments/oof-li-guard-v2.json"
    spec = json.loads(spec_path.read_text())
    if file_hash(spec_path) != manifest["spec_sha256"]:
        raise ValueError(f"{spec_path}: frozen spec SHA differs")
    for relative, expected in manifest["source_sha256"].items():
        path = run / "source" / relative
        if file_hash(path) != expected:
            raise ValueError(f"{path}: frozen source SHA differs")
    _verify_inputs(spec, Path(manifest["origin_root"]))
    return manifest, spec


def execute(run: Path, jobs: int) -> None:
    """Build one guarded NPZ, then score it with the unchanged native scorer."""
    started = time.perf_counter()
    run = run.resolve()
    manifest, spec = _context(run)
    origin = Path(manifest["origin_root"])
    names = manifest["pilot_files"]
    output = run / "guarded.npz"
    if output.exists() or (run / "result.json").exists():
        raise FileExistsError(f"{run}: li guard output already exists")
    source = origin / spec["pinned_inputs"]["source_prediction"]["path"]
    candidate = origin / spec["pinned_inputs"]["full_normal_prediction"]["path"]
    data = origin.parent / "downloaded_huggingface/hf_dataset"
    scorer = origin / spec["pinned_scorer"]["root"]
    mask = scorer / "fusion_scoring/masks/d3d_envelope.npz"
    tasks = [(name, i, manifest["pilot_frame_counts"][i], str(data), str(source),
              str(candidate), str(mask)) for i, name in enumerate(names)]
    arrays: dict[str, np.ndarray] = {}
    per_shot: list[dict[str, Any]] = []
    fallback_count = 0
    fallback_shots = 0
    threshold = float(spec["frozen_protocol"]["fixed_threshold_dimensionless"])
    if threshold != 0.05:
        raise ValueError(f"{SPEC}: threshold {threshold}, expected frozen 0.05")
    try:
        for i, row in enumerate(pimap(scorer_li_for_saved_maps, tasks, jobs)):
            if row["shot"] != names[i] or row["frames"] != tasks[i][2]:
                raise ValueError(f"{names[i]}: scorer-li worker order/frames differ")
            keys = {field: f"shot_{i:04d}_{field}" for field in ("psirz", "q95", "betaN")}
            with (np.load(source, allow_pickle=False) as original,
                  np.load(candidate, allow_pickle=False) as changed):
                first = np.asarray(original[keys["psirz"]])
                second = np.asarray(changed[keys["psirz"]])
                guarded, selected = guarded_maps(
                    first, second, row["source_li"], row["final_li"], threshold)
                arrays[keys["psirz"]] = guarded
                for field in ("q95", "betaN"):
                    a = np.asarray(original[keys[field]])
                    b = np.asarray(changed[keys[field]])
                    if (a.dtype != b.dtype or a.dtype != np.float32
                            or a.shape != (row["frames"],) or a.tobytes() != b.tobytes()):
                        raise ValueError(f"{names[i]}: {field} source/candidate bits differ")
                    arrays[keys[field]] = a
            indices = np.flatnonzero(selected).astype(int).tolist()
            if indices:
                fallback_shots += 1
            fallback_count += len(indices)
            per_shot.append({
                "shot": names[i], "frames": row["frames"],
                "fallback_frames": indices,
                "fallback_abs_li_change": [
                    float(abs(row["final_li"][j] - row["source_li"][j])) for j in indices
                ],
            })
            if (i + 1) % 20 == 0:
                print(f"li guard built {i + 1}/100 TRAIN shots", flush=True)
    finally:
        release()
    if sum(row["frames"] for row in per_shot) != 21667:
        raise ValueError(f"{run}: guard frame count differs from frozen 21667")
    _atomic_npz(output, arrays)
    write_json(output.with_suffix(".json"), {
        "files": names,
        "prediction_sha256": file_hash(output),
        "source_prediction_sha256": file_hash(source),
        "full_normal_prediction_sha256": file_hash(candidate),
        "fallback_frames": fallback_count,
        "fallback_shots": fallback_shots,
        "threshold": threshold,
        "submitted_scalars_byte_identical": True,
        "target_columns_requested_for_guard": [],
    })
    build_seconds = time.perf_counter() - started
    files_path = run / "files.json"
    write_json(files_path, {"files": names})
    scorer_started = time.perf_counter()
    metrics = run / "guarded.metrics.json"
    subprocess.run([
        sys.executable, "-m", "tools.historical_compare", "score",
        "--prediction", str(output), "--files-json", str(files_path),
        "--data-dir", str(data), "--metrics-json", str(metrics),
        "--jobs", str(jobs), "--scorer-root", str(scorer),
    ], cwd=run / "source", env={**os.environ, "PYTHONPATH": str(run / "source")},
        check=True)
    native = json.loads(metrics.read_text())
    if native["files"] != names:
        raise ValueError(f"{metrics}: native scored filenames differ from frozen100")
    source_native = json.loads((origin / spec["pinned_inputs"]["source_native_json"]["path"])
                               .read_text())
    full = json.loads((origin / spec["pinned_inputs"]["full_normal_native_result"]["path"])
                      .read_text())
    control = source_native["results"]["model"]
    treated = native["results"]["model"]
    full_metric = full["native"]["full_normal_truth_assisted"]
    if control != full["native"]["source"]:
        raise ValueError(f"{run}: reused source control native terms differ from full diagnostic")
    delta = treated["S"] - control["S"]
    fallback_fraction = fallback_count / 21667
    native_valid = bool(treated["lcfs_fail_frac"] == 0
                        and treated["cons_fail_frac"] == 0)
    write_json(run / "result.json", {
        "classification": "fixed_input_only_li_guard_on_truth_assisted_full_normal_OOF100",
        "manifest_sha256": file_hash(run / "manifest.json"),
        "spec_sha256": manifest["spec_sha256"],
        "source_prediction_sha256": file_hash(source),
        "full_normal_prediction_sha256": file_hash(candidate),
        "guarded_prediction_sha256": file_hash(output),
        "guarded_native_json_sha256": file_hash(metrics),
        "source_native": control,
        "unguarded_full_normal_native": full_metric,
        "guarded_native": treated,
        "unguarded_delta_S": full_metric["S"] - control["S"],
        "guarded_delta_S": delta,
        "guarded_minus_unguarded_S": treated["S"] - full_metric["S"],
        "fallback_frames": fallback_count,
        "fallback_fraction": fallback_fraction,
        "fallback_shots": fallback_shots,
        "per_shot_fallbacks": per_shot,
        "threshold": threshold,
        "scalars_byte_identical": True,
        "native_zero_failures": native_valid,
        "investment_passed": bool(fallback_fraction <= 0.02
                                  and delta >= 0.0026 and native_valid),
        "build_wall_seconds": build_seconds,
        "native_scorer_wall_seconds": time.perf_counter() - scorer_started,
        "total_wall_seconds": time.perf_counter() - started,
        "gate200_and_tail70_target_access": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "execute"):
        part = sub.add_parser(command)
        part.add_argument("--run", type=Path, required=True)
        part.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 24:
        raise ValueError(f"jobs={args.jobs}, expected 1..24")
    if args.command == "prepare":
        prepare(args.run)
    else:
        execute(args.run, args.jobs)


if __name__ == "__main__":
    main()

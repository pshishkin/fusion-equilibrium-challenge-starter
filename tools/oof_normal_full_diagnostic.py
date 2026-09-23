"""One train-only full-normal truth-assisted contour diagnostic after the K8 null.

The K8 pilot remains failed. This changes only Fourier truncation: the source map,
normal projection, additive psi formula, input-only source-li step, shot list and
native scorer stay fixed. No evaluation-tail or public label is read.
"""

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
import pandas as pd

# The vendored scorer retains upstream flat imports inside its package modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fusion_scoring"))

from experiments import _as_psirz_stack
from fusion_scoring import common, contour, derive, lcfs
from my_experiments.run import source_files
from toolkit.parallel import pimap, release
from toolkit.records import file_hash, write_json
from tools.contour_learnability import (
    canonical_contour,
    correct_map_from_displacement,
    encode_normal_target,
    outward_normals,
)
from tools.geometry_source_li_compensation import _compensate_frame

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "experiments/oof-full-normal-diagnostic.json"
PROFILE_SPEC = ROOT / "experiments/geometry-additive-source-li-compensation-v3.json"


def full_normal_displacement(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Return all 512 matched normal components in source-contour index order."""
    _, normal_scalar = encode_normal_target(pred, truth)
    canonical, start = canonical_contour(pred)
    normals = outward_normals(canonical)
    movement = np.roll(normal_scalar[:, None] * normals, start, axis=0)
    if movement.shape != (512, 2) or not np.isfinite(movement).all():
        raise ValueError(f"full normal displacement shape/finiteness {movement.shape}/"
                         f"{int((~np.isfinite(movement)).sum())}, expected (512,2)/0")
    return movement


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"{path}: output already sealed")
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def prepare(run: Path) -> None:
    """Freeze approved spec, code, original100 input identities and source artifact."""
    started = time.perf_counter()
    run = run.resolve()
    if run.exists():
        raise FileExistsError(f"{run}: diagnostic run already exists")
    spec = json.loads(SPEC.read_text())
    for label, record in spec["pinned_inputs"].items():
        path = ROOT / record["path"]
        if file_hash(path) != record["sha256"]:
            raise ValueError(f"{label} {path}: SHA differs from frozen spec")
    if file_hash(ROOT / "tools/oof_normal_full_diagnostic.py") != spec["helper_sha256"]:
        raise ValueError(f"{SPEC}: full-normal helper source SHA differs")
    old = ROOT / spec["old_campaign"]
    campaign = json.loads((old / "campaign.json").read_text())
    files = json.loads((old / "files.json").read_text())
    names = files["pilot_train_oof_files"]
    if (len(names) != 100 or files["pilot_train_oof_ordered_sha256"]
            != spec["pilot_ordered_names_sha256"]
            or campaign["pilot_full_frames"] != spec["pilot_full_frames"]
            or set(names) & (set(files["gate_files"]) | set(files["reserved_tail_files"]))):
        raise ValueError(f"{SPEC}: frozen original TRAIN pilot filenames/frames overlap "
                         "gate or evaluation tail")
    run.mkdir(parents=True)
    source = run / "source"
    source.mkdir()
    for path in source_files():
        target = source / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for path in (SPEC, PROFILE_SPEC):
        target = source / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    hashes = {str(path.relative_to(ROOT)): file_hash(source / path.relative_to(ROOT))
              for path in source_files()}
    hashes[str(SPEC.relative_to(ROOT))] = file_hash(source / SPEC.relative_to(ROOT))
    hashes[str(PROFILE_SPEC.relative_to(ROOT))] = file_hash(
        source / PROFILE_SPEC.relative_to(ROOT))
    write_json(run / "manifest.json", {
        "classification": "full_normal_512_train_only_diagnostic_frozen_before_labels",
        "origin_root": str(ROOT),
        "old_campaign": str(old),
        "spec_sha256": file_hash(SPEC),
        "source_sha256": hashes,
        "pilot_files": names,
        "pilot_input_frame_counts": campaign["pilot_input_frame_counts"],
        "pilot_full_frames": campaign["pilot_full_frames"],
        "source_prediction_sha256": spec["pinned_inputs"]["source_prediction"]["sha256"],
        "source_native_metrics_sha256": spec["pinned_inputs"]["source_native_metrics"]["sha256"],
        "gate_shots_excluded": len(files["gate_files"]),
        "evaluation_tail_shots_excluded": len(files["reserved_tail_files"]),
        "prepare_wall_seconds": time.perf_counter() - started,
    })
    check = subprocess.run(
        [sys.executable, "-c",
         "import pathlib,tools.oof_normal_full_diagnostic as d; "
         "assert pathlib.Path(d.__file__).resolve()=="
         "pathlib.Path('tools/oof_normal_full_diagnostic.py').resolve()"],
        cwd=source, env={**os.environ, "PYTHONPATH": str(source)},
        capture_output=True, text=True, check=True,
    )
    print(f"Frozen full-normal diagnostic {run}: {len(names)} original-TRAIN shots, "
          f"{campaign['pilot_full_frames']} frames; isolated import rc={check.returncode}")


def _context(run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if ROOT.resolve() != (run / "source").resolve():
        raise ValueError(f"diagnostic imported from {ROOT}, expected {run / 'source'}")
    manifest = json.loads((run / "manifest.json").read_text())
    spec = json.loads((run / "source/experiments/oof-full-normal-diagnostic.json")
                      .read_text())
    if file_hash(run / "source/experiments/oof-full-normal-diagnostic.json") != manifest[
            "spec_sha256"]:
        raise ValueError(f"{run}: frozen full-normal spec SHA differs")
    for relative, expected in manifest["source_sha256"].items():
        path = run / "source" / relative
        if file_hash(path) != expected:
            raise ValueError(f"{path}: frozen source SHA differs")
    origin = Path(manifest["origin_root"])
    for label, record in spec["pinned_inputs"].items():
        path = origin / record["path"]
        if file_hash(path) != record["sha256"]:
            raise ValueError(f"{label} {path}: pinned input SHA differs")
    return manifest, spec


def _one(task: tuple[str, int, int, str, str, str]) -> dict[str, Any]:
    name, index, frames, data_s, source_s, frozen_s = task
    frozen = Path(frozen_s)
    path = Path(data_s) / "data/diii_d_train" / name
    dataset = json.loads((frozen / "my_experiments/diii_d_dataset.json").read_text())
    expected_size, expected_sha = dataset["configs"]["diii_d_train"][name]
    if path.stat().st_size != expected_size or file_hash(path) != expected_sha:
        raise ValueError(f"{path}: size/SHA differ from frozen dataset manifest")
    row = pd.read_parquet(path, columns=["efit_psirz", "efit_grid_R", "efit_grid_Z"]).iloc[0]
    truth = _as_psirz_stack(row["efit_psirz"]).astype(np.float64)
    with np.load(source_s, allow_pickle=False) as saved:
        maps = np.asarray(saved[f"shot_{index:04d}_psirz"])
    if (maps.shape != (frames, 65, 65) or maps.dtype != np.float32
            or truth.shape != maps.shape or not np.isfinite(maps).all()
            or not np.isfinite(truth).all()):
        raise ValueError(f"{path}: source/truth shape/dtype/finiteness "
                         f"{maps.shape}/{maps.dtype}/{truth.shape}, "
                         f"expected ({frames},65,65)/float32/({frames},65,65) finite")
    R = np.asarray(row["efit_grid_R"], dtype=np.float64)
    Z = np.asarray(row["efit_grid_Z"], dtype=np.float64)
    with np.load(frozen / "fusion_scoring/masks/d3d_envelope.npz") as saved:
        mask = saved["mask_coarse"].astype(bool)
        if not np.array_equal(R, saved["grid_R"]) or not np.array_equal(Z, saved["grid_Z"]):
            raise ValueError(f"{path}: physical grid differs from frozen scorer mask")
    sign = int(common.AXIS_SIGN["DIII-D"])
    a_grid = json.loads((frozen / "experiments/geometry-additive-source-li-compensation-v3.json")
                        .read_text())["family"]["candidate_a_grid"]
    result: list[np.ndarray] = []
    movement_max: list[float] = []
    movement_p95: list[float] = []
    for frame, (psi, target) in enumerate(zip(maps, truth, strict=True)):
        c_source, _ = lcfs.extract_lcfs_with_sign(
            psi, R, Z, "DIII-D", mask, mask.astype(float), axis_sign=sign)
        c_true = lcfs.extract_lcfs(target, R, Z, "DIII-D", mask, mask.astype(float))
        if c_source is None or c_true is None:
            raise ValueError(f"{path} frame {frame}: source/true LCFS missing "
                             f"{c_source is None}/{c_true is None}, expected both present")
        try:
            movement = full_normal_displacement(c_source, c_true)
            axis_r, axis_z, _, _ = derive.magnetic_axis(
                psi, R, Z, "DIII-D", mask, sign)
            additive, _ = correct_map_from_displacement(
                psi, c_source, movement, (axis_r, axis_z), R, Z)
            compensated, _ = _compensate_frame(
                psi, additive, R, Z, mask, sign, a_grid, derive, lcfs, contour)
        except ValueError as exc:
            raise ValueError(f"{path} frame {frame}: {exc}") from exc
        norm = np.linalg.norm(movement, axis=1)
        movement_max.append(float(norm.max()))
        movement_p95.append(float(np.quantile(norm, 0.95)))
        result.append(compensated)
    return {
        "shot": name,
        "frames": frames,
        "psi": np.stack(result).astype(np.float32),
        "max_normal_move_m": max(movement_max),
        "median_frame_p95_normal_move_m": float(np.median(movement_p95)),
    }


def execute(run: Path, jobs: int) -> None:
    """Build one fixed full-normal arm and score it against the pinned K8 control."""
    started = time.perf_counter()
    run = run.resolve()
    manifest, spec = _context(run)
    old = Path(manifest["old_campaign"])
    origin = Path(manifest["origin_root"])
    candidate = run / "full_normal.npz"
    if candidate.exists() or (run / "result.json").exists():
        raise FileExistsError(f"{run}: full-normal output already sealed")
    names = manifest["pilot_files"]
    old_source = old / "pilot/source.npz"
    data = origin.parent / "downloaded_huggingface/hf_dataset"
    tasks = [(name, i, manifest["pilot_input_frame_counts"][i], str(data),
              str(old_source), str(run / "source")) for i, name in enumerate(names)]
    arrays: dict[str, np.ndarray] = {}
    movements: list[dict[str, Any]] = []
    try:
        for i, item in enumerate(pimap(_one, tasks, jobs)):
            if item["shot"] != names[i] or item["frames"] != tasks[i][2]:
                raise ValueError(f"{names[i]}: full-normal worker shot/frame order differs")
            arrays[f"shot_{i:04d}_psirz"] = item["psi"]
            movements.append({"shot": names[i], "frames": item["frames"],
                              "max_normal_move_m": item["max_normal_move_m"],
                              "median_frame_p95_normal_move_m":
                              item["median_frame_p95_normal_move_m"]})
            if (i + 1) % 10 == 0:
                print(f"full-normal map built {i + 1}/100 TRAIN shots", flush=True)
    finally:
        release()
    with np.load(old_source, allow_pickle=False) as source:
        for i, name in enumerate(names):
            for field in ("q95", "betaN"):
                key = f"shot_{i:04d}_{field}"
                values = np.asarray(source[key])
                if (values.shape != (manifest["pilot_input_frame_counts"][i],)
                        or values.dtype != np.float32):
                    raise ValueError(f"{name}: source {field} shape/dtype {values.shape}/"
                                     f"{values.dtype}, expected frozen frames/float32")
                arrays[key] = values
    _atomic_npz(candidate, arrays)
    write_json(candidate.with_suffix(".json"), {
        "classification": "same100_truth_assisted_full_normal_maps_sealed",
        "files": names,
        "prediction_sha256": file_hash(candidate),
        "source_prediction_sha256": manifest["source_prediction_sha256"],
        "submitted_scalars_byte_identical_to_source": True,
        "target_columns_used_only_for_train_truth_assisted_map": ["efit_psirz"],
    })
    build_seconds = time.perf_counter() - started
    scorer_root = origin / spec["native_scorer_root"]
    if file_hash(scorer_root / "local_score.py") != spec["native_scorer_source_sha256"]:
        raise ValueError(f"{scorer_root}: pinned native scorer SHA differs")
    files_json = run / "files.json"
    write_json(files_json, {"files": names})
    metrics = run / "full_normal.metrics.json"
    scorer_started = time.perf_counter()
    subprocess.run([
        sys.executable, "-m", "tools.historical_compare", "score",
        "--prediction", str(candidate), "--files-json", str(files_json),
        "--data-dir", str(data), "--metrics-json", str(metrics),
        "--jobs", str(jobs), "--scorer-root", str(scorer_root),
    ], cwd=run / "source", env={**os.environ, "PYTHONPATH": str(run / "source")},
        check=True)
    scored = json.loads(metrics.read_text())
    if scored["files"] != names or scored["n_shots"] != 100:
        raise ValueError(f"{metrics}: scorer shot names/count differ from frozen pilot")
    old_metric_path = origin / spec["pinned_inputs"]["source_native_metrics"]["path"]
    old_scored = json.loads(old_metric_path.read_text())
    if old_scored["files"] != names:
        raise ValueError(f"{old_metric_path}: reused source native files differ")
    control = old_scored["results"]["model"]
    treated = scored["results"]["model"]
    k8 = json.loads((origin / spec["pinned_inputs"]["k8_result"]["path"]).read_text())
    delta = treated["S"] - control["S"]
    write_json(run / "result.json", {
        "classification": "full_normal_512_vs_K8_same100_train_truth_assisted_diagnostic",
        "manifest_sha256": file_hash(run / "manifest.json"),
        "spec_sha256": manifest["spec_sha256"],
        "source_prediction_sha256": manifest["source_prediction_sha256"],
        "full_normal_prediction_sha256": file_hash(candidate),
        "candidate_native_json_sha256": file_hash(metrics),
        "reused_control_native_json_sha256": manifest["source_native_metrics_sha256"],
        "pilot_shots": len(names),
        "pilot_frames": manifest["pilot_full_frames"],
        "source_native": control,
        "full_normal_native": treated,
        "full_normal_delta_S": delta,
        "k8_delta_S_same100": k8["pilot"]["native_delta_S"],
        "full_minus_K8_delta_S": delta - k8["pilot"]["native_delta_S"],
        "reference_investment_margin_not_a_new_gate": 0.0026,
        "full_reaches_reference_margin": bool(delta >= 0.0026),
        "per_shot_normal_movement": movements,
        "build_wall_seconds": build_seconds,
        "native_scorer_wall_seconds": time.perf_counter() - scorer_started,
        "total_wall_seconds": time.perf_counter() - started,
        "gate200_and_tail70_target_read": False,
        "new_fit_or_public_prediction": False,
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

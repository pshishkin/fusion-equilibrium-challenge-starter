"""Train-only scorer-LCFS Fourier calibration before any direct contour fit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from my_experiments.data_check import check_dataset
from my_experiments.run import source_files
from toolkit.parallel import pimap, release
from toolkit.records import file_hash, write_json
from tools.geometry_map_smoke import _frozen_modules, _simple_closed

ROOT = Path(__file__).resolve().parent.parent
SPEC_RELATIVE = Path("experiments/direct-contour-representation-prereg.json")
ARTIFACT_RELATIVE = Path("runs/compact-production-9301af36275620f7/base_s0/baseline.joblib")


def _name_hash(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def _canonical_true(contour: np.ndarray) -> np.ndarray:
    """Orient the scorer's 512-point polygon CCW, then anchor its outboard point."""
    c = np.asarray(contour, dtype=np.float64)
    if c.shape != (512, 2) or not np.isfinite(c).all() or not _simple_closed(c):
        raise ValueError(f"true scorer LCFS shape/finiteness/topology {c.shape}, "
                         "expected simple finite (512,2)")
    nxt = np.roll(c, -1, axis=0)
    signed_area2 = float(np.sum(c[:, 0] * nxt[:, 1] - nxt[:, 0] * c[:, 1]))
    if not np.isfinite(signed_area2) or signed_area2 == 0:
        raise ValueError(f"true scorer LCFS signed area {signed_area2}, expected nonzero")
    if signed_area2 < 0:
        c = c[::-1]
    return np.roll(c, -int(np.argmax(c[:, 0])), axis=0)


def _project(contour: np.ndarray, modes: int) -> tuple[np.ndarray, float]:
    """Orthogonal complex Fourier projection of both physical contour coordinates."""
    if modes not in (8, 16) or contour.shape != (512, 2):
        raise ValueError(f"contour/modes {contour.shape}/{modes}, expected (512,2)/8 or 16")
    spectrum = np.fft.rfft(contour, axis=0)
    spectrum[modes + 1:] = 0
    reconstructed = np.fft.irfft(spectrum, n=512, axis=0)
    centered = contour - contour.mean(axis=0)
    reconstructed_centered = reconstructed - reconstructed.mean(axis=0)
    energy = float(np.sum(reconstructed_centered**2) / np.sum(centered**2))
    if not np.isfinite(energy) or not np.isfinite(reconstructed).all():
        raise ValueError("absolute contour Fourier projection is nonfinite")
    return reconstructed, energy


def _metadata_one(task: tuple[str, str]) -> tuple[str, int]:
    name, data_s = task
    path = Path(data_s) / "data/diii_d_train" / name
    times = np.asarray(pd.read_parquet(path, columns=["efit_times"]).iloc[0]["efit_times"],
                       dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError(f"{path}: efit_times shape/finiteness {times.shape}, expected finite (T,)")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError(f"{path}: efit_times is not strictly increasing")
    return name, len(times)


def _indices(n: int) -> list[int]:
    if n < 3:
        raise ValueError(f"efit_times has {n} frames, expected at least 3 eligible frames")
    result = [int(np.floor(q * (n - 1) + 0.5)) for q in (0.1, 0.5, 0.9)]
    if len(set(result)) != 3:
        raise ValueError(f"{n} efit frames produce duplicate fixed probe indices {result}")
    return result


def prepare(output: Path, jobs: int) -> None:
    """Freeze source and metadata-selected train probes before opening any true ψ."""
    started = time.perf_counter()
    origin = ROOT.resolve()
    spec_path = origin / SPEC_RELATIVE
    spec_sha = file_hash(spec_path)
    spec = json.loads(spec_path.read_text())
    artifact = origin / ARTIFACT_RELATIVE
    if file_hash(artifact) != spec["split_provenance"]["compact_artifact_sha256"]:
        raise ValueError(f"{artifact}: compact artifact SHA differs from preregistration")
    scored_path = origin / spec["split_provenance"]["compact_score_files"]
    if file_hash(scored_path) != spec["split_provenance"]["compact_score_files_sha256"]:
        raise ValueError(f"{scored_path}: evaluation filename record SHA differs")
    packed = joblib.load(artifact)
    train = list(packed["train_files"])
    validation = list(packed["val_files"])
    score = json.loads(scored_path.read_text())["files"]
    pinned = spec["split_provenance"]
    checks = (
        ("train", train, pinned["train_shots"], pinned["train_ordered_names_sha256"]),
        ("validation", validation, pinned["validation_shots"],
         pinned["validation_ordered_names_sha256"]),
        ("evaluation", score, pinned["evaluation_only_tail_shots"],
         pinned["evaluation_only_tail_ordered_names_sha256"]),
    )
    for label, names, expected_n, expected_sha in checks:
        if len(names) != expected_n or _name_hash(names) != expected_sha:
            raise ValueError(f"{artifact}: {label} filenames/count differ from preregistration")
    if (set(train) & set(validation) or set(train) & set(score)
            or set(validation) & set(score)):
        raise ValueError("compact train/validation/evaluation filenames overlap")
    data = (origin / "../downloaded_huggingface/hf_dataset").resolve()
    corpus = check_dataset(data, "diii_d_train")
    if corpus["manifest_sha256"] != pinned["dataset_manifest_sha256"]:
        raise ValueError("pinned DIII-D dataset manifest SHA changed")
    if output.exists():
        raise FileExistsError(f"{output}: frozen diagnostic directory exists")
    output.mkdir(parents=True)
    snapshot = output / "source"
    source_hashes: dict[str, str] = {}
    for path in source_files():
        relative = path.relative_to(origin)
        dest = snapshot / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        source_hashes[str(relative)] = file_hash(dest)
        if source_hashes[str(relative)] != file_hash(path):
            raise ValueError(f"{path}: source snapshot SHA differs")
    frozen_spec = snapshot / SPEC_RELATIVE
    frozen_spec.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(spec_path, frozen_spec)
    source_hashes[str(SPEC_RELATIVE)] = file_hash(frozen_spec)
    if source_hashes[str(SPEC_RELATIVE)] != spec_sha:
        raise ValueError(f"{frozen_spec}: prereg spec SHA differs after copy")
    examined: list[tuple[str, int]] = []
    try:
        for result in pimap(_metadata_one, [(name, str(data)) for name in train[:200]], jobs):
            examined.append(result)
    finally:
        release()
    selected: list[dict[str, Any]] = []
    for name, n in examined:
        if n >= 3:
            selected.append({"file": name, "frames": n, "probe_indices": _indices(n)})
        if len(selected) == 100:
            break
    if len(selected) != 100:
        raise ValueError(f"first 200 train shots yielded {len(selected)} eligible, expected 100")
    names = [item["file"] for item in selected]
    manifest = {
        "status": "frozen_before_truth_psi_read",
        "spec": str(SPEC_RELATIVE),
        "spec_sha256": spec_sha,
        "origin_root": str(origin),
        "data_dir": str(data),
        "source_hashes": source_hashes,
        "compact_artifact_sha256": file_hash(artifact),
        "compact_train_ordered_names_sha256": _name_hash(train),
        "compact_validation_ordered_names_sha256": _name_hash(validation),
        "evaluation_only_ordered_names_sha256": _name_hash(score),
        "selected": selected,
        "selected_ordered_names_sha256": _name_hash(names),
        "selected_probe_count": sum(len(item["probe_indices"]) for item in selected),
        "dataset_preflight": corpus,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "pid_namespace": os.readlink("/proc/self/ns/pid"),
        "prepare_wall_seconds": time.perf_counter() - started,
    }
    write_json(output / "manifest.json", manifest)
    print(f"Frozen {output}: {len(selected)} train-only shots, "
          f"{manifest['selected_probe_count']} fixed probes, "
          f"{manifest['prepare_wall_seconds']:.2f} s")


def _verify_run(output: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    manifest = json.loads((output / "manifest.json").read_text())
    if ROOT.resolve() != (output / "source").resolve():
        raise ValueError(f"runner imported from {ROOT}, expected {output / 'source'}")
    for relative, expected in manifest["source_hashes"].items():
        path = ROOT / relative
        if file_hash(path) != expected:
            raise ValueError(f"{path}: frozen source SHA changed")
    origin = Path(manifest["origin_root"])
    if file_hash(origin / SPEC_RELATIVE) != manifest["spec_sha256"]:
        raise ValueError(f"{origin / SPEC_RELATIVE}: prereg spec SHA changed")
    spec = json.loads((ROOT / SPEC_RELATIVE).read_text())
    names = [item["file"] for item in manifest["selected"]]
    if (_name_hash(names) != manifest["selected_ordered_names_sha256"]
            or len(names) != 100 or set(names) & set(json.loads(
                (origin / spec["split_provenance"]["compact_score_files"]).read_text()
            )["files"])):
        raise ValueError(f"{output}: selected files changed or overlap evaluation tail")
    scorer_root = origin / spec["scorer_provenance"]["frozen_root"]
    for key, relative in (
        ("local_score_py_sha256", "local_score.py"),
        ("derive_py_sha256", "fusion_scoring/derive.py"),
        ("lcfs_py_sha256", "fusion_scoring/lcfs.py"),
        ("contour_py_sha256", "fusion_scoring/contour.py"),
        ("o_point_py_sha256", "fusion_scoring/o_point.py"),
        ("common_py_sha256", "fusion_scoring/common.py"),
        ("diii_d_envelope_npz_sha256", "fusion_scoring/masks/d3d_envelope.npz"),
    ):
        path = scorer_root / relative
        if file_hash(path) != spec["scorer_provenance"][key]:
            raise ValueError(f"{path}: scorer source SHA changed")
    return manifest, spec, scorer_root


def _probe_one(task: tuple[dict[str, Any], str, str, dict[str, Any]]
               ) -> dict[str, Any]:
    selected, data_s, scorer_s, spec = task
    name = selected["file"]
    path = Path(data_s) / "data/diii_d_train" / name
    pinned = json.loads((ROOT / spec["split_provenance"]["dataset_manifest"]).read_text())
    expected_size, expected_sha = pinned["configs"]["diii_d_train"][name]
    if path.stat().st_size != expected_size or file_hash(path) != expected_sha:
        raise ValueError(f"{path}: size/SHA differs from pinned DIII-D manifest")
    scorer = spec["scorer_provenance"]
    modules = {"scorer_provenance": {
        "root": scorer_s,
        **{key: val for key, val in scorer.items() if key.endswith("_py_sha256")},
    }}
    local_score, _, lcfs, contour, _ = _frozen_modules(modules)
    with np.load(Path(scorer_s) / "fusion_scoring/masks/d3d_envelope.npz") as masks:
        mask = masks["mask_coarse"].astype(bool)
        mask_r = masks["grid_R"]
        mask_z = masks["grid_Z"]
    row = pd.read_parquet(path, columns=["efit_times", "efit_grid_R", "efit_grid_Z",
                                         "efit_psirz"]).iloc[0]
    times = np.asarray(row["efit_times"], dtype=np.float64)
    R = np.asarray(row["efit_grid_R"], dtype=np.float64)
    Z = np.asarray(row["efit_grid_Z"], dtype=np.float64)
    if (len(times) != selected["frames"] or not np.array_equal(R, mask_r)
            or not np.array_equal(Z, mask_z)):
        raise ValueError(f"{path}: efit frames/grid differ from frozen metadata/scorer: "
                         f"{len(times)} vs {selected['frames']}")
    truth = local_score._as_psi_stack(row["efit_psirz"])
    if truth.shape != (len(times), 65, 65) or not np.isfinite(truth).all():
        raise ValueError(f"{path}: true psi {truth.shape}/"
                         f"{int((~np.isfinite(truth)).sum())} nonfinite, "
                         f"expected ({len(times)},65,65)/0")
    spacing = float(min(np.diff(R).min(), np.diff(Z).min()))
    rows: list[dict[str, Any]] = []
    for frame in selected["probe_indices"]:
        true_c = lcfs.extract_lcfs(truth[frame], R, Z, "DIII-D", mask,
                                   mask.astype(float))
        if true_c is None:
            raise ValueError(f"{name} frame {frame}: scorer-native true LCFS missing")
        try:
            canonical = _canonical_true(true_c)
        except ValueError as exc:
            raise ValueError(f"{name} frame {frame}: {exc}") from exc
        detail: dict[str, Any] = {"file": name, "frame": frame, "spacing_m": spacing}
        centered_true_energy = float(np.sum((canonical - canonical.mean(axis=0)) ** 2))
        for modes in (8, 16):
            projected, energy = _project(canonical, modes)
            centered_reconstructed_energy = float(
                np.sum((projected - projected.mean(axis=0)) ** 2)
            )
            hausdorff = contour.symmetric_hausdorff(projected, canonical)
            simple = bool(_simple_closed(projected))
            in_grid = bool(
                (projected[:, 0] >= R[0]).all() and (projected[:, 0] <= R[-1]).all()
                and (projected[:, 1] >= Z[0]).all() and (projected[:, 1] <= Z[-1]).all()
            )
            detail[f"K{modes}"] = {
                "centered_shape_energy_retained": energy,
                "centered_shape_energy_numerator": centered_reconstructed_energy,
                "centered_shape_energy_denominator": centered_true_energy,
                "symmetric_hausdorff_m": hausdorff,
                "simple_polygon": simple,
                "within_physical_grid": in_grid,
            }
        rows.append(detail)
    return {"name": name, "frames": len(times), "rows": rows}


def run(output: Path, jobs: int) -> None:
    """Measure only the preregistered 300 TRAIN contour probes."""
    started = time.perf_counter()
    manifest, spec, scorer_root = _verify_run(output)
    if (output / "representation.json").exists():
        raise FileExistsError(f"{output}: representation already measured")
    tasks = [(item, manifest["data_dir"], str(scorer_root), spec)
             for item in manifest["selected"]]
    rows: list[dict[str, Any]] = []
    try:
        for i, response in enumerate(pimap(_probe_one, tasks, jobs)):
            wanted = manifest["selected"][i]
            if response["name"] != wanted["file"] or response["frames"] != wanted["frames"]:
                raise ValueError(f"{wanted['file']}: probe response filename/frame mismatch")
            if [item["frame"] for item in response["rows"]] != wanted["probe_indices"]:
                raise ValueError(f"{wanted['file']}: returned probe indices differ")
            rows.extend(response["rows"])
            if (i + 1) % 20 == 0:
                print(f"train-only contour representation: {i + 1}/100 shots", flush=True)
    finally:
        release()
    if len(rows) != 300:
        raise ValueError(f"{output}: {len(rows)} measured probes, expected 300")
    summaries: dict[str, Any] = {}
    passing: list[int] = []
    for modes in (8, 16):
        label = f"K{modes}"
        h = np.asarray([row[label]["symmetric_hausdorff_m"] for row in rows])
        numerator = sum(row[label]["centered_shape_energy_numerator"] for row in rows)
        denominator = sum(row[label]["centered_shape_energy_denominator"] for row in rows)
        spacings = np.asarray([row["spacing_m"] for row in rows])
        invalid_simple = sum(not row[label]["simple_polygon"] for row in rows)
        outside = sum(not row[label]["within_physical_grid"] for row in rows)
        median_h = float(np.median(h))
        p95_h = float(np.quantile(h, 0.95))
        cell = float(np.min(spacings))
        passed = invalid_simple == 0 and outside == 0 and median_h <= 0.25 * cell \
            and p95_h <= 0.5 * cell
        if passed:
            passing.append(modes)
        summaries[label] = {
            "median_hausdorff_m": median_h,
            "p95_hausdorff_m": p95_h,
            "max_hausdorff_m": float(np.max(h)),
            "median_hausdorff_in_min_grid_cells": median_h / cell,
            "p95_hausdorff_in_min_grid_cells": p95_h / cell,
            "pooled_centered_shape_energy_retained": float(numerator / denominator),
            "minimum_grid_spacing_m": cell,
            "non_simple_polygon_count": invalid_simple,
            "out_of_grid_count": outside,
            "gate_passed": bool(passed),
        }
    chosen = passing[0] if passing else None
    write_json(output / "representation.json", {
        "classification": "absolute_contour_representation_feasible" if chosen is not None
                          else "absolute_contour_representation_gate_failed",
        "spec_sha256": manifest["spec_sha256"],
        "manifest_sha256": file_hash(output / "manifest.json"),
        "train_sample_ordered_names_sha256": manifest["selected_ordered_names_sha256"],
        "train_shots": 100,
        "train_probes": 300,
        "scorer_root": str(scorer_root),
        "K": summaries,
        "selected_smallest_passing_K": chosen,
        "all_probe_rows": rows,
        "wall_seconds": time.perf_counter() - started,
        "no_model_fit_or_score_tail_access": True,
    })
    print(f"Train-only representation: K8 {summaries['K8']}, K16 {summaries['K16']}; "
          f"selected K={chosen}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 4:
        raise ValueError(f"jobs={args.jobs}, expected 1..4")
    output = args.out.resolve()
    if args.stage == "prepare":
        prepare(output, args.jobs)
    else:
        try:
            run(output, args.jobs)
        except Exception as exc:
            write_json(output / "representation_failure.json", {
                "classification": "representation_numeric_or_input_failure",
                "manifest_sha256": file_hash(output / "manifest.json"),
                "exception_type": type(exc).__name__,
                "error": str(exc),
                "not_a_scientific_negative": True,
            })
            raise


if __name__ == "__main__":
    main()

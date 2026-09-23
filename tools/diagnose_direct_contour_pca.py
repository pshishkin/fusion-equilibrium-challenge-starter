"""One frozen train-only PCA contour representation gate after Fourier failure."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from my_experiments.data_check import check_dataset
from my_experiments.run import source_files
from toolkit.parallel import pimap, release
from toolkit.records import file_hash, write_json
from tools.diagnose_direct_contour_representation import (
    _canonical_true,
    _indices,
    _metadata_one,
    _name_hash,
)
from tools.geometry_map_smoke import _frozen_modules, _simple_closed

ROOT = Path(__file__).resolve().parent.parent
SPEC_RELATIVE = Path("experiments/direct-contour-pca-representation-review.json")


def _verify_prereg(origin: Path) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    spec_path = origin / SPEC_RELATIVE
    spec = json.loads(spec_path.read_text())
    prior_result = origin / spec["prior_negative"]["result"]
    if file_hash(prior_result) != spec["prior_negative"]["result_sha256"]:
        raise ValueError(f"{prior_result}: preceding Fourier result SHA changed")
    prior = origin / spec["basis_training_sample"]["frozen_prior_manifest"]
    if file_hash(prior) != spec["basis_training_sample"]["frozen_prior_manifest_sha256"]:
        raise ValueError(f"{prior}: prior train-sample manifest SHA changed")
    prior_manifest = json.loads(prior.read_text())
    prior_names = [item["file"] for item in prior_manifest["selected"]]
    if (len(prior_names) != 100 or _name_hash(prior_names)
            != spec["basis_training_sample"]["ordered_names_sha256"]
            or sum(len(item["probe_indices"]) for item in prior_manifest["selected"]) != 300):
        raise ValueError(f"{prior}: prior sample names or fixed probes changed")
    artifact = origin / spec["fixed_split"]["compact_artifact"]
    if file_hash(artifact) != spec["fixed_split"]["compact_artifact_sha256"]:
        raise ValueError(f"{artifact}: compact artifact SHA differs")
    packed = joblib.load(artifact)
    train = list(packed["train_files"])
    val = list(packed["val_files"])
    fixed = spec["fixed_split"]
    score_path = origin / "runs/compact-production-9301af36275620f7/base_s0/metrics.json"
    score = json.loads(score_path.read_text())["files"]
    if (len(train) != fixed["compact_train_shots"]
            or _name_hash(train) != fixed["compact_train_ordered_names_sha256"]
            or len(val) != fixed["compact_validation_shots"]
            or _name_hash(val) != fixed["compact_validation_ordered_names_sha256"]
            or len(score) != fixed["reserved_evaluation_tail_shots"]
            or _name_hash(score) != fixed["reserved_evaluation_ordered_names_sha256"]
            or set(train) & set(val) or set(train) & set(score) or set(val) & set(score)):
        raise ValueError(f"{artifact}: compact train/validation/evaluation split changed")
    if train[:100] != prior_names:
        raise ValueError(f"{prior}: prior basis sample is not the first 100 compact TRAIN shots")
    return spec, prior_manifest, train


def prepare(output: Path, jobs: int) -> None:
    """Freeze exact second-sample filenames and source before reading its true psi."""
    started = time.perf_counter()
    origin = ROOT.resolve()
    spec, prior, train = _verify_prereg(origin)
    data = (origin / "../downloaded_huggingface/hf_dataset").resolve()
    corpus = check_dataset(data, "diii_d_train")
    if corpus["manifest_sha256"] != spec["fixed_split"]["dataset_manifest_sha256"]:
        raise ValueError("pinned DIII-D corpus manifest SHA changed")
    if output.exists():
        raise FileExistsError(f"{output}: frozen PCA gate directory exists")
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
    frozen_spec = snapshot / SPEC_RELATIVE
    frozen_spec.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origin / SPEC_RELATIVE, frozen_spec)
    hashes[str(SPEC_RELATIVE)] = file_hash(frozen_spec)
    examined: list[tuple[str, int]] = []
    try:
        for item in pimap(_metadata_one,
                          [(name, str(data)) for name in train[100:300]], jobs):
            examined.append(item)
    finally:
        release()
    check: list[dict[str, Any]] = []
    for name, n in examined:
        if n >= 3:
            check.append({"file": name, "frames": n, "probe_indices": _indices(n)})
        if len(check) == 100:
            break
    if len(check) != 100:
        raise ValueError(f"next 200 TRAIN shots yielded {len(check)} eligible, expected 100")
    basis = prior["selected"]
    basis_names = [item["file"] for item in basis]
    check_names = [item["file"] for item in check]
    if set(basis_names) & set(check_names) or set(check_names) & set(train[300:]):
        raise ValueError("basis and representation-check samples overlap or leave fixed window")
    manifest = {
        "status": "frozen_before_second_sample_truth_psi_read",
        "spec_sha256": file_hash(origin / SPEC_RELATIVE),
        "prior_manifest_sha256": file_hash(
            origin / spec["basis_training_sample"]["frozen_prior_manifest"]
        ),
        "origin_root": str(origin),
        "data_dir": str(data),
        "source_hashes": hashes,
        "basis_sample": basis,
        "basis_sample_ordered_names_sha256": _name_hash(basis_names),
        "representation_check_sample": check,
        "representation_check_ordered_names_sha256": _name_hash(check_names),
        "compact_train_ordered_names_sha256": _name_hash(train),
        "dataset_preflight": corpus,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "pid_namespace": os.readlink("/proc/self/ns/pid"),
        "prepare_wall_seconds": time.perf_counter() - started,
    }
    write_json(output / "manifest.json", manifest)
    print(f"Frozen {output}: basis 100/300, check 100/300, "
          f"{manifest['prepare_wall_seconds']:.2f} s")


def _verify_run(output: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    manifest = json.loads((output / "manifest.json").read_text())
    if ROOT.resolve() != (output / "source").resolve():
        raise ValueError(f"runner imported from {ROOT}, expected {output / 'source'}")
    for relative, digest in manifest["source_hashes"].items():
        path = ROOT / relative
        if file_hash(path) != digest:
            raise ValueError(f"{path}: frozen source SHA changed")
    origin = Path(manifest["origin_root"])
    if file_hash(origin / SPEC_RELATIVE) != manifest["spec_sha256"]:
        raise ValueError(f"{origin / SPEC_RELATIVE}: PCA review spec changed")
    spec = json.loads((ROOT / SPEC_RELATIVE).read_text())
    artifact = origin / spec["fixed_split"]["compact_artifact"]
    if file_hash(artifact) != spec["fixed_split"]["compact_artifact_sha256"]:
        raise ValueError(f"{artifact}: compact artifact SHA changed")
    prior_path = origin / spec["basis_training_sample"]["frozen_prior_manifest"]
    if file_hash(prior_path) != manifest["prior_manifest_sha256"]:
        raise ValueError(f"{prior_path}: basis-training sample manifest changed")
    basis = manifest["basis_sample"]
    check = manifest["representation_check_sample"]
    if (len(basis) != 100 or len(check) != 100
            or _name_hash([item["file"] for item in basis])
            != manifest["basis_sample_ordered_names_sha256"]
            or _name_hash([item["file"] for item in check])
            != manifest["representation_check_ordered_names_sha256"]
            or set(item["file"] for item in basis) & set(item["file"] for item in check)):
        raise ValueError(f"{output}: sealed PCA sample identities overlap or changed")
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
            raise ValueError(f"{path}: pinned scorer source SHA differs")
    return manifest, spec, scorer_root


def _contours_one(task: tuple[dict[str, Any], str, str, dict[str, Any]]
                  ) -> dict[str, Any]:
    sample, data_s, scorer_s, spec = task
    name = sample["file"]
    path = Path(data_s) / "data/diii_d_train" / name
    pinned = json.loads((ROOT / spec["fixed_split"]["dataset_manifest"]).read_text())
    expected_size, expected_sha = pinned["configs"]["diii_d_train"][name]
    if path.stat().st_size != expected_size or file_hash(path) != expected_sha:
        raise ValueError(f"{path}: size/SHA differs from pinned DIII-D manifest")
    scorer = spec["scorer_provenance"]
    local_score, _, lcfs, _, _ = _frozen_modules({"scorer_provenance": {
        "root": scorer_s,
        **{key: value for key, value in scorer.items() if key.endswith("_py_sha256")},
    }})
    with np.load(Path(scorer_s) / "fusion_scoring/masks/d3d_envelope.npz") as masks:
        mask = masks["mask_coarse"].astype(bool)
        mask_r = masks["grid_R"]
        mask_z = masks["grid_Z"]
    row = pd.read_parquet(path, columns=["efit_times", "efit_grid_R", "efit_grid_Z",
                                         "efit_psirz"]).iloc[0]
    times = np.asarray(row["efit_times"], dtype=np.float64)
    R = np.asarray(row["efit_grid_R"], dtype=np.float64)
    Z = np.asarray(row["efit_grid_Z"], dtype=np.float64)
    if (len(times) != sample["frames"] or not np.array_equal(R, mask_r)
            or not np.array_equal(Z, mask_z)):
        raise ValueError(f"{path}: efit frames/grid differ from frozen sample/scorer: "
                         f"{len(times)} vs {sample['frames']}")
    truth = local_score._as_psi_stack(row["efit_psirz"])
    if truth.shape != (len(times), 65, 65) or not np.isfinite(truth).all():
        raise ValueError(f"{path}: true psi {truth.shape}/"
                         f"{int((~np.isfinite(truth)).sum())} nonfinite, "
                         f"expected ({len(times)},65,65)/0")
    contours: list[np.ndarray] = []
    for frame in sample["probe_indices"]:
        contour = lcfs.extract_lcfs(truth[frame], R, Z, "DIII-D", mask,
                                    mask.astype(float))
        if contour is None:
            raise ValueError(f"{name} frame {frame}: scorer-native true LCFS missing")
        try:
            contours.append(_canonical_true(contour))
        except ValueError as exc:
            raise ValueError(f"{name} frame {frame}: {exc}") from exc
    return {
        "name": name,
        "frames": len(times),
        "indices": sample["probe_indices"],
        "contours": np.stack(contours),
        "grid_R_minmax": [float(R[0]), float(R[-1])],
        "grid_Z_minmax": [float(Z[0]), float(Z[-1])],
        "spacing_m": float(min(np.diff(R).min(), np.diff(Z).min())),
    }


def _collect(sample: list[dict[str, Any]], data: str, scorer: str,
             spec: dict[str, Any], jobs: int, label: str
             ) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    try:
        for i, response in enumerate(pimap(_contours_one,
                                           [(item, data, scorer, spec) for item in sample], jobs)):
            item = sample[i]
            if (response["name"] != item["file"] or response["frames"] != item["frames"]
                    or response["indices"] != item["probe_indices"]):
                raise ValueError(f"{item['file']}: {label} result order/indices differ")
            rows.append(response)
            if (i + 1) % 20 == 0:
                print(f"PCA {label} contours: {i + 1}/100 shots", flush=True)
    finally:
        release()
    contours = np.concatenate([row["contours"] for row in rows])
    if contours.shape != (300, 512, 2):
        raise ValueError(f"{label}: {contours.shape} contours, expected (300,512,2)")
    return contours, rows


def run(output: Path, jobs: int) -> None:
    """Fit one PCA basis on TRAIN sample A; read only sample B for its gate."""
    started = time.perf_counter()
    manifest, spec, scorer_root = _verify_run(output)
    if (output / "representation.json").exists():
        raise FileExistsError(f"{output}: PCA representation already measured")
    basis, _ = _collect(manifest["basis_sample"], manifest["data_dir"],
                        str(scorer_root), spec, jobs, "basis")
    basis_read_seconds = time.perf_counter() - started
    fit_started = time.perf_counter()
    X_basis = basis.reshape(300, 1024)
    pca = PCA(n_components=64, svd_solver="full", whiten=False, random_state=0)
    pca.fit(X_basis)
    fit_seconds = time.perf_counter() - fit_started
    model_path = output / "contour_pca64.joblib"
    joblib.dump(pca, model_path)
    check_started = time.perf_counter()
    check, check_rows = _collect(manifest["representation_check_sample"],
                                 manifest["data_dir"], str(scorer_root), spec, jobs, "check")
    check_read_seconds = time.perf_counter() - check_started
    _, _, _, scorer_contour, _ = _frozen_modules({"scorer_provenance": {
        "root": str(scorer_root),
        **{key: value for key, value in spec["scorer_provenance"].items()
           if key.endswith("_py_sha256")},
    }})
    X_check = check.reshape(300, 1024)
    codes = pca.transform(X_check)
    identities: list[dict[str, Any]] = []
    for row in check_rows:
        for frame in row["indices"]:
            identities.append({"file": row["name"], "frame": frame,
                               "spacing_m": row["spacing_m"],
                               "grid_R_minmax": row["grid_R_minmax"],
                               "grid_Z_minmax": row["grid_Z_minmax"]})
    if len(identities) != 300:
        raise ValueError(f"{output}: {len(identities)} check frame identities, expected 300")
    summaries: dict[str, Any] = {}
    rows: list[dict[str, Any]] = [{**identity} for identity in identities]
    passing: list[int] = []
    for components in (32, 64):
        reconstructed = (pca.mean_[None, :] + codes[:, :components]
                         @ pca.components_[:components]).reshape(300, 512, 2)
        delta_sq = np.sum((reconstructed - check) ** 2)
        centered_sq = np.sum((check.reshape(300, 1024) - pca.mean_[None, :]) ** 2)
        if centered_sq <= 0 or not np.isfinite(delta_sq):
            raise ValueError(f"PCA{components}: held contour L2 denominator/residual invalid")
        distance: list[float] = []
        invalid_simple = 0
        outside = 0
        for i, (truth, candidate) in enumerate(zip(check, reconstructed, strict=True)):
            h = scorer_contour.symmetric_hausdorff(truth, candidate)
            simple = bool(_simple_closed(candidate))
            Rmin, Rmax = identities[i]["grid_R_minmax"]
            Zmin, Zmax = identities[i]["grid_Z_minmax"]
            in_grid = bool((candidate[:, 0] >= Rmin).all() and (candidate[:, 0] <= Rmax).all()
                           and (candidate[:, 1] >= Zmin).all() and (candidate[:, 1] <= Zmax).all())
            invalid_simple += not simple
            outside += not in_grid
            distance.append(h)
            rows[i][f"PCA{components}"] = {
                "symmetric_hausdorff_m": h,
                "simple_polygon": simple,
                "within_physical_grid": in_grid,
            }
        values = np.asarray(distance)
        spacing = min(row["spacing_m"] for row in identities)
        median = float(np.median(values))
        p95 = float(np.quantile(values, 0.95))
        passed = invalid_simple == 0 and outside == 0 and median <= 0.25 * spacing \
            and p95 <= 0.5 * spacing
        if passed:
            passing.append(components)
        summaries[f"PCA{components}"] = {
            "median_hausdorff_m": median,
            "p95_hausdorff_m": p95,
            "max_hausdorff_m": float(np.max(values)),
            "median_hausdorff_in_min_grid_cells": median / spacing,
            "p95_hausdorff_in_min_grid_cells": p95 / spacing,
            "held_vector_L2_reconstruction_ratio": float(1.0 - delta_sq / centered_sq),
            "minimum_grid_spacing_m": spacing,
            "non_simple_polygon_count": invalid_simple,
            "out_of_grid_count": outside,
            "gate_passed": bool(passed),
        }
    chosen = passing[0] if passing else None
    write_json(output / "representation.json", {
        "classification": "physical_PCA_representation_feasible" if chosen is not None
                          else "physical_PCA_representation_gate_failed",
        "spec_sha256": manifest["spec_sha256"],
        "manifest_sha256": file_hash(output / "manifest.json"),
        "basis_sample_ordered_names_sha256": manifest["basis_sample_ordered_names_sha256"],
        "representation_check_ordered_names_sha256":
            manifest["representation_check_ordered_names_sha256"],
        "basis_shots": 100, "basis_probes": 300,
        "check_shots": 100, "check_probes": 300,
        "pca64_basis_sha256": file_hash(model_path),
        "explained_variance_ratio_64_sum": float(pca.explained_variance_ratio_.sum()),
        "candidates": summaries,
        "selected_smallest_passing_components": chosen,
        "per_probe_rows": rows,
        "basis_read_seconds": basis_read_seconds,
        "pca_fit_seconds": fit_seconds,
        "check_read_seconds": check_read_seconds,
        "total_wall_seconds": time.perf_counter() - started,
        "no_direct_predictor_fit_or_reserved_tail_read": True,
    })
    print(f"Train-only PCA representation: PCA32 {summaries['PCA32']}, "
          f"PCA64 {summaries['PCA64']}; selected components={chosen}")


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
                "classification": "PCA_numeric_or_input_failure",
                "manifest_sha256": file_hash(output / "manifest.json"),
                "exception_type": type(exc).__name__,
                "error": str(exc),
                "not_a_scientific_negative": True,
            })
            raise


if __name__ == "__main__":
    main()

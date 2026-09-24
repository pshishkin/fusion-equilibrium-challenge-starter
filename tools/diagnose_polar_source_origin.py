"""Train-only source-centroid polar LCFS and log-radius PCA64 feasibility gate."""

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

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

from experiments import _as_psirz_stack
from my_experiments import baseline_model
from my_experiments.data_check import check_dataset
from my_experiments.oof_split import names_hash
from my_experiments.polar_source_origin import (
    origin_edge_distance,
    polar_contour,
    polygon_centroid,
    ray_outer_radii,
)
from my_experiments.run import digest, source_files, verify_source
from toolkit.parallel import pimap, release
from toolkit.records import file_hash, write_json
from tools.direct_contour_compact_val_predict import INPUT_COLUMNS
from tools.geometry_map_smoke import _simple_closed

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "experiments/polar-source-origin-review.json"
SAMPLES = ROOT / "experiments/polar-source-origin-samples.json"
DATA = ROOT.parent / "downloaded_huggingface/hf_dataset"
sys.path.insert(0, str(ROOT / "fusion_scoring"))
from fusion_scoring import common, derive, lcfs  # noqa: E402
from fusion_scoring import contour as scorer_contour  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(path.read_text())
    return value


def _reference_checks(origin: Path, spec: dict[str, Any], samples: dict[str, Any]) -> None:
    refs = spec["frozen_inputs"]
    if file_hash(origin / refs["selection_manifest"]) != refs["selection_manifest_sha256"]:
        raise ValueError(f"{origin / refs['selection_manifest']}: selected shot manifest changed")
    for relative, key in (("runs/oof-normal-k8-5fold-20260923/files.json",
                            "oof_files_manifest_sha256"),
                          (refs["fold0_base_artifact"], "fold0_base_artifact_sha256"),
                          ("my_experiments/diii_d_dataset.json", "dataset_manifest_sha256")):
        if file_hash(origin / relative) != refs[key]:
            raise ValueError(f"{origin / relative}: pinned OOF/base/dataset SHA differs")
    scorer = origin / refs["scorer_source"]
    if verify_source(scorer.parent)["source_hash"] != refs["scorer_bundle_sha256"]:
        raise ValueError(f"{scorer}: scorer source bundle SHA differs")
    for relative, key in (("fusion_scoring/lcfs.py", "scorer_lcfs_py_sha256"),
                          ("fusion_scoring/derive.py", "scorer_derive_py_sha256"),
                          ("fusion_scoring/contour.py", "scorer_contour_py_sha256"),
                          ("fusion_scoring/masks/d3d_envelope.npz", "scorer_mask_sha256")):
        if (file_hash(scorer / relative) != refs[key]
                or file_hash(ROOT / relative) != refs[key]):
            raise ValueError(f"{relative}: scorer and polar source implementations differ")
    if (samples["first100"]["ordered_name_sha256"]
            != refs["first100_ordered_names_sha256"]
            or samples["third100"]["ordered_name_sha256"]
            != refs["third100_ordered_names_sha256"]):
        raise ValueError(f"{SAMPLES}: first/third100 ordered sample hashes differ")


def prepare(output: Path) -> None:
    """Freeze code, scorer and exact TRAIN-only probes before any true-psi read."""
    started = time.perf_counter()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"{output}: polar source-origin campaign already exists")
    spec = _read_json(SPEC)
    samples = _read_json(SAMPLES)
    _reference_checks(ROOT, spec, samples)
    oof = _read_json(ROOT / "runs/oof-normal-k8-5fold-20260923/files.json")
    reference: dict[str, Any] = joblib.load(ROOT / (
        "runs/compact-production-9301af36275620f7/base_s0/baseline.joblib"))
    base: dict[str, Any] = joblib.load(ROOT / spec["frozen_inputs"]["fold0_base_artifact"])
    train_set = set(reference["train_files"])
    oof_set = set(oof["folds"][0]["oof_files"])
    sealed = set(oof["gate_files"]) | set(oof["reserved_tail_files"])
    first = samples["first100"]["records"]
    third = samples["third100"]["records"]
    names = [item["file"] for item in first + third]
    if (len(first) != 100 or len(third) != 100 or len(set(names)) != 200
            or set(names) - (train_set & oof_set) or set(names) & sealed
            or set(names) & (set(base["train_files"]) | set(base["val_files"]))
            or base["oof_files"] != oof["folds"][0]["oof_files"]
            or base["val_files"] != []
            or names_hash([item["file"] for item in first])
            != spec["frozen_inputs"]["first100_ordered_names_sha256"]
            or names_hash([item["file"] for item in third])
            != spec["frozen_inputs"]["third100_ordered_names_sha256"]):
        raise ValueError(f"{SAMPLES}: TRAIN/OOF/gate/tail/base-role proof differs")
    dataset_identity = check_dataset(DATA, "diii_d_train")
    files = set(source_files()) | {SPEC, SAMPLES}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".polar-freeze-") as temporary:
        stage = Path(temporary) / "campaign"
        source = stage / "source"
        source.mkdir(parents=True)
        for path in sorted(files):
            relative = path.relative_to(ROOT)
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if file_hash(target) != file_hash(path):
                raise ValueError(f"{path}: changed while freezing polar source")
        hashes = {str(path.relative_to(ROOT)): file_hash(source / path.relative_to(ROOT))
                  for path in sorted(files)}
        write_json(stage / "manifest.json", {
            "classification": "source_centroid_polar_train_only_frozen_before_psi_read",
            "origin_root": str(ROOT),
            "data_dir": str(DATA),
            "spec_sha256": file_hash(SPEC),
            "samples_sha256": file_hash(SAMPLES),
            "source_sha256": hashes,
            "source_bundle_sha256": digest(hashes),
            "dataset_identity": dataset_identity,
            "fold0_base_artifact_sha256": file_hash(
                ROOT / spec["frozen_inputs"]["fold0_base_artifact"]),
            "first100": first,
            "third100": third,
            "gate200_shots_used": 0,
            "reserved_tail70_shots_used": 0,
            "prepare_wall_seconds": time.perf_counter() - started,
        })
        subprocess.run([
            sys.executable, "-I", "-c",
            "import pathlib,sys; root=pathlib.Path(sys.argv[1]).resolve(); "
            "sys.path.insert(0,str(root)); "
            "import my_experiments.polar_source_origin,tools.diagnose_polar_source_origin; "
            "assert pathlib.Path(tools.diagnose_polar_source_origin.__file__).resolve()"
            ".is_relative_to(root)",
            str(source),
        ], cwd="/tmp", env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            check=True)
        stage.rename(output)
    print(f"Frozen {output}: first100/300 and disjoint third100/300 OOF TRAIN probes")


def _context(output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    output = output.resolve()
    if ROOT.resolve() != (output / "source").resolve():
        raise ValueError(f"{output}: polar tool imported from {ROOT}, expected frozen source")
    meta = _read_json(output / "manifest.json")
    spec = _read_json(output / "source/experiments/polar-source-origin-review.json")
    samples = _read_json(output / "source/experiments/polar-source-origin-samples.json")
    if (file_hash(output / "source/experiments/polar-source-origin-review.json")
            != meta["spec_sha256"]
            or file_hash(output / "source/experiments/polar-source-origin-samples.json")
            != meta["samples_sha256"]):
        raise ValueError(f"{output}: frozen spec/sample identity differs")
    for relative, expected in meta["source_sha256"].items():
        if file_hash(output / "source" / relative) != expected:
            raise ValueError(f"{output / 'source' / relative}: frozen source SHA differs")
    _reference_checks(Path(meta["origin_root"]), spec, samples)
    if check_dataset(Path(meta["data_dir"]), "diii_d_train") != meta["dataset_identity"]:
        raise ValueError(f"{meta['data_dir']}: complete dataset identity differs")
    return meta, spec


def _extract(psi: np.ndarray, r: np.ndarray, z: np.ndarray,
             mask: np.ndarray) -> np.ndarray | None:
    contour, _ = lcfs.extract_lcfs_with_sign(
        psi.astype(np.float64), r, z, "DIII-D", mask, mask.astype(float),
        n_points=512, axis_sign=int(common.AXIS_SIGN["DIII-D"]))
    return None if contour is None else np.asarray(contour, dtype=np.float64)


def _probe_one(task: tuple[dict[str, Any], str, str, str]) -> dict[str, Any]:
    item, data_s, artifact_s, mask_s = task
    name = item["file"]
    path = Path(data_s) / "data/diii_d_train" / name
    if (path.stat().st_size != item["parquet_size_bytes"]
            or file_hash(path) != item["parquet_sha256"]):
        raise ValueError(f"{path}: selected TRAIN parquet bytes/SHA differ")
    row = pd.read_parquet(path, columns=INPUT_COLUMNS).iloc[0]
    forbidden = [key for key in row.index if key.startswith("efit_")
                 and key not in ("efit_times", "efit_grid_R", "efit_grid_Z")]
    if forbidden:
        raise ValueError(f"{path}: input-only source row contains EFIT targets {forbidden}")
    times = np.asarray(row["efit_times"], dtype=np.float64)
    indices = item["probe_indices"]
    expected = [int(np.floor(q * (len(times) - 1) + 0.5)) for q in (0.1, 0.5, 0.9)]
    if (len(times) != item["full_frames"] or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or indices != expected
            or len(set(indices)) != 3):
        raise ValueError(f"{path}: efit_times {len(times)} and fixed indices {indices}, "
                         f"expected {item['full_frames']} and {expected} monotone finite")
    with np.load(mask_s) as packed:
        grid_r = np.asarray(packed["grid_R"], dtype=np.float64)
        grid_z = np.asarray(packed["grid_Z"], dtype=np.float64)
        mask = np.asarray(packed["mask_coarse"], dtype=np.bool_)
    if (not np.array_equal(row["efit_grid_R"], grid_r)
            or not np.array_equal(row["efit_grid_Z"], grid_z)):
        raise ValueError(f"{path}: input grid differs from pinned scorer mask")
    source_prediction = baseline_model.predict_row(
        baseline_model.slim_row(row), "DIII-D", "mlp", Path(artifact_s))
    source_stack = np.asarray(source_prediction["psirz"], dtype=np.float32)
    if source_stack.shape != (len(times), 65, 65) or not np.isfinite(source_stack).all():
        raise ValueError(f"{path}: input-only source psi shape/finiteness {source_stack.shape}, "
                         f"expected ({len(times)},65,65) finite")
    # EFIT truth is opened only after the source was predicted from target-stripped inputs.
    true_row = pd.read_parquet(path, columns=["efit_psirz"]).iloc[0]
    true_stack = _as_psirz_stack(true_row["efit_psirz"])
    if true_stack.shape != source_stack.shape or not np.isfinite(true_stack).all():
        raise ValueError(f"{path}: true psi shape/finiteness {true_stack.shape}, "
                         f"expected {source_stack.shape} finite")
    spacing = float(min(np.min(np.diff(grid_r)), np.min(np.diff(grid_z))))
    source_contours = np.full((3, 512, 2), np.nan)
    true_contours = np.full((3, 512, 2), np.nan)
    envelopes = np.full((3, 512, 2), np.nan)
    origins = np.full((3, 2), np.nan)
    log_radii = np.full((3, 512), np.nan)
    frames: list[dict[str, Any]] = []
    for probe, frame in enumerate(indices):
        source = _extract(source_stack[frame], grid_r, grid_z, mask)
        true = _extract(true_stack[frame], grid_r, grid_z, mask)
        detail: dict[str, Any] = {"file": name, "frame": frame,
                                  "probe_index": probe, "spacing_m": spacing,
                                  "source_only_fallback_reason": None,
                                  "hard_representation_reason": None,
                                  "multi_hit_rays": 0, "zero_hit_rays": 0,
                                  "ambiguous_rays": 0,
                                  "source_to_true_centroid_m": None,
                                  "source_hausdorff_m": None,
                                  "outer_envelope_hausdorff_m": None}
        if true is None or not _simple_closed(true):
            detail["hard_representation_reason"] = "true_lcfs_missing_or_invalid"
            frames.append(detail)
            continue
        true_contours[probe] = true
        if source is None or not _simple_closed(source):
            detail["hard_representation_reason"] = "source_lcfs_missing_or_invalid"
            frames.append(detail)
            continue
        source_contours[probe] = source
        detail["source_hausdorff_m"] = scorer_contour.symmetric_hausdorff(true, source)
        try:
            origin = polygon_centroid(source, f"{path} frame {frame} source")
        except ValueError:
            detail["source_only_fallback_reason"] = "source_centroid_invalid_geometry"
            frames.append(detail)
            continue
        try:
            true_center = polygon_centroid(true, f"{path} frame {frame} true")
        except ValueError as exc:
            detail["hard_representation_reason"] = str(exc)
            frames.append(detail)
            continue
        origins[probe] = origin
        detail["source_to_true_centroid_m"] = float(np.linalg.norm(origin - true_center))
        source_inside = bool(derive._points_in_polygon(origin[0:1], origin[1:2], source)[0])
        source_edge = origin_edge_distance(source, origin, f"{path} frame {frame} source")
        if not source_inside or source_edge <= 1e-4 * spacing:
            detail["source_only_fallback_reason"] = "source_centroid_invalid_on_source"
            frames.append(detail)
            continue
        true_inside = bool(derive._points_in_polygon(origin[0:1], origin[1:2], true)[0])
        radii, ray = ray_outer_radii(
            true, origin, spacing, true_inside, f"{path} frame {frame}")
        detail.update({"multi_hit_rays": ray["multi_hit_rays"],
                       "zero_hit_rays": ray["zero_hit_rays"],
                       "ambiguous_rays": ray["ambiguous_rays"],
                       "first_pathology": ray["first_pathology"],
                       "max_inner_to_outer_hit_gap_m":
                       ray.get("max_inner_to_outer_hit_gap_m")})
        if radii is None:
            detail["hard_representation_reason"] = ray["hard_reason"]
            frames.append(detail)
            continue
        envelope = polar_contour(origin, radii, f"{path} frame {frame} outer-envelope")
        if (not _simple_closed(envelope)
                or np.any(envelope[:, 0] < grid_r[0])
                or np.any(envelope[:, 0] > grid_r[-1])
                or np.any(envelope[:, 1] < grid_z[0])
                or np.any(envelope[:, 1] > grid_z[-1])):
            detail["hard_representation_reason"] = "outer_envelope_non_simple_or_out_of_grid"
            frames.append(detail)
            continue
        envelopes[probe] = envelope
        log_radii[probe] = np.log(radii / 1.0)
        detail["outer_envelope_hausdorff_m"] = scorer_contour.symmetric_hausdorff(
            true, envelope)
        frames.append(detail)
    return {"name": name, "full_frames": len(times), "indices": indices,
            "source_contours": source_contours, "true_contours": true_contours,
            "outer_envelopes": envelopes, "source_origins": origins,
            "log_radii": log_radii, "frames": frames}


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"{path}: polar contour probe already exists")
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _collect(output: Path, sample: str, jobs: int,
             meta: dict[str, Any], spec: dict[str, Any]
             ) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], float]:
    """Extract exact physical source/truth contours in frozen shot/frame order."""
    started = time.perf_counter()
    if sample not in ("first100", "third100") or not 1 <= jobs <= 24:
        raise ValueError(f"sample/jobs {sample}/{jobs}, expected first/third100 and 1..24")
    selected = meta[sample]
    origin = Path(meta["origin_root"])
    artifact = origin / spec["frozen_inputs"]["fold0_base_artifact"]
    mask = output / "source/fusion_scoring/masks/d3d_envelope.npz"
    tasks = [(item, meta["data_dir"], str(artifact), str(mask)) for item in selected]
    try:
        responses = list(pimap(_probe_one, tasks, jobs))
    finally:
        release()
    if (len(responses) != 100
            or [row["name"] for row in responses] != [item["file"] for item in selected]
            or any(row["full_frames"] != item["full_frames"]
                   or row["indices"] != item["probe_indices"]
                   for row, item in zip(responses, selected, strict=True))):
        raise ValueError(f"{output}: {sample} source/true shot/frame order differs")
    arrays: dict[str, np.ndarray] = {}
    for key in ("source_contours", "true_contours", "outer_envelopes",
                "source_origins", "log_radii"):
        arrays[key] = np.concatenate([row[key] for row in responses])
    if (arrays["source_contours"].shape != (300, 512, 2)
            or arrays["true_contours"].shape != (300, 512, 2)
            or arrays["source_origins"].shape != (300, 2)
            or arrays["log_radii"].shape != (300, 512)):
        raise ValueError(f"{output}: {sample} contour/origin/radius arrays differ from 300 probes")
    frames = [frame for response in responses for frame in response["frames"]]
    if len(frames) != 300:
        raise ValueError(f"{output}: {sample} frame records {len(frames)}, expected 300")
    _atomic_npz(output / f"{sample}_probes.npz", arrays)
    write_json(output / f"{sample}_frames.json", frames)
    return arrays, frames, time.perf_counter() - started


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("polar summary values must all be finite")
    return {"median": float(np.median(arr)), "p95": float(np.quantile(arr, .95)),
            "max": float(np.max(arr))}


def _frame_counts(frames: list[dict[str, Any]]) -> dict[str, Any]:
    hard: dict[str, int] = {}
    source_only: dict[str, int] = {}
    multi_frames = 0
    multi_rays = 0
    for frame in frames:
        if frame["hard_representation_reason"] is not None:
            key = str(frame["hard_representation_reason"])
            hard[key] = hard.get(key, 0) + 1
        if frame["source_only_fallback_reason"] is not None:
            key = str(frame["source_only_fallback_reason"])
            source_only[key] = source_only.get(key, 0) + 1
        multi_frames += frame["multi_hit_rays"] > 0
        multi_rays += frame["multi_hit_rays"]
    return {"hard_reasons": hard, "hard_frames": sum(hard.values()),
            "source_only_fallback_reasons": source_only,
            "source_only_fallback_frames": sum(source_only.values()),
            "multiple_hit_true_frames": multi_frames,
            "multiple_hit_true_rays": multi_rays}


def first_gate(output: Path, jobs: int) -> None:
    """Stop before PCA/third truth if source-origin topology is not viable."""
    started = time.perf_counter()
    output = output.resolve()
    meta, spec = _context(output)
    result_path = output / "first_result.json"
    if result_path.exists() or (output / "third100_probes.npz").exists():
        raise FileExistsError(f"{output}: first/third polar result already exists")
    arrays, frames, extract_seconds = _collect(output, "first100", jobs, meta, spec)
    counts = _frame_counts(frames)
    source_h = [frame["source_hausdorff_m"] for frame in frames
                if frame["source_hausdorff_m"] is not None]
    envelope_h = [frame["outer_envelope_hausdorff_m"] for frame in frames
                  if frame["outer_envelope_hausdorff_m"] is not None]
    multi_h = [frame["outer_envelope_hausdorff_m"] for frame in frames
               if frame["multi_hit_rays"] > 0
               and frame["outer_envelope_hausdorff_m"] is not None]
    displacement = [frame["source_to_true_centroid_m"] for frame in frames
                    if frame["source_to_true_centroid_m"] is not None]
    passed = counts["hard_frames"] == 0 and counts["source_only_fallback_frames"] <= 6
    basis_sha: str | None = None
    fit_seconds = 0.0
    fit_rows = 0
    if passed:
        valid = [i for i, frame in enumerate(frames)
                 if frame["hard_representation_reason"] is None
                 and frame["source_only_fallback_reason"] is None]
        fit_rows = len(valid)
        if fit_rows < 294 or not np.isfinite(arrays["log_radii"][valid]).all():
            raise ValueError(f"{output}: viable first-sample log-radius rows {fit_rows}, "
                             "expected >=294 finite")
        fit_started = time.perf_counter()
        pca = PCA(n_components=64, svd_solver="full", whiten=False, random_state=0)
        pca.fit(arrays["log_radii"][valid])
        fit_seconds = time.perf_counter() - fit_started
        path = output / "log_radius_pca64.joblib"
        fd, temporary = tempfile.mkstemp(dir=output, suffix=".joblib")
        os.close(fd)
        try:
            joblib.dump(pca, temporary)
            os.link(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        basis_sha = file_hash(path)
    write_json(result_path, {
        "classification": "source_origin_first100_topology_gate",
        "spec_sha256": meta["spec_sha256"],
        "sample_manifest_sha256": meta["samples_sha256"],
        "source_bundle_sha256": meta["source_bundle_sha256"],
        "shots": 100, "frames": 300,
        "frame_counts": counts,
        "source_contour_hausdorff_m": _summary(source_h),
        "outer_star_envelope_hausdorff_m": _summary(envelope_h),
        "outer_star_envelope_multiple_hit_frames_hausdorff_m": _summary(multi_h),
        "source_to_true_centroid_distance_m": _summary(displacement),
        "first_pathologies": [frame for frame in frames
                              if frame["hard_representation_reason"] is not None
                              or frame["source_only_fallback_reason"] is not None
                              or frame["multi_hit_rays"] > 0][:20],
        "first_probes_sha256": file_hash(output / "first100_probes.npz"),
        "first_frame_records_sha256": file_hash(output / "first100_frames.json"),
        "PCA64_fit_rows": fit_rows,
        "PCA64_basis_sha256": basis_sha,
        "passed": passed,
        "timing_seconds": {"extract_and_ray": extract_seconds,
                           "PCA64_fit": fit_seconds,
                           "first_stage_wall": time.perf_counter() - started},
        "no_gate200_or_reserved_tail_read": True,
    })
    print(f"First100 source-origin chart: hard {counts['hard_frames']}, "
          f"source-only fallback {counts['source_only_fallback_frames']}, "
          f"multiple-hit frames {counts['multiple_hit_true_frames']}; pass {passed}", flush=True)


def _local_errors(truth: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    """Describe worst physical points and explicit X-point-like region proxies."""
    true_to_candidate, _ = cKDTree(candidate).query(truth)
    candidate_to_true, _ = cKDTree(truth).query(candidate)
    t_index = int(np.argmax(true_to_candidate))
    c_index = int(np.argmax(candidate_to_true))
    if true_to_candidate[t_index] >= candidate_to_true[c_index]:
        worst = {"direction": "true_to_reconstruction", "vertex_index": t_index,
                 "physical_RZ_m": truth[t_index].tolist(),
                 "distance_m": float(true_to_candidate[t_index])}
    else:
        worst = {"direction": "reconstruction_to_true", "vertex_index": c_index,
                 "physical_RZ_m": candidate[c_index].tolist(),
                 "distance_m": float(candidate_to_true[c_index])}
    low = truth[:, 1] <= np.quantile(truth[:, 1], .10)
    high = truth[:, 1] >= np.quantile(truth[:, 1], .90)
    incoming = truth - np.roll(truth, 1, axis=0)
    outgoing = np.roll(truth, -1, axis=0) - truth
    denominator = np.linalg.norm(incoming, axis=1) * np.linalg.norm(outgoing, axis=1)
    if np.any(denominator <= 0):
        raise ValueError("true contour has a zero adjacent segment in curvature proxy")
    bend = np.arccos(np.clip(np.sum(incoming * outgoing, axis=1) / denominator, -1, 1))
    high_curvature = np.zeros(512, dtype=np.bool_)
    high_curvature[np.argsort(bend)[-26:]] = True
    groups = {"lower_vertical_decile": low, "upper_vertical_decile": high,
              "highest_curvature_26_vertices": high_curvature}
    return {"worst_directed_point": worst,
            "true_vertex_region_errors_m": {
                name: {"p95": float(np.quantile(true_to_candidate[mask], .95)),
                       "max": float(np.max(true_to_candidate[mask]))}
                for name, mask in groups.items()}}


def third_gate(output: Path, jobs: int) -> None:
    """Measure one fixed PCA64 on a disjoint TRAIN sample, including every fallback."""
    started = time.perf_counter()
    output = output.resolve()
    meta, spec = _context(output)
    first = _read_json(output / "first_result.json")
    if not first["passed"] or first["PCA64_fit_rows"] < 294:
        raise ValueError(f"{output}: first100 topology gate failed; third100 truth is sealed")
    result_path = output / "third_result.json"
    if result_path.exists():
        raise FileExistsError(f"{result_path}: held TRAIN representation already measured")
    basis = output / "log_radius_pca64.joblib"
    if file_hash(basis) != first["PCA64_basis_sha256"]:
        raise ValueError(f"{basis}: frozen first-sample PCA64 SHA differs")
    pca: PCA = joblib.load(basis)
    if pca.components_.shape != (64, 512):
        raise ValueError(f"{basis}: PCA component shape {pca.components_.shape}, expected 64x512")
    arrays, frames, extraction_seconds = _collect(output, "third100", jobs, meta, spec)
    counts = _frame_counts(frames)
    spacing = min(frame["spacing_m"] for frame in frames)
    candidate_rows: list[dict[str, Any]] = []
    distances: list[float] = []
    source_distances: list[float] = []
    multiple_hit_distances: list[float] = []
    displacement: list[float] = []
    fallback_reasons: dict[str, int] = {}
    accepted_simple = 0
    accepted_in_grid = 0
    eval_started = time.perf_counter()
    with np.load(output / "source/fusion_scoring/masks/d3d_envelope.npz") as packed:
        grid_r = np.asarray(packed["grid_R"], dtype=np.float64)
        grid_z = np.asarray(packed["grid_Z"], dtype=np.float64)
    for i, frame in enumerate(frames):
        row = dict(frame)
        source = arrays["source_contours"][i]
        truth = arrays["true_contours"][i]
        if frame["source_to_true_centroid_m"] is not None:
            displacement.append(float(frame["source_to_true_centroid_m"]))
        if frame["source_hausdorff_m"] is not None:
            source_distances.append(float(frame["source_hausdorff_m"]))
        if frame["hard_representation_reason"] is not None:
            row["reconstruction_hausdorff_m"] = None
            row["reconstruction"] = "hard_unrepresentable_true_frame"
            candidate_rows.append(row)
            continue
        if frame["source_only_fallback_reason"] is not None:
            candidate = source
            reason = str(frame["source_only_fallback_reason"])
        else:
            log_radius = arrays["log_radii"][i]
            if not np.isfinite(log_radius).all():
                raise ValueError(f"{frame['file']} frame {frame['frame']}: valid log radius "
                                 "row is nonfinite")
            projected = pca.inverse_transform(pca.transform(log_radius[None]))[0]
            radius = np.exp(projected)
            if not np.isfinite(radius).all() or np.any(radius <= 0):
                candidate = source
                reason = "reconstructed_radius_nonfinite_or_nonpositive"
            else:
                proposed = polar_contour(arrays["source_origins"][i], radius,
                                          f"{frame['file']} frame {frame['frame']}")
                in_grid = bool((proposed[:, 0] >= grid_r[0]).all()
                               and (proposed[:, 0] <= grid_r[-1]).all()
                               and (proposed[:, 1] >= grid_z[0]).all()
                               and (proposed[:, 1] <= grid_z[-1]).all())
                simple = bool(_simple_closed(proposed))
                if simple and in_grid:
                    candidate = proposed
                    reason = None
                    accepted_simple += 1
                    accepted_in_grid += 1
                else:
                    candidate = source
                    reason = "reconstructed_polygon_non_simple_or_out_of_grid"
        if reason is not None:
            fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
        if not _simple_closed(candidate):
            raise ValueError(f"{frame['file']} frame {frame['frame']}: selected source/polar "
                             "candidate is not a simple physical polygon")
        hausdorff = scorer_contour.symmetric_hausdorff(truth, candidate)
        distances.append(hausdorff)
        if frame["multi_hit_rays"] > 0:
            multiple_hit_distances.append(hausdorff)
        row["reconstruction"] = "source_fallback" if reason else "polar_PCA64"
        row["fallback_reason"] = reason
        row["reconstruction_hausdorff_m"] = hausdorff
        row["local_error"] = _local_errors(truth, candidate)
        candidate_rows.append(row)
    hard_count = counts["hard_frames"]
    fallback_count = sum(fallback_reasons.values())
    if hard_count == 0 and len(distances) != 300:
        raise ValueError(f"{output}: third100 all-frame distances {len(distances)}, expected 300")
    all_frame = _summary(distances) if hard_count == 0 else {
        "median": None, "p95": None, "max": None}
    passed = bool(hard_count == 0 and fallback_count <= 6
                  and all_frame["median"] is not None
                  and all_frame["p95"] is not None
                  and all_frame["median"] <= .25 * spacing
                  and all_frame["p95"] <= .5 * spacing
                  and accepted_simple == accepted_in_grid
                  and accepted_simple + fallback_count == 300)
    local_regions = ("lower_vertical_decile", "upper_vertical_decile",
                     "highest_curvature_26_vertices")
    local_summary = {
        name: {
            "per_frame_p95_m": _summary([
                row["local_error"]["true_vertex_region_errors_m"][name]["p95"]
                for row in candidate_rows if "local_error" in row]),
            "per_frame_max_m": _summary([
                row["local_error"]["true_vertex_region_errors_m"][name]["max"]
                for row in candidate_rows if "local_error" in row]),
        } for name in local_regions
    }
    write_json(output / "third100_reconstruction_frames.json", candidate_rows)
    write_json(result_path, {
        "classification": "source_origin_polar_PCA64_third100_TRAIN_representation_gate",
        "spec_sha256": meta["spec_sha256"],
        "sample_manifest_sha256": meta["samples_sha256"],
        "source_bundle_sha256": meta["source_bundle_sha256"],
        "first_result_sha256": file_hash(output / "first_result.json"),
        "PCA64_basis_sha256": file_hash(basis),
        "shots": 100, "frames": 300,
        "minimum_grid_spacing_m": spacing,
        "hard_frame_counts": counts,
        "source_only_and_output_fallback_reasons": fallback_reasons,
        "fallback_frames": fallback_count,
        "fallback_fraction": fallback_count / 300,
        "accepted_simple_polar_polygons": accepted_simple,
        "accepted_in_grid_polar_polygons": accepted_in_grid,
        "all_frame_reconstruction_hausdorff_m": all_frame,
        "all_frame_reconstruction_hausdorff_in_grid_cells": {
            key: None if value is None else value / spacing
            for key, value in all_frame.items()},
        "source_contour_hausdorff_m_descriptive": _summary(source_distances),
        "multiple_hit_true_frame_reconstruction_hausdorff_m":
            _summary(multiple_hit_distances),
        "source_to_true_centroid_distance_m": _summary(displacement),
        "local_X_point_like_proxy_true_vertex_errors": local_summary,
        "worst_directed_points": sorted(
            [row["local_error"]["worst_directed_point"] | {
                "file": row["file"], "frame": row["frame"]}
             for row in candidate_rows if "local_error" in row],
            key=lambda value: value["distance_m"], reverse=True)[:10],
        "third_probes_sha256": file_hash(output / "third100_probes.npz"),
        "third_frame_records_sha256": file_hash(output / "third100_frames.json"),
        "third_reconstruction_records_sha256": file_hash(
            output / "third100_reconstruction_frames.json"),
        "passed": passed,
        "timing_seconds": {"extract_and_ray": extraction_seconds,
                           "PCA_and_local_geometry": time.perf_counter() - eval_started,
                           "third_stage_wall": time.perf_counter() - started},
        "no_gate200_or_reserved_tail_read": True,
        "no_learner_fit_or_native_S_score": True,
    })
    print(f"Third100 polar PCA64: hard {hard_count}, fallback {fallback_count}, "
          f"median/p95 cells "
          f"{None if all_frame['median'] is None else all_frame['median'] / spacing}/"
          f"{None if all_frame['p95'] is None else all_frame['p95'] / spacing}; "
          f"pass {passed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", type=Path, required=True)
    for name in ("first", "third"):
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output)
    elif args.command == "first":
        first_gate(args.output, args.jobs)
    else:
        third_gate(args.output, args.jobs)


if __name__ == "__main__":
    main()

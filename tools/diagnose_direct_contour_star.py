"""Frozen first-TRAIN-sample centroid-ray uniqueness check for true LCFS polygons."""

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

from my_experiments.data_check import check_dataset
from my_experiments.run import source_files
from toolkit.parallel import pimap, release
from toolkit.records import file_hash, write_json
from tools.diagnose_direct_contour_pca import _contours_one
from tools.diagnose_direct_contour_representation import _name_hash
from tools.geometry_map_smoke import _frozen_modules

ROOT = Path(__file__).resolve().parent.parent
SPEC = Path("experiments/direct-contour-star-viability-v2.json")
ORIGINAL_SPEC = Path("experiments/direct-contour-star-viability-review.json")


def _polygon_centroid(contour: np.ndarray) -> np.ndarray:
    c = np.asarray(contour, dtype=np.float64)
    if c.shape != (512, 2) or not np.isfinite(c).all():
        raise ValueError(f"true contour shape/finiteness {c.shape}, expected finite (512,2)")
    nxt = np.roll(c, -1, axis=0)
    cross = c[:, 0] * nxt[:, 1] - nxt[:, 0] * c[:, 1]
    area2 = float(cross.sum())
    if not np.isfinite(area2) or area2 <= 0:
        raise ValueError(f"CCW true contour signed double-area {area2}, expected positive")
    centroid = np.array([
        np.sum((c[:, 0] + nxt[:, 0]) * cross),
        np.sum((c[:, 1] + nxt[:, 1]) * cross),
    ]) / (3.0 * area2)
    if not np.isfinite(centroid).all():
        raise ValueError("true contour centroid is nonfinite")
    return centroid


def _ray_uniqueness(contour: np.ndarray, center: np.ndarray, spacing: float,
                    origin_inside: bool) -> dict[str, Any]:
    """Count distinct physical ray/polygon hits, preserving ambiguous cases."""
    if contour.shape != (512, 2) or center.shape != (2,) or spacing <= 0:
        raise ValueError(f"ray geometry {contour.shape}/{center.shape}/{spacing}, "
                         "expected (512,2)/(2,)/positive spacing")
    tau = 1e-4 * spacing
    start = contour
    end = np.roll(contour, -1, axis=0)
    seg = end - start
    lengths = np.linalg.norm(seg, axis=1)
    if np.any(lengths <= 0) or not np.isfinite(lengths).all():
        raise ValueError("true contour has zero/nonfinite polygon segment length")
    delta = center[None, :] - start
    along = np.clip(np.sum(delta * seg, axis=1) / (lengths**2), 0.0, 1.0)
    nearest = start + along[:, None] * seg
    origin_distance = float(np.min(np.linalg.norm(center[None, :] - nearest, axis=1)))
    theta = 2 * np.pi * (np.arange(512, dtype=np.float64) + 0.5) / 512
    directions = np.column_stack([np.cos(theta), np.sin(theta)])
    offset = start - center
    def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    den = cross(directions[:, None, :], seg[None, :, :])
    parallel = np.abs(den) <= 1e-12 * lengths[None, :]
    collinear = parallel & (np.abs(cross(offset[None, :, :],
                                            directions[:, None, :])) <= tau)
    safe_den = np.where(parallel, 1.0, den)
    radius = cross(offset, seg)[None, :] / safe_den
    seg_parameter = cross(offset[None, :, :], directions[:, None, :]) / safe_den
    allowed_segment = ((seg_parameter >= -tau / lengths[None, :])
                       & (seg_parameter <= 1.0 + tau / lengths[None, :]))
    hits = (~parallel) & allowed_segment & (radius > tau)
    counts: list[int] = []
    duplicate_hits = 0
    min_positive = float("inf")
    first_pathology: dict[str, Any] | None = None
    for angle in range(512):
        values = np.sort(radius[angle, hits[angle]])
        unique: list[float] = []
        for value in values:
            if not unique or value - unique[-1] > tau:
                unique.append(float(value))
            else:
                duplicate_hits += 1
        count = len(unique)
        counts.append(count)
        if unique:
            min_positive = min(min_positive, unique[0])
        ambiguous = int(np.count_nonzero(collinear[angle]))
        if first_pathology is None and (count != 1 or ambiguous):
            first_pathology = {
                "angle_index": angle,
                "angle_radians": float(theta[angle]),
                "unique_positive_hits": count,
                "collinear_segments": ambiguous,
                "positive_hit_radii_m": unique[:5],
            }
    zero_count = counts.count(0)
    one_count = counts.count(1)
    multiple_count = 512 - zero_count - one_count
    ambiguous_angles = int(np.count_nonzero(np.any(collinear, axis=1)))
    return {
        "centroid_RZ_m": center.tolist(),
        "centroid_inside": bool(origin_inside),
        "centroid_edge_distance_m": origin_distance,
        "centroid_strictly_inside": bool(origin_inside and origin_distance > tau),
        "physical_tau_m": tau,
        "exactly_one_positive_intersection_angles": one_count,
        "zero_intersection_angles": zero_count,
        "multiple_intersection_angles": multiple_count,
        "collinear_ambiguous_angles": ambiguous_angles,
        "shared_vertex_duplicate_hits_deduplicated": duplicate_hits,
        "minimum_positive_radius_m": min_positive if np.isfinite(min_positive) else None,
        "first_pathology": first_pathology,
    }


def _verify_prereg(origin: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    amended = json.loads((origin / SPEC).read_text())
    original = origin / ORIGINAL_SPEC
    if amended["amends_spec_sha256"] != file_hash(original):
        raise ValueError(f"{original}: amended star spec SHA changed")
    if ("theta_j=2*pi*(j+0.5)/512" not in amended["geometry"]["rays"]
            or amended["geometry"]["viability_gate"].find("153,600") < 0):
        raise ValueError(f"{origin / SPEC}: fixed shifted-angle/gate formula changed")
    prior_path = origin / amended["sample"]["frozen_first100_train_manifest"]
    if file_hash(prior_path) != amended["sample"]["frozen_first100_train_manifest_sha256"]:
        raise ValueError(f"{prior_path}: prior train sample SHA changed")
    prior = json.loads(prior_path.read_text())
    selected = prior["selected"]
    names = [row["file"] for row in selected]
    if (len(selected) != 100 or _name_hash(names) != amended["sample"]["ordered_names_sha256"]
            or sum(len(row["probe_indices"]) for row in selected) != 300):
        raise ValueError(f"{prior_path}: 100 selected train shots/300 frames differ")
    artifact = origin / amended["sample"]["compact_artifact"]
    if file_hash(artifact) != amended["sample"]["compact_artifact_sha256"]:
        raise ValueError(f"{artifact}: compact artifact SHA changed")
    packed = joblib.load(artifact)
    if list(packed["train_files"][:100]) != names:
        raise ValueError(f"{artifact}: star sample is not the first compact TRAIN100")
    if (set(names) & set(packed["val_files"])
            or set(names) & set(json.loads((origin / "runs/compact-production-9301af36275620f7"
                                        "/base_s0/metrics.json").read_text())["files"])):
        raise ValueError("star sample overlaps compact validation or reserved evaluation")
    return amended, prior


def prepare(output: Path) -> None:
    """Freeze exact 300 TRAIN probes and all relevant source before reading truth."""
    started = time.perf_counter()
    origin = ROOT.resolve()
    spec, prior = _verify_prereg(origin)
    data = (origin / "../downloaded_huggingface/hf_dataset").resolve()
    corpus = check_dataset(data, "diii_d_train")
    if corpus["manifest_sha256"] != spec["sample"]["dataset_manifest_sha256"]:
        raise ValueError("DIII-D corpus manifest SHA changed")
    if output.exists():
        raise FileExistsError(f"{output}: frozen star diagnostic exists")
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
            raise ValueError(f"{path}: frozen source SHA differs")
    for relative in (ORIGINAL_SPEC, SPEC):
        dest = snapshot / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin / relative, dest)
        hashes[str(relative)] = file_hash(dest)
    manifest = {
        "status": "frozen_before_train_true_psi_read",
        "spec_sha256": file_hash(origin / SPEC),
        "original_spec_sha256": file_hash(origin / ORIGINAL_SPEC),
        "prior_manifest_sha256": file_hash(
            origin / spec["sample"]["frozen_first100_train_manifest"]
        ),
        "origin_root": str(origin),
        "data_dir": str(data),
        "source_hashes": hashes,
        "selected": prior["selected"],
        "selected_ordered_names_sha256": _name_hash(
            [row["file"] for row in prior["selected"]]
        ),
        "dataset_preflight": corpus,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "pid_namespace": os.readlink("/proc/self/ns/pid"),
        "prepare_wall_seconds": time.perf_counter() - started,
    }
    write_json(output / "manifest.json", manifest)
    print(f"Frozen {output}: 100 compact TRAIN shots, 300 fixed frames, "
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
    if (file_hash(origin / SPEC) != manifest["spec_sha256"]
            or file_hash(origin / ORIGINAL_SPEC) != manifest["original_spec_sha256"]):
        raise ValueError("original or amended star spec changed")
    spec = json.loads((ROOT / SPEC).read_text())
    prior_path = origin / spec["sample"]["frozen_first100_train_manifest"]
    if file_hash(prior_path) != manifest["prior_manifest_sha256"]:
        raise ValueError(f"{prior_path}: selected TRAIN sample changed")
    if (_name_hash([row["file"] for row in manifest["selected"]])
            != manifest["selected_ordered_names_sha256"]
            or len(manifest["selected"]) != 100):
        raise ValueError(f"{output}: frozen 100-shot star sample changed")
    scorer = origin / spec["scorer_provenance"]["frozen_root"]
    for key, relative in (
        ("local_score_py_sha256", "local_score.py"),
        ("derive_py_sha256", "fusion_scoring/derive.py"),
        ("lcfs_py_sha256", "fusion_scoring/lcfs.py"),
        ("contour_py_sha256", "fusion_scoring/contour.py"),
        ("o_point_py_sha256", "fusion_scoring/o_point.py"),
        ("common_py_sha256", "fusion_scoring/common.py"),
        ("diii_d_envelope_npz_sha256", "fusion_scoring/masks/d3d_envelope.npz"),
    ):
        path = scorer / relative
        if file_hash(path) != spec["scorer_provenance"][key]:
            raise ValueError(f"{path}: frozen scorer source SHA changed")
    return manifest, spec, scorer


def _star_one(task: tuple[dict[str, Any], str, str, dict[str, Any]]
              ) -> dict[str, Any]:
    selected, data_s, scorer_s, spec = task
    probe_spec = {**spec, "fixed_split": {
        "dataset_manifest": "my_experiments/diii_d_dataset.json"
    }}
    source = _contours_one((selected, data_s, scorer_s, probe_spec))
    scorer = spec["scorer_provenance"]
    _, derive, _, _, _ = _frozen_modules({"scorer_provenance": {
        "root": scorer_s,
        **{key: value for key, value in scorer.items() if key.endswith("_py_sha256")},
    }})
    rows: list[dict[str, Any]] = []
    for frame, contour in zip(source["indices"], source["contours"], strict=True):
        center = _polygon_centroid(contour)
        inside = bool(derive._points_in_polygon(
            center[0:1], center[1:2], contour
        )[0])
        details = _ray_uniqueness(contour, center, source["spacing_m"], inside)
        rows.append({"file": source["name"], "frame": frame, **details})
    return {"name": source["name"], "frames": source["frames"],
            "indices": source["indices"], "rows": rows}


def run(output: Path, jobs: int) -> None:
    """Measure centroid-ray uniqueness only on the frozen 300 TRAIN probes."""
    started = time.perf_counter()
    manifest, spec, scorer = _verify_run(output)
    if (output / "viability.json").exists():
        raise FileExistsError(f"{output}: star viability already measured")
    tasks = [(item, manifest["data_dir"], str(scorer), spec)
             for item in manifest["selected"]]
    rows: list[dict[str, Any]] = []
    try:
        for i, response in enumerate(pimap(_star_one, tasks, jobs)):
            wanted = manifest["selected"][i]
            if (response["name"] != wanted["file"] or response["frames"] != wanted["frames"]
                    or response["indices"] != wanted["probe_indices"]):
                raise ValueError(f"{wanted['file']}: star probe identity differs")
            rows.extend(response["rows"])
            if (i + 1) % 20 == 0:
                print(f"centroid-ray TRAIN viability: {i + 1}/100 shots", flush=True)
    finally:
        release()
    if len(rows) != 300:
        raise ValueError(f"{output}: {len(rows)} frame rows, expected 300")
    one = sum(row["exactly_one_positive_intersection_angles"] for row in rows)
    zero = sum(row["zero_intersection_angles"] for row in rows)
    multiple = sum(row["multiple_intersection_angles"] for row in rows)
    ambiguous = sum(row["collinear_ambiguous_angles"] for row in rows)
    inside = sum(row["centroid_strictly_inside"] for row in rows)
    if one + zero + multiple != 300 * 512:
        raise ValueError("star ray totals differ from 153,600 frozen angles")
    first_bad = next((row for row in rows if not row["centroid_strictly_inside"]
                      or row["exactly_one_positive_intersection_angles"] != 512
                      or row["collinear_ambiguous_angles"]), None)
    passing = inside == 300 and one == 153600 and ambiguous == 0
    positive = [row["minimum_positive_radius_m"] for row in rows
                if row["minimum_positive_radius_m"] is not None]
    write_json(output / "viability.json", {
        "classification": "centroid_star_shaped_viable" if passing
                          else "centroid_star_shaped_gate_failed",
        "spec_sha256": manifest["spec_sha256"],
        "manifest_sha256": file_hash(output / "manifest.json"),
        "train_sample_ordered_names_sha256": manifest["selected_ordered_names_sha256"],
        "shots": 100, "frames": 300, "rays": 153600,
        "strictly_inside_centroids": inside,
        "unique_positive_intersection_rays": one,
        "zero_intersection_rays": zero,
        "multiple_intersection_rays": multiple,
        "collinear_ambiguous_rays": ambiguous,
        "minimum_positive_radius_m": min(positive) if positive else None,
        "first_pathological_frame": first_bad,
        "gate_passed": passing,
        "per_frame_rows": rows,
        "wall_seconds": time.perf_counter() - started,
        "no_log_radius_PCA_or_model_fit_or_reserved_tail_read": True,
    })
    print(f"Centroid rays: inside {inside}/300, unique {one}/153600, "
          f"zero {zero}, multiple {multiple}, ambiguous {ambiguous}; gate {passing}")


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
        prepare(output)
    else:
        try:
            run(output, args.jobs)
        except Exception as exc:
            write_json(output / "viability_failure.json", {
                "classification": "star_numeric_or_input_failure",
                "manifest_sha256": file_hash(output / "manifest.json"),
                "exception_type": type(exc).__name__,
                "error": str(exc),
                "not_a_scientific_negative": True,
            })
            raise


if __name__ == "__main__":
    main()

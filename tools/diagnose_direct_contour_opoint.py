"""Train-only scorer O-point ray-uniqueness gate for true DIII-D LCFS."""

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

from my_experiments.data_check import check_dataset
from my_experiments.run import source_files
from toolkit.parallel import pimap, release
from toolkit.records import file_hash, write_json
from tools.diagnose_direct_contour_representation import _canonical_true, _name_hash
from tools.diagnose_direct_contour_star import _ray_uniqueness
from tools.geometry_map_smoke import _frozen_modules

ROOT = Path(__file__).resolve().parent.parent
SPEC = Path("experiments/direct-contour-opoint-star-review.json")


def _verify_sample(origin: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = json.loads((origin / SPEC).read_text())
    prior_result = origin / spec["distinct_prior"]["centroid_star_result"]
    if file_hash(prior_result) != spec["distinct_prior"]["centroid_star_result_sha256"]:
        raise ValueError(f"{prior_result}: centroid-star negative result SHA changed")
    prior = origin / spec["sample"]["prior_first100_train_manifest"]
    if file_hash(prior) != spec["sample"]["prior_first100_train_manifest_sha256"]:
        raise ValueError(f"{prior}: first100 TRAIN sample manifest SHA changed")
    manifest = json.loads(prior.read_text())
    selected = manifest["selected"]
    names = [row["file"] for row in selected]
    if (len(names) != 100 or _name_hash(names) != spec["sample"]["ordered_names_sha256"]
            or sum(len(row["probe_indices"]) for row in selected) != 300):
        raise ValueError(f"{prior}: frozen TRAIN sample shots/frames changed")
    artifact = origin / spec["sample"]["compact_artifact"]
    if file_hash(artifact) != spec["sample"]["compact_artifact_sha256"]:
        raise ValueError(f"{artifact}: compact artifact SHA changed")
    packed = joblib.load(artifact)
    if list(packed["train_files"][:100]) != names:
        raise ValueError(f"{artifact}: O-point sample differs from compact TRAIN head")
    score_record = origin / "runs/compact-production-9301af36275620f7/base_s0/metrics.json"
    score_names = json.loads(score_record.read_text())["files"]
    if set(names) & (set(packed["val_files"]) | set(score_names)):
        raise ValueError(f"{artifact}: O-point sample overlaps validation/evaluation")
    return spec, manifest


def prepare(output: Path) -> None:
    """Freeze scorer, code and exact TRAIN frame identities before reading true psi."""
    started = time.perf_counter()
    origin = ROOT.resolve()
    spec, prior = _verify_sample(origin)
    data = (origin / "../downloaded_huggingface/hf_dataset").resolve()
    corpus = check_dataset(data, "diii_d_train")
    if corpus["manifest_sha256"] != spec["sample"]["dataset_manifest_sha256"]:
        raise ValueError("pinned DIII-D corpus manifest SHA changed")
    if output.exists():
        raise FileExistsError(f"{output}: frozen O-point diagnostic exists")
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
            raise ValueError(f"{path}: frozen source copy SHA differs")
    frozen_spec = snapshot / SPEC
    frozen_spec.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origin / SPEC, frozen_spec)
    hashes[str(SPEC)] = file_hash(frozen_spec)
    frozen = {
        "status": "frozen_before_train_true_psi_read",
        "spec_sha256": file_hash(origin / SPEC),
        "prior_manifest_sha256": file_hash(
            origin / spec["sample"]["prior_first100_train_manifest"]
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
    write_json(output / "manifest.json", frozen)
    print(f"Frozen {output}: 100 compact TRAIN shots, 300 O-point probes, "
          f"{frozen['prepare_wall_seconds']:.2f} s")


def _verify_run(output: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    manifest = json.loads((output / "manifest.json").read_text())
    if ROOT.resolve() != (output / "source").resolve():
        raise ValueError(f"runner imported from {ROOT}, expected {output / 'source'}")
    for relative, digest in manifest["source_hashes"].items():
        path = ROOT / relative
        if file_hash(path) != digest:
            raise ValueError(f"{path}: frozen source SHA changed")
    origin = Path(manifest["origin_root"])
    if file_hash(origin / SPEC) != manifest["spec_sha256"]:
        raise ValueError(f"{origin / SPEC}: O-point spec changed")
    spec = json.loads((ROOT / SPEC).read_text())
    prior = origin / spec["sample"]["prior_first100_train_manifest"]
    if file_hash(prior) != manifest["prior_manifest_sha256"]:
        raise ValueError(f"{prior}: train sample manifest changed")
    if (len(manifest["selected"]) != 100
            or _name_hash([row["file"] for row in manifest["selected"]])
            != manifest["selected_ordered_names_sha256"]):
        raise ValueError(f"{output}: 100-shot O-point sample identity changed")
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
            raise ValueError(f"{path}: scorer source SHA changed")
    return manifest, spec, scorer


def _opoint_one(task: tuple[dict[str, Any], str, str, dict[str, Any]]
                ) -> dict[str, Any]:
    selected, data_s, scorer_s, spec = task
    name = selected["file"]
    path = Path(data_s) / "data/diii_d_train" / name
    pinned = json.loads((ROOT / "my_experiments/diii_d_dataset.json").read_text())
    expected_size, expected_sha = pinned["configs"]["diii_d_train"][name]
    if path.stat().st_size != expected_size or file_hash(path) != expected_sha:
        raise ValueError(f"{path}: parquet size/SHA differs from pinned manifest")
    scorer = spec["scorer_provenance"]
    local_score, derive, lcfs, _, _ = _frozen_modules({"scorer_provenance": {
        "root": scorer_s,
        **{key: val for key, val in scorer.items() if key.endswith("_py_sha256")},
    }})
    with np.load(Path(scorer_s) / "fusion_scoring/masks/d3d_envelope.npz") as saved:
        mask = saved["mask_coarse"].astype(bool)
        mask_r = saved["grid_R"]
        mask_z = saved["grid_Z"]
    row = pd.read_parquet(path, columns=["efit_times", "efit_grid_R", "efit_grid_Z",
                                         "efit_psirz"]).iloc[0]
    times = np.asarray(row["efit_times"], dtype=np.float64)
    R = np.asarray(row["efit_grid_R"], dtype=np.float64)
    Z = np.asarray(row["efit_grid_Z"], dtype=np.float64)
    if (len(times) != selected["frames"] or not np.array_equal(R, mask_r)
            or not np.array_equal(Z, mask_z)):
        raise ValueError(f"{path}: EFIT frames/grid differ from frozen scorer, "
                         f"got {len(times)} frames, expected {selected['frames']}")
    truth = local_score._as_psi_stack(row["efit_psirz"])
    if truth.shape != (len(times), 65, 65) or not np.isfinite(truth).all():
        raise ValueError(f"{path}: true psi shape/nonfinite count {truth.shape}/"
                         f"{int((~np.isfinite(truth)).sum())}, "
                         f"expected ({len(times)},65,65)/0")
    spacing = float(min(np.diff(R).min(), np.diff(Z).min()))
    details: list[dict[str, Any]] = []
    for frame in selected["probe_indices"]:
        psi = truth[frame]
        c_true = lcfs.extract_lcfs(psi, R, Z, "DIII-D", mask, mask.astype(float))
        if c_true is None:
            raise ValueError(f"{name} frame {frame}: scorer-native true LCFS missing")
        try:
            c = _canonical_true(c_true)
        except ValueError as exc:
            raise ValueError(f"{name} frame {frame}: {exc}") from exc
        ra, za, *_ = derive.magnetic_axis(psi, R, Z, "DIII-D", mask, axis_sign=None)
        center = np.asarray([ra, za], dtype=np.float64)
        if not np.isfinite(center).all():
            raise ValueError(f"{name} frame {frame}: scorer-native true O-point nonfinite")
        inside = bool(derive._points_in_polygon(center[0:1], center[1:2], c)[0])
        raw = _ray_uniqueness(c, center, spacing, inside)
        detail = {key.replace("centroid", "O_point"): val for key, val in raw.items()}
        details.append({"file": name, "frame": frame, **detail})
    return {"name": name, "frames": len(times), "indices": selected["probe_indices"],
            "rows": details}


def run(output: Path, jobs: int) -> None:
    """Count O-point ray intersections on exactly the 300 frozen TRAIN frames."""
    started = time.perf_counter()
    manifest, spec, scorer = _verify_run(output)
    if (output / "viability.json").exists():
        raise FileExistsError(f"{output}: O-point viability already measured")
    tasks = [(item, manifest["data_dir"], str(scorer), spec)
             for item in manifest["selected"]]
    rows: list[dict[str, Any]] = []
    try:
        for i, response in enumerate(pimap(_opoint_one, tasks, jobs)):
            wanted = manifest["selected"][i]
            if (response["name"] != wanted["file"] or response["frames"] != wanted["frames"]
                    or response["indices"] != wanted["probe_indices"]):
                raise ValueError(f"{wanted['file']}: O-point probe identity differs")
            rows.extend(response["rows"])
            if (i + 1) % 20 == 0:
                print(f"true-O-point TRAIN viability: {i + 1}/100 shots", flush=True)
    finally:
        release()
    if len(rows) != 300:
        raise ValueError(f"{output}: {len(rows)} frame rows, expected 300")
    one = sum(row["exactly_one_positive_intersection_angles"] for row in rows)
    zero = sum(row["zero_intersection_angles"] for row in rows)
    multiple = sum(row["multiple_intersection_angles"] for row in rows)
    ambiguous = sum(row["collinear_ambiguous_angles"] for row in rows)
    inside = sum(row["O_point_strictly_inside"] for row in rows)
    if one + zero + multiple != 153600:
        raise ValueError("O-point ray totals differ from 153,600 frozen angles")
    first_bad = next((row for row in rows if not row["O_point_strictly_inside"]
                      or row["exactly_one_positive_intersection_angles"] != 512
                      or row["collinear_ambiguous_angles"]), None)
    passed = inside == 300 and one == 153600 and ambiguous == 0
    radii = [row["minimum_positive_radius_m"] for row in rows
             if row["minimum_positive_radius_m"] is not None]
    write_json(output / "viability.json", {
        "classification": "true_O_point_star_shaped_viable" if passed
                          else "true_O_point_star_shaped_gate_failed",
        "spec_sha256": manifest["spec_sha256"],
        "manifest_sha256": file_hash(output / "manifest.json"),
        "train_sample_ordered_names_sha256": manifest["selected_ordered_names_sha256"],
        "shots": 100, "frames": 300, "rays": 153600,
        "strictly_inside_O_points": inside,
        "unique_positive_intersection_rays": one,
        "zero_intersection_rays": zero,
        "multiple_intersection_rays": multiple,
        "collinear_ambiguous_rays": ambiguous,
        "minimum_positive_radius_m": min(radii) if radii else None,
        "first_pathological_frame": first_bad,
        "gate_passed": passed,
        "per_frame_rows": rows,
        "wall_seconds": time.perf_counter() - started,
        "no_log_radius_PCA_or_model_fit_or_reserved_tail_read": True,
    })
    print(f"True O-point rays: inside {inside}/300, unique {one}/153600, "
          f"zero {zero}, multiple {multiple}, ambiguous {ambiguous}; gate {passed}")


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
                "classification": "O_point_numeric_or_input_failure",
                "manifest_sha256": file_hash(output / "manifest.json"),
                "exception_type": type(exc).__name__,
                "error": str(exc),
                "not_a_scientific_negative": True,
            })
            raise


if __name__ == "__main__":
    main()

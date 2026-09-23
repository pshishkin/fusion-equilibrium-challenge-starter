"""Fixed contour-local additive psi correction for a truth-assisted mechanism smoke.

True psi is used only to extract a scorer-native target contour. The output is
diagnostic and cannot be used as a public prediction or deployable model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

from toolkit.records import file_hash, write_json
from tools.geometry_map_smoke import (
    NumericalInfeasibility,
    _cyclic_match,
    _edge_anchors,
    _frozen_modules,
    _save,
)
from tools.geometry_taper import boundary_taper, contour_margin


def _correct_map(
    psi: np.ndarray,
    c_pred: np.ndarray,
    c_true: np.ndarray,
    axis: tuple[float, float],
    R: np.ndarray,
    Z: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the preregistered first-order Taylor correction, without true psi pixels."""
    if psi.shape != (65, 65) or psi.dtype != np.float32 or not np.isfinite(psi).all():
        raise NumericalInfeasibility(
            f"source psi shape/dtype/finiteness {psi.shape}/{psi.dtype}/"
            f"{int((~np.isfinite(psi)).sum())}, expected (65,65)/float32/0"
        )
    matched, shift = _cyclic_match(c_pred, c_true)
    movement = np.linalg.norm(matched - c_pred, axis=1)
    max_move = float(movement.max())
    p95_move = float(np.quantile(movement, 0.95))
    lo = np.array([R[0], Z[0]])
    span = np.array([R[-1] - R[0], Z[-1] - Z[0]])
    selected = np.arange(0, 512, 8)
    controls = (c_pred[selected] - lo) / span
    target_displacement = (matched[selected] - c_pred[selected]) / span
    anchors = np.vstack([controls, (np.asarray(axis) - lo) / span, _edge_anchors()])
    displacements = np.vstack([target_displacement, np.zeros((17, 2))])
    if len(np.unique(anchors, axis=0)) != len(anchors):
        raise NumericalInfeasibility("TPS control points are not unique")
    try:
        rbf = RBFInterpolator(
            anchors, displacements, kernel="thin_plate_spline", smoothing=0.0001, degree=1
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise NumericalInfeasibility(f"TPS solve failed: {type(exc).__name__}") from exc
    try:
        margin = contour_margin(c_pred, R, Z)
        contour_taper = boundary_taper(controls, lo, span, margin)
        axis_taper = boundary_taper(((np.asarray(axis) - lo) / span)[None, :], lo, span, margin)
    except ValueError as exc:
        raise NumericalInfeasibility(f"boundary taper invalid: {exc}") from exc
    if not np.all(contour_taper == 1.0) or not np.all(axis_taper == 1.0):
        raise NumericalInfeasibility("contour/axis taper is not full strength")

    grid_r, grid_z = np.meshgrid(R, Z)
    physical = np.column_stack([grid_r.ravel(), grid_z.ravel()])
    unit = (physical - lo) / span
    taper = boundary_taper(unit, lo, span, margin)
    edge = np.isclose(unit[:, 0], 0) | np.isclose(unit[:, 0], 1)
    edge |= np.isclose(unit[:, 1], 0) | np.isclose(unit[:, 1], 1)
    if not np.all(taper[edge] == 0.0):
        raise NumericalInfeasibility("edge taper is nonzero on grid boundary")
    displacement_physical = rbf(unit) * taper[:, None] * span
    if not np.isfinite(displacement_physical).all():
        raise NumericalInfeasibility("TPS displacement is nonfinite")
    width = 2.0 * float(min(np.diff(R).min(), np.diff(Z).min()))
    if width <= 0:
        raise NumericalInfeasibility(f"contour shell width {width} must be positive")
    distance, _ = cKDTree(c_pred).query(physical, k=1)
    shell = np.exp(-0.5 * (distance / width) ** 2)
    dpsi_dz, dpsi_dr = np.gradient(psi.astype(np.float64), Z, R)
    if not np.isfinite(dpsi_dz).all() or not np.isfinite(dpsi_dr).all():
        raise NumericalInfeasibility("physical psi gradient is nonfinite")
    change = -shell * (
        displacement_physical[:, 0] * dpsi_dr.ravel()
        + displacement_physical[:, 1] * dpsi_dz.ravel()
    )
    result = (psi.astype(np.float64).ravel() + change).reshape(65, 65).astype(np.float32)
    if result.shape != psi.shape or not np.isfinite(result).all():
        raise NumericalInfeasibility("additive map has wrong shape or nonfinite values")
    if result.ravel()[edge].tobytes() != psi.ravel()[edge].tobytes():
        raise NumericalInfeasibility("additive map changed grid-edge psi bits")
    return result, {
        "cyclic_shift": float(shift),
        "max_matched_contour_move_m": max_move,
        "p95_matched_contour_move_m": p95_move,
        "boundary_taper_margin_m": margin,
        "shell_width_m": width,
        "max_abs_displacement_m": float(np.linalg.norm(displacement_physical, axis=1).max()),
        "max_abs_delta_psi": float(np.abs(change).max()),
    }


def build(
    spec_path: Path, arm: str, output_dir: Path, extension_path: Path | None = None
) -> dict[str, Any]:
    if arm not in ("control", "additive"):
        raise ValueError(f"unknown arm {arm}")
    started = time.perf_counter()
    spec = json.loads(spec_path.read_text())
    if file_hash(Path(spec["amends_spec"])) != spec["amends_spec_sha256"]:
        raise ValueError(f"{spec_path}: amended source spec SHA differs")
    source = Path(spec["input"]["prediction"])
    if file_hash(source) != spec["input"]["prediction_sha256"]:
        raise ValueError(f"{source}: source NPZ SHA differs from frozen spec")
    files_json = Path(spec["input"]["scored_files_json"])
    if file_hash(files_json) != spec["input"]["scored_files_json_sha256"]:
        raise ValueError(f"{files_json}: scored filenames SHA differs from frozen spec")
    all_files = json.loads(files_json.read_text())["files"]
    if extension_path is None:
        scope = "first3"
        names = spec["smoke"]["files"]
        frame_counts = spec["smoke"]["frames_per_shot"]
        expected_frames = spec["smoke"]["total_frames"]
        extension_sha256 = None
        if all_files[:3] != names or len(names) != 3:
            raise ValueError(f"{files_json}: first three ordered filenames changed")
    else:
        scope = "full70"
        extension = json.loads(extension_path.read_text())
        extension_sha256 = file_hash(extension_path)
        if arm != "additive" or extension["arm"] != "additive":
            raise ValueError(f"{extension_path}: full70 permits only the fixed additive arm")
        if extension["base_spec_sha256"] != file_hash(spec_path):
            raise ValueError(f"{extension_path}: base spec changed")
        if extension["producer_sha256"] != file_hash(Path(__file__)):
            raise ValueError(f"{extension_path}: producer source changed")
        if extension["prediction_sha256"] != spec["input"]["prediction_sha256"]:
            raise ValueError(f"{extension_path}: source prediction changed")
        if extension["files_json_sha256"] != file_hash(files_json):
            raise ValueError(f"{extension_path}: scored file list changed")
        formula_hash = hashlib.sha256(
            json.dumps(spec["construction"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if formula_hash != extension["construction_sha256"]:
            raise ValueError(f"{extension_path}: additive construction changed")
        names = all_files
        frame_counts = extension["frames_per_shot"]
        expected_frames = extension["total_frames"]
        if (
            names != extension["files"]
            or len(names) != extension["shots"]
            or len(frame_counts) != len(names)
            or hashlib.sha256("\n".join(names).encode()).hexdigest()
            != extension["ordered_filenames_sha256"]
        ):
            raise ValueError(f"{extension_path}: ordered filenames/frame lengths changed")
    scorer_spec = {
        "scorer_provenance": {
            "root": spec["scoring"]["frozen_root"],
            **{k: v for k, v in spec["scoring"].items() if k.endswith("_py_sha256")},
        }
    }
    local_score, derive, lcfs, contour, common = _frozen_modules(scorer_spec)
    mask_path = Path(spec["scoring"]["frozen_root"]) / "fusion_scoring/masks/d3d_envelope.npz"
    with np.load(mask_path) as masks:
        mask = masks["mask_coarse"].astype(bool)
        mask_r, mask_z = masks["grid_R"], masks["grid_Z"]
    sign = int(common.AXIS_SIGN["DIII-D"])
    arrays: dict[str, np.ndarray] = {}
    diagnostics: list[dict[str, Any]] = []
    with np.load(source, allow_pickle=False) as packed:
        for i, name in enumerate(names):
            path = Path(spec["input"]["data_dir"]) / "data" / spec["input"]["data_config"] / name
            row = pd.read_parquet(path).iloc[0]
            R = np.asarray(row["efit_grid_R"], dtype=np.float64)
            Z = np.asarray(row["efit_grid_Z"], dtype=np.float64)
            if not np.array_equal(R, mask_r) or not np.array_equal(Z, mask_z):
                raise ValueError(f"{name}: physical grid differs from frozen scorer mask")
            n = frame_counts[i]
            psi_stack = packed[f"shot_{i:04d}_psirz"]
            if psi_stack.shape != (n, 65, 65) or psi_stack.dtype != np.float32:
                raise ValueError(
                    f"{name}: source psi shape/dtype {psi_stack.shape}/{psi_stack.dtype}"
                )
            if arm == "control":
                output_maps = psi_stack.copy()
            else:
                truth_stack = local_score._as_psi_stack(row["efit_psirz"])
                if truth_stack.shape != psi_stack.shape or not np.isfinite(truth_stack).all():
                    raise ValueError(f"{name}: true psi shape/finiteness differs from source")
                output_frames = []
                for frame, (psi, truth) in enumerate(zip(psi_stack, truth_stack, strict=True)):
                    try:
                        c_pred, _ = lcfs.extract_lcfs_with_sign(
                            psi, R, Z, "DIII-D", mask, mask.astype(float), axis_sign=sign
                        )
                        c_true = lcfs.extract_lcfs(truth, R, Z, "DIII-D", mask, mask.astype(float))
                        if c_pred is None or c_true is None:
                            raise NumericalInfeasibility("source or true scorer LCFS is missing")
                        axis_r, axis_z, *_ = derive.magnetic_axis(psi, R, Z, "DIII-D", mask, sign)
                        corrected, detail = _correct_map(
                            psi, c_pred, c_true, (axis_r, axis_z), R, Z
                        )
                        c_new, _ = lcfs.extract_lcfs_with_sign(
                            corrected, R, Z, "DIII-D", mask, mask.astype(float), axis_sign=sign
                        )
                        detail["source_contour_to_true_hausdorff_m"] = contour.symmetric_hausdorff(
                            c_pred, c_true
                        )
                        detail["corrected_lcfs_missing"] = float(c_new is None)
                        detail["corrected_contour_to_true_hausdorff_m"] = (
                            contour.symmetric_hausdorff(c_new, c_true)
                            if c_new is not None
                            else None
                        )
                        output_frames.append(corrected)
                        diagnostics.append({"shot": name, "frame": frame, **detail})
                    except NumericalInfeasibility as exc:
                        raise NumericalInfeasibility(
                            f"additive: {name} frame {frame}: {exc}"
                        ) from exc
                output_maps = np.stack(output_frames)
            arrays[f"shot_{i:04d}_psirz"] = output_maps
            for field in ("q95", "betaN"):
                key = f"shot_{i:04d}_{field}"
                value = packed[key]
                if value.shape != (n,) or value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError(f"{name}: {key} shape/dtype/finiteness differs")
                arrays[key] = value.copy()
            if scope == "full70" and (i + 1) % 5 == 0:
                print(f"additive full70: {i + 1}/{len(names)} shots complete", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if sum(frame_counts) != expected_frames:
        raise ValueError(f"{scope}: {sum(frame_counts)} frames, expected {expected_frames}")
    output = output_dir / f"{arm}-{scope}.npz"
    _save(output, arrays)
    with (
        np.load(source, allow_pickle=False) as original,
        np.load(output, allow_pickle=False) as packed,
    ):
        for i in range(len(names)):
            for field in ("q95", "betaN"):
                key = f"shot_{i:04d}_{field}"
                if packed[key].tobytes() != original[key].tobytes():
                    raise AssertionError(f"{output}: {key} scalar bits changed")
            if arm == "control":
                key = f"shot_{i:04d}_psirz"
                if packed[key].tobytes() != original[key].tobytes():
                    raise AssertionError(f"{output}: source control {key} map bits changed")
    result: dict[str, Any] = {
        "classification": "valid_finite_map_construction",
        "arm": arm,
        "scope": scope,
        "spec": str(spec_path),
        "spec_sha256": file_hash(spec_path),
        "extension_spec_sha256": extension_sha256,
        "producer_sha256": file_hash(Path(__file__)),
        "source_prediction_sha256": spec["input"]["prediction_sha256"],
        "files": names,
        "frames": expected_frames,
        "prediction_npz": str(output),
        "prediction_sha256": file_hash(output),
        "scalar_bits_unchanged": True,
        "control_all_array_bits_exact": arm == "control",
        "true_psi_pixels_copied_into_output": False,
        "corrected_lcfs_missing_frames": int(sum(d["corrected_lcfs_missing"] for d in diagnostics)),
        "per_frame_diagnostics": diagnostics,
        "build_seconds": time.perf_counter() - started,
    }
    write_json(output.with_suffix(".json"), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--arm", choices=("control", "additive"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--extension", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        result = build(args.spec, args.arm, args.out_dir, args.extension)
    except NumericalInfeasibility as exc:
        scope = "full70" if args.extension is not None else "first3"
        write_json(
            args.out_dir / f"{args.arm}-{scope}-failure.json",
            {
                "classification": "numeric_or_source_contour_infeasibility",
                "arm": args.arm,
                "scope": scope,
                "spec_sha256": file_hash(args.spec),
                "extension_spec_sha256": file_hash(args.extension) if args.extension else None,
                "error": str(exc),
                "wall_seconds": time.perf_counter() - started,
                "no_scientific_negative_inferred": True,
            },
        )
        print(f"{args.arm} construction failed: {exc}")
        return 2
    print(
        f"{args.arm}: {len(result['files'])} shots, {result['frames']} frames, "
        f"build {result['build_seconds']:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Match an additive map's li to the ORIGINAL predicted map's own derived li.

This mechanism probe reads only prediction NPZs and the input shot's grid. It does not
read true psi, stored li, or any other target column. The additive input itself remains
truth-assisted and is not a public prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from toolkit.records import file_hash, write_json
from tools.geometry_map_smoke import NumericalInfeasibility, _frozen_modules, _save


def _grid_for_shot(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read input geometry only; parquet target columns are never requested."""
    row = pd.read_parquet(path, columns=["efit_grid_R", "efit_grid_Z"]).iloc[0]
    return np.asarray(row["efit_grid_R"], dtype=np.float64), np.asarray(
        row["efit_grid_Z"], dtype=np.float64
    )


def _exact_segment_distance(points: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """Minimum physical point-to-polygon-segment distance in fixed 64-edge chunks."""
    if points.ndim != 2 or points.shape[1] != 2 or contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError(f"point/contour shapes {points.shape}/{contour.shape}, expected (N,2)")
    if not np.isfinite(points).all() or not np.isfinite(contour).all():
        raise ValueError("point/contour geometry is nonfinite")
    start = contour
    end = np.roll(contour, -1, axis=0)
    segment = end - start
    length_sq = np.sum(segment * segment, axis=1)
    if np.any(length_sq <= 0):
        raise NumericalInfeasibility("additive contour has a degenerate polygon segment")
    best_sq = np.full(len(points), np.inf)
    for lo in range(0, len(contour), 64):
        hi = min(lo + 64, len(contour))
        a = start[lo:hi]
        v = segment[lo:hi]
        delta = points[:, None, :] - a[None, :, :]
        projection = np.clip(np.sum(delta * v[None, :, :], axis=2) / length_sq[None, lo:hi], 0, 1)
        nearest = a[None, :, :] + projection[:, :, None] * v[None, :, :]
        difference = points[:, None, :] - nearest
        best_sq = np.minimum(best_sq, np.min(np.sum(difference * difference, axis=2), axis=1))
    return np.sqrt(best_sq)


def _interior_taper(
    contour: np.ndarray, R: np.ndarray, Z: np.ndarray, derive: Any
) -> tuple[np.ndarray, float, int, float]:
    started = time.perf_counter()
    rr, zz = np.meshgrid(R, Z)
    points = np.column_stack([rr.ravel(), zz.ravel()])
    inside = derive._points_in_polygon(points[:, 0], points[:, 1], contour)
    distance = _exact_segment_distance(points, contour)
    width = float(min(np.diff(R).min(), np.diff(Z).min()))
    if not np.isfinite(width) or width <= 0:
        raise NumericalInfeasibility(f"interior taper width {width}, expected positive finite")
    signed = np.where(inside, distance, -distance)
    t = np.clip(signed / width, 0.0, 1.0)
    taper = (t * t * (3.0 - 2.0 * t)).reshape(65, 65)
    edge = np.zeros((65, 65), dtype=bool)
    edge[0] = edge[-1] = True
    edge[:, 0] = edge[:, -1] = True
    if not np.all(taper[edge] == 0.0):
        raise NumericalInfeasibility("inside-polygon taper is nonzero on grid edge")
    contour_distance = _exact_segment_distance(contour, contour)
    if not np.all(contour_distance == 0.0):
        raise NumericalInfeasibility("inside-polygon taper is nonzero on polygon contour")
    return taper, width, int(np.count_nonzero(inside)), time.perf_counter() - started


def _spatial_profile(
    additive: np.ndarray,
    sign: int,
    axis_phi: float,
    sep_phi: float,
    a: float,
    taper: np.ndarray,
) -> np.ndarray:
    if a == 0:
        return additive.copy()
    span = axis_phi - sep_phi
    if not np.isfinite(span) or span <= 0:
        raise NumericalInfeasibility(f"axis-separatrix span {span}, expected positive finite")
    phi = sign * additive.astype(np.float64)
    u = (axis_phi - phi) / span
    interior = (u >= 0) & (u <= 1)
    g = u.copy()
    g[interior] += a * u[interior] ** 2 * (1 - u[interior]) ** 2
    global_psi = sign * (axis_phi - span * g)
    result = (
        additive.astype(np.float64) + taper * (global_psi - additive.astype(np.float64))
    ).astype(np.float32)
    if result.shape != additive.shape or not np.isfinite(result).all():
        raise NumericalInfeasibility(f"a={a} produced nonfinite/wrong-shape spatial remap")
    return result


def _select(
    valid: list[tuple[float, float, np.ndarray, float, float, float]], target_li: float
) -> tuple[tuple[float, float, np.ndarray, float, float, float], bool]:
    """Choose one fixed-grid candidate and report whether the target lies outside its li span."""
    if not any(item[0] == 0.0 for item in valid):
        raise NumericalInfeasibility("a=0 exact control is missing from valid candidates")
    selected = min(valid, key=lambda item: (abs(item[3] - target_li), abs(item[0]), item[0]))
    values = [item[3] for item in valid]
    return selected, target_li < min(values) or target_li > max(values)


def _compensate_frame(
    source: np.ndarray,
    additive: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    mask: np.ndarray,
    sign: int,
    a_grid: list[float],
    derive: Any,
    lcfs: Any,
    contour: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    if (
        source.shape != (65, 65)
        or additive.shape != (65, 65)
        or source.dtype != np.float32
        or additive.dtype != np.float32
        or not np.isfinite(source).all()
        or not np.isfinite(additive).all()
    ):
        raise NumericalInfeasibility("source/additive psi must be finite float32 (65,65)")
    c_source, _ = lcfs.extract_lcfs_with_sign(
        source, R, Z, "DIII-D", mask, mask.astype(float), axis_sign=sign
    )
    c_add, _ = lcfs.extract_lcfs_with_sign(
        additive, R, Z, "DIII-D", mask, mask.astype(float), axis_sign=sign
    )
    if c_source is None or c_add is None:
        raise NumericalInfeasibility("original or additive source map has no scorer LCFS")
    target_li = float(derive.internal_inductance(source, R, Z, "DIII-D", c_source))
    if not np.isfinite(target_li):
        raise NumericalInfeasibility("original source-map li target is nonfinite")
    axis_r, axis_z, iz, ir = derive.magnetic_axis(additive, R, Z, "DIII-D", mask, sign)
    phi = sign * additive.astype(np.float64)
    axis_phi = float(phi[iz, ir])
    levels = derive._bilinear(phi, R, Z, c_add[:, 0], c_add[:, 1])
    sep_phi = float(np.median(levels))
    span = axis_phi - sep_phi
    if not np.isfinite(levels).all() or not np.isfinite(span) or span <= 0:
        raise NumericalInfeasibility("additive contour level/span is nonfinite or nonpositive")
    spread = np.abs(levels - sep_phi) / span
    taper, interior_width, inside_count, distance_seconds = _interior_taper(c_add, R, Z, derive)
    spacing = float(min(np.diff(R).min(), np.diff(Z).min()))
    edge = np.zeros(additive.shape, dtype=bool)
    edge[0] = edge[-1] = True
    edge[:, 0] = edge[:, -1] = True
    valid: list[tuple[float, float, np.ndarray, float, float, float]] = []
    rejected: list[dict[str, Any]] = []
    for a in a_grid:
        reason: str | None = None
        try:
            candidate = _spatial_profile(additive, sign, axis_phi, sep_phi, a, taper)
        except NumericalInfeasibility:
            candidate = additive.copy()
            reason = "nonfinite_profile_map"
        if reason is None:
            if candidate.shape != additive.shape or candidate.dtype != np.float32:
                reason = "shape_or_dtype"
            elif candidate[edge].tobytes() != additive[edge].tobytes():
                reason = "boundary_bits_changed"
        if reason is None:
            ra, za, *_ = derive.magnetic_axis(candidate, R, Z, "DIII-D", mask, sign)
            axis_shift = float(np.hypot(ra - axis_r, za - axis_z))
            if not np.isfinite(axis_shift) or axis_shift > 0.01 * spacing:
                reason = "axis_moved"
        else:
            axis_shift = float("nan")
        if reason is None:
            c_candidate, _ = lcfs.extract_lcfs_with_sign(
                candidate, R, Z, "DIII-D", mask, mask.astype(float), axis_sign=sign
            )
            if c_candidate is None:
                reason = "lcfs_missing"
        else:
            c_candidate = None
        if reason is None:
            contour_shift = float(contour.symmetric_hausdorff(c_candidate, c_add))
            if not np.isfinite(contour_shift) or contour_shift > 0.1 * spacing:
                reason = "contour_moved"
        else:
            contour_shift = float("nan")
        if reason is None:
            li = float(derive.internal_inductance(candidate, R, Z, "DIII-D", c_candidate))
            if not np.isfinite(li):
                reason = "li_nonfinite"
        else:
            li = float("nan")
        if reason is None:
            valid.append((a, abs(li - target_li), candidate, li, axis_shift, contour_shift))
        else:
            if a == 0:
                raise NumericalInfeasibility(f"a=0 exact control failed: {reason}")
            rejected.append({"a": a, "reason": reason})
    selected, outside = _select(valid, target_li)
    zero = next(item for item in valid if item[0] == 0)
    if zero[2].tobytes() != additive.tobytes():
        raise NumericalInfeasibility("a=0 map lost additive source bits")
    return selected[2], {
        "selected_a": selected[0],
        "source_li_target": target_li,
        "additive_li_before": zero[3],
        "selected_li_after": selected[3],
        "abs_li_target_error_before": abs(zero[3] - target_li),
        "abs_li_target_error_after": abs(selected[3] - target_li),
        "selected_axis_shift_m": selected[4],
        "selected_contour_shift_m": selected[5],
        "target_outside_valid_li_span": outside,
        "valid_a_count": len(valid),
        "rejected": rejected,
        "contour_level_max_spread_over_span": float(np.max(spread)),
        "contour_level_p95_spread_over_span": float(np.quantile(spread, 0.95)),
        "interior_taper_width_m": interior_width,
        "inside_grid_points": inside_count,
        "interior_taper_boundary_max": float(np.max(taper[edge])),
        "exact_segment_distance_seconds": distance_seconds,
    }


def build(
    spec_path: Path, arm: str, output_dir: Path, extension_path: Path | None = None
) -> dict[str, Any]:
    if arm not in ("control", "compensated"):
        raise ValueError(f"unknown arm {arm}")
    started = time.perf_counter()
    spec = json.loads(spec_path.read_text())
    inputs = spec["input"]
    for path_key, hash_key in (
        ("source_prediction", "source_prediction_sha256"),
        ("additive_prediction", "additive_prediction_sha256"),
        ("additive_prediction_record", "additive_prediction_record_sha256"),
        ("additive_native_result", "additive_native_result_sha256"),
        ("files_json", "files_json_sha256"),
    ):
        path = Path(inputs[path_key])
        if file_hash(path) != inputs[hash_key]:
            raise ValueError(f"{path}: SHA differs from frozen source-li spec")
    all_files = json.loads(Path(inputs["files_json"]).read_text())["files"]
    if extension_path is None:
        scope = "first3"
        names = spec["smoke"]["files"]
        frame_counts = spec["smoke"]["frames_per_shot"]
        expected_frames = spec["smoke"]["total_frames"]
        extension_sha256 = None
        if all_files[:3] != names or len(names) != 3:
            raise ValueError("first three ordered filenames differ from frozen spec")
    else:
        scope = "full70"
        extension = json.loads(extension_path.read_text())
        extension_sha256 = file_hash(extension_path)
        if arm != "compensated" or extension["arm"] != "compensated":
            raise ValueError(f"{extension_path}: full70 permits only the compensated arm")
        if extension["base_spec_sha256"] != file_hash(spec_path):
            raise ValueError(f"{extension_path}: base spec changed")
        if extension["producer_sha256"] != file_hash(Path(__file__)):
            raise ValueError(f"{extension_path}: producer source changed")
        if (
            extension["source_prediction_sha256"] != inputs["source_prediction_sha256"]
            or extension["additive_prediction_sha256"] != inputs["additive_prediction_sha256"]
            or extension["files_json_sha256"] != file_hash(Path(inputs["files_json"]))
        ):
            raise ValueError(f"{extension_path}: prediction or file-list identity changed")
        family_hash = hashlib.sha256(
            json.dumps(spec["family"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if extension["family_sha256"] != family_hash:
            raise ValueError(f"{extension_path}: source-li family changed")
        if file_hash(Path(extension["first3_result"])) != extension["first3_result_sha256"]:
            raise ValueError(f"{extension_path}: first3 evidence changed")
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
            raise ValueError(f"{extension_path}: ordered files/frame counts changed")
    scorer_spec = {
        "scorer_provenance": {
            "root": spec["scoring"]["frozen_root"],
            **{k: v for k, v in spec["scoring"].items() if k.endswith("_py_sha256")},
        }
    }
    _, derive, lcfs, contour, common = _frozen_modules(scorer_spec)
    mask_path = Path(spec["scoring"]["frozen_root"]) / "fusion_scoring/masks/d3d_envelope.npz"
    with np.load(mask_path) as masks:
        mask = masks["mask_coarse"].astype(bool)
        mask_r, mask_z = masks["grid_R"], masks["grid_Z"]
    sign = int(common.AXIS_SIGN["DIII-D"])
    arrays: dict[str, np.ndarray] = {}
    frame_details: list[dict[str, Any]] = []
    with (
        np.load(inputs["source_prediction"], allow_pickle=False) as original,
        np.load(inputs["additive_prediction"], allow_pickle=False) as additive,
    ):
        for i, name in enumerate(names):
            path = Path(inputs["data_dir"]) / "data" / inputs["config"] / name
            R, Z = _grid_for_shot(path)
            if not np.array_equal(R, mask_r) or not np.array_equal(Z, mask_z):
                raise ValueError(f"{name}: input grid differs from frozen scorer mask")
            n = frame_counts[i]
            key = f"shot_{i:04d}_psirz"
            source_maps, additive_maps = original[key], additive[key]
            if (
                source_maps.shape != (n, 65, 65)
                or additive_maps.shape != (n, 65, 65)
                or source_maps.dtype != np.float32
                or additive_maps.dtype != np.float32
            ):
                raise ValueError(f"{name}: source/additive map shape or dtype changed")
            if arm == "control":
                output_maps = additive_maps.copy()
            else:
                output_frames = []
                for frame, (source, corrected) in enumerate(
                    zip(source_maps, additive_maps, strict=True)
                ):
                    try:
                        chosen, detail = _compensate_frame(
                            source,
                            corrected,
                            R,
                            Z,
                            mask,
                            sign,
                            spec["family"]["candidate_a_grid"],
                            derive,
                            lcfs,
                            contour,
                        )
                        output_frames.append(chosen)
                        frame_details.append({"shot": name, "frame": frame, **detail})
                    except NumericalInfeasibility as exc:
                        raise NumericalInfeasibility(
                            f"source-li: {name} frame {frame}: {exc}"
                        ) from exc
                output_maps = np.stack(output_frames)
            arrays[key] = output_maps
            for field in ("q95", "betaN"):
                key = f"shot_{i:04d}_{field}"
                left, right = original[key], additive[key]
                if (
                    left.shape != (n,)
                    or left.dtype != np.float32
                    or right.shape != (n,)
                    or right.dtype != np.float32
                    or left.tobytes() != right.tobytes()
                ):
                    raise ValueError(f"{name}: original/additive {field} arrays differ")
                arrays[key] = right.copy()
            if scope == "full70" and (i + 1) % 5 == 0:
                print(f"source-li full70: {i + 1}/{len(names)} shots complete", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if sum(frame_counts) != expected_frames:
        raise ValueError(f"{scope}: {sum(frame_counts)} frames, expected {expected_frames}")
    output = output_dir / f"{arm}-{scope}.npz"
    _save(output, arrays)
    with (
        np.load(inputs["additive_prediction"], allow_pickle=False) as before,
        np.load(output, allow_pickle=False) as after,
    ):
        for i in range(len(names)):
            for field in ("q95", "betaN"):
                key = f"shot_{i:04d}_{field}"
                if after[key].tobytes() != before[key].tobytes():
                    raise AssertionError(f"{output}: {key} scalar bits changed")
            if arm == "control":
                key = f"shot_{i:04d}_psirz"
                if after[key].tobytes() != before[key].tobytes():
                    raise AssertionError(f"{output}: control {key} psi bits changed")
    rejected = Counter(item["reason"] for frame in frame_details for item in frame["rejected"])
    result: dict[str, Any] = {
        "classification": "valid_source_li_mechanism_construction",
        "arm": arm,
        "scope": scope,
        "spec": str(spec_path),
        "spec_sha256": file_hash(spec_path),
        "extension_spec_sha256": extension_sha256,
        "producer_sha256": file_hash(Path(__file__)),
        "source_prediction_sha256": inputs["source_prediction_sha256"],
        "additive_prediction_sha256": inputs["additive_prediction_sha256"],
        "files": names,
        "frames": expected_frames,
        "prediction_npz": str(output),
        "prediction_sha256": file_hash(output),
        "control_all_array_bits_exact_to_additive": arm == "control",
        "scalar_bits_unchanged": True,
        "true_li_or_target_columns_read": False,
        "invalid_nonzero_a_by_reason": dict(sorted(rejected.items())),
        "invalid_nonzero_a_total": sum(rejected.values()),
        "per_frame_diagnostics": frame_details,
        "build_seconds": time.perf_counter() - started,
    }
    write_json(output.with_suffix(".json"), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--arm", choices=("control", "compensated"), required=True)
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
                "classification": "source_li_mechanism_numeric_infeasibility",
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
        f"invalid nonzero a={result['invalid_nonzero_a_total']}, "
        f"build {result['build_seconds']:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

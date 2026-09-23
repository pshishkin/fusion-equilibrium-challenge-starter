"""Truth-assisted, three-shot map-space mechanism smoke from a frozen spec.

Corrected maps are diagnostic only. Their construction reads labeled psi and must never
be used for public prediction. The competition score is delegated to local_score.py.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator, RegularGridInterpolator

from toolkit.records import file_hash, write_json
from tools.historical_compare import _save


class NumericalInfeasibility(ValueError):
    """This fixed correction cannot produce a valid psi map on the selected frame."""


def _frozen_modules(spec: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
    root = Path(spec["scorer_provenance"]["root"]).resolve()
    names = {
        "local_score": "local_score.py",
        "derive": "fusion_scoring/derive.py",
        "lcfs": "fusion_scoring/lcfs.py",
        "contour": "fusion_scoring/contour.py",
        "common": "fusion_scoring/common.py",
        "o_point": "fusion_scoring/o_point.py",
    }
    hashes = spec["scorer_provenance"]
    for name, path in names.items():
        key = f"{name}_py_sha256"
        if file_hash(root / path) != hashes[key]:
            raise ValueError(f"{root / path}: source SHA differs from frozen spec")
    sys.path.insert(0, str(root / "fusion_scoring"))
    sys.path.insert(0, str(root))
    modules = [importlib.import_module(name) for name in names]
    for name, module in zip(names, modules, strict=True):
        if module.__file__ is None or Path(module.__file__).resolve() != root / names[name]:
            raise ValueError(f"{name}: imported {module.__file__}, expected {root / names[name]}")
    local_score, derive, lcfs, contour, common, _ = modules
    return local_score, derive, lcfs, contour, common


def _signed_area(c: np.ndarray) -> float:
    return float(0.5 * np.sum(c[:, 0] * np.roll(c[:, 1], -1) - c[:, 1] * np.roll(c[:, 0], -1)))


def _simple_closed(c: np.ndarray) -> bool:
    """Reject degenerate or crossing edges, without assuming a star-shaped contour."""
    n = len(c)
    if c.shape != (512, 2) or not np.isfinite(c).all():
        return False
    a = c
    b = np.roll(c, -1, axis=0)
    if np.any(np.linalg.norm(b - a, axis=1) <= 1e-10) or abs(_signed_area(c)) <= 1e-10:
        return False
    i, j = np.triu_indices(n, k=2)
    keep = ~((i == 0) & (j == n - 1))
    i, j = i[keep], j[keep]
    ai, bi, aj, bj = a[i], b[i], a[j], b[j]

    def cross(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]

    u = bi - ai
    v = bj - aj
    c1 = cross(u, aj - ai)
    c2 = cross(u, bj - ai)
    c3 = cross(v, ai - aj)
    c4 = cross(v, bi - aj)
    return not bool(np.any((c1 * c2 < 0) & (c3 * c4 < 0)))


def _cyclic_match(pred: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, int]:
    if not _simple_closed(pred) or not _simple_closed(truth):
        raise NumericalInfeasibility("non-simple or degenerate LCFS contour")
    q = truth[::-1] if np.sign(_signed_area(pred)) != np.sign(_signed_area(truth)) else truth
    corr = np.fft.ifft(
        np.conj(np.fft.fft(pred[:, 0])) * np.fft.fft(q[:, 0])
        + np.conj(np.fft.fft(pred[:, 1])) * np.fft.fft(q[:, 1])
    ).real
    shift = int(np.argmax(corr))
    return np.roll(q, -shift, axis=0), shift


def _edge_anchors() -> np.ndarray:
    t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    return np.unique(
        np.vstack(
            [
                np.column_stack([t, np.zeros_like(t)]),
                np.column_stack([t, np.ones_like(t)]),
                np.column_stack([np.zeros_like(t), t]),
                np.column_stack([np.ones_like(t), t]),
            ]
        ),
        axis=0,
    )


def _warp_map(
    psi: np.ndarray,
    pred: np.ndarray,
    truth: np.ndarray,
    axis: tuple[float, float],
    R: np.ndarray,
    Z: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    q, shift = _cyclic_match(pred, truth)
    spacing = float(np.hypot(np.mean(np.diff(R)), np.mean(np.diff(Z))))
    displacement = q - pred
    max_move = float(np.linalg.norm(displacement, axis=1).max())
    if max_move > 4 * spacing:
        raise NumericalInfeasibility(f"matched contour displacement {max_move} > {4 * spacing}")
    lo = np.array([R[0], Z[0]])
    span = np.array([R[-1] - R[0], Z[-1] - Z[0]])
    selected = np.arange(0, 512, 8)
    points = (pred[selected] - lo) / span
    shifts = displacement[selected] / span
    anchors = np.vstack([points, (np.asarray(axis) - lo) / span, _edge_anchors()])
    shifts = np.vstack([shifts, np.zeros((17, 2))])
    if len(np.unique(anchors, axis=0)) != len(anchors):
        raise NumericalInfeasibility("TPS control points are not unique")
    rbf = RBFInterpolator(anchors, shifts, kernel="thin_plate_spline", smoothing=0.0001, degree=1)
    r_norm = (R - R[0]) / span[0]
    z_norm = (Z - Z[0]) / span[1]
    rr, zz = np.meshgrid(r_norm, z_norm)
    fixed = np.column_stack([rr.ravel(), zz.ravel()])
    warped = fixed + rbf(fixed)
    wr, wz = warped[:, 0].reshape(rr.shape), warped[:, 1].reshape(zz.shape)
    dwr_dz, dwr_dr = np.gradient(wr, z_norm, r_norm)
    dwz_dz, dwz_dr = np.gradient(wz, z_norm, r_norm)
    jac = dwr_dr * dwz_dz - dwr_dz * dwz_dr
    min_jac = float(np.min(jac))
    if not np.isfinite(jac).all() or min_jac <= 0.2:
        raise NumericalInfeasibility(f"warp Jacobian minimum {min_jac} <= 0.2")
    inverse = fixed.copy()
    tol = 0.00001 * min(np.diff(R).min(), np.diff(Z).min()) / min(span)
    converged = False
    for _iteration in range(20):
        updated = fixed - rbf(inverse)
        inverse = updated
        if float(np.max(np.linalg.norm(inverse + rbf(inverse) - fixed, axis=1))) <= tol:
            converged = True
            break
    if not converged:
        raise NumericalInfeasibility("inverse TPS warp did not converge in 20 iterations")
    if np.any(inverse < 0) or np.any(inverse > 1):
        raise NumericalInfeasibility("inverse TPS warp leaves the source grid")
    physical = lo + inverse * span
    interpolation = RegularGridInterpolator((Z, R), psi, method="linear", bounds_error=True)
    sampled = interpolation(np.column_stack([physical[:, 1], physical[:, 0]]))
    output = sampled.reshape(psi.shape).astype(np.float32)
    if not np.isfinite(output).all():
        raise NumericalInfeasibility("spatial warp produced nonfinite flux")
    return output, {
        "cyclic_shift": float(shift),
        "max_displacement_m": max_move,
        "min_jacobian": min_jac,
        "inverse_iterations": float(_iteration + 1),
    }


def _remap_map(psi: np.ndarray, sign: int, axis: float, sep: float, a: float) -> np.ndarray:
    if a == 0:
        return psi.copy()
    phi = sign * psi.astype(np.float64)
    span = axis - sep
    if not np.isfinite(span) or span <= 0:
        raise NumericalInfeasibility(f"axis-to-separatrix span {span} must be positive")
    u = (axis - phi) / span
    interior = (u >= 0) & (u <= 1)
    g = u.copy()
    g[interior] += a * u[interior] ** 2 * (1 - u[interior]) ** 2
    result = (sign * (axis - span * g)).astype(np.float32)
    if not np.isfinite(result).all():
        raise NumericalInfeasibility(f"a={a} produced nonfinite flux")
    return result


def _li_remap(
    psi: np.ndarray,
    truth: np.ndarray,
    c_pred: np.ndarray,
    c_true: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    mask: np.ndarray,
    sign: int,
    derive: Any,
    lcfs: Any,
    contour: Any,
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    axis_r, axis_z, iz, ir = derive.magnetic_axis(psi, R, Z, "DIII-D", mask, sign)
    phi = sign * psi.astype(np.float64)
    axis = float(phi[iz, ir])
    levels = derive._bilinear(phi, R, Z, c_pred[:, 0], c_pred[:, 1])
    sep = float(np.median(levels))
    span = axis - sep
    if not np.isfinite(levels).all() or span <= 0:
        raise NumericalInfeasibility("predicted contour has nonfinite level or nonpositive span")
    level_spread = np.abs(levels - sep) / span
    truth_li = float(derive.internal_inductance(truth, R, Z, "DIII-D", c_true))
    if not np.isfinite(truth_li):
        raise NumericalInfeasibility("true scorer-derived li is nonfinite")
    spacing = min(np.diff(R).min(), np.diff(Z).min())
    candidates: list[tuple[float, float, np.ndarray, float]] = []
    for a in params["candidate_grid"]:
        # max |2u(1-u)(1-2u)| = sqrt(3)/9, so |a|<=2 keeps derivative positive.
        if 1 - abs(a) * np.sqrt(3) / 9 <= 0.615:
            raise NumericalInfeasibility(f"a={a} violates strict monotonicity")
        corrected = _remap_map(psi, sign, axis, sep, a)
        if a == 0 and corrected.tobytes() != psi.tobytes():
            raise NumericalInfeasibility("a=0 map differs from the source bits")
        r2, z2, *_ = derive.magnetic_axis(corrected, R, Z, "DIII-D", mask, sign)
        if np.hypot(r2 - axis_r, z2 - axis_z) > 0.01 * spacing:
            raise NumericalInfeasibility(f"a={a} moved the magnetic axis")
        c2, _ = lcfs.extract_lcfs_with_sign(
            corrected, R, Z, "DIII-D", mask, mask.astype(float), axis_sign=sign
        )
        if c2 is None or contour.symmetric_hausdorff(c2, c_pred) > 0.1 * spacing:
            raise NumericalInfeasibility(f"a={a} changed the predicted LCFS")
        li = float(derive.internal_inductance(corrected, R, Z, "DIII-D", c_pred))
        if not np.isfinite(li):
            raise NumericalInfeasibility(f"a={a} produced nonfinite li")
        candidates.append((abs(li - truth_li), a, corrected, li))
    chosen = min(candidates, key=lambda item: (item[0], abs(item[1]), item[1]))
    li_values = [item[3] for item in candidates]
    return chosen[2], {
        "chosen_a": chosen[1],
        "true_li": truth_li,
        "corrected_li": chosen[3],
        "contour_level_max_spread_over_span": float(np.max(level_spread)),
        "contour_level_p95_spread_over_span": float(np.quantile(level_spread, 0.95)),
        "true_li_outside_candidate_range": float(
            truth_li < min(li_values) or truth_li > max(li_values)
        ),
    }


def build(
    spec_path: Path, arm: str, output_dir: Path, extension_path: Path | None = None
) -> dict[str, Any]:
    if arm not in ("control", "A", "B"):
        raise ValueError(f"unknown arm {arm}")
    started = time.perf_counter()
    spec = json.loads(spec_path.read_text())
    source = Path(spec["input"]["prediction"])
    if file_hash(source) != spec["input"]["prediction_sha256"]:
        raise ValueError(f"{source}: prediction SHA differs from frozen spec")
    files_json = Path(spec["input"]["files_json"])
    if file_hash(files_json) != spec["input"]["files_json_sha256"]:
        raise ValueError(f"{files_json}: scored-file source SHA differs from frozen spec")
    all_files = json.loads(files_json.read_text())["files"]
    if extension_path is None:
        scope = "first3"
        names = spec["smoke"]["files"]
        frame_counts = spec["smoke"]["frames_per_shot"]
        expected_frames = spec["smoke"]["total_frames"]
        if all_files[:3] != names or len(names) != 3:
            raise ValueError(f"{files_json}: first three filenames differ from frozen smoke")
        extension_hash = None
    else:
        scope = "full70"
        extension = json.loads(extension_path.read_text())
        extension_hash = file_hash(extension_path)
        if arm not in ("control", "B"):
            raise ValueError("full70 extension permits only the unchanged control and B")
        if extension["base_spec_sha256"] != file_hash(spec_path):
            raise ValueError(f"{extension_path}: base spec changed")
        if extension["producer_sha256"] != file_hash(Path(__file__)):
            raise ValueError(f"{extension_path}: producer source changed")
        if extension["prediction_sha256"] != spec["input"]["prediction_sha256"]:
            raise ValueError(f"{extension_path}: prediction identity changed")
        if extension["files_json_sha256"] != file_hash(files_json):
            raise ValueError(f"{extension_path}: filename source changed")
        source_names_hash = hashlib.sha256("\n".join(all_files).encode()).hexdigest()
        if (
            source_names_hash != extension["ordered_filenames_sha256"]
            or extension["files"] != all_files
        ):
            raise ValueError(f"{extension_path}: ordered filenames changed")
        b_hash = hashlib.sha256(
            json.dumps(spec["B_interior_li_remap"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if b_hash != extension["B_config_sha256"]:
            raise ValueError(f"{extension_path}: B algorithm parameters changed")
        names = all_files
        frame_counts = extension["frames_per_shot"]
        expected_frames = extension["total_frames"]
        if len(names) != extension["shots"] or len(frame_counts) != len(names):
            raise ValueError(f"{extension_path}: shot/frame list length changed")
    local_score, derive, lcfs, contour, common = _frozen_modules(spec)
    mask_path = Path(spec["scorer_provenance"]["root"]) / "fusion_scoring/masks/d3d_envelope.npz"
    with np.load(mask_path) as masks:
        mask = masks["mask_coarse"].astype(bool)
        mask_R = masks["grid_R"]
        mask_Z = masks["grid_Z"]
    arrays: dict[str, np.ndarray] = {}
    diagnostics: list[dict[str, Any]] = []
    sign = int(common.AXIS_SIGN["DIII-D"])
    with np.load(source, allow_pickle=False) as packed:
        for i, name in enumerate(names):
            path = Path(spec["input"]["data_dir"]) / "data" / spec["input"]["data_config"] / name
            row = pd.read_parquet(path).iloc[0]
            truth_stack = local_score._as_psi_stack(row["efit_psirz"])
            R = np.asarray(row["efit_grid_R"], dtype=np.float64)
            Z = np.asarray(row["efit_grid_Z"], dtype=np.float64)
            if not np.array_equal(R, mask_R) or not np.array_equal(Z, mask_Z):
                raise ValueError(f"{name}: shot grid differs from scorer's pinned mask grid")
            n = frame_counts[i]
            psi_stack = packed[f"shot_{i:04d}_psirz"]
            if psi_stack.shape != truth_stack.shape or psi_stack.shape != (n, 65, 65):
                raise ValueError(
                    f"{name}: source/truth shapes {psi_stack.shape}/{truth_stack.shape}, "
                    f"expected {(n, 65, 65)}"
                )
            output_maps = []
            for frame, (psi, truth) in enumerate(zip(psi_stack, truth_stack, strict=True)):
                try:
                    if psi.dtype != np.float32 or not np.isfinite(psi).all():
                        raise NumericalInfeasibility("source flux is not finite float32")
                    if arm == "control":
                        output_maps.append(psi.copy())
                        continue
                    c_pred, _ = lcfs.extract_lcfs_with_sign(
                        psi, R, Z, "DIII-D", mask, mask.astype(float), axis_sign=sign
                    )
                    c_true = lcfs.extract_lcfs(truth, R, Z, "DIII-D", mask, mask.astype(float))
                    if c_pred is None or c_true is None:
                        raise NumericalInfeasibility("source or true scorer LCFS is missing")
                    if arm == "A":
                        axis_r, axis_z, *_ = derive.magnetic_axis(psi, R, Z, "DIII-D", mask, sign)
                        corrected, detail = _warp_map(psi, c_pred, c_true, (axis_r, axis_z), R, Z)
                    else:
                        corrected, detail = _li_remap(
                            psi,
                            truth,
                            c_pred,
                            c_true,
                            R,
                            Z,
                            mask,
                            sign,
                            derive,
                            lcfs,
                            contour,
                            spec["B_interior_li_remap"],
                        )
                    if corrected.shape != psi.shape or not np.isfinite(corrected).all():
                        raise NumericalInfeasibility("corrected map shape/finiteness changed")
                    output_maps.append(corrected)
                    diagnostics.append({"shot": name, "frame": frame, **detail})
                except NumericalInfeasibility as exc:
                    raise NumericalInfeasibility(f"{arm}: {name} frame {frame}: {exc}") from exc
            arrays[f"shot_{i:04d}_psirz"] = np.stack(output_maps)
            for field in ("q95", "betaN"):
                key = f"shot_{i:04d}_{field}"
                value = packed[key]
                if value.shape != (n,) or value.dtype != np.float32:
                    raise ValueError(f"{name}: {key} has shape/dtype {value.shape}/{value.dtype}")
                arrays[key] = value.copy()
            if scope == "full70" and (i + 1) % 5 == 0:
                print(f"{arm} full70: {i + 1}/{len(names)} shots complete", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if sum(frame_counts) != expected_frames:
        raise ValueError(f"{scope}: {sum(frame_counts)} frames, expected {expected_frames}")
    out = output_dir / f"{arm}-{scope}.npz"
    _save(out, arrays)
    with np.load(source, allow_pickle=False) as before, np.load(out, allow_pickle=False) as after:
        for i in range(len(names)):
            for field in ("q95", "betaN"):
                key = f"shot_{i:04d}_{field}"
                if after[key].tobytes() != before[key].tobytes():
                    raise AssertionError(f"{out}: {key} scalar bits changed")
            if arm == "control":
                key = f"shot_{i:04d}_psirz"
                if after[key].tobytes() != before[key].tobytes():
                    raise AssertionError(f"{out}: control map {key} changed")
    result: dict[str, Any] = {
        "classification": "valid_map_construction",
        "arm": arm,
        "scope": scope,
        "spec": str(spec_path),
        "spec_sha256": file_hash(spec_path),
        "extension_spec_sha256": extension_hash,
        "producer_sha256": file_hash(Path(__file__)),
        "source_prediction_sha256": spec["input"]["prediction_sha256"],
        "files": names,
        "frames": expected_frames,
        "prediction_npz": str(out),
        "prediction_sha256": file_hash(out),
        "scalar_bits_unchanged": True,
        "control_map_bits_unchanged": arm == "control",
        "per_frame_diagnostics": diagnostics,
        "build_seconds": time.perf_counter() - started,
    }
    write_json(out.with_suffix(".json"), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--arm", choices=("control", "A", "B"), required=True)
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
                "classification": "numerical_infeasibility_of_fixed_construction",
                "arm": args.arm,
                "scope": scope,
                "spec_sha256": file_hash(args.spec),
                "extension_spec_sha256": file_hash(args.extension) if args.extension else None,
                "error": str(exc),
                "wall_seconds": time.perf_counter() - started,
                "no_scientific_negative_inferred": True,
            },
        )
        print(f"{args.arm} numerical feasibility gate failed: {exc}")
        return 2
    print(
        f"{args.arm}: {len(result['files'])} shots, {result['frames']} frames, "
        f"build {result['build_seconds']:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

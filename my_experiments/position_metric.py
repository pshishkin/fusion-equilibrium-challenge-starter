"""One shared target metric and optional position-conditioned MLP training factors.

Both B11 screen arms probe the same training-only true maps and use the same global target
coordinates. The candidate rotates only the training error within those coordinates. Shot-frame
group IDs are transient fit metadata; an artifact stores the small forms and probe provenance.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating[Any]]
GroupArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class PositionPlan:
    """The shared target factor, optional training factors, and compact artifact diagnostics."""

    global_factor: FloatArray
    loss_factors: FloatArray
    train_groups: GroupArray
    diagnostics: dict[str, Any]


def normalize_config(raw: Any, context: str) -> dict[str, Any]:
    """Parse every numerical choice from YAML; absence means the historical global loss."""
    if not isinstance(raw, dict):
        raise ValueError(f"{context}: expected a mapping, got {type(raw).__name__}")
    expected = {"mode", "strength", "probe_shots", "minimum_kept_frames",
                "minimum_complete_triplets", "anchors", "boundaries"}
    if set(raw) != expected:
        raise ValueError(f"{context}: keys {sorted(raw)}, expected exactly {sorted(expected)}")
    mode = raw["mode"]
    if mode not in ("pooled", "conditioned"):
        raise ValueError(f"{context}.mode={mode!r}, expected 'pooled' or 'conditioned'")
    numeric = ("strength",)
    for key in numeric:
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{context}.{key}={value!r}, expected finite number")
    strength = float(raw["strength"])
    if not 0.0 < strength < 1.0:
        raise ValueError(f"{context}.strength={strength}, expected strictly between 0 and 1")
    counts = ("probe_shots", "minimum_kept_frames", "minimum_complete_triplets")
    for key in counts:
        value = raw[key]
        if type(value) is not int or value < 1:
            raise ValueError(f"{context}.{key}={value!r}, expected positive integer")
    if raw["minimum_kept_frames"] < 3 or not (
        1 <= raw["minimum_complete_triplets"] <= raw["probe_shots"]
    ):
        raise ValueError(f"{context}: minimum frames/triplets {raw['minimum_kept_frames']}/"
                         f"{raw['minimum_complete_triplets']} incompatible with "
                         f"{raw['probe_shots']} probe shots")
    anchors = np.asarray(raw["anchors"], dtype=np.float64)
    boundaries = np.asarray(raw["boundaries"], dtype=np.float64)
    if (anchors.shape != (3,) or boundaries.shape != (2,)
            or not np.isfinite(anchors).all() or not np.isfinite(boundaries).all()
            or not np.all((anchors > 0) & (anchors < 1))
            or not np.all(np.diff(anchors) > 0)
            or not np.allclose(boundaries, (anchors[:-1] + anchors[1:]) / 2,
                                rtol=0, atol=1e-12)):
        raise ValueError(f"{context}: anchors {anchors.tolist()} and boundaries "
                         f"{boundaries.tolist()} must be three sorted interior anchors and "
                         "their exact two midpoints")
    return {"mode": mode, "strength": strength,
            "probe_shots": int(raw["probe_shots"]),
            "minimum_kept_frames": int(raw["minimum_kept_frames"]),
            "minimum_complete_triplets": int(raw["minimum_complete_triplets"]),
            "anchors": anchors.tolist(), "boundaries": boundaries.tolist()}


def validate_preflight(params: Any, fit_mode: str) -> None:
    """Reject unsupported variants before corpus reading or CUDA allocation."""
    config = params.position_metric
    if config is None:
        return
    from my_experiments.models import TorchMLPModel

    if fit_mode != "validation":
        raise ValueError(f"{params.path}: B11 position metric needs validation mode, got "
                         f"{fit_mode!r}")
    if set(params.models) != {"mlp"} or not isinstance(params.models["mlp"], TorchMLPModel):
        raise ValueError(f"{params.path}: B11 requires only the torch MLP named 'mlp', got "
                         f"{[(name, model.kind) for name, model in params.models.items()]}")
    model = params.models["mlp"]
    if model.loss != "mse" or model.frame_ends_strength != 0 or model.stop_on != "loss":
        raise ValueError(f"{params.path}: B11 requires MLP loss=mse, frame_ends_strength=0, "
                         f"stop_on=loss; got {model.loss!r}/{model.frame_ends_strength}/"
                         f"{model.stop_on!r}")
    if model.head_weight <= 0 or model.head_frozen or model.head_cons_weights == "adaptive":
        raise ValueError(f"{params.path}: B11 requires the trainable static functional head; "
                         f"head_weight={model.head_weight}, head_frozen={model.head_frozen}, "
                         f"head_cons_weights={model.head_cons_weights!r}")
    unsupported = {
        "loss.metric": params.loss_metric != "jacobian",
        "loss.calibrate_scalars": params.calibrate_scalars,
        "loss.li_form": params.li_form != "jacobian",
        "loss.boundary": params.boundary,
        "loss.field_weight": params.field_weight != 0,
        "loss.env_weight": params.env_weight != 0,
        "loss.aux_scalars": params.aux_scalars != "none",
    }
    bad = [key for key, present in unsupported.items() if present]
    if bad:
        raise ValueError(f"{params.path}: B11 pooled/conditioned pair does not support "
                         f"{bad}; keep the original global path for these variants")


def frame_groups(shots: npt.NDArray[np.int64], boundaries: tuple[float, float]) -> GroupArray:
    """Nearest-anchor group for every kept frame, with a singleton at position 0.5."""
    if shots.ndim != 1 or len(shots) == 0 or np.any(shots < 1):
        raise ValueError(f"shot frame counts {shots.tolist()}, expected nonempty positive vector")
    pieces = []
    for n in shots:
        count = int(n)
        positions = (np.asarray([0.5]) if count == 1
                     else np.arange(count, dtype=np.float64) / (count - 1))
        pieces.append(np.searchsorted(boundaries, positions, side="right").astype(np.int64))
    return np.concatenate(pieces)


def _selector(anchors: np.ndarray, minimum: int, frame_share: float,
              expected_kept: int, filename: str) -> tuple[
    Callable[[int], np.ndarray], list[int], list[np.ndarray], list[np.ndarray]
]:
    """Map three positions in the kept frame sequence back to the raw cache's full indices."""
    from my_experiments.baseline_model import thin_frames

    full_counts: list[int] = []
    kept_indices: list[np.ndarray] = []
    full_indices: list[np.ndarray] = []

    def keep(n_frames: int) -> np.ndarray:
        full_counts.append(n_frames)
        thinned = thin_frames(n_frames, frame_share)
        if len(thinned) != expected_kept:
            raise ValueError(f"{filename}: base cache selects {len(thinned)} frames from "
                             f"{n_frames} full frames at share {frame_share}, but the loaded "
                             f"training shot has {expected_kept}")
        if len(thinned) < minimum:
            kept_indices.append(np.empty(0, dtype=np.intp))
            full_indices.append(np.empty(0, dtype=np.intp))
            return full_indices[-1]
        chosen = np.rint(anchors * (len(thinned) - 1)).astype(np.intp)
        if len(np.unique(chosen)) != 3:
            raise ValueError(f"{filename}: {len(thinned)} kept frames yield duplicate probe "
                             f"indices {chosen}")
        kept_indices.append(chosen)
        full_indices.append(thinned[chosen])
        return full_indices[-1]

    return keep, full_counts, kept_indices, full_indices


def _read_probe_task(task: tuple[Path, int, float, np.ndarray, int, str]) -> tuple[
    np.ndarray, dict[str, Any]
]:
    """Spawn-safe read of three mapped kept-frame probes from one validated base-cache shot."""
    from my_experiments import shot_cache

    path, expected_kept, frame_share, anchors, minimum, key = task
    keep, counts, kept_indices, full_indices = _selector(
        anchors, minimum, frame_share, expected_kept, path.name
    )
    cached = shot_cache.load(path, key, keep)
    if cached is None:
        raise ValueError(f"{path}: missing/stale base cache key {key}; B11 never builds it "
                         "implicitly")
    if len(counts) != 1 or len(kept_indices) != 1 or len(full_indices) != 1:
        raise ValueError(f"{path}: cache selector reported {len(counts)} full counts, "
                         f"{len(kept_indices)} kept selections and {len(full_indices)} "
                         "full selections, each expected 1")
    psi = np.asarray(cached[1], dtype=np.float64)
    if psi.shape != (3, 65, 65) or not np.isfinite(psi).all():
        raise ValueError(f"{path}: three cached flux maps have shape {psi.shape}, "
                         f"nonfinite={int((~np.isfinite(psi)).sum())}; expected (3,65,65)")
    selected = {"file": path.name, "full_frames": counts[0],
                "kept_frames": expected_kept,
                "kept_indices": kept_indices[0].tolist(),
                "full_indices": full_indices[0].tolist()}
    return psi, selected


def _probe_maps(files: list[Path], shots: npt.NDArray[np.int64], frame_share: float,
                config: dict[str, Any], jobs: int) -> tuple[
    FloatArray, list[dict[str, Any]], list[dict[str, Any]], str
]:
    """Select on loaded shot lengths, then read probes in order through the shared process pool."""
    from my_experiments.baseline_model import _shot_code_key
    from toolkit.parallel import pimap, resolve_jobs

    key = _shot_code_key()
    anchors = np.asarray(config["anchors"], dtype=np.float64)
    eligible = np.flatnonzero(shots >= config["minimum_kept_frames"])
    wanted = int(config["probe_shots"])
    if len(eligible) < wanted:
        raise ValueError(f"{len(eligible)} eligible kept training shots for B11, expected "
                         f"{wanted} from {len(files)} files")
    indices = eligible[:wanted]
    last = int(indices[-1])
    ineligible = [{"file": files[i].name, "kept_frames": int(shots[i])}
                  for i in range(last + 1) if shots[i] < config["minimum_kept_frames"]]
    tasks = [(files[int(i)], int(shots[i]), frame_share, anchors,
              int(config["minimum_kept_frames"]), key) for i in indices]
    selected: list[dict[str, Any]] = []
    maps: list[np.ndarray] = []
    for psi, item in pimap(_read_probe_task, tasks, resolve_jobs(jobs, len(tasks))):
        selected.append(item)
        maps.append(psi)
    return np.stack(maps), selected, ineligible, key


def _consistency_form(jac: np.ndarray, var: np.ndarray, split: np.ndarray,
                      weight: float) -> np.ndarray:
    """The scorer-weighted Jacobian Gram form under one pooled scalar denominator."""
    if not (var > 0).all():
        raise ValueError(f"nonpositive pooled scalar variance {var.tolist()}")
    return np.einsum("nsk,nsl,s->kl", jac, jac, weight * split / var,
                     optimize=True) / len(jac)


def _assemble_metrics(raw: np.ndarray, fractions: np.ndarray, strength: float,
                      psi_ss_tot: float, cons_gain: float) -> dict[str, np.ndarray | float]:
    """Equalise trace, preserve the weighted global form, then rotate losses in its coordinates."""
    from my_experiments.target_metric import factor, metric_form

    if raw.ndim != 3 or raw.shape[0] != 3 or raw.shape[1] != raw.shape[2]:
        raise ValueError(f"B11: raw consistency forms shape {raw.shape}, expected (3,n,n)")
    if fractions.shape != (3,) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError(f"B11: group fractions {fractions}, expected three summing to 1")
    raw_traces = np.trace(raw, axis1=1, axis2=2)
    if not np.isfinite(raw_traces).all() or not (raw_traces > 0).all():
        raise ValueError(f"B11: raw group traces {raw_traces.tolist()}, expected finite positive")
    common_trace = float(fractions @ raw_traces)
    equalized = raw * (common_trace / raw_traces)[:, None, None]
    average = np.einsum("g,gij->ij", fractions, equalized, optimize=True)
    global_metric = metric_form(average, psi_ss_tot, cons_gain=cons_gain)
    global_factor = factor(global_metric)
    group_metrics = np.stack([
        (1 - strength) * global_metric
        + strength * metric_form(equalized[group], psi_ss_tot, cons_gain=cons_gain)
        for group in range(3)
    ])
    weighted_metric = np.einsum("g,gij->ij", fractions, group_metrics, optimize=True)
    if not np.allclose(weighted_metric, global_metric, rtol=1e-10, atol=1e-10):
        raise ValueError("B11: weighted group metric differs from shared global metric")
    inverse = np.linalg.inv(global_factor)
    scaled_metrics = np.stack([inverse @ matrix @ inverse.T for matrix in group_metrics])
    loss_factors = np.stack([factor(matrix) for matrix in scaled_metrics])
    if not np.allclose(np.einsum("g,gij->ij", fractions, scaled_metrics),
                       np.eye(len(global_metric)), rtol=1e-9, atol=1e-9):
        raise ValueError("B11: weighted group metric in target-scaled coordinates is not identity")
    return {
        "raw_traces": raw_traces,
        "common_trace": common_trace,
        "equalized_consistency_forms": equalized,
        "pooled_consistency_form": average,
        "global_metric": global_metric,
        "group_metrics": group_metrics,
        "global_factor": global_factor,
        "scaled_group_metrics": scaled_metrics,
        "scaled_loss_factors": loss_factors,
    }


def build(files: list[Path], shots: npt.NDArray[np.int64], pca: Any,
          params: Any, coil_plan: dict[str, Any], psi_ss_tot: float,
          frame_share: float, jobs: int) -> PositionPlan:
    """Build matched global and group forms from one fixed training-only probe sample."""
    from my_experiments.target_metric import (
        CONS_SCALARS,
        N_CONS,
        W_CONS,
        _jacobian_task,
        scorer_context,
    )
    from toolkit.parallel import pimap, resolve_jobs

    config = params.position_metric
    if config is None:
        raise ValueError("B11 build requires an explicit loss.position_metric configuration")
    if len(shots) != len(files):
        raise ValueError(f"B11: {len(shots)} training shot counts against {len(files)} files")
    groups = frame_groups(shots, tuple(config["boundaries"]))
    counts = np.bincount(groups, minlength=3)
    fractions = counts / counts.sum()
    if len(groups) != int(shots.sum()) or np.any(counts == 0):
        raise ValueError(f"B11: {len(groups)} group IDs against {shots.sum()} frames; "
                         f"group counts {counts.tolist()}")
    psi, selected, ineligible, cache_key = _probe_maps(files, shots, frame_share,
                                                       config, jobs)
    components = np.asarray(pca.pca.components_).reshape(params.n_pca, 65, 65)
    ctx = scorer_context(coil_plan["grid_R"], coil_plan["grid_Z"], coil_plan["machine"])
    flat = psi.reshape(-1, 65, 65)
    workers = resolve_jobs(jobs, len(flat))
    chunks = np.array_split(np.arange(len(flat)), workers * 2)
    delta = params.jacobian_delta * float(np.sqrt(psi_ss_tot / params.n_pca))
    tasks = [(flat[chunk], components, delta, ctx, i)
             for i, chunk in enumerate(chunks) if len(chunk)]
    base_parts = []
    jac_parts = []
    for base, jac, _ in pimap(_jacobian_task, tasks, workers):
        base_parts.append(base)
        jac_parts.append(jac)
    base = np.concatenate(base_parts).reshape(len(selected), 3, N_CONS)
    jac = np.concatenate(jac_parts).reshape(len(selected), 3, N_CONS, params.n_pca)
    finite = np.isfinite(base).all(axis=2) & np.isfinite(jac).all(axis=(2, 3))
    complete = finite.all(axis=1)
    if complete.sum() < config["minimum_complete_triplets"]:
        bad = [selected[i]["file"] for i in np.flatnonzero(~complete)]
        raise ValueError(f"B11: {int(complete.sum())}/{len(selected)} complete finite triplets, "
                         f"need {config['minimum_complete_triplets']}; invalid files {bad[:5]}")
    base_good, jac_good = base[complete], jac[complete]
    variance = base_good.reshape(-1, N_CONS).var(axis=0, ddof=0)
    if not (variance > 0).all():
        raise ValueError(f"B11: nonpositive pooled variance for "
                         f"{[CONS_SCALARS[i] for i in np.flatnonzero(variance <= 0)]}")
    split = np.asarray(params.cons_weights if params.cons_weights is not None
                       else np.ones(N_CONS) / N_CONS, dtype=np.float64)
    if split.shape != (N_CONS,) or not np.isclose(split.sum(), 1.0):
        raise ValueError(f"B11: Consistency split {split}, expected seven weights summing to 1")
    raw = np.stack([_consistency_form(jac_good[:, group], variance, split, W_CONS)
                    for group in range(3)])
    assembled = _assemble_metrics(raw, fractions, float(config["strength"]), psi_ss_tot,
                                  params.cons_gain)
    global_factor = np.asarray(assembled["global_factor"])
    loss_factors = np.asarray(assembled["scaled_loss_factors"])
    diagnostics = {
        "config": dict(config),
        "frame_share": frame_share,
        "probe_delta_wb_per_rad": delta,
        "selected_probes": selected,
        "ineligible_before_selection": ineligible,
        "invalid_triplets": [
            {"file": item["file"], "valid_positions": finite[i].tolist()}
            for i, item in enumerate(selected) if not complete[i]
        ],
        "complete_triplets": int(complete.sum()),
        "base_cache_code_key": cache_key,
        "pooled_scalar_variance": variance,
        "consistency_scalar_order": tuple(CONS_SCALARS),
        "consistency_split": split,
        "train_group_counts": counts,
        "train_group_fractions": fractions,
        "raw_consistency_forms": raw,
        **assembled,
    }
    return PositionPlan(global_factor, loss_factors, groups, diagnostics)

"""B11 controlled replication with a common scalar denominator across shot positions.

This audits a mechanism in the existing Jacobian loss form. It never trains a model, computes a
competition score, or changes the fitted artifact or the decoded-shot cache.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from tqdm import tqdm

from experiments import DEFAULT_LOCAL_DATA_DIR, EFIT_GRID_SIZE
from my_experiments import shot_cache
from my_experiments.baseline_model import _shot_code_key
from my_experiments.data_check import check_dataset
from my_experiments.models import load_params
from my_experiments.target_metric import (
    CONS_SCALARS,
    N_CONS,
    W_CONS,
    _jacobian_task,
    scorer_context,
)
from toolkit.parallel import pimap, release
from toolkit.progress import SHOT_EVERY, bar_kwargs
from toolkit.records import file_hash, write_json

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "experiments/b11_shared_variance_prereg.json"
RESULT_PATH = ROOT / "results/b11_shared_variance_20260922.json"
GROUP_PAIRS = ((0, 1), (0, 2), (1, 2))


def _indices(n_frames: int, fractions: np.ndarray, minimum: int) -> np.ndarray:
    """Three positions in a shot's full EFIT index range, or none if it is ineligible."""
    if n_frames < minimum:
        return np.empty(0, dtype=np.intp)
    indices = np.rint(fractions * (n_frames - 1)).astype(np.intp)
    if len(np.unique(indices)) != len(fractions):
        raise ValueError(f"{n_frames} full frames yield duplicate position indices {indices}")
    return indices


def _keep_selector(fractions: np.ndarray, minimum: int) -> tuple[
    Callable[[int], np.ndarray], list[int]
]:
    """Make a per-shot cache selector that records its full-frame count once."""
    lengths: list[int] = []

    def keep(n_frames: int) -> np.ndarray:
        lengths.append(n_frames)
        return _indices(n_frames, fractions, minimum)

    return keep, lengths


def _selected_probes(artifact: dict[str, Any], spec: dict[str, Any]) -> tuple[
    np.ndarray, list[dict[str, Any]], list[dict[str, Any]]
]:
    """Read three maps per eligible training shot through the validated base cache only."""
    selection = spec["selection"]
    fractions = np.asarray(selection["index_fractions"], dtype=np.float64)
    if fractions.shape != (3,) or not np.all((fractions > 0) & (fractions < 1)):
        raise ValueError(f"{SPEC_PATH}: index fractions {fractions} must be three values in (0,1)")
    minimum = int(selection["eligible_full_frames_minimum"])
    wanted = int(selection["eligible_shots"])
    cache_key = _shot_code_key()
    if cache_key != spec["base_cache_code_key"]:
        raise ValueError(f"base shot-cache code key {cache_key}, preregistered "
                         f"{spec['base_cache_code_key']}; do not rebuild from a changed producer")
    data_dir = DEFAULT_LOCAL_DATA_DIR / "data" / artifact["config"]
    manifest = json.loads((ROOT / "my_experiments/diii_d_dataset.json").read_text())
    entries = manifest["configs"][artifact["config"]]
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    for name in artifact["train_files"]:
        path = data_dir / name
        if name not in entries:
            raise ValueError(f"{path}: absent from pinned dataset manifest")
        size, expected_sha256 = entries[name]
        if path.stat().st_size != size:
            raise ValueError(f"{path}: {path.stat().st_size} bytes, expected {size}")
        keep, lengths = _keep_selector(fractions, minimum)
        cached = shot_cache.load(path, cache_key, keep)
        if cached is None:
            raise ValueError(f"{path}: missing or stale base shot cache for key {cache_key}; "
                             "this diagnostic never writes cache entries")
        if len(lengths) != 1:
            raise ValueError(f"{path}: cache selector saw {len(lengths)} full-frame counts")
        n_full = lengths[0]
        indices = _indices(n_full, fractions, minimum)
        if n_full < minimum:
            excluded.append({"file": name, "full_frames": n_full,
                             "reason": f"fewer than {minimum} full frames"})
            continue
        psi = np.asarray(cached[1], dtype=np.float64)
        if psi.shape != (3, EFIT_GRID_SIZE, EFIT_GRID_SIZE):
            raise ValueError(f"{path}: selected flux shape {psi.shape}, expected "
                             f"(3, {EFIT_GRID_SIZE}, {EFIT_GRID_SIZE})")
        if not np.isfinite(psi).all():
            raise ValueError(f"{path}: selected flux contains "
                             f"{int((~np.isfinite(psi)).sum())} nonfinite values")
        actual_sha256 = file_hash(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"{path}: SHA-256 {actual_sha256}, manifest expects {expected_sha256}")
        selected.append({"file": name, "full_frames": n_full, "indices": indices.tolist(),
                         "parquet_sha256": actual_sha256})
        frames.append(psi)
        if len(selected) == wanted:
            break
    if len(selected) != wanted:
        raise ValueError(f"selected {len(selected)} eligible shots, expected {wanted} from "
                         f"{len(artifact['train_files'])} artifact training filenames")
    return np.stack(frames), selected, excluded


def _form(jacobians: np.ndarray, scalar_var: np.ndarray,
          scalar_split: np.ndarray) -> np.ndarray:
    """Mean Jᵀ diag(W_CONS * split / variance) J on a fixed group of finite frames."""
    if jacobians.ndim != 3 or jacobians.shape[1] != N_CONS:
        raise ValueError(f"Jacobian group shape {jacobians.shape}, expected (N,{N_CONS},50)")
    if not (scalar_var > 0).all():
        raise ValueError(f"scalar variances {scalar_var.tolist()} must be positive")
    weights = W_CONS * scalar_split / scalar_var
    return np.einsum("nsk,nsl,s->kl", jacobians, jacobians, weights,
                     optimize=True) / len(jacobians)


def _trace_normalize(forms: list[np.ndarray]) -> tuple[list[np.ndarray], list[float]]:
    """Remove each form's scalar size while preserving its coefficient directions."""
    traces = [float(np.trace(form)) for form in forms]
    if any(not np.isfinite(trace) or trace <= 0 for trace in traces):
        raise ValueError(f"form traces {traces}, expected finite positive")
    return [form / trace for form, trace in zip(forms, traces, strict=True)], traces


def _distances(forms: list[np.ndarray]) -> tuple[list[float], float]:
    """Three symmetric relative Frobenius distances and their predeclared mean."""
    normal, _ = _trace_normalize(forms)
    distances = []
    for first, second in GROUP_PAIRS:
        a, b = normal[first], normal[second]
        denominator = 0.5 * (np.linalg.norm(a) + np.linalg.norm(b))
        if denominator <= 0:
            raise ValueError(f"pair {first}/{second}: nonpositive Frobenius norm")
        distances.append(float(np.linalg.norm(a - b) / denominator))
    return distances, float(np.mean(distances))


def _angles(forms: list[np.ndarray], rank: int) -> list[dict[str, Any]]:
    """Top-rank principal angles, descriptive only and separate from the gate."""
    bases = []
    eigengaps = []
    for form in forms:
        values, vectors = np.linalg.eigh(form)
        order = np.argsort(values)[::-1]
        bases.append(vectors[:, order[:rank]])
        eigengaps.append(float(values[order[rank - 1]] / values[order[rank]]))
    out = []
    for first, second in GROUP_PAIRS:
        singular = np.linalg.svd(bases[first].T @ bases[second], compute_uv=False)
        degrees = np.degrees(np.arccos(np.clip(singular, -1.0, 1.0)))
        out.append({"pair": [first, second], "degrees": degrees.tolist(),
                    "median_degrees": float(np.median(degrees)),
                    "maximum_degrees": float(np.max(degrees)),
                    "rank7_to_rank8_eigenvalue_ratio": [eigengaps[first], eigengaps[second]]})
    return out


def _shared_forms(jac: np.ndarray, variance: np.ndarray,
                  split: np.ndarray) -> list[np.ndarray]:
    """Three group means with the same seven scalar denominators."""
    return [_form(jac[:, group], variance, split) for group in range(3)]


def _legacy_forms(jac: np.ndarray, base: np.ndarray,
                  split: np.ndarray) -> tuple[list[np.ndarray], list[list[float]]]:
    """Old group-specific denominator on identical complete probes, for confound accounting."""
    forms = []
    variances = []
    for group in range(3):
        var = np.var(base[:, group], axis=0, ddof=0)
        if not (var > 0).all():
            raise ValueError(f"legacy group {group} has zero variance in "
                             f"{[CONS_SCALARS[i] for i in np.flatnonzero(var <= 0)]}")
        forms.append(_form(jac[:, group], var, split))
        variances.append(var.tolist())
    return forms, variances


def _permutation_null(jac: np.ndarray, variance: np.ndarray, split: np.ndarray,
                      repetitions: int, seed: int) -> np.ndarray:
    """Within-shot label permutations preserve every triplet and the pooled denominator."""
    rng = np.random.default_rng(seed)
    scores = np.empty(repetitions, dtype=np.float64)
    shot_indices = np.arange(len(jac))[:, None]
    for repetition in range(repetitions):
        labels = np.stack([rng.permutation(3) for _ in range(len(jac))])
        forms = _shared_forms(jac[shot_indices, labels], variance, split)
        scores[repetition] = _distances(forms)[1]
    return scores


def _bootstrap(jac: np.ndarray, variance: np.ndarray, split: np.ndarray,
               repetitions: int, seed: int) -> np.ndarray:
    """Resample entire complete triplets to show shot-sample uncertainty."""
    rng = np.random.default_rng(seed)
    scores = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        draw = rng.integers(0, len(jac), size=len(jac))
        scores[repetition] = _distances(_shared_forms(jac[draw], variance, split))[1]
    return scores


def main() -> None:
    """Select cached truth, probe existing functionals, and test only the preregistered gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    spec = json.loads(SPEC_PATH.read_text())
    if args.jobs != int(spec["jacobian"]["workers"]):
        raise ValueError(f"{SPEC_PATH}: jobs={args.jobs}, expected "
                         f"{spec['jacobian']['workers']}")
    artifact_path = ROOT / spec["artifact"]
    if file_hash(artifact_path) != spec["artifact_sha256"]:
        raise ValueError(f"{artifact_path}: SHA-256 differs from preregistration")
    manifest = ROOT / "my_experiments/diii_d_dataset.json"
    if file_hash(manifest) != spec["dataset_manifest_sha256"]:
        raise ValueError(f"{manifest}: SHA-256 differs from preregistration")
    dataset_identity = check_dataset(DEFAULT_LOCAL_DATA_DIR)
    artifact = joblib.load(artifact_path)
    params_path = artifact_path.with_name("params.yaml")
    params = load_params(params_path)
    effective = yaml.safe_load(artifact["params_yaml"])
    if yaml.safe_load(params_path.read_text()) != effective:
        raise ValueError(f"{artifact_path}: effective params YAML differs from {params_path}")
    if int(artifact["n_pca"]) != 50 or artifact["coil"]["machine"] != "DIII-D":
        raise ValueError(f"{artifact_path}: expected 50 DIII-D PCA directions")
    split = np.asarray(params.cons_weights if params.cons_weights is not None
                       else np.ones(N_CONS) / N_CONS, dtype=np.float64)
    if split.shape != (N_CONS,) or not np.isclose(split.sum(), 1.0):
        raise ValueError(f"{params_path}: Consistency weights {split}, expected seven summing 1")
    psi, selected, excluded = _selected_probes(artifact, spec)
    components = np.asarray(artifact["pca"].pca.components_).reshape(
        int(artifact["n_pca"]), EFIT_GRID_SIZE, EFIT_GRID_SIZE
    )
    ctx = scorer_context(artifact["coil"]["grid_R"], artifact["coil"]["grid_Z"],
                         artifact["coil"]["machine"])
    delta = float(spec["jacobian"]["probe_delta_wb_per_rad"])
    flat_psi = psi.reshape(-1, EFIT_GRID_SIZE, EFIT_GRID_SIZE)
    chunks = np.array_split(np.arange(len(flat_psi)), int(spec["jacobian"]["ordered_chunks"]))
    tasks = [(flat_psi[chunk], components, delta, ctx, i)
             for i, chunk in enumerate(chunks)]
    base_parts = []
    jac_parts = []
    try:
        for base, jac, _ in tqdm(pimap(_jacobian_task, tasks, args.jobs), total=len(tasks),
                                 desc="B11 Jacobian chunks", unit="chunk",
                                 **bar_kwargs(SHOT_EVERY)):
            base_parts.append(base)
            jac_parts.append(jac)
    finally:
        release()
    base = np.concatenate(base_parts).reshape(len(selected), 3, N_CONS)
    jac = np.concatenate(jac_parts).reshape(len(selected), 3, N_CONS, len(components))
    finite = np.isfinite(base).all(axis=2) & np.isfinite(jac).all(axis=(2, 3))
    complete = finite.all(axis=1)
    invalid = [
        {"file": item["file"], "indices": item["indices"],
         "valid_positions": finite[i].tolist()}
        for i, item in enumerate(selected) if not complete[i]
    ]
    record: dict[str, Any] = {
        "kind": "B11 shared-denominator direction diagnostic; no fit or competition score",
        "preregistration": str(SPEC_PATH.relative_to(ROOT)),
        "preregistration_sha256": file_hash(SPEC_PATH),
        "script_sha256": file_hash(Path(__file__)),
        "target_metric_sha256": file_hash(ROOT / "my_experiments/target_metric.py"),
        "shot_cache_sha256": file_hash(ROOT / "my_experiments/shot_cache.py"),
        "artifact_sha256": file_hash(artifact_path),
        "dataset_identity": dataset_identity,
        "selected": selected,
        "ineligible_before_selection": excluded,
        "invalid_jacobian_triplets": invalid,
        "finite_position_counts": finite.sum(axis=0).tolist(),
        "complete_shot_count": int(complete.sum()),
        "probe_delta_wb_per_rad": delta,
        "scalar_order": list(CONS_SCALARS),
        "normalized_consistency_split": split.tolist(),
        "group_labels": spec["selection"]["group_labels"],
    }
    if int(complete.sum()) < int(spec["jacobian"]["required_complete_finite_triplets"]):
        record["status"] = "inconclusive_insufficient_complete_triplets"
        write_json(RESULT_PATH, record)
        print(f"B11 inconclusive: {int(complete.sum())} complete shots of {len(selected)}")
        return
    base_complete = base[complete]
    jac_complete = jac[complete]
    variance = np.var(base_complete.reshape(-1, N_CONS), axis=0, ddof=0)
    if not (variance > 0).all():
        record["status"] = "inconclusive_nonpositive_pooled_variance"
        record["pooled_scalar_variance"] = variance.tolist()
        write_json(RESULT_PATH, record)
        print(f"B11 inconclusive: pooled scalar variance {variance.tolist()}")
        return
    shared_forms = _shared_forms(jac_complete, variance, split)
    shared_pairs, observed = _distances(shared_forms)
    legacy_forms, legacy_variances = _legacy_forms(jac_complete, base_complete, split)
    legacy_pairs, legacy_dispersion = _distances(legacy_forms)
    null_spec = spec["permutation_null"]
    null = _permutation_null(jac_complete, variance, split,
                             int(null_spec["repetitions"]), int(null_spec["seed"]))
    null_95 = float(np.quantile(null, float(null_spec["percentile"]) / 100,
                                method=null_spec["numpy_quantile_method"]))
    boot_spec = spec["bootstrap_uncertainty"]
    boot = _bootstrap(jac_complete, variance, split, int(boot_spec["repetitions"]),
                      int(boot_spec["seed"]))
    boot_ci = np.quantile(boot, np.asarray(boot_spec["dispersion_interval_percentiles"]) / 100)
    gate_minimum = float(spec["screening_gate"]["dispersion_minimum"])
    record.update({
        "status": "measured",
        "pooled_scalar_variance": variance.tolist(),
        "shared_denominator": {
            "raw_traces": [float(np.trace(form)) for form in shared_forms],
            "trace_normalized_pair_distances": shared_pairs,
            "primary_mean_pair_distance": observed,
            "top7_subspace_angles": _angles(shared_forms, int(spec["comparisons"][
                "descriptive_subspace_rank"
            ])),
        },
        "legacy_group_denominator_control": {
            "group_scalar_variances": legacy_variances,
            "raw_traces": [float(np.trace(form)) for form in legacy_forms],
            "trace_normalized_pair_distances": legacy_pairs,
            "mean_pair_distance": legacy_dispersion,
            "difference_from_shared_mean_pair_distance": legacy_dispersion - observed,
        },
        "permutation_null": {
            "values": null.tolist(), "median": float(np.median(null)),
            "percentile_95": null_95,
            "empirical_upper_tail_p": float((1 + np.count_nonzero(null >= observed)) /
                                            (1 + len(null))),
        },
        "bootstrap_dispersion": {
            "values": boot.tolist(), "percentile_2_5": float(boot_ci[0]),
            "percentile_97_5": float(boot_ci[1]),
        },
        "gate_minimum": gate_minimum,
        "gate_passed": bool(observed > null_95 and observed >= gate_minimum),
    })
    write_json(RESULT_PATH, record)
    print(f"B11 shared-denominator dispersion {observed:.6f}; permutation p95 "
          f"{null_95:.6f}; threshold {gate_minimum:.2f}; "
          f"gate {'PASS' if record['gate_passed'] else 'FAIL'}")
    print(f"B11 legacy-denominator dispersion on the same probes {legacy_dispersion:.6f}")
    print(f"Wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()

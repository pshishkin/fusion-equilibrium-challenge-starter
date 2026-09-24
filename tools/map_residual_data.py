"""Build exact OOF source/truth map pairs and train-only U-Net arrays."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from experiments import _as_psirz_stack
from my_experiments import baseline_model
from my_experiments.map_residual_unet import frame_energies, grid_channels
from toolkit.parallel import pimap, release
from toolkit.records import file_hash, write_json
from tools.direct_contour_compact_val_predict import INPUT_COLUMNS
from tools.map_residual_campaign import verify_campaign


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"{path}: source/truth target cache already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _train_one(task: tuple[str, str, int, str, str, str]) -> dict[str, Any]:
    name, data_s, expected_size, expected_sha, artifact_s, output_s = task
    path = Path(data_s) / "data/diii_d_train" / name
    if path.stat().st_size != expected_size or file_hash(path) != expected_sha:
        raise ValueError(f"{path}: bytes/SHA differ from pinned dataset manifest")
    row = pd.read_parquet(path, columns=INPUT_COLUMNS).iloc[0]
    forbidden = [key for key in row.index if key.startswith("efit_")
                 and key not in ("efit_times", "efit_grid_R", "efit_grid_Z")]
    if forbidden:
        raise ValueError(f"{path}: target-stripped inference row contains {forbidden}")
    n = len(np.asarray(row["efit_times"]))
    slim = baseline_model.slim_row(row)
    artifact: dict[str, Any] = joblib.load(artifact_s)
    if name not in artifact["oof_files"] or name in artifact["train_files"]:
        raise ValueError(f"{path}: fold0 base saw shot or did not declare it OOF")
    features = baseline_model.features_for_row(row)
    selected = baseline_model.build_inputs(artifact["coil"], features)
    prediction = baseline_model.predict_row(slim, "DIII-D", "mlp", Path(artifact_s))
    source = np.asarray(prediction["psirz"], dtype=np.float32)
    q95 = np.asarray(prediction["q95"], dtype=np.float32)
    beta = np.asarray(prediction["betaN"], dtype=np.float32)
    if (source.shape != (n, 65, 65) or selected.shape != (n, 84)
            or q95.shape != (n,) or beta.shape != (n,)
            or not np.isfinite(source).all() or not np.isfinite(selected).all()
            or not np.isfinite(q95).all() or not np.isfinite(beta).all()):
        raise ValueError(f"{path}: source psi/selected84/scalars shapes or finiteness "
                         f"{source.shape}/{selected.shape}/{q95.shape}/{beta.shape}, "
                         f"expected ({n},65,65)/({n},84)/({n},)/({n},) finite")
    # Supervised truth is opened only AFTER source inference has used an input-only row.
    truth_row = pd.read_parquet(path, columns=["efit_psirz"]).iloc[0]
    truth = _as_psirz_stack(truth_row["efit_psirz"]).astype(np.float32)
    if truth.shape != source.shape or not np.isfinite(truth).all():
        raise ValueError(f"{path}: true psi shape/finiteness {truth.shape}/"
                         f"{bool(np.isfinite(truth).all())}, expected {source.shape}/true")
    kept = baseline_model.thin_frames(n, 0.2)
    output = Path(output_s) / f"{Path(name).stem}.npz"
    _atomic_npz(output, {
        "source_psi": source[kept],
        "true_psi": truth[kept],
        "selected84": np.asarray(selected[kept], dtype=np.float32),
        "kept_indices": np.asarray(kept, dtype=np.int32),
    })
    return {
        "shot": name,
        "full_frames": n,
        "kept_frames": len(kept),
        "source_parquet_sha256": expected_sha,
        "prediction_used_efit_target_columns": False,
        "supervised_target_columns": ["efit_psirz"],
        "output_path": str(output),
        "output_sha256": file_hash(output),
    }


def cache_train(campaign: Path, jobs: int) -> None:
    """Extract only fold0 OOF targets through the shared ordered shot pool."""
    started = time.perf_counter()
    campaign = campaign.resolve()
    meta = verify_campaign(campaign)
    if jobs < 1 or jobs > 24:
        raise ValueError(f"jobs={jobs}, expected 1..24")
    ready = campaign / "train_cache_ready.json"
    if ready.exists():
        raise FileExistsError(f"{ready}: OOF train target cache already complete")
    names = meta["train_files"]
    if len(names) != 1355 or set(names) & (set(meta["gate_files"])
                                        | set(meta["reserved_tail_files"])):
        raise ValueError(f"{campaign}: OOF training roles overlap gate/tail")
    artifact = Path(meta["origin_root"]) / "runs/oof-normal-k8-5fold-20260923/fold0/base.joblib"
    data = Path(meta["data_dir"])
    dataset = json.loads((campaign / "source/my_experiments/diii_d_dataset.json").read_text())
    output_dir = campaign / "train_shots"
    output_dir.mkdir(exist_ok=True)
    tasks = [(name, str(data), *dataset["configs"]["diii_d_train"][name],
              str(artifact), str(output_dir)) for name in names]
    try:
        entries = list(pimap(_train_one, tasks, jobs))
    finally:
        release()
    if ([entry["shot"] for entry in entries] != names
            or sum(entry["full_frames"] for entry in entries) != 300793
            or sum(entry["kept_frames"] for entry in entries) != 60171
            or any(entry["prediction_used_efit_target_columns"] for entry in entries)):
        raise ValueError(f"{campaign}: ordered train shots/full/kept frames or "
                         "target-stripped inference differ from 1355/300793/60171")
    write_json(ready, {
        "classification": "fold0_OOF_physical_source_and_true_psi_only_training_cache",
        "campaign_source_bundle_sha256": meta["source_bundle_sha256"],
        "base_artifact_sha256": meta["fold0_artifact_sha256"],
        "train_shots": len(entries),
        "full_frames": sum(entry["full_frames"] for entry in entries),
        "kept_frames": sum(entry["kept_frames"] for entry in entries),
        "gate_or_tail_shots_read": 0,
        "entries": entries,
        "elapsed_seconds": time.perf_counter() - started,
    })


def assemble(campaign: Path) -> None:
    """Publish exact kept-frame memmaps and train-only feature/loss scalers."""
    started = time.perf_counter()
    campaign = campaign.resolve()
    meta = verify_campaign(campaign)
    ready_path = campaign / "train_cache_ready.json"
    ready = json.loads(ready_path.read_text())
    entries = ready["entries"]
    if (ready["campaign_source_bundle_sha256"] != meta["source_bundle_sha256"]
            or [entry["shot"] for entry in entries] != meta["train_files"]
            or ready["kept_frames"] != 60171):
        raise ValueError(f"{ready_path}: OOF role/source or kept-frame count differs")
    params: dict[str, Any] = yaml.safe_load(
        (campaign / "source/baselines/map-residual-unet.params.yaml").read_text())
    with np.load(campaign / "source/fusion_scoring/masks/d3d_envelope.npz") as mask_file:
        grid_r = np.asarray(mask_file["grid_R"], dtype=np.float64)
        grid_z = np.asarray(mask_file["grid_Z"], dtype=np.float64)
        mask = np.asarray(mask_file["mask_coarse"], dtype=np.bool_)
    loss_cfg = params["loss"]
    if (loss_cfg["envelope_inside_weight"] != 2.0
            or loss_cfg["envelope_outside_weight"] != 1.0
            or loss_cfg["value_weight"] != 1.0 or loss_cfg["gradient_weight"] != 1.0
            or loss_cfg["gradient_stencil"] != "physical_forward_difference_no_padding"
            or loss_cfg["normalization"]
            != "equal_shot_training_initial_residual_energy"):
        raise ValueError(f"{campaign}: map residual loss YAML differs from fixed review")
    weight = np.where(mask, 2.0, 1.0).astype(np.float64)
    output = campaign / "fit_data"
    if output.exists() or (campaign / "assembly.json").exists():
        raise FileExistsError(f"{output}: fit arrays already published")
    temporary = Path(tempfile.mkdtemp(dir=campaign, prefix=".map-unet-assembling-"))
    count = 60171
    source_map = np.lib.format.open_memmap(
        temporary / "source_norm.npy", mode="w+", dtype=np.float32, shape=(count, 65, 65))
    target_map = np.lib.format.open_memmap(
        temporary / "target_norm.npy", mode="w+", dtype=np.float32, shape=(count, 65, 65))
    feature_map = np.lib.format.open_memmap(
        temporary / "features86.npy", mode="w+", dtype=np.float32, shape=(count, 86))
    sample_weight = np.empty(count, dtype=np.float64)
    offsets = [0]
    value_total = 0.0
    gradient_total = 0.0
    for entry in entries:
        path = Path(entry["output_path"])
        if file_hash(path) != entry["output_sha256"]:
            raise ValueError(f"{path}: cached source/truth/feature hash differs")
        with np.load(path, allow_pickle=False) as packed:
            source = np.asarray(packed["source_psi"])
            truth = np.asarray(packed["true_psi"])
            selected = np.asarray(packed["selected84"])
            kept = np.asarray(packed["kept_indices"])
        n = entry["kept_frames"]
        expected_indices = baseline_model.thin_frames(entry["full_frames"], 0.2)
        if (source.shape != (n, 65, 65) or truth.shape != source.shape
                or selected.shape != (n, 84) or source.dtype != np.float32
                or truth.dtype != np.float32 or selected.dtype != np.float32
                or not np.array_equal(kept, expected_indices)):
            raise ValueError(f"{path}: kept source/true/feature/index shapes or bits differ "
                             f"from {n} rows and the fixed 0.2 frame selector")
        channels, extra, scale = grid_channels(
            source, grid_r, grid_z, mask,
            int(params["source"]["source_range_min_float32_ulps"]), entry["shot"])
        target = ((truth.astype(np.float64) - source.astype(np.float64))
                  / scale[:, None, None]).astype(np.float32)
        features = np.column_stack([selected, extra]).astype(np.float32)
        if not np.isfinite(target).all() or not np.isfinite(features).all():
            raise ValueError(f"{path}: normalized physical residual or features nonfinite")
        offset = offsets[-1]
        source_map[offset:offset + n] = channels[:, 0]
        target_map[offset:offset + n] = target
        feature_map[offset:offset + n] = features
        sample_weight[offset:offset + n] = 1.0 / n
        values, gradients = frame_energies(target, weight, grid_r, grid_z)
        value_total += float(np.mean(values))
        gradient_total += float(np.mean(gradients))
        offsets.append(offset + n)
    if offsets[-1] != count:
        raise ValueError(f"{campaign}: assembled {offsets[-1]} frames, expected {count}")
    tau_value = value_total / len(entries)
    tau_gradient = gradient_total / len(entries)
    if (not np.isfinite(tau_value) or not np.isfinite(tau_gradient)
            or tau_value <= 0 or tau_gradient <= 0):
        raise ValueError(f"{campaign}: train-only tau value/gradient "
                         f"{tau_value}/{tau_gradient}, expected positive finite")
    scaler = StandardScaler()
    scaler.fit(feature_map, sample_weight=sample_weight)
    if scaler.n_features_in_ != 86 or not np.isfinite(scaler.mean_).all():
        raise ValueError(f"{campaign}: train-only head feature scaler width/mean differs")
    for start in range(0, count, 8192):
        stop = min(count, start + 8192)
        feature_map[start:stop] = scaler.transform(feature_map[start:stop]).astype(np.float32)
    source_map.flush()
    target_map.flush()
    feature_map.flush()
    del source_map, target_map, feature_map
    scaler_path = temporary / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    os.replace(temporary, output)
    files = {path.name: file_hash(path) for path in sorted(output.iterdir())}
    write_json(campaign / "assembly.json", {
        "classification": "train_only_equal_shot_map_residual_arrays_and_constants",
        "source_bundle_sha256": meta["source_bundle_sha256"],
        "cache_ready_sha256": file_hash(ready_path),
        "train_shots": len(entries),
        "train_frames": count,
        "shot_offsets": offsets,
        "tau_value": tau_value,
        "tau_gradient": tau_gradient,
        "files_sha256": files,
        "gate_or_tail_targets_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("cache")
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=24)
    p = sub.add_parser("assemble")
    p.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "cache":
        cache_train(args.campaign, args.jobs)
    else:
        assemble(args.campaign)


if __name__ == "__main__":
    main()

"""Fit the one fixed-step zero-initialized physical-map residual U-Net."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from my_experiments.map_residual_unet import MapResidualUNet, torch_loss
from toolkit.records import file_hash, write_json
from tools.map_residual_campaign import verify_campaign


def _atomic_checkpoint(path: Path, state: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"{path}: fixed U-Net checkpoint already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".pt")
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def fit(campaign: Path) -> None:
    """Sample OOF shots uniformly, keep the fixed last of 10,000 CUDA updates."""
    started = time.perf_counter()
    campaign = campaign.resolve()
    meta = verify_campaign(campaign)
    params_path = campaign / "source/baselines/map-residual-unet.params.yaml"
    params: dict[str, Any] = yaml.safe_load(params_path.read_text())
    cfg = params["train"]
    if (cfg["seed"] != 0 or cfg["batch_frames"] != 32
            or cfg["max_steps"] != 10000 or cfg["optimizer"] != "adamw"
            or cfg["learning_rate"] != 0.0003 or cfg["weight_decay"] != 0.0001
            or cfg["gradient_clip_norm"] != 1.0 or cfg["scheduler"] != "none"
            or cfg["checkpoint"] != "last_step" or cfg["device"] != "cuda"
            or cfg["log_every_steps"] != 500 or cfg["torch_threads"] != 4):
        raise ValueError(f"{params_path}: training hyperparameters differ from frozen review")
    assembly_path = campaign / "assembly.json"
    assembled = json.loads(assembly_path.read_text())
    if (assembled["source_bundle_sha256"] != meta["source_bundle_sha256"]
            or assembled["train_shots"] != 1355
            or assembled["train_frames"] != 60171
            or assembled["gate_or_tail_targets_read"]):
        raise ValueError(f"{assembly_path}: OOF train arrays or source identity differ")
    data = campaign / "fit_data"
    for name, digest in assembled["files_sha256"].items():
        if file_hash(data / name) != digest:
            raise ValueError(f"{data / name}: frozen fit-data SHA differs")
    if not torch.cuda.is_available():
        raise RuntimeError(f"{params_path}: explicit CUDA requested but unavailable")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = MapResidualUNet(params["model"], 86).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
    source = np.load(data / "source_norm.npy", mmap_mode="r")
    target = np.load(data / "target_norm.npy", mmap_mode="r")
    features = np.load(data / "features86.npy", mmap_mode="r")
    if (source.shape != (60171, 65, 65) or target.shape != source.shape
            or features.shape != (60171, 86)
            or source.dtype != np.float32 or target.dtype != np.float32
            or features.dtype != np.float32):
        raise ValueError(f"{data}: source/target/features shapes or dtypes differ")
    offsets = np.asarray(assembled["shot_offsets"], dtype=np.int64)
    if len(offsets) != 1356 or offsets[0] != 0 or offsets[-1] != 60171:
        raise ValueError(f"{assembly_path}: shot offsets {len(offsets)}/"
                         f"{offsets[0]}/{offsets[-1]}, expected 1356/0/60171")
    with np.load(campaign / "source/fusion_scoring/masks/d3d_envelope.npz") as mask_file:
        r = np.asarray(mask_file["grid_R"], dtype=np.float32)
        z = np.asarray(mask_file["grid_Z"], dtype=np.float32)
        mask = np.asarray(mask_file["mask_coarse"], dtype=np.bool_)
    device = torch.device("cuda")
    rr, zz = np.meshgrid(2 * (r - r[0]) / (r[-1] - r[0]) - 1,
                         2 * (z - z[0]) / (z[-1] - z[0]) - 1)
    coords = torch.from_numpy(np.stack([rr, zz, mask.astype(np.float32)])).to(device)
    weight = torch.from_numpy(np.where(mask, 2.0, 1.0).astype(np.float32)).to(device)
    grid_r = torch.from_numpy(r.copy()).to(device)
    grid_z = torch.from_numpy(z.copy()).to(device)
    tau_value = float(assembled["tau_value"])
    tau_gradient = float(assembled["tau_gradient"])
    trace: list[dict[str, float]] = []
    torch.cuda.reset_peak_memory_stats()
    fit_started = time.perf_counter()
    for step in range(1, 10001):
        selected_shots = rng.integers(0, 1355, size=32)
        sampled = offsets[selected_shots] + np.floor(rng.random(32) * (
            offsets[selected_shots + 1] - offsets[selected_shots])).astype(np.int64)
        map_batch = torch.from_numpy(np.array(source[sampled], copy=True)).to(device)
        map_input = torch.cat([map_batch[:, None], coords[None].expand(32, -1, -1, -1)],
                              dim=1)
        feature_batch = torch.from_numpy(np.array(features[sampled], copy=True)).to(device)
        target_batch = torch.from_numpy(np.array(target[sampled], copy=True)).to(device)[:, None]
        optimizer.zero_grad(set_to_none=True)
        predicted = model(map_input, feature_batch)
        loss = torch_loss(predicted, target_batch, weight, grid_r, grid_z,
                          tau_value, tau_gradient)
        if not torch.isfinite(loss):
            raise ValueError(f"{campaign}: nonfinite U-Net loss at fixed step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm):
            raise ValueError(f"{campaign}: nonfinite U-Net gradient at fixed step {step}")
        optimizer.step()
        if step % 500 == 0:
            value = float(loss.item())
            trace.append({"step": float(step), "train_loss": value})
            print(f"U-Net step {step}/10000: train loss {value:.6f}", flush=True)
    torch.cuda.synchronize()
    fit_seconds = time.perf_counter() - fit_started
    output = campaign / "models/fold0_map_residual.pt"
    _atomic_checkpoint(output, {
        "state_dict": model.cpu().state_dict(),
        "step": 10000,
        "seed": 0,
        "params_sha256": file_hash(params_path),
        "spec_sha256": meta["spec_sha256"],
        "assembly_sha256": file_hash(assembly_path),
        "scaler_sha256": assembled["files_sha256"]["scaler.joblib"],
        "base_artifact_sha256": meta["fold0_artifact_sha256"],
        "source_bundle_sha256": meta["source_bundle_sha256"],
    })
    write_json(output.with_suffix(".json"), {
        "classification": "one_fixed_seed0_10000_step_physical_map_residual_unet",
        "checkpoint_sha256": file_hash(output),
        "params_sha256": file_hash(params_path),
        "spec_sha256": meta["spec_sha256"],
        "assembly_sha256": file_hash(assembly_path),
        "train_shots": 1355,
        "train_frames": 60171,
        "tau_value": tau_value,
        "tau_gradient": tau_gradient,
        "steps": 10000,
        "fit_wall_seconds": fit_seconds,
        "total_wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "trace": trace,
        "gate_or_tail_labels_read": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    fit(args.campaign)


if __name__ == "__main__":
    main()

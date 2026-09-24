"""Synthetic CUDA throughput and memory gate for the fixed map residual U-Net."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from my_experiments.map_residual_unet import MapResidualUNet, torch_loss
from toolkit.records import file_hash, write_json


def smoke(params_path: Path, mask_path: Path, output: Path) -> dict[str, Any]:
    """Measure 50 exact-shape CUDA updates with synthetic arrays only."""
    params: dict[str, Any] = yaml.safe_load(params_path.read_text())
    train = params["train"]
    if (train["device"] != "cuda" or train["batch_frames"] != 32
            or train["max_steps"] != 10000 or train["seed"] != 0
            or train["optimizer"] != "adamw" or train["scheduler"] != "none"):
        raise ValueError(f"{params_path}: synthetic smoke requires frozen CUDA/32/10000/seed0")
    if not torch.cuda.is_available():
        raise RuntimeError(f"{params_path}: explicit CUDA device unavailable")
    torch.set_num_threads(int(train["torch_threads"]))
    torch.manual_seed(0)
    with np.load(mask_path) as packed:
        r = np.asarray(packed["grid_R"], dtype=np.float64)
        z = np.asarray(packed["grid_Z"], dtype=np.float64)
        mask = np.asarray(packed["mask_coarse"], dtype=np.bool_)
    if r.shape != (65,) or z.shape != (65,) or mask.shape != (65, 65):
        raise ValueError(f"{mask_path}: DIII-D grid/mask shapes differ from 65/65")
    device = torch.device("cuda")
    model = MapResidualUNet(params["model"], int(params["source"]["input_feature_width"]))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(train["learning_rate"]),
                                  weight_decay=float(train["weight_decay"]))
    batch = int(train["batch_frames"])
    # The values are independent random synthetic tensors, never competition data.
    maps = torch.rand((batch, 4, 65, 65), device=device)
    features = torch.randn((batch, 86), device=device)
    target = torch.randn((batch, 1, 65, 65), device=device) * 0.01
    weight = torch.from_numpy(np.where(mask, 2.0, 1.0).astype(np.float32)).to(device)
    grid_r = torch.from_numpy(r.astype(np.float32)).to(device)
    grid_z = torch.from_numpy(z.astype(np.float32)).to(device)
    torch.cuda.reset_peak_memory_stats()
    warmup = 5
    measured = 50
    started = time.perf_counter()
    last_loss = 0.0
    for step in range(warmup + measured):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(maps, features)
        loss = torch_loss(prediction, target, weight, grid_r, grid_z, 1.0, 1.0)
        if not torch.isfinite(loss):
            raise ValueError(f"{params_path}: synthetic CUDA loss nonfinite at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       float(train["gradient_clip_norm"]))
        optimizer.step()
        last_loss = float(loss.item())
        if step == warmup - 1:
            torch.cuda.synchronize()
            started = time.perf_counter()
    torch.cuda.synchronize()
    measured_seconds = time.perf_counter() - started
    peak_bytes = torch.cuda.max_memory_allocated()
    projected = measured_seconds / measured * int(train["max_steps"])
    result = {
        "classification": "synthetic_only_fixed_unet_cuda_resource_gate",
        "params_sha256": file_hash(params_path),
        "mask_sha256": file_hash(mask_path),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "batch": batch,
        "warmup_updates": warmup,
        "measured_updates": measured,
        "measured_seconds": measured_seconds,
        "projected_10000_step_seconds": projected,
        "peak_allocated_bytes": peak_bytes,
        "last_synthetic_loss": last_loss,
        "resource_gate_passed": projected <= 90 * 60 and peak_bytes <= 18 * 2 ** 30,
        "real_data_read": False,
    }
    write_json(output, result)
    if not result["resource_gate_passed"]:
        raise RuntimeError(f"{output}: fixed synthetic CUDA resource gate failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = smoke(args.params.resolve(), args.mask.resolve(), args.output.resolve())
    print(f"Synthetic U-Net 50-step wall {result['measured_seconds']:.3f}s; "
          f"projected fixed fit {result['projected_10000_step_seconds']:.1f}s; "
          f"peak GPU allocation {result['peak_allocated_bytes'] / 2 ** 30:.3f} GiB")


if __name__ == "__main__":
    main()

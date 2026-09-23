"""One fixed full-physical-contour MLP, with train-only scaling and shot-balanced batches."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn
import yaml

from toolkit.records import file_hash, write_json
from tools.direct_full_contour_cache import ROOT, YAML, verify_campaign


class DirectContourMLP(nn.Module):
    """The sole preregistered 209-to-1024 architecture."""

    def __init__(self, width: int = 512) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(209, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, 1024),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def contour_loss(prediction: torch.Tensor, target: torch.Tensor,
                 worst_vertex_count: int = 26,
                 worst_weight: float = 1.0) -> torch.Tensor:
    """Per-frame pointwise physical MSE plus equal-weight worst-5% vertex MSE."""
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 1024:
        raise ValueError(f"prediction/target shapes {prediction.shape}/{target.shape}, "
                         "expected matching (N,1024)")
    if not 1 <= worst_vertex_count <= 512:
        raise ValueError(f"worst vertex count {worst_vertex_count}, expected 1..512")
    squared = ((prediction - target).reshape(-1, 512, 2) ** 2).sum(dim=2)
    per_frame = squared.mean(dim=1) + worst_weight * torch.topk(
        squared, worst_vertex_count, dim=1
    ).values.mean(dim=1)
    return per_frame


def _cache_records(run: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest, spec = verify_campaign(run)
    ready_path = run / "target_cache_ready.json"
    ready = json.loads(ready_path.read_text())
    if (ready["manifest_sha256"] != file_hash(run / "manifest.json")
            or ready["train_ordered_names_sha256"] != manifest["train_ordered_names_sha256"]
            or ready["validation_ordered_names_sha256"]
            != manifest["validation_ordered_names_sha256"]
            or len(ready["ready_entries"]) != 6971):
        raise ValueError(f"{ready_path}: target-cache completion identity changed")
    for i, record in enumerate(ready["ready_entries"]):
        expected = ((manifest["train_files"] + manifest["validation_files"])[i])
        if record["shot"] != expected or file_hash(Path(record["cache_path"])) != record[
            "cache_sha256"
        ]:
            raise ValueError(f"{record['cache_path']}: cache shot/hash differs from {expected}")
    return manifest, spec, ready


def assemble(run: Path) -> None:
    """Assemble bounded float32 memmaps and fit only TRAIN normalization statistics."""
    started = time.perf_counter()
    manifest, _, ready = _cache_records(run)
    folder = run / "fit"
    if folder.exists():
        raise FileExistsError(f"{folder}: fit arrays or model already exist")
    folder.mkdir(parents=True)
    records = ready["ready_entries"]
    counts = {
        "train": sum(item["kept_frames"] for item in records[:5633]),
        "validation": sum(item["kept_frames"] for item in records[5633:]),
    }
    sum_x = np.zeros(209, dtype=np.float64)
    sum_x2 = np.zeros(209, dtype=np.float64)
    sum_y = np.zeros(1024, dtype=np.float64)
    sum_y2 = 0.0
    shot_offsets: dict[str, list[int]] = {"train": [0], "validation": [0]}
    arrays: dict[str, Any] = {}
    for window in ("train", "validation"):
        arrays[f"X_{window}"] = np.lib.format.open_memmap(
            folder / f"X_{window}.npy", mode="w+", dtype=np.float32,
            shape=(counts[window], 209)
        )
        arrays[f"Y_{window}"] = np.lib.format.open_memmap(
            folder / f"Y_{window}.npy", mode="w+", dtype=np.float32,
            shape=(counts[window], 1024)
        )
    for i, record in enumerate(records):
        window = "train" if i < 5633 else "validation"
        with np.load(record["cache_path"], allow_pickle=False) as saved:
            features = np.asarray(saved["features"])
            targets = np.asarray(saved["contours"]).reshape(-1, 1024)
            indices = saved["indices"]
            full_frames = int(saved["full_frames"])
        n = record["kept_frames"]
        if (features.shape != (n, 209) or targets.shape != (n, 1024)
                or len(indices) != n or full_frames != record["full_frames"]
                or features.dtype != np.float32 or targets.dtype != np.float32
                or not np.isfinite(features).all() or not np.isfinite(targets).all()):
            raise ValueError(f"{record['cache_path']}: assembled arrays/count differ from ready")
        start = shot_offsets[window][-1]
        arrays[f"X_{window}"][start:start + n] = features
        arrays[f"Y_{window}"][start:start + n] = targets
        shot_offsets[window].append(start + n)
        if window == "train":
            sum_x += features.astype(np.float64).mean(axis=0)
            sum_x2 += (features.astype(np.float64) ** 2).mean(axis=0)
            sum_y += targets.astype(np.float64).sum(axis=0)
            sum_y2 += float(np.sum(targets.astype(np.float64) ** 2))
    for window in ("train", "validation"):
        if shot_offsets[window][-1] != counts[window]:
            raise ValueError(f"{window}: assembled {shot_offsets[window][-1]} frames, "
                             f"expected {counts[window]}")
        arrays[f"X_{window}"].flush()
        arrays[f"Y_{window}"].flush()
    mean_x = sum_x / 5633
    variance_x = np.maximum(sum_x2 / 5633 - mean_x**2, 0.0)
    std_x = np.sqrt(variance_x)
    constant_count = int(np.count_nonzero(std_x == 0))
    std_x[std_x == 0] = 1.0  # Defined StandardScaler behavior for genuine constant inputs.
    mean_y = sum_y / counts["train"]
    target_variance = sum_y2 / (counts["train"] * 1024) - float(np.mean(mean_y**2))
    scale_y = float(np.sqrt(target_variance))
    if (not np.isfinite(mean_x).all() or not np.isfinite(std_x).all()
            or not np.isfinite(mean_y).all() or not np.isfinite(scale_y)
            or scale_y <= 0):
        raise ValueError("train-only scaler has nonfinite or zero global target RMS")
    record = {
        "classification": "train_only_scaling_and_memmaps_ready",
        "campaign_manifest_sha256": file_hash(run / "manifest.json"),
        "target_cache_ready_sha256": file_hash(run / "target_cache_ready.json"),
        "train_ordered_names_sha256": manifest["train_ordered_names_sha256"],
        "validation_ordered_names_sha256": manifest["validation_ordered_names_sha256"],
        "train_kept_frames": counts["train"],
        "validation_kept_frames": counts["validation"],
        "train_offsets": shot_offsets["train"],
        "validation_offsets": shot_offsets["validation"],
        "feature_mean": mean_x.tolist(),
        "feature_std": std_x.tolist(),
        "constant_feature_columns": constant_count,
        "contour_mean_RZ_interleaved": mean_y.tolist(),
        "contour_global_RMS_m": scale_y,
        "memmap_sha256": {name: file_hash(folder / f"{name}.npy") for name in arrays},
        "assembly_wall_seconds": time.perf_counter() - started,
    }
    write_json(folder / "assembly.json", record)
    print(f"Fit arrays: train {counts['train']}, val {counts['validation']} frames; "
          f"global target RMS {scale_y:.6f} m, {record['assembly_wall_seconds']:.1f} s")


def _validation_loss(net: DirectContourMLP, X: np.ndarray, Y: np.ndarray,
                     offsets: list[int], mean_x: torch.Tensor, std_x: torch.Tensor,
                     mean_y: torch.Tensor, scale_y: float, batch: int,
                     device: torch.device) -> float:
    net.eval()
    weights = np.empty(len(X), dtype=np.float32)
    for left, right in itertools.pairwise(offsets):
        weights[left:right] = 1.0 / (right - left)
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(X), batch):
            stop = min(start + batch, len(X))
            x = torch.from_numpy(np.asarray(X[start:stop]).copy()).to(device)
            y = torch.from_numpy(np.asarray(Y[start:stop]).copy()).to(device)
            pred = net((x - mean_x) / std_x)
            per_frame = contour_loss(pred, (y - mean_y) / scale_y)
            w = torch.from_numpy(weights[start:stop].copy()).to(device)
            total += float((per_frame * w).sum().item())
    net.train()
    return total / (len(offsets) - 1)


def fit(run: Path) -> None:
    """Fit exactly one seed-0 CUDA model and save its best validation checkpoint."""
    started = time.perf_counter()
    manifest, _, ready = _cache_records(run)
    assembly_path = run / "fit/assembly.json"
    assembly = json.loads(assembly_path.read_text())
    if assembly["target_cache_ready_sha256"] != file_hash(run / "target_cache_ready.json"):
        raise ValueError(f"{assembly_path}: target cache changed after assembly")
    if (run / "fit/direct_contour.joblib").exists():
        raise FileExistsError(f"{run}: model checkpoint already exists")
    plan = yaml.safe_load((ROOT / YAML).read_text())
    if plan["device"] != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("review YAML requires CUDA; CUDA device is unavailable")
    hp = plan["optimizer"]
    if (hp["max_steps"] != 8000 or hp["batch_size"] != 4096
            or hp["validation_every_steps"] != 250 or hp["patience_steps"] != 1500
            or plan["loss"]["worst_vertex_count"] != 26
            or plan["loss"]["worst_vertex_mean_weight"] != 1.0):
        raise ValueError(f"{ROOT / YAML}: fixed training/loss hyperparameters changed")
    folder = run / "fit"
    for name, digest in assembly["memmap_sha256"].items():
        if file_hash(folder / f"{name}.npy") != digest:
            raise ValueError(f"{folder / f'{name}.npy'}: assembled memmap SHA changed")
    X_train = np.load(folder / "X_train.npy", mmap_mode="r")
    Y_train = np.load(folder / "Y_train.npy", mmap_mode="r")
    X_val = np.load(folder / "X_validation.npy", mmap_mode="r")
    Y_val = np.load(folder / "Y_validation.npy", mmap_mode="r")
    offsets = assembly["train_offsets"]
    lengths = np.diff(np.asarray(offsets, dtype=np.int64))
    if (len(lengths) != 5633 or len(assembly["validation_offsets"]) != 1339
            or np.any(lengths <= 0)):
        raise ValueError(f"{assembly_path}: invalid train/validation shot offsets")
    seed = int(plan["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda")
    net = DirectContourMLP(int(plan["model"]["hidden_width"])).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=float(hp["learning_rate"]),
                                  weight_decay=float(hp["weight_decay"]))
    mean_x = torch.tensor(assembly["feature_mean"], dtype=torch.float32, device=device)
    std_x = torch.tensor(assembly["feature_std"], dtype=torch.float32, device=device)
    mean_y = torch.tensor(assembly["contour_mean_RZ_interleaved"],
                          dtype=torch.float32, device=device)
    scale_y = float(assembly["contour_global_RMS_m"])
    best_loss = float("inf")
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    trace: list[dict[str, float]] = []
    validation_seconds = 0.0
    for step in range(1, int(hp["max_steps"]) + 1):
        shots = rng.integers(0, len(lengths), size=int(hp["batch_size"]))
        indices = np.asarray(offsets, dtype=np.int64)[shots] + (
            rng.random(len(shots)) * lengths[shots]
        ).astype(np.int64)
        x = torch.from_numpy(np.asarray(X_train[indices]).copy()).to(device)
        y = torch.from_numpy(np.asarray(Y_train[indices]).copy()).to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = net((x - mean_x) / std_x)
        loss = contour_loss(prediction, (y - mean_y) / scale_y).mean()
        if not torch.isfinite(loss):
            raise ValueError(f"MLP step {step}: train loss is nonfinite")
        loss.backward()
        optimizer.step()
        if step % int(hp["validation_every_steps"]) == 0:
            begin = time.perf_counter()
            val = _validation_loss(net, X_val, Y_val, assembly["validation_offsets"],
                                   mean_x, std_x, mean_y, scale_y, int(hp["batch_size"]),
                                   device)
            validation_seconds += time.perf_counter() - begin
            if not np.isfinite(val):
                raise ValueError(f"MLP step {step}: validation loss is nonfinite")
            trace.append({"step": float(step), "train_loss": float(loss.item()),
                          "validation_loss": val})
            print(f"direct contour step {step}: train {loss.item():.6f}, "
                  f"validation {val:.6f}", flush=True)
            if val < best_loss:
                best_loss, best_step = val, step
                best_state = {key: value.detach().cpu().clone()
                              for key, value in net.state_dict().items()}
            if step - best_step >= int(hp["patience_steps"]):
                break
    if best_state is None:
        raise ValueError("MLP had no finite validation checkpoint")
    checkpoint = {
        "state_dict": best_state,
        "feature_mean": torch.tensor(assembly["feature_mean"], dtype=torch.float32),
        "feature_std": torch.tensor(assembly["feature_std"], dtype=torch.float32),
        "target_mean": torch.tensor(assembly["contour_mean_RZ_interleaved"],
                                     dtype=torch.float32),
        "target_scale": torch.tensor(scale_y, dtype=torch.float32),
        "best_step": best_step,
        "best_validation_loss": best_loss,
        "manifest_sha256": file_hash(run / "manifest.json"),
        "assembly_sha256": file_hash(assembly_path),
        "yaml_sha256": manifest["yaml_sha256"],
        "train_ordered_names_sha256": manifest["train_ordered_names_sha256"],
        "validation_ordered_names_sha256": manifest["validation_ordered_names_sha256"],
    }
    fd, temporary = tempfile.mkstemp(dir=folder, suffix=".joblib")
    os.close(fd)
    try:
        joblib.dump(checkpoint, temporary)
        os.replace(temporary, folder / "direct_contour.joblib")
    finally:
        Path(temporary).unlink(missing_ok=True)
    write_json(folder / "fit.json", {
        "classification": "single_input_only_direct_contour_MLP_fit_complete",
        "checkpoint_sha256": file_hash(folder / "direct_contour.joblib"),
        "manifest_sha256": file_hash(run / "manifest.json"),
        "assembly_sha256": file_hash(assembly_path),
        "train_shots": 5633, "validation_shots": 1338,
        "train_frames": ready["train_kept_frames"],
        "validation_frames": ready["validation_kept_frames"],
        "best_step": best_step, "best_validation_loss": best_loss,
        "completed_steps": step,
        "validation_wall_seconds": validation_seconds,
        "fit_wall_seconds": time.perf_counter() - started,
        "trace": trace,
        "CUDA_device": torch.cuda.get_device_name(0),
        "no_base_prediction_input_or_reserved_eval70_read": True,
    })
    print(f"Direct contour fit best step {best_step}, loss {best_loss:.6f}")


def smoke(use_cuda: bool) -> None:
    """Check fixed forward/loss gradients on CPU, and initial CUDA arithmetic if available."""
    torch.manual_seed(0)
    cpu = DirectContourMLP().cpu()
    x = torch.randn(4, 209, dtype=torch.float32)
    y = torch.randn(4, 1024, dtype=torch.float32)
    cpu_out = cpu(x)
    cpu_loss = contour_loss(cpu_out, y).mean()
    cpu_loss.backward()
    if not torch.isfinite(cpu_loss) or any(
        p.grad is None or not torch.isfinite(p.grad).all() for p in cpu.parameters()
    ):
        raise ValueError("synthetic CPU contour forward/gradient is nonfinite")
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA smoke requested but CUDA is unavailable")
        gpu = DirectContourMLP().cuda()
        gpu.load_state_dict(cpu.state_dict())
        gpu_out = gpu(x.cuda())
        gpu_loss = contour_loss(gpu_out, y.cuda()).mean()
        gpu_loss.backward()
        if not torch.allclose(cpu_out, gpu_out.cpu(), rtol=1e-4, atol=1e-5):
            raise ValueError("initial CPU/CUDA forward arithmetic differs beyond tolerance")
        if not torch.allclose(cpu_loss, gpu_loss.cpu(), rtol=1e-4, atol=1e-5):
            raise ValueError("initial CPU/CUDA loss arithmetic differs beyond tolerance")
        for (name, left), (_, right) in zip(cpu.named_parameters(),
                                            gpu.named_parameters(), strict=True):
            if left.grad is None or right.grad is None or not torch.allclose(
                left.grad, right.grad.cpu(), rtol=3e-4, atol=3e-5
            ):
                raise ValueError(f"initial CPU/CUDA gradient differs for {name}")
    print(f"Direct contour synthetic {'CPU+CUDA' if use_cuda else 'CPU'} "
          f"forward/gradient smoke passed, loss {cpu_loss.item():.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "assemble", "fit"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    if args.stage == "smoke":
        smoke(args.cuda)
    else:
        if args.run is None:
            raise ValueError("--run is required for assemble or fit")
        run = args.run.resolve()
        if args.stage == "assemble":
            assemble(run)
        else:
            fit(run)


if __name__ == "__main__":
    main()

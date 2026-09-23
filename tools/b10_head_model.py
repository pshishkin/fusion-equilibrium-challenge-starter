"""Matched waveform residual heads for the bounded B10 information gate."""

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
from torch import nn

from toolkit.records import file_hash, write_json

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "experiments/b10-two-rate-information-gate-v4.json"


class WaveformResidualNet(nn.Module):
    """Identical capacity for raw and EFIT-reconstructed waveform windows."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        required = {"conv_channels", "kernel_size", "stride", "embedding_size",
                    "hidden_sizes", "activation", "zero_output_init"}
        if set(config) != required:
            raise ValueError(f"B10 model YAML keys {sorted(config)}, expected {sorted(required)}")
        channels = int(config["conv_channels"])
        kernel = int(config["kernel_size"])
        stride = int(config["stride"])
        embedding = int(config["embedding_size"])
        hidden_sizes = list(config["hidden_sizes"])
        if (channels <= 0 or kernel <= 0 or kernel % 2 != 1 or stride <= 0
                or embedding <= 0 or len(hidden_sizes) != 2
                or any(int(size) <= 0 for size in hidden_sizes)
                or config["activation"] != "gelu"
                or config["zero_output_init"] is not True):
            raise ValueError(f"B10 model YAML has invalid architecture values {config}")
        padding = kernel // 2
        self.depth1 = nn.Conv1d(43, 43, kernel_size=kernel, stride=stride,
                                padding=padding, groups=43)
        self.point1 = nn.Conv1d(43, channels, kernel_size=1)
        self.depth2 = nn.Conv1d(channels, channels, kernel_size=kernel,
                                stride=stride, padding=padding, groups=channels)
        self.point2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.project = nn.Linear(2 * channels, embedding)
        self.hidden1 = nn.Linear(84 + 52 + embedding, int(hidden_sizes[0]))
        self.hidden2 = nn.Linear(int(hidden_sizes[0]), int(hidden_sizes[1]))
        self.output = nn.Linear(int(hidden_sizes[1]), 52)
        self.activation = nn.GELU()
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, frame84: torch.Tensor, base52: torch.Tensor,
                window: torch.Tensor, mask: torch.Tensor,
                rates: torch.Tensor) -> torch.Tensor:
        batch = frame84.shape[0]
        if (frame84.shape != (batch, 84) or base52.shape != (batch, 52)
                or window.shape != (batch, 41, 21)
                or mask.shape != (batch, 41) or rates.shape != (batch, 21)):
            raise ValueError(f"B10 batch frame/base/window/mask/rates "
                             f"{frame84.shape}/{base52.shape}/{window.shape}/"
                             f"{mask.shape}/{rates.shape}, expected fixed 84/52/41x21/41/21")
        signal = window.transpose(1, 2)
        rate_channel = rates[:, :, None].expand(-1, -1, 41)
        combined = torch.cat([signal, rate_channel, mask[:, None, :]], dim=1)
        hidden = self.activation(self.point1(self.depth1(combined)))
        hidden = self.activation(self.point2(self.depth2(hidden)))
        pooled = torch.cat([hidden.mean(dim=2), hidden.amax(dim=2)], dim=1)
        embedding = self.activation(self.project(pooled))
        state = torch.cat([frame84, base52, embedding], dim=1)
        state = self.activation(self.hidden1(state))
        state = self.activation(self.hidden2(state))
        return self.output(state)


def decode_residual(source_psi: np.ndarray, source_q95: np.ndarray,
                    source_betaN: np.ndarray, base_scaled: np.ndarray,
                    predicted_scaled_residual: np.ndarray, artifact: dict[str, Any]
                    ) -> dict[str, np.ndarray]:
    """Add a scaled residual through the base artifact's physical PCA basis."""
    psi = np.asarray(source_psi)
    q95 = np.asarray(source_q95)
    beta = np.asarray(source_betaN)
    base = np.asarray(base_scaled, dtype=np.float64)
    residual = np.asarray(predicted_scaled_residual, dtype=np.float64)
    frames = len(psi)
    if (psi.shape != (frames, 65, 65) or psi.dtype != np.float32
            or q95.shape != (frames,) or beta.shape != (frames,)
            or q95.dtype != np.float32 or beta.dtype != np.float32
            or base.shape != (frames, 52) or residual.shape != base.shape
            or not np.isfinite(psi).all() or not np.isfinite(q95).all()
            or not np.isfinite(beta).all() or not np.isfinite(base).all()
            or not np.isfinite(residual).all()):
        raise ValueError(f"B10 decoder source psi/q95/betaN/base/residual "
                         f"{psi.shape}/{q95.shape}/{beta.shape}/{base.shape}/"
                         f"{residual.shape}, expected finite float32 source and 52 targets")
    if np.count_nonzero(residual) == 0:
        return {"psirz": psi.copy(), "q95": q95.copy(), "betaN": beta.copy()}
    scaler = artifact["target_scaler"]
    physical_base = scaler.inverse_transform(base)
    physical_corrected = scaler.inverse_transform(base + residual)
    change = physical_corrected[:, :50] - physical_base[:, :50]
    basis = np.asarray(artifact["pca"].pca.components_, dtype=np.float64)
    if basis.shape != (50, 65 * 65):
        raise ValueError(f"PCA residual basis {basis.shape}, expected (50,4225)")
    map_change = (change @ basis).reshape(frames, 65, 65)
    result = {"psirz": (psi.astype(np.float64) + map_change).astype(np.float32),
              "q95": np.asarray(physical_corrected[:, 50], dtype=np.float32),
              "betaN": np.asarray(physical_corrected[:, 51], dtype=np.float32)}
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError("B10 decoded map/scalar residual contains nonfinite values")
    return result


def _atomic_checkpoint(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"{path}: B10 residual head already fitted")
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".pt")
    try:
        os.close(fd)
        torch.save(value, temporary)
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def train(campaign: Path, arm: str, seed: int) -> None:
    """Fit one frozen last-step residual head with equal-shot frame sampling."""
    started = time.perf_counter()
    campaign = campaign.resolve()
    if ROOT.resolve() != (campaign / "source").resolve():
        raise ValueError(f"head code imported from {ROOT}, expected {campaign / 'source'}")
    spec = json.loads(SPEC.read_text())
    if file_hash(Path(__file__)) != spec["head_helper_sha256"]:
        raise ValueError(f"{campaign}: frozen head source SHA differs")
    params_path = campaign / "source/baselines/b10-two-rate-info-gate.params.yaml"
    if file_hash(params_path) != spec["effective_params_sha256"]:
        raise ValueError(f"{params_path}: frozen B10 model YAML SHA differs")
    params = yaml.safe_load(params_path.read_text())
    train_cfg = params["train"]
    required_train = {"batch_size", "max_steps", "learning_rate", "weight_decay",
                      "gradient_clip", "log_every_steps", "torch_threads", "device"}
    if set(train_cfg) != required_train:
        raise ValueError(f"{params_path}: training keys {sorted(train_cfg)}, "
                         f"expected {sorted(required_train)}")
    if train_cfg["device"] != "cuda":
        raise ValueError(f"{params_path}: B10 device {train_cfg['device']}, expected cuda")
    expected = {("lowpass", 0), ("lowpass", 1), ("raw", 0)}
    if (arm, seed) not in expected:
        raise ValueError(f"B10 arm/seed {arm}/{seed}, expected {sorted(expected)}")
    assembled_path = campaign / "assembly.json"
    assembled = json.loads(assembled_path.read_text())
    if (assembled["train_shots"], assembled["train_frames"]) != (1355, 300793):
        raise ValueError(f"{assembled_path}: shots/frames differ from 1355/300793")
    if not torch.cuda.is_available():
        raise RuntimeError("B10 configured device=cuda but CUDA is unavailable")
    torch.set_num_threads(int(train_cfg["torch_threads"]))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = WaveformResidualNet(params["model"]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(train_cfg["learning_rate"]),
                                  weight_decay=float(train_cfg["weight_decay"]))
    arrays = {name: np.load(campaign / "fit_data" / f"{name}.npy", mmap_mode="r")
              for name in ("x84", "base52", arm, "mask", "rates", "target52")}
    offsets = np.asarray(assembled["shot_offsets"], dtype=np.int64)
    if len(offsets) != 1356 or int(offsets[-1]) != 300793:
        raise ValueError(f"{assembled_path}: shot offsets {len(offsets)}/"
                         f"{int(offsets[-1])}, expected 1356/300793")
    trace: list[dict[str, float]] = []
    fit_started = time.perf_counter()
    batch_size = int(train_cfg["batch_size"])
    max_steps = int(train_cfg["max_steps"])
    for step in range(1, max_steps + 1):
        shots = rng.integers(0, 1355, size=batch_size)
        sampled = offsets[shots] + np.floor(rng.random(batch_size) * (
            offsets[shots + 1] - offsets[shots])).astype(np.int64)
        batch = {name: torch.from_numpy(np.array(array[sampled], copy=True)).cuda()
                 for name, array in arrays.items()}
        optimizer.zero_grad(set_to_none=True)
        predicted = model(batch["x84"], batch["base52"], batch[arm],
                          batch["mask"], batch["rates"])
        loss = torch.mean((predicted - batch["target52"]) ** 2)
        if not torch.isfinite(loss):
            raise ValueError(f"B10 {arm} seed{seed} step {step}: nonfinite loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),
                                 max_norm=float(train_cfg["gradient_clip"]))
        optimizer.step()
        if step % int(train_cfg["log_every_steps"]) == 0:
            value = float(loss.item())
            trace.append({"step": float(step), "train_loss": value})
            print(f"B10 {arm} seed{seed} step {step}/{max_steps} train {value:.6f}",
                  flush=True)
    fit_seconds = time.perf_counter() - fit_started
    path = campaign / "models" / f"{arm}_seed{seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_checkpoint(path, {"state_dict": model.cpu().state_dict(), "arm": arm,
                              "seed": seed, "steps": max_steps,
                              "assembly_sha256": file_hash(assembled_path),
                              "spec_sha256": file_hash(SPEC),
                              "params_sha256": file_hash(params_path)})
    write_json(path.with_suffix(".json"), {
        "classification": "fixed_B10_matched_residual_head_last_step",
        "arm": arm, "seed": seed, "steps": max_steps,
        "artifact_sha256": file_hash(path),
        "params_sha256": file_hash(params_path),
        "assembly_sha256": file_hash(assembled_path),
        "train_shots": 1355, "train_frames": 300793,
        "fit_wall_seconds": fit_seconds,
        "total_wall_seconds": time.perf_counter() - started,
        "trace": trace,
        "validation_or_gate_targets_read": False})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--arm", choices=("lowpass", "raw"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    train(args.campaign, args.arm, args.seed)


if __name__ == "__main__":
    main()

"""One physical-flux residual U-Net for the sealed OOF transfer pilot."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.nn import functional as F

FloatArray = npt.NDArray[np.floating[Any]]


class ConvBlock(nn.Module):
    """Two fixed 3x3 group-normalized convolutions."""

    def __init__(self, input_channels: int, output_channels: int, groups: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels), nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels), nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class MapResidualUNet(nn.Module):
    """Zero-initialized spatial correction with source-independent tabular FiLM."""

    def __init__(self, model_cfg: dict[str, Any], feature_width: int) -> None:
        super().__init__()
        expected = {"input_channels", "encoder_channels", "norm", "norm_groups",
                    "activation", "conv_kernel", "downsample", "upsample",
                    "feature_embedding", "bottleneck_film", "output_channels",
                    "zero_output_init"}
        if set(model_cfg) != expected:
            raise ValueError(f"U-Net model keys {sorted(model_cfg)}, expected {sorted(expected)}")
        channels = [int(value) for value in model_cfg["encoder_channels"]]
        groups = int(model_cfg["norm_groups"])
        if (model_cfg["input_channels"] != 4 or channels != [16, 32, 64, 128]
                or feature_width != 86 or model_cfg["norm"] != "group_norm"
                or groups != 8 or model_cfg["activation"] != "gelu"
                or model_cfg["conv_kernel"] != 3
                or model_cfg["downsample"] != "average_pool_2"
                or model_cfg["upsample"] != "bilinear_align_corners_false_to_skip_shape"
                or model_cfg["feature_embedding"] != 128
                or model_cfg["bottleneck_film"] is not True
                or model_cfg["output_channels"] != 1
                or model_cfg["zero_output_init"] is not True):
            raise ValueError(f"U-Net architecture differs from fixed YAML: {model_cfg}")
        self.enc0 = ConvBlock(4, 16, groups)
        self.enc1 = ConvBlock(16, 32, groups)
        self.enc2 = ConvBlock(32, 64, groups)
        self.enc3 = ConvBlock(64, 128, groups)
        self.film = nn.Sequential(nn.Linear(feature_width, 128), nn.GELU(),
                                  nn.Linear(128, 256))
        self.dec2 = ConvBlock(128 + 64, 64, groups)
        self.dec1 = ConvBlock(64 + 32, 32, groups)
        self.dec0 = ConvBlock(32 + 16, 16, groups)
        self.output = nn.Conv2d(16, 1, 1)
        nn.init.zeros_(self.output.weight)
        assert self.output.bias is not None
        nn.init.zeros_(self.output.bias)

    def forward(self, map_channels: torch.Tensor,
                features86: torch.Tensor) -> torch.Tensor:
        n = len(map_channels)
        if map_channels.shape != (n, 4, 65, 65) or features86.shape != (n, 86):
            raise ValueError(f"U-Net map/features {tuple(map_channels.shape)}/"
                             f"{tuple(features86.shape)}, expected ({n},4,65,65)/({n},86)")
        a = self.enc0(map_channels)
        b = self.enc1(F.avg_pool2d(a, 2))
        c = self.enc2(F.avg_pool2d(b, 2))
        d = self.enc3(F.avg_pool2d(c, 2))
        gamma, beta = self.film(features86).chunk(2, dim=1)
        d = d * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        d = self.dec2(torch.cat([F.interpolate(d, size=c.shape[-2:], mode="bilinear",
                                                align_corners=False), c], dim=1))
        d = self.dec1(torch.cat([F.interpolate(d, size=b.shape[-2:], mode="bilinear",
                                                align_corners=False), b], dim=1))
        d = self.dec0(torch.cat([F.interpolate(d, size=a.shape[-2:], mode="bilinear",
                                                align_corners=False), a], dim=1))
        return self.output(d)


def grid_channels(psi: np.ndarray, grid_r: np.ndarray, grid_z: np.ndarray,
                  envelope: np.ndarray, min_ulps: int,
                  location: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize using source-only flux range and return exact physical output scale."""
    arr = np.asarray(psi)
    r = np.asarray(grid_r, dtype=np.float64)
    z = np.asarray(grid_z, dtype=np.float64)
    mask = np.asarray(envelope)
    frames = len(arr)
    if (arr.shape != (frames, 65, 65) or arr.dtype != np.float32
            or not np.isfinite(arr).all() or r.shape != (65,) or z.shape != (65,)
            or mask.shape != (65, 65) or mask.dtype != np.bool_
            or not np.isfinite(r).all() or not np.isfinite(z).all()
            or np.any(np.diff(r) <= 0) or np.any(np.diff(z) <= 0)
            or min_ulps != 1024):
        raise ValueError(f"{location}: source/grid/mask shape, dtype, finite or "
                         "monotonicity differs from 65x65 float32 DIII-D and 1024 ULPs")
    physical = arr.astype(np.float64)
    minimum = np.min(physical, axis=(1, 2))
    maximum = np.max(physical, axis=(1, 2))
    scale = maximum - minimum
    required = min_ulps * np.finfo(np.float32).eps * np.maximum(
        1.0, np.maximum(np.abs(minimum), np.abs(maximum)))
    bad = np.flatnonzero(~np.isfinite(scale) | (scale <= required))
    if len(bad):
        i = int(bad[0])
        raise ValueError(f"{location}: frame {i} source flux range {scale[i]:.9g}, "
                         f"required > {required[i]:.9g} from 1024 float32 ULPs")
    normalized = ((physical - minimum[:, None, None]) / scale[:, None, None]).astype(np.float32)
    rr = np.broadcast_to((2 * (r - r[0]) / (r[-1] - r[0]) - 1)[None, :], (65, 65))
    zz = np.broadcast_to((2 * (z - z[0]) / (z[-1] - z[0]) - 1)[:, None], (65, 65))
    channels = np.stack([normalized,
                         np.broadcast_to(rr.astype(np.float32), normalized.shape),
                         np.broadcast_to(zz.astype(np.float32), normalized.shape),
                         np.broadcast_to(mask.astype(np.float32), normalized.shape)], axis=1)
    extra = np.column_stack([minimum, np.log(scale)]).astype(np.float32)
    if not np.isfinite(channels).all() or not np.isfinite(extra).all():
        raise ValueError(f"{location}: normalized source channels or minimum/log-range nonfinite")
    return channels, extra, scale


def apply_delta(source: np.ndarray, normalized_delta: np.ndarray,
                source_scale: np.ndarray, location: str) -> np.ndarray:
    """Add a physical residual with one cast, with a strict zero-bit identity path."""
    psi = np.asarray(source)
    delta = np.asarray(normalized_delta)
    scale = np.asarray(source_scale)
    frames = len(psi)
    if (psi.shape != (frames, 65, 65) or psi.dtype != np.float32
            or delta.shape != psi.shape or delta.dtype != np.float32
            or scale.shape != (frames,) or not np.isfinite(psi).all()
            or not np.isfinite(delta).all() or not np.isfinite(scale).all()
            or np.any(scale <= 0)):
        raise ValueError(f"{location}: source/delta/scale shape, dtype or finite mismatch "
                         f"{psi.shape}/{delta.shape}/{scale.shape}")
    if not np.count_nonzero(delta):
        return psi.copy()
    out = (psi.astype(np.float64)
           + scale[:, None, None].astype(np.float64) * delta.astype(np.float64))
    if not np.isfinite(out).all():
        raise ValueError(f"{location}: physical residual addition produced nonfinite flux")
    return out.astype(np.float32)


def frame_energies(error: np.ndarray, weight: np.ndarray,
                   grid_r: np.ndarray, grid_z: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact value and physical-gradient terms for train-only normalization."""
    e = np.asarray(error, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    r = np.asarray(grid_r, dtype=np.float64)
    z = np.asarray(grid_z, dtype=np.float64)
    if (e.ndim != 3 or e.shape[1:] != (65, 65) or w.shape != (65, 65)
            or r.shape != (65,) or z.shape != (65,)
            or not np.isfinite(e).all() or not np.isfinite(w).all()
            or np.any(w <= 0) or np.any(np.diff(r) <= 0) or np.any(np.diff(z) <= 0)):
        raise ValueError(f"energy error/weight/grid {e.shape}/{w.shape}/{r.shape}/{z.shape}, "
                         "expected finite T/65/65, 65/65, 65, 65 with positive steps")
    value = np.mean(w[None] * e * e, axis=(1, 2))
    dr = (e[:, :, 1:] - e[:, :, :-1]) / np.diff(r)[None, None, :]
    dz = (e[:, 1:, :] - e[:, :-1, :]) / np.diff(z)[None, :, None]
    wr = (w[:, 1:] + w[:, :-1]) / 2
    wz = (w[1:, :] + w[:-1, :]) / 2
    gradient = 0.5 * (np.mean(wr[None] * dr * dr, axis=(1, 2))
                      + np.mean(wz[None] * dz * dz, axis=(1, 2)))
    return value, gradient


def torch_loss(predicted: torch.Tensor, target: torch.Tensor,
               weight: torch.Tensor, grid_r: torch.Tensor, grid_z: torch.Tensor,
               tau_value: float, tau_gradient: float) -> torch.Tensor:
    """One fixed equally normalized value-plus-gradient residual objective."""
    if (predicted.shape != target.shape or predicted.ndim != 4
            or predicted.shape[1:] != (1, 65, 65)
            or weight.shape != (65, 65) or grid_r.shape != (65,)
            or grid_z.shape != (65,) or not np.isfinite(tau_value)
            or not np.isfinite(tau_gradient)
            or tau_value <= 0 or tau_gradient <= 0):
        raise ValueError(f"loss tensors {tuple(predicted.shape)}/{tuple(target.shape)}/"
                         f"{tuple(weight.shape)}, tau {tau_value}/{tau_gradient}, "
                         "expected B/1/65/65, 65/65 and positive normalizers")
    error = predicted - target
    value = torch.mean(weight[None, None] * error.square())
    dr = (error[:, :, :, 1:] - error[:, :, :, :-1]) / torch.diff(grid_r)[None, None, None, :]
    dz = (error[:, :, 1:, :] - error[:, :, :-1, :]) / torch.diff(grid_z)[None, None, :, None]
    wr = (weight[:, 1:] + weight[:, :-1]) / 2
    wz = (weight[1:, :] + weight[:-1, :]) / 2
    gradient = 0.5 * (torch.mean(wr[None, None] * dr.square())
                      + torch.mean(wz[None, None] * dz.square()))
    return value / tau_value + gradient / tau_gradient

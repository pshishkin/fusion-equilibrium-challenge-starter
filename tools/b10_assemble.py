"""Assemble bounded B10 memmaps and fit one shared train-only raw scaler."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from toolkit.records import file_hash, write_json

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "experiments/b10-two-rate-information-gate-v4.json"
SHAPES = {
    "x84": (84,), "base52": (52,), "target52": (52,),
    "raw": (41, 21), "lowpass": (41, 21), "mask": (41,), "rates": (21,),
}


def assemble(campaign: Path) -> None:
    """Only the 1355 genuinely OOF training shots contribute statistics or targets."""
    started = time.perf_counter()
    campaign = campaign.resolve()
    if ROOT.resolve() != (campaign / "source").resolve():
        raise ValueError(f"assembly imported from {ROOT}, expected {campaign / 'source'}")
    spec = json.loads(SPEC.read_text())
    if file_hash(Path(__file__)) != spec["assembly_helper_sha256"]:
        raise ValueError(f"{campaign}: assembly helper SHA differs")
    manifest = json.loads((campaign / "manifest.json").read_text())
    sidecars = json.loads((campaign / "waveform_sidecars_ready.json").read_text())
    targets = json.loads((campaign / "targets_and_base_ready.json").read_text())
    if (sidecars["shots"] != targets["train_shots"] + targets["gate_shots"]
            or targets["train_shots"] != 1355 or targets["train_frames"] != 300793
            or manifest["head_train_files"] != [row["shot"] for row in targets["entries"][:1355]]
            or manifest["head_train_files"] != [row["shot"] for row in sidecars["entries"][:1355]]):
        raise ValueError(f"{campaign}: sidecar/target/manifest training identities differ")
    output = campaign / "assembly.json"
    folder = campaign / "fit_data"
    if output.exists() or folder.exists():
        raise FileExistsError(f"{campaign}: B10 assembly already started or completed")
    folder.mkdir(parents=True)
    arrays: dict[str, np.memmap] = {}
    for name, shape in SHAPES.items():
        arrays[name] = open_memmap(folder / f"{name}.npy.tmp", mode="w+", dtype=np.float32,
                                   shape=(300793, *shape))
    offsets = [0]
    sum_raw = np.zeros(21, dtype=np.float64)
    count_raw = 0
    for i in range(1355):
        source = sidecars["entries"][i]
        target = targets["entries"][i]
        if (source["shot"] != target["shot"]
                or source["frames"] != target["frames"]
                or file_hash(Path(source["sidecar_path"])) != source["sidecar_sha256"]
                or file_hash(Path(target["output_path"])) != target["output_sha256"]):
            raise ValueError(f"{source['shot']}: frozen sidecar/target hash or frames differ")
        frames = source["frames"]
        lo, hi = offsets[-1], offsets[-1] + frames
        with (np.load(source["sidecar_path"], allow_pickle=False) as windows,
              np.load(target["output_path"], allow_pickle=False) as fitted):
            for name in SHAPES:
                if name == "target52":
                    value = np.asarray(fitted["target_scaled52"])
                elif name in ("x84", "base52"):
                    value = np.asarray(fitted["base_scaled52" if name == "base52" else "x84"])
                elif name == "rates":
                    value = np.broadcast_to(windows["rates"], (frames, 21))
                else:
                    value = np.asarray(windows[name])
                if (value.shape != (frames, *SHAPES[name])
                        or value.dtype != np.float32 or not np.isfinite(value).all()):
                    raise ValueError(f"{source['shot']}: {name} shape/dtype/finiteness "
                                     f"{value.shape}/{value.dtype}/"
                                     f"{int((~np.isfinite(value)).sum())}, "
                                     f"expected ({frames},{SHAPES[name]})/float32/0")
                arrays[name][lo:hi] = value
            sum_raw += np.sum(np.asarray(windows["raw"], dtype=np.float64), axis=(0, 1))
            count_raw += frames * 41
        offsets.append(hi)
        if (i + 1) % 100 == 0:
            print(f"B10 memmap assembly {i + 1}/1355 OOF shots", flush=True)
    if offsets[-1] != 300793 or count_raw != 300793 * 41:
        raise ValueError(f"{campaign}: assembled frames/raw bins {offsets[-1]}/{count_raw}, "
                         "expected 300793/12332513")
    mean = sum_raw / count_raw
    sum_sq = np.zeros(21, dtype=np.float64)
    for lo in range(0, 300793, 5000):
        hi = min(lo + 5000, 300793)
        raw = np.asarray(arrays["raw"][lo:hi], dtype=np.float64)
        sum_sq += np.sum((raw - mean) ** 2, axis=(0, 1))
    std = np.sqrt(sum_sq / count_raw)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError(f"{campaign}: raw train-only scaler constant/nonfinite channel "
                         f"{np.flatnonzero((std <= 0) | ~np.isfinite(std)).tolist()}")
    for lo in range(0, 300793, 5000):
        hi = min(lo + 5000, 300793)
        for arm in ("raw", "lowpass"):
            normalized = (np.asarray(arrays[arm][lo:hi], dtype=np.float64) - mean) / std
            if not np.isfinite(normalized).all():
                raise ValueError(f"{campaign}: {arm} normalized frames {lo}:{hi} nonfinite")
            arrays[arm][lo:hi] = normalized.astype(np.float32)
    for array in arrays.values():
        array.flush()
    del arrays
    hashes: dict[str, str] = {}
    for name in SHAPES:
        temporary = folder / f"{name}.npy.tmp"
        final = folder / f"{name}.npy"
        os.replace(temporary, final)
        hashes[name] = file_hash(final)
    write_json(output, {
        "classification": "B10_train_only_shared_raw_scaler_and_matched_memmaps_ready",
        "campaign_manifest_sha256": file_hash(campaign / "manifest.json"),
        "sidecar_ready_sha256": file_hash(campaign / "waveform_sidecars_ready.json"),
        "target_ready_sha256": file_hash(campaign / "targets_and_base_ready.json"),
        "train_shots": 1355, "train_frames": 300793,
        "shot_offsets": offsets,
        "raw_channel_mean_train_only": mean.tolist(),
        "raw_channel_std_train_only": std.tolist(),
        "raw_signal_samples_for_scaling": count_raw,
        "npy_sha256": hashes,
        "gate_or_tail_target_read_for_scaling": False,
        "wall_seconds": time.perf_counter() - started,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.campaign)


if __name__ == "__main__":
    main()

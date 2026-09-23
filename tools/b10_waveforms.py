"""Input-only, two-clock 0.5 ms waveform sidecars for the bounded B10 gate."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments import D3D_MAGNETICS_SIGNALS, interpolate_magnetics_to_efit
from my_experiments import baseline_model
from toolkit.parallel import pimap, release
from toolkit.records import file_hash, write_json
from tools.b10_raw_support import INPUT_COLUMNS, classify_clock

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "experiments/b10-two-rate-information-gate-v4.json"
COMMON = D3D_MAGNETICS_SIGNALS[:-1]
if len(COMMON) != 20 or D3D_MAGNETICS_SIGNALS[-1] != "plasma_current":
    raise ValueError(f"DIII-D common magnetics/Ip order "
                     f"{len(COMMON)}/{D3D_MAGNETICS_SIGNALS[-1]}, "
                     "expected 20 common channels followed by plasma_current")


def trapezoid_bins(times: np.ndarray, values: np.ndarray, centres: np.ndarray,
                   half_bin: float) -> np.ndarray:
    """Integrate linearly interpolated raw traces across fixed physical time bins."""
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    q = np.asarray(centres, dtype=np.float64)
    if (t.ndim != 1 or v.ndim != 2 or len(v) != len(t) or len(t) < 2
            or q.ndim != 1 or not np.isfinite(t).all() or not np.isfinite(v).all()
            or not np.isfinite(q).all() or not np.all(np.diff(t) > 0)
            or not np.isfinite(half_bin) or half_bin <= 0):
        raise ValueError(f"raw trapezoid axis/values/centres/half-bin "
                         f"{t.shape}/{v.shape}/{q.shape}/{half_bin}, expected finite "
                         "strict axis, matched 2D values and positive half-bin")
    left = q - half_bin
    right = q + half_bin
    if left[0] < t[0] or right[-1] > t[-1]:
        raise ValueError(f"requested bin bounds {left[0]}..{right[-1]} ms, "
                         f"raw support {t[0]}..{t[-1]} ms")
    dt = np.diff(t)
    increments = 0.5 * (v[:-1] + v[1:]) * dt[:, None]
    cumulative = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(increments, axis=0)])

    def integral_at(points: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(t, points, side="right") - 1
        idx = np.minimum(idx, len(t) - 2)
        delta = points - t[idx]
        next_value = v[idx] + (v[idx + 1] - v[idx]) * (delta / dt[idx])[:, None]
        return cumulative[idx] + 0.5 * (v[idx] + next_value) * delta[:, None]

    result = (integral_at(right) - integral_at(left)) / (2.0 * half_bin)
    if result.shape != (len(q), v.shape[1]) or not np.isfinite(result).all():
        raise ValueError(f"integrated waveform shape/finiteness {result.shape}/"
                         f"{int((~np.isfinite(result)).sum())}, "
                         f"expected ({len(q)},{v.shape[1]})/0")
    return result.astype(np.float32)


def cadence_control(efit_times: np.ndarray, sampled: np.ndarray,
                    query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate only EFIT-rate values; pad its edges and expose the mask."""
    t = np.asarray(efit_times, dtype=np.float64)
    v = np.asarray(sampled, dtype=np.float64)
    q = np.asarray(query, dtype=np.float64)
    if (t.ndim != 1 or len(t) == 0 or v.shape != (len(t), 21)
            or q.ndim != 2 or q.shape[0] != len(t) or not np.isfinite(t).all()
            or not np.isfinite(v).all() or not np.isfinite(q).all()
            or (len(t) > 1 and not np.all(np.diff(t) > 0))):
        raise ValueError(f"cadence control times/values/queries "
                         f"{t.shape}/{v.shape}/{q.shape}, expected finite (T,)/(T,21)/(T,41)")
    flat = q.ravel()
    window = np.column_stack([np.interp(flat, t, v[:, channel])
                              for channel in range(21)])
    result = window.reshape(len(t), q.shape[1], 21).astype(np.float32)
    supported = ((q >= t[0]) & (q <= t[-1])).astype(np.float32)
    return result, supported


def _atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"{path}: sidecar already ready")
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


def _one(task: tuple[str, str, str, int, float, str, dict[str, float], str]
         ) -> dict[str, Any]:
    name, window, data_s, expected_frames, expected_rate, expected_sha, numeric, out_s = task
    path = Path(data_s) / "data/diii_d_train" / name
    if file_hash(path) != expected_sha:
        raise ValueError(f"{path}: source parquet SHA differs from complete preflight")
    row = pd.read_parquet(path, columns=INPUT_COLUMNS).iloc[0]
    shot = baseline_model.inputs_only_shot(row)
    efit = np.asarray(shot["efit_times"], dtype=np.float64)
    if len(efit) != expected_frames:
        raise ValueError(f"{path}: EFIT frames {len(efit)}, expected {expected_frames}")
    shared = np.asarray(shot["magnetics"]["ECOILA"]["times"], dtype=np.float64)
    ip = np.asarray(shot["magnetics"]["plasma_current"]["times"], dtype=np.float64)
    rate, _, _ = classify_clock(
        shared, f"{path}: common magnetics", numeric["sample_rate_tolerance"],
        numeric["maximum_step_factor"])
    if rate != expected_rate:
        raise ValueError(f"{path}: common rate class {rate}, expected preflight {expected_rate}")
    centres = np.arange(-10.0, 10.0 + 0.25, 0.5)
    query = efit[:, None] + centres[None, :]
    flat = query.ravel()
    shared_values = np.column_stack([
        np.asarray(shot["magnetics"][signal]["values"], dtype=np.float64)
        for signal in COMMON
    ])
    raw_common = trapezoid_bins(shared, shared_values, flat,
                                numeric["half_bin_ms"]).reshape(len(efit), 41, 20)
    ip_values = np.asarray(shot["magnetics"]["plasma_current"]["values"],
                           dtype=np.float64)[:, None]
    raw_ip = trapezoid_bins(ip, ip_values, flat, numeric["half_bin_ms"])
    raw = np.concatenate([raw_common, raw_ip.reshape(len(efit), 41, 1)], axis=2)
    sampled = interpolate_magnetics_to_efit(shot)
    control, supported = cadence_control(efit, sampled, query)
    if (raw.shape != control.shape or raw.shape != (len(efit), 41, 21)
            or supported.shape != (len(efit), 41)
            or not np.isfinite(raw).all() or not np.isfinite(control).all()):
        raise ValueError(f"{path}: raw/control/support shape/finiteness "
                         f"{raw.shape}/{control.shape}/{supported.shape}, "
                         f"expected ({len(efit)},41,21)/({len(efit)},41)")
    rates = np.zeros(21, dtype=np.float32)
    rates[:20] = np.float32(np.log10(0.5 / rate))
    output = Path(out_s) / window / f"{Path(name).stem}.npz"
    _atomic(output, {"raw": raw, "lowpass": control, "mask": supported, "rates": rates})
    return {"shot": name, "window": window, "frames": len(efit),
            "source_sha256": expected_sha, "sidecar_path": str(output),
            "sidecar_sha256": file_hash(output), "common_rate_class_ms": rate,
            "efit_edge_padded_bins": int(np.sum(supported == 0))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 24:
        raise ValueError(f"jobs={args.jobs}, expected 1..24")
    campaign = args.campaign.resolve()
    meta = json.loads((campaign / "manifest.json").read_text())
    origin = Path(meta["origin_root"])
    if ROOT.resolve() != (campaign / "source").resolve():
        raise ValueError(f"waveform producer imported from {ROOT}, "
                         f"expected {campaign / 'source'}")
    for relative, expected in meta["source_sha256"].items():
        path = campaign / "source" / relative
        if file_hash(path) != expected:
            raise ValueError(f"{path}: frozen waveform source SHA differs")
    spec = json.loads((campaign / "source/experiments/b10-two-rate-information-gate-v4.json")
                      .read_text())
    if file_hash(Path(__file__)) != spec["waveform_helper_sha256"]:
        raise ValueError(f"{campaign}: frozen waveform helper SHA differs")
    audit = json.loads((origin / spec["support_audit"]["path"]).read_text())
    if file_hash(origin / spec["support_audit"]["path"]) != spec["support_audit"]["sha256"]:
        raise ValueError(f"{campaign}: full raw support audit SHA differs")
    if not audit["raw_support_complete"] or len(audit["per_shot"]) != 1555:
        raise ValueError(f"{campaign}: raw support audit is incomplete or failed")
    if (audit["train_shots"], audit["gate_shots"]) != (1355, 200):
        raise ValueError(f"{campaign}: raw support windows differ from 1355/200")
    output = campaign / "waveform_sidecars_ready.json"
    if output.exists():
        raise FileExistsError(f"{output}: raw waveform sidecars already recorded")
    records = audit["per_shot"]
    data = Path(meta["data_dir"])
    dataset = json.loads((campaign / "source/my_experiments/diii_d_dataset.json")
                         .read_text())
    numeric = spec["numeric_preflight"]
    tasks = [
        (record["shot"], "head_train" if i < 1355 else "gate", str(data),
         record["frames"], record["shared_rate_class_ms"],
         dataset["configs"]["diii_d_train"][record["shot"]][1], numeric,
         str(campaign / "waveforms"))
        for i, record in enumerate(records)
    ]
    started = time.perf_counter()
    ready: list[dict[str, Any]] = []
    try:
        for i, item in enumerate(pimap(_one, tasks, args.jobs)):
            if item["shot"] != records[i]["shot"] or item["frames"] != records[i]["frames"]:
                raise ValueError(f"{records[i]['shot']}: sidecar order/frames differ")
            if item["efit_edge_padded_bins"] != records[i][
                    "efit_edge_padded_window_bins_per_signal"]:
                raise ValueError(f"{item['shot']}: EFIT-control edge mask count "
                                 f"{item['efit_edge_padded_bins']}, expected "
                                 f"{records[i]['efit_edge_padded_window_bins_per_signal']}")
            ready.append(item)
            if (i + 1) % 100 == 0:
                print(f"B10 waveform sidecars {i + 1}/1555 shots", flush=True)
    finally:
        release()
    write_json(output, {"classification": "B10_full_input_only_raw_and_EFIT_control_sidecars",
                        "support_audit_sha256": spec["support_audit"]["sha256"],
                        "campaign_manifest_sha256": file_hash(campaign / "manifest.json"),
                        "shots": len(ready),
                        "train_frames": sum(row["frames"] for row in ready[:1355]),
                        "gate_frames": sum(row["frames"] for row in ready[1355:]),
                        "entries": ready,
                        "wall_seconds": time.perf_counter() - started,
                        "target_columns_requested": []})


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Diagnostic C6 in ideas.md — how far back does the sequence model actually reach?

    uv run python my_experiments/diagnose_memory.py --shots 40

The architecture cannot answer it. A GRU's update gate decides, per channel and per frame, how much
of the state survives one step, so the reach is a property of the trained weights and not of the
design. "In principle unbounded" is true and useless.

The measurement needs no training and no scoring. Run the fitted encoder and GRU over real shots,
collect the update gate `z` at every frame and channel, and read the time constant off it:

    h_t = (1 - z) * n + z * h_{t-1}        (PyTorch's convention: z KEEPS the old state)
    tau = -1 / ln(z) frames,  one frame = 20 ms

| z | tau | in shot time |
|---|---|---|
| 0.5 | 1.4 frames | 0.03 s |
| 0.9 | 9.5 frames | 0.19 s |
| 0.99 | 99.5 frames | 2.0 s |

Two outcomes that mean opposite things. A spread reaching seconds says the recurrence carries slow
plasma and vessel state, which is the mechanism it was built for and the one A22 is chasing.
Everything piled near z ~ 0.5 says it learned a two-frame smoother and whatever it contributes to
the ensemble comes from somewhere else.

The gates are recomputed here from the stored weights rather than read out of torch, because
`nn.GRU` returns only the output sequence — the gates are internal and there is no hook that
exposes them.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from experiments import DEFAULT_LOCAL_DATA_DIR, HF_TRAIN_CONFIG  # noqa: E402
from my_experiments.baseline_model import (  # noqa: E402
    ARTIFACT,
    FRAME_STEP_MS,
    build_inputs,
    features_for_row,
    sorted_shots,
    take_share,
)
from my_experiments.progress import install_timestamps  # noqa: E402

FloatArray = npt.NDArray[np.float64]


def _mlp_forward(state: dict[str, np.ndarray], prefix: str, x: FloatArray,
                 activation: str) -> FloatArray:
    """The stored `build_mlp` stack, in numpy. Layers are `<prefix>.<i>.0` then a bare final one."""
    acts: dict[str, Callable[[FloatArray], FloatArray]] = {"relu": lambda v: np.maximum(v, 0.0),
           "gelu": lambda v: 0.5 * v * (1.0 + np.tanh(np.sqrt(2 / np.pi)
                                                      * (v + 0.044715 * v ** 3))),
           "silu": lambda v: v / (1.0 + np.exp(-v)),
           "tanh": np.tanh}
    act = acts[activation]
    i = 0
    while f"{prefix}.{i}.0.weight" in state:
        x = act(x @ state[f"{prefix}.{i}.0.weight"].T + state[f"{prefix}.{i}.0.bias"])
        i += 1
    return x @ state[f"{prefix}.{i}.weight"].T + state[f"{prefix}.{i}.bias"]


def _gates(state: dict[str, np.ndarray], seq: FloatArray, reverse: bool) -> FloatArray:
    """(T, hidden) update gate `z` at every frame, for one direction of the GRU.

    PyTorch stacks the three gates as [r, z, n] in the first dimension of the weight matrices.
    """
    tag = "_reverse" if reverse else ""
    w_ih, w_hh = state[f"gru.weight_ih_l0{tag}"], state[f"gru.weight_hh_l0{tag}"]
    b_ih, b_hh = state[f"gru.bias_ih_l0{tag}"], state[f"gru.bias_hh_l0{tag}"]
    hidden = w_hh.shape[1]
    order = range(len(seq) - 1, -1, -1) if reverse else range(len(seq))
    h = np.zeros(hidden)
    out = np.zeros((len(seq), hidden))
    for t in order:
        gi = seq[t] @ w_ih.T + b_ih
        gh = h @ w_hh.T + b_hh
        r = 1.0 / (1.0 + np.exp(-(gi[:hidden] + gh[:hidden])))
        z = 1.0 / (1.0 + np.exp(-(gi[hidden:2 * hidden] + gh[hidden:2 * hidden])))
        n = np.tanh(gi[2 * hidden:] + r * gh[2 * hidden:])
        h = (1.0 - z) * n + z * h
        out[t] = z
    return out


def main() -> int:
    install_timestamps()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", type=int, default=40, help="held-out shots to run (default 40)")
    ap.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR)
    ap.add_argument("--config", default=HF_TRAIN_CONFIG)
    args = ap.parse_args()

    art = joblib.load(ARTIFACT)
    if "seq" not in art["models"]:
        raise SystemExit(f"{ARTIFACT} holds no sequence model, so there is no gate to read")
    model = art["models"]["seq"]
    state = model._state
    if state is None:
        raise SystemExit("the sequence model in this artifact was never fitted")

    files = take_share(sorted_shots(args.local_data_dir, args.config, int(art["split_salt"])),
                       0.01, "tail")[:args.shots]
    print(f"Reading the update gate over {len(files)} held-out shots, "
          f"{state['gru.weight_hh_l0'].shape[1]} channels per direction")

    zs: dict[bool, list[FloatArray]] = {False: [], True: []}
    for path in files:
        row = pd.read_parquet(path).iloc[0]
        feats = build_inputs(art["coil"], features_for_row(row))
        x = art["scaler"].transform(feats)
        seq = _mlp_forward(state, "enc", x, model.activation)
        for reverse in (False, True):
            zs[reverse].append(_gates(state, seq, reverse))

    print(f"\n  a frame is {FRAME_STEP_MS:.0f} ms; tau = -1/ln(z) frames\n")
    for reverse in (False, True):
        z = np.concatenate(zs[reverse])
        # Per CHANNEL, over frames: the question is what each channel decided to be, and a channel
        # that holds z steady is the one carrying a long time constant.
        per_channel = np.median(z, axis=0)
        tau = np.where(per_channel < 1.0, -1.0 / np.log(np.clip(per_channel, 1e-12, 1 - 1e-12)),
                       np.inf)
        name = "backward" if reverse else "forward "
        q = np.percentile(tau, [50, 75, 90, 99])
        print(f"  {name}  median tau {q[0]:6.2f} frames ({q[0] * FRAME_STEP_MS / 1000:.2f} s), "
              f"p75 {q[1]:6.2f}, p90 {q[2]:7.2f}, p99 {q[3]:8.2f} "
              f"({q[3] * FRAME_STEP_MS / 1000:.2f} s)")
        bands = [(0, 2, "under 2 frames (40 ms)"), (2, 10, "2-10 frames"),
                 (10, 50, "10-50 frames (0.2-1 s)"), (50, 1e9, "over 50 frames (1 s+)")]
        for lo, hi, label in bands:
            share = float(((tau >= lo) & (tau < hi)).mean())
            print(f"      {label:>26s}   {share:5.1%} of channels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

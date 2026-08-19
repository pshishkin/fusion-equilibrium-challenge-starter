#!/usr/bin/env python3
"""The only file that knows both challenges exist.

One submission enters both leaderboards, so somewhere the two machines have to meet. This is that
place, and it is deliberately the whole of it — four dispatch functions, no logic:

    DIII-D  ->  my_experiments/   a fitted model, 7041 labelled shots, 84 features per frame
    MAST    ->  mast/             a free-boundary Grad-Shafranov solve, no training data at all

Neither package imports the other. They share the vendored scorer in `fusion_scoring/` and nothing
else, which is the point: MAST has one of DIII-D's 21 magnetics signals, a flux grid that starts at
R = 0.06 m and an aspect ratio of 1.3 against 2.7, so a shared feature pipeline would be a shared
set of assumptions and every one of them is false on one side or the other.

Callers — `local_score.py`, `submission_skeleton.py`, `validate_submission.py` — import from here.
"""
from __future__ import annotations

from typing import Any

MACHINES = ("DIII-D", "MAST")


def machine_of(row: Any) -> str:
    """Which challenge this row belongs to, from its own `source` column."""
    if "source" not in row:
        raise ValueError("row has no `source` column, so there is no way to tell the machines "
                         "apart; every released config carries one")
    source = str(row["source"])
    if source not in MACHINES:
        raise ValueError(f"source {source!r} is neither of {MACHINES}")
    return source


def slim_row(row: Any) -> dict:
    """Keep only what inference reads, so a scored fold fits in memory.

    A full DIII-D row retains 101 MB against the ~6 MB inference touches, and every scored shot
    stays resident for the whole run because the metric pools R2 across the fold.
    """
    if machine_of(row) == "MAST":
        from mast.predict import slim_row as slim_mast
        return slim_mast(row)
    from my_experiments.baseline_model import slim_row as slim_d3d
    return slim_d3d(row)


def predict_row(row: Any, model: str | None = None) -> dict:
    """`{"psirz": (T, nZ, nR), "q95": (T,), "betaN": (T,)}` for one shot of either machine."""
    if machine_of(row) == "MAST":
        from mast.predict import predict_row as predict_mast
        return predict_mast(row)
    from my_experiments.baseline_model import ENSEMBLE
    from my_experiments.baseline_model import predict_row as predict_d3d
    return predict_d3d(row, "DIII-D", model if model is not None else ENSEMBLE)

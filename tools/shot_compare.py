#!/usr/bin/env python3
"""Per-shot flux error of two linear stacks: do the vessel members fix the long-ramp-down shots?

    uv run python tools/shot_compare.py runs/cv5_fixcos --a <members> --b <members> \
        --basis-from runs/cap_mlp_p100_3x512 --out runs/shotcmp_22_34

Public detail pages (2026-10-07): the leaders' worst shots are far better than ours, and our worst
(shot_0621, R2_psi 0.9858) spends 56% of its frames in the current ramp-down. On CV, per fold: the
two stacks' cross-fold linear maps (cached, `stack_delta_shrink.build_maps`), the truth, and per
shot its SSE, its own-mean SST and its ramp-down share (`error_regimes.shot_regimes`). Reported:
by ramp-down bin, the share of pooled SSE and the SSE ratio b / a; the worst shots of stack a and
what stack b does on them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cv_oof import _fold_files
from error_regimes import shot_regimes
from my_experiments import baseline_model as B
from stack_delta_shrink import FOLDS, build_maps
from toolkit.parallel import pmap


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign", type=Path)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--basis-from", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=20)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ca, cb = args.out / "a", args.out / "b"
    build_maps(args.campaign, args.a.split(","), args.basis_from, ca, args.jobs)
    build_maps(args.campaign, args.b.split(","), args.basis_from, cb, args.jobs)
    rows = []
    for f in FOLDS:
        files = _fold_files(joblib.load(args.campaign / f"fold{f}" / "baseline.joblib"))
        _, Y, _, lengths = B._read_shots(files, f"fold{f}", 1.0, args.jobs)
        if not np.array_equal(lengths, np.load(ca / f"lengths_fold{f}.npy")):
            raise ValueError(f"fold {f}: frames differ from the cached maps")
        ma = np.load(ca / f"maps_fold{f}.npy", mmap_mode="r")
        mb = np.load(cb / f"maps_fold{f}.npy", mmap_mode="r")
        reg = pmap(shot_regimes, files, args.jobs)
        bounds = np.concatenate([[0], np.cumsum(lengths)])
        for i, fl in enumerate(files):
            a, b = bounds[i], bounds[i + 1]
            y = Y[a:b].astype(np.float64)
            sst = ((y - y.mean(0)) ** 2).sum()
            rows.append({"fold": f, "shot": fl.name, "frames": int(b - a),
                         "rampdown": float((reg[i][:, 0] == 2).mean()),
                         "sse_a": float(((ma[a:b] - y) ** 2).sum()),
                         "sse_b": float(((mb[a:b] - y) ** 2).sum()), "sst": float(sst)})
        del Y
        print(f"fold {f}: {len(files)} shots", flush=True)
    (args.out / "rows.json").write_text(json.dumps(rows))
    sa = sum(r["sse_a"] for r in rows)
    sb = sum(r["sse_b"] for r in rows)
    print(f"\npooled SSE b / a = {sb / sa:.4f}  ({len(rows)} shots)")
    bins = [(0.0, 0.0), (1e-9, 0.1), (0.1, 0.3), (0.3, 1.01)]
    print(f"{'ramp-down share':>16s} {'shots':>6s} {'SSE share a':>12s} {'SSE share b':>12s} {'b/a':>7s} "
          f"{'per-shot R2 a':>14s} {'per-shot R2 b':>14s}")
    summary = []
    for lo, hi in bins:
        sel = [r for r in rows if (r["rampdown"] == 0 if hi == 0 else lo <= r["rampdown"] < hi)]
        if not sel:
            continue
        ea, eb = sum(r["sse_a"] for r in sel), sum(r["sse_b"] for r in sel)
        # single-frame shots have no own-mean spread: left out of the per-shot mean only
        r2a = np.mean([1 - r["sse_a"] / r["sst"] for r in sel if r["sst"] > 0])
        r2b = np.mean([1 - r["sse_b"] / r["sst"] for r in sel if r["sst"] > 0])
        label = "0" if hi == 0 else f"[{lo:.1f}, {min(hi, 1):.1f})"
        print(f"{label:>16s} {len(sel):6d} {ea / sa:12.3f} {eb / sb:12.3f} {eb / ea:7.3f} "
              f"{r2a:14.6f} {r2b:14.6f}")
        summary.append({"bin": label, "shots": len(sel), "sse_share_a": ea / sa,
                        "sse_share_b": eb / sb, "ratio": eb / ea, "r2_a": r2a, "r2_b": r2b})
    worst = sorted(rows, key=lambda r: -r["sse_a"])[:25]
    print("\nworst 25 shots of a (by SSE): share of pooled SSE, ramp-down share, SSE b/a")
    for r in worst:
        print(f"  fold {r['fold']} {r['shot']}: {r['sse_a'] / sa:.4f} rd {r['rampdown']:.2f} "
              f"frames {r['frames']:3d}  b/a {r['sse_b'] / r['sse_a']:.3f}")
    (args.out / "report.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

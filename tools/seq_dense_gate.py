#!/usr/bin/env python3
"""Gate for Astra's `time-and-geometry` item 1: do sequence members lose on shots with frame gaps?

    uv run python tools/seq_dense_gate.py runs/cap_attn_p100_t4resln runs/cap_gru_p100_t1024 \
        --fold 0 --gap-shots 100 --controls 50 --out runs/seqdense_f0

Sequence members see frame index, not time: attention positions and conv distances are frames, and
the gap feature is clipped at 5 steps (100 ms), so a 120 ms and a 2.6 s gap look alike. Here a
trained member predicts each held-out shot twice: on its EFIT frames (as scored), and on a DENSE
sequence — the EFIT frames plus filler frames on the 20 ms clock inside every gap, inputs computed
from the raw signals exactly as for real frames — keeping only the original frames' predictions.
No training. The Ip time axis is aligned once on the original frames and reused for the dense
sequence, so only the frame sequence changes. Reported per member: pooled flux SSE dense / orig on
gap shots and on gap-free controls (where the two must agree exactly), per-shot ratios, q95/betaN.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cv_oof import _fold_files
from my_experiments import baseline_model as B

GAP_MS = 1.5 * B.FRAME_STEP_MS


def dense_times(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(dense times, indices of the original frames in it)."""
    parts = [t[:1]]
    for a, b in zip(t[:-1], t[1:], strict=True):
        if b - a > GAP_MS:
            parts.append(np.arange(a + B.FRAME_STEP_MS, b - 0.5 * B.FRAME_STEP_MS, B.FRAME_STEP_MS))
        parts.append(np.array([b]))
    d = np.concatenate(parts)
    idx = np.searchsorted(d, t)
    if not np.array_equal(d[idx], t):
        raise ValueError("original frames lost in the dense grid")
    return d, idx


def features_at(row: pd.Series, times: np.ndarray, ip_times: np.ndarray) -> np.ndarray:
    r = row.copy()
    r["efit_times"] = times
    keep = B.align_ip_times
    B.align_ip_times = lambda *a, **k: ip_times
    try:
        return B.features_for_row(r)
    finally:
        B.align_ip_times = keep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("members", nargs="+", type=Path)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--model", default="seq")
    ap.add_argument("--gap-shots", type=int, default=100)
    ap.add_argument("--controls", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    arts = {m: joblib.load(m / f"fold{args.fold}" / "baseline.joblib") for m in args.members}
    files = _fold_files(next(iter(arts.values())))
    for m, a in arts.items():
        if [f.name for f in _fold_files(a)] != [f.name for f in files]:
            raise ValueError(f"{m}: a different held-out fold")
    # Shots by their largest gap; the gap shots with the most gap time first, controls gap-free.
    info = []
    for f in files:
        t = np.asarray(pd.read_parquet(f, columns=["efit_times"]).iloc[0]["efit_times"], float)
        d = np.diff(t)
        info.append((f, float((d[d > GAP_MS]).sum()), len(t)))
    gap = sorted([x for x in info if x[1] > 0], key=lambda x: -x[1])
    free = [x for x in info if x[1] == 0]
    print(f"fold {args.fold}: {len(files)} shots, {len(gap)} with gaps > {GAP_MS:g} ms, "
          f"{len(free)} gap-free", flush=True)
    rng = np.random.default_rng(0)
    take_gap = gap[:args.gap_shots]
    take_free = [free[i] for i in rng.choice(len(free), min(args.controls, len(free)), replace=False)]
    rows = []
    for kind, sel in (("gap", take_gap), ("control", take_free)):
        for f, gap_ms, n in sel:
            row = pd.read_parquet(f).iloc[0]
            t = np.asarray(row["efit_times"], dtype=np.float64)
            feats_c, psi, scal = B._read_training_shot(f)
            ip_times = B.align_ip_times(
                t, np.asarray(row["magnetics_plasma_current_times"], dtype=np.float64),
                np.asarray(row["magnetics_plasma_current"], dtype=np.float64),
                np.asarray(row["magnetics_time"], dtype=np.float64))
            feats_o = features_at(row, t, ip_times)
            if not np.allclose(feats_o, feats_c, rtol=1e-5, atol=1e-6):
                raise ValueError(f"{f.name}: recomputed features differ from the shot cache")
            td, idx = dense_times(t)
            feats_d = features_at(row, td, ip_times)
            q_true = scal[:, :2]
            entry = {"shot": f.name, "kind": kind, "frames": n, "dense": len(td), "gap_ms": gap_ms}
            for m, art in arts.items():
                nps = int(art["n_pca"])
                outs = {}
                for tag, X, pick in (("orig", feats_o, slice(None)), ("dense", feats_d, idx)):
                    Xs = art["scaler"].transform(B.build_inputs(art["coil"], X))
                    P = B._predict_targets(art, args.model, Xs)[pick]
                    flux = (art["pca"].inverse_transform(P[:, :nps]).reshape(-1, 65, 65)
                            + B.coil_flux(art["coil"], feats_o))
                    outs[tag] = {"sse": float(((flux - psi) ** 2).sum()),
                                 "qb_se": ((P[:, nps:nps + 2] - q_true) ** 2).sum(0).tolist()}
                entry[m.name] = outs
            entry["sst"] = float(((psi - psi.mean(0)) ** 2).sum())
            rows.append(entry)
            msg = " ".join(f"{m.name}: {entry[m.name]['dense']['sse'] / entry[m.name]['orig']['sse']:.3f}"
                           for m in arts)
            print(f"{kind} {f.name} frames {n} -> {len(td)} (gaps {gap_ms:.0f} ms) dense/orig {msg}",
                  flush=True)
    (args.out / "rows.json").write_text(json.dumps(rows))
    summary = {}
    for m in arts:
        for kind in ("gap", "control"):
            sel = [r for r in rows if r["kind"] == kind]
            so = sum(r[m.name]["orig"]["sse"] for r in sel)
            sd = sum(r[m.name]["dense"]["sse"] for r in sel)
            ratios = np.array([r[m.name]["dense"]["sse"] / r[m.name]["orig"]["sse"] for r in sel])
            qo = np.sum([r[m.name]["orig"]["qb_se"] for r in sel], axis=0)
            qd = np.sum([r[m.name]["dense"]["qb_se"] for r in sel], axis=0)
            s = {"shots": len(sel), "pooled": sd / so, "median": float(np.median(ratios)),
                 "better": int((ratios < 1).sum()), "q95": float(qd[0] / qo[0]),
                 "betaN": float(qd[1] / qo[1])}
            summary[f"{m.name}/{kind}"] = s
            print(f"{m.name:32s} {kind:8s} shots {s['shots']:3d} pooled flux SSE dense/orig "
                  f"{s['pooled']:.4f} median {s['median']:.4f} better {s['better']}/{s['shots']} "
                  f"q95 {s['q95']:.3f} betaN {s['betaN']:.3f}")
    (args.out / "report.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

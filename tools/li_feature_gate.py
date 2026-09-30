#!/usr/bin/env python3
"""Gate: do current-diffusion features explain the stack's derived-scalar residuals?

    uv run python tools/li_feature_gate.py --out runs/li_feature_gate

The magnetic diagnostics EFIT fitted are withheld, so the interior current profile — which is
what the scored li error is (tools/li_decompose.py: the interior <Bp^2>, not the contour) — can
only be inferred physically: resistivity ~ Te^-1.5, so current settles where the plasma is hot,
at a rate set by the Ip history. Today's inputs carry Te and ne as summaries and the pressure
Te*ne as a profile; no Te shape, no Te^1.5, no explicit Ip history.

Per frame this builds (a) the Te and ne core profiles normalised to their peak on a fixed chord
grid, Te^1.5 moments, and (b) Ip history: time since the current rose, integral of Ip, Ip over
its running and shot maximum, dIp/dt over 0.1-1 s windows. A HistGBM per scalar predicts the
residual `truth - map scalar` of the cached 11-member linear map (runs/consgate_fc11), shot-wise
5-fold, from BASE = model inputs + the map's own seven scalars, and from BASE + the new block.
The difference in R2_cons is the gate.
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
from toolkit.parallel import pmap

NAMES = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
N_GRID = 12


def _profile(vals: np.ndarray, coord: np.ndarray, times: np.ndarray, efit_t: np.ndarray,
             power: float) -> np.ndarray:
    """(T, N_GRID + 2): peak-normalised profile^power on a fixed chord grid, its mean and width."""
    order = np.argsort(coord)
    c, v = coord[order], vals[:, order]
    grid = np.linspace(c[0], c[-1], N_GRID)
    out = np.full((len(times), N_GRID + 2), np.nan)
    for t in range(len(times)):
        live = v[t] > 0
        if live.sum() < 4:
            continue
        prof = np.interp(grid, c[live], v[t, live]) ** power
        pk = prof.max()
        if pk <= 0:
            continue
        prof = prof / pk
        w = prof / prof.sum()
        cbar = (w * grid).sum()
        out[t] = np.concatenate([prof, [prof.mean(), np.sqrt((w * (grid - cbar) ** 2).sum())]])
    res = np.zeros((len(efit_t), out.shape[1]), np.float32)
    for j in range(out.shape[1]):
        ok = np.isfinite(out[:, j])
        if ok.sum() >= 2:
            res[:, j] = np.interp(efit_t, times[ok], out[ok, j])
    return res


def shot_features(path: Path) -> np.ndarray:
    r = pd.read_parquet(path).iloc[0]
    et = np.asarray(r["efit_times"], dtype=np.float64)
    names = [str(x) for x in np.asarray(r["thomson_chord_name"])]
    z = np.asarray(r["thomson_chord_Z"], dtype=np.float64)
    te = np.stack([np.asarray(x, dtype=np.float64) for x in r["thomson_core_Te"]])
    ne = np.stack([np.asarray(x, dtype=np.float64) for x in r["thomson_core_ne"]])
    tt = np.asarray(r["thomson_core_times"], dtype=np.float64)
    core = [i for i, n in enumerate(names) if n.startswith("TS_core")][:te.shape[1]]
    zc = z[core]
    blocks = [_profile(te, zc, tt, et, 1.0), _profile(te, zc, tt, et, 1.5),
              _profile(np.where(te > 0, ne, 0.0), zc, tt, et, 1.0)]
    ip = np.asarray(r["magnetics_plasma_current"], dtype=np.float64)
    # The shipped Ip axis is a shared template on ~70% of shots (README): align it as the models do.
    it = np.asarray(B.align_ip_times(et, np.asarray(r["magnetics_plasma_current_times"], dtype=np.float64),
                                     ip, np.asarray(r["magnetics_time"], dtype=np.float64)), dtype=np.float64)
    ipe = np.interp(et, it, ip)
    aip = np.abs(ip)
    start = it[np.argmax(aip > 0.3 * aip.max())] if aip.max() > 0 else it[0]
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (aip[1:] + aip[:-1]) * np.diff(it))])
    runmax = np.maximum.accumulate(aip)
    hist = [et - start, np.interp(et, it, cum) / max(aip.max(), 1e-9), np.abs(ipe) / max(aip.max(), 1e-9),
            np.abs(ipe) / np.maximum(np.interp(et, it, runmax), 1e-9)]
    for w in (100.0, 300.0, 1000.0):
        hist.append((np.abs(ipe) - np.interp(et - w, it, aip)) / w)
    return np.column_stack([*blocks, np.column_stack(hist)]).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=30)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from sklearn.ensemble import HistGradientBoostingRegressor
    base_X, new_X, resid, true_all, shot_ids = [], [], [], [], []
    for k in range(5):
        art = joblib.load(f"runs/cv5_fixcos/fold{k}/baseline.joblib")
        files = _fold_files(art)
        X, _, _, lens = B._read_shots(files, f"reading fold{k}", 1.0, args.jobs)
        Xs = art["scaler"].transform(B.build_inputs(art["coil"], X))
        z = np.load(f"runs/consgate_fc11/scalars_fold{k}.npz")
        feats = np.concatenate(pmap(shot_features, files, args.jobs))
        if len(feats) != len(X):
            raise ValueError(f"fold {k}: {len(feats)} feature rows for {len(X)} frames")
        base_X.append(np.column_stack([Xs, z["pred"]]).astype(np.float32))
        new_X.append(feats)
        resid.append(z["true"] - z["pred"])
        true_all.append(z["true"])
        shot_ids.append(np.repeat(np.arange(len(lens)) + 10000 * k, lens))
        print(f"fold {k}: {len(X)} frames, base {base_X[-1].shape[1]} + new {feats.shape[1]} features",
              flush=True)
    folds = [np.full(len(r), k) for k, r in enumerate(resid)]
    Xb, Xn, R, T, fold = (np.concatenate(a) for a in (base_X, new_X, resid, true_all, folds))
    report = {}
    for j, name in enumerate(NAMES):
        ok = np.isfinite(R[:, j]) & np.isfinite(T[:, j])
        sst = ((T[ok, j] - T[ok, j].mean()) ** 2).sum()
        before = 1 - (R[ok, j] ** 2).sum() / sst
        out = {"before": before}
        for label, X in (("base", Xb), ("base+new", np.column_stack([Xb, Xn]))):
            pred = np.zeros(len(R))
            for k in range(5):
                tr = ok & (fold != k)
                rows = np.flatnonzero(tr)[::args.every]
                m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.06, max_leaf_nodes=63,
                                                  min_samples_leaf=200, l2_regularization=1.0,
                                                  random_state=j)
                m.fit(X[rows], R[rows, j])
                te = ok & (fold == k)
                pred[te] = m.predict(X[te])
            out[label] = 1 - ((R[ok, j] - pred[ok]) ** 2).sum() / sst
        report[name] = out
        print(f"{name:>8}: R2_cons {before:.5f} | residual model on base {out['base']:.5f} "
              f"({out['base'] - before:+.5f}) | base+new {out['base+new']:.5f} "
              f"({out['base+new'] - before:+.5f}) | new adds {out['base+new'] - out['base']:+.5f}",
              flush=True)
        (args.out / "report.json").write_text(json.dumps(report, indent=1))
    gain = np.mean([report[n]["base+new"] - report[n]["base"] for n in NAMES])
    print(f"mean dR2_cons from the new block {gain:+.5f}, dS {0.20 * gain:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

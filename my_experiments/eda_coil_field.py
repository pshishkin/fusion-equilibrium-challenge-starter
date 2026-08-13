#!/usr/bin/env python3
"""
What is left of psi once the coils' own field is subtracted.

    uv run python my_experiments/eda_coil_field.py [--shots 10] [--fit-frames 20]

One row per shot: the stored flux map, the analytic coil field of `coil_field.py`, and their
difference -- the part a model would actually have to predict if the vacuum field were computed
instead of learned. Written to `results/eda_coil_field.png`, with the numbers that decide whether
this is worth building on printed to stdout:

  * the gains, fitted OUTSIDE the plasma boundary with the plasma's own filament in the design.
    These are a calibration CHECK, not a result -- the model subtracts an unfitted coil field. A
    group gain of 1.0 for the F-coils says the shipped ampere-turns, the shipped rectangles, the
    elliptic integrals and the storage sign all already agree with the stored flux. `ECOILA` is
    expected to be far off (kA, turn count not folded in, plus the unshipped `ECOILB`), and the
    plasma filament slightly under 1 (a 1 MA plasma is not a filament).
  * R^2 of the unfitted coil field against the stored flux, outside the boundary and over the
    whole grid, alongside the fitted version.
  * how much of the flux VARIANCE the subtraction removes, which is what matters for modelling:
    the PCA the model regresses onto has to span whatever is left.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import D3D_MAGNETICS_SIGNALS, DEFAULT_LOCAL_DATA_DIR, _as_psirz_stack
from my_experiments.baseline_model import features_for_row, sorted_shots, thin_frames
from my_experiments.coil_field import (
    Calibration,
    CoilBasis,
    FloatArray,
    build_basis,
    filament_flux,
    fit_flux_gains,
    vacuum_residual,
)

OUT = Path(__file__).resolve().parent.parent / "results" / "eda_coil_field.png"


def lcfs_mask(row: Any, frames: FloatArray, basis: CoilBasis) -> FloatArray:
    """(T, nZ, nR) bools: grid nodes OUTSIDE the last closed flux surface.

    That is where the coil field has to explain the flux on its own, so it is where a calibration
    fit is honest. Inside the boundary the plasma's own flux is both large and correlated with the
    shaping currents, and a fit there would launder plasma into the coil gains.
    """
    from matplotlib.path import Path as MplPath

    rr, zz = np.meshgrid(basis.grid_R, basis.grid_Z)
    nodes = np.column_stack([rr.ravel(), zz.ravel()])
    n = np.asarray(row["efit_lcfs_n"])
    r_all = np.asarray(row["efit_lcfs_r"], dtype=object)
    z_all = np.asarray(row["efit_lcfs_z"], dtype=object)
    out = np.zeros((frames.size, basis.grid_Z.size, basis.grid_R.size), dtype=bool)
    for i, t in enumerate(frames):
        k = int(n[t])
        if k < 3:
            raise ValueError(f"frame {t}: LCFS has {k} points, cannot bound a region")
        poly = np.column_stack([np.asarray(r_all[t])[:k], np.asarray(z_all[t])[:k]])
        inside = MplPath(poly).contains_points(nodes).reshape(rr.shape)
        out[i] = ~inside
    return out


def load_shot(path: Path, n_frames: int) -> tuple[Any, Any, FloatArray]:
    """One shot thinned to `n_frames`: the row, the frame indices, and psi (T, nZ, nR)."""
    row = pd.read_parquet(path).iloc[0]
    psi_all = _as_psirz_stack(row["efit_psirz"])
    if not np.isfinite(psi_all).all():
        raise ValueError(f"{path.name}: efit_psirz carries non-finite values")
    frames = thin_frames(psi_all.shape[0], min(1.0, n_frames / psi_all.shape[0]))
    return row, frames, psi_all[frames]


def currents_for(row: Any, basis: CoilBasis, frames: FloatArray) -> tuple[FloatArray, FloatArray]:
    """(T, C) currents in basis-column order and (T,) plasma current, on the EFIT time base.

    Through `features_for_row`, so these are the very columns the model is fed — including the
    plasma-current time-base correction, which would otherwise misalign one of them by 3 s.
    """
    feats = features_for_row(row)                                   # (T, 21), signal order
    index = {sig: i for i, sig in enumerate(D3D_MAGNETICS_SIGNALS)}
    cols = []
    for column in basis.columns:
        key = column.removeprefix("magnetics_")
        if key not in index:
            raise ValueError(
                f"coil geometry joins to {column!r}, which is not among the "
                f"{len(D3D_MAGNETICS_SIGNALS)} magnetics signals {D3D_MAGNETICS_SIGNALS}"
            )
        cols.append(feats[frames, index[key]].astype(np.float64))
    ip = feats[frames, index["plasma_current"]].astype(np.float64)
    return np.column_stack(cols), ip


def pca_ceiling(psi: FloatArray, resid: FloatArray, coil: FloatArray) -> None:
    """Where the truncation ceiling sits for each target, in the metric's own R2_psi.

    The hope that motivates subtracting the coil field is that the residual, being a single smooth
    blob following the plasma instead of that blob plus the coils' hot spots at the grid edge,
    needs fewer components. It does -- at ten. By 50, the number `params.yaml` actually uses, both
    targets are exact to six decimals, so the model's R2_psi of 0.999 is entirely REGRESSION
    error and none of it is representation error. Whatever this decomposition is worth, it is not
    worth a more compact basis; do not re-argue that one.
    """
    from sklearn.decomposition import PCA

    ss = float(((psi - psi.mean()) ** 2).sum())
    print("PCA truncation ceiling on these frames, as the metric's own R2_psi:")
    for name, target in (("psi", psi), ("residual", resid)):
        flat = target.reshape(len(target), -1)
        for n in (10, 25, 50):
            if n >= min(flat.shape):
                continue
            p = PCA(n_components=n, random_state=0).fit(flat)
            rec = p.inverse_transform(p.transform(flat)).reshape(target.shape)
            back = rec + coil if name == "residual" else rec
            r2 = 1.0 - float(((psi - back) ** 2).sum()) / ss
            print(f"  {name:9s} n={n:3d}   R2_psi {r2:.6f}")


def panel(ax: Any, field: FloatArray, basis: CoilBasis, title: str, poly: FloatArray) -> None:
    lim = float(np.abs(field).max())
    ax.pcolormesh(basis.grid_R, basis.grid_Z, field, cmap="RdBu_r", vmin=-lim, vmax=lim,
                  shading="nearest", rasterized=True)
    ax.contour(basis.grid_R, basis.grid_Z, field, levels=12, colors="#44444455", linewidths=0.4)
    ax.plot(poly[:, 0], poly[:, 1], color="#111111", lw=1.2)
    ax.set_title(f"{title}   ±{lim:.3f} Wb/rad", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", type=int, default=10,
                    help="shots to look at, one frame plotted each")
    ap.add_argument("--fit-frames", type=int, default=20, help="frames per shot used to fit gains")
    ap.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR)
    args = ap.parse_args(argv)

    files = sorted_shots(args.local_data_dir)[: args.shots]
    print(f"{len(files)} shots, {args.fit_frames} frames each for the fit\n")

    basis: CoilBasis | None = None
    psi_parts, cur_parts, mask_parts, fil_parts, shown = [], [], [], [], []
    for path in files:
        row, frames, psi = load_shot(path, args.fit_frames)
        if basis is None:
            basis = build_basis(row)
            print(f"basis: {len(basis.columns)} current columns on {basis.machine}, grid "
                  f"{basis.grid_R[0]:.2f}..{basis.grid_R[-1]:.2f} m x "
                  f"{basis.grid_Z[0]:+.2f}..{basis.grid_Z[-1]:+.2f} m, closest filament-to-node "
                  f"gap {basis.min_gap * 1e3:.1f} mm\n")
        currents, ip = currents_for(row, basis, frames)
        psi_parts.append(psi)
        cur_parts.append(currents)
        mask_parts.append(lcfs_mask(row, frames, basis))
        # The plasma's own field, for the calibration fit only: a filament at the magnetic axis
        # carrying Ip. Both coordinates are labels; nothing downstream of this reads them.
        fil_parts.append(np.stack([
            filament_flux(float(np.asarray(row["efit_r_axis"])[t]),
                          float(np.asarray(row["efit_z_axis"])[t]), float(ip[i]), basis)
            for i, t in enumerate(frames)]))
        mid = frames.size // 2
        k = int(np.asarray(row["efit_lcfs_n"])[frames[mid]])
        poly = np.column_stack([np.asarray(row["efit_lcfs_r"], dtype=object)[frames[mid]][:k],
                                np.asarray(row["efit_lcfs_z"], dtype=object)[frames[mid]][:k]])
        shown.append((path.stem, psi[mid], currents[mid], poly))

    assert basis is not None
    psi = np.concatenate(psi_parts)
    currents = np.concatenate(cur_parts)
    mask = np.concatenate(mask_parts)
    plasma = np.concatenate(fil_parts)

    names = [c.removeprefix("magnetics_") for c in basis.columns]
    cal = Calibration(len(names))
    ones = np.ones_like(psi[0])
    for t in range(len(psi)):
        # Per frame, the plasma gets its own amplitude and its own constant. Both are nuisances:
        # a filament is a crude 1 MA plasma, and pooling its amplitude across frames pushes the
        # misfit into the coil gains, which is the number being measured.
        cal.add(basis.maps, currents[t], psi[t], mask[t],
                np.stack([plasma[t], ones], axis=-1))
    group = np.zeros((len(names), 2))
    group[[n.startswith("F") for n in names], 0] = 1.0
    group[names.index("ECOILA"), 1] = 1.0
    g2, r2_g2 = cal.solve(group)
    gains, r2_fit = cal.solve()

    print("calibration gains (1.0 = shipped units, geometry, elliptic integrals and storage sign\n"
          "already agree with the stored flux), outside the boundary, with a per-frame plasma\n"
          "filament and a per-frame constant projected out:\n")
    print(f"  F-coils as a group {g2[0]:+.4f}     R2 {r2_g2:.4f}   ({cal.n_nodes} nodes)")
    print("  per coil           " + "  ".join(f"{n}={g:+.2f}" for n, g in zip(names, gains,
                                                                             strict=True)))
    print(f"  {' ' * 19}R2 {r2_fit:.4f}")
    print(f"\n  ECOILA is not identifiable and is not evidence either way: its field over this "
          f"grid is\n  nearly degenerate with the constant, and its gain swings from +142 to "
          f"{gains[names.index('ECOILA')]:+.0f} with\n  that constant free. The pipeline never "
          f"uses these gains — it fits its own, see coil_field.\n")
    print(f"  Grad-Shafranov self-check, one filament away from itself: "
          f"|Delta* psi| / |psi| = {vacuum_residual(basis.grid_R, basis.grid_Z):.1e}\n")

    # What the pipeline actually subtracts, so the panels show the real decomposition: the same
    # maps under the gains `train()` fits, which minimise what is left rather than recover a
    # physical constant.
    scaled = basis.maps * fit_flux_gains(basis.maps, currents, psi)[:, None, None]
    coil = np.tensordot(currents, scaled, axes=(1, 0))
    resid = psi - coil

    flat = mask.reshape(-1)
    b = psi.reshape(-1)[flat]
    r2_coil_only = (1.0 - float((resid.reshape(-1)[flat] ** 2).sum())
                    / float(((b - b.mean()) ** 2).sum()))
    print(f"under the gains the pipeline fits, the coil field alone explains R2 = "
          f"{r2_coil_only:+.4f}\nof the stored flux outside the boundary. Read that as a split, "
          f"not as an approximation:\nthose gains minimise what is left over rather than recover "
          f"a physical constant, and a 1 MA\nplasma outweighs ~140 kA-turn per shaping coil, so "
          f"its own field reaches well past the\nboundary. Under the calibration gains above the "
          f"same number is negative, and neither is\na failure -- whatever is subtracted is added "
          f"back unchanged.\n")
    print(f"flux std   {psi.std():.4f} Wb/rad  ->  residual std {resid.std():.4f} Wb/rad "
          f"({resid.std() / psi.std():.1%} of it left)")
    per_frame = psi.reshape(psi.shape[0], -1).mean(axis=1)
    per_frame_r = resid.reshape(resid.shape[0], -1).mean(axis=1)
    print(f"per-frame mean spread  {per_frame.std():.4f}  ->  {per_frame_r.std():.4f}\n")
    pca_ceiling(psi, resid, coil)

    n = len(shown)
    fig, axes = plt.subplots(n, 3, figsize=(9.5, 3.1 * n), constrained_layout=True)
    for i, (name, psi_i, cur_i, poly) in enumerate(shown):
        coil_i = np.tensordot(cur_i, scaled, axes=(0, 0))
        panel(axes[i][0], psi_i, basis, f"{name}\nstored psi", poly)
        panel(axes[i][1], coil_i, basis, "coil field (analytic)", poly)
        panel(axes[i][2], psi_i - coil_i, basis, "residual = plasma", poly)
    fig.suptitle("psi, its analytic coil field, and what is left for a model to predict",
                 fontsize=11)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

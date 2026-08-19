#!/usr/bin/env python3
"""Run the solver on DIII-D, where the truth exists in quantity, and see what it is worth.

    uv run python solver/validate_d3d.py --fit 20 --eval 40 --rounds 5 --jobs 16

Three MAST demo shots have now refuted three decisions that were fitted on them — the last one, the
Thomson-driven pressure term, gained +0.021 of R2_psi locally and lost 0.075 on the 1206-shot fold.
The disease is not any one of those choices; it is that every constant in `mast/` is chosen from a
sample of three. DIII-D has **7041 labelled shots** and the same equation, so it is where a choice
can be tested before it is spent on a leaderboard.

Two things this measures and one it does not.

* **Is the solver right at all?** MAST's board R2_psi is 0.8895 against a leader's 0.9732, and with
  three shots there was no way to tell a solver bug from a MAST-specific limitation. Here a bug
  shows up against thousands of frames.
* **How fragile is a choice to being fitted on three shots?** `--fit 3` with a random draw, scored
  on shots it never saw, repeated — that is the MAST situation reproduced where it can be measured.
  A parameterisation that wins on 3-shot draws is the one to carry over, and it need not be the one
  that fits best.
* It does **not** measure machine transfer. DIII-D runs at aspect ratio 2.7 and MAST at 1.3, and a
  source profile that suits one need not suit the other. That risk is separate and stays open.

The gains are fitted the same alternating way as MAST's — solve, subtract the plasma flux the solve
produced, refit the coils on what is left, damp, repeat — because a gain fitted against the raw
truth launders the plasma's own field into the coils.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from solver import d3d  # noqa: E402
from solver.greens import CoilBasis, build_basis, grid_green  # noqa: E402
from solver.gs import Profile, operator, solve  # noqa: E402
from solver.machine import D3D, Machine  # noqa: E402
from toolkit.parallel import pimap, resolve_jobs  # noqa: E402
from toolkit.progress import install_timestamps  # noqa: E402

FloatArray = npt.NDArray[np.floating]
DATA = HERE.parent.parent / "downloaded_huggingface" / "hf_dataset" / "data" / "diii_d_train"
# Frames below this carry no confined plasma — the first and last moments of a discharge, where a
# free-boundary solve has nothing to find. MAST's own rate of "no closed surface" frames was 0.12%,
# all of them under 250 kA against a flat top of 600-800.
MIN_IP_KA = 150.0


def load(paths: list[Path]) -> tuple[CoilBasis, list[d3d.D3DShot]]:
    basis = None
    shots = []
    for path in paths:
        row = pd.read_parquet(path).iloc[0]
        if basis is None:
            basis = build_basis(row, D3D)
        shots.append(d3d.read(row, basis.columns))
    if basis is None:
        raise SystemExit("no shots given")
    return basis, shots


def live(s: d3d.D3DShot) -> npt.NDArray[np.bool_]:
    """Frames with a plasma worth solving for."""
    return np.abs(s.ip) / 1e3 >= MIN_IP_KA


def truth_of(s: d3d.D3DShot) -> FloatArray:
    """The shipped flux on the live frames. Absent truth is a caller error, not a missing value."""
    if s.psi is None:
        raise ValueError("this shot carries no efit_psirz; validation needs the training config")
    return s.psi[live(s)]


def pooled_r2(truth: list[FloatArray], pred: list[FloatArray]) -> float:
    t = np.concatenate([x.ravel() for x in truth])
    p = np.concatenate([x.ravel() for x in pred])
    return 1.0 - float(((t - p) ** 2).sum()) / float(((t - t.mean()) ** 2).sum())


def coil_flux(basis: CoilBasis, s: d3d.D3DShot, gains: FloatArray) -> FloatArray:
    return basis.flux(s.currents * gains[None, :])


def _solve_task(args: tuple) -> FloatArray:
    """One shot's solved flux, in a worker. Returns (n_live, nZ, nR).

    The `Machine` travels in the task rather than being read from the module. A worker is a fresh
    interpreter: it re-imports `solver.machine` and sees the shipped defaults, so a sweep that
    mutates `MACHINES` in the parent silently scores the default in every cell. That happened —
    four values of `z_band` all returned R2 +0.35631 to five decimals, which is what caught it.
    """
    from threadpoolctl import threadpool_limits

    coil, ip, profile, n_iter, relax, grid_R, grid_Z, machine = args
    op = operator(machine, grid_R, grid_Z)
    with threadpool_limits(limits=1):
        out = []
        for k in range(len(ip)):
            try:
                out.append(solve(coil[k], float(ip[k]), op, profile=profile,
                                 n_iter=n_iter, relax=relax).psi)
            except ValueError:
                out.append(coil[k])
    return np.stack(out)


def solve_all(basis: CoilBasis, shots: list[d3d.D3DShot], gains: FloatArray, profile: Profile,
              n_iter: int, relax: float, jobs: int, machine: Machine = D3D) -> list[FloatArray]:
    """Every live frame of every shot, solved, over a process pool."""
    tasks = [(coil_flux(basis, s, gains)[live(s)], s.ip[live(s)], profile, n_iter, relax,
              basis.grid_R, basis.grid_Z, machine) for s in shots if live(s).any()]
    return list(pimap(_solve_task, tasks, resolve_jobs(jobs, len(tasks))))


def fit_gains(basis: CoilBasis, shots: list[d3d.D3DShot], plasma: list[FloatArray],
              outside: list[npt.NDArray[np.bool_]], prior: FloatArray, damp: float,
              per_frame: bool = False) -> FloatArray:
    """One gain per coil column, damped toward `prior`.

    Two modes for what to do about the plasma's own field, and the difference is not cosmetic.

    * `per_frame=False` — the plasma flux is KNOWN (it came out of a solve) and is subtracted.
    * `per_frame=True` — the plasma flux is a crude filament whose amplitude nobody knows, so each
      FRAME gets its own free amplitude. It is not fitted as a column but projected out of both the
      design and the target, frame by frame, which is the same estimator and costs one dot product
      instead of a parameter per frame.

    Per-frame is what this fork already measured as necessary on DIII-D: fitting the coil gains
    with no plasma term returned values scattered from -0.15 to -0.57 and both signs; with ONE
    amplitude pooled over frames it read 0.997 on ten shots and 0.911 on forty — a number that
    moved with the sample, which is the signature of a misspecified fit; with a per-frame amplitude
    it settles at **0.87-0.89 at any sample size tried**.

    `outside` restricts every frame to the cells beyond the plasma boundary, and it is the other
    half of the same lesson: inside the boundary the flux is a distributed current that one
    filament cannot represent, so a fit there launders the plasma into the coils however free its
    amplitude is. Fitting over the whole grid here returned F-coil gains from -1.4 to +2.8 and an
    alternation that oscillated instead of converging.
    """
    n_c = len(basis.columns)
    xs, ys = [], []
    for s, pl, use in zip(shots, plasma, outside, strict=True):
        keep = live(s)
        cur = s.currents[keep]
        n_f = int(keep.sum())
        x = np.stack([(cur[:, i][:, None, None] * basis.maps[i]).reshape(n_f, -1)
                      for i in range(n_c)], axis=-1)          # (n_f, pixels, n_c)
        y = truth_of(s).reshape(n_f, -1)
        g = pl.reshape(n_f, -1)
        w = use.reshape(n_f, -1)                              # cells outside the boundary
        if per_frame:
            gw = np.where(w, g, 0.0)
            gg = np.maximum((gw * gw).sum(axis=1), 1e-300)
            proj = (gw[:, :, None] * x).sum(axis=1) / gg[:, None]
            x = x - g[:, :, None] * proj[:, None, :]
            y = y - g * ((gw * y).sum(axis=1) / gg)[:, None]
        else:
            y = y - g
        xs.append(x[w])
        ys.append(y[w])
    fitted, *_ = np.linalg.lstsq(np.concatenate(xs), np.concatenate(ys), rcond=None)
    return (1.0 - damp) * prior + damp * np.asarray(fitted, dtype=np.float64)


def bootstrap_plasma(basis: CoilBasis, shots: list[d3d.D3DShot]) -> list[FloatArray]:
    """Round zero: one filament at the vessel centre carrying Ip, with a FREE amplitude.

    The amplitude has to be free, and this fork has the measurement to prove it: fitting DIII-D's
    coil gains without a plasma term at all returned values scattered from -0.15 to -0.57 and both
    signs, and pooling one amplitude across frames read 0.997 on ten shots and 0.911 on forty. So
    the filament is a column in the least-squares fit rather than a fixed subtraction, and its
    coefficient is thrown away afterwards — it exists only to stop the coils laundering the
    plasma's own field into themselves.
    """
    r0 = 0.5 * float(basis.grid_R[0] + basis.grid_R[-1])
    g = grid_green(np.array([[r0, 0.0]]), basis.grid_R, basis.grid_Z)
    out: list[FloatArray] = []
    for s in shots:
        out.append(np.asarray(g[None] * s.ip[live(s)][:, None, None], dtype=np.float64))
    return out


def main() -> int:
    install_timestamps()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", type=int, default=20, help="shots the gains are fitted on")
    ap.add_argument("--eval", type=int, default=40, help="further shots, never fitted on")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--damp", type=float, default=0.4)
    ap.add_argument("--gains-json", type=Path,
                    help="use these coil gains instead of fitting any; the file is written by "
                         "the standalone estimator, which fits a per-frame plasma filament at the "
                         "SHIPPED magnetic axis over the cells outside the shipped LCFS")
    ap.add_argument("--fixed-gain", type=float, default=-1.0,
                    help="skip the fit and give every coil this gain; 1.0 is what DIII-D's one "
                         "rectangle per coil implies, and it isolates the SOLVER from the "
                         "calibration")
    ap.add_argument("--seed", type=int, default=0, help="which random draw of shots")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--relax", type=float, default=0.4)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--edge", type=float, default=0.0,
                    help="how far past the last closed surface the source runs, as a fraction of "
                         "the flux span; MAST's composite sweep chose 0.02 and this is where that "
                         "choice is checked against a machine with thousands of labelled frames")
    ap.add_argument("--z-pin", type=float, help="override the machine's vertical axis pin")
    ap.add_argument("--z-band", type=float, help="override how far the axis may wander from it")
    ap.add_argument("--data", type=Path, default=DATA)
    args = ap.parse_args()

    files = sorted(args.data.glob("*.parquet"))
    if len(files) < args.fit + args.eval:
        raise SystemExit(f"{len(files)} shots in {args.data}, need {args.fit + args.eval}")
    pick = np.random.default_rng(args.seed).permutation(len(files))[:args.fit + args.eval]
    fit_paths = [files[i] for i in pick[:args.fit]]
    eval_paths = [files[i] for i in pick[args.fit:]]
    profile = Profile(alpha=args.alpha, gamma=args.gamma, beta=args.beta, edge=args.edge)
    machine = D3D
    if args.z_pin is not None or args.z_band is not None:
        d = dict(D3D.__dict__)
        d["z_pin"] = args.z_pin if args.z_pin is not None else D3D.z_pin
        d["z_band"] = args.z_band if args.z_band is not None else D3D.z_band
        machine = Machine(**d)
        print(f"  vertical pin overridden: z_pin={d['z_pin']} band={d['z_band']}")

    print(f"DIII-D: fitting on {len(fit_paths)} shots, evaluating on {len(eval_paths)} never "
          f"fitted on, draw seed {args.seed}, {profile}")
    basis, fit_shots = load(fit_paths)
    _, eval_shots = load(eval_paths)
    n_fit = sum(int(live(s).sum()) for s in fit_shots)
    n_eval = sum(int(live(s).sum()) for s in eval_shots)
    print(f"  {len(basis.columns)} current columns; {n_fit} fit frames and {n_eval} eval frames "
          f"above {MIN_IP_KA:.0f} kA")

    fit_truth = [truth_of(s) for s in fit_shots]
    eval_truth = [truth_of(s) for s in eval_shots]
    gains: FloatArray = np.full(len(basis.columns), args.fixed_gain, dtype=np.float64)
    if args.gains_json is not None:
        import json
        d = json.loads(args.gains_json.read_text())
        gains = np.array([d[c] for c in basis.columns], dtype=np.float64)
        print(f"  gains from {args.gains_json.name}: F-coil median "
              f"{np.median([v for c, v in d.items() if 'F' in c[10:]]):.4f}, "
              f"ECOILA {d['magnetics_ECOILA']:.1f}")
    elif args.fixed_gain <= 0:
        outside = [d3d.outside_lcfs(s, live(s)) for s in fit_shots]
        gains = fit_gains(basis, fit_shots, bootstrap_plasma(basis, fit_shots), outside,
                          np.ones(len(basis.columns)), 1.0, per_frame=True)
    n_rounds = 0 if (args.fixed_gain > 0 or args.gains_json is not None) else args.rounds
    for rnd in range(1, n_rounds + 1):
        got = solve_all(basis, fit_shots, gains, profile, args.n_iter, args.relax,
                        args.jobs, machine)
        r2_fit = pooled_r2(fit_truth, got)
        plasma: list[FloatArray] = [np.asarray(g - coil_flux(basis, s, gains)[live(s)],
                                               dtype=np.float64)
                                    for g, s in zip(got, fit_shots, strict=True)]
        gains = fit_gains(basis, fit_shots, plasma, outside, gains, args.damp)
        print(f"  round {rnd}: R2_psi on the FIT shots {r2_fit:+.5f}   "
              f"gains {np.array2string(gains[:4], precision=2)}... "
              f"ECOILA {gains[basis.columns.index('magnetics_ECOILA')]:.1f}")

    held = solve_all(basis, eval_shots, gains, profile, args.n_iter, args.relax, args.jobs, machine)
    r2_held = pooled_r2(eval_truth, held)
    final = solve_all(basis, fit_shots, gains, profile, args.n_iter, args.relax, args.jobs, machine)
    print(f"\n  R2_psi   fitted-on {pooled_r2(fit_truth, final):+.5f}   "
          f"HELD OUT {r2_held:+.5f}")
    print("  gains:")
    for c, v in zip(basis.columns, gains, strict=True):
        print(f"    {c[10:]:<10} {v:>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""L3's free gate: is a learned surrogate's GRADIENT good enough to teach the model anything?

    uv run python my_experiments/surrogate_gate.py --artifact artefacts/lanes/a.joblib --shots 12

`aux-surrogate` puts a small net `g(c) -> (kappa, li)` on top of the 50 predicted PCA coefficients,
so the auxiliary gradient can only reach a lower loss THROUGH those coefficients. Plain `aux` cannot
force that: its auxiliary scalars are extra rows of the same output layer, and the trunk may satisfy
them by routes the coefficients never see. D1 measured that failure directly — a head fitted to the
seven functionals knew kappa **+0.0119** better than the map did, and the transport would have
realised about half of it.

**THE GATE IS ON GRADIENT FIDELITY, NOT VALUE ACCURACY, and that is the whole point.** The surrogate
is consumed at PREDICTED coefficients, which sit off the manifold of true equilibria. A `g` that
predicts kappa beautifully but whose derivative points the wrong way teaches the model to move the
wrong way, and a value R2 cannot see it. So `g` is fitted on coefficients PERTURBED around the truth
at the model's own error scale — the regime the gradient is actually consumed in, the same argument
`jacobian_form` makes for its own isotropic probe — and then judged on HELD-OUT shots by how well it
predicts the ACTUAL change of the functional along the real error direction `dc = c_pred - c_true`.

Three predictors are compared on the same frames:
  * `linear`    — a least-squares linearisation fitted on the same perturbed sample. This is the
                  stand-in for what the loss already does, and the number to beat.
  * `surrogate` — the two-layer net.
  * `constant`  — predicts the mean change. Present so a correlation can be read against something
                  that knows nothing; a method that cannot beat it is not measuring the direction.

Pre-registered threshold, from L3 and unchanged: **proceed only if kappa's correlation exceeds 0.90**
(the linearisation reached 0.77 when L3 was written). Below that, both surrogate arms close and no
training code is written.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "fusion_scoring"))

from common import CONS_SCALARS  # noqa: E402
from derive import derive_frame  # noqa: E402
from lcfs import extract_lcfs  # noqa: E402

from experiments import DEFAULT_LOCAL_DATA_DIR, HF_TRAIN_CONFIG  # noqa: E402
from my_experiments.baseline_model import sorted_shots, take_share  # noqa: E402
from my_experiments.target_metric import scorer_context  # noqa: E402
from toolkit.parallel import pimap  # noqa: E402

WANT = ("kappa", "li")
D3D = "DIII-D"


def _derive_maps(args: tuple) -> np.ndarray:
    """(T, len(WANT)) — the wanted scalars for a stack of maps, by the scorer's own code."""
    import local_score
    psi, ctx = args
    idx = [CONS_SCALARS.index(w) for w in WANT]
    out = np.full((len(psi), len(WANT)), np.nan)
    for k in range(len(psi)):
        c = extract_lcfs(psi[k], ctx["grid_R"], ctx["grid_Z"], ctx["machine"],
                         ctx["mask_coarse"], ctx["mask_f"], n_points=local_score.N_POINTS)
        vals = derive_frame(psi[k], ctx["grid_R"], ctx["grid_Z"], ctx["machine"],
                            ctx["mask_coarse"], ctx["mask_f"], contour=c, with_li=True)
        for j, name in enumerate(WANT):
            out[k, j] = vals[name]
    return out


def derive_all(psi: np.ndarray, ctx: dict, jobs: int, desc: str) -> np.ndarray:
    """Derive over a big stack, chunked so each worker gets a reasonable slice."""
    chunks = np.array_split(psi, max(jobs, 1) * 4)
    print(f"    {desc}: {len(psi)} frames over {jobs} processes")
    return np.vstack(list(pimap(_derive_maps, [(c, ctx) for c in chunks if len(c)], jobs)))


def fit_linear(C: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Least squares with an intercept — the linearisation the loss effectively carries."""
    A = np.hstack([C, np.ones((len(C), 1))])
    return np.linalg.lstsq(A, Y, rcond=None)[0]


def apply_linear(W: np.ndarray, C: np.ndarray) -> np.ndarray:
    return np.hstack([C, np.ones((len(C), 1))]) @ W


def fit_surrogate(C: np.ndarray, Y: np.ndarray, hidden: int, steps: int, seed: int):
    """Two hidden layers, plain MSE on standardised targets. Small on purpose: this is a gate."""
    import torch
    torch.manual_seed(seed)
    cm, cs = C.mean(0), C.std(0) + 1e-12
    ym, ys = Y.mean(0), Y.std(0) + 1e-12
    Xt = torch.tensor((C - cm) / cs, dtype=torch.float32)
    Yt = torch.tensor((Y - ym) / ys, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(C.shape[1], hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    n = len(Xt)
    for i in range(steps):
        idx = torch.randint(0, n, (min(4096, n),))
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(Xt[idx]), Yt[idx])
        loss.backward()
        opt.step()
    net.eval()

    def predict(Cq: np.ndarray) -> np.ndarray:
        import torch as T
        with T.no_grad():
            z = net(T.tensor((Cq - cm) / cs, dtype=T.float32)).numpy()
        return z * ys + ym

    return predict, float(loss.item())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", type=Path, default=Path("artefacts/lanes/a.joblib"))
    ap.add_argument("--shots", type=int, default=12, help="clean-tail shots to probe")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--share", type=float, default=0.05)
    ap.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR)
    ap.add_argument("--config", default=HF_TRAIN_CONFIG)
    a = ap.parse_args()

    import pandas as pd
    from experiments import _as_psirz_stack
    from my_experiments.baseline_model import predict_row

    art = joblib.load(a.artifact)
    salt = int(art.get("split_salt", 0))
    seen = set(art.get("train_files", ())) | set(art.get("val_files", ()))
    tail = take_share(sorted_shots(a.local_data_dir, a.config, salt), a.share, "tail")
    if [p for p in tail if p.name in seen]:
        raise SystemExit("tail overlaps the fit — the error scale would be the training one")
    files = tail[:a.shots]
    print(f"  artifact {a.artifact}  salt {salt}  probing {len(files)} clean-tail shots")

    pca = art["pca"]
    rows, psi_true, psi_pred = [], [], []
    for p in files:
        row = pd.read_parquet(p).iloc[0]
        rows.append(row)
        psi_true.append(_as_psirz_stack(row["efit_psirz"]).astype(np.float64))
        psi_pred.append(np.asarray(predict_row(row, D3D, "ensemble", a.artifact)["psirz"]))
    psi_true = np.vstack(psi_true)
    psi_pred = np.vstack(psi_pred)
    print(f"  {len(psi_true)} frames")

    # Coefficients. The basis is orthonormal, so transform(decode(c)) == c and this is exact.
    c_true = np.asarray(pca.transform(psi_true), dtype=np.float64)
    c_pred = np.asarray(pca.transform(psi_pred), dtype=np.float64)
    dc = c_pred - c_true
    scale = dc.std(axis=0)
    print(f"  model error per coefficient: median {np.median(scale):.4g}, "
          f"max {scale.max():.4g}")

    mask = np.load(HERE.parent / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    ctx = scorer_context(mask["grid_R"], mask["grid_Z"], D3D)
    # dec(c) rather than the raw truth: the question is how the functional responds to a change
    # IN THE BASIS, which is the only thing the model can move.
    dec = lambda C: np.asarray(pca.inverse_transform(C))
    print("  deriving the actual functional values")
    f_true = derive_all(dec(c_true), ctx, a.jobs, "at c_true")
    f_pred = derive_all(dec(c_pred), ctx, a.jobs, "at c_pred")

    # Fit both predictors on coefficients PERTURBED at the model's own error scale — the regime the
    # gradient is consumed in. A separate draw per frame, isotropic in the same per-coefficient
    # scale the model actually errs by.
    rng = np.random.default_rng(0)
    c_pert = c_true + rng.normal(size=c_true.shape) * scale
    print("  deriving the perturbed sample that both predictors are fitted on")
    f_pert = derive_all(dec(c_pert), ctx, a.jobs, "at c_pert")

    ok = np.isfinite(f_true).all(1) & np.isfinite(f_pred).all(1) & np.isfinite(f_pert).all(1)
    print(f"  {ok.sum()} of {len(ok)} frames have all values finite\n")
    half = ok.nonzero()[0]
    rng.shuffle(half)
    fit_idx, test_idx = half[: len(half) // 2], half[len(half) // 2:]

    W = fit_linear(c_pert[fit_idx], f_pert[fit_idx])
    g, gl = fit_surrogate(c_pert[fit_idx], f_pert[fit_idx], a.hidden, a.steps, 0)
    print(f"  surrogate final training loss (standardised): {gl:.5f}\n")

    actual = f_pred[test_idx] - f_true[test_idx]
    preds = {
        "linear": apply_linear(W, c_pred[test_idx]) - apply_linear(W, c_true[test_idx]),
        "surrogate": g(c_pred[test_idx]) - g(c_true[test_idx]),
        "constant": np.zeros_like(actual) + actual.mean(0),
    }
    print(f"  === correlation of PREDICTED vs ACTUAL change, on {len(test_idx)} held-out frames ===")
    print(f"  {'method':<12}" + "".join(f"{w:>12}" for w in WANT))
    for name, P in preds.items():
        cors = []
        for j in range(len(WANT)):
            x, y = P[:, j], actual[:, j]
            cors.append(np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else float("nan"))
        print(f"  {name:<12}" + "".join(f"{c:>12.3f}" for c in cors))
    ks = WANT.index("kappa")
    kc = np.corrcoef(preds["surrogate"][:, ks], actual[:, ks])[0, 1]
    print(f"\n  GATE: kappa correlation for the surrogate is {kc:.3f}; L3 requires > 0.90")
    print(f"  -> {'PROCEED' if kc > 0.90 else 'BOTH SURROGATE ARMS CLOSE'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

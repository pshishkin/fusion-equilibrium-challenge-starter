"""`mapnet` — the flux map predicted DIRECTLY, with no PCA basis anywhere.

The other two learned models in this repo predict a target VECTOR that a fixed PCA basis then
decodes. This one predicts the 65x65 map itself: features -> encoder -> a learned latent -> a
decoder that is a learned linear map plus a convolutional refinement, with q95 and betaN read off
the same latent. Nothing about the representation is chosen in advance
the latent is whatever
training makes it.

It is deliberately NOT a `TargetModel`. The pipeline's model interface assumes coefficients, and
that assumption is the thing being tested. Instead this writes predictions to an `.npz` and hands
them to `local_score.py --pred`, so every number it produces is the competition composite computed
by the scorer's own code on the same held-out tail production is scored on.

WHAT IS ESTABLISHED, measured 2026-08-31 on 352 tail shots of salt 0 (list sha1 0d4a15c2aab058ed,
verified identical to production's, not merely "the same tail"):

    one member        0.9855-0.9872 over four seeds, mean 0.986501, sd 0.00058
    four averaged     0.989124
    A, four MLPs      0.992976
    A + seq           0.993453

Three things moved it and are larger than the seed spread: a LEARNED LINEAR BRANCH beside the
convolutions (PCA proves a linear decoder reaches R2_psi 0.9998, and asking transposed convolutions
to rediscover that from a 9x9 seed wastes their capacity)
weighting pixels INSIDE the plasma
envelope, which covers 57.8% of the grid so a plain loss spends 42% of itself where nothing is
scored
and MATCHING GRADIENTS, because every scored functional reads a level set whose position is
delta-psi over |grad psi| — a smooth map is right on average and unreadable as geometry.

Three things did not, and all lie within the seed spread rather than being refuted: bilinear
upsampling instead of transposed convolutions, coordinate channels, and weighting the gradient term
by 1/R. Latent 256 IS refuted, at -0.0116.

The gap to A is one term: Consistency 0.9437 against 0.9740, which is 85% of the shortfall.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, 'my_experiments')
sys.path.insert(0, '.')


MAPNET_KIND = "mapnet"


def build_nets(n_in: int, latent: int, grid: int, upsample: str, coords: bool, dev: str,
               seed: int = 0) -> dict:
    """Every module of a mapnet, built the same way for training and for inference.

    One function rather than two because a decoder rebuilt slightly differently at predict time is
    a silent corruption: it loads, it runs, and it returns a map that is nobody's model.
    """
    import torch
    torch.manual_seed(seed)
    enc = torch.nn.Sequential(torch.nn.Linear(n_in, 512), torch.nn.GELU(),
                              torch.nn.Linear(512, 512), torch.nn.GELU(),
                              torch.nn.Linear(512, latent)).to(dev)
    lin = torch.nn.Linear(latent, grid * grid).to(dev)
    seed_mod = torch.nn.Sequential(torch.nn.Linear(latent, 64 * 9 * 9), torch.nn.GELU(),
                                   torch.nn.Unflatten(1, (64, 9, 9))).to(dev)
    extra = 2 if coords else 0

    class Up(torch.nn.Module):
        def __init__(self, cin, cout, mode):
            super().__init__()
            self.mode = mode
            self.op = (torch.nn.ConvTranspose2d(cin, cout, 4, 2, 1) if mode == "convt"
                       else torch.nn.Conv2d(cin, cout, 3, 1, 1))

        def forward(self, x):
            if self.mode == "bilinear":
                x = torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear",
                                                    align_corners=False)
            return self.op(x)

    body = torch.nn.Sequential(
        Up(64 + extra, 48, upsample), torch.nn.GELU(),
        Up(48, 32, upsample), torch.nn.GELU(),
        Up(32, 16, upsample), torch.nn.GELU(),
        torch.nn.Conv2d(16, 1, 3, 1, 1)).to(dev)
    qb = torch.nn.Sequential(torch.nn.Linear(latent, 128), torch.nn.GELU(),
                             torch.nn.Linear(128, 2)).to(dev)
    return {"enc": enc, "lin": lin, "seed": seed_mod, "body": body, "qb": qb}


def mapnet_predict(art: dict, Xs, grid: int):
    """Decoded maps and scalars from a mapnet artifact. Runs on the CPU, like every other member.

    Mirrors `TorchMLPModel.predict`'s promise: a submission is scored on a machine this fork does
    not control, so inference must never require a GPU.
    """
    import numpy as np
    import torch
    cfg = art["mapnet"]
    nets = build_nets(cfg["n_in"], cfg["latent"], grid, cfg["upsample"], cfg["coords"], "cpu")
    for k, mod in nets.items():
        mod.load_state_dict({kk: torch.from_numpy(vv) for kk, vv in cfg["state"][k].items()})
        mod.eval()
    coord = None
    if cfg["coords"]:
        rr, zz = np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9), indexing="xy")
        coord = torch.from_numpy(np.stack([rr, zz]).astype(np.float32))[None]
    out_m, out_q = [], []
    with torch.no_grad():
        for i in range(0, len(Xs), 4096):
            x = torch.from_numpy(np.ascontiguousarray(Xs[i:i + 4096], dtype=np.float32))
            z = nets["enc"](x)
            h = nets["seed"](z)
            if coord is not None:
                h = torch.cat([h, coord.expand(len(z), -1, -1, -1)], dim=1)
            m = nets["lin"](z) + nets["body"](h)[:, 0, :grid, :grid].reshape(len(z), -1)
            out_m.append(m.numpy())
            out_q.append(nets["qb"](z).numpy())
    m = np.concatenate(out_m) * cfg["ys"] + cfg["ym"]
    q = np.concatenate(out_q) * cfg["qs"] + cfg["qm"]
    return m.reshape(len(Xs), grid, grid), q


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=3000, help="training shots to read")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--latent", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--salt", type=int, default=0)
    ap.add_argument("--mask-weight", type=float, default=0.0,
                    help="weight pixels INSIDE the scorer's plasma envelope this many times more "
                         "than those outside; 0 keeps a uniform pixel loss")
    ap.add_argument("--sobolev", type=float, default=0.0,
                    help="weight on matching the map's GRADIENTS as well as its values")
    ap.add_argument("--upsample", default="convt", choices=["convt", "bilinear"],
                    help="convt = transposed convolutions (checkerboard); bilinear = resize then "
                         "a plain convolution, which has no periodic amplitude artefact")
    ap.add_argument("--coords", action="store_true",
                    help="append normalised R and Z channels to the decoder's seed, so the "
                         "convolutions know WHERE they are")
    ap.add_argument("--sob-physical", action="store_true",
                    help="weight the gradient term by 1/R, because B_p = |grad psi| / R: the same "
                         "flux slope is twice the field at half the radius")
    ap.add_argument("--seed", type=int, default=0,
                    help="init seed; members of a C ensemble differ only in this")
    ap.add_argument("--tag", default="v")
    ap.add_argument("--save-artifact", default="",
                    help="also write a self-contained artifact usable as an ensemble "
                         "member via salts:<name>+<name>")
    ap.add_argument("--out", default="/tmp/encdec_pred.npz")
    a = ap.parse_args()

    import baseline_model as B
    files = B.sorted_shots(salt=a.salt)
    train_files, _ = B.split_train_val(files, 0.80, 0.15)
    score_files = B.take_share(files, 0.05, "tail")
    train_files = train_files[:a.shots]
    print(f"  {len(train_files)} training shots, {len(score_files)} scoring shots", flush=True)

    Xt, psit, qbt, _ = B._read_shots(train_files, "train", 1.0, 10)
    psit = psit.astype(np.float32)
    plan = B.coil_plan_from_artifact() if hasattr(B, 'coil_plan_from_artifact') else None
    import joblib
    art = joblib.load('artefacts/lanes/baseline_q_base_s0.joblib')
    plan = art['coil']
    G = psit.shape[1]
    St = art['scaler'].transform(B.build_inputs(plan, Xt)).astype(np.float32)
    Yt = (psit - B.coil_flux(plan, Xt).astype(np.float32)).reshape(len(psit), -1)
    del psit
    ym, ys = float(Yt.mean()), float(Yt.std())
    qm, qs = qbt.mean(axis=0), qbt.std(axis=0) + 1e-12
    print(f"  train X {St.shape}, maps {Yt.shape}, {Yt.nbytes/2**30:.1f} GiB", flush=True)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(a.seed)
    enc = torch.nn.Sequential(torch.nn.Linear(St.shape[1], 512), torch.nn.GELU(),
                              torch.nn.Linear(512, 512), torch.nn.GELU(),
                              torch.nn.Linear(512, a.latent)).to(dev)
    # TWO PATHS to the map, and the linear one is the point of this version.
    #
    # PCA proves a LINEAR decoder reaches R2_psi 0.9998: a weighted sum of 50 whole-grid pictures
    # is almost the whole job. v1 asked transposed convolutions to rediscover that from a 9x9 seed,
    # which is a hard way to learn something a single matrix expresses exactly — and v1 landed at
    # R2_psi 0.9955 against production's 0.9998.
    #
    # `lin` is that matrix, but LEARNED jointly with the encoder rather than fitted to explain
    # variance. It is not PCA: PCA's basis is chosen before any model exists and maximises
    # reconstructed variance; this one is chosen by the composite's own gradient. The convolutional
    # branch then does what convolutions are for — local structure, which is where the boundary
    # lives and where a global basis is weakest.
    class Up(torch.nn.Module):
        """One upsampling stage.

        `convt` is a transposed convolution: kernel 4, stride 2, which is the standard recipe and
        the standard source of CHECKERBOARD artefacts — a periodic amplitude ripple where the
        kernel's stride does not divide its width evenly. Averaged over pixels that ripple is
        small, which is why v1-v6 reached R2_psi 0.9996 with it. But every scored functional reads
        a LEVEL SET, and a ripple crossing the contour displaces the contour: exactly the shape of
        this model's failure, Consistency 0.943 against a map that is otherwise nearly right.

        `bilinear` resizes first and then convolves, which cannot produce that pattern.
        """
        def __init__(self, cin, cout, mode):
            super().__init__()
            self.mode = mode
            self.op = (torch.nn.ConvTranspose2d(cin, cout, 4, 2, 1) if mode == "convt"
                       else torch.nn.Conv2d(cin, cout, 3, 1, 1))

        def forward(self, x):
            if self.mode == "bilinear":
                x = torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear",
                                                    align_corners=False)
            return self.op(x)

    # COORDINATES. A convolution is translation-equivariant and shares its weights across the grid,
    # but the plasma sits at a definite place on a definite machine and the physics carries an
    # explicit 1/R. The linear branch knows position by construction — each latent number owns a
    # whole picture — and the convolutional branch does not. Two channels holding normalised R and
    # Z give it that, at the cost of two numbers.
    seed_extra = 2 if a.coords else 0
    if a.coords:
        rr, zz = np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9), indexing="xy")
        coord = torch.from_numpy(np.stack([rr, zz]).astype(np.float32)).to(dev)[None]
    lin = torch.nn.Linear(a.latent, G * G).to(dev)
    seed = torch.nn.Sequential(torch.nn.Linear(a.latent, 64 * 9 * 9), torch.nn.GELU(),
                               torch.nn.Unflatten(1, (64, 9, 9))).to(dev)
    body = torch.nn.Sequential(
        Up(64 + seed_extra, 48, a.upsample), torch.nn.GELU(),
        Up(48, 32, a.upsample), torch.nn.GELU(),
        Up(32, 16, a.upsample), torch.nn.GELU(),
        torch.nn.Conv2d(16, 1, 3, 1, 1)).to(dev)
    dec = torch.nn.ModuleList([seed, body]).to(dev)

    def decode(z):
        h = seed(z)
        if a.coords:
            h = torch.cat([h, coord.expand(len(z), -1, -1, -1)], dim=1)
        return body(h)
    # Zeroed so the model STARTS as the linear decoder alone and the convolutions have to earn
    # every change they make — the same discipline as seq's residual head.
    torch.nn.init.zeros_(body[-1].weight)
    torch.nn.init.zeros_(body[-1].bias)
    qb = torch.nn.Sequential(torch.nn.Linear(a.latent, 128), torch.nn.GELU(),
                             torch.nn.Linear(128, 2)).to(dev)
    ps = (list(enc.parameters()) + list(dec.parameters()) + list(lin.parameters())
          + list(qb.parameters()))
    opt = torch.optim.Adam(ps, lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    # PIXEL WEIGHTS. The scorer's envelope covers 57.8% of the grid, so a uniform pixel loss spends
    # 42% of itself outside the plasma, where no scored functional reads anything. Normalised to
    # mean 1 so the loss keeps its scale and only its SHAPE changes.
    # 1/R for the gradient term. `poloidal_field_form` on the PCA side has carried this since it
    # was written — B_p = |grad psi| / R, so the same flux slope means twice the poloidal field at
    # half the major radius, and a plain sum of squared differences weights the inboard side as if
    # it were the outboard. Applied as sqrt(1/R) on each difference, which squares to the 1/R the
    # field error actually carries.
    rw = None
    if a.sob_physical:
        r1 = np.sqrt(1.0 / np.asarray(plan['grid_R'], dtype=np.float32))
        rw = torch.from_numpy(r1).to(dev)                       # (G,) along the R axis
        print(f"  gradient term weighted by 1/R: "
              f"{float(r1.min()**2):.3f}..{float(r1.max()**2):.3f}", flush=True)
    pw = None
    if a.mask_weight:
        from target_metric import scorer_context
        _ctx = scorer_context(plan['grid_R'], plan['grid_Z'], plan['machine'])
        w = np.asarray(_ctx['mask_f'], dtype=np.float32).reshape(-1)
        w = 1.0 + (a.mask_weight - 1.0) * w
        w = w / w.mean()
        pw = torch.from_numpy(w).to(dev)
        print(f"  pixel weights: inside {float(w[w > w.min()].mean()):.3f}, "
              f"outside {float(w.min()):.3f}, mean {float(w.mean()):.3f}", flush=True)
    Xg = torch.from_numpy(St).to(dev)
    Qg = torch.from_numpy(((qbt - qm) / qs).astype(np.float32)).to(dev)
    Yc = torch.from_numpy(Yt)                        # stays on the CPU: too large for the GPU

    def fwd(x):
        z = enc(x)
        return lin(z) + decode(z)[:, 0, :G, :G].reshape(len(x), -1), qb(z)

    for k in range(a.steps):
        ib = torch.randint(0, len(Xg), (a.batch,))
        yb = ((Yc[ib] - ym) / ys).to(dev, non_blocking=True)
        m, q = fwd(Xg[ib.to(dev)])
        opt.zero_grad()
        mloss = (((m - yb) ** 2).mean() if pw is None else (((m - yb) ** 2) * pw).mean())
        if a.sobolev:
            # THE CONSISTENCY TERM IS GEOMETRY, AND GEOMETRY IS DERIVATIVES. All seven scored
            # scalars come from the LCFS contour and moments over it — a level set, whose position
            # is set by where the flux crosses a value and whose shape is set by the gradient
            # there. A pixel MSE constrains VALUES and is blind to the field's slope, which is why
            # v2 reached R2_psi 0.9996 while Consistency sat at 0.9306: the map is right on average
            # and locally too smooth for a contour to be read out of it.
            M2 = m.reshape(-1, G, G)
            Y2 = yb.reshape(-1, G, G)
            gx = (M2[:, :, 1:] - M2[:, :, :-1]) - (Y2[:, :, 1:] - Y2[:, :, :-1])
            gy = (M2[:, 1:, :] - M2[:, :-1, :]) - (Y2[:, 1:, :] - Y2[:, :-1, :])
            if rw is not None:
                gx = gx * rw[None, None, 1:]
                gy = gy * rw[None, None, :]
            mloss = mloss + a.sobolev * ((gx ** 2).mean() + (gy ** 2).mean())
        loss = mloss + 0.1 * torch.nn.functional.mse_loss(q, Qg[ib.to(dev)])
        loss.backward()
        opt.step()
        sch.step()
        if (k + 1) % 10000 == 0:
            print(f"    step {k+1:6d}  loss {float(loss):.6f}", flush=True)
    enc.eval()
    dec.eval()
    lin.eval()
    qb.eval()

    print("  predicting the scoring shots...", flush=True)
    out = {}
    for i, f in enumerate(score_files):
        X, _psi, _, _ = B._read_shots([f], "", 1.0, 1)
        S = art['scaler'].transform(B.build_inputs(plan, X)).astype(np.float32)
        with torch.no_grad():
            m, q = fwd(torch.from_numpy(S).to(dev))
        M = m.cpu().numpy() * ys + ym
        M = M.reshape(-1, G, G) + B.coil_flux(plan, X)
        Q = q.cpu().numpy() * qs + qm
        out[f"shot_{i:04d}_psirz"] = M.astype(np.float32)
        out[f"shot_{i:04d}_q95"] = Q[:, 0].astype(np.float32)
        out[f"shot_{i:04d}_betaN"] = Q[:, 1].astype(np.float32)
    if a.save_artifact:
        # SELF-CONTAINED, because a blend member that borrows another artifact's scaler is a member
        # that silently changes when that artifact is retrained. Weights are stored as numpy so the
        # artifact unpickles without torch, matching every other model in this repo.
        import joblib as _jl
        nets = {"enc": enc, "lin": lin, "seed": seed, "body": body, "qb": qb}
        state = {k: {kk: vv.detach().cpu().numpy() for kk, vv in mod.state_dict().items()}
                 for k, mod in nets.items()}
        _jl.dump({
            "kind": MAPNET_KIND,
            "scaler": art["scaler"], "coil": art["coil"], "n_pca": art["n_pca"],
            "train_files": [str(f) for f in train_files],
            "mapnet": {"n_in": St.shape[1], "latent": a.latent, "upsample": a.upsample,
                       "coords": bool(a.coords), "state": state,
                       "ym": ym, "ys": ys, "qm": qm, "qs": qs},
        }, a.save_artifact)
        print(f"  wrote artifact {a.save_artifact} — usable as salts:<other>+"
              f"{Path(a.save_artifact).stem}", flush=True)
    np.savez(a.out, **out)
    print(f"  wrote {a.out}", flush=True)
    print("  score it with:")
    print(f"    uv run python local_score.py --source local --pred {a.out} --files " +
          " ".join(str(f) for f in score_files[:2]) + " ...")
    with open('/tmp/encdec_files.txt', 'w') as fh:
        fh.write("\n".join(str(f) for f in score_files))
    print("  scoring file list at /tmp/encdec_files.txt", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

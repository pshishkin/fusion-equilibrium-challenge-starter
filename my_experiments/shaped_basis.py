"""The PCA basis fitted under the metric that SCORES it, instead of under pixel variance.

`experiments.TargetPCA` chooses its 50 directions to explain the most squared error per pixel,
counted equally everywhere on the 65x65 grid. Nothing in the competition metric counts that way:

  * 42.2% of the grid lies OUTSIDE the scorer's plasma envelope, where no scored functional reads
    anything at all, and a plain basis spends its capacity there in proportion to the variance it
    happens to find.
  * Six of the seven Consistency functionals and D_LCFS are read off a CONTOUR — a level set, whose
    position and shape are set by the flux GRADIENT where it crosses. A basis chosen to reproduce
    VALUES is free to be locally too smooth, and the encoder-decoder series measured exactly that:
    `c_v1` reached R2_psi 0.9996 with Consistency at 0.9278.

`mapnet` v7 answered both with one loss — the envelope weighted 16x and a term matching first
differences (`c_v5` 0.985139, `c_v6` 0.985652, `c_v7` 0.985811 from 0.982821 without them). That
loss is a QUADRATIC FORM in the map error, so it can be handed to the basis directly rather than
only to a network:

    ||r||_A^2  =  (1/G^2) sum_i w_i r_i^2  +  lambda/(G(G-1)) * (||Dx r||^2 + ||Dy r||^2)

with `w` the envelope weights normalised to mean 1 and `Dx`, `Dy` the same one-pixel forward
differences v7 uses. Both normalisers are v7's own `.mean()`s, so `lambda` here is v7's `--sobolev`
on the same scale: the map's global standardisation `(Y - ym) / ys` multiplies both terms by the
same `1/ys^2` and therefore cancels out of their ratio.

**The A-optimal subspace, by one Cholesky.** `A = R^T R`, so `||r||_A = ||R r||_2` and the rank-k
subspace minimising the A-weighted reconstruction error is just ORDINARY PCA of the transformed
maps `z = R y`, pulled back through `R^-1`. No iteration, no new hyper-parameter beyond the two the
form already has.

**Why the coordinates are then rebuilt, and this is the important part.** The pull-back basis is
not orthonormal in pixel space, and three things downstream depend on the basis being orthonormal:
`TargetScaler` divides the whole psi block by one number, which is Parseval and false otherwise;
the Jacobian metric's identity term IS that same Parseval statement; and `R2_psi` is measured in
pixels, not in coefficients. So this module keeps the SUBSPACE the metric chose and throws its
coordinate frame away: re-orthonormalise (QR), then re-diagonalise the training covariance inside
the subspace (eigh). What comes back is orthonormal, decorrelated and variance-ordered — a PCA in
every respect that anything downstream can observe — differing from `TargetPCA` in one thing only,
WHICH 50 directions it spans.

That restriction is deliberate and it costs something: reconstruction stays the ORTHOGONAL
projection onto the subspace, where the A-optimal reconstruction would be the A-orthogonal one.
The A-oblique variant is a strictly larger change — it breaks the three assumptions above and has
to pay for repairing them — and is worth building only if choosing the subspace alone moves the
score. This measures that first.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cholesky, solve_triangular

from experiments import EFIT_GRID_SIZE, TargetPCA

FloatArray = np.ndarray


def _difference_operator(n: int) -> sp.csr_matrix:
    """(n-1, n) forward difference, the same one v7 takes on the raw pixel index.

    Not divided by the grid spacing: v7's gradient term is a difference between NEIGHBOURING
    PIXELS, and dividing by dR and dZ separately would silently reweight the two axes against each
    other by (dR/dZ)^2 relative to what was measured there.
    """
    return sp.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n)).tocsr()


def metric_matrix(mask_f: FloatArray, envelope: float, sobolev: float,
                  grid: int = EFIT_GRID_SIZE) -> FloatArray:
    """(P, P) SPD `A` with `r^T A r` = v7's map loss on the error `r`, flattened row-major.

    `envelope` is v7's `--mask-weight` (1.0 = uniform), `sobolev` its `--sobolev` (0.0 = off).
    At (1.0, 0.0) this is `I / G^2` and the basis below reduces to `TargetPCA` exactly, which is
    what the smoke test asserts.
    """
    if not envelope > 0:
        raise ValueError(f"envelope must be positive, got {envelope!r}")
    if sobolev < 0:
        raise ValueError(f"sobolev must be non-negative, got {sobolev!r}")
    p = grid * grid
    w = 1.0 + (envelope - 1.0) * np.asarray(mask_f, dtype=np.float64).reshape(-1)
    if w.size != p:
        raise ValueError(f"mask has {w.size} pixels, grid says {p}")
    w = w / w.mean()
    a = sp.diags(w / p)
    if sobolev:
        d1 = _difference_operator(grid)
        eye = sp.identity(grid, format="csr")
        dx = sp.kron(eye, d1, format="csr")      # along R, within each Z row
        dy = sp.kron(d1, eye, format="csr")      # along Z
        a = a + (sobolev / (grid * (grid - 1))) * (dx.T @ dx + dy.T @ dy)
    return np.asarray(a.toarray(), dtype=np.float64)


class ShapedTargetPCA(TargetPCA):
    """`TargetPCA` whose subspace is chosen under `metric_matrix`, not under pixel variance.

    Drop-in: it fills the same `self.pca.components_` / `mean_` that everything downstream reads,
    so the artifact, `predict_row`, the Jacobian metric and every diagnostic are untouched.
    """

    def __init__(self, n_components: int, mask_f: FloatArray, envelope: float,
                 sobolev: float) -> None:
        super().__init__(n_components=n_components)
        self.mask_f = np.asarray(mask_f, dtype=np.float64)
        self.envelope = float(envelope)
        self.sobolev = float(sobolev)

    def fit(self, Y: FloatArray) -> ShapedTargetPCA:
        yf = Y.reshape(len(Y), -1).astype(np.float64)   # our own copy, overwritten in place below
        n, p = yf.shape
        if self.n_components > min(n, p):
            raise ValueError(f"{self.n_components} components from {n} frames of {p} pixels")
        grid = round(p ** 0.5)
        # `A = R^T R`, R upper triangular. A is positive definite by construction (the envelope
        # term alone is), so a failure here means the mask is not what it claims to be rather than
        # a numerical accident — hence no fallback.
        r = cholesky(metric_matrix(self.mask_f, self.envelope, self.sobolev, grid), lower=False)
        # z = R y, IN PLACE, accumulating the plain moments on the way past. At production this
        # array is 250k frames of 4225 pixels — 8 GB in float64 — and holding y and z at once, on
        # top of the centred copy sklearn makes inside `fit`, is what would make a shaped basis
        # cost more memory than a plain one. It does not. The moments have to be taken BEFORE the
        # overwrite: `explained_variance_ratio_` is measured against the variance of psi in PIXELS,
        # and after the transform that number is not recoverable from what is left.
        ysum = np.zeros(p, dtype=np.float64)
        ysq = 0.0
        for lo in range(0, n, 4096):
            block = yf[lo:lo + 4096]
            ysum += block.sum(axis=0)
            ysq += float(np.einsum("ij,ij->", block, block))
            yf[lo:lo + 4096] = block @ r.T
        mean = ysum / n
        total_var = (ysq - n * float(mean @ mean)) / (n - 1)
        self.pca.fit(yf)                                          # yf holds z now, not y
        # Pull the z-space components back to pixels: b_i = R^-1 w_i, then orthonormalise. The QR
        # keeps the SPAN and throws away the frame, which is the whole design (see the docstring).
        pulled = solve_triangular(r, np.asarray(self.pca.components_).T, lower=False)   # (P, k)
        q, _ = np.linalg.qr(pulled)                                                     # (P, k)
        # Coefficients of the training maps in that subspace, still without restoring y:
        # c = (R^-1 z - mean) Q = z (R^-T Q) - mean Q.
        qt = solve_triangular(r, q, lower=False, trans="T")                             # (P, k)
        offset = mean @ q
        c = np.empty((n, self.n_components), dtype=np.float64)
        for lo in range(0, n, 4096):
            c[lo:lo + 4096] = yf[lo:lo + 4096] @ qt - offset
        # Re-diagonalise inside the subspace, so what comes out is decorrelated and
        # variance-ordered exactly as `TargetPCA`'s is.
        evals, evecs = np.linalg.eigh((c.T @ c) / (n - 1))
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]
        self.pca.components_ = np.ascontiguousarray((q @ evecs).T)
        self.pca.mean_ = mean
        self.pca.explained_variance_ = evals
        self.pca.explained_variance_ratio_ = evals / total_var
        self.pca.singular_values_ = np.sqrt(np.maximum(evals, 0.0) * (n - 1))
        self.pca.n_components_ = self.n_components
        self.pca.n_features_in_ = p
        self.pca.n_samples_ = n
        self.pca.noise_variance_ = 0.0
        self._fitted = True
        return self


def build_target_pca(n_pca: int, mask_f: FloatArray | None, envelope: float,
                     sobolev: float) -> TargetPCA:
    """`TargetPCA` when the metric is uniform, `ShapedTargetPCA` when it is not.

    Returning the plain class at (1.0, 0.0) rather than the shaped one with a trivial metric is not
    an optimisation: it keeps the DEFAULT path byte-identical to every number measured before this
    module existed, so a control arm is the old code and not a re-derivation of it.
    """
    if envelope == 1.0 and sobolev == 0.0:
        return TargetPCA(n_components=n_pca)
    if mask_f is None:
        raise ValueError("a shaped basis needs the scorer's envelope mask")
    return ShapedTargetPCA(n_pca, mask_f, envelope, sobolev)

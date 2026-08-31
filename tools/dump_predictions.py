"""A's predictions on the same 352 shots, in the same .npz layout, so A, C and their blends all go
through one scorer and differ only in what produced the map."""
import sys

import joblib
import numpy as np

sys.path.insert(0, 'my_experiments')
sys.path.insert(0, '.')
from pathlib import Path

import baseline_model as B

art = joblib.load(sys.argv[1])
out = sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else 'ensemble'
with open('/tmp/encdec_files_v7_m16sob.txt') as fh:
    files = [Path(x.strip()) for x in fh]
plan = art['coil']
n = art['n_pca']
res = {}
for i, f in enumerate(files):
    X, psi, _, _ = B._read_shots([f], "", 1.0, 1)
    S = art['scaler'].transform(B.build_inputs(plan, X))
    P = B._predict_targets(art, model, S)
    M = art['pca'].inverse_transform(P[:, :n]).reshape(len(X), psi.shape[1], -1) \
        + B.coil_flux(plan, X)
    res[f"shot_{i:04d}_psirz"] = M.astype(np.float32)
    res[f"shot_{i:04d}_q95"] = P[:, n].astype(np.float32)
    res[f"shot_{i:04d}_betaN"] = P[:, n + 1].astype(np.float32)
np.savez(out, **res)
print(f"  wrote {out} for model={model} over {len(files)} shots")

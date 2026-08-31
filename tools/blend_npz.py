"""Average several prediction .npz files into one, the way the production blend averages members.

`salts:a+b+c` averages DECODED predictions, not coefficients, because different fits have different
bases and their coefficients are not commensurable. These files are already decoded maps, so the
same rule applies trivially — and it is what makes a C member mixable with an A member at all.
"""
import sys

import numpy as np

out = sys.argv[1]
srcs = [np.load(p) for p in sys.argv[2:]]
keys = list(srcs[0].keys())
for s in srcs[1:]:
    if list(s.keys()) != keys:
        raise SystemExit("the inputs do not hold the same shots in the same order")
acc = {}
for k in keys:
    stack = np.stack([np.asarray(s[k], dtype=np.float64) for s in srcs])
    acc[k] = stack.mean(axis=0).astype(np.float32)
np.savez(out, **acc)
print(f"  averaged {len(srcs)} files over {len(keys)//3} shots -> {out}")

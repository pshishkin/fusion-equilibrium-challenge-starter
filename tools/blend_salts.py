#!/usr/bin/env python3
"""Average per-salt blends that were each computed once.

    uv run python tools/blend_salts.py --out submission_3salts \\
        --inputs .blend_cache/s1 .blend_cache/s2 .blend_cache/s3

**The identity this rests on.** Every salt carries the SAME composition with the SAME per-model
weights, so each salt's blend is a mean over the same total weight. The mean of equal-sized means is
the overall mean, exactly — averaging three per-salt blends is the thirty-artifact weighted blend,
not an approximation of it. It stops being exact the moment one salt carries a different
composition, which is why `--inputs` are checked for identical shot keys and shapes.

**Why this is the safe cache and coefficients were not.** What is stored is a decoded MAP. A map has
no invariant left to violate: nothing downstream can reinterpret it against the wrong basis, which
is the failure that killed `coef-cache` on 2026-09-04. A stale entry here is a previous valid blend
rather than a nonsense one — still wrong, still guarded by the key, but wrong in a way that cannot
masquerade as a modelling result.

**What it buys.** A submission over N salts costs one salt's predictions, not N. Adding salt 4 to an
existing three-salt blend costs salt 4 alone. On 2026-09-04 a three-salt build was seven hours in
and had recomputed salt 1, which had already been predicted for the single-salt submission.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.submission_format import (
    PSI_DTYPE,
    SCALAR_DTYPE,
    check_file,
    require_submission_dtypes,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", type=Path, required=True,
                    help="directories, one per salt, each holding <config>.npz")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--configs", nargs="+",
                    default=["diii_d_public_test", "mast_public_test"])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    for config in a.configs:
        paths = [d / f"{config}.npz" for d in a.inputs]
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"  {config}: skipped, missing {[str(p) for p in missing]}")
            continue
        out: dict[str, np.ndarray] = {}
        with ExitStack() as stack:
            zs = [stack.enter_context(np.load(p, allow_pickle=False)) for p in paths]
            for z, p in zip(zs, paths, strict=True):
                require_submission_dtypes(z, p)
            keys = sorted(zs[0].files)
            for z, p in zip(zs[1:], paths[1:], strict=True):
                if sorted(z.files) != keys:
                    raise SystemExit(f"{p} holds different shots than {paths[0]} — these are not "
                                     "the same submission over different salts, and averaging "
                                     "them would silently mix two shot orders")
            for k in keys:
                arrs = [z[k] for z in zs]
                shp = {tuple(x.shape) for x in arrs}
                if len(shp) != 1:
                    raise SystemExit(f"{k}: shapes differ across salts {shp}")
                acc = np.zeros(arrs[0].shape, dtype=np.float64)
                for x in arrs:
                    acc += x.astype(np.float64)
                dtype = PSI_DTYPE if k.endswith("_psirz") else SCALAR_DTYPE
                out[k] = (acc / len(arrs)).astype(dtype)
        dst = a.out / f"{config}.npz"
        np.savez_compressed(dst, **out)  # type: ignore[arg-type]
        check_file(dst)
        print(f"  {config}: {len(a.inputs)} salts x {len(keys) // 3} shots -> {dst} "
              f"({dst.stat().st_size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

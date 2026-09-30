#!/usr/bin/env python3
"""Which members does the linear stack need? Class and member ablation from sufficient statistics.

    uv run python tools/stack_ablation.py runs/linvar_fc24/stats.npz --members mlp,seq,... \
        --out runs/ablation_fc24

The ridgecv_block linear stage's held-out error depends only on per-fold, per-column statistics
G_j = A_j'A_j, b_j = A_j'y_j (`stack_linear_variants.fold_stats`), and a member subset is a
sub-block of them. So every subset is refitted on four folds and its held-out error on the fifth
is exact, in seconds, without loading a table:

1. drop one model CLASS (mlp, gru, lstm, conv, attn) at a time;
2. drop one member at a time;
3. greedy backward elimination: repeatedly drop the member whose removal lowers the pooled
   held-out psi MSE most, while it does.

Reported per subset: pooled psi MSE (basis units, the in-basis pixel MSE) and q95 / betaN MSE as
ratios to the full library, and on how many folds psi improves. This is a proxy: the score reads
Consistency off the map, so subsets worth keeping are scored natively with
`stack_linear_variants.py --members <subset> --score ridgecv_block`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stack_linear_variants import fit_ridge_cv, mse

FOLDS = [0, 1, 2, 3, 4]


def member_class(name: str) -> str:
    """mlp / gru / lstm / conv / attn, from the member spec `[run:]model`."""
    run, _, model = name.rpartition(":")
    if model == "mlp":
        return "mlp"
    for cls in ("lstm", "conv", "attn", "gru"):
        if cls in run:
            return cls
    return "gru"  # the default seq cell: cv5_fixcos seq, fc_seqseed1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stats", type=Path)
    ap.add_argument("--members", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    members = args.members.split(",")
    z = np.load(args.stats)
    stats = {f: (z[f"G_{f}"], z[f"b_{f}"], z[f"yy_{f}"], int(z[f"n_{f}"])) for f in FOLDS}
    J, M = stats[0][1].shape
    if M != len(members):
        raise ValueError(f"{args.stats} holds {M} members, --members names {len(members)}")
    k = J - 2
    cols = np.arange(J)
    groups = [cols[a:min(a + 10, k)] for a in range(0, k, 10)] + [np.array([k]), np.array([k + 1])]

    def held_out(keep: list[int]) -> np.ndarray:
        """(5 folds, J) held-out MSE of ridgecv_block on the member subset `keep`."""
        idx = np.asarray(keep)
        sub = {f: (G[:, idx][:, :, idx], b[:, idx], yy, n) for f, (G, b, yy, n) in stats.items()}
        out = []
        for f in FOLDS:
            W = fit_ridge_cv([sub[j] for j in FOLDS if j != f], groups)
            out.append(mse(W, sub[f]) * sub[f][3])
        return np.array(out)

    base = held_out(list(range(M)))
    report: dict = {"members": members, "rows": []}

    def row(label: str, keep: list[int]) -> dict:
        e = held_out(keep)
        psi = e[:, :k].sum(1) / base[:, :k].sum(1)
        r = {"label": label, "n": len(keep), "keep": [members[i] for i in keep],
             "psi": float(e[:, :k].sum() / base[:, :k].sum()),
             "q95": float(e[:, k].sum() / base[:, k].sum()),
             "betaN": float(e[:, k + 1].sum() / base[:, k + 1].sum()),
             "psi_folds": psi.tolist(), "psi_better": int((psi < 1).sum())}
        print(f"{label:44s} n={len(keep):2d} psi {r['psi']:.4f} ({r['psi_better']}/5 better) "
              f"q95 {r['q95']:.4f} betaN {r['betaN']:.4f}", flush=True)
        report["rows"].append(r)
        (args.out / "report.json").write_text(json.dumps(report, indent=1))
        return r

    classes = sorted({member_class(m) for m in members})
    print("classes: " + ", ".join(f"{c} {sum(member_class(m) == c for m in members)}"
                                  for c in classes), flush=True)
    for c in classes:
        row(f"drop class {c}", [i for i, m in enumerate(members) if member_class(m) != c])
    for c in classes:
        row(f"only class {c}", [i for i, m in enumerate(members) if member_class(m) == c])
    for i, m in enumerate(members):
        row(f"drop {m}", [j for j in range(M) if j != i])
    keep = list(range(M))
    cur = 1.0
    while len(keep) > 2:
        trials = {i: held_out([j for j in keep if j != i]) for i in keep}
        best = min(trials, key=lambda i: trials[i][:, :k].sum())
        score = trials[best][:, :k].sum() / base[:, :k].sum()
        if score >= cur:
            break
        keep.remove(best)
        cur = score
        row(f"greedy: -{members[best]}", keep)
    print(f"greedy stops at {len(keep)} members: " + ",".join(members[i] for i in keep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

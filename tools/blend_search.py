#!/usr/bin/env python3
"""Search the collection for the blend that scores best: a coarse pass, then greedy addition.

    uv run python tools/blend_search.py --salt 1 --jobs 20

**The user's design, 2026-09-03.** Rank the members by their own quality, blend the top 25/50/75/100
percent equally and keep the best of the four; then add members one at a time, keeping whatever
improves. The coarse pass exists because the greedy pass alone would be long.

**Why this is fast.** Scoring a blend through `evaluate.py` re-predicts every member and re-derives
the ground truth every time, and that is where the twelve minutes went. Here the ground truth's LCFS
extraction runs ONCE, each member's predictions are computed ONCE, and a combination costs only an
average of arrays plus the contour extraction on the result — the irreducible part. That turns a
greedy search from days into hours.

**The unit is an ARTIFACT, not a model, and this matters.** `salts:a+b` averages whole artifacts, so
`z_npca_base` — itself a five-model ensemble — is one unit and `c_conv`, a single sequence model, is
also one unit. Equal weighting therefore gives four single models four fifths of the weight against
production's five, which is the confound that made the `mixers` rung look bad on 2026-09-03. The
search is run over units because that is what the submission path can express, and the asymmetry is
stated rather than hidden.

**Selection is not evidence.** Everything here is chosen on ONE salt's tail, and a search over
combinations on the set it is measured on is exactly the shape that made the lambda sweep report
+0.000054 in-sample and -0.000300 out. The winner has to be confirmed on the OTHER salts, which the
search never saw, and by the untouched `local_score` CLI rather than by this file's fast path — a
scorer reimplemented for speed is a scorer that can disagree with the one that counts.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_score as LS
from my_experiments.baseline_model import (
    DEFAULT_LOCAL_DATA_DIR,
    HF_TRAIN_CONFIG,
    artifact_path,
    predict_row,
    sorted_shots,
    take_share,
)


def ground_truth_setup(shots, jobs):
    """Everything the scorer derives from the truth alone — computed once, reused by every blend."""
    machine, R, Z, mask_coarse, mask_f = LS._machine_of(shots)
    refs = LS._map(LS._ref_task,
                   [(s["psi"], R, Z, mask_coarse, mask_f, machine) for s in shots],
                   jobs, "  references")
    psi_sum = psi_sumsq = 0.0
    psi_n = 0
    scal_sum, scal_sumsq, scal_n = (np.zeros(len(LS.SCALARS)), np.zeros(len(LS.SCALARS)),
                                    np.zeros(len(LS.SCALARS), dtype=int))
    cons_sum, cons_sumsq, cons_n = np.zeros(LS.N_CONS), np.zeros(LS.N_CONS), np.zeros(LS.N_CONS)
    for s, ref in zip(shots, refs, strict=True):
        g = s["psi"].astype(np.float64)
        fin = np.isfinite(g)
        psi_sum += float(g[fin].sum())
        psi_sumsq += float((g[fin] ** 2).sum())
        psi_n += int(fin.sum())
        for j, name in enumerate(LS.SCALARS):
            v = np.asarray(s[name], dtype=np.float64)
            v = v[np.isfinite(v)]
            scal_sum[j] += v.sum()
            scal_sumsq[j] += (v ** 2).sum()
            scal_n[j] += v.size
        cons_gt, cmask = ref[1], ref[2]
        for j in range(LS.N_CONS):
            v = cons_gt[cmask[:, j], j]
            cons_sum[j] += v.sum()
            cons_sumsq[j] += (v ** 2).sum()
            cons_n[j] += v.size
    ref_stats = {"psi_sum": psi_sum, "psi_sumsq": psi_sumsq, "psi_n": psi_n,
                 "scal_sum": scal_sum, "scal_sumsq": scal_sumsq, "scal_n": scal_n,
                 "cons_sum": cons_sum, "cons_sumsq": cons_sumsq, "cons_n": cons_n}
    mean_psi = psi_sum / psi_n if psi_n else 0.0
    means = (mean_psi,
             np.where(scal_n > 0, scal_sum / np.where(scal_n > 0, scal_n, 1), 0.0),
             np.where(cons_n > 0, cons_sum / np.where(cons_n > 0, cons_n, 1), 0.0))
    return machine, R, Z, mask_coarse, mask_f, refs, ref_stats, means, mean_psi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--salt", type=int, default=1)
    ap.add_argument("--members", nargs="+", help="artifact names; default = every one of this salt")
    ap.add_argument("--share", type=float, default=0.05)
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=6, help="greedy additions to attempt")
    ap.add_argument("--out", type=Path, default=Path("results/x_blend_search.csv"))
    a = ap.parse_args()

    names = a.members or sorted(
        p.stem.removeprefix("baseline_")
        for p in Path("artefacts/lanes").glob(f"baseline_*_s{a.salt}.joblib")
        if p.stem.removeprefix("baseline_").startswith(("c_", "z_npca_", "z_sbasis_")))
    files = take_share(sorted_shots(DEFAULT_LOCAL_DATA_DIR, HF_TRAIN_CONFIG, a.salt),
                       a.share, "tail")
    print(f"  salt {a.salt}: {len(names)} artifacts, {len(files)} tail shots")

    # The members must not have seen the shots they are scored on. Checked, not assumed.
    seen = set()
    # THE WEIGHT OF AN ARTIFACT IS HOW MANY MODELS IT HOLDS. `z_npca_base` is a five-model
    # ensemble and `c_conv` is one sequence model; giving them the same say puts one model against
    # five and calls it a blend. That is what made the `mixers` rung of x_blend_eval lose on all
    # three salts — four single models took four fifths of the weight from production.
    #
    # Each artifact's internal ensemble is already the EQUAL average of its members, so weighting
    # the artifact by its member count is exactly "every model carries weight 1", without paying
    # to predict each model separately.
    # DEDUPLICATED BY FITTED WEIGHTS, not by name. Two lanes that ran their control arm at the same
    # salt produce the same models byte for byte — `fit()` seeds itself — so z_sbasis_base_s1 and
    # z_npca_base_s1 hash identically. Blending both gives that ensemble weight 10 instead of 5 and
    # calls it diversity. Names cannot catch this; the weights can.
    n_models, by_hash = {}, {}
    for n in names:
        art = joblib.load(artifact_path(n))
        if int(art["split_salt"]) != a.salt:
            raise SystemExit(f"{n} was fitted under salt {art['split_salt']}, not {a.salt}")
        h = hashlib.sha256()
        for k in sorted(art.get("models", {})):
            st = getattr(art["models"][k], "_state", None)
            if st:
                for _, v in sorted(st.items()):
                    h.update(np.ascontiguousarray(v).tobytes())
        key = h.hexdigest()
        if key in by_hash:
            print(f"    dropping {n}: identical weights to {by_hash[key]}")
            continue
        by_hash[key] = n
        n_models[n] = len(art["ensemble"]) if "ensemble" in art else 1
        seen |= set(art["train_files"]) | set(art.get("val_files", []))
    names = list(n_models)
    overlap = {p.name for p in files} & seen
    if overlap:
        raise SystemExit(f"{len(overlap)} scored shots were in some member's fit")

    print(f"  models per artifact: "
          f"{', '.join(f'{k}={v}' for k, v in sorted(n_models.items(), key=lambda kv: -kv[1]))}")
    print(f"  {sum(n_models.values())} models in total, each carrying weight 1\n")
    shots = LS.load_shots(len(files), 0, "local", DEFAULT_LOCAL_DATA_DIR, HF_TRAIN_CONFIG, files)
    machine, R, Z, mask_coarse, mask_f, refs, ref_stats, means, mean_psi = ground_truth_setup(
        shots, a.jobs)

    print("  predicting each member once")
    preds: dict[str, list[dict]] = {}
    for n in names:
        path = artifact_path(n)
        preds[n] = [{k: v.astype(np.float32) for k, v in
                     predict_row(s["row"], machine, "ensemble", path).items()} for s in shots]
        print(f"    {n}", flush=True)

    def score(members: list[str]) -> float:
        tot = float(sum(n_models[n] for n in members))
        w = {n: n_models[n] / tot for n in members}
        blend = [{k: sum(w[n] * preds[n][i][k] for n in members)
                  for k in ("psirz", "q95", "betaN")} for i in range(len(shots))]
        sign = LS.choose_psi_sign(shots, blend, mean_psi)
        acc = LS.Accum()
        acc.psi_sign = sign
        tasks = [({"psi": s["psi"], "q95": s["q95"], "betaN": s["betaN"]},
                  ref, p, R, Z, mask_coarse, mask_f, sign, means, machine)
                 for s, ref, p in zip(shots, refs, blend, strict=True)]
        for part in LS._map(LS._score_task, tasks, a.jobs, ""):
            acc.add(part)
        return float(LS.finalize_machine(acc, ref_stats)["S"])

    rows = []
    solo = {n: score([n]) for n in names}
    for n, s in sorted(solo.items(), key=lambda kv: -kv[1]):
        print(f"    solo {n:<24s} {s:.6f}")
        rows.append((a.salt, "solo", 1, s, n))
    order = [n for n, _ in sorted(solo.items(), key=lambda kv: -kv[1])]

    print("\n  coarse pass: top 25/50/75/100% by solo score, equal weights")
    best_set, best_s = None, -1.0
    for frac in (0.25, 0.50, 0.75, 1.0):
        k = max(1, round(frac * len(order)))
        s = score(order[:k])
        print(f"    top {frac:>4.0%} ({k:2d} members): {s:.6f}")
        rows.append((a.salt, f"top{int(frac * 100)}", k, s, "+".join(order[:k])))
        if s > best_s:
            best_set, best_s = order[:k], s

    print(f"\n  greedy from the coarse winner ({len(best_set)} members, {best_s:.6f})")
    cur, cur_s = list(best_set), best_s
    for r in range(a.rounds):
        cands = [n for n in order if n not in cur]
        if not cands:
            break
        gains = {n: score([*cur, n]) for n in cands}
        n_best = max(gains, key=gains.get)
        if gains[n_best] <= cur_s:
            print(f"    round {r + 1}: best addition {n_best} gives {gains[n_best]:.6f}, "
                  f"no improvement on {cur_s:.6f} — stopping")
            break
        cur, cur_s = [*cur, n_best], gains[n_best]
        print(f"    round {r + 1}: +{n_best} -> {cur_s:.6f}")
        rows.append((a.salt, f"greedy{r + 1}", len(cur), cur_s, "+".join(cur)))

    a.out.parent.mkdir(exist_ok=True)
    new = not a.out.exists()
    with a.out.open("a") as fh:
        if new:
            fh.write("salt,rung,n,S,members\n")
        for salt, rung, n, s, m in rows:
            fh.write(f"{salt},{rung},{n},{s:.6f},{m}\n")
    print(f"\n  WINNER on salt {a.salt}: {cur_s:.6f} with {len(cur)} members")
    print(f"    salts:{'+'.join(cur)}")
    print("  This is a SELECTION number. Confirm it on the other salts, with local_score itself.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

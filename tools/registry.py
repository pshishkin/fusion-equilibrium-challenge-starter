#!/usr/bin/env python3
"""The model collection: what is on disk, what seed it carries, and what a blend of it can be
scored on.

    uv run python tools/registry.py                      # the catalogue, grouped by split
    uv run python tools/registry.py --blend a b c        # can THESE be blended, and scored where

**Why the third column is the important one.** `salts:a+b+c` averages the decoded predictions of
several artifacts, and `evaluate.py` refuses the run unless NO member has seen a scored shot. Each
fit takes 95% of the corpus (80% train + 15% validation), so two artifacts under different salts
have between them seen almost everything. A collection is only as useful as the set of shots none of
its members has touched, and that set is what this prints.

Everything is read from the artifacts themselves — `params_yaml` is the EFFECTIVE configuration the
fit ran under, not the file as it stands today — so a model catalogued here says what produced it
even after params.yaml has moved on.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from my_experiments.baseline_model import sorted_shots

LANES = Path("artefacts/lanes")
RESULTS = Path("results")


def recorded_scores(root: Path) -> dict[str, dict]:
    """Every score already on disk, keyed by the artifact it belongs to.

    A lane writes `<name>,<S>,<R2_psi>,<R2_qb>,<1-D_LCFS>,<Cons>` and names the artifact
    `baseline_<name>.joblib`, so the join is exact and free — no artifact needs re-scoring to appear
    here with a number.

    **What the number IS, stated because the CSV does not say it.** It is that lane's own scoring
    pass on the 352-shot tail of that artifact's salt, under whatever `--score-models` the lane
    used: `ensemble` for a production lane, a single member for a screen. So it is comparable ACROSS
    rows of one lane and WITHIN one salt, which is exactly what the lanes were built for, and it is
    not a per-member figure for a multi-member artifact. Later rows win, since a re-run supersedes.
    """
    out: dict[str, dict] = {}
    for f in sorted(root.glob("*.csv")):
        try:
            rows = list(csv.DictReader(f.open()))
        except Exception:
            continue
        for r in rows:
            name, s_ = r.get("name"), r.get("S")
            if not name or not s_:
                continue
            try:
                out[name] = {"S": float(s_), "src": f.name,
                             **{k: (float(r[k]) if r.get(k) else None)
                                for k in ("R2_psi", "R2_qb", "D_LCFS", "Cons")}}
            except ValueError:
                continue
    return out


def flat(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


def summarise(path: Path, base: dict) -> dict | None:
    try:
        a = joblib.load(path)
    except Exception:
        return None
    if "train_files" not in a or "split_salt" not in a:
        return None
    cfg = flat(yaml.safe_load(a.get("params_yaml") or "") or {})
    delta = {k: v for k, v in cfg.items()
             if k in base and v != base[k] and not k.endswith(".enabled")}
    seeds = {n: getattr(m, "seed", None) for n, m in a.get("models", {}).items()
             if getattr(m, "seed", None) is not None}
    return {
        "name": path.stem.removeprefix("baseline_"),
        "salt": int(a["split_salt"]),
        "models": list(a.get("models", {})),
        "seeds": seeds,
        "seen": set(a["train_files"]) | set(a["val_files"]),
        "n_train": a.get("n_train_shots"), "n_val": a.get("n_val_shots"),
        "holdout": float(a.get("holdout_share") or 0.0),
        "delta": delta,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blend", nargs="+", metavar="NAME",
                    help="artifact names (as printed in the catalogue) to test as one blend")
    ap.add_argument("--lanes", type=Path, default=LANES)
    args = ap.parse_args()

    base = flat(yaml.safe_load(Path("params.yaml").read_text()))
    corpus = {p.name for p in sorted_shots()}
    rows = [r for r in (summarise(p, base) for p in sorted(args.lanes.glob("*.joblib")))
            if r is not None]
    scores = recorded_scores(RESULTS)
    for r in rows:
        r["score"] = scores.get(r["name"])
    if not rows:
        raise SystemExit(f"no readable artifacts under {args.lanes}")

    if args.blend:
        by_name = {r["name"]: r for r in rows}
        missing = [n for n in args.blend if n not in by_name]
        if missing:
            raise SystemExit(f"not in the collection: {missing}")
        chosen = [by_name[n] for n in args.blend]
        seen = set().union(*(r["seen"] for r in chosen))
        free = corpus - seen
        print(f"  blend of {len(chosen)}: {', '.join(r['name'] for r in chosen)}")
        print(f"  salts involved: {sorted({r['salt'] for r in chosen})}")
        print(f"  shots unseen by EVERY member: {len(free)} of {len(corpus)} "
              f"({len(free) / len(corpus):.1%})")
        if not free:
            print("  -> this blend CANNOT be scored locally. Every shot was in some member's fit,\n"
                  "     so evaluate.py will refuse it, and it is right to: a score on shots a\n"
                  "     member trained on is that member's memory. Reserve a holdout before\n"
                  "     fitting the members, not after.")
        return 0

    groups = defaultdict(list)
    for r in rows:
        groups[(r["salt"], frozenset(r["seen"]))].append(r)
    n_mod = sum(len([m for m in r["models"] if m != "ridge"]) for r in rows)
    n_scored = sum(1 for r in rows if r["score"])
    print(f"  {len(rows)} artifacts holding {n_mod} fitted models (ridge excluded), "
          f"{len(groups)} distinct splits, corpus {len(corpus)} shots")
    print(f"  {n_scored} of {len(rows)} carry a recorded score; the rest were never scored on "
          f"their own tail\n")
    for (salt, seen), members in sorted(groups.items(), key=lambda kv: (kv[0][0], -len(kv[1]))):
        free = len(corpus - seen)
        print(f"  salt {salt}   {len(members):2d} artifact(s)   "
              f"{free} shots unseen by these ({free / len(corpus):.1%}) — blendable and scorable "
              f"together")
        # Best first, because the catalogue is read to choose blend members.
        for r in sorted(members, key=lambda r: (-(r["score"] or {}).get("S", -1), r["name"])):
            seeds = ",".join(f"{n}={s}" for n, s in sorted(r["seeds"].items()))
            d = " ".join(f"{k.removeprefix('models.').removeprefix('features.')}={v}"
                         for k, v in sorted(r["delta"].items())) or "-"
            sc = r["score"]
            got = (f"S {sc['S']:.6f}  qb {sc['R2_qb']:.5f}  1-D {sc['D_LCFS']:.5f}  "
                   f"cons {sc['Cons']:.5f}" if sc and sc.get("R2_qb") else
                   (f"S {sc['S']:.6f}" if sc else "not scored"))
            print(f"      {r['name']:<28s} {got}")
            print(f"      {'':<28s} {','.join(r['models'])}  seeds[{seeds}]"
                  + (f"  [{sc['src']}]" if sc else ""))
            if d != "-":
                print(f"      {'':<28s} differs: {d}")
        print()
    everywhere = corpus - set().union(*(r["seen"] for r in rows))
    print("  A blend is assembled WITHIN a salt, and validated by repeating it across salts: the\n"
          "  ensemble at salt 0 scored on salt 0's tail against the ensemble at salt 1 on salt\n"
          "  1's, paired the way every other measurement here is. Each group above is therefore\n"
          "  a ready-made blend with a clean 352-shot tail, and no holdout is involved.\n")
    print(f"  (Recorded so the question is not re-asked: {len(everywhere)} shots are unseen by "
          f"EVERY artifact, so a blend mixing MEMBERS FROM DIFFERENT SALTS has no clean scoring\n"
          f"  set at all — each fit takes 95% of the corpus and 12 salts between them cover it.\n"
          f"  That would need `split.holdout_share` non-zero before the members were fitted, and\n"
          f"  it is 0.0 in params.yaml. It is not needed for the design above.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

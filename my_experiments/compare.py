#!/usr/bin/env python3
"""Paired deltas out of a sweep CSV, against the threshold selection alone would clear.

    uv run python my_experiments/compare.py results/i_capacity.csv --control a15_w512_d2_p0

Two things this exists to stop, both of which have already happened in this fork:

* **Comparing against an estimated control.** A delta was once reported against a baseline that
  was reasoned about rather than run; the measured control turned out 0.0010 away, and the
  headline number was wrong by half. The control here is a ROW, or the run fails.
* **Reading the best of many arms as a result.** Over `n` arms of pure noise the largest is
  expected at `sigma * sqrt(2 ln n)` above the control — +0.0029 over twelve arms at this fork's
  sigma of 0.0013. Thirty screened configurations produced a best of +0.0013, i.e. LESS than
  selection alone would give, and two arms carried past that point both went negative on unseen
  salts. The threshold is printed beside every delta, so no arm can be read without it.

Salts are handled by refusing to mix them: absolute scores are not comparable across splits — the
scored shots differ — so `--salt-suffix` groups rows by the trailing `_sN` of their name and
compares each group against its own control.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

# One net's spread at production, measured 2026-08-13 over three salts by three seeds.
SIGMA = 0.0013
SALT = re.compile(r"_s(\d+)$")


def threshold(n_arms: int, sigma: float = SIGMA) -> float:
    """What the best of `n_arms` noise draws is expected to clear the control by."""
    return sigma * math.sqrt(2 * math.log(n_arms)) if n_arms > 1 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, help="a sweep result CSV")
    ap.add_argument("--control", required=True,
                    help="the row every other row is read against, by name (or its salt-stripped "
                         "stem when --salt-suffix is given)")
    ap.add_argument("--salt-suffix", action="store_true",
                    help="group rows by the trailing _sN and compare within each salt")
    ap.add_argument("--sigma", type=float, default=SIGMA)
    args = ap.parse_args()

    with args.csv.open() as fh:
        rows = [r for r in csv.DictReader(fh) if r["S"] not in ("", "FAILED")]
    if not rows:
        raise SystemExit(f"{args.csv}: no scored rows")

    def split(name: str) -> tuple[str, str]:
        m = SALT.search(name) if args.salt_suffix else None
        return (name[:m.start()], m.group(1)) if m else (name, "")

    groups: dict[str, list[tuple[str, float]]] = {}
    for r in rows:
        stem, salt = split(r["name"])
        groups.setdefault(salt, []).append((stem, float(r["S"])))

    n_arms = max(len({stem for stem, _ in g}) - 1 for g in groups.values())
    bar = threshold(n_arms, args.sigma)
    print(f"{args.csv.name}: {len(rows)} runs, {n_arms} arms against '{args.control}'")
    print(f"selection alone clears +{bar:.4f} over {n_arms} arms "
          f"(sigma {args.sigma:.4f} x sqrt(2 ln n)); an arm below that has shown nothing\n")

    means: dict[str, list[float]] = {}
    for salt in sorted(groups):
        scored = dict(groups[salt])
        if args.control not in scored:
            raise SystemExit(f"salt {salt or '-'}: no control row '{args.control}' — it holds "
                             f"{sorted(scored)}. A control is measured, never assumed.")
        base = scored[args.control]
        head = f"salt {salt}" if salt else "all runs"
        print(f"  {head}   control {args.control} = {base:.4f}")
        for stem, s in sorted(scored.items(), key=lambda kv: -kv[1]):
            if stem == args.control:
                continue
            d = s - base
            means.setdefault(stem, []).append(d)
            print(f"    {stem:28s} {s:.4f}  {d:+.4f}  {'OVER' if d > bar else '':4s}")
        print()

    if len(groups) > 1:
        print("  across salts (the number that matters — signs that disagree are noise):")
        for stem, ds in sorted(means.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
            mean = sum(ds) / len(ds)
            agree = all(d > 0 for d in ds) or all(d < 0 for d in ds)
            print(f"    {stem:28s} mean {mean:+.4f} over {len(ds)}  "
                  f"[{', '.join(f'{d:+.4f}' for d in ds)}]  "
                  f"{'signs agree' if agree else 'SIGNS DISAGREE'}")
            # The spread of the paired deltas themselves, which is the honest uncertainty of the
            # effect — and the right test when the arm was pre-registered, where the selection
            # threshold above measures nothing because nothing was selected.
            if len(ds) > 2:
                var = sum((d - mean) ** 2 for d in ds) / (len(ds) - 1)
                se = math.sqrt(var / len(ds))
                t = mean / se if se > 0 else float("inf")
                print(f"    {'':28s} standard error {se:.4f}, t = {t:+.2f}"
                      f"   {'resolved at 2 sigma' if abs(t) >= 2 else 'BELOW what this resolves'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Read paired multi-salt sweeps and apply the gate that was fixed before they ran.

    uv run python my_experiments/paired.py results/x_base_16salts.csv \
        results/x_jac_16salts.csv results/x_ema_16salts.csv results/x_aux_16salts.csv

The first CSV is the baseline; every other is an arm. Rows are matched **by salt**, never by
position, and absolute scores are never compared across salts — a salt reshuffles which shots are
scored, so only the within-salt difference means anything.

**Why the gate has two halves.** The mean says how big the effect is; the sign count says whether
it is one salt carrying it. Both were fixed in ideas.md ("X. The 16-salt program") before the first
run: accept at mean paired dS >= +0.00065 AND at least 12 of 16 signs positive.

**And why +0.00065 rather than 2 SE of 0.00033.** Over three arms the best-from-noise alone is
SE*sqrt(2 ln 3) = 0.00049. A threshold that does not move with the number of arms turns a finer
instrument into a finer noise detector, which is what the thirty-arm grid of 2026-08-14 measured
the hard way: its best arm (+0.0013) came in BELOW the +0.0029 selection alone would have produced.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

SALT = re.compile(r"_s(\d+)$")
GATE_MEAN = 0.00065
GATE_SIGNS = 12


TERMS = ("S", "R2_psi", "R2_qb", "D_LCFS", "Cons")
# The composite's weights, so a per-term difference can be quoted in the units that matter --
# what it contributes to S -- rather than in the term's own scale. D_LCFS enters as (1 - D).
WEIGHT = {"S": 1.0, "R2_psi": 0.55, "R2_qb": 0.15,
          # The column is HEADED "D_LCFS" but the runners write `1 - D_LCFS` into it, because that
          # is what local_score.py:481 prints. In S the term is +0.10 * (1 - D_LCFS), so the weight
          # here is +0.10, not -0.10. Caught 2026-08-21 by arithmetic: with -0.10 the huber
          # decomposition summed to +0.00079 against a measured +0.00088; with +0.10 it sums to
          # +0.00087. Higher is better in this column, like every other one.
          "D_LCFS": 0.10, "Cons": 0.20}


def read(path: Path, term: str = "S") -> dict[int, float]:
    """`{salt: value}` for one arm and one term, keyed off the trailing `_s<N>` of the row name."""
    out: dict[int, float] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            m = SALT.search(row["name"])
            if m is None:
                raise SystemExit(f"{path}: row {row['name']!r} has no _s<salt> suffix")
            salt = int(m.group(1))
            if salt in out:
                raise SystemExit(f"{path}: salt {salt} appears twice")
            # sweep.py writes FAILED for a run that died, and a dead run is missing data rather
            # than a zero — dropping the salt keeps the pairing honest instead of scoring the
            # arm as if the failure were a result. Its OTHER columns come back empty rather than
            # FAILED, so both forms have to be caught: reading `S` alone is not enough once
            # --terms asks for the rest.
            if row[term] in ("FAILED", "", None):
                continue
            out[salt] = float(row[term])
    return out


def stats(d: list[float]) -> tuple[float, float, int]:
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else float("nan")
    return mean, sd / math.sqrt(n) if n > 1 else float("nan"), sum(1 for x in d if x > 0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline", type=Path)
    ap.add_argument("arms", type=Path, nargs="+")
    ap.add_argument("--terms", action="store_true",
                    help="also decompose each arm across the four scored terms, in units of the "
                         "contribution to S -- which is where a gain came from, not just how big")
    a = ap.parse_args(argv)
    base = read(a.baseline)
    print(f"baseline {a.baseline.name}: {len(base)} salts, "
          f"S {min(base.values()):.4f}..{max(base.values()):.4f}\n")
    print(f"{'arm':<26}{'n':>4}{'mean dS':>10}{'SE':>9}{'t':>7}{'signs':>8}   verdict")
    for path in a.arms:
        arm = read(path)
        shared = sorted(set(base) & set(arm))
        d = [arm[s] - base[s] for s in shared]
        n = len(d)
        if n < 2:
            print(f"{path.stem:<26}{n:>4}   too few paired salts yet")
            continue
        mean = sum(d) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
        se = sd / math.sqrt(n)
        pos = sum(1 for x in d if x > 0)
        ok = mean >= GATE_MEAN and pos >= GATE_SIGNS and n >= 16
        verdict = ("ACCEPTED" if ok else
                   "incomplete" if n < 16 else
                   "refuted at its gate")
        print(f"{path.stem:<26}{n:>4}{mean:>+10.5f}{se:>9.5f}"
              f"{(mean / se if se > 0 else float('nan')):>7.1f}{pos:>4}/{n:<3}   {verdict}")
        print(f"{'':>26}  per salt: " + " ".join(f"{x*1e4:+.0f}" for x in d) + "   (units 1e-4)")
        if a.terms:
            for term in TERMS[1:]:
                tb, ta = read(a.baseline, term), read(path, term)
                dt = [ta[s] - tb[s] for s in shared]
                m2, se2, pos2 = stats(dt)
                print(f"{'':>28}{term:<8}{m2:+9.5f} raw"
                      f"{m2 * WEIGHT[term]:+10.5f} of S"
                      f"   SE {se2:.5f}   {pos2}/{len(dt)} up")
    print(f"\ngate, fixed before the runs: mean >= {GATE_MEAN:+.5f} AND "
          f"{GATE_SIGNS}/16 signs positive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

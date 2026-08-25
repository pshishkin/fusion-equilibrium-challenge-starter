"""Read a production pair file: both arms in ONE csv, rows named <tag>_<arm>_s<salt>.

`paired.py` matches a baseline file against arm files. The production confirmations instead run
both arms inside a single lane so a pair always completes together, which puts them in one file.
The arithmetic is the same: match by SALT, difference within the salt, never compare across salts.
"""
import argparse, csv, re, statistics as st
from pathlib import Path

TERMS = {"S": 1.0, "R2_psi": 0.55, "R2_qb": 0.15,
          # The column is HEADED "D_LCFS" but the runners write `1 - D_LCFS` into it, because that
          # is what local_score.py:481 prints. In S the term is +0.10 * (1 - D_LCFS), so the weight
          # here is +0.10, not -0.10. Caught 2026-08-21 by arithmetic: with -0.10 the huber
          # decomposition summed to +0.00079 against a measured +0.00088; with +0.10 it sums to
          # +0.00087. Higher is better in this column, like every other one.
          "D_LCFS": 0.10, "Cons": 0.20}


def load(path, base_arm):
    rows = {}
    for r in csv.DictReader(Path(path).open()):
        m = re.match(r"^(.+?)_(.+?)_s(\d+)$", r["name"])
        if not m:
            continue
        _, arm, salt = m.groups()
        if r["S"] in ("FAILED", "", None):      # a dead run is not a zero
            continue
        rows.setdefault(arm, {})[int(salt)] = r
    arms = sorted(rows, key=lambda a: a != base_arm)
    if base_arm not in rows:
        raise SystemExit(f"no arm {base_arm!r}; file holds {sorted(rows)}")
    return rows, arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--base", default="base")
    ap.add_argument("--terms", action="store_true")
    a = ap.parse_args()
    rows, arms = load(a.path, a.base)
    base = rows[a.base]
    print(f"  {'arm':<12}{'n':>4}{'mean':>10}{'SE':>9}{'t':>7}  signs")
    for arm in arms:
        if arm == a.base:
            continue
        salts = sorted(set(base) & set(rows[arm]))
        if not salts:
            print(f"  {arm:<12}   0   no completed pairs yet")
            continue
        d = [float(rows[arm][s]["S"]) - float(base[s]["S"]) for s in salts]
        n = len(d)
        mean = st.fmean(d)
        se = st.stdev(d) / n ** 0.5 if n > 1 else float("nan")
        pos = sum(x > 0 for x in d)
        t = mean / se if n > 1 and se else float("nan")
        print(f"  {arm:<12}{n:>4}{mean:>+10.5f}{se:>9.5f}{t:>7.1f}  {pos}/{n}")
        print(f"      per salt: " + " ".join(f"{x*1e4:+.1f}" for x in d) + "   (units 1e-4)")
        if a.terms:
            for term, w in TERMS.items():
                if term == "S":
                    continue
                dt = [float(rows[arm][s][term]) - float(base[s][term]) for s in salts]
                print(f"      {term:<8}{st.fmean(dt):+.5f}  -> S {w*st.fmean(dt):+.5f}")


if __name__ == "__main__":
    main()

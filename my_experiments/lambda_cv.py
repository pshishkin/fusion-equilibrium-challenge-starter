"""Is the test-time coefficient scale a real gain, or the best of eleven noise draws?

The first sweep found lambda* = 1.005 worth +0.000075 — but it PICKED lambda on the same 70 shots it
scored on, so the number is an in-sample optimum and biased upward by construction. This project has
thrown work away for exactly that before (a twelve-feature model that cross-validated at +0.972).

The honest test costs nothing extra: `artefacts/tailnoise/base_s0.joblib` was trained under
0.80/0.15/0.05, so it has a clean 352-shot tail. Split those shots into two disjoint halves,
round-robin through the sha1 order so neither half is one corner of it; choose lambda on half A,
report what it earns on half B, and vice versa. The reported number is the mean of the two
out-of-sample gains, which no amount of sweeping can inflate.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib

from experiments import DEFAULT_LOCAL_DATA_DIR, HF_TRAIN_CONFIG
from my_experiments.baseline_model import sorted_shots, take_share

GRID = ["0.9900", "0.9950", "1.0000", "1.0025", "1.0050", "1.0075", "1.0100", "1.0150", "1.0250"]


def score(files, lam, art, jobs, data_dir, config):
    env = dict(os.environ, FUSION_ARTIFACT=str(art), FUSION_COEF_LAMBDA=lam,
               FUSION_COEF_LAMBDA_FROM="0")
    out = subprocess.run(
        [sys.executable, "local_score.py", "--source", "local", "--local-data-dir", str(data_dir),
         "--config", config, "--mode", "model", "--jobs", str(jobs), "--models", "ensemble",
         "--files", *[str(p) for p in files]],
        capture_output=True, text=True, env=env, cwd=Path(__file__).resolve().parent.parent)
    m = re.search(r"COMPOSITE S = ([\d.]+)", out.stdout)
    if not m:
        sys.stderr.write(out.stdout[-3000:] + out.stderr[-2000:])
        raise SystemExit(f"no score for lambda={lam}")
    return float(m.group(1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", type=Path, default=Path("artefacts/tailnoise/base_s0.joblib"))
    ap.add_argument("--share", type=float, default=0.05)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR)
    ap.add_argument("--config", default=HF_TRAIN_CONFIG)
    a = ap.parse_args()

    art = joblib.load(a.artifact)
    salt = int(art.get("split_salt", 0))
    seen = set(art.get("train_files", ())) | set(art.get("val_files", ()))
    tail = take_share(sorted_shots(a.local_data_dir, a.config, salt), a.share, "tail")
    if [p for p in tail if p.name in seen]:
        raise SystemExit("tail overlaps the fit — a score on those shots means nothing")
    halves = [tail[0::2], tail[1::2]]
    print(f"  {len(tail)} clean tail shots -> halves of {len(halves[0])} and {len(halves[1])}")

    S = [{}, {}]
    for h in (0, 1):
        for lam in GRID:
            S[h][lam] = score(halves[h], lam, a.artifact, a.jobs, a.local_data_dir, a.config)
        print(f"  half {h}: " + "  ".join(f"{l}:{S[h][l]:.6f}" for l in GRID))

    print(f"\n  {'fit on':>8}{'lambda*':>9}{'in-sample':>12}{'OUT-OF-SAMPLE':>15}")
    oos = []
    for h in (0, 1):
        other = 1 - h
        star = max(GRID, key=lambda l: S[h][l])
        ins = S[h][star] - S[h]["1.0000"]
        out = S[other][star] - S[other]["1.0000"]
        oos.append(out)
        print(f"  {'half ' + str(h):>8}{star:>9}{ins:>+12.6f}{out:>+15.6f}")
    print(f"\n  mean out-of-sample gain: {sum(oos)/2:+.6f}")
    print("  (the in-sample column is what a sweep reports; the last column is what it is worth)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

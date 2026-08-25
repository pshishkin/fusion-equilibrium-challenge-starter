"""What weight should `seq` carry in the ensemble? Answered by RESCORING, with no training at all.

Every member ships at 1.0, so `seq` gets exactly 1/5. Those weights were fixed before `seq` gained
its cosine schedule (+0.00010 in production, t = 5.1), and nothing has revisited them since. The
mechanism for expecting a different answer is measured: `mlp` alone scores 0.9931, four seeds 0.9940,
`mlp+seq` **0.9943** — one extra architecture beats three extra seeds, because seeds correlate at
0.56 and a different architecture does not.

**This is not A21, which is closed.** A21 FITTED all five weights and cross-validated at -0.00002,
worse than equal: five free parameters on a handful of shots overfit. This sweeps ONE weight coarsely.

**The gate is two-column, and it is not optional.** The lambda sweep returned +0.000075 in-sample and
-0.00015 on shots it had not seen, because the best of eleven noise draws is positive by
construction. So: choose the weight on one half of the clean tail, report what it earns on the other,
and swap. The number that counts is the mean of the two out-of-sample columns.
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

# seq weight w against four MLPs at 1.0 each; w=1.0 is what ships (seq = 1/5 of the blend).
GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]


def spec(w: float) -> str:
    mlps = "+".join(f"{m}*1.0" for m in ("mlp", "mlp1", "mlp2", "mlp3"))
    return mlps if w == 0.0 else f"{mlps}+seq*{w}"


def score(files, w, art, jobs, data_dir, config):
    env = dict(os.environ, FUSION_ARTIFACT=str(art))
    out = subprocess.run(
        [sys.executable, "local_score.py", "--source", "local", "--local-data-dir", str(data_dir),
         "--config", config, "--mode", "model", "--jobs", str(jobs), "--models", spec(w),
         "--files", *[str(p) for p in files]],
        capture_output=True, text=True, env=env, cwd=Path(__file__).resolve().parent.parent)
    m = re.search(r"COMPOSITE S = ([\d.]+)", out.stdout)
    if not m:
        sys.stderr.write(out.stdout[-3000:] + out.stderr[-2000:])
        raise SystemExit(f"no score for seq weight {w}")
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
        for w in GRID:
            S[h][w] = score(halves[h], w, a.artifact, a.jobs, a.local_data_dir, a.config)
        print(f"  half {h}: " + "  ".join(f"{w}:{S[h][w]:.6f}" for w in GRID))

    print(f"\n  {'fit on':>8}{'w*':>7}{'in-sample':>12}{'OUT-OF-SAMPLE':>15}   (vs w=1.0, what ships)")
    oos = []
    for h in (0, 1):
        o = 1 - h
        star = max(GRID, key=lambda w: S[h][w])
        print(f"  {'half ' + str(h):>8}{star:>7}{S[h][star]-S[h][1.0]:>+12.6f}"
              f"{S[o][star]-S[o][1.0]:>+15.6f}")
        oos.append(S[o][star] - S[o][1.0])
    print(f"\n  mean out-of-sample gain over the shipped weight: {sum(oos)/2:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

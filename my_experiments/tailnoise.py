"""How much of a paired comparison's noise is the SHOT SAMPLE, and does a bigger tail cut it?

The screens moved from a 1% tail (70 shots) to a 5% tail (352) on the argument that 86% of a
production comparison's paired variance is the sample rather than the training, so five times the
shots should cut the paired sd by 44%. That argument came from a decomposition; this measures the
thing directly.

Cut the tail into K disjoint blocks, score every block, and the spread of a score ACROSS blocks is
the sample noise at that block size, measured rather than modelled. Blocks share no shots, so the
sd at the full tail follows as block_sd / sqrt(K).

TWO MODES, and the difference between them was learned the hard way:

  --models a b ...   several members of ONE artifact, scored on identical shots. Cheap: no training.
                     This was first used as a stand-in for an arm comparison, on the argument that
                     two seeds "differ only in initialisation". MEASURED, THAT PROXY IS WRONG — it
                     overstates the paired sample noise about fivefold (pooled paired sd 0.00125 at
                     70 shots, against 0.00025 for the real base-vs-ema production difference). A
                     seed change is a large perturbation and its effect varies a lot shot to shot;
                     a loss change at fixed seed is a small one. Use this mode for how the ABSOLUTE
                     score varies, which is what it honestly measures.

  --artifact-b PATH  two artifacts differing in ONE setting, same salt, same split. This is the real
                     arm comparison and the only mode whose paired numbers mean what they say. It
                     costs two training runs to produce the artifacts.

Blocks are taken ROUND-ROBIN through the sha1 order, not as contiguous chunks: contiguous blocks
would each be one corner of the ordering, and the sha1 order is not meaningless — it is what the
train/val/test windows are carved from.
"""
from __future__ import annotations

import argparse
import os
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib

from experiments import DEFAULT_LOCAL_DATA_DIR, HF_TRAIN_CONFIG
from my_experiments.baseline_model import sorted_shots, take_share

ROW = re.compile(r"^\s+(\S+)\s+([\d.]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s*$")


def score_block(files: list[Path], models: list[str], jobs: int, artifact: Path,
                data_dir: Path, config: str) -> dict[str, float]:
    """Run the real scorer on exactly these shots and return {model: S}."""
    env = dict(os.environ, FUSION_ARTIFACT=str(artifact))
    cmd = [sys.executable, "local_score.py", "--source", "local",
           "--local-data-dir", str(data_dir), "--config", config,
           "--mode", "model", "--jobs", str(jobs), "--models", *models,
           "--files", *[str(p) for p in files]]
    out = subprocess.run(cmd, capture_output=True, text=True, env=env,
                         cwd=Path(__file__).parent.parent)
    if out.returncode != 0:
        sys.stderr.write(out.stdout[-4000:] + out.stderr[-4000:])
        raise SystemExit(f"scorer failed on a block of {len(files)}")
    got = {}
    for line in out.stdout.splitlines():
        line = re.sub(r"^\[[^\]]*\]", "", line)
        m = ROW.match(line)
        if m and m.group(1) in models:
            got[m.group(1)] = float(m.group(2))
    missing = set(models) - set(got)
    if missing:
        sys.stderr.write(out.stdout[-4000:])
        raise SystemExit(f"no score row for {sorted(missing)}")
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", type=Path,
                    default=Path("artefacts/candidate/baseline_prod.joblib"))
    ap.add_argument("--artifact-b", type=Path,
                    help="second artifact differing in ONE setting; every block is scored under "
                         "both and the paired difference is the real arm comparison")
    ap.add_argument("--share", type=float, default=0.05, help="the tail to cut up (default 5%%)")
    ap.add_argument("--blocks", type=int, default=5, help="disjoint blocks to cut it into")
    ap.add_argument("--models", nargs="+", default=["mlp", "mlp1", "mlp2", "mlp3"])
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR)
    ap.add_argument("--config", default=HF_TRAIN_CONFIG)
    a = ap.parse_args()

    art = joblib.load(a.artifact)
    salt = int(art.get("split_salt", 0))
    seen = set(art.get("train_files", ())) | set(art.get("val_files", ()))
    tail = take_share(sorted_shots(a.local_data_dir, a.config, salt), a.share, "tail")
    overlap = [p for p in tail if p.name in seen]
    if overlap:
        raise SystemExit(f"{len(overlap)} of the tail shots were TRAINED or VALIDATED on — "
                         f"a score on those means nothing")
    blocks = [tail[i::a.blocks] for i in range(a.blocks)]
    print(f"  {len(tail)} tail shots (salt {salt}) -> {a.blocks} disjoint blocks of "
          f"{min(len(b) for b in blocks)}-{max(len(b) for b in blocks)}")

    if a.artifact_b:
        artb = joblib.load(a.artifact_b)
        if int(artb.get("split_salt", 0)) != salt:
            raise SystemExit(f"salt mismatch: {a.artifact} is {salt}, {a.artifact_b} is "
                             f"{artb.get('split_salt')} — different splits are not a paired arm")
        seen_b = set(artb.get("train_files", ())) | set(artb.get("val_files", ()))
        if seen_b != seen:
            raise SystemExit("the two artifacts trained on DIFFERENT shots — then a difference "
                             "between them is not one setting changed")

    per: list[dict[str, float]] = []
    for i, b in enumerate(blocks):
        got = score_block(b, a.models, a.jobs, a.artifact, a.local_data_dir, a.config)
        if a.artifact_b:
            gb = score_block(b, a.models, a.jobs, a.artifact_b, a.local_data_dir, a.config)
            got = {f"A:{k}": v for k, v in got.items()} | {f"B:{k}": v for k, v in gb.items()}
        per.append(got)
        print(f"  block {i}: " + "  ".join(f"{m} {got[m]:.6f}" for m in sorted(got)))
    models = sorted(per[0]) if a.artifact_b else a.models

    k = len(per)
    n_block = len(blocks[0])
    print(f"\n  === sd of S across {k} disjoint blocks of ~{n_block} shots ===")
    for m in models:
        v = [p[m] for p in per]
        print(f"  {m:<8} sd {st.stdev(v):.5f}   -> at {len(tail)} shots: {st.stdev(v)/k**0.5:.5f}")

    print("\n  === sd of a PAIRED difference between two members ===")
    print("  (two seeds differ only in initialisation — the same structure as an arm comparison)")
    if a.artifact_b:
        # Only A-vs-B of the SAME member is an arm comparison; A:mlp vs B:mlp1 mixes both changes.
        pairs = [(f"B:{m}", f"A:{m}") for m in a.models]
        print("  (artifact B against artifact A, same member, same split — one setting changed)")
    else:
        pairs = [(x, y) for i, x in enumerate(a.models) for y in a.models[i + 1:]]
        print("  WARNING: these are two SEEDS, not two arms. Measured to overstate the paired "
              "sample noise ~5x — read the module docstring before using these numbers.")
    sds = []
    for x, y in pairs:
        d = [p[x] - p[y] for p in per]
        sd = st.stdev(d)
        sds.append(sd)
        print(f"  {x}-{y:<6} mean {st.fmean(d):+.5f}  sd {sd:.5f}"
              f"   -> at {len(tail)} shots: {sd/k**0.5:.5f}")
    if sds:
        pooled = (sum(s * s for s in sds) / len(sds)) ** 0.5
        print(f"\n  pooled paired sd at ~{n_block} shots: {pooled:.5f}")
        print(f"  pooled paired sd at {len(tail)} shots: {pooled/k**0.5:.5f}"
              f"   ({100*(1-1/k**0.5):.0f}% lower, if the sample is the only source)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Fill the decoded-shot cache for every shot, so no later run pays for parquet again.

    uv run python my_experiments/warm_cache.py            # every training shot
    uv run python my_experiments/warm_cache.py --config diii_d_public_test

Worth doing once on a fresh machine, and once before a sweep over `split.salt`: the cache is keyed
by shot, but the salt decides WHICH shots land in the training window, so a new salt otherwise
re-reads whatever it pulls in that salt 0 never touched.

Frames are asked for at the smallest share there is. That is not a shortcut — `shot_cache` stores
every frame whatever the caller wants, and the share only decides how much comes back through the
pool. Keeping it tiny is what lets this walk 7041 shots without holding 26 GB of flux in memory.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import DEFAULT_LOCAL_DATA_DIR, HF_TRAIN_CONFIG
from my_experiments.baseline_model import _read_task, sorted_shots
from my_experiments.shot_cache import CACHE_DIR
from toolkit.parallel import pimap, resolve_jobs
from toolkit.progress import SHOT_EVERY, bar_kwargs, install_timestamps


def main() -> int:
    install_timestamps()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=HF_TRAIN_CONFIG)
    ap.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR)
    ap.add_argument("--jobs", type=int, default=0,
                    help="worker processes (default 0 = cores - 2)")
    args = ap.parse_args()

    files = sorted_shots(args.local_data_dir, args.config)
    jobs = resolve_jobs(args.jobs, len(files))
    print(f"Warming {CACHE_DIR / args.config} for {len(files)} shots of {args.config} "
          f"on {jobs} process(es)")

    # The tiniest share there is: one frame per shot comes back, every frame goes into the cache.
    tasks = [(path, 1e-9) for path in files]
    for _ in tqdm(pimap(_read_task, tasks, jobs), total=len(files), unit="shot",
                  **bar_kwargs(SHOT_EVERY),
                  desc=f"caching x{jobs}"):
        pass

    total = sum(p.stat().st_size for p in (CACHE_DIR / args.config).glob("*"))
    print(f"Cache holds {total / 2 ** 30:.1f} GiB for {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
